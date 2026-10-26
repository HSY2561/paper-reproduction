"""Call the unchanged official trainer after isolating initialization RNG use.

This file changes no augmentation or optimizer formula. NumPy/Python are used
only by the official data transforms; model dropout uses Paddle's RNG. An
audited dataset proxy records the actual first two batches and all sample
indices. The official sampler owns an independent RandomState(epoch).
"""
import hashlib
import importlib.util
import json
import os
import pickle
import random
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src' / 'PaddleSeg-2.10.0'
sys.path.insert(0, str(SRC))
import numpy as np
import paddle
from paddleseg.utils import utils


def digest(value):
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def main():
    data_audit_only = '--data-audit-only' in sys.argv
    if data_audit_only:
        sys.argv.remove('--data-audit-only')
    spec = importlib.util.spec_from_file_location('w11_official_train', SRC / 'tools/train.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    original = official.train
    args = official.parse_args()
    if data_audit_only:
        args.device = 'cpu'
        def original(model, dataset, *positional, **kwargs):
            loader = paddle.io.DataLoader(dataset, batch_size=4, shuffle=False,
                                          num_workers=0, return_list=True)
            for batch, data in enumerate(loader):
                if list(data['img'].shape) != [4, 3, 400, 400]:
                    raise RuntimeError('Unexpected actual DataLoader batch shape')
                if batch == 1:
                    break
            print('DATA_AUDIT_ONLY_PASS: model/dataset/optimizer/loss constructed; two actual batches read; no backward/optimizer/GPU', flush=True)
    if args.seed not in (42, 43, 44) or args.num_workers != 0:
        raise RuntimeError('W11 requires seed42/43/44 and num_workers0')
    if args.precision != 'fp32' or args.resume_model is not None:
        raise RuntimeError('W11 supports only fresh uninterrupted FP32 runs')
    run = Path(args.save_dir).name
    witness_path = ROOT / 'provenance/rng' / (run + '.json')
    order_path = ROOT / 'provenance/rng' / (run + '_actual_indices.jsonl')
    if witness_path.exists() or order_path.exists():
        raise RuntimeError('Existing RNG evidence must be manually isolated: ' + run)
    order_path.parent.mkdir(parents=True, exist_ok=True)

    def audited_train(model, train_dataset, *positional, **kwargs):
        if kwargs.get('batch_size') != 4 or kwargs.get('precision') != 'fp32':
            raise RuntimeError('Protocol batch size or precision changed')
        if train_dataset.mode != 'train' or len(train_dataset) != 1896:
            raise RuntimeError('Only the registered training dataset is accepted')
        if kwargs.get('num_workers') != 0 or not kwargs.get('use_ema'):
            raise RuntimeError('Protocol workers/EMA changed')
        if kwargs.get('iters') not in (20, 18000):
            raise RuntimeError('Only 20iter smoke or 18000iter formal training is accepted')
        # This reset occurs after model, optimizer, loss and datasets are built.
        utils.set_seed(args.seed)
        state = dict(run=run, seed=args.seed, status='running', test_used=False,
                     source_root=str(SRC), official_train_sha256=hashlib.sha256(
                         (SRC / 'tools/train.py').read_bytes()).hexdigest(),
                     protocol='official trainer; post-construction set_seed; unchanged transforms/sampler',
                     data_audit_only=data_audit_only,
                     start_numpy_state_sha256=digest(np.random.get_state()),
                     start_python_state_sha256=digest(random.getstate()),
                     first_samples=[], epoch_first_samples=[],
                     actual_indices_path=str(order_path), batch_size=4,
                     expected_training_samples=kwargs['iters'] * 4)
        sequence_sha = hashlib.sha256()
        training_sequence_sha = hashlib.sha256()
        audit_lock = threading.Lock()
        sample_count = 0
        stream = order_path.open('x', encoding='utf-8')

        class DatasetAudit(paddle.io.Dataset):
            def __init__(self, source):
                super().__init__()
                self.source = source

            def __len__(self):
                return len(self.source)

            def __getattr__(self, name):
                if name == 'source':
                    raise AttributeError(name)
                return getattr(self.source, name)

            def __getitem__(self, index):
                nonlocal sample_count
                # A zero-worker DataLoader uses one reader thread. The lock
                # protects evidence writes and consumes no random numbers.
                with audit_lock:
                    sample_no = sample_count
                    in_budget = sample_no < state['expected_training_samples']
                    detailed = in_budget and (sample_no < 8 or sample_no % len(self) < 8)
                    before_np = digest(np.random.get_state()) if detailed else None
                    before_py = digest(random.getstate()) if detailed else None
                    data = self.source[index]
                    sequence_sha.update((str(int(index)) + '\n').encode('ascii'))
                    if in_budget:
                        training_sequence_sha.update((str(int(index)) + '\n').encode('ascii'))
                    stream.write(str(int(index)) + '\n')
                    if detailed:
                        image, label = self.source.file_list[index]
                        row = dict(sample_no=sample_no, batch=sample_no // 4,
                                   index=int(index), image=image, label=label,
                                   img=array_digest(data['img']),
                                   transformed_label=array_digest(data['label']),
                                   numpy_before=before_np, python_before=before_py,
                                   numpy_after=digest(np.random.get_state()),
                                   python_after=digest(random.getstate()))
                        state['epoch_first_samples'].append(row)
                        if sample_no < 8:
                            state['first_samples'].append(row)
                    sample_count += 1
                    if sample_count == state['expected_training_samples']:
                        # Reader prefetch may request extra samples after the
                        # last training batch. Compare the exact budget prefix.
                        state.update(actual_training_indices_sha256=training_sequence_sha.hexdigest(),
                                     training_budget_numpy_state_sha256=digest(np.random.get_state()),
                                     training_budget_python_state_sha256=digest(random.getstate()))
                    if sample_count <= 8 or sample_count % len(self) == 0:
                        state.update(observed_samples=sample_count,
                                     actual_indices_sha256=sequence_sha.hexdigest())
                        stream.flush()
                        write_json(witness_path, state)
                    return data

        write_json(witness_path, state)
        try:
            value = original(model, DatasetAudit(train_dataset), *positional, **kwargs)
            state['status'] = 'complete'
            return value
        except BaseException as error:
            state.update(status='failed', error=repr(error))
            raise
        finally:
            stream.flush()
            stream.close()
            state.update(observed_samples=sample_count,
                         actual_indices_sha256=sequence_sha.hexdigest(),
                         final_numpy_state_sha256=digest(np.random.get_state()),
                         final_python_state_sha256=digest(random.getstate()))
            write_json(witness_path, state)

    official.train = audited_train
    official.main(args)


if __name__ == '__main__':
    main()
