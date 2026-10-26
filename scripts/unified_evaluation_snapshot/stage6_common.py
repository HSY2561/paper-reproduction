"""Read-only metadata helpers for the separate final-evaluation stage."""
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECOVERY = ROOT.parent
CONTROL = RECOVERY / 'control_init_witness_20261025'
WAVE = RECOVERY.parent
PROJECT = WAVE.parent.parent
SOURCE = WAVE / 'src/PaddleSeg-2.10.0'
EXTERNAL = PROJECT / 'external/PaddleSeg-2.10.0'
PUBLIC = PROJECT / 'experiments/segformer_public_wave9_20261023'
FORMAL = PUBLIC / 'sm120_ubuntu24_formal'
W4 = PROJECT / 'experiments/retrain_wave4_20261007'
W7 = PROJECT / 'experiments/segformer_aligned_wave7_20261018'
FIELDS = ('precision', 'recall', 'f1', 'crackIoU', 'mIoU', 'boundary_f1')
PADDLE_VERSION = '3.3.0.dev20251209'
WSL_DISTRO = 'Ubuntu-24.04'
WSL_PYTHON = '${WSL_ENV}/venv/bin/python'
LD_LIBRARY_PATH = '/usr/lib/wsl/lib:${WSL_ENV}/venv/lib/python3.10/site-packages/nvidia/cu13/lib:${WSL_ENV}/venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${WSL_ENV}/venv/lib/python3.10/site-packages/nvidia/nccl/lib:${WSL_ENV}/venv/lib/python3.10/site-packages/nvidia/cusparselt/lib'


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')


def local_path(value):
    """Map retained Windows metadata to the same files mounted by WSL."""
    value = str(value)
    if os.name != 'nt' and re.match(r'^[A-Za-z]:[\\/]', value):
        return Path('/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/'))
    return Path(value)


def wsl_path(value):
    value = str(value)
    if re.match(r'^[A-Za-z]:[\\/]', value):
        return '/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/')
    return value.replace('\\', '/')


def read(path):
    return json.loads(local_path(path).read_text(encoding='utf-8-sig'))


def write(path, data, exclusive=False):
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def sha(path):
    digest = hashlib.sha256()
    with local_path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_stage_release():
    release = read(ROOT / 'provenance/release.json')
    for path, digest in release['files'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen stage/source changed: ' + path)


def manifest():
    p = ROOT / 'provenance/final_evaluation_manifest.json'
    data = read(p)
    if not data.get('selection_uses_test') is False:
        raise RuntimeError('Final manifest must be validation-only selected')
    return data


def item_by_id(ident):
    return next(x for x in manifest()['items'] if x['id'] == ident)


def verify_checkpoint(item):
    if sha(item['checkpoint']) != item['checkpoint_sha256']:
        raise RuntimeError('Frozen checkpoint SHA mismatch: ' + item['id'])


def model_and_dataset(item, split, device='gpu:0'):
    """Lazy Paddle import: metadata preparation never imports or starts CUDA."""
    cfg_path = ROOT / 'configs' / (item['id'] + '_' + split + '.yml')
    expected_cfg = item['val_config_sha256'] if split == 'val' else require_test_authorization()['config_sha256'][item['id']]
    if sha(cfg_path) != expected_cfg:
        raise RuntimeError('Frozen inference config SHA mismatch: ' + item['id'] + ' ' + split)
    import sys
    sys.path.insert(0, str(SOURCE))
    import paddle
    import numpy as np
    from paddleseg.cvlibs import Config, SegBuilder
    if paddle.__version__ != PADDLE_VERSION:
        raise RuntimeError('Unexpected final-evaluation Paddle version: ' + paddle.__version__)
    paddle.set_device(device)
    verify_checkpoint(item)
    builder = SegBuilder(Config(str(cfg_path)))
    model = builder.model
    weights = paddle.load(str(local_path(item['checkpoint'])))
    weights.pop('StructuredToParameterName@@', None)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(weights))
    extra = sorted(set(weights) - set(expected))
    shapes = {key: {'expected': list(expected[key].shape), 'found': list(weights[key].shape)}
              for key in set(expected) & set(weights)
              if list(expected[key].shape) != list(weights[key].shape)}
    if missing or extra or shapes:
        raise RuntimeError('Strict checkpoint key/shape mismatch: ' + json.dumps(
            {'id': item['id'], 'missing': missing, 'extra': extra, 'shapes': shapes}))
    model.set_state_dict(weights)
    model.eval()
    return paddle, np, model, builder, cfg_path


def require_test_authorization():
    gate = read(ROOT / 'provenance/test_authorization.json')
    if not gate.get('test_allowed'):
        raise RuntimeError('Final test is not authorized by the validation gate')
    if gate['manifest_sha256'] != sha(ROOT / 'provenance/final_evaluation_manifest.json'):
        raise RuntimeError('Final manifest changed after validation authorization')
    if gate['validation_gate_sha256'] != sha(ROOT / 'results/unified_val_gate.json'):
        raise RuntimeError('Unified validation gate changed after authorization')
    if not read(ROOT / 'results/unified_val_gate.json')['passed']:
        raise RuntimeError('Unified validation gate failed')
    return gate
