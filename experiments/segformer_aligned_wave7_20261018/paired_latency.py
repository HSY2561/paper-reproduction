"""Interleaved network-forward benchmark; no dataset or model changes."""
import json
import os
from pathlib import Path
import random
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'segformer_balance_wave6_20261016'
os.environ['PYTHONPATH'] = str(ROOT / 'src/PaddleSeg-2.10.0')
import sys
sys.path.insert(0, os.environ['PYTHONPATH'])
import numpy as np
import paddle
from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import utils


def snapshot():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,name,utilization.gpu,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw', '--format=csv'], capture_output=True, text=True, timeout=15)
        return {'exit_code': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
    except Exception as e:
        return {'error': str(e)}


def main():
    paddle.set_device('gpu:0')
    paddle.seed(42)
    ids = ['B0', 'W128', 'M13_Aligned', 'M10_GloRe']
    models = {}
    records = []
    before = snapshot()
    for ident in ids:
        weights = (ROOT.parent / 'retrain_wave4_20261007/checkpoints/segformer_b0/model.pdparams'
                   if ident == 'B0' else OLD / 'checkpoints' / ident / 'model.pdparams')
        model = SegBuilder(Config(str(OLD / 'configs' / (ident + '.yml')))).model
        utils.load_entire_model(model, str(weights))
        model.eval()
        models[ident] = model
    rng = random.Random(20261018)
    with paddle.no_grad():
        for size in (400, 480):
            x = paddle.randn([1, 3, size, size], dtype='float32')
            for model in models.values():
                for _ in range(40):
                    model(x)
            paddle.device.cuda.synchronize()
            # Paired rounds: each model measured once per round, random order.
            for repeat in range(12):
                order = ids.copy()
                rng.shuffle(order)
                for ident in order:
                    for _ in range(5):
                        models[ident](x)
                    paddle.device.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(30):
                        models[ident](x)
                    paddle.device.cuda.synchronize()
                    ms = (time.perf_counter() - start) * 1000 / 30
                    records.append(dict(size=size, round=repeat, id=ident, ms=ms, order=order))
                print('BENCHMARK', size, repeat, flush=True)
    summary = []
    for size in (400, 480):
        baseline = {r['round']: r['ms'] for r in records if r['id'] == 'B0' and r['size'] == size}
        for ident in ids:
            rows = [r for r in records if r['id'] == ident and r['size'] == size]
            values = [r['ms'] for r in rows]
            ratios = [r['ms'] / baseline[r['round']] for r in rows]
            summary.append(dict(id=ident, size=size, median_ms=float(np.median(values)),
                                p10_ms=float(np.percentile(values,10)), p90_ms=float(np.percentile(values,90)),
                                paired_median_ratio_to_B0=float(np.median(ratios)),
                                paired_min_ratio=float(min(ratios)), paired_max_ratio=float(max(ratios)),
                                params=sum(int(np.prod(p.shape)) for p in models[ident].parameters())))
    result = dict(protocol='one process, FP32, batch1, fixed input; warmup40; 12 randomized paired rounds x30 forwards, warmup5 between models, GPU sync',
                  exclusive_gpu_training=True, exclusive_machine=False,
                  limitation='Desktop applications remain running; CPU saturation previously observed. This is a conditional paired measurement, not proof of deployment speed.',
                  before=before, after=snapshot(), raw=records, summary=summary, test_used=False)
    (ROOT / 'results/paired_latency.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
