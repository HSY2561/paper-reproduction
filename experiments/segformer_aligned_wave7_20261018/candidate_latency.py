"""Frozen-candidate paired timing. No training, data access or threshold search."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'followup_20261019'
sys.path.insert(0,str(ROOT/'src/PaddleSeg-2.10.0'))
import numpy as np
import paddle
from paddleseg.cvlibs import Config,SegBuilder
from paddleseg.utils import utils
from paired_latency import snapshot


def main():
    frozen=json.loads((OUT/'provenance/frozen_candidates.json').read_text(encoding='utf-8'))
    paddle.set_device('gpu:0');paddle.seed(42)
    models={};checks={};rows=[];before=snapshot()
    for ident,rec in frozen['timing_models'].items():
        path=Path(rec['checkpoint'])
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=rec['sha256']:raise RuntimeError('Frozen SHA changed: '+ident)
        model=SegBuilder(Config(rec['config'])).model
        utils.load_entire_model(model,str(path));model.eval()
        models[ident]=model;checks[ident]=actual
    ids=list(models);rng=random.Random(20261019)
    with paddle.no_grad():
        for size in (400,480):
            x=paddle.randn([1,3,size,size],dtype='float32')
            for m in models.values():
                for _ in range(40):m(x)
            for block in range(3):
                base_order=ids.copy();rng.shuffle(base_order)
                for rotation in range(len(ids)):
                    order=base_order[rotation:]+base_order[:rotation]
                    for ident in order:
                        for _ in range(5):models[ident](x)
                        paddle.device.cuda.synchronize();t=time.perf_counter()
                        for _ in range(30):models[ident](x)
                        paddle.device.cuda.synchronize()
                        rows.append(dict(id=ident,size=size,round=block*len(ids)+rotation,ms=(time.perf_counter()-t)*1000/30,order=order))
                    print('PAIRED',size,block,rotation,flush=True)
    summary=[]
    for size in (400,480):
        b={r['round']:r['ms'] for r in rows if r['id']=='B0' and r['size']==size}
        for ident in ids:
            rs=[r for r in rows if r['id']==ident and r['size']==size]
            values=np.array([r['ms'] for r in rs]);ratios=np.array([r['ms']/b[r['round']] for r in rs])
            spread=float((np.percentile(ratios,75)-np.percentile(ratios,25))/np.median(ratios))
            stable=float(np.percentile(values,90)/np.percentile(values,10))<=1.15 and spread<=.1
            summary.append(dict(id=ident,size=size,median_ms=float(np.median(values)),p10_ms=float(np.percentile(values,10)),p90_ms=float(np.percentile(values,90)),paired_median_ratio_to_B0=float(np.median(ratios)),paired_iqr_relative=spread,stability_pass=stable))
    result=dict(models_sha256=checks,raw=rows,summary=summary,before=before,after=snapshot(),test_used=False,
                protocol='FP32 batch1; 40 warmup, 15 balanced Latin-order paired rounds x30 forwards, 5 transition warmup, GPU synchronization',
                stable_for_all=all(r['stability_pass'] for r in summary),
                preregistered_stability='per-model P90/P10<=1.15 and paired-ratio IQR/median<=0.10; before observing these timings',
                limitation='Other desktop applications remain active; no GPU training concurrently. Failure of stability gate means no acceleration claim.')
    (OUT/'results/candidate_latency.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
