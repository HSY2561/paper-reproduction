"""Independent gradient and compatibility audit of official loss components."""
from pathlib import Path
import json
import os
import sys
import traceback
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src/PaddleSeg-2.10.0'))
import numpy as np
import paddle
from paddleseg.cvlibs import Config,SegBuilder


def trial(loss):
    paddle.seed(42)
    logits=paddle.randn([1,2,128,128],dtype='float32');logits.stop_gradient=False
    labels=paddle.zeros([1,128,128],dtype='int64')
    labels[:,30:50,20:100]=1;labels[:,:,63:66]=1;labels[:,:4,:4]=255
    before=labels.clone()
    value=loss(logits,labels)
    if isinstance(value,list):value=sum(value)
    finite=bool(paddle.isfinite(value).all().item());value.backward();grad=logits.grad
    valid=finite and grad is not None and bool(paddle.isfinite(grad).all().item()) and float(paddle.abs(grad).sum().item())>0
    result=dict(type=loss.__class__.__name__,value=float(value.item()),finite=finite,gradient_valid=valid,
        gradient_abs_sum=float(paddle.abs(grad).sum().item()) if grad is not None else None,
        labels_changed=int((labels!=before).sum().item()))
    if not valid:raise RuntimeError('Disconnected/zero/nonfinite logit gradient: '+str(result))
    return result


def main():
    paddle.set_device(os.environ.get('LOSS_PREFLIGHT_DEVICE','gpu:0'));rows=[]
    jobs=json.loads((ROOT/'provenance/jobs.json').read_text(encoding='utf-8'))
    for job in jobs:
        row=dict(id=job['id'],role=job['role'],components=[])
        try:
            layer=SegBuilder(Config(str(ROOT/'configs'/(job['id']+'.yml')))).loss['types'][0]
            components=layer.losses if layer.__class__.__name__=='MixedLoss' else [layer]
            for loss in components:row['components'].append(trial(loss))
            row['combined']=trial(layer)
            if any(loss.__class__.__name__=='DualTaskLoss' for loss in components):
                repeated=trial(layer)
                row['seed42_repeat']=repeated
                if abs(row['combined']['value']-repeated['value'])>1e-7 or abs(row['combined']['gradient_abs_sum']-repeated['gradient_abs_sum'])>1e-5:raise RuntimeError('Seeded DualTask repeat mismatch')
            row['pass']=True
        except Exception as e:
            row.update({'pass':False,'error':repr(e),'traceback':traceback.format_exc()})
        rows.append(row)
        print('GRADIENT_PREFLIGHT',row['id'],row['pass'],flush=True)
    (ROOT/'results'/os.environ.get('LOSS_PREFLIGHT_OUTPUT','gradient_preflight.json')).write_text(json.dumps(dict(rows=rows,test_used=False,device=paddle.device.get_device()),indent=2),encoding='utf-8')
    print('PREFLIGHT_DONE',sum(r['pass'] for r in rows),'/',len(rows),flush=True)


if __name__=='__main__':main()
