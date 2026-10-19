"""Independent gradient and compatibility audit of official loss components."""
from pathlib import Path
import json
import sys
import traceback
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src/PaddleSeg-2.10.0'))
import numpy as np
import paddle
from paddleseg.cvlibs import Config,SegBuilder


def main():
    paddle.set_device('gpu:0');rows=[]
    jobs=json.loads((ROOT/'provenance/jobs.json').read_text(encoding='utf-8'))
    for job in jobs:
        row=dict(id=job['id'],role=job['role'],components=[])
        try:
            layer=SegBuilder(Config(str(ROOT/'configs'/(job['id']+'.yml')))).loss['types'][0]
            components=layer.losses if layer.__class__.__name__=='MixedLoss' else [layer]
            for i,loss in enumerate(components):
                paddle.seed(42)
                logits=paddle.randn([1,2,128,128],dtype='float32');logits.stop_gradient=False
                labels=paddle.zeros([1,128,128],dtype='int64')
                labels[:,30:50,20:100]=1;labels[:,:,63:66]=1;labels[:,:4,:4]=255
                value=loss(logits,labels.clone())
                finite=bool(paddle.isfinite(value).all().item())
                value.backward()
                grad=logits.grad
                valid=finite and grad is not None and bool(paddle.isfinite(grad).all().item()) and float(paddle.abs(grad).sum().item())>0
                row['components'].append(dict(type=loss.__class__.__name__,value=float(value.item()),finite=finite,gradient_valid=valid))
                if not valid:raise RuntimeError('Disconnected/zero/nonfinite logit gradient')
            row['pass']=True
        except Exception as e:
            row.update({'pass':False,'error':repr(e),'traceback':traceback.format_exc()})
        rows.append(row)
        print('GRADIENT_PREFLIGHT',row['id'],row['pass'],flush=True)
    (ROOT/'results/gradient_preflight.json').write_text(json.dumps(dict(rows=rows,test_used=False,device=paddle.device.cuda.get_device_name()),indent=2),encoding='utf-8')
    print('PREFLIGHT_DONE',sum(r['pass'] for r in rows),'/',len(rows),flush=True)


if __name__=='__main__':main()
