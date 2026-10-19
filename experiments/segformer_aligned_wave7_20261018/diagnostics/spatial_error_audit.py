"""Frozen validation prediction audit, never a training or threshold search."""
from pathlib import Path
import os
import sys
import csv
import json
import gc
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'segformer_balance_wave6_20261016'
sys.path.insert(0,str(OLD/'src/PaddleSeg-2.10.0'))
import cv2
import numpy as np
import paddle
from paddleseg.cvlibs import Config,SegBuilder
from paddleseg.core import infer
from paddleseg.utils import utils


def main():
    paddle.set_device('gpu:0')
    output=[]
    summaries={}
    for ident in ['B0','M13_Aligned','M10_GloRe']:
        builder=SegBuilder(Config(str(OLD/'configs'/(ident+'.yml'))))
        model=builder.model
        weights=(ROOT.parent/'retrain_wave4_20261007/checkpoints/segformer_b0/model.pdparams' if ident=='B0' else OLD/'checkpoints'/ident/'model.pdparams')
        utils.load_entire_model(model,str(weights));model.eval()
        dataset=builder.val_dataset
        assert dataset.mode=='val' and len(dataset)==348
        loader=paddle.io.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0,return_list=True)
        cm=np.zeros((2,2),dtype=np.int64)
        rows=[]
        with paddle.no_grad():
            for i,data in enumerate(loader):
                pred,_=infer.inference(model,data['img'],trans_info=data['trans_info'])
                p=np.asarray(pred).squeeze().astype(np.uint8)
                gt=np.asarray(data['label']).squeeze().astype(np.uint8)
                cm+=np.bincount((gt.ravel()*2+p.ravel()).astype(np.int64),minlength=4).reshape(2,2)
                fp=(p==1)&(gt==0);fn=(p==0)&(gt==1)
                distance=cv2.distanceTransform(1-gt,cv2.DIST_L2,cv2.DIST_MASK_PRECISE) if gt.any() else np.full(gt.shape,np.inf)
                padded=np.pad(gt,1)
                inner=cv2.distanceTransform(padded,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
                row=dict(model=ident,image=os.path.basename(dataset.file_list[i][0]),
                         fp_near_2px=int((fp&(distance<=2)).sum()),fp_2_to_5px=int((fp&(distance>2)&(distance<=5)).sum()),
                         fp_far_5px=int((fp&(distance>5)).sum()),fn_gt_inner_distance_le2=int((fn&(inner<=2)).sum()),
                         fn_gt_inner_distance_gt2=int((fn&(inner>2)).sum()),empty_pred=int(not p.any()))
                rows.append(row)
                if i%50==0:print('SPATIAL_AUDIT',ident,i,flush=True)
        expected=json.loads((OLD/'results'/(ident+'_val.json')).read_text())['confusion_matrix']
        if cm.tolist()!=expected:raise RuntimeError('Frozen inference mismatch: '+ident)
        summaries[ident]=dict(confusion_matrix=cm.tolist(),counts={k:sum(r[k] for r in rows) for k in rows[0] if k not in ('model','image')},n=len(rows),original_confusion_reproduced=True)
        output.extend(rows)
        del loader,dataset,model,builder
        gc.collect();paddle.device.cuda.empty_cache()
    with (ROOT/'diagnostics/spatial_error_all348.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=output[0].keys());writer.writeheader();writer.writerows(output)
    result=dict(models=summaries,test_used=False,protocol='official inference argmax and original restoration; exact euclidean distance to GT foreground; FN distance bands describe region interior, not crack width',
                limitations='Far false positives are spatial categories, not automatic shadow/leaf scene labels. No threshold or architecture selection from this audit.')
    (ROOT/'diagnostics/spatial_error_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
