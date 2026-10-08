import json, os, csv
import cv2, numpy as np, paddle
from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.core import infer
from paddleseg.utils import utils

cfg = Config(r'C:\\local_wave3\\configs\\C4_inputcrop.yml')
builder = SegBuilder(cfg)
model = builder.model
ck = r'C:\\local_wave3\\checkpoints\\C4_seed43\\model.pdparams'
utils.load_entire_model(model, ck)
model.eval()
ds = builder.val_dataset
loader = paddle.io.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, return_list=True)

def boundary(mask):
    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    er = cv2.erode(m, np.ones((3,3), np.uint8), iterations=1)
    return (m - er).astype(np.uint8)

def matched(a, b, radius=2):
    if not a.any():
        return 0
    d = cv2.distanceTransform((1 - b.astype(np.uint8)), cv2.DIST_L2, 3)
    return int(np.logical_and(a > 0, d <= radius + 1e-6).sum())

rows=[]
cm=np.zeros((2,2), dtype=np.int64)
empty=0
with paddle.no_grad():
    for i, data in enumerate(loader):
        pred, _ = infer.inference(model, data['img'], trans_info=data['trans_info'])
        p=np.asarray(pred).squeeze().astype(np.uint8)
        g=np.asarray(data['label']).squeeze().astype(np.uint8)
        cm += np.bincount((g.ravel()*2+p.ravel()).astype(np.int64), minlength=4).reshape(2,2)
        pb, gb = boundary(p), boundary(g)
        tp=int(((p==1)&(g==1)).sum()); fp=int(((p==1)&(g==0)).sum()); fn=int(((p==0)&(g==1)).sum())
        bp=matched(pb,gb); bg=matched(gb,pb)
        bp_n=int(pb.sum()); bg_n=int(gb.sum())
        if p.sum()==0: empty+=1
        path=ds.file_list[i][0]
        rows.append(dict(image=os.path.basename(path), tp=tp, fp=fp, fn=fn,
                         crack_iou=tp/(tp+fp+fn) if tp+fp+fn else 1.0,
                         precision=tp/(tp+fp) if tp+fp else 0.0,
                         recall=tp/(tp+fn) if tp+fn else 0.0,
                         boundary_tp_pred=bp, boundary_tp_gt=bg,
                         boundary_pred=bp_n, boundary_gt=bg_n,
                         boundary_f1=(2*(bp+bg)/(bp_n+bg_n+bp+bg)) if (bp_n+bg_n+bp+bg) else 1.0))
inter=np.diag(cm).astype(float)
den=cm.sum(1)+cm.sum(0)-inter
iou=inter/np.maximum(den,1)
tp=cm[1,1]; fp=cm[0,1]; fn=cm[1,0]
bp=sum(r['boundary_tp_pred'] for r in rows); bg=sum(r['boundary_tp_gt'] for r in rows)
bpn=sum(r['boundary_pred'] for r in rows); bgn=sum(r['boundary_gt'] for r in rows)
out=dict(n=len(rows), confusion_matrix=cm.tolist(), mIoU=float(iou.mean()), crackIoU=float(iou[1]),
         precision=float(tp/(tp+fp)), recall=float(tp/(tp+fn)), f1=float(2*tp/(2*tp+fp+fn)),
         boundary_counts=[bp,bg,bpn,bgn], boundary_f1=float(2*(bp+bg)/(bpn+bgn+bp+bg)),
         empty_predictions=empty, test_used=False)
root=r'C:\\local_wave3\\results'
with open(root+'\\\\C4_seed43_val_per_image.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
with open(root+'\\\\C4_seed43_validation_metrics.json','w',encoding='utf-8') as f:
    json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
