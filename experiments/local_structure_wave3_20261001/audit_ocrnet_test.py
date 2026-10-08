import json, os, csv
import cv2, numpy as np, paddle
from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.core import infer
from paddleseg.utils import utils
cfg = Config(r'C:\local_wave3\configs\public_ocrnet_hrnetw18_test.yml'); builder = SegBuilder(cfg)
model=builder.model; utils.load_entire_model(model,r'C:\local_wave3\checkpoints\public_ocrnet_hrnetw18\model.pdparams'); model.eval(); ds=builder.val_dataset
loader=paddle.io.DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,return_list=True)
def bd(m):
 m=(m.astype(np.uint8)>0).astype(np.uint8); return m-cv2.erode(m,np.ones((3,3),np.uint8),iterations=1)
def mt(a,b):
 if not a.any(): return 0
 d=cv2.distanceTransform(1-b.astype(np.uint8),cv2.DIST_L2,3); return int(np.logical_and(a>0,d<=2.000001).sum())
cm=np.zeros((2,2),np.int64); bp=bg=bpn=bgn=empty=0
with paddle.no_grad():
 for i,data in enumerate(loader):
  pred,_=infer.inference(model,data['img'],trans_info=data['trans_info']); p=np.asarray(pred).squeeze().astype(np.uint8); g=np.asarray(data['label']).squeeze().astype(np.uint8)
  cm+=np.bincount((g.ravel()*2+p.ravel()).astype(np.int64),minlength=4).reshape(2,2); pb,gb=bd(p),bd(g); bp+=mt(pb,gb); bg+=mt(gb,pb); bpn+=int(pb.sum()); bgn+=int(gb.sum()); empty+=int(p.sum()==0)
inter=np.diag(cm).astype(float); den=cm.sum(1)+cm.sum(0)-inter; iou=inter/np.maximum(den,1); tp=cm[1,1]; fp=cm[0,1]; fn=cm[1,0]; pr=bp/bpn; rc=bg/bgn
out={'images':len(ds),'confusion_matrix':cm.tolist(),'mIoU':float(iou.mean()),'crackIoU':float(iou[1]),'precision':float(tp/(tp+fp)),'recall':float(tp/(tp+fn)),'f1':float(2*tp/(2*tp+fp+fn)),'boundary_counts_pm_pn_gm_gn':[bp,bpn,bg,bgn],'boundary_precision':pr,'boundary_recall':rc,'boundary_F1':2*pr*rc/(pr+rc),'empty_predictions':empty,'test_used':True}
with open(r'C:\local_wave3\results\public_ocrnet_hrnetw18_test_metrics.json','w') as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))

