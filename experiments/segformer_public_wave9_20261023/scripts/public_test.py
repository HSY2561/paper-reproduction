import argparse, csv, hashlib, json, os, time
import cv2, numpy as np, paddle
from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.core import infer
from paddleseg.utils import utils

def boundary(mask):
    m=(mask.astype(np.uint8)>0).astype(np.uint8)
    return m-cv2.erode(m,np.ones((3,3),np.uint8),iterations=1)

def match(a,b):
    if not np.any(a): return 0
    d=cv2.distanceTransform((1-b.astype(np.uint8)),cv2.DIST_L2,3)
    return int(np.logical_and(a>0,d<=2.000001).sum())

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''): h.update(x)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--model',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    cfg=Config(args.config); builder=SegBuilder(cfg); model=builder.model
    utils.load_entire_model(model,args.checkpoint); model.eval(); ds=builder.val_dataset
    loader=paddle.io.DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,return_list=True)
    cm=np.zeros((2,2),np.int64); bp=bg=bpn=bgn=empty=0; per=[]; t0=time.perf_counter()
    with paddle.no_grad():
      for i,data in enumerate(loader):
        pred,_=infer.inference(model,data['img'],trans_info=data['trans_info'])
        p=np.asarray(pred).squeeze().astype(np.uint8); g=np.asarray(data['label']).squeeze().astype(np.uint8)
        cm += np.bincount((g.ravel()*2+p.ravel()).astype(np.int64),minlength=4).reshape(2,2)
        pb,gb=boundary(p),boundary(g); a=match(pb,gb); b=match(gb,pb); bp+=a; bg+=b; bpn+=int(pb.sum()); bgn+=int(gb.sum()); empty+=int(p.sum()==0)
        tp=int(np.logical_and(p==1,g==1).sum()); fp=int(np.logical_and(p==1,g==0).sum()); fn=int(np.logical_and(p==0,g==1).sum()); inter=tp; union=tp+fp+fn
        per.append({'index':i,'tp':tp,'fp':fp,'fn':fn,'iou':float(inter/max(union,1)),'boundary_pred':int(pb.sum()),'boundary_gt':int(gb.sum()),'boundary_match_pred':a,'boundary_match_gt':b})
    elapsed=time.perf_counter()-t0
    inter=np.diag(cm).astype(float); den=cm.sum(1)+cm.sum(0)-inter; iou=inter/np.maximum(den,1); tp=cm[1,1]; fp=cm[0,1]; fn=cm[1,0]; pr=float(tp/max(tp+fp,1)); rc=float(tp/max(tp+fn,1)); f1=float(2*tp/max(2*tp+fp+fn,1)); bpr=float(bp/max(bpn,1)); brc=float(bg/max(bgn,1)); bf1=float(2*bpr*brc/max(bpr+brc,1e-12))
    out={'model':args.model,'config':os.path.abspath(args.config),'checkpoint':os.path.abspath(args.checkpoint),'checkpoint_sha256':sha(args.checkpoint),'images':len(ds),'confusion_matrix':cm.tolist(),'mIoU':float(iou.mean()),'crackIoU':float(iou[1]),'precision':pr,'recall':rc,'f1':f1,'boundary_counts':[bp,bpn,bg,bgn],'boundary_precision':bpr,'boundary_recall':brc,'boundary_F1':bf1,'empty_predictions':empty,'elapsed_s':elapsed,'ms_per_image':1000*elapsed/max(len(ds),1),'test_used':True}
    with open(args.out,'w',encoding='utf-8') as f: json.dump(out,f,indent=2)
    with open(os.path.splitext(args.out)[0]+'_per_image.csv','w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=per[0].keys()); w.writeheader(); w.writerows(per)
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
