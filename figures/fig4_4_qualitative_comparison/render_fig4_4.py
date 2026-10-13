"""Render a four-scenario qualitative comparison from real Crack500 validation data."""
from pathlib import Path
import sys, os, json, gc, hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import paddle
from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.core.infer import inference
from paddleseg.transforms import Compose, Normalize

ROOT=Path(r'D:\论文复现\experiments\retrain_wave4_20261007')
SRC=ROOT/'src/PaddleSeg-2.10.0'; sys.path.insert(0,str(SRC)); os.environ['PYTHONPATH']=str(SRC)
DATA=Path(r'C:\Users\user\algolab_data'); OUT=Path(__file__).resolve().parent; OUT.mkdir(parents=True,exist_ok=True)
names={'Fine crack':'20160407_165001_1_1081.jpg','Shadow / low contrast':'20160307_162438_1921_1081.jpg','Dense texture':'20160302_155857_641_721.jpg','Wide crack':'20160328_153829_1_1.jpg'}
val_pairs={}
for line in (DATA/'val.txt').read_text().splitlines():
    a,b=line.split(); val_pairs[Path(a).name]=(DATA/a,DATA/b)
for n in names.values():
    if n not in val_pairs: raise FileNotFoundError(n)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_model(cfg_name, ckpt):
    cfg=Config(str(ROOT/'configs'/cfg_name)); cfg.dic['model']['backbone']['pretrained']=None
    model=SegBuilder(cfg).model; model.set_state_dict(paddle.load(str(ckpt))); model.eval(); return model
def predict(model, image_path):
    sample=Compose([Normalize()])({'img':str(image_path),'trans_info':[]})
    with paddle.no_grad():
        pred,_=inference(model,paddle.to_tensor(sample['img'][None]),trans_info=sample.get('trans_info',[]))
    return pred.numpy().squeeze().astype(np.uint8)==1
def iou(p,g):
    inter=np.logical_and(p,g).sum(); union=np.logical_or(p,g).sum(); return float(inter/union) if union else 1.0

ckpts={'SegFormer-B0':('public_segformer_b0.yml',ROOT/'checkpoints/segformer_b0/model.pdparams'),'UAFM':('segformer_uafm_crack500.yml',ROOT/'checkpoints/segformer_uafm_crack500/model.pdparams'),'PPContext':('segformer_ppcontext_crack500.yml',ROOT/'training/segformer_ppcontext_crack500_compat/ema_best_model/ema_model.pdparams')}
preds={k:{} for k in ckpts}; hashes={}; paddle.set_device('gpu')
for model_name,(cfg,ckpt) in ckpts.items():
    hashes[model_name]={'config':str(ROOT/'configs'/cfg),'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt)}
    model=load_model(cfg,ckpt)
    for category,n in names.items(): preds[model_name][category]=predict(model,val_pairs[n][0])
    del model; gc.collect(); paddle.device.cuda.empty_cache()

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':10})
fig,axes=plt.subplots(4,5,figsize=(13.5,9.0)); columns=['Input','Ground truth','SegFormer-B0','UAFM','PPContext']
for ax,title in zip(axes[0],columns): ax.set_title(title,pad=9,fontsize=12)
records=[]
for row,(category,n) in enumerate(names.items()):
    ip,gp=val_pairs[n]; rgb=np.array(Image.open(ip).convert('RGB')); raw=np.array(Image.open(gp)); gt=raw==1
    rowrec={'category':category,'image':n,'image_path':str(ip),'label_path':str(gp),'image_sha256':sha(ip),'label_sha256':sha(gp),'shape':list(rgb.shape),'label_values':np.unique(raw).tolist(),'models':{}}
    axes[row,0].imshow(rgb)
    for col in range(5):
        axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
        for sp in axes[row,col].spines.values(): sp.set_visible(False)
    axes[row,0].set_ylabel(category,fontsize=10); axes[row,1].imshow(gt,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
    for col,model_name in enumerate(['SegFormer-B0','UAFM','PPContext'],start=2):
        p=preds[model_name][category]; value=iou(p,gt); axes[row,col].imshow(p,cmap='gray',vmin=0,vmax=1,interpolation='nearest'); axes[row,col].set_xlabel(f'IoU {value:.3f}',fontsize=9,labelpad=3)
        rowrec['models'][model_name]={'crack_iou':value,'tp':int((p&gt).sum()),'fp':int((p&~gt).sum()),'fn':int((~p&gt).sum())}
    records.append(rowrec)
fig.subplots_adjust(left=.095,right=.995,top=.945,bottom=.06,hspace=.24,wspace=.035)
for ext in ['pdf','png','svg']: fig.savefig(OUT/f'Fig_4_4_Crack500_qualitative_comparison.{ext}',dpi=300,facecolor='white')
metadata={'figure':'Fig. 4-4','dataset':'Crack500 validation','protocol':'same image, label, Normalize-only official inference, FP32, argmax, no TTA or postprocessing','columns':columns,'categories':list(names.keys()),'checkpoints':hashes,'test_accessed':False,'samples':records}
(OUT/'fig4_4_source_manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(metadata,ensure_ascii=False,indent=2))
