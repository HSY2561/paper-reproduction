"""Render a five-scenario qualitative ablation comparison from real Crack500 validation data."""
from pathlib import Path
import sys, os, json, gc, hashlib, csv
ROOT=Path(r'D:\论文复现\experiments\retrain_wave4_20261007')
SRC=ROOT/'src/PaddleSeg-2.10.0'
sys.path.insert(0,str(SRC)); os.environ['PYTHONPATH']=str(SRC)
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import paddle
import paddleseg
assert Path(paddleseg.__file__).resolve().is_relative_to(SRC.resolve()), paddleseg.__file__
from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.core.infer import inference
from paddleseg.transforms import Compose, Normalize

ROOT=Path(r'D:\论文复现\experiments\retrain_wave4_20261007')
SRC=ROOT/'src/PaddleSeg-2.10.0'; sys.path.insert(0,str(SRC)); os.environ['PYTHONPATH']=str(SRC)
DATA=Path(r'C:\Users\user\algolab_data'); OUT=Path(__file__).resolve().parent; OUT.mkdir(parents=True,exist_ok=True)
names={'Fine crack':'20160407_165001_1_1081.jpg','Low contrast':'20160328_152022_641_721.jpg','Shadow / occlusion':'20160307_162438_1921_1081.jpg','Dense texture':'20160302_155857_641_721.jpg','Wide crack':'20160328_153829_1_1.jpg'}
val_pairs={}
for line in (DATA/'val.txt').read_text().splitlines():
    a,b=line.split(); val_pairs[Path(a).name]=(DATA/a,DATA/b)
for n in names.values():
    if n not in val_pairs: raise FileNotFoundError(n)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_before={str(p):sha(p) for n in names.values() for p in val_pairs[n]}
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
expected={'SegFormer-B0':'07f6048e0a6de174a2c3ca1915df4bcd635565a706d978288fba946ef240f2d5','UAFM':'a96a8e62320f0db0bf5d783d6a28ea6bfea24794d6d3bffa44c1b218be0efa47','PPContext':'5c8a9ca5765c59896383b2bef2f4e539ef6356d88d499d3d12557160227fb067'}
# Cross-check source records as well as immutable checkpoint identities.
reference={}
for key,file in [('SegFormer-B0','segformer_b0_val_boundary_per_image.csv'),('PPContext','segformer_ppcontext_compat_val_boundary_per_image.csv')]:
    with (ROOT/'results'/file).open(encoding='utf-8-sig',newline='') as f:
        reference[key]={r['image']:float(r['crack_iou']) for r in csv.DictReader(f)}
for model_name,(cfg,ckpt) in ckpts.items():
    hashes[model_name]={'config':str(ROOT/'configs'/cfg),'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt)}
    assert hashes[model_name]['checkpoint_sha256']==expected[model_name], (model_name,hashes[model_name])
    print('Predicting',model_name,flush=True)
    model=load_model(cfg,ckpt)
    for category,n in names.items(): preds[model_name][category]=predict(model,val_pairs[n][0])
    del model; gc.collect(); paddle.device.cuda.empty_cache()

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':10})
fig,axes=plt.subplots(5,5,figsize=(13.5,9.0)); columns=['Input','Ground truth','SegFormer-B0','UAFM','PPContext']
for ax,title in zip(axes[0],columns): ax.set_title(title,pad=9,fontsize=12)
records=[]
for row,(category,n) in enumerate(names.items()):
    ip,gp=val_pairs[n]; rgb=np.array(Image.open(ip).convert('RGB')); raw=np.array(Image.open(gp)); gt=raw==1
    assert set(np.unique(raw))<= {0,1}
    panel=OUT/'panels'/Path(n).stem; panel.mkdir(parents=True,exist_ok=True)
    Image.fromarray(rgb).save(panel/'input.png'); Image.fromarray(gt.astype(np.uint8)*255).save(panel/'ground_truth.png')
    rowrec={'category':category,'image':n,'image_path':str(ip),'label_path':str(gp),'image_sha256':sha(ip),'label_sha256':sha(gp),'shape':list(rgb.shape),'label_values':np.unique(raw).tolist(),'models':{}}
    axes[row,0].imshow(rgb)
    for col in range(5):
        axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
        for sp in axes[row,col].spines.values(): sp.set_visible(False)
    axes[row,0].set_ylabel(category,fontsize=10); axes[row,1].imshow(gt,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
    for col,model_name in enumerate(['SegFormer-B0','UAFM','PPContext'],start=2):
        p=preds[model_name][category]; value=iou(p,gt); axes[row,col].imshow(p,cmap='gray',vmin=0,vmax=1,interpolation='nearest'); axes[row,col].set_xlabel(f'IoU {value:.3f}',fontsize=9,labelpad=3)
        assert p.shape==gt.shape
        if model_name in reference:
            assert abs(value-reference[model_name][n])<1e-5, (model_name,n,value,reference[model_name][n])
        pred_path=panel/(model_name+'.png'); Image.fromarray(p.astype(np.uint8)*255).save(pred_path)
        rowrec['models'][model_name]={'crack_iou':value,'tp':int((p & gt).sum()),'fp':int((p & ~gt).sum()),'fn':int((~p & gt).sum()),'mask':str(pred_path),'mask_sha256':sha(pred_path)}
    records.append(rowrec)
fig.subplots_adjust(left=.12,right=.995,top=.945,bottom=.055,hspace=.25,wspace=.035)
for ext in ['pdf','png','svg']: fig.savefig(OUT/f'Fig_5_Qualitative_Crack500_ablation.{ext}',dpi=300,facecolor='white')
metadata={'figure':'Fig. 5','dataset':'Crack500 validation','protocol':'same image, label, Normalize-only official inference, FP32, argmax, no TTA or postprocessing','columns':columns,'categories':list(names.keys()),'category_note':'Shadow / occlusion denotes a dark illumination or occlusion region in the selected validation image; it is not a claim that the full dataset has a formal shadow label.','checkpoints':hashes,'test_accessed':False,'samples':records}
assert all(sha(p)==h for p,h in source_before.items()), 'Source data changed'
metadata['source_hashes_unchanged']=True
metadata['paddleseg_source']=paddleseg.__file__
(OUT/'fig5_source_manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(metadata,ensure_ascii=False,indent=2))
