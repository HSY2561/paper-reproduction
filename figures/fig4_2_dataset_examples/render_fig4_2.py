"""Reproduce a data-derived paper figure; official SegFormer inference only."""
import os, sys, csv, json, hashlib
from pathlib import Path
ROOT = Path(r'D:\论文复现\experiments\retrain_wave4_20261007')
SRC = ROOT / 'src/PaddleSeg-2.10.0'
sys.path.insert(0, str(SRC))
os.environ['PYTHONPATH'] = str(SRC)
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paddle
from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.core.infer import inference
from paddleseg.transforms import Compose, Normalize

OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True, parents=True)
DATA = Path(r'C:\Users\user\algolab_data')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ckpt = ROOT / 'checkpoints/segformer_b0/model.pdparams'
selection = json.loads((ckpt.parent/'selection.json').read_text(encoding='utf-8-sig'))
assert sha(ckpt) == selection['sha256']
rows = list(csv.DictReader((ROOT/'results/segformer_b0_val_boundary_per_image.csv').open(encoding='utf-8-sig')))
rows.sort(key=lambda r: (float(r['crack_iou']), r['image']))
pairs = {}
for line in (DATA/'val.txt').read_text().splitlines():
    a,b = line.split()
    pairs[Path(a).name] = (DATA/a, DATA/b)
chosen=[]; used=set()
for q in [0.2,0.4,0.6,0.8]:
    target=round(q*(len(rows)-1))
    for idx in sorted(range(len(rows)), key=lambda j: abs(j-target)):
        r=rows[idx]; source='_'.join(r['image'].split('_')[:2])
        if source not in used:
            used.add(source); chosen.append((q,r)); break
paddle.set_device('gpu')
cfg=Config(str(ROOT/'configs/public_segformer_b0.yml'))
cfg.dic['model']['backbone']['pretrained']=None
model=SegBuilder(cfg).model
model.set_state_dict(paddle.load(str(ckpt))); model.eval()
transform=Compose([Normalize()])
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':11})
fig, axes=plt.subplots(4,4,figsize=(12.6,8.5))
for ax, title in zip(axes[0],['(a) Original image','(b) Ground-truth label','(c) SegFormer-B0 prediction','(d) Boundary overlay']):
    ax.set_title(title,pad=10,fontsize=12)
manifest=[]
for i,(q,r) in enumerate(chosen):
    ip,gp=pairs[r['image']]; before={str(ip):sha(ip),str(gp):sha(gp)}
    rgb=np.array(Image.open(ip).convert('RGB')); raw=np.array(Image.open(gp))
    assert set(np.unique(raw)).issubset({0,1,255}), np.unique(raw)
    gt=raw==1
    assert gt.any(), 'Expected crack foreground label 1'
    sample=transform({'img':str(ip),'trans_info':[]})
    with paddle.no_grad():
        pred,_=inference(model,paddle.to_tensor(sample['img'][None]),trans_info=sample.get('trans_info',[]))
    p=pred.numpy().squeeze()==1
    assert p.shape==gt.shape
    tp=int((p&gt).sum()); fp=int((p&~gt).sum()); fn=int((~p&gt).sum())
    actual=tp/(tp+fp+fn)
    assert abs(actual-float(r['crack_iou'])) < 1e-5, (r['image'],actual,r['crack_iou'])
    def edge(m): return m & ~cv2.erode(m.astype('uint8'),np.ones((3,3),np.uint8)).astype(bool)
    bg,bp=edge(gt),edge(p)
    overlay=rgb.copy(); overlay[bg]=[0,220,255]; overlay[bp]=[255,0,180]; overlay[bg&bp]=[255,255,255]
    d=OUT/'panels'/f'row_{i+1}'; d.mkdir(parents=True,exist_ok=True)
    for name,arr in [('original',rgb),('label',(gt*255).astype('uint8')),('prediction',(p*255).astype('uint8')),('boundary_overlay',overlay)]:
        Image.fromarray(arr).save(d/(name+'.png'))
    for j,arr in enumerate([rgb,gt,p,overlay]):
        axes[i,j].imshow(arr,cmap='gray' if j in [1,2] else None,vmin=0 if j in [1,2] else None,vmax=1 if j in [1,2] else None,interpolation='nearest')
        axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
        for spine in axes[i,j].spines.values(): spine.set_visible(False)
    axes[i,0].set_ylabel(f'Example {i+1}',fontsize=11)
    axes[i,2].set_xlabel(f'Crack IoU = {actual:.3f}',fontsize=10,labelpad=3)
    assert all(sha(k)==v for k,v in before.items())
    manifest.append({'row':i+1,'selection_target_quantile':q,'source_image':str(ip),'source_label':str(gp),'source_sha256':before,'image':r['image'],'crack_iou':actual,'tp':tp,'fp':fp,'fn':fn,'original_shape':list(rgb.shape),'label_values':np.unique(raw).tolist(),'exported_panels':str(d)})
fig.legend(handles=[Line2D([0],[0],color='#00dcff',lw=3,label='Ground-truth boundary'),Line2D([0],[0],color='#ff00b4',lw=3,label='Predicted boundary'),Line2D([0],[0],color='#aaaaaa',lw=3,label='Coincident boundaries: white')],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(0.52,0.009))
fig.subplots_adjust(left=.05,right=.995,top=.965,bottom=.055,hspace=.14,wspace=.04)
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'Fig_4_2_Crack500_examples.{ext}',dpi=300,facecolor='white')
fig.savefig(OUT/'preview.png',dpi=110,facecolor='white')
record={'dataset':'Crack500 validation','model':'SegFormer-B0','checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt),'selected_iteration':11000,'config':str(ROOT/'configs/public_segformer_b0.yml'),'source_code':str(SRC),'paddle_version':paddle.__version__,'selection_rule':'Nearest image IoU ranks to quantiles .2,.4,.6,.8; distinct filename source prefixes; not class-balanced or representative prevalence. No test data accessed.','inference':'Official inference, FP32, Normalize only, argmax, no TTA or postprocessing','boundary':'mask minus 3x3 erosion; exact pixel overlap shown white; no matching tolerance in display','source_hashes_verified_unchanged':True,'samples':manifest}
(OUT/'source_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))
