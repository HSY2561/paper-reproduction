"""Read-only aggregation of existing experiment evidence; no training or inference."""
from __future__ import annotations
import base64
import csv
import datetime as dt
import hashlib
import html
import io
import json
from pathlib import Path
import re
import shutil
import statistics as st
import zipfile

ROOT = Path(r'D:\论文复现')
OUT = Path(__file__).resolve().parent
EX = ROOT / 'experiments'
W4 = EX / 'retrain_wave4_20261007'
W5 = EX / 'retrain_wave5_20261012'
W6 = EX / 'segformer_balance_wave6_20261016'
W7 = EX / 'segformer_aligned_wave7_20261018'
FU = W7 / 'followup_20261019'
W8 = EX / 'segformer_loss_wave8_20261019'
TZ = dt.timezone(dt.timedelta(hours=8))
NOW = dt.datetime.now(TZ)
STAMP = NOW.strftime('%Y%m%d_%H%M%S')
for folder in ('refine-logs', 'tables', 'figures', 'provenance/snapshots'):
    (OUT / folder).mkdir(parents=True, exist_ok=True)
SOURCES = {}
METRICS = ('precision', 'recall', 'f1', 'crackIoU', 'mIoU', 'boundary_f1')
CN = dict(precision='Precision', recall='Recall', f1='F1', crackIoU='裂缝IoU', mIoU='mIoU', boundary_f1='边界F1')
ALIASES = {'precision': ('precision', 'Precision'), 'recall': ('recall', 'Recall'),
           'f1': ('f1', 'F1'), 'crackIoU': ('crackIoU', 'crack_IoU'), 'mIoU': ('mIoU',),
           'boundary_f1': ('boundary_f1', 'boundary_F1')}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def read(p, optional=False):
    p = Path(p)
    if not p.exists():
        if optional:
            return None
        raise FileNotFoundError(p)
    data = p.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    rel = str(p.relative_to(ROOT)).replace('\\', '/')
    dest = OUT / 'provenance/snapshots' / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    SOURCES[str(p)] = dict(path=str(p), sha256=digest, bytes=len(data), snapshot=str(dest),
                           modified=dt.datetime.fromtimestamp(p.stat().st_mtime, TZ).isoformat())
    return json.loads(data.decode('utf-8-sig'))

def text_source(p):
    p = Path(p)
    if not p.exists():
        return ''
    data=p.read_bytes()
    SOURCES[str(p)] = dict(path=str(p), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                           snapshot='', modified=dt.datetime.fromtimestamp(p.stat().st_mtime, TZ).isoformat())
    return data.decode('utf-8-sig', errors='replace')

def norm(d):
    result={}
    for key, names in ALIASES.items():
        for name in names:
            if name in d and isinstance(d[name], (int,float)):
                result[key]=float(d[name]); break
    if 'confusion_matrix' in d:
        a=d['confusion_matrix']; tn,fp,fn,tp=a[0][0],a[0][1],a[1][0],a[1][1]
        result.update(tn=tn,fp=fp,fn=fn,tp=tp)
        computed=dict(precision=tp/(tp+fp),recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),
                      crackIoU=tp/(tp+fp+fn),mIoU=(tp/(tp+fp+fn)+tn/(tn+fp+fn))/2)
        for k,v in computed.items(): result.setdefault(k,v)
    return result

def pct(v): return '—' if v is None else f'{100*v:.4f}'
def delta(a,b,k): return a[k]-b[k] if k in a and k in b else None
def pp(v): return '—' if v is None else f'{v*100:+.4f}'
def mparam(v): return '—' if v is None else f'{v/1e6:.6f}'
def status_cn(v):
    return {'complete':'完成','smoke_failed':'反向smoke失败','compatibility_deferred':'兼容暂缓',
            'gradient_preflight_failed':'梯度预检失败','training':'正式训练中','pending':'待运行',
            'smoke_running':'smoke中','smoke_passed':'smoke通过，未完成正式训练',
            'needs_review':'待核验','train_failed':'正式训练失败','verifying':'独立核验中'}.get(v,v)

def table(headers, rows):
    def cell(x): return str(x).replace('|','／').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(cell,r))+' |' for r in rows])+'\n'

def csvout(name, rows, fields=None):
    p=OUT/'tables'/name
    if fields is None:
        fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');wr.writeheader();wr.writerows(rows)
    return p

def jsonout(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return p

comp6=read(W6/'results/comparison.json')
comp7=read(W7/'results/comparison.json')
seed_data=read(FU/'results/three_seed_summary.json')
comp8=read(W8/'results/comparison.json')
protocol8=read(W8/'provenance/protocol.json')
heartbeat8=read(W8/'results/heartbeat.json')
recovery=read(W8/'dice_compat_recovery_20261020/results/heartbeat.json',True)
latency=read(FU/'results/candidate_latency.json')
baseline=norm(read(W6/'results/B0_val.json'))
baseline['params']=read(W6/'results/B0_profile.json')['params_total']
parent6=norm(read(W6/'results/W128_val.json'))
parent6['params']=read(W6/'results/W128_profile.json')['params_total']
a0=norm(read(W6/'results/M13_Aligned_val.json'))
a0['params']=read(W6/'results/M13_Aligned_profile.json')['params_total']

rows6=[]; rows7=[]; rows8=[]; AUDITS=[]
def normalize_rows(comp, folder, phase):
    result=[]
    for item in comp['rows']:
        r=dict(item)
        s=read(folder/'results'/f"{r['id']}_status.json",True)
        if s: r['status']=s['status']
        r.update(phase=phase,seed=42,split='Crack500 val',n=348)
        if r['status']=='complete':
            metrics=read(folder/'results'/f"{r['id']}_val.json")
            r.update(norm(metrics))
            prof=read(folder/'results'/f"{r['id']}_profile.json")
            r['params']=prof['params_total']
            r['empty_predictions']=metrics.get('empty_predictions')
            r['source']=str(folder/'results'/f"{r['id']}_val.json")
            receipts={}
            for step in ('train','official_val','pixel_boundary','profile'):
                receipt=read(folder/'results'/f"{r['id']}_{step}_exit.json",True)
                receipts[step]=receipt.get('exit_code') if receipt else None
            sel=read(folder/'checkpoints'/r['id']/'selection.json')
            ckpt=folder/'checkpoints'/r['id']/'model.pdparams'
            log=text_source(folder/'logs'/f"{r['id']}_train.log")
            matrix=metrics.get('confusion_matrix')
            errors=[]
            if not all(code==0 for code in receipts.values()): errors.append('missing/nonzero receipt')
            if not (folder/'training'/r['id']/'iter_18000/model.pdparams').exists(): errors.append('missing final checkpoint')
            if 'iter: 18000/18000' not in log: errors.append('missing final iteration in log')
            if not ckpt.exists() or sha(ckpt).lower()!=sel['sha256'].lower(): errors.append('frozen SHA mismatch')
            if abs(metrics['mIoU']-sel['selected']['mIoU'])>1e-6: errors.append('val mismatch')
            if sel['selected']['mIoU']!=max(x['mIoU'] for x in sel['candidates']): errors.append('wrong ordinary/EMA selection')
            if metrics.get('test_used') is not False: errors.append('missing/true test_used')
            if abs(r['f1']-2*r['crackIoU']/(1+r['crackIoU']))>1e-8: errors.append('inconsistent F1/IoU')
            if matrix:
                tn,fp,fn,tp=matrix[0][0],matrix[0][1],matrix[1][0],matrix[1][1]
                if abs(r['precision']-tp/(tp+fp))>1e-8: errors.append('confusion matrix mismatch')
            AUDITS.append(dict(phase=phase,id=r['id'],receipts=receipts,errors=errors,
                               selection_variant=sel['selected']['variant'],selection_iter=sel['selected']['iter'],
                               sha256=sel['sha256'],frozen_checkpoint=str(ckpt)))
            r['selected_variant']=sel['selected']['variant'];r['selected_iter']=sel['selected']['iter'];r['sha256']=sel['sha256']
        result.append(r)
    return result

rows6=normalize_rows(comp6,W6,'wave6')
rows7=normalize_rows(comp7,W7,'wave7')
rows8=normalize_rows(comp8,W8,'wave8')

follow=[]
for row in seed_data['rows']:
    r=dict(row,phase='followup',split='Crack500 val',n=348)
    if r['status']=='complete':
        metrics=read(FU/'results'/f"{r['id']}_val.json");r.update(norm(metrics))
        profile=read(FU/'results'/f"{r['id']}_profile.json");r['params']=profile['params_total']
        receipts={step:read(FU/'results'/f"{r['id']}_{step}_exit.json")['exit_code'] for step in ('train','official_val','pixel_boundary','profile')}
        sel=read(FU/'checkpoints'/r['id']/'selection.json')
        ckpt=FU/'checkpoints'/r['id']/'model.pdparams'
        errors=[]
        if not all(v==0 for v in receipts.values()):errors.append('nonzero receipt')
        if sha(ckpt).lower()!=sel['sha256'].lower():errors.append('SHA mismatch')
        if sel['selected']['mIoU']!=max(c['mIoU'] for c in sel['candidates']):errors.append('wrong selection')
        if abs(r['mIoU']-sel['selected']['mIoU'])>1e-6:errors.append('val mismatch')
        if not (FU/'training'/r['id']/'iter_18000/model.pdparams').exists():errors.append('final checkpoint missing')
        if 'iter: 18000/18000' not in text_source(FU/'logs'/f"{r['id']}_train.log"):errors.append('final log missing')
        AUDITS.append(dict(phase='followup',id=r['id'],receipts=receipts,errors=errors,
                           sha256=sel['sha256'],selection_variant=sel['selected']['variant'],selection_iter=sel['selected']['iter']))
        r['sha256']=sel['sha256'];r['source']=str(FU/'results'/f"{r['id']}_val.json")
    follow.append(r)

all_new=rows6+rows7+follow+rows8
for r in all_new:
    ref=parent6 if r['phase']=='wave6' else a0
    for key in METRICS:
        if key in r:
            r['delta_B0_'+key+'_pp']=100*(r[key]-baseline[key])
            r['delta_parent_'+key+'_pp']=100*(r[key]-ref[key])
            r[key+'_percent']=100*r[key]
    if 'params' in r:r['param_reduction_B0_percent']=100*(1-r['params']/baseline['params'])

def export_rows(rows,name):
    cols=['phase','id','label','source','seed','status','split','n','precision','recall','f1','crackIoU','mIoU','boundary_f1','params',
          *[k+'_percent' for k in METRICS],*[f'delta_B0_{k}_pp' for k in METRICS],*[f'delta_parent_{k}_pp' for k in METRICS],
          'param_reduction_B0_percent','selected_variant','selected_iter','sha256']
    csvout(name,rows,cols)
export_rows(all_new,'全部新实验_原始数值与涨点.csv')
export_rows(rows6,'01_Wave6单模块与宽度.csv')
export_rows(rows7,'02_Wave7_Aligned组合.csv')
export_rows(follow,'03_归因补充与多种子新增运行.csv')
export_rows(rows8,'04_Wave8损失函数.csv')

test=read(W4/'results/baseline_test_summary.json')['models']
costs=read(W4/'results/compute_costs_current.json')['models']
costmap={x['model']:x['params'] for x in costs}
testrows=[]
for x in test:
    raw=read(W4/'results'/('segformer_b0_verified_metrics.json' if x['model']=='SegFormer-B0' else 'ocrnet_hrnetw18_verified_metrics.json'))
    r=dict(model=x['model'],split='Crack500 test',n=raw['images'],params=costmap.get(x['model']),**norm(raw))
    r['sha256']=raw['checkpoint_sha256'];testrows.append(r)
csvout('05_已完成原始test.csv',testrows)

wave4=read(W4/'results/crack500_segformer_ablation_summary.json')
wave5=read(W5/'results/wave5_summary.json')
corrections=read(W6/'provenance/historical_corrections.json')
hist=[]
uafm=next(x for x in wave4['models'] if x['model']=='SegFormer-B0+UAFM_SpAtten')
hist.append(dict(id='UAFM_SpAtten',**{k:uafm.get('val_'+('F1' if k=='f1' else k)) for k in METRICS if k!='boundary_f1'},params=costmap['SegFormer-B0+UAFM_SpAtten'],note='历史汇总四位精度，delta重新计算，未测test'))
ppctx=read(W4/'results/segformer_ppcontext_compat_val_boundary_metrics.json')
hist.append(dict(id='PPContext兼容版',**norm(ppctx),params=costmap['SegFormer-B0+PPContextModule'],note='18k与独立验证存在；正式训练exit未知；旧BF1口径不纳入当前对比'))
for name,prefix,param in [('EMAU','A_EMAU_context',3847266),('DualAttention','B_Dual_context',3976883),('LightHam','C_Ham_decoder',3431138)]:
    variant='ema' if name=='LightHam' else 'ordinary'
    m=read(W5/'results'/f'{prefix}_{variant}_verified.json')
    hist.append(dict(id=name,**norm(m),params=param,note='LightHam按EMA iter10000纠正；与旧汇总不同' if name=='LightHam' else '历史独立验证；原边界口径不同'))
for r in hist:
    for k in METRICS:
        if isinstance(r.get(k),(int,float)):r['delta_B0_'+k+'_pp']=100*(r[k]-baseline[k])
csvout('06_早期SegFormer历史结果及修正.csv',hist)

seedrows=[];means=[];paired=[]
params_by={'B0':baseline['params'],'A0':a0['params'],'A21':3503330,'A10':3656994}
for model,g in seed_data['groups'].items():
    for x in g['seeds']:seedrows.append(dict(model=model,params=params_by[model],**x))
    r=dict(model=model,n=len(g['seeds']),params=params_by[model])
    for k in METRICS:
        values=[s[k] for s in g['seeds']]
        r[k+'_mean']=st.mean(values);r[k+'_sample_std']=st.stdev(values)
        assert abs(r[k+'_mean']-g['mean'][k])<1e-12
        assert abs(r[k+'_sample_std']-g['sample_std'][k])<1e-12
    means.append(r)
means_by={r['model']:r for r in means}
for cand in ('A0','A21','A10'):
    for ref in (('B0',) if cand=='A0' else ('B0','A0')):
        a={x['seed']:x for x in seed_data['groups'][cand]['seeds']};b={x['seed']:x for x in seed_data['groups'][ref]['seeds']}
        for sd in sorted(set(a)&set(b)):
            paired.append(dict(candidate=cand,reference=ref,seed=sd,**{k+'_delta_pp':100*(a[sd][k]-b[sd][k]) for k in METRICS}))
csvout('07_三种子逐次原始结果.csv',seedrows)
csvout('08_三种子均值与样本标准差.csv',means)
csvout('09_同种子成对涨点.csv',paired)

single_afnb=next(r for r in follow if r['id']=='X_AFNB_s42')
single_psa=next(r for r in rows6 if r['id']=='M02_PSA')
joint_psa=next(r for r in rows7 if r['id']=='A21_Polarized')
joint_afnb=next(r for r in rows7 if r['id']=='A10_AFNB')
factorial=[]
for name,single,joint in [('Polarized',single_psa,joint_psa),('AFNB',single_afnb,joint_afnb)]:
    for label,r in [('W128',parent6),('W128 + Aligned',a0),('W128 + '+name,single),('W128 + Aligned + '+name,joint)]:
        factorial.append(dict(family=name,configuration=label,params=r['params'],**{k:r[k] for k in METRICS}))
csvout('10_两模块四行归因.csv',factorial)

# Existing per-image CSV only: no dataset, prediction, test, or inference access.
image_summaries=[];paired_images=[]
selected_images={}
for label,folder,ident in [('B0',W6,'B0'),('W128',W6,'W128'),('A0',W6,'M13_Aligned'),('A21',W7,'A21_Polarized'),('A10',W7,'A10_AFNB'),('L06',W8,'L06_CE_Lovasz')]:
    p=folder/'results'/f'{ident}_val_per_image.csv'
    if not p.exists():continue
    with p.open(encoding='utf-8-sig',newline='') as f: images=list(csv.DictReader(f))
    text_source(p)
    mapping={x['image']:x for x in images};selected_images[label]=mapping
    ious=[float(x['crack_iou']) for x in images]
    image_summaries.append(dict(model=label,n=len(images),macro_IoU=st.mean(ious),empty_predictions=sum(int(x['tp'])+int(x['fp'])==0 for x in images),zero_overlap=sum(int(x['tp'])==0 and int(x['tp'])+int(x['fn'])>0 for x in images),sum_tp=sum(int(x['tp']) for x in images),sum_fp=sum(int(x['fp']) for x in images),sum_fn=sum(int(x['fn']) for x in images)))
if 'B0' in selected_images:
    for cand,images in selected_images.items():
        if cand=='B0':continue
        for ident,x in images.items():
            b=selected_images['B0'][ident]
            def bf(z):
                p=int(z['boundary_pred']);g=int(z['boundary_gt']);mp=int(z['boundary_tp_pred']);mg=int(z['boundary_tp_gt'])
                if not p and not g:return 1.0
                if not p or not g:return 0.0
                precision=mp/p;recall=mg/g
                return 2*precision*recall/(precision+recall) if precision+recall else 0.0
            paired_images.append(dict(model=cand,image=ident,IoU_B0=float(b['crack_iou']),IoU_candidate=float(x['crack_iou']),IoU_delta_pp=100*(float(x['crack_iou'])-float(b['crack_iou'])),BF1_delta_pp=100*(bf(x)-bf(b)),FP_delta=int(x['fp'])-int(b['fp']),FN_delta=int(x['fn'])-int(b['fn']),TP_delta=int(x['tp'])-int(b['tp'])))
csvout('11_逐图诊断摘要.csv',image_summaries)
csvout('12_逐图涨点与误检漏检变化.csv',paired_images)
spatial=read(W7/'diagnostics/spatial_error_summary.json')

# Historical BiSeNet and custom data: kept in a separate appendix, never pooled with the main protocol.
legacy=[]
def add_legacy(label,p,key=None):
    d=read(p,True)
    if d is None:return
    if key:d=d[key]
    r=norm(d)
    if 'mIoU' not in r:return
    legacy.append(dict(model=label,split='历史Crack500 val',**r,source=str(p)))
add_legacy('BiSeNetV2 C0',EX/'local_structure_20260930/results/C0_boundary_metrics.json')
add_legacy('BiSeNetV2 + OCR C1',EX/'local_structure_20260930/results/C1_validation_metrics.json')
add_legacy('BiSeNetV2 + Shape C2',EX/'local_structure_20260930/results/C2_validation_metrics.json')
for key,label in [('C1b','BiSeNetV2 + OCR融合 C1b'),('C2b','BiSeNetV2 + PointRend C2b')]:
    add_legacy(label,EX/'local_structure_wave2_20260930/results/wave2_validation_metrics.json',key)
add_legacy('BiSeNetV2 C4 输入方案（历史）',EX/'local_structure_wave3_20261001/results/C4_validation_metrics.json')
add_legacy('BiSeNetV2 C4 seed43（历史）',EX/'local_structure_wave3_20261001/results/C4_seed43_validation_metrics.json')
add_legacy('BiSeNetV2 C4 seed44（历史）',EX/'local_structure_wave3_20261001/results/C4_seed44_validation_metrics.json')
custom0=read(EX/'local_segformer_wave1_20261006/results/S0_seed42_validation_metrics.json')
custom=[dict(model='自建 SegFormer-B0 S0',split='自建 val 177',**norm(custom0))]
verified=read(W4/'results/report_verified_metrics.json')
for r in verified:
    if r['split']=='custom val':custom.append(dict(model='自建 SegFormer + UAFM',split='自建 val 177',**norm(r)))
legacytest=[dict(model='BiSeNetV2-C4（排除主线）',split='历史Crack500 test',**norm(r)) for r in verified if r['model']=='bisenetv2_c4']
csvout('13_历史BiSeNet结果_排除主线.csv',legacy+legacytest)
csvout('14_历史自建数据结果_不参与Crack500排名.csv',custom)

# Machine-readable evidence checks are separate from the fresh-agent semantic audit.
jsonout('provenance/结果确定性核验.json',dict(generated_at=NOW.isoformat(),runs=AUDITS,checked=len(AUDITS),
    errors=[dict(phase=a['phase'],id=a['id'],errors=a['errors']) for a in AUDITS if a['errors']],
    scope='No inference/training; checks existing checkpoint hashes, receipts, final iteration, ordinary/EMA selection and val arithmetic'))
assert not any(a['errors'] for a in AUDITS), 'Evidence mismatch: resolve before delivering'

# Figures generated with CPU Matplotlib only.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
COLORS={'B0':'#586577','A0':'#267b9e','A21':'#db8b24','A10':'#9570ae'}
FIGS=[]
def savefig(fig,name):
    fig.tight_layout();fig.savefig(OUT/'figures'/f'{name}.png',bbox_inches='tight');fig.savefig(OUT/'figures'/f'{name}.svg',bbox_inches='tight');plt.close(fig);FIGS.append(name)

fig,ax=plt.subplots(figsize=(9.4,5.4))
ax.scatter([r['params']/1e6 for r in rows7 if r['status']=='complete'],[100*r['crackIoU'] for r in rows7 if r['status']=='complete'],c='#afc9d7',s=45,label='Wave7 completed combinations')
for label,r in [('B0',baseline),('A0',a0),('A21',joint_psa),('A10',joint_afnb)]:
    ax.scatter(r['params']/1e6,100*r['crackIoU'],s=90,color=COLORS[label],zorder=3);ax.annotate(label,(r['params']/1e6,100*r['crackIoU']),xytext=(5,5),textcoords='offset points')
ax.set(xlabel='Parameters (M)',ylabel='Crack IoU (%)',title='Crack500 validation: parameter and region-accuracy tradeoff, seed42');ax.legend(loc='lower right');ax.grid(alpha=.15);savefig(fig,'01_parameter_iou')

completed7=[r for r in rows7 if r['status']=='complete'];completed7.sort(key=lambda r:r['crackIoU']-a0['crackIoU'])
fig,ax=plt.subplots(figsize=(9.8,8.2));vals=[100*(r['crackIoU']-a0['crackIoU']) for r in completed7]
ax.barh([r['id'] for r in completed7],vals,color=['#c7655b' if v<0 else '#378a9b' for v in vals]);ax.axvline(.2,color='#db8b24',ls='--',label='IoU part of preregistered gate (+0.2 pp)');ax.axvline(0,color='black',lw=.7)
ax.set(xlabel='Crack IoU change vs A0 (percentage points)',title='Wave7 second-module contribution, seed42');ax.legend(fontsize=8);savefig(fig,'02_combo_iou_changes')

fig,axes=plt.subplots(1,3,figsize=(12.2,4.0))
for ax,k in zip(axes,['crackIoU','f1','mIoU']):
    for model in ('B0','A0','A21','A10'):
        seeds=seed_data['groups'][model]['seeds'];ax.plot([x['seed'] for x in seeds],[100*x[k] for x in seeds],marker='o',label=model,color=COLORS[model])
    ax.set(xticks=[42,43,44],xlabel='Matched random seed',ylabel=f'{CN[k] if k in ("f1","mIoU") else "Crack IoU"} (%)',title=k);ax.grid(alpha=.15)
axes[0].legend(fontsize=8);fig.suptitle('Three independent training seeds, validation only',y=1.04);savefig(fig,'03_matched_seeds')

loss_done=[r for r in rows8 if r['status']=='complete']
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,keys in zip(axes,[['crackIoU','f1'],['precision','recall']]):
    x=np.arange(len(loss_done));w=.35
    for i,k in enumerate(keys):ax.bar(x+(i-.5)*w,[100*(r[k]-a0[k]) for r in loss_done],width=w,label={'crackIoU':'Crack IoU','f1':'F1','precision':'Precision','recall':'Recall'}[k])
    ax.set(xticks=x,xticklabels=[r['id'].split('_')[0] for r in loss_done],ylabel='Change vs A0 + CE (pp)');ax.axhline(0,color='black',lw=.7);ax.legend()
fig.suptitle('Completed loss recipes: region gains and precision/recall cost, seed42');savefig(fig,'04_loss_tradeoffs')

fig,ax=plt.subplots(figsize=(8,4.5));x=np.arange(4);w=.35
for i,size in enumerate((400,480)):
    rr={r['id']:r for r in latency['summary'] if r['size']==size};ax.bar(x+(i-.5)*w,[rr[k]['median_ms'] for k in ('B0','A0','A21','A10')],width=w,label=f'{size} x {size}')
ax.set(xticks=x,xticklabels=['B0','A0','A21','A10'],ylabel='GPU forward latency (ms)',title='Paired timing: stability failed; no acceleration claim');ax.legend();savefig(fig,'05_latency_limitations')

fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for ax,cand in zip(axes,('A21','L06')):
    rr=[r for r in paired_images if r['model']==cand];ax.scatter([r['IoU_delta_pp'] for r in rr],[r['BF1_delta_pp'] for r in rr],s=12,alpha=.5,color='#267b9e');ax.axhline(0,color='#999',lw=.8);ax.axvline(0,color='#999',lw=.8);ax.set(xlabel='Per-image IoU change vs B0 (pp)',ylabel='Per-image boundary F1 change (pp)',title=f'{cand}: all 348 validation images')
savefig(fig,'06_image_level_changes')

def metric_table(rows,label='id',bf=True,delta_ref=None):
    headers=['配置','状态','P%','R%','F1%','裂缝IoU%','mIoU%']+(['BF1%'] if bf else [])+['Params M']
    if delta_ref:headers+=['ΔIoU pp','ΔF1 pp','ΔmIoU pp']
    body=[]
    for r in rows:
        row=[r.get(label,r.get('model','')),status_cn(r.get('status','历史／复用'))]+[pct(r.get(k)) for k in ('precision','recall','f1','crackIoU','mIoU')]
        if bf:row+=[pct(r.get('boundary_f1'))]
        row+=[mparam(r.get('params'))]
        if delta_ref:row +=[pp(delta(r,delta_ref,k)) for k in ('crackIoU','f1','mIoU')]
        body.append(row)
    return table(headers,body)

parts=[]
def add(s=''):parts.append(s.strip()+'\n')
def figure(name,caption):
    add(f'![{caption}]({(OUT/"figures"/(name+".png")).as_posix()})\n\n{caption}')
add('# SegFormer裂缝分割全部实验结果与涨点总结报告')
add(f'结果快照时间：{NOW:%Y年%m月%d日 %H:%M:%S}（北京时间）。本报告供实验审核、同门协作和后续论文写作使用。所有数字由既有JSON/CSV重新读取、按对应母体重新计算，不启动训练或推理，不更改原始图像、标签、运行中队列或筛选门槛。报告是这一时刻的结果快照，运行中的损失实验后续会继续产生新结果。')
add('## 一 目前能确认的结论')
add(f'已经完成Wave6的13种官方单模块及2个宽度控制、Wave7的22种第二模块组合，以及后续9项归因和匹配随机种子训练。本次快照中Wave8的20种损失配置已完成{len(loss_done)}项，其余项目按真实状态列出。以上属于不同配置与训练运行，存在模块复用及多种子复跑，不能合计成几十种独立新算法。')
add('当前最清楚的正结果是：A0（SegFormer-B0，解码器宽度128＋Aligned）和A21（在A0上加入Polarized）在三种子平均区域指标上优于原始B0。A21对B0的平均裂缝IoU提高0.4871个百分点、F1提高0.3437个百分点、mIoU提高0.2527个百分点，参数减少5.7020%；Precision略降而Recall提高，不能描述为所有指标同时上涨。A21第二模块相对A0的平均IoU提升仅0.1907895个百分点，低于预登记0.2个百分点，不能四舍五入放行。A10（AFNB组合）未显示稳定的第二模块增益。')
add('当前不能宣称最终模型、完整精度效率目标通过或领域SOTA。候选的稳定速度条件未通过，且实际配对中位延迟慢于B0；新的结构/损失候选尚未做最终test。已有原始test结果主要是SegFormer-B0和OCRNet，六个原始公开模型尚未齐全。论文应保留负结果、协议差异、验证搜索偏差以及数据来源互斥复核尚待完成这些限制。')
add('## 二 数据和训练评价协议')
add(table(['部分','清单样本数','实际作用','本轮访问情况'],[['Crack500 train',1896,'更新网络权重','所有新候选使用'],['Crack500 val',348,'普通/EMA选择、结构/损失筛选','所有完成候选独立验证'],['Crack500 test',1124,'冻结模型后统一最终评估','历史B0/OCRNet已测；新候选未测'],['自建数据集','历史train1400、val177','早期独立实验','当前Wave6–8不使用']]))
add('当前Crack500清单合计3368个图像—标签样本，其中训练/验证2244个。样本包含裁剪，不能当作3368张相互独立的原始道路照片。只读入口为C:/Users/user/algolab_data；历史数据清单有4488个train/val源文件哈希（图像和标签各计一文件）。本报告只读取已有结果，不重新读取测试图像或标签。自建集历史有精确图像/D4去重处理，原始源帧身份尚未完整建立；其分数不能与Crack500横向排名。')
add('统一主要协议：本机RTX5060 8GB，Paddle3.4/PaddleSeg-2.10.0，FP32，batch4，AdamW，初始学习率6e-5，PolynomialDecay end_lr=0，18000 iter，EMA，每1000 iter验证。训练增强沿用0.75–1.25缩放、400×400裁剪、水平翻转、颜色扰动和Normalize；验证不做这些随机增强。新结构/损失训练从共同ImageNet MiT-B0预训练初始化，并非从训练好的对照模型接着微调。多种子为42/43/44，预算保持一致。')
add('评价：先聚合348张图的混淆矩阵，再计算Precision、Recall、F1和前景IoU；mIoU是背景IoU与裂缝IoU的均值。F1=2×IoU/(1+IoU)，因此F1和裂缝IoU不是两份独立证据。BF1为前景减3×3腐蚀所得内边界，图像外为零，精确欧氏距离≤2px双向匹配，全数据集计数后计算调和平均。逐图宏平均和全局指标单独报告。')
add('选模规则：普通best与EMA best比较精确验证mIoU，选择最高者，复制冻结checkpoint并计算SHA256，随后官方val.py和独立像素/边界评估复核。不同模型不能依据test成绩反向选权重。本报告中Δ均表示绝对百分点（pp），不是相对百分比；例如IoU从67.8%到68.3%为+0.5pp。参数变化百分比则另按相对B0计算。')
add('## 三 阶段完成清单与结果可信度')
inventory=[['早期SegFormer UAFM/PPContext/EMAU/DualAttention/LightHam','5项历史候选','不同早期记录完整性；LightHam已纠正EMA选模','见历史章节，边界不混比'],['Wave6 宽度与单模块',f"{sum(r['status']=='complete' for r in rows6)}/16完成",'13单模块＋2宽度；PSP smoke失败','真实18k、退出、冻结和独立验证'],['Wave7 Aligned＋第二模块',f"{sum(r['status']=='complete' for r in rows7)}/24完成",'22完成；DAPPM失败，PSP暂缓','不把同一模块换接入点计为全新机制'],['归因补充与匹配seed',f"{len([r for r in follow if r['status']=='complete'])}/9完成",'AFNB独立seed42；四模型seed43/44','形成四模型各3seed'],['Wave8 固定A0损失',f'{len(loss_done)}/20完成',f'{len(loss_done)}项正式结果；兼容恢复另列','超参数配置不计独立模型机制'],['最终六个原始公开对照','尚未齐全','B0/OCRNet test已有；PP-LiteSeg val已有','PSPNet/DeepLab/FastSCNN未有完整统一test']]
add(table(['阶段','快照状态','包含内容','解释'],inventory))
add(f'本次确定性核验共检查{len(AUDITS)}个完成运行：训练/官方验证/独立像素边界/profile真实exit0、最终18000日志与checkpoint、冻结SHA、普通/EMA最高选择、精确验证复现和指标一致性；发现错误0。该计数是“运行数”，含控制组和多种子，不代表独立模型数量。核验详情见provenance/结果确定性核验.json。另已尝试独立审阅，但指定审阅代理因使用配额限制未执行完成，因此独立语义审计状态是BLOCKED，不是PASS；详见EXPERIMENT_AUDIT.md。历史缺少真实训练exit的条目不伪造exit0。')
add('## 四 对照与宽度压缩')
controls=[dict(id='B0 原始width256',status='冻结复用',**baseline),*rows6[:2],dict(id='A0 W128 + Aligned',status='冻结复用',**a0)]
add(metric_table(controls,delta_ref=baseline))
add('W128相对原B0减少约7.09%参数，但seed42的区域与边界略降；W64减少约9.31%，区域损失更大，且已有计时不能证明更快。Aligned接在W128后，恢复了部分区域精度，但参数节约主要来自解码宽度压缩，不能全部归因于Aligned。后续组合的直接母体为A0，不是原始width256模型。')
add('## 五 早期SegFormer模块历史结果及选模修正')
add(metric_table([dict(r,status='历史已验证') for r in hist],bf=False,delta_ref=baseline))
add('UAFM旧汇总含错误的缓存delta，本表从绝对指标重新相减：裂缝IoU相对B0实际下降约0.2974pp，不能沿用旧文档中的“增加0.0030”。该行原始汇总仅保留四位小数，精度低于后续逐像素结果。PPContext兼容版完整迭代/独立验证有证据，但原启动器没有真实训练退出JSON，退出码保留未知；兼容性修复不等于性能通过。')
add('LightHam旧汇总错误选了普通权重（mIoU82.3293%）；正确按最高验证mIoU应选EMA iter10000（mIoU82.4242%、裂缝IoU67.1499%、F1 80.3469%）。修正后仍低于原B0，不改变负结果性质。旧波BF1使用默认外延/近似距离，部分更早记录还存在非调和平均公式错误；本表不混入其BF1，不能把重评口径变化解释为模型提升。')
add('## 六 Wave6 全部单模块与宽度实验')
add('本节除B0外均为width128上的单模块，W64是单独压缩控制。每行P/R/F1/IoU/mIoU/BF1均为seed42验证百分数；涨点以原B0为参照。')
add(metric_table(rows6,delta_ref=baseline))
add('### 6.1 相对W128的模块净贡献')
add(table(['模块','ΔP pp','ΔR pp','ΔF1 pp','ΔIoU pp','ΔmIoU pp','ΔBF1 pp'],[[r['id']]+[pp(delta(r,parent6,k)) for k in METRICS] for r in rows6 if r.get('modules')]))
add('观察：Aligned单独在W128上带来约+0.1992pp IoU和+0.1416pp F1；其余单模块多数表现为Precision/Recall工作点变化或很小的区域收益。NonLocal的Recall提高，但Precision损失明显；独立Polarized在该母体上并未提高IoU。单模块无效并不保证其在Aligned母体上也无效，反之也不能自动假定联合互补，需要下一节完整训练验证。PSP只有兼容失败，不存在合法性能分数。')
add('### 6.2 每项模块的作用与真实观察')
for r in rows6:
    if not r.get('modules'):continue
    if r['status']!='complete':add(f"- {r['id']}：{status_cn(r['status'])}，保留失败证据；不填写涨点。")
    else:add(f"- {r['id']}（{r['label']}）：对W128，IoU {pp(delta(r,parent6,'crackIoU'))}pp、F1 {pp(delta(r,parent6,'f1'))}pp、P {pp(delta(r,parent6,'precision'))}pp、R {pp(delta(r,parent6,'recall'))}pp、BF1 {pp(delta(r,parent6,'boundary_f1'))}pp；对原B0，IoU {pp(delta(r,baseline,'crackIoU'))}pp。参数{mparam(r['params'])}M。")
add('## 七 Wave7 Aligned与第二模块的全部组合')
add('全部组合结构为MiT-B0＋width128解码器＋Aligned＋表中第二模块，数据、损失CE、初始化和训练预算保持一致。预登记24种组合，22项完成，2项技术失败/暂缓。')
add(metric_table(rows7))
add('### 7.1 对直接母体A0的涨点')
add(table(['组合','ΔP pp','ΔR pp','ΔF1 pp','ΔIoU pp','ΔmIoU pp','ΔBF1 pp','原单seed母体门槛'],[[r['id']]+[pp(delta(r,a0,k)) for k in METRICS]+[str(r.get('parent_accuracy_gate','—'))] for r in rows7]))
add('### 7.2 对原始SegFormer-B0的涨点')
add(table(['组合','ΔP pp','ΔR pp','ΔF1 pp','ΔIoU pp','ΔmIoU pp','ΔBF1 pp','减参%'],[[r['id']]+[pp(delta(r,baseline,k)) for k in METRICS]+[f"{100*(1-r['params']/baseline['params']):.3f}" if 'params' in r else '—'] for r in rows7]))
add('### 7.3 每个组合的结果解读')
for r in rows7:
    if r['status']!='complete':add(f"- {r['id']}：{status_cn(r['status'])}，未形成正式性能结论。")
    else:add(f"- {r['id']}（{r['label']}）：相对A0的IoU {pp(delta(r,a0,'crackIoU'))}pp、F1 {pp(delta(r,a0,'f1'))}pp，P {pp(delta(r,a0,'precision'))}pp、R {pp(delta(r,a0,'recall'))}pp、BF1 {pp(delta(r,a0,'boundary_f1'))}pp。原单seed完整母体门槛{'通过' if r.get('parent_accuracy_gate') else '未通过'}；相对B0的IoU {pp(delta(r,baseline,'crackIoU'))}pp、参数{mparam(r['params'])}M。")
add('最高seed42区域IoU来自Aligned＋PAM（68.3507%），但Precision和BF1明显下降，不得仅按最高IoU选择最终模型。APNB、Hamburger等也存在母体Precision限制。原先仅A21 Polarized与A10 AFNB进入冻结复核，不等于两者最终通过。GloRe/PPContext/ContextEmbedding等负结果保留；组合名称代表官方模块的适配组合，不是宣称完整复现该模块原论文的整网。')
figure('01_parameter_iou','图1 单seed参数与裂缝IoU的取舍。高IoU不自动等于满足全部门槛。')
figure('02_combo_iou_changes','图2 第二模块相对A0的IoU变化。虚线只表示IoU分项门槛，不能替代Precision/Recall/BF1等完整条件。')
add('## 八 两模块四行消融与交互归因')
add(metric_table([dict(id=r['configuration'],status='完成／复用',**r) for r in factorial]))
interaction=[]
for name,single,joint in [('Polarized',single_psa,joint_psa),('AFNB',single_afnb,joint_afnb)]:
    interaction.append([name]+[pp((joint[k]-a0[k])-(single[k]-parent6[k])) for k in ('crackIoU','f1','mIoU')])
add(table(['第二模块','IoU交互差 pp','F1交互差 pp','mIoU交互差 pp'],interaction))
add('交互差=(联合−Aligned母体)−(第二模块单独−W128)。它只是seed42的描述性差分，不是统计显著性检验，不能证明作用机制。Polarized单独较弱、与Aligned组合较好说明效果依赖接入母体；AFNB有单独对照后才能讨论其联合贡献。组合减参仍主要来自width256→128，损失优化没有参与这些结构行。')
add('## 九 三随机种子完整复核')
add('四个模型均具备seed42/43/44结果。以下标准差为样本标准差（ddof=1），不能将三seed宣称为充分的统计显著性证明，也不能消除反复使用同一验证集的搜索偏差。')
add(metric_table([dict(id=f"{r['model']} seed{r['seed']}",status='完成／冻结复用',**r) for r in seedrows]))
add('### 9.1 均值与样本标准差')
add(table(['模型','P% 均值±SD','R% 均值±SD','F1% 均值±SD','IoU% 均值±SD','mIoU% 均值±SD','BF1% 均值±SD','Params M'],[[r['model']]+[f"{100*r[k+'_mean']:.4f} ± {100*r[k+'_sample_std']:.4f}" for k in METRICS]+[mparam(r['params'])] for r in means]))
add('### 9.2 匹配种子的涨点')
add(table(['候选','参照','seed','ΔP pp','ΔR pp','ΔF1 pp','ΔIoU pp','ΔmIoU pp','ΔBF1 pp'],[[r['candidate'],r['reference'],r['seed']]+[f"{r[k+'_delta_pp']:+.4f}" for k in METRICS] for r in paired]))
add('### 9.3 原门槛的最终审核')
gate_rows=[]
for cand in ('A0','A21','A10'):
    for ref in (('B0',) if cand=='A0' else ('B0','A0')):
        ds={k:means_by[cand][k+'_mean']-means_by[ref][k+'_mean'] for k in METRICS}
        rr=[r for r in paired if r['candidate']==cand and r['reference']==ref]
        gate_rows.append([cand,ref]+[pp(ds[k]) for k in METRICS]+[f"{sum(r['crackIoU_delta_pp']>0 for r in rr)}/3"])
add(table(['候选','参照','平均ΔP pp','平均ΔR pp','平均ΔF1 pp','平均ΔIoU pp','平均ΔmIoU pp','平均ΔBF1 pp','IoU正增益seed'],gate_rows))
add('A21相对B0的平均IoU增益0.4871338pp、F1增益0.3437393pp，3/3 seed为正。相对A0，平均IoU增益精确为0.1907895284pp，要求至少0.2pp，因此第二模块稳定贡献门槛未通过；差距很小也不能通过四舍五入放行。A10相对A0平均IoU下降0.2332866pp，仅1/3 seed为正，第二模块不稳定。A0相对B0平均IoU增加0.2963443pp；但没有匹配的W128 seed43/44，三seed只能支持“A0整套压缩与Aligned方案”的比较，不能把全部均值增益归因于Aligned单模块。')
add('因此此前阶段报告中的“进入候选/单seed通过”已被三seed结果进一步限定：A21可记录为有重复区域收益的参数折中研究结果，但不能写为完整通过的最终模型；A10不宜继续作为稳定第二模块。当前最新路线是固定A0，检验损失配方，见下一节。')
figure('03_matched_seeds','图3 同seed对比可区分训练波动与模型差值；不能挑选某一seed代替三seed均值。')
add('## 十 Wave8 固定A0的20个损失配置')
add(f"本节快照中当前任务为{heartbeat8['tag']}，最后心跳{heartbeat8['timestamp']}，最近训练迭代{heartbeat8.get('latest_iter')}。共{len(loss_done)}项完整结果；运行中行以真实状态文件覆盖可能过时的汇总pending状态。C0就是A0＋CE的同seed42冻结控制；参数恒3.470434M，模型结构、推理阈值、输入与训练预算未变。损失函数不会单独降低推理参数量或自然带来加速。")
add(metric_table([dict(id='C0 A0 + CE',status='冻结复用',**a0)]+rows8,delta_ref=a0))
add('### 10.1 全部损失配方及状态')
recipes=[]
for r in rows8:
    failed=[]
    if r['status']=='complete':
        for k,threshold in protocol8['main_gate'].items():
            if r[k]-a0[k] < threshold-1e-12: failed.append(CN[k]+f" {pp(r[k]-a0[k])}pp")
    recipes.append([r['id'],r['label'],status_cn(r['status']),('通过' if r.get('main_gate') else '未通过') if r['status']=='complete' else '尚无性能结论','；'.join(failed) if failed else '—'])
add(table(['编号','配方','真实状态','完整门槛','未满足项'],recipes))
add('CE＋Lovasz的权重分别是[1,0.1]、[1,0.25]、[1,0.5]，并非归一化到权重和为1；因此也改变总梯度幅值。L06的裂缝IoU由A0的67.8520%到68.6376%，增加0.7856pp，F1增加0.5551pp；但Precision由82.6766%降到80.4899%，损失2.1867pp，超过预登记允许下降0.5pp，仍未通过。L07也提高Recall，但Precision损失更大。不能把这些行称为最终有效配方。')
add('加权CE类别顺序为[背景,裂缝]。L08/L09增加背景权重，提高Precision却牺牲Recall/IoU；L09的Precision85.3079%不是整体最好模型。L10增加裂缝权重，Recall达到84.4415%，但Precision降至77.4460%，BF1也下降；“Recall最高”不能替代区域与边界的综合质量。TopK/OHEM/Focal/DualTask/RMI等尚无本快照完整结果，不能预判涨点。')
add('L01–L04在原MixedLoss组合中发生标签原地修改相关梯度版本冲突，未进入合法性能比较；计划中的L18也涉及同类组合路径。独立兼容副本仅在各损失调用处传labels.clone()隔离标签突变，不改变损失数学定义或权重。已通过的CPU预检不等于GPU或正式训练通过。恢复队列当前'+str(recovery.get('stage') if recovery else '未取得心跳')+'，gpu_started='+str(recovery.get('gpu_started') if recovery else None)+'；它等待父队列结束，不应与当前训练重复占GPU。原失败与修复来源必须分别披露，恢复原配置不计新算法。')
figure('04_loss_tradeoffs','图4 已完成损失配方的区域收益与Precision/Recall代价，参考为同seed42的A0＋CE。')
add('## 十一 参数和同机推理效率')
add(table(['模型','输入','中位延迟ms','P10 ms','P90 ms','配对B0比值','稳定条件'],[[r['id'],f"{r['size']}×{r['size']}",f"{r['median_ms']:.3f}",f"{r['p10_ms']:.3f}",f"{r['p90_ms']:.3f}",f"{r['paired_median_ratio_to_B0']:.3f}",'通过' if r['stability_pass'] else '未通过'] for r in latency['summary']]))
add('配对协议：同一RTX5060、FP32、batch1，四个冻结模型、400/480输入，预热40次、15轮平衡顺序，每轮30次同步前向，转场预热5次；不含读图、模型加载或机器人控制开销。事先要求每模型P90/P10≤1.15且配对比值IQR/median≤0.10。当前条件不满足，桌面其他应用共享资源，不能声明稳定独占性能。')
add('实测480输入B0/A0/A21/A10中位约13.93/15.61/16.91/18.39ms，对应配对比值约1.00/1.18/1.25/1.35。此处比值为逐轮配对比值中位数，不一定等于两总体中位数直接相除。可以如实写“参数更少”，但不能写“更快”“实时效率全面提升”或“最低参数”。旧单项profile因测量热态/时钟差异出现大输入反而更快等现象，只保留原始记录，不用最短一轮替代本表。FLOPs中部分EMAU/NMF算子计数不完整，未作为完整算力结论。')
figure('05_latency_limitations','图5 已有速度测量的真实成本。稳定性未通过，图中变慢结果仍完整保留。')
add('## 十二 真正的Crack500 test结果与公开模型状态')
add('本节才是1124张原始test的结果。它与前面348张验证集的分数不能混排；例如A21 val 68%以上IoU不能拿来直接宣称超过OCRNet test 57%左右IoU。历史test在项目中已经被观察过，后续应如实披露，不能声称是全项目从未见过的最终集合。')
add(metric_table([dict(id=r['model'],status='历史冻结test完成',**r) for r in testrows],bf=False))
add(table(['六个原始公开模型','现有证据','本轮统一test状态'],[['SegFormer-B0','18k、冻结、历史test已完成','可核SHA复用，待统一汇总'],['OCRNet-HRNet-W18','18k、冻结、历史test已完成','可核SHA复用，待统一汇总'],['PP-LiteSeg-STDC1','18k与独立val已完成，原训练exit未知','尚未完成；作为对照不因val弱删行'],['PSPNet-ResNet50','未取得当前协议完整训练和最终test','待兼容与完整实验'],['DeepLabV3+-ResNet50','历史GPU反向smoke失败','待兼容与完整实验'],['FastSCNN','未取得当前协议完整训练和最终test','待完整实验']]))
add('PP-LiteSeg现有val：P82.5001%、R77.2995%、F1 79.8152%、裂缝IoU66.4104%、mIoU82.0509%，参数8.039122M；该行不是test。原PIDNet-S也曾GPU smoke失败，留作历史工程记录，不临时冒充本轮预选六模型之一。所有原始对照保持官方结构；没有原始训练/test回执的模型不能填引用论文数字，也不能用师兄论文表格补齐。')
add('## 十三 逐图误差和空间归因')
add(table(['模型','验证图','逐图宏IoU%','空预测图','零重叠图','TP像素','FP像素','FN像素'],[[r['model'],r['n'],f"{100*r['macro_IoU']:.4f}",r['empty_predictions'],r['zero_overlap'],r['sum_tp'],r['sum_fp'],r['sum_fn']] for r in image_summaries]))
add('全局IoU是像素量加权意义的混淆矩阵结果；逐图宏IoU让每张图同权，两者可朝不同方向变化。Aligned相对B0全局IoU略高，但空预测增加，小前景/边缘段仍可能完全漏检。不能只报告整体涨点而隐去完全变差的样本。逐图CSV完整保留，未删除失败图片。')
sp=spatial['models'];sb=sp['B0']['counts']
add(table(['模型','≤2px附近FP','2–5px FP','>5px远处FP','GT内部距离≤2px FN','GT内部距离>2px FN','空预测'],[[key]+[value['counts'][k] for k in ('fp_near_2px','fp_2_to_5px','fp_far_5px','fn_gt_inner_distance_le2','fn_gt_inner_distance_gt2','empty_pred')] for key,value in sp.items()]))
add('该空间统计源自既有官方inference与恢复流程的独立诊断，能描述误检距GT的位置，却不能自动把远处FP命名为“落叶/阴影”。FN按GT内部距离分带不等于物理裂缝宽度分组。已有代表图只人工查看12张，尚不足以估计全数据集的阴影/落叶场景占比或证明特定模块解决了这些场景。误差原因解释在当前阶段仍是待验证假设。')
figure('06_image_level_changes','图6 全348图的区域与边界变化。左上象限表示边界改善但区域退化，右上象限表示两者改善，左下象限表示同时变差。')
for cand in ('A21','L06'):
    rr=[r for r in paired_images if r['model']==cand]
    selections=[('边界改善但IoU下降',sorted([r for r in rr if r['BF1_delta_pp']>0 and r['IoU_delta_pp']<0],key=lambda r:r['BF1_delta_pp'],reverse=True)[:3]),('IoU与边界同时改善',sorted([r for r in rr if r['BF1_delta_pp']>0 and r['IoU_delta_pp']>0],key=lambda r:r['IoU_delta_pp'],reverse=True)[:3]),('IoU下降最多',sorted(rr,key=lambda r:r['IoU_delta_pp'])[:3])]
    add(f'### {cand} 的代表性定量案例')
    add(table(['类型','图像ID','ΔIoU pp','ΔBF1 pp','ΔFP','ΔFN'],[[label,r['image'],f"{r['IoU_delta_pp']:+.3f}",f"{r['BF1_delta_pp']:+.3f}",r['FP_delta'],r['FN_delta']] for label,items in selections for r in items]))
add('这些案例按已经存在的验证指标选取，只是诊断列表；不是随机场景样本，也没有在本报告中重新推理或将GT叠加误标为预测。完整逐图变化见tables/12_逐图涨点与误检漏检变化.csv。')
add('## 十四 历史BiSeNet与自建集结果保留')
add('下列属于已经退出当前主线的历史协议，仅供审计和回溯；C4不是当前SegFormer基线或最终对照，不与Wave6–8混合排序。部分旧指标/边界实现、增强与预算不同，不能从本表推导本轮新模型的净收益。')
add(metric_table([dict(id=r['model'],status='历史排除主线',**r) for r in legacy],bf=False))
add('历史BiSeNet-C4原始test：'+', '.join(f"{CN[k]}={pct(legacytest[0][k])}%" for k in ('precision','recall','f1','crackIoU','mIoU'))+'；仅在历史附录保留，不进入六个原始公开模型表。' if legacytest else '')
add('### 历史自建集验证')
add(metric_table([dict(id=r['model'],status='历史自建val',**r) for r in custom],bf=False))
add('自建集val为177张，历史训练清单1400对，与Crack500不是同一划分。自建UAFM的高分不能证明其Crack500有效；相反，跨数据差异是继续做域泛化和来源互斥复核的理由。本次没有访问自建test。服务器RTX4090结果、SCSegamba已作废或被用户移除的对照、师兄论文表格不用于当前本机主结果；未经本次独立原始回执核验的历史数字不编入本表，见原历史复盘文档。')
add('## 十五 失败与尚未完成项')
add('1. 结构兼容：Wave6 PSP反向smoke失败；Wave7 DAPPM失败、PSP暂缓。失败不等于模型精度差，不能填写性能为0。\n2. 旧记录问题：UAFM缓存delta、LightHam ordinary/EMA选择、历史BF1公式/边界口径，以及部分训练缺真实exit；本报告逐项说明并采用更精确证据。\n3. 队列包装器与单组训练分开：旧系统任务曾异常结束或缺末端wrapper回执，不能把wrapper写为exit0；单组已有真实exit0/最终权重/独立验证的运行仍可据其证据核验。\n4. 损失兼容恢复尚未完成；CPU预检与等待心跳不计GPU训练成功。\n5. 所有新模型尚无最终统一test；六模型尚未齐全；速度稳定条件尚未通过。\n6. source-frame-disjoint、自建集最终测试、相机—底盘—三轴标定及机器人循迹/宽度/修复覆盖率实验尚不能由分割分数替代，本文不填机器人结果。')
add('## 十六 论文可采用的结论和后续任务')
add('可以写：在相同本机Crack500 train/val协议下，已系统比较宽度控制、官方结构模块组合和匹配种子；参数压缩与特征对齐方案具有小幅且可重复的区域指标收益，但部分组合在Precision、边界或延迟上存在代价。该说法同时保留正结果与局限。')
add('不能写：已实现所有指标最优、领域SOTA、参数最小、已显著加速、已解决阴影落叶或已提高机器人修复精度。当前证据不支持这些表述。更多配置数量不自动增加方法新颖性；最终方法需要清楚的任务动机、因果消融、稳定取舍及真实应用验证。')
add('接下来按现有授权计划：完成正在进行的20损失配置及原配置兼容恢复；只对完整门槛通过的配方冻结最多top2，再做匹配seed43/44；若没有通过者如实报告，不根据观察成绩降低Precision/BF1门槛。之后补齐六个原始公开对照，按val冻结并统一原始test，保留弱对照；再做源帧互斥复核及用户安排的自建集和机器人阶段。因新一轮20损失训练还在进行，以前“结构复核结束后再需8–16小时”的估计不再覆盖当前新增范围，不能沿用为全部任务截止时间。')
add('建议最终论文正文采用三类表：①结构与压缩消融（B0/W128/A0/有效第二模块及联合，保留各自母体）；②固定结构的损失消融（CE及真实有效配方、匹配种子）；③统一test公开模型表（六个官方原始模型与冻结最终模型）。全量负结果可放补充材料，正文选择不能隐瞒未达到预登记门槛的事实。')
add('## 十七 报告附件和复现入口')
add('tables/目录提供所有新实验的原始小数、百分数、对B0和直接母体的逐指标涨点、参数变化，以及逐seed/均值/配对差值、2×2归因、原始test、逐图误检漏检、历史BiSeNet与自建集独立表。figures/提供6组PNG和SVG图，SVG可继续编辑。provenance/snapshots/保留读取时的原始JSON快照及SHA索引，避免运行中汇总变化造成报告不可追溯。')
add('本报告生成器build_results_report.py仅聚合既有产物，CPU绘图，不运行模型。源代码、完整训练/验证日志和原始checkpoint仍在对应实验目录；交付压缩包不包含数据集、环境、预训练或训练大权重。复现索引见同目录复现索引.md。')

from module_analysis import append_analysis
module_notes=append_analysis(add,table,rows6,rows7,parent6,a0,baseline)
csvout('15_每个模块优劣与改进方向.csv',module_notes)

report='\n'.join(parts)
report_name='实验结果总报告'
version=OUT/'refine-logs'/f'{report_name}_{STAMP}.md'
version.write_text(report,encoding='utf-8')
latest=OUT/'refine-logs'/f'{report_name}.md';shutil.copy2(version,latest)
root_report=ROOT/'当前全部实验结果总报告_20261020.md';shutil.copy2(version,root_report)

# Portable HTML renderer for this report's headings, tables, lists, links and images.
def inline(s):
    s=html.escape(s)
    s=re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',s)
    s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
    return s
lines=report.splitlines();body=[];i=0
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('|') and i+1<len(lines) and re.match(r'^\|[\s:|\-]+\|$',lines[i+1]):
        headers=[x.strip() for x in line.strip('|').split('|')];i+=2;rr=[]
        while i<len(lines) and lines[i].lstrip().startswith('|'):
            rr.append([x.strip() for x in lines[i].strip().strip('|').split('|')]);i+=1
        body.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+inline(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in row)+'</tr>' for row in rr)+'</tbody></table></div>');continue
    img=re.match(r'!\[(.*?)\]\((.*?)\)',line)
    if img:
        p=Path(img[2]);encoded=base64.b64encode(p.read_bytes()).decode('ascii');body.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(img[1])}"><figcaption>{inline(img[1])}</figcaption></figure>')
    elif line.startswith('#'):
        n=min(len(line)-len(line.lstrip('#')),6);body.append(f'<h{n}>'+inline(line[n:].strip())+f'</h{n}>')
    elif line.startswith('- '):body.append('<p class="item">• '+inline(line[2:])+'</p>')
    else:body.append('<p>'+inline(line)+'</p>')
    i+=1
css='''body{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;max-width:1320px;margin:36px auto;padding:0 28px;color:#18212d;line-height:1.7}h1{font-size:30px;color:#111}h2{font-size:23px;margin-top:38px;border-bottom:1px solid #bac7d4;padding-bottom:8px}h3{font-size:18px;margin-top:26px}p{margin:12px 0}.table-wrap{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.5;font-variant-numeric:tabular-nums}th{background:#edf2f6;color:#173954;text-align:center;padding:9px 7px;border-bottom:2px solid #758da2}td{padding:7px 6px;border-bottom:1px solid #d8e0e6;text-align:right}td:first-child,td:nth-child(2){text-align:left}tbody tr:nth-child(even){background:#f8fafb}figure{margin:26px 0}img{width:100%;max-height:800px;object-fit:contain}figcaption{font-size:13px;color:#475568}code{font-family:Consolas,monospace;font-size:.9em}a{color:#166782}@media print{body{margin:0;padding:0;max-width:none;font-size:10pt}h2{break-after:avoid}thead{display:table-header-group}tr{break-inside:avoid}figure{break-inside:avoid}table{font-size:7.5pt}.table-wrap{overflow:visible}a{color:inherit} @page{size:A4 landscape;margin:15mm}}'''
html_doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SegFormer全部实验结果报告</title><style>'+css+'</style><body>'+''.join(body)+'</body></html>'
(OUT/'实验结果总报告.html').write_text(html_doc,encoding='utf-8')
jsonout('provenance/报告汇总数据.json',dict(snapshot_at=NOW.isoformat(),baseline=baseline,parent6=parent6,a0=a0,wave6=rows6,wave7=rows7,followup=follow,wave8=rows8,three_seed_means=means,paired=paired,latency=latency['summary'],test=testrows,history=hist,legacy=legacy,custom=custom,heartbeat_at_snapshot=heartbeat8,dice_recovery=recovery,independent_audit='BLOCKED: designated reviewer usage limit; deterministic evidence checks completed separately'))
csvout('来源文件SHA256索引.csv',list(SOURCES.values()))
index='''# 复现与文件索引

主报告：refine-logs/实验结果总报告.md；可打印离线版：实验结果总报告.html。

表格：tables/01–14；全部新实验包括小数原值、百分数和百分点涨点。来源文件SHA索引同在tables中。

结果依据：provenance/snapshots内是生成时JSON快照；确定性核验见provenance/结果确定性核验.json。第三方审阅见独立审计文件。

现有工程：
- Wave4：D:/论文复现/experiments/retrain_wave4_20261007
- Wave5：D:/论文复现/experiments/retrain_wave5_20261012
- Wave6：D:/论文复现/experiments/segformer_balance_wave6_20261016
- Wave7：D:/论文复现/experiments/segformer_aligned_wave7_20261018
- 三seed：Wave7/followup_20261019
- Wave8：D:/论文复现/experiments/segformer_loss_wave8_20261019

生成器只聚合已有结果和CPU绘图；运行该生成器不会训练模型。不要为查看报告重启训练队列。复训前按各工程运行契约确认GPU、Python、PYTHONPATH、种子、原始清单和权重SHA。完整源码和大权重在现有工程中，不在压缩包中重复复制。
'''
(OUT/'复现索引.md').write_text(index,encoding='utf-8')
manifest=OUT/'MANIFEST.md'
entries=[p for p in OUT.rglob('*') if p.is_file() and p.name!='MANIFEST.md']
manifest.write_text('# 报告交付文件清单\n\n'+table(['路径','大小bytes','SHA256'],[[str(p.relative_to(OUT)).replace('\\','/'),p.stat().st_size,sha(p)] for p in sorted(entries)]),encoding='utf-8')
print(json.dumps(dict(report=str(latest),root_report=str(root_report),html=str(OUT/'实验结果总报告.html'),snapshot_at=NOW.isoformat(),completed_run_checks=len(AUDITS),errors=0,wave8_complete=len(loss_done),source_files=len(SOURCES),report_chars=len(report),figures=len(FIGS)),ensure_ascii=False))
