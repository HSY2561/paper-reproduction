"""Render report figures from committed aggregate results. No model or data loading."""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'Noto Sans CJK SC', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': 10})

def load(name):
    return json.loads((ROOT / 'results' / name).read_text(encoding='utf-8-sig'))

def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=160, bbox_inches='tight')
    plt.close(fig)

fusion = load('fusion20_seed42.json')
rows, base = fusion['rows'], fusion['baseline']
fig, ax = plt.subplots(figsize=(10, 5.5))
for r in rows:
    passed = r.get('accuracy_gate_seed42', r.get('accuracy_gate', False))
    color = '#168b79' if passed and r['params'] <= .99 * base['params'] else '#416c96' if passed else '#bdc6ce'
    ax.scatter(r['params']/1e6, r['crackIoU']*100, color=color, s=55)
    if r['id'] in ('W96', 'W80', 'W64', 'F08_Branch32', 'F04_Pair128', 'F17_Branch64'):
        ax.annotate(r['id'], (r['params']/1e6, r['crackIoU']*100), xytext=(5, 4), textcoords='offset points', fontsize=8)
ax.scatter(base['params']/1e6, base['crackIoU']*100, marker='*', s=150, color='#ce8037', label='A0 control')
ax.axvline(.99*base['params']/1e6, color='#8f9ca5', ls='--', label='1% fewer parameters vs A0')
ax.set(xlabel='Parameters (M)', ylabel='Crack IoU (%)', title='Crack500 val: all 20 seed42 configurations; multiseed conclusions separate')
ax.legend(fontsize=8); ax.grid(alpha=.2); save(fig, 'fusion20_pareto')

multi = load('fusion_multiseed.json')['rows']
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, key, title in zip(axs, ('crackIoU', 'boundary_f1'), ('Crack IoU', 'Exact 2px boundary F1')):
    xx = np.arange(len(multi))
    ax.bar(xx, [r[key]*100 for r in multi], yerr=[r['sample_std_'+key]*100 for r in multi],
           color=['#889aa8', '#397ca4', '#c2b794', '#1b8a75'], capsize=5)
    ax.set(xticks=xx, xticklabels=[r['id'] for r in multi], ylabel=title+' (%)', title='Matched seeds 42/43/44: mean ± sample SD')
    vals=[r[key]*100 for r in multi]
    ax.set_ylim(min(vals)-1.2, max(vals)+1.2); ax.grid(axis='y', alpha=.2)
fig.suptitle('F08 passes accuracy balance; W96 fails; F08 extra parameter reduction <1%')
save(fig, 'fusion_multiseed')

keys = ['precision', 'recall', 'f1', 'crackIoU', 'mIoU', 'boundary_f1']
arr = np.array([[100*(r[k]-base[k]) for k in keys] for r in rows])
fig, ax = plt.subplots(figsize=(10, 8.5))
im=ax.imshow(arr, cmap='RdBu', vmin=-4, vmax=4, aspect='auto')
ax.set(xticks=range(6), xticklabels=['P', 'R', 'F1', 'Crack IoU', 'mIoU', 'Boundary F1'],
       yticks=range(len(rows)), yticklabels=[r['id'] for r in rows], title='Seed42 validation differences vs A0 (percentage points)')
for i in range(len(rows)):
    for j in range(6): ax.text(j, i, f'{arr[i,j]:+.2f}', ha='center', va='center', fontsize=8, color='white' if abs(arr[i,j])>2.6 else '#173043')
fig.colorbar(im, ax=ax, label='Δ pp'); save(fig, 'fusion20_metric_deltas')

with (ROOT/'results/public_models_historical_test.csv').open(encoding='utf-8-sig', newline='') as f:
    public=list(csv.DictReader(f))
names={'SegFormer_B0':'B0', 'OCRNet_HRNetW18':'OCRNet', 'PPLiteSeg_STDC1':'PP-LiteSeg',
       'PSPNet_ResNet50':'PSPNet', 'DeepLabV3P_ResNet50':'DeepLabV3P', 'FastSCNN':'Fast-SCNN', 'A0':'A0'}
xx=np.arange(len(public)); labels=[names.get(r['id'], r['id']) for r in public]
fig,axs=plt.subplots(1,2,figsize=(12,5))
for j,key in enumerate(('f1_pct','crack_iou_pct','miou_pct')):
    axs[0].bar(xx+(j-1)*.24,[float(r[key]) for r in public],.24,label=['F1','Crack IoU','mIoU'][j])
axs[0].set(xticks=xx,xticklabels=labels,ylabel='Historical test (%)',ylim=(50,85));axs[0].legend(fontsize=8)
axs[1].bar(xx,[float(r['params_M']) for r in public],color='#418c9b')
axs[1].set(xticks=xx,xticklabels=labels,ylabel='Parameters (M)')
for ax in axs:ax.tick_params(axis='x',rotation=35);ax.grid(axis='y',alpha=.2)
fig.suptitle('Six original references + A0: historical test; recipes/backends differ; unified test pending')
save(fig,'public_historical_test')

with (ROOT/'results/loss20.csv').open(encoding='utf-8-sig',newline='') as f:loss=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(11,4.8));xx=np.arange(len(loss))
for j,key in enumerate(('crackIoU','precision')):
    ax.bar(xx+(j-.5)*.36,[100*(float(r[key])-base[key]) for r in loss],.36,label=['Δ Crack IoU','Δ Precision'][j])
ax.set(xticks=xx,xticklabels=[r['id'].split('_')[0] for r in loss],ylabel='Difference vs A0 (pp)',title='20 loss recipes on A0: seed42 val; no parameter change')
ax.axhline(0,color='#777',lw=.8);ax.legend();ax.grid(axis='y',alpha=.2);save(fig,'loss20_tradeoff')

status=load('current_status.json')
lat=status['latency']['summary']
fig,axs=plt.subplots(1,2,figsize=(11,4.8))
for ax,size in zip(axs,(400,480)):
    rr=[r for r in lat if r['size']==size]
    ax.bar(range(len(rr)),[r['median_ms'] for r in rr],yerr=[r['sample_sd_ms'] for r in rr],capsize=5,color='#498d9a')
    ax.set(xticks=range(len(rr)),xticklabels=[r['model'] for r in rr],ylabel='Latency (ms)',title=f'{size}×{size}, FP32 batch1')
    ax.tick_params(axis='x',rotation=15);ax.grid(axis='y',alpha=.2)
fig.suptitle('Interleaved latency measured; candidate paired CV >10%; no stable acceleration claim')
save(fig,'latency_measured_unstable')
print(json.dumps({'figure_pairs':6,'source':'committed aggregate results only','training_or_inference':False}))
