"""Generate Fig. 4-3 from the SegFormer-B0 training log."""
from pathlib import Path
import csv, json, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r'D:\论文复现\experiments\retrain_wave4_20261007')
LOG = ROOT / 'logs/segformer_b0_train.log'
SEL = ROOT / 'checkpoints/segformer_b0/selection.json'
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

text = LOG.read_text(encoding='utf-8', errors='replace')
iter_pat = re.compile(r'iter:\s*(\d+)/18000')
eval_pat = re.compile(r'\[EVAL\]\s+#Images:\s*348\s+mIoU:\s*([0-9.]+)')
points = []
current_iter = None
for line in text.splitlines():
    m = iter_pat.search(line)
    if m:
        current_iter = int(m.group(1))
    m = eval_pat.search(line)
    if m:
        if current_iter is None:
            raise RuntimeError('An evaluation record has no preceding training iteration')
        points.append((current_iter, float(m.group(1))))
if len(points) != 36 or len(points) % 2:
    raise RuntimeError(f'Expected 36 paired validation records, found {len(points)}')

records = []
for i in range(0, len(points), 2):
    it1, ordinary = points[i]
    it2, ema = points[i+1]
    if it1 != it2:
        raise RuntimeError(f'Mismatched ordinary/EMA iteration: {it1}, {it2}')
    records.append({'iteration': it1, 'ordinary_mIoU': ordinary, 'ema_mIoU': ema})

selected = json.loads(SEL.read_text(encoding='utf-8-sig'))['selected']
selected_iter = int(selected['state']['iter'])
selected_miou = float(selected['state']['mIoU'])
ordinary_best = max(records, key=lambda x: x['ordinary_mIoU'])
ema_best = max(records, key=lambda x: x['ema_mIoU'])
if ordinary_best['iteration'] != selected_iter or abs(ordinary_best['ordinary_mIoU'] - selected_miou) > 2e-4:
    raise RuntimeError('Parsed ordinary curve does not agree with selection.json')

with (OUT/'fig4_3_checkpoint_selection.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['iteration','ordinary_mIoU','ema_mIoU'])
    for r in records: w.writerow([r['iteration'], f"{r['ordinary_mIoU']:.6f}", f"{r['ema_mIoU']:.6f}"])
metadata = {
    'figure': 'Fig. 4-3', 'model': 'SegFormer-B0', 'dataset': 'Crack500 validation',
    'log': str(LOG), 'selection_json': str(SEL), 'n_validation_events': len(records),
    'selection_metric': 'validation mIoU', 'selected_frozen_variant': 'ordinary best_model',
    'selected_frozen_iteration': selected_iter, 'selected_frozen_mIoU': selected_miou,
    'ordinary_best': ordinary_best, 'ema_best': ema_best,
    'test_used_for_curve_or_selection': False,
    'note': 'All values are parsed from the recorded log; no interpolation or smoothing was applied.'
}
(OUT/'fig4_3_checkpoint_selection.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'mathtext.fontset': 'stix',
})
xs = [r['iteration']/1000 for r in records]
ys_o = [r['ordinary_mIoU'] for r in records]
ys_e = [r['ema_mIoU'] for r in records]
fig, ax = plt.subplots(figsize=(5.25, 3.35))
ax.plot(xs, ys_o, color='#0072B2', linewidth=1.9, marker='o', markersize=3.5, label='Ordinary model')
ax.plot(xs, ys_e, color='#D55E00', linewidth=1.9, marker='s', markersize=3.2, linestyle='--', label='EMA model')
ax.axvline(selected_iter/1000, color='#0072B2', linewidth=1.1, linestyle=':', alpha=.9)
ax.scatter([selected_iter/1000], [selected_miou], color='#0072B2', s=42, zorder=4, edgecolor='white', linewidth=.7)
ax.scatter([ema_best['iteration']/1000], [ema_best['ema_mIoU']], color='#D55E00', s=40, zorder=4, edgecolor='white', linewidth=.7)
ax.annotate(f'Frozen ordinary\n{selected_iter:,} iter, {selected_miou:.4f}',
            xy=(selected_iter/1000, selected_miou), xytext=(selected_iter/1000+1.0, selected_miou-.0045),
            arrowprops=dict(arrowstyle='->', color='#0072B2', lw=.9), color='#005A8D', fontsize=8.5,
            ha='left', va='top')
ax.annotate(f'Best EMA\n{ema_best["iteration"]:,} iter, {ema_best["ema_mIoU"]:.4f}',
            xy=(ema_best['iteration']/1000, ema_best['ema_mIoU']), xytext=(ema_best['iteration']/1000+1.0, ema_best['ema_mIoU']+.0022),
            arrowprops=dict(arrowstyle='->', color='#D55E00', lw=.9), color='#A84600', fontsize=8.5,
            ha='left', va='bottom')
ax.set_xlabel('Training iteration (×10$^3$)')
ax.set_ylabel('Validation mIoU')
ax.set_xlim(0.5, 18.5); ax.set_ylim(0.785, 0.833)
ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 18])
ax.grid(axis='y', color='#D9D9D9', linewidth=.55, alpha=.7)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(frameon=False, loc='lower right')
fig.tight_layout(pad=.6)
for ext in ['pdf','png','svg']:
    fig.savefig(OUT/f'Fig_4_3_checkpoint_selection.{ext}', dpi=300, facecolor='white')
print(json.dumps(metadata, ensure_ascii=False, indent=2))
