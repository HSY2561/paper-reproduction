"""Read-only wave6 validation CSV analysis. No inference, training or test access."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
WAVE6 = OUT.parent.parent / 'segformer_balance_wave6_20261016'
DATA = Path('C:/Users/user/algolab_data')
MODELS = ['B0', 'W128', 'M13_Aligned', 'M10_GloRe']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.


def load(model):
    p = WAVE6 / 'results' / (model + '_val_per_image.csv')
    records = {}
    with p.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            n = r.pop('image')
            d = {k: float(v) for k, v in r.items()}
            bp, bg = d['boundary_pred'], d['boundary_gt']
            # Explicit per-image empty-boundary convention: both empty=1;
            # only one empty=0. Micro aggregate is computed separately.
            if not bp and not bg:
                bf = 1.
            elif not bp or not bg:
                bf = 0.
            else:
                bf = f1(d['boundary_tp_pred'] / bp, d['boundary_tp_gt'] / bg)
            d['boundary_f1'] = bf
            d['f1'] = 2 * d['tp'] / max(2*d['tp'] + d['fp'] + d['fn'], 1)
            d['empty_prediction'] = int(d['tp'] + d['fp'] == 0)
            d['zero_overlap'] = int(d['tp'] == 0 and d['tp'] + d['fn'] > 0)
            assert n not in records
            records[n] = d
    return records


def summary_delta(values):
    a = np.asarray(values, dtype=float)
    return dict(increase=int((a > 1e-12).sum()), decrease=int((a < -1e-12).sum()),
                equal=int((np.abs(a) <= 1e-12).sum()), mean=float(a.mean()),
                median=float(np.median(a)), p10=float(np.quantile(a, .1)),
                p90=float(np.quantile(a, .9)), minimum=float(a.min()), maximum=float(a.max()))


def montage(rows, number, paths, data):
    font_path = 'C:/Windows/Fonts/arial.ttf'
    font = ImageFont.truetype(font_path, 15)
    small = ImageFont.truetype(font_path, 13)
    W, H = 1120, 4 * 360 + 55
    canvas = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 8), 'Validation examples only | Original / GT / GT overlay (no prediction masks)', font=font, fill='black')
    draw.text((15, 30), 'Numbers are paired whole-image metric deltas in percentage points vs B0; FP/FN are pixel deltas.', font=small, fill='black')
    entries = []
    for j, item in enumerate(rows):
        name, reason = item
        ip, lp = paths[name]
        im = Image.open(ip).convert('RGB')
        lab0 = np.asarray(Image.open(lp))
        if lab0.ndim == 3:
            lab0 = lab0[..., 0]
        mask = lab0 > 0
        rgb = np.asarray(im).copy()
        rgb[mask] = (rgb[mask] * .4 + np.array([255, 0, 0]) * .6).astype(np.uint8)
        label = Image.fromarray(np.uint8(mask) * 255).convert('RGB')
        y = 55 + j * 360
        draw.text((12, y), f'{name} | {reason}', font=font, fill='black')
        for col, pic in enumerate((im, label, Image.fromarray(rgb))):
            pic.thumbnail((355, 270), Image.Resampling.BILINEAR if col != 1 else Image.Resampling.NEAREST)
            canvas.paste(pic, (12 + col * 368, y + 24))
        a, g, b = data['M13_Aligned'][name], data['M10_GloRe'][name], data['B0'][name]
        for k, (tag, d) in enumerate((('Aligned', a), ('GloRe', g))):
            msg = f"{tag}: IoU {100*(d['crack_iou']-b['crack_iou']):+.2f} pp, BF1 {100*(d['boundary_f1']-b['boundary_f1']):+.2f} pp, FP {int(d['fp']-b['fp']):+d}, FN {int(d['fn']-b['fn']):+d}"
            draw.text((12, y+299+k*21), msg, font=small, fill='black')
        entries.append(dict(image=name, selection=reason, image_path=str(ip), label_path=str(lp),
                            image_sha256=sha(ip), label_sha256=sha(lp), label_values=np.unique(lab0).tolist(),
                            size=list(im.size)))
    dest = OUT / f'val_examples_{number}.png'
    canvas.save(dest)
    return entries


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = {m: load(m) for m in MODELS}
    names = list(d['B0'])
    assert len(names) == 348
    for m in MODELS:
        assert set(d[m]) == set(names)
        for name in names:
            assert d[m][name]['tp']+d[m][name]['fn'] == d['B0'][name]['tp']+d['B0'][name]['fn']
            assert d[m][name]['boundary_gt'] == d['B0'][name]['boundary_gt']
    paths={}
    for line in (DATA/'val.txt').read_text().splitlines():
        p,l=line.split(); paths[Path(p).name]=(DATA/p,DATA/l)
    assert set(names)==set(paths)
    metadata={}
    for n in names:
        raw=np.asarray(Image.open(paths[n][1]))
        if raw.ndim==3: raw=raw[...,0]
        fg=raw>0; h,w=fg.shape
        assert int(fg.sum())==int(d['B0'][n]['tp']+d['B0'][n]['fn'])
        rim=np.ones_like(fg,dtype=bool)
        if h>32 and w>32:rim[16:-16,16:-16]=False
        frac=float(fg.mean()); edge=float((fg & rim).sum()/max(fg.sum(),1))
        metadata[n]={'width':w,'height':h,'gt_area_fraction':frac,
                     'gt_fraction_within_16px_border':edge,
                     'gt_size_bin':'<=0.5%' if frac<=.005 else ('0.5-2%' if frac<=.02 else ('2-5%' if frac<=.05 else '>5%'))}
    pairs = [('M13_Aligned','B0'), ('M10_GloRe','B0'), ('M13_Aligned','W128'), ('M10_GloRe','W128')]
    full = []
    for n in names:
        row = {'image': n, **metadata[n]}
        for m in MODELS:
            row.update({m+'_'+k: v for k,v in d[m][n].items()})
        for m, b in pairs:
            row.update({m+'_vs_'+b+'_delta_'+k: d[m][n][k]-d[b][n][k]
                        for k in ['crack_iou','boundary_f1','f1','precision','recall','fp','fn']})
        full.append(row)
    with (OUT/'paired_val_all348.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=full[0].keys()); w.writeheader(); w.writerows(full)
    out = dict(n=348, split='val', test_used=False, inference_run=False,
               pixel_masks_exported=False,
               boundary_protocol='inner boundary, zero outside, exact Euclidean <=2px; per-image both-empty=1, one-empty=0; official micro numbers separately retained',
               source_csv_sha256={m:sha(WAVE6/'results'/(m+'_val_per_image.csv')) for m in MODELS},
               global_metrics={}, pair_comparisons={})
    for m in MODELS:
        out['global_metrics'][m] = json.loads((WAVE6/'results'/(m+'_val.json')).read_text(encoding='utf-8'))
        out['global_metrics'][m]['zero_overlap_image_count'] = sum(d[m][n]['zero_overlap'] for n in names)
    for m, b in pairs:
        deltas = {k: [d[m][n][k]-d[b][n][k] for n in names]
                  for k in ['crack_iou','boundary_f1','f1','precision','recall','fp','fn']}
        io, bf = np.array(deltas['crack_iou']), np.array(deltas['boundary_f1'])
        out['pair_comparisons'][m+'_vs_'+b] = {
            'deltas':{k:summary_delta(v) for k,v in deltas.items()},
            'both_iou_bf_improve':int(((io>1e-12)&(bf>1e-12)).sum()),
            'both_iou_bf_decrease':int(((io< -1e-12)&(bf< -1e-12)).sum()),
            'boundary_up_iou_down':int(((io< -1e-12)&(bf>1e-12)).sum()),
            'iou_up_boundary_down':int(((io>1e-12)&(bf< -1e-12)).sum()),
            'iou_gain_at_least_1pp':int((io>=.01).sum()),
            'iou_drop_at_least_1pp':int((io<=-.01).sum()),
            'fp_total_delta':int(sum(deltas['fp'])), 'fn_total_delta':int(sum(deltas['fn'])),
            'macro_iou_delta_note':'mean of paired per-image IoU; not global confusion IoU',
            'top10_iou_gain':sorted(names, key=lambda n:d[m][n]['crack_iou']-d[b][n]['crack_iou'], reverse=True)[:10],
            'top10_iou_drop':sorted(names, key=lambda n:d[m][n]['crack_iou']-d[b][n]['crack_iou'])[:10]}
    out['gt_area_groups']={}
    for group in ['<=0.5%','0.5-2%','2-5%','>5%']:
        group_names=[n for n in names if metadata[n]['gt_size_bin']==group]
        group_stats={'n':len(group_names)}
        for m in ['M13_Aligned','M10_GloRe']:
            group_stats[m]={
                'mean_iou_delta_vs_B0':float(np.mean([d[m][n]['crack_iou']-d['B0'][n]['crack_iou'] for n in group_names])),
                'mean_boundary_f1_delta_vs_B0':float(np.mean([d[m][n]['boundary_f1']-d['B0'][n]['boundary_f1'] for n in group_names])),
                'empty_predictions':sum(d[m][n]['empty_prediction'] for n in group_names),
                'iou_better_count':sum(d[m][n]['crack_iou']>d['B0'][n]['crack_iou']+1e-12 for n in group_names),
                'fp_delta':int(sum(d[m][n]['fp']-d['B0'][n]['fp'] for n in group_names)),
                'fn_delta':int(sum(d[m][n]['fn']-d['B0'][n]['fn'] for n in group_names))}
        out['gt_area_groups'][group]=group_stats
    out['small_edge_foreground']={}
    for m in ['M13_Aligned','M10_GloRe']:
        subset=[n for n in names if metadata[n]['gt_area_fraction']<=.005 and metadata[n]['gt_fraction_within_16px_border']>=.5]
        out['small_edge_foreground'][m]={
            'definition':'GT area<=0.5% and >=50% GT within 16px image border',
            'n':len(subset), 'images':subset,
            'empty_predictions':sum(d[m][n]['empty_prediction'] for n in subset),
            'B0_empty_predictions':sum(d['B0'][n]['empty_prediction'] for n in subset),
            'mean_iou_delta_vs_B0':float(np.mean([d[m][n]['crack_iou']-d['B0'][n]['crack_iou'] for n in subset]))}
    picks=[]; used=set()
    for m in ['M13_Aligned','M10_GloRe']:
        def delta(n,k): return d[m][n][k]-d['B0'][n][k]
        choices=[('largest IoU loss',lambda n:True, lambda n:delta(n,'crack_iou')),
                 ('largest IoU gain',lambda n:True, lambda n:-delta(n,'crack_iou')),
                 ('boundary gain / area loss',lambda n:delta(n,'crack_iou')<0 and delta(n,'boundary_f1')>0, lambda n:-delta(n,'boundary_f1')),
                 ('boundary and area loss',lambda n:delta(n,'crack_iou')<0 and delta(n,'boundary_f1')<0, lambda n:delta(n,'boundary_f1')),
                 ('boundary and area gain',lambda n:delta(n,'crack_iou')>0 and delta(n,'boundary_f1')>0, lambda n:-delta(n,'boundary_f1')),
                 ('zero overlap',lambda n:d[m][n]['zero_overlap']>0, lambda n:-d['B0'][n]['crack_iou'])]
        for reason, filt, key in choices:
            avail=sorted([n for n in names if n not in used and filt(n)],key=key)
            if avail:
                n=avail[0];used.add(n);picks.append((n,m+': '+reason))
    for idx in range(0,len(picks),4):
        entries=montage(picks[idx:idx+4],idx//4+1,paths,d)
        out.setdefault('visual_examples',[]).extend(entries)
    (OUT/'paired_val_summary.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:{q:v[q] for q in ['both_iou_bf_improve','both_iou_bf_decrease','boundary_up_iou_down','iou_up_boundary_down','fp_total_delta','fn_total_delta']} for k,v in out['pair_comparisons'].items()},indent=2))
    print(json.dumps(picks,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
