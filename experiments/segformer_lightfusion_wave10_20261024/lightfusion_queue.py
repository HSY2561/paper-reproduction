"""Wave10: lightweight multi-scale fusion replacements for SegFormer+Aligned.

This is experiment orchestration plus configuration of standard tensor/conv
operators.  It never reads the Crack500 test list.  The source snapshot is
isolated from all prior waves and every candidate keeps the same backbone,
augmentation, loss, optimizer and 18k iteration budget.
"""
import copy
import csv
import datetime as dt
import importlib.util
import json
import os
import shutil
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('wave6_runtime', ROOT / 'wave6_runtime.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
q.ROOT = ROOT
q.SRC = ROOT / 'src' / 'PaddleSeg-2.10.0'
q.DATA = Path(r'C:\Users\user\algolab_data')
q.DATA_MANIFEST = Path(r'D:\论文复现\experiments\local_structure_20260930\provenance\train_val_manifest.json')
q.__file__ = __file__
q.BASE_WEIGHTS = ROOT / 'checkpoints' / 'A0' / 'model.pdparams'
q.BASE_SHA = ''

def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')

def read(path):
    return q.read_json(path)

def write(path, obj):
    return q.write_json(path, obj)


JOBS = [
    dict(id='W96', width=96, variant='base', rank=32, label='Aligned base width96', family='width_control'),
    dict(id='W80', width=80, variant='base', rank=32, label='Aligned base width80', family='width_control'),
    dict(id='W64', width=64, variant='base', rank=32, label='Aligned base width64', family='width_control'),
    dict(id='F01_Sum128', width=128, variant='sum', rank=32, label='four-branch sum + 1x1', family='sum'),
    dict(id='F02_Weighted128', width=128, variant='weighted_sum', rank=32, label='learned scalar weighted sum', family='weighted_sum'),
    dict(id='F03_ChannelWeighted128', width=128, variant='channel_weighted_sum', rank=32, label='channel weighted sum', family='channel_weighted_sum'),
    dict(id='F04_Pair128', width=128, variant='pair_sum', rank=32, label='pairwise low/high sum', family='pair_sum'),
    dict(id='F05_Hierarchical128', width=128, variant='hierarchical', rank=32, label='hierarchical pair fusion', family='hierarchical'),
    dict(id='F06_LowRank32', width=128, variant='lowrank_concat', rank=32, label='low-rank concat rank32', family='lowrank'),
    dict(id='F07_LowRank16', width=128, variant='lowrank_concat16', rank=16, label='low-rank concat rank16', family='lowrank'),
    dict(id='F08_Branch32', width=128, variant='branch_bottleneck', rank=32, label='per-branch bottleneck rank32', family='branch_bottleneck'),
    dict(id='F09_Branch16', width=128, variant='branch_bottleneck16', rank=16, label='per-branch bottleneck rank16', family='branch_bottleneck'),
    dict(id='F10_DWSum128', width=128, variant='dw_sum', rank=32, label='depthwise sum fusion', family='depthwise'),
    dict(id='F11_Separable128', width=128, variant='separable_sum', rank=32, label='separable sum fusion', family='separable'),
    dict(id='F12_GatedPair128', width=128, variant='gated_pair', rank=32, label='learned gated pair fusion', family='gated_pair'),
    dict(id='F13_Sum96', width=96, variant='sum', rank=32, label='sum fusion width96', family='sum_width'),
    dict(id='F14_LowRank96', width=96, variant='lowrank_concat', rank=24, label='low-rank concat width96 rank24', family='lowrank_width'),
    dict(id='F15_DWSum96', width=96, variant='dw_sum', rank=32, label='depthwise sum width96', family='depthwise_width'),
    dict(id='F16_Sum64', width=64, variant='sum', rank=32, label='sum fusion width64', family='sum_width'),
    dict(id='F17_Branch64', width=64, variant='branch_bottleneck16', rank=16, label='branch bottleneck width64 rank16', family='branch_width'),
]

def prepare():
    import yaml
    template = yaml.safe_load((ROOT / 'configs' / 'baseline_template.yml').read_text(encoding='utf-8-sig'))
    template['model']['backbone']['pretrained'] = str(ROOT / 'weights' / 'mit_b0_imagenet.pdparams')
    template['model']['balance_modules'] = ['aligned']
    template['model']['fusion_variant'] = 'base'
    template['model']['fusion_rank'] = 32
    # A0 is a frozen, independently verified control copied from wave7.
    old = Path(r'D:\论文复现\experiments\segformer_aligned_wave7_20261018')
    old_ckpt = Path(r'D:\论文复现\experiments\segformer_balance_wave6_20261016\checkpoints\M13_Aligned\model.pdparams')
    (ROOT / 'checkpoints' / 'A0').mkdir(parents=True, exist_ok=True)
    shutil.copy2(old_ckpt, ROOT / 'checkpoints' / 'A0' / 'model.pdparams')
    q.BASE_SHA = q.sha(ROOT / 'checkpoints' / 'A0' / 'model.pdparams')
    shutil.copy2(old / 'results' / 'A0_val.json', ROOT / 'results' / 'A0_val.json')
    shutil.copy2(old / 'results' / 'A0_profile.json', ROOT / 'results' / 'A0_profile.json')
    a0p = read(ROOT / 'results' / 'A0_profile.json')
    a0p['checkpoint'] = str(ROOT / 'checkpoints' / 'A0' / 'model.pdparams')
    a0p['checkpoint_sha256'] = q.BASE_SHA
    write(ROOT / 'results' / 'A0_profile.json', a0p)
    write(ROOT / 'checkpoints' / 'A0' / 'selection.json', dict(model='A0', source='wave7 M13_Aligned',
         checkpoint=str(ROOT / 'checkpoints' / 'A0' / 'model.pdparams'), sha256=q.BASE_SHA,
         selection='reused independent validation selection from wave7', test_used=False, frozen_at=now()))
    write(ROOT / 'results' / 'A0_status.json', dict(status='complete_reused', updated_at=now(), source='wave7_M13_Aligned'))
    configs = []
    for job in JOBS:
        cfg = copy.deepcopy(template)
        cfg['model']['embedding_dim'] = job['width']
        cfg['model']['fusion_variant'] = job['variant']
        cfg['model']['fusion_rank'] = job['rank']
        cfg_path = ROOT / 'configs' / (job['id'] + '.yml')
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
        configs.append(cfg_path.name)
    write(ROOT / 'provenance' / 'job_specs.json', dict(created_at=now(), parent='SegFormer-B0 width128 + Aligned',
         jobs=JOBS, controls=['A0'], module_count=len(JOBS), test_used=False))
    write(ROOT / 'provenance' / 'protocol.json', dict(created_at=now(), parent='A0=width128+Aligned',
         seed=42, batch_size=4, iters=18000, precision='fp32', optimizer='AdamW', learning_rate=6e-5,
         end_lr=0, crop=[400,400], train=1896, val=348, test_used=False,
         candidate_rule='one fusion_variant or width change; common ImageNet MiT-B0 initialization',
         gate_vs_A0={'mIoU':-0.002,'crackIoU':-0.002,'f1':-0.002,'precision':-0.005,'recall':-0.005,'boundary_f1':-0.005},
         efficiency_target='params <= 0.99*A0 preferred; balanced Pareto allows accuracy within gate and reports exact latency',
         boundary='zero-extension exact 2px symmetric boundary matching',
         selection='highest exact validation mIoU among ordinary/EMA; freeze SHA; no test until candidate selected'))
    write(ROOT / 'provenance' / 'source_hashes.json', {str(p.relative_to(q.SRC)): q.sha(p) for p in (q.SRC/'paddleseg').rglob('*.py')})
    write(ROOT / 'provenance' / 'config_hashes.json', {p.name: q.sha(p) for p in (ROOT/'configs').glob('*.yml')})
    write(ROOT / 'provenance' / 'data_manifest_reference.json', read(q.DATA_MANIFEST))
    write(ROOT / 'results' / 'queue_state.json', dict(state='prepared', jobs=len(JOBS), completed=0, test_used=False, updated_at=now()))
    print('PREPARED', len(JOBS), 'fusion/width candidates; A0 reused', flush=True)

def status(ident, value, **kwargs):
    write(ROOT / 'results' / (ident + '_status.json'), dict(status=value, updated_at=now(), **kwargs))

def report():
    base = read(ROOT / 'results' / 'A0_val.json')
    bp = read(ROOT / 'results' / 'A0_profile.json')
    rows=[]
    for j in JOBS:
        ident=j['id']; sf=ROOT/'results'/(ident+'_status.json'); vf=ROOT/'results'/(ident+'_val.json'); pf=ROOT/'results'/(ident+'_profile.json')
        row=dict(id=ident,label=j['label'],width=j['width'],variant=j['variant'],status=read(sf)['status'] if sf.exists() else 'pending')
        if vf.exists() and pf.exists() and row['status']=='complete':
            m=read(vf); p=read(pf); keys=('precision','recall','f1','crackIoU','mIoU','boundary_f1')
            row.update({k:m[k] for k in keys}); row['params']=p['params_total']; row['latency480_ms']=p['latency'][1]['median_ms']
            row['delta_vs_A0']={k:m[k]-base[k] for k in keys}; row['param_ratio']=p['params_total']/bp['params_total']
            d=row['delta_vs_A0']; row['accuracy_gate']=all([d['mIoU']>=-.002,d['crackIoU']>=-.002,d['f1']>=-.002,d['precision']>=-.005,d['recall']>=-.005,d['boundary_f1']>=-.005])
            row['lightweight_gate']=row['accuracy_gate'] and row['param_ratio']<=.99
            row['pareto_candidate']=row['accuracy_gate'] and row['params']<=bp['params_total']
        rows.append(row)
    write(ROOT/'results/comparison.json',dict(baseline=base,baseline_profile=bp,rows=rows,test_used=False,updated_at=now()))
    cols=['id','label','width','variant','status','precision','recall','f1','crackIoU','mIoU','boundary_f1','params','latency480_ms','accuracy_gate','lightweight_gate','pareto_candidate']
    with (ROOT/'results/ablation_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    lines=['# SegFormer-B0+Aligned 轻量多尺度融合波次10','','更新时间：'+now(),'',
           '筛选仅使用Crack500 train/val；A0为wave7独立冻结控制。', '',
           '| ID | 候选 | 状态 | P | R | F1 | crackIoU | mIoU | BF1 | Params M | Latency480 ms |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    lines.append('| A0 | width128 + Aligned（复用） | complete_reused | '+' | '.join(f'{base[k]:.6f}' for k in ('precision','recall','f1','crackIoU','mIoU','boundary_f1'))+f" | {bp['params_total']/1e6:.6f} | {bp['latency'][1]['median_ms']:.3f} |")
    for r in rows:
        if 'mIoU' in r:
            lines.append('| '+r['id']+' | '+r['label']+' | '+r['status']+' | '+' | '.join(f'{r[k]:.6f}' for k in ('precision','recall','f1','crackIoU','mIoU','boundary_f1'))+f" | {r['params']/1e6:.6f} | {r['latency480_ms']:.3f} |")
        else: lines.append('| '+r['id']+' | '+r['label']+' | '+r['status']+' | — | — | — | — | — | — | — | — |')
    lines += ['', '候选均为新波融合配置或宽度控制，不把旧wave7注意力加法模块重复计数。lightweight_gate需要精度门槛和参数≤A0的实测证据。', '未完成/失败候选不填性能；不得据此声称SOTA。']
    q.versioned_text(ROOT/'refine-logs/EXPERIMENT_RESULTS.md','\n'.join(lines)+'\n')

def run_job(j):
    ident=j['id']; smoke=ROOT/'results'/(ident+'_smoke_exit.json')
    if not smoke.exists():
        status(ident,'smoke_running'); code=q.run_command(q.train_args(ident,True),ident+'_smoke',timeout=1800)
        if code: status(ident,'smoke_failed'); return False
        status(ident,'smoke_passed')
    if not (ROOT/'results'/(ident+'_train_exit.json')).exists():
        if (ROOT/'training'/ident).exists():
            status(ident,'needs_review',error='unfinished training directory without exit receipt'); return False
        status(ident,'training'); code=q.run_command(q.train_args(ident,False),ident+'_train',timeout=43200)
        if code: status(ident,'train_failed'); return False
    if not (ROOT/'training'/ident/'iter_18000/model.pdparams').exists():
        status(ident,'needs_review',error='iter_18000 checkpoint missing'); return False
    log=(ROOT/'logs'/(ident+'_train.log')).read_text(encoding='utf-8',errors='replace')
    if 'iter: 18000/18000' not in log:
        status(ident,'needs_review',error='final iteration missing'); return False
    selection=q.choose_checkpoint(ident); status(ident,'verifying',selection=selection)
    q.evaluate(ident,selection['checkpoint'])
    measured=read(ROOT/'results'/(ident+'_val.json'))
    if abs(measured['mIoU']-selection['selected']['mIoU'])>1e-6:
        status(ident,'needs_review',error='independent validation mismatch'); return False
    # Compress the frozen tensor file after all independent checks.  The raw
    # SHA remains in selection.json; the archive is the retained reproducible
    # artifact and can be expanded for later seed/test stages.
    frozen = ROOT / 'checkpoints' / ident / 'model.pdparams'
    archive = frozen.with_suffix('.pdparams.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(frozen, arcname='model.pdparams')
    selection['raw_checkpoint_sha256'] = q.sha(frozen)
    selection['archive'] = str(archive)
    selection['archive_sha256'] = q.sha(archive)
    selection['checkpoint'] = str(archive)
    write(ROOT / 'checkpoints' / ident / 'selection.json', selection)
    frozen.unlink(missing_ok=True)
    status(ident,'complete',selection=selection)
    # Keep the frozen winner and selection evidence, release redundant
    # iter_1000 checkpoints so the long sequential sweep stays on the data
    # disk without crossing the 4 GiB safety floor.
    shutil.rmtree(ROOT / 'training' / ident, ignore_errors=True)
    return True

def run_queue():
    lock=ROOT/'results/queue.lock'
    with lock.open('x') as f: json.dump(dict(pid=os.getpid(),started_at=now()),f)
    try:
        if not q.BASE_WEIGHTS.exists(): raise RuntimeError('A0 frozen checkpoint missing')
        q.BASE_SHA = q.sha(q.BASE_WEIGHTS)
        q.audit_data('before')
        for j in JOBS:
            ident=j['id']; sf=ROOT/'results'/(ident+'_status.json')
            if sf.exists() and read(sf).get('status')=='complete': continue
            try: run_job(j)
            except Exception as e: status(ident,'needs_review',error=str(e),traceback=traceback.format_exc())
            report(); write(ROOT/'results/heartbeat.json',dict(tag=ident,timestamp=now(),free_gb=round(shutil.disk_usage(ROOT).free/2**30,2),test_used=False))
            if shutil.disk_usage(ROOT).free < 4*2**30: raise RuntimeError('disk below 4GiB; queue paused')
        q.audit_data('after'); report()
        states={j['id']:read(ROOT/'results'/(j['id']+'_status.json')).get('status') for j in JOBS if (ROOT/'results'/(j['id']+'_status.json')).exists()}
        write(ROOT/'results/queue_done.json',dict(updated_at=now(),states=states,complete=sum(v=='complete' for v in states.values()),planned=len(JOBS),test_used=False))
    finally: lock.unlink(missing_ok=True)

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--run',action='store_true'); p.add_argument('--report',action='store_true'); p.add_argument('--profile'); p.add_argument('--weights')
    a=p.parse_args()
    if a.prepare: prepare()
    elif a.run: run_queue()
    elif a.report: report()
    elif a.profile: q.profile(a.profile,a.weights)
    else: p.error('choose --prepare, --run, --report or --profile')
