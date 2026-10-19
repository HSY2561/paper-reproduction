"""Matched seeds and missing component ablation, official training/evaluation."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import pickle
import re
import sys
import traceback
ROOT=Path(__file__).resolve().parent
MAIN=ROOT.parent
sys.path.insert(0,str(MAIN))
import aligned_queue as a
q=a.q
a.ROOT=ROOT;q.ROOT=ROOT;q.SRC=MAIN/'src/PaddleSeg-2.10.0';q.__file__=__file__
q.run_command=a.run_command
read,write,sha=a.read,a.write,a.sha
FIELDS=('precision','recall','f1','crackIoU','mIoU','boundary_f1')


def prepare():
    import yaml
    for name in ('configs','results','provenance','logs','training','checkpoints','tmp','refine-logs'):(ROOT/name).mkdir(parents=True,exist_ok=True)
    catalog=[]
    configurations=[('X_AFNB_s42','A10_AFNB',42,['afnb'])]
    for seed in (43,44):
        for label,source in [('B0','B0'),('A0','A0'),('A21','A21_Polarized'),('A10','A10_AFNB')]:
            configurations.append((label+'_s'+str(seed),source,seed,None))
    for ident,source,seed,modules in configurations:
        cfg=yaml.safe_load((MAIN/'configs'/(source+'.yml')).read_text(encoding='utf-8-sig'))
        if modules is not None:cfg['model']['balance_modules']=modules
        (ROOT/'configs'/(ident+'.yml')).write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
        catalog.append(dict(id=ident,source=source,seed=seed,modules=cfg['model']['balance_modules']))
    write(ROOT/'provenance/jobs.json',catalog)
    write(ROOT/'provenance/config_hashes.json',{p.name:sha(p) for p in (ROOT/'configs').glob('*.yml')})
    report()
    print('PREPARED_FOLLOWUP',len(catalog),flush=True)


def report():
    if not (ROOT/'provenance/jobs.json').exists():return
    jobs=read(ROOT/'provenance/jobs.json');rows=[]
    for job in jobs:
        ident=job['id'];p=ROOT/'results'/(ident+'_status.json')
        row=dict(job,status=read(p)['status'] if p.exists() else 'pending')
        if row['status']=='complete':
            m=read(ROOT/'results'/(ident+'_val.json'));row.update({k:m[k] for k in FIELDS})
        rows.append(row)
    with (ROOT/'results/seed_runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','seed','status',*FIELDS],extrasaction='ignore');w.writeheader();w.writerows(rows)
    lines=['# 冻结候选的归因与匹配多seed进度','','更新时间：'+q.now(),'','当前只用train/val；seed42候选已冻结；未齐三seed不计算均值±标准差。','',
        '| 运行 | seed | 状态 | P% | R% | F1% | IoU% | mIoU% | BF1% |','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        numbers=' | '.join(f'{r[k]*100:.4f}' for k in FIELDS) if 'mIoU' in r else '— | — | — | — | — | —'
        lines.append(f"| {r['id']} | {r['seed']} | {r['status']} | {numbers} |")
    groups={};seed42={'B0':'B0','A0':'A0','A21':'A21_Polarized','A10':'A10_AFNB'}
    for label,origin in seed42.items():
        values=[dict(seed=42,**{k:read(MAIN/'results'/(origin+'_val.json'))[k] for k in FIELDS})]
        for r in rows:
            if r['id'].startswith(label+'_s') and r['status']=='complete':values.append({k:r[k] for k in ['seed',*FIELDS]})
        groups[label]=dict(n=len(values),seeds=values)
        if len(values)==3:
            import statistics
            groups[label]['mean']={k:statistics.mean(r[k] for r in values) for k in FIELDS}
            groups[label]['sample_std']={k:statistics.stdev(r[k] for r in values) for k in FIELDS}
    write(ROOT/'results/three_seed_summary.json',dict(groups=groups,rows=rows,test_used=False,updated_at=q.now()))
    q.versioned_text(ROOT/'refine-logs/RESULTS.md','\n'.join(lines)+'\n')


def run():
    a.ROOT=MAIN
    a.verify_release()
    a.ROOT=ROOT
    frozen=read(ROOT/'provenance/frozen_candidates.json')
    for r in frozen['timing_models'].values():
        if sha(r['checkpoint'])!=r['sha256']:raise RuntimeError('Frozen candidate SHA mismatch')
    for name,expected in read(ROOT/'provenance/config_hashes.json').items():
        if sha(ROOT/'configs'/name)!=expected:raise RuntimeError('Followup config changed')
    lock=ROOT/'results/queue.lock'
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),started_at=q.now()),f)
    try:
        q.audit_data('before')
        if not (ROOT/'results/candidate_latency_exit.json').exists():
            code=a.run_command([MAIN/'candidate_latency.py'],'candidate_latency',3600)
            write(ROOT/'results/speed_stage.json',dict(exit_code=code,precision_gate_not_relaxed=True,test_used=False))
        for job in read(ROOT/'provenance/jobs.json'):
            ident=job['id'];state=ROOT/'results'/(ident+'_status.json')
            if state.exists() and read(state)['status'] in ('complete','smoke_failed','train_failed','needs_review'):continue
            def args(smoke):
                result=q.train_args(ident,smoke);result[result.index('--seed')+1]=str(job['seed']);return result
            try:
                smoke=ROOT/'results'/(ident+'_smoke_exit.json')
                if not smoke.exists():
                    q.status(ident,'smoke_running')
                    if a.run_command(args(True),ident+'_smoke',1800):q.status(ident,'smoke_failed');report();continue
                elif read(smoke)['exit_code']!=0:q.status(ident,'smoke_failed');continue
                receipt=ROOT/'results'/(ident+'_train_exit.json')
                if not receipt.exists():
                    if (ROOT/'training'/ident).exists():raise RuntimeError('Prior unfinished training requires audit')
                    q.status(ident,'training')
                    if a.run_command(args(False),ident+'_train'):q.status(ident,'train_failed');report();continue
                elif read(receipt)['exit_code']!=0:q.status(ident,'train_failed');continue
                if not (ROOT/'training'/ident/'iter_18000/model.pdparams').exists() or 'iter: 18000/18000' not in (ROOT/'logs'/(ident+'_train.log')).read_text(errors='replace'):
                    raise RuntimeError('No full18k evidence')
                chosen=q.choose_checkpoint(ident);q.status(ident,'verifying')
                q.evaluate(ident,chosen['checkpoint'])
                if abs(read(ROOT/'results'/(ident+'_val.json'))['mIoU']-chosen['selected']['mIoU'])>1e-6:raise RuntimeError('Independent validation mismatch')
                q.status(ident,'complete',selection=chosen)
            except Exception as e:
                q.status(ident,'needs_review',error=str(e),traceback=traceback.format_exc());report()
                if 'disk_below' in str(e) or 'log_stale' in str(e):raise
            report()
        q.audit_data('after');report()
        states={j['id']:read(ROOT/'results'/(j['id']+'_status.json'))['status'] for j in read(ROOT/'provenance/jobs.json')}
        write(ROOT/'results/queue_done.json',dict(states=states,updated_at=q.now(),test_used=False,next_step='review three-seed gates and speed, then original-public-model phase'))
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--profile');p.add_argument('--weights');args=p.parse_args()
    if args.prepare:prepare()
    elif args.run:run()
    elif args.report:report()
    elif args.profile:q.profile(args.profile,args.weights)
