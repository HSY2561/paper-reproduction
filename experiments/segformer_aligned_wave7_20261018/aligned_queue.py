"""Aligned-parent ablation orchestration; existing official PaddleSeg blocks."""
import argparse
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'segformer_balance_wave6_20261016'
spec = importlib.util.spec_from_file_location('wave6_runtime', OLD / 'experiment_queue.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
q.ROOT = ROOT
q.SRC = ROOT / 'src/PaddleSeg-2.10.0'
q.__file__ = __file__
read, write, sha = q.read_json, q.write_json, q.sha
METRICS = ('precision','recall','f1','crackIoU','mIoU','boundary_f1')


def jobs():
    return read(ROOT / 'provenance/job_specs.json')['jobs']


def prepare():
    import yaml
    catalog = read(ROOT / 'provenance/module_catalog.json')
    assert len(catalog) >= 21 and len({r['module'] for r in catalog}) == len(catalog)
    template = yaml.safe_load((ROOT / 'configs/baseline_template.yml').read_text(encoding='utf-8-sig'))
    template['model']['backbone']['pretrained'] = str(ROOT / 'weights/mit_b0_imagenet.pdparams')
    template['model'].pop('decoder_variant',None)
    controls = [('B0',256,[]),('W128',128,[]),('A0',128,['aligned']),('GloRe',128,['glore'])]
    pending = [dict(r,width=128,modules=['aligned',r['module']],parent='A0',
                   skip_reason=('Prior PSP backward cuDNN incompatibility; no performance result' if r['module']=='psp_pool' else None)) for r in catalog]
    for ident,width,modules in controls + [(r['id'],r['width'],r['modules']) for r in pending]:
        cfg = copy.deepcopy(template)
        cfg['model']['embedding_dim'] = width
        cfg['model']['balance_modules'] = modules
        (ROOT / 'configs' / (ident+'.yml')).write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    write(ROOT/'provenance/job_specs.json',dict(created_at=q.now(),jobs=pending,modules=len(pending),full_training_planned=sum(not r['skip_reason'] for r in pending)))
    mapping={'B0':'B0','W128':'W128','A0':'M13_Aligned','GloRe':'M10_GloRe'}
    records={}
    for ident,old in mapping.items():
        for suffix in ('val.json','val_per_image.csv','profile.json'):
            source=OLD/'results'/(old+'_'+suffix)
            shutil.copy2(source,ROOT/'results'/(ident+'_'+suffix))
        weights = q.BASE_WEIGHTS if ident=='B0' else OLD/'checkpoints'/old/'model.pdparams'
        selection = None if ident=='B0' else read(OLD/'checkpoints'/old/'selection.json')
        records[ident]=dict(source_id=old,checkpoint=str(weights),sha256=sha(weights),selection=selection,
                            source_val_sha256=sha(OLD/'results'/(old+'_val.json')),reused=True)
    write(ROOT/'provenance/reused_controls.json',records)
    write(ROOT/'provenance/protocol.json',dict(created_at=q.now(),parent='A0=SegFormer-B0 width128 + Aligned',
          seed=42,batch_size=4,iters=18000,precision='fp32',optimizer='AdamW',lr=6e-5,end_lr=0,
          train=1896,val=348,crop=[400,400],test_used=False,initialization='same ImageNet MiT-B0; no fine-tuning from trained A0',
          checkpoint_selection='maximum exact validation mIoU among ordinary/EMA; frozen SHA; independent official val and pixel/boundary verification',
          source_scope='one additional module per Aligned model; no width variants counted as modules',
          preregistered_parent_gate={'crackIoU':.002,'f1':.001,'mIoU':0,'precision':-.005,'recall':-.005,'boundary_f1':-.005},
          original_B0_target_gate={'crackIoU':.002,'f1':.001,'mIoU':0,'recall':-.005,'boundary_f1':-.005,'max_param_ratio':.95,'max_latency_ratio':1.1},
          balanced_backup='pass parent accuracy gate, Params<=B0, latency<=1.1x B0; separate from strict -5% parameter target',
          speed_policy='provisional until same-process paired stable measurement using current frozen checkpoints',
          selection_policy='Pareto top2 frozen before seed43/44; no test used for search'))
    write(ROOT/'provenance/source_hashes.json',{str(p.relative_to(q.SRC)):sha(p) for p in (q.SRC/'paddleseg').rglob('*.py')})
    write(ROOT/'provenance/config_hashes.json',{p.name:sha(p) for p in (ROOT/'configs').glob('*.yml')})
    report()
    print('PREPARED',len(pending),'mechanisms;',sum(not r['skip_reason'] for r in pending),'formal attempts planned',flush=True)


def run_command(args,tag,timeout=7200):
    timeout=43200 if tag.endswith('_train') else max(timeout,1800 if tag.endswith('_smoke') else 3600)
    log=ROOT/'logs'/(tag+'.log')
    command=[str(q.PYTHON)]+[str(a) for a in args]
    started=time.time()
    with log.open('wb') as stream:
        child=subprocess.Popen(command,cwd=ROOT,env=q.env(),stdout=stream,stderr=subprocess.STDOUT,
                               creationflags=0x20 if os.name=='nt' else 0)
        write(ROOT/'results/active_process.json',dict(tag=tag,pid=child.pid,runner_pid=os.getpid(),started_at=q.now(),command=command,log=str(log),test_used=False))
        last_update=0;last_change=started;last_size=0;stop_reason=None
        while child.poll() is None:
            elapsed=time.time()-started;size=log.stat().st_size
            if size!=last_size:last_change=time.time();last_size=size
            free=shutil.disk_usage(ROOT).free/2**30
            if free<4:stop_reason='disk_below_4GiB'
            elif elapsed>timeout:stop_reason='stage_timeout'
            elif time.time()-last_change>1800:stop_reason='log_stale_over_30min'
            if stop_reason:
                child.terminate();child.wait(timeout=60);break
            if time.time()-last_update>20:
                found=re.findall(r'iter:\s*(\d+)/18000',q.tail(log))
                write(ROOT/'results/heartbeat.json',dict(tag=tag,pid=child.pid,runner_pid=os.getpid(),timestamp=q.now(),
                    latest_iter=int(found[-1]) if found else None,elapsed_seconds=round(elapsed,1),log_size=size,free_gb=round(free,2)))
                last_update=time.time()
            time.sleep(1)
        code=child.returncode
    write(ROOT/'results'/(tag+'_exit.json'),dict(exit_code=code,stop_reason=stop_reason,elapsed_seconds=round(time.time()-started,2),ended_at=q.now(),command=command,test_used=False))
    print(tag,'exit',code,'reason',stop_reason,flush=True)
    if stop_reason:raise RuntimeError(tag+': '+stop_reason)
    return code


def report():
    if not (ROOT/'provenance/job_specs.json').exists():return
    base=read(ROOT/'results/B0_val.json');parent=read(ROOT/'results/A0_val.json')
    bp=read(ROOT/'results/B0_profile.json')['params_total'];rows=[]
    for job in jobs():
        ident=job['id'];state=ROOT/'results'/(ident+'_status.json')
        row=dict(id=ident,label=job['label'],modules=job['modules'],status=read(state)['status'] if state.exists() else ('compatibility_deferred' if job['skip_reason'] else 'pending'))
        if row['status']=='complete':
            m=read(ROOT/'results'/(ident+'_val.json'));p=read(ROOT/'results'/(ident+'_profile.json'))
            row.update({k:m[k] for k in METRICS})
            row.update(params=p['params_total'],delta_vs_A0={k:m[k]-parent[k] for k in METRICS},delta_vs_B0={k:m[k]-base[k] for k in METRICS})
            d=row['delta_vs_A0'];b=row['delta_vs_B0']
            row['parent_accuracy_gate']=d['crackIoU']>=.002 and d['f1']>=.001 and d['mIoU']>=0 and d['precision']>=-.005 and d['recall']>=-.005 and d['boundary_f1']>=-.005
            row['B0_strict_metrics_params_gate']=b['crackIoU']>=.002 and b['f1']>=.001 and b['mIoU']>=0 and b['recall']>=-.005 and b['boundary_f1']>=-.005 and row['params']<=.95*bp
            row['balanced_metrics_params_gate']=row['parent_accuracy_gate'] and row['params']<=bp
            row['full_target_pass']=None;row['latency_status']='pending stable paired measurement'
        rows.append(row)
    write(ROOT/'results/comparison.json',dict(baseline=base,parent=parent,rows=rows,test_used=False,updated_at=q.now(),all_speed_gates_provisional=True))
    columns=['id','label','status',*METRICS,'params','parent_accuracy_gate','B0_strict_metrics_params_gate','balanced_metrics_params_gate','latency_status']
    with (ROOT/'results/ablation_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    lines=['# Aligned母体第二模块消融进度','','更新时间：'+q.now(),'','所有结果只使用Crack500验证集。待完成、兼容性失败不填性能；旧运行速度仅为临时测量。','',
           '| 编号 | 模块 | 状态 | P% | R% | F1% | IoU% | mIoU% | BF1% | Params M |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for ident in ('B0','W128','A0','GloRe'):
        m=read(ROOT/'results'/(ident+'_val.json'));p=read(ROOT/'results'/(ident+'_profile.json'))
        lines.append('| '+ident+' | 复用冻结对照 | 已核验 | '+' | '.join(f'{m[k]*100:.3f}' for k in METRICS)+f" | {p['params_total']/1e6:.6f} |")
    for r in rows:
        nums=' | '.join(f'{r[k]*100:.3f}' for k in METRICS)+f" | {r['params']/1e6:.6f}" if 'mIoU' in r else '— | — | — | — | — | — | —'
        lines.append(f"| {r['id']} | {r['label']} | {r['status']} | {nums} |")
    lines+=['','24个登记机制、23个当前可尝试项，PSP兼容性暂缓；数量以真实exit0/18k/冻结/独立验证完成数为准。',
            '每项相对Aligned母体和原始B0分别归因；速度确认前full_target_pass为空。筛选结束后匹配三seed、统一同机速度、六个原始模型及最终test。']
    q.versioned_text(ROOT/'refine-logs/EXPERIMENT_RESULTS.md','\n'.join(lines)+'\n')
    current='\n'.join(lines)+'\n\n本轮计划：`SegFormer_Aligned第二模块消融执行计划_20261018.md`。逐图/空间归因在活动工程`diagnostics`中；速度交错复测仍有波动，未确认加速。\n'
    (q.PROJECT/'SegFormer_Aligned第二模块消融进展报告_20261018.md').write_text(current,encoding='utf-8')
    main=q.PROJECT/'模型消融文档.md'
    if main.exists():
        old=main.read_text(encoding='utf-8-sig')
        snapshot=ROOT/'provenance/模型消融文档_进入wave7前.md'
        if not snapshot.exists():snapshot.write_text(old,encoding='utf-8')
        start='<!-- ALIGNED_WAVE7_CURRENT_START -->';end='<!-- ALIGNED_WAVE7_CURRENT_END -->'
        section=start+'\n'+current+end
        if start in old and end in old:
            old=re.sub(re.escape(start)+r'.*?'+re.escape(end),lambda _:section,old,flags=re.S)
        else:old=section+'\n\n---\n\n'+old
        main.write_text(old,encoding='utf-8')
    manifest=['# 实验输出索引','','| 相对路径 | 字节数 |','|---|---:|']
    for folder in ('results','provenance','configs','diagnostics','refine-logs','checkpoints'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and p.suffix not in ('.tmp','.lock'):manifest.append(f'| {p.relative_to(ROOT).as_posix()} | {p.stat().st_size} |')
    (ROOT/'MANIFEST.md').write_text('\n'.join(manifest)+'\n',encoding='utf-8')


def verify_release():
    release=read(ROOT/'provenance/RELEASE.json')
    if release.get('released') is not True or release.get('test_used') is not False:raise RuntimeError('Review release missing or invalid')
    for name,folder in [('source_hashes',q.SRC),('config_hashes',ROOT/'configs')]:
        manifest=ROOT/'provenance'/(name+'.json')
        if sha(manifest)!=release[name+'_sha256']:raise RuntimeError('Release manifest hash changed: '+name)
        for relative,expected in read(manifest).items():
            if sha(folder/relative)!=expected:raise RuntimeError('Frozen source/config changed: '+relative)
    for control in read(ROOT/'provenance/reused_controls.json').values():
        if sha(control['checkpoint'])!=control['sha256']:raise RuntimeError('Reused checkpoint hash changed')


def run_queue():
    verify_release();lock=ROOT/'results/queue.lock'
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),started_at=q.now()),f)
    q.run_command=run_command
    try:
        q.audit_data('before')
        for job in jobs():
            ident=job['id'];state=ROOT/'results'/(ident+'_status.json')
            if job['skip_reason']:
                q.status(ident,'compatibility_deferred',reason=job['skip_reason']);continue
            prior=read(state)['status'] if state.exists() else 'pending'
            if prior in ('complete','smoke_failed','train_failed','needs_review'):continue
            try:
                smoke=ROOT/'results'/(ident+'_smoke_exit.json')
                if not smoke.exists():
                    q.status(ident,'smoke_running')
                    if run_command(q.train_args(ident,True),ident+'_smoke',1800):
                        q.status(ident,'smoke_failed');report();continue
                elif read(smoke)['exit_code']!=0:q.status(ident,'smoke_failed');continue
                receipt=ROOT/'results'/(ident+'_train_exit.json')
                if not receipt.exists():
                    if (ROOT/'training'/ident).exists():raise RuntimeError('Prior incomplete training without real exit; review before retry')
                    q.status(ident,'training')
                    if run_command(q.train_args(ident),ident+'_train'):
                        q.status(ident,'train_failed');report();continue
                elif read(receipt)['exit_code']!=0:q.status(ident,'train_failed');continue
                if not (ROOT/'training'/ident/'iter_18000/model.pdparams').exists() or 'iter: 18000/18000' not in (ROOT/'logs'/(ident+'_train.log')).read_text(encoding='utf-8',errors='replace'):
                    raise RuntimeError('Final18k checkpoint/log absent')
                selection=q.choose_checkpoint(ident);q.status(ident,'verifying',selection=selection)
                q.evaluate(ident,selection['checkpoint']);measured=read(ROOT/'results'/(ident+'_val.json'))
                if abs(measured['mIoU']-selection['selected']['mIoU'])>1e-6:raise RuntimeError('Independent validation mismatch')
                q.status(ident,'complete',selection=selection)
            except Exception as e:
                q.status(ident,'needs_review',error=str(e),traceback=traceback.format_exc());report()
                if 'disk_below' in str(e) or 'log_stale' in str(e):raise
            report()
        q.audit_data('after');report()
        states={j['id']:read(ROOT/'results'/(j['id']+'_status.json'))['status'] for j in jobs()}
        write(ROOT/'results/queue_done.json',dict(updated_at=q.now(),states=states,complete=sum(s=='complete' for s in states.values()),test_used=False,next_step='review failures, paired timing, Pareto top2 and matched seeds before public controls/test'))
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--profile');p.add_argument('--weights')
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.run:run_queue()
    elif a.report:report()
    elif a.profile:q.profile(a.profile,a.weights)
    else:p.error('choose an action')
