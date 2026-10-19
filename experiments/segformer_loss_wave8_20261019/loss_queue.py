"""Loss-only ablation queue using official classes and unchanged A0 network."""
import argparse
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'segformer_aligned_wave7_20261018'
sys.path.insert(0,str(PREV))
import aligned_queue as a
q=a.q;a.ROOT=ROOT;q.ROOT=ROOT;q.SRC=ROOT/'src/PaddleSeg-2.10.0';q.__file__=__file__;q.run_command=a.run_command
read,write,sha=a.read,a.write,a.sha
FIELDS=a.METRICS


def prepare():
    import yaml
    from loss_catalog import catalog
    jobs=catalog();template=yaml.safe_load((PREV/'configs/A0.yml').read_text(encoding='utf-8-sig'))
    template['model']['backbone']['pretrained']=str(ROOT/'weights/mit_b0_imagenet.pdparams')
    for job in [dict(id='C0',loss=dict(type='CrossEntropyLoss'))]+jobs:
        cfg=copy.deepcopy(template);cfg['loss']=dict(types=[job['loss']],coef=[1.])
        (ROOT/'configs'/(job['id']+'.yml')).write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    write(ROOT/'provenance/jobs.json',jobs)
    for s in ['val.json','val_per_image.csv','profile.json']:shutil.copy2(PREV/'results'/('A0_'+s),ROOT/'results'/('C0_'+s))
    weight=ROOT.parent/'segformer_balance_wave6_20261016/checkpoints/M13_Aligned/model.pdparams'
    ref=read(PREV/'provenance/reused_controls.json')['A0']
    if sha(weight)!=ref['sha256']:raise RuntimeError('A0 reference SHA mismatch')
    write(ROOT/'provenance/control.json',dict(checkpoint=str(weight),sha256=ref['sha256'],seed42_metrics=read(ROOT/'results/C0_val.json'),
        source=str(PREV/'provenance/reused_controls.json'),three_seed_reference=read(PREV/'followup_20261019/results/three_seed_summary.json')['groups']['A0']))
    write(ROOT/'provenance/source_hashes.json',{str(p.relative_to(q.SRC)):sha(p) for p in (q.SRC/'paddleseg').rglob('*.py')})
    write(ROOT/'provenance/config_hashes.json',{p.name:sha(p) for p in (ROOT/'configs').glob('*.yml')})
    write(ROOT/'provenance/protocol.json',dict(parent='SegFormer-B0 width128+Aligned',params=3470434,seed=42,iters=18000,batch=4,
        optimizer='AdamW',lr=6e-5,end_lr=0,model_precision='FP32',test_used=False,train=1896,val=348,
        initialization='common ImageNet backbone, not fine-tuning trained C0',target_completed=20,
        trials='20 main configurations, 2 preregistered engineering-only reserves; not 20 independent algorithms',
        changes='only official loss config; fixed architecture, crops, augmentation, optimizer and training budget',
        main_gate={'crackIoU':.002,'f1':.001,'mIoU':0,'precision':0,'recall':-.005,'boundary_f1':-.005},
        precision_track={'precision':.005,'crackIoU':0,'f1':0,'mIoU':0,'recall':-.005,'boundary_f1':-.005},
        selection='exact ordinary/EMA maximum val mIoU, frozen SHA, independent val/boundary; top2 recipes frozen before seeds43/44',
        efficiency='network is identical; loss does not reduce parameters or accelerate inference; record training cost separately'))
    report();print('PREPARED_20_MAIN_2_ENGINEERING_RESERVES',flush=True)


def report():
    if not (ROOT/'provenance/jobs.json').exists():return
    base=read(ROOT/'results/C0_val.json');rows=[]
    for job in read(ROOT/'provenance/jobs.json'):
        ident=job['id'];p=ROOT/'results'/(ident+'_status.json')
        r=dict(job,status=read(p)['status'] if p.exists() else ('reserve' if job['role']=='reserve' else 'pending'))
        if r['status']=='complete':
            m=read(ROOT/'results'/(ident+'_val.json'));profile=read(ROOT/'results'/(ident+'_profile.json'))
            r.update({k:m[k] for k in FIELDS});r['params']=profile['params_total']
            d={k:m[k]-base[k] for k in FIELDS};r['delta']=d
            r['main_gate']=d['crackIoU']>=.002 and d['f1']>=.001 and d['mIoU']>=0 and d['precision']>=0 and d['recall']>=-.005 and d['boundary_f1']>=-.005
            r['precision_gate']=d['precision']>=.005 and d['crackIoU']>=0 and d['f1']>=0 and d['mIoU']>=0 and d['recall']>=-.005 and d['boundary_f1']>=-.005
            r['full_final_status']='pending matched-seed verification and final protocol'
        rows.append(r)
    write(ROOT/'results/comparison.json',dict(parent=base,rows=rows,updated_at=q.now(),test_used=False))
    columns=['id','label','family','role','status',*FIELDS,'params','main_gate','precision_gate']
    with (ROOT/'results/loss_ablation.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    lines=['# SegFormer＋Aligned 损失函数消融进度','','更新时间：'+q.now(),'','仅Crack500验证集；20主方案＋2技术备选。所有模型结构和推理参数相同。权重变体不计独立算法。','',
        '| 编号 | 配方 | 状态 | P% | R% | F1% | IoU% | mIoU% | BF1% | Params M |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|',
        '| C0 | A0 + CE | 冻结复用 | '+' | '.join(f'{base[k]*100:.4f}' for k in FIELDS)+' | 3.470434 |']
    for r in rows:
        numbers=' | '.join(f'{r[k]*100:.4f}' for k in FIELDS)+f" | {r['params']/1e6:.6f}" if 'mIoU' in r else '— | — | — | — | — | — | —'
        lines.append(f"| {r['id']} | {r['label']} | {r['status']} | {numbers} |")
    lines+=['','普通/EMA按最高精确val mIoU选模，冻结后独立验证。筛选不读test或自建数据；失败不填性能。','不同损失的绝对标量不可横向当作精度；参数和推理算子不变，不能将损失优化称为减参或加速。']
    content='\n'.join(lines)+'\n';q.versioned_text(ROOT/'refine-logs/RESULTS.md',content)
    (q.PROJECT/'SegFormer损失函数消融进展_20261019.md').write_text(content,encoding='utf-8')
    target=q.PROJECT/'模型消融文档.md'
    if target.exists():
        text=target.read_text(encoding='utf-8-sig');start='<!-- LOSS_WAVE8_START -->';end='<!-- LOSS_WAVE8_END -->'
        section=start+'\n'+content+end
        if start in text and end in text:
            i=text.index(start);j=text.index(end,i)+len(end);text=text[:i]+section+text[j:]
        else:text=section+'\n\n---\n\n'+text
        target.write_text(text,encoding='utf-8')
    manifest=['# 输出索引','','| 路径 | 字节 |','|---|---:|']
    for folder in ['configs','provenance','results','logs','checkpoints','refine-logs']:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and p.suffix not in ['.tmp','.lock']:manifest.append(f'| {p.relative_to(ROOT).as_posix()} | {p.stat().st_size} |')
    (ROOT/'MANIFEST.md').write_text('\n'.join(manifest)+'\n',encoding='utf-8')


def verify():
    done=read(PREV/'followup_20261019/results/queue_done.json')
    receipt=read(PREV/'followup_20261019/results/scheduled_followup_exit.json')
    if not all(s=='complete' for s in done['states'].values()) or receipt['exit_code']!=0:raise RuntimeError('Previous queue has not completed successfully')
    for name,folder in [('source_hashes',q.SRC),('config_hashes',ROOT/'configs')]:
        for relative,expected in read(ROOT/'provenance'/(name+'.json')).items():
            if sha(folder/relative)!=expected:raise RuntimeError('Frozen source/config changed: '+relative)
    control=read(ROOT/'provenance/control.json')
    if sha(control['checkpoint'])!=control['sha256']:raise RuntimeError('Control weights changed')


def run():
    verify();lock=ROOT/'results/queue.lock'
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),started_at=q.now()),f)
    try:
        q.audit_data('before')
        if not (ROOT/'results/C0_recheck.json').exists():
            control=read(ROOT/'provenance/control.json')
            if a.run_command([ROOT/'evaluate_val_boundary.py','--config',ROOT/'configs/C0.yml','--model_path',control['checkpoint'],'--output',ROOT/'results/C0_recheck'],'C0_recheck',3600):raise RuntimeError('New source control recheck failed')
            if read(ROOT/'results/C0_recheck.json')['confusion_matrix']!=read(ROOT/'results/C0_val.json')['confusion_matrix']:raise RuntimeError('C0 no longer reproduces')
        if not (ROOT/'results/gradient_preflight.json').exists():
            if a.run_command([ROOT/'gradient_preflight.py'],'gradient_preflight',3600):raise RuntimeError('Preflight engine failed')
        checks={r['id']:r for r in read(ROOT/'results/gradient_preflight.json')['rows']}
        jobs=read(ROOT/'provenance/jobs.json')
        completed=0
        for job in jobs:
            ident=job['id'];state=ROOT/'results'/(ident+'_status.json')
            if state.exists() and read(state)['status']=='complete':completed+=1;continue
            if job['role']=='reserve' and completed>=20:q.status(ident,'reserve_not_needed');continue
            if not checks[ident]['pass']:q.status(ident,'gradient_preflight_failed',evidence=checks[ident]);report();continue
            if state.exists() and read(state)['status'] in ['smoke_failed','train_failed','needs_review']:continue
            try:
                smoke=ROOT/'results'/(ident+'_smoke_exit.json')
                if not smoke.exists():
                    q.status(ident,'smoke_running')
                    if a.run_command(q.train_args(ident,True),ident+'_smoke',1800):q.status(ident,'smoke_failed');report();continue
                elif read(smoke)['exit_code']!=0:q.status(ident,'smoke_failed');continue
                receipt=ROOT/'results'/(ident+'_train_exit.json')
                if not receipt.exists():
                    if (ROOT/'training'/ident).exists():raise RuntimeError('Prior interrupted run: do not restart blindly')
                    q.status(ident,'training')
                    if a.run_command(q.train_args(ident),ident+'_train'):q.status(ident,'train_failed');report();continue
                elif read(receipt)['exit_code']!=0:q.status(ident,'train_failed');continue
                if not (ROOT/'training'/ident/'iter_18000/model.pdparams').exists() or 'iter: 18000/18000' not in (ROOT/'logs'/(ident+'_train.log')).read_text(encoding='utf-8',errors='replace'):raise RuntimeError('Missing final18k evidence')
                selection=q.choose_checkpoint(ident);q.status(ident,'verifying');q.evaluate(ident,selection['checkpoint'])
                result=read(ROOT/'results'/(ident+'_val.json'));profile=read(ROOT/'results'/(ident+'_profile.json'))
                if abs(result['mIoU']-selection['selected']['mIoU'])>1e-6:raise RuntimeError('Independent validation mismatch')
                if profile['params_total']!=3470434:raise RuntimeError('Architecture unexpectedly changed')
                q.status(ident,'complete',selection=selection);completed+=1
            except Exception as e:
                q.status(ident,'needs_review',error=str(e),traceback=traceback.format_exc());report()
                if 'disk_below' in str(e) or 'log_stale' in str(e):raise
            report()
        q.audit_data('after');report()
        states={j['id']:read(ROOT/'results'/(j['id']+'_status.json'))['status'] for j in jobs}
        write(ROOT/'results/queue_done.json',dict(states=states,complete=completed,ended_at=q.now(),test_used=False,next_step='freeze top2 qualifying loss recipes; matched seeds43/44; compare original public models only after this selection'))
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--profile');p.add_argument('--weights');args=p.parse_args()
    if args.prepare:prepare()
    elif args.run:run()
    elif args.report:report()
    elif args.profile:q.profile(args.profile,args.weights)
