"""Recover the five pre-registered loss recipes after the parent queue exits.

The only algorithm-source change is storage isolation of official MixedLoss
labels. No objective, coefficient, model, data or budget is changed.
"""
import argparse
import csv
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
MAIN = ROOT.parent
PREV = MAIN.parent / 'segformer_aligned_wave7_20261018'
sys.path.insert(0, str(PREV))
import aligned_queue as a
q = a.q
a.ROOT = ROOT
q.ROOT = ROOT
q.SRC = ROOT / 'src/PaddleSeg-2.10.0'
q.__file__ = __file__
q.run_command = a.run_command
read, write, sha = a.read, a.write, a.sha
FIELDS = a.METRICS


def running(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return False
        raise OSError(error, 'Cannot verify parent process exit')
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


def wait_parent():
    started = time.time()
    while True:
        done = MAIN / 'results/queue_done.json'
        blocked = not done.exists() or (MAIN / 'results/queue.lock').exists()
        if not blocked:
            active = read(MAIN / 'results/active_process.json')
            receipt = MAIN / 'results' / (active['tag'] + '_exit.json')
            blocked = not receipt.exists() or running(active['runner_pid']) or running(active['pid'])
        if not blocked:
            write(ROOT / 'provenance/parent_exit_verified.json', dict(
                checked_at=q.now(), active=active,
                final_stage_receipt=read(receipt), queue_done=read(done),
                note='Parent lock gone and actual runner/child exited; safe to use GPU'))
            return
        write(ROOT / 'results/heartbeat.json', dict(
            stage='waiting_for_parent', timestamp=q.now(),
            elapsed_seconds=round(time.time()-started), gpu_started=False))
        if time.time()-started > 48*3600:
            raise RuntimeError('Parent wait exceeded 48h; do not overlap GPU jobs')
        time.sleep(20)


def verify():
    release = read(ROOT / 'provenance/RELEASE.json')
    for name, expected in release['code'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('Recovery runner changed: ' + name)
    source = read(ROOT / 'provenance/source_hashes.json')
    original = read(MAIN / 'provenance/source_hashes.json')
    changed = [name for name in source if source[name] != original[name]]
    if changed != ['paddleseg\\models\\losses\\mixed_loss.py'] and changed != ['paddleseg/models/losses/mixed_loss.py']:
        raise RuntimeError('Unexpected source differences: '+str(changed))
    for name, expected in source.items():
        if sha(q.SRC / name) != expected:
            raise RuntimeError('Recovery source hash mismatch')
    for name, expected in read(ROOT/'provenance/config_hashes.json').items():
        if sha(ROOT/'configs'/name) != expected or sha(MAIN/'configs'/name) != expected:
            raise RuntimeError('Recovery config diverged from registered recipe')
    control=read(ROOT/'provenance/control.json')
    if sha(control['checkpoint'])!=control['sha256']:
        raise RuntimeError('Control weights changed')
    if not all(row['pass'] for row in read(ROOT/'results/cpu_gradient_preflight.json')['rows']):
        raise RuntimeError('CPU compatibility witness failed')


def report():
    base=read(MAIN/'results/C0_val.json')
    rows=[]
    for job in read(MAIN/'provenance/jobs.json'):
        ident=job['id']
        original=MAIN/'results'/(ident+'_status.json')
        repaired=ROOT/'results'/(ident+'_status.json')
        original_status=read(original)['status'] if original.exists() else 'pending'
        repaired_status=read(repaired)['status'] if repaired.exists() else None
        location=ROOT if repaired_status=='complete' else MAIN
        status='complete_compat_clone' if repaired_status=='complete' else original_status
        row=dict(id=ident,label=job['label'],original_status=original_status,
                 recovery_status=repaired_status,status=status,
                 evidence_root=str(location),test_used=False)
        if status in ['complete','complete_compat_clone']:
            m=read(location/'results'/(ident+'_val.json'))
            p=read(location/'results'/(ident+'_profile.json'))
            row.update({key:m[key] for key in FIELDS},params=p['params_total'])
            delta={key:m[key]-base[key] for key in FIELDS}
            row['delta']=delta
            row['main_gate']=(delta['crackIoU']>=.002 and delta['f1']>=.001 and
                delta['mIoU']>=0 and all(delta[k]>=-.005 for k in ['precision','recall','boundary_f1']))
        rows.append(row)
    write(ROOT/'results/consolidated_comparison.json', dict(
        updated_at=q.now(),parent=base,rows=rows,test_used=False,
        note='Original failure evidence retained; compat_clone changes storage only, not formula'))
    columns=['id','label','status','original_status','recovery_status',*FIELDS,'params','main_gate','evidence_root']
    with (ROOT/'results/loss_ablation_consolidated.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    lines=['# SegFormer＋Aligned：20配置损失消融与兼容恢复','','更新时间：'+q.now(),
        '', '固定Aligned，只优化损失；仅Crack500 train/val。原始反向失败保留，兼容恢复只将每项损失的标签复制，未改公式或系数。',
        '', '| 配置 | 状态 | P% | R% | F1% | 裂缝IoU% | mIoU% | BF1% | Params M | 完整门槛 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|',
        '| C0=A0+CE | 冻结复用 | '+' | '.join(f'{base[k]*100:.4f}' for k in FIELDS)+' | 3.470434 | 对照 |']
    for row in rows:
        if 'mIoU' in row:
            numbers=' | '.join(f'{row[k]*100:.4f}' for k in FIELDS)
            lines.append(f"| {row['id']} ({row['label']}) | {row['status']} | {numbers} | 3.470434 | {row['main_gate']} |")
        else:
            lines.append(f"| {row['id']} ({row['label']}) | {row['status']} /恢复:{row['recovery_status']} | — | — | — | — | — | — | — | 未完成 |")
    lines+=['','20配置不等于20独立算法；全部候选参数和推理图相同。loss不带来额外减参或推理加速。',
        '单seed过门槛也只算候选；全部配置完成后再冻结最多top2，匹配43/44复核，然后继续六个原始对照及统一test。',
        '技术修复只因真实梯度错误触发；原失败日志/状态不覆盖。未完成或工程失败的性能留空。']
    content='\n'.join(lines)+'\n'
    (ROOT/'refine-logs/RESULTS.md').write_text(content,encoding='utf-8')
    (q.PROJECT/'SegFormer损失函数消融进展_20261019.md').write_text(content,encoding='utf-8')
    path=q.PROJECT/'模型消融文档.md'
    text=path.read_text(encoding='utf-8-sig')
    start='<!-- LOSS_WAVE8_START -->';end='<!-- LOSS_WAVE8_END -->'
    section=start+'\n'+content+end
    if start in text and end in text:
        i=text.index(start);j=text.index(end,i)+len(end)
        text=text[:i]+section+text[j:]
    else:text=section+'\n\n'+text
    path.write_text(text,encoding='utf-8')


def run():
    lock=ROOT/'results/queue.lock'
    with lock.open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),started_at=q.now()),f)
    try:
        verify();wait_parent();q.audit_data('before')
        control=read(ROOT/'provenance/control.json')
        if not (ROOT/'results/C0_recheck.json').exists():
            if a.run_command([ROOT/'evaluate_val_boundary.py','--config',ROOT/'configs/C0.yml',
                '--model_path',control['checkpoint'],'--output',ROOT/'results/C0_recheck'],'C0_recheck',3600):
                raise RuntimeError('Recovery C0 recheck failed')
            if read(ROOT/'results/C0_recheck.json')['confusion_matrix']!=read(ROOT/'results/C0_val.json')['confusion_matrix']:
                raise RuntimeError('Recovery source changes control predictions')
        if not (ROOT/'results/gradient_preflight.json').exists():
            if a.run_command([ROOT/'gradient_preflight.py'],'gradient_preflight',3600):
                raise RuntimeError('GPU compatibility preflight engine failed')
        checks={r['id']:r for r in read(ROOT/'results/gradient_preflight.json')['rows']}
        for job in read(ROOT/'provenance/jobs.json'):
            ident=job['id'];status=ROOT/'results'/(ident+'_status.json')
            if status.exists() and read(status)['status']=='complete':continue
            if not checks[ident]['pass']:
                q.status(ident,'gradient_preflight_failed',evidence=checks[ident]);report();continue
            try:
                for smoke in [True,False]:
                    stage=ident+('_smoke' if smoke else '_train')
                    receipt=ROOT/'results'/(stage+'_exit.json')
                    if receipt.exists():
                        if read(receipt)['exit_code']!=0:raise RuntimeError('Preserved nonzero stage '+stage)
                        continue
                    target=ROOT/'training'/(('smoke_' if smoke else '')+ident)
                    if target.exists():raise RuntimeError('Interrupted stage needs audit; no blind restart')
                    q.status(ident,'smoke_running' if smoke else 'training')
                    if a.run_command(q.train_args(ident,smoke),stage):raise RuntimeError('Nonzero stage '+stage)
                if not (ROOT/'training'/ident/'iter_18000/model.pdparams').exists():raise RuntimeError('Missing final checkpoint')
                if 'iter: 18000/18000' not in (ROOT/'logs'/(ident+'_train.log')).read_text(encoding='utf-8',errors='replace'):
                    raise RuntimeError('Missing 18k log evidence')
                selection=q.choose_checkpoint(ident);q.status(ident,'verifying');q.evaluate(ident,selection['checkpoint'])
                metrics=read(ROOT/'results'/(ident+'_val.json'))
                profile=read(ROOT/'results'/(ident+'_profile.json'))
                if abs(metrics['mIoU']-selection['selected']['mIoU'])>1e-6 or profile['params_total']!=3470434:
                    raise RuntimeError('Independent metric/parameter mismatch')
                q.status(ident,'complete',selection=selection,compatibility_version='labels_clone_only')
            except Exception as error:
                q.status(ident,'needs_review',error=str(error),traceback=traceback.format_exc())
                if 'disk_below' in str(error) or 'log_stale' in str(error):raise
            report()
        q.audit_data('after');report()
        states={j['id']:read(ROOT/'results'/(j['id']+'_status.json'))['status'] for j in read(ROOT/'provenance/jobs.json')}
        write(ROOT/'results/queue_done.json',dict(states=states,ended_at=q.now(),test_used=False,
            next_step='Audit merged 20 recipes; freeze main-gate top2 before matched seed43/44; no test selection'))
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--profile');parser.add_argument('--weights')
    args=parser.parse_args()
    if args.run:run()
    elif args.report:report()
    elif args.verify:verify();print('RECOVERY_CPU_CONTRACT_OK',flush=True)
    elif args.profile:q.profile(args.profile,args.weights)
