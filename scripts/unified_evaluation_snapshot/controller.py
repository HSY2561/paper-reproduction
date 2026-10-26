"""Wait for the separate training workflow, then evaluate frozen models only.

No training, task registration, source edits, outcome-driven test selection,
or automatic deletion is implemented in this controller.
"""
import argparse
import copy
import csv
import ctypes
import difflib
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import traceback

from stage6_common import *


def source_audit():
    names = ['pspnet.py', 'deeplab.py', 'fast_scnn.py', 'ocrnet.py', 'pp_liteseg.py']
    rows = []
    known_pair = ('9ff5697e2389e4917c216f702e2db23fc0a5744001bef26acfd38948df0ee746',
                  '1d2c3c43b478fa1c5dc83dccff237fb69d948966c73c3af9a268f85229385f53')
    for name in names:
        a = SOURCE / 'paddleseg/models' / name
        b = EXTERNAL / 'paddleseg/models' / name
        pair = (sha(a), sha(b))
        exact = pair[0] == pair[1]
        reviewed = name == 'pp_liteseg.py' and pair == known_pair
        rows.append({'file': name, 'wave10_sha256': pair[0], 'external_sha256': pair[1],
                     'identical': exact, 'reviewed_value_equivalent_compatibility_exception': reviewed})
        if not exact:
            text = ''.join(difflib.unified_diff(b.read_text(encoding='utf-8').splitlines(True),
                                               a.read_text(encoding='utf-8').splitlines(True),
                                               fromfile='external/' + name, tofile='wave10/' + name))
            (ROOT / 'provenance' / (name + '.diff')).write_text(text, encoding='utf-8')
        if not exact and not reviewed:
            raise RuntimeError('Unreviewed original-public-model source difference: ' + name)
    write(ROOT / 'provenance/source_comparison.json', {
        'at': now(), 'rows': rows, 'passed': True,
        'exception_review': 'Parent approved exact PP-LiteSeg hash pair: 1x1 bilinear equals broadcast; clone preserves values. No parameter/architecture change. Strict loading and backend val CM checks still required.'})


def public_items():
    p6 = PROJECT / 'experiments/segformer_balance_wave6_20261016'
    specs = [
        ('SegFormer_B0', W4/'configs/public_segformer_b0.yml', W4/'checkpoints/segformer_b0/model.pdparams',
         '07f6048e0a6de174a2c3ca1915df4bcd635565a706d978288fba946ef240f2d5',
         {'initialization': 'official MiT-B0 ImageNet', 'loss': 'CE', 'EMA': True},
         read(p6/'results/B0_val.json')),
        ('OCRNet_HRNetW18', W4/'configs/public_ocrnet_hrnetw18.yml', W4/'checkpoints/ocrnet_hrnetw18/model.pdparams',
         'de78e90ca3d5921c385cc3f82a8a694f8ef4b1d6d055f49e31750c1361416570',
         {'initialization': 'random; HRNet_W18 pretrained=None', 'loss': 'CE main/aux [1,.4]', 'EMA': True},
         {'mIoU': 0.8227357600895859}),
        ('PPLiteSeg_STDC1', W4/'configs/ppliteseg_stdc1_crack500_formal.yml', W4/'checkpoints/ppliteseg_stdc1_crack500_formal/model.pdparams',
         '42da2e8fdb5859589731b494e021fb5705946ac7268462a4618450aad5bd3f12',
         {'initialization': 'official PP_STDCNet1 pretrained', 'loss': 'OHEM three heads [1,1,1]', 'EMA': True},
         {'mIoU': 0.8205094481314313, 'precision': 0.8250009671155241, 'recall': 0.7729951323946548,
          'f1': 0.7981518002411377, 'crackIoU': 0.6641036699986556}),
    ]
    rows = []
    for ident, cfg, ckpt, digest, recipe, reference in specs:
        rows.append({'id': ident, 'family': 'original_public', 'config': str(cfg),
                     'checkpoint': str(ckpt), 'checkpoint_sha256': digest, 'recipe': recipe,
                     'historical_val_reference': reference, 'include_in_test': True,
                     'include_in_profile': True, 'seed': 42})
    for ident, cfg, loss in [('PSPNet_ResNet50','linux_pspnet_r50_crack500.yml','OHEM main/aux [1,.4]'),
                             ('DeepLabV3P_ResNet50','linux_deeplabv3p_r50_crack500.yml','OHEM single head'),
                             ('FastSCNN','linux_fastscnn_crack500.yml','OHEM main/aux [1,.4]')]:
        sel = read(FORMAL/'frozen'/ident/'selection.json')
        for suffix in ['_exit', '_independent_val_exit', '_frozen_test_exit']:
            if read(FORMAL/'results'/(ident+suffix+'.json'))['exit_code'] != 0:
                raise RuntimeError('Successful historical WSL receipt missing: '+ident+suffix)
        rows.append({'id': ident, 'family': 'original_public', 'config': str(FORMAL/'configs'/cfg),
                     'checkpoint': sel['checkpoint'], 'checkpoint_sha256': sel['sha256'],
                     'recipe': {'initialization': 'random; pretrained:null', 'loss': loss, 'EMA': False},
                     'historical_val_reference': {'mIoU': sel['best_val_mIoU']},
                     'include_in_test': True, 'include_in_profile': True, 'seed': 42})
    for row in rows:
        verify_checkpoint(row)
    return rows


def clear_pretrained(value):
    if isinstance(value, dict):
        return {k: None if k == 'pretrained' else clear_pretrained(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clear_pretrained(x) for x in value]
    return value


def config_for(item, split):
    import yaml
    path = ROOT/'configs'/(item['id']+'_'+split+'.yml')
    original = yaml.safe_load(local_path(item['config']).read_text(encoding='utf-8-sig'))
    cfg = copy.deepcopy(original)
    # Only inference paths / unused initialization downloads change. All model
    # keys, head counts, align_corners and feature widths retain their values.
    cfg['model'] = clear_pretrained(cfg['model'])
    dataset = copy.deepcopy(cfg['val_dataset'])
    dataset['dataset_root'] = '${DATA_ROOT}'
    dataset['val_path'] = '${DATA_ROOT}/'+split+'.txt'
    dataset['mode'] = 'val'
    cfg['val_dataset'] = dataset
    if 'train_dataset' in cfg:
        cfg['train_dataset']['dataset_root'] = '${DATA_ROOT}'
        cfg['train_dataset']['train_path'] = '${DATA_ROOT}/train.txt'
    content = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    if path.exists():
        if path.read_text(encoding='utf-8') != content:
            raise RuntimeError('Previously generated config changed: '+str(path))
    else:
        path.write_text(content, encoding='utf-8')
    return path


def prepare():
    for folder in ['configs','logs','results','provenance','refine-logs']:
        (ROOT/folder).mkdir(parents=True, exist_ok=True)
    release = ROOT/'provenance/release.json'
    if release.exists():
        verify_stage_release()
        print('Stage6 already prepared; frozen release verified; no GPU started')
        return
    source_audit()
    pubs = public_items()
    for item in pubs:
        if item['id'] != 'SegFormer_B0':
            config_for(item, 'val')
    write(ROOT/'provenance/public_frozen_manifest.json', {'at':now(), 'items':pubs,
          'test_accessed':False, 'training_repeated':False}, exclusive=True)
    files = [ROOT/'controller.py', ROOT/'stage6_common.py', ROOT/'evaluate_exact.py',
             ROOT/'profile_models.py', ROOT/'protocol.json', ROOT/'run_stage6_once.ps1', ROOT/'README.md', ROOT/'provenance/source_comparison.json',
             ROOT/'provenance/public_frozen_manifest.json']
    files += list((SOURCE/'paddleseg').rglob('*.py'))
    files += list((ROOT/'configs').glob('*.yml'))
    write(release, {'at':now(), 'files':{str(p):sha(p) for p in files},
                   'runtime': {'distro':WSL_DISTRO,'python':WSL_PYTHON,'paddle':PADDLE_VERSION},
                   'GPU_started':False,'test_accessed':False}, exclusive=True)
    print('Prepared frozen stage6 metadata; no Paddle/CUDA imported; no dataset images accessed')


def pid_alive(pid):
    if not pid:
        return False
    if os.name != 'nt':
        raise RuntimeError('This waiting controller must run on Windows')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # invalid PID: definitely gone
            return False
        if error == 5:
            raise RuntimeError('Cannot prove parent-owned process has exited (access denied): '+str(pid))
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise RuntimeError('Cannot read parent-owned process exit status: '+str(pid))
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


def parent_ready():
    done_path = RECOVERY/'results/queue_done.json'
    exit_path = RECOVERY/'results/wrapper_exit.json'
    if exit_path.exists() and read(exit_path).get('exit_code') != 0:
        raise RuntimeError('Parent wrapper ended nonzero; preserve stage6 waiting state')
    if not done_path.exists() or not exit_path.exists():
        return False, 'waiting_parent_queue_done_and_real_wrapper_exit0'
    done, wrapper = read(done_path), read(exit_path)
    if not done.get('stages_1_to_4_complete') or done.get('test_used'):
        raise RuntimeError('Parent completion contract invalid')
    if (RECOVERY/'results/recovery.lock').exists():
        return False, 'waiting_parent_lock_release'
    pids = [wrapper.get('wrapper_pid')]
    if (RECOVERY/'results/active_process.json').exists():
        active = read(RECOVERY/'results/active_process.json')
        pids += [active.get('runner_pid'),active.get('pid')]
    if any(pid_alive(pid) for pid in set(pids) if pid):
        return False, 'waiting_actual_parent_owned_process_exit'
    control_done = CONTROL/'results/queue_done.json'
    control_exit = CONTROL/'results/wrapper_exit.json'
    if control_exit.exists() and read(control_exit).get('exit_code')!=0:
        raise RuntimeError('Initial-weight witness control wrapper ended nonzero')
    if not control_done.exists() or not control_exit.exists():
        return False, 'waiting_additional_B0_initialization_witness_control'
    if any((CONTROL/'results').glob('*.lock')):
        return False, 'waiting_B0_witness_lock_release'
    witness=read(control_done)
    for key in ['baseline_config','baseline_checkpoint','baseline_checkpoint_sha256','baseline_val_json']:
        if not witness.get(key):
            raise RuntimeError('B0 initial-weight witness completion contract missing: '+key)
    cpids=[read(control_exit).get('wrapper_pid')]
    if (CONTROL/'results/active_process.json').exists():
        cactive=read(CONTROL/'results/active_process.json')
        cpids += [cactive.get('runner_pid'),cactive.get('pid')]
    if any(pid_alive(pid) for pid in set(cpids) if pid):
        return False, 'waiting_actual_B0_witness_owned_process_exit'
    if shutil.disk_usage(ROOT).free < 4*2**30:
        return False, 'paused_disk_below_4GiB'
    return True, 'ready'


def wait_parent(wait, max_hours=48):
    start = time.monotonic()
    consecutive = 0
    while True:
        ready, reason = parent_ready()
        write(ROOT/'results/heartbeat.json', {'stage':reason,'at':now(), 'pid':os.getpid(),
                                            'elapsed_wait_s':time.monotonic()-start,
                                            'free_gib':shutil.disk_usage(ROOT).free/2**30,'test_used':False})
        consecutive = consecutive + 1 if ready else 0
        if ready and consecutive >= 2:
            return
        if not wait and not ready:
            raise RuntimeError('Parent is not ready: '+reason)
        if time.monotonic()-start > max_hours*3600:
            raise RuntimeError('Parent wait exceeded 48h; no GPU launched')
        time.sleep(5 if ready else 20)


def freeze_final_manifest():
    target = ROOT/'provenance/final_evaluation_manifest.json'
    if target.exists():
        data = read(target)
        for path,digest in data['selection_evidence_sha256'].items():
            if sha(path) != digest:
                raise RuntimeError('Training/selection evidence changed after final evaluation freeze')
        for item in data['items']:
            verify_checkpoint(item)
            if sha(item['config'])!=item['source_config_sha256']:
                raise RuntimeError('Original inference model config changed after final freeze')
        return data
    freeze_path = RECOVERY/'provenance/frozen_candidates.json'
    seed_path = RECOVERY/'results/three_seed_summary.json'
    done_path = RECOVERY/'results/queue_done.json'
    candidates = read(freeze_path)['candidates']
    if len(candidates) > 2:
        raise RuntimeError('More than two preregistered candidates')
    summary = read(seed_path)
    if not summary.get('controls_reproduced') or summary.get('test_used'):
        raise RuntimeError('Matched controls were not reproduced before final stage')
    items = copy.deepcopy(read(ROOT/'provenance/public_frozen_manifest.json')['items'])
    witness=read(CONTROL/'results/queue_done.json')
    if witness.get('test_used') is True:
        raise RuntimeError('B0 initialization control unexpectedly used test')
    corrected=read(CONTROL/'results/corrected_control_groups.json')
    corrected_groups=corrected.get('groups',corrected)
    if 'B0' not in corrected_groups or {s['seed'] for s in corrected_groups['B0']['seeds']}!={42,43,44}:
        raise RuntimeError('Corrected B0 matched control group is incomplete')
    newbaseline=read(witness['baseline_val_json'])
    seed42=next(x for x in corrected_groups['B0']['seeds'] if x['seed']==42)
    if any(abs(seed42[k]-newbaseline[k])>1e-12 for k in FIELDS):
        raise RuntimeError('Corrected B0 seed42 statistics do not match the new witnessed baseline')
    b0=next(x for x in items if x['id']=='SegFormer_B0')
    b0.update(config=witness['baseline_config'],checkpoint=witness['baseline_checkpoint'],
              checkpoint_sha256=witness['baseline_checkpoint_sha256'],
              historical_val_reference=read(witness['baseline_val_json']),
              recipe={'initialization':'same local ImageNet checkpoint with binary-SHA witness', 'loss':'CE','EMA':True},
              baseline_initialization_witness=str(CONTROL),legacy_B0_seed42_excluded_from_formal_test=True)
    eligible = []; negative = []
    for candidate in candidates:
        group = summary['groups'][candidate['id']]
        passed = group.get('mean_accuracy_gate') is True and group.get('at_least_two_iou_within_tolerance') is True
        record = {'id':candidate['id'],'selection_track':candidate['selection_track'],
                  'params':candidate['params'],'three_seed':group, 'passed':passed}
        (eligible if passed else negative).append(record)
    # Frozen pre-test priority: retain the preregistered primary lightweight
    # track if eligible, otherwise the eligible secondary, otherwise A0.
    primary = next((x['id'] for x in eligible if x['selection_track']=='lightweight_balance'),
                   eligible[0]['id'] if eligible else 'A0')
    for seed in (42,43,44):
        ident = 'A0' if seed==42 else 'A0_s'+str(seed)
        ckpt = WAVE/'checkpoints/A0/model.pdparams' if seed==42 else W7/f'followup_20261019/checkpoints/A0_s{seed}/model.pdparams'
        reference = read(RECOVERY/f'results/A0_control_s{seed}_val.json')
        items.append({'id':ident,'family':'aligned_control','config':str(RECOVERY/'configs/A0_recheck.yml'),
                      'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt),'seed':seed,
                      'historical_val_reference':reference,'include_in_test':seed==42,
                      'include_in_profile':seed==42,
                      'recipe':{'initialization':'common MiT-B0 ImageNet','loss':'CE','EMA':True}})
    for c in candidates:
        if c['id'] not in [x['id'] for x in eligible]:
            continue
        for seed in (42,43,44):
            ident = c['id'] if seed==42 else c['id']+'_s'+str(seed)
            if seed==42:
                ckpt=c['checkpoint'];digest=c['checkpoint_sha256']
                cfg=local_path(c['evidence_root'])/'configs'/(c['run_id']+'.yml')
                reference=read(local_path(c['evidence_root'])/'results'/(c['run_id']+'_val.json'))
            else:
                sel=read(RECOVERY/'checkpoints'/ident/'selection.json')
                ckpt=sel['checkpoint'];digest=sel['sha256'];cfg=RECOVERY/'configs'/(ident+'.yml')
                reference=read(RECOVERY/'results'/(ident+'_val.json'))
            items.append({'id':ident,'candidate_id':c['id'],'family':'final_candidate',
                          'config':str(cfg),'checkpoint':str(ckpt),'checkpoint_sha256':digest,
                          'historical_val_reference':reference,'seed':seed,
                          'include_in_test':seed==42,'include_in_profile':seed==42,
                          'recipe':{'initialization':'common MiT-B0 ImageNet','loss':'CE','EMA':True}})
    for item in items:
        verify_checkpoint(item)
        item['source_config_sha256']=sha(item['config'])
        item['val_config_sha256']=sha(config_for(item,'val'))
    data={'at':now(),'selection_uses_test':False,'candidate_rule':'source-three-seed mean_accuracy_gate and at_least_two_iou_within_tolerance; maximum two',
          'primary_model_frozen_before_test':primary,'primary_rule':'eligible preregistered lightweight track, else eligible secondary, else A0 reference',
          'eligible_candidates':eligible,'negative_candidates_validation_only':negative,'items':items,
          'L01':'independent retained loss alternative; not combined or inserted here',
          'selection_evidence_sha256':{str(p):sha(p) for p in [freeze_path,seed_path,done_path,RECOVERY/'results/wrapper_exit.json',CONTROL/'results/queue_done.json',CONTROL/'results/wrapper_exit.json',CONTROL/'results/corrected_control_groups.json']}}
    write(target,data,exclusive=True)
    return data


def run_logged(script, args, tag, output, timeout=3600):
    verify_stage_release()
    receipt=ROOT/'results'/(tag+'_exit.json'); log=ROOT/'logs'/(tag+'.log')
    context={'manifest_sha256':sha(ROOT/'provenance/final_evaluation_manifest.json'),
             'script_sha256':sha(ROOT/script),'args':args}
    if '--id' in args and '--split' in args:
        config_id=args[args.index('--id')+1];config_split=args[args.index('--split')+1]
        context['config_sha256']=sha(ROOT/'configs'/(config_id+'_'+config_split+'.yml'))
    if receipt.exists():
        old=read(receipt)
        if old.get('exit_code')!=0 or old.get('context')!=context or not output.exists() or sha(output)!=old.get('output_sha256'):
            raise RuntimeError('Prior stage failed/changed; preserve evidence and register a new isolated attempt: '+tag)
        cached=read(output)
        if cached.get('per_image_sha256'):
            per_image=output.with_name(output.stem+'_per_image.csv')
            if not per_image.exists() or sha(per_image)!=cached['per_image_sha256']:
                raise RuntimeError('Completed per-image evidence missing/changed: '+tag)
        return
    if log.exists() or output.exists():
        raise RuntimeError('Partial earlier stage exists without complete receipt: '+tag)
    if shutil.disk_usage(ROOT).free<4*2**30:
        raise RuntimeError('Data disk below4GiB; no new WSL process started')
    command=[WSL_PYTHON,'-u',wsl_path(ROOT/script),*args]
    shell=('export LD_LIBRARY_PATH='+shlex.quote(LD_LIBRARY_PATH)+'; export PYTHONPATH='+shlex.quote(wsl_path(SOURCE))+
           '; export PYTHONUNBUFFERED=1; exec timeout --signal=TERM --kill-after=30s '+str(timeout)+'s '+
           ' '.join(shlex.quote(x) for x in command))
    cmd=['wsl.exe','--distribution',WSL_DISTRO,'--exec','bash','-lc',shell]
    start=time.time();last_update=0
    with log.open('xb') as stream:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        write(ROOT/'results/active_process.json',{'pid':child.pid,'runner_pid':os.getpid(),'tag':tag,'started_at':now(),'command':cmd})
        while child.poll() is None:
            if time.time()-last_update>20:
                write(ROOT/'results/heartbeat.json',{'stage':tag,'at':now(),'pid':child.pid,'runner_pid':os.getpid(),
                    'elapsed_s':time.time()-start,'log_bytes':log.stat().st_size,'test_used':'_test' in tag})
                last_update=time.time()
            if shutil.disk_usage(ROOT).free<4*2**30 and not (ROOT/'results/STOP_REQUEST.json').exists():
                write(ROOT/'results/STOP_REQUEST.json',{'reason':'disk_below_4GiB','at':now()})
            time.sleep(1)
        code=child.returncode
    write(receipt,{'exit_code':code,'started_epoch':start,'ended_at':now(),'elapsed_seconds':time.time()-start,
                   'command':cmd,'context':context,'log_sha256':sha(log),
                   'output_sha256':sha(output) if output.exists() else None},exclusive=True)
    if code!=0 or not output.exists():
        raise RuntimeError('Actual WSL stage failed: '+tag+' exit='+str(code))


def unified_val_gate(data):
    target=ROOT/'results/unified_val_gate.json'
    if target.exists():
        old=read(target)
        if not old['passed']:
            raise RuntimeError('Prior cross-backend gate failed; test remains blocked')
        return old
    observations=[];results={}
    for item in data['items']:
        new=read(ROOT/'results'/(item['id']+'_val.json'));old=item['historical_val_reference'];results[item['id']]=new
        cmold=old.get('confusion_matrix')
        diff=[[new['confusion_matrix'][i][j]-cmold[i][j] for j in range(2)] for i in range(2)] if cmold else None
        observations.append({'id':item['id'],'reference_CM_available':cmold is not None,
                             'CM_difference_new_minus_historical':diff,
                             'delta_metrics':{k:new[k]-old[k] for k in FIELDS if k in old},
                             'BF1_reference_note':'Only current exact val references are comparable; old public BF1 deliberately absent',
                             'new_config_sha256':new['config_sha256']})
    thresholds=read(WAVE/'provenance/protocol.json')['gate_vs_A0'];verdicts=[]
    for c in data['eligible_candidates']:
        pairs=[]
        for seed in (42,43,44):
            ident=c['id'] if seed==42 else c['id']+'_s'+str(seed)
            a0id='A0' if seed==42 else 'A0_s'+str(seed)
            pairs.append({'seed':seed,**{k:results[ident][k]-results[a0id][k] for k in FIELDS}})
        delta={k:statistics.mean(x[k] for x in pairs) for k in FIELDS}
        passmean=all(delta[k]>=v for k,v in thresholds.items())
        passtwo=sum(x['crackIoU']>=-.002 for x in pairs)>=2
        historical=c['three_seed']
        changed=(passmean!=historical['mean_accuracy_gate'] or passtwo!=historical['at_least_two_iou_within_tolerance'])
        verdicts.append({'id':c['id'],'paired_new_backend':pairs,'mean_delta_A0':delta,
                         'new_mean_accuracy_gate':passmean,'new_at_least_two_iou_within_tolerance':passtwo,
                         'historical_mean_accuracy_gate':historical['mean_accuracy_gate'],
                         'historical_at_least_two_iou_within_tolerance':historical['at_least_two_iou_within_tolerance'],
                         'decision_changed':changed})
    passed=all(not r['decision_changed'] for r in verdicts)
    result={'at':now(),'passed':passed,'observations':observations,'candidate_verdicts':verdicts,
            'decision_rule':'Stop before test if unified-backend validation changes any candidate mean/two-seed pass; never replace the candidate using test.',
            'fallback_reference_A0':not data['eligible_candidates'],'test_used':False}
    write(target,result,exclusive=True)
    if not passed:
        raise RuntimeError('Unified WSL validation changes candidate gate; no test allowed')
    return result


def comparison(data):
    rows=[]
    timings=read(ROOT/'results/interleaved_profile.json') if (ROOT/'results/interleaved_profile.json').exists() else None
    for item in data['items']:
        if not item['include_in_test']:
            continue
        val=read(ROOT/'results'/(item['id']+'_val.json'))
        test=read(ROOT/'results'/(item['id']+'_test.json'))
        row={'model':item['id'],'family':item['family'],'primary_frozen':item['id']==data['primary_model_frozen_before_test'],
                     'seed':item['seed'],'initialization':item['recipe']['initialization'],'training_loss':item['recipe']['loss'],
                     'training_EMA':item['recipe']['EMA'],'evaluation_backend':PADDLE_VERSION+' / WSL2 Ubuntu24 / RTX5060 FP32',
                     'val_mIoU':val['mIoU'],'params':test['params'],**{k:test[k] for k in FIELDS},
                     'checkpoint_sha256':item['checkpoint_sha256'],'test_used':True}
        for size in (400,480):
            timing=next((x for x in timings['summary'] if x['model']==item['id'] and x['size']==size),None) if timings else None
            row[f'latency_{size}_median_ms']=timing['median_ms'] if timing else None
            row[f'latency_{size}_paired_cv']=timing['paired_ratio_B0_cv'] if timing else None
        memory=ROOT/'results'/(item['id']+'_memory_profile.json')
        row['memory_profile']=str(memory) if memory.exists() else None
        row['FLOPs_status']='not measured; complete op support not verified'
        rows.append(row)
    result={'at':now(),'rows':rows,'primary_model_frozen_before_test':data['primary_model_frozen_before_test'],
            'negative_candidates_validation_only':data['negative_candidates_validation_only'],
            'training_protocol':'Common18k/seed42/batch4/AdamW6e-5/end_lr0/augmentation/split; original loss/init/EMA differ and are disclosed per row',
            'evaluation_protocol':'Same source, WSL Paddle, frozen weights, 1124 test, official infer restore, globalCM, exact zero-extension2px BF1',
            'SOTA_claim':False,'test_used_for_selection':False,'historical_test_previously_observed':True}
    write(ROOT/'results/comparison.json',result)
    columns=list(rows[0])
    with (ROOT/'results/comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    lines=['# 阶段6：冻结后的统一WSL评价结果','','更新时间：'+now(),'',
           '训练配方不同按行披露；评价后端、源码和精确边界口径统一。主模型在test之前冻结，不依据本表择优。', '',
           '| 模型 | P% | R% | F1% | IoU% | mIoU% | 精确BF1% | Params M | 初始化 | Loss | EMA |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|']
    for r in rows:
        lines.append('| '+r['model']+' | '+' | '.join(f'{100*r[k]:.4f}' for k in FIELDS)+
                     f" | {r['params']/1e6:.6f} | {r['initialization']} | {r['training_loss']} | {r['training_EMA']} |")
    lines += ['', '主模型（test前冻结）：'+data['primary_model_frozen_before_test'],
              '未通过候选仅保留验证负结果：'+', '.join(x['id'] for x in data['negative_candidates_validation_only']),
              '', '旧BF1近似结果不混入此表；计时另见可选profile接口，不把IO整体耗时写成网络延迟。',
              '详细跨后端CM差值：../results/unified_val_gate.json。主三seed及全20融合表保留在../parent_exports。',
              '此目录由独立controller生成，不调用硬编码旧状态的历史总报告builder。']
    (ROOT/'refine-logs/RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    export=ROOT/'parent_exports';export.mkdir(exist_ok=True)
    for p in [RECOVERY/'results/three_seed_summary.json',RECOVERY/'results/consolidated_comparison.json',
              RECOVERY/'results/ablation_table.csv',RECOVERY/'provenance/frozen_candidates.json',
              CONTROL/'results/corrected_control_groups.json']:
        shutil.copy2(p,export/p.name)
    groups=read(RECOVERY/'results/three_seed_summary.json')['groups']
    corrected=read(CONTROL/'results/corrected_control_groups.json')
    corrected_groups=corrected.get('groups',corrected)
    if 'B0' in corrected_groups:
        groups['B0']=corrected_groups['B0']
    seedrows=[]
    for model,g in groups.items():
        for s in g['seeds']:seedrows.append({'model':model,**s})
    with (export/'three_seed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['model','seed',*FIELDS],extrasaction='ignore');writer.writeheader();writer.writerows(seedrows)
    (ROOT/'DELIVERY_INDEX.md').write_text('# 当前统一评价交付索引\n\n- refine-logs/RESULTS.md：最新6公开模型及冻结候选表。\n- results/comparison.json/csv：统一test原值与配方列。\n- results/*_val/test.json及*_per_image.csv：精确指标与全逐图记录。\n- results/*_exit.json：真实WSL退出与输入/输出/日志SHA。\n- provenance/final_evaluation_manifest.json、test_authorization.json：test前冻结与门禁。\n- parent_exports：全20配置与匹配三seed导出。\n- 该索引不代替最终总报告/图压缩包人工审核。\n',encoding='utf-8')


def run(wait=False, profile=False):
    if os.name!='nt':
        raise RuntimeError('Run this controller from the Windows orchestration environment')
    verify_stage_release()
    if (ROOT/'results/queue_done.json').exists():
        if read(ROOT/'results/queue_done.json').get('evaluation_complete'):
            print('Stage6 already complete; no duplicate evaluation/profile process launched',flush=True)
            return
        raise RuntimeError('Prior stage6 completion marker is inconsistent; preserve and review')
    lock=ROOT/'results/controller.lock'
    with lock.open('x',encoding='utf-8') as stream:
        json.dump({'pid':os.getpid(),'started_at':now()},stream)
    try:
        wait_parent(wait)
        data=freeze_final_manifest()
        for item in data['items']:
            config_for(item,'val')
            run_logged('evaluate_exact.py',['--id',item['id'],'--split','val'],item['id']+'_val',ROOT/'results'/(item['id']+'_val.json'))
        unified_val_gate(data)
        test_configs={item['id']:sha(config_for(item,'test')) for item in data['items'] if item['include_in_test']}
        auth=ROOT/'provenance/test_authorization.json'
        if not auth.exists():
            write(auth,{'at':now(),'test_allowed':True,'manifest_sha256':sha(ROOT/'provenance/final_evaluation_manifest.json'),
                        'validation_gate_sha256':sha(ROOT/'results/unified_val_gate.json'),
                        'config_sha256':test_configs,
                        'candidate_primary_frozen':data['primary_model_frozen_before_test'],
                        'not_test_selection':True},exclusive=True)
        for item in data['items']:
            if not item['include_in_test']:
                continue
            config_for(item,'test')
            run_logged('evaluate_exact.py',['--id',item['id'],'--split','test'],item['id']+'_test',ROOT/'results'/(item['id']+'_test.json'))
        comparison(data)
        if profile:
            run_profiles(data)
            comparison(data)
        write(ROOT/'results/queue_done.json',{'at':now(),'evaluation_complete':True,'test_used_for_selection':False,
              'original_public_count':6,'final_candidate_count':len(data['eligible_candidates']),
              'fallback_A0_reference':not data['eligible_candidates'],'profile_requested':profile},exclusive=True)
    except BaseException as exc:
        write(ROOT/'results/attention_required.json',{'at':now(),'error':str(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        if lock.exists():
            lock.unlink()


def run_profiles(data):
    run_logged('profile_models.py',['--interleaved'],'interleaved_profile',ROOT/'results/interleaved_profile.json',3600)
    for item in data['items']:
        if item.get('include_in_profile'):
            run_logged('profile_models.py',['--one',item['id']],item['id']+'_memory_profile',ROOT/'results'/(item['id']+'_memory_profile.json'),900)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--wait',action='store_true')
    parser.add_argument('--run-ready',action='store_true');parser.add_argument('--with-profile',action='store_true')
    parser.add_argument('--profile-ready',action='store_true');args=parser.parse_args()
    if args.prepare:
        prepare()
    elif args.wait or args.run_ready:
        run(args.wait,args.with_profile)
    elif args.profile_ready:
        wait_parent(False);verify_stage_release();run_profiles(manifest())
    else:
        parser.error('Choose --prepare, --wait, --run-ready or --profile-ready')
