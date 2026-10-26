"""Isolated W11 queue: 4 smoke runs followed by 4 cells x 3 seeds.

No test/custom data, resume, result-dependent skipping, or evidence deletion.
The parent controller owns every child it can stop and records real exit codes.
"""
import argparse
import copy
import csv
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
SRC = ROOT / 'src/PaddleSeg-2.10.0'
PYTHON = Path(r'${WINDOWS_ENV}\Scripts\python.exe')
DATA = Path(r'${DATA_ROOT}')
DATA_MANIFEST = PROJECT / 'experiments/local_structure_20260930/provenance/train_val_manifest.json'
A0_WEIGHTS = PROJECT / 'experiments/segformer_lightfusion_wave10_20261024/checkpoints/A0/model.pdparams'
A0_SHA = '000b90f9668e35d0a7cae0fd535134802c5a64a7ced2eab05f869cef726cd5ea'
A0_CM = [[76223138, 763607], [963066, 3644333]]
A0_MIOU = .8281842714310546
SEEDS = (42, 43, 44)
FIELDS = ('precision', 'recall', 'f1', 'crackIoU', 'mIoU', 'boundary_f1',
          'boundary_f1_1px', 'hard_cldice', 'skeleton_precision', 'skeleton_recall')


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def jobs():
    core = [dict(id=f'W11_E{group}_s{seed}', init_id=f'E{group}_s{seed}',
                 group=f'E{group}', seed=seed, use_m1=bool(group % 2),
                 use_m2=group >= 2)
            for seed in SEEDS for group in range(4)]
    controls = [dict(id=f'W11_R0_s{seed}', init_id=f'R0_s{seed}', group='R0',
                     seed=seed, use_m1=False, use_m2=False) for seed in SEEDS]
    return core + controls


def state(stage, **extra):
    write(ROOT / 'results/workflow_state.json',
          dict(stage=stage, updated_at=now(), test_used=False, **extra))


def status(ident, value, **extra):
    write(ROOT / f'results/{ident}_status.json',
          dict(status=value, updated_at=now(), test_used=False, **extra))


def environment():
    result = dict(os.environ)
    result.update(PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE='1',
                  PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
                  TEMP=str(ROOT / 'tmp'), TMP=str(ROOT / 'tmp'),
                  CUDA_CACHE_PATH=str(ROOT / 'tmp/cuda_cache'))
    return result


def prepare():
    import yaml
    import numpy as np
    for folder in ('configs', 'logs', 'results', 'checkpoints', 'training',
                   'provenance/rng', 'tmp/cuda_cache'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    template = yaml.safe_load((PROJECT / 'experiments/segformer_lightfusion_wave10_20261024/configs/baseline_template.yml').read_text(encoding='utf-8-sig'))
    init_manifest_path = ROOT / 'provenance/initialization_manifest.json'
    init_manifest = read(init_manifest_path) if init_manifest_path.exists() else None
    matched_path = ROOT / 'provenance/matched_R0_initialization.json'
    matched = read(matched_path) if matched_path.exists() else None
    entries = []
    for job in jobs():
        cfg = copy.deepcopy(template)
        init = ROOT / f"weights/init/{job['init_id']}.pdparams"
        cfg['model'] = dict(type='SegFormerComplementarity',
                            backbone=dict(type='MixVisionTransformer_B0' if job['group'] == 'R0' else 'MixVisionTransformer_B0_S4FFN768', pretrained=None),
                            embedding_dim=128, num_classes=2, align_corners=False,
                            use_m1=job['use_m1'], use_m2=job['use_m2'], pretrained=str(init))
        text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        dest = ROOT / f"configs/{job['id']}.yml"
        if dest.exists() and dest.read_text(encoding='utf-8') != text:
            raise RuntimeError('Existing different config requires isolation: ' + str(dest))
        if not dest.exists():
            dest.write_text(text, encoding='utf-8')
        entry = dict(job, config=str(dest), config_sha256=sha(dest),
                     initialization=str(init), initialization_sha256=sha(init) if init.exists() else None)
        manifest = matched if job['group'] == 'R0' else init_manifest
        if manifest:
            expected = manifest['runs'][job['init_id']]
            if entry['initialization_sha256'] != expected['sha256']:
                raise RuntimeError('Initialization SHA differs from manifest: ' + job['id'])
        entries.append(entry)
    cfg = copy.deepcopy(template)
    cfg['model'] = dict(type='SegFormer',
                        backbone=dict(type='MixVisionTransformer_B0', pretrained=None),
                        embedding_dim=128, num_classes=2,
                        balance_modules=['aligned'], fusion_variant='base', fusion_rank=32)
    dest = ROOT / 'configs/R0_reproduction.yml'
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    if dest.exists() and dest.read_text(encoding='utf-8') != text:
        raise RuntimeError('Existing R0 config differs')
    if not dest.exists():
        dest.write_text(text, encoding='utf-8')
    # Exactly the Paddle3.4 nranks1 DistributedBatchSampler algorithm: epoch
    # starts at zero and owns a local RandomState(epoch), independent of seed.
    batches = []
    epoch = 0
    while len(batches) < 18000:
        order = np.arange(1896)
        np.random.RandomState(epoch).shuffle(order)
        batches.extend(order.reshape(-1, 4).tolist())
        epoch += 1
    order_path = ROOT / 'provenance/official_sampler_order.json'
    order_record = dict(batch_size=4, samples=1896, epochs=epoch,
                        shuffle='np.random.RandomState(epoch); official epoch starts0; nranks1',
                        train_iters=18000, batches=batches[:18000], test_used=False)
    if not order_path.exists():
        write(order_path, order_record)
    elif read(order_path) != order_record:
        raise RuntimeError('Sampler order manifest differs')
    write(ROOT / 'provenance/job_specs.json', dict(created_at=now(), jobs=entries,
          execution_order='seed42 E0-E3; seed43 E0-E3; seed44 E0-E3; matched R0 seed42,43,44',
          sampler_order=str(order_path), sampler_order_sha256=sha(order_path),
          initializer_manifest_sha256=sha(init_manifest_path) if init_manifest else None,
          matched_R0_manifest_sha256=sha(matched_path) if matched else None,
          test_used=False))
    print('PREPARED 12 core + 3 matched R0 runs; no GPU work launched', flush=True)


def audit_data(tag):
    reference = read(DATA_MANIFEST)
    failures = []
    checked = 0
    for split in ('train', 'val'):
        listing = DATA / (split + '.txt')
        if sha(listing) != reference[split]['list_sha256']:
            failures.append(str(listing))
        for relative, expected in reference[split]['files'].items():
            path = DATA / relative
            if not path.is_file() or sha(path) != expected:
                failures.append(str(path))
            checked += 1
    report = dict(at=now(), checked_files=checked, mismatches=failures,
                  reference=str(DATA_MANIFEST), reference_sha256=sha(DATA_MANIFEST),
                  test_used=False)
    write(ROOT / f'provenance/data_{tag}.json', report)
    if failures or checked != 4488:
        raise RuntimeError('Registered 4488 train/val files or list hashes differ')
    return report


def freeze_release():
    init = read(ROOT / 'provenance/initialization_manifest.json')
    if init['status'] != 'passed':
        raise RuntimeError('Initialization audit has not passed')
    if read(ROOT / 'provenance/matched_R0_initialization.json')['status'] != 'passed':
        raise RuntimeError('Matched R0 initialization audit has not passed')
    specs = read(ROOT / 'provenance/job_specs.json')
    if any(not item['initialization_sha256'] for item in specs['jobs']):
        raise RuntimeError('Run --prepare again after initialization is ready')
    files = list(SRC.rglob('*.py')) + list((ROOT / 'configs').glob('*.yml'))
    files += [ROOT / name for name in ('protocol.json', 'w11_workflow.py', 'train_entry.py',
                                      'evaluate_val_boundary.py', 'run_workflow.ps1')]
    files += [ROOT / 'provenance/job_specs.json', ROOT / 'provenance/initialization_manifest.json',
              ROOT / 'provenance/official_sampler_order.json', Path(init['derived_backbone']['path']),
              Path(init['indices']['path'])]
    files += [Path(item['initialization']) for item in specs['jobs']]
    files += [ROOT / 'provenance/evaluation_versions.json']
    files += [ROOT / 'prepare_initialization.py', ROOT / 'provenance/compression_audit.json',
              ROOT / 'provenance/compression_tensor_ledger.json',
              ROOT / 'provenance/s4ffn768_scores_float64.json']
    files += [ROOT / 'prepare_matched_controls.py', ROOT / 'provenance/matched_R0_initialization.json']
    files += [ROOT / 'provenance/execution_addendum.json',
              ROOT / 'refine-logs/EXECUTION_ADDENDUM_20261025.md']
    files += list((ROOT / 'provenance/initialization_tensors').glob('*.json'))
    files += list((ROOT / 'provenance/matched_R0_tensors').glob('*.json'))
    files += list((ROOT / 'provenance').glob('s4_block*_channel_scores.csv'))
    gate_path = ROOT / 'provenance/execution_release.json'
    if not gate_path.exists() or not read(gate_path).get('approved'):
        raise RuntimeError('Controller CPU/source release evidence is required')
    files.append(gate_path)
    hashes = {str(path.resolve()): sha(path) for path in sorted(set(files))}
    destination = ROOT / 'provenance/release.json'
    if destination.exists():
        old = read(destination)
        if old['files'] != hashes:
            raise RuntimeError('Frozen release differs; isolate and review existing release')
        return
    write(destination, dict(frozen_at=now(), files=hashes,
                            a0_checkpoint=str(A0_WEIGHTS), a0_sha256=A0_SHA,
                            test_used=False))
    print('RELEASE_FROZEN', len(hashes), 'files', sha(destination), flush=True)


def verify_release():
    release = read(ROOT / 'provenance/release.json')
    for path, expected in release['files'].items():
        if not Path(path).is_file() or sha(path) != expected:
            raise RuntimeError('Frozen source/config/init/runtime changed: ' + path)
    if sha(A0_WEIGHTS) != A0_SHA:
        raise RuntimeError('Read-only A0 checkpoint SHA changed')
    gate = read(ROOT / 'provenance/execution_release.json')
    if not gate.get('approved'):
        raise RuntimeError('Execution gate has not passed')
    for evidence in gate.get('evidence', []):
        if sha(evidence['path']) != evidence['sha256']:
            raise RuntimeError('Execution gate evidence changed: ' + evidence['path'])


def tail(path, count=20000):
    with Path(path).open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - count))
        return stream.read().decode('utf-8', errors='replace')


def command(args, tag, timeout=43200):
    """No automatic reattempt or log truncation; an exit0 stage is reusable."""
    receipt = ROOT / f'results/{tag}_exit.json'
    if receipt.exists():
        previous = read(receipt)
        if previous.get('exit_code') != 0 or previous.get('guard_reason'):
            raise RuntimeError('Prior nonzero stage requires manual isolation: ' + tag)
        if previous.get('release_sha256') != sha(ROOT / 'provenance/release.json'):
            raise RuntimeError('Prior stage belongs to a different release: ' + tag)
        return previous
    stdout = ROOT / f'logs/{tag}.stdout.log'
    stderr = ROOT / f'logs/{tag}.stderr.log'
    if stdout.exists() or stderr.exists():
        raise RuntimeError('Unfinished prior stage evidence requires manual isolation: ' + tag)
    verify_release()
    if shutil.disk_usage(ROOT).free < 4 * 2**30:
        raise RuntimeError('Disk below 4GiB before launch; evidence preserved')
    argv = [str(PYTHON), *map(str, args)]
    started = time.time()
    last_progress = started
    last_size = -1
    last_heartbeat = 0
    reason = None
    child = None
    with stdout.open('xb') as out, stderr.open('xb') as err:
        try:
            child = subprocess.Popen(argv, cwd=ROOT, env=environment(), stdout=out, stderr=err)
            active = dict(tag=tag, pid=child.pid, runner_pid=os.getpid(), started_at=now(),
                          command=argv, stdout=str(stdout), stderr=str(stderr),
                          release_sha256=sha(ROOT / 'provenance/release.json'), test_used=False)
            write(ROOT / 'results/active_process.json', active)
            while child.poll() is None:
                current = time.time()
                size = stdout.stat().st_size + stderr.stat().st_size
                if size != last_size:
                    last_size, last_progress = size, current
                if current - started > timeout:
                    reason = 'stage_timeout_' + str(timeout) + 's'
                if current - last_progress > 1800:
                    reason = 'stdout_stderr_stalled_30min'
                if shutil.disk_usage(ROOT).free < 4 * 2**30:
                    reason = 'disk_below_4GiB'
                if reason:
                    # Only the Popen child owned by this controller is stopped.
                    stop = subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                                          capture_output=True, text=True)
                    write(ROOT / f'results/{tag}_owned_stop.json',
                          dict(at=now(), pid=child.pid, guard_reason=reason,
                               taskkill_exit=stop.returncode, stdout=stop.stdout, stderr=stop.stderr))
                    child.wait(timeout=60)
                    break
                if current - last_heartbeat > 20:
                    iterations = re.findall(r'iter:\s*(\d+)/(\d+)', tail(stdout) + tail(stderr))
                    write(ROOT / 'results/heartbeat.json',
                          dict(at=now(), tag=tag, pid=child.pid, runner_pid=os.getpid(),
                               latest_iter=int(iterations[-1][0]) if iterations else None,
                               total_iters=int(iterations[-1][1]) if iterations else None,
                               elapsed_seconds=round(current - started, 2),
                               seconds_since_log_progress=round(current - last_progress, 2),
                               stdout_bytes=stdout.stat().st_size, stderr_bytes=stderr.stat().st_size,
                               free_gib=round(shutil.disk_usage(ROOT).free / 2**30, 3), test_used=False))
                    last_heartbeat = current
                time.sleep(1)
        except BaseException:
            if child is not None and child.poll() is None:
                stop = subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                                      capture_output=True, text=True)
                write(ROOT / f'results/{tag}_exception_owned_stop.json',
                      dict(at=now(), pid=child.pid, taskkill_exit=stop.returncode,
                           stdout=stop.stdout, stderr=stop.stderr))
                child.wait(timeout=60)
            raise
        finally:
            if child is not None:
                result = dict(tag=tag, pid=child.pid, exit_code=child.returncode,
                              guard_reason=reason, ended_at=now(),
                              elapsed_seconds=round(time.time() - started, 2), command=argv,
                              stdout=str(stdout), stderr=str(stderr),
                              release_sha256=sha(ROOT / 'provenance/release.json'), test_used=False)
                write(receipt, result)
                write(ROOT / 'results/active_process.json', dict(result, active=False))
    print(tag, 'exit', result['exit_code'], flush=True)
    if result['exit_code'] != 0 or reason:
        raise RuntimeError(tag + ' failed; preserve and review: ' + str(result))
    return result


def train_args(job, smoke=False):
    ident = job['id']
    return [ROOT / 'train_entry.py', '--config', ROOT / f'configs/{ident}.yml',
            '--save_dir', ROOT / 'training' / (('smoke_' if smoke else '') + ident),
            '--iters', '20' if smoke else '18000', '--do_eval', '--use_ema',
            '--save_interval', '20' if smoke else '1000', '--seed', str(job['seed']),
            '--log_iters', '10' if smoke else '100', '--num_workers', '0',
            '--keep_checkpoint_max', '1', '--precision', 'fp32', '--batch_size', '4']


def verify_rng(group_jobs, smoke=False, full=False):
    records = []
    key = 'smoke_' if smoke else ''
    for job in group_jobs:
        path = ROOT / f"provenance/rng/{key}{job['id']}.json"
        record = read(path)
        if record['status'] != 'complete' or len(record['first_samples']) != 8:
            raise RuntimeError('Missing complete first-two-batch RNG witness: ' + job['id'])
        records.append(record)
    first = records[0]
    passed = all(r['first_samples'] == first['first_samples'] for r in records[1:])
    if full:
        fields = ('actual_training_indices_sha256', 'training_budget_numpy_state_sha256',
                  'training_budget_python_state_sha256')
        passed = passed and all(all(r.get(f) == first.get(f) and r.get(f) is not None for f in fields)
                                for r in records)
        passed = passed and all(r['epoch_first_samples'] == first['epoch_first_samples'] for r in records)
        planned = read(ROOT / 'provenance/official_sampler_order.json')['batches']
        planned_sha = hashlib.sha256(''.join(str(index) + '\n' for batch in planned for index in batch).encode('ascii')).hexdigest()
        passed = passed and first.get('actual_training_indices_sha256') == planned_sha
    report = dict(at=now(), passed=passed, smoke=smoke, full_training=full,
                  runs=[r['run'] for r in records], first_two_batches_identical=passed,
                  witness_files=[str(ROOT / f"provenance/rng/{key}{j['id']}.json") for j in group_jobs],
                  test_used=False)
    control_suffix = '_R0' if any(j['group'] == 'R0' for j in group_jobs) else ''
    write(ROOT / f"provenance/rng/paired_{'smoke' if smoke else 'formal'}{control_suffix}_s{group_jobs[0]['seed']}.json", report)
    if not passed:
        raise RuntimeError('Paired data ordering/augmentation RNG witness differs')


def ensure_fresh_train_dir(ident, tag):
    if (ROOT / 'training' / ident).exists() and not (ROOT / f'results/{tag}_exit.json').exists():
        raise RuntimeError('Existing partial training requires manual isolation: ' + ident)


def choose_checkpoint(job):
    ident = job['id']
    folder = ROOT / 'training' / ident
    candidates = []
    for variant, state_name, weight_name in (
        ('ordinary', 'model.pdstates', 'model.pdparams'),
        ('ema', 'ema_model.pdstates', 'ema_model.pdparams')):
        sub = folder / ('best_model' if variant == 'ordinary' else 'ema_best_model')
        state_path = sub / state_name
        with state_path.open('rb') as stream:
            exact = pickle.load(stream)
        metric = float(exact['mIoU'])
        if not math.isfinite(metric):
            raise RuntimeError('Non-finite best metric: ' + ident)
        weight = sub / weight_name
        candidates.append(dict(variant=variant, mIoU=metric, iter=int(exact['iter']),
                               weight=str(weight), weight_sha256=sha(weight),
                               state_path=str(state_path), state_sha256=sha(state_path)))
    selected = min(candidates, key=lambda row: (-row['mIoU'], row['iter'], row['variant'] != 'ordinary'))
    dest = ROOT / 'checkpoints' / ident
    record_path = dest / 'selection.json'
    if record_path.exists():
        old = read(record_path)
        if old['selected'] != selected or sha(old['checkpoint']) != old['sha256']:
            raise RuntimeError('Existing frozen selection changed: ' + ident)
        return old
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError('Partial checkpoint freeze requires manual isolation: ' + ident)
    dest.mkdir(parents=True, exist_ok=True)
    checkpoint = dest / 'model.pdparams'
    shutil.copy2(selected['weight'], checkpoint)
    for candidate in candidates:
        shutil.copy2(candidate['state_path'], dest / (candidate['variant'] + '_' + Path(candidate['state_path']).name))
    record = dict(run_id=ident, group=job['group'], seed=job['seed'], candidates=candidates,
                  selected=selected, checkpoint=str(checkpoint), sha256=sha(checkpoint),
                  initialization_sha256=sha(ROOT / f"weights/init/{job['init_id']}.pdparams"),
                  config_sha256=sha(ROOT / f'configs/{ident}.yml'), frozen_at=now(),
                  selection='exact val mIoU desc; iteration asc; ordinary before EMA; no boundary selection',
                  release_sha256=sha(ROOT / 'provenance/release.json'), test_used=False)
    write(record_path, record)
    return record


def evaluate(ident, weights):
    cfg = ROOT / f'configs/{ident}.yml'
    command([SRC / 'tools/val.py', '--config', cfg, '--model_path', weights,
             '--num_workers', '0'], ident + '_official_val', 1800)
    command([ROOT / 'evaluate_val_boundary.py', '--config', cfg, '--model_path', weights,
             '--output', ROOT / f'results/{ident}_val'], ident + '_pixel_boundary', 1800)


def baseline():
    state('A0_full348_reproduction')
    evaluate('R0_reproduction', A0_WEIGHTS)
    measured = read(ROOT / 'results/R0_reproduction_val.json')
    passed = (measured['n'] == 348 and measured['confusion_matrix'] == A0_CM
              and abs(measured['mIoU'] - A0_MIOU) < 1e-12)
    report = dict(at=now(), passed=passed, expected_cm=A0_CM, expected_mIoU=A0_MIOU,
                  actual_cm=measured['confusion_matrix'], actual_mIoU=measured['mIoU'],
                  checkpoint=str(A0_WEIGHTS), checkpoint_sha256=sha(A0_WEIGHTS),
                  test_used=False)
    write(ROOT / 'results/R0_reproduction.json', report)
    if not passed:
        raise RuntimeError('A0 source control full348 confusion matrix failed reproduction')


def smoke_matrix():
    matrix = []
    for job in jobs()[:4]:
        ident = job['id']
        state('four_cell_GPU_smoke', active=job)
        ensure_fresh_train_dir('smoke_' + ident, ident + '_smoke')
        command(train_args(job, True), ident + '_smoke', 1800)
        log = tail(ROOT / f'logs/{ident}_smoke.stdout.log', 100000) + tail(ROOT / f'logs/{ident}_smoke.stderr.log', 100000)
        if 'iter: 20/20' not in log or not (ROOT / f'training/smoke_{ident}/iter_20/model.pdparams').exists():
            raise RuntimeError('GPU smoke missing final iteration/checkpoint: ' + ident)
        if re.search(r'loss[:=]\s*(?:nan|inf)\b', log, flags=re.I):
            raise RuntimeError('Non-finite GPU smoke loss: ' + ident)
        matrix.append(dict(run_id=ident, passed=True, iterations=20, exit_code=0))
    verify_rng(jobs()[:4], smoke=True)
    write(ROOT / 'results/smoke_matrix.json', dict(at=now(), passed=True, rows=matrix,
                                                 test_used=False, performance_evidence=False))


def train_one(job):
    ident = job['id']
    prior = ROOT / f'results/{ident}_status.json'
    if prior.exists() and read(prior)['status'] == 'complete':
        selection = read(ROOT / f'checkpoints/{ident}/selection.json')
        if sha(selection['checkpoint']) != selection['sha256']:
            raise RuntimeError('Completed frozen checkpoint changed: ' + ident)
        return
    ensure_fresh_train_dir(ident, ident + '_train')
    state('formal_training', active=job)
    status(ident, 'training', seed=job['seed'], group=job['group'])
    command(train_args(job), ident + '_train')
    log = (ROOT / f'logs/{ident}_train.stdout.log').read_text(encoding='utf-8', errors='replace')
    log += (ROOT / f'logs/{ident}_train.stderr.log').read_text(encoding='utf-8', errors='replace')
    if 'iter: 18000/18000' not in log or not (ROOT / f'training/{ident}/iter_18000/model.pdparams').exists():
        raise RuntimeError('Exit0 lacks full18000 training evidence: ' + ident)
    selection = choose_checkpoint(job)
    status(ident, 'validating', selection=selection)
    evaluate(ident, selection['checkpoint'])
    measured = read(ROOT / f'results/{ident}_val.json')
    if measured['n'] != 348 or abs(measured['mIoU'] - selection['selected']['mIoU']) > 1e-9:
        raise RuntimeError('Independent exact validation differs from selected pdstates: ' + ident)
    status(ident, 'complete', selection=selection, group=job['group'], seed=job['seed'])


def summary():
    rows = []
    grouped = {}
    for job in jobs():
        path = ROOT / f"results/{job['id']}_status.json"
        row = dict(job, status=read(path)['status'] if path.exists() else 'pending')
        if row['status'] == 'complete':
            result = read(ROOT / f"results/{job['id']}_val.json")
            selection = read(ROOT / f"checkpoints/{job['id']}/selection.json")
            row.update({key: result.get(key) for key in FIELDS})
            row.update(checkpoint_sha256=selection['sha256'],
                       initialization_sha256=selection['initialization_sha256'],
                       selected_variant=selection['selected']['variant'],
                       selected_iter=selection['selected']['iter'])
        rows.append(row)
    for group in ('E0', 'E1', 'E2', 'E3', 'R0'):
        complete = [r for r in rows if r['group'] == group and r['status'] == 'complete']
        grouped[group] = dict(complete_seeds=len(complete), seeds=complete)
        if len(complete) == 3:
            valid_fields = [key for key in FIELDS if all(r[key] is not None for r in complete)]
            grouped[group].update(mean={k: statistics.mean(r[k] for r in complete) for k in valid_fields},
                                  sample_sd={k: statistics.stdev(r[k] for r in complete) for k in valid_fields})
    pairs = []
    interactions = []
    for seed in SEEDS:
        cells = {r['group']: r for r in rows if r['seed'] == seed and r['status'] == 'complete'}
        for lhs, rhs in (('E1', 'E0'), ('E2', 'E0'), ('E3', 'E0'), ('E3', 'E1'), ('E3', 'E2')):
            if lhs in cells and rhs in cells:
                pairs.append(dict(seed=seed, contrast=lhs + '-' + rhs,
                                  **{k: cells[lhs][k] - cells[rhs][k] for k in FIELDS
                                     if cells[lhs][k] is not None and cells[rhs][k] is not None}))
        if all(group in cells for group in ('E0', 'E1', 'E2', 'E3')):
            interactions.append(dict(seed=seed, formula='E3-E1-E2+E0',
                                     **{k: cells['E3'][k] - cells['E1'][k] - cells['E2'][k] + cells['E0'][k]
                                        for k in FIELDS if all(cells[g][k] is not None for g in ('E0', 'E1', 'E2', 'E3'))}))
    interaction_summary = None
    if len(interactions) == 3:
        values = [r['crackIoU'] for r in interactions]
        interaction_summary = dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values),
                                   positive_seeds=sum(x > 0 for x in values),
                                   consistent_positive_interaction=statistics.mean(values) > 0 and sum(x > 0 for x in values) >= 2,
                                   statistical_significance_claim=False)
    comparisons = {}
    thresholds = dict(crackIoU=.002, f1=.001, mIoU=0, precision=-.005, recall=-.005, boundary_f1=-.005)
    if all(grouped[g]['complete_seeds'] == 3 for g in ('E0', 'E1', 'E2', 'E3')):
        for lhs, rhs in (('E1', 'E0'), ('E2', 'E0'), ('E3', 'E0'), ('E3', 'E1'), ('E3', 'E2')):
            subset = [r for r in pairs if r['contrast'] == lhs + '-' + rhs]
            means = {k: statistics.mean(r[k] for r in subset) for k in thresholds}
            gates = dict(thresholds)
            if lhs == 'E2':
                gates['boundary_f1'] = .01
            if lhs == 'E3' and rhs != 'E0':
                gates.update(crackIoU=.001, f1=0)
            direction = {k: sum(r[k] > 0 for r in subset) for k in ('crackIoU', 'f1', 'boundary_f1')}
            sign_keys = ('crackIoU',) if lhs == 'E3' and rhs != 'E0' else (('crackIoU', 'f1', 'boundary_f1') if lhs == 'E2' else ('crackIoU', 'f1'))
            comparisons[lhs + '-' + rhs] = dict(mean_delta=means, thresholds=gates,
                mean_effect_gate=all(means[k] >= gates[k] for k in gates),
                positive_seeds=direction, direction_gate=all(direction[k] >= 2 for k in sign_keys))
    lightweight = {}
    if all(grouped[g]['complete_seeds'] == 3 for g in grouped):
        init_manifest = read(ROOT / 'provenance/initialization_manifest.json')
        matched_manifest = read(ROOT / 'provenance/matched_R0_initialization.json')
        for group in ('E0', 'E1', 'E2', 'E3'):
            contrasts = []
            for seed in SEEDS:
                cell = next(r for r in rows if r['group'] == group and r['seed'] == seed)
                control = next(r for r in rows if r['group'] == 'R0' and r['seed'] == seed)
                contrasts.append(dict(seed=seed, **{k: cell[k] - control[k] for k in thresholds}))
            means = {k: statistics.mean(r[k] for r in contrasts) for k in thresholds}
            gate = dict(crackIoU=-.002, f1=-.002, mIoU=-.002, precision=-.005, recall=-.005, boundary_f1=-.005)
            params = init_manifest['runs'][group + '_s42']
            control_params = matched_manifest['runs']['R0_s42']
            registered_ratio = params['registered_numel'] / control_params['registered_numel']
            trainable_ratio = params['trainable_numel'] / control_params['trainable_numel']
            signs = sum(r['crackIoU'] >= -.002 for r in contrasts)
            lightweight[group] = dict(paired_deltas=contrasts, mean_delta=means,
                registered_param_ratio=registered_ratio, trainable_param_ratio=trainable_ratio,
                iou_within_tolerance_seeds=signs,
                passed=registered_ratio <= .9 and trainable_ratio <= .9 and signs >= 2
                       and all(means[k] >= gate[k] for k in gate))
    strong_modules = (all(item['mean_effect_gate'] and item['direction_gate'] for item in comparisons.values())
                      if len(comparisons) == 5 else None)
    result = dict(updated_at=now(), planned_runs=15, core_runs=12, matched_R0_runs=3,
                  complete_runs=sum(r['status'] == 'complete' for r in rows),
                  rows=rows, groups=grouped, paired_deltas=pairs, interactions=interactions,
                  interaction_crackIoU=interaction_summary, preregistered_comparisons=comparisons,
                  matched_R0_lightweight_comparisons=lightweight,
                  supports_two_modules_plus_combination=strong_modules,
                  test_used=False, thresholds_relaxed=False)
    write(ROOT / 'results/four_cell_three_seed_summary.json', result)
    with (ROOT / 'results/four_cell_runs.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'group', 'seed', 'status', *FIELDS,
                 'selected_variant', 'selected_iter', 'checkpoint_sha256', 'initialization_sha256'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    for name, records in (('paired_deltas', pairs), ('interactions', interactions)):
        if records:
            with (ROOT / f'results/{name}.csv').open('w', newline='', encoding='utf-8-sig') as stream:
                columns = list(dict.fromkeys(k for row in records for k in row))
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(records)
    return result


def run(smoke_only=False):
    lock = ROOT / 'results/workflow.lock'
    lock_value = dict(pid=os.getpid(), started_at=now(), source=str(__file__))
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(lock_value, stream)
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        verify_release()
        audit_data('queue_before')
        baseline()
        smoke_matrix()
        if smoke_only:
            state('smoke_complete_formal_not_started')
            return
        for seed in SEEDS:
            seed_jobs = [j for j in jobs() if j['seed'] == seed and j['group'] != 'R0']
            for job in seed_jobs:
                train_one(job)
                summary()
            verify_rng(seed_jobs, full=True)
        for job in [j for j in jobs() if j['group'] == 'R0']:
            train_one(job)
            summary()
            paired = [j for j in jobs() if j['seed'] == job['seed'] and j['group'] in ('E0', 'R0')]
            verify_rng(paired, full=True)
        audit_data('queue_after')
        result = summary()
        write(ROOT / 'results/queue_done.json', dict(at=now(), complete_runs=result['complete_runs'],
              full_training_complete=result['complete_runs'] == 15, profile_pending=True, test_used=False))
        state('all12_core_and3_matchedR0_training_and_val_complete_profile_pending')
    except BaseException as error:
        state('needs_review', error=repr(error), traceback=traceback.format_exc())
        write(ROOT / 'results/workflow_exit.json', dict(at=now(), exit_code=1, error=repr(error), test_used=False))
        raise
    finally:
        # Only this process' own lock is removed. Training evidence is retained.
        if lock.exists() and read(lock).get('pid') == os.getpid():
            lock.unlink()
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    write(ROOT / 'results/workflow_exit.json', dict(at=now(), exit_code=0, test_used=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--freeze-release', action='store_true')
    group.add_argument('--run', action='store_true')
    group.add_argument('--smoke-only', action='store_true')
    group.add_argument('--refresh-report', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.freeze_release:
        freeze_release()
    elif args.refresh_report:
        summary()
    else:
        run(smoke_only=args.smoke_only)
