"""Dataset-free W11 efficiency measurements and evidence/report packaging.

This is a separately versioned postprocessing tool. It neither changes the
training controller/release nor starts training. GPU commands require all 15
registered runs to be completed and their frozen checkpoints to be verified.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src' / 'PaddleSeg-2.10.0'
PYTHON = Path(r'C:\venv_bisenet\Scripts\python.exe')
GROUPS = ('R0', 'E0', 'E1', 'E2', 'E3')
SIZES = (400, 480)
METRICS = ('precision', 'recall', 'f1', 'crackIoU', 'mIoU', 'boundary_f1',
           'boundary_f1_1px', 'hard_cldice', 'skeleton_precision', 'skeleton_recall')
SUMMARY = ROOT / 'results' / 'four_cell_three_seed_summary.json'
FINAL_REPORT = ROOT.parent.parent / 'SegFormer_E0-E3两模块互补实验报告_20261025.md'
RESULT_REPORT = ROOT / 'refine-logs' / 'EXPERIMENT_RESULTS.md'
LATEST = ROOT / 'results' / 'postprocess_latest.json'


def now():
    return dt.datetime.now().astimezone().isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def csv_write(path, records):
    records = list(records)
    fields = list(dict.fromkeys(key for row in records for key in row))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def command_capture(command, timeout=20):
    try:
        receipt = subprocess.run([str(x) for x in command], capture_output=True,
                                 text=True, encoding='utf-8', errors='replace',
                                 timeout=timeout)
        return dict(command=[str(x) for x in command], exit_code=receipt.returncode,
                    stdout=receipt.stdout, stderr=receipt.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=[str(x) for x in command], exit_code=None,
                    stdout=getattr(error, 'stdout', None), stderr=str(error))


def telemetry():
    return dict(at=now(), pid=os.getpid(), gpu=command_capture([
        'nvidia-smi', '--query-gpu=index,name,driver_version,pstate,temperature.gpu,'
        'clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,memory.used',
        '--format=csv,noheader,nounits']),
        compute_processes=command_capture([
            'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits']))


def assert_idle(snapshot):
    receipt = snapshot['compute_processes']
    if receipt['exit_code'] != 0:
        raise RuntimeError('Cannot verify idle GPU: ' + str(receipt))
    foreign = []
    for line in (receipt.get('stdout') or '').splitlines():
        first = line.split(',', 1)[0].strip()
        if first.isdigit() and int(first) != os.getpid():
            foreign.append(line)
    if foreign:
        raise RuntimeError('Foreign GPU compute processes; leave them running and defer profile: '
                           + repr(foreign))
    return True


def guard_disk():
    if shutil.disk_usage(ROOT).free < 4 * 2 ** 30:
        raise RuntimeError('Less than 4GiB free: defer postprocessing, delete nothing')
    if (ROOT / 'results' / 'STOP_REQUEST.json').exists():
        raise RuntimeError('STOP_REQUEST present')


def load_summary():
    if not SUMMARY.is_file():
        raise FileNotFoundError('Controller summary not present: ' + str(SUMMARY))
    result = read(SUMMARY)
    if result.get('test_used'):
        raise RuntimeError('W11 postprocessing expects validation-only summary')
    return result


def verify_training_release():
    path = ROOT / 'provenance' / 'release.json'
    release = read(path)
    mismatches = [filename for filename, digest in release['files'].items()
                  if not Path(filename).is_file() or sha(filename) != digest]
    if mismatches:
        raise RuntimeError('Frozen training release changed: ' + repr(mismatches))
    return dict(path=str(path), sha256=sha(path), files_verified=len(release['files']))


def checkpoint_evidence(group, seed=42):
    ident = f'W11_{group}_s{seed}'
    folder = ROOT / 'checkpoints' / ident
    selection = read(folder / 'selection.json')
    path = folder / 'model.pdparams'
    config = ROOT / 'configs' / (ident + '.yml')
    digest = sha(path)
    if digest != selection['sha256'] or sha(config) != selection['config_sha256']:
        raise RuntimeError('Frozen checkpoint/config changed: ' + ident)
    if Path(selection['checkpoint']).resolve() != path.resolve():
        raise RuntimeError('Unexpected frozen checkpoint path: ' + ident)
    return dict(group=group, seed=seed, id=ident, checkpoint=str(path),
                sha256=digest, config=str(config), config_sha256=sha(config),
                selection=str(folder / 'selection.json'),
                selection_sha256=sha(folder / 'selection.json'))


def require_complete(summary):
    guard_disk()
    expected = {(group, seed) for group in GROUPS for seed in (42, 43, 44)}
    complete = {(row['group'], row['seed']) for row in summary['rows']
                if row.get('status') == 'complete'}
    if summary.get('complete_runs') != 15 or complete != expected:
        raise RuntimeError('Full GPU profile requires all 15 complete runs; currently '
                           + str(summary.get('complete_runs')))
    done = read(ROOT / 'results' / 'queue_done.json')
    if not done.get('full_training_complete') or done.get('complete_runs') != 15:
        raise RuntimeError('Missing final queue_done full-training receipt')
    if read(ROOT / 'results' / 'workflow_exit.json').get('exit_code') != 0:
        raise RuntimeError('Missing controller exit0')
    if (ROOT / 'results' / 'workflow.lock').exists():
        raise RuntimeError('Training workflow lock still present')
    evidence = []
    for row in summary['rows']:
        ident = row['id']
        status = read(ROOT / 'results' / (ident + '_status.json'))
        receipt = read(ROOT / 'results' / (ident + '_train_exit.json'))
        if status.get('status') != 'complete' or receipt.get('exit_code') != 0:
            raise RuntimeError('Incomplete real training receipt: ' + ident)
        frozen = checkpoint_evidence(row['group'], row['seed'])
        if row.get('checkpoint_sha256') != frozen['sha256']:
            raise RuntimeError('Summary checkpoint SHA mismatch: ' + ident)
        evidence.append(frozen)
    return dict(training_release=verify_training_release(), checkpoints=evidence,
                test_read=False, images_read=False)


def tensor_counts(model):
    values = list(model.parameters())
    registered = sum(math.prod(value.shape) for value in values)
    trainable = sum(math.prod(value.shape) for value in values if not value.stop_gradient)
    return dict(registered=registered, trainable=trainable,
                nontrainable_statistics=registered - trainable)


def build_gpu_model(group):
    # Config parsing/building is lazy: only builder.model is accessed, never a dataset.
    sys.path.insert(0, str(SRC))
    os.environ['PYTHONPATH'] = str(SRC)
    import paddle
    from paddleseg.cvlibs import Config, SegBuilder
    from paddleseg.models.backbones.mit_b0_s4ffn768 import load_exact_state
    import paddleseg
    if not str(Path(paddleseg.__file__).resolve()).startswith(str(SRC.resolve())):
        raise RuntimeError('Wrong editable PaddleSeg source')
    if not paddle.is_compiled_with_cuda():
        raise RuntimeError('CUDA Paddle required for actual GPU profile')
    paddle.set_device('gpu:0')
    item = checkpoint_evidence(group)
    config = Config(item['config'])
    config.dic['model']['pretrained'] = None
    config.dic['model']['backbone']['pretrained'] = None
    model = SegBuilder(config).model
    load_exact_state(model, item['checkpoint'])
    model.eval()
    if any(str(value.dtype) != 'paddle.float32' for value in model.parameters()):
        raise RuntimeError('Profile requires actual FP32 parameters')
    return paddle, model, dict(**item, counts=tensor_counts(model),
                               source=str(paddleseg.__file__))


def percentile(values, q):
    values = sorted(float(x) for x in values)
    if not values:
        return None
    position = (len(values) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    return values[low] + (values[high] - values[low]) * (position - low)


def coefficient_of_variation(values):
    return statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else None


def process_state(pid):
    """Read-only PID query; never terminate an observed training/user process."""
    if not isinstance(pid, int) or pid <= 0:
        return dict(pid=pid, running=None, reason='PID not recorded')
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            return dict(pid=pid, running=False if error in (87, 1168) else None,
                        winerror=error, reason='OpenProcess read-only query')
        try:
            code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return dict(pid=pid, running=None, winerror=ctypes.get_last_error())
            return dict(pid=pid, running=code.value == 259, observed_exit_code=code.value)
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return dict(pid=pid, running=True)
    except ProcessLookupError:
        return dict(pid=pid, running=False)
    except PermissionError:
        return dict(pid=pid, running=None, reason='permission denied')


def wait_and_profile(max_seconds=24 * 3600):
    """Wait without holding a GPU context, then own exactly one postprocess child."""
    started = time.time()
    waiter_lock = ROOT / 'results' / 'postprocess_wait.lock'
    state_path = ROOT / 'results' / 'postprocess_wait_state.json'
    controller_pid = None
    with waiter_lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), at=now(), timeout_seconds=max_seconds), stream)
    if os.name == 'nt':
        # Keep this scheduler helper at normal priority; no visible window is launched here.
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x20):
            raise RuntimeError('Cannot set owned waiter to normal priority')
    try:
        while time.time() - started < max_seconds:
            guard_disk()
            summary = load_summary()
            workflow_lock = ROOT / 'results' / 'workflow.lock'
            if workflow_lock.exists():
                controller_pid = read(workflow_lock).get('pid', controller_pid)
            heartbeat_path = ROOT / 'results' / 'heartbeat.json'
            if controller_pid is None and heartbeat_path.exists():
                controller_pid = read(heartbeat_path).get('runner_pid')
            workflow_exit = ROOT / 'results' / 'workflow_exit.json'
            workflow_state = ROOT / 'results' / 'workflow_state.json'
            failed = (workflow_exit.exists() and read(workflow_exit).get('exit_code') not in (None, 0)
                      and (workflow_exit.stat().st_mtime >= started or
                           (workflow_state.exists() and read(workflow_state).get('stage') == 'needs_review')))
            if failed:
                write(state_path, dict(at=now(), status='blocked_training_failure',
                    workflow_exit=read(workflow_exit), GPU_started=False, training_restarted=False))
                return 2
            state = dict(at=now(), status='waiting', completed_runs=summary.get('complete_runs'),
                         elapsed_seconds=round(time.time() - started, 1),
                         maximum_seconds=max_seconds, poll_seconds=30, GPU_started=False,
                         workflow_lock_present=workflow_lock.exists(), controller_pid=controller_pid,
                         training_restarted=False)
            if summary.get('complete_runs') == 15 and not workflow_lock.exists():
                # queue_done plus workflow exit still do not prove the wrapper/PIDs have exited.
                done_path = ROOT / 'results' / 'queue_done.json'
                wrapper_receipts = []
                if done_path.exists():
                    for candidate in (ROOT / 'results').glob('controller_*_exit.json'):
                        receipt = read(candidate)
                        if receipt.get('mode') == 'run' and candidate.stat().st_mtime >= done_path.stat().st_mtime - 1:
                            wrapper_receipts.append((candidate, receipt))
                if wrapper_receipts:
                    wrapper_path, wrapper = max(wrapper_receipts, key=lambda pair: pair[0].stat().st_mtime)
                    if wrapper.get('exit_code') != 0:
                        write(state_path, dict(at=now(), status='blocked_controller_wrapper_failure',
                              receipt=str(wrapper_path), actual_exit_code=wrapper.get('exit_code'),
                              GPU_started=False, training_restarted=False))
                        return 2
                    active_path = ROOT / 'results' / 'active_process.json'
                    active = read(active_path) if active_path.exists() else {}
                    checks = [process_state(controller_pid), process_state(active.get('pid')),
                              process_state(wrapper.get('runner_pid'))]
                    state.update(wrapper_exit_receipt=str(wrapper_path), process_checks=checks,
                                 last_child_active_flag=active.get('active'))
                    if active.get('active') is False and active.get('exit_code') == 0 and all(
                            check['running'] is False for check in checks):
                        require_complete(summary)
                        state.update(status='training_finished_launching_profile', all_training_processes_exited=True)
                        write(state_path, state)
                        command = [str(PYTHON), str(Path(__file__).resolve()), '--profile-and-report']
                        stdout = ROOT / 'logs' / 'postprocess_after_training.stdout.log'
                        stderr = ROOT / 'logs' / 'postprocess_after_training.stderr.log'
                        if stdout.exists() or stderr.exists():
                            raise RuntimeError('Prior automatic profile logs exist; preserve and review before retry')
                        with stdout.open('x', encoding='utf-8') as out, stderr.open('x', encoding='utf-8') as err:
                            child = subprocess.Popen(command, stdout=out, stderr=err, cwd=ROOT,
                                env=dict(os.environ, PYTHONPATH=str(SRC), PYTHONIOENCODING='utf-8'))
                            remaining = max(1, int(max_seconds - (time.time() - started)))
                            timed_out = False
                            try:
                                child.wait(timeout=remaining)
                            except subprocess.TimeoutExpired:
                                timed_out = True
                                # Stop only this owned postprocess process tree, including its memory children.
                                if os.name == 'nt':
                                    stop = command_capture(['taskkill', '/PID', str(child.pid), '/T', '/F'])
                                    write(ROOT / 'results' / 'postprocess_owned_timeout_stop.json', stop)
                                else:
                                    child.kill()
                                child.wait(timeout=60)
                        receipt = dict(at=now(), command=command, pid=child.pid,
                                       exit_code=child.returncode, timed_out=timed_out,
                                       stdout=str(stdout), stderr=str(stderr), training_restarted=False)
                        write(ROOT / 'results' / 'automatic_postprocess_child_exit.json', receipt)
                        delivered = child.returncode == 0 and not timed_out
                        write(state_path, dict(at=now(), status='complete' if delivered else 'blocked_profile_failure',
                              child_receipt=receipt, training_restarted=False))
                        if delivered:
                            latest = read(LATEST)
                            delivered = latest.get('stage') == 'full_profile_and_report_complete'
                            write(ROOT / 'results' / 'final_delivery_done.json', dict(
                                at=now(), complete=delivered, postprocess=latest,
                                actual_child_exit_code=child.returncode,
                                notification_sent=False, original_controller_profile_pending_unchanged=True))
                        return 0 if delivered else 2
                else:
                    state['waiting_reason'] = 'Wait for the actual run-mode PowerShell wrapper exit receipt'
            write(state_path, state)
            time.sleep(30)
        write(state_path, dict(at=now(), status='blocked_24h_wait_limit', maximum_seconds=max_seconds,
                              GPU_started=False, training_restarted=False))
        return 2
    except BaseException as error:
        write(state_path, dict(at=now(), status='blocked_wait_or_profile_error', error=repr(error),
                              traceback=traceback.format_exc(), training_restarted=False))
        raise
    finally:
        if waiter_lock.exists() and read(waiter_lock).get('pid') == os.getpid():
            waiter_lock.unlink()


def memory_one(group, size, output):
    summary = load_summary()
    require_complete(summary)
    before = telemetry()
    assert_idle(before)
    paddle, model, item = build_gpu_model(group)
    cuda = paddle.device.cuda
    x = paddle.randn([1, 3, size, size], dtype='float32')
    api_names = ('memory_allocated', 'memory_reserved', 'max_memory_allocated',
                 'max_memory_reserved', 'reset_max_memory_allocated',
                 'reset_max_memory_reserved', 'empty_cache')
    apis = {name: hasattr(cuda, name) for name in api_names}
    def api(name):
        return int(getattr(cuda, name)()) if apis.get(name) else None
    with paddle.no_grad():
        for _ in range(100):
            model(x)
        cuda.synchronize()
        if apis['empty_cache']:
            cuda.empty_cache()
        for name in ('reset_max_memory_allocated', 'reset_max_memory_reserved'):
            if apis[name]:
                getattr(cuda, name)()
        baseline_allocated, baseline_reserved = api('memory_allocated'), api('memory_reserved')
        for _ in range(10):
            model(x)
        cuda.synchronize()
        peak_allocated, peak_reserved = api('max_memory_allocated'), api('max_memory_reserved')
    after = telemetry()
    assert_idle(after)
    if sha(item['checkpoint']) != item['sha256']:
        raise RuntimeError('Frozen weight changed during memory profile')
    result = dict(at=now(), size=size, pid=os.getpid(), **item,
                  actual_api_availability=apis, baseline_allocated_bytes=baseline_allocated,
                  baseline_reserved_bytes=baseline_reserved, peak_allocated_bytes=peak_allocated,
                  peak_reserved_bytes=peak_reserved,
                  increment_allocated_bytes=(peak_allocated - baseline_allocated
                    if peak_allocated is not None and baseline_allocated is not None else None),
                  allocated_peak_window=('post-warmup reset' if apis['reset_max_memory_allocated']
                                         else 'whole fresh process; reset API unavailable'),
                  reserved_peak_window=('post-warmup reset' if apis['reset_max_memory_reserved']
                                        else 'whole fresh process; reset API unavailable'),
                  protocol='Fresh process/model/input; FP32 batch1; 100warmup; 10measured forwards; no images/test',
                  paddle=paddle.__version__, hardware=cuda.get_device_name(),
                  telemetry_before=before, telemetry_after=after, images_read=False,
                  test_read=False, FLOPs=dict(status='NA', complete_coverage=False,
                    reason='No verified complete counter for grid_sample/interpolation/gates; no partial total reported'))
    write(output, result)
    return result


def run_memory_children(directory):
    receipts, records = [], []
    for size in SIZES:
        for group in GROUPS:
            guard_disk()
            tag = f'memory_{group}_{size}'
            output = directory / (tag + '.json')
            stdout, stderr = directory / (tag + '.stdout.log'), directory / (tag + '.stderr.log')
            command = [str(PYTHON), str(Path(__file__).resolve()), '--memory-one', group,
                       str(size), '--output', str(output)]
            started = now()
            timed_out = False
            with stdout.open('w', encoding='utf-8') as out, stderr.open('w', encoding='utf-8') as err:
                child = subprocess.Popen(command, stdout=out, stderr=err,
                                         env=dict(os.environ, PYTHONPATH=str(SRC),
                                                  PYTHONIOENCODING='utf-8'))
                try:
                    child.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    child.kill()
                    child.wait()
            receipt = dict(command=command, started_at=started, ended_at=now(),
                           exit_code=child.returncode, timed_out=timed_out,
                           stdout=str(stdout), stderr=str(stderr))
            write(directory / (tag + '_exit.json'), receipt)
            receipts.append(receipt)
            if child.returncode != 0 or not output.is_file():
                raise RuntimeError('Memory child failed: ' + tag + '; receipt retained')
            records.append(read(output))
    write(directory / 'memory_summary.json', dict(records=records, child_receipts=receipts,
                                                   test_read=False, images_read=False))
    return records


def interleaved(directory):
    # The parent has not imported Paddle or created a GPU context before the memory children.
    metadata, models = {}, {}
    first = telemetry()
    assert_idle(first)
    for group in GROUPS:
        paddle, model, item = build_gpu_model(group)
        models[group], metadata[group] = model, item
    cuda = paddle.device.cuda
    rows, snapshots = [], [first]
    with paddle.no_grad():
        for size in SIZES:
            x = paddle.randn([1, 3, size, size], dtype='float32')
            for group in GROUPS:
                for _ in range(100):
                    models[group](x)
            cuda.synchronize()
            for round_index in range(5):
                order = list(GROUPS[round_index:] + GROUPS[:round_index])
                for group in order:
                    guard_disk()
                    before = telemetry()
                    assert_idle(before)
                    values = []
                    for _ in range(100):
                        cuda.synchronize()
                        started = time.perf_counter_ns()
                        models[group](x)
                        cuda.synchronize()
                        values.append((time.perf_counter_ns() - started) / 1e6)
                    after = telemetry()
                    assert_idle(after)
                    row = dict(size=size, round=round_index, group=group, order=order,
                               individual_ms=values, P50_ms=percentile(values, .5),
                               P95_ms=percentile(values, .95), telemetry_before=before,
                               telemetry_after=after)
                    rows.append(row)
                    write(directory / 'interleaved_partial.json', dict(rows=rows,
                          status='running', metadata=metadata, images_read=False, test_read=False))
                    print('PROFILE', size, round_index, group, row['P50_ms'], flush=True)
    summaries = []
    for size in SIZES:
        for group in GROUPS:
            subset = [row for row in rows if row['size'] == size and row['group'] == group]
            values = [number for row in subset for number in row['individual_ms']]
            ratios = [row['P50_ms'] / next(r['P50_ms'] for r in rows
                      if r['size'] == size and r['round'] == row['round'] and r['group'] == 'R0')
                      for row in subset]
            cv = coefficient_of_variation(ratios)
            summaries.append(dict(size=size, P50_ms=percentile(values, .5),
                P95_ms=percentile(values, .95), round_P50_ms=[row['P50_ms'] for row in subset],
                paired_P50_ratios_R0=ratios, paired_P50_ratio_R0_median=statistics.median(ratios),
                paired_ratio_sample_CV=cv, stable_CV_le_5pct=(cv is not None and cv <= .05),
                **metadata[group]))
    result = dict(at=now(), status='complete', protocol=dict(shapes=[400, 480], batch=1,
                  precision='fp32', warmup_per_model_per_size=100, rounds=5,
                  forwards_each_round=100, synchronize_each_forward=True,
                  order='fixed cyclic rotation', images_read=False, test_read=False),
                  summary=summaries, rows=rows, metadata=metadata, telemetry=snapshots,
                  hardware=cuda.get_device_name(), paddle=paddle.__version__,
                  speed_claim='Descriptive measurement only; CV and telemetry must be disclosed.')
    for item in metadata.values():
        if sha(item['checkpoint']) != item['sha256']:
            raise RuntimeError('Frozen weight changed during latency profile')
    write(directory / 'interleaved_profile.json', result)
    return result


def known_counts():
    counts = {}
    for filename in ('initialization_manifest.json', 'matched_R0_initialization.json'):
        path = ROOT / 'provenance' / filename
        if not path.is_file():
            continue
        for ident, item in read(path).get('runs', {}).items():
            if ident.endswith('_s42'):
                group = ident.rsplit('_s', 1)[0]
                counts[group] = dict(registered=item.get('registered_numel'),
                                     trainable=item.get('trainable_numel'),
                                     source='complete CPU initial model construction manifest')
    return counts


def mean_text(group, metric):
    mean = group.get('mean', {}).get(metric)
    sd = group.get('sample_sd', {}).get(metric)
    return (f'{100*mean:.4f} ± {100*sd:.4f}'
            if mean is not None and sd is not None else '待完成')


def make_figures(directory, summary, counts, profile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    complete = [(group, summary.get('groups', {}).get(group, {})) for group in GROUPS]
    valid = [(group, row) for group, row in complete
             if row.get('mean', {}).get('crackIoU') is not None and counts.get(group, {}).get('registered')]
    figures = []
    if valid:
        fig, axis = plt.subplots(figsize=(7, 4.5), layout='constrained')
        for group, row in valid:
            x = counts[group]['registered'] / 1e6
            y = row['mean']['crackIoU'] * 100
            axis.errorbar(x, y, yerr=row['sample_sd']['crackIoU'] * 100, fmt='o', capsize=3)
            axis.annotate(group, (x, y), xytext=(5, 5), textcoords='offset points')
        axis.set(xlabel='Registered parameters (M; includes BN statistics)',
                 ylabel='Validation crack IoU (%)', title='W11 three-seed mean and sample SD')
        axis.grid(alpha=.2)
        path = directory / 'pareto_parameters_iou.png'
        fig.savefig(path, dpi=220)
        plt.close(fig)
        figures.append(path)
    if profile and valid:
        for size in SIZES:
            fig, axis = plt.subplots(figsize=(7, 4.5), layout='constrained')
            for group, row in valid:
                point = next(p for p in profile['summary'] if p['group'] == group and p['size'] == size)
                axis.scatter(point['P50_ms'], row['mean']['crackIoU'] * 100)
                axis.annotate(group + ('' if point['stable_CV_le_5pct'] else ' [unstable]'),
                              (point['P50_ms'], row['mean']['crackIoU'] * 100),
                              xytext=(5, 5), textcoords='offset points')
            axis.set(xlabel='Same-machine FP32 batch1 P50 (ms)', ylabel='Validation crack IoU (%)',
                     title=f'W11 {size}x{size}: descriptive accuracy/latency tradeoff')
            axis.grid(alpha=.2)
            path = directory / f'pareto_latency_iou_{size}.png'
            fig.savefig(path, dpi=220)
            plt.close(fig)
            figures.append(path)
    interactions = summary.get('interactions', [])
    if len(interactions) == 3:
        fig, axis = plt.subplots(figsize=(6, 3.6), layout='constrained')
        axis.bar([str(row['seed']) for row in interactions], [100*row['crackIoU'] for row in interactions])
        axis.axhline(0, color='black', linewidth=.7)
        axis.set(xlabel='Seed', ylabel='IoU interaction (percentage points)',
                 title='E3 - E1 - E2 + E0; descriptive, not significance')
        path = directory / 'interaction_iou_three_seeds.png'
        fig.savefig(path, dpi=220)
        plt.close(fig)
        figures.append(path)
    return figures


def report(directory, summary, profile=None, memory=None):
    counts = known_counts()
    if profile:
        counts.update({group: item['counts'] for group, item in profile['metadata'].items()})
    records = []
    for group in GROUPS:
        item = summary.get('groups', {}).get(group, {})
        row = dict(group=group, complete_seeds=item.get('complete_seeds', 0), **counts.get(group, {}))
        for metric in METRICS:
            row[metric + '_mean'] = item.get('mean', {}).get(metric)
            row[metric + '_sample_SD'] = item.get('sample_sd', {}).get(metric)
        if profile:
            for size in SIZES:
                measured = next(p for p in profile['summary'] if p['group'] == group and p['size'] == size)
                for key in ('P50_ms', 'P95_ms', 'paired_P50_ratio_R0_median', 'paired_ratio_sample_CV', 'stable_CV_le_5pct'):
                    row[f'{size}_{key}'] = measured[key]
        if memory:
            for measured in memory:
                if measured['group'] == group:
                    for key in ('peak_allocated_bytes', 'peak_reserved_bytes', 'increment_allocated_bytes'):
                        row[f"{measured['size']}_{key}"] = measured[key]
        records.append(row)
    csv_write(directory / 'ablation_three_seed.csv', records)
    csv_write(directory / 'all_seed_runs.csv', [dict((key, row.get(key)) for key in
              ('id', 'group', 'seed', 'status', *METRICS, 'selected_iter', 'selected_variant',
               'checkpoint_sha256', 'initialization_sha256')) for row in summary['rows']])
    write(directory / 'gates_and_interactions.json', {key: summary.get(key) for key in (
        'paired_deltas', 'interactions', 'interaction_crackIoU', 'preregistered_comparisons',
        'matched_R0_lightweight_comparisons', 'supports_two_modules_plus_combination', 'thresholds_relaxed')})
    figures = make_figures(directory, summary, counts, profile)
    full = summary.get('complete_runs') == 15
    lines = ['# SegFormer E0–E3 两模块互补实验报告', '', f'更新时间：{now()}。', '',
        f"当前真实完成 {summary.get('complete_runs', 0)}/15 项（四格12项＋新匹配R0三项）。"
        + ('本表为完整三seed验证结果。' if full else '未完成项保持待测，不据部分结果给稳定贡献结论。'), '',
        '本轮只使用Crack500 train/val。历史test已被观察；本后处理不读取原图、标签、test或自建集，'
        '效率输入为随机张量。没有领域SOTA结论。', '',
        '## 结构与参数', '',
        'R0是本轮匹配的原MiT-B0＋原A0头；E0–E3固定S4FFN768压缩骨干。'
        'E1=M1轻量多尺度解码器，E2=M2浅层门控恢复，E3=M1＋M2。旧历史A0三seed另作历史证据，'
        '不替代本轮R0。骨干压缩的减参不能全归为解码器贡献。', '',
        '|模型|完成seed|注册总量|可训练量|', '|---|---:|---:|---:|']
    for row in records:
        lines.append(f"|{row['group']}|{row['complete_seeds']}/3|{row.get('registered', '待测')}|{row.get('trainable', '待测')}|")
    lines += ['', '注册总量延续项目口径，包含Paddle注册但不训练的BN运行统计；可训练量另列。'
              '参数来自完整模型初始化核算，GPU阶段如已完成则另核冻结模型。', '',
              '## 三seed验证结果', '', '百分数均值 ± 样本SD（ddof=1）；三seed不足以声称统计显著性。', '',
              '|模型|Precision|Recall|F1|裂缝IoU|mIoU|BF1@2px|BF1@1px|hard clDice|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in GROUPS:
        item = summary.get('groups', {}).get(group, {})
        lines.append('|' + group + '|' + '|'.join(mean_text(item, metric) for metric in METRICS[:8]) + '|')
    lines += ['', '全348图先聚合像素/边界计数，再计算指标；hard clDice为固定骨架协议的全局诊断，'
              '不是额外训练loss，也不保证没有错误桥接。逐seed及全348图CSV在复现索引中保留。', '',
              '## 单模块、联合和交互判断', '']
    comparisons = summary.get('preregistered_comparisons', {})
    if not comparisons:
        lines.append('四格三seed尚未齐全，预登记门槛判断待完成。')
    else:
        lines += ['|比较|均值效应门槛|方向门槛|IoU均值差（百分点）|', '|---|---|---|---:|']
        for name, item in comparisons.items():
            lines.append(f"|{name}|{item['mean_effect_gate']}|{item['direction_gate']}|{item['mean_delta']['crackIoU']*100:+.4f}|")
    lines += ['', f"完整‘两个单项有效且组合更好’门槛：{summary.get('supports_two_modules_plus_combination')}。",
              'False表示该强主张未获支持，不删除负结果，也不事后放宽阈值。']
    interaction = summary.get('interaction_crackIoU')
    if interaction:
        lines += ['', f"IoU交互 I=E3−E1−E2＋E0，三seed均值 {interaction['mean']*100:+.4f} 百分点，"
                  f"样本SD {interaction['sample_sd']*100:.4f}，正向seed {interaction['positive_seeds']}/3；"
                  f"一致正交互标志 {interaction['consistent_positive_interaction']}。",
                  '组合最好不等于正交互；以上是描述性检验，不写统计证明。']
    lines += ['', '## 相对新匹配 R0 的轻量非劣门槛', '']
    lightweight = summary.get('matched_R0_lightweight_comparisons', {})
    if not lightweight:
        lines.append('新匹配R0与四格三seed尚未齐全，不能用旧A0历史均值替代通过结论。')
    else:
        lines += ['|模型|注册参数比R0|可训练参数比R0|IoU容差内seed|完整门槛|', '|---|---:|---:|---:|---|']
        for group, item in lightweight.items():
            lines.append(f"|{group}|{item['registered_param_ratio']:.6f}|{item['trainable_param_ratio']:.6f}|"
                         f"{item['iou_within_tolerance_seeds']}/3|{item['passed']}|")
    lines += ['', '## 同机效率与测量限制', '']
    if not profile:
        lines.append('同机交错延迟和独立进程峰值显存待测。参数减少不能替代速度结果。')
    else:
        lines += ['FP32、batch1、400/480，所有五组同进程；每模型每尺寸100预热，5轮轮换顺序，'
                  '每轮100次逐次CUDA同步。P50/P95统计单次延迟，配对比值由同轮P50相对R0得到；'
                  '样本CV≤5%才标稳定。完整温度/时钟/功率与外部进程回执保留。', '',
                  '|输入|模型|P50 ms|P95 ms|配对P50/R0|比值CV|CV≤5%|', '|---|---|---:|---:|---:|---:|---|']
        for item in profile['summary']:
            lines.append(f"|{item['size']}|{item['group']}|{item['P50_ms']:.4f}|{item['P95_ms']:.4f}|"
                         f"{item['paired_P50_ratio_R0_median']:.4f}|{item['paired_ratio_sample_CV']:.4f}|{item['stable_CV_le_5pct']}|")
        lines += ['', '这些是本机冻结checkpoint的描述性测量；不把减参写成加速，也不隐去不稳定或变慢组。']
    if memory:
        lines += ['', '|输入|模型|峰值allocated MiB|峰值reserved MiB|allocated增量 MiB|', '|---|---|---:|---:|---:|']
        for item in memory:
            def mib(key):
                return f"{item[key]/2**20:.3f}" if item.get(key) is not None else 'NA'
            lines.append(f"|{item['size']}|{item['group']}|{mib('peak_allocated_bytes')}|{mib('peak_reserved_bytes')}|{mib('increment_allocated_bytes')}|")
        lines += ['', '每行由独立新进程测量。实际内存API及峰值reset支持情况在对应JSON；'
                  '无reset API时标明整进程峰值，不假称纯前向窗口。不是系统显示占用或其他应用显存。']
    lines += ['', 'FLOPs：NA。尚无覆盖grid_sample、插值、门控全部算子的可靠计数器，'
              '不把不完整卷积算量当完整GFLOPs。', '', '## 图表、证据与后续', '']
    for figure in figures:
        lines += [f'![{figure.stem}]({figure.as_posix()})', '']
    lines += [f'数值、三seed、门槛与交互：`{directory}`。',
              'checkpoint、初始化、源码、配置和真实exit SHA随交付manifest提供；压缩包不包含原始数据或大权重。', '',
              '后续只在真实门槛支持时决定是否保留两模块。若组合未超过单项，优先简单方案；'
              '若只达到轻量非劣，按折中贡献表述。冻结最终候选后再登记统一test/额外泛化复核，'
              '不据test反选模型；自建集、source-frame-disjoint与机器人实验仍属后续阶段。', '',
              '六个既有原始公开模型的历史统一对照保留原报告版本，本轮不重新训练或改写其数字。']
    text = '\n'.join(lines) + '\n'
    local_report = directory / 'EXPERIMENT_RESULTS.md'
    local_report.write_text(text, encoding='utf-8')
    RESULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    RESULT_REPORT.write_text(text, encoding='utf-8')
    FINAL_REPORT.write_text(text, encoding='utf-8')
    return dict(local_report=str(local_report), experiment_report=str(RESULT_REPORT),
                root_report=str(FINAL_REPORT), figures=[str(path) for path in figures],
                csv=str(directory / 'ablation_three_seed.csv'))


def package(directory, delivery):
    # Explicit text/source evidence allow-list; never enumerate raw data or binary weights.
    files = set(path for path in directory.rglob('*') if path.is_file())
    files.update(ROOT.glob('*.py'))
    files.update(ROOT.glob('protocol.json'))
    files.update((ROOT / 'configs').glob('*.yml'))
    files.update((ROOT / 'results').glob('*.json'))
    files.update((ROOT / 'results').glob('*.csv'))
    files.update((ROOT / 'provenance').rglob('*.json'))
    files.update((ROOT / 'provenance').rglob('*.md'))
    files.update((ROOT / 'provenance').rglob('*.csv'))
    files.update((ROOT / 'provenance' / 'rng').glob('*.jsonl'))
    files.update(SRC.rglob('*.py'))
    files.update((ROOT / 'checkpoints').glob('*/selection.json'))
    files.update((ROOT / 'logs').glob('*.log'))
    files.update((ROOT / 'refine-logs').glob('*.md'))
    files.add(FINAL_REPORT)
    entries, omitted = [], []
    for path in sorted(files):
        if path.suffix.lower() in ('.zip', '.pdparams', '.pdopt', '.pdstates', '.png', '.jpg', '.jpeg') and path not in [Path(p) for p in delivery['figures']]:
            continue
        if path.stat().st_size > 64 * 2 ** 20:
            omitted.append(dict(path=str(path), reason='individual evidence file >64MiB'))
            continue
        archive_name = str(path.relative_to(ROOT)).replace('\\', '/') if path.is_relative_to(ROOT) else 'reports/' + path.name
        entries.append(dict(path=str(path), archive_name=archive_name,
                            size_bytes=path.stat().st_size, sha256=sha(path)))
    manifest = directory / 'bundle_manifest.json'
    write(manifest, dict(at=now(), files=entries, omitted=omitted,
                         raw_data_included=False, large_weights_included=False,
                         checkpoint_hashes_in_selection_records=True))
    archive = directory / 'W11_reports_and_reproduction.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for item in entries:
            z.write(item['path'], item['archive_name'])
        z.write(manifest, 'postprocessing/bundle_manifest.json')
    receipt = dict(path=str(archive), sha256=sha(archive), size_bytes=archive.stat().st_size,
                   manifest=str(manifest), manifest_sha256=sha(manifest), files=len(entries),
                   raw_data_included=False, large_weights_included=False)
    write(directory / 'bundle_receipt.json', receipt)
    return receipt


def self_test():
    assert percentile([1, 2, 3, 4], .5) == 2.5
    assert abs(percentile([1, 2, 3, 4], .95) - 3.85) < 1e-12
    assert coefficient_of_variation([1, 1, 1, 1, 1]) == 0
    assert FINAL_REPORT.parent == ROOT.parent.parent
    incomplete_rejected = None
    if load_summary().get('complete_runs') != 15:
        try:
            require_complete(load_summary())
        except RuntimeError:
            incomplete_rejected = True
        else:
            raise AssertionError('Incomplete training did not reject GPU profiling')
    assert process_state(os.getpid())['running'] is True
    result = dict(at=now(), passed=True, scope='Python syntax/import and pure arithmetic/PID query; no Paddle/GPU/imported images',
                  file_sha256=sha(__file__), GPU_started=False, training_started=False,
                  full_profile_ready=load_summary().get('complete_runs') == 15,
                  incomplete_training_blocks_GPU=incomplete_rejected)
    write(ROOT / 'provenance' / 'postprocess_self_test.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--profile-and-report', action='store_true')
    group.add_argument('--report-only', action='store_true')
    group.add_argument('--wait-and-profile', action='store_true')
    group.add_argument('--self-test', action='store_true')
    group.add_argument('--memory-one', nargs=2, metavar=('GROUP', 'SIZE'))
    parser.add_argument('--output', type=Path, help='Memory child JSON destination')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False), flush=True)
        return 0
    if args.wait_and_profile:
        return wait_and_profile()
    if args.memory_one:
        model, size = args.memory_one
        size = int(size)
        if model not in GROUPS or size not in SIZES or args.output is None:
            parser.error('--memory-one needs R0/E0/E1/E2/E3, 400/480 and --output')
        memory_one(model, size, args.output.resolve())
        return 0
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = ROOT / 'results' / 'postprocess' / stamp
    directory.mkdir(parents=True, exist_ok=False)
    lock = ROOT / 'results' / 'postprocess.lock'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), directory=str(directory), at=now()), stream)
    try:
        guard_disk()
        summary = load_summary()
        inputs = {'script': str(Path(__file__).resolve()), 'summary': str(SUMMARY),
                  'protocol': str(ROOT / 'protocol.json')}
        release = dict(at=now(), purpose='separate postprocessing release; training release unchanged',
                       inputs={key: dict(path=value, sha256=sha(value)) for key, value in inputs.items()},
                       command=sys.argv, GPU_requested=args.profile_and_report, images_read=False,
                       test_read=False)
        if args.profile_and_report:
            release['training_evidence'] = require_complete(summary)
        write(directory / 'postprocess_release.json', release)
        profile, memory = None, None
        if args.profile_and_report:
            # No parent GPU context yet: every peak-memory row is a fresh child.
            memory = run_memory_children(directory)
            profile = interleaved(directory)
            require_complete(load_summary())
            if sha(SUMMARY) != release['inputs']['summary']['sha256']:
                raise RuntimeError('Summary changed during profile')
            write(ROOT / 'results' / 'efficiency_profile.json', dict(
                  latency=str(directory / 'interleaved_profile.json'),
                  memory=str(directory / 'memory_summary.json'), postprocess_release=str(directory / 'postprocess_release.json'),
                  summary_sha256=sha(SUMMARY), status='complete', test_read=False))
        delivery = report(directory, summary, profile, memory)
        bundle = package(directory, delivery)
        result = dict(at=now(), exit_code=0, stage='full_profile_and_report_complete' if profile else 'partial_or_validation_report_only',
                      completed_training_runs=summary.get('complete_runs'), directory=str(directory),
                      delivery=delivery, bundle=bundle, images_read=False, test_read=False)
        write(directory / 'postprocess_exit.json', result)
        write(LATEST, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as error:
        write(directory / 'postprocess_exit.json', dict(at=now(), exit_code=1, error=repr(error),
              traceback=traceback.format_exc(), images_read=False, test_read=False))
        raise
    finally:
        if lock.exists() and read(lock).get('pid') == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    sys.exit(main())
