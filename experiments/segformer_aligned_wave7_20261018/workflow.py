"""Durable orchestration for existing official model implementations."""
import argparse
import ctypes
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'segformer_balance_wave6_20261016'
PYTHON = r'C:\venv_bisenet\Scripts\python.exe'


def write(name, value):
    target = ROOT / 'results' / name
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, target)


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def load_legacy():
    spec = importlib.util.spec_from_file_location('legacy_queue', OLD / 'experiment_queue.py')
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    original = q.run_command
    def run(args, tag, timeout=7200):
        # Runtime limit only; model, batch, workers, data and budget unchanged.
        limit = 12 * 3600 if tag.endswith('_train') else max(timeout, 3600)
        return original(args, tag, timeout=limit)
    q.run_command = run
    return q


def legacy_se():
    q = load_legacy()
    state = q.read_json(OLD / 'results/M01_SE_status.json')
    if state['status'] == 'complete':
        return
    if (OLD / 'results/M01_SE_train_exit.json').exists():
        raise RuntimeError('SE old exit must be archived with evidence before retry')
    q.run_queue()
    state = q.read_json(OLD / 'results/M01_SE_status.json')
    if state['status'] != 'complete':
        raise RuntimeError('SE did not complete; retain evidence, review before advancing')


def run():
    # Task Scheduler priority must be normal before spawning venv/real Python.
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.SetPriorityClass.restype = ctypes.c_int
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x20):
            raise RuntimeError('Could not set NORMAL_PRIORITY_CLASS')
    lock = ROOT / 'results/workflow.lock'
    with lock.open('x') as f:
        json.dump(dict(pid=os.getpid(), started_at=now()), f)
    try:
        write('workflow_state.json', dict(stage='legacy_SE', pid=os.getpid(), updated_at=now()))
        legacy_se()
        write('workflow_state.json', dict(stage='awaiting_release', pid=os.getpid(), updated_at=now()))
        # The release file is created only after source review, full diagnostic
        # evidence, config preparation and smoke-interface checks are complete.
        deadline = time.time() + 24 * 3600
        while not (ROOT / 'provenance/RELEASE.json').exists():
            if time.time() > deadline:
                raise RuntimeError('Wave7 review release not received within 24h')
            time.sleep(30)
        write('workflow_state.json', dict(stage='wave7', pid=os.getpid(), updated_at=now()))
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src/PaddleSeg-2.10.0'),
                   PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
        with (ROOT / 'logs/queue.log').open('ab') as stream:
            result = subprocess.run([PYTHON, str(ROOT / 'aligned_queue.py'), '--run'],
                                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    creationflags=0x20 if os.name == 'nt' else 0)
        write('queue_exit.json', dict(exit_code=result.returncode, ended_at=now()))
        if result.returncode:
            raise RuntimeError('Wave7 queue exited nonzero')
        write('workflow_state.json', dict(stage='queue_finished_requires_review', pid=os.getpid(), updated_at=now()))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        run()
    except BaseException as e:
        write('workflow_exit.json', dict(exit_code=1, ended_at=now(), error=repr(e)))
        raise
    else:
        write('workflow_exit.json', dict(exit_code=0, ended_at=now()))
