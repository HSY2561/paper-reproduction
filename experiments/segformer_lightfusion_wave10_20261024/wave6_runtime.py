"""Experiment orchestration, not a new segmentation algorithm.

Model changes live in the snapshot's existing segformer.py and call existing
PaddleSeg blocks. This runner never reads test/custom dataset lists.
"""
import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src' / 'PaddleSeg-2.10.0'
PYTHON = Path(r'C:\venv_bisenet\Scripts\python.exe')
PROJECT = ROOT.parent.parent
BASE_WEIGHTS = PROJECT / 'experiments/retrain_wave4_20261007/checkpoints/segformer_b0/model.pdparams'
BASE_SHA = '07f6048e0a6de174a2c3ca1915df4bcd635565a706d978288fba946ef240f2d5'
DATA = Path(r'C:\Users\user\algolab_data')
DATA_MANIFEST = PROJECT / 'experiments/local_structure_20260930/provenance/train_val_manifest.json'
MODULES = [
    ('M01_SE', 'se', 'MobileNetV3 SE', 'backbones/mobilenetv3.py', '融合后通道选择'),
    ('M02_PSA', 'polarized', 'Polarized Self Attention', 'layers/polaried_self_attention.py', '融合后空间与通道判别'),
    ('M03_GC', 'gc', 'Global Context Block', 'gcnet.py', 'C4全局上下文'),
    ('M04_CCA', 'cca', 'Criss Cross Attention', 'ccnet.py', 'C4行列长距离依赖'),
    ('M05_PAM', 'pam', 'Position Attention Module', 'layers/attention.py', 'C4空间位置关系'),
    ('M06_CAM', 'cam', 'Channel Attention Module', 'layers/attention.py', 'C4通道关系'),
    ('M07_NonLocal', 'nonlocal', 'NonLocal2D', 'layers/nonlocal2d.py', 'C4成对上下文'),
    ('M08_MSCA', 'msca', 'MSCAN SpatialAttention', 'backbones/mscan.py', '融合后多尺度大核条带卷积'),
    ('M09_Encoding', 'encoding', 'EncModule', 'encnet.py', 'C4字典编码上下文'),
    ('M10_GloRe', 'glore', 'GloRe GruModule', 'glore.py', 'C4图关系推理'),
    ('M11_ASPPsep', 'aspp_sep', 'Separable ASPP', 'layers/pyramid_pool.py', '投影C4多尺度空洞卷积'),
    ('M12_PSP', 'psp_pool', 'PSP PPModule', 'layers/pyramid_pool.py', '投影C4金字塔池化'),
    ('M13_Aligned', 'aligned', 'SFNet AlignedModule', 'sfnet.py', 'C1与C4语义流对齐'),
    ('M14_UAFMchannel', 'uafm_channel', 'UAFM Channel Attention', 'layers/tensor_fusion.py', 'C1与融合特征通道融合'),
]
MODULE_ORDER = ['M08_MSCA', 'M13_Aligned', 'M03_GC', 'M06_CAM', 'M11_ASPPsep',
                'M02_PSA', 'M04_CCA', 'M05_PAM', 'M07_NonLocal', 'M09_Encoding',
                'M10_GloRe', 'M14_UAFMchannel', 'M01_SE', 'M12_PSP']
MODULES.sort(key=lambda row: MODULE_ORDER.index(row[0]))


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def versioned_text(path, text):
    path = Path(path)
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    history = path.with_name(path.stem + '_' + stamp + path.suffix)
    history.write_text(text, encoding='utf-8')
    shutil.copy2(history, path)


def env():
    values = dict(os.environ)
    values.update(PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE='1',
                  PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
                  TEMP=str(ROOT / 'tmp'), TMP=str(ROOT / 'tmp'),
                  CUDA_CACHE_PATH=str(ROOT / 'tmp/cuda_cache'))
    return values


def spec_list():
    result = [dict(id='W128', width=128, modules=[], family='width_control', label='MLP decoder width128'),
              dict(id='W64', width=64, modules=[], family='width_control', label='MLP decoder width64')]
    for ident, module, label, source, purpose in MODULES:
        result.append(dict(id=ident, width=128, modules=[module], family=module,
                           label=label, source=source, purpose=purpose, parent='W128'))
    return result


def prepare():
    import yaml
    template = yaml.safe_load((ROOT / 'configs/baseline_template.yml').read_text(encoding='utf-8-sig'))
    template['model']['backbone']['pretrained'] = str(ROOT / 'weights/mit_b0_imagenet.pdparams')
    template['model'].pop('decoder_variant', None)
    jobs = spec_list()
    for job in [dict(id='B0', width=256, modules=[])] + jobs:
        config = copy.deepcopy(template)
        config['model']['embedding_dim'] = job['width']
        config['model']['balance_modules'] = job['modules']
        (ROOT / 'configs' / (job['id'] + '.yml')).write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    write_json(ROOT / 'provenance/job_specs.json', dict(created_at=now(), jobs=jobs,
                                                      distinct_module_families=len(MODULES)))
    sources = {str(p.relative_to(SRC)): sha(p) for p in (SRC / 'paddleseg').rglob('*.py')}
    write_json(ROOT / 'provenance/source_hashes.json', sources)
    write_json(ROOT / 'provenance/protocol.json', dict(
        date=now(), seed=42, batch_size=4, iters=18000, precision='fp32',
        optimizer='AdamW', lr=6e-5, end_lr=0, train=1896, val=348,
        crop=[400, 400], scaling=[0.75, 1.25], test_used=False,
        baseline_checkpoint=str(BASE_WEIGHTS), baseline_sha256=BASE_SHA,
        pretrained_sha256=sha(ROOT / 'weights/mit_b0_imagenet.pdparams'),
        module_family_count=len(MODULES), width_controls_count=2,
        public_comparison_policy='all six preregistered controls get test after val-only freezing, regardless of rank'))
    print('PREPARED', len(jobs), 'runs;', len(MODULES), 'distinct new module mechanisms', flush=True)


def audit_data(tag):
    reference = read_json(DATA_MANIFEST)
    mismatches = []
    checked = 0
    for split in ('train', 'val'):
        if sha(DATA / (split + '.txt')) != reference[split]['list_sha256']:
            mismatches.append(split + '.txt')
        for relative, expected in reference[split]['files'].items():
            if sha(DATA / relative) != expected:
                mismatches.append(relative)
            checked += 1
    result = dict(checked_files=checked, mismatches=mismatches, test_used=False, timestamp=now())
    write_json(ROOT / ('provenance/data_' + tag + '.json'), result)
    if mismatches:
        raise RuntimeError('Source train/val hashes changed; stop queue')
    return result


def tail(path, count=12000):
    with Path(path).open('rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - count))
        return f.read().decode('utf-8', errors='replace')


def run_command(args, tag, timeout=7200):
    log = ROOT / 'logs' / (tag + '.log')
    started = time.time()
    command = [str(PYTHON)] + [str(a) for a in args]
    with log.open('wb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT), env=env())
        write_json(ROOT / 'results/active_process.json', dict(
            tag=tag, pid=child.pid, runner_pid=os.getpid(), command=command,
            started_at=now(), log=str(log), test_used=False))
        last_update = 0
        while child.poll() is None:
            if time.time() - started > timeout:
                child.terminate()
                child.wait(timeout=60)
                code = child.returncode
                write_json(ROOT / 'results' / (tag + '_exit.json'), dict(
                    exit_code=code, timed_out=True, elapsed_seconds=time.time() - started,
                    ended_at=now(), command=command, test_used=False))
                raise RuntimeError('Child timeout: ' + tag)
            if time.time() - last_update > 20:
                matches = re.findall(r'iter:\s*(\d+)/18000', tail(log))
                write_json(ROOT / 'results/heartbeat.json', dict(
                    tag=tag, pid=child.pid, runner_pid=os.getpid(), timestamp=now(),
                    latest_iter=int(matches[-1]) if matches else None,
                    elapsed_seconds=round(time.time() - started, 1),
                    log_size=log.stat().st_size, free_gb=round(shutil.disk_usage(ROOT).free / 2**30, 2)))
                last_update = time.time()
            time.sleep(1)
        code = child.returncode
    result = dict(exit_code=code, elapsed_seconds=round(time.time() - started, 2),
                  ended_at=now(), command=command, test_used=False)
    write_json(ROOT / 'results' / (tag + '_exit.json'), result)
    print(tag, 'exit', code, 'seconds', result['elapsed_seconds'], flush=True)
    return code


def model_for(ident, weights):
    import paddle
    from paddleseg.cvlibs import Config, SegBuilder
    from paddleseg.utils import utils
    paddle.seed(42)
    paddle.set_device('gpu:0')
    model = SegBuilder(Config(str(ROOT / 'configs' / (ident + '.yml')))).model
    if weights:
        utils.load_entire_model(model, str(weights))
    model.eval()
    return model


def profile(ident, weights):
    import numpy as np
    import paddle
    model = model_for(ident, weights)
    param_total = sum(int(np.prod(p.shape)) for p in model.parameters())
    param_trainable = sum(int(np.prod(p.shape)) for p in model.parameters() if not p.stop_gradient)
    rows = []
    with paddle.no_grad():
        for size in (400, 480):
            x = paddle.randn([1, 3, size, size], dtype='float32')
            for _ in range(20):
                model(x)
            timings = []
            for _ in range(3):
                paddle.device.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(50):
                    model(x)
                paddle.device.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000 / 50)
            rows.append(dict(input_shape=[1, 3, size, size], repeat_ms=timings,
                             median_ms=float(np.median(timings))))
        feats = model.backbone(paddle.zeros([1, 3, 400, 400]))
        feat_shapes = [list(f.shape) for f in feats]
    output = dict(model=ident, params_total=param_total, params_trainable=param_trainable,
                  latency=rows, checkpoint=str(weights), feature_shapes_at400=feat_shapes,
                  hardware=paddle.device.cuda.get_device_name(), paddle=paddle.__version__,
                  protocol='FP32 GPU network forward; batch1; sync; warmup20; 3x50; no I/O/postprocess',
                  test_used=False, timestamp=now())
    write_json(ROOT / 'results' / (ident + '_profile.json'), output)
    print(json.dumps(output, ensure_ascii=False), flush=True)


def choose_checkpoint(ident):
    folder = ROOT / 'training' / ident
    choices = []
    for variant, state_name, weight_name in (
        ('best_model', 'model.pdstates', 'model.pdparams'),
        ('ema_best_model', 'ema_model.pdstates', 'ema_model.pdparams')):
        state_path = folder / variant / state_name
        with state_path.open('rb') as stream:
            state = pickle.load(stream)
        choices.append(dict(variant=variant, mIoU=float(state['mIoU']),
                            iter=int(state['iter']), weight=str(folder / variant / weight_name),
                            state_sha256=sha(state_path)))
    chosen = max(choices, key=lambda row: row['mIoU'])
    destination = ROOT / 'checkpoints' / ident
    destination.mkdir(exist_ok=True)
    shutil.copy2(chosen['weight'], destination / 'model.pdparams')
    record = dict(model=ident, candidates=choices, selected=chosen,
                  sha256=sha(destination / 'model.pdparams'),
                  checkpoint=str(destination / 'model.pdparams'),
                  selection='highest exact validation mIoU from ordinary and EMA state',
                  test_used=False, frozen_at=now())
    write_json(destination / 'selection.json', record)
    return record


def evaluate(ident, weights):
    cfg = ROOT / 'configs' / (ident + '.yml')
    code = run_command([SRC / 'tools/val.py', '--config', cfg, '--model_path', weights,
                        '--num_workers', '0'], ident + '_official_val', timeout=600)
    if code:
        raise RuntimeError(ident + ': official validation failed')
    code = run_command([ROOT / 'evaluate_val_boundary.py', '--config', cfg,
                        '--model_path', weights, '--output', ROOT / 'results' / (ident + '_val')],
                       ident + '_pixel_boundary', timeout=600)
    if code:
        raise RuntimeError(ident + ': pixel boundary verification failed')
    code = run_command([__file__, '--profile', ident, '--weights', weights],
                       ident + '_profile', timeout=600)
    if code:
        raise RuntimeError(ident + ': parameter/latency profile failed')


def decisions():
    base = read_json(ROOT / 'results/B0_val.json')
    bp = read_json(ROOT / 'results/B0_profile.json')
    rows = []
    parent_file = ROOT / 'results/W128_val.json'
    parent = read_json(parent_file) if parent_file.exists() else None
    jobs = read_json(ROOT / 'provenance/job_specs.json')['jobs']
    for job in jobs:
        ident = job['id']
        result = ROOT / 'results' / (ident + '_val.json')
        prof = ROOT / 'results' / (ident + '_profile.json')
        status_file = ROOT / 'results' / (ident + '_status.json')
        row = dict(id=ident, label=job['label'], modules=job['modules'], width=job['width'],
                   status=read_json(status_file)['status'] if status_file.exists() else 'pending')
        if result.exists() and prof.exists() and row['status'] == 'complete':
            m = read_json(result)
            p = read_json(prof)
            row.update({k: m[k] for k in ('mIoU', 'crackIoU', 'precision', 'recall', 'f1', 'boundary_f1')})
            row.update(params=p['params_total'], latency480=p['latency'][1]['median_ms'])
            row['delta'] = {k: m[k] - base[k] for k in ('mIoU', 'crackIoU', 'precision', 'recall', 'f1', 'boundary_f1')}
            if job['modules'] and parent:
                row['delta_vs_W128'] = {k: m[k] - parent[k] for k in ('mIoU', 'crackIoU', 'precision', 'recall', 'f1', 'boundary_f1')}
            row['param_ratio'] = p['params_total'] / bp['params_total']
            row['latency_ratio'] = row['latency480'] / bp['latency'][1]['median_ms']
            row['accuracy_pass'] = (row['delta']['crackIoU'] >= .003 and row['delta']['f1'] >= .002
                                    and row['delta']['mIoU'] >= 0 and row['delta']['recall'] >= -.005
                                    and row['delta']['boundary_f1'] >= -.005
                                    and row['param_ratio'] <= 1.05 and row['latency_ratio'] <= 1.2)
            row['efficiency_pass'] = (row['param_ratio'] <= .95 and row['latency_ratio'] <= .9
                                      and row['delta']['crackIoU'] >= -.002 and row['delta']['f1'] >= -.002
                                      and row['delta']['recall'] >= -.01 and row['delta']['boundary_f1'] >= -.01)
            row['target_pass'] = (row['param_ratio'] <= .95 and row['delta']['crackIoU'] >= .002
                                  and row['delta']['f1'] >= .001 and row['delta']['mIoU'] >= 0
                                  and row['delta']['recall'] >= -.005 and row['delta']['boundary_f1'] >= -.005
                                  and row['latency_ratio'] <= 1.1)
        rows.append(row)
    write_json(ROOT / 'results/comparison.json', dict(baseline=base, baseline_profile=bp,
                                                    rows=rows, test_used=False, updated_at=now()))
    columns = ['id', 'label', 'width', 'status', 'precision', 'recall', 'f1', 'crackIoU',
               'mIoU', 'params', 'latency480', 'boundary_f1', 'target_pass']
    with (ROOT / 'results/ablation_table.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# SegFormer-B0 Crack500 精度效率消融进度', '', '更新时间：' + now(), '',
             '以下都是验证集结果。待训练/失败条目不填写性能；普通/EMA以精确验证mIoU选择。', '',
             '| 编号 | 模块 | 状态 | Precision | Recall | F1 | IoU | mIoU | Params M | 480延迟ms |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|',
             f"| B0 | SegFormer-B0 | 冻结复用 | {base['precision']:.6f} | {base['recall']:.6f} | {base['f1']:.6f} | {base['crackIoU']:.6f} | {base['mIoU']:.6f} | {bp['params_total']/1e6:.4f} | {bp['latency'][1]['median_ms']:.3f} |"]
    for row in rows:
        if 'mIoU' in row:
            nums = ' | '.join(f'{row[k]:.6f}' for k in ('precision', 'recall', 'f1', 'crackIoU', 'mIoU'))
            lines.append(f"| {row['id']} | {row['label']} | {row['status']} | {nums} | {row['params']/1e6:.4f} | {row['latency480']:.3f} |")
        else:
            lines.append(f"| {row['id']} | {row['label']} | {row['status']} | — | — | — | — | — | — | — |")
    lines += ['', '14个不同模块机制与2个宽度控制分开计数。默认每个完成18k；smoke失败不等于性能差。',
              '精度/效率通过与最终目标通过分开记录，详见comparison.json和预注册计划。六个公开模型由候选确定后统一执行，均保留原始结构。',
              '不能从筛选验证集的偶然最大值推断SOTA；最终需要多种子、冻结test与源帧互斥复核。']
    versioned_text(ROOT / 'refine-logs/EXPERIMENT_RESULTS.md', '\n'.join(lines) + '\n')
    tracker = ['# 实验执行跟踪', '', '更新时间：' + now(), '',
               '| 编号 | 机制 | 宽度 | 状态 |', '|---|---|---:|---|']
    tracker += [f"| {r['id']} | {r['label']} | {r['width']} | {r['status']} |" for r in rows]
    versioned_text(ROOT / 'refine-logs/EXPERIMENT_TRACKER.md', '\n'.join(tracker) + '\n')
    manifest = ROOT / 'MANIFEST.md'
    paths_file = ROOT / 'provenance/manifest_paths.json'
    known = set(read_json(paths_file)) if paths_file.exists() else set()
    if not manifest.exists():
        manifest.write_text('# Research Output Manifest\n\n| Timestamp | Skill | File | Stage | Description |\n|---|---|---|---|---|\n', encoding='utf-8')
    with manifest.open('a', encoding='utf-8') as stream:
        for subdir in ('configs', 'results', 'provenance', 'refine-logs', 'logs', 'checkpoints'):
            for path in sorted((ROOT / subdir).rglob('*')):
                if not path.is_file() or path.suffix in ('.tmp', '.lock'):
                    continue
                rel = path.relative_to(ROOT).as_posix()
                if rel not in known:
                    stream.write(f'| {now()} | /run-experiment | {rel} | implementation | experiment evidence |\n')
                    known.add(rel)
    write_json(paths_file, sorted(known))


def status(ident, value, **kwargs):
    write_json(ROOT / 'results' / (ident + '_status.json'), dict(status=value, updated_at=now(), **kwargs))


def train_args(ident, smoke=False):
    return [SRC / 'tools/train.py', '--config', ROOT / 'configs' / (ident + '.yml'),
            '--save_dir', ROOT / 'training' / (('smoke_' if smoke else '') + ident),
            '--iters', '20' if smoke else '18000', '--do_eval', '--use_ema',
            '--save_interval', '20' if smoke else '1000', '--seed', '42',
            '--log_iters', '10' if smoke else '100', '--num_workers', '0',
            '--keep_checkpoint_max', '1', '--precision', 'fp32']


def run_queue():
    lock = ROOT / 'results/queue.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), started_at=now())))
    try:
        audit_data('before')
        if sha(BASE_WEIGHTS) != BASE_SHA:
            raise RuntimeError('Baseline checkpoint hash mismatch')
        if not (ROOT / 'results/B0_val.json').exists() or not (ROOT / 'results/B0_profile.json').exists():
            evaluate('B0', BASE_WEIGHTS)
        baseline = read_json(ROOT / 'results/B0_val.json')
        if abs(baseline['mIoU'] - .8280206270288883) > 1e-7:
            raise RuntimeError('Baseline no longer reproduces its pixel mIoU')
        jobs = read_json(ROOT / 'provenance/job_specs.json')['jobs']
        # Phase one checks all actual trainer, eval and checkpoint paths first.
        for job in jobs:
            ident = job['id']
            path = ROOT / 'results' / (ident + '_smoke_exit.json')
            if path.exists():
                continue
            status(ident, 'smoke_running')
            if run_command(train_args(ident, True), ident + '_smoke', timeout=900):
                status(ident, 'smoke_failed', evidence=str(path))
            else:
                status(ident, 'smoke_passed')
            decisions()
        write_json(ROOT / 'results/smoke_matrix_complete.json', dict(timestamp=now(), jobs=len(jobs)))
        for job in jobs:
            ident = job['id']
            state = read_json(ROOT / 'results' / (ident + '_status.json'))
            if state['status'] in ('complete', 'smoke_failed'):
                continue
            if shutil.disk_usage(ROOT).free < 4 * 2**30:
                raise RuntimeError('Disk below 4GiB; queue paused without deleting evidence')
            try:
                receipt = ROOT / 'results' / (ident + '_train_exit.json')
                if not receipt.exists():
                    if (ROOT / 'training' / ident).exists():
                        raise RuntimeError('Unfinished prior training directory; do not restart blindly')
                    status(ident, 'training')
                    if run_command(train_args(ident), ident + '_train'):
                        status(ident, 'train_failed', evidence=str(receipt))
                        decisions()
                        continue
                elif read_json(receipt)['exit_code'] != 0:
                    continue
                if not (ROOT / 'training' / ident / 'iter_18000/model.pdparams').exists():
                    raise RuntimeError('Training exit0 without iter18000 checkpoint')
                if 'iter: 18000/18000' not in (ROOT / 'logs' / (ident + '_train.log')).read_text(encoding='utf-8', errors='replace'):
                    raise RuntimeError('Missing final iteration in trainer log')
                selection = choose_checkpoint(ident)
                status(ident, 'verifying')
                evaluate(ident, selection['checkpoint'])
                measured = read_json(ROOT / 'results' / (ident + '_val.json'))
                if abs(measured['mIoU'] - selection['selected']['mIoU']) > 1e-6:
                    raise RuntimeError('Frozen checkpoint validation differs from selection state')
                status(ident, 'complete', selection=selection)
            except Exception as error:
                status(ident, 'needs_review', error=str(error), traceback=traceback.format_exc())
                print(ident, 'needs_review:', error, flush=True)
            decisions()
        audit_data('after')
        decisions()
        states = [read_json(ROOT / 'results' / (j['id'] + '_status.json'))['status'] for j in jobs]
        write_json(ROOT / 'results/queue_done.json', dict(timestamp=now(), states=states,
                   full_training_complete=states.count('complete'), planned_modules=len(MODULES), test_used=False,
                   next_step='review Pareto table; choose complementary pair before joint training'))
        print('QUEUE_DONE', states.count('complete'), '/', len(jobs), flush=True)
    finally:
        # Delete only the runner-owned lock, never training evidence/data.
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--profile')
    parser.add_argument('--weights')
    parser.add_argument('--refresh-report', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.profile:
        profile(args.profile, args.weights)
    elif args.run:
        run_queue()
    elif args.refresh_report:
        decisions()
    else:
        parser.error('Choose --prepare, --profile or --run')
