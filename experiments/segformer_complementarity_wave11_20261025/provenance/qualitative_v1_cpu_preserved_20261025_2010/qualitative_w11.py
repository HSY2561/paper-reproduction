"""Separate fixed24 W11 qualitative supplement; --prepare is CPU-only.

--render is refused until all 15 trainings and automatic postprocessing exit0,
all owned GPU processes have exited, and the recorded WDDM guard is clear.
No prediction lists are created. Only registered val Dataset items are used.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import zipfile

import cv2
import numpy as np
import skimage
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src/PaddleSeg-2.10.0'
DATA = Path(r'C:\Users\user\algolab_data')
REFERENCE = ROOT.parent / 'local_structure_20260930/provenance/train_val_manifest.json'
FIXED = ROOT / 'provenance/fixed24_val_cases.json'
RELEASE = ROOT / 'provenance/qualitative_release.json'
GROUPS = ('R0', 'E0', 'E1', 'E2', 'E3')


def now():
    return dt.datetime.now().astimezone().isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            value.update(chunk)
    return value.hexdigest()


def mask_sha(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def csv_write(path, records):
    with Path(path).open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in records for k in row)))
        writer.writeheader()
        writer.writerows(records)


def postprocess():
    spec = importlib.util.spec_from_file_location('w11_frozen_postprocess', ROOT / 'postprocess_w11.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_cases():
    record = read(FIXED)
    cases = record['cases']
    names = [x['image'] for x in cases]
    if len(cases) != 24 or len(set(names)) != 24:
        raise RuntimeError('Frozen fixed24 must contain exactly 24 unique images')
    if any(Path(name).name != name or not name.isascii() for name in names):
        raise RuntimeError('Fixed case filenames must be simple ASCII basenames')
    return record


def verify_release():
    release = read(RELEASE)
    bad = [name for name, expected in release['files'].items()
           if not Path(name).is_file() or sha(name) != expected]
    if bad:
        raise RuntimeError('Separate qualitative release changed: ' + repr(bad))
    return dict(path=str(RELEASE), sha256=sha(RELEASE), version=release['version'])


def require_finished(pp):
    """All checks happen before importing Paddle or creating a GPU context."""
    summary = read(ROOT / 'results/four_cell_three_seed_summary.json')
    if summary.get('complete_runs') != 15:
        raise RuntimeError('QUALITATIVE_RENDER_REFUSED: training is incomplete ('
                           + str(summary.get('complete_runs')) + '/15); no GPU import/inference')
    for name in ('workflow.lock', 'postprocess.lock', 'postprocess_wait.lock'):
        if (ROOT / 'results' / name).exists():
            raise RuntimeError('QUALITATIVE_RENDER_REFUSED: active lock ' + name)
    training = pp.require_complete(summary)
    final = read(ROOT / 'results/final_delivery_done.json')
    receipt = read(ROOT / 'results/automatic_postprocess_child_exit.json')
    latest = read(ROOT / 'results/postprocess_latest.json')
    if (not final.get('complete') or final.get('actual_child_exit_code') != 0
            or receipt.get('exit_code') != 0 or receipt.get('timed_out')
            or latest.get('exit_code') != 0 or latest.get('stage') != 'full_profile_and_report_complete'):
        raise RuntimeError('QUALITATIVE_RENDER_REFUSED: automatic profile/report has not genuinely exited0')
    if final['postprocess'] != latest:
        raise RuntimeError('Final delivery and latest automatic postprocess evidence disagree')
    active = read(ROOT / 'results/active_process.json')
    if active.get('active') is not False or active.get('exit_code') != 0:
        raise RuntimeError('Last training child is still active or lacks exit0')
    pid_checks = [pp.process_state(receipt.get('pid')), pp.process_state(active.get('pid')),
                  pp.process_state(active.get('runner_pid'))]
    wrapper_receipts = []
    final_time = (ROOT / 'results/final_delivery_done.json').stat().st_mtime
    for path in (ROOT / 'results').glob('postprocess_waiter_*_exit.json'):
        record = read(path)
        if path.stat().st_mtime >= final_time - 1 and record.get('mode') == 'wait-and-profile':
            wrapper_receipts.append((path, record))
    if not wrapper_receipts:
        raise RuntimeError('Automatic postprocess waiter wrapper has no final exit receipt')
    wrapper_path, wrapper = max(wrapper_receipts, key=lambda pair: pair[0].stat().st_mtime)
    if wrapper.get('exit_code') != 0:
        raise RuntimeError('Automatic postprocess waiter wrapper did not exit0')
    pid_checks.append(pp.process_state(wrapper.get('runner_pid')))
    for path in Path(latest['directory']).glob('memory_*_exit.json'):
        memory = read(path)
        if memory.get('exit_code') != 0:
            raise RuntimeError('A profile child has nonzero exit: ' + str(path))
        pid_checks.append(pp.process_state(memory.get('pid')))
    if any(row['running'] is not False for row in pid_checks):
        raise RuntimeError('Owned training/profile PID still running or exit unknown: ' + repr(pid_checks))
    bundle = latest['bundle']
    if sha(bundle['path']) != bundle['sha256'] or sha(bundle['manifest']) != bundle['manifest_sha256']:
        raise RuntimeError('Original postprocess report bundle/manifest SHA changed')
    pp.verify_postprocess_release()
    return dict(training=training, postprocess=latest, process_checks=pid_checks,
                wrapper_receipt=str(wrapper_path), original_report_bundle=bundle)


def audit_data(directory, stage):
    reference = read(REFERENCE)
    mismatches = []
    count = 0
    for split in ('train', 'val'):
        if sha(DATA / (split + '.txt')) != reference[split]['list_sha256']:
            mismatches.append(split + '.txt')
        for relative, expected in reference[split]['files'].items():
            path = DATA / relative
            if not path.is_file() or sha(path) != expected:
                mismatches.append(relative)
            count += 1
    result = dict(at=now(), checked_files=count, mismatches=mismatches,
                  reference_sha256=sha(REFERENCE), test_read=False, custom_read=False)
    write(directory / ('data_' + stage + '.json'), result)
    if mismatches or count != 4488:
        raise RuntimeError('Registered 4488 train/val source SHA audit failed')
    return result


def counts(predicted, target):
    if predicted.ndim != 2 or predicted.shape != target.shape:
        raise RuntimeError('Prediction and GT shapes differ')
    if not np.isin(predicted, (0, 1)).all() or not np.isin(target, (0, 1)).all():
        raise RuntimeError('Expected binary argmax/GT masks with only 0/1 values')
    cm = np.bincount((target.ravel() * 2 + predicted.ravel()).astype(np.int64), minlength=4)
    tn, fp, fn, tp = map(int, cm)
    return dict(tn=tn, fp=fp, fn=fn, tp=tp)


def error_rgb(predicted, target):
    # BGR colors for OpenCV output: TP green, FP red, FN cyan, TN dark gray.
    image = np.full((*target.shape, 3), 24, np.uint8)
    image[(predicted == 1) & (target == 1)] = (60, 200, 60)
    image[(predicted == 1) & (target == 0)] = (30, 30, 240)
    image[(predicted == 0) & (target == 1)] = (240, 220, 30)
    return image


def skeleton_rgb(predicted, target):
    ps = skeletonize(predicted.astype(bool), method='zhang')
    gs = skeletonize(target.astype(bool), method='zhang')
    image = np.zeros((*target.shape, 3), np.uint8)
    image[gs] = (255, 220, 0)
    image[ps] = (220, 0, 255)
    image[ps & gs] = (255, 255, 255)
    return image, ps, gs


def mask_bgr(mask):
    return cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)


def png(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError('Refuse to overwrite an existing qualitative image: ' + str(path))
    ok, encoded = cv2.imencode('.png', array)
    if not ok:
        raise RuntimeError('PNG encode failed')
    encoded.tofile(str(path))


def tile(image, label, size=(256, 144), mask=False):
    width, height = size
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_AREA
    small = cv2.resize(image, (width, height), interpolation=interpolation)
    result = np.full((height + 26, width, 3), 245, np.uint8)
    result[26:, :] = small
    cv2.putText(result, label[:42], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .42,
                (20, 20, 20), 1, cv2.LINE_AA)
    return result


def panel(case, source, gt, predictions):
    top = np.concatenate([tile(source, 'Input'), tile(mask_bgr(gt), 'GT', mask=True),
                          tile(skeleton_rgb(gt, gt)[0], 'GT skeleton', mask=True),
                          tile(error_rgb(gt, gt), 'TP green | FP red | FN cyan', mask=True)], axis=1)
    rows = [top]
    for group in GROUPS:
        pred = predictions[group]
        c = counts(pred, gt)
        skeleton = skeleton_rgb(pred, gt)[0]
        overlay = source.copy()
        overlay[pred > 0] = (overlay[pred > 0].astype(np.uint16) // 2 + np.array([30, 110, 30])).clip(0, 255).astype(np.uint8)
        label = f'{group}: TP{c["tp"]} FP{c["fp"]} FN{c["fn"]}'
        rows.append(np.concatenate([tile(mask_bgr(pred), label, mask=True),
                    tile(error_rgb(pred, gt), group + ' errors', mask=True),
                    tile(skeleton, group + ' skeleton: GT cyan / pred pink', mask=True),
                    tile(overlay, group + ' prediction overlay')], axis=1))
    image = np.concatenate(rows, axis=0)
    header = np.full((32, image.shape[1], 3), 255, np.uint8)
    cv2.putText(header, case['image'], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6,
                (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def prepare():
    # Small generated arrays are kept in memory; no synthetic images are exported.
    target = np.array([[0, 1, 0], [1, 1, 0]], np.uint8)
    pred = np.array([[0, 0, 1], [1, 1, 0]], np.uint8)
    assert counts(pred, target) == dict(tn=2, fp=1, fn=1, tp=2)
    errors = error_rgb(pred, target)
    assert errors[0, 2].tolist() == [30, 30, 240]
    assert errors[0, 1].tolist() == [240, 220, 30]
    skeleton, ps, gs = skeleton_rgb(pred, target)
    assert ps.dtype == gs.dtype == bool and skeleton.shape == (2, 3, 3)
    ok, encoded = cv2.imencode('.png', errors)
    assert ok and np.array_equal(cv2.imdecode(encoded, cv2.IMREAD_COLOR), errors)
    example = panel(dict(image='in_memory_self_test'), np.zeros((2, 3, 3), np.uint8), target,
                    {group: pred for group in GROUPS})
    assert example.shape == (1052, 1024, 3)
    cases = fixed_cases()
    version = read(ROOT / 'provenance/evaluation_versions.json')
    assert skimage.__version__ == version['skimage'] and cv2.__version__ == version['cv2']
    pp = postprocess()
    refused = None
    if read(ROOT / 'results/four_cell_three_seed_summary.json').get('complete_runs') != 15:
        try:
            require_finished(pp)
        except RuntimeError as error:
            refused = str(error)
        assert refused and 'training is incomplete' in refused
    if 'paddle' in sys.modules:
        raise RuntimeError('CPU preparation unexpectedly imported Paddle')
    self_test = dict(at=now(), passed=True, image_array_tests=True, PNG_roundtrip_exact=True,
                     fixed_case_count=len(cases['cases']), training_incomplete_render_refusal=refused,
                     real_images_read=False, GPU_imported=False, GPU_inference=False, test_read=False,
                     script_sha256=sha(__file__))
    self_path = ROOT / 'provenance/qualitative_self_test.json'
    if self_path.exists():
        raise RuntimeError('Prior qualitative preparation evidence exists; isolate before a new version')
    write(self_path, self_test)
    files = [Path(__file__), FIXED, ROOT / 'postprocess_w11.py', ROOT / 'provenance/release.json',
             ROOT / 'provenance/postprocess_release.json', ROOT / 'provenance/evaluation_versions.json',
             ROOT / 'provenance/postprocess_desktop_baseline.json', REFERENCE, self_path,
             ROOT / 'QUALITATIVE_API.md']
    if RELEASE.exists():
        raise RuntimeError('Prior qualitative release exists; preserve before making a new version')
    write(RELEASE, dict(version='W11-qualitative-v1', frozen_at=now(),
          files={str(path.resolve()): sha(path) for path in files},
          independent_of_training_and_postprocess=True,
          purpose='fixed24 val / frozen five models seed42 / official infer.inference / no new threshold',
          render_requires='15 complete + automatic profile/report child/waiter exit0 + all owned PID exited + WDDM guard',
          GPU_execution_in_preparation=False, training_modified=False, test_read=False))
    print(json.dumps(self_test, ensure_ascii=False), flush=True)


def render():
    release = verify_release()
    pp = postprocess()
    finished = require_finished(pp)
    pp.guard_disk()
    quiet = pp.quiet_window()
    if not quiet['passed']:
        raise RuntimeError('QUALITATIVE_RENDER_DEFERRED: measured desktop GPU background is not quiet/stable')
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = ROOT / 'results/qualitative' / stamp
    directory.mkdir(parents=True, exist_ok=False)
    lock = ROOT / 'results/qualitative.lock'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), at=now(), directory=str(directory)), stream)
    rows = []
    after_audit = None
    try:
        write(directory / 'execution_inputs.json', dict(at=now(), release=release,
              completion_evidence=finished, quiet_window=quiet, fixed24_sha256=sha(FIXED),
              data_list_created=False, argmax_only=True, threshold_search=False, test_read=False))
        audit_data(directory, 'before')
        sys.path.insert(0, str(SRC))
        import paddle
        import paddleseg
        from paddleseg.cvlibs import Config, SegBuilder
        from paddleseg.core import infer
        from paddleseg.utils import utils
        if not Path(paddleseg.__file__).resolve().is_relative_to(SRC.resolve()):
            raise RuntimeError('Editable install won precedence over the isolated source')
        utils.set_device('gpu')
        fixed = fixed_cases()['cases']
        evidence = {group: pp.checkpoint_evidence(group, 42) for group in GROUPS}
        sources, targets, prediction_maps, case_folders = {}, {}, {}, {}
        input_evidence = {}
        for group in GROUPS:
            pp.guard_disk()
            pp.assert_idle(pp.telemetry())
            item = evidence[group]
            csv_path = ROOT / f'results/{item["id"]}_val_per_image.csv'
            val_path = ROOT / f'results/{item["id"]}_val.json'
            with csv_path.open(encoding='utf-8-sig', newline='') as stream:
                reference_rows = list(csv.DictReader(stream))
            expected = {row['image']: row for row in reference_rows}
            val = read(val_path)
            if len(reference_rows) != 348 or len(expected) != 348 or val['checkpoint_sha256'] != item['sha256']:
                raise RuntimeError('Frozen full348 CSV/checkpoint evidence mismatch: ' + group)
            input_evidence[group] = dict(checkpoint=item, full348_csv_sha256=sha(csv_path),
                                         full348_metrics_sha256=sha(val_path), exact_mask_hash_columns='prediction_sha256' in expected[fixed[0]['image']])
            cfg = Config(item['config'])
            settings = cfg.dic['val_dataset']
            if settings['mode'] != 'val' or 'test_path' in settings or Path(settings['val_path']).resolve() != (DATA / 'val.txt').resolve():
                raise RuntimeError('Only the registered official validation dataset is accepted')
            builder = SegBuilder(cfg)
            dataset, model = builder.val_dataset, builder.model
            lookup = {Path(pair[0]).name: index for index, pair in enumerate(dataset.file_list)}
            if len(dataset) != 348 or any(case['image'] not in lookup for case in fixed):
                raise RuntimeError('Frozen24 file names do not map uniquely into val348')
            weights = paddle.load(item['checkpoint'])
            actual = model.state_dict()
            if set(weights) != set(actual) or any(list(weights[k].shape) != list(actual[k].shape) for k in actual):
                raise RuntimeError('Frozen model must match every state key/shape strictly')
            model.set_state_dict(weights)
            model.eval()

            class SelectedVal(paddle.io.Dataset):
                def __len__(self):
                    return len(fixed)
                def __getitem__(self, index):
                    return dataset[lookup[fixed[index]['image']]]

            loader = paddle.io.DataLoader(SelectedVal(), batch_size=1, shuffle=False,
                                          num_workers=0, return_list=True)
            with paddle.no_grad():
                for index, data in enumerate(loader):
                    case = fixed[index]
                    name = case['image']
                    prediction, _ = infer.inference(model, data['img'], trans_info=data['trans_info'])
                    pred = np.asarray(prediction).squeeze().astype(np.uint8)
                    gt = np.asarray(data['label']).squeeze().astype(np.uint8)
                    measured = counts(pred, gt)
                    source_row = expected[name]
                    counts_match = all(measured[k] == int(source_row[k]) for k in ('tp', 'fp', 'fn', 'tn'))
                    gt_sha = mask_sha(gt)
                    pred_sha = mask_sha(pred)
                    gt_match = gt_sha == case['gt_mask_sha256'] == source_row['gt_mask_sha256']
                    old_pred_sha = source_row.get('prediction_sha256')
                    pixelwise_match = pred_sha == old_pred_sha if old_pred_sha else None
                    passed = counts_match and gt_match and pixelwise_match is not False
                    row = dict(case_index=index + 1, image=name, group=group, **measured,
                               counts_match_full348_csv=counts_match, gt_hash_match=gt_match,
                               prior_prediction_mask_sha256=old_pred_sha or None,
                               prediction_mask_sha256=pred_sha, pixelwise_hash_match=pixelwise_match,
                               gt_mask_sha256=gt_sha, verification_passed=passed)
                    rows.append(row)
                    # Preserve every fixed case and failed verification. No metric-based filtering.
                    case_folder = directory / 'cases' / f'{index+1:02d}_{Path(name).stem}'
                    case_folders[name] = case_folder
                    if name not in sources:
                        image_path = Path(dataset.file_list[lookup[name]][0])
                        if not image_path.resolve().is_relative_to(DATA.resolve()):
                            raise RuntimeError('Registered val input escaped the data root')
                        original = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if original is None or original.shape[:2] != gt.shape:
                            raise RuntimeError('Actual input/GT shape mismatch')
                        sources[name], targets[name], prediction_maps[name] = original, gt, {}
                        case_folder.mkdir(parents=True, exist_ok=False)
                        shutil.copy2(image_path, case_folder / ('input_original' + image_path.suffix.lower()))
                        png(case_folder / 'input.png', original)
                        png(case_folder / 'GT.png', gt * 255)
                        png(case_folder / 'GT_skeleton.png', skeletonize(gt.astype(bool), method='zhang').astype(np.uint8) * 255)
                    elif not np.array_equal(gt, targets[name]):
                        raise RuntimeError('Ground truth changed between model evaluations')
                    prediction_maps[name][group] = pred
                    png(case_folder / group / 'prediction.png', pred * 255)
                    png(case_folder / group / 'FP_FN.png', error_rgb(pred, gt))
                    skeleton, ps, gs = skeleton_rgb(pred, gt)
                    png(case_folder / group / 'skeleton_comparison.png', skeleton)
                    png(case_folder / group / 'prediction_skeleton.png', ps.astype(np.uint8) * 255)
                    if not passed:
                        write(case_folder / group / 'VERIFICATION_FAILED.json', row)
                    print('QUALITATIVE', group, index + 1, '/24', 'PASS' if passed else 'FAILED_PRESERVED', flush=True)
            del loader, dataset, builder, model, weights, actual
            paddle.device.cuda.empty_cache()
        overview_rows = []
        for case in fixed:
            name = case['image']
            png(case_folders[name] / 'comparison.png', panel(case, sources[name], targets[name], prediction_maps[name]))
            tiles = [tile(sources[name], name, (160, 90)), tile(mask_bgr(targets[name]), 'GT', (160, 90), True)]
            tiles += [tile(mask_bgr(prediction_maps[name][group]), group, (160, 90), True) for group in GROUPS]
            overview_rows.append(np.concatenate(tiles, axis=1))
        png(directory / 'overview_all24.png', np.concatenate(overview_rows, axis=0))
        for page in range(3):
            png(directory / f'overview_page{page+1}.png', np.concatenate(overview_rows[page*8:(page+1)*8], axis=0))
        after_audit = audit_data(directory, 'after')
        csv_write(directory / 'per_image_verification.csv', rows)
        original_bundle = finished['original_report_bundle']
        if sha(original_bundle['path']) != original_bundle['sha256'] or sha(original_bundle['manifest']) != original_bundle['manifest_sha256']:
            raise RuntimeError('Original report package SHA changed while rendering')
        failed = [row for row in rows if not row['verification_passed']]
        write(directory / 'verification.json', dict(passed=not failed, fixed_cases=24, model_count=5,
              rows=len(rows), failed_rows=failed, counts_exact=True,
              pixelwise_hash_comparison='performed only when full348 CSV contains prediction_sha256',
              all_fixed_cases_retained=True, original_report_bundle_unchanged=True,
              original_report_bundle=original_bundle, test_read=False))
        (directory / 'README.md').write_text(
            '# W11 fixed24 qualitative supplement\n\n'
            'All 24 cases were fixed from input/GT before formal training. Five frozen seed42 models are shown. '
            'All cases and any verification failures are retained. These examples do not replace the full348 metrics.\n\n'
            'Prediction/GT PNGs contain black background and white foreground. Error colors: TP green, FP red, FN cyan. '
            'Skeleton comparison: GT cyan, prediction pink, exact overlap white. Overview panels use nearest-neighbor '
            'mask thumbnails; full-size masks are available per case.\n\n'
            'Counts are compared exactly with the corresponding full348 CSV. Pixelwise identity is claimed only where '
            'a prior prediction-mask SHA is present and matches. Source files and the original report bundle are unchanged.\n', encoding='utf-8')
        file_entries = [dict(path=path.relative_to(directory).as_posix(), sha256=sha(path), size_bytes=path.stat().st_size)
                        for path in sorted(directory.rglob('*')) if path.is_file()]
        manifest = dict(at=now(), qualitative_release=release, cases=24, models=list(GROUPS), seed=42,
                        source_case_policy=read(FIXED)['policy'], files=file_entries,
                        original_report_bundle_unchanged=original_bundle,
                        includes_only_selected24_input_exports=True, full_raw_dataset_included=False,
                        weights_included=False, verification_passed=not failed, test_read=False)
        write(directory / 'manifest.json', manifest)
        archive = directory / 'W11_fixed24_qualitative_supplement.zip'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for item in file_entries:
                bundle.write(directory / item['path'], item['path'])
            bundle.write(directory / 'manifest.json', 'manifest.json')
        result = dict(at=now(), exit_code=0 if not failed else 2, complete=not failed, directory=str(directory),
                      manifest=str(directory / 'manifest.json'), manifest_sha256=sha(directory / 'manifest.json'),
                      archive=str(archive), archive_sha256=sha(archive), rows=120,
                      failed_verifications=len(failed), original_report_bundle_unchanged=True, test_read=False)
        write(directory / 'qualitative_exit.json', result)
        write(ROOT / 'results/qualitative_latest.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result['exit_code']
    except BaseException as error:
        write(directory / 'qualitative_exit.json', dict(at=now(), exit_code=1, complete=False,
              error=repr(error), traceback=traceback.format_exc(), partial_outputs_preserved=True,
              observations=rows, test_read=False))
        raise
    finally:
        if after_audit is None:
            try:
                audit_data(directory, 'after_failure')
            except BaseException as error:
                write(directory / 'data_after_failure_error.json', dict(error=repr(error)))
        if lock.exists() and read(lock).get('pid') == os.getpid():
            lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', action='store_true')
    actions.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return 0
    return render()


if __name__ == '__main__':
    sys.exit(main())
