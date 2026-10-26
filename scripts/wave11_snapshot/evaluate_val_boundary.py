"""W11 validation-only exact global pixels, boundaries and hard skeleton counts."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import cv2
import numpy as np
import skimage
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent
VERSIONS = {'scikit-image': '0.26.0', 'opencv-python-headless': '4.11.0.86',
            'numpy': '2.4.1'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def mask_sha(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def versions():
    actual = {package: importlib.metadata.version(package) for package in VERSIONS}
    if actual != VERSIONS:
        raise RuntimeError('Pinned evaluation versions changed: ' + repr(actual))
    return dict(packages=actual, cv2=cv2.__version__, skimage=skimage.__version__,
                skeleton_algorithm='skimage.morphology.skeletonize(method=zhang)',
                boundary_distance='cv2.DIST_L2 / DIST_MASK_PRECISE',
                gt_radius_border='pad1 constant0; exact distanceTransform; crop1',
                test_used=False)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def harmonic(precision, recall):
    if precision is None or recall is None:
        return None
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def similarity(matched_pred, matched_gt, pred_count, gt_count):
    if pred_count == 0 and gt_count == 0:
        return dict(precision=None, recall=None, f1=None, empty='both')
    if pred_count == 0 or gt_count == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0,
                    empty='pred' if pred_count == 0 else 'gt')
    precision = matched_pred / pred_count
    recall = matched_gt / gt_count
    return dict(precision=float(precision), recall=float(recall),
                f1=harmonic(precision, recall), empty='neither')


def boundary(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1,
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask - eroded


def boundary_counts(predicted, target):
    pb, gb = boundary(predicted), boundary(target)
    pn, gn = int(pb.sum()), int(gb.sum())
    result = dict(boundary_pred=pn, boundary_gt=gn)
    if pn and gn:
        to_gt = cv2.distanceTransform(1 - gb, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        to_pred = cv2.distanceTransform(1 - pb, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    for radius in (1, 2):
        mp = int(np.logical_and(pb > 0, to_gt <= radius + 1e-6).sum()) if pn and gn else 0
        mg = int(np.logical_and(gb > 0, to_pred <= radius + 1e-6).sum()) if pn and gn else 0
        result[f'boundary_match_pred_{radius}px'] = mp
        result[f'boundary_match_gt_{radius}px'] = mg
        result[f'boundary_f1_{radius}px'] = similarity(mp, mg, pn, gn)['f1']
    return result


def skeleton_counts(predicted, target):
    ps = skeletonize(predicted.astype(bool), method='zhang')
    gs = skeletonize(target.astype(bool), method='zhang')
    pn, gn = int(ps.sum()), int(gs.sum())
    mp = int(np.logical_and(ps, target > 0).sum())
    mg = int(np.logical_and(gs, predicted > 0).sum())
    result = dict(pred_skeleton_count=pn, gt_skeleton_count=gn,
                  pred_skeleton_in_gt=mp, gt_skeleton_in_pred=mg,
                  hard_cldice=similarity(mp, mg, pn, gn)['f1'],
                  pred_skeleton_sha256=mask_sha(ps), gt_skeleton_sha256=mask_sha(gs))
    padded = np.pad(target.astype(np.uint8), 1, mode='constant', constant_values=0)
    radius = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    edge = np.zeros(target.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    result.update(gt_crop_border_skeleton_count=int(np.logical_and(gs, edge).sum()),
                  gt_crop_border_skeleton_in_pred=int(np.logical_and(gs & edge, predicted > 0).sum()),
                  gt_skeleton_mean_radius=float(radius[gs].mean()) if gn else None)
    for name, condition in (
        ('r_le1', radius <= 1 + 1e-6),
        ('r_1to2', (radius > 1 + 1e-6) & (radius <= 2 + 1e-6)),
        ('r_gt2', radius > 2 + 1e-6)):
        points = gs & condition
        border = points & edge
        result[name + '_gt'] = int(points.sum())
        result[name + '_hit'] = int(np.logical_and(points, predicted > 0).sum())
        result[name + '_border_gt'] = int(border.sum())
        result[name + '_border_hit'] = int(np.logical_and(border, predicted > 0).sum())
        result[name + '_recall'] = ratio(result[name + '_hit'], result[name + '_gt'])
    return result


def image_counts(predicted, target):
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise RuntimeError('Prediction/GT must share a 2D shape')
    if not np.isin(predicted, (0, 1)).all() or not np.isin(target, (0, 1)).all():
        raise RuntimeError('This registered binary val set requires labels 0/1 only')
    cm = np.bincount((target.ravel() * 2 + predicted.ravel()).astype(np.int64),
                     minlength=4).reshape(2, 2)
    tn, fp, fn, tp = map(int, cm.ravel())
    both_empty = tp + fp + fn == 0
    result = dict(tn=tn, fp=fp, fn=fn, tp=tp, height=int(target.shape[0]), width=int(target.shape[1]),
                  crack_iou=None if both_empty else ratio(tp, tp + fp + fn),
                  precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                  f1=None if both_empty else ratio(2 * tp, 2 * tp + fp + fn),
                  gt_foreground_fraction=float(target.mean()),
                  empty_prediction=tp + fp == 0, empty_gt=tp + fn == 0,
                  empty_both=both_empty, zero_foreground_overlap=tp == 0 and tp + fp + fn > 0,
                  negative_sample_fp=fp if tp + fn == 0 else 0,
                  prediction_sha256=mask_sha(predicted), gt_mask_sha256=mask_sha(target))
    result.update(boundary_counts(predicted, target))
    result.update(skeleton_counts(predicted, target))
    return result, cm


def aggregate(rows, cm):
    tp, fp, fn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0])
    diagonal = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - diagonal
    ious = [ratio(diagonal[i], union[i]) for i in range(2)]
    result = dict(n=len(rows), confusion_matrix=cm.tolist(),
                  mIoU=float(np.mean([x for x in ious if x is not None])),
                  crackIoU=ious[1], precision=ratio(tp, tp + fp),
                  recall=ratio(tp, tp + fn), f1=ratio(2 * tp, 2 * tp + fp + fn),
                  empty_predictions=sum(r['empty_prediction'] for r in rows),
                  empty_gt=sum(r['empty_gt'] for r in rows),
                  empty_both=sum(r['empty_both'] for r in rows),
                  zero_foreground_overlap=sum(r['zero_foreground_overlap'] for r in rows),
                  negative_sample_fp=sum(r['negative_sample_fp'] for r in rows))
    count_fields = ['boundary_pred', 'boundary_gt', 'pred_skeleton_count', 'gt_skeleton_count',
                    'pred_skeleton_in_gt', 'gt_skeleton_in_pred',
                    'gt_crop_border_skeleton_count', 'gt_crop_border_skeleton_in_pred']
    count_fields += [f'boundary_match_{side}_{r}px' for side in ('pred', 'gt') for r in (1, 2)]
    count_fields += [name + '_' + side for name in ('r_le1', 'r_1to2', 'r_gt2')
                     for side in ('gt', 'hit', 'border_gt', 'border_hit')]
    sums = {key: sum(r[key] for r in rows) for key in count_fields}
    for radius in (1, 2):
        metric = similarity(sums[f'boundary_match_pred_{radius}px'], sums[f'boundary_match_gt_{radius}px'],
                            sums['boundary_pred'], sums['boundary_gt'])
        result[f'boundary_f1_{radius}px'] = metric['f1']
        result[f'boundary_precision_{radius}px'] = metric['precision']
        result[f'boundary_recall_{radius}px'] = metric['recall']
    result.update(boundary_f1=result['boundary_f1_2px'],
                  boundary_precision=result['boundary_precision_2px'],
                  boundary_recall=result['boundary_recall_2px'])
    metric = similarity(sums['pred_skeleton_in_gt'], sums['gt_skeleton_in_pred'],
                        sums['pred_skeleton_count'], sums['gt_skeleton_count'])
    result.update(hard_cldice=metric['f1'], skeleton_precision=metric['precision'],
                  skeleton_recall=metric['recall'], global_counts=sums)
    result['gt_radius_strata'] = {
        name: dict(gt=sums[name + '_gt'], hit=sums[name + '_hit'],
                   recall=ratio(sums[name + '_hit'], sums[name + '_gt']),
                   crop_border_gt=sums[name + '_border_gt'],
                   crop_border_hit=sums[name + '_border_hit'],
                   crop_border_recall=ratio(sums[name + '_border_hit'], sums[name + '_border_gt']))
        for name in ('r_le1', 'r_1to2', 'r_gt2')}
    result['crop_border_skeleton_recall'] = ratio(sums['gt_crop_border_skeleton_in_pred'],
                                                  sums['gt_crop_border_skeleton_count'])
    return result


def fixed_cases(rows):
    """Freeze 24 examples using only input/GT, before any formal run."""
    descriptors = [dict(image=r['image'], image_path=r['image_path'],
                        gt_mask_sha256=r['gt_mask_sha256'],
                        gt_foreground_fraction=r['gt_foreground_fraction'],
                        gray_std=r['gray_std'], fg_bg_contrast=r['fg_bg_contrast'],
                        thin_fraction=ratio(r['r_le1_gt'], r['gt_skeleton_count']) or 0.0)
                   for r in rows]
    thin_order = sorted(descriptors, key=lambda r: (r['thin_fraction'], r['image']))
    contrast_order = sorted(descriptors, key=lambda r: (r['fg_bg_contrast'], r['image']))
    for rank, row in enumerate(thin_order):
        row['thin_tertile'] = min(2, rank * 3 // len(rows))
    for rank, row in enumerate(contrast_order):
        row['contrast_tertile'] = min(2, rank * 3 // len(rows))
    strata = {key: sorted([r for r in descriptors if (r['thin_tertile'], r['contrast_tertile']) == key],
                          key=lambda r: r['image']) for key in [(a, b) for a in range(3) for b in range(3)]}
    chosen = []
    while len(chosen) < 24:
        for key in sorted(strata):
            if strata[key] and len(chosen) < 24:
                chosen.append(strata[key].pop(0))
    record = dict(policy='GT skeleton radius<=1 fraction tertile x input fg/bg contrast tertile; filename sorted round-robin24',
                  cases=chosen, all_descriptors=sorted(descriptors, key=lambda r: r['image']), test_used=False)
    path = ROOT / 'provenance/fixed24_val_cases.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8-sig'))
        if previous != record:
            raise RuntimeError('Input/GT case manifest changed; preserve and review')
    else:
        write(path, record)
    return dict(path=str(path), sha256=sha(path), selected_images=[r['image'] for r in chosen])


def self_test():
    empty = np.zeros((7, 9), np.uint8)
    line = empty.copy()
    line[0:7, 4] = 1
    same, cm = image_counts(line, line)
    assert same['boundary_f1_1px'] == same['boundary_f1_2px'] == same['hard_cldice'] == 1
    assert same['gt_crop_border_skeleton_count'] == 2
    assert same['r_le1_gt'] == 7 and same['r_le1_hit'] == 7
    both, _ = image_counts(empty, empty)
    assert both['boundary_f1_1px'] is None and both['hard_cldice'] is None
    one, _ = image_counts(empty, line)
    assert one['boundary_f1_1px'] == one['hard_cldice'] == 0
    shifted = np.roll(line, 2, axis=1)
    counts, _ = image_counts(shifted, line)
    assert counts['boundary_f1_1px'] == 0 and counts['boundary_f1_2px'] == 1
    full = np.ones((3, 4), np.uint8)
    assert int(boundary(full).sum()) == 10
    version = versions()
    write(ROOT / 'provenance/evaluation_versions.json', version)
    write(ROOT / 'results/evaluation_self_test.json', dict(passed=True, cases=6,
          zero_border=True, exact_euclidean=True, empty_semantics=True, test_used=False))
    print('EVALUATOR_SELF_TEST_PASS', json.dumps(version), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model_path')
    parser.add_argument('--output')
    parser.add_argument('--device', default='gpu')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.config, args.model_path, args.output)):
        parser.error('--config, --model_path and --output are required')
    version = versions()
    import paddle
    from paddleseg.cvlibs import Config, SegBuilder
    from paddleseg.core import infer
    from paddleseg.utils import utils
    cfg = Config(args.config)
    if cfg.dic['val_dataset']['mode'] != 'val' or 'test_path' in cfg.dic['val_dataset']:
        raise RuntimeError('This evaluator accepts only the registered validation dataset')
    if Path(cfg.dic['val_dataset']['val_path']).resolve() != Path(r'${DATA_ROOT}\val.txt').resolve():
        raise RuntimeError('Unregistered validation list')
    utils.set_device(args.device)
    builder = SegBuilder(cfg)
    dataset = builder.val_dataset
    model = builder.model
    checkpoint = paddle.load(args.model_path)
    actual = model.state_dict()
    if set(checkpoint) != set(actual) or any(list(checkpoint[k].shape) != list(actual[k].shape) for k in actual):
        raise RuntimeError('Checkpoint must match every model state key/shape exactly')
    model.set_state_dict(checkpoint)
    model.eval()
    if len(dataset) != 348:
        raise RuntimeError('Registered val348 count changed')
    loader = paddle.io.DataLoader(dataset, batch_size=1, shuffle=False,
                                  num_workers=0, return_list=True)
    cm = np.zeros((2, 2), dtype=np.int64)
    rows = []
    with paddle.no_grad():
        for index, data in enumerate(loader):
            prediction, _ = infer.inference(model, data['img'], trans_info=data['trans_info'])
            predicted = np.asarray(prediction).squeeze().astype(np.uint8)
            target = np.asarray(data['label']).squeeze().astype(np.uint8)
            row, image_cm = image_counts(predicted, target)
            cm += image_cm
            image_path = dataset.file_list[index][0]
            raw = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if raw is None or raw.shape != target.shape:
                raise RuntimeError('Cannot read registered input for GT-based descriptors')
            fg, bg = raw[target > 0], raw[target == 0]
            row.update(index=index, image=os.path.basename(image_path), image_path=image_path,
                       gray_std=float(raw.std()),
                       fg_bg_contrast=float(abs(float(fg.mean()) - float(bg.mean()))) if len(fg) and len(bg) else 0.0)
            rows.append(row)
            if (index + 1) % 50 == 0:
                print('VAL_PROGRESS', index + 1, '/348', flush=True)
    result = aggregate(rows, cm)
    result.update(test_used=False, config=str(Path(args.config).resolve()), config_sha256=sha(args.config),
                  model_path=str(Path(args.model_path).resolve()), checkpoint_sha256=sha(args.model_path),
                  versions=version, fixed24_cases=fixed_cases(rows),
                  boundary_protocol='inner3x3 constant0; exact Euclidean<=1/2px; accumulate match counts then harmonic mean',
                  skeleton_protocol='hard global counts: skeleton(pred) in GT; skeleton(GT) in pred; Zhang0.26.0; argmax masks',
                  radius_protocol='GT pad1 constant0 exact distance then crop; radius<=1,1<r<=2,r>2; crop-border points separately')
    write(args.output + '.json', result)
    with open(args.output + '_per_image.csv', 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
