"""Frozen-only val/test evaluation with exact zero-extension 2px boundaries.

This is a separately versioned evaluator. It never replaces the historical
approximate public_test.py or changes a model, checkpoint, or dataset.
"""
import argparse
import csv
import os
import shutil
import time

from stage6_common import (ROOT, SOURCE, now, read, write, sha, manifest,
                          item_by_id, verify_stage_release, model_and_dataset,
                          require_test_authorization)


def boundary(mask):
    import cv2
    import numpy as np
    foreground = (mask > 0).astype(np.uint8)
    eroded = cv2.erode(foreground, np.ones((3, 3), np.uint8), iterations=1,
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return foreground - eroded


def matches(a, b, radius=2):
    import cv2
    import numpy as np
    if not np.any(a) or not np.any(b):
        return 0
    distance = cv2.distanceTransform((1 - b).astype(np.uint8), cv2.DIST_L2,
                                    cv2.DIST_MASK_PRECISE)
    return int(((a > 0) & (distance <= radius + 1e-6)).sum())


def ratios(cm, counts):
    import numpy as np
    cm = np.asarray(cm, dtype=np.int64)
    tp, fp, fn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    diag = np.diag(cm).astype(np.float64)
    unions = cm.sum(0) + cm.sum(1) - diag
    iou = diag / np.maximum(unions, 1)
    mp, mg, npred, ngt = counts
    bp = mp / npred if npred else (1.0 if not ngt else 0.0)
    br = mg / ngt if ngt else (1.0 if not npred else 0.0)
    return {'precision': p, 'recall': r,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            'crackIoU': float(iou[1]), 'mIoU': float(iou.mean()),
            'boundary_precision': bp, 'boundary_recall': br,
            'boundary_f1': 2 * bp * br / (bp + br) if bp + br else 0.0}


def self_check():
    """CPU-only checks; no Paddle import, GPU, datasets or image files."""
    import numpy as np
    blank = np.zeros((11, 11), np.uint8)
    full = np.ones((11, 11), np.uint8)
    assert boundary(blank).sum() == 0
    assert boundary(full).sum() == 40  # zero-extension keeps perimeter
    a = blank.copy(); a[5, 5] = 1
    b = blank.copy(); b[5, 7] = 1
    c = blank.copy(); c[7, 7] = 1
    assert matches(a, b) == 1  # exact distance 2
    assert matches(a, c) == 0  # sqrt(8) is outside the radius
    assert matches(a, blank) == matches(blank, a) == 0
    assert ratios([[10, 0], [0, 5]], [3, 3, 3, 3])['boundary_f1'] == 1
    return {'passed': True, 'tests': ['zero-outside perimeter', 'exact Euclidean 2px',
                                    'diagonal rejection', 'empty boundaries', 'global ratio'],
            'gpu_started': False, 'dataset_accessed': False}


def evaluate(ident, split):
    verify_stage_release()
    if split == 'test':
        require_test_authorization()
    item = item_by_id(ident)
    if split == 'test' and not item.get('include_in_test'):
        raise RuntimeError('This matched-seed diagnostic is validation-only')
    paddle, np, model, builder, cfg = model_and_dataset(item, split)
    from paddleseg.core import infer
    dataset = builder.val_dataset
    expected_n = 348 if split == 'val' else 1124
    if len(dataset) != expected_n:
        raise RuntimeError('Unexpected dataset length: ' + str(len(dataset)))
    loader = paddle.io.DataLoader(dataset, batch_size=1, shuffle=False,
                                   num_workers=0, return_list=True)
    cm = np.zeros((2, 2), dtype=np.int64)
    rows = []; counts = np.zeros(4, dtype=np.int64); empty = 0
    start = time.perf_counter()
    with paddle.no_grad():
        for index, data in enumerate(loader):
            if (ROOT / 'results/STOP_REQUEST.json').exists():
                raise RuntimeError('Controller requested a stop; partial logs retained')
            if shutil.disk_usage(ROOT).free < 4 * 2**30:
                raise RuntimeError('Data disk below 4GiB; stop before further output')
            pred, _ = infer.inference(model, data['img'], trans_info=data['trans_info'])
            p = np.asarray(pred).squeeze().astype(np.uint8)
            g = np.asarray(data['label']).squeeze().astype(np.uint8)
            if p.shape != g.shape or p.ndim != 2:
                raise RuntimeError('Restored pixel geometry mismatch')
            if not np.all((p == 0) | (p == 1)) or not np.all((g == 0) | (g == 1)):
                raise RuntimeError('Current binary-mask protocol requires labels 0/1')
            cm += np.bincount((g.ravel() * 2 + p.ravel()).astype(np.int64),
                              minlength=4).reshape(2, 2)
            pb, gb = boundary(p), boundary(g)
            c = [matches(pb, gb), matches(gb, pb), int(pb.sum()), int(gb.sum())]
            counts += c
            tp = int(((p == 1) & (g == 1)).sum())
            fp = int(((p == 1) & (g == 0)).sum())
            fn = int(((p == 0) & (g == 1)).sum())
            union = tp + fp + fn
            empty += int(not p.any())
            rows.append({'image': os.path.basename(dataset.file_list[index][0]),
                         'tp': tp, 'fp': fp, 'fn': fn,
                         'crack_iou': tp / union if union else 1.0,
                         'precision': tp / (tp + fp) if tp + fp else 0.0,
                         'recall': tp / (tp + fn) if tp + fn else 0.0,
                         'boundary_tp_pred': c[0], 'boundary_tp_gt': c[1],
                         'boundary_pred': c[2], 'boundary_gt': c[3]})
            if (index + 1) % 50 == 0:
                print(f'{ident} {split} {index + 1}/{expected_n}', flush=True)
    output = ROOT / 'results' / (ident + '_' + split + '.json')
    per_image = output.with_name(output.stem + '_per_image.csv')
    if output.exists() or per_image.exists():
        raise RuntimeError('Existing final output cannot be overwritten')
    with per_image.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    result = {'id': ident, 'split': split, 'n': len(rows),
              'confusion_matrix': cm.tolist(), 'boundary_counts_pred_gt': counts.tolist(),
              **ratios(cm, counts), 'empty_predictions': empty,
              'params': sum(int(np.prod(p.shape)) for p in model.parameters()),
              'checkpoint': item['checkpoint'], 'checkpoint_sha256': item['checkpoint_sha256'],
              'config_sha256': sha(cfg), 'per_image_sha256': sha(per_image),
              'source': str(SOURCE), 'paddle': paddle.__version__,
              'opencv': __import__('cv2').__version__, 'hardware': paddle.device.cuda.get_device_name(),
              'boundary_protocol': 'inner 3x3; border constant0; DIST_L2 DIST_MASK_PRECISE; <=2px+1e-6; symmetric global counts',
              'test_used': split == 'test', 'selection_uses_test': False,
              'elapsed_seconds_with_IO_postprocess': time.perf_counter() - start,
              'at': now()}
    write(output, result, exclusive=True)
    print(__import__('json').dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id'); parser.add_argument('--split', choices=['val', 'test'])
    parser.add_argument('--self-check', action='store_true'); args = parser.parse_args()
    if args.self_check:
        print(__import__('json').dumps(self_check()))
    elif args.id and args.split:
        evaluate(args.id, args.split)
    else:
        parser.error('Choose --self-check or --id and --split')
