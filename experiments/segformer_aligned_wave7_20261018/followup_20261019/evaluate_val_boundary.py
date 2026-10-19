import argparse
import csv
import json
import os

import cv2
import numpy as np
import paddle

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.core import infer
from paddleseg.utils import utils


def boundary(mask):
    mask = (mask > 0).astype(np.uint8)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1,
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask - eroded


def matched(pred_boundary, target_boundary, radius=2):
    if not pred_boundary.any() or not target_boundary.any():
        return 0
    distance = cv2.distanceTransform(
        (1 - target_boundary).astype(np.uint8), cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE)
    return int(np.logical_and(pred_boundary > 0,
                              distance <= radius + 1e-6).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='gpu')
    args = parser.parse_args()

    cfg = Config(args.config)
    builder = SegBuilder(cfg)
    if builder.val_dataset.mode != 'val':
        raise ValueError('This screening evaluator accepts validation only')
    utils.set_device(args.device)
    utils.load_entire_model(builder.model, args.model_path)
    builder.model.eval()
    loader = paddle.io.DataLoader(builder.val_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=0,
                                  return_list=True)

    cm = np.zeros((2, 2), dtype=np.int64)
    rows = []
    empty_predictions = 0
    with paddle.no_grad():
        for index, data in enumerate(loader):
            pred, _ = infer.inference(builder.model,
                                      data['img'],
                                      trans_info=data['trans_info'])
            predicted = np.asarray(pred).squeeze().astype(np.uint8)
            target = np.asarray(data['label']).squeeze().astype(np.uint8)
            cm += np.bincount((target.ravel() * 2 + predicted.ravel()).astype(
                np.int64), minlength=4).reshape(2, 2)
            pred_boundary = boundary(predicted)
            target_boundary = boundary(target)
            tp = int(((predicted == 1) & (target == 1)).sum())
            fp = int(((predicted == 1) & (target == 0)).sum())
            fn = int(((predicted == 0) & (target == 1)).sum())
            boundary_tp_pred = matched(pred_boundary, target_boundary)
            boundary_tp_gt = matched(target_boundary, pred_boundary)
            boundary_pred = int(pred_boundary.sum())
            boundary_gt = int(target_boundary.sum())
            if predicted.sum() == 0:
                empty_predictions += 1
            image = os.path.basename(builder.val_dataset.file_list[index][0])
            rows.append({
                'image': image,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'crack_iou': tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                'precision': tp / (tp + fp) if tp + fp else 0.0,
                'recall': tp / (tp + fn) if tp + fn else 0.0,
                'boundary_tp_pred': boundary_tp_pred,
                'boundary_tp_gt': boundary_tp_gt,
                'boundary_pred': boundary_pred,
                'boundary_gt': boundary_gt,
            })

    diagonal = np.diag(cm).astype(np.float64)
    union = cm.sum(axis=1) + cm.sum(axis=0) - diagonal
    iou = diagonal / np.maximum(union, 1)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    boundary_tp_pred = sum(r['boundary_tp_pred'] for r in rows)
    boundary_tp_gt = sum(r['boundary_tp_gt'] for r in rows)
    boundary_pred = sum(r['boundary_pred'] for r in rows)
    boundary_gt = sum(r['boundary_gt'] for r in rows)
    boundary_precision = (boundary_tp_pred / boundary_pred
                          if boundary_pred else 1.0)
    boundary_recall = (boundary_tp_gt / boundary_gt if boundary_gt else 1.0)
    boundary_f1 = (2 * boundary_precision * boundary_recall /
                   (boundary_precision + boundary_recall)
                   if boundary_precision + boundary_recall else 0.0)
    result = {
        'n': len(rows),
        'confusion_matrix': cm.tolist(),
        'mIoU': float(iou.mean()),
        'crackIoU': float(iou[1]),
        'precision': float(tp / (tp + fp)),
        'recall': float(tp / (tp + fn)),
        'f1': float(2 * tp / (2 * tp + fp + fn)),
        'boundary_counts_pred_gt': [boundary_tp_pred, boundary_tp_gt,
                                    boundary_pred, boundary_gt],
        'boundary_precision': float(boundary_precision),
        'boundary_recall': float(boundary_recall),
        'boundary_f1': float(boundary_f1),
        'empty_predictions': empty_predictions,
        'test_used': False,
        'boundary_protocol': 'inner boundary, zero outside image; exact Euclidean <=2px; global counts then harmonic mean',
        'config': args.config,
        'model_path': args.model_path,
    }
    with open(args.output + '.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
    with open(args.output + '_per_image.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
