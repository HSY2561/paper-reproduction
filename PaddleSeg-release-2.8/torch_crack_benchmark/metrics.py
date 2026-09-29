import numpy as np


def confusion_matrix_from_masks(pred, label, num_classes=2):
    pred = pred.astype(np.int64).reshape(-1)
    label = label.astype(np.int64).reshape(-1)
    valid = (label >= 0) & (label < num_classes)
    hist = np.bincount(
        num_classes * label[valid] + pred[valid],
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    return hist


def metrics_from_hist(hist, foreground_class=1):
    hist = hist.astype(np.float64)
    true_positive = np.diag(hist)
    pred_area = hist.sum(axis=0)
    label_area = hist.sum(axis=1)
    union = pred_area + label_area - true_positive
    class_iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=union > 0)
    class_precision = np.divide(true_positive, pred_area, out=np.zeros_like(true_positive), where=pred_area > 0)
    class_recall = np.divide(true_positive, label_area, out=np.zeros_like(true_positive), where=label_area > 0)
    class_dice = np.divide(
        2 * true_positive,
        pred_area + label_area,
        out=np.zeros_like(true_positive),
        where=(pred_area + label_area) > 0,
    )
    total = hist.sum()
    accuracy = 0.0 if total == 0 else float(true_positive.sum() / total)
    miou = float(class_iou.mean())
    mdice = float(class_dice.mean())
    pe = 0.0 if total == 0 else float((pred_area * label_area).sum() / (total**2))
    po = accuracy
    kappa = 0.0 if abs(1.0 - pe) < 1e-12 else float((po - pe) / (1.0 - pe))
    fg = min(foreground_class, len(class_iou) - 1)
    fg_precision = float(class_precision[fg])
    fg_recall = float(class_recall[fg])
    fg_f1 = 0.0 if fg_precision + fg_recall == 0 else 2 * fg_precision * fg_recall / (fg_precision + fg_recall)
    return {
        "foreground_class": fg,
        "miou": miou,
        "mdice": mdice,
        "accuracy": accuracy,
        "kappa": kappa,
        "class_iou": [float(v) for v in class_iou],
        "class_dice": [float(v) for v in class_dice],
        "class_precision": [float(v) for v in class_precision],
        "class_recall": [float(v) for v in class_recall],
        "metrics": {
            "foreground_iou": float(class_iou[fg]),
            "foreground_dice": float(class_dice[fg]),
            "foreground_precision": fg_precision,
            "foreground_recall": fg_recall,
            "foreground_f1": float(fg_f1),
        },
    }
