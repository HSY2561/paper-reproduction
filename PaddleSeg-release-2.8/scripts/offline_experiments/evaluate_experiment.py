import argparse
from pathlib import Path

import numpy as np
import paddle
import paddle.nn.functional as F

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paddleseg.core import infer
from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import metrics, utils

from scripts.offline_experiments.common import dump_json


def merge_test_config(cfg, args):
    test_config = cfg.test_config
    if args.aug_eval:
        test_config["aug_eval"] = True
        test_config["scales"] = args.scales
        test_config["flip_horizontal"] = args.flip_horizontal
        test_config["flip_vertical"] = args.flip_vertical
    if args.is_slide:
        test_config["is_slide"] = True
        test_config["crop_size"] = args.crop_size
        test_config["stride"] = args.stride
    return test_config


def _override_val_dataset(cfg, dataset_root=None, list_path=None):
    if dataset_root:
        cfg.dic["val_dataset"]["dataset_root"] = dataset_root
    if list_path:
        cfg.dic["val_dataset"]["val_path"] = list_path
    return cfg


def evaluate_dataset(model, eval_dataset, test_config, num_workers=0, foreground_class=1):
    model.eval()
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        return_list=True,
    )

    intersect_area_all = paddle.zeros([eval_dataset.num_classes], dtype="int64")
    pred_area_all = paddle.zeros([eval_dataset.num_classes], dtype="int64")
    label_area_all = paddle.zeros([eval_dataset.num_classes], dtype="int64")
    logits_all = None
    label_all = None
    aug_eval = test_config.get("aug_eval", False)
    scales = test_config.get("scales", 1.0)
    flip_horizontal = test_config.get("flip_horizontal", False)
    flip_vertical = test_config.get("flip_vertical", False)
    is_slide = test_config.get("is_slide", False)
    stride = test_config.get("stride")
    crop_size = test_config.get("crop_size")
    auc_roc = test_config.get("auc_roc", False)

    with paddle.no_grad():
        for data in loader:
            label = data["label"].astype("int64")
            if aug_eval:
                pred, logits = infer.aug_inference(
                    model,
                    data["img"],
                    trans_info=data["trans_info"],
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size,
                )
            else:
                pred, logits = infer.inference(
                    model,
                    data["img"],
                    trans_info=data["trans_info"],
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size,
                )

            intersect_area, pred_area, label_area = metrics.calculate_area(
                pred,
                label,
                eval_dataset.num_classes,
                ignore_index=eval_dataset.ignore_index,
            )
            intersect_area_all += intersect_area
            pred_area_all += pred_area
            label_area_all += label_area

            if auc_roc:
                probs = F.softmax(logits, axis=1)
                logits_all = probs.numpy() if logits_all is None else np.concatenate([logits_all, probs.numpy()])
                label_all = label.numpy() if label_all is None else np.concatenate([label_all, label.numpy()])

    class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all, label_area_all)
    acc, class_precision, class_recall = metrics.class_measurement(
        intersect_area_all, pred_area_all, label_area_all
    )
    class_dice, mdice = metrics.dice(intersect_area_all, pred_area_all, label_area_all)
    kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)

    fg_index = min(foreground_class, len(class_iou) - 1)
    fg_precision = float(class_precision[fg_index])
    fg_recall = float(class_recall[fg_index])
    fg_f1 = 0.0 if fg_precision + fg_recall == 0 else 2 * fg_precision * fg_recall / (fg_precision + fg_recall)

    result = {
        "num_images": len(eval_dataset),
        "foreground_class": fg_index,
        "miou": float(miou),
        "mdice": float(mdice),
        "accuracy": float(acc),
        "kappa": float(kappa),
        "class_iou": [float(value) for value in class_iou],
        "class_dice": [float(value) for value in class_dice],
        "class_precision": [float(value) for value in class_precision],
        "class_recall": [float(value) for value in class_recall],
        "metrics": {
            "foreground_iou": float(class_iou[fg_index]),
            "foreground_dice": float(class_dice[fg_index]),
            "foreground_precision": fg_precision,
            "foreground_recall": fg_recall,
            "foreground_f1": float(fg_f1),
        },
    }
    if auc_roc and logits_all is not None and label_all is not None:
        result["auc_roc"] = float(
            metrics.auc_roc(logits_all, label_all, num_classes=eval_dataset.num_classes)
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate offline segmentation experiments.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu", "xpu", "npu", "mlu"], type=str)
    parser.add_argument("--dataset_root", default=None, type=str)
    parser.add_argument("--list_path", default=None, type=str)
    parser.add_argument("--dataset_name", default="", type=str)
    parser.add_argument("--experiment_name", default="", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--foreground_class", default=1, type=int)
    parser.add_argument("--aug_eval", action="store_true")
    parser.add_argument("--scales", nargs="+", type=float, default=1.0)
    parser.add_argument("--flip_horizontal", action="store_true")
    parser.add_argument("--flip_vertical", action="store_true")
    parser.add_argument("--is_slide", action="store_true")
    parser.add_argument("--crop_size", nargs=2, type=int)
    parser.add_argument("--stride", nargs=2, type=int)
    parser.add_argument("--auc_roc", action="store_true")
    parser.add_argument("--opts", nargs="+", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(args.config, opts=args.opts)
    cfg = _override_val_dataset(cfg, dataset_root=args.dataset_root, list_path=args.list_path)
    builder = SegBuilder(cfg)

    utils.set_device(args.device)
    model = builder.model
    utils.load_entire_model(model, args.model_path)

    test_config = merge_test_config(cfg, args)
    if args.auc_roc:
        test_config["auc_roc"] = True

    results = evaluate_dataset(
        model,
        builder.val_dataset,
        test_config=test_config,
        num_workers=args.num_workers,
        foreground_class=args.foreground_class,
    )
    results["config_path"] = str(Path(args.config).as_posix())
    results["best_model_path"] = str(Path(args.model_path).as_posix())
    results["dataset"] = args.dataset_name
    results["experiment_name"] = args.experiment_name
    results["dataset_root"] = args.dataset_root
    results["list_path"] = args.list_path
    dump_json(results, args.save_path)
    print(results)


if __name__ == "__main__":
    main()
