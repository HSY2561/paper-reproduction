"""Threshold diagnostics for AW; model selection remains val-only."""
import argparse
import json
from pathlib import Path

import numpy as np
import paddle
from paddle.nn import functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="gpu")
    args = parser.parse_args()

    from paddleseg.cvlibs import Config, SegBuilder
    from paddleseg.core import infer
    from paddleseg.utils import utils

    cfg = Config(args.config)
    utils.set_device(args.device)
    builder = SegBuilder(cfg)
    model = builder.model
    model.set_state_dict(paddle.load(args.model_path))
    model.eval()
    dataset = builder.val_dataset
    loader = paddle.io.DataLoader(dataset, batch_size=1, shuffle=False,
                                  num_workers=0, return_list=True)
    thresholds = np.arange(0.30, 0.701, 0.025)
    counts = {float(t): [0, 0, 0, 0] for t in thresholds}
    with paddle.no_grad():
        for data in loader:
            _, logits = infer.inference(model, data["img"], trans_info=data["trans_info"])
            prob = F.softmax(logits, axis=1)[:, 1].numpy().reshape(-1)
            target = np.asarray(data["label"]).reshape(-1).astype(np.uint8)
            for threshold in thresholds:
                pred = (prob >= threshold).astype(np.uint8)
                tp = int(np.logical_and(pred == 1, target == 1).sum())
                fp = int(np.logical_and(pred == 1, target == 0).sum())
                fn = int(np.logical_and(pred == 0, target == 1).sum())
                tn = int(np.logical_and(pred == 0, target == 0).sum())
                c = counts[float(threshold)]
                c[0] += tn; c[1] += fp; c[2] += fn; c[3] += tp
    rows = []
    for threshold, (tn, fp, fn, tp) in counts.items():
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        bg_iou = tn / (tn + fp + fn) if tn + fp + fn else 0.0
        rows.append(dict(threshold=threshold, tn=tn, fp=fp, fn=fn, tp=tp,
                         precision=precision, recall=recall, f1=f1,
                         crackIoU=iou, mIoU=(bg_iou + iou) / 2))
    result = dict(split=args.split, config=str(Path(args.config).resolve()),
                  model_path=str(Path(args.model_path).resolve()),
                  rows=rows, test_used=args.split == "test")
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

