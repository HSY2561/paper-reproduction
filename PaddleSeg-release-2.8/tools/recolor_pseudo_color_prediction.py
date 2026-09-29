# coding: utf8
import argparse
import os
import os.path as osp

import numpy as np
from PIL import Image


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recolor pseudo_color_prediction from red/green to black/white.")
    parser.add_argument(
        "--input_dir",
        default=r"output\result\pseudo_color_prediction",
        help="Directory containing pseudo color predictions.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Default: input_dir + '_bw'.")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite files in input_dir.")
    parser.add_argument(
        "--tol",
        type=int,
        default=0,
        help="Tolerance for matching red/green (0-255).")
    parser.add_argument(
        "--auto_binary",
        action="store_true",
        help="If only two colors exist, map the dominant color to black and the other to white.")
    return parser.parse_args()


def is_image_file(path):
    return osp.splitext(path)[1].lower() in SUPPORTED_EXTS


def recolor_image(path, tol, auto_binary):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8, copy=True)

    if auto_binary:
        colors, counts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
        if colors.shape[0] == 2:
            idx = np.argsort(-counts)
            bg = colors[idx[0]]
            fg = colors[idx[1]]
            bg_mask = np.all(arr == bg, axis=2)
            fg_mask = np.all(arr == fg, axis=2)
            arr[bg_mask] = (0, 0, 0)
            arr[fg_mask] = (255, 255, 255)
            return Image.fromarray(arr, mode="RGB")

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Match by channel dominance to handle non-pure colors.
    r16 = r.astype(np.int16)
    g16 = g.astype(np.int16)
    b16 = b.astype(np.int16)
    red_mask = (r16 - g16 >= tol) & (r16 - b16 >= tol)
    green_mask = (g16 - r16 >= tol) & (g16 - b16 >= tol)

    arr[red_mask] = (0, 0, 0)
    arr[green_mask] = (255, 255, 255)

    return Image.fromarray(arr, mode="RGB")


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = input_dir if args.inplace else args.output_dir
    if output_dir is None:
        output_dir = input_dir.rstrip("\\/") + "_bw"

    if not osp.isdir(input_dir):
        raise SystemExit("Input directory not found: {}".format(input_dir))

    for root, _, files in os.walk(input_dir):
        rel = osp.relpath(root, input_dir)
        target_dir = output_dir if rel == "." else osp.join(output_dir, rel)
        if not osp.exists(target_dir):
            os.makedirs(target_dir)

        for name in files:
            src = osp.join(root, name)
            if not is_image_file(src):
                continue
            dst = osp.join(target_dir, name)
            out_img = recolor_image(src, args.tol, args.auto_binary)
            out_img.save(dst)
            print("Saved:", dst)


if __name__ == "__main__":
    main()
