import json
import os
import numpy as np
import cv2
from PIL import Image


def coco_json_to_png(json_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 解析COCO格式JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 建立图像ID到文件名的映射
    img_id_to_name = {img['id']: img['file_name'] for img in data['images']}
    # 按图像ID分组标注
    annotations_by_img = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_img:
            annotations_by_img[img_id] = []
        annotations_by_img[img_id].append(ann)

    # 处理每张图像
    for img in data['images']:
        img_id = img['id']
        img_name = img_id_to_name[img_id]
        height, width = img['height'], img['width']
        # 创建空白掩码（初始值0，对应背景）
        mask = np.zeros((height, width), dtype=np.uint8)

        # 绘制该图像的所有标注
        if img_id in annotations_by_img:
            for ann in annotations_by_img[img_id]:
                category_id = ann['category_id']
                # 解析多边形（COCO格式的segmentation是多边形列表）
                for seg in ann['segmentation']:
                    # 转换为(x,y)坐标对
                    polygon = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    # 填充多边形区域为类别ID
                    cv2.fillPoly(mask, [polygon], color=category_id)

        # 保存为PNG（文件名与原图对应，替换扩展名）
        png_name = os.path.splitext(img_name)[0] + '.png'
        output_path = os.path.join(output_dir, png_name)
        Image.fromarray(mask, mode='L').save(output_path)
        print(f"已生成掩码: {output_path}")


# 配置路径
json_path = r"D:\dataset\3d_bearing.v1i.coco\annotations\val\_annotations.coco.json"
output_dir = r"D:\dataset\3d_bearing.v1i.coco\annotations\val"

# 执行转换
coco_json_to_png(json_path, output_dir)