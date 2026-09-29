import cv2
import numpy as np
from pathlib import Path


## 采用ORB+RANSAC估计单应性（Homography）进行图像配准，并裁切为正方形 ###
##  Oriented FAST and Rotated BRIEF （ORB)、Random Sample Consensus（随机采样一致性）


# ====== 1) 读取图像 ======
before_path = Path("before.jpg")
after_path  = Path("after.jpg")

img1 = cv2.imread(str(before_path))
img2 = cv2.imread(str(after_path))
assert img1 is not None and img2 is not None, "图片读取失败"

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
if (w1 != w2) or (h1 != h2):
    target_w, target_h = min(w1, w2), min(h1, h2)
    img1 = cv2.resize(img1, (target_w, target_h))
    img2 = cv2.resize(img2, (target_w, target_h))

g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ====== 2) 特征匹配 + 单应性 ======
orb = cv2.ORB_create(6000)
k1, d1 = orb.detectAndCompute(g1, None)
k2, d2 = orb.detectAndCompute(g2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(d1, d2, k=2)
good = [m for m, n in matches if m.distance < 0.75 * n.distance]
assert len(good) >= 8, "匹配点不足"

src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, inliers = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)
assert H is not None, "单应性估计失败"

aligned = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
cv2.imwrite("aligned_after.png", aligned)

matches_debug = cv2.drawMatches(
    img1, k1, img2, k2, good[:200], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("matches_debug.png", matches_debug)

# ====== 3) 计算无黑边重叠区域 ======
h, w = img1.shape[:2]
geom_mask_ref = np.ones((h, w), np.uint8) * 255
geom_mask_aln = cv2.warpPerspective(geom_mask_ref, H, (w, h))

thr = 10
nb_ref = (cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) > thr).astype(np.uint8) * 255
nb_aln = (cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) > thr).astype(np.uint8) * 255

overlap_mask = cv2.bitwise_and(geom_mask_ref, geom_mask_aln)
overlap_mask = cv2.bitwise_and(overlap_mask, nb_ref)
overlap_mask = cv2.bitwise_and(overlap_mask, nb_aln)
overlap_mask = cv2.erode(overlap_mask, np.ones((7, 7), np.uint8), iterations=1)

ys, xs = np.where(overlap_mask > 0)
x0, x1 = int(xs.min()), int(xs.max())
y0, y1 = int(ys.min()), int(ys.max())

# ====== 4) 裁剪无黑边矩形 + 最大正方形 ======
crop_before = img1[y0:y1+1, x0:x1+1].copy()
crop_after  = aligned[y0:y1+1, x0:x1+1].copy()
cv2.imwrite("crop_before.png", crop_before)
cv2.imwrite("crop_after.png",  crop_after)

Hc, Wc = crop_before.shape[:2]
side = min(Hc, Wc)
x_start = (Wc - side) // 2
y_start = (Hc - side) // 2
square_before = crop_before[y_start:y_start+side, x_start:x_start+side]
square_after  = crop_after [y_start:y_start+side, x_start:x_start+side]
cv2.imwrite("square_before.png", square_before)
cv2.imwrite("square_after.png",  square_after)

# ====== 5) 快速叠加图 ======
gray = cv2.cvtColor(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
overlay = gray.copy()
overlay[..., 2] = aligned[..., 2]
cv2.imwrite("overlay.png", overlay)

# ====== 6) 中心裁剪原尺寸的90% ======
def crop_center_percent(img, percent: float):
    """从中心裁剪指定比例的图像区域"""
    h, w = img.shape[:2]
    new_h, new_w = int(h * percent), int(w * percent)
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    return img[y0:y0 + new_h, x0:x0 + new_w].copy()

percent = 0.8  # 改为0.8可裁80%
sb_90 = crop_center_percent(square_before, percent)
sa_90 = crop_center_percent(square_after,  percent)
cv2.imwrite("square_p80_before.png", sb_90)
cv2.imwrite("square_p80_after.png",  sa_90)

print(f"✅ 处理完成：已自动配准、裁去黑边并中心裁剪 {percent*100:.0f}% 区域")
