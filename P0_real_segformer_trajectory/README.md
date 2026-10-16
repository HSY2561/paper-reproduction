# P0：真实 SegFormer 掩膜到裂缝骨架与修复轨迹

本目录由冻结的 Crack500 SegFormer-B0 checkpoint 对真实 `valcrop/20160222_080850_1281_361.jpg` 推理生成，未使用模拟掩膜，也未访问 test 集。

## 主图

- `P0_real_segformer_skeleton_trajectory.png`：论文实验图，依次展示原图、真实 SegFormer 掩膜、骨架、裂缝宽度图、坐标轨迹和离线平滑修复轨迹。
- `P0_real_segformer_skeleton_trajectory.pdf`：同版矢量输出。

## 中间结果

- `20160222_080850_1281_361_segformer_mask_raw.png`：模型原始二值预测。
- `20160222_080850_1281_361_segformer_mask_clean.png`：仅移除小于 8 像素孤立连通域后的显示掩膜。
- `20160222_080850_1281_361_skeleton.png`：`skimage.morphology.skeletonize` 生成的单像素骨架。
- `20160222_080850_1281_361_width_px.npy`：由距离变换得到的局部宽度估计（像素）。
- `20160222_080850_1281_361_trajectory_pixels.csv`：骨架最大连通分量端点路径，经 B 样条平滑后的像素轨迹。
- `20160222_080850_1281_361_trajectory_coordinates_provisional_mm.csv`：坐标映射接口输出；当前仍是占位比例，不能直接驱动机器人。
- `P0_metadata.json`：输入、模型哈希、后处理、像素统计和坐标状态。

## 可复现输入

- 配置：`experiments/retrain_wave4_20261007/configs/public_segformer_b0.yml`
- 权重：`experiments/retrain_wave4_20261007/checkpoints/segformer_b0/model.pdparams`
- 权重 SHA256：`07f6048e0a6de174a2c3ca1915df4bcd635565a706d978288fba946ef240f2d5`
- 数据根目录：`C:/Users/user/algolab_data`
- 执行脚本：`P0_real_pipeline.py`

## 坐标说明

仓库中没有本课题相机到路面工作平面的实测标定矩阵、毫米每像素比例或喷嘴外参。因此图 (e) 使用明确标记的 identity placeholder（1 px = 1 mm）来验证数据接口和绘图链路，不能作为真实物理尺寸或电机指令。获得相机内参、平面标定点和相机—三轴/喷嘴外参后，应把 `P0_real_pipeline.py` 中的坐标变换替换为实测单应矩阵或平面映射，并重新生成图 (e)、CSV 和 metadata。

## 当前 P0 结论

真实 SegFormer 掩膜、骨架和离线轨迹链路已经跑通；真实毫米坐标和机器人可执行轨迹仍需一次实测标定后才能宣称完成。当前轨迹是从最大骨架连通分量端点提取并用 B 样条平滑的离线中心线，不包含底盘闭环、喷嘴偏置补偿、速度/流量控制或在线重规划。
