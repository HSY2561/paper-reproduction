# WTConv 标准 Haar 小波实现与兼容说明

## 检查结论

当前工程中原来的 `WTConv` 名称对应的是一个工程候选块：低频路径为 `AvgPool2D(stride=2)` 后接 `1×1` 卷积，高频路径为 depthwise `5×5` 卷积，二者相加后经过 pointwise、BatchNorm、ReLU 和残差连接。该实现没有小波分析滤波器，也没有 `LL/LH/HL/HH` 四个子带和逆小波合成，因此不能称为标准小波卷积。

原 `candidate: wtconv` 配置和历史 checkpoint 保持不变，以保证已有 Wave14/Wave17/VOC 结果仍然可复现。新增实现使用显式候选名 `candidate: wtconv_wavelet`，不会把旧结果重新解释成标准 WTConv 结果。

## 新实现

新代码位于：

`experiments/segformer_priority_wave18_20261107/src/src/PaddleSeg-2.10.0/paddleseg/models/wtconv_wavelet.py`

并同步到当前 Crack500、VOCdevkit3000、自建域和 CameraStudio 的运行时源码中。

### 1. Haar 分解

对输入特征逐通道执行一层正交二维 Haar 分解：

\[
LL=(x_{00}+x_{01}+x_{10}+x_{11})/2
\]

\[
LH=(x_{00}-x_{01}+x_{10}-x_{11})/2
\]

\[
HL=(x_{00}+x_{01}-x_{10}-x_{11})/2
\]

\[
HH=(x_{00}-x_{01}-x_{10}+x_{11})/2
\]

其中 `LL` 表示低频上下文，`LH/HL/HH` 分别保留水平、垂直和对角细节。

### 2. 子带卷积与逆变换

每个子带使用 depthwise 卷积和逐通道可学习缩放因子处理，然后通过固定的逆 Haar 公式重建原分辨率。逆变换为：

\[
x_{00}=(LL+LH+HL+HH)/2,
\quad x_{01}=(LL-LH+HL-HH)/2
\]

\[
x_{10}=(LL+LH-HL-HH)/2,
\quad x_{11}=(LL-LH-HL+HH)/2
\]

重建后再通过轻量局部 depthwise 分支、pointwise 投影和残差连接输出。这样模块同时具备显式的上下文分解、局部细节处理和逆小波重建。

### 3. 奇数尺寸

当输入高或宽为奇数时，先复制最后一行或最后一列完成偶数化；逆变换后裁回原始高宽。因此模块不会改变输入尺寸，也不会因为 `400×400` 以外的奇数尺寸而发生错位。

### 4. 多层模式

`WTConvWavelet(..., levels=1)` 是当前轻量默认配置。`levels>1` 时只递归分解低频 `LL`，并按从粗到细的顺序逆向重建。SegFormer 配置支持：

```yaml
candidate: wtconv_wavelet
wavelet_levels: 1
wavelet_kernel: 3
```

默认一层配置的 `C=128` 模块约有 `24.2k` 个可训练参数（含 BatchNorm 运行统计时状态张量约 `24.4k`），不包含固定 Haar 变换参数；旧 AvgPool candidate 的历史结果不能直接用于新模块的性能结论，必须重新训练。

## 配置与验证

新增配置示例：

- `experiments/segformer_vocdevkit3000_aligned_wtconv_ablation_20261031/configs/C4_Aligned_WTConvWavelet.yml`
- `experiments/segformer_priority_wave18_20261107/configs/CFD/C3_Aligned_WTConvWavelet.yml`
- `experiments/segformer_priority_wave18_20261107/configs/DeepCrack/C3_Aligned_WTConvWavelet.yml`

CPU 验证脚本：

`experiments/segformer_priority_wave18_20261107/tests/test_wtconv_wavelet.py`

验证内容包括：偶数尺寸、奇数尺寸、极小尺寸的 DWT/IWT 重建误差，以及 `levels=1/2/3` 的前向形状和反向梯度。当前 Paddle 3.4 CPU 结果：重建最大绝对误差约 `2.4×10⁻⁷`，所有测试通过。

## 结果解释边界

这次改动只完成标准 Haar DWT/IWT 模块和可复现实验入口，没有把新模块的精度、延迟或 FLOPs 写入旧实验结果。旧 `wtconv` checkpoint 的参数结构也不能直接当作 `wtconv_wavelet` checkpoint 使用。下一步若要报告收益，应在同一数据划分、初始化、训练预算、EMA 和选模协议下重新训练至少 `B0`、`Aligned`、`WTConvWavelet` 和 `Aligned+WTConvWavelet` 四组，再进行统一验证和部署测速。

