# E0–E3 CPU 参数与接口审计（2026-10-25）

本审计只在 CPU 构造现有 A0、原始/压缩骨干及临时原生层容器，读取预训练权重形状并计数。没有修改既有源码，没有实现 E0–E3 完整网络，没有读取图像，没有运行 GPU、训练或推理速度测试。命令真实 exit code 为 0；设备回执为 `cpu`，Paddle 3.4.0。组件可构造不等于完整网络已实现或 GPU smoke 已通过。

源码：`${PROJECT_ROOT}\experiments\segformer_lightfusion_wave10_20261024\src\PaddleSeg-2.10.0`。详细机器可读证据见同目录 `fourcell_parameter_audit_20261025.json`。

## 1. 计数口径与最终预算

沿用本项目历史口径：`sum(numel(p) for p in model.parameters())`，包含 Paddle 注册在 parameters 中、但 `stop_gradient=True` 的 BN running mean/variance。另列可训练数量，不能将 BN 统计称作可训练参数。LayerNorm 权重和偏置均为可训练参数。

**M1 固定使用四个 Conv1×1-BN-ReLU 投影，Conv bias=False。** 其头部实测构造计数为 **59,394**，不是原参考预算 58,722。

| 配置 | 定义 | 注册总量（项目口径） | 可训练量 | BN 非训练统计 | 相对 A0 减少 |
|---|---|---:|---:|---:|---:|
| R0/A0 | 原 MiT-B0 + 原 A0 解码器 | 3,470,434 | 3,470,178 | 256 | 0% |
| E0 | 压缩骨干 + 原 A0 解码器 | 3,202,658 | 3,202,402 | 256 | 7.7159% |
| E1 | 压缩骨干 + M1 | 3,111,010 | 3,110,178 | 832 | 10.3567% |
| E2 | 压缩骨干 + 原 A0 解码器 + M2 | 3,208,148 | 3,207,540 | 608 | 7.5577% |
| E3 | 压缩骨干 + M1 + M2 | 3,116,500 | 3,115,316 | 1,184 | 10.1986% |

表中 R0/A0 是现有网络实际构造计数；E0–E3 是核实骨干与头部组件后相加的**完整网络预算**，不是 E0–E3 完整网络实现或结果。压缩骨干 3,051,616；原骨干 3,319,392；原 A0 解码器 151,042（可训练 150,786）。E1/E3 在预算层面可满足相对 A0 至少减参 10%，不能据此推断更快或精度不降。

## 2. S4 FFN 压缩与预训练裁剪

本地 `mix_transformer.py` 的 MixFFN 使用 `nn.Linear`，**不是把 fc1/fc2 当作 Conv2D**。两个 stage4 block 的通道 C=256，hidden H=1024→768；所有 stage 输出维度不变，仍为 [32,64,160,256]。

每个 FFN 的注册量为 `2*C*H + 11*H + C`：两个有 bias 的 Linear、一个有 bias 的 3×3 depthwise convolution。单块从 535,808 降至 401,920，减少 133,888；两块合计 **267,776**。实际 CPU 构造骨干差值也为 267,776。

对 `block4.0` 和 `block4.1` 各自使用预先固定的同一组 768 个索引 I，裁剪轴必须为：

| 张量 | 原形状 → 压缩形状 | 裁剪方式 |
|---|---|---|
| `mlp.fc1.weight` | [256,1024] → [256,768] | `[:, I]` |
| `mlp.fc1.bias` | [1024] → [768] | `[I]` |
| `mlp.dwconv.dwconv.weight` | [1024,1,3,3] → [768,1,3,3] | `[I, :, :, :]` |
| `mlp.dwconv.dwconv.bias` | [1024] → [768] | `[I]` |
| `mlp.fc2.weight` | [1024,256] → [768,256] | `[I, :]` |
| `mlp.fc2.bias` | [256] → [256] | 不裁剪 |

同一个 block 内必须联动相同隐藏索引，否则破坏特征对应。两个 block 的索引可按预登记规则独立确定，但 E0/E1/E2/E3 必须共享完全相同的压缩初始化 tensor 和 SHA；不得用 val/test 结果选择裁剪索引，也不得将 A0 微调后权重当成与 ImageNet 初始化同预算的起点。

官方工厂 `MixVisionTransformer_B0(**kwargs)` 已硬编码 `mlp_ratios=[4,4,4,4]`。实际 CPU 调用 `MixVisionTransformer_B0(mlp_ratios=[4,4,4,3])` 报 `got multiple values for keyword argument 'mlp_ratios'`。因此不能只修改 YAML 并假设支持；未来隔离实现应使用与 B0 参数一致的底层 `MixVisionTransformer` 构造或明确命名的压缩适配构造器。

预训练文件 `weights\mit_b0_imagenet.pdparams` SHA256：`211e15d5157b8d48b1931102c40094da0b578de1c153c5706360f7ad561a774e`。加载到 CPU 内存后形状与上表一致。

## 3. M1：四投影 CBR 版

除 classifier 外全部 Conv bias=False；ConvBNReLU 的 BN 项目口径为 4×输出通道（scale、bias、running mean、running variance）。

| 组件 | 注册量 | 可训练量 |
|---|---:|---:|
| C1：32→128，1×1 CBR | 4,608 | 4,352 |
| C2：64→32，1×1 CBR | 2,176 | 2,112 |
| C3：160→32，1×1 CBR | 5,248 | 5,184 |
| C4：256→32，1×1 CBR | 8,320 | 8,256 |
| C2/C3/C4 在 C2 尺度拼接 96→64，1×1 CBR | 6,400 | 6,272 |
| Aligned low：128→32，1×1 Conv | 4,096 | 4,096 |
| Aligned high：64→32，1×1 Conv | 2,048 | 2,048 |
| flow：64→2，3×3 Conv | 1,152 | 1,152 |
| low128 与 warped-high64 拼接，192→128，1×1 CBR | 25,088 | 24,832 |
| classifier：128→2，1×1 Conv，有 bias | 258 | 258 |
| 合计 | **59,394** | **58,562** |

Dropout、插值、拼接、warp 不增加参数。官方 `sfnet.AlignedModule(inplane,outplane)` 的 low/high 共用输入通道数；不能直接传 low128/high64。必须在未来的新隔离适配路径中保留官方 flow/warp 算法而显式定义两个输入投影，不能把官方同通道接口已通过构造视为不对称接口已通过。

**原 58,722 数字的来源已定位：**四个投影若改为 SegFormer 原式有 bias 的 Linear（无 BN/ReLU），临时组件构造总量恰为 58,722、可训练 58,338。CBR 与 Linear 四投影相差 672。该替代版本只用于解释预算差异，**不进入本轮主配置，也没有为迁就旧数字而替换设计。**

空间接口建议用明确目标 `size`：C3/C4 先恢复到 C2，再生成 high64；flow 生成时 high64 投影恢复到 C1，最终 warp 的源是未预先放大到 C1 的 high64。400×400 输入对应 C1=100×100、C2=50×50、C3=25×25、C4=13×13；不能用固定倍数假定 C4→C2 恰好是四倍。400 裁剪和奇数原图恢复均应单独做形状/坐标 smoke。

## 4. M2：5490 的构成

`D=CBR(C1:32→16)`；`S=CBR(F:128→16)`；`g=sigmoid(Conv1×1([D,S]:32→1))`；`R=ConvBN16→128(DW-CBR16(D))`；`F'=F+alpha*g*R`。alpha 为单个可训练标量，初始 0.1。D 在 gate 与 R 中共享同一投影，不额外复制。

| 组件 | 注册量 | 可训练量 |
|---|---:|---:|
| D：32→16，1×1 CBR | 576 | 544 |
| S：128→16，1×1 CBR | 2,112 | 2,080 |
| g：32→1，1×1 Conv，有 bias | 33 | 33 |
| DW：16→16，3×3 depthwise CBR | 208 | 176 |
| R 输出：16→128，1×1 ConvBN，无最后 ReLU | 2,560 | 2,304 |
| alpha | 1 | 1 |
| 合计 | **5,490** | **5,138** |

BN 非训练统计合计 352。可训练 5,138 与项目口径 5,490 不能混用。alpha=0.1 使初始残差较小且分支第一步仍可获得梯度；若改成 alpha=0，则第一步 D/S/g/R 权重梯度可能为零，不能把该现象直接当作永久死分支。

## 5. 死分支与审计门槛

1. E1/E3 应完整替换原 A0 的四个 Linear、512→128 fusion、原同通道 Aligned 和 classifier 注册路径；不得新旧头都实例化后仅跳过旧 forward，否则参数统计包含死分支并破坏预算。
2. E2 必须让 M2 输出送入原 dropout/classifier；E3 让同一 M2 作用于 M1 的128通道、C1尺度输出。不能计算后丢弃，也不能 detach g/R。
3. S 只用于 gate，D 同时用于 gate/R；二者均有前向作用。alpha 固定为参数、不能只用 Python 数值冒充已计数标量。
4. 正式隔离实现后要核 state_dict、实际完整模型总量和可训练量，并在非饱和输入上检查可训练参数梯度覆盖。BN running mean/variance 不参与梯度，不能把它们误判为死分支。
5. E0–E3 需要完整网络 CPU/GPU forward、backward、奇数尺寸恢复与相同初始化 SHA 核验后才可训练。预算更低不代表 grid_sample 更快；需最终同机交错延迟、峰值显存和覆盖说明明确的 FLOPs。

没有任何精度、速度、互补性或 SOTA 结果由本审计产生。四格效应应在真实训练后由 E0/E1/E2/E3 同协议结果检验。
