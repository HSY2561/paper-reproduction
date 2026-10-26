# W11 初始化审计

2026-10-25，`${WINDOWS_ENV}\Scripts\python.exe` / Paddle 3.4 / CPU。实际子进程 exit 0，见 [initialization_exit.json](initialization_exit.json) 和 完整日志（本地来源：`../logs/prepare_initialization.log`，未公开副本）。初始化过程未读数据集、未训练、未执行 GPU 运算。

## 原始权重与固定裁剪

原始 ImageNet SHA256 为 `211e15d5157b8d48b1931102c40094da0b578de1c153c5706360f7ad561a774e`，与预登记一致。两个 S4 block 分别按预登记 CPU float64 归一化 L1 打分，score 降序、原索引升序打破并列，取 top768 后按原索引升序保存。Paddle Linear 的 [in,out] 排列已联动裁剪 `fc1[:,I]`、`fc1.bias[I]`、`DW[I]`、`DW.bias[I]`、`fc2[I,:]`；`fc2.bias` 保留。

- 176键严格派生骨干（本地来源：`../weights/mit_b0_s4ffn768_imagenet.pdparams`，未公开副本）：SHA256 `11498f1d0a5ec9bcc853b814ec312571ff4b83ca12845310af6756d92a9c58d1`。
- [通道索引](../weights/s4ffn768_indices.json)：SHA256 `b3f070c23686a76cbc24b253c11abfcbd122c96074363ee9c0d0df200feeb1e8`。
- [float64分数](s4ffn768_scores_float64.json)、[block0逐通道CSV](s4_block0_channel_scores.csv)、[block1逐通道CSV](s4_block1_channel_scores.csv)。
- [逐张量转换账本](compression_tensor_ledger.json) 记录原始/派生形状、dtype、原始字节 SHA、转换轴和包含关系；[压缩审计JSON](compression_audit.json) 记录全部文件 SHA 与误差。

原始文件178键，其中 ImageNet 分类 `head.weight`、`head.bias` 不属于分割骨干的176键。它们被**显式排除**于严格模型文件，并原样保留于 178键派生全源文件（本地来源：`../weights/mit_b0_s4ffn768_imagenet_full_source.pdparams`，未公开副本）。全源文件共修改10个张量，另外168个张量逐项 bitwise 相同；严格骨干内部对应10个修改和166个原样张量。没有静默跳过键或形状不匹配。

## FFN 等价验收

每个 block 使用原1024通道 FFN，并将未选中通道的 fc2 权重行置零，构成 masked-original 参照。原参照与768通道压缩 FFN 均使用 eval、dropout=0；CPU FP32 固定随机输入覆盖 `(B,H,W)=(1,13,13),(2,7,9),(1,1,1)`。六项全部通过 `atol=1e-5,rtol=1e-5`，最大绝对误差 `3.814697265625e-06`。该验收不声称压缩 FFN 等于完整1024通道 FFN。

## 十二组完整初始化

输出为 `weights/init/E{0..3}_s{42..44}.pdparams`，全部是完整模型 state_dict。每个文件都经过保存/回读、全部键显式回灌，以及正式运行采用的 `SegFormerComplementarity(pretrained=完整init)` 严格构造回读；逐项 bitwise 相同。

| 组别 | 完整state键数 | 注册量 | 可训练量 |
| --- | ---: | ---: | ---: |
| E0 | 194 | 3,202,658 | 3,202,402 |
| E1 | 211 | 3,111,010 | 3,110,178 |
| E2 | 217 | 3,208,148 | 3,207,540 |
| E3 | 234 | 3,116,500 | 3,115,316 |

每个 top-level/具名子组件在原生 Paddle 构造前立即设置 `SHA256(W11-v1|initialization|seed|组件名)` 派生种子。完整模型构造使用不同、随后丢弃的随机流，再显式覆盖每个注册张量。组件逆序重建在三个 seed 上均逐张量 bitwise 一致，证明新增构造的随机消耗不会改变已共享张量。

每seed五项共享检查全部通过，合计15项：四组骨干相同、四组分类器相同、E0/E2旧头相同、E1/E3 M1头相同、E2/E3 M2相同。M2 alpha 是精确 float32 `0.1`；全部 BN 运行均值为0、运行方差为1并单独登记。

[初始化manifest](initialization_manifest.json) 保存每个文件 SHA、逐张量清单路径/SHA、具名种子、组件聚合 SHA、三项构造顺序检查和15项配对检查。逐张量清单目录（本地来源：`initialization_tensors`，未公开副本） 保存十二组完整 shape/dtype/numel/字节 SHA。

训练接口：`model.type=SegFormerComplementarity`，`model.pretrained=对应完整init`，`backbone.type=MixVisionTransformer_B0_S4FFN768`，`backbone.pretrained=null`。模型加载后必须重新设置运行 seed，再开始正式数据排列/增强；数据与训练随机流审计由 runtime 单独记录。
