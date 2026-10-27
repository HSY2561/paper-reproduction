## 最新进度（2026-10-27）

- **Wave12 首轮验证快照**：D01-D10、C01/C02 共 12 个 seed42 配置完成 18,000 iter 训练、官方验证和像素/边界核验，相关回执均 exit=0；全部为 val348，`test_used=false`。
- **测速阶段未形成可用放行结论**：phase2 profile 在一次内存子进程延期后失败（workflow exit=1），失败回执保留；没有最终模型选择，也没有加速声明。
- **门槛结果**：12 行当前 `accuracy.passed=false`，因此不把 C01 等较高单次 val 分数称为候选或最终模型。权重、数据集、原始日志、缓存和 profile 目录未上传。详见 [Wave12 快照](docs/wave12_snapshot/README.md) 与 [结果索引](results/wave12_20261027/comparison.json)。

- **Wave12 首轮验证快照**：D01-D10、C01/C02 共 12 个 seed42 配置完成 18,000 iter 训练、官方验证和像素/边界核验，相关回执均 exit=0；全部为 val348，`test_used=false`。
- **测速阶段未形成可用放行结论**：phase2 profile 在一次内存子进程延期后失败（workflow exit=1），失败回执保留；没有最终模型选择，也没有加速声明。
- **门槛结果**：12 行当前 `accuracy.passed=false`，因此不把 C01 等较高单次 val 分数称为候选或最终模型。权重、数据集、原始日志、缓存和 profile 目录未上传。详见 [Wave12 快照](docs/wave12_snapshot/README.md) 与 [结果索引](results/wave12_20261027/comparison.json)。

- **Wave12 首轮验证快照**：D01-D10、C01/C02 共 12 个 seed42 配置完成 18,000 iter 训练、官方验证和像素/边界核验，相关回执均 exit=0；全部为 val348，`test_used=false`。
- **测速阶段未形成可用放行结论**：phase2 profile 在一次内存子进程延期后失败（workflow exit=1），失败回执保留；没有最终模型选择，也没有加速声明。
- **门槛结果**：12 行当前 `accuracy.passed=false`，因此不把 C01 等较高单次 val 分数称为候选或最终模型。权重、数据集、原始日志、缓存和 profile 目录未上传。详见 [Wave12 快照](docs/wave12_snapshot/README.md) 与 [结果索引](results/wave12_20261027/comparison.json)。

# Crack500 裂缝分割：SegFormer 精度与效率改进

当前研发基准固定为 **A0 = SegFormer-B0 + width128 + Aligned**。目标是兼顾裂缝分割精度、参数量和机器人部署效率。原始SegFormer-B0（width256）保留为公开模型对照。

## 最新进度（2026-10-26）

新增已有进度快照，不重新训练或评估：

- **统一终评已完成**：6 个公开模型＋A0＋冻结 F08，共 8 组 test×1124 图及 12 组 val×348 图，统一队列和包装器 exit=0。F08 的 test 裂缝 IoU 为 **58.5583%**、F1 **73.8634%**、mIoU **77.7341%**；参数减少并不代表加速。旧历史 test 表保留，与本次同后端结果分开。
- **Wave11全部完成**：15个18k（三seed四格＋匹配R0）、独立验证、同机400/480交错延迟/显存及固定24图均已核验。45份训练/验证和科学量化子进程退出为0；原队列的`profile_pending:true`保留为训练结束时快照，新的完成状态另记。
- **结论未放宽**：M2相对E0三个seed均改善，但IoU仅+0.1490、BF1+0.2314个百分点，未达到预设幅度；E3没有稳定互补，减参10.20%伴随IoU−0.5601个百分点和更高延迟。480输入显存约减53%，不等同加速或完整非劣。
- [W11最终报告与完整消融表](docs/wave11_final_20261026.md) · [348逐图/三seed/效率CSV](results/wave11_20261026/final/README.md) · [24图预测和误差画廊](figures/wave11_qualitative_20261026/README.md) · [审计摘要](evidence/wave11_final_20261026/audit_summary.json)。W11未访问test，不替换前一阶段冻结F08的统一test结论。

以下保留 2026-10-25 的轻量融合筛选阶段背景；其中“最终 test 待完成”状态已由上述完成回执取代，不与 Wave11 的新匹配实验混用。

本轮 **20个轻量融合/宽度配置及4次额外种子训练均完成18,000 iter和独立验证**。20配置包含宽度控制及机制/宽度复用，不等于20个独立算法。

- **F08_Branch32通过匹配三seed精度门槛**；相对A0平均裂缝IoU提高0.3761个百分点、F1提高0.2650个百分点。参数减少0.9295%，尚未达到本轮额外减参至少1%的目标。
- **W96三seed未通过完整门槛**；虽然相对A0减参1.5306%，但平均IoU和Recall下降超过预设容差。
- 同进程交错测速已有真实退出回执，但候选延迟波动未达到稳定性标准，F08也没有证明比原始B0更快。参数更少不能代替加速证据。
- B0共同初始化补控已完整18k并独立验证；冻结候选与六个原始模型的统一评价已结束，终表见[统一比较 CSV](results/unified_evaluation_20261025/comparison.csv)。已有公开模型历史test单独列示。
- L01（CE＋0.1 Dice）是通过匹配三seed均值门槛的独立损失备选，尚未与F08联合训练。

以下均为 **Crack500 val348，seed42/43/44均值**，不能和历史test分数直接比较：

| 模型 | 裂缝IoU(%) | F1(%) | mIoU(%) | Params(M) | 当前结论 |
|---|---:|---:|---:|---:|---|
| 原始SegFormer-B0（历史控制） | 68.0389 | 80.9798 | 82.9082 | 3.715170 | 原始公开对照 |
| A0：width128＋Aligned | 68.3353 | 81.1890 | 83.0631 | 3.470434 | 后续研发基准 |
| W96 | 68.0957 | 81.0198 | 82.9464 | 3.417314 | 多seed完整门槛未通过 |
| F08_Branch32 | 68.7114 | 81.4540 | 83.2626 | 3.438178 | 精度通过；减参不足1%；统一test已另表发布 |

B0共同初始化补控已于16:15完成，其验证mIoU为82.7134%；更新后的三seed控制单独列于[补控记录](results/b0_verified_initialization_control.json)。上表B0仍为明确标注的历史控制，最终统一test已使用新冻结B0；A0与候选的成对门槛保持原登记。

本项目目前不宣称领域SOTA、所有指标最优或所有对照中参数最少。完整Precision、Recall、F1、IoU、mIoU、边界F1、样本SD和逐seed差值见[结果报告](docs/results.md)。

![轻量融合与参数折中](figures/fusion20_pareto.png)

## 从哪里开始

| 内容 | 入口 |
|---|---|
| 当前进度与已核验范围 | [状态JSON](results/current_status.json) / [结果报告](docs/results.md) |
| 训练、选模、指标与门槛 | [实验协议](docs/protocol.md) |
| A0复现与环境准备 | [运行说明](runtime/README.md) |
| 20配置及历史结构/损失消融 | [results目录](results/) |
| 六个原始公开模型及已有test | [公开模型对照与配方](docs/public_models.md) |
| 冻结权重SHA、真实退出核验 | [evidence目录](evidence/) |
| 后续实验和放行顺序 | [后续任务](docs/next_steps.md) |
| 实物应用规划 | [机器人阶段](docs/robotics_scope.md) |
| 当前证据限制 | [限制与口径](docs/limitations.md) |

## 仓库内容

`overlays/`保存与固定官方PaddleSeg版本配套的模型改动；`scripts/`提供源码准备、配置生成和官方入口调用；`configs/`保存A0家族及公开模型的实际配方模板。`third_party/`记录来源、上游归档SHA和许可证。

`results/`、`figures/`和`evidence/`只保存小型结果、图和复现索引。数据集、权重、环境、训练输出及机器专用路径由参数指定，按`.gitignore`排除。历史失败完整保留在汇总表中，旧BiSeNet/C4/SCSegamba进度和重复资料已移出当前工作树；[本次整理范围](docs/repository_scope.md)说明了保留边界。

当前训练硬件为本地RTX5060，FP32、batch4、AdamW、lr6e-5、end_lr=0、18k iter、seed42/43/44。公开模型存在初始化、损失、EMA和后端差异，比较时必须连同配方披露。裁块之间还存在源帧相关风险；多轮val筛选不等于未见数据泛化证明。
