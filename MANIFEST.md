## 2026-10-25：当前视觉阶段最终交付

- [最终报告](D:/论文复现/最终视觉模型实验报告_20261025.md)、[离线图文版](D:/论文复现/最终视觉实验交付_20261025/最终视觉模型实验报告.html)、[完整表图包](D:/论文复现/SegFormer最终实验报告与完整表图_20261025.zip)。
- 20配置、三seed、补控及六原始模型统一精确test/profile已完成；最终F08，W96失败。
- 有精度/参数/显存收益，延迟退化；source-frame-disjoint/自建/机器人仍待完成，不声称SOTA。

# 交付索引

此索引从本次交付开始登记，不声称枚举此前全部工程文件。

## 2026-10-20：Wave8 首项完整训练与后续进度

- [当前代码与证据入口](experiments/segformer_loss_wave8_20261019/README.md)、[损失消融进度](SegFormer损失函数消融进展_20261019.md)、[冻结执行计划](SegFormer损失函数优化20方案执行计划_20261019.md)
- 04:22 后更新：L05–L07 三项 CE + Lovasz 配置均完成 18k 及独立验证；L08 加权 CE 最近状态为训练中；L01–L04 梯度预检失败，保留证据不填性能。
- 原先两个替补配置已在训练前停用并移入 `provenance/prelaunch_disabled_reserves/` 保存，不按验证成绩扩充本轮搜索。

## 2026-10-19：Wave8 损失实验准备快照

- [损失目录与预检代码](experiments/segformer_loss_wave8_20261019/)
- [上一轮九项复核审计](experiments/segformer_loss_wave8_20261019/provenance/previous_followup_audit.json)
- 历史状态：这是准备阶段快照；当前正式训练进度见本页顶部 10 月 20 日条目。

## 2026-10-19 20:19：归因与三 seed 队列完成

- [最新完成说明与三 seed 汇总](experiments/segformer_aligned_wave7_20261018/followup_20261019/README.md)
- [全部 9 项完成回执](experiments/segformer_aligned_wave7_20261018/followup_20261019/results/queue_done.json)、[均值与样本标准差](experiments/segformer_aligned_wave7_20261018/followup_20261019/results/three_seed_summary.json)
- 状态：9 项新训练均 complete；B0/A0/A21/A10 的三 seed 齐备。速度条件与预登记门槛仍须复核，六模型及本轮统一 test 尚未完成。

## 2026-10-19：Crack500 文件核验与复核进度更新

- [数据来源、目录与完整性结论](analysis/crack500_integrity_20261019/README.md)
- [机器可读核验结果](analysis/crack500_integrity_20261019/audit.json)、[逐文件 SHA256](analysis/crack500_integrity_20261019/file_sha256.json)
- 3,368 对裁剪图像/标签全部存在并可解码，与本地原包的清单和前景标签一致；文件名前缀跨划分风险独立记录，未擅自调整划分。
- 此处为较早交付记录；20:19 后续 9 项队列已全部完成，最新状态以本页顶部完成说明为准。

## 2026-10-19：Wave7 筛选完成与候选复核

- [阶段报告](SegFormer_Aligned模块筛选阶段报告_20261019.md)、[代码与证据入口](experiments/segformer_aligned_wave7_20261018/README.md)
- [全量结果 CSV](experiments/segformer_aligned_wave7_20261018/results/ablation_percent_20261019.csv)、[独立审计](experiments/segformer_aligned_wave7_20261018/results/audit_complete_20261019.json)
- [归因与多 seed 复核进度](experiments/segformer_aligned_wave7_20261018/followup_20261019/refine-logs/RESULTS.md)
- 状态：22 项完整筛选训练已完成；后续三 seed 训练已补齐，速度稳定性未通过，六个公开模型和本轮统一 test 仍待完成。

## 2026-10-17：Wave6 模块消融训练结果

- [最新结果与改进建议](SegFormer模块消融结果与改进建议_20261017.md)
- [代码、配置与证据入口](experiments/segformer_balance_wave6_20261016/README.md)
- [完整消融结果表](experiments/segformer_balance_wave6_20261016/refine-logs/EXPERIMENT_RESULTS.md)
- 10 月 18 日更新：SE 已重新完成 18k 训练与验证，累计 15 个完整组（13 模块、2 宽度）；PSP smoke 失败。10 月 17 日超时仅是历史记录，当前结论仍限于单 seed 验证集筛选。

## 2026-10-16：10×6.5 m 场地方案

- [完整实验方案](10m×6.5m场地_麦克纳姆扫描建图与定点修复完整实验方案_20261016.md)
- [本轮数据、图表、模板与证据清单](experiments/field_10x6p5_20261016/MANIFEST.md)
- 状态：方案与数值规划已完成；单张航拍推理成功执行但视觉结果不合格；实车采集、标定、建图和修复待执行。

## 2026-10-17：缩小验证区域与效率改进

- [文献依据及执行建议](实验范围缩小与效率改进_文献依据及执行建议_20261017.md)
- [小区域路线计算](experiments/field_efficiency_20261017/roi_calculations.json)
- [采集预算图](experiments/field_efficiency_20261017/采集时间预算_范围与视野对照.png)
- 状态：文献核验与数值规划完成；推荐2×1.5 m先导、3×2 m主验证。实际区域待量测，未运行实车。


## 2026-10-19：局部修复路线选择与空鼓检测论文借鉴

- [三种修复方案比较、参考文献与下一步](修复实验路线选择与空鼓检测论文借鉴_20261019.md)
- [来源核验记录](experiments/repair_options_20261019/review_provenance.json)；下载的论文全文和证据页保留本地，不上传仓库。
- 状态：文献与方案完成；推荐单窗口先导、多独立站点主验证、长裂缝作为扩展。未运行实车、未改变训练。

## 2026-10-25：下一轮 E0–E3 两模块互补设计（仅规划）

- [详细计划](SegFormer_E0-E3两模块互补实验计划_20261025.md)、[本轮目录](experiments/segformer_complementarity_wave11_20261025/README.md)、[机器协议](experiments/segformer_complementarity_wave11_20261025/protocol.json)。
- [运行跟踪](experiments/segformer_complementarity_wave11_20261025/refine-logs/EXPERIMENT_TRACKER.md)、[CPU预算核验](experiments/segformer_complementarity_wave11_20261025/provenance/parameter_audit.md)、[来源快照](experiments/segformer_complementarity_wave11_20261025/provenance/referenced_conversation.json)。
- 已读取《模型模块贡献与替换方案》，冻结固定S4FFN768骨干的四格设计；M1为CBR低分辨率多尺度解码，M2为浅层门控细节残差，A0为总对照。
- CPU组件预算E3为3,116,500（相对A0减10.1986%），完整网络未实现，正式训练0/12；精度/速度/互补性均待测。未启动新训练任务或监控，当前最终交付不被本轮规划替换。

## 2026-10-25 19:00：W11正式启动

- 用户已授权开始训练；新网络、共享初始化、数据SHA、新源A0复现及四格GPU smoke通过。首个E0正式18k已实际迭代；后续状态见[跟踪表](experiments/segformer_complementarity_wave11_20261025/refine-logs/EXPERIMENT_TRACKER.md)和[真实心跳](experiments/segformer_complementarity_wave11_20261025/results/heartbeat.json)。
- [配对控制补充](experiments/segformer_complementarity_wave11_20261025/refine-logs/EXECUTION_ADDENDUM_20261025.md)：旧A0初始化/训练随机流不严格匹配，12核心之后新增3个R0，总15次；旧结果保留，不混用新旧均值。
- 一次性训练任务`Codex-SegFormer-W11-fourcell-once-20261025`；静默健康监控`segformer-w11`。训练release272文件冻结，未完成结果不得填数，禁止筛选读取test/自建集。后处理独立放行，不修改运行中的训练源码。

| 2026-10-27 00:09:58 | experiment-plan/run-experiment | SegFormer_W12替换简化与温和压缩执行计划_20261027.md | implementation | W12 12组替换/简化+条件温和S4压缩，待GPU |
| 2026-10-27 00:09:58 | experiment-plan/run-experiment | experiments/segformer_simplify_wave12_20261026/protocol.json | implementation | 预登记门槛/数据/资源/预算 |
| 2026-10-27 00:09:58 | experiment-plan/run-experiment | experiments/segformer_simplify_wave12_20261026/experiment_specs.json | implementation | 32定义96初始状态，最多18正式训练 |
| 2026-10-27 00:09:58 | experiment-plan/run-experiment | experiments/segformer_simplify_wave12_20261026/refine-logs/EXPERIMENT_PLAN.md | implementation | 分阶段执行计划 |
| 2026-10-27 00:09:58 | experiment-plan/run-experiment | experiments/segformer_simplify_wave12_20261026/refine-logs/EXPERIMENT_TRACKER.md | implementation | 状态跟踪 |
| 2026-10-27 00:09:58 | experiment-plan/run-experiment | experiments/segformer_simplify_wave12_20261026/provenance/model_preflight_w12.json | implementation | CPU完整模型/尺寸/梯度/几何检查真实exit0 |

## 2026-10-27 00:48：W12替换、简化与温和压缩正式启动

- [预登记详细计划](SegFormer_W12替换简化与温和压缩执行计划_20261027.md)，首轮12组＋至多2条件压缩＋至多4匹配多seed，总新增正式18k上限18；只用train/val。
- [96份初始化与逐tensor索引](experiments/segformer_simplify_wave12_20261026/provenance/initialization_manifest.json)、[独立执行审查](experiments/segformer_simplify_wave12_20261026/provenance/independent_workflow_review_d3348d35_v2_approved.json)、[560项冻结版本](experiments/segformer_simplify_wave12_20261026/provenance/release.json)。96初始状态不等于96训练。
- [实际进程及日志](experiments/segformer_simplify_wave12_20261026/results/heartbeat.json)、[运行交接](experiments/segformer_simplify_wave12_20261026/provenance/RUNTIME.md)、[最终报告约定](experiments/segformer_simplify_wave12_20261026/refine-logs/FINAL_DELIVERY.md)。
- R0新源全348精确复现和R0/D02短训练通过；D00数值门槛未过已拒绝。D02于00:47:37正式启动，真实日志已300/18000，最终指标尚无。
- 唯一一次性任务`Codex-SegFormer-W12-once-20261027`，静默健康监控`segformer-w12`。所有用户现有应用保持运行；仅管理本轮自有进程，不重启旧W11。

## 2026-10-27 02:18：W12定性报告辅助CPU准备

- [操作说明](experiments/segformer_simplify_wave12_20261026/postprocess_prepare_20261027/QUALITATIVE_API.md)、[独立审查](experiments/segformer_simplify_wave12_20261026/postprocess_prepare_20261027/independent_review.json)。仅CPU准备和拒绝门控验收通过；未执行渲染/新GPU，主训练冻结版本不改。
- 须等真实全部训练、profile和wrapper退出，并取得外部退出观察回执及单独helper release后使用；定性图不代替科学放行。
-
## 2026-10-27 09:11：W12效率阶段真实失败

- [阶段失败回执](experiments/segformer_simplify_wave12_20261026/provenance/phase2_profile_failure_20261027.json)、[原始工作流失败证据](experiments/segformer_simplify_wave12_20261026/results/workflow_failure.json)。首轮12项训练和验证已完成；phase2在129次保留的独立尝试后因磁盘保护阈值真实exit1，不能填充完整延迟/显存结果。
- 训练checkpoint、逐图CSV、源码/config/init/checkpoint SHA和阶段一profile证据保留。没有重启重复任务、删除失败轮次、停止用户应用或访问test；后续恢复需独立审查和用户决定。

- [效率阶段恢复说明](W12效率阶段恢复说明_20261027.md)：不重训；等待用户提供无未知GPU上下文的稳定窗口后，独立恢复phase2约30–60分钟。

## 2026-10-27：W12阶段总结报告

- [阶段总结报告](W12阶段总结报告_20261027.md)、[首轮汇总CSV](experiments/segformer_simplify_wave12_20261026/results/W12_stage_summary.csv)。报告覆盖12组seed42验证指标、参数变化、门槛判定、D00失败、phase1测速和phase2真实失败状态。
- ## 2026-10-27：W13 SegFormer-B0+Aligned八组配对消融

- [W13阶段报告](W13八组配对消融实验报告_20261027.md)、[W13工程](experiments/segformer_aligned_pair_wave13_20261027/README.md)。8组train/val已完成；当前验证集候选为B+A+RepDW，待同机稳定速度与多种子复核。
