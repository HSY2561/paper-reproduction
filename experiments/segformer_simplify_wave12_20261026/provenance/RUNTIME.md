# W12 运行交接

2026-10-27，Asia/Shanghai；工程目录于10月26日创建，后缀不改。

本轮由用户明确授权：以SegFormer为基础，替换/简化融合与Aligned，并有控制地压缩S4 Mix-FFN。首轮12组全部独立seed42，随后最多一个解码家族的960/896组合，再最多两个候选补43/44。新增正式18k上限18次。96份初始化与配置不代表96次训练。只用Crack500 train1896/val348，禁止筛选访问test或自建集。

环境：`C:/venv_bisenet/Scripts/python.exe`；PYTHONPATH必须为本工程`src/PaddleSeg-2.10.0`。Paddle3.4.0、FP32/batch4、AdamW6e-5/end_lr0、CE、EMA、18k、workers0保持不变；缓存与输出留D盘。本次有效FLAGS已写kernel见证，历史W11未显式记录FLAGS，不能声称历史旗标完全证明一致。

CPU构造/奇数尺寸/有限梯度、96初始化及独立审查已完成。核验见`model_preflight_w12.json`、`initialization_manifest.json`、`initialization_independent_review.json`。评估6个CPU验收案例真实exit0；kernel算子/反向见证真实exit0（chunk2bca69），只证明环境可运行，不等于正式模型训练。

受保护桌面进程只读身份fallback已审，profile源SHA为d489652019f76fb04e3f1791a240aa2b44b4f3d4cdddb755d2afcb9831728102；新baseline在00:38后重采，42个C+G已核、2个用户CPU控制台宿主，未停止任何应用。CIM在默认沙箱拒绝访问的初次回执为exit1，已通过获批只读全局核验成功；不把初次拒绝当训练故障。CPU服务PID可能复用，必须比较创建身份/实际命令和只读状态，不按旧PID结束应用。

唯一新任务名称`Codex-SegFormer-W12-once-20261027`，纯Python隐藏launcher→workflow，启动/结束证据分别在`results/launch_receipt.json`、`workflow_exit.json`、`launch_wrapper_exit.json`。系统任务注册或Running不是GPU训练证据。`register_run_once.ps1`只创建本轮手动一次任务，48h上限、IgnoreNew；开始前须完成release/independent review。

每阶段先核release SHA；先4488数据SHA、R0新源全348精确CM重现、R0 smoke数据RNG、D00全348数值重排对照，然后逐项smoke20→18k→普通/EMA精确mIoU冻结→官方val和全348独立区域/边界/骨架指标。任何未知真实exit/未退出的owned child阻断后续GPU阶段，保留null，不拼接中断日志。只管理自己Popen的PID+创建时间+可执行身份；用户其他进程不动。

完整首轮顺序D02,D04,D05,D08,C01,C02,D06,D07,D01,D03,D09,D10。D03仅容量对照，不可入选轻量候选；D00不训练/不减参/不计模块。NativeSum、flow简化与S4通道取法见计划及初始化逐tensorledger。

首轮全齐后，独立新进程显存与同进程400/480交错测速。轮次N*ceil(5/N)，每模型/尺寸100预热、每轮100同步，全部轮次保留。背景不稳整次defer3并新目录重试；不得删坏轮或杀桌面。参数更少不等于速度更快。精度保护、条件组合、top2三seed门槛见protocol，不得放宽。

队列完成只表示定量阶段。仍须独立审查、固定代表图、完整报告/复现索引/无数据大权重ZIP及给用户可点击链接；按`refine-logs/FINAL_DELIVERY.md`做。失败与负结果都交付，不声称SOTA。

当前最终实际状态以results和真实日志为准；本段是启动前静态交接，不预填训练成功。

## 00:48 实际启动回执

封版560项文件，releaseSHA `157d0af16466f15e0b274174260fb5380e0cafca5690cd5ceeb50cbfc6d17cbe`，真实CPU封版exit0（chunk2d12a1），独立controller审查为`independent_workflow_review_d3348d35_v2_approved.json`。唯一任务00:45:29注册启动，纯Pythonlauncher/科学controller均实际创建，非仅Task Running。

R0官方val于00:45:53真实exit0、全348独立CM/边界于00:46:15真实exit0；精确CM和mIoU与W11 R0_s42一致。R0 smoke20于00:46:39真实exit0，数据RNG/首8增强与80实际采样顺序一致。

D00部署重排实际完整348执行exit0，但科学门槛failed：159/81594144像素不同，预测一致率0.9999980513低于0.999999，IoU差−0.0000025390；未通过预登记极严数值门槛，已拒绝D00，不称逐位等价、不纳入后续速度候选。12个独立训练不受其影响，不放宽该门槛。

D02（无Aligned/width128/S4=1024）smoke20于00:47:35真实exit0，随后00:47:37正式18k进程启动；00:48:07日志已到300/18000，FP32/batch4，新源码及GPU实际运行。正式退出码仍未知/运行中，不能写exit0或已有最终指标。GPU日志约0.061秒/iter，阶段间验证会另耗时。

已建立当前聊天静默健康监控`segformer-w12`（30分钟）；正常无实质变化不通知。只在重要结果、故障/必要操作或最终交付时通知。GitHub监控另行保留，旧W11任务不复启。

## 02:18 健康检查与交付辅助准备

D02、D04、D05均已有真实18k/exit0/终点/冻结和独立348验证；本seed三项均未过完整精度保护，保留负结果。D05的BF1@2下降0.6569个百分点、半径≤1骨架Recall下降1.8119个百分点；未测稳定延迟，不声称加速。D08于02:13:52继续正式训练，02:17:52心跳2800/18000，日志持续更新，盘余约12GiB；未重复启动。

固定24图辅助已在独立`postprocess_prepare_20261027`目录完成CPU准备和独立审查，不需重写。先读该目录`QUALITATIVE_API.md`。helper SHA `8071d36bd3cf7d78b4ff951f66489f96cb26da9ba678fd4dcf6dc3684464ed38`，API SHA `78f2499683604dfa070ee99614b1cad75988135602d12e89a5e6a4e08394f4e5`，独立审查`independent_review.json` SHA `ae5f8f1ac2ec5cd73d75a8ae3e5acb40c2b1f4d41dc630fd887600f42b5bcb06`。CPU自检及审查真实exit0，11组/19类错误拒绝通过，未GPU/未读原图/未渲染，也尚无helper release。

未来全部训练和profile实际完成、controller和launcher/Popen宿主身份均已退出后，须按API提供外部`completion_observer.json`（本轮确切wrapper PID+birth+exe、独立实际exit0，来源可为唯一一次性任务LastTaskResult=0且任务已结束的真实查询或预持进程句柄回执），保持原launch_wrapper_exit记录不改写。随后独立冻结helper release绑定全部最终动态证据，才可--render。不能把finally中的child exit0推断成wrapper外部exit0，缺证据就拒绝。

可视化只展示R0与queue_done.accepted_groups中的seed42模型；若无合格者，按预定val IoU降序/id升序展示一个明确NEGATIVE_FAILED示例，不改变候选或触发训练。固定24沿用W11原列表。helper核348 CSV预测/GT/骨架SHA与计数、4488前后SHA，所有本地进程明细留local_only不入公开ZIP。训练560个冻结文件及release SHA保持不变。
## 09:11 实际阶段失败回执

首轮12组seed42及每组训练、选模、官方与独立验证已完成；未进入条件组合或额外seed，因为phase2统一效率筛选没有完成。phase2从attempt001至attempt129均保留：前128次由受保护桌面GPU上下文触发安全defer3，第129次memory_R0_400触发预设`disk_below4GiB`保护，工作流真实`workflow_exit.exit_code=1`。失败信息见`results/workflow_failure.json`和`provenance/phase2_profile_failure_20261027.json`。

失败发生时回执记录的瞬时余量约6.43GiB；当前检查余量约10.45GiB，这不改写第129次的真实保护判定。launcher记录科学child真实exit1，没有改写成exit0。训练checkpoint、逐图CSV、阶段一测速和phase2部分证据均保留；未删除轮次、未停止桌面应用、未重启重复任务、未访问test。

后续若恢复效率阶段，须先处理桌面GPU上下文和临时空间，再走独立审查的恢复流程；当前不能填充完整延迟/显存结果，也不能据此声称候选放行或SOTA。
