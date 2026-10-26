# Wave11 执行源码快照：只供复核，不自动运行

这些脚本是已结束实验的源码副本，不是将Wave11接入旧A0运行器的实现。本次未导入或执行Paddle；保留原始`ROOT/src/PaddleSeg-2.10.0`布局假设，机器绝对路径仅替换为占位符。禁止把它们不经配置直接作为当前A0入口执行。

复现需在独立环境中先准备固定PaddleSeg 2.10.0，组装原布局，将当前`overlays/paddleseg/models/segformer.py`与`overlays/wave11_snapshot/paddleseg/`内的新增模型/骨干及两个注册入口放入相应目录，再将本目录脚本、`configs/wave11_snapshot/`配置和独立初始化权重按原布局准备。W11所依赖的SegFormer源码SHA与当前发布A0 overlay相同；其余冻结来源、初始化审计和完整原始文件SHA见`evidence/wave11_20261026/provenance/`。本次仅静态发布，不保证自动组装或一键复现已验证。

仅M1、仅M2、两者组合及新匹配R0遵守同一协议，不能把旧A0或F08结果替代成新四格的一行。数据、初始/选模权重不托管；执行前须提供本地资源并核对记录SHA。请先阅读`docs/wave11_snapshot/EXPERIMENT_PLAN.md`和`EXECUTION_ADDENDUM_20261025.md`。

`scripts/unified_evaluation_snapshot/`也只是已完成统一评价的源码快照，仍依赖原来的独立布局与冻结资源，不由本次同步运行。
