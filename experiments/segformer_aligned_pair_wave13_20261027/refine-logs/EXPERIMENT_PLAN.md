# W13 执行计划

引用：`D:\论文复现\SegFormer_Aligned四模块八组配对实验计划_20261027.md`。

工程独立于旧 W12；源码、配置、训练输出和报告均放在本目录。首轮顺序：P01_RepDW、P01A_RepDW_Aligned、P02_Ghost、P02A_Ghost_Aligned、P03_PDC、P03A_PDC_Aligned、P04_MSCA、P04A_MSCA_Aligned。每组先CPU构造/反向smoke，再20iter GPU smoke，随后18000iter；不得复用旧W12的训练checkpoint作为新组权重。

当前状态：接口代码与配置准备中，正式GPU训练尚未开始。真实状态只以本目录results、logs和exit回执为准。
