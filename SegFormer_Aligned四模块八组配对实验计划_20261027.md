# SegFormer-B0+Aligned 四模块八组配对实验计划

本轮按第五章的完整归因设计，以原版 SegFormer-B0（B）和 SegFormer-B0+Aligned（B+A）作为共同对照。不使用 W12 的 R0 命名，也不把 width128 当作本轮基线；B 与 B+A 都使用完整256维解码投影。RepDW、Ghost、PDC 三个候选各做单独和联合两组，共八组配置；MSCA保留为前三类无可信信号时的第二批候选。

## 首轮八组

|配置|结构|主要验证问题|
|---|---|---|
|B|原版 SegFormer-B0，256维，无Aligned|共同精度与资源基线|
|B+A|原版 SegFormer-B0，256维+Aligned|验证Aligned在新基线上的独立贡献|
|B+RepDW|RepDW融合替换，无Aligned|局部融合能否独立改善细裂缝并减计算|
|B+A+RepDW|Aligned后使用RepDW融合替换|验证对齐与局部结构恢复是否互补|
|B+Ghost|Ghost融合替换，无Aligned|验证低成本特征生成的独立效果|
|B+A+Ghost|Aligned后使用Ghost融合替换|验证低成本融合与对齐的互补性|
|B+PDC|PDC融合替换，无Aligned|验证像素差分边界收益及纹理误检风险|
|B+A+PDC|Aligned后使用PDC融合替换|验证差分边界与对齐的互补性|

RepDW、Ghost、PDC组都替换原来的4D→D最终融合层，而不是在原融合层后面追加模块；联合组保留Aligned的C1/C4对齐，再使用同一个替换融合层。八组不叠加第三模块。

## 固定协议

- Crack500 train/val，1896/348张；禁止筛选访问test和自建集；4488源文件哈希前后核对。
- FP32、RTX5060、seed42、batch4、AdamW、6e-5、PolynomialDecay end_lr=0、CE、EMA、18000iter、workers0。
- 400裁剪、0.75–1.25缩放、原翻转/颜色扰动、同一ImageNet权重；普通/EMA按精确val mIoU选模。
- 记录Precision、Recall、F1、裂缝IoU、mIoU、2px边界F1、hard clDice、细线Recall、参数量、训练exit、checkpoint/config SHA和逐图CSV。

## 放行与解释

相对B+A要求裂缝IoU至少+0.002、F1至少+0.001、mIoU不降，Precision/Recall/边界F1下降不超过0.005；相对B同时报告联合模型的净参数和延迟变化。单独组和联合组都必须单独报告，联合优于两个单模块才支持“互补”解释。任何未通过者保留负结果，不按验证结果事后修改门槛。

## 时间估计

八组纯训练按已有约30–35分钟/组估计约4–6小时；加验证、参数核验、逐图边界和同机速度约6–10小时。若GPU背景占用或兼容性失败，按真实日志延长；不会重训旧W12或关闭用户桌面程序。首轮完成后才决定是否对最多两个候选做seed43/44复核；只有精度和效率同时满足才进入统一test和公开模型对照。最终不预先声称SOTA。
