# PSPNet / DeepLabV3P / Fast-SCNN 最新状态

三组均完成GPU 18,000 iter、独立val、验证冻结及test，训练/验证/test九项真实退出码均0。训练完成时间2026-10-25 02:28:21；三组test完成时间09:56:19。

- 真实回执、测试JSON与逐图CSV：results/
- 训练和独立评价日志：logs/
- 选模SHA与冻结时间：frozen/<model>/selection.json
- 本目录环境：Ubuntu24.04、SM120 Paddle3.3.0.dev20251209、RTX5060，随机初始化/OHEM/无EMA；与旧Windows配方不同。
- public_test.py的BF1为legacy近似口径，循环耗时不是GPU纯推理延迟。
- 最新完整报告：D:/论文复现/全部训练结果汇总报告_20261025.md

三组完成不代表后续Wave10多seed、效率和统一后端终评全部完成。
