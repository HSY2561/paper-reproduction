# AW+bgCE 主线清理清单

更新时间：2026-11-13（Asia/Shanghai）

本次仓库视图固定为 **SegFormer-B0 + Aligned + 标准 Haar WTConv + background-weighted CE（AW+bgCE）**。

## 保留内容

- Crack500 当前 AW+bgCE 训练、精确验证、test、六个公开模型对照及必要源代码；
- Crack500 当前标准 Haar WTConv-only 对照；
- VOCdevkit3000 当前标准 Haar WTConv+bgCE 结果；
- VOCdevkit3000 旧四组历史消融（明确标注为旧 WTConv candidate + 普通 CE）；
- DeepCrack 当前四组 Haar+bgCE 消融；
- 已冻结的六个公开模型参考结果；
- 主线报告、消融表、评估脚本、配置、数据清单哈希、checkpoint SHA 和 exit 回执。

## 从当前 GitHub 视图移除的内容

- W11–W18 的结构搜索、损失搜索、轻量融合搜索及其重复中间产物；
- 旧的 RepDW、Ghost、PDC、MSCA 等候选模块实验；
- 与当前裂缝分割主线无关的 Bisenet、相机工具、机器人服务、临时分析和旧报告压缩包；
- 未进入主线报告的重复日志、缓存、训练中间文件和本机环境文件。

## 数据与权重边界

原始数据集、派生数据 staging、训练权重、优化器状态和本机运行环境不提交 GitHub。它们仍由本机实验目录或外部数据路径管理；仓库保留复现实验所需的配置、清单、哈希、评估代码和结果摘要。旧实验的完整历史仍可从 Git 提交历史审计，不在当前工作树中作为主线材料展示。
