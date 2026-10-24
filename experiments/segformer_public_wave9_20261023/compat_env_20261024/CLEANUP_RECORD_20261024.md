# 2026-10-24 D盘清理记录

已删除的临时/可重建文件：

- D:\论文复现\_wheels：Paddle wheel 下载包；
- D:\论文复现\_wheel_extract：wheel 解压临时目录；
- D:\论文复现\tmp_pip331：pip 临时目录；
- D:\论文复现\envs\paddle331win：仅用于兼容性 smoke 的独立环境；
- D:\论文复现\experiments\segformer_public_wave9_20261023\compat_env_20261024\venv：失败的空 venv 临时目录。

保留内容：

- 数据集、源代码、正式 checkpoint、配置、训练/评估日志；
- compat_env_20261024\results 中所有真实 exit 和失败证据；
- RUNTIME.md、最新对照 JSON/CSV 和中文报告。

删除前未发现这些路径被正式实验配置或报告依赖。wheel 和兼容环境可由官方 PaddleX wheel URL 重新构建。
