# 本地 checkpoint 与 Python 环境接入说明

GitHub 仓库保存源码、配置、退出回执、指标和 checkpoint SHA256；训练得到的 `.pdparams` 二进制和完整虚拟环境目录保留在本地或由外部制品存储提供。这样可以避免把大文件、机器缓存、CUDA 驱动和用户路径写入 Git 历史，同时让软件侧能够稳定复现加载流程。

## 1. 环境

仓库记录了 2026-10-25 观察到的 Windows AMD64 环境：Python 3.12.10、PaddlePaddle GPU 3.4.0、NumPy 2.4.1、Pillow 12.1.0，以及 `requirements/selected-windows.txt` 中的其余包。该文件是已观测包集合，不是跨机器完整锁定；Paddle GPU wheel、CUDA/cuDNN、显卡驱动和硬件必须按目标机器单独匹配。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements/selected-windows.txt
```

安装后可运行：

```powershell
.\.venv\Scripts\python.exe -B scripts/capture_environment.py
```

该命令只记录包元数据，不下载数据、不启动训练。首次运行官方 PaddleSeg 代码时，先按 `runtime/README.md` 用已核验的 PaddleSeg 归档生成 `runtime/PaddleSeg-2.10.0`。

## 2. checkpoint 放置与校验

把外部提供的 checkpoint 放在本机目录，例如：

```text
D:\checkpoints\W12_C01_s42\model.pdparams
```

软件左侧的模型选择器应直接选择这个 `.pdparams` 文件；配置文件中的 `${CHECKPOINT_ROOT}` 只是可移植占位符，不能当作真实路径。使用前先校验 SHA256：

```powershell
Get-FileHash D:\checkpoints\W12_C01_s42\model.pdparams -Algorithm SHA256
```

Wave12 的 12 个 seed42 checkpoint、选模迭代、配置 SHA 和预期权重 SHA 见 [`runtime/checkpoint_manifest_wave12.json`](../runtime/checkpoint_manifest_wave12.json)。历史 Wave11、统一评估和公开模型的 SHA 见 [`evidence/checkpoints.json`](../evidence/checkpoints.json)。manifest 中 `bundled:false` 是有意的：GitHub 不包含权重本体。

## 3. 软件接入约定

软件应让用户选择三项：

1. Python 解释器（含 PaddlePaddle、Pillow、NumPy）；
2. 模型配置 YAML；
3. 与配置和 manifest SHA 匹配的 `.pdparams`。

加载前检查文件存在、SHA256、配置中的模型注册名和类别数（Crack500 为 2 类），再启动推理。不要把验证集 checkpoint 当成统一 test 最终模型：Wave12 记录明确 `test_used=false`，且 profile 阶段失败、没有最终选模。若需要发布二进制，请通过外部制品存储或 GitHub Release/LFS 按组织的存储配额单独分发，并在 `checkpoint_manifest_wave12.json` 中更新 SHA；不要把 `.venv`、CUDA 安装目录、数据集或训练缓存打包进仓库。
