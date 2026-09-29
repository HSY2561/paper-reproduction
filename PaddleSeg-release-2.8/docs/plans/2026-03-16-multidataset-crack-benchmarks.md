# Multi-Dataset Crack Benchmarks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prepare relative-path benchmark code for `Crackseg9k` and `Crack500_kaggle`, keeping `SegFormer-B0` on Paddle and adding PyTorch wrappers for `CrackFormer-II`, `DeepCrack`, `Efficientnet`, and `Mobilenetv3`, with early stopping on validation `mIoU`.

**Architecture:** Keep the Paddle branch inside `scripts/offline_experiments`, and add a new generic PyTorch package that reads the same split files as Paddle. Reuse the existing wrapper style from `lmm_original_crack500`, but make it dataset-agnostic and model-agnostic.

**Tech Stack:** Python, PaddleSeg, PyTorch cu121, `unittest`, YAML configs

---

### Task 1: Add failing tests for generic benchmark behavior

**Files:**
- Create: `tests/torch_crack_benchmark/test_config.py`
- Create: `tests/torch_crack_benchmark/test_model.py`
- Create: `tests/torch_crack_benchmark/test_early_stopping.py`
- Create: `tests/offline_experiments/test_public_benchmark_configs.py`

**Step 1: Write tests**

- Verify PyTorch config loader preserves relative dataset paths.
- Verify model resolver accepts `crackformer_ii`, `deepcrack`, `efficientnet`, `mobilenetv3`.
- Verify early-stop controller stops after `patience` evaluations without sufficient improvement.
- Verify dataset-specific SegFormer config generation targets `dataset/Crackseg9k` and `dataset/Crack500_kaggle`.

**Step 2: Run tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.torch_crack_benchmark.test_config tests.torch_crack_benchmark.test_model tests.torch_crack_benchmark.test_early_stopping tests.offline_experiments.test_public_benchmark_configs
```

Expected:
- FAIL because the package and config generation logic do not exist yet.

### Task 2: Implement the generic PyTorch benchmark package

**Files:**
- Create: `torch_crack_benchmark/__init__.py`
- Create: `torch_crack_benchmark/common.py`
- Create: `torch_crack_benchmark/config.py`
- Create: `torch_crack_benchmark/dataset.py`
- Create: `torch_crack_benchmark/model.py`
- Create: `torch_crack_benchmark/metrics.py`
- Create: `torch_crack_benchmark/evaluate.py`
- Create: `torch_crack_benchmark/model_stats.py`
- Create: `torch_crack_benchmark/benchmark_fps.py`
- Create: `torch_crack_benchmark/summarize.py`
- Create: `torch_crack_benchmark/train.py`
- Create: `torch_crack_benchmark/run.py`

**Step 1: Implement config and dataset loading**

- Read relative dataset paths from YAML.
- Read `train.txt / val.txt / test.txt` split files directly.

**Step 2: Implement model resolver**

- Load original-source modules from:
  - `Lightweight-Modular-Model-main/Codes`
  - extracted `CrackFormer-II` source tree

**Step 3: Implement early stopping**

- Track best validation `mIoU`.
- Stop after `patience` stale evaluations.
- Log stop reason.

**Step 4: Implement train/eval/stats/fps/summary flow**

- Keep outputs aligned with the existing offline experiment JSON files.

### Task 3: Add dataset-specific config generation and Paddle runner support

**Files:**
- Modify: `scripts/offline_experiments/generate_ablation_configs.py`
- Create: `scripts/offline_experiments/run_public_benchmarks.py`
- Create: `configs/torch_crack_benchmark/crackseg9k/*.yml`
- Create: `configs/torch_crack_benchmark/crack500_kaggle/*.yml`

**Step 1: Add SegFormer dataset-specific config generation**

- Generate relative-path configs for:
  - `dataset/Crackseg9k`
  - `dataset/Crack500_kaggle`

**Step 2: Add PyTorch config templates**

- Create benchmark configs for four PyTorch models on both datasets.

**Step 3: Add runner script**

- Provide one entrypoint for public benchmark runs and summary export.

### Task 4: Install PyTorch cu121 into `.venv` and verify environment

**Files:**
- Environment only

**Step 1: Install packages**

- Install `torch`, `torchvision`, `torchaudio` with CUDA 12.1 wheels into `.venv`.

**Step 2: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Expected:
- CUDA-enabled build
- `torch.cuda.is_available()` returns `True`

### Task 5: Run targeted tests and smoke checks

**Files:**
- Verify: new tests under `tests/torch_crack_benchmark`
- Verify: `scripts/offline_experiments/run_public_benchmarks.py`
- Verify: `scripts/torch_crack_benchmark/run_benchmark.py` or package runner

**Step 1: Run unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.torch_crack_benchmark.test_config tests.torch_crack_benchmark.test_model tests.torch_crack_benchmark.test_early_stopping tests.offline_experiments.test_public_benchmark_configs
```

Expected:
- PASS

**Step 2: Smoke-check CLI help**

Run:

```powershell
.\.venv\Scripts\python.exe scripts/offline_experiments/run_public_benchmarks.py --help
```

Expected:
- CLI renders dataset/model options with relative-path defaults.
