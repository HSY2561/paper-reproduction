# LMM Torch Crack500 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone PyTorch LMM experiment pipeline for Crack500 that mirrors the current Paddle Crack500 experiment protocol and output files.

**Architecture:** The implementation lives under `experiments/lmm_torch/` and does not depend on `paddleseg/` at runtime. It reuses `dataset/Crack500/*.txt`, writes results into `output/offline_experiments_torch/crack500/lmnet/`, and exports the same metrics/summary JSON structure used by the Paddle offline experiment scripts.

**Tech Stack:** Python, PyTorch, PIL, NumPy, JSON/YAML, unittest.

---

### Task 1: Add failing tests for config and summary behavior

**Files:**
- Create: `tests/lmm_torch/test_config.py`
- Create: `tests/lmm_torch/test_summary.py`

**Step 1: Write the failing test**

Add a test that imports planned config helpers and asserts:
- Crack500 defaults use `batch_size=8`
- Training crop is `720`
- Output directory points to `output/offline_experiments_torch/crack500/lmnet`

Add a test that builds fake `metrics.json`, `model_stats.json`, and `fps.json` and expects a `summary_row.json` with fields:
- `Experiment`
- `Dataset`
- `Config`
- `BestModelPath`
- `Precision`
- `Recall`
- `F1`
- `IoU`
- `mIoU`
- `Params_M`
- `FPS`

**Step 2: Run test to verify it fails**

Run:
```bash
.\.venv\Scripts\python.exe -m unittest tests.lmm_torch.test_config tests.lmm_torch.test_summary
```

Expected: import failures because the new package does not exist yet.

**Step 3: Commit**

```bash
git add tests/lmm_torch/test_config.py tests/lmm_torch/test_summary.py
git commit -m "test: add lmm torch config and summary tests"
```

### Task 2: Create the standalone LMM Torch package

**Files:**
- Create: `experiments/lmm_torch/__init__.py`
- Create: `experiments/lmm_torch/config.py`
- Create: `experiments/lmm_torch/common.py`
- Create: `experiments/lmm_torch/model.py`

**Step 1: Implement minimal package**

- `config.py`: define default Crack500-aligned config values and load/merge helpers
- `common.py`: JSON/YAML IO, directory creation, checkpoint discovery
- `model.py`: vendor the original PyTorch LMM model with local imports only

**Step 2: Run tests**

Run:
```bash
.\.venv\Scripts\python.exe -m unittest tests.lmm_torch.test_config tests.lmm_torch.test_summary
```

Expected: config test still fails on missing summary logic; config test passes once defaults exist.

**Step 3: Commit**

```bash
git add experiments/lmm_torch
git commit -m "feat: add standalone lmm torch package skeleton"
```

### Task 3: Add Crack500 dataset adapter and transforms

**Files:**
- Create: `experiments/lmm_torch/dataset.py`
- Create: `tests/lmm_torch/test_dataset.py`

**Step 1: Write the failing test**

Test parsing of `dataset/Crack500/train.txt`-style lines into image/mask paths and verify returned tensors/shapes for a synthetic sample.

**Step 2: Run test to verify it fails**

Run:
```bash
.\.venv\Scripts\python.exe -m unittest tests.lmm_torch.test_dataset
```

Expected: import failure or missing dataset class failure.

**Step 3: Implement dataset**

- Parse PaddleSeg list files
- Provide train/val/test datasets
- Apply Crack500-aligned transforms, including `720` crop for training

**Step 4: Run test to verify it passes**

Run:
```bash
.\.venv\Scripts\python.exe -m unittest tests.lmm_torch.test_dataset
```

**Step 5: Commit**

```bash
git add experiments/lmm_torch/dataset.py tests/lmm_torch/test_dataset.py
git commit -m "feat: add lmm torch crack500 dataset adapter"
```

### Task 4: Add training, evaluation, stats, and FPS scripts

**Files:**
- Create: `experiments/lmm_torch/train.py`
- Create: `experiments/lmm_torch/evaluate.py`
- Create: `experiments/lmm_torch/model_stats.py`
- Create: `experiments/lmm_torch/benchmark_fps.py`
- Create: `experiments/lmm_torch/summarize.py`

**Step 1: Implement training/eval flow**

- `train.py`: train LMM on Crack500 and save checkpoints plus `best_model.pt`
- `evaluate.py`: compute Crack500 metrics matching Paddle JSON field names
- `model_stats.py`: count params in millions
- `benchmark_fps.py`: run timed forward passes
- `summarize.py`: emit `summary_row.json`

**Step 2: Smoke-test script imports**

Run:
```bash
E:\anaconda3\envs\cuda11-py310\python.exe -m py_compile experiments/lmm_torch/train.py experiments/lmm_torch/evaluate.py experiments/lmm_torch/model_stats.py experiments/lmm_torch/benchmark_fps.py experiments/lmm_torch/summarize.py
```

**Step 3: Commit**

```bash
git add experiments/lmm_torch/train.py experiments/lmm_torch/evaluate.py experiments/lmm_torch/model_stats.py experiments/lmm_torch/benchmark_fps.py experiments/lmm_torch/summarize.py
git commit -m "feat: add lmm torch training and evaluation scripts"
```

### Task 5: Add a single entry-point runner and usage doc

**Files:**
- Create: `experiments/lmm_torch/run_crack500_lmm.py`
- Create: `experiments/lmm_torch/README.md`

**Step 1: Implement runner**

- Reuse config defaults
- Accept `--iters`, `--batch_size`, `--device`, `--skip_*`, `--resume_failed`
- Produce output under `output/offline_experiments_torch/crack500/lmnet/`

**Step 2: Smoke-test no-op stages**

Run:
```bash
E:\anaconda3\envs\cuda11-py310\python.exe experiments/lmm_torch/run_crack500_lmm.py --skip_train --skip_eval --skip_fps
```

Expected: graceful message or checkpoint-not-found guard, not a traceback from imports or path handling.

**Step 3: Document usage**

Document:
- required interpreter
- dataset assumptions
- example train/eval command
- output files

**Step 4: Commit**

```bash
git add experiments/lmm_torch/run_crack500_lmm.py experiments/lmm_torch/README.md
git commit -m "feat: add lmm torch crack500 runner"
```

### Task 6: Final verification

**Files:**
- Verify: `docs/plans/2026-03-12-lmm-torch-crack500-design.md`
- Verify: `docs/plans/2026-03-12-lmm-torch-crack500.md`

**Step 1: Run unit tests**

```bash
.\.venv\Scripts\python.exe -m unittest tests.lmm_torch.test_config tests.lmm_torch.test_summary tests.lmm_torch.test_dataset
```

**Step 2: Run smoke stats check**

```bash
E:\anaconda3\envs\cuda11-py310\python.exe experiments/lmm_torch/model_stats.py --save_path output/offline_experiments_torch/crack500/lmnet/model_stats.json
```

**Step 3: Record environment caveat**

Document whether the selected PyTorch interpreter is CPU-only or GPU-enabled.

**Step 4: Commit**

```bash
git add docs/plans/2026-03-12-lmm-torch-crack500-design.md docs/plans/2026-03-12-lmm-torch-crack500.md
git commit -m "docs: add lmm torch crack500 design and implementation plan"
```
