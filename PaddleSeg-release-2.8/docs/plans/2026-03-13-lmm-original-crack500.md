# LMM Original-Source Crack500 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a minimal wrapper around the downloaded original `LM_Net` source so it can run Crack500 with the same training protocol used by the current HrSegNet experiments.

**Architecture:** Keep `Lightweight-Modular-Model-main/Codes/LM_Net.py` unchanged and add a separate experiment layer under `Lightweight-Modular-Model-main/experiments/crack500/`. The wrapper owns config parsing, dataset adaptation from Paddle-style txt lists, training/evaluation orchestration, and standardized result export.

**Tech Stack:** Python, PyTorch, PIL, NumPy, existing local Crack500 split files.

---

### Task 1: Add failing tests for the wrapper contract

**Files:**
- Create: `tests/lmm_original/__init__.py`
- Create: `tests/lmm_original/test_config.py`
- Create: `tests/lmm_original/test_summary.py`

**Step 1: Write the failing tests**

Cover:

- default config matches Crack500 HrSegNet protocol
- summary writer exports the expected columns

**Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests.lmm_original.test_config tests.lmm_original.test_summary`

Expected: FAIL because the new wrapper modules do not exist yet.

**Step 3: Write minimal implementation**

Create config and summary modules under `Lightweight-Modular-Model-main/experiments/crack500/`.

**Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m unittest tests.lmm_original.test_config tests.lmm_original.test_summary`

Expected: PASS.

### Task 2: Implement dataset and metric helpers

**Files:**
- Create: `Lightweight-Modular-Model-main/experiments/crack500/common.py`
- Create: `Lightweight-Modular-Model-main/experiments/crack500/config.py`
- Create: `Lightweight-Modular-Model-main/experiments/crack500/dataset.py`
- Create: `Lightweight-Modular-Model-main/experiments/crack500/metrics.py`
- Create: `Lightweight-Modular-Model-main/experiments/crack500/summarize.py`

**Step 1: Write minimal helper implementations**

Implement:

- YAML/JSON load and dump
- dataset parsing from `dataset/Crack500/*.txt`
- train transforms aligned with HrSegNet protocol
- binary segmentation metric aggregation
- summary row export

**Step 2: Run tests**

Run: `.\.venv\Scripts\python.exe -m unittest tests.lmm_original.test_config tests.lmm_original.test_summary`

Expected: PASS.

### Task 3: Implement original-source LM_Net training and evaluation wrappers

**Files:**
- Create: `Lightweight-Modular-Model-main/experiments/crack500/train.py`
- Create: `Lightweight-Modular-Model-main/experiments/crack500/evaluate.py`
- Create: `Lightweight-Modular-Model-main/experiments/crack500/model_stats.py`
- Create: `Lightweight-Modular-Model-main/experiments/crack500/benchmark_fps.py`
- Create: `Lightweight-Modular-Model-main/experiments/crack500/run.py`
- Create: `Lightweight-Modular-Model-main/experiments/crack500/configs/lmnet_crack500.yml`
- Create: `Lightweight-Modular-Model-main/experiments/crack500/README.md`

**Step 1: Use original model source**

Import `LM_Net` from `Lightweight-Modular-Model-main/Codes/LM_Net.py`.

**Step 2: Implement training protocol**

Use:

- `SGD`
- polynomial lr decay + warmup
- configurable micro-batch and gradient accumulation
- `BCEWithLogitsLoss` on binary mask logits

**Step 3: Implement evaluation and export**

Write result files with the same schema as the current Crack500 experiment outputs.

**Step 4: Run smoke verification**

Run one small smoke pass on GPU:

`E:\anaconda3\python.exe Lightweight-Modular-Model-main/experiments/crack500/train.py --iters 1 --batch_size 1 --accumulation_steps 8 --device cuda --skip_val`

Expected: completes and saves a checkpoint.

### Task 4: Verify and document usage

**Files:**
- Modify: `Lightweight-Modular-Model-main/experiments/crack500/README.md`

**Step 1: Run validation**

Run:

- `.\.venv\Scripts\python.exe -m unittest tests.lmm_original.test_config tests.lmm_original.test_summary`
- `E:\anaconda3\python.exe -m py_compile Lightweight-Modular-Model-main/experiments/crack500/common.py Lightweight-Modular-Model-main/experiments/crack500/config.py Lightweight-Modular-Model-main/experiments/crack500/dataset.py Lightweight-Modular-Model-main/experiments/crack500/metrics.py Lightweight-Modular-Model-main/experiments/crack500/train.py Lightweight-Modular-Model-main/experiments/crack500/evaluate.py Lightweight-Modular-Model-main/experiments/crack500/model_stats.py Lightweight-Modular-Model-main/experiments/crack500/benchmark_fps.py Lightweight-Modular-Model-main/experiments/crack500/run.py`

**Step 2: Add final usage commands**

Document the exact training command for the user.
