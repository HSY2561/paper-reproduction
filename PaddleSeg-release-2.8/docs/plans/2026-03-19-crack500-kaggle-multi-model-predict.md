# Crack500 Kaggle Multi-Model Predict Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a unified prediction script that scans usable Crack500-Kaggle experiment outputs and writes binary crack masks for user-provided images.

**Architecture:** The script will discover experiments from result directories plus metadata files, then route inference to Paddle or Torch loaders based on the resolved model weight suffix. Binary mask writing and manifest generation will be shared across both branches.

**Tech Stack:** Python, PaddleSeg, PyTorch, PIL, unittest

---

### Task 1: Discovery Tests

**Files:**
- Create: `tests/offline_experiments/test_predict_crack500_kaggle_models.py`

**Step 1: Write the failing test**

Cover:
- discovery resolves config/model paths from `metrics.json` when local `runtime_config.yml` is missing
- discovery classifies `.pdparams` as paddle and `.pt` as torch
- binary mask saving writes only `0` and `255`

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/offline_experiments/test_predict_crack500_kaggle_models.py -q`
Expected: FAIL because the module does not exist yet.

**Step 3: Write minimal implementation**

Create the prediction module with the tested helper functions.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/offline_experiments/test_predict_crack500_kaggle_models.py -q`
Expected: PASS

### Task 2: Unified Prediction Script

**Files:**
- Create: `scripts/offline_experiments/predict_crack500_kaggle_models.py`

**Step 1: Implement Paddle branch**

Use `Config`, `SegBuilder`, `Compose(builder.val_transforms)`, and `infer.inference` to predict one image and save a binary mask.

**Step 2: Implement Torch branch**

Use `torch_crack_benchmark.config.load_config`, `torch_crack_benchmark.model.build_model`, `extract_logits`, and the benchmark normalization recipe to predict one image and save a binary mask.

**Step 3: Implement CLI and manifest**

Support image files and directories, scan the default model root, and write outputs to `output/offline_experiments_torch/crack500_kaggle_predictions`.

**Step 4: Run focused tests**

Run: `python -m pytest tests/offline_experiments/test_predict_crack500_kaggle_models.py -q`
Expected: PASS

### Task 3: Runtime Verification

**Files:**
- Create: `output/offline_experiments_torch/crack500_kaggle_predictions/...`

**Step 1: Run on the requested validation image**

Run:

```bash
python scripts/offline_experiments/predict_crack500_kaggle_models.py --images dataset/custom/images/train/1.jpg
```

Expected: Binary PNG masks are written for each discovered model.

**Step 2: Inspect produced files**

Confirm output masks exist under model-specific directories and manifest JSON lists them.
