# Crack500 Kaggle Hybrid Benchmarks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Crack500-Kaggle benchmark support for PaddleSeg `pspnet`/`unet` and original-source `shufflenetv2`/`deepcrack`, with all outputs written under `output/offline_experiments_torch/crack500_kaggle`.

**Architecture:** Public benchmark config generation will describe both PaddleSeg and original-source experiment families. PaddleSeg experiments will reuse the existing offline experiment utilities; original-source experiments will reuse `lmm_original_crack500.run` after extending its model registry to include `deepcrack`.

**Tech Stack:** Python, PaddleSeg, PyTorch, YAML configs, unittest

---

### Task 1: Cover New Public Benchmark Specs

**Files:**
- Modify: `tests/offline_experiments/test_public_benchmark_configs.py`

**Step 1: Write the failing test**

Add assertions that `build_public_benchmark_specs()` exposes Crack500-Kaggle entries for Paddle `pspnet` and `unet`, plus original-source `shufflenetv2` and `deepcrack`, all with output roots under `output/offline_experiments_torch/crack500_kaggle`.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/offline_experiments/test_public_benchmark_configs.py -q`
Expected: FAIL because the new model entries do not exist yet.

**Step 3: Write minimal implementation**

Update `scripts/offline_experiments/run_public_benchmarks.py` to describe the added model families and generate their config paths.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/offline_experiments/test_public_benchmark_configs.py -q`
Expected: PASS

### Task 2: Cover Original-Source DeepCrack Registration

**Files:**
- Modify: `tests/lmm_original/test_model.py`
- Modify: `tests/lmm_original/test_config.py`

**Step 1: Write the failing test**

Add assertions that `lmm_original_crack500.model.resolve_model_spec("deepcrack")` resolves to `DeepCrack.define_deepcrack` and that config loading can override `model_type`, dataset paths, experiment name, and output root for Crack500-Kaggle.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/lmm_original/test_model.py tests/lmm_original/test_config.py -q`
Expected: FAIL because `deepcrack` is not supported yet.

**Step 3: Write minimal implementation**

Update `lmm_original_crack500/model.py` and any supporting config defaults needed for Crack500-Kaggle overrides.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/lmm_original/test_model.py tests/lmm_original/test_config.py -q`
Expected: PASS

### Task 3: Add Crack500-Kaggle Config Materialization

**Files:**
- Create: `configs/offline_experiments/crack500_kaggle/unet.yml`
- Create: `configs/offline_experiments/crack500_kaggle/pspnet.yml`
- Create: `configs/lmm_original_crack500/crack500_kaggle/shufflenetv2.yml`
- Create: `configs/lmm_original_crack500/crack500_kaggle/deepcrack.yml`
- Modify: `scripts/offline_experiments/run_public_benchmarks.py`

**Step 1: Write the failing test**

Extend the public benchmark config test so generated config paths and output directories match the new Crack500-Kaggle layout.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/offline_experiments/test_public_benchmark_configs.py -q`
Expected: FAIL until config generation produces the new files.

**Step 3: Write minimal implementation**

Generate Paddle configs from existing PaddleSeg templates and original-source configs from benchmark YAML templates, keeping dataset and output paths relative.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/offline_experiments/test_public_benchmark_configs.py -q`
Expected: PASS

### Task 4: Add Unified Crack500-Kaggle Runner

**Files:**
- Create: `scripts/offline_experiments/run_crack500_kaggle.py`
- Modify: `scripts/offline_experiments/run_public_benchmarks.py`

**Step 1: Write the failing test**

If practical, add a focused test for runner spec selection; otherwise rely on config/spec unit coverage and keep the runner thin.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/offline_experiments/test_public_benchmark_configs.py -q`
Expected: FAIL only if runner-specific expectations were added.

**Step 3: Write minimal implementation**

Dispatch Paddle models through the Paddle offline experiment utilities and original-source models through `python -m lmm_original_crack500.run`, then summarize selected roots into `output/offline_experiments_torch/tables`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/offline_experiments/test_public_benchmark_configs.py -q`
Expected: PASS

### Task 5: Run Targeted Verification

**Files:**
- Modify: `tests/offline_experiments/test_public_benchmark_configs.py`
- Modify: `tests/lmm_original/test_model.py`
- Modify: `tests/lmm_original/test_config.py`

**Step 1: Run focused tests**

Run: `python -m pytest tests/offline_experiments/test_public_benchmark_configs.py tests/lmm_original/test_model.py tests/lmm_original/test_config.py -q`
Expected: PASS

**Step 2: Sanity-check generated configs**

Run: `python scripts/offline_experiments/run_public_benchmarks.py --generate_only`
Expected: Generated config files include Crack500-Kaggle Paddle and original-source models.

**Step 3: Document execution commands**

Record the exact command to run later:

```bash
python scripts/offline_experiments/run_crack500_kaggle.py --models pspnet unet shufflenetv2 deepcrack
```

**Step 4: Commit**

This workspace currently has no `.git` root available, so skip commit unless the repository is reattached to git metadata.
