# Offline Segmentation Experiments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an isolated offline experiment workflow for crack segmentation ablation and Crack500 generalization experiments, with reproducible configs, batch runners, and paper-ready summaries.

**Architecture:** Add a dedicated offline experiment layer around PaddleSeg. Keep `data/` and deployed logic unchanged, generate new configs under `configs/offline_experiments/`, add B16 ablation model variants, and store structured metrics in `output/offline_experiments/`.

**Tech Stack:** Python, PaddleSeg, YAML, pandas, argparse, unittest

---

### Task 1: Create the scaffolding and backups

**Files:**
- Create: `backup/offline_experiments/2026-03-10/.gitkeep`
- Create: `scripts/offline_experiments/.gitkeep`
- Create: `configs/offline_experiments/ablation/.gitkeep`
- Create: `configs/offline_experiments/crack500/.gitkeep`
- Test: none

**Step 1: Add the directory scaffolding**

Create the backup, script, and config directories with placeholder files.

**Step 2: Verify directory creation**

Run: `Get-ChildItem backup/offline_experiments/2026-03-10, scripts/offline_experiments, configs/offline_experiments/ablation, configs/offline_experiments/crack500`

Expected: All four directories exist.

### Task 2: Write failing tests for offline experiment utilities

**Files:**
- Create: `tests/offline_experiments/test_prepare_datasets.py`
- Create: `tests/offline_experiments/test_generate_configs.py`
- Create: `tests/offline_experiments/test_summarize_results.py`
- Test: `tests/offline_experiments/test_prepare_datasets.py`
- Test: `tests/offline_experiments/test_generate_configs.py`
- Test: `tests/offline_experiments/test_summarize_results.py`

**Step 1: Write failing tests**

Cover:

- Crack500 txt generation
- Custom dataset copy plan
- YAML config generation with consistent dataset roots and model types
- Summary aggregation from per-experiment JSON files

**Step 2: Run tests to verify failure**

Run: `python -m unittest discover -s tests/offline_experiments -v`

Expected: Fail because the target modules do not exist yet.

### Task 3: Implement dataset preparation utilities and CLI

**Files:**
- Create: `scripts/offline_experiments/prepare_datasets.py`
- Modify: none
- Test: `tests/offline_experiments/test_prepare_datasets.py`

**Step 1: Implement pure utility functions**

Add functions for:

- Mirroring `data/` into `dataset/custom/`
- Validating image-mask pairing
- Writing PaddleSeg txt files for Crack500

**Step 2: Add CLI wrapper**

Support:

- preparing custom copy
- preparing Crack500 splits
- `--custom_src`, `--custom_dst`, `--crack500_root`, `--dry_run`

**Step 3: Run tests**

Run: `python -m unittest tests.offline_experiments.test_prepare_datasets -v`

Expected: PASS.

### Task 4: Implement B16 ablation model variants

**Files:**
- Create: `paddleseg/models/hrsegnet_b16_ablation.py`
- Modify: `paddleseg/models/__init__.py`
- Test: `tests/offline_experiments/test_generate_configs.py`

**Step 1: Reuse the existing building blocks**

Implement B16-family variants with configurable:

- ASPP branch on/off
- lite decoder on/off

**Step 2: Register dedicated model classes**

Register:

- `HrSegNetB16ASPP`
- `HrSegNetB16Decoder`
- `HrSegNetB16A`

**Step 3: Add a config-build smoke check**

Generated configs must instantiate through PaddleSeg `Config`.

### Task 5: Implement config generation

**Files:**
- Create: `scripts/offline_experiments/generate_ablation_configs.py`
- Test: `tests/offline_experiments/test_generate_configs.py`

**Step 1: Read base configs**

Use:

- `configs/aa/hrsegnetb16.yml`
- `configs/aa/hrsegnetb16_asppdelite.yml`
- `configs/aa/unet.yml`
- `configs/aa/deeplabv3p.yml`

**Step 2: Generate offline configs**

Generate:

- Ablation configs on custom dataset
- Crack500 configs for three HrSegNet variants, UNet, and DeepLabV3P

**Step 3: Validate generated YAML**

Run YAML parse and optional PaddleSeg `Config` build checks.

### Task 6: Implement structured evaluation and stats collection

**Files:**
- Create: `scripts/offline_experiments/evaluate_experiment.py`
- Create: `scripts/offline_experiments/benchmark_fps.py`
- Create: `scripts/offline_experiments/model_stats.py`
- Test: `tests/offline_experiments/test_summarize_results.py`

**Step 1: Implement evaluator**

Compute and save:

- foreground IoU
- foreground Dice
- foreground Precision
- foreground Recall
- F1
- raw class metrics

**Step 2: Implement model stats helper**

Build config and compute Params (M), optionally FLOPs if available.

**Step 3: Implement benchmark helper**

Provide a best-effort FPS benchmark interface with output JSON and graceful fallback to `NA`.

### Task 7: Implement batch runners

**Files:**
- Create: `scripts/offline_experiments/run_ablation.py`
- Create: `scripts/offline_experiments/run_crack500.py`
- Modify: `tools/train.py` only if required for compatibility
- Modify: `tools/model/analyze_model.py` only if required for compatibility
- Test: smoke execution commands

**Step 1: Add reusable subprocess helpers**

Run:

- train
- eval
- model stats
- fps benchmark

**Step 2: Standardize output folders**

Each run writes logs and JSON outputs into experiment-specific directories.

**Step 3: Support partial experiment selection**

Allow `--models` filters and `--skip_train`, `--skip_eval`, `--skip_fps` options.

### Task 8: Implement result summarization

**Files:**
- Create: `scripts/offline_experiments/summarize_results.py`
- Test: `tests/offline_experiments/test_summarize_results.py`

**Step 1: Read per-experiment JSON files**

Collect:

- metrics
- params
- fps
- config path
- best model path

**Step 2: Write summary tables**

Write:

- CSV
- XLSX

under `output/offline_experiments/tables/`.

### Task 9: Run verification

**Files:**
- Modify: none
- Test: generated files and smoke test outputs

**Step 1: Run unit tests**

Run: `python -m unittest discover -s tests/offline_experiments -v`

Expected: PASS.

**Step 2: Run dataset preparation**

Run: `python scripts/offline_experiments/prepare_datasets.py --prepare_custom --prepare_crack500`

Expected: `dataset/custom/` mirrored and Crack500 txt files generated.

**Step 3: Run config generation**

Run: `python scripts/offline_experiments/generate_ablation_configs.py`

Expected: offline configs created and parse successfully.

**Step 4: Run smoke tests**

Run minimal-iteration custom and Crack500 commands with `--iters 2` or config overrides.

Expected: train and eval paths complete, JSON metrics are written.

### Task 10: Document usage

**Files:**
- Create: `docs/plans/2026-03-10-offline-segmentation-experiments-usage.md`

**Step 1: Write concise usage reference**

Include:

- preparation command
- ablation run command
- Crack500 run command
- summary command
- output locations
