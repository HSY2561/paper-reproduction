# Multi-Dataset Crack Benchmarks Design

## Goal

Prepare a portable benchmark branch that can be moved to another server and run on two datasets:

- `dataset/Crackseg9k`
- `dataset/Crack500_kaggle`

The benchmark should compare:

- Paddle: `SegFormer-B0`
- PyTorch wrappers around original-source models:
  - `CrackFormer-II`
  - `DeepCrack`
  - `Efficientnet`
  - `Mobilenetv3`

All paths must remain relative. PyTorch training should support automatic early stopping based on validation convergence.

## Constraints

- Keep existing Paddle offline experiment workflow intact.
- Reuse existing wrapper patterns where possible.
- Avoid modifying original model source files unless absolutely necessary.
- Target CUDA 12.1 for `.venv` PyTorch installation.

## Recommended Architecture

### Paddle branch

Keep Paddle-only models in the current framework:

- add dataset-specific `SegFormer-B0` configs for:
  - `Crackseg9k`
  - `Crack500_kaggle`
- add a lightweight runner that can launch those dataset-specific Paddle comparisons using relative dataset paths.

### PyTorch branch

Generalize `lmm_original_crack500` into a reusable multi-model, multi-dataset wrapper:

- new package: `torch_crack_benchmark`
- supported models:
  - `crackformer_ii`
  - `deepcrack`
  - `efficientnet`
  - `mobilenetv3`
- dataset config fields:
  - `dataset_name`
  - `dataset_root`
  - `train_list`
  - `val_list`
  - `test_list`
- outputs:
  - `train.log`
  - `metrics.json`
  - `model_stats.json`
  - `fps.json`
  - `summary_row.json`

## Early Stopping

Use validation `mIoU`:

- `max_iters = 18000`
- `save_interval = eval_interval = 1000`
- `early_stop_metric = val mIoU`
- `early_stop_min_delta = 0.001`
- `early_stop_patience = 8`

Behavior:

- If best `mIoU` fails to improve by at least `min_delta` for `patience` consecutive evaluations, stop training.
- Keep saving `best_model`.
- Log the early-stop decision clearly in `train.log`.

## Data Assumptions

Each dataset should follow Paddle-style split files:

- `train.txt`
- `val.txt`
- `test.txt`

Each line:

- `relative/image/path relative/mask/path`

The PyTorch wrapper will read those same split files directly, so Paddle and PyTorch comparisons use identical splits.

## Config Strategy

### Paddle configs

Generate:

- `configs/offline_experiments/crackseg9k/segformer_b0.yml`
- `configs/offline_experiments/crack500_kaggle/segformer_b0.yml`

### PyTorch configs

Generate dataset/model configs under:

- `configs/torch_crack_benchmark/crackseg9k/*.yml`
- `configs/torch_crack_benchmark/crack500_kaggle/*.yml`

Each config keeps all paths relative.

## Runners

### Paddle runner

Add a small dataset-oriented runner for `SegFormer-B0`:

- `scripts/offline_experiments/run_public_benchmarks.py`

### PyTorch runner

Add a generic benchmark runner:

- `scripts/torch_crack_benchmark/run_benchmark.py`

This runner should:

- accept `--dataset crackseg9k|crack500_kaggle`
- accept `--models crackformer_ii deepcrack efficientnet mobilenetv3`
- support `--resume_failed`
- support `--device cuda`
- export a ranked summary table

## Verification

Test only behavior that does not require real datasets:

- config loading
- model resolution
- early-stop controller logic
- runner command construction / config generation

PyTorch installation verification in `.venv` should confirm:

- `torch.__version__`
- `torch.version.cuda == "12.1"` compatible build
- `torch.cuda.is_available()`
