# Crack500 Kaggle Multi-Model Predict Design

**Goal**

Add a single script that takes one or more input images and runs inference with every usable model under `output/offline_experiments_torch/crack500_kaggle`, then writes black-background / white-crack binary masks.

**Inputs**

- One image path
- Multiple image paths
- Directory of images

**Outputs**

- Per-model binary PNG masks under `output/offline_experiments_torch/crack500_kaggle_predictions/<model_name>/`
- A manifest JSON describing discovered models, resolved config/model paths, and saved output files

**Model Discovery**

The script must not rely only on `runtime_config.yml` inside each experiment directory because some result folders only keep metrics/stat summaries locally. Discovery order:

1. Look for local `runtime_config.yml`
2. Fall back to `metrics.json` / `model_stats.json` / `summary_row.json` config paths
3. Resolve model weights from metadata first, then from local default locations
4. Determine model family by weight suffix:
   - `.pdparams` -> Paddle
   - `.pt` -> Torch

**Inference**

- Paddle models use `Config`, `SegBuilder`, `Compose(builder.val_transforms)`, and `infer.inference`.
- Torch models use `torch_crack_benchmark` model loading and the same mean/std normalization as benchmark evaluation.
- All predictions are converted to a single-channel mask with `0` for background and `255` for crack.

**Validation**

- Add focused unit tests for experiment discovery and binary mask saving.
- Run the script on `dataset/custom/images/train/1.jpg`.
