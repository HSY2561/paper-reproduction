# Crack500 Kaggle Hybrid Benchmarks Design

**Goal**

Add a single Crack500-Kaggle benchmark setup that mixes PaddleSeg models and original-source PyTorch models while writing all experiment artifacts under `output/offline_experiments_torch/crack500_kaggle`.

**Scope**

- PaddleSeg models: `pspnet`, `unet`
- Original-source models from `Lightweight-Modular-Model-main/Codes`: `shufflenetv2`, `deepcrack`
- No training run in this task
- Post-run artifacts must remain compatible with existing summary tooling: `runtime_config.yml`, `metrics.json`, `model_stats.json`, `fps.json`, `summary_row.json`

**Approach**

Use model-family-specific runners instead of forcing all four models through one training implementation:

- PaddleSeg models keep using the existing offline experiment evaluation/stat/FPS scripts.
- Original-source models keep using the existing `lmm_original_crack500` runtime style, extended to support `deepcrack`.
- A thin public benchmark layer generates the right config files and exposes a single batch runner for Crack500-Kaggle.

**Config Layout**

- `configs/offline_experiments/crack500_kaggle/unet.yml`
- `configs/offline_experiments/crack500_kaggle/pspnet.yml`
- `configs/lmm_original_crack500/crack500_kaggle/shufflenetv2.yml`
- `configs/lmm_original_crack500/crack500_kaggle/deepcrack.yml`

All four configs target `dataset/Crack500_kaggle` and write results into `output/offline_experiments_torch/crack500_kaggle/<model>`.

**Runner Layout**

- Extend public benchmark config generation so Crack500-Kaggle knows about:
  - Paddle: `segformer_b0`, `pspnet`, `unet`
  - Original-source: `shufflenetv2`, `deepcrack`
- Add a unified Crack500-Kaggle runner that dispatches:
  - Paddle configs to PaddleSeg train/eval/stat/FPS flow
  - Original-source configs to `lmm_original_crack500.run`

**Data Flow**

1. Generate relative-path configs for Crack500-Kaggle.
2. Run selected experiments.
3. Each experiment materializes `runtime_config.yml` into its own output directory.
4. Train/eval/stat/FPS stages write standard JSON artifacts.
5. Summary generation writes `summary_row.json` per experiment and a merged table under `output/offline_experiments_torch/tables`.

**Error Handling**

- Reject unknown model names early.
- Keep resume behavior compatible with current runners.
- Keep model lookup strict so missing checkpoints fail clearly before eval/FPS.

**Testing**

- Update unit tests for public benchmark specs/config generation.
- Update original-source model spec tests to include `deepcrack`.
- Add tests for Crack500-Kaggle config paths and output roots.

**Constraints**

- Preserve existing Crackseg9k behavior.
- Do not move existing output roots for already supported experiments.
- Avoid changing training implementations unless required for model registration.
