# Offline Segmentation Experiments Design

## Goal

Build an isolated offline experiment workflow for PaddleSeg crack segmentation that supports:

- Ablation study on the custom dataset
- Generalization experiments on Crack500
- Automatic result aggregation into paper-ready tables

This workflow must not disturb the existing `data/` layout, the deployed final model, or later robot experiments.

## Constraints

- Keep `data/` unchanged.
- Keep current deployed model logic unchanged.
- Preserve existing `configs/aa/...` files.
- Favor new scripts, generated configs, and minimal model additions over wide refactors.
- Back up any original script before modifying it.

## Naming

- `HrSegNet-B16`: baseline
- `HrSegNet-B16 + ASPP`: ASPP only
- `HrSegNet-B16 + Decoder`: lite decoder only
- `HrSegNet-B16-A`: ASPP + lite decoder, final paper model

Existing `HrSegNet_Lite` remains for compatibility. The offline experiment workflow will use a new B16-family ablation model class with structure switches.

## Data Layout

### Existing data

- `data/`
  - Remains untouched for legacy training and robot workflows

### Offline experiment datasets

- `dataset/custom/`
  - A copied mirror of `data/`
  - Same structure:
    - `images/train|val|test`
    - `annotations/train|val|test`
    - `train.txt`, `val.txt`, `test.txt`

- `dataset/Crack500/`
  - Uses the existing split layout:
    - `train/images`, `train/masks`
    - `val/images`, `val/masks`
    - `test/images`, `test/masks`
  - Add generated:
    - `train.txt`, `val.txt`, `test.txt`

The generated txt files use PaddleSeg relative paths from dataset root.

## Architecture

### Model layer

Add `paddleseg/models/hrsegnet_b16_ablation.py`:

- Reuse existing `SegBlock`, `SegHead`, and ASPP-style modules
- Expose structure switches:
  - `use_aspp`
  - `use_lite_decoder`
- Register dedicated classes for:
  - `HrSegNetB16ASPP`
  - `HrSegNetB16Decoder`
  - `HrSegNetB16A`

Baseline continues to use `HrSegNetB16`.

### Script layer

Add `scripts/offline_experiments/` with:

- `prepare_datasets.py`
- `generate_ablation_configs.py`
- `evaluate_experiment.py`
- `benchmark_fps.py`
- `run_ablation.py`
- `run_crack500.py`
- `summarize_results.py`

### Config layer

Generate experiment configs under:

- `configs/offline_experiments/ablation/`
- `configs/offline_experiments/crack500/`

All ablation configs share the same training strategy. Only structure differs.

## Metrics Strategy

Paper-facing metrics use crack foreground class values:

- IoU
- Dice
- Precision
- Recall
- F1

Deployment-facing metrics:

- Params (M)
- FPS

`mIoU` can still be kept in raw metrics files, but summary tables will default to foreground IoU for binary crack segmentation.

## Tooling Strategy

Reuse PaddleSeg entry points where helpful:

- `tools/train.py`
- `tools/model/analyze_model.py`

Evaluation will use a dedicated offline experiment evaluator so metrics can be saved as structured JSON without forcing a broad change to PaddleSeg core behavior.

## Output Layout

- `output/offline_experiments/ablation/<experiment_name>/`
- `output/offline_experiments/crack500/<experiment_name>/`
- `output/offline_experiments/tables/`

Each experiment directory should contain:

- `train.log`
- `val.log`
- `metrics.json`
- `model_stats.json`
- `fps.json`
- `summary_row.json`

## Verification

- Dataset preparation dry-run and file count checks
- YAML generation validation by parsing generated configs
- Model build smoke tests through PaddleSeg `Config`
- Lightweight train/val smoke test on generated configs
- Summary generation test on synthetic result files

## Backup Policy

Before modifying any original script:

- `backup/offline_experiments/2026-03-10/tools/train.py`
- `backup/offline_experiments/2026-03-10/tools/val.py`
- `backup/offline_experiments/2026-03-10/tools/model/analyze_model.py`

If a script does not need modification, no backup is created for it.
