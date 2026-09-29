# LMM Original-Source Crack500 Design

## Goal

Add a minimal experiment wrapper around `Lightweight-Modular-Model-main/Codes` so the original GitHub `LM_Net` can be trained and evaluated on `dataset/Crack500` under the same Crack500 protocol used by the current HrSegNet experiments.

## Chosen Approach

Create a new wrapper layer under `Lightweight-Modular-Model-main/experiments/crack500/` and keep `Codes/LM_Net.py` as the model source of truth. The wrapper will:

- read Paddle-style `train.txt / val.txt / test.txt`
- apply the same training augmentation and crop policy used by HrSegNet Crack500 experiments
- use the same high-level training protocol:
  - `crop_size = 720`
  - `iters = 12000`
  - `effective batch size = 8`
  - `SGD + PolynomialDecay + warmup`
- export result files in the same schema used by the existing offline experiment summaries

## Why This Approach

This gives the user a defensible "original-source LMM under unified protocol" comparison without rewriting the model body or mixing it into PaddleSeg internals. It also avoids large invasive edits to the downloaded repository.

## Directory Layout

New files will live under:

- `Lightweight-Modular-Model-main/experiments/crack500/`
- `tests/lmm_original/`

The wrapper layer will own config loading, dataset adaptation, training, evaluation, fps benchmarking, and summary export.

## Training Protocol

The wrapper will align to the current HrSegNet Crack500 setup:

- dataset root: `dataset/Crack500`
- list files: `dataset/Crack500/train.txt`, `val.txt`, `test.txt`
- resize-step scaling:
  - `min_scale_factor = 0.5`
  - `max_scale_factor = 2.0`
  - `scale_step_size = 0.25`
- train crop:
  - `720 x 720`
- train augment:
  - random horizontal flip
  - random distort with brightness/contrast/saturation ranges matching HrSegNet
- optimizer:
  - `SGD(momentum=0.9, weight_decay=0.0005)`
- lr scheduler:
  - `PolynomialDecay(learning_rate=0.01, end_lr=0.0, power=0.9)`
  - warmup `1000` iters from `1e-5`
- default effective batch size:
  - `8`

Because LMM is much more memory-heavy than HrSegNet at `720`, the wrapper may use micro-batch plus gradient accumulation while keeping effective batch size equal to `8`.

## Metrics and Outputs

The wrapper will output:

- `metrics.json`
- `model_stats.json`
- `fps.json`
- `summary_row.json`
- `best_model.pt`
- `train.log`
- `eval_test.log`

Summary fields will match the existing comparison schema:

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

## Risks

- Original `LM_Net` may still be slow at `720`, even with matched training protocol.
- The model may require `batch_size=1` with accumulation on a 12 GB GPU.
- The wrapper should stay independent from Paddle scripts so failures do not affect current robot-bound segmentation experiments.
