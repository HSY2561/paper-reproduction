# LMM Original Wrapper Multi-Model Extension

## Goal

Extend the existing original-source Crack500 wrapper so it can run not only `LM_Net`, but also the lightweight baselines already shipped in the upstream repository:

- `Shufflenetv2`
- `Mobilenetv3`

All of them should reuse the same Crack500 protocol already aligned to HrSegNet.

## Scope

Keep the current wrapper architecture:

- original model code stays in `Lightweight-Modular-Model-main/Codes`
- the wrapper layer stays in `lmm_original_crack500`
- `Lightweight-Modular-Model-main/experiments/crack500` keeps the run-facing config and launch scripts

## Changes

- add a model-type resolver in the wrapper
- make training, evaluation, stats, and fps scripts build models from config instead of hard-coding `LM_Net`
- add two new configs:
  - `shufflenetv2_crack500.yml`
  - `mobilenetv3_crack500.yml`

## Protocol

The wrapper should keep the same Crack500 protocol as HrSegNet:

- dataset split: current `dataset/Crack500/*.txt`
- crop size: `720`
- optimizer: `SGD`
- lr schedule: `PolynomialDecay + warmup`
- total iters: `12000`

For lightweight backbones that fit memory, use direct batch `8`. If a model needs it, gradient accumulation remains available.
