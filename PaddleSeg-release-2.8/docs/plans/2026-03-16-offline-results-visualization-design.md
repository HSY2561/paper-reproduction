# Offline Results Visualization Design

## Goal

Generate paper-ready comparison figures directly from existing offline experiment outputs under:

- `output/offline_experiments/ablation`
- `output/offline_experiments/crack500`

## Output

Create four PNG figures:

- `ablation_metrics.png`
- `ablation_efficiency.png`
- `crack500_metrics.png`
- `crack500_efficiency.png`

## Visual Design

- Metrics figures:
  - horizontal grouped bars
  - metrics: `Precision`, `Recall`, `F1`, `IoU`, `mIoU`
  - models sorted by `F1` descending
- Efficiency figures:
  - two horizontal bar subplots
  - left: `Params_M`
  - right: `FPS`

## Implementation Boundary

- add one standalone script under `scripts/offline_experiments`
- reuse existing `summary_row.json`
- do not change training or evaluation flows
