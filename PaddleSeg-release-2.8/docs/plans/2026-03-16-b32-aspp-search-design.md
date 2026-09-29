# B32 ASPP Search Design

## Goal

Add a dedicated offline experiment entrypoint to search `HrSegNet-B32` ASPP hyperparameters on the custom dataset without changing the main model logic or destabilizing the existing ablation workflow.

## Constraints

- Reuse the current PaddleSeg-style offline experiment framework.
- Keep training recipe aligned with current ablation settings on `dataset/custom`.
- Avoid modifying model implementation unless absolutely necessary.
- Keep the change small and isolated.

## Options Considered

### Option 1: Extend `run_ablation.py` only

- Add `b32_aspp` and `b32_final` to the existing ASPP-search base choices.
- Run search through the existing `--aspp_rates` / `--aspp_out_channels` interface.

Pros:
- Minimal code change.
- Reuses existing summary and execution flow.

Cons:
- User-facing entrypoint remains generic and slightly cumbersome for repeated B32 search runs.

### Option 2: Add a dedicated wrapper script over `run_ablation.py`

- Keep the existing generic search logic.
- Add `scripts/offline_experiments/run_b32_aspp_search.py` with B32-specific defaults.

Pros:
- Minimal implementation risk.
- Cleaner user-facing workflow for repeated B32 search.
- No refactor of the current ablation runner.

Cons:
- Adds one small extra script.

### Option 3: Build a brand-new standalone runner

- Duplicate train/eval/fps/summary orchestration for B32 search.

Pros:
- Full freedom over output layout.

Cons:
- Duplicates stable logic.
- Higher maintenance cost and more drift risk.

## Recommended Design

Use Option 2.

Implementation shape:

- Expand `run_ablation.py` ASPP search base choices to include `b32_aspp` and `b32_final`.
- Add a dedicated wrapper script `run_b32_aspp_search.py` that:
  - targets the custom dataset by default,
  - defaults to `b32_final`,
  - accepts command-line `--rates` and `--channels`,
  - delegates execution to `run_ablation.py`,
  - uses a B32-specific summary prefix.
- Keep generated configs under the existing ablation-search config root used by `run_ablation.py`.
- Keep result output under the existing `output/offline_experiments/ablation/<experiment_name>` structure to preserve compatibility with current summary tooling.

## Data Flow

1. User runs `run_b32_aspp_search.py`.
2. Wrapper converts rate/channel arguments into the format expected by `run_ablation.py`.
3. `run_ablation.py` generates search configs from `b32_final.yml` or `b32_aspp.yml`.
4. Existing training/eval/fps/summary pipeline executes unchanged.
5. Existing summary export writes ranked tables to `output/offline_experiments/tables/`.

## Error Handling

- Reject invalid ASPP rate strings.
- Reject non-positive output channel values.
- Fail fast if an unknown B32 base variant is requested.
- Preserve existing `--resume_failed` behavior.

## Testing

- Unit test that `run_ablation.py` accepts `b32_aspp` and `b32_final` as ASPP search bases.
- Unit test that the new wrapper builds the expected delegated command.
- Keep existing ASPP search config generation tests as the behavior baseline.
