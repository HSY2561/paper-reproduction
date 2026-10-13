# Wave 5 artifact manifest

- Configs: `configs/A_EMAU_context.yml`, `configs/B_Dual_context.yml`, `configs/C_Ham_decoder.yml`.
- Formal exits: `results/A_EMAU_context_exit.json`, `results/B_Dual_context_exit.json`, `results/C_Ham_decoder_exit.json`.
- Independent verification: `results/A_EMAU_context_ordinary_verified.json`, `results/B_Dual_context_ordinary_verified.json`, `results/C_Ham_decoder_ordinary_verified.json` and corresponding EMA files.
- Selection records: `results/A_EMAU_selection.json`, `results/B_Dual_context_selection.json`, `results/C_Ham_decoder_selection.json`.
- Protocol and data evidence: `experiment_plan_20261012.md`, `provenance/preflight_data_hashes.json`.
- Aggregate result: `results/wave5_summary.json`.

All formal candidates used train/val only; `test_used` is false in their result records. The candidate checkpoint copies are under `checkpoints/` and have SHA256 values in the selection records.
