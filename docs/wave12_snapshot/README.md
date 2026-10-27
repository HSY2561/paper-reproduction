# Wave12 snapshot (2026-10-27)

This is a read-only evidence snapshot. Twelve seed-42 configurations (D01-D10, C01/C02) completed the registered 18,000-iteration train, official validation and pixel-boundary stages with exit code 0. The comparison is validation-only (`test_used=false`).

The phase-2 profiling controller later failed after a deferred memory child; the failure receipt and deferred-attempt marker are retained. No latency release or final-model selection is asserted. All twelve rows remain `accuracy.passed=false` under the registered matched-R0 gates. Checkpoints, datasets, raw logs, caches and active profile directories are intentionally excluded.

See [results](../../results/wave12_20261027/comparison.json), [failure receipt](../../results/wave12_20261027/workflow_failure.json), and the frozen [experiment plan](EXPERIMENT_PLAN_20261027.md).
