# Wave18 / Crack500 evaluator audit (read-only, 2026-11-09)

This audit only read existing JSON files. No training or evaluator process was
started, and no existing result was changed.

## Wave18 six-dataset status

Results root: `C:\Users\user\Desktop\segformer_priority_wave18_20261107\results`.

| Dataset | Formal val evaluator | Formal test evaluator | Current evidence / gap |
|---|---:|---:|---|
| DeepCrack | 8 (4 ordinary + 4 EMA) | 4 ordinary | Complete; val `n=45`, each `fixed24_cases` has 24 images; test `n=237`, `test_used=true`. Ordinary val mIoU is higher for all four models. |
| CFD | 0 | 0 | Only train/status and evaluator exit receipts. |
| CrackTree260 | 0 | 0 | Only training completion/smoke receipts. |
| CrackLS315 | 0 | 0 | Only train/status receipts. |
| CRKWH100 | 0 | 0 | Only recovery train/monitor receipts. |
| GAPS384 | 0 | 0 | Only train/status receipts. |

DeepCrack also has `v2_` ordinary duplicate evaluator outputs. EMA val JSONs
carry a model/variant suffix in their `dataset` field while ordinary JSONs use
`DeepCrack`; filenames, split, metrics and provenance are present.

## Crack500 wave17 status

Results root: `D:\论文复现\experiments\segformer_crack500_aligned_wtconv_ablation_wave17_20261030\results`.

All four models have ordinary and EMA val metrics (`n=348`,
`test_used=false`, 24 fixed cases each) and one formal test metrics JSON
(`n=1124`, `test_used=true`). The four selection files all apply the exact
ordinary/EMA validation mIoU rule and select ordinary. Crack500 test JSONs omit
the `dataset` field, but retain config, checkpoint, test flag, sample count and
metrics; this is a metadata cleanup item rather than a missing evaluator.
