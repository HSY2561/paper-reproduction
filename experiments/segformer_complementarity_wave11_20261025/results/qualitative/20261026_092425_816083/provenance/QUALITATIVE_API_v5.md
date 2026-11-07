# W11 qualitative supplement API

This is a separate optional supplement. It never changes the training or profiling release, starts training, creates scheduled tasks, or overwrites the original report package.

CPU preparation (already performed once):

```powershell
C:\venv_bisenet\Scripts\python.exe qualitative_w11_v5.py --prepare
```

This checks small in-memory mask/color/PNG arrays and the frozen24 manifest, tests incomplete-training refusal, and freezes `provenance/qualitative_release_v5.json`. No generated test images are exported; no actual input images, Paddle, GPU context or model inference are used during preparation.

Render only after the 15 training runs and automatic profiling/report delivery have genuinely completed:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src\PaddleSeg-2.10.0'
C:\venv_bisenet\Scripts\python.exe qualitative_w11_v5.py --render
```

Run in the experiment directory. The renderer requires full training receipts, automatic postprocess child and waiter wrapper exit0, absent training/profile locks, exited owned PIDs, unchanged releases/checkpoints/report package, and the frozen WDDM background guard. During training `--render` refuses before importing Paddle. It never stops another application.

The renderer uses the existing official val348 Dataset and indexes the preregistered24 cases in memory. It creates no dataset or absolute-path inference list and calls `paddleseg.core.infer.inference`, with argmax and no new threshold. All five models use the frozen seed42 checkpoints. For every case/model, TP/FP/FN/TN must match its full348 CSV. The actual W11 evaluator CSV has `prediction_sha256` and `gt_mask_sha256`; this renderer requires and compares both. Counts alone do not prove pixelwise identity.

Outputs are in a fresh English directory `results/qualitative/YYYYmmdd_HHMMSS_microseconds/`: 24 full comparison panels, full-size original/GT/prediction/error/skeleton images, overview PNGs, exact verification CSV/JSON, 4488-file before/after audits, manifest, and `W11_fixed24_qualitative_supplement.zip`. The archive includes only the selected24 input exports and derived masks/panels, no full raw dataset or weights. All cases and failed verifications are preserved. A failed comparison exits2 after packaging evidence; technical failure exits1 and preserves partial outputs.

Latest successful or verification-failed completion is indexed in `results/qualitative_latest.json`. This separate supplement does not alter the original report archive or its SHA. The heartbeat may call `--render` once the guards pass; no new scheduled task is needed.

The first CPU-only preparation was preserved in `provenance/qualitative_v1_cpu_preserved_20261025_2010/` before v2 corrected the reader of the existing profile-child/controller PID evidence. No GPU inference occurred in that version, and the training/profile releases were unchanged.

## V3 isolated desktop-baseline compatibility

The original frozen training, postprocess v2 and qualitative v2 files remain unchanged. V3 uses the separately verified current desktop C+G PID/name/path baseline. It still blocks foreign C and unknown C+G/Python processes, applies the same low/stable background window and CV rule, and stops no user applications. No metrics, model weights, training protocol, sampling, thresholds or inference formulas change. The old deferred waiter and its real nonzero stop receipt remain preserved. V3 uses its own source, release and CPU self-test paths.

## V4 exact executable identity compatibility

V2 and V3 remain unchanged. V4 permits a desktop C+G process with a changed PID only when its absolute path and basename match an explicitly frozen binary identity and the current executable SHA256 matches that identity. Pure C, suspicious Python/training names, new PIDs without full paths, unknown paths and changed binary hashes remain blocked. Classification records the exact identity and hash evidence; low/stable background samples and CV rules remain unchanged. No user applications are stopped.

The V3 waiter stop wrapper produced `exit_code: null` because native stderr interrupted the old PowerShell native invocation; that code remains unknown and is not rewritten as exit1. V4 uses Start-Process with redirected output, WaitForExit and the process ExitCode; its CPU stderr/exit7 probe returned exactly7. This repairs receipt capture only.

## V5 explicit transport-unknown CPU handoff

The actual profile/report child and all ten memory children have genuine exit0 receipts. The v4 transport wrapper still recorded exit_code null; it remains unknown and is never changed to zero. Only that exact v4 wrapper version with explicit null can use a separately executed CPU-only verification witness. The witness and a second receipt recording tools.exec_command actual returned exit0 bind the unchanged wrapper, automatic child receipt and final-delivery receipt by SHA256. All owned PIDs, the original ZIP/manifest, checkpoints, full training receipts and all existing data/CSV gates must independently pass before this compatibility branch is accepted. Missing/changed witness, nonzero exit or any other unknown wrapper version is rejected. The witness and exec receipt are frozen into the v5 release and copied into the supplement provenance/manifest. V5 still uses frozen postprocess v4; it starts no repeated profiling.
