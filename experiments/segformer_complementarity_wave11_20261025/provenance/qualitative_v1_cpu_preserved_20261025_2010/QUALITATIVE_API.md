# W11 qualitative supplement API

This is a separate optional supplement. It never changes the training or profiling release, starts training, creates scheduled tasks, or overwrites the original report package.

CPU preparation (already performed once):

```powershell
C:\venv_bisenet\Scripts\python.exe qualitative_w11.py --prepare
```

This checks small in-memory mask/color/PNG arrays and the frozen24 manifest, tests incomplete-training refusal, and freezes `provenance/qualitative_release.json`. No generated test images are exported; no actual input images, Paddle, GPU context or model inference are used during preparation.

Render only after the 15 training runs and automatic profiling/report delivery have genuinely completed:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src\PaddleSeg-2.10.0'
C:\venv_bisenet\Scripts\python.exe qualitative_w11.py --render
```

Run in the experiment directory. The renderer requires full training receipts, automatic postprocess child and waiter wrapper exit0, absent training/profile locks, exited owned PIDs, unchanged releases/checkpoints/report package, and the frozen WDDM background guard. During training `--render` refuses before importing Paddle. It never stops another application.

The renderer uses the existing official val348 Dataset and indexes the preregistered24 cases in memory. It creates no dataset or absolute-path inference list and calls `paddleseg.core.infer.inference`, with argmax and no new threshold. All five models use the frozen seed42 checkpoints. For every case/model, TP/FP/FN/TN must match its full348 CSV. If prior mask hashes exist, they are compared too; counts alone do not prove pixelwise identity.

Outputs are in a fresh English directory `results/qualitative/YYYYmmdd_HHMMSS_microseconds/`: 24 full comparison panels, full-size original/GT/prediction/error/skeleton images, overview PNGs, exact verification CSV/JSON, 4488-file before/after audits, manifest, and `W11_fixed24_qualitative_supplement.zip`. The archive includes only the selected24 input exports and derived masks/panels, no full raw dataset or weights. All cases and failed verifications are preserved. A failed comparison exits2 after packaging evidence; technical failure exits1 and preserves partial outputs.

Latest successful or verification-failed completion is indexed in `results/qualitative_latest.json`. This separate supplement does not alter the original report archive or its SHA. The heartbeat may call `--render` once the guards pass; no new scheduled task is needed.
