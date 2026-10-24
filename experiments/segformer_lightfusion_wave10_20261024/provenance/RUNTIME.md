# Runtime witness

- Host: local RTX5060, Windows
- Python: `C:\venv_bisenet\Scripts\python.exe`
- Source: `src/PaddleSeg-2.10.0` in this wave; `PYTHONPATH` is set by the queue to this snapshot.
- Data: read-only `C:\Users\user\algolab_data`; train/val only during this wave.
- Protocol: FP32, batch 4, AdamW, lr 6e-5, polynomial end_lr 0, seed 42, 18,000 iterations, crop 400×400.
- Safety: the queue keeps the selected frozen checkpoint and selection SHA, prunes redundant iteration checkpoints after independent validation, and pauses below 4 GiB free space. Original data and prior-wave evidence are not modified.
- Source and config hashes are written to `provenance/source_hashes.json` and `provenance/config_hashes.json` after preparation.
