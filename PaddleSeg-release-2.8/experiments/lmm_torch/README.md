# LMM Torch Crack500

This folder provides a standalone PyTorch branch for running the LMM model on the same Crack500 split used by the Paddle offline experiments.

## Protocol

- Dataset lists: `dataset/Crack500/train.txt`, `val.txt`, `test.txt`
- Default training crop: `720 x 720`
- Default micro batch size: `1`
- Default gradient accumulation: `8`
- Effective batch size: `8`
- Default iters: `12000`
- Output root: `output/offline_experiments_torch/crack500/lmnet`
- Summary schema matches Paddle `output/offline_experiments/crack500/*/summary_row.json`

## Interpreter

Current checked interpreter with PyTorch:

```powershell
E:\anaconda3\envs\cuda11-py310\python.exe
```

At the time of implementation this interpreter had `torch 2.5.1+cpu`, so training will run on CPU unless you install a CUDA-enabled PyTorch build.

## Run

```powershell
E:\anaconda3\envs\cuda11-py310\python.exe experiments/lmm_torch/run_crack500_lmm.py --iters 12000 --device cpu
```

If you later install a CUDA-enabled PyTorch build in that interpreter, change `--device cpu` to `--device cuda`.

## Smoke test

```powershell
E:\anaconda3\envs\cuda11-py310\python.exe experiments/lmm_torch/train.py --config experiments/lmm_torch/configs/crack500_lmnet.yml --iters 1 --batch_size 1 --device cpu --output_dir output/offline_experiments_torch/crack500/lmnet_smoke --skip_val
```

## Output files

- `train.log`
- `eval_test.log`
- `metrics.json`
- `model_stats.json`
- `fps.json`
- `summary_row.json`
- `best_model.pt`
