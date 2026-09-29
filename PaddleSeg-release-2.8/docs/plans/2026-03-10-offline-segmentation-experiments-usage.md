# Offline Segmentation Experiments Usage

## Prepare datasets

```powershell
python scripts/offline_experiments/prepare_datasets.py --prepare_custom --prepare_crack500
```

## Generate configs

```powershell
python scripts/offline_experiments/generate_ablation_configs.py
```

## Run ablation experiments

```powershell
python scripts/offline_experiments/run_ablation.py --device gpu
```

## Run Crack500 experiments

```powershell
python scripts/offline_experiments/run_crack500.py --device gpu
```

## Summarize results

```powershell
python scripts/offline_experiments/summarize_results.py --result_roots output/offline_experiments/ablation --output_dir output/offline_experiments/tables --table_prefix ablation_results
python scripts/offline_experiments/summarize_results.py --result_roots output/offline_experiments/crack500 --output_dir output/offline_experiments/tables --table_prefix crack500_results
```

## Smoke test examples

```powershell
python scripts/offline_experiments/run_ablation.py --models b16_baseline --iters 2 --save_interval 2 --skip_fps
python scripts/offline_experiments/run_crack500.py --models hrsegnet_b16 --iters 2 --save_interval 2 --skip_fps
```
