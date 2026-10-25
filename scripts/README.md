# Reproduction commands

These scripts do not start a queue. `run_official.py` prints its command by
default; add `--execute` only when ready to run it. All examples use paths you
provide. No training data, initialization weights, or trained checkpoints are
bundled.

1. Use Python 3.12 and an appropriate PaddlePaddle installation for your platform.
   `requirements/selected-windows.txt` records the observed Windows packages;
   `requirements/README.md` explains its scope and the distinct historical WSL
   backend. `PyYAML` is needed by the configuration helpers.
2. Obtain the official [PaddleSeg v2.10.0 ZIP](https://github.com/PaddlePaddle/PaddleSeg/archive/refs/tags/v2.10.0.zip).
   Its required SHA256 is
   `8ad8541e3ebb7fe9fc670f4026972be77385e73764f66e94390297e794c268d6`.
   GitHub may regenerate archive bytes; a different digest is refused and must
   be reviewed separately instead of silently replacing the pin.
3. Bootstrap the local runtime and check the portable source package:

```console
python -B scripts/bootstrap_runtime.py --archive /path/to/PaddleSeg-v2.10.0.zip
python -B scripts/check_static.py
```

The runtime contains the upstream `paddleseg` package, official `tools/train.py`
and `tools/val.py`, requirements and LICENSE. Only `paddleseg/models/segformer.py`
is overwritten by the committed overlay. Existing output directories are refused.
`--output` can select a different runtime directory; pass that same directory as
`--runtime` to the runner. No setup script is executed by bootstrap.

The data root must contain `train.txt` and `val.txt` in PaddleSeg's original
two-column image/label list format, with paths relative to the dataset root.
Preserve the original splits and label semantics (background 0, crack 1).
Materialization does not inspect those lists or any images. The artifact root
holds initialization weights under `pretrained/`; see `configs/recipes.json`
for required filenames, known hashes, and the original public download URLs.

```console
python -B scripts/materialize_config.py --config configs/A0.yml --output runtime/configs/A0.yml --data-root /path/to/data --artifact-root /path/to/artifacts
python -B scripts/run_official.py train --recipe A0 --config runtime/configs/A0.yml --save-dir /path/to/artifacts/runs/A0_s42 --seed 42
```

The train runner supplies FP32, seed 42 by default, validation every 1,000
iterations, log interval 100, zero data workers, and the recipe's EMA setting.
The A0 family keeps at most one iteration checkpoint; public recipes leave
retention at the upstream default of five. The template supplies batch
size 4 and 18,000 iterations. Set `--seed 43` or `--seed 44` to repeat a seed.
It checks that the materialized config matches the selected recipe except for
the declared local paths. It sets `PYTHONPATH` to the bootstrapped runtime and
disables Python bytecode output. Add `--execute` to the printed command invocation
to actually start the official training process.

For full-checkpoint validation, the explicit `--for-evaluation` option sets only
existing `pretrained` entries to null, avoiding an unnecessary initialization
file load before loading the full trained checkpoint. All other model and data
fields stay the same. Use a separate generated config for training.

```console
python -B scripts/materialize_config.py --config configs/A0.yml --output runtime/configs/A0_eval.yml --data-root /path/to/data --artifact-root /path/to/artifacts --for-evaluation
python -B scripts/run_official.py val --recipe A0 --config runtime/configs/A0_eval.yml --model-path /path/to/artifacts/A0/model.pdparams
python -B scripts/run_official.py exact --recipe A0 --config runtime/configs/A0_eval.yml --model-path /path/to/artifacts/A0/model.pdparams --output /path/to/artifacts/evaluations/A0_val
```

The exact evaluator is a byte-for-byte copy of the original validation evaluator
(`evaluate_val_boundary.py`, SHA256
`97ac482974536c1cd9be268b126313741eedbb6697b5f9af65adc873be7f53a6`).
It accepts validation mode only and writes a JSON aggregate and a per-image CSV.
Its boundary implementation uses 3x3 erosion with a constant-zero border and
symmetric matching within Euclidean distance 2 pixels. The original framework
checkpoint loader is preserved, including its handling of missing or extra keys;
check loading logs when supplying a checkpoint. This package does not strengthen
or alter that algorithm.

`--device cpu` is available for CPU execution; the default is GPU. Merely
materializing a configuration, bootstrapping, running `check_static.py`, or
capturing the environment never imports Paddle or starts model execution.

```console
python -B scripts/capture_environment.py --output runtime/environment_local.json
```
