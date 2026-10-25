# Observed environment

`environment_windows.json` was captured on 2026-10-25 with
`importlib.metadata`; no Paddle import or CUDA probe was performed.
`selected-windows.txt` pins those selected installed packages, not the full
transitive environment. The recorded Python version is 3.12.10 on Windows AMD64.

`paddleseg-v2.10.0-upstream.txt` preserves the archive's original requirements
as provenance. Its older `opencv-python <= 4.6.0` constraint differs from the
observed `opencv-python-headless 4.11.0.86` environment. Do not treat the two
lists as a combined tested installer specification.

Install an appropriate PaddlePaddle GPU wheel and GPU runtime for the intended
platform before running models. The wheel index, CUDA/cuDNN libraries, driver,
and hardware are not inferred from distribution version metadata. A package
version list alone does not establish identical numerical behavior.

The original PSPNet, DeepLabV3+ and Fast-SCNN public runs used WSL2 Ubuntu 24.04
and PaddlePaddle GPU `3.3.0.dev20251209`. This directory does not invent a full
lock file for that separate historical backend. The original EMA and loss
differences remain documented in `configs/recipes.json`.

Use `python -B scripts/capture_environment.py` to capture the same selected
distribution metadata on another system. The script does not record user names,
home paths, host names, environment variables, or credentials.
