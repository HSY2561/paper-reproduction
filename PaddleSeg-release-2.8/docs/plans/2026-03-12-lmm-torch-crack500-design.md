# LMM Torch Crack500 Design

**Goal:** Add a standalone PyTorch-based LMM experiment folder for Crack500 that matches the current Paddle Crack500 experiment protocol and produces directly comparable result artifacts.

**Scope:**
- Reuse `dataset/Crack500/train.txt`, `val.txt`, and `test.txt`.
- Match the current Crack500 training recipe where possible: `batch_size=8`, `crop_size=720`, same split files, same `iters` override style, same reported metrics.
- Keep Paddle experiment code unchanged except for optional cross-framework result summarization hooks if needed.
- Do not run LMM through Paddle.

**Architecture:**
- Create a new isolated folder: `experiments/lmm_torch/`.
- Vendor the original PyTorch LMM model implementation into that folder and keep it separate from `paddleseg/`.
- Add thin PyTorch-specific scripts for config loading, dataset adaptation, training, evaluation, FPS benchmarking, and summary generation.
- Mirror the Paddle offline experiment output shape: each run writes `metrics.json`, `model_stats.json`, `fps.json`, and `summary_row.json`.

**Data Flow:**
- Read Crack500 list files already prepared under `dataset/Crack500`.
- Build a PyTorch dataset that parses `image mask` lines and applies training transforms compatible with the existing Paddle setup.
- Train/evaluate under `output/offline_experiments_torch/crack500/lmnet/`.

**Constraints:**
- Current checked PyTorch env is `E:\anaconda3\envs\cuda11-py310\python.exe` with `torch 2.5.1+cpu`. The implementation should not assume GPU availability.
- The new branch must be runnable independently of Paddle.

**Validation:**
- Unit-test config parsing / summary generation / dataset line parsing.
- Smoke-test config loading and result file generation without requiring full training.
