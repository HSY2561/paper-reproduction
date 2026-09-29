# B32 ASPP Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a dedicated `HrSegNet-B32` ASPP search script for the custom dataset while reusing the current offline ablation pipeline.

**Architecture:** Extend the generic ASPP-search entry in `run_ablation.py` to accept B32 bases, then add a thin wrapper script with B32-specific defaults that delegates to the existing runner. Keep summaries and outputs in the current offline experiment structure.

**Tech Stack:** Python, PaddleSeg offline experiment scripts, `unittest`

---

### Task 1: Add failing tests for B32 ASPP search entrypoints

**Files:**
- Modify: `tests/offline_experiments/test_run_ablation.py`
- Create: `tests/offline_experiments/test_run_b32_aspp_search.py`

**Step 1: Write the failing tests**

- Add a test asserting `run_ablation.parse_args()` accepts `--aspp_search_base b32_final`.
- Add a test asserting the new wrapper script builds a delegated command targeting `run_ablation.py` with `b32_final`.

**Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.offline_experiments.test_run_ablation tests.offline_experiments.test_run_b32_aspp_search
```

Expected:
- `run_ablation` test fails because `b32_final` is not an allowed choice yet.
- wrapper-script test fails because the module does not exist yet.

### Task 2: Implement the minimal B32 ASPP search support

**Files:**
- Modify: `scripts/offline_experiments/run_ablation.py`
- Create: `scripts/offline_experiments/run_b32_aspp_search.py`

**Step 1: Extend `run_ablation.py`**

- Add `b32_aspp` and `b32_final` to `--aspp_search_base` choices.

**Step 2: Add wrapper script**

- Parse B32-specific arguments.
- Provide defaults for rates, channels, base variant, and summary prefix.
- Build and run a delegated `run_ablation.py` command using `run_command`.

**Step 3: Keep implementation minimal**

- Reuse existing helper behavior.
- Do not duplicate train/eval/fps/summary orchestration.

### Task 3: Verify behavior and document usage

**Files:**
- Verify: `tests/offline_experiments/test_run_ablation.py`
- Verify: `tests/offline_experiments/test_run_b32_aspp_search.py`
- Reference: `scripts/offline_experiments/run_b32_aspp_search.py`

**Step 1: Run tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.offline_experiments.test_run_ablation tests.offline_experiments.test_run_b32_aspp_search
```

Expected:
- PASS

**Step 2: Smoke-check help output**

Run:

```powershell
.\.venv\Scripts\python.exe scripts/offline_experiments/run_b32_aspp_search.py --help
```

Expected:
- CLI renders with rate/channel/base-variant options.

**Step 3: Provide run command**

- Give the user one ready-to-run command for the custom dataset B32 ASPP search.
