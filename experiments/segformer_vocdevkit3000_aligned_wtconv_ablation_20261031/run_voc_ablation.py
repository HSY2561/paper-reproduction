import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"C:\venv_bisenet\Scripts\python.exe")
SRC = ROOT / "src" / "PaddleSeg-2.10.0"
JOBS = [
    ("C0_B0", "C0_B0.yml"),
    ("C1_Aligned", "C1_Aligned.yml"),
    ("C2_WTConv", "C2_WTConv.yml"),
    ("C3_Aligned_WTConv", "C3_Aligned_WTConv.yml"),
]


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def env():
    e = os.environ.copy()
    e.update(
        PYTHONPATH=str(SRC),
        PYTHONUNBUFFERED="1",
        PYTHONIOENCODING="utf-8",
        TEMP=str(ROOT / "tmp"),
        TMP=str(ROOT / "tmp"),
    )
    (ROOT / "tmp").mkdir(exist_ok=True)
    return e


def run_stage(tag, argv, timeout_seconds=43200):
    receipt = ROOT / "results" / f"{tag}_exit.json"
    if receipt.exists():
        old = json.loads(receipt.read_text(encoding="utf-8"))
        if old.get("exit_code") == 0:
            return old
        raise RuntimeError(f"Existing failed stage requires isolated repair: {tag}")
    out = ROOT / "logs" / f"{tag}.stdout.log"
    err = ROOT / "logs" / f"{tag}.stderr.log"
    out.parent.mkdir(exist_ok=True)
    start = time.time()
    child = None
    reason = None
    try:
        with out.open("xb") as so, err.open("xb") as se:
            child = subprocess.Popen(
                [str(x) for x in argv], cwd=ROOT, env=env(), stdout=so, stderr=se
            )
            write(
                ROOT / "results" / "active_process.json",
                {"tag": tag, "pid": child.pid, "started_at": now(), "active": True, "test_used": False},
            )
            while child.poll() is None:
                elapsed = time.time() - start
                if elapsed > timeout_seconds:
                    reason = "timeout"
                    child.terminate()
                    break
                if shutil.disk_usage(ROOT).free < 4 * 2**30:
                    reason = "disk_below4GiB"
                    child.terminate()
                    break
                time.sleep(2)
            if child.poll() is None:
                child.wait(timeout=120)
    finally:
        code = child.returncode if child is not None else None
        result = {
            "tag": tag,
            "exit_code": code,
            "guard_reason": reason,
            "ended_at": now(),
            "command": [str(x) for x in argv],
            "stdout": str(out),
            "stderr": str(err),
            "test_used": False,
        }
        write(receipt, result)
        write(ROOT / "results" / "active_process.json", dict(result, active=False))
    if result["exit_code"] != 0 or reason:
        raise RuntimeError(f"{tag} failed: {result}")
    return result


def train_args(job, smoke=False):
    config = ROOT / "configs" / dict(JOBS)[job]
    save = ROOT / "training" / (("smoke_" if smoke else "") + job)
    return [
        PYTHON,
        SRC / "tools" / "train.py",
        "--config",
        config,
        "--save_dir",
        save,
        "--iters",
        "20" if smoke else "18000",
        "--do_eval",
        "--use_ema",
        "--save_interval",
        "20" if smoke else "1000",
        "--seed",
        "42",
        "--log_iters",
        "10" if smoke else "100",
        "--num_workers",
        "0",
        "--keep_checkpoint_max",
        "1",
        "--precision",
        "fp32",
    ]


def select_checkpoint(job):
    folder = ROOT / "training" / job
    candidates = []
    for variant, subdir, state_name, weight_name in [
        ("ordinary", "best_model", "model.pdstates", "model.pdparams"),
        ("ema", "ema_best_model", "ema_model.pdstates", "ema_model.pdparams"),
    ]:
        state = folder / subdir / state_name
        weight = folder / subdir / weight_name
        if not state.exists() or not weight.exists():
            raise FileNotFoundError(f"Missing {variant} checkpoint for {job}: {state}")
        obj = pickle.load(state.open("rb"))
        candidates.append(
            {
                "variant": variant,
                "mIoU": float(obj["mIoU"]),
                "iter": int(obj["iter"]),
                "weight": str(weight),
                "state": str(state),
                "weight_sha256": sha(weight),
                "state_sha256": sha(state),
            }
        )
    selected = min(candidates, key=lambda x: (-x["mIoU"], x["iter"], x["variant"] != "ordinary"))
    destination = ROOT / "checkpoints" / job
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "model.pdparams"
    shutil.copy2(selected["weight"], checkpoint)
    record = {
        "job": job,
        "candidates": candidates,
        "selected": selected,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "config_sha256": sha(ROOT / "configs" / dict(JOBS)[job]),
        "seed": 42,
        "test_used": False,
    }
    write(destination / "selection.json", record)
    return record


def evaluate(job, selection):
    config = ROOT / "configs" / dict(JOBS)[job]
    checkpoint = selection["checkpoint"]
    run_stage(
        job + "_official_val",
        [PYTHON, SRC / "tools" / "val.py", "--config", config, "--model_path", checkpoint, "--num_workers", "0"],
        1800,
    )
    run_stage(
        job + "_custom_val_exact",
        [
            PYTHON,
            ROOT / "evaluate_custom_boundary.py",
            "--config",
            config,
            "--model_path",
            checkpoint,
            "--output",
            ROOT / "results" / f"{job}_custom_val",
            "--expected_val_path",
            ROOT / "configs" / "val.txt",
            "--expected_count",
            "587",
            "--dataset_name",
            f"custom_in_domain_{job}",
        ],
        1800,
    )
    return json.loads((ROOT / "results" / f"{job}_custom_val.json").read_text(encoding="utf-8"))


def main():
    write(ROOT / "results" / "workflow_state.json", {"stage": "starting", "jobs": [x[0] for x in JOBS], "test_used": False})
    rows = []
    for job, _ in JOBS:
        write(ROOT / "results" / "workflow_state.json", {"stage": "smoke", "job": job, "test_used": False})
        run_stage(job + "_smoke", train_args(job, smoke=True), 1800)
        write(ROOT / "results" / "workflow_state.json", {"stage": "train", "job": job, "test_used": False})
        run_stage(job + "_train", train_args(job, smoke=False), 43200)
        selection = select_checkpoint(job)
        metrics = evaluate(job, selection)
        row = {"job": job, "status": "complete", "selection": selection, "metrics": metrics, "test_used": False}
        rows.append(row)
        write(ROOT / "results" / "comparison.json", {"updated_at": now(), "records": rows, "test_used": False})
    write(ROOT / "results" / "queue_done.json", {"updated_at": now(), "successful_jobs": len(rows), "test_used": False})
    write(ROOT / "results" / "workflow_state.json", {"stage": "complete", "successful_jobs": len(rows), "test_used": False})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write(ROOT / "results" / "workflow_failure.json", {"updated_at": now(), "error": repr(exc), "test_used": False})
        write(ROOT / "results" / "workflow_state.json", {"stage": "failed", "error": repr(exc), "test_used": False})
        raise
