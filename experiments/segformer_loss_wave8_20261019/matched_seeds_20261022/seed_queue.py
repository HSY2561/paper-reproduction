"""Matched seed 43/44 verification for the two loss recipes that passed seed42.

This runner reuses the official PaddleSeg trainer and evaluator.  The source
snapshot is the Dice compatibility clone; the only source delta in that clone
is labels.clone() inside MixedLoss, which isolates the official in-place
ignore-label side effect.  No test or custom-data list is read here.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
RECOVERY = ROOT.parent / "dice_compat_recovery_20261020"
WAVE6 = ROOT.parent.parent / "segformer_balance_wave6_20261016"
sys.path.insert(0, str(WAVE6))
spec = importlib.util.spec_from_file_location("wave6_queue", WAVE6 / "experiment_queue.py")
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
q.ROOT = ROOT
q.SRC = RECOVERY / "src/PaddleSeg-2.10.0"
q.__file__ = str(__file__)
read, write, sha = q.read_json, q.write_json, q.sha
FIELDS = ("precision", "recall", "f1", "crackIoU", "mIoU", "boundary_f1")
JOBS = [
    ("L01_CE_Dice", "L01_CE_Dice", 43),
    ("L01_CE_Dice", "L01_CE_Dice", 44),
    ("L02_CE_Dice", "L02_CE_Dice", 43),
    ("L02_CE_Dice", "L02_CE_Dice", 44),
]


def now():
    return q.now()


def ident_for(recipe, seed):
    return f"{recipe}_s{seed}"


def env():
    values = dict(os.environ)
    values.update(
        PYTHONPATH=str(q.SRC),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONIOENCODING="utf-8",
        PYTHONUNBUFFERED="1",
        TEMP=str(ROOT / "tmp"),
        TMP=str(ROOT / "tmp"),
        CUDA_CACHE_PATH=str(ROOT / "tmp/cuda_cache"),
    )
    return values


def status(ident, value, **kwargs):
    write(ROOT / "results" / f"{ident}_status.json", dict(status=value, updated_at=now(), **kwargs))


def run_stage(args, tag, timeout=43200):
    log = ROOT / "logs" / f"{tag}.log"
    command = [str(q.PYTHON)] + [str(a) for a in args]
    started = time.time()
    with log.open("wb") as stream:
        child = subprocess.Popen(command, cwd=str(ROOT), env=env(), stdout=stream, stderr=subprocess.STDOUT)
        write(ROOT / "results/active_process.json", dict(tag=tag, pid=child.pid, runner_pid=os.getpid(),
            started_at=now(), command=command, log=str(log), test_used=False))
        last_size = -1
        last_change = started
        last_heartbeat = 0
        stop_reason = None
        while child.poll() is None:
            elapsed = time.time() - started
            size = log.stat().st_size if log.exists() else 0
            if size != last_size:
                last_size = size
                last_change = time.time()
            free = shutil.disk_usage(ROOT).free / (2 ** 30)
            if free < 4:
                stop_reason = "disk_below_4GiB"
            elif elapsed > timeout:
                stop_reason = "stage_timeout"
            elif time.time() - last_change > 1800:
                stop_reason = "log_stale_over_30min"
            if stop_reason:
                child.terminate()
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=30)
                break
            if time.time() - last_heartbeat > 20:
                tail = log.read_text(encoding="utf-8", errors="replace")[-20000:] if log.exists() else ""
                matches = re.findall(r"iter:\s*(\d+)/18000", tail)
                write(ROOT / "results/heartbeat.json", dict(tag=tag, pid=child.pid, runner_pid=os.getpid(),
                    timestamp=now(), latest_iter=int(matches[-1]) if matches else None,
                    elapsed_seconds=round(elapsed, 1), log_size=size, free_gb=round(free, 2)))
                last_heartbeat = time.time()
            time.sleep(1)
        code = child.returncode
    write(ROOT / "results" / f"{tag}_exit.json", dict(exit_code=code, stop_reason=stop_reason,
        elapsed_seconds=round(time.time() - started, 2), ended_at=now(), command=command, test_used=False))
    print(tag, "exit", code, "reason", stop_reason, flush=True)
    if code != 0 or stop_reason:
        raise RuntimeError(f"{tag}: exit={code}, reason={stop_reason}")
    return code


def prepare():
    import yaml
    for name in ("configs", "results", "provenance", "logs", "training", "checkpoints", "tmp", "refine-logs"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    jobs = []
    for recipe, source, seed in JOBS:
        ident = ident_for(recipe, seed)
        cfg = yaml.safe_load((RECOVERY / "configs" / f"{source}.yml").read_text(encoding="utf-8-sig"))
        (ROOT / "configs" / f"{ident}.yml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        jobs.append(dict(id=ident, recipe=recipe, source=source, seed=seed, modules=cfg["model"].get("balance_modules", [])))
    shutil.copy2(RECOVERY / "evaluate_val_boundary.py", ROOT / "evaluate_val_boundary.py")
    parent = read(RECOVERY / "results" / "consolidated_comparison.json")
    write(ROOT / "provenance/jobs.json", jobs)
    write(ROOT / "provenance/parent_candidates.json", {
        "L01": next(r for r in parent["rows"] if r["id"] == "L01_CE_Dice"),
        "L02": next(r for r in parent["rows"] if r["id"] == "L02_CE_Dice"),
    })
    write(ROOT / "provenance/runtime.json", dict(
        created_at=now(), parent=str(RECOVERY), source=str(q.SRC), python=str(q.PYTHON),
        seeds=[43, 44], recipes=["L01_CE_Dice", "L02_CE_Dice"], iters=18000, batch=4,
        test_used=False, selection="ordinary/EMA maximum exact val mIoU, then independent val and boundary"))
    write(ROOT / "provenance/source_hashes.json", {str(p.relative_to(q.SRC)): sha(p) for p in (q.SRC / "paddleseg").rglob("*.py")})
    write(ROOT / "provenance/config_hashes.json", {p.name: sha(p) for p in (ROOT / "configs").glob("*.yml")})
    write(ROOT / "provenance/data_manifest_reference.json", read(RECOVERY / "provenance" / "data_before.json"))
    report()
    print("PREPARED_MATCHED_SEEDS", len(jobs), flush=True)


def train_args(ident, smoke=False, seed=42):
    return [q.SRC / "tools/train.py", "--config", ROOT / "configs" / f"{ident}.yml",
            "--save_dir", ROOT / "training" / (("smoke_" if smoke else "") + ident),
            "--iters", "20" if smoke else "18000", "--do_eval", "--use_ema",
            "--save_interval", "20" if smoke else "1000", "--seed", str(seed),
            "--log_iters", "10" if smoke else "100", "--num_workers", "0",
            "--keep_checkpoint_max", "1", "--precision", "fp32"]


def audit_data():
    manifest = read(RECOVERY / "provenance" / "data_before.json")
    if manifest.get("mismatches"):
        raise RuntimeError("Recovery parent train/val hash witness already had mismatches")
    write(ROOT / "provenance/data_before.json", dict(manifest, checked_at=now(), test_used=False))


def verify_static():
    for name, expected in read(ROOT / "provenance/config_hashes.json").items():
        if sha(ROOT / "configs" / name) != expected:
            raise RuntimeError("Config changed: " + name)
    for relative, expected in read(ROOT / "provenance/source_hashes.json").items():
        if sha(q.SRC / relative) != expected:
            raise RuntimeError("Source changed: " + relative)
    audit_data()
    for recipe in ("L01_CE_Dice", "L02_CE_Dice"):
        recovered = RECOVERY / "checkpoints" / recipe / "model.pdparams"
        selection = read(RECOVERY / "checkpoints" / recipe / "selection.json")
        if sha(recovered) != selection["sha256"]:
            raise RuntimeError("Seed42 recovered checkpoint changed: " + recipe)


def choose_checkpoint(ident):
    folder = ROOT / "training" / ident
    choices = []
    import pickle
    for variant, state_name, weight_name in (("best_model", "model.pdstates", "model.pdparams"),
                                              ("ema_best_model", "ema_model.pdstates", "ema_model.pdparams")):
        with (folder / variant / state_name).open("rb") as stream:
            state = pickle.load(stream)
        choices.append(dict(variant=variant, mIoU=float(state["mIoU"]), iter=int(state["iter"]),
                            weight=str(folder / variant / weight_name), state_sha256=sha(folder / variant / state_name)))
    selected = max(choices, key=lambda r: r["mIoU"])
    dst = ROOT / "checkpoints" / ident
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected["weight"], dst / "model.pdparams")
    record = dict(model=ident, seed=int(ident.rsplit("s", 1)[1]), candidates=choices, selected=selected,
                  checkpoint=str(dst / "model.pdparams"), sha256=sha(dst / "model.pdparams"),
                  selection="highest exact validation mIoU from ordinary and EMA state", test_used=False, frozen_at=now())
    write(dst / "selection.json", record)
    return record


def evaluate(ident, checkpoint):
    run_stage([q.SRC / "tools/val.py", "--config", ROOT / "configs" / f"{ident}.yml", "--model_path", checkpoint,
               "--num_workers", "0"], ident + "_official_val", 1800)
    run_stage([ROOT / "evaluate_val_boundary.py", "--config", ROOT / "configs" / f"{ident}.yml", "--model_path", checkpoint,
               "--output", ROOT / "results" / (ident + "_val")], ident + "_pixel_boundary", 1800)
    # q.profile is an official model-forward profile and does not read test.
    run_stage([__file__, "--profile", ident, "--weights", checkpoint], ident + "_profile", 1800)


def profile(ident, weights):
    # Use the shared official profile implementation with this queue's root/source.
    q.profile(ident, weights)


def report():
    jobs_path = ROOT / "provenance/jobs.json"
    if not jobs_path.exists():
        return
    parent = read(RECOVERY / "results" / "consolidated_comparison.json")
    parent_by = {r["id"]: r for r in parent["rows"] if r["id"] in ("L01_CE_Dice", "L02_CE_Dice")}
    rows = []
    for job in read(jobs_path):
        ident = job["id"]
        state_path = ROOT / "results" / f"{ident}_status.json"
        row = dict(job, status=read(state_path)["status"] if state_path.exists() else "pending")
        if row["status"] == "complete" and (ROOT / "results" / f"{ident}_val.json").exists():
            metrics = read(ROOT / "results" / f"{ident}_val.json")
            profile_data = read(ROOT / "results" / f"{ident}_profile.json")
            row.update({k: metrics[k] for k in FIELDS}, params=profile_data["params_total"],
                       boundary_protocol=metrics.get("boundary_protocol"))
            row["delta_vs_seed42"] = {k: metrics[k] - parent_by[job["recipe"]][k] for k in FIELDS}
            row["main_gate_against_seed42"] = (
                row["delta_vs_seed42"]["crackIoU"] >= .002 and row["delta_vs_seed42"]["f1"] >= .001 and
                row["delta_vs_seed42"]["mIoU"] >= 0 and all(row["delta_vs_seed42"][k] >= -.005 for k in ("precision", "recall", "boundary_f1")))
        rows.append(row)
    with (ROOT / "results/seed_runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "recipe", "seed", "status", *FIELDS, "params", "main_gate_against_seed42"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    groups = {}
    for recipe in ("L01_CE_Dice", "L02_CE_Dice"):
        values = [{"seed": 42, **{k: parent_by[recipe][k] for k in FIELDS}}]
        for row in rows:
            if row["recipe"] == recipe and row["status"] == "complete":
                values.append({"seed": row["seed"], **{k: row[k] for k in FIELDS}})
        group = {"n": len(values), "seeds": values}
        if len(values) == 3:
            import statistics
            group["mean"] = {k: statistics.mean(v[k] for v in values) for k in FIELDS}
            group["sample_std"] = {k: statistics.stdev(v[k] for v in values) for k in FIELDS}
            group["mean_delta_vs_seed42"] = {k: group["mean"][k] - values[0][k] for k in FIELDS}
        groups[recipe] = group
    write(ROOT / "results/three_seed_summary.json", dict(groups=groups, rows=rows, test_used=False, updated_at=now()))
    lines = ["# Dice兼容恢复候选匹配 seed43/44", "", "只使用Crack500 train/val；没有访问test或自建数据。", "",
             "| 运行 | 配方 | seed | 状态 | P% | R% | F1% | 裂缝IoU% | mIoU% | BF1% | Params M |", "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if "mIoU" in row:
            numbers = " | ".join(f"{row[k]*100:.4f}" for k in FIELDS) + f" | {row['params']/1e6:.6f}"
        else:
            numbers = "— | — | — | — | — | — | —"
        lines.append(f"| {row['id']} | {row['recipe']} | {row['seed']} | {row['status']} | {numbers} |")
    lines += ["", "seed42 来自兼容恢复独立验证；完整20项合并表仍保留原始失败与compat_clone证据。", "", "" ]
    (ROOT / "refine-logs/RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(groups, ensure_ascii=False, indent=2), flush=True)


def run():
    verify_static()
    lock = ROOT / "results/queue.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), started_at=now(), test_used=False), stream)
    try:
        for job in read(ROOT / "provenance/jobs.json"):
            ident = job["id"]
            state_path = ROOT / "results" / f"{ident}_status.json"
            if state_path.exists() and read(state_path)["status"] == "complete":
                continue
            try:
                for smoke in (True, False):
                    stage = ident + ("_smoke" if smoke else "_train")
                    receipt = ROOT / "results" / f"{stage}_exit.json"
                    if receipt.exists():
                        if read(receipt)["exit_code"] != 0:
                            raise RuntimeError("preserved nonzero stage " + stage)
                        continue
                    target = ROOT / "training" / (("smoke_" if smoke else "") + ident)
                    if target.exists():
                        raise RuntimeError("unfinished training directory requires audit: " + str(target))
                    status(ident, "smoke_running" if smoke else "training")
                    run_stage(train_args(ident, smoke, job["seed"]), stage, 3600 if smoke else 43200)
                final_dir = ROOT / "training" / ident / "iter_18000"
                train_log = ROOT / "logs" / f"{ident}_train.log"
                if not (final_dir / "model.pdparams").exists() or "iter: 18000/18000" not in train_log.read_text(encoding="utf-8", errors="replace"):
                    raise RuntimeError("missing full 18k evidence")
                selection = choose_checkpoint(ident)
                status(ident, "verifying", selection=selection)
                evaluate(ident, selection["checkpoint"])
                metric = read(ROOT / "results" / f"{ident}_val.json")
                prof = read(ROOT / "results" / f"{ident}_profile.json")
                if abs(metric["mIoU"] - selection["selected"]["mIoU"]) > 1e-6:
                    raise RuntimeError("independent validation differs from checkpoint selection")
                if prof["params_total"] != 3470434:
                    raise RuntimeError("parameter count changed")
                status(ident, "complete", selection=selection)
            except Exception as error:
                status(ident, "needs_review", error=str(error), traceback=traceback.format_exc())
                report()
                if "disk_below" in str(error) or "log_stale" in str(error):
                    raise
            report()
        audit_data()
        report()
        states = {j["id"]: read(ROOT / "results" / f"{j['id']}_status.json")["status"] for j in read(ROOT / "provenance/jobs.json")}
        write(ROOT / "results/queue_done.json", dict(states=states, complete=sum(v == "complete" for v in states.values()), ended_at=now(), test_used=False,
            next_step="audit three-seed mean gates before original public model/test phase"))
    finally:
        (ROOT / "results/queue.lock").unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--profile")
    parser.add_argument("--weights")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.run:
        run()
    if args.report:
        report()
    if args.profile:
        profile(args.profile, args.weights)
