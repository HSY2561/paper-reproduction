#!/usr/bin/env python3
"""Aggregate wave18 evaluator JSON files without starting or changing training.

The evaluator writes one JSON object per validation/test run.  This script only
reads those files, chooses the ordinary/EMA validation result with the larger
*unrounded* ``mIoU``, and writes a fixed 6-dataset x 4-model report.  Missing
evaluator files are represented as ``missing`` rows so that the report can be
generated before a run has completed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


DATASETS = (
    "DeepCrack",
    "CFD",
    "CrackTree260",
    "CrackLS315",
    "CRKWH100",
    "GAPS384",
)
MODELS = (
    "C0_B0",
    "C1_Aligned",
    "C2_WTConv",
    "C3_Aligned_WTConv",
)

# Longest first avoids matching C1_Aligned inside a future, longer model name.
MODEL_ALIASES = {
    "C3_Aligned_WTConv": ("c3_aligned_wtconv", "c3-aligned-wtconv", "c3alignedwtconv"),
    "C2_WTConv": ("c2_wtconv", "c2-wtconv", "c2wtconv"),
    "C1_Aligned": ("c1_aligned", "c1-aligned", "c1aligned"),
    "C0_B0": ("c0_b0", "c0-b0", "c0b0"),
}
DATASET_ALIASES = {
    "DeepCrack": ("deepcrack", "deep_crack"),
    "CFD": ("cfd",),
    "CrackTree260": ("cracktree260", "crack_tree260", "cracktree"),
    "CrackLS315": ("crackls315", "crack_ls315", "crackls"),
    "CRKWH100": ("crkwh100", "crkwh_100", "crkwh", "crackwh100", "crackwh"),
    "GAPS384": ("gaps384", "gaps_384", "gaps"),
}

METRIC_KEYS = (
    "mIoU",
    "crackIoU",
    "precision",
    "recall",
    "f1",
    "boundary_f1_2px",
    "boundary_f1",
    "hard_cldice",
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, Path)):
        return str(value)
    return ""


def _flatten_text(payload: Mapping[str, Any]) -> str:
    """Return text from identity/provenance fields, excluding large metric arrays."""
    fields = (
        "dataset",
        "dataset_name",
        "data_name",
        "id",
        "model",
        "model_id",
        "variant",
        "checkpoint_kind",
        "weight_variant",
        "weights",
        "split",
        "stage",
        "set",
        "evaluation_split",
        "config",
        "model_path",
        "output",
        "validation_list",
        "train_list",
        "test_list",
    )
    values = [_text(payload.get(key)) for key in fields]
    # A few launchers put metadata in a nested ``metadata`` object.
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        values.extend(_text(metadata.get(key)) for key in fields)
    return " ".join(x for x in values if x)


def _metric_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Support both the evaluator's flat schema and ``{"metrics": {...}}``."""
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        # Keep identity/provenance at the top level while allowing metrics to win.
        merged = dict(payload)
        merged.update(metrics)
        return merged
    return payload


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_exact_number(value: Any) -> Optional[Decimal]:
    """Parse a metric for selection without rounding through binary float."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return number if number.is_finite() else None


def _find_dataset(path: Path, payload: Mapping[str, Any]) -> Optional[str]:
    # Explicit dataset fields have precedence over a path token.
    candidates = []
    for key in ("dataset", "dataset_name", "data_name"):
        value = _text(payload.get(key))
        if value:
            candidates.append(value)
    candidates.extend((_flatten_text(payload), str(path)))
    for candidate in candidates:
        normalized = candidate.lower().replace("\\", "/")
        for canonical, aliases in DATASET_ALIASES.items():
            for alias in aliases:
                # Dataset names are separated from neighbouring identifiers by
                # punctuation/path boundaries.  ``s2t`` is intentionally
                # matched as a token rather than as an arbitrary substring.
                pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
                if re.search(pattern, normalized):
                    return canonical
    return None


def _find_model(path: Path, payload: Mapping[str, Any]) -> Optional[str]:
    candidates = []
    for key in ("model", "model_id", "id", "architecture", "config", "model_path"):
        value = _text(payload.get(key))
        if value:
            candidates.append(value)
    candidates.extend((_flatten_text(payload), str(path)))
    for candidate in candidates:
        normalized = candidate.lower().replace("\\", "/")
        for canonical in MODELS:
            for alias in MODEL_ALIASES[canonical]:
                if re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", normalized):
                    return canonical
    return None


def _find_split(path: Path, payload: Mapping[str, Any]) -> str:
    for key in ("split", "evaluation_split", "stage", "set"):
        value = _text(payload.get(key)).lower()
        if value in {"train", "training"}:
            return "train"
        if value in {"val", "valid", "validation", "dev"}:
            return "val"
        if value in {"test", "testing"}:
            return "test"
    text = f"{path.name} {_flatten_text(payload)}".lower()
    if re.search(r"(^|[_\-.])(train|training)([_\-.]|$)", text):
        return "train"
    if re.search(r"(^|[_\-.])(test|testing)([_\-.]|$)", text):
        return "test"
    if re.search(r"(^|[_\-.])(val|valid|validation|dev)([_\-.]|$)", text):
        return "val"
    test_used = payload.get("test_used")
    if isinstance(test_used, bool):
        return "test" if test_used else "val"
    # A metrics JSON without a split marker is generally the validation
    # evaluator; this fallback keeps future evaluator naming compatible.
    return "val"


def _find_variant(path: Path, payload: Mapping[str, Any]) -> str:
    explicit = payload.get("variant")
    if isinstance(explicit, str) and explicit.strip():
        lower = explicit.lower()
        if "ema" in lower:
            return "ema"
        if any(x in lower for x in ("ordinary", "normal", "raw", "base")):
            return "ordinary"
    for key in ("checkpoint_kind", "weight_variant", "weights"):
        value = _text(payload.get(key)).lower()
        if value:
            if "ema" in value:
                return "ema"
            if any(x in value for x in ("ordinary", "normal", "raw", "base")):
                return "ordinary"
    for key in ("ema", "use_ema", "is_ema", "ema_enabled"):
        if key in payload and isinstance(payload[key], bool):
            return "ema" if payload[key] else "ordinary"
    text = f"{path.name} {_flatten_text(payload)}".lower()
    if re.search(r"(^|[_\-.])ema([_\-.]|$)|ema_model", text):
        return "ema"
    if re.search(r"(^|[_\-.])(ordinary|normal|raw)([_\-.]|$)", text):
        return "ordinary"
    # ``best_model/model.pdparams`` is the ordinary checkpoint in PaddleSeg.
    return "ordinary"


def _is_evaluator(payload: Mapping[str, Any]) -> bool:
    merged = _metric_payload(payload)
    return _as_number(merged.get("mIoU")) is not None


def _candidate_payloads(payload: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], Optional[str]]]:
    """Yield direct metrics and wave17-style selection JSON nested metrics."""
    if _is_evaluator(payload):
        return [(payload, None)]
    candidates: list[tuple[Mapping[str, Any], Optional[str]]] = []
    for variant in ("ordinary", "ema"):
        nested = payload.get(variant)
        if isinstance(nested, Mapping) and _is_evaluator(nested):
            merged = dict(payload)
            merged.update(nested)
            # The override makes selection files unambiguous even when their
            # filename is simply ``*_selection.json``.
            merged["variant"] = variant
            candidates.append((merged, variant))
    return candidates


@dataclass(frozen=True)
class Record:
    path: Path
    payload: Mapping[str, Any]
    dataset: str
    model: str
    split: str
    variant: str
    mIoU: Decimal
    priority: tuple[int, int, str] = field(compare=False)

    def metric(self, key: str) -> Optional[float]:
        return _as_number(_metric_payload(self.payload).get(key))


def _record_priority(path: Path, payload: Mapping[str, Any]) -> tuple[int, int, str]:
    """Prefer the boundary evaluator, then a complete metrics file, deterministically."""
    lower = path.name.lower()
    boundary = int("boundary" in lower or "boundary_protocol" in payload)
    # Exit/receipt files are filtered by mIoU, but this keeps future duplicates
    # deterministic if a receipt happens to include metrics.
    receipt_penalty = int(any(x in lower for x in ("_exit", "receipt", "summary", "selection")))
    return (boundary, -receipt_penalty, str(path).lower())


def scan_records(results_dir: Path) -> tuple[list[Record], dict[tuple[str, str], Path], Counter[str], list[str]]:
    counters: Counter[str] = Counter()
    errors: list[str] = []
    records: list[Record] = []
    train_files: dict[tuple[str, str], Path] = {}
    if not results_dir.exists():
        counters["missing_results_dir"] = 1
        return records, train_files, counters, errors
    for path in sorted(results_dir.rglob("*.json")):
        counters["json_files_scanned"] += 1
        try:
            with path.open("r", encoding="utf-8-sig") as stream:
                # Decimal preserves the evaluator's printed mIoU for exact
                # ordinary-vs-EMA ordering; report formatting happens later.
                payload = json.load(stream, parse_float=Decimal)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            counters["malformed_json"] += 1
            errors.append(f"{path}: {exc}")
            continue
        if not isinstance(payload, Mapping):
            counters["ignored_non_object"] += 1
            continue
        candidate_payloads = _candidate_payloads(payload)
        if not candidate_payloads:
            # Training receipts normally contain only exit/status metadata.  A
            # receipt is still useful provenance in the CSV, so retain the
            # best deterministic train path while ignoring it as a metric.
            dataset = _find_dataset(path, payload)
            model = _find_model(path, payload)
            if dataset and model and _find_split(path, payload) == "train":
                key = (dataset, model)
                prior = train_files.get(key)
                if prior is None or _record_priority(path, payload) > _record_priority(prior, {}):
                    train_files[key] = path
                counters["train_receipt"] += 1
            counters["ignored_non_evaluator"] += 1
            continue
        for candidate, variant_override in candidate_payloads:
            dataset = _find_dataset(path, candidate)
            model = _find_model(path, candidate)
            if dataset is None or model is None:
                counters["unidentified_evaluator"] += 1
                missing = ", ".join(x for x, value in (("dataset", dataset), ("model", model)) if value is None)
                errors.append(f"{path}: could not infer {missing}")
                continue
            split = _find_split(path, candidate)
            variant = variant_override or _find_variant(path, candidate)
            miou = _as_exact_number(_metric_payload(candidate).get("mIoU"))
            assert miou is not None
            records.append(
                Record(
                    path=path,
                    payload=candidate,
                    dataset=dataset,
                    model=model,
                    split=split,
                    variant=variant,
                    mIoU=miou,
                    priority=_record_priority(path, candidate),
                )
            )
            counters["evaluator_json"] += 1
    return records, train_files, counters, errors


def _deduplicate(records: Iterable[Record]) -> tuple[dict[tuple[str, str, str, str], Record], Counter[str]]:
    selected: dict[tuple[str, str, str, str], Record] = {}
    duplicate_counts: Counter[str] = Counter()
    for record in records:
        key = (record.dataset, record.model, record.split, record.variant)
        prior = selected.get(key)
        if prior is None or record.priority > prior.priority:
            if prior is not None:
                duplicate_counts["duplicate_replaced"] += 1
            selected[key] = record
        else:
            duplicate_counts["duplicate_ignored"] += 1
    return selected, duplicate_counts


def _fmt(value: Optional[float | Decimal], digits: int = 12) -> str:
    if value is None:
        return ""
    # Keep exact JSON decimals visible for the val selection audit.  Other
    # metrics are floats and use a stable 12-decimal display.
    if isinstance(value, Decimal):
        return format(value, "f")
    return f"{value:.{digits}f}"


def _rel(path: Optional[Path], root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def build_rows(
    root: Path,
    records: list[Record],
    duplicate_counts: Counter[str],
    train_files: Optional[Mapping[tuple[str, str], Path]] = None,
) -> list[dict[str, Any]]:
    by_key, _ = _deduplicate(records)
    train_files = train_files or {}
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for model in MODELS:
            ordinary = by_key.get((dataset, model, "val", "ordinary"))
            ema = by_key.get((dataset, model, "val", "ema"))
            if ordinary and ema:
                # Deliberately compare raw floats before any CSV formatting.
                if ema.mIoU > ordinary.mIoU:
                    chosen_val, reason = ema, "ema_higher"
                elif ordinary.mIoU > ema.mIoU:
                    chosen_val, reason = ordinary, "ordinary_higher"
                else:
                    chosen_val, reason = ordinary, "tie_ordinary"
            elif ema:
                chosen_val, reason = ema, "ema_only"
            elif ordinary:
                chosen_val, reason = ordinary, "ordinary_only"
            else:
                chosen_val, reason = None, "missing_val"

            test_variant = chosen_val.variant if chosen_val else None
            test = by_key.get((dataset, model, "test", test_variant)) if test_variant else None
            test_note = ""
            if chosen_val and test is None:
                # A test evaluator may not encode EMA/ordinary in its filename;
                # use it only as an explicit fallback and disclose the fallback.
                test = by_key.get((dataset, model, "test", "ordinary")) or by_key.get((dataset, model, "test", "ema"))
                if test:
                    test_note = "test_variant_fallback"

            status = "ok" if chosen_val else "missing"
            if chosen_val and test is None:
                status = "val_only"
            row: dict[str, Any] = {
                "dataset": dataset,
                "model": model,
                "status": status,
                "val_variant_selected": chosen_val.variant if chosen_val else "",
                "val_selection_reason": reason,
                "ordinary_val_mIoU": _fmt(ordinary.mIoU if ordinary else None),
                "ema_val_mIoU": _fmt(ema.mIoU if ema else None),
                "selected_val_mIoU": _fmt(chosen_val.mIoU if chosen_val else None),
                "selected_val_crackIoU": _fmt(chosen_val.metric("crackIoU") if chosen_val else None),
                "selected_val_precision": _fmt(chosen_val.metric("precision") if chosen_val else None),
                "selected_val_recall": _fmt(chosen_val.metric("recall") if chosen_val else None),
                "selected_val_f1": _fmt(chosen_val.metric("f1") if chosen_val else None),
                "selected_val_boundary_f1_2px": _fmt(chosen_val.metric("boundary_f1_2px") if chosen_val else None),
                "val_n": chosen_val.payload.get("n", "") if chosen_val else "",
                "test_variant": test.variant if test else "",
                "test_mIoU": _fmt(test.metric("mIoU") if test else None),
                "test_crackIoU": _fmt(test.metric("crackIoU") if test else None),
                "test_precision": _fmt(test.metric("precision") if test else None),
                "test_recall": _fmt(test.metric("recall") if test else None),
                "test_f1": _fmt(test.metric("f1") if test else None),
                "test_boundary_f1_2px": _fmt(test.metric("boundary_f1_2px") if test else None),
                "test_n": test.payload.get("n", "") if test else "",
                "train_json": _rel(train_files.get((dataset, model)), root),
                "val_json": _rel(chosen_val.path if chosen_val else None, root),
                "test_json": _rel(test.path if test else None, root),
                "notes": test_note,
            }
            rows.append(row)
    return rows


CSV_FIELDS = (
    "dataset",
    "model",
    "status",
    "val_variant_selected",
    "val_selection_reason",
    "ordinary_val_mIoU",
    "ema_val_mIoU",
    "selected_val_mIoU",
    "selected_val_crackIoU",
    "selected_val_precision",
    "selected_val_recall",
    "selected_val_f1",
    "selected_val_boundary_f1_2px",
    "val_n",
    "test_variant",
    "test_mIoU",
    "test_crackIoU",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_boundary_f1_2px",
    "test_n",
    "train_json",
    "val_json",
    "test_json",
    "notes",
)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_cell(value: Any) -> str:
    return str(value if value not in (None, "") else "—").replace("|", "\\|").replace("\n", " ")


def write_markdown(
    rows: list[dict[str, Any]],
    path: Path,
    root: Path,
    results_dir: Path,
    counters: Counter[str],
    errors: list[str],
    csv_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    status_counts = Counter(row["status"] for row in rows)
    lines = [
        "# wave18 六数据集 × 四模型汇总",
        "",
        f"生成时间（UTC）：`{generated}`",
        "",
        "本报告由只读汇总脚本生成；脚本不会启动训练，也不会改写 `results/` 中已有 evaluator JSON。",
        f"输入目录：`{results_dir.resolve()}`；CSV：`{_rel(csv_path, root)}`。",
        "",
        "## 选择规则",
        "",
        "对每个数据集/模型组合，直接比较普通（ordinary）与 EMA 验证 evaluator JSON 中的原始 JSON 数值 `mIoU`；较高者作为 `selected_val_mIoU`。完全相等时保留 ordinary，并标记 `tie_ordinary`。测试结果优先使用同一 variant；若测试 JSON 没有 variant 标记，报告中标记 `test_variant_fallback`。",
        "",
        "## 结果",
        "",
        "| 数据集 | 模型 | 状态 | 选中验证权重 | ordinary val mIoU | EMA val mIoU | 选中 val mIoU | 选中 val crackIoU | 选中 val F1 | test mIoU | test F1 | 备注 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(row[key])
                for key in (
                    "dataset",
                    "model",
                    "status",
                    "val_variant_selected",
                    "ordinary_val_mIoU",
                    "ema_val_mIoU",
                    "selected_val_mIoU",
                    "selected_val_crackIoU",
                    "selected_val_f1",
                    "test_mIoU",
                    "test_f1",
                    "notes",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 数据与样本量披露",
            "",
            "- `CRKWH100` 为 `CHCrack5K` 派生数据集/划分；它不应被表述为与 CHCrack5K 完全独立的新数据来源。",
            "- `GAPS384` 的 validation split 仅包含 5 张图像；对应 val mIoU 方差较大，不能按大样本验证集解读。",
            "",
            "## 输入扫描",
            "",
            f"扫描 JSON：{counters.get('json_files_scanned', 0)}；识别 evaluator：{counters.get('evaluator_json', 0)}；忽略非 evaluator：{counters.get('ignored_non_evaluator', 0)}；重复记录忽略/替换：{counters.get('duplicate_ignored', 0)}/{counters.get('duplicate_replaced', 0)}。",
            f"固定输出行数：{len(rows)}（应为 {len(DATASETS) * len(MODELS)}）；状态统计：{dict(status_counts)}。",
        ]
    )
    if errors:
        lines.extend(["", "## 扫描提示", ""])
        for error in errors[:20]:
            escaped_error = error.replace("`", "\\`")
            lines.append(f"- `{escaped_error}`")
        if len(errors) > 20:
            lines.append(f"- 其余 {len(errors) - 20} 条提示未展开。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="wave18 experiment directory (default: directory containing this script)",
    )
    parser.add_argument("--results-dir", type=Path, default=None, help="evaluator JSON directory; defaults to ROOT/results")
    parser.add_argument("--output-dir", type=Path, default=None, help="report directory; defaults to ROOT/reports")
    parser.add_argument("--strict", action="store_true", help="return exit code 2 if any of the 24 rows lacks validation JSON")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    results_dir = (args.results_dir or (root / "results")).expanduser().resolve()
    output_dir = (args.output_dir or (root / "reports")).expanduser().resolve()

    records, train_files, counters, errors = scan_records(results_dir)
    records_by_key, duplicate_counts = _deduplicate(records)
    counters.update(duplicate_counts)
    rows = build_rows(root, records, duplicate_counts, train_files)
    csv_path = output_dir / "wave18_summary.csv"
    markdown_path = output_dir / "wave18_report.md"
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path, root, results_dir, counters, errors, csv_path)

    missing_val = sum(row["status"] == "missing" for row in rows)
    print(f"wave18 summary: {len(records)} evaluator JSON, {len(rows)} rows, {missing_val} missing val rows")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")
    if args.strict and missing_val:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
