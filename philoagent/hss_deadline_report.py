"""Export the explicitly shortened, deadline experiment without altering native logs.

``build_report`` accepts the deadline directory, or its original experiment
parent. Only completed numeric scores are exported as results. Missing work is
listed explicitly; an execution failure is scored zero only by the worker.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import threading
from typing import Any


PROTOCOL_NOTE = "原生候选比赛由压缩学习/反思替代；本报告是截止时间压缩实验，不代表原生完整协议全部完成。"
GROUP_LABELS = {"native": "原生 N", "fixed": "固定哲学 F", "full": "可修订哲学 R"}
DOMAIN_LABELS = {"humanities": "人文", "social": "社会", "business": "商业"}


def _read(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def _write(path: Path, value: Any, *, raw: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    body = value if raw else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, path)


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    except ValueError:
        return None


def _resolve_root(value: str | Path) -> Path:
    root = Path(value).resolve()
    child = root / "deadline_20261030"
    return child if child.is_dir() and not root.name.startswith("deadline_") else root


def _job_specs(root: Path, manifest: dict) -> list[dict]:
    jobs = manifest.get("jobs", [])
    if isinstance(jobs, dict):
        jobs = [{"job_id": key, **value} for key, value in jobs.items()]
    if jobs:
        return jobs
    # The formal matrix is known even if a controller has not yet copied the
    # manifest. This fallback declares coverage, never creates task scores.
    return [
        {"job_id": f"{domain}-{group}-r{repetition}", "domain": domain,
         "group": group, "repetition": repetition}
        for domain in DOMAIN_LABELS for group in GROUP_LABELS for repetition in (1, 2, 3)
    ]


def _aggregate(rows: list[dict]) -> dict:
    heldout = [row["score"] for row in rows if row["phase"] == "heldout"]
    learning = [row["score"] for row in rows if row["phase"] == "learning"]
    model_count = sum("model" in row["score_method"].lower() or
                      "llm" in row["score_method"].lower() for row in rows)
    heuristic_count = sum("heuristic" in row["score_method"].lower() for row in rows)
    return {"scored_tasks": len(rows), "learning_scored": len(learning),
            "heldout_scored": len(heldout), "quality_mean": _mean([row["score"] for row in rows]),
            "learning_mean": _mean(learning), "heldout_mean": _mean(heldout),
            "failed_tasks": sum(row["execution_failed"] for row in rows),
            "valid_work_tasks": sum(not row["execution_failed"] for row in rows),
            "model_scored_tasks": model_count, "heuristic_scored_tasks": heuristic_count,
            "reused_tasks": sum(row["reused"] for row in rows)}


def build_report(root: str | Path) -> dict:
    """Write current JSON/Markdown results and return the complete summary.

    The function is read-only with respect to worker files and original records.
    A partial snapshot always exports successfully and identifies every gap.
    """
    root = _resolve_root(root)
    manifest = _read(root / "manifest.json", {})
    parent_manifest = _read(root.parent / "manifest.json", {})
    # A deadline manifest may contain only controller configuration. Preserve
    # the original matrix's task IDs and ordering when it does not copy jobs.
    matrix = manifest if manifest.get("jobs") else parent_manifest
    specs = _job_specs(root, matrix)
    now = datetime.now(timezone.utc)
    rows: list[dict] = []
    job_summaries: list[dict] = []
    gaps: list[dict] = []
    extras: list[dict] = []
    starts: list[datetime] = []
    finishes: list[datetime] = []

    for spec in specs:
        job_id = spec["job_id"]
        directory = root / "jobs" / job_id
        result = _read(directory / "result.json", {})
        start = _time(result.get("started_utc"))
        finish = _time(result.get("completed_utc"))
        if start:
            starts.append(start)
        if finish:
            finishes.append(finish)
        learning_ids = spec.get("learning_order", [])
        heldout_ids = spec.get("evaluation_order", spec.get("heldout_order", []))
        expected = {str(task_id): "learning" for task_id in learning_ids}
        expected.update({str(task_id): "heldout" for task_id in heldout_ids})
        expected_count = 12
        expected_valid = len(learning_ids) == 8 and len(heldout_ids) == 4 and len(expected) == 12
        job_rows = []
        seen: set[str] = set()
        invalid = []
        for path in sorted((directory / "tasks").glob("*.json")):
            record = _read(path)
            if not isinstance(record, dict):
                invalid.append({"path": str(path), "reason": "unreadable_task_json"})
                continue
            task_id = str(record.get("task_id", path.stem))
            phase = record.get("phase", record.get("stage"))
            if phase == "evaluation":
                phase = "heldout"
            if expected_valid and task_id not in expected:
                extras.append({"job_id": job_id, "task_id": task_id, "path": str(path),
                               "reason": "task_not_in_expected_12"})
                continue
            if task_id in seen:
                invalid.append({"task_id": task_id, "reason": "duplicate_task_id", "path": str(path)})
                continue
            score = record.get("score", record.get("quality_score"))
            if not _number(score) or not 0 <= score <= 100:
                invalid.append({"task_id": task_id, "reason": "missing_or_invalid_numeric_score", "path": str(path)})
                continue
            if expected_valid and phase != expected[task_id]:
                invalid.append({"task_id": task_id, "reason": "phase_does_not_match_manifest", "path": str(path)})
                continue
            if phase not in ("learning", "heldout"):
                invalid.append({"task_id": task_id, "reason": "unknown_phase", "path": str(path)})
                continue
            status = str(record.get("status", ""))
            if status not in ("completed", "complete", "failed_execution", "failed"):
                invalid.append({"task_id": task_id, "reason": "nonterminal_task_status", "path": str(path)})
                continue
            failed = status in ("failed_execution", "failed")
            if failed and score != 0:
                invalid.append({"task_id": task_id, "reason": "failed_execution_requires_explicit_zero", "path": str(path)})
                continue
            seen.add(task_id)
            row = {**record, "job_id": job_id, "domain": spec.get("domain", record.get("domain")),
                   "group": spec.get("group", record.get("group", record.get("arm"))),
                   "repetition": spec.get("repetition", record.get("repetition", record.get("repeat"))),
                   "task_id": task_id, "phase": phase, "score": float(score),
                   "execution_failed": failed, "reused": bool(record.get("reused", False)),
                   "score_method": str(record.get("score_method", "unspecified")),
                   "record_path": str(path), "protocol_changed": True}
            rows.append(row)
            job_rows.append(row)

        missing = sorted(set(expected) - seen) if expected_valid else []
        counts = _aggregate(job_rows)
        coverage_complete = (expected_valid and not missing and len(job_rows) == expected_count
                             and counts["learning_scored"] == 8 and counts["heldout_scored"] == 4)
        job_summaries.append({"job_id": job_id, "domain": spec.get("domain"), "group": spec.get("group"),
                              "repetition": spec.get("repetition"), "worker_status": result.get("status", "pending"),
                              "status": "completed_with_failures" if coverage_complete and counts["failed_tasks"] else
                                        "completed" if coverage_complete else "pending",
                              "complete": coverage_complete, "expected_tasks": expected_count,
                              **counts, "pending_count": max(0, expected_count - len(job_rows)),
                              "pending_task_ids": missing, "expected_task_ids_verified": expected_valid,
                              "invalid_records": invalid})
        if not coverage_complete:
            gaps.append({"job_id": job_id, "pending_count": max(0, expected_count - len(job_rows)),
                         "pending_task_ids": missing, "expected_task_ids_verified": expected_valid,
                         "invalid_records": invalid})

    grouped = []
    for domain in DOMAIN_LABELS:
        for group in GROUP_LABELS:
            selected = [row for row in rows if row["domain"] == domain and row["group"] == group]
            grouped.append({"domain": domain, "group": group, **_aggregate(selected)})
    differences = []
    for domain in DOMAIN_LABELS:
        by_group = {entry["group"]: entry for entry in grouped if entry["domain"] == domain}
        for left, right in (("fixed", "native"), ("full", "native"), ("full", "fixed")):
            a, b = by_group[left]["heldout_mean"], by_group[right]["heldout_mean"]
            left_rows = {(row["repetition"], row["task_id"]): row["score"] for row in rows
                         if row["domain"] == domain and row["group"] == left and row["phase"] == "heldout"}
            right_rows = {(row["repetition"], row["task_id"]): row["score"] for row in rows
                          if row["domain"] == domain and row["group"] == right and row["phase"] == "heldout"}
            common = set(left_rows) & set(right_rows)
            differences.append({"domain": domain, "comparison": f"{left}_minus_{right}",
                                "heldout_mean_difference": round(a - b, 4) if a is not None and b is not None else None,
                                "paired_heldout_count": len(common),
                                "paired_heldout_mean_difference": _mean([left_rows[key] - right_rows[key] for key in common])})

    controller = _read(root / "controller.json", {})
    start = (_time(manifest.get("started_utc")) or _time(manifest.get("start_utc"))
             or _time(controller.get("started_utc")) or (min(starts) if starts else None))
    deadline = (_time(manifest.get("deadline_utc")) or _time(manifest.get("deadline"))
                or _time(controller.get("deadline_utc")))
    complete = len(job_summaries) == 27 and all(job["complete"] for job in job_summaries)
    finish = max(finishes) if complete and len(finishes) == 27 else None
    model_count = sum("model" in row["score_method"].lower() or "llm" in row["score_method"].lower() for row in rows)
    heuristic_count = sum("heuristic" in row["score_method"].lower() for row in rows)
    summary = {
        "schema_version": 1, "mode": "deadline_rapid_v1", "generated_utc": now.isoformat(),
        "root": str(root), "original_experiment_root": str(root.parent),
        "protocol_changed": True, "protocol_note": PROTOCOL_NOTE,
        "original_records_modified": False,
        "status": "completed_with_failures" if complete and any(row["execution_failed"] for row in rows) else
                  "completed" if complete else "in_progress",
        "complete": complete, "expected_jobs": 27, "expected_tasks": 324,
        "expected_learning_tasks": 216, "expected_heldout_tasks": 108,
        "completed_jobs": sum(job["complete"] for job in job_summaries),
        "pending_jobs": sum(not job["complete"] for job in job_summaries),
        "pending_tasks": max(0, 324 - len(rows)), **_aggregate(rows),
        "model_scored_tasks": model_count, "heuristic_scored_tasks": heuristic_count,
        "other_scored_tasks": len(rows) - sum(("model" in row["score_method"].lower() or
                                               "llm" in row["score_method"].lower() or
                                               "heuristic" in row["score_method"].lower()) for row in rows),
        "reused_tasks": sum(row["reused"] for row in rows),
        "newly_executed_tasks": sum(not row["reused"] for row in rows),
        "failed_execution_tasks": sum(row["execution_failed"] for row in rows),
        "started_utc": start.isoformat() if start else None,
        "deadline_utc": deadline.isoformat() if deadline else None,
        "completed_utc": finish.isoformat() if finish else None,
        "elapsed_seconds": round(((finish or now) - start).total_seconds(), 2) if start else None,
        "remaining_seconds": max(0, round((deadline - now).total_seconds(), 2)) if deadline else None,
        "finished_within_deadline": finish <= deadline if finish and deadline else None,
        "by_domain_group": grouped,
        "by_domain": [{"domain": domain, **_aggregate([row for row in rows if row["domain"] == domain])}
                      for domain in DOMAIN_LABELS],
        "by_group": [{"group": group, **_aggregate([row for row in rows if row["group"] == group])}
                     for group in GROUP_LABELS],
        "heldout_differences": differences, "jobs": job_summaries,
        "pending": gaps, "extra_records_excluded": extras,
    }
    _write(root / "results.jsonl", "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), raw=True)
    _write(root / "summary.json", summary)
    _write(root / "preliminary_results.md", _markdown(summary), raw=True)
    return summary


def _markdown(summary: dict) -> str:
    def show(value: Any) -> str:
        return "尚无评分" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)

    lines = ["# 一小时压缩实验初步结果", "", PROTOCOL_NOTE, "",
             f"更新时间：{summary['generated_utc']}；状态：{summary['status']}。",
             f"已完成 {summary['completed_jobs']}/27 条谱系、{summary['scored_tasks']}/324 道答题评分；"
             f"剩余 {summary['pending_tasks']} 道，其中学习题 {summary['learning_scored']}/216、留出题 {summary['heldout_scored']}/108 已评分。", "",
             f"复用 {summary['reused_tasks']} 份，新执行 {summary['newly_executed_tasks']} 份；"
             f"模型评分 {summary['model_scored_tasks']} 份，启发式评分 {summary['heuristic_scored_tasks']} 份，"
             f"其他评分 {summary['other_scored_tasks']} 份；真实执行失败 {summary['failed_execution_tasks']} 份（明确计 0 分）。", "",
             f"已用时 {show(summary['elapsed_seconds'])} 秒；截止时间 {summary['deadline_utc'] or '未设置'}。", "",
             "| 领域 | 组别 | 已评分/36 | 全题均分 | 学习均分 | 留出均分 | 执行失败 | 启发式 | 复用 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for entry in summary["by_domain_group"]:
        lines.append(f"| {DOMAIN_LABELS[entry['domain']]} | {GROUP_LABELS[entry['group']]} | {entry['scored_tasks']}/36 | "
                     f"{show(entry['quality_mean'])} | {show(entry['learning_mean'])} | {show(entry['heldout_mean'])} | {entry['failed_tasks']} | "
                     f"{entry['heuristic_scored_tasks']} | {entry['reused_tasks']} |")
    lines.extend(["", "| 领域 | 留出比较 | 共同题数 | 共同题平均分差 |", "|---|---|---:|---:|"])
    labels = {"fixed_minus_native": "F − N", "full_minus_native": "R − N", "full_minus_fixed": "R − F"}
    for entry in summary["heldout_differences"]:
        lines.append(f"| {DOMAIN_LABELS[entry['domain']]} | {labels[entry['comparison']]} | {entry['paired_heldout_count']} | "
                     f"{show(entry['paired_heldout_mean_difference'])} |")
    lines.extend(["", "学习题和留出题按原任务清单匹配。表中分差使用同一重复、同一道留出题配对；"
                  "启发式与模型评分混用，仅用于本次快速初步观察。原始完整实验记录保留在父目录，未改写。", ""])
    if summary["pending"]:
        lines.extend(["## 明确的未完成项", ""])
        for gap in summary["pending"]:
            identifiers = ", ".join(gap["pending_task_ids"]) or "清单尚未验证"
            lines.append(f"- {gap['job_id']}：剩余 {gap['pending_count']} 道；{identifiers}。")
        lines.append("")
    if summary["failed_execution_tasks"]:
        lines.extend(["执行失败记录可在 results.jsonl 中按 execution_failed=true 检索，保留原 error 和答案；"
                      "0 分表示执行失败，不表示获得有效作品。", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    summary = build_report(arguments.root)
    print(json.dumps({key: summary[key] for key in ("status", "completed_jobs", "scored_tasks", "pending_tasks",
                                                   "failed_execution_tasks", "model_scored_tasks", "heuristic_scored_tasks")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
