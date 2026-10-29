"""Independent, paired evaluation of frozen local agent runtimes.

Only the public HSS rubric is shared with the research implementation.  This
module never opens a formal experiment, its budget, or its supervisor.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time
import unicodedata
import uuid

from philoagent.hss_evaluation import rubric, validate_grade

from .storage import copy_runtime, runtime_fingerprint


def _normalized(value):
    if isinstance(value, str):
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    return value


def public_task_hash(task):
    """Identify the same public task even after its ID/formatting changes."""
    public = {"domain": task.get("domain"), "question": task.get("question"),
              "materials": task.get("materials", []),
              "deliverable": task.get("deliverable", ""),
              "task_criteria": task.get("task_criteria", [])}
    encoded = json.dumps(_normalized(public), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _learning_tasks(workspace):
    for path in sorted((workspace / "operations").glob("*/request.json")):
        try:
            request = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise ValueError("Cannot verify learning/evaluation separation: unreadable operation request") from exc
        if not isinstance(request, dict):
            raise ValueError("Cannot verify learning/evaluation separation: invalid operation request")
        if request.get("operation", request.get("command")) not in ("run", "evolve"):
            continue
        payload = request.get("payload", request)
        if not isinstance(payload, dict):
            raise ValueError("Cannot verify learning/evaluation separation: invalid learning payload")
        tasks = payload.get("tasks", [])
        if "task" in payload:
            tasks = [payload["task"], *(tasks if isinstance(tasks, list) else [])]
        if not isinstance(tasks, list) or any(not isinstance(task, dict) for task in tasks):
            raise ValueError("Cannot verify learning/evaluation separation: invalid learning tasks")
        yield from tasks


def _check_disjoint(workspace, tasks):
    learning = list(_learning_tasks(workspace))
    ids = {_normalized(task["task_id"]) for task in learning if task.get("task_id")}
    hashes = {public_task_hash(task) for task in learning}
    for task in tasks:
        if ((task.get("task_id") and _normalized(task["task_id"]) in ids)
                or public_task_hash(task) in hashes):
            raise ValueError("Evaluation tasks overlap learning history by task_id or public task content")


def _provider(runtime):
    state_path = runtime / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8-sig")) if state_path.exists() else {}
    provider = state.get("provider")
    if not provider:
        provider = json.loads((runtime / "run.json").read_text(encoding="utf-8-sig"))["initial_provider"]
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("A frozen runtime must identify its provider")
    return provider


def _snapshot_evidence(runtime):
    return {"provider": _provider(runtime), "path": str(runtime),
            "runtime_sha256": runtime_fingerprint(runtime)}


def _json_safe(value):
    # Failed judge values can themselves be non-finite. Preserve their identity
    # in valid JSON without ever allowing them into a numeric aggregate.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2,
                                    allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _integer(value):
    return type(value) is int and value >= 0


def _usage(results):
    """Expose partial confirmed totals while leaving an unknown total unknown."""
    usages = [item.get("api_usage") for item in results]
    fields = ("request_count", "prompt_tokens", "completion_tokens", "total_tokens",
              "reserved_unknown_tokens", "unknown_usage_attempts")
    sums = {}
    for field in fields:
        values = [usage[field] for usage in usages
                  if isinstance(usage, dict) and _integer(usage.get(field))]
        sums[field] = sum(values) if values else None
    complete = [isinstance(usage, dict) and usage.get("usage_complete") is True
                and _integer(usage.get("total_tokens"))
                and not usage.get("unknown_usage_attempts")
                and not usage.get("pending_request_count") for usage in usages]
    return {"request_count": sums["request_count"],
            "confirmed_prompt_tokens": sums["prompt_tokens"],
            "confirmed_completion_tokens": sums["completion_tokens"],
            "confirmed_total_tokens": sums["total_tokens"],
            "total_tokens": sums["total_tokens"] if all(complete) and usages else None,
            "reserved_unknown_tokens": sums["reserved_unknown_tokens"],
            "unknown_usage_attempts": sums["unknown_usage_attempts"],
            "usage_complete": bool(usages) and all(complete),
            "unknown_usage_runs": sum(not item for item in complete),
            "runs": len(results)}


def _run_arm(agent, task, provider, directory, runtime):
    started = time.monotonic()
    try:
        raw = agent._invoke("evaluate_task", {"task": deepcopy(task), "provider": provider},
                            operation_dir=directory, runtime_root=runtime)
        if not isinstance(raw, dict):
            raise ValueError("Worker returned no operation result")
        result = deepcopy(raw)
    except Exception as exc:
        result = {"status": "failed", "operation": "evaluate_task", "provider": provider,
                  "error_type": type(exc).__name__, "failure_stage": "worker", "api_usage": None}
    result["elapsed_seconds"] = round(time.monotonic() - started, 6)
    record = result.get("record")
    if not isinstance(result.get("api_usage"), dict) and isinstance(record, dict):
        result["api_usage"] = deepcopy(record.get("api_usage"))
    grade = record.get("grade") if isinstance(record, dict) else None
    result["missing_grade"] = grade is None
    result["validated_grade"] = None
    if result.get("status") == "completed":
        try:
            if isinstance(grade, dict) and "domain" in grade and grade["domain"] != task["domain"]:
                raise ValueError("Saved grade belongs to a different domain")
            validated = validate_grade(grade, task["domain"])
            if isinstance(grade, dict) and "quality_score" in grade and grade["quality_score"] != validated["quality_score"]:
                raise ValueError("Saved total differs from its rubric dimensions")
            result["validated_grade"] = validated
        except (ValueError, TypeError, KeyError):
            result["worker_status"] = result["status"]
            result.update(status="failed", error_type="InvalidGrade", failure_stage="grade_validation")
    _write_json(directory / "evaluation_result.json", result)
    return result


def _mean(values):
    return statistics.mean(values) if values else None


def _summarize_arm(tasks, arm):
    results = [task["arms"][arm] for task in tasks]
    valid = [(task["domain"], task["arms"][arm]["validated_grade"]) for task in tasks
             if task["arms"][arm].get("status") == "completed"
             and task["arms"][arm].get("validated_grade")]
    domains = defaultdict(list)
    for domain, grade in valid:
        domains[domain].append(grade)
    by_domain = {}
    for domain, grades in domains.items():
        by_domain[domain] = {
            "graded_tasks": len(grades),
            "mean_quality": _mean([grade["quality_score"] for grade in grades]),
            "dimension_means": {name: _mean([grade["dimensions"][name]["score"] for grade in grades])
                                for name in rubric(domain)},
        }
    return {"total_tasks": len(results), "completed_tasks": len(valid),
            "failed_tasks": sum(item.get("status") != "completed" for item in results),
            "missing_grade_tasks": sum(item["missing_grade"] for item in results),
            "mean_quality": _mean([grade["quality_score"] for _, grade in valid]),
            "by_domain": by_domain,
            "elapsed_seconds": round(sum(item["elapsed_seconds"] for item in results), 6),
            "api_usage": _usage(results)}


def _paired(tasks):
    pairs = [task for task in tasks if task["paired"]]
    by_domain = {}
    for domain in sorted({task["domain"] for task in tasks}):
        selected = [task for task in pairs if task["domain"] == domain]
        by_domain[domain] = {"paired_tasks": len(selected),
                             "mean_delta": _mean([task["quality_delta"] for task in selected])}
    return {"total_tasks": len(tasks), "paired_tasks": len(pairs),
            "unpaired_tasks": len(tasks) - len(pairs),
            "mean_initial_quality": _mean([task["arms"]["initial"]["validated_grade"]["quality_score"] for task in pairs]),
            "mean_current_quality": _mean([task["arms"]["current"]["validated_grade"]["quality_score"] for task in pairs]),
            "mean_delta": _mean([task["quality_delta"] for task in pairs]), "by_domain": by_domain}


def _markdown(report):
    def display(value):
        return "未知 / 无有效结果" if value is None else str(round(value, 4) if isinstance(value, float) else value)
    lines = ["# 智能体独立评测报告", "", f"状态：{report['status']}；任务数：{report['task_count']}。",
             "每题从相同冻结快照的独立副本运行；评测不摄入记忆、不演化。",
             "连续质量分范围为 0–100；失败和缺失评分不填零。", "",
             "| 快照 | 有效评分 / 总题数 | 平均质量 | 确认 tokens | 用量未知的运行 |",
             "| --- | --- | --- | --- | --- |"]
    for arm, summary in report["arms"].items():
        lines.append(f"| {arm} | {summary['completed_tasks']} / {summary['total_tasks']} | "
                     f"{display(summary['mean_quality'])} | {display(summary['api_usage']['confirmed_total_tokens'])} | "
                     f"{summary['api_usage']['unknown_usage_runs']} |")
    if report["paired"] is not None:
        paired = report["paired"]
        lines.extend(["", f"配对成功：{paired['paired_tasks']} / {paired['total_tasks']}；"
                      f"配对平均差值（当前 − 初始）：{display(paired['mean_delta'])}。",
                      "差值仅包含双方均完成且评分合法的同题配对；单侧有效结果仍保留在分组均值中。"])
    lines.extend(["", "| 任务 | 领域 | initial | current | 差值 |", "| --- | --- | --- | --- | --- |"])
    for task in report["tasks"]:
        cells = []
        for arm in ("initial", "current"):
            result = task["arms"].get(arm)
            if result is None:
                cells.append("未启用")
            elif result.get("validated_grade"):
                cells.append(display(result["validated_grade"]["quality_score"]))
            else:
                cells.append(f"失败：{result.get('failure_stage', 'worker')} / {result.get('error_type', 'UnknownError')}")
        identifier = str(task["task_id"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {identifier} | {task['domain']} | {cells[0]} | {cells[1]} | {display(task['quality_delta'])} |")
    lines.extend(["", "完整逐题作品、评分依据、失败状态和原始用量保存在同目录 report.json。",
                  "这是产品能力评测，不是当前正式实验的新增结果，也不据此自动宣称统计显著提升。", ""])
    return "\n".join(lines)


def evaluate(agent, tasks, output_dir, compare_initial=True):
    """Evaluate normalized tasks while the caller holds the agent workspace lock."""
    tasks = deepcopy(list(tasks))
    if not tasks:
        raise ValueError("Evaluation requires at least one task")
    workspace = Path(agent.workspace).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    _check_disjoint(workspace, tasks)
    seen_ids, seen_hashes = set(), set()
    for index, task in enumerate(tasks, 1):
        rubric(task.get("domain"))
        if not isinstance(task.get("question"), str) or not task["question"].strip():
            raise ValueError("Evaluation requires a nonempty public question")
        identifier = task.get("task_id", f"task-{index}")
        content = public_task_hash(task)
        if identifier in seen_ids or content in seen_hashes:
            raise ValueError("Evaluation tasks must have distinct IDs and public task content")
        seen_ids.add(identifier)
        seen_hashes.add(content)
        task["task_id"] = identifier
    if (output_dir / "report.json").exists() or any((output_dir / f"frozen-{arm}").exists()
                                                   for arm in ("current", "initial")):
        raise FileExistsError("Evaluation output already contains a frozen run; choose a fresh output directory")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    arms = ("initial", "current") if compare_initial else ("current",)
    snapshots = {}
    for arm in arms:
        source = workspace / "initial" if arm == "initial" else workspace
        destination = output_dir / f"frozen-{arm}"
        source_sha256 = runtime_fingerprint(source)
        copy_runtime(source, destination)
        snapshots[arm] = _snapshot_evidence(destination)
        snapshots[arm]["source_runtime_sha256"] = source_sha256
    records = []
    for index, task in enumerate(tasks, 1):
        entry = {"task_id": task["task_id"], "domain": task["domain"],
                 "public_task_sha256": public_task_hash(task), "arms": {},
                 "paired": False, "quality_delta": None}
        # Counterbalance order; each arm remains an entirely separate process.
        order = arms if index % 2 else tuple(reversed(arms))
        for arm in order:
            directory = output_dir / "tasks" / f"{index:04d}" / arm
            runtime = directory / "runtime"
            try:
                copy_runtime(Path(snapshots[arm]["path"]), runtime)
                # Budget receipts use operation_id globally across all forks.
                # A fresh ID prevents later evaluations from claiming old calls.
                worker_operation = runtime / "operations" / f"eval-{arm}-{uuid.uuid4().hex}"
                result = _run_arm(agent, task, snapshots[arm]["provider"], worker_operation, runtime)
            except Exception as exc:
                result = {"status": "failed", "operation": "evaluate_task",
                          "provider": snapshots[arm]["provider"], "error_type": type(exc).__name__,
                          "failure_stage": "runtime_copy", "api_usage": None,
                          "missing_grade": True, "validated_grade": None, "elapsed_seconds": 0.0}
                _write_json(directory / "evaluation_result.json", result)
            entry["arms"][arm] = result
        if compare_initial and all(entry["arms"][arm].get("validated_grade")
                                   and entry["arms"][arm].get("status") == "completed" for arm in arms):
            entry["paired"] = True
            entry["quality_delta"] = round(entry["arms"]["current"]["validated_grade"]["quality_score"]
                                           - entry["arms"]["initial"]["validated_grade"]["quality_score"], 6)
        records.append(entry)
    for arm in arms:
        source = workspace / "initial" if arm == "initial" else workspace
        snapshots[arm]["source_runtime_preserved"] = (
            runtime_fingerprint(source) == snapshots[arm]["source_runtime_sha256"])
        if not snapshots[arm]["source_runtime_preserved"]:
            raise RuntimeError("Independent evaluation changed its source runtime")
    summaries = {arm: _summarize_arm(records, arm) for arm in arms}
    report = {"schema_version": 1, "kind": "independent_agent_evaluation",
              "status": "complete" if all(summary["failed_tasks"] == 0 for summary in summaries.values()) else "incomplete",
              "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": round(time.monotonic() - started, 6),
              "compare_initial": bool(compare_initial), "task_count": len(tasks),
              "quality_metric": "continuous_0_to_100", "heldout": True,
              "heldout_ingestion": False, "independent_snapshot_per_task": True,
              "snapshots": snapshots, "arms": summaries, "paired": _paired(records) if compare_initial else None,
              "tasks": records, "api_usage": _usage([result for task in records for result in task["arms"].values()]),
              "report_json": str(output_dir / "report.json"), "report_markdown": str(output_dir / "report.md")}
    report = _json_safe(report)
    _write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report
