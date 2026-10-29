"""Read-only cost analysis of the fixed interim snapshot; no experiment imports."""
import collections
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "hss-business-interim2-20261029.snapshot.json"
OUTPUT = HERE / "hss-business-interim2-cost-20261029.json"
snapshot = json.loads(SOURCE.read_text(encoding="utf-8"))
attempts = snapshot["attempt_min"]
budget = snapshot["budget"]
jobs = {row["job_id"]: row for row in snapshot["manifest"]["jobs"]}


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return round(ordered[lower] * (upper - index) + ordered[upper] * (index - lower), 6) if lower != upper else round(ordered[lower], 6)


def summarize(rows):
    result = {
        "settled_requests": len(rows),
        "successful_requests": sum(row.get("status") == "success" for row in rows),
        "failed_requests": sum(row.get("status") != "success" for row in rows),
        "known_usage_requests": sum(isinstance(row.get("usage"), dict) for row in rows),
        "unknown_usage_requests": sum(not isinstance(row.get("usage"), dict) for row in rows),
        "reserved_unknown_tokens": sum(row.get("reserved_unknown_tokens", 0) for row in rows),
    }
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        result[field] = sum((row.get("usage") or {}).get(field, 0) for row in rows)
    result["confirmed_token_share_percent"] = round(100 * result["total_tokens"] / budget["actual_tokens"], 6)
    result["input_token_percent"] = round(100 * result["prompt_tokens"] / result["total_tokens"], 6) if result["total_tokens"] else None
    result["confirmed_tokens_per_settled_request"] = round(result["total_tokens"] / len(rows), 6) if rows else None
    duration = [row["duration_seconds"] for row in rows if isinstance(row.get("duration_seconds"), (int, float))]
    result["sdk_dispatch_duration_seconds"] = {
        "n": len(duration),
        "sum": round(sum(duration), 6),
        "mean": round(sum(duration) / len(duration), 6) if duration else None,
        "p50": quantile(duration, 0.5),
        "p90": quantile(duration, 0.9),
        "p95": quantile(duration, 0.95),
        "max": max(duration) if duration else None,
    }
    return result


def group_by(key):
    groups = collections.defaultdict(list)
    for row in attempts:
        groups[key(row)].append(row)
    return {name: summarize(rows) for name, rows in sorted(groups.items())}


completed_ids = set()
other_ids = set()
for row in snapshot["records"]:
    record = row["record"]
    identifiers = set(record.get("task_execution_ids", []))
    if record.get("task_execution_id"):
        identifiers.add(record["task_execution_id"])
    (completed_ids if record.get("status") == "completed" else other_ids).update(identifiers)
other_ids -= completed_ids


def attribution(row):
    identifier = row.get("task_execution_id")
    if not identifier:
        return "native_evolution_or_other_without_task_execution_id"
    if identifier in completed_ids:
        return "linked_to_active_completed_task_including_that_task_recovery"
    if identifier in other_ids:
        return "linked_to_active_incomplete_task"
    return "other_task_execution_not_present_in_active_records_not_assumed_waste"


known_invalid_execution_ids = {
    "865647581e594f11b4f63d6e0ed63673": "social-native-r1 old WB0133 after invalidated WB0682 feedback",
    "a498b6533d694ec0aec14f811fe3ed45": "social-fixed-r1 old WB0133 after invalidated WB0682 feedback",
    "f208823d721d4a009f398bf072eed28d": "humanities-fixed-r1 incomplete Aether TEXT-interface failure before controlled retry",
}
invalid_rows = [row for row in attempts if row.get("task_execution_id") in known_invalid_execution_ids]
error_rows = [row for row in attempts if row.get("status") != "success"]
lower_bound = {row["attempt"]: row for row in invalid_rows + error_rows}
retry_rows = [row for row in attempts if row.get("recovery_index", 0) > 0 or row.get("retry_of_attempt") is not None]

result = {
    "source": str(SOURCE),
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "captured_start_utc": snapshot["captured_start_utc"],
    "captured_end_utc": snapshot["captured_end_utc"],
    "all_settled": summarize(attempts),
    "budget": {
        "attempt_count_including_pending": budget["attempt_count"],
        "pending_requests": len(budget["pending"]),
        "failed_requests": budget["failed_requests"],
        "confirmed_tokens": budget["actual_tokens"],
        "reserved_unknown_tokens": budget["reserved_unknown_tokens"],
        "effective_tokens": budget["actual_tokens"] + budget["reserved_unknown_tokens"],
        "effective_token_budget_percent": round(100 * (budget["actual_tokens"] + budget["reserved_unknown_tokens"]) / budget["limits"]["max_tokens"], 6),
        "call_budget_percent": round(100 * budget["attempt_count"] / budget["limits"]["max_calls"], 6),
        "sdk_failure_percent_of_settled": round(100 * budget["failed_requests"] / len(attempts), 6),
        "blocked_reason": budget["blocked_reason"],
    },
    "by_role": group_by(lambda row: row.get("role", "unscoped")),
    "by_group": group_by(lambda row: jobs[row["job_id"]]["group"]),
    "by_domain": group_by(lambda row: jobs[row["job_id"]]["domain"]),
    "by_job": group_by(lambda row: row["job_id"]),
    "by_stage": group_by(lambda row: row.get("stage", "unscoped")),
    "by_artifact_attribution": group_by(attribution),
    "sdk_error_cost": summarize(error_rows),
    "sdk_errors": [{key: row.get(key) for key in ("attempt", "job_id", "role", "status", "error_type", "usage", "reserved_unknown_tokens", "duration_seconds")} for row in error_rows],
    "explicit_transport_retry_requests": summarize(retry_rows),
    "explicit_transport_retry_attempt_ids": [row["attempt"] for row in retry_rows],
    "known_invalid_task_execution_cost": {
        identifier: {"reason": reason, **summarize([row for row in attempts if row.get("task_execution_id") == identifier])}
        for identifier, reason in known_invalid_execution_ids.items()
    },
    "confirmed_historical_error_or_invalid_execution_lower_bound_union": summarize(list(lower_bound.values())),
    "lower_bound_attempt_ids": sorted(lower_bound),
    "method_notes": [
        "The budget boundary excludes six pending requests from attempt_min; all settled confirmed usage reconciles exactly.",
        "Durations measure SDK dispatch through completion; they exclude time waiting for global request slots and do not measure whole task duration.",
        "Percentiles use linearly interpolated order statistics (equivalent to inclusive quantiles). Five interrupted unknown requests have no SDK duration.",
        "Unknown reservations are conservative protection, not confirmed provider usage. Pending usage is not yet known.",
        "Active-completed linkage includes earlier executions retained in task_execution_ids and therefore includes recovery cost; it is not a clean no-repair treatment cost.",
        "Missing active task linkage does not mean waste; it includes historical artifacts, candidate operations, summaries and native evolution.",
        "The historical error/invalid lower bound counts each attempt once. It does not estimate all repair overhead, and it does not discard reused actor answers.",
        "Totals by group/domain are not causal efficiency comparisons because active lineage counts, progress, candidate architectures and recovery differ.",
        "No model pricing or completion ETA is inferred.",
    ],
}
assert result["all_settled"]["total_tokens"] == budget["actual_tokens"]
assert len(attempts) + len(budget["pending"]) == budget["attempt_count"]
assert result["all_settled"]["successful_requests"] == budget["successful_requests"]
assert result["all_settled"]["failed_requests"] == budget["failed_requests"]
OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(OUTPUT), "all_settled": result["all_settled"], "budget": result["budget"], "by_role": result["by_role"], "by_group": result["by_group"], "by_domain": result["by_domain"], "attribution": result["by_artifact_attribution"], "sdk_error_cost": result["sdk_error_cost"], "retry": result["explicit_transport_retry_requests"], "historical_lower_bound": result["confirmed_historical_error_or_invalid_execution_lower_bound_union"]}, ensure_ascii=False))
