"""Resume a stopped decision analysis by exact replay and audited prompt compression.

Only successful analysis calls 19 and 20 are replayed. Their messages and model
must match exactly. Subsequent requests keep the existing budget and replace
only the full frozen provider source with its identity. Cached trajectory
summaries must already exist; another summary request is refused.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .cli import PROVIDER, environment
from .live_smoke import activate_workspace, write_json
from .orientation import load
from .smoke_budget import Budget, wrap_native_model


class ReplayRefused(RuntimeError):
    """Resume cannot prove equivalence; no replacement request is permitted."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def compress_messages(value, source, replacement):
    """Recursively replace only exact provider-source substrings; return count."""
    if not source:
        raise ValueError("An empty source cannot be compressed")
    if isinstance(value, str):
        return value.replace(source, replacement), value.count(source)
    if isinstance(value, list):
        transformed = [compress_messages(item, source, replacement) for item in value]
        return [item for item, _ in transformed], sum(count for _, count in transformed)
    if isinstance(value, tuple):
        transformed = [compress_messages(item, source, replacement) for item in value]
        return tuple(item for item, _ in transformed), sum(count for _, count in transformed)
    if isinstance(value, dict):
        transformed = {key: compress_messages(item, source, replacement) for key, item in value.items()}
        return {key: item for key, (item, _) in transformed.items()}, sum(count for _, count in transformed.values())
    return value, 0


def restored_completion(audit):
    """Reconstruct a typed SDK response without changing paid usage or content."""
    from openai.types.chat import ChatCompletion

    response = audit.get("response") or {}
    choices = response.get("choices")
    usage = audit.get("usage")
    if audit.get("status") != "success" or not choices or not isinstance(usage, dict):
        raise ReplayRefused("Cached call is not a successful response with usage")
    return ChatCompletion.model_validate({
        "id": "offline-replay-attempt-" + str(audit["attempt"]),
        "object": "chat.completion", "created": 0,
        "model": audit["request"]["model"], "usage": usage,
        "choices": [{**deepcopy(choice), "index": index} for index, choice in enumerate(choices)],
    })


class ReplayState:
    def __init__(self, audits, source, source_sha256, event_path):
        if [item.get("attempt") for item in audits] != [19, 20]:
            raise ReplayRefused("Resume requires exactly saved attempts 19 and 20")
        if any(item.get("label") != "decision-evolve-analysis" for item in audits):
            raise ReplayRefused("Saved attempts are not native analysis calls")
        self.audits = deepcopy(audits)
        self.source = source
        self.source_sha256 = source_sha256
        self.replacement = (
            "[Frozen provider P: existing lightweight memory implementation; "
            "source_sha256=" + source_sha256 + ". Full source omitted for input size only. "
            "P and its effective configuration remain fixed; assess the inherited philosophical "
            "system using the supplied training evidence.]"
        )
        self.event_path = Path(event_path)
        self.replayed = 0
        self.live_requests = 0

    def record(self, event):
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


class ReplayCompressClient:
    """Outer adapter; all live requests still pass through the budgeted client."""

    def __init__(self, inner, state, label):
        self.inner = inner
        self.state = state
        self.label = label
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        state = self.state
        if self.label == "decision-evolve-trajectory-summary":
            state.record({"event": "summary_repayment_refused", "label": self.label})
            raise ReplayRefused("Successful trajectory summaries must be read from training cache")
        if self.label not in {"decision-evolve-analysis", "decision-evolve-generation"}:
            raise ReplayRefused("Unexpected client role during resume")
        if state.replayed < 2:
            if self.label != "decision-evolve-analysis":
                raise ReplayRefused("Generation cannot precede exact analysis replay")
            audit = state.audits[state.replayed]
            saved = audit["request"]
            actual = {key: kwargs.get(key) for key in ("messages", "model")}
            expected = {key: saved.get(key) for key in ("messages", "model")}
            if actual != expected:
                state.record({"event": "cache_replay_refused", "source_attempt": audit["attempt"],
                              "actual_messages_model_sha256": _digest(actual),
                              "expected_messages_model_sha256": _digest(expected)})
                raise ReplayRefused("Cached analysis messages or model differ; no request was sent")
            # Match the actual paid request hash after the same deterministic
            # changes made by the inner budget wrapper, including any extra
            # semantic fields such as stop/tools/response_format.
            normalized = deepcopy(kwargs)
            normalized["max_tokens"] = min(normalized.get("max_tokens", 4096), 4096)
            normalized["temperature"] = 0
            normalized.pop("timeout", None)
            if str(normalized.get("model", "")).lower().startswith("deepseek"):
                normalized["extra_body"] = {
                    **(normalized.get("extra_body") or {}), "thinking": {"type": "disabled"}}
            if _digest(normalized) != audit["request_sha256"]:
                state.record({"event": "cache_replay_refused", "source_attempt": audit["attempt"],
                              "reason": "normalized_request_hash_mismatch"})
                raise ReplayRefused("Cached normalized request differs; no request was sent")
            result = restored_completion(audit)
            state.record({"event": "cache_replay", "source_attempt": audit["attempt"],
                          "messages_model_sha256": _digest(actual),
                          "usage_reused_for_native_metrics": audit["usage"],
                          "budget_increment": 0, "network_dispatched": False,
                          "cache_replay_not_billable": True})
            state.replayed += 1
            return result

        original = kwargs.get("messages")
        compressed, count = compress_messages(original, state.source, state.replacement)
        request = dict(kwargs)
        request["messages"] = compressed
        before = len(json.dumps(original, ensure_ascii=False).encode("utf-8"))
        after = len(json.dumps(compressed, ensure_ascii=False).encode("utf-8"))
        state.record({"event": "live_request_compression", "label": self.label,
                      "source_sha256": state.source_sha256, "source_occurrences_replaced": count,
                      "messages_before_sha256": _digest(original),
                      "messages_after_sha256": _digest(compressed),
                      "messages_utf8_bytes_before": before, "messages_utf8_bytes_after": after,
                      "budget_policy_unchanged": True})
        result = self.inner.chat.completions.create(**request)
        state.live_requests += 1
        return result


def resume(run_dir, values):
    """Activate the original run and return evolve_resume/summary.json contents."""
    from . import decision_smoke_evolve

    run_dir = Path(run_dir).resolve(strict=True)
    root = run_dir / "decision_smoke"
    budget_root = root / "budget"
    original_failure = root / "evolve/failure.json"
    if not original_failure.is_file() or (root / "evolve/summary.json").exists():
        raise ReplayRefused("Resume requires the preserved failed evolution without a candidate")
    if (root / "evolve_resume").exists() or (root / "resume_protocol.json").exists():
        raise ReplayRefused("Resume was already attempted; automatic retries are disabled")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("formal_matrix_authorized") or values.get("DEFAULT_MODEL") != manifest["model"]:
        raise ReplayRefused("Model or minimal experimental scope changed")
    budget = Budget(budget_root, max_calls=100, max_tokens=300000)
    before = budget.summary()
    if before["blocked_reason"] or before["unknown_usage_attempts"]:
        raise ReplayRefused("Budget has a blocked or unknown paid outcome")
    audits = [json.loads((budget_root / "attempts" / f"{number:04d}.json").read_text(encoding="utf-8"))
              for number in (19, 20)]
    for audit in audits:
        restored_completion(audit)
    for task_id in ("train_01", "train_02"):
        log = json.loads((root / "train/task_logs" / f"{task_id}.json").read_text(encoding="utf-8"))
        if not isinstance(log.get("trajectory_summary"), str) or not log["trajectory_summary"].strip():
            raise ReplayRefused("Both successful training trajectory summaries must already be cached")
    workspace = run_dir / "workspace"
    source_path = workspace / "EvolveLab/providers/lightweight_memory_provider.py"
    source = source_path.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    original_manifest = json.loads((root / "evolve/input_manifest.json").read_text(encoding="utf-8"))
    if source_hash != original_manifest["parent_code_sha256"]:
        raise ReplayRefused("Frozen provider source changed since the stopped analysis")
    if load(run_dir / "lineage", PROVIDER).to_dict() != original_manifest["parent_orientation_snapshot"]:
        raise ReplayRefused("Parent philosophy changed since the stopped analysis")
    current_memory = decision_smoke_evolve._files(workspace / "storage" / PROVIDER)
    if current_memory != original_manifest["ordinary_memory"]:
        raise ReplayRefused("Ordinary training memory changed since the stopped analysis")
    state = ReplayState(audits, source, source_hash, root / "evolve_resume/replay_events.jsonl")
    values = dict(values)
    values.update(ANALYSIS_MODEL=values["DEFAULT_MODEL"], GENERATION_MODEL=values["DEFAULT_MODEL"],
                  DEFAULT_SUMMARY_MODEL=values["DEFAULT_MODEL"], PHILO_SMOKE_THINKING="disabled")
    os.environ.update({key: str(value) for key, value in values.items() if value is not None})
    activate_workspace(run_dir)
    protocol = {
        "kind": "exact_analysis_replay_then_frozen_source_compression",
        "replayed_attempts": [19, 20], "prohibit_new_trajectory_summary_requests": True,
        "source_sha256": source_hash, "budget_before": before,
        "budget_algorithm_and_limits_changed": False, "original_failure_preserved": str(original_failure),
        "compression_scope": "Exact full parent provider source substrings inside messages only",
        "output": str(root / "evolve_resume"), "formal_matrix": False,
    }
    write_json(root / "resume_protocol.json", protocol)

    def wrap(model, shared_budget, snapshot, label, max_output=4096):
        result = wrap_native_model(model, shared_budget, snapshot, label, max_output)
        result.client = ReplayCompressClient(result.client, state, label)
        return result

    with (root / "evolve_resume.native.log").open("x", encoding="utf-8") as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            with patch.object(decision_smoke_evolve, "wrap_native_model", wrap):
                result = decision_smoke_evolve.evolve(run_dir, values, budget_root, output_name="evolve_resume")
    if state.replayed != 2:
        raise ReplayRefused("Native analysis did not consume both exact cached responses")
    write_json(root / "evolve_resume/resume_summary.json", {
        **protocol, "success": True, "replayed_count": state.replayed,
        "successful_live_requests": state.live_requests,
        "budget_after": budget.summary(), "candidate_provider": result["candidate_provider"],
    })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Original absolute run directory")
    args = parser.parse_args(argv)
    try:
        result = resume(args.run, environment())
    except Exception as exc:
        print(json.dumps({"success": False, "phase": "evolve_resume", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
