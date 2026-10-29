"""Lossless evidence projection for the bounded, Phi-only native reflection.

No model calls, environment loading, evaluator imports or inferred counterfactuals.
The raw logs stay immutable; a manifest links each projected record to its source.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace


TRAIN_IDS = ("train_01", "train_02")
PHI_BEGIN = "<current_phi>"
PHI_END = "</current_phi>"


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class ReflectionCoverage:
    """Coverage comes from successful native tool returns, never model claims."""

    def __init__(self):
        self.summaries = set()
        self.steps = {task: set() for task in TRAIN_IDS}

    def trajectories(self, requested, result):
        if not isinstance(result, dict):
            return
        for task in requested:
            item = result.get(task)
            if (task in TRAIN_IDS and isinstance(item, dict) and "error" not in item
                    and isinstance(item.get("trajectory_summary"), str)
                    and item["trajectory_summary"].strip()):
                self.summaries.add(task)

    def specific_steps(self, task, requested, result):
        if task not in TRAIN_IDS or not isinstance(result, dict) or result.get("task_id") != task:
            return
        returned = result.get("requested_steps", {})
        if not isinstance(returned, dict):
            return
        for index in requested:
            item = returned.get(f"step_{index}")
            if (type(index) is int and index >= 0 and isinstance(item, dict) and "error" not in item
                    and item.get("name") in {"plan", "action", "summary"}
                    and item.get("source_step") == index):
                self.steps[task].add(index)

    def report(self):
        return {"basis": "successful_native_tool_returns", "required_tasks": list(TRAIN_IDS),
                "summary_tasks": sorted(self.summaries),
                "step_indices": {task: sorted(indices) for task, indices in self.steps.items()},
                "complete": self.summaries == set(TRAIN_IDS) and all(self.steps.values())}

    def require_complete(self):
        if not self.report()["complete"]:
            raise ValueError("Native analysis must read both training summaries and at least one valid step per task")


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else compact(value).encode("utf-8")).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(compact(value) + "\n", encoding="utf-8")


def phi_reference(snapshot, index):
    item = snapshot.items[index]
    match = re.search(r"(?:METHOD-\d|ONT|EPI|MET|VAL|ACT|REV)", item)
    label = match.group() if match else f"item-{index + 1}"
    return f"[Phi {snapshot.sha256[:12]} / {label}; full text in current_phi]"


def replace_phi(value, snapshot):
    """Replace only exact existing paragraphs, leaving all ordinary M intact."""
    if isinstance(value, str):
        for index, item in enumerate(snapshot.items):
            value = value.replace(item, phi_reference(snapshot, index))
        return value
    if isinstance(value, list):
        return [replace_phi(item, snapshot) for item in value]
    if isinstance(value, dict):
        return {key: replace_phi(item, snapshot) for key, item in value.items()}
    return value


def project_record(data, task_id, snapshot):
    if task_id not in TRAIN_IDS or data.get("evaluation", {}).get("task_id") != task_id:
        raise ValueError("Only the matching completed training record may be projected")
    if data.get("orientation_snapshot") != snapshot.to_dict():
        raise ValueError("Training Phi differs from the current parent")
    evaluation = data["evaluation"]
    # Whitelist realized outcome fields. Never include alternative action tables.
    realized = {key: deepcopy(evaluation[key]) for key in (
        "task_id", "done", "time", "deadline", "reward", "cost", "utility",
        "success", "completed", "feedback", "violations", "observations") if key in evaluation}
    audit = [{key: deepcopy(event[key]) for key in ("tool", "argument", "result") if key in event}
             for event in evaluation.get("audit", [])]
    steps = []
    for index, source in enumerate(data.get("agent_trajectory", [])):
        step = {key: deepcopy(source[key]) for key in (
            "name", "value", "think", "tool_calls", "obs", "memory_guidance")
                if source.get(key) not in (None, "", [])}
        step = replace_phi(step, snapshot)
        # final_answer repeats agent_result; retain its location and decision
        # rationale but refer to the one complete answer in the projected log.
        if step.get("name") == "action" and step.get("tool_calls"):
            if all(call.get("name") == "final_answer" for call in step["tool_calls"]):
                for call in step["tool_calls"]:
                    call["arguments"] = "[see agent_result: actor self-report]"
                step["obs"] = "[see agent_result: actor self-report, not environment feedback]"
        step["source_step"] = index
        if "memory_guidance" in step:
            step["memory_guidance_provenance"] = (
                "Native provider output, possibly retrieved experience or immediate model advice. "
                "Only exact Phi paragraphs were replaced by references; all other text is retained. "
                "Advice and the actor's claimed philosophical influence are not causal evidence.")
        steps.append(step)
    summary = {"kind": "deterministic_observed_events_only", "actual_environment_events": audit,
               "realized_outcome": realized,
               "step_index": [{"index": i, "name": step.get("name"),
                               "tools": [call.get("name") for call in step.get("tool_calls", [])]}
                              for i, step in enumerate(steps)]}
    return {"kind": "native_phi_reflection_projection", "task_id": task_id,
            "status": data.get("status"), "question": data["question"],
            "agent_result": replace_phi(data.get("agent_result", ""), snapshot),
            "agent_result_provenance": "Actor self-report; actual results are in trajectory_summary.",
            "is_correct": data.get("is_correct", False), "judgement": data.get("judgement"),
            "agent_trajectory": steps, "trajectory_summary": compact(summary),
            "phi_reference": snapshot.sha256}


def build_view(training, output, snapshot):
    training, output = Path(training).resolve(strict=True), Path(output).resolve()
    if {path.name for path in training.glob("*.json")} != {f"{task}.json" for task in TRAIN_IDS}:
        raise ValueError("Reflection requires exactly the two training logs")
    if output.exists():
        raise FileExistsError("Reflection projection already exists")
    output.mkdir(parents=True)
    mapping = {}
    for task_id in TRAIN_IDS:
        source = training / f"{task_id}.json"
        raw = source.read_bytes()
        projected = project_record(json.loads(raw.decode("utf-8")), task_id, snapshot)
        target = output / "task_logs" / source.name
        _write(target, projected)
        mapping[task_id] = {"source": str(source), "source_sha256": sha(raw),
                            "source_bytes": len(raw), "projection": str(target),
                            "projection_sha256": sha(target.read_bytes()),
                            "projection_bytes": target.stat().st_size,
                            "retained_fields": ["question", "agent_result", "agent_trajectory.name/value/think/tool_calls/obs/memory_guidance",
                                                "evaluation.realized_fields", "evaluation.audit.tool/argument/result"],
                            "omitted_fields": ["agent_messages", "budget_before", "budget_after", "metrics",
                                               "memory_ingestion", "trajectory_summary.generated", "orientation_snapshot.items"],
                            "transforms": ["exact Phi paragraph -> version/paragraph reference",
                                           "final_answer repeated payload -> agent_result reference",
                                           "deterministic summary from actual audit; no inferred conclusion"]}
    _write(output / "current_phi.json", snapshot.to_dict())
    manifest = {"kind": "training_only_reflection_view_v1", "sources": mapping,
                "phi_sha256": snapshot.sha256, "generated_summary_calls": 0,
                "raw_logs_unchanged": True, "no_held_out_inputs": True}
    _write(output / "source_map.json", manifest)
    return manifest


def deduplicate_phi(messages):
    """The native planner embeds the task again; keep one verbatim Phi block."""
    seen = False
    pattern = re.compile(re.escape(PHI_BEGIN) + r".*?" + re.escape(PHI_END), re.S)

    def visit(value):
        nonlocal seen
        if isinstance(value, str):
            def substitute(match):
                nonlocal seen
                if not seen:
                    seen = True
                    return match.group()
                return "[current_phi repeated by native task history; use the complete earlier block]"
            return pattern.sub(substitute, value)
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        return value
    return visit(deepcopy(messages))


class ReflectionClient:
    """Deduplicate before the budget/audit boundary, retaining the actual request."""

    def __init__(self, audited_client):
        self.audited_client = audited_client
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        kwargs["messages"] = deduplicate_phi(kwargs.get("messages", []))
        return self.audited_client.chat.completions.create(**kwargs)


def analysis_prompt(snapshot, parent, view, memory_files):
    return (
        "Analyze the revisable philosophical system using only these two completed training episodes. "
        "Provider implementation P and its effective configuration are frozen. Do not design provider code "
        "or memory pipelines. Do not inspect, predict or reference held-out tasks.\n"
        "Read both view_trajectories results and at least one relevant view_specific_steps entry for EACH "
        "training task before final_answer; reading both tasks' steps in one batch of tool calls is allowed. "
        "The summaries deterministically reproduce actual observations and realized outcomes; they do not "
        "infer causes. Preserve the distinction among observed feedback, actor self-report, ordinary native "
        "memory advice and philosophical interpretation. Ordinary memory advice may itself propose actions. "
        "A reference or self-report does not establish that Phi caused a decision.\n"
        "Assess which philosophical position, relationship or scope condition is supported, challenged or "
        "underdetermined. Explain alternatives and what evidence would distinguish them. Keeping Phi unchanged "
        "is valid. Do not infer the outcomes of unchosen actions. Task goals, permissions and requirements "
        "remain external constraints and cannot be revoked by philosophical revision or a higher score. "
        "Respect the six connected philosophical chapters and optional methods as a system: do not collapse "
        "it to slogans, task-specific answers or a mandatory pipeline. Revisions can qualify, connect or "
        "reorganize claims where the observed evidence warrants it; no text change is required.\n"
        "Give a concise report with observed decisions/outcomes, supported or uncertain interpretation, "
        "proposed revision or retention with paragraph references, implications for related claims, and limits.\n"
        + compact({"training_ids": list(TRAIN_IDS), "projected_task_logs": str(Path(view) / "task_logs"),
                   "readable_memory_files": [str(path) for path in memory_files],
                   "fixed_P_sha256": parent["source_sha256"]})
        + "\n" + PHI_BEGIN + "\n" + compact(snapshot.to_dict()) + "\n" + PHI_END
    )


def generation_prompt(analysis_text, snapshot, existing_names):
    # This is the native parser's exact format. The frozen-P policy replaces
    # the parsed placeholder before native creation, so no source is executed.
    return (
        "Produce one native-compatible Phi candidate from the training-only analysis below. P is frozen "
        "and your provider placeholder will be discarded before creation. Do not design implementation, "
        "retrieval or storage changes. Retaining the current Phi is valid; a change is not evidence of improvement. "
        "If revising, explain which claims, relations and scope conditions change and why, keeping unsupported "
        "alternatives unresolved. Preserve the coherent six-chapter worldview/methodology and optional methods "
        "unless a specific revision is justified; do not compress the system into a task recipe. "
        "Task goals, permissions and requirements are external and cannot be revoked by Phi. "
        "Do not reference or anticipate held-out tasks.\n"
        "Output rationale as plain prose, followed by exactly these native headings with NEW unique identifiers "
        "and a minimal nonexecuted Python placeholder (under 20 lines):\n"
        "**Class Name**: UniquePhiCandidate\n**Module Name**: unique_phi_candidate\n"
        "```python\nclass UniquePhiCandidate:\n    pass\n```\n"
        "**Enum Name**: UNIQUE_PHI_CANDIDATE\n**Enum Value**: unique_phi_candidate\n"
        "Configuration: omit orientation_items to inherit all current items unchanged; or emit "
        "**orientation_items**: followed by the complete single-line JSON list of strings. "
        "An explicit [] removes all Phi and would require evidence-based justification. "
        "Do not use other bold key/value fields for prose. No other config fields are needed.\n"
        "Existing identifiers: " + compact(existing_names) + "\nTraining analysis:\n"
        + replace_phi(analysis_text, snapshot) + "\n" + PHI_BEGIN + "\n"
        + compact(snapshot.to_dict()) + "\n" + PHI_END
    )
