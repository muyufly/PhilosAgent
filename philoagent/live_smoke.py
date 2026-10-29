"""Real-model integration smoke test over synthetic local fixtures, never WebWalkerQA.

Create an isolated run with ``philoagent init`` first. The three phases intentionally
separate generated code from its execution: tasks, evolve, then candidate after
the candidate source has been inspected and its SHA-256 supplied explicitly.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
from functools import partial
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

from .cli import PROVIDER, environment
from .orientation import load


FIXTURES = {
    "release-initial": "SYNTHETIC LOCAL FIXTURE. 09:00 preliminary bulletin: release lot CEDAR-27; pressure test passed. Check release-update before deciding.",
    "release-update": "SYNTHETIC LOCAL FIXTURE. 10:00 authoritative correction supersedes release-initial. The test sensor was uncalibrated. Lot CEDAR-27 must be held pending a calibrated retest. Decision code HOLD.",
    "route-rules": "SYNTHETIC LOCAL FIXTURE. Routine route ALPHA is shorter but unavailable to refrigerated cargo. For refrigerated cargo use route BETA if its current reading is at most 4 C and it is open. Otherwise HOLD. Do not copy a prior HOLD decision without checking present conditions.",
    "route-current": "SYNTHETIC LOCAL FIXTURE. Present shipment is refrigerated. ALPHA is open, reading 2 C. BETA is open, reading 3 C. Route rules remain current. BETA meets the refrigerated requirement. Decision code BETA.",
}
TASKS = (
    {"id": "evidence_reversal", "documents": ["release-initial", "release-update"], "expected": "HOLD",
     "question": "Decide whether to release lot CEDAR-27 using the preliminary bulletin and the correction. Explain which evidence changes the initial conclusion."},
    {"id": "context_constraint", "documents": ["route-rules", "route-current"], "expected": "BETA",
     "question": "Choose the present refrigerated shipment's route. Apply the cargo constraint and present readings; do not reuse an earlier decision mechanically."},
)


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


class EmptyModelContentError(ValueError):
    """Transport returned successfully but supplied no usable assistant output."""


class AuditedClient:
    """Audit the actual SDK request boundary without persisting credentials/URLs."""

    def __init__(self, client, path, orientation_items, *, label, timeout=60, max_tokens=4096):
        self.client = client.with_options(timeout=timeout, max_retries=0)
        self.path = Path(path)
        self.orientation_items = tuple(orientation_items)
        self.label = label
        self.max_tokens = max_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        # Clamp upstream generation's large default to a deliberate smoke budget.
        kwargs["max_tokens"] = min(kwargs.get("max_tokens", self.max_tokens), self.max_tokens)
        thinking = os.environ.get("PHILO_SMOKE_THINKING", "disabled")
        apply_thinking = str(kwargs.get("model", "")).lower().startswith("deepseek") and thinking != "provider-default"
        if apply_thinking:
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body["thinking"] = {"type": thinking}
            kwargs["extra_body"] = extra_body
        messages = kwargs.get("messages", [])
        prompt = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
        event = {"label": self.label, "model": kwargs.get("model"), "prompt_sha256": digest(messages),
                 "orientation_item_count": len(self.orientation_items),
                 "orientation_present_in_request": bool(self.orientation_items) and all(x in prompt for x in self.orientation_items),
                 "max_tokens": kwargs["max_tokens"], "thinking_selection": thinking,
                 "thinking_applied": apply_thinking, "status": "started", "status_scope": "transport_only"}
        started = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = self.client.chat.completions.create(**kwargs)
            usage = getattr(result, "usage", None)
            has_content = any(bool(getattr(choice.message, "content", None) and choice.message.content.strip()) for choice in result.choices)
            has_tools = any(bool(getattr(choice.message, "tool_calls", None)) for choice in result.choices)
            event.update(status="success", usage=usage.model_dump() if hasattr(usage, "model_dump") else None,
                         response_id=getattr(result, "id", None),
                         response_has_content=has_content, response_has_tool_calls=has_tools,
                         finish_reason=[getattr(choice, "finish_reason", None) for choice in result.choices],
                         content_sha256=digest([choice.message.model_dump() for choice in result.choices]))
            if not has_content and not has_tools:
                raise EmptyModelContentError("Model response has no content or tool calls; inspect API audit finish_reason and usage")
            return result
        except Exception as exc:
            if event["status"] != "success":
                event["status"] = "error"
            event["error_type"] = type(exc).__name__
            raise
        finally:
            event["duration_seconds"] = round(time.monotonic() - started, 3)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def make_model(native, values, output, snapshot, label, *, model_key="DEFAULT_MODEL", max_tokens=4096, timeout=60):
    model = native.OpenAIServerModel(model_id=values[model_key], api_base=values.get("OPENAI_API_BASE"),
                                     api_key=values["OPENAI_API_KEY"], temperature=0, max_tokens=max_tokens)
    model.client = AuditedClient(model.client, output / "api_calls.jsonl", snapshot.items,
                                 label=label, timeout=timeout, max_tokens=max_tokens)
    return model


def parse_answer(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return {}
    return {}


def check_answer(task, value, reads):
    answer = parse_answer(value)
    evidence = answer.get("evidence", [])
    return (answer.get("decision") == task["expected"]
            and isinstance(evidence, list) and all(isinstance(item, str) for item in evidence)
            and set(task["documents"]).issubset(set(evidence))
            and set(task["documents"]).issubset(set(reads)))


def run_tasks(run_dir, values, *, provider_id=PROVIDER, output_name="tasks", max_steps=5, timeout=60):
    import FlashOAgents as native
    from base_agent import BaseAgent
    from EvolveLab.memory_types import TrajectoryData
    from run_flash_searcher_webwalkerqa import load_memory_provider

    output = run_dir / "live_smoke" / output_name
    if (output / "summary.json").exists() or (output / "api_calls.jsonl").exists():
        raise ValueError("This smoke output already exists; use a fresh isolated run for a repeat")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "task_logs"
    snapshot = load(run_dir / "lineage", provider_id)
    write_json(output / "fixtures.json", {"kind": "synthetic_local_fixtures", "fixtures": FIXTURES, "tasks": TASKS})
    model = make_model(native, values, output, snapshot, "task-and-native-memory", timeout=timeout)
    provider = load_memory_provider(provider_id, model)
    results = []
    for task in TASKS:
        reads = []

        class ReadFixture(native.Tool):
            name = "read_fixture"
            description = "Read one synthetic local fixture document by its exact document_id. No web or filesystem access."
            inputs = {"document_id": {"type": "string", "description": "Exact fixture document ID listed in this task."}}
            output_type = "string"

            def forward(self, document_id: str) -> str:
                if document_id not in task["documents"]:
                    raise ValueError("Unknown fixture document for this task")
                reads.append(document_id)
                return FIXTURES[document_id]

        agent = native.ToolCallingAgent(tools=[ReadFixture()], model=model, memory_provider=provider,
                                        max_steps=max_steps, summary_interval=None, verbosity_level=0)
        question = ("This is an integration test using SYNTHETIC LOCAL FIXTURES, not a web benchmark. "
                    + task["question"] + " Read every listed document with read_fixture before answering. "
                    + "Document IDs: " + ", ".join(task["documents"])
                    + '. Call final_answer with answer as a JSON object containing "decision" (the exact decision code), '
                    + '"evidence" (a list of the document IDs you used), and "reason" (a brief explanation).')
        model.reset_total_counts()
        answer = agent.run(question)
        correct = check_answer(task, answer, reads)
        messages = agent.write_memory_to_messages(include_system_prompt=False)
        before_calls = model.get_total_counts()["total_api_calls"]
        ingestion = provider.take_in_memory(TrajectoryData(query=question, trajectory=messages, result=answer,
                       metadata={"status": "success", "is_correct": correct, "synthetic_local_fixture": True,
                                 "full_query": question, "task_id": task["id"]}))
        wrapper = BaseAgent(model)
        wrapper.agent_fn = agent
        record = {"kind": "synthetic_local_fixture", "status": "success", "question": question,
                  "agent_result": answer, "golden_answer": task["expected"], "is_correct": correct,
                  "judgement": "correct" if correct else "incorrect", "fixture_reads": reads,
                  "agent_messages": messages, **wrapper.capture_trajectory(), "memory_ingestion": list(ingestion),
                  "ingestion_api_calls": model.get_total_counts()["total_api_calls"] - before_calls,
                  "metrics": model.get_total_counts(), "orientation_snapshot": snapshot.to_dict()}
        write_json(logs / f"{task['id']}.json", record)
        results.append({"id": task["id"], "correct": correct, "fixture_reads": reads,
                        "memory_ingested": bool(ingestion[0]), "ingestion_api_calls": record["ingestion_api_calls"]})
    api_calls = [json.loads(line) for line in (output / "api_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    orientation_sent = any(call["orientation_present_in_request"] for call in api_calls)
    summary = {"kind": "synthetic_local_integration_smoke", "formal_webwalkerqa": False, "provider": provider_id,
               "task_count": len(results), "correct_count": sum(x["correct"] for x in results), "tasks": results,
               "api_call_count": len(api_calls), "successful_api_call_count": sum(x["status"] == "success" for x in api_calls),
               "orientation_entered_model_request": orientation_sent,
               "thinking_selection": os.environ.get("PHILO_SMOKE_THINKING", "disabled"),
               "api_status_scope": "transport_only",
               "passed": all(x["correct"] and x["memory_ingested"] and x["ingestion_api_calls"] > 0 for x in results)
                         and (orientation_sent or not snapshot.items),
               "orientation_snapshot": snapshot.to_dict(), "output_dir": str(output)}
    write_json(output / "summary.json", summary)
    return summary


def activate_workspace(run_dir):
    workspace = (run_dir / "workspace").resolve(strict=True)
    if not (run_dir / "run.json").is_file():
        raise ValueError("Expected a run created by philoagent init")
    # Avoid labelling cached vendor modules as the isolated run's code.
    for name in ("FlashOAgents", "EvolveLab", "MemEvolve", "run_flash_searcher_webwalkerqa", "base_agent"):
        module = sys.modules.get(name)
        if module and getattr(module, "__file__", None) and workspace not in Path(module.__file__).resolve().parents:
            raise ValueError("Native modules already loaded from another workspace; start a fresh Python process")
    sys.path.insert(0, str(workspace))
    os.chdir(workspace)
    os.environ["PHILO_LINEAGE_ROOT"] = str(run_dir / "lineage")
    os.environ["PHILO_EVIDENCE_DIR"] = str(run_dir / "live_smoke" / "runtime_evidence")
    return workspace


def validate_candidate(config, workspace):
    """Validate filenames and syntax without importing/executing generated code."""
    provider = config["provider_code"]
    memory_type = config["memory_type_info"]
    for name in (provider["class_name"], provider["module_name"], memory_type["enum_name"], memory_type["enum_value"]):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("Generated candidate contains an invalid Python identifier")
    if (workspace / "EvolveLab" / "providers" / f"{provider['module_name']}.py").exists():
        raise ValueError("Generated candidate must use a new provider filename")
    ast.parse(provider["code"])
    return digest(provider["code"])


def validate_analysis(analysis):
    """Reject the native wrapper's error/empty results before generation."""
    if not analysis.get("success") or analysis.get("error"):
        raise ValueError("Native analysis failed; generation was not started")
    report = analysis.get("agent_result")
    if isinstance(report, dict):
        if report.get("error"):
            raise ValueError("Native analysis agent returned an error; generation was not started")
        report = report.get("agent_result")
    if not report or (isinstance(report, str) and not report.strip()):
        raise ValueError("Native analysis produced no report; generation was not started")


def run_evolution(run_dir, values, *, timeout=60):
    import FlashOAgents as native
    from openai import OpenAI
    from MemEvolve.phases import phase_analyzer
    from MemEvolve.phases.phase_generator import PhaseGenerator
    from MemEvolve.phases.memory_creator import MemorySystemCreator
    from MemEvolve.utils import trajectory_tools

    output = run_dir / "live_smoke" / "evolve"
    task_output = run_dir / "live_smoke" / "tasks"
    if not (task_output / "summary.json").exists():
        raise ValueError("Complete tasks phase before evolve")
    if output.exists():
        raise ValueError("Evolution smoke output already exists; use a fresh isolated run for a repeat")
    output.mkdir(parents=True)
    workspace = run_dir / "workspace"
    snapshot = load(run_dir / "lineage", PROVIDER)
    summary_model = values.get("DEFAULT_SUMMARY_MODEL") or values["ANALYSIS_MODEL"]
    original_agent = phase_analyzer.AnalysisAgent
    original_client = trajectory_tools.OpenAI

    def summary_client(**kwargs):
        return AuditedClient(OpenAI(api_key=values["OPENAI_API_KEY"], base_url=values.get("OPENAI_API_BASE")),
                             output / "api_calls.jsonl", snapshot.items, label="native-trajectory-summary", timeout=timeout)

    # Native AnalysisAgent otherwise has a hardcoded summary model default.
    phase_analyzer.AnalysisAgent = partial(original_agent, summary_model_id=summary_model)
    trajectory_tools.OpenAI = summary_client
    try:
        analyzer = phase_analyzer.PhaseAnalyzer(values["ANALYSIS_MODEL"], str(task_output / "task_logs"), PROVIDER)
        analyzer.base_model = make_model(native, values, output, snapshot, "native-analysis",
                                         model_key="ANALYSIS_MODEL", timeout=timeout)
        analysis = analyzer.run_analysis(max_steps=4)
    finally:
        phase_analyzer.AnalysisAgent = original_agent
        trajectory_tools.OpenAI = original_client
    write_json(output / "analysis.json", analysis)
    validate_analysis(analysis)
    client = AuditedClient(OpenAI(api_key=values["OPENAI_API_KEY"], base_url=values.get("OPENAI_API_BASE")),
                           output / "api_calls.jsonl", snapshot.items, label="native-generation", timeout=timeout, max_tokens=12000)
    generator = PhaseGenerator(client, values["GENERATION_MODEL"], output, PROVIDER)
    original_prompt = generator._build_generation_prompt

    def bounded_prompt(*args, **kwargs):
        return original_prompt(*args, **kwargs) + (
            "\n\nThis is a small SYNTHETIC LOCAL FIXTURE integration smoke, not formal WebWalkerQA evaluation. "
            "Generate a compact provider (prefer at most 220 Python lines), using only Python stdlib and the "
            "existing EvolveLab/FlashOAgents interfaces. Use config.get('model') for LLM calls; do not construct "
            "external clients, download resources, execute shell commands, or make network calls yourself. "
            "Store only under a relative storage/<new-provider-name> directory. Accept current principles from "
            "config['philo_orientation_items']; never hardcode seed text or change lineage. For native memory "
            "extraction use the supplied model and real trajectory. Include **orientation_items**: followed by "
            "a single-line JSON list, explicitly choosing to retain or revise the current principles based on "
            "the analysis. An empty list removes them. This field belongs outside the Python code block. "
            "Give the new provider unique class/module/enum identifiers. Focus on one concrete, inspectable "
            "improvement to retrieval or extraction; do not add a framework or dependencies."
        )

    generator._build_generation_prompt = bounded_prompt
    generation = generator.run_generation(analysis, creativity_index=0.2)
    write_json(output / "generation.json", generation)
    if not generation.get("success"):
        raise RuntimeError("Native generation did not produce a parseable candidate")
    config = generation["config"]
    generated_text_hash = validate_candidate(config, workspace)
    # A standalone copy makes inspection possible before any generated import.
    source_path = output / "candidate_provider.py"
    source_path.write_text(config["provider_code"]["code"], encoding="utf-8")
    created = MemorySystemCreator.create_memory_system(config, base_dir=str(workspace))
    write_json(output / "creation.json", created)
    if not created.get("success"):
        raise RuntimeError("Native creator failed; inspect creation.json")
    provider_path = workspace / "EvolveLab" / "providers" / f"{config['provider_code']['module_name']}.py"
    provider_bytes = provider_path.read_bytes()
    source_path.write_bytes(provider_bytes)
    code_hash = hashlib.sha256(provider_bytes).hexdigest()
    candidate_id = created["memory_type"]
    candidate_snapshot = load(run_dir / "lineage", candidate_id)
    parent_source = workspace / "EvolveLab" / "providers" / "lightweight_memory_provider.py"
    result = {"kind": "synthetic_local_integration_smoke", "formal_webwalkerqa": False,
              "thinking_selection": os.environ.get("PHILO_SMOKE_THINKING", "disabled"),
              "api_status_scope": "transport_only",
              "generated_code_executed": False, "candidate_provider": candidate_id,
              "candidate_source": str(source_path), "candidate_code_sha256": code_hash,
              "generated_text_sha256": generated_text_hash,
              "source_changed_vs_parent": code_hash != hashlib.sha256(parent_source.read_bytes()).hexdigest(),
              "parent_orientation_snapshot": snapshot.to_dict(),
              "candidate_orientation_snapshot": candidate_snapshot.to_dict(),
              "orientation_changed": snapshot.items != candidate_snapshot.items,
              "next_step": "Inspect generated source and config, then invoke candidate phase with --reviewed-code-sha256 " + code_hash}
    write_json(output / "summary.json", result)
    return result


def run_candidate(run_dir, values, reviewed_hash, *, max_steps=5, timeout=60):
    summary_path = run_dir / "live_smoke" / "evolve" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_hash = summary["candidate_code_sha256"]
    if not reviewed_hash or reviewed_hash != expected_hash:
        raise ValueError("Candidate execution requires the SHA-256 of the reviewed generated source")
    config = json.loads((summary_path.parent / "generation.json").read_text(encoding="utf-8"))["config"]
    source = run_dir / "workspace" / "EvolveLab" / "providers" / f"{config['provider_code']['module_name']}.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Candidate provider changed since generation/review")
    return run_tasks(run_dir, values, provider_id=summary["candidate_provider"], output_name="candidate",
                     max_steps=max_steps, timeout=timeout)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase", choices=["tasks", "evolve", "candidate"], default="tasks")
    parser.add_argument("--env-file")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-steps", type=int, choices=range(4, 7), default=5)
    parser.add_argument("--reviewed-code-sha256")
    parser.add_argument("--thinking", choices=["disabled", "enabled", "provider-default"], default="disabled",
                        help="DeepSeek smoke requests only; default disables thinking to preserve the small output budget")
    args = parser.parse_args(argv)
    try:
        run_dir = Path(args.run_dir).resolve(strict=True)
        values = environment(args.env_file)
        required = ["DEFAULT_MODEL", "OPENAI_API_KEY"]
        if args.phase == "evolve":
            required.extend(["ANALYSIS_MODEL", "GENERATION_MODEL"])
        missing = [key for key in required if not values.get(key)]
        if missing:
            raise ValueError("Missing configuration: " + ", ".join(missing))
        if not 1 <= args.timeout <= 180:
            raise ValueError("Timeout must be between 1 and 180 seconds")
        os.environ.update(values)
        os.environ["PHILO_SMOKE_THINKING"] = args.thinking
        activate_workspace(run_dir)
        logging.disable(logging.CRITICAL)
        # Native prompts/outputs are retained in task artifacts, without printing
        # SDK exceptions (which can contain the endpoint or sensitive headers).
        with redirect_stdout(io.StringIO()):
            if args.phase == "tasks":
                result = run_tasks(run_dir, values, max_steps=args.max_steps, timeout=args.timeout)
            elif args.phase == "evolve":
                result = run_evolution(run_dir, values, timeout=args.timeout)
            else:
                result = run_candidate(run_dir, values, args.reviewed_code_sha256,
                                       max_steps=args.max_steps, timeout=args.timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        # Deliberately omit arbitrary API/LLM exception strings from console.
        print(json.dumps({"success": False, "error_type": type(exc).__name__,
                          "message": str(exc) if isinstance(exc, (ValueError, FileNotFoundError, FileExistsError)) else "See saved smoke artifacts; no credential-bearing exception text is printed"}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
