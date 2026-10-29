"""Offline native train -> analyze -> generate -> alias -> fresh-probe closure.

Only the OpenAI transport is fake. Registry loading, native agent tools, memory
ingestion, proposal parsing, factor projection, creation and budget auditing run
normally in isolated child processes. Both a revised Phi and legal inheritance
must reach actual candidate requests without changing P or leaking probe data.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


_CHILD = r'''
import json, os, shutil, socket, sys
from pathlib import Path
from types import SimpleNamespace
import dotenv

def forbidden(*args, **kwargs):
    raise AssertionError("Offline closure test attempted environment-file or network access")

dotenv.load_dotenv = forbidden
dotenv.dotenv_values = forbidden
socket.create_connection = forbidden
socket.socket.connect = forbidden
from philoagent.cli import init_run, PROVIDER, PROJECT
from philoagent.live_smoke import activate_workspace, write_json, digest
from philoagent import decision_smoke
from philoagent.hss_budget import process_identity
from philoagent.orientation import load
decision_smoke.environment = forbidden
root, phase, revision = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
run = root / ("native-closure-" + revision)
if phase == "train":
    run = Path(init_run(run.name, "dialectical", output_root=root,
                        seed_file=PROJECT / "seeds/systems/dialectical-with-methods-v1.json")["run_dir"])
budget_root = run / "decision_smoke/budget"
values = {"DEFAULT_MODEL": "deepseek-offline", "OPENAI_API_KEY": "offline-placeholder"}
if phase == "train":
    run_manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    write_json(run / "decision_smoke/manifest.json", {
        "formal_matrix_authorized": False, "adapter_sha256": run_manifest["adapter_sha256"],
        "max_calls": 100, "max_tokens_soft_stop": 300000, "model": values["DEFAULT_MODEL"]})
marker = "REVISION_WITNESS: Separate observations from inherited expectations and revise applicability from contrary evidence."
common_fact = "TRAINED_MEMORY_WITNESS: The two observed batches had different current error states."
calls, actor_actions, extractions = [], [], []

if phase.startswith("probe_"):
    candidate = json.loads((run / "decision_smoke/evolve/summary.json").read_text(encoding="utf-8"))["candidate_provider"]
    condition, number = phase.split("_")[1:]
    provider_id = PROVIDER if condition == "old" else candidate
    task_id = "probe_" + number
    trial = decision_smoke.clone_probe(run, task_id, provider_id, phase)
    active_run = trial
else:
    active_run, provider_id = run, PROVIDER
assert not any(name == package or name.startswith(package + ".")
               for name in sys.modules for package in ("FlashOAgents", "EvolveLab", "MemEvolve")), "Native registry was imported before this child selected its workspace"
activate_workspace(active_run)
import openai
from openai.types.chat import ChatCompletion

def proposal():
    parent = load(run / "lineage", PROVIDER)
    text = """**Class Name**: OfflineUnexecutedProvider
**Module Name**: offline_unexecuted_provider
```python
raise AssertionError('GENERATED_PROVIDER_MUST_NOT_EXECUTE')
class OfflineUnexecutedProvider:
    pass
```
**Enum Name**: OFFLINE_UNEXECUTED_PROVIDER
**Enum Value**: offline_unexecuted_provider
**storage_dir**: ./storage/offline_unexecuted_provider
"""
    if revision == "change":
        text += "**orientation_items**: " + json.dumps([*parent.items, marker], ensure_ascii=False) + "\n"
    return text

class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **kwargs):
        assert kwargs == {"timeout": 90, "max_retries": 0}
        return self

    def create(self, **kwargs):
        assert kwargs["model"] == "deepseek-offline"
        assert kwargs["temperature"] == 0
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
        texts = []
        for message in kwargs["messages"]:
            content = message.get("content", "")
            texts.append(content if isinstance(content, str) else " ".join(part.get("text", "") for part in content))
        prompt = "\n".join(texts)
        calls.append({"max_tokens": kwargs["max_tokens"], "prompt": prompt})
        if phase == "evolve":
            assert "probe_01" not in prompt and "probe_02" not in prompt, "Probe IDs leaked to evolution"
            assert "展览布展" not in prompt and "馆藏整理" not in prompt, "Probe task leaked to evolution"
            assert "class LightweightMemoryProvider" not in prompt, "Frozen P source leaked into reflection"
            assert all(prompt.count(item) == 1 for item in load(run / "lineage", PROVIDER).items), "Reflection must carry each complete Phi paragraph exactly once"
            if kwargs["max_tokens"] >= 12000:
                output = proposal()
            elif "Summarize this agent task execution trajectory step by step" in prompt:
                output = "Step 0: Planned to use present observations. Step 1: Inspected current state. Step 2: Chose the available resource. Step 3: Reported the result."
            elif "world-class planning expert" in texts[0]:
                output = "Review the two training records; retain useful principles and state any evidence-supported correction."
            elif "Results for tool call 'view_trajectories'" not in prompt:
                output = {"think": "Read only completed training evidence.", "tools": [{"name": "view_trajectories", "arguments": {"task_ids": ["train_01", "train_02"]}}]}
            elif "Results for tool call 'view_specific_steps'" not in prompt:
                output = {"think": "Check actual observations, actions and memory guidance for both training episodes.", "tools": [
                    {"name": "view_specific_steps", "arguments": {"task_id": task, "step_indices": [0, 1, 2]}}
                    for task in ("train_01", "train_02")]}
            else:
                output = {"think": "Observed evidence supports condition-sensitive interpretation.", "tools": [{"name": "final_answer", "arguments": {"answer": "Both training tasks inspected current state and completed. Retain the worldview; make the distinction between observations and inherited expectation explicit only if useful. No efficacy claim follows from two cases."}}]}
        elif "Extract reusable learnings from this successful execution." in prompt:
            assert phase == "train", "Probe was ingested"
            extractions.append(prompt)
            output = {"strategic": [common_fact], "operational": []}
        elif "You are a memory guidance system." in prompt:
            output = {"selected_indices": [1], "guidance": common_fact}
        elif "You are analyzing the current step of task execution." in prompt:
            output = {"step_summary": "Use the actual observation from this episode.", "key_extracts": []}
        elif "world-class planning expert" in texts[0]:
            output = "Inspect relevant current evidence, execute an admissible action, then report actual outcome."
        else:
            if phase == "train":
                current_task = "train_02" if '"task_id": "train_02"' in prompt else "train_01"
            else:
                current_task = task_id
            source, option = {
                "train_01": ("current_meter", "primary"),
                "train_02": ("current_meter", "reserve"),
                "probe_01": ("sensor", "hall"),
                "probe_02": ("material_record", "dry"),
            }[current_task]
            if "Results for tool call 'act'" in prompt:
                name, arguments = "final_answer", {"answer": "The stated action was executed using observed evidence; report the tool outcome."}
            elif "Results for tool call 'inspect'" in prompt:
                name, arguments = "act", {"option": option}
            else:
                name, arguments = "inspect", {"source_id": source}
            actor_actions.append([current_task, name])
            output = {"think": "Use observed conditions with the current working principles.", "tools": [{"name": name, "arguments": arguments}]}
        content = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        return ChatCompletion.model_validate({
            "id": "offline-closure-" + str(len(calls)), "created": 0,
            "model": "deepseek-offline", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": content, "reasoning_content": ""}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

openai.OpenAI = lambda **kwargs: FakeClient()
import FlashOAgents as native
assert Path(native.__file__).resolve().is_relative_to(active_run / "workspace")

if phase == "train":
    result = decision_smoke.run_episodes(run, values, budget_root, ("train_01", "train_02"),
                                         PROVIDER, run / "decision_smoke/train", ingest=True)
    assert len(extractions) == 2, len(extractions)
    assert all(task["success"] and task["completed"] for task in result["tasks"])
    frozen = run / "decision_smoke/frozen_memory"
    shutil.copytree(run / "workspace/storage" / PROVIDER, frozen)
    files = decision_smoke.file_manifest(frozen)
    assert files, "Training did not persist ordinary memory"
    assert common_fact in (frozen / "longterm_memory.json").read_text(encoding="utf-8")
    write_json(run / "decision_smoke/frozen_memory.json", {"files": files, "sha256": digest(files)})
elif phase == "evolve":
    from philoagent.decision_smoke_evolve import evolve
    result = evolve(run, values, budget_root)
    assert result["success"] and result["policy_fixed"]
    assert result["orientation_changed"] == (revision == "change")
    assert result["generated_code_executed"] is False
    candidate = load(run / "lineage", result["candidate_provider"])
    parent = load(run / "lineage", PROVIDER)
    assert candidate.parent_sha256 == parent.sha256
    assert (marker in candidate.items) == (revision == "change")
    assert not (run / "workspace/EvolveLab/providers/offline_unexecuted_provider.py").exists()
    evidence = json.loads((run / "decision_smoke/evolve/policy_fixed_evidence.json").read_text(encoding="utf-8"))
    assert evidence["ordinary_memory_unchanged"]
    assert evidence["parent_code_sha256"] == evidence["candidate_code_sha256"]
    assert evidence["parent_effective_config_sha256"] == evidence["candidate_effective_config_sha256"]
    assert any("view_trajectories" in call["prompt"] for call in calls)
    assert any("Results for tool call 'view_specific_steps'" in call["prompt"] for call in calls)
else:
    initial = json.loads((run / "decision_smoke/frozen_memory.json").read_text(encoding="utf-8"))["files"]
    result = decision_smoke.run_episodes(active_run, values, budget_root, (task_id,), provider_id,
                                         active_run / "result", ingest=False)
    assert result["memory_before"] == initial, "Trial did not start with common M bytes"
    assert result["ingestion_enabled"] is False and not extractions
    assert result["tasks"][0]["success"] and result["tasks"][0]["completed"]
    snapshot = load(active_run / "lineage", provider_id)
    assert result["phi_items_sha256"] == digest(list(snapshot.items))
    assert any(all(item in call["prompt"] for item in snapshot.items) for call in calls), "Actual request missing active Phi"
    assert any(common_fact in call["prompt"] for call in calls), "Common training memory was not read"
    assert any(marker in call["prompt"] for call in calls) == (condition == "new" and revision == "change")
    events = [json.loads(line) for path in (active_run / "live_smoke/runtime_evidence").glob("*.jsonl")
              for line in path.read_text(encoding="utf-8").splitlines()]
    assert events and all(event["details"]["provider"] == provider_id for event in events)
    assert all(event["disk_matches_loaded_code"] for event in events)
    assert not any(event["operation"] == "take_in_memory" for event in events), "Probe runtime ingested memory"
    parent_code = (run / "workspace/EvolveLab/providers/lightweight_memory_provider.py").read_bytes()
    import hashlib
    assert all(event["loaded_code_sha256"] == hashlib.sha256(parent_code).hexdigest() for event in events)
    record = json.loads((active_run / "result/task_logs" / (task_id + ".json")).read_text(encoding="utf-8"))
    assert record["memory_ingestion"] is None
    assert actor_actions == [[task_id, "inspect"], [task_id, "act"], [task_id, "final_answer"]]
assert result["budget"]["unknown_usage_attempts"] == 0
assert result["budget"]["blocked_reason"] is None
assert result["budget"]["actual_tokens"] == result["budget"]["attempt_count"] * 15
write_json(run / "decision_smoke" / (phase + ".offline_assertions.json"), {
    "phase": phase, "pid": os.getpid(), "process_identity": process_identity(os.getpid()),
    "requests": len(calls), "extractions": len(extractions),
    "actor_actions": actor_actions, "budget": result["budget"]})
print("OFFLINE_NATIVE_CLOSURE_OK", phase, revision)
'''


class NativeDecisionClosureTests(unittest.TestCase):
    def test_native_chain_revised_and_retained_phi_in_fresh_processes(self):
        project = Path(__file__).resolve().parents[1]
        phases = ("train", "evolve", "probe_old_01", "probe_new_01", "probe_new_02", "probe_old_02")
        env = {key: os.environ[key] for key in
               ("SystemRoot", "WINDIR", "PATH", "TEMP", "TMP", "COMSPEC", "PATHEXT")
               if key in os.environ}
        env.update(PYTHONPATH=str(project), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                   PHILO_SMOKE_THINKING="disabled", OPENAI_API_KEY="offline-placeholder")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "offline_native_closure.py"
            script.write_text(_CHILD, encoding="utf-8")
            for revision in ("change", "retain"):
                with self.subTest(revision=revision):
                    process_births = []
                    for phase in phases:
                        completed = subprocess.run(
                            [sys.executable, str(script), str(root), phase, revision],
                            cwd=project, env=env, capture_output=True, text=True,
                            encoding="utf-8", timeout=120)
                        self.assertEqual(completed.returncode, 0,
                                         f"{revision}/{phase}\n" + (completed.stdout + completed.stderr)[-18000:])
                        self.assertIn("OFFLINE_NATIVE_CLOSURE_OK", completed.stdout)
                        record = json.loads((root / ("native-closure-" + revision) / "decision_smoke" /
                                             (phase + ".offline_assertions.json")).read_text(encoding="utf-8"))
                        identity = record["process_identity"]
                        self.assertIsInstance(identity, dict)
                        self.assertEqual(identity["pid"], record["pid"])
                        self.assertIsInstance(identity.get("started_unix"), (int, float))
                        self.assertGreater(identity["started_unix"], 0)
                        process_births.append((record["pid"], identity["started_unix"]))
                    # Windows may reuse a PID after its child exits. Birth time
                    # distinguishes those fresh interpreters without rejecting reuse.
                    self.assertEqual(len(set(process_births)), len(phases), "Each phase needs a fresh process and native module registry")
                    from philoagent.decision_smoke import finalize
                    run = root / ("native-closure-" + revision)
                    closed = finalize(run)
                    self.assertTrue(closed["protocol_completed"])
                    self.assertTrue(closed["all_episodes_reached_terminal_state"])
                    self.assertEqual(closed["decision_success_count"], 6)
                    self.assertEqual(closed["orientation_changed"], revision == "change")
                    self.assertFalse(closed["formal_matrix_started"])
                    self.assertEqual(set(closed["runtime_checks"]), set(phases[2:]))
                    # The acceptance routine must reject an otherwise complete
                    # trial whose record claims that held-out feedback was ingested.
                    record_path = run / "decision_smoke/trials/probe_new_01/result/task_logs/probe_01.json"
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    record["memory_ingestion"] = [True, "unexpected feedback"]
                    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "ingested"):
                        finalize(run)


if __name__ == "__main__":
    unittest.main()
