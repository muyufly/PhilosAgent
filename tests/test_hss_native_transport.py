"""Real native HSS actor/provider/quality path, with only SDK transport faked."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


SCRIPT = r'''
from contextlib import ExitStack
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import dotenv

forbidden_calls = []
def forbidden(*args, **kwargs):
    forbidden_calls.append("environment-or-network-access")
    raise AssertionError("Offline HSS native transport test attempted dotenv or network access")

dotenv.load_dotenv = forbidden
dotenv.dotenv_values = forbidden
socket.create_connection = forbidden
socket.socket.connect = forbidden

from philoagent import cli
cli.environment = forbidden
from philoagent.live_smoke import activate_workspace
from philoagent.hss_budget import install_hss_audit
from philoagent.hss_evaluation import RUBRICS
from philoagent.hss_tasks_runtime import NativeHSSRunner
from philoagent.hss_tool_calls import install_tool_decoding

root = Path(sys.argv[1])
run_dir = Path(cli.init_run("offline-native-hss", "engineering", output_root=root)["run_dir"])
workspace = activate_workspace(run_dir)
os.environ.update(DEFAULT_MODEL="offline-model", DEFAULT_JUDGE_MODEL="offline-judge",
                  OPENAI_API_KEY="offline-placeholder", WEB_SEARCH_PROVIDER="deepseek",
                  WEB_ACCESS_PROVIDER="deepseek", LITELLM_LOCAL_MODEL_COST_MAP="True")
assert not list(workspace.glob(".env*")), "An isolated run must not copy credential files"

import openai
from openai.types.chat import ChatCompletion

calls = []
patterns = []
clients = []
decoder_mode = sys.argv[2] if len(sys.argv) > 2 else "normal"
learned_insight = "offline-learned-quality-method: compare evidence before proposing a strategy"
report = "先根据给定需求比较两种增长路径，再以预算和可检验的指标安排分阶段试行。"

class OfflineClient:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.api_key = kwargs.get("api_key", "offline-placeholder")
        self.base_url = kwargs.get("base_url", "https://offline.invalid")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        clients.append(self)

    def with_options(self, **kwargs):
        self.options.update(kwargs)
        return self

    def create(self, **kwargs):
        calls.append(kwargs)
        assert kwargs["model"] in ("offline-model", "offline-judge")
        assert kwargs["max_tokens"] == 8192
        assert kwargs["timeout"] == 30
        texts = []
        for message in kwargs["messages"]:
            content = message.get("content", "")
            texts.append(content if isinstance(content, str) else " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)))
        prompt = "\n".join(texts)
        if "你是文社商开放任务的质量评审" in prompt:
            patterns.append("judge")
            payload = json.loads(texts[-1])
            assert payload["anonymous_report"] == report
            assert "SECRET_GROUP" not in prompt and "SECRET_PHI" not in prompt
            assert "agent_trajectory" not in prompt
            output = {"dimensions": {name: {"score": 3.75, "reason": "Report makes a concrete, assessable comparison."}
                                      for name in RUBRICS["business"]},
                      "overall_comment": "A coherent, implementable report with identifiable limitations."}
        elif "You are a memory guidance system." in prompt:
            patterns.append("memory_guidance")
            output = {"selected_indices": [1], "guidance": "Consider comparing evidence and implementation costs."}
        elif "You are analyzing the current step of task execution." in prompt:
            patterns.append("step_extraction")
            output = {"step_summary": "The report compares growth choices using supplied material.", "key_extracts": []}
        elif "Extract reusable learnings from this successful execution." in prompt:
            patterns.append("memory_ingestion")
            assert "HELDOUT_ONLY" not in prompt
            output = {"strategic": [learned_insight], "operational": []}
        elif "world-class planning expert" in texts[0]:
            patterns.append("planning")
            output = "Compare the supplied choices, assess the evidence, and deliver a decision report."
        elif "Each action must contain" in prompt:
            patterns.append("actor")
            if decoder_mode == "malformed_then_dsml" and patterns.count("actor") == 1:
                output = {"tools": ["PLAN"]}
            elif decoder_mode == "malformed_then_dsml" and patterns.count("actor") == 2:
                assert "Tool output format error" in prompt
                output = ('[Analysis] Completed the task.\n[Tools]\n<｜｜DSML｜｜ calls>'
                          '<｜｜DSML｜｜ invoke name="final_answer">'
                          '<｜｜DSML｜｜ parameter name="arguments" string="false">'
                          + json.dumps({"answer": report}, ensure_ascii=False)
                          + '</｜｜DSML｜｜ parameter></｜｜DSML｜｜ invoke></｜｜DSML｜｜ calls>')
            else:
                output = {"think": "Compare the provided options and deliver the requested work.",
                          "tools": [{"name": "final_answer", "arguments": {"answer": report}}]}
        else:
            raise AssertionError("Unexpected native model prompt: " + prompt[:1000])
        content = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        return ChatCompletion.model_validate({
            "id": "offline-hss-" + str(len(calls)), "created": 0,
            "model": kwargs["model"], "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": content, "reasoning_content": ""}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

# Only replace the SDK transport. All native model conversion, prompt logic,
# tool schema validation, provider learning and BaseAgent capture remain real.
openai.OpenAI = OfflineClient
import FlashOAgents as native
import run_flash_searcher_webwalkerqa as runner
from base_agent import BaseAgent
from EvolveLab.providers.lightweight_memory_provider import LightweightMemoryProvider
assert Path(native.__file__).resolve().is_relative_to(workspace)
assert Path(sys.modules[LightweightMemoryProvider.__module__].__file__).resolve().is_relative_to(workspace)

task = {"task_id": "OFFLINE_LEARNING", "domain": "business", "split": "learning",
        "question": "根据给定预算，比较两种增长路径并写出可实施的建议。",
        "materials": [{"content": "预算为100，路径甲成本60，路径乙成本80。"}],
        "task_criteria": ["Compare costs and implementation."],
        "group": "SECRET_GROUP", "phi": "SECRET_PHI"}
learning_file = run_dir / "learning.jsonl"
learning_file.write_text(json.dumps(task, ensure_ascii=False) + "\n", encoding="utf-8")
evaluation_file = run_dir / "evaluation.jsonl"
evaluation_file.write_text(json.dumps({**task, "task_id": "OFFLINE_HELDOUT", "split": "evaluation",
                                       "question": "HELDOUT_ONLY: compare another pair of growth plans."},
                                      ensure_ascii=False) + "\n", encoding="utf-8")
protocol = {"model": "offline-model", "judge_model": "offline-judge",
            "actor_max_steps": 32, "actor_max_output_tokens": 8192,
            "judge_max_output_tokens": 8192, "request_timeout_seconds": 30,
            "max_calls": 100, "max_tokens": 1000000}
config = {"model_id": "offline-model", "api_key": "offline-placeholder", "temperature": 0,
          "custom_role_conversions": {"tool-call": "assistant", "tool-response": "user"}}
with ExitStack() as stack:
    install_tool_decoding(stack, native)
    sink = install_hss_audit(stack, run_dir, workspace, root / "shared_budget", protocol)
    callback = NativeHSSRunner(run_dir, workspace, protocol, config, runner=runner, native=native,
                               base_agent_class=BaseAgent, usage_reader=lambda: sink.events)
    learning_logs = callback("writingbench", cli.PROVIDER, [0], learning_file,
                             run_dir / "evolution" / "round_1" / "base_logs")
    learning = json.loads((learning_logs / "1.json").read_text(encoding="utf-8"))
    assert learning["status"] == "completed", learning
    assert learning["grade"]["quality_score"] == 75
    assert learning["score"] == .75
    assert learning["memory_ingestion_complete"] is True
    assert learning["agent_result"] == report
    if decoder_mode == "malformed_then_dsml":
        assert len(learning["actor_decoding_events"]) == 2
        assert learning["actor_decoding_events"][0]["status"] == "invalid"
        assert learning["actor_decoding_events"][1]["format"] == "dsml"
        assert learning["actor_decoding_events"][0]["original_content"] == '{"tools": ["PLAN"]}'
    assert {step["name"] for step in learning["agent_trajectory"]} == {"plan", "action"}
    assert learning["agent_messages"]
    assert Path(learning["runtime_artifact"]["source_path"]).resolve().is_relative_to(workspace)
    assert learning["api_usage"]["usage_complete"]
    quality = json.loads((learning_logs / "_quality" / "summary.json").read_text(encoding="utf-8"))
    assert quality["summary"]["accuracy"] == .75
    storage = workspace / "storage" / cli.PROVIDER / "longterm_memory.json"
    learned_db = json.loads(storage.read_text(encoding="utf-8"))
    assert any(memory["content"] == learned_insight for memory in learned_db["strategic"])
    assert patterns.count("memory_ingestion") == 1
    learning_event_count = len(sink.events)
    callback.heldout = True
    heldout_logs = callback("writingbench", cli.PROVIDER, [0], evaluation_file,
                            run_dir / "terminal" / "task_logs")
    heldout = json.loads((heldout_logs / "1.json").read_text(encoding="utf-8"))
    assert heldout["status"] == "completed", heldout
    assert heldout["grade"]["quality_score"] == 75
    assert heldout["stage"] == "heldout"
    assert heldout["memory_evolution_enabled"] is False
    assert heldout["memory_ingestion_complete"] is False
    assert patterns.count("memory_ingestion") == 1
    # Reopening the actual provider exposed the learned memory to retrieval.
    heldout_requests = sink.events[learning_event_count:]
    assert any(learned_insight in json.dumps(event["request"], ensure_ascii=False)
               for event in heldout_requests)
    assert not any(event["role"] == "memory_ingestion" for event in heldout_requests)
    assert learning["total_model_tokens"] + heldout["total_model_tokens"] == 15 * len(calls)
    budget = sink.budget.summary()
    assert budget["attempt_count"] == len(calls)
    assert budget["actual_tokens"] == 15 * len(calls)
    assert budget["successful_requests"] == len(calls)
    assert budget["failed_requests"] == 0 and budget["pending_count"] == 0
    assert budget["blocked_reason"] is None
    assert len(sink.events) == len(calls)
    assert all(event["status"] == "success" for event in sink.events)
    assert any(event["orientation_present_in_request"] for event in sink.events if event["role"] == "actor")
    assert {event["role"] for event in sink.events} == {"actor", "judge", "memory_ingestion"}
    assert all(client.options["max_retries"] == 0 for client in clients)
    evidence = [json.loads(line) for path in (run_dir / "live_smoke" / "runtime_evidence").glob("calls-*.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines()]
    assert sum(item["operation"] == "take_in_memory" for item in evidence) == 1
    assert sum(item["operation"] == "provide_memory" for item in evidence) >= 4
    assert not forbidden_calls, forbidden_calls
    assert {"planning", "actor", "judge", "memory_guidance", "step_extraction", "memory_ingestion"}.issubset(set(patterns))
    print("OFFLINE_HSS_NATIVE_TRANSPORT_OK", len(calls), budget["actual_tokens"])
'''


def _run_native_transport(decoder_mode):
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temporary:
        script = Path(temporary) / "offline_hss_native_transport.py"
        script.write_text(SCRIPT, encoding="utf-8")
        # Keep only the operating-system essentials; no inherited credentials,
        # endpoint configuration, experiment state or proxy settings enter it.
        env = {key: os.environ[key] for key in
               ("SystemRoot", "WINDIR", "PATH", "TEMP", "TMP", "COMSPEC", "PATHEXT")
               if key in os.environ}
        env.update(PYTHONPATH=str(project), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                   LITELLM_LOCAL_MODEL_COST_MAP="True", OPENAI_API_KEY="offline-placeholder")
        completed = subprocess.run([sys.executable, str(script), str(Path(temporary) / "runs"), decoder_mode],
                                   cwd=project, env=env, capture_output=True, text=True,
                                   encoding="utf-8", timeout=90)
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-18000:]
    assert "OFFLINE_HSS_NATIVE_TRANSPORT_OK" in completed.stdout


def test_actual_native_learning_and_heldout_via_audited_fake_sdk_transport():
    _run_native_transport("normal")


def test_actual_native_step_continues_after_plan_string_and_decodes_dsml_final_answer():
    _run_native_transport("malformed_then_dsml")
