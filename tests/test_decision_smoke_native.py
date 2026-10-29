"""Native decision actor exercised through a fake SDK transport, entirely offline."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class NativeDecisionSmokeTests(unittest.TestCase):
    def test_native_tools_actor_budget_and_task_log_without_ingestion(self):
        script = r'''
import json, os, socket, sys
from pathlib import Path
from types import SimpleNamespace
import dotenv

def forbidden(*args, **kwargs):
    raise AssertionError("Offline test attempted environment-file or network access")

dotenv.load_dotenv = forbidden
dotenv.dotenv_values = forbidden
socket.create_connection = forbidden
socket.socket.connect = forbidden
from philoagent.cli import init_run, PROVIDER
from philoagent.live_smoke import activate_workspace
from philoagent import decision_smoke
decision_smoke.environment = forbidden
root = Path(sys.argv[1])
run = Path(init_run("offline-native-decision", "engineering", output_root=root)["run_dir"])
activate_workspace(run)
import openai
from openai.types.chat import ChatCompletion

calls, actor_actions, clients = [], [], []

class FakeClient:
    def __init__(self):
        self.options = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        clients.append(self)

    def with_options(self, **kwargs):
        self.options = kwargs
        return self

    def create(self, **kwargs):
        assert kwargs["model"] == "deepseek-offline"
        assert kwargs["temperature"] == 0
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
        calls.append(kwargs)
        texts = []
        for message in kwargs["messages"]:
            content = message.get("content", "")
            texts.append(content if isinstance(content, str) else " ".join(x.get("text", "") for x in content))
        prompt = "\n".join(texts)
        if "You are a memory guidance system." in prompt:
            output = {"selected_indices": [], "guidance": ""}
        elif "You are analyzing the current step of task execution." in prompt:
            output = {"step_summary": "Use the recorded observations of this episode.", "key_extracts": []}
        elif "world-class planning expert" in texts[0]:
            output = "Inspect the current capacity, execute a suitable action, then report the observed result."
        else:
            if "Results for tool call 'act'" in prompt:
                name, arguments = "final_answer", {"answer": "已读取当前容量并将作业交付主机池；12单位按时完成。"}
            elif "Results for tool call 'inspect'" in prompt:
                name, arguments = "act", {"option": "primary"}
            else:
                name, arguments = "inspect", {"source_id": "current_meter"}
            actor_actions.append(name)
            output = {"think": "Use the observed episode state.", "tools": [{"name": name, "arguments": arguments}]}
        content = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        return ChatCompletion.model_validate({
            "id": "offline-decision-" + str(len(calls)), "created": 0,
            "model": "deepseek-offline", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": content, "reasoning_content": ""}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

# Keep the actual native OpenAIServerModel, ToolCallingAgent, Tool schema
# validation, provider, and budget adapter. Only the SDK transport is replaced.
openai.OpenAI = lambda **kwargs: FakeClient()
import FlashOAgents as native
assert Path(native.__file__).resolve().is_relative_to(run / "workspace")
budget_root = root / "shared-budget"
output = run / "decision_smoke/offline"
summary = decision_smoke.run_episodes(
    run, {"DEFAULT_MODEL": "deepseek-offline", "OPENAI_API_KEY": "offline-placeholder"},
    budget_root, ("train_01",), PROVIDER, output, ingest=False)
assert actor_actions == ["inspect", "act", "final_answer"], actor_actions
assert len(calls) >= 4, len(calls)
assert clients and all(client.options == {"timeout": 90, "max_retries": 0} for client in clients)
assert summary["ingestion_enabled"] is False
assert summary["tasks"][0]["success"] is True, summary
assert summary["tasks"][0]["utility"] == 11
assert [(item["tool"], item["argument"]) for item in summary["tasks"][0]["audit"]] == [
    ("inspect", "current_meter"), ("act", "primary")]
assert summary["budget"]["attempt_count"] == len(calls)
assert summary["budget"]["actual_tokens"] == 15 * len(calls)
assert summary["budget"]["blocked_reason"] is None
ledger = json.loads((budget_root / "budget.json").read_text(encoding="utf-8"))
assert len(ledger["attempts"]) == len(calls)
assert all(item["status"] == "success" for item in ledger["attempts"])
audit = [json.loads(line) for line in (budget_root / "api_calls.jsonl").read_text(encoding="utf-8").splitlines()]
assert any(item["orientation_present_in_request"] for item in audit), audit
record = json.loads((output / "task_logs/train_01.json").read_text(encoding="utf-8"))
assert record["memory_ingestion"] is None
assert record["is_correct"] is True
assert record["evaluation"]["completed"] is True
assert record["metrics"]["total_api_calls"] == len(calls)
assert record["budget_after"]["attempt_count"] == len(calls)
assert (output / "summary.json").is_file()
print("OFFLINE_NATIVE_DECISION_OK", len(calls))
'''
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            script_path = Path(temporary) / "offline_native_decision.py"
            script_path.write_text(script, encoding="utf-8")
            # Explicitly exclude inherited credentials and all experiment config.
            env = {key: os.environ[key] for key in
                   ("SystemRoot", "WINDIR", "PATH", "TEMP", "TMP", "COMSPEC", "PATHEXT")
                   if key in os.environ}
            env.update(PYTHONPATH=str(project), PYTHONUTF8="1",
                       PYTHONDONTWRITEBYTECODE="1", PHILO_SMOKE_THINKING="disabled",
                       OPENAI_API_KEY="offline-placeholder")
            completed = subprocess.run(
                [sys.executable, str(script_path), str(Path(temporary) / "runs")],
                cwd=project, env=env, capture_output=True, text=True, encoding="utf-8", timeout=90)
        self.assertEqual(completed.returncode, 0, (completed.stdout + completed.stderr)[-14000:])
        self.assertIn("OFFLINE_NATIVE_DECISION_OK", completed.stdout)


if __name__ == "__main__":
    unittest.main()
