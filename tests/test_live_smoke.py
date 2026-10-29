"""Offline checks for the real-model smoke harness; no .env or network access."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from philoagent import live_smoke


class AuditedClientTests(unittest.TestCase):
    def test_audit_records_actual_request_usage_and_phi_without_secrets(self):
        calls = []

        class Client:
            api_key = "offline-secret-do-not-log"

            def with_options(self, **kwargs):
                self.options = kwargs
                return self

        client = Client()

        def complete(**kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content="done", tool_calls=None, model_dump=lambda: {"role": "assistant", "content": "done"})
            return SimpleNamespace(id="offline-call", usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 12, "completion_tokens": 3}),
                                   choices=[SimpleNamespace(message=message)])

        client.chat = SimpleNamespace(completions=SimpleNamespace(create=complete))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calls.jsonl"
            audited = live_smoke.AuditedClient(client, path, ["check current evidence"], label="offline-test", max_tokens=50)
            audited.chat.completions.create(model="offline-model", messages=[{"role": "user", "content": "check current evidence"}], max_tokens=60000)
            text = path.read_text(encoding="utf-8")
            event = json.loads(text)
        self.assertTrue(event["orientation_present_in_request"])
        self.assertEqual(event["usage"]["prompt_tokens"], 12)
        self.assertEqual(calls[0]["max_tokens"], 50)
        self.assertEqual(client.options, {"timeout": 60, "max_retries": 0})
        self.assertNotIn(client.api_key, text)
        self.assertNotIn("check current evidence", text)

    def test_error_audit_contains_only_exception_type(self):
        class Client:
            def with_options(self, **kwargs):
                return self

        client = Client()

        def fail(**kwargs):
            raise RuntimeError("secret-endpoint-and-key")

        client.chat = SimpleNamespace(completions=SimpleNamespace(create=fail))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calls.jsonl"
            audited = live_smoke.AuditedClient(client, path, [], label="offline-error")
            with self.assertRaises(RuntimeError):
                audited.create(model="offline", messages=[])
            text = path.read_text(encoding="utf-8")
        self.assertEqual(json.loads(text)["error_type"], "RuntimeError")
        self.assertNotIn("secret-endpoint-and-key", text)

    def test_deepseek_thinking_selection_preserves_other_extra_body_fields(self):
        for model, selection, applied in (("deepseek-flash", "disabled", True),
                                          ("DeepSeek-flash", "enabled", True),
                                          ("deepseek-flash", "provider-default", False),
                                          ("other-model", "disabled", False)):
            with self.subTest(model=model, selection=selection), tempfile.TemporaryDirectory() as temporary:
                calls = []

                class Client:
                    def with_options(self, **kwargs):
                        return self

                def complete(**kwargs):
                    calls.append(kwargs)
                    message = SimpleNamespace(content="ok", tool_calls=None, model_dump=lambda: {"content": "ok"})
                    return SimpleNamespace(id="offline", usage=None, choices=[SimpleNamespace(message=message, finish_reason="stop")])

                client = Client()
                client.chat = SimpleNamespace(completions=SimpleNamespace(create=complete))
                path = Path(temporary) / "calls.jsonl"
                with patch.dict(os.environ, {"PHILO_SMOKE_THINKING": selection}):
                    live_smoke.AuditedClient(client, path, [], label="offline").create(
                        model=model, messages=[], extra_body={"other_setting": 7})
                self.assertEqual(calls[0]["extra_body"]["other_setting"], 7)
                self.assertEqual("thinking" in calls[0]["extra_body"], applied)
                if applied:
                    self.assertEqual(calls[0]["extra_body"]["thinking"], {"type": selection})
                event = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(event["thinking_selection"], selection)
                self.assertEqual(event["thinking_applied"], applied)
                self.assertEqual(event["finish_reason"], ["stop"])

    def test_empty_response_keeps_transport_success_but_raises_validation_error(self):
        class Client:
            def with_options(self, **kwargs):
                return self

        client = Client()
        message = SimpleNamespace(content=None, tool_calls=None, model_dump=lambda: {"content": None})
        response = SimpleNamespace(id="offline-empty", usage=None, choices=[SimpleNamespace(message=message, finish_reason="length")])
        client.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calls.jsonl"
            with self.assertRaises(live_smoke.EmptyModelContentError):
                live_smoke.AuditedClient(client, path, [], label="offline-empty").create(model="deepseek-flash", messages=[])
            event = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(event["status"], "success")
        self.assertEqual(event["status_scope"], "transport_only")
        self.assertFalse(event["response_has_content"])
        self.assertFalse(event["response_has_tool_calls"])
        self.assertEqual(event["finish_reason"], ["length"])
        self.assertEqual(event["error_type"], "EmptyModelContentError")

    def test_failed_or_empty_native_analysis_is_rejected(self):
        for analysis in ({"success": False}, {"success": True, "agent_result": ""},
                         {"success": True, "agent_result": {"error": "empty response"}},
                         {"success": True, "agent_result": {"agent_result": " "}}):
            with self.subTest(analysis=analysis), self.assertRaises(ValueError):
                live_smoke.validate_analysis(analysis)
        live_smoke.validate_analysis({"success": True, "agent_result": {"agent_result": "A usable report"}})

    def test_fixture_answer_requires_evidence_and_real_fixture_reads(self):
        task = live_smoke.TASKS[0]
        answer = {"decision": "HOLD", "evidence": task["documents"]}
        self.assertTrue(live_smoke.check_answer(task, answer, task["documents"]))
        self.assertFalse(live_smoke.check_answer(task, answer, []))
        self.assertFalse(live_smoke.check_answer(task, {"decision": "RELEASE", "evidence": task["documents"]}, task["documents"]))

    def test_generated_filename_and_source_validated_without_execution(self):
        config = {"provider_code": {"class_name": "Candidate", "module_name": "candidate", "code": "raise AssertionError('must not execute')\n"},
                  "memory_type_info": {"enum_name": "CANDIDATE", "enum_value": "candidate"}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(live_smoke.validate_candidate(config, root), live_smoke.digest(config["provider_code"]["code"]))
            config["provider_code"]["module_name"] = "../escape"
            with self.assertRaisesRegex(ValueError, "identifier"):
                live_smoke.validate_candidate(config, root)

    def test_candidate_cannot_execute_without_matching_review_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            live_smoke.write_json(run_dir / "live_smoke/evolve/summary.json", {"candidate_code_sha256": "expected"})
            with self.assertRaisesRegex(ValueError, "reviewed"):
                live_smoke.run_candidate(run_dir, {}, None)


class NativeOfflineSmokeTests(unittest.TestCase):
    def test_native_task_loop_fixture_calls_and_memory_ingestion(self):
        # A fresh process proves imports come from the isolated workspace. Every
        # completion crosses AuditedClient, then an in-memory fake SDK transport.
        script = r'''
import hashlib, json, os, sys
from pathlib import Path
from philoagent.cli import init_run
from philoagent import live_smoke
root = Path(sys.argv[1])
run = Path(init_run("offline", "engineering", output_root=root)["run_dir"])
live_smoke.activate_workspace(run)
os.environ["OPENAI_API_KEY"] = "offline-never-network"
import FlashOAgents as native
from openai.types.chat import ChatCompletion
generated = "\n".join([
    "**Class Name**: OfflineCandidate", "**Module Name**: offline_candidate_provider",
    "**Enum Name**: OFFLINE_CANDIDATE", "**Enum Value**: offline_candidate",
    "```python", "raise AssertionError('evolve phase executed generated code')", "class OfflineCandidate:", "    pass", "```",
    "**storage_dir**: ./storage/offline_candidate",
    '**orientation_items**: ["Check the latest correction before retaining an earlier judgment."]'])
class FakeClient:
    def with_options(self, **kwargs):
        return self
    @property
    def chat(self):
        return self
    @property
    def completions(self):
        return self
    def create(self, **kwargs):
        texts = []
        for message in kwargs["messages"]:
            content = message.get("content", "")
            texts.append(content if isinstance(content, str) else " ".join(x.get("text", "") for x in content))
        prompt = "\n".join(texts)
        if kwargs["model"] == "offline-generation":
            output = generated
        elif kwargs["model"] == "native-analysis":
            if "world-class planning expert" in texts[0]:
                output = "Inspect task feedback and identify one memory improvement."
            else:
                output = {"think": "Both synthetic tasks succeeded; current constraints remain relevant.", "tools": [{"name": "final_answer", "arguments": {"answer": "Retain context checks, and prioritize current corrections in extracted memories."}}]}
        elif "Extract reusable learnings from this successful execution." in prompt:
            output = {"strategic": ["Check current constraints before reusing prior decisions."], "operational": ["Read both the initial fixture and correction."]}
        elif "You are a memory guidance system." in prompt:
            output = {"selected_indices": [1], "guidance": "Consider checking present evidence."}
        elif "world-class planning expert" in texts[0]:
            output = "Read all listed fixtures, then compare current evidence and constraints."
        else:
            route = "Document IDs: route-rules, route-current" in prompt
            documents = ["route-rules", "route-current"] if route else ["release-initial", "release-update"]
            if "Results for tool call 'read_fixture'" not in prompt:
                output = {"think": "Read the required documents.", "tools": [{"name": "read_fixture", "arguments": {"document_id": item}} for item in documents]}
            else:
                output = {"think": "Use the current evidence.", "tools": [{"name": "final_answer", "arguments": {"answer": {"decision": "BETA" if route else "HOLD", "evidence": documents, "reason": "The current correction or cargo restriction determines the result."}}}]}
        text = output if isinstance(output, str) else json.dumps(output)
        return ChatCompletion.model_validate({"id": "offline", "created": 0, "model": "offline-model", "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text, "reasoning_content": ""}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
def offline_model(native, values, output, snapshot, label, **kwargs):
    model = native.OpenAIServerModel(model_id=label, api_key="offline-never-network", max_tokens=4096)
    model.client = live_smoke.AuditedClient(FakeClient(), output / "api_calls.jsonl", snapshot.items, label=label)
    return model
live_smoke.make_model = offline_model
summary = live_smoke.run_tasks(run, {"DEFAULT_MODEL": "offline-model"})
assert summary["correct_count"] == 2, summary
assert all(item["memory_ingested"] and item["ingestion_api_calls"] > 0 for item in summary["tasks"]), summary
calls = [json.loads(line) for line in (run / "live_smoke/tasks/api_calls.jsonl").read_text(encoding="utf-8").splitlines()]
assert any(call["orientation_present_in_request"] for call in calls)
import openai
openai.OpenAI = lambda **kwargs: FakeClient()
from MemEvolve.phases.memory_creator import MemorySystemCreator
original_create = MemorySystemCreator.create_memory_system
def create_with_crlf(config, base_dir):
    result = original_create(config, base_dir)
    provider = Path(base_dir) / "EvolveLab/providers" / (config["provider_code"]["module_name"] + ".py")
    text = provider.read_text(encoding="utf-8")
    provider.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))
    return result
MemorySystemCreator.create_memory_system = staticmethod(create_with_crlf)
evolution = live_smoke.run_evolution(run, {"OPENAI_API_KEY": "offline-never-network", "ANALYSIS_MODEL": "offline-analysis", "GENERATION_MODEL": "offline-generation"})
MemorySystemCreator.create_memory_system = staticmethod(original_create)
assert evolution["generated_code_executed"] is False, evolution
assert evolution["orientation_changed"] is True, evolution
assert evolution["source_changed_vs_parent"] is True, evolution
assert Path(evolution["candidate_source"]).is_file()
provider_bytes = (run / "workspace/EvolveLab/providers/offline_candidate_provider.py").read_bytes()
assert b"\r\n" in provider_bytes
assert Path(evolution["candidate_source"]).read_bytes() == provider_bytes
assert evolution["candidate_code_sha256"] == hashlib.sha256(provider_bytes).hexdigest()
assert evolution["generated_text_sha256"] != evolution["candidate_code_sha256"]
original_run_tasks = live_smoke.run_tasks
live_smoke.run_tasks = lambda *args, **kwargs: {"review_gate_passed": True, "provider": kwargs["provider_id"]}
reviewed = live_smoke.run_candidate(run, {}, evolution["candidate_code_sha256"])
assert reviewed == {"review_gate_passed": True, "provider": "offline_candidate"}, reviewed
live_smoke.run_tasks = original_run_tasks
class EmptyClient(FakeClient):
    calls = 0
    def create(self, **kwargs):
        self.calls += 1
        return ChatCompletion.model_validate({"id": "offline-empty", "created": 0, "model": "offline", "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": None, "reasoning_content": "budget consumed"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 4096, "total_tokens": 4106}})
empty_client = EmptyClient()
empty_model = native.OpenAIServerModel(model_id="deepseek-flash", api_key="offline-never-network")
empty_model.client = live_smoke.AuditedClient(empty_client, run / "empty-call.jsonl", [], label="offline-empty")
try:
    empty_model([{"role": "user", "content": "empty transport test"}])
except live_smoke.EmptyModelContentError:
    pass
else:
    raise AssertionError("empty content was accepted")
assert empty_client.calls == 1, empty_client.calls
print("OFFLINE_NATIVE_SMOKE_OK")
'''
        with tempfile.TemporaryDirectory() as temporary:
            script_path = Path(temporary) / "offline.py"
            script_path.write_text(script, encoding="utf-8")
            project = Path(__file__).resolve().parents[1]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(project)
            env["PYTHONUTF8"] = "1"
            result = subprocess.run([sys.executable, str(script_path), str(Path(temporary) / "runs")],
                                    cwd=project, env=env, capture_output=True, text=True, encoding="utf-8", timeout=90)
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-10000:])
        self.assertIn("OFFLINE_NATIVE_SMOKE_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
