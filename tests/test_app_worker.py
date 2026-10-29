"""Product worker integration: actual native agents, local tools, fake SDK only."""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from philoagent.cli import init_run
from philoagent.hss_evaluation import RUBRICS
from philoagent_app.worker import _validate_request


PROJECT = Path(__file__).resolve().parents[1]
INITIAL_TEXT = "完整原文第一行。\n第二行保持证据与判断之间的区别。\n终行：材料不足时说明局限。"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _task(number=1, split="learning"):
    return {"task_id": f"task-{number}", "domain": "humanities", "split": split,
            "question": f"根据人物选择材料{number}，写一篇有证据的解释。", "materials": "人物拒绝捷径。",
            "task_criteria": ["比较两种竞争解释，并指出证据边界。"]}


@pytest.fixture
def agent_workspace(tmp_path):
    seed = tmp_path / "seed.json"
    _write(seed, {"schema_version": 1, "items": [INITIAL_TEXT]})
    root = Path(init_run("agent", "engineering", seed_file=seed, output_root=tmp_path)["run_dir"])
    _write(root / "state.json", {"provider": "lightweight_memory"})
    _write(root / "agent.json", {"initial_provider": "lightweight_memory"})
    return root


def _request(root, identifier, operation="run", **payload):
    path = root / "operations" / identifier / "request.json"
    _write(path, {"operation": operation, "operation_id": identifier, "payload": payload})
    return path


def _launch(root, request, mode="normal"):
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(PROJECT)}
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--offline-app",
                               str(root), str(request), mode], cwd=PROJECT,
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=120, env=env)
    result = _json(request.parent / "result.json")
    assert completed.returncode == (0 if result["status"] == "completed" else 1), (completed.stdout + completed.stderr)[-10000:]
    return result


def _manifest(path):
    return {str(item.relative_to(path)): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in path.rglob("*") if item.is_file()} if path.exists() else {}


def test_run_uses_native_tools_full_philosophy_and_persistent_memory(agent_workspace):
    root = agent_workspace
    first = _launch(root, _request(root, "first", task=_task()))
    assert first["status"] == "completed", first
    assert first["record"]["memory_ingestion_complete"] is True
    assert "离线检索证据" in json.dumps(first["record"]["agent_trajectory"], ensure_ascii=False)
    transport = _json(root / "operations/first/transport.json")
    assert transport["full_philosophy_in_actor"] is True
    assert transport["task_criteria_in_actor"] is True
    assert first["record"]["question"] == _task()["question"]
    assert transport["web_search_executions"] >= 1
    memory = root / "workspace/storage/lightweight_memory/longterm_memory.json"
    first_memory = _json(memory)
    second = _launch(root, _request(root, "second", task=_task(2)))
    assert second["status"] == "completed", second
    assert second["record"]["memory_ingestion_complete"] is True
    assert _json(memory) != first_memory
    assert second["api_usage"]["request_count"] < len((root / "api_calls.jsonl").read_text(encoding="utf-8").splitlines())
    ledger = _json(root / "budget/budget.json")
    assert ledger["max_inflight_requests"] == 1
    assert ledger["limits"] == {"max_calls": 60000, "max_tokens": 350000000}


@pytest.mark.parametrize("mode,stage", [("judge-failure", "judge"), ("ingestion-failure", "ingestion")])
def test_failures_preserve_paid_answer_and_do_not_replay(agent_workspace, mode, stage):
    root = agent_workspace
    request = _request(root, mode, task=_task())
    failed = _launch(root, request, mode)
    assert failed["status"] == "failed" and failed["failure_stage"] == stage, failed
    assert failed["record"]["answer"] == "材料支持这一解释，并需检验竞争解释。"
    assert not failed["record"]["memory_ingestion_complete"]
    assert "must-not-leak" not in (request.parent / "result.json").read_text(encoding="utf-8")
    count = len((root / "api_calls.jsonl").read_text(encoding="utf-8").splitlines())
    assert _launch(root, request, mode) == failed
    assert len((root / "api_calls.jsonl").read_text(encoding="utf-8").splitlines()) == count


def test_evaluation_fork_disables_ingestion_and_preserves_source(agent_workspace, tmp_path):
    root = agent_workspace
    assert _launch(root, _request(root, "learning", task=_task()))["status"] == "completed"
    before = _manifest(root)
    fork = tmp_path / "fork"
    shutil.copytree(root, fork)
    heldout = _launch(fork, _request(fork, "heldout", "evaluate_task", task=_task(5, "heldout")))
    assert heldout["status"] == "completed", heldout
    assert heldout["record"]["stage"] == "heldout"
    assert heldout["record"]["memory_ingestion_complete"] is False
    assert "memory_ingestion" not in heldout["api_usage"]["roles"]
    assert _manifest(root) == before


def test_one_native_evolution_round_adopts_complete_winner_and_next_run(agent_workspace):
    root = agent_workspace
    assert _launch(root, _request(root, "warmup", task=_task()))["status"] == "completed"
    parent_memory = root / "workspace/storage/lightweight_memory/longterm_memory.json"
    before = _json(parent_memory)
    evolved = _launch(root, _request(root, "evolve", "evolve", tasks=[_task(i) for i in range(4)], rounds=1))
    assert evolved["status"] == "completed", evolved
    assert evolved["rounds_completed"] == 1 and evolved["dataset_cursor"] == 4
    assert evolved["winner"].startswith("app_candidate_")
    assert _json(root / "state.json")["provider"] == evolved["winner"]
    assert (root / "lineage" / evolved["winner"] / "orientation.json").is_file()
    assert (root / "workspace/storage" / evolved["winner"] / "memory.json").is_file()
    assert (root / "operations/evolve/evolution/round_00/checkpoint.json").is_file()
    retained = {item["content"]: item for item in _json(parent_memory)["strategic"]}
    assert all(item["content"] in retained and retained[item["content"]]["usage_count"] >= item["usage_count"]
               for item in before["strategic"])
    follow = _launch(root, _request(root, "after-evolution", task=_task(10)))
    assert follow["status"] == "completed", follow
    assert follow["record"]["provider"] == evolved["winner"]
    assert follow["record"]["memory_ingestion_complete"] is True


def test_heldout_and_insufficient_evolution_tasks_fail_before_native_import(agent_workspace):
    root = agent_workspace
    request = _request(root, "leak", "evolve", tasks=[_task(i, "heldout") for i in range(4)])
    with pytest.raises(ValueError, match="cannot enter"):
        _validate_request(root, request)
    request = _request(root, "short", "evolve", tasks=[_task(i) for i in range(3)])
    with pytest.raises(ValueError, match="four development"):
        _validate_request(root, request)
    assert not (root / "budget").exists()


def test_interrupted_marker_blocks_replay_without_spending(agent_workspace):
    root = agent_workspace
    request = _request(root, "interrupted", task=_task())
    _write(request.parent / "worker_started.json", {"pid": 0})
    result = _launch(root, request)
    assert result["error_type"] == "InterruptedOperation"
    assert result["failure_stage"] == "interrupted"
    assert not (root / "budget").exists()


def _offline_worker(root, request, mode):
    """Fresh interpreter: patch the SDK transport, retain actual agent code."""
    import openai
    from openai.types.chat import ChatCompletion
    from unittest.mock import patch
    from philoagent.ablation_runtime import _CONTEXT
    from philoagent.hss_evaluation import validate_grade

    os.environ.update(OPENAI_API_KEY="offline-must-not-leak", DEFAULT_MODEL="offline-fake",
                      ANALYSIS_MODEL="offline-fake", GENERATION_MODEL="offline-fake",
                      DEFAULT_JUDGE_MODEL="offline-fake", WEB_SEARCH_PROVIDER="deepseek", WEB_ACCESS_PROVIDER="deepseek",
                      PHILO_LINEAGE_ROOT=str(root / "lineage"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(root / "workspace"))
    import FlashOAgents as native

    generated, calls, actors = [0], [], {}
    evidence = {"full_philosophy_in_actor": False, "task_criteria_in_actor": False, "web_search_executions": 0}

    def search(self, query):
        evidence["web_search_executions"] += 1
        return "离线检索证据：人物选择符合材料，存在竞争解释。"

    def candidate():
        generated[0] += 1
        number = generated[0]
        enum, cls = f"app_candidate_{number}", f"AppCandidate{number}"
        code = f'''import json
import os
from EvolveLab.base_memory import BaseMemoryProvider
from EvolveLab.memory_types import MemoryType, MemoryRequest, MemoryResponse, MemoryItem, TrajectoryData

class {cls}(BaseMemoryProvider):
    def __init__(self, config=None):
        super().__init__(MemoryType("{enum}"), config)
        self.storage_dir = self.config["storage_dir"]
        self.path = os.path.join(self.storage_dir, "memory.json")
    def initialize(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        self.count = 0
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as stream:
                self.count = json.load(stream)["count"]
        return True
    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        items = [MemoryItem(id="trained", content="Use supporting evidence and examine alternatives.", metadata={{"count": self.count}})]
        return MemoryResponse(memories=items, memory_type=self.memory_type, total_count=len(items))
    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple:
        self.config["model"]([{{"role": "user", "content": "Extract writing experience."}}])
        self.count += 1
        with open(self.path, "w", encoding="utf-8") as stream:
            json.dump({{"count": self.count}}, stream)
        return True, "Stored experience"
'''
        return (f"**Class Name**: {cls}\n**Module Name**: {enum}_provider\n**Enum Name**: {enum.upper()}\n"
                f"**Enum Value**: {enum}\n```python\n{code.rstrip()}\n```\n**storage_dir**: ./storage/{enum}\n"
                f'**orientation_items**: ["Examine evidence and alternatives, revision {number}."]\n')

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def with_options(self, **kwargs):
            return self

        def create(self, **kwargs):
            context = dict(_CONTEXT.get())
            role, provider = context.get("role"), context.get("provider", "")
            calls.append(role)
            message = {"role": "assistant", "content": "A completed report."}
            if role == "generation":
                message["content"] = candidate()
            elif role == "judge":
                requested_dimensions = set(json.loads(kwargs["messages"][-1]["content"])["rubric"])
                domain = next(name for name, dimensions in RUBRICS.items() if requested_dimensions == set(dimensions))
                score = 4.5 if provider.startswith("app_candidate_") else 3.25
                grade = validate_grade({"dimensions": {name: {"score": score, "reason": "Specific evidence supports this grade."}
                                                      for name in RUBRICS[domain]},
                                        "overall_comment": "Evidence and competing interpretations are addressed."}, domain)
                message["content"] = "malformed grade" if mode == "judge-failure" else json.dumps(grade)
            elif role in {"actor", "analysis"}:
                serialized = json.dumps(kwargs.get("messages", []), ensure_ascii=False)
                if role == "actor" and json.dumps(INITIAL_TEXT, ensure_ascii=False)[1:-1] in serialized:
                    evidence["full_philosophy_in_actor"] = True
                if role == "actor" and "比较两种竞争解释，并指出证据边界。" in serialized:
                    evidence["task_criteria_in_actor"] = True
                execution = context.get("task_execution_id")
                actors[execution] = actors.get(execution, 0) + int(role == "actor")
                if role == "actor" and actors[execution] <= 2:
                    tool, arguments = "web_search", {"query": "离线测试材料"}
                else:
                    tool, arguments = "final_answer", {"answer": "材料支持这一解释，并需检验竞争解释。"}
                message["content"] = json.dumps({"think": "Use the available evidence.", "tools": [
                    {"id": f"offline-{len(calls)}", "name": tool, "arguments": arguments}]}, ensure_ascii=False)
            else:
                message["content"] = json.dumps({"strategic": ["Ground interpretation in the supplied material."],
                    "operational": [], "selected_indices": [1], "guidance": "Examine evidence and alternatives.",
                    "key_extracts": [], "step_summary": "Completed the task."})
            return ChatCompletion.model_validate({"id": f"offline-{len(calls)}", "object": "chat.completion", "created": 1,
                "model": "offline-fake", "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}})

    from philoagent_app.worker import main
    from philoagent.runtime import OrientedProvider
    with ExitStack() as stack:
        stack.enter_context(patch.object(openai, "OpenAI", FakeOpenAI))
        stack.enter_context(patch.object(openai, "Client", FakeOpenAI))
        stack.enter_context(patch.object(native.WebSearchTool, "forward", search))
        if mode == "ingestion-failure":
            stack.enter_context(patch.object(OrientedProvider, "take_in_memory", side_effect=RuntimeError("offline-must-not-leak")))
        code = main(["--workspace", str(root), "--request", str(request)])
    _write(request.parent / "transport.json", {**evidence, "roles": calls,
           "adapter_module_path": str(Path(sys.modules["philoagent.hss_tasks_runtime"].__file__).resolve())})
    return code


if __name__ == "__main__" and sys.argv[1:2] == ["--offline-app"]:
    raise SystemExit(_offline_worker(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]))
