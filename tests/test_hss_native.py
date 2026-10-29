"""Native HSS closure tests; all SDK traffic uses an in-process fake transport."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from philoagent.hss_evaluation import RUBRICS, native_quality_summary, validate_grade
from philoagent.hss_native import (
    ContinuousQualityAggregator, _remap_storage_config, _terminal_storage,
)


def _grade(score=4):
    return validate_grade({"dimensions": {name: {"score": score, "reason": "Grounded in the submitted work."}
                                         for name in RUBRICS["humanities"]},
                           "overall_comment": "The report develops a supported interpretation."}, "humanities")


def test_native_aggregation_preserves_fractional_quality(tmp_path):
    directory = tmp_path / "_quality"
    directory.mkdir()
    result = native_quality_summary([
        {"status": "completed", "grade": _grade(1.25), "total_model_tokens": 10},
        {"status": "completed", "grade": _grade(4.5), "total_model_tokens": 20},
    ])
    (directory / "summary.json").write_text(json.dumps(result), encoding="utf-8")
    assert ContinuousQualityAggregator(tmp_path).aggregate()["summary"]["accuracy"] == .575
    result["success"] = False
    (directory / "summary.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="complete continuous"):
        ContinuousQualityAggregator(tmp_path).aggregate()


def test_terminal_storage_is_independent_and_keeps_trained_memory(tmp_path):
    workspace, frozen = tmp_path / "workspace", tmp_path / "frozen"
    (workspace / "storage" / "parent").mkdir(parents=True)
    (frozen / "parent").mkdir(parents=True)
    for directory in (workspace / "storage" / "parent", frozen / "parent"):
        (directory / "memory.json").write_text('["trained"]', encoding="utf-8")
    config = {"storage_dir": "./storage/parent", "memory_path": str(workspace / "storage/parent/memory.json"),
              "filename": "memory.json", "prompt": "ordinary non-path instructions"}
    original = lambda _: config
    runner = SimpleNamespace(get_memory_config=original)
    previous = Path.cwd()
    for number in range(2):
        snapshot = tmp_path / f"snapshot{number}"
        with _terminal_storage(workspace, frozen, snapshot, runner):
            mapped = runner.get_memory_config("parent")
            assert Path.cwd() == snapshot
            assert json.loads(Path(mapped["memory_path"]).read_text(encoding="utf-8")) == ["trained"]
            Path(mapped["memory_path"]).write_text('["retrieval mutation"]', encoding="utf-8")
            assert mapped["filename"] == "memory.json"
    assert runner.get_memory_config is original
    assert Path.cwd() == previous
    assert json.loads((workspace / "storage/parent/memory.json").read_text(encoding="utf-8")) == ["trained"]
    assert _remap_storage_config(config, workspace, tmp_path / "elsewhere")["prompt"] == config["prompt"]


def _offline_native_worker(root):
    """Executed in a fresh interpreter; retain actual native code and SDK audit."""
    import openai
    from openai.types.chat import ChatCompletion
    from unittest.mock import patch
    from philoagent.ablation_runtime import _CONTEXT
    from philoagent.cli import init_run
    from philoagent.hss_native import run_job

    generated = [0]
    transport_requests = []

    def candidate_response():
        generated[0] += 1
        number = generated[0]
        enum, cls = f"mock_candidate_{number}", f"MockCandidate{number}"
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
        items = [MemoryItem(id="trained", content="Use supporting material and examine alternatives.", metadata={{"count": self.count}})]
        return MemoryResponse(memories=items, memory_type=self.memory_type, total_count=len(items))
    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple:
        self.config["model"]([{{"role": "user", "content": "Extract writing experience."}}])
        self.count += 1
        with open(self.path, "w", encoding="utf-8") as stream:
            json.dump({{"count": self.count}}, stream)
        return True, "Stored writing experience"
'''
        return (f"**Class Name**: {cls}\n**Module Name**: {enum}_provider\n"
                f"**Enum Name**: {enum.upper()}\n**Enum Value**: {enum}\n"
                f"```python\n{code.rstrip()}\n```\n**storage_dir**: ./storage/{enum}\n"
                f'**orientation_items**: ["Revise interpretations against specific textual evidence, revision {number}."]\n')

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def with_options(self, **kwargs):
            return self

        def create(self, **kwargs):
            context = dict(_CONTEXT.get())
            role, provider = context.get("role"), context.get("provider", "")
            transport_requests.append({"role": role, "max_tokens": kwargs.get("max_tokens", kwargs.get("max_completion_tokens"))})
            tools = kwargs.get("tools")
            message = {"role": "assistant", "content": "A completed, evidence-based report."}
            if role == "generation":
                message["content"] = candidate_response()
            elif role == "judge":
                number = int(provider.rsplit("_", 1)[-1]) if provider.startswith("mock_candidate_") else 0
                message["content"] = json.dumps(_grade(3.25 + .15 * number))
            elif role in ("actor", "analysis"):
                # This pinned FlashOAgents revision implements its tool calls
                # as JSON text in content rather than the SDK tool_calls field.
                message["content"] = json.dumps({"think": "The work is ready.", "tools": [{
                    "id": "offline-final", "name": "final_answer",
                    "arguments": {"answer": "材料支持这一解释，并需认真检验竞争解释。"}}]}, ensure_ascii=False)
            else:
                message["content"] = json.dumps({"strategic": ["Ground interpretations in the actual supplied material."],
                                                 "operational": [], "selected_indices": [1],
                                                 "guidance": "Use direct evidence and examine alternatives.",
                                                 "key_extracts": [], "step_summary": "Completed the writing task."})
            return ChatCompletion.model_validate({"id": f"offline-{len(transport_requests)}", "object": "chat.completion",
                "created": 1, "model": "offline-fake", "choices": [{"index": 0, "message": message,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}})

    root.mkdir(parents=True, exist_ok=True)
    initialized = init_run("offline-full", "dialectical", output_root=root)
    run = Path(initialized["run_dir"])
    learning = run / "learning.jsonl"
    heldout = root / "heldout.jsonl"
    for path, count, split in ((learning, 8, "learning"), (heldout, 4, "holdout")):
        path.write_text("".join(json.dumps({"task_id": f"{split}-{index}", "domain": "humanities",
            "question": f"根据材料分析人物命运和叙述方式。材料：人物面对选择{index}。", "split": split},
            ensure_ascii=False) + "\n" for index in range(count)), encoding="utf-8")
    os.environ.update(OPENAI_API_KEY="offline-fake-credential", DEFAULT_MODEL="offline-fake",
                      ANALYSIS_MODEL="offline-fake", GENERATION_MODEL="offline-fake",
                      DEFAULT_JUDGE_MODEL="offline-fake", WEB_SEARCH_PROVIDER="deepseek", WEB_ACCESS_PROVIDER="deepseek",
                      HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
    protocol = {"model": "offline-fake", "max_calls": 60000, "max_tokens": 350000000, "actor_max_steps": 32,
                "actor_max_output_tokens": 8192, "analysis_max_steps": 20,
                "analysis_max_output_tokens": 8192, "generation_max_output_tokens": 60000,
                "validation_max_output_tokens": 32768, "thinking": "provider-default"}
    job = {"job_id": "offline-full", "code_evolve": True, "orientation_evolve": True,
           "learning_path": str(learning), "heldout_path": str(heldout), "group": "full"}
    with patch.object(openai, "OpenAI", FakeOpenAI), patch.object(openai, "Client", FakeOpenAI):
        result = run_job(run, job, protocol, root / "budget")
    audit = [json.loads(line) for line in (run / "api_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["status"] == "complete"
    assert result["rounds_completed"] == 2 and result["dataset_cursor"] == 8
    assert generated[0] == 4
    assert result["terminal"]["training_storage_preserved"]
    assert result["terminal"]["heldout_ingestion"] is False
    assert result["winner"].startswith("mock_candidate_")
    assert len(audit) == len(transport_requests)
    assert all(event["status"] == "success" for event in audit)
    assert {"actor", "judge", "analysis", "generation", "memory_ingestion"} <= {event.get("role") for event in audit}
    assert all(event["max_tokens"] < 350000000 for event in audit)
    assert any(round["winner_code_changed"] for round in result["rounds"])
    assert any(round["winner_orientation_items_changed"] for round in result["rounds"])
    heldout_records = [json.loads(path.read_text(encoding="utf-8")) for path in (run / "terminal/task_logs").glob("*/*.json")]
    assert len(heldout_records) == 4
    assert all(record["memory_ingestion_complete"] is False for record in heldout_records)
    return result


def test_real_native_two_rounds_with_fake_openai_in_fresh_process(tmp_path):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--offline-native", str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=120,
                            env={**os.environ, "PYTHONUTF8": "1", "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
    assert result.returncode == 0, (result.stdout + "\n" + result.stderr)[-30000:]
    assert '"status": "complete"' in result.stdout


if __name__ == "__main__" and sys.argv[1:2] == ["--offline-native"]:
    output = _offline_native_worker(Path(sys.argv[2]).resolve())
    print(json.dumps({"status": output["status"], "rounds_completed": output["rounds_completed"]}))
