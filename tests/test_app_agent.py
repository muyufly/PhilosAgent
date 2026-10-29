"""Offline facade tests over real immutable orientations and copied runtimes."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from philoagent.orientation import load
from philoagent_app import Agent, AgentBusyError, AgentStateError
from philoagent_app.storage import copy_runtime, file_lock, read_json, runtime_fingerprint, write_json
from philoagent_app.systems import normalize_task, parse_system, read_tasks


@pytest.mark.parametrize("filename,content", [
    ("system.md", "# 对象观\r\n关系与条件。\r\n\r\n# 自我修正\r\n保留反证。\r\n"),
    ("system.txt", "保留原文，包括结尾换行。\n"),
    ("system.json", '{"schema_version":1,"items":["对象与关系","可修订认识"]}'),
])
def test_import_is_lossless_and_native_compatible(tmp_path, filename, content):
    agent = Agent.create_from_text(tmp_path / "agent", filename, content)
    system = agent.info()["system"]
    original = agent.workspace / "system_source" / filename
    assert original.read_bytes() == content.encode("utf-8")
    expected = json.loads(content)["items"] if filename.endswith("json") else [content]
    assert system["items"] == expected
    assert load(agent.workspace / "lineage", "lightweight_memory").items == tuple(expected)
    assert read_json(agent.workspace / "initial" / "state.json")["provider"] == "lightweight_memory"
    assert (agent.workspace / "adapter_snapshot" / "philoagent" / "hss_tasks_runtime.py").is_file()
    with pytest.raises(FileExistsError):
        Agent.create_from_text(agent.workspace, "replacement.txt", "Do not overwrite learned memory")
    assert original.read_bytes() == content.encode("utf-8")


@pytest.mark.parametrize("filename,text", [("x.pdf", "not text"), ("x.md", "  "),
    ("x.json", "[]"), ("x.json", '{"schema_version":true,"items":["x"]}'),
    ("x.json", '{"schema_version":1,"items":[]}')])
def test_invalid_import_cannot_create_a_workspace(tmp_path, filename, text):
    with pytest.raises(ValueError):
        Agent.create_from_text(tmp_path / "agent", filename, text)
    assert not (tmp_path / "agent").exists()


def test_protects_existing_experiments_and_sources():
    root = Path(__file__).resolve().parents[1]
    for name in ("runs", "vendor", "philoagent", "philoagent_app"):
        with pytest.raises(ValueError, match="protected"):
            Agent(root / name / "do-not-create")


@pytest.fixture
def agent(tmp_path):
    return Agent.create_from_text(tmp_path / "agent", "critical.md", "全文进入当前哲学状态。\n")


TASK = {"domain": "business", "question": "用给定材料写一份决策备忘录。"}


def test_idempotency_and_phase_visibility_preserve_partial_work(agent, monkeypatch):
    calls = []
    def invoke(operation, payload, operation_dir=None, runtime_root=None):
        calls.append((operation, payload))
        record = {"task_id": payload["task"]["task_id"], "answer": "Already paid for this report",
                  "status": "judge_failed", "memory_ingestion_complete": False}
        write_json(operation_dir / "output" / "1.json", record)
        return {"status": "failed", "record": record, "operation": operation, "operation_id": operation_dir.name}
    monkeypatch.setattr(agent, "_invoke", invoke)
    result = agent.run(TASK, request_id="same-request")
    assert agent.run(TASK, request_id="same-request") == result
    assert len(calls) == 1
    info = agent.info()
    assert info["tasks"][0]["answer"] == "Already paid for this report"
    assert info["tasks"][0]["record"]["status"] == "judge_failed"
    assert info["last_operation"]["status"] == "failed"
    with pytest.raises(ValueError, match="different operation"):
        agent.run({**TASK, "question": "Another task"}, request_id="same-request")


def test_interrupted_request_is_never_replayed_and_ambiguous_ingestion_blocks_new_learning(agent, monkeypatch):
    task = normalize_task(TASK)
    from philoagent_app.storage import digest
    directory = agent.workspace / "operations" / "interrupted"
    write_json(directory / "request.json", {"operation": "run", "payload": {"task": task},
               "identity": digest({"operation": "run", "payload": {"task": task}})})
    write_json(directory / "output" / "1.json", {"answer": "saved", "memory_ingestion_started": True,
                                                  "memory_ingestion_complete": False})
    monkeypatch.setattr(agent, "_invoke", lambda *a, **kw: pytest.fail("must not dispatch"))
    assert agent.run(TASK, request_id="interrupted")["status"] == "interrupted"
    with pytest.raises(AgentStateError, match="memory ingestion"):
        agent.run({**TASK, "question": "new task"})


def test_process_lock_covers_an_orphaned_worker(agent):
    with file_lock(agent.workspace / ".worker.lock"):
        assert agent.info()["busy"]
        with pytest.raises(AgentBusyError):
            agent.run(TASK)
    assert not agent.info()["busy"]


def test_snapshot_copies_storage_and_rebases_config_without_touching_parent(agent):
    root = agent.workspace
    path = root / "workspace" / "storage" / "test" / "memory.json"
    write_json(path, {"experience": ["previous task"]})
    config = root / "workspace" / "snapshot_config.json"
    write_json(config, {"storage_path": str(path)})
    before = runtime_fingerprint(root)
    fork = copy_runtime(root, root / "operations" / "evaluation" / "runtime")
    mapped = Path(read_json(fork / "workspace" / "snapshot_config.json")["storage_path"])
    assert mapped.is_relative_to(fork)
    write_json(mapped, {"experience": ["evaluation must remain isolated"]})
    assert read_json(path)["experience"] == ["previous task"]
    assert runtime_fingerprint(root) == before
    assert not (fork / "budget").exists()


def test_evolution_input_requires_development_tasks(agent, monkeypatch):
    monkeypatch.setattr(agent, "_invoke", lambda *a, **kw: pytest.fail("must validate first"))
    with pytest.raises(ValueError, match="four"):
        agent.evolve([TASK])
    tasks = [{**TASK, "task_id": str(i), "split": "heldout"} for i in range(4)]
    with pytest.raises(ValueError, match="Evaluation"):
        agent.evolve(tasks)
    with pytest.raises(ValueError, match="Evaluation"):
        agent.run({**TASK, "split": "test"})


def test_unconfigured_evolution_does_not_block_all_future_learning(agent, monkeypatch):
    def unconfigured(*args, **kwargs):
        raise ValueError("Missing configuration: ANALYSIS_MODEL")
    monkeypatch.setattr(agent, "_invoke", unconfigured)
    tasks = [{**TASK, "task_id": str(i), "question": f"Development task {i}"} for i in range(4)]
    with pytest.raises(ValueError, match="Missing configuration"):
        agent.evolve(tasks, request_id="unconfigured")
    monkeypatch.setattr(agent, "_invoke", lambda *a, **kw: {"status": "completed"})
    assert agent.run(TASK, request_id="configured")["status"] == "completed"


def test_python_import_never_loads_native_clients_or_credentials(tmp_path):
    project = Path(__file__).resolve().parents[1]
    code = "import sys; from philoagent_app import Agent; assert 'FlashOAgents' not in sys.modules; assert 'dotenv' not in sys.modules; assert 'openai' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], cwd=project, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_task_criteria_and_nonfinite_material_validation():
    with pytest.raises(ValueError):
        normalize_task({**TASK, "task_criteria": "not a list"})
    with pytest.raises(ValueError):
        normalize_task({**TASK, "materials": [{"number": float("nan")} ]})
    with pytest.raises(ValueError, match="unique"):
        read_tasks([TASK, TASK])


def test_facade_frozen_adapter_worker_and_pair_evaluation_end_to_end(agent, tmp_path, monkeypatch):
    """Exercise the real facade/worker contract with the SDK replaced in child.

    The dispatcher alone is redirected into the existing offline worker harness;
    initialization, snapshots, child environment, native actor, grading, memory
    and report aggregation all use their real application implementations.
    """
    env_file = tmp_path / "offline.env"
    env_file.write_text("DEFAULT_MODEL=offline-fake\nOPENAI_API_KEY=offline-must-not-leak\n"
                        "ANALYSIS_MODEL=offline-fake\nGENERATION_MODEL=offline-fake\n"
                        "WEB_SEARCH_PROVIDER=deepseek\nWEB_ACCESS_PROVIDER=deepseek\n", encoding="utf-8")
    agent.env_file = env_file
    real_run = subprocess.run
    project = Path(__file__).resolve().parents[1]
    seen = []

    def offline_dispatch(command, **kwargs):
        assert command[2:4] == ["-m", "philoagent_app.worker"]
        root, request = command[5], command[7]
        seen.append((root, request))
        assert kwargs["env"]["PYTHONPATH"].split(__import__("os").pathsep)[0] == str(agent.workspace / "adapter_snapshot")
        assert kwargs["cwd"] == agent.workspace / "adapter_snapshot"
        return real_run([sys.executable, str(project / "tests" / "test_app_worker.py"),
                         "--offline-app", root, request, "normal"], **kwargs)

    monkeypatch.setattr("philoagent_app.agent.subprocess.run", offline_dispatch)
    learned = agent.run({"task_id": "development-real", "domain": "humanities", "question": "解释人物为何放弃捷径。"})
    assert learned["status"] == "completed", learned
    assert learned["record"]["memory_ingestion_complete"]
    transport = read_json(Path(seen[0][1]).parent / "transport.json")
    assert Path(transport["adapter_module_path"]).is_relative_to(agent.workspace / "adapter_snapshot")
    before = runtime_fingerprint(agent.workspace)
    report = agent.evaluate([{"task_id": "heldout-real", "domain": "humanities", "question": "比较叙事视角中的两种解释。"}])
    assert report["status"] == "complete", report
    assert report["paired"]["paired_tasks"] == 1
    assert report["api_usage"]["usage_complete"]
    assert runtime_fingerprint(agent.workspace) == before
    assert len(seen) == 3
    for arm in ("initial", "current"):
        result = report["tasks"][0]["arms"][arm]
        assert result["record"]["stage"] == "heldout"
        assert not result["record"]["memory_ingestion_complete"]
        assert result["api_usage"]["job_id"] != learned["api_usage"]["job_id"]
