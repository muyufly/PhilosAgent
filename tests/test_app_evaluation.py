"""Product evaluation: paired grades, held-out forks and honest partial reports."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from philoagent.hss_evaluation import RUBRICS, validate_grade
from philoagent_app import evaluation


def task(identifier="eval-1", question="Compare the supplied proposals.", domain="business"):
    return {"task_id": identifier, "domain": domain, "question": question,
            "materials": ["Proposal A costs 20; proposal B costs 30."],
            "deliverable": "A short decision memo.", "task_criteria": ["Use only supplied evidence."]}


def grade(domain, value):
    return validate_grade({"dimensions": {name: {"score": value, "reason": "Evidence-based assessment."}
                                          for name in RUBRICS[domain]},
                           "overall_comment": "Assessable work with a supported recommendation."}, domain)


def runtime(root, provider):
    root.mkdir(parents=True)
    (root / "run.json").write_text(json.dumps({"initial_provider": "initial-provider"}), encoding="utf-8")
    (root / "state.json").write_text(json.dumps({"provider": provider}), encoding="utf-8")
    (root / "lineage").mkdir()
    (root / "lineage" / "phi.json").write_text('{"items":["Compare alternatives"]}', encoding="utf-8")
    (root / "workspace" / "storage").mkdir(parents=True)
    (root / "workspace" / "storage" / "memory.txt").write_text("frozen memory", encoding="utf-8")


def fake_copy_runtime(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True)
    for name in ("run.json", "state.json", "lineage", "workspace"):
        current = source / name
        if current.is_dir():
            shutil.copytree(current, destination / name)
        elif current.exists():
            shutil.copyfile(current, destination / name)


class FakeAgent:
    def __init__(self, root, behavior=None):
        self.workspace = root
        self.calls = []
        self.behavior = behavior

    def _invoke(self, operation, payload, operation_dir=None, runtime_root=None):
        assert operation == "evaluate_task"
        assert runtime_root != self.workspace
        memory = runtime_root / "workspace" / "storage" / "memory.txt"
        # Retrieval-time writes from a previous task must never reach this one.
        assert memory.read_text(encoding="utf-8") == "frozen memory"
        memory.write_text("retrieval-time mutation", encoding="utf-8")
        self.calls.append((deepcopy(payload), operation_dir, runtime_root))
        if self.behavior is not None:
            return self.behavior(payload)
        quality = 2.5 if payload["provider"] == "initial-provider" else 4
        return {"status": "completed", "operation": operation, "provider": payload["provider"],
                "record": {"status": "completed", "answer": "A completed work.",
                           "grade": grade(payload["task"]["domain"], quality)},
                "api_usage": {"request_count": 2, "prompt_tokens": 80, "completion_tokens": 20,
                              "total_tokens": 100, "unknown_usage_attempts": 0,
                              "reserved_unknown_tokens": 0, "usage_complete": True}}


@pytest.fixture
def agent(tmp_path, monkeypatch):
    root = tmp_path / "agent"
    runtime(root, "current-provider")
    runtime(root / "initial", "initial-provider")
    monkeypatch.setattr(evaluation, "copy_runtime", fake_copy_runtime)
    return FakeAgent(root)


def output(agent):
    return agent.workspace / "operations" / "eval-001"


def request(agent, operation, payload):
    directory = agent.workspace / "operations" / "learning-001"
    directory.mkdir(parents=True)
    (directory / "request.json").write_text(json.dumps({"operation": operation, "payload": payload}), encoding="utf-8")


def test_paired_evaluation_forks_each_task_and_preserves_agent(agent):
    directory = output(agent)
    directory.mkdir(parents=True)
    parent_request = '{"operation":"evaluate","payload":{}}'
    (directory / "request.json").write_text(parent_request, encoding="utf-8")
    report = evaluation.evaluate(agent, [task(), task("eval-2", "Choose a suitable pilot.", "social")], directory)
    assert report["status"] == "complete"
    assert report["paired"]["paired_tasks"] == 2
    assert report["paired"]["mean_delta"] == 30
    assert report["arms"]["initial"]["mean_quality"] == 50
    assert report["arms"]["current"]["mean_quality"] == 80
    assert report["api_usage"]["total_tokens"] == 400
    assert report["heldout"] is True and report["heldout_ingestion"] is False
    assert len({call[2] for call in agent.calls}) == 4
    assert (agent.workspace / "workspace/storage/memory.txt").read_text() == "frozen memory"
    assert (agent.workspace / "initial/workspace/storage/memory.txt").read_text() == "frozen memory"
    assert (directory / "request.json").read_text() == parent_request
    assert json.loads((directory / "report.json").read_text(encoding="utf-8"))["paired"]["mean_delta"] == 30
    assert "当前 − 初始" in (directory / "report.md").read_text(encoding="utf-8")


def test_partial_failure_is_unpaired_and_unknown_usage_is_not_zero(agent):
    def behavior(payload):
        if payload["task"]["task_id"] == "eval-1" and payload["provider"] == "initial-provider":
            return {"status": "failed", "error_type": "APITimeoutError", "failure_stage": "actor",
                    "api_usage": {"request_count": 1, "usage_complete": False,
                                  "unknown_usage_attempts": 1, "reserved_unknown_tokens": 500}}
        quality = 2.5 if payload["provider"] == "initial-provider" else 4
        return {"status": "completed", "record": {"grade": grade(payload["task"]["domain"], quality)},
                "api_usage": {"request_count": 2, "prompt_tokens": 80, "completion_tokens": 20,
                              "total_tokens": 100, "unknown_usage_attempts": 0, "usage_complete": True}}
    agent.behavior = behavior
    report = evaluation.evaluate(agent, [task(), task("eval-2", "Assess the alternative proposal.")], output(agent))
    assert report["status"] == "incomplete"
    assert report["paired"]["paired_tasks"] == 1
    assert report["paired"]["mean_delta"] == 30
    assert report["arms"]["current"]["mean_quality"] == 80
    assert report["arms"]["initial"]["mean_quality"] == 50
    assert report["arms"]["initial"]["failed_tasks"] == 1
    assert report["tasks"][0]["quality_delta"] is None
    assert report["api_usage"]["confirmed_total_tokens"] == 300
    assert report["api_usage"]["total_tokens"] is None
    assert report["api_usage"]["reserved_unknown_tokens"] == 500
    assert report["api_usage"]["unknown_usage_runs"] == 1
    assert report["tasks"][0]["arms"]["initial"]["error_type"] == "APITimeoutError"


@pytest.mark.parametrize("damage", ["missing", "out_of_range", "total", "nan"])
def test_invalid_or_missing_grade_is_retained_without_zero_imputation(agent, damage):
    def behavior(payload):
        value = grade("business", 4)
        if damage == "missing":
            value = None
        elif damage == "total":
            value["quality_score"] = 99
        else:
            value["dimensions"]["argument"]["score"] = float("nan") if damage == "nan" else 9
        return {"status": "completed", "record": {"grade": value}, "api_usage": None}
    agent.behavior = behavior
    report = evaluation.evaluate(agent, [task()], output(agent))
    assert report["paired"]["paired_tasks"] == 0
    assert report["paired"]["mean_delta"] is None
    assert report["arms"]["current"]["mean_quality"] is None
    assert report["arms"]["current"]["failed_tasks"] == 1
    assert report["api_usage"]["confirmed_total_tokens"] is None
    assert report["api_usage"]["total_tokens"] is None
    result = report["tasks"][0]["arms"]["current"]
    assert result["status"] == "failed" and result["failure_stage"] == "grade_validation"
    assert result["missing_grade"] == (damage == "missing")
    json.dumps(report, allow_nan=False)
    json.loads((output(agent) / "report.json").read_text(encoding="utf-8"),
               parse_constant=lambda value: pytest.fail("Report is not strict JSON"))


def test_current_only_evaluation_does_not_require_an_initial_snapshot(agent):
    shutil.rmtree(agent.workspace / "initial")
    report = evaluation.evaluate(agent, [task()], output(agent), compare_initial=False)
    assert report["paired"] is None
    assert set(report["arms"]) == {"current"}
    assert len(agent.calls) == 1
    assert not (output(agent) / "frozen-initial").exists()


@pytest.mark.parametrize("matching", ["id", "content"])
def test_evaluation_rejects_learning_overlap_before_creating_snapshots(agent, matching):
    learned = task()
    request(agent, "run", {"task": learned})
    candidate = task(question="An unrelated task.") if matching == "id" else task("different-id")
    if matching == "content":
        candidate["question"] = "  Compare  the supplied\nproposals.  "
    with pytest.raises(ValueError, match="overlap learning history"):
        evaluation.evaluate(agent, [candidate], output(agent))
    assert not (output(agent) / "frozen-current").exists()
    assert not agent.calls


def test_evaluation_rejects_evolution_tasks_even_when_learning_failed(agent):
    request(agent, "evolve", {"tasks": [task()]})
    with pytest.raises(ValueError, match="overlap learning history"):
        evaluation.evaluate(agent, [task("new-id")], output(agent))


def test_corrupt_history_is_not_silently_ignored(agent):
    request(agent, "run", {"task": task()})
    path = agent.workspace / "operations/learning-001/request.json"
    path.write_text("incomplete JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot verify learning/evaluation separation"):
        evaluation.evaluate(agent, [task("new-id", "A different task")], output(agent))


def test_worker_exception_preserves_failure_and_continues(agent):
    def behavior(payload):
        if payload["task"]["task_id"] == "eval-1":
            raise RuntimeError("worker could not launch")
        return {"status": "completed", "record": {"grade": grade("business", 3)}, "api_usage": None}
    agent.behavior = behavior
    report = evaluation.evaluate(agent, [task(), task("eval-2", "Consider another question.")], output(agent))
    assert report["status"] == "incomplete"
    assert report["paired"]["paired_tasks"] == 1
    assert report["arms"]["current"]["mean_quality"] == 60
    assert report["tasks"][0]["arms"]["current"]["error_type"] == "RuntimeError"
    assert len(agent.calls) == 4


def test_real_runtime_copy_and_frozen_source_identity(agent, monkeypatch):
    from philoagent_app.storage import copy_runtime
    monkeypatch.setattr(evaluation, "copy_runtime", copy_runtime)
    report = evaluation.evaluate(agent, [task()], output(agent))
    assert report["status"] == "complete"
    for arm in ("initial", "current"):
        identity = report["snapshots"][arm]
        assert len(identity["source_runtime_sha256"]) == 64
        assert len(identity["runtime_sha256"]) == 64
        assert identity["source_runtime_preserved"] is True
    for _, directory, fork in agent.calls:
        assert directory.parent == fork / "operations"
    assert len({directory.name for _, directory, _ in agent.calls}) == 2


def test_duplicate_tasks_and_existing_output_are_refused(agent):
    with pytest.raises(ValueError, match="distinct IDs"):
        evaluation.evaluate(agent, [task(), task("another-id")], output(agent))
    evaluation.evaluate(agent, [task()], output(agent))
    with pytest.raises(FileExistsError, match="fresh output directory"):
        evaluation.evaluate(agent, [task()], output(agent))


def test_examples_are_disjoint_original_fixed_material_tasks():
    root = Path(__file__).resolve().parents[1] / "examples" / "agent"
    development = [json.loads(line) for line in (root / "development.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    heldout = [json.loads(line) for line in (root / "evaluation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(development) >= 4 and len(heldout) == 6
    assert {task["domain"] for task in heldout} == set(RUBRICS)
    assert all(sum(task["domain"] == domain for task in heldout) == 2 for domain in RUBRICS)
    assert not {task["task_id"] for task in development} & {task["task_id"] for task in heldout}
    assert not {evaluation.public_task_hash(task) for task in development} & {evaluation.public_task_hash(task) for task in heldout}
    assert all(task["materials"] and "仅使用" in task["deliverable"] for task in development + heldout)
