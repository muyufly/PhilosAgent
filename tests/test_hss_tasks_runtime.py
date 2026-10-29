"""Offline native callback tests: no dotenv reads or provider requests."""

from concurrent.futures import ThreadPoolExecutor
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from philoagent.ablation_runtime import _CONTEXT
from philoagent.hss_evaluation import RUBRICS
from philoagent.hss_tasks_runtime import HSSResumeRequired, NativeHSSRunner, task_prompt


def test_current_reserved_connection_error_and_success_keep_actual_and_reserved_cost_separate(tmp_path):
    reservation = {"method": "conservative_text_request_bound", "confirmed_actual_usage": False,
                   "original_receipt_sha256": "a" * 64, "reserved_unknown_tokens": 30000}
    events = [{"task_execution_id": "current", "attempt": 101, "status": "error",
               "error_type": "APIConnectionError", "usage": None,
               "reserved_unknown_tokens": 30000, "usage_reservation": reservation},
              {"task_execution_id": "current", "attempt": 102, "status": "success",
               "usage": {"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12},
               "recovered_attempts": [101]},
              {"task_execution_id": "other", "attempt": 103, "status": "error", "usage": None}]
    callback = object.__new__(NativeHSSRunner)
    callback.usage_reader = lambda: events
    usage = callback._usage("current")
    assert usage["total_tokens"] == 12 and usage["reserved_unknown_tokens"] == 30000
    assert usage["unknown_usage_attempts"] == 1
    assert usage["reserved_unknown_attempt_ids"] == [101]
    assert usage["usage_complete"] is False and usage["usage_accounted"] is True
    assert usage["failed_request_count"] == 1 and usage["unrecovered_failed_request_count"] == 0
    prior = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5,
             "request_count": 2, "failed_request_count": 1, "usage_complete": False,
             "usage_accounted": True, "reserved_unknown_tokens": 20000,
             "reserved_unknown_attempt_ids": [99], "unknown_usage_attempts": 1,
             "unrecovered_failed_attempt_ids": [], "unrecovered_failed_request_count": 0}
    record = {"task_execution_id": "current", "prior_api_usage": prior}
    callback._save(tmp_path / "task.json", record, time.monotonic())
    saved = json.loads((tmp_path / "task.json").read_text(encoding="utf8"))
    assert saved["api_usage"]["reserved_unknown_tokens"] == 50000
    assert saved["api_usage"]["unknown_usage_attempts"] == 2
    assert saved["api_usage"]["reserved_unknown_attempt_ids"] == [99, 101]
    assert saved["total_model_tokens"] == 17 and saved["effective_model_tokens"] == 50017
    assert saved["api_usage"]["usage_complete"] is False
    assert saved["api_usage"]["usage_accounted"] is True
    assert saved["api_usage"]["failed_request_count"] == 2


@pytest.mark.parametrize("damage", ["missing_receipt", "wrong_method", "known_usage", "bad_hash"])
def test_current_unknown_usage_cannot_gain_accounted_status_from_an_unsupported_reservation(damage):
    receipt = {"method": "conservative_text_request_bound", "confirmed_actual_usage": False,
               "original_receipt_sha256": "a" * 64, "reserved_unknown_tokens": 30000}
    event = {"task_execution_id": "current", "attempt": 101, "status": "error",
             "error_type": "APIConnectionError", "usage": None,
             "reserved_unknown_tokens": 30000, "usage_reservation": receipt}
    if damage == "missing_receipt": event.pop("usage_reservation")
    elif damage == "wrong_method": receipt["method"] = "unreviewed_guess"
    elif damage == "known_usage": event["usage"] = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
    else: receipt["original_receipt_sha256"] = "not-a-pinned-hash"
    callback = object.__new__(NativeHSSRunner)
    callback.usage_reader = lambda: [event]
    usage = callback._usage("current")
    if damage == "known_usage":
        assert usage["total_tokens"] == 5 and usage["reserved_unknown_tokens"] == 0
        assert usage["unknown_usage_attempts"] == 0
    else:
        assert usage["usage_accounted"] is False


def grade_output(domain="business", score=3.75):
    return json.dumps({"dimensions": {name: {"score": score, "reason": "Specific evidence supports this grade."}
                                      for name in RUBRICS[domain]},
                       "overall_comment": "The report addresses the task with concrete reasoning."})


@pytest.fixture
def harness(tmp_path):
    events, ingestions, agents, models, histories = [], [], [], [], {}
    state = SimpleNamespace(judge_output=grade_output(), ingest_success=True, actor_failure=False,
                            capture_failure=False, omit_usage=False, artifact_changed=False,
                            actor_answer="A substantive anonymous report.", actor_finish_reason=None,
                            actor_transport_recovered=False, actor_event_error=False)

    def event(tokens):
        item = {**_CONTEXT.get(), "status": "success"}
        if not state.omit_usage:
            item["usage"] = {"prompt_tokens": tokens - 1, "completion_tokens": 1, "total_tokens": tokens}
        events.append(item)
        return item

    class Model:
        def __init__(self, **config):
            self.config = config
            models.append(self)

        def __call__(self, messages, **kwargs):
            self.messages, self.kwargs = messages, kwargs
            event(3)
            return SimpleNamespace(content=state.judge_output)

    class Memory:
        def __init__(self, provider):
            self.provider = provider
            self.history = histories.setdefault(provider, [])

        def take_in_memory(self, trajectory):
            ingestions.append(trajectory)
            event(2)
            if state.ingest_success:
                self.history.append(trajectory.query)
            return state.ingest_success, "test result"

    class Agent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.memory_provider = kwargs["memory_provider"]
            agents.append(self)

        def run(self, prompt):
            self.prompt, self.history_before = prompt, list(self.memory_provider.history)
            item = event(7)
            if state.actor_finish_reason:
                item["finish_reason"] = state.actor_finish_reason
            if state.actor_event_error:
                item.update(status="error", attempt=231, error_type="EmptyModelContentError")
            if state.actor_transport_recovered:
                item.update(attempt=101, status="error", response_finish_reasons=["length"])
                recovered = event(4)
                recovered.update(attempt=102, recovered_attempts=[101], response_finish_reasons=["stop"])
            if state.actor_failure:
                raise RuntimeError("secret-key-must-not-enter-record")
            return state.actor_answer

        def write_memory_to_messages(self, **kwargs):
            assert kwargs == {"include_system_prompt": False}
            return [{"role": "assistant", "content": "native trajectory message"}]

    class Base:
        def __init__(self, model):
            self.model = model

        def capture_trajectory(self):
            if state.capture_failure:
                raise ValueError("capture failed")
            return {"agent_trajectory": [{"name": "action", "tool_calls": [], "obs": "evidence"}]}

    native = SimpleNamespace(OpenAIServerModel=Model, ToolCallingAgent=Agent,
                             WebSearchTool=lambda: "search", CrawlPageTool=lambda **kw: "fetch")
    runner = SimpleNamespace(load_memory_provider=lambda provider, model: Memory(provider),
                             TrajectoryData=lambda **kw: SimpleNamespace(**kw))
    callback = NativeHSSRunner(tmp_path / "run", tmp_path / "workspace", {},
                               {"model_id": "offline-model", "api_key": "never-called"},
                               runner=runner, native=native, base_agent_class=Base,
                               usage_reader=lambda: events)
    artifact_calls = []

    def artifact(provider):
        artifact_calls.append(provider)
        return {"provider": provider, "code_sha256": "changed" if state.artifact_changed and len(artifact_calls) > 1 else "fixed",
                "orientation_items_sha256": "phi"}

    callback._artifact = artifact
    tasks = [{"task_id": "WB1", "domain": "business", "question": "Write a business decision report.",
              "split": "learning", "group": "SECRET_GROUP", "phi": "SECRET_PHI"},
             {"task_id": "WB2", "domain": "business", "question": "Compare two growth strategies.", "split": "learning"}]
    dataset = tmp_path / "learning.jsonl"

    def save_tasks():
        dataset.write_text("\n".join(json.dumps(task) for task in tasks), encoding="utf-8")

    save_tasks()
    return SimpleNamespace(callback=callback, events=events, ingestions=ingestions, agents=agents,
                           models=models, histories=histories, state=state, tasks=tasks, dataset=dataset,
                           output=tmp_path / "run" / "evolution" / "round_1" / "base_logs",
                           save_tasks=save_tasks)


def run(h, indices=(0,), provider="parent", output=None):
    return h.callback("writingbench", provider, indices, h.dataset, output or h.output)


def record(h, index=0, output=None):
    return json.loads(((output or h.output) / f"{index + 1}.json").read_text(encoding="utf-8"))


def authorize_recovery(h, *, allow_actor=False, known_failed=()):
    """Represent root's explicit backed-up migration; never a callback shortcut."""
    h.callback.protocol.update(resume_incomplete_tasks=True, actor_recovery_max_attempts=1)
    saved = record(h)
    saved["task_identity"] = h.callback._task_identity(
        h.tasks[0], 0, "parent", h.callback._artifact("parent"), "learning", "writingbench")
    saved["recovery_authorized"] = {"revision": "repair-r2", "reason": "Reviewed paid failure",
                                    "allow_actor_retry": allow_actor,
                                    "known_failed_attempts": list(known_failed)}
    (h.output / "1.json").write_text(json.dumps(saved), encoding="utf-8")


def test_authorized_judge_resume_reuses_paid_answer_and_trajectory_across_processes(harness):
    h = harness
    h.state.judge_output = "invalid paid judge output"
    with pytest.raises(ValueError):
        run(h)
    original = record(h)
    authorize_recovery(h)
    h.events.clear()  # A replacement worker cannot see its predecessor's in-memory sink.
    h.state.judge_output = grade_output()
    run(h)
    saved = record(h)
    assert len(h.agents) == 1 and len(h.ingestions) == 1
    assert [event["role"] for event in h.events] == ["judge", "memory_ingestion"]
    assert saved["answer"] == original["answer"]
    assert saved["agent_messages"] == original["agent_messages"]
    assert saved["total_model_tokens"] == 15
    assert saved["api_usage"]["request_count"] == 4
    assert original["task_execution_id"] in saved["task_execution_ids"]
    assert len(list((h.output / "_recovery" / "1").glob("*.json"))) == 1
    assert saved["memory_ingestion_complete"]
    before = list(h.events)
    run(h)
    assert h.events == before and len(h.ingestions) == 1


def test_authorized_saved_grade_resumes_ingestion_without_rejudging(harness):
    h = harness
    h.state.judge_output = "invalid paid judge output"
    with pytest.raises(ValueError):
        run(h)
    from philoagent.hss_evaluation import validate_grade
    saved = record(h)
    saved.update(status="graded", grade=validate_grade(grade_output(), "business"))
    (h.output / "1.json").write_text(json.dumps(saved), encoding="utf-8")
    authorize_recovery(h)
    h.events.clear()
    run(h)
    assert [event["role"] for event in h.events] == ["memory_ingestion"]
    assert record(h)["total_model_tokens"] == 12


def test_old_paid_judge_format_is_repaired_without_a_new_judge_request(harness):
    h = harness
    h.state.judge_output = "invalid paid judge output"
    with pytest.raises(ValueError):
        run(h)
    saved = record(h)
    raw = grade_output()[:-1]
    saved["judge_output"] = raw
    (h.output / "1.json").write_text(json.dumps(saved), encoding="utf-8")
    authorize_recovery(h)
    h.events.clear()
    run(h)
    saved = record(h)
    assert len(h.agents) == 1
    assert [event["role"] for event in h.events] == ["memory_ingestion"]
    assert saved["judge_output"] == raw
    assert saved["judge_normalization"]["repaired"]
    assert saved["judge_normalization"]["repairs"] == [{"operation": "append_missing_object_closures", "count": 1}]
    assert saved["quality_score"] == 75
    assert saved["total_model_tokens"] == 12


def test_authorized_actor_retry_archives_failure_and_has_one_task_retry_limit(harness):
    h = harness
    h.state.actor_failure = True
    with pytest.raises(RuntimeError):
        run(h)
    authorize_recovery(h, allow_actor=True)
    h.events.clear()
    with pytest.raises(RuntimeError):
        run(h)
    assert record(h)["actor_recovery_attempts"] == 1
    assert record(h)["total_model_tokens"] == 14
    before = len(h.events)
    with pytest.raises(HSSResumeRequired, match="actor.*limit"):
        run(h)
    assert len(h.events) == before


def test_authorized_actor_retry_can_finish_with_cumulative_failed_work_cost(harness):
    h = harness
    h.state.actor_failure = True
    with pytest.raises(RuntimeError):
        run(h)
    authorize_recovery(h, allow_actor=True)
    h.events.clear()
    h.state.actor_failure = False
    run(h)
    saved = record(h)
    assert len(h.agents) == 2 and len(h.ingestions) == 1
    assert saved["total_model_tokens"] == 19
    assert saved["actor_recovery_attempts"] == 1


def test_known_old_actor_error_is_resolved_only_by_explicit_successful_phase_receipt(harness):
    h = harness
    h.state.actor_failure = h.state.actor_event_error = True
    with pytest.raises(RuntimeError):
        run(h)
    original = record(h)
    assert original["api_usage"]["unrecovered_failed_attempt_ids"] == [231]
    authorize_recovery(h, allow_actor=True)
    before = list(h.events)
    with pytest.raises(HSSResumeRequired, match="recovery links"):
        run(h)
    assert h.events == before
    authorize_recovery(h, allow_actor=True, known_failed=[231])
    h.events.clear()
    h.state.actor_failure = h.state.actor_event_error = False
    run(h)
    saved = record(h)
    assert saved["total_model_tokens"] == 19
    assert saved["api_usage"]["failed_request_count"] == 1
    assert saved["api_usage"]["unrecovered_failed_request_count"] == 0
    assert saved["phase_recovery_receipts"] == [{
        "phase": "actor", "successful_task_execution_id": saved["task_execution_id"],
        "recovered_attempts": [231], "authorization_revision": "repair-r2"}]
    archive = next((h.output / "_recovery" / "1").glob("*.json"))
    assert json.loads(archive.read_text(encoding="utf-8"))["api_usage"]["failed_request_count"] == 1


def test_ingestion_ambiguous_failure_cannot_be_replayed_after_authorization(harness):
    h = harness
    h.state.ingest_success = False
    with pytest.raises(RuntimeError):
        run(h)
    assert record(h)["memory_ingestion_started"] is True
    authorize_recovery(h)
    before = list(h.events)
    with pytest.raises(HSSResumeRequired, match="ingestion"):
        run(h)
    assert h.events == before and len(h.ingestions) == 1


def test_ingestion_complete_receipt_skips_memory_mutation_after_integrity_failure(harness):
    h = harness
    h.state.artifact_changed = True
    with pytest.raises(RuntimeError):
        run(h)
    original = record(h)
    assert original["memory_ingestion_complete"] and len(h.ingestions) == 1
    h.state.artifact_changed = False
    authorize_recovery(h)
    h.events.clear()
    run(h)
    assert not h.events
    assert len(h.agents) == 1 and len(h.ingestions) == 1
    assert record(h)["total_model_tokens"] == original["total_model_tokens"]
    assert record(h)["status"] == "completed"


def test_audited_truncation_is_saved_but_never_judged_or_ingested(harness):
    h = harness
    h.state.actor_finish_reason = "length"
    with pytest.raises(RuntimeError, match="truncat"):
        run(h)
    saved = record(h)
    assert saved["status"] == "actor_failed"
    assert saved["answer"] and saved["response_truncated"]
    assert saved["actor_output_issue"] == "audited_finish_reason_length"
    assert not h.ingestions
    assert [event["role"] for event in h.events] == ["actor"]


def test_empty_actor_output_has_explicit_failure_reason(harness):
    h = harness
    h.state.actor_answer = ""
    with pytest.raises(RuntimeError, match="empty"):
        run(h)
    assert record(h)["actor_output_issue"] == "empty_final_answer"
    assert not h.ingestions


def test_only_linked_successful_transport_recovery_makes_failed_request_eligible(harness):
    h = harness
    h.state.actor_transport_recovered = True
    run(h)
    saved = record(h)
    assert saved["status"] == "completed"
    assert saved["total_model_tokens"] == 16
    assert saved["api_usage"]["failed_request_count"] == 1
    assert saved["api_usage"]["unrecovered_failed_request_count"] == 0


def test_authorized_resume_still_rejects_unknown_prior_usage(harness):
    h = harness
    h.state.omit_usage = True
    with pytest.raises(RuntimeError):
        run(h)
    authorize_recovery(h)
    h.state.omit_usage = False
    before = list(h.events)
    with pytest.raises(HSSResumeRequired, match="Unknown"):
        run(h)
    assert h.events == before


def test_explicit_orphan_reservation_recovers_without_claiming_confirmed_usage(harness):
    h = harness
    h.state.actor_failure = h.state.omit_usage = True
    with pytest.raises(RuntimeError):
        run(h)
    authorize_recovery(h, allow_actor=True, known_failed=[700])
    saved = record(h)
    saved["api_usage"].update(usage_complete=False, usage_accounted=True,
                              reserved_unknown_tokens=20000, unknown_usage_attempts=1,
                              reserved_unknown_attempt_ids=[700], failed_request_count=1,
                              unrecovered_failed_attempt_ids=[700], unrecovered_failed_request_count=1)
    path = h.output / "1.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    before = list(h.events)
    with pytest.raises(HSSResumeRequired, match="Unknown"):
        run(h)
    assert h.events == before
    saved["recovery_authorized"]["reserved_unknown_attempts"] = [700]
    path.write_text(json.dumps(saved), encoding="utf-8")
    h.state.actor_failure = h.state.omit_usage = False
    h.events.clear()
    run(h)
    saved = record(h)
    assert saved["status"] == "completed"
    assert saved["total_model_tokens"] == saved["confirmed_model_tokens"] == 12
    assert saved["effective_model_tokens"] == 20012
    assert saved["api_usage"]["usage_complete"] is False
    assert saved["api_usage"]["usage_accounted"] is True
    assert saved["api_usage"]["reserved_unknown_tokens"] == 20000
    assert saved["api_usage"]["unknown_usage_attempts"] == 1
    assert saved["api_usage"]["unrecovered_failed_request_count"] == 0
    assert saved["phase_recovery_receipts"][0]["reserved_unknown_attempts"] == [700]


def test_real_native_interfaces_are_wired_with_persistent_learning_and_continuous_quality(harness):
    h = harness
    assert run(h, (0, 1)) == h.output
    assert h.agents[0].history_before == []
    assert h.agents[1].history_before == [h.tasks[0]["question"]]
    assert h.histories["parent"] == [task["question"] for task in h.tasks]
    for agent in h.agents:
        assert agent.kwargs["max_steps"] == 32
        assert agent.kwargs["tools"] == ["search", "fetch"]
    assert [model.config["max_tokens"] for model in h.models] == [32768, 8192] * 2
    first = record(h)
    assert first["status"] == "completed"
    assert first["quality_score"] == 75
    assert first["score"] == .75
    assert first["total_model_tokens"] == 12
    assert first["agent_trajectory"][0]["name"] == "action"
    assert first["agent_messages"][0]["role"] == "assistant"
    assert first["memory_ingestion_complete"]
    assert h.ingestions[0].metadata["is_correct"]
    assert h.ingestions[0].metadata["quality_grade"]["quality_score"] == 75
    assert h.ingestions[0].metadata["judge_comment"]
    summary = json.loads((h.output / "_quality" / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["accuracy"] == .75
    assert summary["summary"]["tokens"]["total_tokens"] == 24
    assert {path.name for path in h.output.glob("*.json")} == {"1.json", "2.json"}
    assert [event["role"] for event in h.events] == ["actor", "judge", "memory_ingestion"] * 2
    assert len({event["task_execution_id"] for event in h.events}) == 2


def test_official_query_and_judge_anonymity_are_preserved(harness):
    h = harness
    run(h)
    assert h.agents[0].prompt == h.tasks[0]["question"]
    judge_model = h.models[1]
    payload = json.dumps(judge_model.messages)
    assert "SECRET_GROUP" not in payload and "SECRET_PHI" not in payload
    assert "native trajectory message" not in payload
    assert "A substantive anonymous report." in payload
    assert judge_model.kwargs == {"response_format": {"type": "json_object"}}


@pytest.mark.parametrize("heldout_mode", ["explicit", "terminal"])
def test_evaluation_never_ingests_or_resets_provider_memory(harness, heldout_mode):
    h = harness
    h.histories["parent"] = ["Previously learned report"]
    h.tasks[0]["split"] = "evaluation"
    h.save_tasks()
    output = h.output
    if heldout_mode == "explicit":
        h.callback.heldout = True
    else:
        output = h.output.parents[2] / "terminal" / "task_logs"
    run(h, output=output)
    saved = record(h, output=output)
    assert saved["stage"] == "heldout"
    assert not saved["memory_evolution_enabled"]
    assert not saved["memory_ingestion_complete"]
    assert not h.ingestions
    assert h.agents[0].history_before == ["Previously learned report"]
    assert h.histories["parent"] == ["Previously learned report"]
    assert saved["total_model_tokens"] == 10


def test_completed_resume_reuses_answer_and_grade_without_any_new_requests(harness):
    h = harness
    run(h)
    before = list(h.events)
    before_bytes = (h.output / "1.json").read_bytes()
    run(h)
    assert h.events == before
    assert (h.output / "1.json").read_bytes() == before_bytes
    assert len(h.ingestions) == 1
    h.tasks[0]["question"] += " Changed task."
    h.save_tasks()
    with pytest.raises(HSSResumeRequired, match="differs"):
        run(h)
    assert h.events == before


@pytest.mark.parametrize("phase", ["judge", "ingestion", "trajectory", "actor", "integrity"])
def test_failures_preserve_paid_work_and_refuse_automatic_replay(harness, phase):
    h = harness
    if phase == "judge":
        h.state.judge_output = "invalid paid judge output"
    elif phase == "ingestion":
        h.state.ingest_success = False
    elif phase == "trajectory":
        h.state.capture_failure = True
    elif phase == "actor":
        h.state.actor_failure = True
    else:
        h.state.artifact_changed = True
    with pytest.raises((RuntimeError, ValueError)):
        run(h)
    saved = record(h)
    assert saved["status"] == f"{phase}_failed"
    assert saved["error_type"] in ("RuntimeError", "ValueError")
    assert "secret-key" not in json.dumps(saved)
    if phase != "actor":
        assert saved["answer"] == "A substantive anonymous report."
    if phase == "judge":
        assert saved["judge_output"] == "invalid paid judge output"
    if phase in ("ingestion", "integrity"):
        assert saved["grade"]["quality_score"] == 75
    event_count = len(h.events)
    with pytest.raises(HSSResumeRequired):
        run(h)
    assert len(h.events) == event_count
    assert not (h.output / "_quality" / "summary.json").exists()


def test_no_positive_score_shortcut_or_ingestion_success_threshold_in_native_fitness(harness):
    h = harness
    h.state.judge_output = grade_output(score=1.25)
    run(h)
    saved = record(h)
    assert saved["quality_score"] == 25
    assert saved["judgement"] == "incorrect"
    assert not h.ingestions[0].metadata["is_correct"]
    summary = json.loads((h.output / "_quality" / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["accuracy"] == .25


def test_unknown_usage_cannot_complete_as_zero_cost(harness):
    h = harness
    h.state.omit_usage = True
    with pytest.raises(RuntimeError, match="audited SDK usage"):
        run(h)
    saved = record(h)
    assert saved["status"] == "integrity_failed"
    assert not saved["api_usage"]["usage_complete"]
    assert saved["answer"]


def test_candidate_calls_are_scoped_independently_under_parallel_execution(harness):
    h = harness
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, h, (0,), provider, h.output.parent / provider) for provider in ("candidate_a", "candidate_b")]
        for future in futures:
            future.result()
    for provider in ("candidate_a", "candidate_b"):
        saved = record(h, output=h.output.parent / provider)
        assert saved["total_model_tokens"] == 12
        assert saved["api_usage"]["request_count"] == 3
        matching = [event for event in h.events if event["task_execution_id"] == saved["task_execution_id"]]
        assert all(event["provider"] == provider for event in matching)


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [True], [2]])
def test_invalid_batches_fail_before_any_paid_work(harness, indices):
    h = harness
    with pytest.raises(ValueError):
        run(h, indices)
    assert not h.events


@pytest.mark.parametrize("split", ["evaluation", "heldout", "holdout", "test"])
def test_evaluation_split_cannot_accidentally_be_used_for_learning(harness, split):
    h = harness
    h.tasks[0]["split"] = split
    h.save_tasks()
    with pytest.raises(ValueError, match="Evaluation data"):
        run(h)
    assert not h.events


def test_started_attempt_is_not_silently_retried(harness):
    h = harness
    run(h)
    saved = record(h)
    saved["status"] = "started"
    (h.output / "1.json").write_text(json.dumps(saved), encoding="utf-8")
    count = len(h.events)
    with pytest.raises(HSSResumeRequired):
        run(h)
    assert len(h.events) == count


def test_task_prompt_uses_supplied_public_material_without_run_metadata():
    prompt = task_prompt({"question": "Analyze a text.", "domain": "humanities",
                          "materials": [{"content": "A source passage."}], "deliverable": "A review.",
                          "phi": "SECRET"})
    assert prompt.startswith("Analyze a text.")
    assert "A source passage." in prompt and "A review." in prompt
    assert "SECRET" not in prompt


def test_audit_reader_is_required_without_importing_native_modules(tmp_path):
    with pytest.raises(ValueError, match="usage_reader"):
        NativeHSSRunner(tmp_path, tmp_path, {}, {})


def test_global_budget_is_never_used_as_per_request_output_or_step_limit(harness):
    h = harness
    h.callback.protocol.update(max_tokens=350_000_000, max_steps=999_999,
                               actor_max_output_tokens=8192, actor_max_steps=32,
                               judge_max_output_tokens=6000, judge_model="offline-judge",
                               quality_success_threshold=80)
    run(h)
    assert h.models[0].config["max_tokens"] == 8192
    assert h.models[1].config["max_tokens"] == 6000
    assert h.models[1].config["model_id"] == "offline-judge"
    assert h.agents[0].kwargs["max_steps"] == 32
    assert not h.ingestions[0].metadata["is_correct"]
    assert record(h)["quality_score"] == 75


def test_incomplete_later_record_blocks_new_earlier_task_before_spending(harness):
    h = harness
    run(h, (1,))
    saved = record(h, 1)
    saved["status"] = "started"
    (h.output / "2.json").write_text(json.dumps(saved), encoding="utf-8")
    before = len(h.events)
    with pytest.raises(HSSResumeRequired):
        run(h, (0, 1))
    assert len(h.events) == before
    assert not (h.output / "1.json").exists()


def test_stale_paid_task_logs_are_preserved_and_cannot_contaminate_another_batch(harness):
    h = harness
    run(h, (0,))
    paid_bytes = (h.output / "1.json").read_bytes()
    before = len(h.events)
    with pytest.raises(HSSResumeRequired, match="Unexpected saved"):
        run(h, (1,))
    assert len(h.events) == before
    assert (h.output / "1.json").read_bytes() == paid_bytes
