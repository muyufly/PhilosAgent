"""Saved generation recovery reuses native parsing with no paid transport."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from philoagent.cli import init_run
from philoagent.ablation_runtime import digest
from philoagent.hss_experiment import read_json, write_json
from philoagent.hss_generation_recovery import recover_saved_candidate
from philoagent.orientation import load


@pytest.fixture
def saved_candidate(tmp_path):
    experiment = tmp_path / "experiment"
    initialized = init_run("offline-fixed-r1", "dialectical", output_root=experiment / "jobs")
    job_dir = Path(initialized["run_dir"])
    round_dir = job_dir / "evolution/round_00"
    round_dir.mkdir(parents=True)
    job = {"job_id": job_dir.name, "code_evolve": True, "orientation_evolve": False}
    write_json(experiment / "manifest.json", {"jobs": [job], "protocol": {
        "generation_model": "offline-test", "creativity_index": .5}})
    write_json(experiment / "budget/budget.json", {
        "blocked_reason": "maintenance_repair_test", "pending": {}, "unknown_usage_attempts": 0})
    write_json(experiment / "supervisor.json", {"pid": -1})
    write_json(job_dir / "job_status.json", {"pid": -1, "status": "failed"})
    snapshot = load(job_dir / "lineage", "lightweight_memory")
    analysis = {"success": True, "agent_result": "A completed evidence-based analysis.",
                "stats": {"tasks": 2}, "orientation_snapshot": snapshot.to_dict()}
    analysis_path = round_dir / "analysis_report.json"
    write_json(analysis_path, analysis)
    state = {"phases": {"analyze": {"completed": True, "output": str(analysis_path),
        "default_provider": "lightweight_memory"}, "generate": {"completed": False},
        "create": {"completed": False}, "validate": {"completed": False}}}
    write_json(round_dir / "state.json", state)
    write_json(round_dir / "checkpoint.json", {"parent_provider": "lightweight_memory",
        "parent_orientation_sha256": snapshot.sha256, "step_completed": 1, "batch_indices": [0, 1]})
    response = (
        "**Class Name**: SavedProvider\n**Module Name**: saved_candidate_provider\n"
        "**Enum Name**: SAVED_CANDIDATE\n**Enum Value**: saved_candidate\n"
        "```python\nimport threading\nimport os\n"
        "raise RuntimeError('Generated code must not execute during recovery')\n"
        "class SavedProvider:\n    def save(self, a, b):\n        os.replace(a, b)\n```\n"
        "**storage_dir**: ./storage/saved_candidate\n"
        '**orientation_items**: ["A model-proposed revision that the fixed group must override."]\n')
    raw = round_dir / "llm_raw_output_20261029_183606.txt"
    raw.write_text("Temperature: 0.75\nModel: offline-test\nTimestamp: 20261029_183606\n"
                   + "=" * 80 + "\n\n" + response, encoding="utf-8")
    attempt_path = experiment / "budget/attempts/000123.json"
    write_json(attempt_path, {"attempt": 123, "job_id": job_dir.name, "role": "generation",
        "provider": "lightweight_memory", "round_dir": str(round_dir), "status": "success",
        "request_sha256": "paid-original-request", "usage": {
            "prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        "response": {"choices": [{"finish_reason": "stop", "message": {"content": response}}]}})
    return experiment, job, round_dir, raw, attempt_path, state, snapshot


def test_actual_native_generator_reuses_settled_response_without_sdk_or_code_execution(saved_candidate):
    experiment, job, round_dir, raw, attempt_path, before, snapshot = saved_candidate
    raw_before, attempt_before = raw.read_bytes(), attempt_path.read_bytes()
    script = r'''
import json, socket, openai, sys
from pathlib import Path
from philoagent.hss_generation_recovery import recover_saved_candidate
def forbidden(*args, **kwargs):
    raise AssertionError("Offline generation recovery attempted a real model/network request")
openai.OpenAI.__init__ = forbidden
socket.socket.connect = forbidden
result = recover_saved_candidate(Path(sys.argv[1]), sys.argv[2])
assert recover_saved_candidate(Path(sys.argv[1]), sys.argv[2]) == result
print(json.dumps(result))
'''
    completed = subprocess.run([sys.executable, "-X", "utf8", "-c", script, str(experiment), job["job_id"]],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    receipt = json.loads(completed.stdout.splitlines()[-1])
    assert receipt["new_model_requests"] == 0
    assert receipt["original_attempt"] == 123
    assert receipt["original_raw_sha256"] == digest(raw_before)
    assert receipt["original_attempt_usage"]["total_tokens"] == 300
    assert receipt["generated_code_executed"] is False
    config = read_json(round_dir / "generated_system_1.json")
    assert config["orientation_items"] == list(snapshot.items)
    assert config["_generation_metadata"] == {"temperature": .75, "model_id": "offline-test",
        "timestamp": "20261029_183606", "parent_provider": "lightweight_memory"}
    assert config["_raw_llm_output"].startswith("**Class Name**: SavedProvider")
    assert config["_orientation_parent"]["snapshot"] == snapshot.to_dict()
    assert config["_generation_recovery"]["native_creation_and_validation_required"]
    state = read_json(round_dir / "state.json")
    assert state["phases"]["generate"]["completed"]
    assert state["phases"]["create"] == before["phases"]["create"]
    assert state["phases"]["validate"] == before["phases"]["validate"]
    checkpoint = read_json(round_dir / "checkpoint.json")
    assert checkpoint["evolution_completed"]["analysis"]
    assert checkpoint["step_completed"] == 1
    assert raw.read_bytes() == raw_before
    assert attempt_path.read_bytes() == attempt_before
    assert read_json(round_dir / "_saved_generation_recovery/state.before.json") == before


def test_matching_explicit_unknown_usage_reservations_do_not_block_settled_candidate_reuse(saved_candidate):
    experiment, job, round_dir, _, _, _, _ = saved_candidate
    budget = read_json(experiment / "budget/budget.json")
    budget.update(unknown_usage_attempts=2, reserved_unknown_attempts=[121, 122], reserved_unknown_tokens=100000)
    write_json(experiment / "budget/budget.json", budget)
    script = (
        "import sys; from philoagent.hss_generation_recovery import recover_saved_candidate; "
        "r=recover_saved_candidate(sys.argv[1],sys.argv[2]); assert r['original_attempt']==123")
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", script, str(experiment), job["job_id"]],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr + result.stdout
    assert (round_dir / "generated_system_1.json").exists()


def test_second_candidate_reuses_paid_raw_with_other_jobs_running_and_preserves_first(saved_candidate):
    experiment, job, round_dir, raw, attempt_path, before_state, snapshot = saved_candidate
    first = round_dir / "generated_system_1.json"
    first_config = {"memory_type_info": {"enum_name": "PRIOR_CANDIDATE", "enum_value": "prior_candidate"},
                    "_generation_metadata": {"parent_provider": "lightweight_memory"},
                    "_orientation_parent": {"snapshot": snapshot.to_dict()}}
    write_json(first, first_config)
    first_bytes = first.read_bytes()
    before_checkpoint = read_json(round_dir / "checkpoint.json")
    before_checkpoint.update(selection_metadata={"must": "survive"}, evolution_completed={"analysis": True})
    write_json(round_dir / "checkpoint.json", before_checkpoint)
    state = deepcopy(before_state)
    state["phases"]["generate"] = {"completed": True, "output": str(first), "prior_metadata": "keep"}
    state["phases"]["create"] = {"completed": False, "keep": "creation-state"}
    write_json(round_dir / "state.json", state)
    budget = read_json(experiment / "budget/budget.json")
    budget.update(blocked_reason=None, pending={"999": {"pid": -1, "job_id": "other-job"}})
    write_json(experiment / "budget/budget.json", budget)
    write_json(experiment / "budget/attempts/000999.json", {"attempt": 999, "job_id": "other-job", "status": "pending"})
    script = r'''
import json, socket, openai, sys
from pathlib import Path
from philoagent.hss_generation_recovery import recover_saved_candidate
def forbidden(*args, **kwargs):
    raise AssertionError("Offline second-candidate recovery attempted network or model access")
openai.OpenAI.__init__ = forbidden
socket.socket.connect = forbidden
kwargs = dict(candidate_index=2, raw_path=Path(sys.argv[3]), recovery_scope="stopped_job",
              expected_existing_hashes={"generated_system_1.json":sys.argv[4]})
result=recover_saved_candidate(sys.argv[1],sys.argv[2],**kwargs)
assert recover_saved_candidate(sys.argv[1],sys.argv[2],**kwargs)==result
print(json.dumps(result))
'''
    completed = subprocess.run([sys.executable, "-X", "utf8", "-c", script, str(experiment), job["job_id"],
                                str(raw), digest(first_bytes)], cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    receipt = json.loads(completed.stdout.splitlines()[-1])
    assert receipt["candidate_index"] == 2 and receipt["new_model_requests"] == 0
    assert receipt["preserved_existing_candidates"] == {"generated_system_1.json": digest(first_bytes)}
    assert first.read_bytes() == first_bytes
    assert (round_dir / "generated_system_2.json").exists()
    after = read_json(round_dir / "state.json")
    assert after["phases"]["analyze"] == state["phases"]["analyze"]
    assert after["phases"]["create"] == state["phases"]["create"]
    assert after["phases"]["generate"]["prior_metadata"] == "keep"
    assert read_json(round_dir / "checkpoint.json")["selection_metadata"] == {"must":"survive"}
    assert read_json(round_dir / "saved_generation_recovery_2.json") == receipt


@pytest.mark.parametrize("damage", ["first_hash", "missing_first", "target_pending", "not_terminal", "no_explicit_raw"])
def test_second_candidate_isolation_guards_preserve_all_artifacts(saved_candidate, damage):
    experiment, job, round_dir, raw, _, _, _ = saved_candidate
    first = round_dir / "generated_system_1.json"
    write_json(first, {"existing": True})
    expected = {first.name: digest(first.read_bytes())}
    if damage == "first_hash": expected[first.name] = "wrong"
    elif damage == "missing_first": first.unlink()
    elif damage == "target_pending":
        budget = read_json(experiment / "budget/budget.json")
        budget["pending"] = {"999": {"job_id": job["job_id"]}}
        write_json(experiment / "budget/budget.json", budget)
        write_json(experiment / "budget/attempts/000999.json", {"attempt": 999, "job_id": job["job_id"]})
    elif damage == "not_terminal": write_json(round_dir.parent.parent / "job_status.json", {"status":"running", "pid":-1})
    before = {p.relative_to(round_dir):p.read_bytes()for p in round_dir.rglob('*')if p.is_file()}
    with pytest.raises(ValueError):
        recover_saved_candidate(experiment, job["job_id"], candidate_index=2,
                                raw_path=None if damage=="no_explicit_raw" else raw,
                                recovery_scope="stopped_job", expected_existing_hashes=expected)
    assert before == {p.relative_to(round_dir):p.read_bytes()for p in round_dir.rglob('*')if p.is_file()}


@pytest.mark.parametrize("damage", ["length", "unknown_usage", "response_mismatch", "pending",
                                  "parent_change", "extra_raw", "existing_generated"])
def test_ambiguous_or_incomplete_paid_artifacts_are_refused_before_any_recovery_writes(saved_candidate, damage):
    experiment, job, round_dir, raw, attempt_path, _, _ = saved_candidate
    attempt = read_json(attempt_path)
    if damage == "length":
        attempt["response"]["choices"][0]["finish_reason"] = "length"
    elif damage == "unknown_usage":
        attempt.pop("usage")
    elif damage == "response_mismatch":
        attempt["response"]["choices"][0]["message"]["content"] += " changed"
    elif damage == "pending":
        state = read_json(experiment / "budget/budget.json")
        state["pending"] = {"124": {"pid": -1}}
        write_json(experiment / "budget/budget.json", state)
    elif damage == "parent_change":
        analysis = read_json(round_dir / "analysis_report.json")
        analysis["orientation_snapshot"]["sha256"] = "changed"
        write_json(round_dir / "analysis_report.json", analysis)
    elif damage == "extra_raw":
        (round_dir / "llm_raw_output_other.txt").write_bytes(raw.read_bytes())
    elif damage == "existing_generated":
        write_json(round_dir / "generated_system_2.json", {"existing": True})
    write_json(attempt_path, attempt)
    before = {p.relative_to(round_dir): p.read_bytes() for p in round_dir.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        recover_saved_candidate(experiment, job["job_id"])
    after = {p.relative_to(round_dir): p.read_bytes() for p in round_dir.rglob("*") if p.is_file()}
    assert before == after
