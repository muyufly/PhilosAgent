r"""Read-only regressions over the preserved r00 run; no credentials or network.

Run ``.venv\Scripts\python.exe -m pytest tests/test_reflection_regression.py -q -s``.
The historical run is optional and these checks skip when its artifacts are not
available. Temporary copies reproduce budget refusal without changing that run.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from philoagent.decision_smoke_resume import compress_messages
from philoagent.smoke_budget import Budget, BudgetRefused, wrap_native_model


HISTORICAL_RUN = (Path(__file__).resolve().parents[1] / "runs" /
                  "philosophy-system-smoke-20261029-r00")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _prompt(messages):
    texts = []
    for message in messages:
        content = message.get("content", "")
        texts.append(content if isinstance(content, str) else
                     "\n".join(item.get("text", "") for item in content))
    return "\n".join(texts)


def _serialized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


@pytest.fixture
def historical():
    run = HISTORICAL_RUN
    required = [
        "decision_smoke/budget/budget.json",
        "decision_smoke/budget/attempts/0001.json",
        "decision_smoke/budget/attempts/0002.json",
        "decision_smoke/budget/attempts/0019.json",
        "decision_smoke/budget/attempts/0020.json",
        "decision_smoke/evolve/failure.json",
        "decision_smoke/evolve/input_manifest.json",
        "decision_smoke/train/task_logs/train_01.json",
        "workspace/EvolveLab/providers/lightweight_memory_provider.py",
    ]
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        pytest.skip("Optional r00 historical artifacts unavailable: " + ", ".join(missing))
    source_path = run / required[-1]
    original = _read(run / "decision_smoke/evolve/input_manifest.json")
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == original["parent_code_sha256"]
    return {
        "run": run,
        "source": source_path.read_text(encoding="utf-8"),
        "phi": original["parent_orientation_snapshot"]["items"],
        "audits": {number: _read(run / "decision_smoke/budget/attempts" / f"{number:04d}.json")
                   for number in (1, 2, 19, 20)},
        "failure": _read(run / "decision_smoke/evolve/failure.json"),
    }


def test_recorded_reflection_prompt_repeats_full_provider_source(historical):
    audits, source = historical["audits"], historical["source"]
    counts = {number: _prompt(audits[number]["request"]["messages"]).count(source)
              for number in (19, 20)}
    assert counts == {19: 1, 20: 2}
    request = deepcopy(audits[20]["request"])
    request["extra_body"] = {"thinking": {"type": "disabled"}}
    serialized = _serialized(request)
    assert hashlib.sha256(serialized).hexdigest() == audits[20]["request_sha256"]
    exact_reservation = len(serialized) + request["max_tokens"] + Budget.safety_overhead
    assert exact_reservation == audits[20]["reserved_tokens"]
    print(json.dumps({
        "regression": "repeated_frozen_provider_source", "source_occurrences": counts,
        "attempt_20_reserved_tokens": exact_reservation,
        "attempt_20_actual_tokens": audits[20]["usage"]["total_tokens"],
    }))


def test_source_projection_preserves_philosophy_and_reduces_recorded_input(historical):
    original = historical["audits"][20]["request"]["messages"]
    projected, count = compress_messages(original, historical["source"], "[P frozen; full source omitted]")
    assert count == 2
    assert historical["source"] not in _prompt(projected)
    assert all(item in _prompt(projected) for item in historical["phi"])
    assert len(_serialized(projected)) < len(_serialized(original)) / 2
    # Exact substitution also ensures no other task/feedback text was removed.
    assert _prompt(projected) == _prompt(original).replace(historical["source"], "[P frozen; full source omitted]")
    print(json.dumps({
        "regression": "source_only_projection", "source_occurrences_removed": count,
        "messages_bytes_before": len(_serialized(original)),
        "messages_bytes_after": len(_serialized(projected)),
        "phi_items_retained": len(historical["phi"]),
    }))


def test_recorded_large_prompt_no_longer_fits_later_remaining_allowance(historical, tmp_path, monkeypatch):
    """Reproduce refusal of the paid #20 shape at the saved stop balance.

    The next refused prompt was never admitted/audited, so this deliberately
    does not claim to reconstruct that unrecorded request exactly.
    """
    run = historical["run"]
    original_bytes = (run / "decision_smoke/budget/budget.json").read_bytes()
    failure = historical["failure"]["budget"]
    ledger = json.loads(original_bytes)
    ledger["attempts"] = ledger["attempts"][:failure["attempt_count"]]
    ledger["actual_tokens"] = sum(item["usage"]["total_tokens"] for item in ledger["attempts"])
    ledger["blocked_reason"] = None
    ledger["unknown_usage_attempts"] = 0
    assert ledger["actual_tokens"] == failure["actual_tokens"]
    assert ledger["limits"] == {"max_calls": 100, "max_tokens": 300000}
    temporary_budget = tmp_path / "budget"
    temporary_budget.mkdir()
    (temporary_budget / "budget.json").write_text(json.dumps(ledger), encoding="utf-8")
    dispatches = []

    class NeverDispatch:
        def with_options(self, **options):
            assert options == {"timeout": 90, "max_retries": 0}
            return self

        @property
        def chat(self):
            return SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            dispatches.append(kwargs)
            raise AssertionError("Historical budget refusal must precede network dispatch")

    monkeypatch.setenv("PHILO_SMOKE_THINKING", "disabled")
    budget = Budget(temporary_budget, max_calls=100, max_tokens=300000)
    model = SimpleNamespace(model_id=ledger["model"], client=NeverDispatch())
    wrap_native_model(model, budget, SimpleNamespace(items=historical["phi"]), "offline-regression", 4096)
    with pytest.raises(BudgetRefused, match="Token reservation"):
        model.client.chat.completions.create(**historical["audits"][20]["request"])
    assert dispatches == []
    assert budget.summary()["attempt_count"] == failure["attempt_count"]
    assert budget.summary()["actual_tokens"] == failure["actual_tokens"]
    assert (run / "decision_smoke/budget/budget.json").read_bytes() == original_bytes
    print(json.dumps({
        "regression": "large_prompt_refused_before_dispatch", "network_calls": 0,
        "spent_at_stop": failure["actual_tokens"],
        "remaining_at_stop": ledger["limits"]["max_tokens"] - failure["actual_tokens"],
        "recorded_prompt_reservation": historical["audits"][20]["reserved_tokens"],
        "unrecorded_refused_request_reconstructed": False,
    }))


def test_cold_start_synthesis_is_distinct_from_experience_memory_and_phi(historical):
    first, planning = (historical["audits"][number] for number in (1, 2))
    prompt = _prompt(first["request"]["messages"])
    candidates = prompt.split("**Available Memories:**", 1)[1].split("**Your Tasks:**", 1)[0]
    candidate_count = len(re.findall(r"^\d+\. \[(?:STRATEGIC|OPERATIONAL)\]", candidates, re.MULTILINE))
    assert candidate_count == 7
    assert "trial" not in candidates.lower()
    assert not all(item in prompt for item in historical["phi"])
    synthesis = json.loads(first["response"]["choices"][0]["message"]["content"])
    assert synthesis["selected_indices"] == [1, 2, 5]
    assert "trial" in synthesis["guidance"].lower()
    planning_prompt = _prompt(planning["request"]["messages"])
    assert synthesis["guidance"] in planning_prompt
    assert all(item in planning_prompt for item in historical["phi"])
    print(json.dumps({
        "regression": "retrieval_synthesis_provenance", "cold_start_candidates": candidate_count,
        "selected_indices": synthesis["selected_indices"],
        "phi_visible_to_synthesis": False, "phi_visible_to_planner": True,
        "trial_guidance_origin": "on_demand_synthesis_from_task_and_cold_start_memories",
        "is_recorded_prior_episode": False,
    }))
