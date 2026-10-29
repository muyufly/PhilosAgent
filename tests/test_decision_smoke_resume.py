"""Offline checks for exact cached replay and the narrowly scoped compression."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from philoagent.decision_smoke_resume import (
    ReplayCompressClient, ReplayRefused, ReplayState, compress_messages,
)


def _audit(attempt, content):
    request = {
        "messages": [{"role": "user", "content": content}],
        "model": "deepseek-offline", "max_tokens": 4096, "temperature": 0,
    }
    paid = {**request, "extra_body": {"thinking": {"type": "disabled"}}}
    request_hash = hashlib.sha256(json.dumps(
        paid, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        "attempt": attempt, "label": "decision-evolve-analysis", "status": "success",
        "request": request, "request_sha256": request_hash,
        "usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
        "response": {"choices": [{
            "finish_reason": "stop", "message": {
                "role": "assistant", "content": "cached answer " + str(attempt),
                "reasoning_content": "cached reasoning",
            },
        }]},
    }


class FakeClient:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return "offline live response"


@pytest.fixture
def replay(tmp_path):
    audits = [_audit(19, "first analysis"), _audit(20, "second analysis")]
    source = 'def parent_policy():\n    return "frozen"\n'
    state = ReplayState(audits, source, "a" * 64, tmp_path / "events.jsonl")
    inner = FakeClient()
    outer = ReplayCompressClient(inner, state, "decision-evolve-analysis")
    return audits, state, inner, outer


def test_exact_replay_preserves_content_usage_without_dispatch(replay):
    audits, state, inner, outer = replay
    for audit in audits:
        response = outer.create(**deepcopy(audit["request"]))
        assert response.usage.total_tokens == audit["usage"]["total_tokens"]
        assert response.choices[0].message.content == audit["response"]["choices"][0]["message"]["content"]
        assert response.choices[0].message.reasoning_content == "cached reasoning"
    assert state.replayed == 2
    assert inner.calls == []
    events = [json.loads(line) for line in state.event_path.read_text(encoding="utf-8").splitlines()]
    assert [event["source_attempt"] for event in events] == [19, 20]
    assert all(event["budget_increment"] == 0 and not event["network_dispatched"]
               and event["cache_replay_not_billable"] for event in events)


@pytest.mark.parametrize("field", ["model", "messages", "stop", "response_format", "max_tokens"])
def test_changed_request_is_refused_without_dispatch(replay, field):
    audits, state, inner, outer = replay
    request = deepcopy(audits[0]["request"])
    changes = {
        "model": "another-model", "messages": [{"role": "user", "content": "changed"}],
        "stop": ["changed-stop"], "response_format": {"type": "json_object"},
        "max_tokens": 100,
    }
    request[field] = changes[field]
    with pytest.raises(ReplayRefused):
        outer.create(**request)
    assert state.replayed == 0
    assert inner.calls == []


def test_summary_repayment_is_refused(replay):
    _, state, inner, _ = replay
    client = ReplayCompressClient(inner, state, "decision-evolve-trajectory-summary")
    with pytest.raises(ReplayRefused):
        client.create(model="deepseek-offline", messages=[])
    assert inner.calls == []


def test_live_request_reaches_inner_with_only_source_compressed(replay):
    audits, state, inner, outer = replay
    for audit in audits:
        outer.create(**deepcopy(audit["request"]))
    messages = [{"role": "user", "content": [
        "Phi: worldview and methodology retained",
        {"text": "TRAIN: observed feedback retained\n" + state.source},
    ]}]
    preserved = deepcopy(messages)
    result = outer.create(model="deepseek-offline", messages=messages, max_tokens=4096)
    assert result == "offline live response"
    assert len(inner.calls) == 1
    sent = inner.calls[0]["messages"]
    assert sent[0]["content"][0] == preserved[0]["content"][0]
    assert sent[0]["content"][1]["text"] == "TRAIN: observed feedback retained\n" + state.replacement
    assert messages == preserved
    assert state.live_requests == 1


def test_recursive_compression_preserves_phi_training_and_structure():
    source = "COMPLETE FROZEN SOURCE"
    data = {"messages": [{"content": "Phi intact\n" + source}],
            "training": ("feedback intact", {"source": source}), "count": 2}
    original = deepcopy(data)
    changed, count = compress_messages(data, source, "P frozen: sha256=abc")
    assert count == 2
    assert changed == {
        "messages": [{"content": "Phi intact\nP frozen: sha256=abc"}],
        "training": ("feedback intact", {"source": "P frozen: sha256=abc"}), "count": 2,
    }
    assert data == original
    with pytest.raises(ValueError):
        compress_messages(data, "", "replacement")
