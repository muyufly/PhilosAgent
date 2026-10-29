"""An interrupted comparison cannot discard competitors and select a winner."""

from types import SimpleNamespace

import pytest

from philoagent.hss_experiment import read_json


def test_incomplete_competitor_blocks_selection_and_retains_attempt(tmp_path):
    try:
        from philoagent.hss_comparison import guard_comparison, HSSComparisonIncomplete
    except ImportError:
        class HSSComparisonIncomplete(RuntimeError):
            pass
        guard_comparison = lambda original: original
    evaluated = []
    result = {"parent": {"success": True, "summary": {"quality_score": 86.3}},
              "child": {"success": False, "error": "Shared budget blocked: unknown_usage"}}

    def evaluate(self, providers, tasks, round_dir, checkpoint=None):
        evaluated.extend(providers)
        return result

    with pytest.raises(HSSComparisonIncomplete, match="child"):
        guard_comparison(evaluate)(SimpleNamespace(), ["parent", "child"], [0, 2, 3], tmp_path)
    assert evaluated == ["parent", "child"]
    assert read_json(tmp_path / "_comparison_incomplete.json")["results"] == result


def test_failed_checkpoint_entry_is_retried_and_successful_entry_reused(tmp_path):
    from philoagent.hss_comparison import guard_comparison
    original = {"eval_results": {"parent": {"success": True}, "child": {"success": False}}}

    def evaluate(self, providers, tasks, round_dir, checkpoint=None):
        assert checkpoint == {"eval_results": {"parent": {"success": True}}}
        return {"parent": {"success": True}, "child": {"success": True}}

    result = guard_comparison(evaluate)(SimpleNamespace(), providers=["parent", "child"],
                                       tasks=[0, 1], round_dir=tmp_path, checkpoint=original)
    assert all(item["success"] for item in result.values())
    assert original["eval_results"]["child"]["success"] is False


def test_duplicate_provider_names_cannot_masquerade_as_two_candidates(tmp_path):
    from philoagent.hss_comparison import guard_comparison, HSSComparisonIncomplete

    def forbidden(self, providers, tasks, round_dir, checkpoint=None):
        raise AssertionError("duplicate aliases reached native evaluation")

    with pytest.raises(HSSComparisonIncomplete, match="Duplicate"):
        guard_comparison(forbidden)(SimpleNamespace(), ["same", "same"], [0, 1], tmp_path)
