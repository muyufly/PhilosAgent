"""Preserve all scheduled competitors across interrupted native comparisons."""

from copy import deepcopy
from functools import wraps
import inspect
from pathlib import Path

from .evidence import append_event
from .hss_experiment import write_json, now


class HSSComparisonIncomplete(RuntimeError):
    """Paid comparison artifacts remain resumable; no winner is eligible yet."""


def guard_comparison(original):
    signature = inspect.signature(original)

    @wraps(original)
    def evaluate(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        providers = list(bound.arguments["providers"])
        if len(providers) != len(set(providers)):
            raise HSSComparisonIncomplete("Duplicate provider aliases cannot form independent competitors")
        checkpoint = bound.arguments.get("checkpoint")
        if checkpoint:
            checkpoint = deepcopy(checkpoint)
            checkpoint["eval_results"] = {name: value for name, value in checkpoint.get("eval_results", {}).items()
                                          if name in providers and value.get("success") is True}
            bound.arguments["checkpoint"] = checkpoint
        result = original(*bound.args, **bound.kwargs)
        missing = [name for name in providers if result.get(name, {}).get("success") is not True]
        directory = Path(bound.arguments["round_dir"])
        if missing:
            evidence = {"recorded_utc": now(), "providers": providers,
                        "task_indices": list(bound.arguments["tasks"]), "incomplete": missing,
                        "results": result, "winner_selection_allowed": False}
            write_json(directory / "_comparison_incomplete.json", evidence)
            append_event(directory / "comparison_attempts.jsonl", evidence)
            raise HSSComparisonIncomplete("Scheduled comparison incomplete: " + ", ".join(missing))
        return result
    return evaluate
