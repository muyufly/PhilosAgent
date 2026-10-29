"""Enforce the native creator's actual postcondition before validation."""

from functools import wraps
import json
from pathlib import Path

from .evidence import append_event


def guard_creation(original):
    @wraps(original)
    def create(self, *args, **kwargs):
        config_path = kwargs.get("config_path") or (args[0] if args else None)
        if config_path is None:
            config_path = self.state["phases"]["generate"]["output"]
        config = json.loads(Path(config_path).read_text(encoding="utf8"))
        expected = config["memory_type_info"]["enum_value"]
        result = original(self, *args, **kwargs)
        created = result.get("created", [])
        actual_success = bool(result.get("success")) and expected in created
        if not actual_success:
            errors = [item.get("error", "Unknown creation failure") for item in result.get("failed", [])]
            result = {**result, "success": False,
                      "error": "; ".join(errors) or f"Creation returned no artifact for {expected}"}
            phase = self.state["phases"]["create"]
            phase.update(completed=False, error=result["error"])
            # Keep prior successfully created artifacts visible in the state;
            # an empty current attempt must not erase their provenance.
            aggregate = Path(self.work_dir) / "created_system.json"
            if aggregate.exists():
                phase["created_systems"] = json.loads(aggregate.read_text(encoding="utf8")).get("created", [])
            self._save_state()
        append_event(Path(self.work_dir) / "creation_contract.jsonl", {
            "event": "native_creation_postcondition", "provider": expected,
            "success": actual_success, "created_this_attempt": created,
            "failed": result.get("failed", []), "error": result.get("error"),
        })
        return result
    return create
