"""Local credential failover without exposing keys or changing experiments.

Workers share an active slot. Only an explicit provider balance/quota error
marks that slot exhausted; rate limits and transport failures never switch it.
The state file contains slot names only. Credentials remain in a local file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from uuid import uuid4


@dataclass(frozen=True)
class CredentialChoice:
    slot: str
    api_key: str = field(repr=False)


def is_quota_exhausted(error):
    """Inspect provider classification in memory; never retain its message."""
    if getattr(error, "status_code", None) != 402:
        return False
    body = getattr(error, "body", None)
    text = json.dumps(body, ensure_ascii=False).lower() if isinstance(body, (dict, list, str)) else ""
    markers = ("insufficient balance", "insufficient_balance", "insufficient quota", "insufficient_quota",
               "quota exhausted", "quota_exhausted", "balance exhausted", "balance_exhausted",
               "insufficient credit", "insufficient_credit", "余额不足", "额度耗尽", "额度不足")
    return any(marker in text for marker in markers)


class CredentialPool:
    def __init__(self, config_path, state_path):
        self.config_path = Path(config_path).resolve()
        self.state_path = Path(state_path).resolve()

    @classmethod
    def from_environment(cls, values=None):
        values = os.environ if values is None else values
        path, state = values.get("PHILO_HSS_CREDENTIAL_POOL"), values.get("PHILO_HSS_CREDENTIAL_STATE")
        if not path:
            return None
        if not state:
            raise ValueError("Credential pool requires a shared local state path")
        return cls(path, state)

    def _config(self):
        try:
            value = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            raise ValueError("Cannot read local credential pool") from None
        if not isinstance(value, dict) or value.get("version") != 1:
            raise ValueError("Invalid local credential pool schema")
        slots = value.get("slots")
        if (not isinstance(slots, dict) or not slots or set(slots) - {"A", "B"}
                or value.get("active_slot", "A") not in slots):
            raise ValueError("Credential pool must define slots A and/or B and an active slot")
        for slot in slots.values():
            if (not isinstance(slot, dict) or set(slot) != {"api_key"}
                    or not isinstance(slot["api_key"], str) or not slot["api_key"].strip()):
                raise ValueError("Credential slots must contain only a nonempty API key")
        return value

    def _state(self, config):
        if not self.state_path.exists():
            return {"version": 1, "active_slot": config.get("active_slot", "A"),
                    "exhausted_slots": [], "switch_count": 0}
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError("Cannot read shared credential routing state") from None
        if (not isinstance(state, dict) or state.get("version") != 1
                or state.get("active_slot") not in config["slots"]
                or not isinstance(state.get("exhausted_slots"), list)
                or any(slot not in {"A", "B"} for slot in state["exhausted_slots"])):
            raise ValueError("Invalid shared credential routing state")
        return state

    def select(self):
        config = self._config()
        state = self._state(config)
        slot = state["active_slot"]
        if slot in state["exhausted_slots"]:
            replacement = self.exhausted(slot)
            if replacement is None:
                raise RuntimeError("No usable credential remains; no unexhausted backup is configured")
            return replacement
        return CredentialChoice(slot, config["slots"][slot]["api_key"])

    def exhausted(self, slot):
        """Atomically mark an explicit quota failure and choose one backup.

        A late failure from the old slot cannot switch a new active slot back.
        None means all configured slots have explicit exhaustion evidence.
        """
        from .hss_budget import _file_lock

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock(self.state_path.with_suffix(".lock"), 30):
            config = self._config()
            state = self._state(config)
            if slot not in config["slots"]:
                raise ValueError("Unknown credential slot")
            exhausted = set(state["exhausted_slots"]) | {slot}
            active = state["active_slot"]
            previous_active = active
            if active in exhausted:
                active = next((name for name in sorted(config["slots"]) if name not in exhausted), active)
            state.update(active_slot=active, exhausted_slots=sorted(exhausted),
                         switch_count=state.get("switch_count", 0) + int(active != previous_active),
                         updated_unix=time.time(), reason="provider_explicit_402_quota_exhausted")
            temporary = self.state_path.with_name(f"{self.state_path.name}.{uuid4().hex}.tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.state_path)
            if active in exhausted:
                return None
            return CredentialChoice(active, config["slots"][active]["api_key"])
