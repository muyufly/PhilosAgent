"""Expose a candidate's immutable orientation through existing memory interfaces.

This adapter never decides what to remember or how to evolve memory code.
The inner task loop cannot update orientation; only candidate creation can.
"""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

from .orientation import assert_unchanged, env_context


def orientation_text(items):
    if not items:
        return ""
    return (
        "Current revisable working principles (not facts or mandatory conclusions):\n"
        + "\n".join(f"- {item}" for item in items)
    )


def configure_provider(provider_id, config):
    """Return a new config; leave legacy invocations unchanged when not enabled."""
    snapshot = env_context(provider_id)
    result = dict(config)
    if snapshot is not None:
        result["philo_orientation_items"] = list(snapshot.items)
        result["philo_orientation_sha256"] = snapshot.sha256
        if provider_id == "lightweight_memory":
            result["enable_longterm_provision"] = True
    return result, snapshot


class OrientedProvider:
    """Delegate native memory operations, with state integrity and call evidence."""

    def __init__(self, provider, snapshot, code_path, loaded_code_sha256):
        self._provider = provider
        self._snapshot = snapshot
        self._code_path = Path(code_path).resolve()
        self._loaded_hash = loaded_code_sha256
        self._check()
        self._record("loaded")

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def _check(self):
        assert_unchanged(self._snapshot)
        actual = hashlib.sha256(self._code_path.read_bytes()).hexdigest()
        if actual != self._loaded_hash:
            raise RuntimeError("Provider source changed during a task/candidate evaluation")

    def _record(self, operation, **details):
        evidence_dir = os.environ.get("PHILO_EVIDENCE_DIR")
        if evidence_dir:
            from .evidence import record_runtime_call

            record_runtime_call(
                Path(evidence_dir) / f"calls-{os.getpid()}.jsonl",
                self._code_path,
                operation,
                orientation_path=self._snapshot.path,
                loaded_code_sha256=self._loaded_hash,
                details={"provider": self._snapshot.provider_id, **details},
            )

    def initialize(self):
        self._check()
        result = self._provider.initialize()
        self._check()
        self._record("initialize", success=bool(result))
        return result

    def provide_memory(self, request):
        from EvolveLab.memory_types import MemoryItem, MemoryItemType, MemoryStatus

        self._check()
        params = dict(request.additional_params or {})
        params["philo_orientation_items"] = list(self._snapshot.items)
        response = self._provider.provide_memory(replace(request, additional_params=params))
        self._check()
        if request.status == MemoryStatus.BEGIN and self._snapshot.items:
            item = MemoryItem(
                id="philo-orientation:" + self._snapshot.sha256,
                content=orientation_text(self._snapshot.items),
                metadata={"kind": "current_orientation", "sha256": self._snapshot.sha256},
                type=MemoryItemType.TEXT,
            )
            response = replace(response, memories=[item, *response.memories], total_count=len(response.memories) + 1)
        self._record(
            "provide_memory",
            phase=request.status.value,
            query_sha256=hashlib.sha256(request.query.encode("utf-8")).hexdigest(),
            memory_ids=[item.id for item in response.memories],
            response_sha256=hashlib.sha256(
                json.dumps([m.content for m in response.memories], ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
        )
        return response

    def take_in_memory(self, trajectory):
        self._check()
        metadata = dict(trajectory.metadata or {})
        metadata["philo_orientation_items"] = list(self._snapshot.items)
        result = self._provider.take_in_memory(replace(trajectory, metadata=metadata))
        self._check()
        self._record("take_in_memory", success=bool(result[0]))
        return result
