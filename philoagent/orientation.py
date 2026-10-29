"""Immutable, candidate-scoped method memory for the native outer evolution loop.

``root`` is a run's lineage directory, not its episodic-memory directory.  Task
actors may read snapshots; only bootstrap and candidate creation write them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ENV_ROOT = "PHILO_LINEAGE_ROOT"
MISSING = object()
_PROVIDER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_RESERVED_IDS = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
}
_FIELDS = {
    "schema_version", "initialized", "provider_id", "parent_provider_id",
    "parent_sha256", "items", "sha256",
}


class OrientationError(ValueError):
    """Invalid or conflicting orientation state."""


class OrientationValidationError(OrientationError):
    """Input does not satisfy the orientation schema."""


class OrientationConflictError(OrientationError):
    """An immutable namespace already contains a different snapshot."""


class OrientationIntegrityError(OrientationError):
    """A saved snapshot is malformed or changed after it was captured."""


@dataclass(frozen=True)
class OrientationSnapshot:
    provider_id: str
    items: tuple[str, ...]
    sha256: str
    parent_provider_id: str | None
    parent_sha256: str | None
    path: Path
    schema_version: int = 1
    initialized: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return portable, JSON-safe state; the run path is kept separately."""
        return {
            "schema_version": self.schema_version,
            "initialized": self.initialized,
            "provider_id": self.provider_id,
            "parent_provider_id": self.parent_provider_id,
            "parent_sha256": self.parent_sha256,
            "items": list(self.items),
            "sha256": self.sha256,
        }


def _validate_id(provider_id: str) -> None:
    if (
        not isinstance(provider_id, str)
        or not _PROVIDER_ID.fullmatch(provider_id)
        or provider_id.casefold() in _RESERVED_IDS
    ):
        raise OrientationValidationError(f"Invalid provider id: {provider_id!r}")


def _items(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise OrientationValidationError("items must be a sequence of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise OrientationValidationError("Each item must be a nonempty string; use [] to clear")
    return result


def _resolved(path: Path) -> Path:
    value = str(path.resolve())
    # Windows may return the extended-length spelling while another process
    # creates the path. Normalize both spellings before containment checks.
    if os.name == "nt":
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
    return Path(value)


def _state_path(root: str | os.PathLike[str], provider_id: str) -> Path:
    _validate_id(provider_id)
    if not os.fspath(root):
        raise OrientationValidationError("The lineage root must not be empty")
    root_path = _resolved(Path(root).expanduser())
    namespace = root_path / provider_id
    path = namespace / "orientation.json"
    if namespace.is_symlink() or path.is_symlink():
        raise OrientationValidationError("Orientation namespaces must not use symbolic links")
    resolved_path = _resolved(path)
    if not resolved_path.is_relative_to(root_path):
        raise OrientationValidationError(
            f"Orientation path escapes the lineage root: {resolved_path} (root: {root_path})"
        )
    return path


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _new_snapshot(
    path: Path, provider_id: str, items: Sequence[str],
    parent: OrientationSnapshot | None = None,
) -> OrientationSnapshot:
    values = _items(items)
    payload = {
        "schema_version": 1,
        "initialized": True,
        "provider_id": provider_id,
        "parent_provider_id": parent.provider_id if parent else None,
        "parent_sha256": parent.sha256 if parent else None,
        "items": list(values),
    }
    return OrientationSnapshot(
        provider_id=provider_id, items=values, sha256=_digest(payload),
        parent_provider_id=payload["parent_provider_id"],
        parent_sha256=payload["parent_sha256"], path=path,
    )


def load(root: str | os.PathLike[str], provider_id: str) -> OrientationSnapshot:
    """Read and verify an existing snapshot. Missing state is never re-seeded."""
    path = _state_path(root, provider_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OrientationIntegrityError(f"Unreadable orientation state: {path}") from exc
    if not isinstance(data, dict) or set(data) != _FIELDS:
        raise OrientationIntegrityError(f"Unexpected orientation schema: {path}")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise OrientationIntegrityError("Unsupported orientation schema_version")
    if data["initialized"] is not True or data["provider_id"] != provider_id:
        raise OrientationIntegrityError("Orientation namespace or initialization mismatch")
    if not isinstance(data["items"], list):
        raise OrientationIntegrityError("Saved items must be a JSON list")
    try:
        values = _items(data["items"])
        parent_id = data["parent_provider_id"]
        parent_hash = data["parent_sha256"]
        if parent_id is None:
            if parent_hash is not None:
                raise OrientationValidationError("A root snapshot cannot have a parent hash")
        else:
            _validate_id(parent_id)
            if parent_id == provider_id:
                raise OrientationValidationError("A snapshot cannot parent itself")
            if not isinstance(parent_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", parent_hash):
                raise OrientationValidationError("Invalid parent snapshot hash")
    except OrientationValidationError as exc:
        raise OrientationIntegrityError(str(exc)) from exc
    payload = {key: value for key, value in data.items() if key != "sha256"}
    if data["sha256"] != _digest(payload):
        raise OrientationIntegrityError(f"Orientation hash mismatch: {path}")
    return OrientationSnapshot(
        provider_id=provider_id, items=values, sha256=data["sha256"],
        parent_provider_id=parent_id, parent_sha256=parent_hash, path=path,
    )


def _store_once(snapshot: OrientationSnapshot) -> Path:
    """Publish a complete file atomically, without replacing an existing file."""
    path = snapshot.path
    path.parent.mkdir(parents=True, exist_ok=True)
    _state_path(path.parent.parent, snapshot.provider_id)
    if path.exists():
        existing = load(path.parent.parent, snapshot.provider_id)
        if existing.to_dict() != snapshot.to_dict():
            raise OrientationConflictError(f"Orientation already initialized differently: {path}")
        return path
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=".orientation-", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(snapshot.to_dict(), temporary, ensure_ascii=False, sort_keys=True, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            # Unlike os.replace, link is atomic AND refuses to clobber a winner
            # if two candidate-creation processes reach the same namespace.
            os.link(temporary_path, path)
        except FileExistsError:
            existing = load(path.parent.parent, snapshot.provider_id)
            if existing.to_dict() != snapshot.to_dict():
                raise OrientationConflictError(f"Concurrent orientation conflict: {path}")
        return path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def bootstrap(
    root: str | os.PathLike[str], provider_id: str, items: Sequence[str],
) -> Path:
    """Initialize one root candidate once; identical retries are safe."""
    return _store_once(_new_snapshot(_state_path(root, provider_id), provider_id, items))


def fork(
    root: str | os.PathLike[str], parent_provider_id: str, child_provider_id: str,
    orientation_items: Sequence[str] | object = MISSING,
) -> Path:
    """Fork the current parent, replacing items only when explicitly supplied."""
    _validate_id(parent_provider_id)
    path = _state_path(root, child_provider_id)
    if child_provider_id.casefold() == parent_provider_id.casefold():
        raise OrientationValidationError("Parent and child need distinct namespaces")
    parent = load(root, parent_provider_id)
    values = parent.items if orientation_items is MISSING else _items(orientation_items)
    return _store_once(_new_snapshot(path, child_provider_id, values, parent))


def fork_from_snapshot(
    root: str | os.PathLike[str], parent_snapshot: Mapping[str, Any], child_provider_id: str,
    orientation_items: Sequence[str] | object = MISSING,
) -> Path:
    """Resume candidate creation only if the saved parent still matches this run."""
    if not isinstance(parent_snapshot, Mapping) or "provider_id" not in parent_snapshot:
        raise OrientationValidationError("A complete parent snapshot is required")
    parent = load(root, parent_snapshot["provider_id"])
    if parent.to_dict() != dict(parent_snapshot):
        raise OrientationConflictError("Saved parent snapshot differs from the active lineage")
    return fork(root, parent.provider_id, child_provider_id, orientation_items)


def prompt(root: str | os.PathLike[str], provider_id: str) -> str:
    """Render the current candidate's method memory, including a real empty state."""
    snapshot = load(root, provider_id)
    if not snapshot.items:
        return ""
    return "Current method memory (read-only during this task):\n" + "\n".join(
        f"- {item}" for item in snapshot.items
    )


def env_context(
    provider_id: str, *, environ: Mapping[str, str] | None = None,
) -> OrientationSnapshot | None:
    """Resolve an explicitly named candidate; absent env preserves legacy runs."""
    environment = os.environ if environ is None else environ
    if ENV_ROOT not in environment:
        return None
    return load(environment[ENV_ROOT], provider_id)


def context_prompt(provider_id: str) -> str:
    """Convenience environment entry point for existing provider integrations."""
    snapshot = env_context(provider_id)
    return "" if snapshot is None else prompt(snapshot.path.parent.parent, provider_id)


def assert_unchanged(snapshot: OrientationSnapshot) -> None:
    """Reject task-time deletion, replacement or semantic mutation of a snapshot."""
    try:
        current = load(snapshot.path.parent.parent, snapshot.provider_id)
    except (OSError, OrientationError) as exc:
        raise OrientationIntegrityError(f"Orientation changed during task: {snapshot.path}") from exc
    if current.to_dict() != snapshot.to_dict():
        raise OrientationIntegrityError(f"Orientation changed during task: {snapshot.path}")
