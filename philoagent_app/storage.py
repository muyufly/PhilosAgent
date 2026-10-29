"""Local state helpers. No experiment scheduler or model imports live here."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


PROJECT = Path(__file__).resolve().parent.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def assert_product_path(path):
    """A product runtime must never be an existing experiment or source tree."""
    path = Path(path).expanduser().resolve()
    if path == PROJECT or PROJECT.is_relative_to(path):
        raise ValueError("Choose a separate agent workspace, not the repository root")
    for name in ("philoagent", "philoagent_app", "vendor", "runs", ".venv", "tests", "seeds", "docs", "research", "data"):
        if path.is_relative_to(PROJECT / name):
            raise ValueError(f"Agent data cannot be written inside the protected {name} directory")
    return path


@contextmanager
def file_lock(path):
    """Nonblocking process lock, automatically released when a process exits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            raise BlockingIOError("This agent is already running a task or evolution") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def runtime_fingerprint(root):
    """Hash the state that must not be changed by independent evaluation."""
    root = Path(root)
    entries = []
    for name in ("run.json", "state.json", "workspace", "lineage"):
        base = root / name
        paths = [base] if base.is_file() else sorted(base.rglob("*")) if base.exists() else []
        for path in paths:
            if path.is_file() and not any(x in {"__pycache__", ".git"} for x in path.relative_to(root).parts):
                entries.append((path.relative_to(root).as_posix(), digest(path.read_bytes())))
    return digest(entries)


def _rebase_text(text, source, destination):
    replacements = []
    # Native generated configurations sometimes serialize absolute paths. Rebase
    # only the old runtime's own paths; preserve unrelated material verbatim.
    for old, new in ((source, destination),):
        replacements.extend(((str(old), str(new)), (old.as_posix(), destination.as_posix()),
                             (str(old).replace("\\", "\\\\"), str(new).replace("\\", "\\\\"))))
    for old, new in sorted(set(replacements), key=lambda pair: len(pair[0]), reverse=True):
        text = text.replace(old, new)
    return text


def copy_runtime(source, destination):
    """Copy all provider registrations, Phi and M into a fresh native runtime.

    Operation logs, credentials, budgets and the recursively nested initial
    snapshot do not belong in a fork. All storage bytes are copied per task.
    """
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Runtime fork already exists: {destination}")
    # Check before mkdir/copy so symlinks cannot make an evaluation write back
    # into the original runtime or an experiment directory.
    for name in ("workspace", "lineage"):
        tree = source / name
        if tree.is_symlink() or any(path.is_symlink() for path in tree.rglob("*")):
            raise ValueError("Runtime snapshots must not contain symbolic links")
    destination.mkdir(parents=True)
    for name in ("run.json", "state.json"):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    for name in ("workspace", "lineage"):
        shutil.copytree(source / name, destination / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".env", ".env.*", ".git"))
    for path in (destination / "workspace").rglob("*"):
        if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}:
            try:
                before = path.read_text(encoding="utf-8")
            except UnicodeError:
                continue
            after = _rebase_text(before, source, destination)
            if after != before:
                path.write_text(after, encoding="utf-8")
    return destination
