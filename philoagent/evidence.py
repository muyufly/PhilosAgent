"""Conservative source-change and runtime evidence; never infer scientific causality."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any

_WRITE_LOCK = threading.Lock()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _RemoveDocstrings(ast.NodeTransformer):
    def _visit_scope(self, node: ast.AST) -> ast.AST:
        self.generic_visit(node)
        body = getattr(node, "body", [])
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:]
        return node

    visit_Module = _visit_scope
    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_ClassDef = _visit_scope


class _MaskStrings(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="<string-literal>"), node)
        if isinstance(node.value, bytes):
            return ast.copy_location(ast.Constant(value=b"<bytes-literal>"), node)
        return node


def classify_code_change(before: str, after: str) -> dict[str, Any]:
    """Classify syntax only; strings may implement real keys, queries, or prompts.

    A structural AST difference does not establish a changed executed path or a
    causal performance benefit. Docstrings are excluded for this code-mechanism
    diagnostic even though programs can introspect __doc__.
    """
    result: dict[str, Any] = {
        "before_sha256": hashlib.sha256(before.encode("utf-8")).hexdigest(),
        "after_sha256": hashlib.sha256(after.encode("utf-8")).hexdigest(),
        "causal_effect_established": False,
        "behavioral_change_established": False,
        "non_string_ast_changed": False,
    }
    if before == after:
        return {**result, "category": "unchanged", "requires_review": False}
    trees: list[ast.AST] = []
    errors: dict[str, str] = {}
    for label, text in (("before", before), ("after", after)):
        try:
            trees.append(_RemoveDocstrings().visit(ast.parse(text)))
        except (SyntaxError, ValueError) as exc:
            errors[label] = str(exc)
    if errors:
        return {**result, "category": "invalid_python", "requires_review": True, "parse_errors": errors}
    if ast.dump(trees[0], include_attributes=False) == ast.dump(trees[1], include_attributes=False):
        return {**result, "category": "format_or_docstring_only", "requires_review": False}
    masked = [ast.dump(_MaskStrings().visit(tree), include_attributes=False) for tree in trees]
    if masked[0] == masked[1]:
        return {**result, "category": "string_literal_changes_require_review", "requires_review": True}
    return {**result, "category": "executable_syntax_changed", "non_string_ast_changed": True, "requires_review": True}


def append_event(log_path: str | Path, event: dict[str, Any]) -> dict[str, Any]:
    """Append one complete JSONL record, thread-safe within this process.

    Use one log per worker/process or one coordinator writer. This intentionally
    does not claim cross-process atomicity or tamper-proof storage.
    """
    if not isinstance(event, dict):
        raise TypeError("Evidence event must be a dictionary")
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
    line = (json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK, path.open("ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    return record


def _code_identity(code_path: str | Path, orientation_path: str | Path | None) -> dict[str, Any]:
    code = Path(code_path).resolve(strict=True)
    orientation = Path(orientation_path).resolve(strict=True) if orientation_path is not None else None
    return {
        "code_path": str(code),
        "code_sha256": sha256_file(code),
        "orientation_path": str(orientation) if orientation is not None else None,
        "orientation_sha256": sha256_file(orientation) if orientation is not None else None,
    }


def record_code_snapshot(
    log_path: str | Path, code_path: str | Path, orientation_path: str | Path | None = None, **metadata: Any
) -> dict[str, Any]:
    """Record an on-disk source identity, without claiming it was imported/executed."""
    return append_event(log_path, {"event": "code_snapshot", **_code_identity(code_path, orientation_path), "metadata": metadata})


def record_runtime_call(
    log_path: str | Path,
    code_path: str | Path,
    operation: str,
    *,
    orientation_path: str | Path | None = None,
    loaded_code_sha256: str | None = None,
    task_id: str | None = None,
    phase: str | None = None,
    group: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call from the real execution hook, passing the hash captured at code load.

    The disk hash alone is not runtime identity. A caller-supplied loaded hash is
    recorded as such, including a mismatch when disk changed after import.
    """
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("operation must be a non-empty string")
    if loaded_code_sha256 is not None and (len(loaded_code_sha256) != 64 or any(c not in "0123456789abcdef" for c in loaded_code_sha256)):
        raise ValueError("loaded_code_sha256 must be a lowercase SHA-256 hex digest")
    identity = _code_identity(code_path, orientation_path)
    return append_event(log_path, {
        "event": "runtime_call", **identity,
        "operation": operation,
        "loaded_code_sha256": loaded_code_sha256,
        "loaded_hash_source": "caller_execution_hook" if loaded_code_sha256 is not None else "not_provided",
        "disk_matches_loaded_code": identity["code_sha256"] == loaded_code_sha256 if loaded_code_sha256 is not None else None,
        "task_id": task_id, "phase": phase, "group": group, "details": details or {},
        "causal_effect_established": False,
    })
