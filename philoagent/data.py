"""Validated, reproducible WebWalkerQA splits without model or network access."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def _parse_jsonl(data: bytes, source: str = "<bytes>") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_nonfinite)
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON objects; blank lines are ignored, malformed rows are rejected."""
    path = Path(path)
    return _parse_jsonl(path.read_bytes(), str(path))


def normalize_root_url(value: str) -> str:
    """Group equivalent root URLs; retain path/query so distinct roots stay distinct."""
    try:
        url = urlsplit(value.strip())
        if url.scheme.lower() not in {"http", "https"} or not url.hostname:
            raise ValueError("root_url must be an absolute HTTP(S) URL")
        if url.username is not None or url.password is not None:
            raise ValueError("root_url must not contain credentials")
        hostname = url.hostname.lower()
        if ":" in hostname:
            hostname = f"[{hostname}]"
        port = url.port
        scheme = url.scheme.lower()
        netloc = hostname
        if port is not None and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            netloc += f":{port}"
        return urlunsplit((scheme, netloc, url.path.rstrip("/"), url.query, ""))
    except ValueError as exc:
        raise ValueError(f"Invalid root_url {value!r}: {exc}") from exc


def validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy rows, preserve runner fields, and assign stable, unambiguous task IDs."""
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    task_keys: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, 1):
        for key in ("question", "answer", "root_url"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Row {index}: {key} must be a non-empty string")
        root = normalize_root_url(row["root_url"])
        external_id = row.get("task_id", row.get("id"))
        if "task_id" in row or "id" in row:
            if isinstance(external_id, bool) or not isinstance(external_id, (str, int)) or not str(external_id).strip():
                raise ValueError(f"Row {index}: task_id/id must be a non-empty string or integer")
            task_id = str(external_id).strip()
        else:
            task_id = "wwqa-" + _sha256(_canonical(row).encode("utf-8"))
        if task_id in ids:
            raise ValueError(f"Duplicate task ID: {task_id}")
        task_key = (root, " ".join(row["question"].split()))
        if task_key in task_keys:
            raise ValueError(f"Duplicate question at the same root_url: row {index}")
        ids.add(task_id)
        task_keys.add(task_key)
        copied = dict(row)
        copied["task_id"] = task_id
        result.append(copied)
    if len(result) < 2:
        raise ValueError("At least two tasks are required for disjoint splits")
    return result


def split_rows(
    rows: list[dict[str, Any]], seed: int = 42, heldout_fraction: float = 0.3
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split whole root_url groups; no row-level fallback can leak a root across splits."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(heldout_fraction, bool) or not isinstance(heldout_fraction, (int, float)) or not math.isfinite(heldout_fraction) or not 0 < heldout_fraction < 1:
        raise ValueError("heldout_fraction must be finite and strictly between 0 and 1")
    validated = validate_rows(rows)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in validated:
        groups.setdefault(normalize_root_url(row["root_url"]), []).append(row)
    roots = sorted(groups)
    if len(roots) < 2:
        raise ValueError("At least two distinct root_url groups are required; refusing a leaking split")
    rng = random.Random(seed)
    rng.shuffle(roots)
    target = len(validated) * heldout_fraction
    counts: list[int] = []
    running = 0
    for root in roots[:-1]:
        running += len(groups[root])
        counts.append(running)
    boundary = min(range(len(counts)), key=lambda i: (abs(counts[i] - target), i)) + 1
    heldout_roots = set(roots[:boundary])
    dev = sorted((row for root in roots if root not in heldout_roots for row in groups[root]), key=lambda row: row["task_id"])
    heldout = sorted((row for root in roots if root in heldout_roots for row in groups[root]), key=lambda row: row["task_id"])
    rng.shuffle(dev)
    rng.shuffle(heldout)
    return dev, heldout


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return ("\n".join(_canonical(row) for row in rows) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_dataset(
    source: str | Path, output_dir: str | Path, seed: int = 42, heldout_fraction: float = 0.3
) -> dict[str, Any]:
    """Pin actual local source bytes and write immutable, deterministic split artifacts.

    This does not certify that an arbitrary local file came from the upstream URL.
    Callers must record download provenance separately. Only dev.jsonl belongs in
    evolution workspaces; heldout.jsonl must stay in the evaluator's data directory.
    """
    source_path = Path(source).resolve(strict=True)
    source_bytes = source_path.read_bytes()
    dev, heldout = split_rows(_parse_jsonl(source_bytes, str(source_path)), seed, heldout_fraction)
    output = Path(output_dir).resolve()
    splits: dict[str, Any] = {}
    artifacts: dict[Path, bytes] = {}
    for name, rows in (("dev", dev), ("heldout", heldout)):
        data = _jsonl_bytes(rows)
        filename = f"{name}.jsonl"
        artifacts[output / filename] = data
        splits[name] = {
            "file": filename,
            "sha256": _sha256(data),
            "count": len(rows),
            "task_ids": [row["task_id"] for row in rows],
            "root_urls": sorted({normalize_root_url(row["root_url"]) for row in rows}),
        }
    manifest = {
        "schema_version": 1,
        "dataset": "WebWalkerQA",
        "declared_upstream": "https://huggingface.co/datasets/callanwu/WebWalkerQA",
        "source": {"path": str(source_path), "sha256": _sha256(source_bytes), "size_bytes": len(source_bytes)},
        "seed": seed,
        "split_method": "normalized_root_url_groups_seeded_prefix_nearest_target_v1",
        "requested_heldout_fraction": heldout_fraction,
        "actual_heldout_fraction": len(heldout) / (len(dev) + len(heldout)),
        "total_count": len(dev) + len(heldout),
        "splits": splits,
        "evolution_input": "dev.jsonl",
        "heldout_policy": "evaluator_only; do_not_copy_into_evolution_workspaces",
    }
    manifest_path = output / "manifest.json"
    artifacts[manifest_path] = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    for path, content in artifacts.items():
        if path == source_path:
            raise ValueError("Output artifacts must not overwrite the source dataset")
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite a different dataset artifact: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for path, content in artifacts.items():
        if not path.exists():
            _atomic_write(path, content)
    return {"manifest_path": str(manifest_path), "dev_path": str(output / "dev.jsonl"), "heldout_path": str(output / "heldout.jsonl"), "manifest": manifest}


def verify_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Verify source, partition bytes, complete membership and root-level isolation."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset manifest schema")
    source_bytes = Path(manifest["source"]["path"]).read_bytes()
    if _sha256(source_bytes) != manifest["source"]["sha256"] or len(source_bytes) != manifest["source"]["size_bytes"]:
        raise ValueError("Source dataset hash/size mismatch")
    source_rows = validate_rows(_parse_jsonl(source_bytes))
    expected_dev, expected_heldout = split_rows(source_rows, manifest["seed"], manifest["requested_heldout_fraction"])
    for name, expected in (("dev", expected_dev), ("heldout", expected_heldout)):
        entry = manifest["splits"][name]
        path = (manifest_path.parent / entry["file"]).resolve()
        if not path.is_relative_to(manifest_path.parent):
            raise ValueError("Split file must stay within the manifest directory")
        data = path.read_bytes()
        if _sha256(data) != entry["sha256"]:
            raise ValueError(f"{name} split hash mismatch")
        rows = _parse_jsonl(data, str(path))
        if rows != expected:
            raise ValueError(f"{name} split does not match the pinned deterministic source partition")
        if [row["task_id"] for row in rows] != entry["task_ids"] or len(rows) != entry["count"]:
            raise ValueError(f"{name} membership/count mismatch")
        if sorted({normalize_root_url(row["root_url"]) for row in rows}) != entry["root_urls"]:
            raise ValueError(f"{name} root group mismatch")
    if manifest["total_count"] != len(source_rows):
        raise ValueError("Manifest total_count mismatch")
    return manifest
