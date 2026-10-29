"""Offline construction and isolation of the formal matrix's ordinary memory.

These helpers do not import a model, initialize a provider, or run an experiment.
The initial database retains the native seven priors and places the four shared
methods in ordinary strategic memory. Phi is loaded separately, without methods.
Fresh copies isolate every M0 episode and every held-out/probe execution; callers
must also disable ingestion and discard each copy after that execution.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = PROJECT / "seeds/systems/common-methods.json"
DEFAULT_PHI = PROJECT / "seeds/systems/dialectical-v1.json"
PROVIDER_SOURCE = (PROJECT / "vendor/MemEvolve/Flash-Searcher-main/EvolveLab/providers/"
                   "lightweight_memory_provider.py")
PHI_CHAPTERS = ("ONT", "EPI", "MET", "VAL", "ACT", "REV")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value) -> str:
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8"))


def _items(path, count: int) -> tuple[str, ...]:
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    items = document.get("items") if isinstance(document, dict) else None
    if (not isinstance(document, dict) or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1
            or not isinstance(items, list) or len(items) != count
            or any(not isinstance(item, str) or not item.strip() for item in items)):
        raise ValueError("Seed must have schema_version 1 and the required nonempty items")
    return tuple(items)


def phi_seed(path=DEFAULT_PHI) -> tuple[str, ...]:
    """Read the six-chapter initial system; reject the old combined seed."""
    items = _items(path, len(PHI_CHAPTERS))
    if any(not text.startswith("【" + chapter + "｜") or "METHOD-" in text
           for chapter, text in zip(PHI_CHAPTERS, items)):
        raise ValueError("Initial Phi must contain only ONT/EPI/MET/VAL/ACT/REV in order")
    return items


def build_initial_memory(methods_path=DEFAULT_METHODS, provider_source=PROVIDER_SOURCE) -> dict:
    """Build the native JSON schema without executing provider code.

    Cold-start literals come from the pinned provider source, including their
    native tags and signatures. Shared methods have stable source identifiers;
    no timestamps, task text, outcomes, or philosophical chapters are inserted.
    """
    methods_path, provider_source = Path(methods_path), Path(provider_source)
    methods = _items(methods_path, 4)
    if any(not text.startswith(f"【可选方法资源 METHOD-{index}｜")
           for index, text in enumerate(methods, 1)):
        raise ValueError("Shared methods must contain METHOD-1 through METHOD-4 in order")
    source_bytes = provider_source.read_bytes()
    constants = {}
    names = {"COLDSTART_STRATEGIC_MEMORIES": ("strategic", 5),
             "COLDSTART_OPERATIONAL_MEMORIES": ("operational", 2)}
    for node in ast.parse(source_bytes.decode("utf-8-sig")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    if target.id in constants:
                        raise ValueError("Duplicate native cold-start declaration")
                    constants[target.id] = ast.literal_eval(node.value)
    db = {"strategic": [], "operational": [], "meta": {"version": 1}}
    for name, (kind, count) in names.items():
        records = constants.get(name)
        if not isinstance(records, list) or len(records) != count:
            raise ValueError("Pinned native cold-start schema changed")
        for index, record in enumerate(records, 1):
            if (not isinstance(record, dict) or not isinstance(record.get("content"), str)
                    or not record["content"].strip()
                    or not isinstance(record.get("tags"), list)
                    or any(not isinstance(tag, str) for tag in record["tags"])):
                raise ValueError("Invalid native cold-start record")
            db[kind].append({"content": record["content"], "tags": deepcopy(record["tags"]),
                             "usage_count": 0, "success_count": 0,
                             "signature": _sha(record["content"].lower().encode("utf-8")),
                             "source_kind": "native_cold_start_prior",
                             "source_id": f"native:{kind}:{index}"})
    methods_sha = _sha(methods_path.read_bytes())
    for index, text in enumerate(methods, 1):
        db["strategic"].append({"content": text, "tags": ["shared_method"],
                                "usage_count": 0, "success_count": 0,
                                "signature": _sha(text.lower().encode("utf-8")),
                                "source_kind": "shared_method_resource",
                                "source_id": f"METHOD-{index}", "source_sha256": methods_sha})
    db["meta"]["formal_initial_memory"] = {
        "schema_version": 1, "native_priors": 7, "shared_methods": 4,
        "provider_source_sha256": _sha(source_bytes),
        "common_methods_source_sha256": methods_sha,
        "ordinary_learning_rule": "native_success_only_extraction",
    }
    return db


def _is_link(path: Path) -> bool:
    info = path.lstat()
    return (stat.S_ISLNK(info.st_mode)
            or bool(getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)))


def _safe_path(path) -> Path:
    raw = Path(path)
    if ".." in raw.parts:
        raise ValueError("Parent path traversal is not allowed")
    absolute = Path(os.path.abspath(raw))
    # Reject even ancestor links/junctions, including a dangling final link.
    for candidate in (*reversed(absolute.parents), absolute):
        if os.path.lexists(candidate) and _is_link(candidate):
            raise ValueError("Memory paths must not contain links or reparse points")
    return absolute.resolve()


def _destination(destination, boundary_root) -> Path:
    boundary = _safe_path(boundary_root)
    if not boundary.is_dir():
        raise ValueError("An existing directory boundary is required")
    target = _safe_path(destination)
    if target == boundary or not target.is_relative_to(boundary):
        raise ValueError("Memory destination must stay inside its directory boundary")
    if os.path.lexists(target):
        raise FileExistsError("Memory destinations cannot be overwritten")
    return target


def snapshot_manifest(root) -> dict:
    """Hash all regular files and preserve the directory inventory; reject empty M."""
    root = _safe_path(root)
    if not root.is_dir():
        raise ValueError("Memory snapshot must be an existing directory")
    files, directories = {}, []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(current)
        for name in sorted(dirnames):
            path = current / name
            if _is_link(path) or not path.is_dir():
                raise ValueError("Memory snapshot contains a linked or invalid directory")
            directories.append(path.relative_to(root).as_posix())
        for name in sorted(filenames):
            path = current / name
            if _is_link(path) or not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("Memory snapshot contains a link or nonregular file")
            files[path.relative_to(root).as_posix()] = _sha(path.read_bytes())
    if not files:
        raise ValueError("Empty memory snapshots are not allowed")
    content = {"files": dict(sorted(files.items())), "directories": sorted(directories)}
    return {**content, "sha256": _digest(content)}


def write_initial_memory(destination, *, boundary_root, methods_path=DEFAULT_METHODS,
                         provider_source=PROVIDER_SOURCE) -> dict:
    """Create a new canonical M directory, returning its immutable manifest."""
    target = _destination(destination, boundary_root)
    db = build_initial_memory(methods_path, provider_source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    (target / "longterm_memory.json").write_text(
        json.dumps(db, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return snapshot_manifest(target)


def clone_memory(source, destination, *, expected_manifest, boundary_root) -> dict:
    """Copy an entire verified M into a fresh per-episode directory.

    Source verification occurs before and after copying. A failed verification
    never deletes evidence or silently overwrites a partially created trial.
    No retrieval state from an earlier episode is reused by this operation.
    """
    source = _safe_path(source)
    target = _destination(destination, boundary_root)
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("Source and destination memory trees must be disjoint")
    before = snapshot_manifest(source)
    if before != expected_manifest:
        raise ValueError("Source memory differs from its frozen manifest")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, symlinks=True)
    result = snapshot_manifest(target)
    if snapshot_manifest(source) != before or result != before:
        raise ValueError("Memory changed while copying; preserve the failed copy for inspection")
    return result
