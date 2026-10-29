"""Lock an offline factorial design before any model result is observed.

Only task identifiers and normalized root URLs participate in task selection.
Held-out rows are written under ``evaluation_only`` and never into job workspaces.
This module does not load credentials, contact models, or execute candidates.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Any

from .data import normalize_root_url, read_jsonl


SEEDS = ("engineering", "dialectical", "critical", "none")
REPETITIONS = 3
DEFAULT_SELECTION_SEED = 20261029
PROJECT = Path(__file__).resolve().parent.parent
SOURCE_FILES = {
    "dev": "data/webwalkerqa/prepared/dev.jsonl",
    "heldout": "data/webwalkerqa/prepared/heldout.jsonl",
}
PROTOCOL = {
    "rounds": 2,
    "batch_size": 1,
    "candidates": 1,
    "top_t": 2,
    "extra_sample_y": 1,
    "max_steps": 6,
    "thinking": "disabled",
    "development_task_count": 4,
    "heldout_task_count": 4,
    "ordinary_memory_learning": True,
    "initial_memory": "native_provider_defaults_after_storage_reset",
    "comparison_memory_policy": "each_comparison_reset_then_same_warmup",
    "selection": "native",
    "heldout_ingestion": False,
    "heldout_access": "evaluator_only",
    "orientation_update_timing": "between_tasks_at_candidate_creation",
    "initial_orientation_reseeded": False,
    "force_code_or_orientation_change": False,
    "winner_policy": "native_selection_may_retain_parent",
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _items_sha(items: list[str]) -> str:
    # Semantic identity excludes provider IDs and parent-snapshot metadata.
    return _sha(json.dumps(items, ensure_ascii=False, separators=(",", ":"),
                           allow_nan=False).encode("utf-8"))


def _rows_bytes(rows: list[dict[str, Any]]) -> bytes:
    return ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False) + "\n"
                    for row in rows)).encode("utf-8")


def _source(project: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    relative = SOURCE_FILES[split]
    path = project / relative
    contents = path.read_bytes()
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Empty {split} source")
    identifiers = set()
    for row in rows:
        identifier = row.get("task_id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{split} source requires prepared nonempty task_id values")
        if identifier in identifiers:
            raise ValueError(f"Duplicate {split} task_id: {identifier}")
        identifiers.add(identifier)
        normalize_root_url(row.get("root_url", ""))
    return {"path": relative, "sha256": _sha(contents), "count": len(rows)}, rows


def _choose(rows: list[dict[str, Any]], seed: int, excluded_roots: set[str]) -> list[dict[str, Any]]:
    # Never inspect question text, answer text, difficulty, or model results.
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        root = normalize_root_url(row["root_url"])
        if root not in excluded_roots:
            groups.setdefault(root, []).append(row)
    if len(groups) < 4:
        raise ValueError("At least four unused distinct root_url groups are required per split")
    rng = random.Random(seed)
    roots = rng.sample(sorted(groups), 4)
    return [rng.choice(sorted(groups[root], key=lambda row: row["task_id"])) for root in roots]


def build_design(project: str | Path, *, selection_seed: int = DEFAULT_SELECTION_SEED) -> dict[str, Any]:
    """Return a portable, deterministic 48-job manifest without writing files.

    The first two prepared development rows, and every row sharing their root
    URLs, are excluded because that earlier acceptance batch has been observed.
    Three paired repetitions shuffle the same locked four tasks in each split.
    """
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise ValueError("selection_seed must be an integer")
    project = Path(project).resolve(strict=True)
    dev_source, dev = _source(project, "dev")
    heldout_source, heldout = _source(project, "heldout")
    roots = {
        "dev": {normalize_root_url(row["root_url"]) for row in dev},
        "heldout": {normalize_root_url(row["root_url"]) for row in heldout},
    }
    if roots["dev"] & roots["heldout"]:
        raise ValueError("Development and heldout root_url groups overlap")
    if {row["task_id"] for row in dev} & {row["task_id"] for row in heldout}:
        raise ValueError("Development and heldout task IDs overlap")
    prior = dev[:2]
    excluded_roots = {normalize_root_url(row["root_url"]) for row in prior}
    selected = {
        "dev": _choose(dev[2:], selection_seed, excluded_roots),
        "heldout": _choose(heldout, selection_seed, set()),
    }
    seed_entries = {}
    for seed in SEEDS:
        relative = None if seed == "none" else f"seeds/{seed}.json"
        contents = None if relative is None else (project / relative).read_bytes()
        if contents is None:
            items = []
        else:
            document = json.loads(contents.decode("utf-8-sig"))
            items = document.get("items") if isinstance(document, dict) else None
            if (not isinstance(items, list) or not items
                    or any(not isinstance(item, str) or not item.strip() for item in items)):
                raise ValueError(f"Invalid initial philosophy items: {seed}")
        seed_entries[seed] = {
            "file": relative,
            "file_sha256": None if contents is None else _sha(contents),
            "items": items,
            "items_sha256": _items_sha(items),
        }
    repetitions = []
    for repetition in range(REPETITIONS):
        order_seed = selection_seed + 1000 + repetition
        rng = random.Random(order_seed)
        entry: dict[str, Any] = {"replicate": repetition, "order_seed": order_seed}
        for split in ("dev", "heldout"):
            rows = list(selected[split])
            rng.shuffle(rows)
            directory = "development" if split == "dev" else "evaluation_only"
            entry[split] = {
                "file": f"{directory}/rep-{repetition:02d}/{split}.jsonl",
                "sha256": _sha(_rows_bytes(rows)),
                "task_ids": [row["task_id"] for row in rows],
                "root_urls": [normalize_root_url(row["root_url"]) for row in rows],
                "count": len(rows),
            }
        repetitions.append(entry)
    jobs = []
    for seed, code_evolve, orientation_evolve, repetition in itertools.product(
            SEEDS, (False, True), (False, True), range(REPETITIONS)):
        identifier = f"{seed}-p{int(code_evolve)}-phi{int(orientation_evolve)}-r{repetition:02d}"
        jobs.append({
            "job_id": identifier,
            "seed": seed,
            "code_evolve": code_evolve,
            "orientation_evolve": orientation_evolve,
            "replicate": repetition,
            "workspace": f"jobs/{identifier}/workspace",
            "dev_file": repetitions[repetition]["dev"]["file"],
            "heldout_file": repetitions[repetition]["heldout"]["file"],
            "ordinary_memory_learning": True,
        })
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "complete_factorial_small_webwalkerqa_preexperiment",
        "selection_seed": selection_seed,
        "selection_fields": ["task_id", "normalized_root_url"],
        "selection_policy": "seeded_distinct_root_groups_before_any_model_result",
        "root_isolation_scope": "normalized_root_url_groups_not_entire_domains",
        "protocol": dict(PROTOCOL),
        "sources": {"dev": dev_source, "heldout": heldout_source},
        "excluded_previous_acceptance": {
            "dev_first_rows": 2,
            "task_ids": [row["task_id"] for row in prior],
            "root_urls": sorted(excluded_roots),
        },
        "seeds": seed_entries,
        "selected_tasks": {
            split: {"task_ids": [row["task_id"] for row in rows],
                    "root_urls": [normalize_root_url(row["root_url"]) for row in rows]}
            for split, rows in selected.items()
        },
        "repetitions": repetitions,
        "jobs": jobs,
        "job_count": len(jobs),
        "inference_limit": "four_independent_heldout_tasks; repetitions_and_API_calls_are_not_new_tasks",
    }
    manifest["design_sha256"] = _sha(_json_bytes(manifest))
    return manifest


def write_design(
    experiment_root: str | Path,
    design: dict[str, Any] | None = None,
    *,
    project: str | Path | None = None,
) -> dict[str, Any]:
    """Write locked inputs; identical retries resume, differing files are refused.

    All referenced sources and seed files must still match ``build_design``.
    The function creates no per-job workspace and writes no heldout question or
    answer to development files or to the public manifest.
    """
    project = PROJECT if project is None else Path(project)
    expected = build_design(project, selection_seed=(
        DEFAULT_SELECTION_SEED if design is None else design["selection_seed"]))
    if design is not None and design != expected:
        raise ValueError("Design differs from current pinned sources, seeds, or protocol")
    design = expected
    root = Path(experiment_root).resolve()
    artifacts: dict[Path, bytes] = {root / "manifest.json": _json_bytes(design)}
    rows_by_split = {}
    for split in ("dev", "heldout"):
        entry, rows = _source(Path(project), split)
        if entry != design["sources"][split]:
            raise ValueError(f"{split} source changed while writing design")
        rows_by_split[split] = {row["task_id"]: row for row in rows}
    for repetition in design["repetitions"]:
        for split in ("dev", "heldout"):
            entry = repetition[split]
            content = _rows_bytes([rows_by_split[split][identifier] for identifier in entry["task_ids"]])
            if _sha(content) != entry["sha256"]:
                raise ValueError(f"{split} task artifact differs from locked design")
            path = (root / entry["file"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Design artifact escapes experiment root")
            artifacts[path] = content
    # Inspect every destination before creating any file, so a conflict does not
    # leave a newly written mixture of two designs.
    for path, content in artifacts.items():
        if path.exists() and (not path.is_file() or path.read_bytes() != content):
            raise ValueError(f"Refusing to overwrite a different design artifact: {path}")
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(content)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"Concurrent design artifact conflict: {path}") from None
    return design
