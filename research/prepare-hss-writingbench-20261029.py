"""Export the selected WritingBench questions for the value ablation, offline.

Run from any directory with the repository Python. Original queries/checklists
are preserved; this script has no credentials, model client or network access.
"""

from pathlib import Path
import argparse
import hashlib
import json
import random


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "ae2d5176449b7b769815482641d35926f26793eb"
RAW_SHA256 = "18fee37c645166eb2e206b36366b2e354265b1e4201db2c86e759e825eaddcbe"
DOMAINS = ("humanities", "social", "business")
GROUPS = {
    "native": {"initial_phi": "none", "code_evolve": True, "orientation_evolve": False},
    "fixed": {"initial_phi": "dialectical-v1", "code_evolve": True, "orientation_evolve": False},
    "full": {"initial_phi": "dialectical-v1", "code_evolve": True, "orientation_evolve": True},
}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def prepare(output, repetitions):
    if repetitions not in (1, 3):
        raise ValueError("Choose one or three complete repetitions")
    source_dir = ROOT / "data/writingbench" / COMMIT
    source_file = source_dir / "benchmark_all.jsonl"
    raw = source_file.read_bytes()
    if sha(raw) != RAW_SHA256:
        raise ValueError("Official source differs from its pinned snapshot")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    by_id = {row["index"]: row for row in rows}
    if len(rows) != 1000 or len(by_id) != 1000:
        raise ValueError("Unexpected official question inventory")
    artifacts, selected, selections = {}, {}, {}
    seen = set()
    for domain in DOMAINS:
        selection_path = ROOT / "research" / f"hss-writingbench-{domain}-selection-20261029.json"
        selection_raw = selection_path.read_bytes()
        selection = json.loads(selection_raw)
        if selection["source_commit"] != COMMIT or selection["domain"] != domain:
            raise ValueError("Selection source or domain differs")
        tasks = []
        for entry in selection["tasks"]:
            identifier = entry["source_index"]
            if identifier in seen or identifier not in by_id:
                raise ValueError("Repeated or unknown source question")
            seen.add(identifier)
            row = by_id[identifier]
            if row["lang"] != "zh" or row["domain2"] != entry["domain2"]:
                raise ValueError("Selected question metadata differs")
            split = entry["split"]
            if split == "heldout":
                split = "holdout"
            if split not in ("learning", "holdout"):
                raise ValueError("Unknown task split")
            if not isinstance(row["checklist"], list) or len(row["checklist"]) != 5:
                raise ValueError("Official question criteria are missing")
            tasks.append({
                "task_id": f"WB{identifier:04d}", "domain": domain, "split": split,
                "title": entry["title"], "question": row["query"], "materials": [],
                "material_location": "embedded_in_unmodified_official_query",
                "task_criteria": [{"name": item["name"], "description": item["criteria_description"]}
                                  for item in row["checklist"]],
                "original_checklist": row["checklist"],
                "source": {"dataset": "WritingBench", "commit": COMMIT,
                           "index": identifier, "domain1": row["domain1"], "domain2": row["domain2"],
                           "url": f"https://github.com/X-PLUG/WritingBench/blob/{COMMIT}/benchmark_query/benchmark_all.jsonl"},
            })
        if len(tasks) != 12 or sum(item["split"] == "learning" for item in tasks) != 8:
            raise ValueError("Each domain requires eight learning and four evaluation questions")
        selected[domain] = tasks
        selections[domain] = {"path": selection_path.relative_to(ROOT).as_posix(), "sha256": sha(selection_raw)}
        for split in ("learning", "holdout"):
            directory = "learning" if split == "learning" else "evaluation"
            subset = [item for item in tasks if item["split"] == split]
            artifacts[f"{directory}/{domain}.jsonl"] = "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in subset).encode("utf-8")
    artifacts["LICENSE.WritingBench"] = (source_dir / "LICENSE").read_bytes()
    jobs = []
    for repetition in range(repetitions):
        for domain_index, domain in enumerate(DOMAINS):
            train_ids = [row["task_id"] for row in selected[domain] if row["split"] == "learning"]
            holdout_ids = [row["task_id"] for row in selected[domain] if row["split"] == "holdout"]
            order_seed = 20261029 + 100 * repetition + domain_index
            rng = random.Random(order_seed)
            rng.shuffle(train_ids)
            rng.shuffle(holdout_ids)
            for group, factors in GROUPS.items():
                jobs.append({"job_id": f"{domain}-{group}-r{repetition + 1}", "group": group,
                             "domain": domain, "repetition": repetition + 1, **factors,
                             "learning_file": f"learning/{domain}.jsonl", "evaluation_file": f"evaluation/{domain}.jsonl",
                             "learning_order": list(train_ids), "evaluation_order": list(holdout_ids),
                             "order_seed": order_seed})
    manifest = {
        "kind": "hss_business_full_method_ablation_inputs", "version": "20261029-v1",
        "main_plan": "planv0.0.6.md", "status": "questions_selected_runtime_integration_pending",
        "source_commit": COMMIT, "source_sha256": RAW_SHA256, "official_source_questions": 1000,
        "selected_questions": 36, "learning_questions": 24, "evaluation_questions": 12,
        "original_queries_modified": False, "model_results_used_for_selection": False,
        "new_paid_model_calls": 0, "repetitions": repetitions, "jobs": jobs,
        "groups": GROUPS, "selection_files": selections,
        "native_protocol": {"rounds": 2, "task_batch_x": 2, "candidates": 2,
                            "top_t": 2, "extra_sample_y": 1, "clear_storage_per_round": False,
                            "memory_mode": "native_persistent", "actor_max_steps": 32,
                            "actor_max_output_tokens": 8192, "analysis_max_steps": 20,
                            "per_task_spend_stop": False, "shadow_reflection": False},
        "planned_actor_executions": 32 * len(jobs),
        "evaluation": {"main_metric": "continuous_quality_0_to_100", "group_metadata_hidden_from_judge": True,
                       "official_leaderboard_score": False, "module": "philoagent/hss_evaluation.py"},
        "artifact_sha256": {path: sha(data) for path, data in sorted(artifacts.items())},
    }
    manifest["manifest_sha256"] = sha(json_bytes(manifest))
    artifacts["manifest.json"] = json_bytes(manifest)
    output = Path(output).resolve()
    for name, data in artifacts.items():
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Refusing to replace differing input artifact: {path}")
    for name, data in artifacts.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
    for name, data in artifacts.items():
        if (output / name).read_bytes() != data:
            raise ValueError("Written source artifact differs")
    return {"output": str(output), "questions": 36, "jobs": len(jobs),
            "planned_actor_executions": manifest["planned_actor_executions"],
            "source_commit": COMMIT, "manifest_sha256": manifest["manifest_sha256"],
            "original_queries_modified": False, "new_paid_model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, choices=(1, 3), default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or ROOT / f"data/hss_business/writingbench-20261029-r{args.repetitions}"
    print(json.dumps(prepare(destination, args.repetitions), ensure_ascii=False, indent=2))
