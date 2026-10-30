"""Read-only plans for two precisely identified failed native ingestions.

This module does not mutate records/storage, launch processes or call an API.
The calling recovery helper must back up and hash-pin all files before applying.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def content(event):
    return event["response"]["choices"][0]["message"]["content"]


def norm(query):
    text = re.sub(r"\s+", " ", str(query or "").strip().lower())
    return re.sub(r"[，。；：、！？“”‘’（）【】《》,.;:!?\"'()\[\]{}]", "", text)[:2000]


def request_text(event):
    parts = []
    for message in event["request"]["messages"]:
        val = message.get("content", "")
        parts.extend([val] if isinstance(val, str) else [x.get("text", "") for x in val])
    return "\n".join(parts)


def _rollback(root, path, restored):
    text = json.dumps(restored, ensure_ascii=False, indent=2) + "\n"
    return {"path": str(path.relative_to(root)), "before_sha256": sha(path),
            "after_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "restored_utf8": text}


def plan_ingestion(root, job_id, record, events):
    """Return an auditable helper-compatible rollback plan, or reject drift."""
    root = Path(root).resolve(strict=True)
    job = root / "jobs" / job_id
    execution = record["task_execution_id"]
    current = [e for e in events if e.get("task_execution_id") == execution]
    judges = [e for e in current if e.get("role") == "judge" and e.get("status") == "success"]
    assert judges, "Missing saved independent judge"
    judge_end = max(e["finished_unix"] for e in judges)
    ingest = [e for e in current if e.get("role") == "memory_ingestion"]
    assert record["status"] == "ingestion_failed" and record["memory_ingestion_started"]
    assert not record["memory_ingestion_complete"]
    assert isinstance(record["grade"], dict) and record["answer"].strip()

    if job_id == "humanities-full-r1" and execution == "38060b89244d46ce993235fc76d01b23":
        code = job / "workspace/EvolveLab/providers/credence_lattice_provider.py"
        assert sha(code) == "55cb962e5440655fc0d3556682df24af17da2ac3ed02bce998fc0a82feaeaad8"
        assert [e["attempt"] for e in ingest] == [11228, 11297, 11308, 11330]
        extraction, *failed = ingest
        assert extraction["status"] == "success"
        assert extraction["response"]["choices"][0]["finish_reason"] == "stop"
        assert all(e.get("http_status") == 429 and e.get("response") is None for e in failed)
        assert len({json.dumps(e["request"], sort_keys=True) for e in failed}) == 1
        payload = json.loads(content(extraction))
        assert len(payload["semantic"]) == len(payload["anti_patterns"]) == 3
        # Native deterministic consolidation only merges >=.55 semantic/.60 anti.
        # The exact failed request lists all neighbours >=.22, so its maximum
        # overlap proves that the recorded fallback could only append all six.
        overlaps = [float(x) for x in re.findall(r"overlap=(\d+\.\d+)", request_text(failed[0]))]
        assert overlaps and max(overlaps) < .54
        begin = next(e for e in current if e.get("attempt") == 10552)
        selected = json.loads(content(begin))
        selected_sem = selected["selected_ids"]
        selected_anti = selected["guardrail_ids"]
        assert len(selected_sem) == 5 and len(selected_anti) == 3
        # Exploration is enforced in native code. All actual selected entries are
        # independently identified by their last_used timestamp in this BEGIN.
        store = job / "workspace/storage/credence_lattice"
        semantic_path, anti_path, snapshots_path = [store / n for n in
            ("semantic_ledger.json", "anti_pattern_ledger.json", "task_snapshots.json")]
        semantic, anti, snapshots = [read(p) for p in (semantic_path, anti_path, snapshots_path)]
        assert semantic["meta"]["task_counter"] == 14 and snapshots == {}
        assert 60 <= record["quality_score"] < 90  # native partial credit = .5
        restored_files = []
        for path, ledger, chosen, incoming, field in (
            (semantic_path, semantic, selected_sem, payload["semantic"], "claim"),
            (anti_path, anti, selected_anti, payload["anti_patterns"], "failure"),
        ):
            created = [x for x in ledger["items"] if x["stats"].get("created_task") == 14]
            assert len(created) == 3
            # Compare native signature values to exact paid extraction contents.
            assert Counter(x[field] for x in created) == Counter(x[field] for x in incoming)
            assert all(x.get("version") == 1 and x["stats"].get("usage") == 0
                       and x["stats"].get("success") == 0 and x["stats"].get("fail") == 0 for x in created)
            in_begin = [x for x in ledger["items"] if x["stats"].get("last_used", 0) >= begin["started_unix"]]
            assert {x["id"] for x in in_begin} == set(chosen)
            assert len({x["stats"]["last_used"] for x in in_begin}) == 1
            assert all(begin["finished_unix"] <= x["stats"]["last_used"] < judge_end for x in in_begin)
            assert path.stat().st_mtime > max(e["finished_unix"] for e in failed)
            # No pruning could have happened: semantic subsets <=48, anti <=32.
            assert len(ledger["items"]) <= (48 if field == "claim" else 32)
            out = deepcopy(ledger)
            out["items"] = [x for x in out["items"] if x["stats"].get("created_task") != 14]
            for item in out["items"]:
                if item["id"] in chosen:
                    assert item["stats"]["success"] >= .5
                    item["stats"]["success"] -= .5
            restored_files.append(_rollback(root, path, out))
        snapshot = {"task_id": "cl-task-00014", "task_counter": 14,
                    "created": begin["finished_unix"], "semantic_ids": selected_sem,
                    "anti_ids": selected_anti}
        rebuilt = {norm(q): snapshot for q in (record["enhanced_question"], record["question"]) if norm(q)}
        restored_files.append(_rollback(root, snapshots_path, rebuilt))
        return {"method": "rollback_completed_ingestion_transaction_then_retry",
                "reason": "Paid extraction succeeded; exact native fallback appended six items and partial credit before SDK integrity rejection. Undo only these writes and rebuild the consumed task snapshot.",
                "judge_finished_unix": judge_end, "provider_sha256": sha(code),
                "paid_extraction_attempt": 11228, "failed_attempts": [e["attempt"] for e in failed],
                "runtime_context": None,
                "pinned_files": {str(p.relative_to(root)): sha(p) for p in [code, semantic_path, anti_path, snapshots_path]},
                "storage_rollbacks": restored_files, "saved_answer_and_grade_reused": True}

    if job_id == "business-native-r1" and execution == "697d702042624019a607e129b9479295":
        code = job / "workspace/EvolveLab/providers/lightweight_memory_provider.py"
        assert sha(code) == "ce0c1f36ca9fb095c8448fab9b3e35e3c97e5928d115fdf98d56f48c842750f4"
        assert [e["attempt"] for e in ingest] == [11329, 11339]
        assert [e.get("http_status") for e in ingest] == [429, 402]
        assert all(e.get("response") is None for e in ingest)
        begin = next(e for e in current if e.get("attempt") == 9722)
        assert json.loads(content(begin))["selected_indices"] == [12, 14, 40]
        store = job / "workspace/storage/lightweight_memory/longterm_memory.json"
        baseline = root / "native_recovery_20261030/native_storage_before_resume/business-native-r1/storage/lightweight_memory/longterm_memory.json"
        now, old = read(store), read(baseline)
        ids = ["strategic_11", "strategic_13", "operational_16"]
        chosen = {("strategic", 11), ("strategic", 13), ("operational", 16)}
        assert len(old["strategic"]) == len(now["strategic"]) == 23
        assert len(old["operational"]) == len(now["operational"]) == 20
        differences = []
        for name in ("strategic", "operational"):
            for index, (a, b) in enumerate(zip(now[name], old[name])):
                if a != b:
                    differences.append((name, index))
                    assert (name, index) in chosen
                    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"usage_count", "success_count"}
                    assert a["usage_count"] == b["usage_count"] + 1
                    assert a["success_count"] == b["success_count"] + 1
        assert set(differences) == chosen and record["quality_score"] >= 60
        restored = deepcopy(now)
        for name, index in chosen:
            restored[name][index]["success_count"] -= 1
        return {"method": "rollback_completed_ingestion_transaction_then_retry",
                "reason": "Extraction failed without a model response, but native fallback saved three success counters. Undo only success credit; retain valid Actor retrieval usage. Restore RAM task IDs before retrying ingestion.",
                "judge_finished_unix": judge_end, "provider_sha256": sha(code),
                "failed_attempts": [11329, 11339],
                "runtime_context": {"kind": "lightweight_used_memory_ids", "used_memory_ids": ids,
                                    "query": record["enhanced_question"]},
                "pinned_files": {str(p.relative_to(root)): sha(p) for p in [code, store, baseline]},
                "storage_rollbacks": [_rollback(root, store, restored)],
                "saved_answer_and_grade_reused": True}
    raise ValueError("No audited plan for this task execution")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    events = [read(p) for p in sorted((args.root / "budget/attempts").glob("*.json"))]
    result = []
    for job, rel in (("humanities-full-r1", "round_01/eval_finalists/credence_lattice/7.json"),
                     ("business-native-r1", "round_01/eval_batch/lightweight_memory/6.json")):
        record = read(args.root / "jobs" / job / "evolution" / rel)
        plan = plan_ingestion(args.root, job, record, events)
        # Public audit omits complete memory content; helper function retains it.
        result.append({"job_id": job, "record": rel, **{k: v for k, v in plan.items() if k != "storage_rollbacks"},
                       "storage_rollbacks": [{k: v for k, v in x.items() if k != "restored_utf8"}
                                             for x in plan["storage_rollbacks"]]})
    output = {"status": "plans_verified_no_mutation", "plans": result}
    if args.output:
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "jobs": [x["job_id"] for x in result]}, ensure_ascii=False))
