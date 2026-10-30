"""Incident-only JSON encoding repair of the already paid WB0804 grade.

Dry-run by default. Never calls a model or marks task/ingestion completed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from philoagent.hss_evaluation import normalize_grade_output, validate_grade
from philoagent.hss_experiment import process_owned

JOB = "humanities-full-r2"
RELATIVE_RECORD = Path("jobs") / JOB / "evolution/round_00/base_logs/2.json"
PAID_ATTEMPT = 10232
RAW_SHA256 = "241d6fc2d04e7aa42292f288e2d7169b7bf3fc990bbdc5201a768af8afd1c866"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def serialized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def check_job_stopped(root):
    source = root / "jobs" / JOB / "worker_source.json"
    if source.exists() and process_owned(json.loads(source.read_text(encoding="utf-8"))):
        raise ValueError("Target worker is still running; do not edit its record")
    ledger = root / "budget/budget.json"
    if ledger.exists():
        for number in json.loads(ledger.read_text(encoding="utf-8")).get("pending", {}):
            receipt = root / "budget/attempts" / f"{int(number):06d}.json"
            if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")).get("job_id") == JOB:
                raise ValueError("Target job still has an in-flight request")


def repair(root, *, expected_record_sha256, apply=False):
    root = Path(root).resolve()
    path = root / RELATIVE_RECORD
    before = path.read_bytes()
    if sha(before) != expected_record_sha256:
        raise ValueError("Current record bytes differ from the explicitly reviewed SHA256")
    record = json.loads(before)
    if (record.get("task_id") != "WB0804" or record.get("domain") != "humanities"
            or record.get("status") != "judge_failed" or record.get("grade") is not None
            or record.get("memory_ingestion_started") or record.get("memory_ingestion_complete")):
        raise ValueError("Record is outside this stopped pre-ingestion Judge incident")
    raw = record["judge_output"]
    if sha(raw.encode("utf-8")) != RAW_SHA256 or raw.count('"救救孩子"') != 1:
        raise ValueError("Paid Judge body differs from this exact known encoding incident")
    paid = json.loads((root / "budget/attempts" / f"{PAID_ATTEMPT:06d}.json").read_text(encoding="utf-8"))
    choice = paid["response"]["choices"][0]
    if (paid.get("job_id") != JOB or paid.get("role") != "judge" or paid.get("status") != "success"
            or choice.get("finish_reason") != "stop" or choice["message"].get("content") != raw):
        raise ValueError("Original complete paid Judge receipt does not match")
    fixed = raw.replace('"救救孩子"', '\\"救救孩子\\"')
    parsed = json.loads(fixed)
    normalized, normalization = normalize_grade_output(fixed, "humanities")
    if normalization["repaired"] or normalized != parsed:
        raise ValueError("This incident must not require score or rubric structure repair")
    grade = validate_grade(normalized, "humanities")
    if grade["quality_score"] != 81.8:
        raise ValueError("Repaired encoding differs from the reviewed original 81.8 grade")
    canonical = serialized(normalized)
    if json.loads(canonical) != parsed:
        raise ValueError("Canonical encoding changed original Judge values")
    check_job_stopped(root)
    evidence = {
        "method": "escape_two_unescaped_internal_quote_characters_only",
        "paid_attempt_id": PAID_ATTEMPT, "record_sha256_before": sha(before),
        "raw_sha256": RAW_SHA256, "escaped_sha256": sha(fixed.encode("utf-8")),
        "normalized_sha256": sha(canonical.encode("utf-8")),
        "quality_score_validated": grade["quality_score"],
        "scores": {name: item["score"] for name, item in normalized["dimensions"].items()},
        "visible_assessments_unchanged": True, "model_called": False,
        "task_status_unchanged": True, "ingestion_flags_unchanged": True,
        "record_path": str(path), "applied": False,
    }
    if not apply:
        return evidence
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = root / "native_recovery_20261030/paid_judge_encoding" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    (backup / "record.before.json").write_bytes(before)
    (backup / "judge.original.txt").write_text(raw, encoding="utf-8")
    evidence.update(applied=True, backup_dir=str(backup), applied_utc=datetime.now(timezone.utc).isoformat())
    updated = deepcopy(record)
    updated.update(judge_output=canonical, judge_output_original=raw, judge_encoding_repair=evidence)
    unchanged = {key: value for key, value in updated.items()
                 if key not in {"judge_output", "judge_output_original", "judge_encoding_repair"}}
    original = {key: value for key, value in record.items()
                if key not in {"judge_output", "judge_output_original", "judge_encoding_repair"}}
    if unchanged != original:
        raise ValueError("Unexpected record mutation")
    check_job_stopped(root)
    if path.read_bytes() != before:
        raise ValueError("Record changed after backup; target was not edited")
    encoded = (json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".encoding-" + uuid4().hex + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    receipt = {**evidence, "record_sha256_after": sha(encoded)}
    (backup / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-record-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(repair(args.root, expected_record_sha256=args.expected_record_sha256,
                            apply=args.apply), ensure_ascii=False, indent=2))
