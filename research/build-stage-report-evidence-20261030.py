"""Recompute stage-report figures from existing artifacts; no model or network calls."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/hss-business-formal-20261029-r01"
RECOVERY = RUN / "native_recovery_20261030"
OUTPUT = ROOT / "research/stage-report-evidence-20261030.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def proof(path: Path) -> dict:
    return {"path": relative(path), "sha256": sha(path)}


def main():
    progress = read(RECOVERY / "native_progress.json")
    cost = read(RECOVERY / "current_analysis_cost.json")
    pair = read(RECOVERY / "first_paired_native_result.json")
    rows = [json.loads(line) for line in (RECOVERY / "native_results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == progress["completed_works"] == 403
    assert progress["compressed_results_included"] is False
    assert progress["completed_rounds"] == 17
    assert progress["heldout_completed"] == 17
    assert len({r["source_path"] for r in rows}) == len(rows)
    assert len({r["task_execution_id"] for r in rows}) == len(rows)

    verified = []
    heldout = {}
    for row in rows:
        path = Path(row["source_path"])
        assert path.is_relative_to(RUN / "jobs")
        raw = read(path)
        assert raw["status"] == row["status"] == "completed"
        assert raw["task_execution_id"] == row["task_execution_id"]
        dims = raw["grade"]["dimensions"]
        assert sum(v["weight"] for v in dims.values()) == 100
        recomputed = sum(v["score"] / 5 * v["weight"] for v in dims.values())
        assert abs(recomputed - raw["quality_score"]) < 1e-8
        assert abs(recomputed - row["quality_score"]) < 1e-8
        assert abs(recomputed - raw["grade"]["quality_score"]) < 1e-8
        record = {
            **proof(path),
            "job_id": row["job_id"],
            "domain": row["domain"],
            "group": row["group"],
            "repetition": row["repetition"],
            "task_id": row["task_id"],
            "execution_id": row["task_execution_id"],
            "phase": row["phase"],
            "quality_score": round(recomputed, 3),
        }
        verified.append(record)
        if row["phase"] == "heldout":
            heldout.setdefault(row["job_id"], []).append({
                **record,
                "dimensions": {k: {"score": v["score"], "weight": v["weight"]} for k, v in dims.items()},
                "confirmed_model_tokens": raw["confirmed_model_tokens"],
            })

    completed = {}
    for job, records in heldout.items():
        if len(records) == 4:
            completed[job] = {
                "heldout_count": len(records),
                "heldout_mean": round(mean(r["quality_score"] for r in records), 3),
                "heldout_confirmed_tokens": sum(r["confirmed_model_tokens"] for r in records),
                "records": sorted(records, key=lambda r: r["task_id"]),
                "result_proof": proof(RUN / "jobs" / job / "result.json"),
            }
    assert len(completed) == 4
    native = completed["humanities-native-r1"]
    full = completed["humanities-full-r1"]
    assert native["heldout_mean"] == 87.075
    assert full["heldout_mean"] == 91.55
    assert native["heldout_confirmed_tokens"] == 1038644
    assert full["heldout_confirmed_tokens"] == 2138271
    assert completed["social-native-r1"]["heldout_mean"] == 81.85
    assert completed["business-native-r1"]["heldout_mean"] == 83.75
    comparison = []
    for n, r in zip(native["records"], full["records"]):
        assert n["task_id"] == r["task_id"]
        comparison.append({"task_id": n["task_id"], "native": n["quality_score"], "full": r["quality_score"], "difference": round(r["quality_score"] - n["quality_score"], 3)})
    dims = {}
    for key in native["records"][0]["dimensions"]:
        dims[key] = {
            "weight": native["records"][0]["dimensions"][key]["weight"],
            "native_mean": round(mean(r["dimensions"][key]["score"] for r in native["records"]), 3),
            "full_mean": round(mean(r["dimensions"][key]["score"] for r in full["records"]), 3),
        }
    assert cost["native_historical_sdk"]["confirmed_total_tokens"] == 301742425
    learning_audit = read(RECOVERY / "current_analysis_learning.json")
    coverage = []
    for domain in ("humanities", "social", "business"):
        for group in ("native", "fixed", "full"):
            selected = [r for r in rows if r["domain"] == domain and r["group"] == group]
            jobs = [j for j in learning_audit["counts"]["by_job"] if j["job_id"].startswith(f"{domain}-{group}-")]
            coverage.append({
                "domain": domain, "group": group,
                "completed_records": len(selected),
                "learning_records": sum(r["phase"] != "heldout" for r in selected),
                "heldout_records": sum(r["phase"] == "heldout" for r in selected),
                "completed_rounds": sum(j["completed_rounds"] for j in jobs),
                "completed_jobs": sum(job.startswith(f"{domain}-{group}-") for job in completed),
            })
    assert sum(c["completed_rounds"] for c in coverage) == 17

    learning_panels = {}
    for phase_name, round_number in (("initial_all_repetitions", 0), ("after_first_evolution_repetition_1", 1)):
        grouped = defaultdict(dict)
        for row in rows:
            if row["phase"] == "base" and row["round"] == round_number and (round_number == 0 or row["repetition"] == 1):
                key = (row["domain"], row["repetition"], row["task_id"])
                assert row["group"] not in grouped[key]
                grouped[key][row["group"]] = row
        assert len(grouped) == (18 if round_number == 0 else 6)
        cells = []
        for (domain, repetition, task), group_rows in sorted(grouped.items()):
            assert set(group_rows) == {"native", "fixed", "full"}
            cells.append({
                "domain": domain, "repetition": repetition, "task_id": task,
                "scores": {g: r["quality_score"] for g, r in group_rows.items()},
                "source_paths": {g: relative(Path(r["source_path"])) for g, r in group_rows.items()},
            })
        domain_means = {}
        for domain in ("humanities", "social", "business"):
            selected = [c for c in cells if c["domain"] == domain]
            domain_means[domain] = {
                "matched_triplets": len(selected),
                "distinct_questions": len({c["task_id"] for c in selected}),
                "means": {g: round(mean(c["scores"][g] for c in selected), 6) for g in ("native", "fixed", "full")},
            }
        learning_panels[phase_name] = {"cells": cells, "by_domain": domain_means}
    business_sensitivity = [c for c in learning_panels["initial_all_repetitions"]["cells"] if c["domain"] == "business" and not (c["repetition"] == 1 and c["task_id"] == "WB0100")]
    assert len(business_sensitivity) == 5
    sensitivity = {g: round(mean(c["scores"][g] for c in business_sensitivity), 3) for g in ("native", "fixed", "full")}
    assert sensitivity == {"native": 85.72, "fixed": 84.72, "full": 85.98}
    sources = [
        RECOVERY / "native_progress.json", RECOVERY / "native_results.jsonl",
        RECOVERY / "current_analysis_cost.json", RECOVERY / "first_paired_native_result.json",
        RECOVERY / "current_analysis_learning.json",
        RECOVERY / "heartbeat_20261030_0405.json", RUN / "budget/budget.json",
        ROOT / "research/hss-business-interim2-results-audit-20261029.json",
        ROOT / "research/hss-business-interim2-cost-20261029.json",
        ROOT / "research/budget7-analysis-data-20261029.json",
        ROOT / "research/compression-structure-audit-20261030.json",
        ROOT / "research/compression-performance-audit-20261030.json",
        ROOT / "research/compression-measurement-audit-20261030.json",
        RUN / "jobs/humanities-full-r1/evolution/round_00/round_summary.json",
        RUN / "jobs/humanities-full-r1/evolution/round_01/round_summary.json",
        ROOT / "planv0.0.6.md", ROOT / "vendor/MemEvolve.UPSTREAM.json",
    ]
    evidence = {
        "kind": "philoagent_stage_report_fixed_evidence",
        "report_cutoff_hong_kong": "2026-10-30T10:29:21+08:00",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_progress_updated_utc": progress["updated_utc"],
        "no_network_or_model_calls": True,
        "experimental_files_modified": False,
        "scope": "Only existing completed native records; compressed diagnostics are excluded from primary outcomes.",
        "progress": {
            "complete_jobs": 4, "planned_jobs": 27,
            "completed_rounds": 17, "planned_rounds": 54,
            "completed_heldout": 17, "planned_heldout": 108,
            "completed_records": len(rows),
            "distinct_questions": len({r["task_id"] for r in rows}),
            "phase_counts": dict(Counter(r["phase"] for r in rows)),
            "grade_recomputation_mismatches": 0,
        },
        "completed_jobs": completed,
        "all_domain_group_coverage": coverage,
        "all_domain_learning_panels": learning_panels,
        "business_initial_sensitivity_excluding_entire_zero_score_triplet": {
            "matched_triplets": 5, "means": sensitivity,
            "primary_zero_score_retained": True,
            "interpretation": "Sensitivity only, does not replace the full six-triplet primary result.",
        },
        "partial_job_heldout": {job: records for job, records in heldout.items() if job not in completed},
        "humanities_pair": {
            "pairs": comparison,
            "mean_difference": round(mean(p["difference"] for p in comparison), 3),
            "positive_tasks": sum(p["difference"] > 0 for p in comparison),
            "negative_tasks": sum(p["difference"] < 0 for p in comparison),
            "dimension_means": dims,
            "same_task_inputs": pair["same_task_order_and_full_input_rows"],
            "same_frozen_protocol": pair["identical_frozen_protocol"],
            "single_repetition": True,
            "independent_phi_causal_effect_established": False,
            "heldout_confirmed_token_ratio_full_over_native": full["heldout_confirmed_tokens"] / native["heldout_confirmed_tokens"],
        },
        "cost": {k: cost[k] for k in ["shared_ledger_recorded", "compressed_diagnostic_sdk", "native_historical_sdk", "native_sdk_by_role", "native_sdk_by_domain", "complete_humanities_pair_deltas_full_minus_native"]},
        "cost_scope_note": "Historical native requests include failures, recovery and incomplete jobs; unknown usage is not zero; no currency amount estimated.",
        "sources": [proof(p) for p in sources],
        "verified_completed_records": verified,
    }
    OUTPUT.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": relative(OUTPUT), "verified_records": len(rows), "complete_jobs": len(completed), "paired_difference": evidence["humanities_pair"]["mean_difference"], "sha256": sha(OUTPUT)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
