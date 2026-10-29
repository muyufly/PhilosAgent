"""Restore complete paid answers and rewind learning based on parser damage."""

from copy import deepcopy
from pathlib import Path
import shutil

from .hss_experiment import read_json, write_json, initialize_job
from .hss_repair_r3 import (amend as migrate, release as release_revision,
                            repaired_protocol as r3_protocol, task_usage)
from .hss_tool_calls import recover_final_answer


REVISION = "repair-r4"
MAINTENANCE = "maintenance_answer_integrity_20261029"
AFFECTED = ("social-native-r1", "social-fixed-r1")


def repaired_protocol(protocol):
    result = r3_protocol(protocol)
    result.update(revision=REVISION, actor_recovery_max_attempts=3,
                  tool_decoding="lossless_final_answer_v2", implicit_repair_client_retries=0)
    return result


def corrected_record(old, payload, extraction, usage, revision_root, archived_job):
    answer = payload["tools"][0]["arguments"]["answer"]
    if not answer.startswith(old["answer"]) or len(answer) <= len(old["answer"]):
        raise ValueError("Paid answer does not prove the observed prefix loss")
    result = deepcopy(old)
    original_execution = result["task_execution_id"]
    for key in ("grade", "quality_score", "score", "judgement", "judge_output", "judge_normalization",
                "judge_recovery_attempts", "memory_ingestion_started", "error_type", "prior_api_usage",
                "response_truncated", "actor_output_issue"):
        result.pop(key, None)
    result.update(answer=answer, agent_result=answer, status="answer_saved",
                  memory_ingestion_complete=False, api_usage=usage,
                  total_model_tokens=usage["total_tokens"],
                  answer_extraction_recovery={"revision": REVISION, "extraction": extraction,
                     "old_answer_chars": len(old["answer"]), "new_answer_chars": len(answer),
                     "old_grade": deepcopy(old.get("grade")), "archived_job": str(archived_job),
                     "native_memory_reset": True, "new_actor_requests": 0})
    result["recovery_authorized"] = {"revision": REVISION, "allow_actor_retry": False,
        "reason": "Recover complete paid answer, freshly judge it and ingest once into restored initial memory",
        "known_failed_attempts": usage["unrecovered_failed_attempt_ids"],
        "reserved_unknown_attempts": usage["reserved_unknown_attempt_ids"]}
    result["recovery_history"] = [{"archive": str(archived_job / "evolution/round_00/base_logs/1.json"),
        "prior_status": old["status"], "prior_task_execution_id": original_execution,
        "resumed_phase": "judge", "revision": REVISION}]
    return result


def amend(root):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    revision = root / "revisions" / REVISION
    if manifest["protocol"].get("revision") == REVISION:
        evidence = read_json(revision / "amendment.json")
        if evidence.get("answer_restoration_complete") is not True:
            raise ValueError("Partial answer restoration exists; inspect before proceeding")
        return evidence
    attempts = [read_json(path) for path in sorted((root / "budget/attempts").glob("*.json"))]
    plans = []
    for job_id in AFFECTED:
        job = root / "jobs" / job_id
        old = read_json(job / "evolution/round_00/base_logs/1.json")
        ids = set(old.get("task_execution_ids", [])) | {old["task_execution_id"]}
        responses = [event for event in attempts if event.get("task_execution_id") in ids
                     and event.get("role") == "actor" and event.get("status") == "success"]
        response = max(responses, key=lambda item: item["attempt"])
        choices = response["response"]["choices"]
        if len(choices) != 1 or choices[0]["finish_reason"] != "stop":
            raise ValueError("Paid final answer is incomplete")
        payload, extraction = recover_final_answer(choices[0]["message"], {"final_answer"})
        extraction["source_attempt"] = response["attempt"]
        archived_job = revision / "restarted_jobs" / job_id
        if not job.resolve().is_relative_to(root / "jobs") or not archived_job.resolve().is_relative_to(revision):
            raise ValueError("Restoration archive escaped the experiment")
        corrected_record(old, payload, extraction, task_usage(old, attempts), revision, archived_job)
        plans.append((job_id, old, payload, extraction, archived_job))
    evidence = migrate(root, revision_name=REVISION, maintenance_reason=MAINTENANCE,
                       protocol_builder=repaired_protocol)
    new_manifest = read_json(root / "manifest.json")
    restored = []
    for job_id, original, payload, extraction, archived_job in plans:
        job = root / "jobs" / job_id
        migrated = read_json(job / "evolution/round_00/base_logs/1.json")
        archived_job.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(job), str(archived_job))
        spec = next(item for item in new_manifest["jobs"] if item["job_id"] == job_id)
        target = initialize_job(root, spec)
        record = corrected_record(migrated, payload, extraction, task_usage(original, attempts),
                                  revision, archived_job)
        write_json(target / "evolution/round_00/base_logs/1.json", record)
        restored.append({"job_id": job_id, "source_attempt": extraction["source_attempt"],
            "old_answer_chars": len(original["answer"]), "new_answer_chars": len(record["answer"]),
            "old_score_invalidated": original.get("quality_score"), "archive": str(archived_job),
            "actor_replayed": False, "later_tasks_and_derived_candidates_restarted": True})
    evidence.update(restored_answers=restored, answer_restoration_complete=True,
                    user_message="继续跟进，可以分析一下阶段性数据；此前授权异常修复")
    write_json(revision / "amendment.json", evidence)
    return evidence


def release(root):
    evidence = read_json(Path(root) / "revisions" / REVISION / "amendment.json")
    if not evidence.get("answer_restoration_complete"):
        raise ValueError("Restoration has not completed")
    release_revision(root, revision_name=REVISION, maintenance_reason=MAINTENANCE)


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    result = amend(args.root)
    if args.release:
        release(args.root)
    print(json.dumps(result["restored_answers"], ensure_ascii=False))
