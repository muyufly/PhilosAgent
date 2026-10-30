"""Run only the prepared Verity ingestion with networking disabled.

First publish r21 and prepare the selected incident with resume_prepare.py.
Default writes nothing and emits a pinned plan. --apply --plan FILE verifies
that plan and invokes only NativeHSSRunner._run_task for the saved task.
It never starts an Actor, Judge, whole lineage, or partial batch summary.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from philoagent.ablation_runtime import digest, items_digest
from philoagent.cli import source_hash
from philoagent.hss_experiment import process_owned, read_json, write_json
from philoagent.hss_tasks_runtime import NativeHSSRunner, classify_recovery_record
from philoagent.orientation import load
from philoagent.runtime import OrientedProvider
from research.verity_committed_recovery_20261030 import METHOD, SOURCE_SHA256

JOB = "social-fixed-r2"
PROVIDER = "verity_loom_memory"
RELATIVE = Path("jobs/social-fixed-r2/evolution/round_00/eval_batch/verity_loom_memory/1.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forbidden(*args, **kwargs):
    raise RuntimeError("Offline Verity recovery forbids a model, Actor, Judge or network request")


@contextmanager
def no_network():
    with patch.object(socket.socket, "connect", forbidden), patch.object(socket.socket, "connect_ex", forbidden), \
            patch.object(socket, "create_connection", forbidden), patch.object(socket, "getaddrinfo", forbidden), \
            patch.dict(os.environ, {"PHILO_HSS_OBSERVE_ACCOUNTING": "1", "PHILO_HSS_DATA_FIRST": "0",
                                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
        yield


def artifact(root, record):
    job = root / "jobs" / JOB
    old = record["runtime_artifact"]
    source = job / "workspace/EvolveLab/providers/verity_loom_memory_provider.py"
    assert sha(source) == old["code_sha256"] == SOURCE_SHA256
    snapshot = load(job / "lineage", PROVIDER)
    result = {"provider": PROVIDER, "module": old["module"], "class_name": old["class_name"],
              "source_path": str(source.resolve()), "code_sha256": sha(source),
              "orientation_snapshot": snapshot.to_dict(), "orientation_items_sha256": items_digest(snapshot.items)}
    assert result == old, "Saved source or philosophical orientation changed"
    return result


def plan(root):
    root = Path(root).resolve(strict=True)
    path = root / RELATIVE
    record = read_json(path)
    assert record["provider"] == PROVIDER and record["task_id"] == "WB0333"
    assert record["quality_score"] == 75.4 and len(record["answer"]) == 3796
    assert classify_recovery_record(record) == "ingestion" and record["status"] == "graded"
    assert record["native_ingestion_recovery"]["method"] == METHOD
    assert NativeHSSRunner._verified_ingestion_rollback(record, record["recovery_authorized"])
    job = root / "jobs" / JOB
    for name in ("job_status.json", "worker_source.json"):
        if (job / name).exists():
            assert not process_owned(read_json(job / name)), "The target lineage still has an owned worker"
    manifest = read_json(root / "manifest.json")
    assert manifest["adapter_revision"] == "repair-r21"
    assert source_hash(PROJECT / "philoagent") == manifest["adapter_sha256"], "Publish r21 before applying offline recovery"
    current = artifact(root, record)
    paths = [path, root / "manifest.json", job / "learning.jsonl", Path(current["source_path"]),
             Path(record["native_ingestion_recovery"]["receipt_path"]), job / "lineage" / PROVIDER / "orientation.json"]
    paths += [p for p in (job / "workspace/storage" / PROVIDER).iterdir() if p.is_file()]
    # Confirm model-request accounting is unchanged by this local operation.
    paths += [p for p in (job / "api_calls.jsonl", root / "budget/budget.json") if p.exists()]
    return {"kind": "verity_single_task_exact_offline_ingestion", "root": str(root),
            "task_record": str(path), "record_sha256": sha(path),
            "pinned_files": {str(p.resolve()): sha(p) for p in paths},
            "saved_api_usage": record["api_usage"], "quality_score": 75.4,
            "answer_sha256": digest(record["answer"].encode("utf8")),
            "new_model_request": False, "batch_summary_published": False}


def make_runner(root, record):
    job = root / "jobs" / JOB
    workspace = job / "workspace"
    sys.path.insert(0, str(workspace))
    from EvolveLab.config import get_memory_config
    from EvolveLab.memory_types import MemoryType, TrajectoryData
    current = artifact(root, record)
    source = Path(current["source_path"])
    spec = importlib.util.spec_from_file_location("r21_exact_original_verity_provider", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    snapshot = load(job / "lineage", PROVIDER)
    def provider_loader(provider, model):
        assert provider == PROVIDER
        config = deepcopy(get_memory_config(MemoryType(PROVIDER)))
        config.update(model=model, philo_orientation_items=list(snapshot.items))
        inner = module.VerityLoomMemoryProvider(config)
        outer = OrientedProvider(inner, snapshot, source, SOURCE_SHA256)
        assert outer.initialize()
        return outer
    manifest = read_json(root / "manifest.json")
    callback = NativeHSSRunner(job, workspace, manifest["protocol"], {"model_id": manifest["protocol"]["model"]},
        runner=SimpleNamespace(load_memory_provider=provider_loader, TrajectoryData=TrajectoryData),
        native=SimpleNamespace(ToolCallingAgent=forbidden, OpenAIServerModel=forbidden),
        base_agent_class=forbidden, usage_reader=lambda: [])
    callback._new_model = forbidden
    callback._artifact = lambda provider: artifact(root, record) if provider == PROVIDER else forbidden()
    return callback


def apply(value):
    assert value["kind"] == "verity_single_task_exact_offline_ingestion"
    root = Path(value["root"]).resolve(strict=True)
    assert plan(root) == value, "Reviewed offline plan changed"
    path = root / RELATIVE
    before = read_json(path)
    output = root / "native_recovery_20261030/verity_offline_r21" / uuid4().hex
    output.mkdir(parents=True)
    shutil.copy2(path, output / "record.before.json")
    dataset = [json.loads(line) for line in (root / "jobs" / JOB / "learning.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    index = before["global_task_index"]
    assert index == 0 and dataset[index]["task_id"] == before["task_id"]
    previous_cwd = Path.cwd()
    old_evidence = os.environ.get("PHILO_EVIDENCE_DIR")
    try:
        os.chdir(root / "jobs" / JOB / "workspace")
        os.environ["PHILO_EVIDENCE_DIR"] = str(output / "runtime_evidence")
        with no_network():
            callback = make_runner(root, before)
            result = callback._run_task(dataset[index], index, PROVIDER, path.parent,
                artifact=before["runtime_artifact"], stage=before["stage"], stage_id="r21-exact-local-ingestion", dataset_name="writingbench")
    finally:
        os.chdir(previous_cwd)
        if old_evidence is None:
            os.environ.pop("PHILO_EVIDENCE_DIR", None)
        else:
            os.environ["PHILO_EVIDENCE_DIR"] = old_evidence
    assert result["status"] == "completed" and result["memory_ingestion_complete"]
    assert result["answer"] == before["answer"] and result["grade"] == before["grade"]
    assert result["api_usage_current_execution"]["request_count"] == 0
    assert [x["original_attempt"] for x in result["local_provider_replays"]] == [15359, 15391]
    for key in ("request_count", "failed_request_count", "unknown_usage_attempts", "total_tokens"):
        assert result["api_usage"][key] == before["api_usage"][key]
    for name, digest_before in value["pinned_files"].items():
        if name.endswith("api_calls.jsonl") or name.endswith("budget/budget.json") or name.endswith("budget\\budget.json"):
            assert sha(name) == digest_before
    report = {"status": "completed", "record_path": str(path), "record_after_sha256": sha(path),
              "original_record_backup_path": str(output / "record.before.json"),
              "original_record_backup_sha256": value["record_sha256"],
              "ingestion_rollback_receipt": before["native_ingestion_recovery"]["receipt_path"],
              "original_successful_attempts_reused": [15359, 15391], "new_model_request": False,
              "new_sdk_success": False, "api_usage": result["api_usage"],
              "quality_score": result["quality_score"], "answer_sha256": value["answer_sha256"],
              "batch_summary_published": False}
    write_json(output / "verification.json", report)
    return {**report, "verification": str(output / "verification.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.apply:
        assert args.plan, "--apply requires the saved reviewed plan"
        print(json.dumps(apply(read_json(args.plan)), ensure_ascii=False))
    else:
        print(json.dumps(plan(args.root), ensure_ascii=False))
