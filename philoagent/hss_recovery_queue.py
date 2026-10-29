"""Persistent, one-pass repair dispatch beside an already drained supervisor.

Only explicitly selected lineages may start. Workers retain the existing native
manager, task identities and shared request budget. This module does no model
work and never initializes a job directory in the parent process.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .hss_budget import _file_lock, process_identity, process_matches
from .hss_experiment import FLASH, PROJECT, now, process_owned, read_json, source_hash, write_json


STATE_NAME = "recovery_supervisor.r9.json"
KIND = "hss_recovery_queue"


def _optional_json(path):
    return read_json(path) if path.exists() else {}


def _job_snapshot(root, jobs):
    result = {}
    for job in jobs:
        identifier = job["job_id"]
        target = root / "jobs" / identifier
        state = _optional_json(target / "job_status.json")
        saved = _optional_json(target / "result.json")
        complete = saved.get("status") == "complete" or state.get("status") == "complete"
        owned = bool(state.get("pid") and process_owned(state))
        status = "complete" if complete else state.get("status", "queued")
        if status == "running" and not owned:
            status = "interrupted"
        result[identifier] = {"status": status,
                              "owned": owned, "pid": state.get("pid")}
    return result


def _legacy_evidence(root, legacy, selected):
    """The old supervisor may keep its futures, but cannot still own these IDs.

    Its one-pass iterator has already consumed every selected ID only when its
    dispatch log contains their job_finished events. Pin this receipt with the
    legacy process identity. The caller supplies this narrow repair selection.
    """
    if not legacy or not process_owned(legacy):
        return {}
    path = root / "supervisor.repair-r7.log"
    content = path.read_bytes() if path.exists() else b""
    covered, evidence = set(), []
    for number, line in enumerate(content.splitlines(), 1):
        try:
            item = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if item.get("event") == "job_finished" and item.get("job_id") in selected:
            covered.add(item["job_id"])
            evidence.append({"line": number, "job_id": item["job_id"], "exit_code": item.get("exit_code")})
    if covered != set(selected):
        raise ValueError("Live legacy supervisor has no consumed dispatch evidence for every selected job")
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(),
            "legacy_pid": legacy.get("pid"), "legacy_process_identity": legacy.get("process_identity"),
            "covered_job_ids": list(selected), "events": evidence, "recorded_utc": now()}


def _source_wait(manifest):
    if source_hash(PROJECT / "philoagent") != manifest["adapter_sha256"]:
        return "adapter_source_not_frozen"
    if source_hash(FLASH) != manifest["upstream_sha256"]:
        return "native_source_not_frozen"
    return None


def _budget_wait(root):
    # Read only; HSSBudget remains the authority for admission and its six slots.
    state = _optional_json(root / "budget/budget.json")
    if state.get("blocked_reason"):
        return "budget_blocked:" + str(state["blocked_reason"])
    limits = state.get("limits", {})
    if limits.get("max_calls") and state.get("attempt_count", 0) >= limits["max_calls"]:
        return "budget_request_cap"
    effective = state.get("actual_tokens", 0) + state.get("reserved_unknown_tokens", 0)
    if limits.get("max_tokens") and effective >= limits["max_tokens"]:
        return "budget_token_stop"
    return None


def _event(root, identifier, event, **fields):
    value = {"event": event, "job_id": identifier, "recorded_utc": now(), **fields}
    with (root / "queue_logs" / f"{identifier}.events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _publish(root, state, snapshot, active, waiting=None):
    state["updated_utc"] = now()
    state["waiting_reason"] = waiting
    state["attempted_job_ids"] = [i for i in state["job_ids"] if state["jobs"][i].get("attempted")]
    state["skipped_job_ids"] = [i for i in state["job_ids"] if state["jobs"][i]["status"].startswith("skipped_")]
    state["pending_job_ids"] = [i for i in state["job_ids"] if state["jobs"][i]["status"] == "pending"]
    owned = {i for i, item in snapshot.items() if item["owned"]} | set(active)
    state["owned_running_job_ids"] = sorted(owned)
    state["counts"] = {key: sum(item["status"] == key for item in snapshot.values())
                       for key in ("queued", "running", "complete", "failed", "interrupted")}
    state["counts"]["owned_running"] = len(owned)
    state["failed_job_ids"] = [i for i in state["job_ids"] if state["jobs"][i]["status"] in
                               {"failed", "launch_failed", "ambiguous_dispatch"}]
    state["unstarted_job_ids"] = [i for i in state["job_ids"] if not state["jobs"][i].get("attempted")]
    write_json(root / STATE_NAME, state)
    # Independent state is canonical; refresh the UI alias after a legacy finish.
    write_json(root / "supervisor.json", state)


def run_queue(root, job_ids, max_running=10, *, poll_seconds=5):
    """Run an explicit persistent repair queue; each ID gets at most one spawn.

    Restart with the same ID set to observe previous attempts, never replay them.
    New failure recovery requires a new explicitly authorized queue receipt.
    """
    root = Path(root).resolve(strict=True)
    if type(max_running) is not int or max_running <= 0 or poll_seconds <= 0:
        raise ValueError("Positive worker cap and polling interval are required")
    selected = list(dict.fromkeys(job_ids))
    manifest = read_json(root / "manifest.json")
    known = {j["job_id"] for j in manifest["jobs"]}
    if not selected or set(selected) - known:
        raise ValueError("Recovery IDs must be nonempty and belong to the frozen manifest")
    # Manifest IDs become log filenames; reject path traversal even in fixtures.
    if any(Path(i).name != i or "/" in i or "\\" in i for i in selected):
        raise ValueError("Unsafe recovery job identifier")
    (root / "queue_logs").mkdir(exist_ok=True)
    # Hold one OS lifetime lock. PID reuse or a stale JSON status cannot grant it.
    with _file_lock(root / "queue_logs/recovery-r9.lock", 0):
        # Read after acquiring: a previous owner may have consumed IDs while we waited.
        previous = _optional_json(root / STATE_NAME)
        if previous and (previous.get("kind") != KIND or set(previous.get("job_ids", [])) != set(selected)):
            raise ValueError("Existing recovery queue has a different explicit selection")
        legacy = previous.get("legacy_supervisor", {}) if previous else _optional_json(root / "supervisor.json")
        if legacy.get("kind") == KIND:
            legacy = {}
        evidence = previous.get("legacy_dispatch_evidence", {}) if previous else _legacy_evidence(root, legacy, selected)
        state = previous or {"schema_version": 1, "kind": KIND, "revision": "repair-r9",
            "started_utc": now(), "job_ids": selected, "legacy_supervisor": legacy,
            "legacy_dispatch_evidence": evidence,
            "jobs": {i: {"status": "pending", "attempted": False} for i in selected}}
        state.update(status="running", pid=os.getpid(), process_identity=process_identity(os.getpid()),
                     max_running=max_running)
        active = {}
        for identifier, receipt in state["jobs"].items():
            if receipt["status"] == "dispatching":
                receipt.update(status="ambiguous_dispatch", error_type="UnconfirmedSpawnAfterAdmission")
                _event(root, identifier, "ambiguous_dispatch", error_type=receipt["error_type"])
            elif receipt["status"] == "running":
                active[identifier] = None  # Resume monitoring its exact saved process identity.
        try:
            while True:
                manifest = read_json(root / "manifest.json")
                snapshot = _job_snapshot(root, manifest["jobs"])
                for identifier, child in list(active.items()):
                    receipt = state["jobs"][identifier]
                    if child is None:
                        observed = receipt.get("observed_unix")
                        if observed is None and receipt.get("started_utc"):
                            observed = datetime.fromisoformat(receipt["started_utc"]).timestamp()
                        identity = receipt.get("process_identity")
                        alive = bool(identity or observed is not None) and process_matches(
                            receipt.get("pid"), identity=identity, started_unix=observed, executable="python.exe")
                        if alive:
                            continue
                        code = None
                    else:
                        code = child.poll()
                        if code is None:
                            continue
                    current = _job_snapshot(root, [{"job_id": identifier}])[identifier]
                    complete = current["status"] == "complete" and (code == 0 or child is None)
                    receipt.update(status="complete" if complete else "failed", exit_code=code, finished_utc=now())
                    if not complete:
                        receipt["error_type"] = "WorkerExited" if code else "WorkerCompletionUnconfirmed"
                    _event(root, identifier, "worker_finished", **receipt)
                    del active[identifier]
                snapshot = _job_snapshot(root, manifest["jobs"])
                # Check the full matrix, including workers owned by the legacy supervisor.
                for identifier in selected:
                    receipt = state["jobs"][identifier]
                    if receipt["status"] != "pending":
                        continue
                    current = snapshot[identifier]
                    if current["status"] == "complete" or current["owned"]:
                        receipt.update(status="skipped_complete" if current["status"] == "complete" else "skipped_running",
                                       skipped_utc=now(), observed_pid=current["pid"])
                        _event(root, identifier, receipt["status"], observed_pid=current["pid"])
                pending = [i for i in selected if state["jobs"][i]["status"] == "pending"]
                owned = {i for i, item in snapshot.items() if item["owned"]} | set(active)
                if not pending and not owned:
                    if legacy and process_owned(legacy):
                        # Its final write may replace the alias; wait and publish last.
                        _publish(root, state, snapshot, active, "legacy_supervisor_finishing")
                        time.sleep(poll_seconds)
                        continue
                    all_complete = all(item["status"] == "complete" for item in snapshot.values())
                    any_failed = any(r["attempted"] and r["status"] != "complete" for r in state["jobs"].values())
                    state["status"] = "complete" if all_complete and not any_failed else "needs_attention"
                    state["finished_utc"] = now()
                    _publish(root, state, snapshot, active)
                    return state
                waiting = "worker_cap" if len(owned) >= max_running else None
                if pending and not waiting:
                    waiting = _source_wait(manifest) or _budget_wait(root)
                if pending and not waiting:
                    identifier = pending[0]
                    # Recheck immediately before admitting; no directory is created here.
                    snapshot = _job_snapshot(root, manifest["jobs"])
                    owned = {i for i, item in snapshot.items() if item["owned"]} | set(active)
                    if snapshot[identifier]["owned"] or snapshot[identifier]["status"] == "complete" or len(owned) >= max_running:
                        continue
                    receipt = state["jobs"][identifier]
                    receipt.update(status="dispatching", attempted=True, admitted_utc=now(),
                                   adapter_sha256=manifest["adapter_sha256"], adapter_revision=manifest.get("adapter_revision"),
                                   log_path=str(root / "queue_logs" / f"{identifier}.log"))
                    _publish(root, state, snapshot, active)
                    _event(root, identifier, "dispatch_admitted", **receipt)
                    try:
                        env = os.environ.copy()
                        env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
                        with Path(receipt["log_path"]).open("ab") as log:
                            child = subprocess.Popen([sys.executable, "-u", "-m", "philoagent.hss_experiment", "run-job", str(root), identifier],
                                cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
                        active[identifier] = child
                        receipt.update(status="running", pid=child.pid, observed_unix=time.time(),
                                       process_identity=process_identity(child.pid), started_utc=now())
                        _event(root, identifier, "worker_started", **receipt)
                    except Exception as exc:
                        receipt.update(status="launch_failed", error_type=type(exc).__name__, finished_utc=now())
                        _event(root, identifier, "launch_failed", error_type=type(exc).__name__)
                    _publish(root, state, snapshot, active)
                    continue
                _publish(root, state, snapshot, active, waiting or "waiting_existing_workers")
                time.sleep(poll_seconds)
        except BaseException as exc:
            # Leave children alone. Their exact identities permit safe observation on restart.
            state.update(status="failed", error_type=type(exc).__name__, finished_utc=now())
            _publish(root, state, _job_snapshot(root, manifest["jobs"]), active)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("job_ids", nargs="+")
    parser.add_argument("--max-running", type=int, default=10)
    arguments = parser.parse_args(argv)
    result = run_queue(arguments.root, arguments.job_ids, arguments.max_running)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "attempted_job_ids": result["attempted_job_ids"]}), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
