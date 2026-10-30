"""Evidence-driven recovery of complete paid Actors after handled connection errors."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from philoagent.hss_tool_calls import recover_final_answer

def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode("utf8")).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_actor_completion(root, record, events):
    """Promote captured complete trajectory; leave failed memory calls failed."""
    execution = record["task_execution_id"]
    if (record.get("status") != "actor_failed" or not isinstance(record.get("answer"), str)
            or not record["answer"].strip() or record.get("response_truncated")
            or record.get("actor_output_issue") or record.get("memory_ingestion_started")
            or record.get("memory_ingestion_complete")):
        return None
    root = Path(root).resolve()
    assert record["status"] == "actor_failed"
    assert not record.get("memory_ingestion_started") and not record.get("memory_ingestion_complete")
    assert not record.get("response_truncated") and not record.get("actor_output_issue")
    by_id = {e.get("attempt", e.get("attempt_id")): e for e in events}
    current = [e for e in events if e.get("task_execution_id") == execution]
    actors = [e for e in current if e.get("role") == "actor"]
    if not actors:
        return None
    final = max(actors, key=lambda e:e["attempt"])
    final_attempt = final["attempt"]
    assert final["task_execution_id"] == execution and final["role"] == "actor" and final["status"] == "success"
    assert all(c.get("finish_reason") == "stop" for c in final["response"]["choices"])
    decoded, _ = recover_final_answer(final["response"]["choices"][0]["message"],
                                       {"final_answer", "web_search", "crawl_page"})
    assert decoded["tools"][0]["arguments"]["answer"] == record["answer"] == record["agent_result"]
    messages = record.get("agent_messages", record.get("partial_agent_messages"))
    trajectory = record.get("agent_trajectory", record.get("partial_agent_trajectory"))
    assert isinstance(messages, list) and messages and isinstance(trajectory, list) and trajectory
    assert messages[-1]["role"] == "tool-response"
    assert any(record["answer"] in p.get("text", "") for p in messages[-1]["content"])
    assert trajectory[-1]["tool_calls"] == [{"name": "final_answer", "arguments": {"answer": record["answer"]}}]
    exact = []
    matched = set()
    for failed in current:
        if failed.get("status") != "error":
            continue
        fid = failed.get("attempt", failed.get("attempt_id"))
        success = next((e for e in current if e.get("status") == "success"
                        and e["attempt"] > fid and failed.get("request_sha256")
                        and e.get("request_sha256") == failed["request_sha256"]
                        and e.get("prompt_sha256") == failed.get("prompt_sha256")
                        and e.get("role") == failed.get("role")), None)
        if success:
            assert failed.get("response") is None
            exact.append({"failed_attempt": fid, "successful_attempt": success["attempt"]})
            matched.add(fid)
    fallback_ids = [e["attempt"] for e in current if e.get("status") == "error" and e["attempt"] not in matched]
    for fid in fallback_ids:
        event = by_id[fid]
        assert event["task_execution_id"] == execution and event["role"] == "memory_retrieval"
        assert event.get("status") == "error"
        assert event.get("response") is None and fid not in matched
    assert {e["attempt"] for e in current if e.get("status") == "error"} == matched | set(fallback_ids)
    promoted = deepcopy(record)
    promoted.update(status="answer_saved", agent_messages=deepcopy(messages), agent_trajectory=deepcopy(trajectory))
    existing = {i for x in promoted.get("phase_recovery_receipts", []) for i in x.get("recovered_attempts", [])}
    for link in exact:
        if link["failed_attempt"] not in existing:
            promoted.setdefault("phase_recovery_receipts", []).append({
                "phase": "same_audited_request", "successful_task_execution_id": execution,
                "successful_attempt": link["successful_attempt"],
                "recovered_attempts": [link["failed_attempt"]], "authorization_revision": "repair-r18"})
    # Lodestar's previous failed Actor was explicitly authorized and a whole new
    # Actor actually ran. Completion of that replacement is separate from an
    # exact SDK retry; it must never be credited to the later Judge.
    replay_ids = record.get("prior_api_usage", {}).get("unrecovered_failed_attempt_ids", [])
    if replay_ids:
        history = record["recovery_history"][-1]
        assert history["resumed_phase"] == "actor" and history["authorization"]["allow_actor_retry"]
        assert set(history["authorization"]["known_failed_attempts"]) == set(replay_ids)
        promoted.setdefault("phase_recovery_receipts", []).append({
            "phase": "actor", "successful_task_execution_id": execution,
            "recovered_attempts": list(replay_ids), "authorization_revision": history["authorization"]["revision"]})
    source = Path(record["runtime_artifact"]["source_path"])
    assert sha(source) == record["runtime_artifact"]["code_sha256"]
    receipt_ids = {final_attempt, *fallback_ids, *replay_ids}
    for link in exact:
        receipt_ids.update(link.values())
    pinned = {str(source.resolve()): sha(source)}
    for identifier in receipt_ids:
        path = root / "budget/attempts" / f"{identifier:06d}.json"
        pinned[str(path.resolve())] = sha(path)
    evidence = {"method": "complete_native_actor_after_standby", "revision": "repair-r18",
                "task_identity": record["task_identity"], "provider": record["provider"],
                "actor_execution_id": execution, "final_attempt": final_attempt,
                "saved_answer_sha256": hashlib.sha256(record["answer"].encode("utf8")).hexdigest(),
                "saved_messages_sha256": json_hash(messages), "saved_trajectory_sha256": json_hash(trajectory),
                "provider_sha256": record["runtime_artifact"]["code_sha256"],
                "exact_request_recoveries": exact, "completed_actor_replay_failures": replay_ids,
                "native_fallback_failed_attempt_ids": fallback_ids, "pinned_files": pinned,
                "memory_degradation": bool(fallback_ids),
                "degradation_reason": "Original native provider caught failed memory retrieval and continued with its own fallback; model request remains failed." if fallback_ids else None}
    promoted["native_actor_completion_recovery"] = deepcopy(evidence)
    return promoted, evidence
