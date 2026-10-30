"""Read-only exact rollback plan for Cairn's committed, credit-incomplete task.

The three failed credit requests remain failed. A replacement credential must
complete credit normally; only the already successful extraction may replay.
"""
from copy import deepcopy
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import threading
from types import SimpleNamespace
import typing

JOB = "social-full-r2"
EXECUTION = "0eda9581acf54d52928c04189f2f2a58"
SOURCE_SHA256 = "994e0c9b287aecf42397b2a31b25f984ad0edaad89768d6a2909933e125ed2b5"
STEP_ATTEMPTS = [18527, 18555, 18623, 18707, 18775, 18838]
KINDS = ("strategic", "operational", "boundaries")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt(event):
    return event["request"]["messages"][0]["content"][0]["text"]


def response(event):
    choice = event["response"]["choices"][0]
    assert event["status"] == "success" and choice["finish_reason"] == "stop"
    assert event["usage"]["total_tokens"] > 0
    return json.loads(choice["message"]["content"])


def _semantic(value):
    value = deepcopy(value)
    for kind in KINDS:
        for item in value[kind]:
            item["tags"] = sorted(item.get("tags", []))
    return value


def _forward_replay(source_text, restored, observed, state, record, generated, by_id):
    """Execute native methods in RAM with exactly the recorded failed/successful replies."""
    tree = ast.parse(source_text)
    provider = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CairnMemoryProvider")
    names = {"take_in_memory", "_assess_delivery_compliance", "_extract_memories", "_admit_extracted",
             "_admit_memory", "_new_item", "_advance_lifecycle", "_prune_kind", "_utility_score",
             "_find_item_by_id", "_find_by_signature", "_find_similar", "_text_similarity", "_normalize",
             "_char_ngrams", "_signature", "_detect_language", "_trajectory_summary", "_compact_text",
             "_trajectory_outcome", "_outcome_label", "_parse_json", "_task_key"}
    methods = [node for node in provider.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    module = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name="Isolated", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[]))
    new_items = [item for kind in KINDS for item in observed[kind] if item["created_task_index"] == 2]
    generated_index = [-1]
    def replay_uuid():
        generated_index[0] += 1
        item = new_items[generated_index[0]]
        return SimpleNamespace(hex=item["id"].split("_", 1)[1] + "0" * 22)
    fake_time = SimpleNamespace(time=lambda: new_items[generated_index[0]]["created_at"])
    namespace = {**vars(typing), "time": fake_time, "uuid": SimpleNamespace(uuid4=replay_uuid),
                 "hashlib": hashlib, "math": math, "re": re, "json": json, "TrajectoryData": object}
    exec(compile(module, "isolated-cairn-v2-transaction", "exec"), namespace)
    memory = namespace["Isolated"]()
    memory._db, memory._task_counter = deepcopy(restored), 2
    key = hashlib.sha256(record["question"].encode()).hexdigest()[:32]
    memory._task_states, memory._lock = {key: deepcopy(state)}, threading.RLock()
    memory._initialized, memory.model = True, object()
    config = generated["config_updates"]
    for field, default in (("similarity_threshold", .72), ("max_extract_chars", 16000),
                           ("enable_residual_extraction", True), ("dormancy_after_tasks", 4),
                           ("retire_after_tasks", 10), ("max_strategic_memories", 24),
                           ("max_operational_memories", 24), ("max_boundary_memories", 24),
                           ("enable_delivery_compliance", True)):
        setattr(memory, field, config.get(field, default))
    orientation = record["runtime_artifact"]["orientation_snapshot"]["items"]
    memory.config = {"philo_orientation_items": orientation}
    memory.logger = SimpleNamespace(info=lambda *args: None, error=lambda *args, **kwargs: None)
    called = []
    def replay_call(text):
        if text == prompt(by_id[18970]):
            called.append("original_credit_failure")
            return ""
        if text == prompt(by_id[18996]):
            called.append("paid_extraction_18996")
            return by_id[18996]["response"]["choices"][0]["message"]["content"]
        raise AssertionError("Native transaction requested a new or mismatching prompt")
    memory._call_llm = replay_call
    saved = []
    memory._save_db = lambda: saved.append(deepcopy(memory._db))
    outcome = record["quality_score"] >= record["learning_success_threshold"]
    trajectory = SimpleNamespace(query=record["question"], trajectory=record["agent_messages"], result=record["agent_result"],
        metadata={"item_index": record["global_task_index"], "status": "success", "is_correct": outcome,
                  "quality_score": record["quality_score"], "quality_grade": record["grade"],
                  "judge_comment": record["grade"]["overall_comment"], "domain": record["domain"],
                  "full_query": record["enhanced_question"], "learning_success_threshold": record["learning_success_threshold"],
                  "fitness_semantics": "continuous_quality_not_binary_correctness", "philo_orientation_items": orientation})
    success, description = memory.take_in_memory(trajectory)
    assert success and "added=8, merged=0" in description and len(saved) == 1
    assert called == ["original_credit_failure", "paid_extraction_18996"]
    assert _semantic(saved[0]) == _semantic(observed), "Inverse transaction does not replay to the recorded committed state"
    assert generated_index[0] == 7 and memory._task_states == {}
    return {"native_transaction_reproduced": True, "new_model_requests": 0, "durable_writes": 0,
            "new_memory_records": 8, "merges": 0, "comparison": "all fields exact; tags compared as sets because native improvements construct tags through set()"}


def plan_cairn_full_ingestion(root, record, events):
    root = Path(root).resolve()
    job = root / "jobs" / JOB
    assert record["task_execution_id"] == EXECUTION and record["provider"] == "cairn_memory"
    assert record["status"] == "ingestion_failed" and record["memory_ingestion_started"] and not record["memory_ingestion_complete"]
    assert record["grade"]["quality_score"] == record["quality_score"] == 67.8
    source = job / "workspace/EvolveLab/providers/cairn_memory_provider.py"
    assert sha(source) == record["runtime_artifact"]["code_sha256"] == SOURCE_SHA256
    current = [event for event in events if event.get("task_execution_id") == EXECUTION]
    assert len(current) == 26
    by_id = {event["attempt"]: event for event in current}
    assert {event["attempt"] for event in current if event["status"] == "error"} == {18970, 18974, 18985, 18987, 18990}
    failed_credit = [by_id[number] for number in [18970, 18974, 18985]]
    assert all(event["role"] == "memory_ingestion" and event["http_status"] == 429
               and event["error_type"] == "RateLimitError" and event.get("response") is None and event.get("usage") is None for event in failed_credit)
    assert len({event["request_sha256"] for event in failed_credit}) == len({event["prompt_sha256"] for event in failed_credit}) == 1
    extracted = response(by_id[18996])
    assert by_id[18996]["recovered_attempts"] == [18987, 18990]
    assert {kind: len(extracted[kind]) for kind in (*KINDS, "improvements")} == {"strategic": 2, "operational": 2, "boundaries": 2, "improvements": 2}
    selected = response(by_id[18479])["selected_ids"]
    delivered = re.findall(r"^- id: (\S+)", prompt(by_id[18970]), re.M)
    assert len(selected) == len(delivered) == 6 and selected == delivered
    begin_rows = re.findall(r"- id: (\S+)\n  kind: ([^;]+); level: ([^;]+); state: (\S+)\n  confidence: ([\d.]+) \(delivered=(\d+), honoured=(\d+), violated=(\d+)\)", prompt(by_id[18479]))
    assert len(begin_rows) == 30
    begin = {row[0]: {"state": row[3], "delivered_count": int(row[5]), "honoured_count": int(row[6]), "violated_count": int(row[7])} for row in begin_rows}
    store_path = job / "workspace/storage/cairn_memory/cairn_store.json"
    observed = json.loads(store_path.read_text(encoding="utf8"))
    assert observed["meta"]["task_counter"] == 3
    added = {kind: [item for item in observed[kind] if item["created_task_index"] == 2] for kind in KINDS}
    assert {kind: len(items) for kind, items in added.items()} == {"strategic": 2, "operational": 2, "boundaries": 4}
    restored = deepcopy(observed)
    for kind in KINDS:
        restored[kind] = [item for item in restored[kind] if item["created_task_index"] != 2]
    assert {item["id"] for kind in KINDS for item in restored[kind]} == set(begin)
    for kind in KINDS:
        for item in restored[kind]:
            previous = begin[item["id"]]
            assert item["state"] == previous["state"] and item["honoured_count"] == previous["honoured_count"]
            assert item["violated_count"] == previous["violated_count"] == 0
            assert item["delivered_count"] == previous["delivered_count"] + int(item["id"] in delivered)
            if item["id"] in delivered:
                item["delivered_count"] -= 1
                assert item["usage_count"] >= 1 and item["last_selected_task_index"] == 2
    restored["meta"]["task_counter"] = 2
    state = {"query": record["question"], "current_step": 7, "delivered_memory_ids": delivered,
             "step_summaries": [{"step": step, "summary": response(by_id[number])["step_summary"].strip(), "phase": "in"}
                                for step, number in enumerate(STEP_ATTEMPTS, 2)]}
    generated_path = next(path for path in (job / "evolution/round_00").glob("generated_system_*.json")
                          if json.loads(path.read_text(encoding="utf8"))["memory_type_info"]["enum_value"] == "cairn_memory")
    generated = json.loads(generated_path.read_text(encoding="utf8"))
    proof = _forward_replay(source.read_text(encoding="utf8"), restored, observed, state, record, generated, by_id)
    text = json.dumps(restored, ensure_ascii=False, indent=2) + "\n"
    receipt_path = root / "budget/attempts/018996.json"
    key = hashlib.sha256(record["question"].encode()).hexdigest()[:32]
    return {"method": "rollback_cairn_full_v2_committed_ingestion_then_retry_credit",
            "reason": "Credit assessment failed three times; native fallback nevertheless committed eight extracted items. Undo only that transaction, preserve Actor usage, then require real credit recovery.",
            "provider_sha256": SOURCE_SHA256, "failed_attempts": [18970, 18974, 18985],
            "runtime_context": {"kind": "cairn_full_v2_preingestion", "db": restored, "task_counter": 2,
                                "task_states": {key: state}, "query": record["question"],
                                "paid_ingestion_replays": [{"attempt": 18996, "receipt_path": str(receipt_path), "receipt_sha256": sha(receipt_path)}]},
            "storage_rollbacks": [{"path": str(store_path.relative_to(root)), "before_sha256": sha(store_path),
                                   "after_sha256": hashlib.sha256(text.encode()).hexdigest(), "restored_utf8": text}],
            "pinned_files": {str(path.relative_to(root)): sha(path) for path in [source, store_path, generated_path,
                              *(root / "budget/attempts" / f"{event['attempt']:06d}.json" for event in current)]},
            "saved_answer_and_grade_reused": True, "new_credit_request_required": True,
            "offline_ingestion_retry_allowed": False, "failed_credit_requests_remain_failed": True,
            "forward_transaction_proof": proof}
