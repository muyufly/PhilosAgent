"""Native open-writing task callback for the v0.0.6 experiments.

The controller owns provider refresh, model-client auditing and evolution. This
callback preserves each provider's ordinary storage and writes paid answers
before running the independent quality judge or memory ingestion.
"""

from __future__ import annotations

import importlib
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
from copy import deepcopy

from .ablation_runtime import _provider_artifact, audit_scope, digest, write_json
from .hss_evaluation import judge_messages, native_quality_summary, normalize_grade_output, rubric, validate_grade

VERITY_LOOM_SOURCE_SHA256 = "fd2cfd885e682654a8e34b85b1813c5c89b9c20dac324460f8c62c3acb3bcc7e"
CAIRN_SOURCE_SHA256 = "7843293e36224af76e99e613aa91c63d8c56b95cd33a7fae15e15a10fbfc91fc"
CAIRN_FULL_V2_SOURCE_SHA256 = "994e0c9b287aecf42397b2a31b25f984ad0edaad89768d6a2909933e125ed2b5"


class HSSResumeRequired(RuntimeError):
    """An incomplete paid attempt needs an explicit recovery decision."""


class _ExactPaidReplayMismatch(BaseException):
    """Exit native catch-Exception fallbacks before they can commit empty work.

    Only the signed, offline Verity incident uses this sentinel. The task
    boundary catches it and preserves the failed record normally.
    """


def _response_usable_without_usage(event):
    """Only the audited complete-response exception is an accounting-only error."""
    return (event.get("error_type") == "MissingOrInvalidUsage"
            and event.get("response_accepted_without_usage") is True
            and bool(event.get("response_has_content") or event.get("response_has_tool_calls"))
            and "length" not in event.get("response_finish_reasons", []))


def classify_recovery_record(record):
    """Return the next safe task phase without changing paid artifacts or memory.

    An ingestion call can mutate storage before failing. Its start marker is
    therefore a hard ambiguity boundary, rather than permission to try again.
    """
    if record.get("status") == "completed":
        return "completed"
    if (record.get("memory_ingestion_started") or record.get("status") == "ingestion_failed") and not record.get("memory_ingestion_complete"):
        return "blocked_ingestion"
    if record.get("response_truncated") or record.get("actor_output_issue") in (
        "empty_final_answer", "audited_finish_reason_length"
    ) or not isinstance(record.get("answer"), str) or not record["answer"].strip():
        return "actor"
    if record.get("memory_ingestion_complete"):
        return "integrity"
    if record.get("stage") != "heldout" and not isinstance(record.get("agent_messages"), list):
        return "blocked_trajectory"
    return "ingestion" if isinstance(record.get("grade"), dict) else "judge"


def task_prompt(task):
    """Preserve the official query; add separately supplied public material only."""
    question = task.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("An HSS task requires a nonempty question")
    rubric(task.get("domain"))
    parts = [question]
    if task.get("materials"):
        parts.append("\n材料：\n" + json.dumps(task["materials"], ensure_ascii=False))
    if task.get("deliverable"):
        parts.append("\n交付要求：\n" + str(task["deliverable"]))
    return "\n".join(parts)


class NativeHSSRunner:
    """AutoEvolver ``run_provider`` callback using real native agents/providers.

    ``usage_reader`` must return the current process's audited SDK event list.
    Every task has a unique audit scope so simultaneous candidate evaluations do
    not claim one another's tokens. A fresh provider instance per task reopens
    its persistent M; storage is never cleared or rebuilt here.
    """

    def __init__(self, run_dir, workspace, protocol, model_config, *, runner=None,
                 native=None, base_agent_class=None, usage_reader=None, heldout=False):
        if usage_reader is None:
            raise ValueError("HSS execution requires an audited usage_reader")
        # Native modules load dotenv outside the isolated experiment runtime.
        # Do not accidentally read local credentials during offline imports.
        if (runner is None or base_agent_class is None) and not os.getenv("PHILO_LINEAGE_ROOT"):
            raise RuntimeError("Native imports require the isolated PHILO_LINEAGE_ROOT runtime")
        self.run_dir = Path(run_dir).resolve()
        self.workspace = Path(workspace).resolve()
        self.protocol = dict(protocol)
        self.model_config = dict(model_config)
        self.runner = runner or importlib.import_module("run_flash_searcher_webwalkerqa")
        self.native = native or importlib.import_module("FlashOAgents")
        self.base_agent_class = base_agent_class or importlib.import_module("base_agent").BaseAgent
        self.usage_reader = usage_reader
        self.heldout = heldout
        # Explicit operator override: collect useful results while retaining
        # incomplete accounting instead of stopping the entire experiment.
        self.data_first = os.getenv("PHILO_HSS_DATA_FIRST") == "1"
        self.stage_events = []
        self._locks = {}
        self._locks_lock = threading.Lock()

    @property
    def observe_accounting(self):
        """Unknown cost may remain unknown without disabling task quality checks."""
        return self.data_first or os.getenv("PHILO_HSS_OBSERVE_ACCOUNTING") == "1"

    def _lock(self, provider):
        with self._locks_lock:
            return self._locks.setdefault(provider, threading.Lock())

    def _artifact(self, provider):
        return _provider_artifact(self.run_dir, self.workspace, provider)

    def _usage(self, execution_id):
        events = [event for event in self.usage_reader()
                  if event.get("task_execution_id") == execution_id]
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        complete = accounted = True
        unknown, reserved_tokens, reserved_ids, unreserved_ids = 0, 0, [], []
        for event in events:
            usage = event.get("usage")
            if not isinstance(usage, dict) or any(
                type(usage.get(key)) is not int or usage[key] < 0 for key in totals
            ) or usage["total_tokens"] < usage["prompt_tokens"] + usage["completion_tokens"]:
                complete = False
                unknown += 1
                receipt = event.get("usage_reservation", {})
                identifier = event.get("attempt_id", event.get("attempt"))
                reserved = event.get("reserved_unknown_tokens")
                receipt_hash = receipt.get("original_receipt_sha256") if isinstance(receipt, dict) else None
                valid_reservation = (
                    event.get("status") == "error" and event.get("error_type") in ("APIConnectionError", "APITimeoutError")
                    and type(identifier) is int and identifier > 0 and type(reserved) is int and reserved > 0
                    and isinstance(receipt, dict) and receipt.get("reserved_unknown_tokens") == reserved
                    and receipt.get("method") == "conservative_text_request_bound"
                    and receipt.get("confirmed_actual_usage") is False
                    and isinstance(receipt_hash, str) and len(receipt_hash) == 64
                    and all(character in "0123456789abcdef" for character in receipt_hash))
                if valid_reservation:
                    reserved_tokens += reserved
                    reserved_ids.append(identifier)
                else:
                    accounted = False
                    unreserved_ids.append(identifier)
                continue
            for key in totals:
                totals[key] += usage[key]
        recovered = {attempt for event in events
                     if event.get("status") == "success" or _response_usable_without_usage(event)
                     for attempt in event.get("recovered_attempts", [])}
        failures = [event for event in events if event.get("status") != "success"]
        # The SDK explicitly distinguishes a complete usable response with an
        # absent usage receipt from a response/transport failure. Keep its cost
        # unknown and its accounting error visible; do not call it a failed work.
        accounting_only = [event for event in failures if _response_usable_without_usage(event)]
        unresolved = [event for event in failures if event.get("attempt_id", event.get("attempt")) not in recovered
                      and event not in accounting_only]
        return {**totals, "request_count": len(events), "usage_complete": complete, "usage_accounted": accounted,
                "unknown_usage_attempts": unknown, "reserved_unknown_tokens": reserved_tokens,
                "unreserved_unknown_attempt_ids": unreserved_ids,
                "reserved_unknown_attempt_ids": reserved_ids, "effective_tokens": totals["total_tokens"] + reserved_tokens,
                "failed_request_count": len(failures),
                "unrecovered_failed_request_count": len(unresolved),
                "unrecovered_failed_attempt_ids": [event.get("attempt_id", event.get("attempt")) for event in unresolved]}

    def _save(self, path, record, started):
        current = self._usage(record["task_execution_id"])
        prior = record.get("prior_api_usage", {})
        usage = {key: prior.get(key, 0) + current[key]
                 for key in ("prompt_tokens", "completion_tokens", "total_tokens", "request_count", "failed_request_count")}
        usage["usage_complete"] = prior.get("usage_complete", True) and current["usage_complete"]
        usage["usage_accounted"] = prior.get("usage_accounted", prior.get("usage_complete", True)) and current["usage_accounted"]
        usage["reserved_unknown_tokens"] = prior.get("reserved_unknown_tokens", 0) + current["reserved_unknown_tokens"]
        usage["reserved_unknown_attempt_ids"] = list(prior.get("reserved_unknown_attempt_ids", [])) + current["reserved_unknown_attempt_ids"]
        usage["unknown_usage_attempts"] = prior.get("unknown_usage_attempts", 0) + current["unknown_usage_attempts"]
        usage["unreserved_unknown_attempt_ids"] = (
            prior.get("unreserved_unknown_attempt_ids", []) + current["unreserved_unknown_attempt_ids"])
        if self.observe_accounting:
            usage["accounting_policy"] = "observe_only"
            record["data_collection_mode"] = "data_first" if self.data_first else "native_quality_checked"
        usage["effective_tokens"] = usage["total_tokens"] + usage["reserved_unknown_tokens"]
        usage["unrecovered_failed_request_count"] = (
            prior.get("unrecovered_failed_request_count", prior.get("failed_request_count", 0))
            + current["unrecovered_failed_request_count"])
        usage["unrecovered_failed_attempt_ids"] = (
            prior.get("unrecovered_failed_attempt_ids", []) + current["unrecovered_failed_attempt_ids"])
        fallback_ids = self._verified_actor_completion(record)
        usage["blocking_failed_attempt_ids"] = [identifier for identifier in usage["unrecovered_failed_attempt_ids"]
                                                if identifier not in fallback_ids]
        usage["blocking_failed_request_count"] = (usage["unrecovered_failed_request_count"]
                                                   - sum(identifier in fallback_ids for identifier in usage["unrecovered_failed_attempt_ids"]))
        record.update(total_model_tokens=usage["total_tokens"], api_usage=usage,
                      confirmed_model_tokens=usage["total_tokens"], effective_model_tokens=usage["effective_tokens"],
                      api_usage_current_execution=current,
                      metrics={**{key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
                               "elapsed_time": round(record.get("prior_elapsed_time", 0) + time.monotonic() - started, 3)})
        write_json(path, record)

    def _new_model(self, *, judge=False):
        field = "judge_max_output_tokens" if judge else "actor_max_output_tokens"
        config = {**self.model_config, "max_tokens": self.protocol.get(field, 8192 if judge else 32768)}
        if judge:
            config["model_id"] = self.protocol.get("judge_model", config["model_id"])
        return self.native.OpenAIServerModel(**config)

    @staticmethod
    def _native_memory_target(memory):
        # OrientedProvider forwards reads only. Keep its public/audited methods
        # intact, but apply native state and method changes to its actual owner.
        from .runtime import OrientedProvider
        return memory._provider if isinstance(memory, OrientedProvider) else memory

    @staticmethod
    def _apply_provider_compatibility(memory, record, phase):
        """Ignore one non-persisted selector flag in the pinned Verity provider."""
        artifact = record.get("runtime_artifact", {})
        memory = NativeHSSRunner._native_memory_target(memory)
        if record.get("provider") == "cairn_memory" and artifact.get("code_sha256") == CAIRN_SOURCE_SHA256:
            if hashlib.sha256(Path(artifact["source_path"]).read_bytes()).hexdigest() != CAIRN_SOURCE_SHA256:
                raise HSSResumeRequired("Cairn compatibility provider source changed")
            if not getattr(memory, "_hss_cairn_jaccard_compatibility", False):
                original = memory._jaccard
                def compatible_jaccard(a, b):
                    return original(set(a) if isinstance(a, list) else a, set(b) if isinstance(b, list) else b)
                memory._jaccard = compatible_jaccard
                memory._hss_cairn_jaccard_compatibility = True
            metadata = record.setdefault("provider_runtime_compatibility", {
                "revision": "repair-r20", "provider_sha256": CAIRN_SOURCE_SHA256,
                "method": "cairn_jaccard_accept_serialized_token_lists",
                "provider_source_unchanged": True, "phases": []})
            if phase not in metadata["phases"]:
                metadata["phases"].append(phase)
            return
        if record.get("provider") != "verity_loom_memory" or artifact.get("code_sha256") != VERITY_LOOM_SOURCE_SHA256:
            return
        if hashlib.sha256(Path(artifact["source_path"]).read_bytes()).hexdigest() != VERITY_LOOM_SOURCE_SHA256:
            raise HSSResumeRequired("Verity compatibility provider source changed")
        if not getattr(memory, "_hss_verity_make_item_compatibility", False):
            original = memory._make_item
            def compatible(*args, **kwargs):
                kwargs.pop("applies_now", None)
                return original(*args, **kwargs)
            memory._make_item = compatible
            memory._hss_verity_make_item_compatibility = True
        metadata = record.setdefault("provider_runtime_compatibility", {
            "revision": "repair-r19", "provider_sha256": VERITY_LOOM_SOURCE_SHA256,
            "method": "verity_make_item_ignore_nonpersisted_applies_now", "ignored_keywords": ["applies_now"],
            "provider_source_unchanged": True, "phases": []})
        if phase not in metadata["phases"]:
            metadata["phases"].append(phase)

    @staticmethod
    def _verified_actor_completion(record):
        """Recognize signed incident evidence without relabeling failed memory calls."""
        evidence = record.get("native_actor_completion_recovery")
        if evidence is None:
            return set()
        try:
            receipt_path = Path(evidence["receipt_path"])
            raw = receipt_path.read_bytes()
            if not receipt_path.is_absolute() or hashlib.sha256(raw).hexdigest() != evidence["receipt_sha256"]:
                raise ValueError("Actor recovery receipt changed")
            receipt = json.loads(raw)
            original_path = Path(receipt["backed_up_record"])
            original_bytes = original_path.read_bytes()
            if (not original_path.is_absolute()
                    or hashlib.sha256(original_bytes).hexdigest() != receipt["source_record_before_sha256"]):
                raise ValueError("Actor recovery source record changed")
            original = json.loads(original_bytes)
            json_hash = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                                               separators=(",", ":")).encode("utf8")).hexdigest()
            expected = {"method": "complete_native_actor_after_standby", "task_identity": record["task_identity"],
                        "provider": record["provider"], "provider_sha256": record["runtime_artifact"]["code_sha256"],
                        "saved_answer_sha256": hashlib.sha256(record["answer"].encode("utf8")).hexdigest(),
                        "saved_messages_sha256": json_hash(record["agent_messages"]),
                        "saved_trajectory_sha256": json_hash(record["agent_trajectory"])}
            if (any(receipt.get(k) != v or evidence.get(k) != v for k, v in expected.items())
                    or receipt["actor_execution_id"] not in record["task_execution_ids"]
                    or original.get("task_identity") != record["task_identity"]
                    or original.get("task_execution_id") != receipt["actor_execution_id"]
                    or original.get("answer") != record["answer"]
                    or original.get("agent_messages", original.get("partial_agent_messages")) != record["agent_messages"]
                    or original.get("agent_trajectory", original.get("partial_agent_trajectory")) != record["agent_trajectory"]
                    or receipt["pinned_files"].get(str(Path(record["runtime_artifact"]["source_path"]).resolve()))
                    != record["runtime_artifact"]["code_sha256"]):
                raise ValueError("Actor recovery work differs")
            events = {}
            for filename, digest_value in receipt["pinned_files"].items():
                path = Path(filename)
                data = path.read_bytes()
                if not path.is_absolute() or hashlib.sha256(data).hexdigest() != digest_value:
                    raise ValueError("Actor recovery paid receipt or provider changed")
                if path.parent.name == "attempts" and path.suffix == ".json":
                    event = json.loads(data)
                    events[event["attempt"]] = event
            final = events[receipt["final_attempt"]]
            if (final["status"] != "success" or final["role"] != "actor"
                    or final["task_execution_id"] != receipt["actor_execution_id"]
                    or any(c.get("finish_reason") != "stop" for c in final["response"]["choices"])):
                raise ValueError("Actor final response is incomplete")
            from .hss_tool_calls import recover_final_answer
            decoded, _ = recover_final_answer(final["response"]["choices"][0]["message"],
                                               {"final_answer", "web_search", "crawl_page"})
            if decoded["tools"][0]["arguments"]["answer"] != record["answer"]:
                raise ValueError("Actor final answer differs")
            for link in receipt["exact_request_recoveries"]:
                failed, success = events[link["failed_attempt"]], events[link["successful_attempt"]]
                if (failed["status"] != "error" or success["status"] != "success"
                        or not failed.get("request_sha256")
                        or any(failed.get(k) != success.get(k) for k in
                               ("request_sha256", "prompt_sha256", "role", "task_execution_id"))):
                    raise ValueError("Paid SDK replacement is not the same request")
            fallback_ids = receipt["native_fallback_failed_attempt_ids"]
            if (not isinstance(fallback_ids, list) or len(set(fallback_ids)) != len(fallback_ids)
                    or any(type(identifier) is not int or identifier <= 0 for identifier in fallback_ids)
                    or fallback_ids != evidence["native_fallback_failed_attempt_ids"]):
                raise ValueError("Native fallback identities differ from their receipt")
            for identifier in fallback_ids:
                failed = events[identifier]
                if (failed.get("status") != "error" or failed.get("role") != "memory_retrieval"
                        or failed.get("response") is not None
                        or failed.get("task_execution_id") != receipt["actor_execution_id"]):
                    raise ValueError("Native fallback does not match a failed retrieval")
            return set(fallback_ids)
        except (KeyError, IndexError, TypeError, ValueError, OSError):
            raise HSSResumeRequired("Complete Actor recovery evidence is missing or changed") from None

    def _task_identity(self, task, index, provider, artifact, stage, dataset_name):
        return digest({"task": task, "index": index, "provider": provider,
                       "artifact": artifact, "stage": stage, "dataset": dataset_name,
                       "protocol": self.protocol, "model": self.model_config.get("model_id")})

    @staticmethod
    def _verified_ingestion_rollback(record, authorization):
        """Verify the helper's completed storage rollback before replaying ingestion."""
        try:
            evidence = record["native_ingestion_recovery"]
            receipt_path = Path(evidence["receipt_path"])
            raw = receipt_path.read_bytes()
            if not receipt_path.is_absolute() or hashlib.sha256(raw).hexdigest() != evidence["receipt_sha256"]:
                raise ValueError("Rollback receipt hash differs")
            receipt = json.loads(raw)
            expected = {"status": "storage_restored_for_ingestion_retry",
                        "task_identity": record["task_identity"],
                        "prior_task_execution_id": record["task_execution_id"],
                        "provider": record["provider"],
                        "failed_attempt_ids": record["api_usage"].get("unrecovered_failed_attempt_ids", [])}
            if (any(receipt.get(key) != value or evidence.get(key) != value for key, value in expected.items())
                    or receipt.get("revision") != authorization["revision"]
                    or receipt.get("next_phase") != "ingestion" or authorization.get("next_phase") != "ingestion"
                    or receipt["failed_attempt_ids"] != authorization.get("known_failed_attempts", [])
                    or receipt.get("method") != evidence.get("method")
                    or receipt.get("method") not in {
                        "restart_ingestion_before_first_dispatched_request",
                        "rollback_unique_credit_tail_then_restart_ingestion",
                        "rollback_completed_ingestion_transaction_then_retry",
                        "restore_verity_loom_volatile_preingestion_then_retry",
                        "rollback_verity_committed_fallback_then_replay_paid_ingestion",
                        "rollback_cairn_full_v2_committed_ingestion_then_retry_credit",
                        "restore_cairn_volatile_preingestion_then_retry"}):
                raise ValueError("Rollback identity or failed request authorization differs")
            context = evidence.get("runtime_context")
            if receipt.get("runtime_context") != context:
                raise ValueError("Rollback runtime context differs from its receipt")
            if context is not None:
                verity = context.get("kind") == "verity_loom_preingestion"
                cairn = context.get("kind") == "cairn_preingestion"
                cairn_v2 = context.get("kind") == "cairn_full_v2_preingestion"
                offline_verity = receipt["method"] == "rollback_verity_committed_fallback_then_replay_paid_ingestion"
                if verity or cairn or cairn_v2:
                    original_path = Path(receipt["backed_up_record"])
                    original_bytes = original_path.read_bytes()
                    pinned_hash = (VERITY_LOOM_SOURCE_SHA256 if verity else
                                   CAIRN_FULL_V2_SOURCE_SHA256 if cairn_v2 else CAIRN_SOURCE_SHA256)
                    expected_provider = "verity_loom_memory" if verity else "cairn_memory"
                    expected_method = ("rollback_verity_committed_fallback_then_replay_paid_ingestion" if offline_verity else
                                       "restore_verity_loom_volatile_preingestion_then_retry" if verity else
                                       "rollback_cairn_full_v2_committed_ingestion_then_retry_credit" if cairn_v2 else
                                       "restore_cairn_volatile_preingestion_then_retry")
                    if (receipt["method"] != expected_method or record["provider"] != expected_provider
                            or receipt.get("provider_sha256") != pinned_hash
                            or record["runtime_artifact"]["code_sha256"] != pinned_hash
                            or (verity and context["query"] != record["enhanced_question"])
                            or not original_path.is_absolute()
                            or hashlib.sha256(original_bytes).hexdigest() != receipt["source_record_before_sha256"]):
                        raise ValueError("Verity restoration does not match its pinned source record")
                    original = json.loads(original_bytes)
                    if (original["task_identity"] != record["task_identity"] or original["answer"] != record["answer"]
                            or original["grade"] != record["grade"] or original["status"] != "ingestion_failed"):
                        raise ValueError("Verity paid work differs from its interrupted ingestion")
                    pins = receipt["pinned_files_before"]
                    if (not isinstance(pins, dict) or (verity and len(pins) < 24)
                            or (cairn and len(pins) < 5) or (cairn_v2 and len(pins) < 28)):
                        raise ValueError("Verity restoration requires its original paid evidence")
                    for filename, expected_hash in pins.items():
                        path = Path(filename)
                        if not path.is_absolute() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                            raise ValueError("Verity restoration paid evidence changed")
                    if offline_verity and (not verity
                            or context.get("paid_ingestion_replay_policy") != "exact_sequence_offline_only"
                            or [x["attempt"] for x in context.get("paid_ingestion_replays", [])] != [15359, 15391]):
                        raise ValueError("Offline Verity recovery requires both original paid responses")
                    if context.get("paid_ingestion_replay_policy") == "exact_sequence_offline_only" and not offline_verity:
                        raise ValueError("Only the committed Verity incident permits strict offline replay")
                    if offline_verity or cairn_v2:
                        rollbacks = receipt.get("storage_rollback_evidence", [])
                        if len(rollbacks) != (3 if offline_verity else 1):
                            raise ValueError("Committed ingestion lacks its exact transaction backups")
                        for rollback in rollbacks:
                            backup = Path(rollback["backed_up_file"])
                            if (not backup.is_absolute() or hashlib.sha256(backup.read_bytes()).hexdigest() != rollback["before_sha256"]):
                                raise ValueError("Committed transaction backup changed")
                        if cairn_v2 and (context.get("task_counter") != 2
                                or [x["attempt"] for x in context.get("paid_ingestion_replays", [])] != [18996]
                                or context.get("paid_ingestion_replay_policy") == "exact_sequence_offline_only"):
                            raise ValueError("Cairn v2 requires a real replacement credit response")
                    if verity and (context["unknown_nonbehavioral_fields"] != ["task.start_ts", "store.items[3:9].created_ts"]
                            or context["task"]["start_ts"] is not None or len(context["store"]["items"]) != 9
                            or any(item["created_ts"] is not None for item in context["store"]["items"][3:])
                            or context["task"]["surfaced_ids"] != context["snapshots"][context["task_key"]]["ids"]):
                        raise ValueError("Verity restored meaningful fields or unknown timestamps differ")
                elif (receipt.get("method") != "rollback_completed_ingestion_transaction_then_retry"
                        or record["provider"] != "lightweight_memory"
                        or receipt.get("provider_sha256") != record["runtime_artifact"]["code_sha256"]
                        or context.get("kind") != "lightweight_used_memory_ids"
                        or context.get("query") != record["enhanced_question"]
                        or set(context) != {"kind", "used_memory_ids", "query"}
                        or not isinstance(context.get("used_memory_ids"), list)
                        or not context["used_memory_ids"]
                        or len(set(context["used_memory_ids"])) != len(context["used_memory_ids"])):
                    raise ValueError("Rollback runtime context is not a supported native snapshot")
            answer_hash = hashlib.sha256(record["answer"].encode("utf-8")).hexdigest()
            grade_hash = hashlib.sha256(json.dumps(record["grade"], ensure_ascii=False, sort_keys=True,
                                                  separators=(",", ":")).encode("utf-8")).hexdigest()
            if receipt.get("saved_answer_sha256") != answer_hash or receipt.get("saved_grade_sha256") != grade_hash:
                raise ValueError("Rollback saved work differs")
            files = receipt.get("storage_files_after")
            if not isinstance(files, dict) or not files:
                raise ValueError("Rollback storage evidence is missing")
            for name, expected_hash in files.items():
                path = Path(name)
                if not path.is_absolute() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                    raise ValueError("Storage changed after rollback")
            for name in receipt.get("absent_storage_files_after", []):
                if not Path(name).is_absolute() or Path(name).exists():
                    raise ValueError("A removed transaction file reappeared after rollback")
        except (KeyError, TypeError, ValueError, OSError):
            raise HSSResumeRequired("Ingestion recovery requires verified completed storage rollback evidence") from None
        return True

    @staticmethod
    def _restore_ingestion_context(memory, record):
        """Restore only a hash-pinned RAM snapshot lost by an audited interruption."""
        evidence = record.get("native_ingestion_recovery", {})
        context = evidence.get("runtime_context")
        if context is None:
            return
        # Recovery records get a new execution ID after the original rollback
        # proof is checked. Re-check against that preserved predecessor here.
        prior = deepcopy(record)
        prior["task_execution_id"] = record["recovery_history"][-1]["prior_task_execution_id"]
        NativeHSSRunner._verified_ingestion_rollback(prior, record["recovery_authorized"])
        memory = NativeHSSRunner._native_memory_target(memory)
        try:
            if context["kind"] == "verity_loom_preingestion":
                for target, key in (("_store", "store"), ("_task", "task"), ("_snapshots", "snapshots"),
                                    ("_id_counter", "id_counter"), ("_task_counter", "task_counter"),
                                    ("_ingestions_since_maintenance", "ingestions_since_maintenance")):
                    setattr(memory, target, deepcopy(context[key]))
                record["actor_memory_degradation"] = deepcopy(context["actor_memory_degradation"])
                record["volatile_memory_restoration"] = {
                    "meaningful_fields_restored": True, "bitwise_original_state_claimed": False,
                    "unknown_nonbehavioral_fields": list(context["unknown_nonbehavioral_fields"])}
                if context.get("paid_ingestion_replay_policy") == "exact_sequence_offline_only":
                    NativeHSSRunner._install_paid_ingestion_replay(memory, record, context["paid_ingestion_replays"], strict=True)
                return
            if context["kind"] == "cairn_full_v2_preingestion":
                memory._db = deepcopy(context["db"])
                memory._task_states = deepcopy(context["task_states"])
                memory._task_counter = context["task_counter"]
                NativeHSSRunner._install_paid_ingestion_replay(memory, record, context["paid_ingestion_replays"])
                return
            if context["kind"] == "cairn_preingestion":
                memory.db = deepcopy(context["db"])
                memory._by_id = {item["mem_id"]: item for item in memory.db["memories"]}
                memory.task_context = deepcopy(context["task_context"])
                memory.task_snapshots = deepcopy(context["task_snapshots"])
                memory._task_id_counter = context["task_id_counter"]
                NativeHSSRunner._install_paid_ingestion_replay(memory, record, context.get("paid_ingestion_replays", []))
                return
            ids = context["used_memory_ids"]
            for identity in ids:
                kind, index = identity.rsplit("_", 1)
                if kind not in ("strategic", "operational") or not index.isdecimal():
                    raise ValueError("Invalid lightweight memory identity")
                if int(index) >= len(memory.longterm_db[kind]):
                    raise ValueError("Lightweight memory identity is outside the restored store")
            memory.task_context["used_memory_ids"] = list(ids)
            memory.task_context["query"] = context["query"]
            memory._task_snapshots[memory._compute_signature(context["query"])] = list(ids)
        except (AttributeError, KeyError, TypeError, ValueError):
            raise HSSResumeRequired("Restored native ingestion context does not match its provider") from None

    @staticmethod
    def _install_paid_ingestion_replay(memory, record, references, *, strict=False):
        """Reuse pinned complete responses for exact prompts without new usage."""
        memory = NativeHSSRunner._native_memory_target(memory)
        paid = []
        for reference in references:
            path = Path(reference["receipt_path"])
            raw = path.read_bytes()
            event = json.loads(raw)
            if (not path.is_absolute() or hashlib.sha256(raw).hexdigest() != reference["receipt_sha256"]
                    or event["attempt"] != reference["attempt"] or event["role"] != "memory_ingestion"
                    or event["status"] != "success" or len(event["response"]["choices"]) != 1
                    or event["response"]["choices"][0]["finish_reason"] != "stop"
                    or event["task_execution_id"] not in record["task_execution_ids"]):
                raise HSSResumeRequired("Paid ingestion replay response evidence changed")
            paid.append((reference, event))
        original = memory._call_llm
        state = {"consumed": 0, "expected": len(paid)}
        if strict:
            memory._philo_exact_paid_ingestion_state = state
        def call_llm(prompt, *args, **kwargs):
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            candidates = paid[state["consumed"]:state["consumed"] + 1] if strict else paid
            for reference, event in candidates:
                if event["request"]["messages"] == messages:
                    state["consumed"] += 1
                    record.setdefault("local_provider_replays", []).append({
                        "original_attempt": event["attempt"], "original_receipt_sha256": reference["receipt_sha256"],
                        "original_request_sha256": event["request_sha256"], "new_model_request": False,
                        "new_usage_counted": False, "phase": "memory_ingestion"})
                    return event["response"]["choices"][0]["message"]["content"]
            if strict:
                raise _ExactPaidReplayMismatch("Signed offline ingestion prompt differs from the next original paid request; no model call or native fallback is allowed")
            return original(prompt, *args, **kwargs)
        memory._call_llm = call_llm

    def _saved_task(self, path, identity):
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("task_identity") != identity:
                raise HSSResumeRequired("Saved HSS attempt differs from the requested task or provider")
            self._verified_actor_completion(previous)
            if previous.get("status") == "completed":
                native_quality_summary([previous])
                return previous
            authorization = previous.get("recovery_authorized")
            if not self.protocol.get("resume_incomplete_tasks") or not isinstance(authorization, dict) or not authorization.get("revision"):
                raise HSSResumeRequired("Saved HSS attempt is incomplete; paid work is preserved and will not be replayed")
            phase = classify_recovery_record(previous)
            if phase == "blocked_ingestion":
                raise HSSResumeRequired("Native ingestion may already have changed memory; review or restore its storage before recovery")
            if phase == "blocked_trajectory":
                raise HSSResumeRequired("Saved answer has no native ingestion trajectory; actor work will not be replayed")
            if isinstance(previous.get("grade"), dict):
                validated = validate_grade(previous["grade"], previous.get("domain"))
                if previous["grade"].get("quality_score") != validated["quality_score"]:
                    raise HSSResumeRequired("Saved quality differs from its dimension grades")
            usage = previous.get("api_usage", {})
            if not usage.get("usage_complete") and not self.observe_accounting:
                reserved_ids = usage.get("reserved_unknown_attempt_ids", [])
                if (not usage.get("usage_accounted") or not reserved_ids
                        or set(reserved_ids) != set(authorization.get("reserved_unknown_attempts", []))
                        or type(usage.get("reserved_unknown_tokens")) is not int
                        or usage["reserved_unknown_tokens"] <= 0):
                    raise HSSResumeRequired("Unknown paid usage requires reconciliation before recovery")
            unresolved_count = usage.get("unrecovered_failed_request_count", usage.get("failed_request_count", 0))
            if (phase == "judge" and previous.get("status") == "actor_failed" and unresolved_count
                    and not previous.get("native_actor_completion_recovery")):
                raise HSSResumeRequired("Complete Actor has failed requests requiring explicit audited recovery")
            ingestion_rollback = (phase == "ingestion" and previous.get("native_ingestion_recovery") is not None
                                  and self._verified_ingestion_rollback(previous, authorization))
            if unresolved_count:
                unresolved_ids = usage.get("unrecovered_failed_attempt_ids", [])
                authorized_ids = authorization.get("known_failed_attempts", [])
                signed_fallback = self._verified_actor_completion(previous)
                completed_ingestion_integrity = (phase == "integrity" and previous.get("memory_ingestion_complete") is True
                    and isinstance(previous.get("grade"), dict) and signed_fallback
                    and set(unresolved_ids) <= signed_fallback)
                if ((phase not in ("actor", "judge") and not ingestion_rollback and not completed_ingestion_integrity) or len(unresolved_ids) != unresolved_count
                        or any(attempt is None for attempt in unresolved_ids)
                        or set(unresolved_ids) != set(authorized_ids)):
                    raise HSSResumeRequired("Failed audited requests need explicit successful recovery links")
            if phase == "actor":
                if previous.get("memory_ingestion_complete"):
                    raise HSSResumeRequired("An incomplete actor answer already entered memory; restore storage before actor recovery")
                if not authorization.get("allow_actor_retry"):
                    raise HSSResumeRequired("Actor replay requires an explicit actor recovery authorization")
                limit = self.protocol.get("actor_recovery_max_attempts", 1)
                if previous.get("actor_recovery_attempts", 0) >= limit and not self.data_first:
                    raise HSSResumeRequired("Authorized actor retry limit exhausted")
            if (phase == "judge" and not self.data_first
                    and previous.get("judge_recovery_attempts", 0) >= self._judge_recovery_limit(previous)):
                try:
                    validate_grade(previous.get("judge_output"), previous.get("domain"))
                except ValueError:
                    raise HSSResumeRequired("Authorized judge recovery limit exhausted") from None
            return previous
        return None

    def _judge_recovery_limit(self, record):
        """One extra grade request only after a pinned HTTP402 returned no grade."""
        limit = self.protocol.get("judge_recovery_max_attempts", 1)
        authorization = record.get("recovery_authorized", {})
        grant = authorization.get("additional_judge_recovery_attempt")
        if not isinstance(grant, dict):
            return limit
        try:
            denied_path = Path(grant["denied_receipt_path"])
            raw = denied_path.read_bytes()
            denied = json.loads(raw)
            old_execution = record["task_execution_id"]
            if record.get("status") == "judge_recovery_started":
                old_execution = record["recovery_history"][-1]["prior_task_execution_id"]
            prior_counter = record.get("judge_recovery_attempts", 0)
            if (not denied_path.is_absolute() or hashlib.sha256(raw).hexdigest() != grant["denied_receipt_sha256"]
                    or grant.get("task_identity") != record["task_identity"]
                    or grant.get("prior_task_execution_id") != old_execution
                    or grant.get("prior_counter") != limit or prior_counter != limit
                    or grant.get("allowed_counter") != limit + 1
                    or grant.get("saved_answer_sha256") != hashlib.sha256(record["answer"].encode("utf-8")).hexdigest()
                    or authorization.get("next_phase") != "judge"
                    or record.get("grade") is not None or "judge_output" in record
                    or record.get("memory_ingestion_started") or record.get("memory_ingestion_complete")
                    or denied.get("attempt") != grant["denied_attempt_id"]
                    or denied.get("role") != "judge" or denied.get("status") != "error"
                    or denied.get("http_status") != 402 or denied.get("response") is not None
                    or denied.get("task_execution_id") != old_execution
                    or authorization.get("known_failed_attempts") != [grant["denied_attempt_id"]]
                    or record.get("api_usage", {}).get("unrecovered_failed_attempt_ids") != [grant["denied_attempt_id"]]):
                return limit
            return limit + 1
        except (KeyError, TypeError, ValueError, OSError):
            return limit

    def _resume_record(self, path, previous):
        """Archive before mutating; carry old usage into a new audited execution."""
        record = deepcopy(previous)
        phase = classify_recovery_record(record)
        history = record.setdefault("recovery_history", [])
        archive = path.parent / "_recovery" / path.stem / f"{len(history) + 1:02d}-{record['task_execution_id']}.json"
        if archive.exists():
            raise HSSResumeRequired("Recovery archive already exists; inspect the interrupted recovery before continuing")
        write_json(archive, previous)
        history.append({"archive": str(archive), "prior_status": record.get("status"),
                        "prior_task_execution_id": record["task_execution_id"], "resumed_phase": phase,
                        "authorization": deepcopy(record["recovery_authorized"])})
        record["prior_api_usage"] = deepcopy(record["api_usage"])
        record["prior_elapsed_time"] = record.get("metrics", {}).get("elapsed_time", 0)
        ids = record.setdefault("task_execution_ids", [record["task_execution_id"]])
        record["task_execution_id"] = uuid.uuid4().hex
        ids.append(record["task_execution_id"])
        record.pop("error_type", None)
        if phase == "actor":
            record["actor_recovery_attempts"] = record.get("actor_recovery_attempts", 0) + 1
            for key in ("answer", "agent_result", "agent_trajectory", "agent_messages", "grade", "judge_output",
                        "quality_score", "score", "judgement", "response_truncated", "actor_output_issue"):
                record.pop(key, None)
        record["status"] = f"{phase}_recovery_started"
        return record, phase

    def _actor_truncated(self, execution_id):
        events = [event for event in self.usage_reader()
                  if event.get("task_execution_id") == execution_id and event.get("role") == "actor"]
        recovered = {attempt for event in events
                     if event.get("status") == "success" or _response_usable_without_usage(event)
                     for attempt in event.get("recovered_attempts", [])}
        return any(event.get("attempt_id", event.get("attempt")) not in recovered and (
            event.get("finish_reason") == "length" or "length" in event.get("finish_reasons", [])
            or "length" in event.get("response_finish_reasons", [])
        ) for event in events)

    @staticmethod
    def _grade_output(record, domain):
        normalized, evidence = normalize_grade_output(record["judge_output"], domain)
        record["judge_normalization"] = evidence
        return validate_grade(normalized, domain)

    @staticmethod
    def _resolve_prior_phase_failures(record, phase):
        """Link authorized old errors only after replacement phase succeeds."""
        prior = record.get("prior_api_usage", {})
        attempts = prior.get("unrecovered_failed_attempt_ids", [])
        if not attempts:
            return
        authorization = record["recovery_authorized"]
        fallback_ids = NativeHSSRunner._verified_actor_completion(record)
        if set(attempts) != set(authorization.get("known_failed_attempts", [])):
            raise HSSResumeRequired("Prior phase failures have no matching recovery authorization")
        remaining = [identifier for identifier in attempts if identifier in fallback_ids]
        attempts = [identifier for identifier in attempts if identifier not in fallback_ids]
        if not attempts:
            return
        if phase == "ingestion" and (not record.get("memory_ingestion_complete")
                                     or authorization.get("next_phase") != "ingestion"
                                     or not record.get("native_ingestion_recovery")):
            raise HSSResumeRequired("Ingestion failures may only link after verified rollback and successful ingestion")
        record.setdefault("phase_recovery_receipts", []).append({
            "phase": phase, "successful_task_execution_id": record["task_execution_id"],
            "recovered_attempts": list(attempts), "authorization_revision": authorization["revision"],
            **({"rollback_receipt_sha256": record["native_ingestion_recovery"]["receipt_sha256"]}
               if phase == "ingestion" else {}),
            **({"completion_method": "exact_paid_response_replay",
                "original_successful_attempts": [x["original_attempt"] for x in record.get("local_provider_replays", [])],
                "new_model_request": False, "new_sdk_success": False}
               if phase == "ingestion" and record.get("native_ingestion_recovery", {}).get("runtime_context", {}).get("paid_ingestion_replay_policy") == "exact_sequence_offline_only" else {}),
            **({"reserved_unknown_attempts": list(prior.get("reserved_unknown_attempt_ids", [])),
                "reserved_unknown_tokens": prior["reserved_unknown_tokens"],
                "usage_complete": False} if prior.get("reserved_unknown_tokens") else {}),
        })
        prior["unrecovered_failed_attempt_ids"] = remaining
        prior["unrecovered_failed_request_count"] = len(remaining)

    def _run_task(self, task, index, provider, output_dir, *, artifact, stage, stage_id, dataset_name):
        prompt = task_prompt(task)
        path = output_dir / f"{index + 1}.json"
        identity = self._task_identity(task, index, provider, artifact, stage, dataset_name)
        previous = self._saved_task(path, identity)
        if previous is not None and previous.get("status") == "completed":
            return previous

        started = time.monotonic()
        memory = None
        if previous is not None:
            record, phase = self._resume_record(path, previous)
        else:
            execution_id = uuid.uuid4().hex
            phase = "actor"
            record = {"task_identity": identity, "task_execution_id": execution_id,
                      "task_execution_ids": [execution_id],
                      "task_id": task.get("task_id", str(index)), "global_task_index": index,
                      "item_index": index, "question": task["question"], "enhanced_question": prompt,
                      "domain": task["domain"], "provider": provider, "stage": stage,
                      "runtime_artifact": artifact, "status": "started",
                      "memory_evolution_enabled": stage != "heldout", "memory_ingestion_complete": False,
                      "learning_success_threshold": self.protocol.get("quality_success_threshold", 60),
                      "fitness_semantics": "mean_continuous_quality_divided_by_100"}
        execution_id = record["task_execution_id"]
        resumed_phase = phase
        # Every recovery has its own execution scope and carries the predecessor's
        # settled usage. Saving here also makes interrupted recovery visible.
        self._save(path, record, started)
        with audit_scope(provider=provider, task_execution_id=execution_id, task_id=record["task_id"],
                         task_index=index, stage=stage, stage_id=stage_id,
                         code_sha256=artifact.get("code_sha256"),
                         orientation_items_sha256=artifact.get("orientation_items_sha256")):
            try:
                if phase == "actor":
                    with audit_scope(role="actor"):
                        model = self._new_model()
                        memory = self.runner.load_memory_provider(provider, model)
                        if memory is None:
                            raise RuntimeError("HSS provider initialization returned no provider")
                        self._apply_provider_compatibility(memory, record, "actor")
                        agent = self.base_agent_class(model)
                        agent.agent_fn = self.native.ToolCallingAgent(
                            model=model,
                            tools=[self.native.WebSearchTool(), self.native.CrawlPageTool(model=model)],
                            summary_interval=self.protocol.get("summary_interval", 8),
                            prompts_type=self.protocol.get("prompts_type", "default"),
                            max_steps=self.protocol.get("actor_max_steps", 32), memory_provider=memory,
                        )
                        def save_decoding(evidence):
                            record.setdefault("actor_decoding_events", []).append(deepcopy(evidence))
                            self._save(path, record, started)

                        agent.agent_fn._hss_decoding_observer = save_decoding
                        # One native actor invocation. Full actor replay is only
                        # allowed by the separately recorded repair authorization.
                        answer = agent.agent_fn.run(prompt)
                        record["agent_result"] = answer
                        record["answer"] = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
                        if not record["answer"].strip() or answer is None:
                            record["actor_output_issue"] = "empty_final_answer"
                            raise RuntimeError("HSS actor returned an empty final work")
                        # The native Actor may finish after a handled retrieval
                        # failure. Preserve its whole paid work before deciding
                        # whether request integrity needs operator recovery.
                        record["status"] = "answer_saved"
                        self._save(path, record, started)
                        phase = "trajectory"
                        record["agent_messages"] = agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
                        self._save(path, record, started)
                        record.update(agent.capture_trajectory())
                        self._save(path, record, started)
                        phase = "actor"
                        if self._actor_truncated(execution_id):
                            record.update(response_truncated=True, actor_output_issue="audited_finish_reason_length")
                            raise RuntimeError("HSS actor final work follows an unrecovered truncated response")
                        if (not self.data_first
                                and self._usage(execution_id)["unrecovered_failed_request_count"]):
                            raise RuntimeError("HSS actor has failed audited SDK usage")
                        self._resolve_prior_phase_failures(record, "actor")
                        self._save(path, record, started)
                    phase = "judge"

                if phase in ("judge", "ingestion"):
                    phase = "judge"
                    if not isinstance(record.get("grade"), dict):
                        # Formatting-only normalization is provided by validate_grade.
                        # A usable paid grade is always preferred to another request.
                        if "judge_output" in record:
                            try:
                                record["grade"] = self._grade_output(record, task["domain"])
                            except ValueError:
                                pass
                        if "grade" not in record:
                            if previous is not None:
                                attempts = record.get("judge_recovery_attempts", 0)
                                if attempts >= self._judge_recovery_limit(record) and not self.data_first:
                                    raise HSSResumeRequired("Authorized judge recovery limit exhausted")
                                record["judge_recovery_attempts"] = attempts + 1
                            with audit_scope(role="judge"):
                                judge = self._new_model(judge=True)
                                response = judge(judge_messages(task, record["answer"]),
                                                 response_format={"type": "json_object"})
                                # Preserve every malformed paid judge output in the
                                # task archive before a later repair can replace it.
                                record["judge_output"] = response.content
                                self._save(path, record, started)
                                record["grade"] = self._grade_output(record, task["domain"])
                    grade = validate_grade(record["grade"], task["domain"])
                    if record["grade"].get("quality_score") != grade["quality_score"]:
                        raise ValueError("Saved quality differs from its dimension grades")
                    if resumed_phase != "ingestion":
                        self._resolve_prior_phase_failures(record, "judge")
                    quality = grade["quality_score"]
                    success_threshold = record["learning_success_threshold"]
                    record.update(status="graded", quality_score=quality, score=quality / 100,
                                  judgement="correct" if quality >= success_threshold else "incorrect")
                    self._save(path, record, started)

                    phase = "ingestion"
                    if (not self.data_first
                            and self._usage(execution_id)["unrecovered_failed_request_count"]):
                        raise RuntimeError("HSS task has failed audited SDK usage before memory ingestion")
                    if stage != "heldout" and not record.get("memory_ingestion_complete"):
                        with audit_scope(role="memory_ingestion"):
                            if memory is None:
                                context = record.get("native_ingestion_recovery", {}).get("runtime_context", {})
                                if context.get("paid_ingestion_replay_policy") == "exact_sequence_offline_only":
                                    # _saved_task already verified this signed incident.
                                    # Creating a normal client would reject exhausted
                                    # credentials before the two local replays run.
                                    def offline_model(*args, **kwargs):
                                        raise _ExactPaidReplayMismatch("Signed offline ingestion attempted an unexpected model request")
                                    model = offline_model
                                else:
                                    model = self._new_model()
                                memory = self.runner.load_memory_provider(provider, model)
                            if memory is None:
                                raise RuntimeError("HSS provider initialization returned no provider")
                            self._apply_provider_compatibility(memory, record, "ingestion")
                            self._restore_ingestion_context(memory, record)
                            trajectory_data = self.runner.TrajectoryData(
                                query=task["question"], trajectory=record["agent_messages"],
                                result=record["agent_result"], metadata={
                                    "item_index": index, "status": "success", "is_correct": quality >= success_threshold,
                                    "quality_score": quality, "quality_grade": record["grade"],
                                    "judge_comment": record["grade"]["overall_comment"],
                                    "domain": task["domain"], "full_query": prompt,
                                    "learning_success_threshold": success_threshold,
                                    "fitness_semantics": "continuous_quality_not_binary_correctness",
                                })
                            record["memory_ingestion_started"] = True
                            self._save(path, record, started)
                            success, _message = memory.take_in_memory(trajectory_data)
                            replay_state = getattr(self._native_memory_target(memory), "_philo_exact_paid_ingestion_state", None)
                            if replay_state is not None and replay_state["consumed"] != replay_state["expected"]:
                                raise _ExactPaidReplayMismatch("Offline ingestion did not consume every original paid response")
                            if not success:
                                raise RuntimeError("Native HSS memory ingestion returned false")
                            if (not self.data_first
                                    and self._usage(execution_id)["unrecovered_failed_request_count"]):
                                raise RuntimeError("Native HSS memory ingestion has failed audited SDK usage")
                            record["memory_ingestion_complete"] = True
                            self._resolve_prior_phase_failures(record, "ingestion")
                            self._save(path, record, started)

                phase = "integrity"
                if self._artifact(provider) != artifact:
                    raise RuntimeError("Provider implementation or philosophy changed within an HSS task")
                self._save(path, record, started)
                usage = record["api_usage"]
                if (not usage["request_count"]
                        or (not self.data_first and ((not self.observe_accounting and not usage["usage_accounted"])
                                                     or usage["blocking_failed_request_count"]))):
                    raise RuntimeError("HSS task has incomplete or failed audited SDK usage")
                record["status"] = "completed"
                self._save(path, record, started)
                return record
            except (Exception, _ExactPaidReplayMismatch) as exc:
                if phase == "actor" and "agent" in locals():
                    # Preserve paid native messages even when a later step fails.
                    # This is evidence, not authority to replay memory mutation.
                    try:
                        record["partial_agent_messages"] = agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
                        record["partial_agent_trajectory"] = agent.capture_trajectory().get("agent_trajectory", [])
                    except Exception:
                        record["partial_trajectory_capture_failed"] = True
                if phase == "actor" and self._actor_truncated(execution_id):
                    record.update(response_truncated=True, actor_output_issue="audited_finish_reason_length")
                record.update(status=f"{phase}_failed", error_type=type(exc).__name__)
                self._save(path, record, started)
                if isinstance(exc, _ExactPaidReplayMismatch):
                    raise HSSResumeRequired(str(exc)) from None
                raise

    def __call__(self, dataset_name, provider_name, task_indices, dataset_file, output_dir):
        if dataset_name not in ("writingbench", "hss", "hss_business"):
            raise ValueError("NativeHSSRunner supports only the locked WritingBench HSS tasks")
        indices = list(task_indices)
        if not indices or len(set(indices)) != len(indices) or any(type(index) is not int for index in indices):
            raise ValueError("HSS task indices must be distinct nonempty integers")
        rows = [json.loads(line) for line in Path(dataset_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(index < 0 or index >= len(rows) for index in indices):
            raise ValueError("HSS task index outside the locked dataset")
        # Validate the entire requested batch before making the first paid call.
        for index in indices:
            task_prompt(rows[index])
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stage = "heldout" if self.heldout or "terminal" in output_dir.parts else "learning"
        if stage != "heldout" and any(rows[index].get("split") in ("evaluation", "heldout", "holdout", "test") for index in indices):
            raise ValueError("Evaluation data cannot enter a learning callback")
        stage_id = digest(str(output_dir))[:16]
        with self._lock(provider_name):
            artifact = self._artifact(provider_name)
            expected = {f"{index + 1}.json" for index in indices}
            if any(path.name not in expected for path in output_dir.glob("*.json")):
                raise HSSResumeRequired("Unexpected saved task records require review; no paid work was deleted")
            # Check every saved record before starting any new task in a resumed
            # batch. An unresolved old failure must not trigger fresh spending.
            for index in indices:
                self._saved_task(output_dir / f"{index + 1}.json",
                                 self._task_identity(rows[index], index, provider_name, artifact, stage, dataset_name))
            event = {"output_dir": str(output_dir), "provider": provider_name, "stage": stage,
                     "task_indices": indices, "runtime_artifact": artifact, "status": "started",
                     "memory_policy": "persistent_native_storage_no_reset"}
            event_path = self.run_dir / "evidence" / "stages" / f"{stage_id}.json"
            write_json(event_path, event)
            records = []
            try:
                for index in indices:
                    records.append(self._run_task(rows[index], index, provider_name, output_dir,
                                                 artifact=artifact, stage=stage, stage_id=stage_id,
                                                 dataset_name=dataset_name))
                summary = native_quality_summary(records)
                if not summary["success"]:
                    raise RuntimeError("Incomplete HSS batch is ineligible for selection")
                # A subdirectory keeps native trajectory viewers from treating
                # the aggregate as another task JSON.
                write_json(output_dir / "_quality" / "summary.json", summary)
                event.update(status="completed", summary=summary, task_count=len(records))
                self.stage_events.append(event)
                return output_dir
            except Exception as exc:
                event.update(status="failed", error_type=type(exc).__name__, completed_tasks=len(records))
                raise
            finally:
                write_json(event_path, event)
