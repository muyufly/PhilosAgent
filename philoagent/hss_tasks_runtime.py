"""Native open-writing task callback for the v0.0.6 experiments.

The controller owns provider refresh, model-client auditing and evolution. This
callback preserves each provider's ordinary storage and writes paid answers
before running the independent quality judge or memory ingestion.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
from copy import deepcopy

from .ablation_runtime import _provider_artifact, audit_scope, digest, write_json
from .hss_evaluation import judge_messages, native_quality_summary, normalize_grade_output, rubric, validate_grade


class HSSResumeRequired(RuntimeError):
    """An incomplete paid attempt needs an explicit recovery decision."""


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
        recovered = {attempt for event in events if event.get("status") == "success"
                     for attempt in event.get("recovered_attempts", [])}
        failures = [event for event in events if event.get("status") != "success"]
        unresolved = [event for event in failures if event.get("attempt_id", event.get("attempt")) not in recovered]
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
        if self.data_first:
            usage["accounting_policy"] = "observe_only"
            record["data_collection_mode"] = "data_first"
        usage["effective_tokens"] = usage["total_tokens"] + usage["reserved_unknown_tokens"]
        usage["unrecovered_failed_request_count"] = (
            prior.get("unrecovered_failed_request_count", prior.get("failed_request_count", 0))
            + current["unrecovered_failed_request_count"])
        usage["unrecovered_failed_attempt_ids"] = (
            prior.get("unrecovered_failed_attempt_ids", []) + current["unrecovered_failed_attempt_ids"])
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

    def _task_identity(self, task, index, provider, artifact, stage, dataset_name):
        return digest({"task": task, "index": index, "provider": provider,
                       "artifact": artifact, "stage": stage, "dataset": dataset_name,
                       "protocol": self.protocol, "model": self.model_config.get("model_id")})

    def _saved_task(self, path, identity):
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("task_identity") != identity:
                raise HSSResumeRequired("Saved HSS attempt differs from the requested task or provider")
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
            if not usage.get("usage_complete") and not self.data_first:
                reserved_ids = usage.get("reserved_unknown_attempt_ids", [])
                if (not usage.get("usage_accounted") or not reserved_ids
                        or set(reserved_ids) != set(authorization.get("reserved_unknown_attempts", []))
                        or type(usage.get("reserved_unknown_tokens")) is not int
                        or usage["reserved_unknown_tokens"] <= 0):
                    raise HSSResumeRequired("Unknown paid usage requires reconciliation before recovery")
            unresolved_count = usage.get("unrecovered_failed_request_count", usage.get("failed_request_count", 0))
            if unresolved_count:
                unresolved_ids = usage.get("unrecovered_failed_attempt_ids", [])
                authorized_ids = authorization.get("known_failed_attempts", [])
                if (phase not in ("actor", "judge") or len(unresolved_ids) != unresolved_count
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
                    and previous.get("judge_recovery_attempts", 0) >= self.protocol.get("judge_recovery_max_attempts", 1)):
                try:
                    validate_grade(previous.get("judge_output"), previous.get("domain"))
                except ValueError:
                    raise HSSResumeRequired("Authorized judge recovery limit exhausted") from None
            return previous
        return None

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
        recovered = {attempt for event in events if event.get("status") == "success"
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
        if set(attempts) != set(authorization.get("known_failed_attempts", [])):
            raise HSSResumeRequired("Prior phase failures have no matching recovery authorization")
        record.setdefault("phase_recovery_receipts", []).append({
            "phase": phase, "successful_task_execution_id": record["task_execution_id"],
            "recovered_attempts": list(attempts), "authorization_revision": authorization["revision"],
            **({"reserved_unknown_attempts": list(prior.get("reserved_unknown_attempt_ids", [])),
                "reserved_unknown_tokens": prior["reserved_unknown_tokens"],
                "usage_complete": False} if prior.get("reserved_unknown_tokens") else {}),
        })
        prior["unrecovered_failed_attempt_ids"] = []
        prior["unrecovered_failed_request_count"] = 0

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
                        if self._actor_truncated(execution_id):
                            record.update(response_truncated=True, actor_output_issue="audited_finish_reason_length")
                            raise RuntimeError("HSS actor final work follows an unrecovered truncated response")
                        record["status"] = "answer_saved"
                        self._save(path, record, started)
                        phase = "trajectory"
                        # Save native messages independently of the display adapter.
                        # They are sufficient for exact native ingestion recovery.
                        record["agent_messages"] = agent.agent_fn.write_memory_to_messages(include_system_prompt=False)
                        self._save(path, record, started)
                        record.update(agent.capture_trajectory())
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
                                if attempts >= self.protocol.get("judge_recovery_max_attempts", 1) and not self.data_first:
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
                    self._resolve_prior_phase_failures(record, "judge")
                    quality = grade["quality_score"]
                    success_threshold = record["learning_success_threshold"]
                    record.update(status="graded", quality_score=quality, score=quality / 100,
                                  judgement="correct" if quality >= success_threshold else "incorrect")
                    self._save(path, record, started)

                    phase = "ingestion"
                    if stage != "heldout" and not record.get("memory_ingestion_complete"):
                        with audit_scope(role="memory_ingestion"):
                            if memory is None:
                                memory = self.runner.load_memory_provider(provider, self._new_model())
                            if memory is None:
                                raise RuntimeError("HSS provider initialization returned no provider")
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
                            if not success:
                                raise RuntimeError("Native HSS memory ingestion returned false")
                            record["memory_ingestion_complete"] = True
                            self._save(path, record, started)

                phase = "integrity"
                if self._artifact(provider) != artifact:
                    raise RuntimeError("Provider implementation or philosophy changed within an HSS task")
                self._save(path, record, started)
                usage = record["api_usage"]
                if (not usage["request_count"]
                        or (not self.data_first and (not usage["usage_accounted"]
                                                     or usage["unrecovered_failed_request_count"]))):
                    raise RuntimeError("HSS task has incomplete or failed audited SDK usage")
                record["status"] = "completed"
                self._save(path, record, started)
                return record
            except Exception as exc:
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
