"""Deadline fast lane: reuse paid work, then answer and grade every remaining task.

This is an explicitly shortened experiment. It never edits native records and
does not claim to complete the native generated-program tournament.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

from .cli import environment
from .hss_budget import HSSBudget, process_identity
from .hss_deadline_scoring import score_work


MODE = "deadline_rapid_v1"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _tasks(root, spec, phase):
    relative = spec["learning_file"] if phase == "learning" else spec.get("heldout_file", spec["evaluation_file"])
    records = [json.loads(line) for line in (root / relative).read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    indexed = {record["task_id"]: record for record in records}
    order = spec["learning_order"] if phase == "learning" else spec.get("evaluation_order", spec.get("heldout_order"))
    return [indexed[task_id] for task_id in order]


def _initial_state(root, spec):
    original = root / "jobs" / spec["job_id"]
    checkpoint = _read(original / "evolution" / "evolve_state.json", {})
    winner = checkpoint.get("best_provider", "lightweight_memory")
    seed = _read(root / "inputs" / "phi0.json", {}).get("items", [])
    if spec["group"] == "native":
        phi = []
    elif spec["group"] == "fixed":
        phi = seed
    else:
        phi = _read(original / "lineage" / winner / "orientation.json", {}).get("items", seed)
    return {"memory": [], "retrieval_policy": "优先使用与当前题目相关的经验；依题目要求组织作品。",
            "phi": phi, "starting_native_provider": winner,
            "native_checkpoint": checkpoint, "mode": MODE}


def _existing_work(root, spec, winner):
    """Select one original completed work per task within this exact job only."""
    original = root / "jobs" / spec["job_id"]
    chosen = {}
    paths = []
    for round_dir in (original / "evolution").glob("round_[0-9][0-9]"):
        for pattern in ("base_logs/[0-9]*.json", "eval_batch/*/[0-9]*.json",
                        "eval_finalists/*/[0-9]*.json"):
            paths.extend(round_dir.glob(pattern))
    paths.extend((original / "terminal" / "task_logs").glob("*/[0-9]*.json"))
    for path in paths:
        # Native task logs are integer-named JSON files. Do not load generated
        # programs, checkpoints, receipts or archived incomplete attempts.
        if not path.stem.isdigit() or any(part in {"recovery_archive", "archives", "archive"} for part in path.parts):
            continue
        value = _read(path)
        if not isinstance(value, dict) or value.get("status") not in {"completed", "complete"}:
            continue
        task_id = value.get("task_id")
        answer = value.get("answer", value.get("agent_result"))
        grade = value.get("grade", {})
        score = value.get("quality_score", grade.get("quality_score") if isinstance(grade, dict) else None)
        if not task_id or not isinstance(answer, str) or not answer.strip() or not _number(score):
            continue
        if not 0 <= score <= 100:
            continue
        relative = path.relative_to(original).as_posix()
        priority = (3 if "base_logs/" in relative else 2 if value.get("provider") == winner else 1,
                    path.stat().st_mtime)
        if task_id not in chosen or priority > chosen[task_id][0]:
            chosen[task_id] = (priority, path, value)
    return chosen


def _content(response):
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    return "" if content is None else str(content)


class _Caller:
    def __init__(self, root, spec, deadline_unix):
        from openai import OpenAI
        values = environment()
        self.client = OpenAI(api_key=values["OPENAI_API_KEY"],
                             base_url=values.get("OPENAI_BASE_URL"), max_retries=0)
        self.model = values.get("DEFAULT_MODEL", "deepseek-flash")
        self.deadline = deadline_unix
        self.spec = spec
        self.task_id = None
        self.budget = HSSBudget(root / "budget", max_inflight=16)
        self.receipts = []

    def __call__(self, messages, role="actor", max_tokens=4096, timeout=90):
        remaining = self.deadline - time.time()
        if remaining <= 5:
            raise TimeoutError("Deadline reached before model dispatch")
        timeout = max(1, min(float(timeout), remaining - 3))
        request = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": 0, "stream": False}
        with self.budget.serial_call():
            if time.time() >= self.deadline - 3:
                raise TimeoutError("Deadline reached while waiting for request slot")
            attempt = self.budget.begin(request, job_id=self.spec["job_id"],
                                        domain=self.spec["domain"], group=self.spec["group"],
                                        task_id=self.task_id, role="deadline_" + role, mode=MODE)
            try:
                response = self.client.chat.completions.create(**request, timeout=timeout)
            except Exception as exc:
                receipt = self.budget.finish(attempt, status="error", error_type=type(exc).__name__,
                                              http_status=getattr(exc, "status_code", None), mode=MODE)
                self.receipts.append(receipt)
                raise
            usage = response.usage.model_dump() if response.usage is not None else None
            receipt = self.budget.finish(attempt, usage=usage, response=response.model_dump(), mode=MODE)
            self.receipts.append(receipt)
        return _content(response)


def _messages(task, state):
    system = ("你是承担连续学习写作任务的助手。直接完成用户题目并提交可使用的完整作品，"
              "严格优先满足题目要求的格式和长度。若题目未规定长度，约1200至2000个中文字。"
              "基于提供材料完成，不编造有出处的引文和具体数据；允许明确合理假设。"
              "本次使用已有知识直接作答，无需报告工作流程。")
    context = {"最近相关经验": state["memory"][-3:], "经验使用方法": state["retrieval_policy"]}
    if state["phi"]:
        context["指导判断的哲学状态"] = state["phi"]
    return [{"role": "system", "content": system},
            {"role": "system", "content": json.dumps(context, ensure_ascii=False)},
            {"role": "user", "content": str(task.get("question", ""))}]


def _reflect(state, records, spec, caller, deadline_unix):
    before = json.loads(json.dumps(state, ensure_ascii=False))
    payload = {"经验": state["memory"][-4:], "当前经验使用方法": state["retrieval_policy"],
               "哲学状态": state["phi"], "group": spec["group"]}
    instruction = ("根据刚完成任务的经验，形成可迁移到后续题目的简短反思。仅输出JSON："
                   '{"memory_summary":"200字内","retrieval_policy":"150字内"')
    if spec["group"] == "full":
        instruction += ',"phi":["可保留或修订的哲学原则；总计800字以内"]'
    instruction += "}。不要只追求高分；指出经验适用范围。"
    raw, error = None, None
    try:
        if time.time() > deadline_unix - 30:
            raise TimeoutError("Deadline near; retain learned experience without model reflection")
        raw = caller([{"role": "system", "content": instruction},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                     role="reflection", max_tokens=1536, timeout=45)
        decoder = json.JSONDecoder()
        parsed = None
        for index, char in enumerate(raw):
            if char == "{":
                try:
                    parsed, _ = decoder.raw_decode(raw[index:])
                    if isinstance(parsed, dict):
                        break
                except ValueError:
                    continue
        if not isinstance(parsed, dict):
            raise ValueError("Reflection JSON missing")
        policy = parsed.get("retrieval_policy")
        if isinstance(policy, str) and policy.strip():
            state["retrieval_policy"] = policy[:1600]
        summary = parsed.get("memory_summary")
        if isinstance(summary, str) and summary.strip():
            state["memory"].append({"type": "round_reflection", "summary": summary[:2000]})
        phi = parsed.get("phi")
        if spec["group"] == "full" and isinstance(phi, list) and all(isinstance(item, str) for item in phi):
            state["phi"] = [item[:1600] for item in phi[:8] if item.strip()]
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)[:300]
    return {"mode": MODE, "status": "completed", "raw_reflection": raw, "error": error,
            "before": before, "after": state, "learning_task_ids": [record["task_id"] for record in records],
            "completed_utc": _now(), "native_tournament_replaced": True}


def run_job(root, job_id, deadline_unix):
    root = Path(root).resolve()
    if root.name.startswith("deadline_"):
        root = root.parent
    manifest = _read(root / "manifest.json", {})
    spec = next(job for job in manifest["jobs"] if job["job_id"] == job_id)
    out = root / "deadline_20261030" / "jobs" / job_id
    prior_result = _read(out / "result.json", {})
    started = prior_result.get("started_utc", _now())
    state = _initial_state(root, spec)
    existing = _existing_work(root, spec, state["starting_native_provider"])
    caller = None
    initialization_error = None
    try:
        caller = _Caller(root, spec, deadline_unix)
    except Exception as exc:
        initialization_error = type(exc).__name__ + ": " + str(exc)[:300]
    base = {"job_id": job_id, "domain": spec["domain"], "group": spec["group"],
            "repetition": spec["repetition"], "mode": MODE, "protocol_changed": True}
    records = []
    _write(out / "result.json", {**base, "status": "running", "started_utc": started,
                                 "pid": os.getpid(), "process_identity": process_identity(os.getpid()),
                                 "deadline_unix": deadline_unix})
    for phase in ("learning", "heldout"):
        frozen = json.loads(json.dumps(state, ensure_ascii=False))
        tasks = _tasks(root, spec, phase)
        for index, task in enumerate(tasks):
            task_id = task["task_id"]
            path = out / "tasks" / (task_id + ".json")
            record = _read(path)
            if not isinstance(record, dict) or record.get("status") not in {"completed", "failed_execution"}:
                record = {**base, "task_id": task_id, "stage": phase, "phase": phase,
                          "question": task.get("question"), "title": task.get("title"),
                          "started_utc": _now(), "reused": False}
                if task_id in existing:
                    _, source, value = existing[task_id]
                    grade = value.get("grade", {})
                    score = float(value.get("quality_score", grade.get("quality_score")))
                    record.update(answer=value.get("answer", value.get("agent_result")),
                                  score=score, quality_score=score, grade=grade,
                                  score_method="original_native_judge", status="completed", reused=True,
                                  source=str(source), original_provider=value.get("provider"),
                                  original_task_execution_id=value.get("task_execution_id"), error=None)
                else:
                    answer, errors = "", []
                    if caller:
                        caller.task_id = task_id
                        for attempt in range(2):
                            try:
                                answer = caller(_messages(task, state if phase == "learning" else frozen),
                                                role="actor", max_tokens=4096 if attempt == 0 else 8192,
                                                timeout=90)
                                if answer.strip():
                                    break
                                errors.append("EmptyActorResponse")
                            except Exception as exc:
                                errors.append(type(exc).__name__ + ": " + str(exc)[:300])
                            if time.time() > deadline_unix - 10:
                                break
                    else:
                        errors.append(initialization_error or "Model caller unavailable")
                    grade = score_work(task, answer, caller, deadline_unix)
                    failed = bool(grade.get("evaluator_flags", {}).get("execution_failure"))
                    record.update(answer=answer, score=grade["quality_score"],
                                  quality_score=grade["quality_score"], grade=grade,
                                  score_method=grade["score_method"],
                                  status="failed_execution" if failed else "completed",
                                  source="deadline_direct_model", error=errors or None,
                                  state_used=state if phase == "learning" else frozen)
                record["completed_utc"] = _now()
                _write(path, record)
            records.append(record)
            if phase == "learning":
                state["memory"].append({"task_id": task_id, "quality_score": record["quality_score"],
                                        "comment": str(record.get("grade", {}).get("overall_comment", ""))[:900],
                                        "answer_summary": str(record.get("answer", ""))[:1400],
                                        "execution_failed": record["status"] == "failed_execution"})
                if (index + 1) % 4 == 0:
                    round_id = (index + 1) // 4 - 1
                    reflection_path = out / "rounds" / f"round_{round_id:02d}.json"
                    prior_reflection = _read(reflection_path)
                    if isinstance(prior_reflection, dict) and isinstance(prior_reflection.get("after"), dict):
                        state = prior_reflection["after"]
                    else:
                        reflection = _reflect(state, records[-4:], spec, caller, deadline_unix)
                        _write(reflection_path, reflection)
                _write(out / "state.json", state)
            print(json.dumps({"job_id": job_id, "task_id": task_id, "phase": phase,
                              "status": record["status"], "score": record["score"],
                              "reused": record.get("reused"), "finished": len(records)}, ensure_ascii=False), flush=True)
    failures = [record["task_id"] for record in records if record["status"] == "failed_execution"]
    heldout = [record["score"] for record in records if record["phase"] == "heldout"]
    result = {**base, "status": "completed_with_failures" if failures else "complete",
              "started_utc": started, "completed_utc": _now(), "tasks_terminal": len(records),
              "expected_tasks": 12, "failed_task_ids": failures, "failed_task_count": len(failures),
              "reused_task_count": sum(bool(record.get("reused")) for record in records),
              "quality_mean": sum(record["score"] for record in records) / len(records),
              "heldout_mean": sum(heldout) / len(heldout), "final_state": state,
              "initialization_error": initialization_error,
              "native_tournament_replaced": True, "deadline_unix": deadline_unix,
              "finished_before_deadline": time.time() <= deadline_unix,
              "pid": os.getpid(), "process_identity": process_identity(os.getpid())}
    _write(out / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("job_id")
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args(argv)
    result = run_job(args.root, args.job_id, args.deadline_unix)
    print(json.dumps({"job_id": args.job_id, "status": result["status"],
                      "tasks_terminal": result["tasks_terminal"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
