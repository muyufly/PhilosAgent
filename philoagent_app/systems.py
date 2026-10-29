"""Lossless initial-system imports, compatible with native orientation seeds."""

import json
from pathlib import Path

from .storage import digest

MAX_SYSTEM_BYTES = 1024 * 1024


def parse_system(filename, content):
    filename = Path(str(filename).replace("\\", "/")).name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".json", ".md", ".txt"}:
        raise ValueError("Initial philosophy must be a JSON, Markdown or TXT file")
    raw = content if isinstance(content, bytes) else content.encode("utf-8")
    if len(raw) > MAX_SYSTEM_BYTES:
        raise ValueError("Initial philosophy exceeds the 1 MiB limit")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError:
        raise ValueError("Initial philosophy must be UTF-8 text") from None
    if not text.strip():
        raise ValueError("Initial philosophy must not be empty")
    if suffix == ".json":
        try:
            document = json.loads(text)
        except ValueError:
            raise ValueError("Initial philosophy is not valid JSON") from None
        items = document.get("items") if isinstance(document, dict) else None
        if (not isinstance(document, dict) or type(document.get("schema_version")) is not int
                or document["schema_version"] != 1 or not isinstance(items, list) or not items
                or any(not isinstance(item, str) or not item.strip() for item in items)):
            raise ValueError("JSON philosophy requires schema_version=1 and nonempty string items")
    else:
        items = [text]
    return {"filename": filename, "sha256": digest(raw), "items": list(items),
            "format": suffix[1:], "size_bytes": len(raw)}, raw


def normalize_task(task):
    if not isinstance(task, dict):
        raise ValueError("A task must be a JSON object")
    from philoagent.hss_tasks_runtime import task_prompt
    task = dict(task)
    task_prompt(task)
    if "materials" in task and not isinstance(task["materials"], (list, str, dict)):
        raise ValueError("Task materials must be text, a list or an object")
    if "deliverable" in task and not isinstance(task["deliverable"], str):
        raise ValueError("Task deliverable must be text")
    if "task_criteria" in task and (not isinstance(task["task_criteria"], list)
                                    or any(not isinstance(x, str) for x in task["task_criteria"])):
        raise ValueError("task_criteria must be a list of strings")
    identifier = task.get("task_id")
    if identifier is not None and (not isinstance(identifier, str) or not identifier.strip()):
        raise ValueError("task_id must be a nonempty string")
    # Never put arbitrary request metadata (including credentials) into tasks.
    task = {key: value for key, value in task.items()
            if key in {"task_id", "domain", "question", "materials", "deliverable", "task_criteria", "split"}}
    if identifier is None:
        task["task_id"] = "task-" + digest(task)[:16]
    json.dumps(task, ensure_ascii=False, allow_nan=False)
    return task


def read_tasks(tasks):
    if isinstance(tasks, (str, Path)):
        text = Path(tasks).read_text(encoding="utf-8-sig")
        try:
            tasks = [json.loads(line) for line in text.splitlines() if line.strip()]
        except ValueError:
            raise ValueError("Tasks must be valid UTF-8 JSONL") from None
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("A nonempty task list or JSONL file is required")
    result = [normalize_task(task) for task in tasks]
    identifiers = [task["task_id"] for task in result]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Task identifiers must be unique")
    return result
