"""Repair the already paid PraxisLoom candidate; dry-run is the default.

Only two AST changes are permitted: bind the loaded DB before cold-start
construction, and let _new_item read a cold-start item's content field. No
model call, regenerated candidate, task completion or memory ingestion occurs.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import typing
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from philoagent.hss_experiment import process_owned

JOB = "business-fixed-r3"
CANDIDATE = "praxis_loom_memory"
SIBLING = "verity_forge_memory"
ROUND = Path("evolution/round_00")
PROVIDER = Path("workspace/EvolveLab/providers/praxis_loom_memory_provider.py")
GENERATED = ROUND / "generated_system_1.json"
TEMP_PROVIDER = ROUND / ".temp_validation/praxis_loom_memory_isolated_env/EvolveLab/providers/praxis_loom_memory_provider.py"
PAID_ATTEMPT = 12991
PAID_SHA256 = "34915cd3cf9ee63add8493a461b57fd32a95e4a883a0ebd7d324a786244389e1"
ORIGINAL_CODE_SHA256 = "6c19f3e3418441b7f0d5e35e6fe1313311dd48546ec6ddb11589eddddb2fdece"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def normalized(text):
    return text.replace("\r\n", "\n")


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def ast_key(node):
    return ast.dump(node, include_attributes=False)


def patch_provider(source):
    """Verify exactly two changes against an independently transformed AST."""
    tree = ast.parse(source)
    expected = deepcopy(tree)
    provider = next(node for node in expected.body if isinstance(node, ast.ClassDef)
                    and node.name == "PraxisLoomMemoryProvider")
    loader = next(node for node in provider.body if isinstance(node, ast.FunctionDef) and node.name == "_load_db")
    new_item = next(node for node in provider.body if isinstance(node, ast.FunctionDef) and node.name == "_new_item")
    condition = ast.parse('not db.get("items") and not db.get("archive")', mode="eval").body
    matches = [(index, node) for index, node in enumerate(loader.body)
               if isinstance(node, ast.If) and ast_key(node.test) == ast_key(condition)]
    if len(matches) != 1:
        raise ValueError("Cold-start branch differs from the reviewed candidate")
    index, branch = matches[0]
    loader.body.insert(index, ast.parse("self.db = db").body[0])
    original_assignment = ast.parse('text = str(candidate.get("text", "")).strip()').body[0]
    assignments = [(index, node) for index, node in enumerate(new_item.body)
                   if ast_key(node) == ast_key(original_assignment)]
    if len(assignments) != 1:
        raise ValueError("_new_item text construction differs from the reviewed candidate")
    index, assignment = assignments[0]
    replacement = 'text = str(candidate.get("text", candidate.get("content", ""))).strip()'
    new_item.body[index] = ast.parse(replacement).body[0]
    lines = source.splitlines(keepends=True)
    old_line = lines[assignment.lineno - 1]
    ending = "\r\n" if old_line.endswith("\r\n") else "\n"
    lines[assignment.lineno - 1] = " " * assignment.col_offset + replacement + ending
    lines.insert(branch.lineno - 1, " " * branch.col_offset + "self.db = db" + ending)
    patched = "".join(lines)
    if ast_key(ast.parse(patched)) != ast_key(expected):
        raise ValueError("Provider patch changed more than the two reviewed AST statements")
    compile(patched, "reviewed-praxis-candidate", "exec")
    return patched


def verify_coldstart(source):
    """Execute only three methods in memory; filesystem existence is false."""
    tree = ast.parse(source)
    provider = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "PraxisLoomMemoryProvider")
    methods = [node for node in provider.body if isinstance(node, ast.FunctionDef)
               and node.name in {"_empty_db", "_load_db", "_new_item"}]
    subset = ast.ClassDef(name="Isolated", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[subset], type_ignores=[]))
    seeds = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "COLDSTART_ITEMS" for target in node.targets))
    scope = {**vars(typing), "time": time, "COLDSTART_ITEMS": seeds,
             "os": SimpleNamespace(path=SimpleNamespace(exists=lambda _: False))}
    exec(compile(module, "isolated-praxis-coldstart", "exec"), scope)
    instance = scope["Isolated"]()
    instance.db = None
    instance.db_path = "no-filesystem-access"
    instance._coldstart_injected = False
    instance._compute_signature = lambda text: hashlib.sha256(text.encode()).hexdigest()
    db = instance._load_db()
    if ([item["content"] for item in db["items"]] != [item["content"] for item in seeds]
            or not all(item["content"] for item in db["items"])
            or len({item["id"] for item in db["items"]}) != len(seeds)
            or db["meta"]["next_id"] != len(seeds) + 1):
        raise ValueError("Cold-start repair did not preserve the original seed contents and IDs")
    return {"seed_count": len(seeds), "seed_content_preserved": True, "next_id": db["meta"]["next_id"],
            "filesystem_or_model_calls": 0}


def check_stopped(root, directory):
    for relative in ("worker_source.json", "job_status.json"):
        path = directory / relative
        if path.exists() and process_owned(json.loads(path.read_text(encoding="utf-8"))):
            raise ValueError("Target worker is still running; candidate repair is refused")
    ledger = root / "budget/budget.json"
    if ledger.exists():
        for number in json.loads(ledger.read_text(encoding="utf-8")).get("pending", {}):
            receipt = root / "budget/attempts" / f"{int(number):06d}.json"
            if json.loads(receipt.read_text(encoding="utf-8")).get("job_id") == JOB:
                raise ValueError("Target job still has an in-flight request")


def repair(root, *, expected_provider_sha256, apply=False):
    root = Path(root).resolve()
    directory = root / "jobs" / JOB
    check_stopped(root, directory)
    relative_paths = [PROVIDER, GENERATED, TEMP_PROVIDER, ROUND / "validated_systems.json", ROUND / "state.json"]
    before = {relative.as_posix(): (directory / relative).read_bytes() for relative in relative_paths}
    if sha(before[PROVIDER.as_posix()]) != expected_provider_sha256:
        raise ValueError("Provider bytes differ from the explicitly reviewed SHA256")
    generated = json.loads(before[GENERATED.as_posix()])
    source = before[PROVIDER.as_posix()].decode("utf-8")
    temporary_source = before[TEMP_PROVIDER.as_posix()].decode("utf-8")
    original_code = generated["provider_code"]["code"]
    if (sha(normalized(original_code).encode()) != ORIGINAL_CODE_SHA256
            or normalized(source) != normalized(original_code)
            or normalized(temporary_source) != normalized(original_code)
            or generated["memory_type_info"] != {"enum_name": "PRAXIS_LOOM_MEMORY", "enum_value": CANDIDATE}):
        raise ValueError("Generated, workspace and validation candidate copies differ")
    paid_path = root / "budget/attempts" / f"{PAID_ATTEMPT:06d}.json"
    paid_bytes = paid_path.read_bytes()
    paid = json.loads(paid_bytes)
    choices = (paid.get("response") or {}).get("choices", [])
    if (sha(paid_bytes) != PAID_SHA256 or paid.get("job_id") != JOB or paid.get("role") != "generation"
            or paid.get("status") != "success" or len(choices) != 1
            or choices[0].get("finish_reason") != "stop"
            or choices[0].get("message", {}).get("content") != generated.get("_raw_llm_output")
            or paid.get("usage") != {"prompt_tokens": 26546, "completion_tokens": 31462, "total_tokens": 58008}):
        raise ValueError("Candidate does not match its original complete settled generation receipt")
    for pattern in ("round_*/base_logs/*.json", "round_*/eval_batch/*/*.json", "round_*/eval_finalists/*/*.json"):
        for path in (directory / "evolution").glob(pattern):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("provider") == CANDIDATE:
                raise ValueError("Candidate already owns a learning record; do not change its paid identity")
    validation = json.loads(before[(ROUND / "validated_systems.json").as_posix()])
    state = json.loads(before[(ROUND / "state.json").as_posix()])
    if CANDIDATE not in validation.get("failed", []) or CANDIDATE in validation.get("validated", []):
        raise ValueError("Candidate is not exclusively awaiting failed-validation recovery")
    if CANDIDATE not in state.get("phases", {}).get("validate", {}).get("failed_systems", []):
        raise ValueError("Controller validation state no longer identifies this failed candidate")
    protected_paths = [ROUND / "generated_system_2.json", ROUND / "created_system.json", ROUND / "checkpoint.json",
                       Path("workspace/EvolveLab/memory_types.py"), Path("workspace/EvolveLab/providers/verity_forge_memory_provider.py")]
    protected = {relative.as_posix(): sha((directory / relative).read_bytes()) for relative in protected_paths}
    patched_source = patch_provider(source)
    patched_generated = deepcopy(generated)
    patched_generated["provider_code"]["code"] = patch_provider(original_code)
    if normalized(patched_source) != normalized(patched_generated["provider_code"]["code"]):
        raise ValueError("Patched candidate copies differ")
    coldstart = verify_coldstart(patched_generated["provider_code"]["code"])
    validation["failed"] = [name for name in validation["failed"] if name != CANDIDATE]
    validation.get("details", {}).pop(CANDIDATE, None)
    phase = state["phases"]["validate"]
    phase["completed"] = False
    phase["failed_systems"] = [name for name in phase.get("failed_systems", []) if name != CANDIDATE]
    after = {PROVIDER.as_posix(): patched_source.encode("utf-8"), GENERATED.as_posix(): encoded(patched_generated),
             TEMP_PROVIDER.as_posix(): patch_provider(temporary_source).encode("utf-8"),
             (ROUND / "validated_systems.json").as_posix(): encoded(validation),
             (ROUND / "state.json").as_posix(): encoded(state)}
    evidence = {"job_id": JOB, "candidate": CANDIDATE, "sibling_preserved": SIBLING,
                "paid_attempt": PAID_ATTEMPT, "paid_receipt_sha256": PAID_SHA256,
                "input_hashes": {name: sha(value) for name, value in before.items()},
                "output_hashes": {name: sha(value) for name, value in after.items()},
                "preserved_hashes": protected, "original_raw_llm_output_unchanged": True,
                "changes": ["bind_loaded_db_before_coldstart", "new_item_text_falls_back_to_content",
                            "clear_only_praxis_failed_validation_cache"],
                "coldstart_check": coldstart, "model_calls": 0, "learning_records_changed": False, "applied": False}
    check_stopped(root, directory)
    if any((directory / name).read_bytes() != value for name, value in before.items()):
        raise ValueError("Candidate artifacts changed while preparing recovery")
    if any(sha((directory / name).read_bytes()) != value for name, value in protected.items()):
        raise ValueError("Protected sibling/controller state changed while preparing recovery")
    if not apply:
        return evidence
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = root / "native_recovery_20261030/candidate_repairs" / f"{JOB}-{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    for name, value in before.items():
        path = backup / "before" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    (backup / "paid_generation.before.json").write_bytes(paid_bytes)
    staged = {}
    for name, value in after.items():
        path = directory / name
        temporary = path.with_name(path.name + ".candidate-repair-" + uuid4().hex + ".tmp")
        temporary.write_bytes(value)
        staged[name] = temporary
    check_stopped(root, directory)
    if any((directory / name).read_bytes() != value for name, value in before.items()):
        raise ValueError("Candidate artifacts changed after backup; live artifacts were not edited")
    for name, temporary in staged.items():
        os.replace(temporary, directory / name)
    if any(sha((directory / name).read_bytes()) != value for name, value in protected.items()):
        raise RuntimeError("Protected sibling/controller state changed during candidate recovery")
    evidence.update(applied=True, backup_dir=str(backup), applied_utc=datetime.now(timezone.utc).isoformat())
    (backup / "receipt.json").write_bytes(encoded(evidence))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--expected-provider-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(repair(args.root, expected_provider_sha256=args.expected_provider_sha256,
                            apply=args.apply), ensure_ascii=False, indent=2))
