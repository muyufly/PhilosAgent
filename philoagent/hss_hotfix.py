"""Publish a provenance-recorded controller fix without interrupting live jobs.

This operation does not change the experimental protocol or task identities.
Already running interpreters retain their loaded adapter version; future workers
validate and record the newly frozen sources at startup.
"""

from copy import deepcopy
from pathlib import Path
import shutil

from .cli import PROJECT, FLASH, source_hash
from .hss_experiment import read_json, write_json, now, process_owned, file_sha


def _loaded_source(root, manifest, state, fallback_snapshot):
    source = state.get("worker_source")
    if source:
        return {"loaded_adapter_revision": source["adapter_revision"],
                "loaded_adapter_sha256": source["adapter_sha256"],
                "source_snapshot": source["snapshot_path"]}
    for amendment in reversed(manifest.get("amendments", [])):
        path = Path(amendment.get("evidence", ""))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            continue
        for previous in read_json(path).get("live_workers_preserved", []):
            if previous.get("pid") != state.get("pid"):
                continue
            old_birth = (previous.get("process_identity") or {}).get("started_unix")
            current_birth = (state.get("process_identity") or {}).get("started_unix")
            if old_birth is None or current_birth is None or abs(old_birth - current_birth) > .01:
                continue
            return {key: previous[key] for key in
                    ("loaded_adapter_revision", "loaded_adapter_sha256", "source_snapshot")}
    return {"loaded_adapter_revision": manifest.get("adapter_revision") or manifest["protocol"].get("revision"),
            "loaded_adapter_sha256": manifest["adapter_sha256"], "source_snapshot": str(fallback_snapshot)}


def publish(root, *, revision="repair-r5", reason):
    root = Path(root).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    target = root / "revisions" / revision
    if manifest.get("adapter_revision") == revision:
        if source_hash(PROJECT / "philoagent") != manifest["adapter_sha256"]:
            raise ValueError("Published hotfix sources changed")
        return read_json(target / "amendment.json")
    if target.exists():
        raise ValueError("Partial hotfix exists; inspect it before proceeding")
    if source_hash(FLASH) != manifest["upstream_sha256"]:
        raise ValueError("Native source changed; a controller-only hotfix cannot include it")
    for relative, expected in manifest["input_hashes"].items():
        if file_sha(root / relative) != expected:
            raise ValueError("Frozen experiment input changed")
    old_revision = manifest.get("adapter_revision") or manifest["protocol"].get("revision")
    old_snapshot = root / manifest.get("adapter_snapshot", f"revisions/{old_revision}/code_snapshot.after/philoagent")
    if not old_snapshot.exists():
        old_snapshot = root / "code_snapshot/philoagent"
    if source_hash(old_snapshot) != manifest["adapter_sha256"]:
        raise ValueError("Previous source snapshot no longer matches its freeze")
    current = PROJECT / "philoagent"
    names = {path.relative_to(current) for path in current.rglob("*.py")}
    names |= {path.relative_to(old_snapshot) for path in old_snapshot.rglob("*.py")}
    changed = [name.as_posix() for name in sorted(names)
               if not (current / name).exists() or not (old_snapshot / name).exists()
               or (current / name).read_bytes() != (old_snapshot / name).read_bytes()]
    live = []
    for path in (root / "jobs").glob("*/job_status.json"):
        state = read_json(path)
        if state.get("status") == "running" and process_owned(state):
            live.append({"job_id": path.parent.name, "process_identity": state.get("process_identity"),
                         "pid": state["pid"], **_loaded_source(root, manifest, state, old_snapshot)})
    target.mkdir(parents=True)
    write_json(target / "manifest.before.json", manifest)
    shutil.copytree(current, target / "code_snapshot.after/philoagent",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    after = deepcopy(manifest)
    after.update(adapter_revision=revision, adapter_sha256=source_hash(current),
                 adapter_snapshot=f"revisions/{revision}/code_snapshot.after/philoagent")
    after.setdefault("amendments", []).append({"revision": revision, "kind": "controller_only_hotfix",
        "user_message": "出现报错，请修改", "reason": reason, "recorded_utc": now(),
        "protocol_unchanged": True, "evidence": str(target / "amendment.json")})
    evidence = {"revision": revision, "kind": "controller_only_hotfix", "recorded_utc": now(),
        "reason": reason, "changed_files": changed, "live_workers_preserved": live,
        "prior_adapter_sha256": manifest["adapter_sha256"], "adapter_sha256": after["adapter_sha256"],
        "protocol_unchanged": after["protocol"] == manifest["protocol"],
        "jobs_unchanged": after["jobs"] == manifest["jobs"],
        "task_records_modified": False, "budget_ledger_modified": False,
        "same_snapshot_for_future_workers": after["adapter_snapshot"]}
    write_json(target / "amendment.json", evidence)
    write_json(root / "manifest.json", after)
    return evidence


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--revision", default="repair-r5")
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    evidence = publish(args.root, revision=args.revision, reason=args.reason)
    print(json.dumps({"revision": evidence["revision"], "changed_files": evidence["changed_files"],
                      "live_workers_preserved": len(evidence["live_workers_preserved"])}, ensure_ascii=False))
