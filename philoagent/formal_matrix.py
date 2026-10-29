"""Prepare and verify a reviewable formal matrix without credentials or calls.

This is deliberately an offline entrypoint. A proposal hash freezes the material
being reviewed, not user approval. There is no live-run or approval command here.
The existing decision_smoke entrypoint is not a formal-matrix executor.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys


PROJECT = Path(__file__).resolve().parent.parent
GROUPS = {
    "M0": {"phi": False, "ordinary_learning": False, "system_reflection": False,
           "adopt_phi": False, "memory_start": "initial_per_task"},
    "M1": {"phi": False, "ordinary_learning": True, "system_reflection": False,
           "adopt_phi": False, "memory_start": "previous_primary_learning_task"},
    "S0": {"phi": True, "ordinary_learning": True, "system_reflection": True,
           "adopt_phi": False, "memory_start": "previous_primary_learning_task"},
    "S1": {"phi": True, "ordinary_learning": True, "system_reflection": True,
           "adopt_phi": True, "memory_start": "previous_primary_learning_task"},
}

# A rotation balances group position in a proposed three-stream run as closely
# as possible. Do not shuffle the causal order inside a learning stream.
GROUP_ORDERS = (("M0", "M1", "S0", "S1"), ("M1", "S0", "S1", "M0"),
                ("S0", "S1", "M0", "M1"))
SAMPLING = {"model": "deepseek-flash", "temperature": 0,
            "thinking": "disabled", "api_seed": None,
            "deterministic_model_outputs": False,
            "model_changes_require_new_proposal": True}
LIMITS = {"actor_max_steps": 6, "actor_max_output_tokens": 3072,
          "analysis_max_steps": 4, "analysis_max_output_tokens": 4096,
          "generation_max_output_tokens": 12000,
          "validation_max_output_tokens": 3072, "native_workers": 1,
          "maximum_inflight_requests": 1,
          "candidates_per_system_lineage": 1, "candidate_auto_repairs": 0,
          "sdk_retries": 0, "task_retries": 0}


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def protocol():
    """Explicit pending protocol; ordinary reflection uses the native extractor."""
    return {
        "groups": deepcopy(GROUPS), "sampling": deepcopy(SAMPLING), "limits": deepcopy(LIMITS),
        "p": "same_native_lightweight_provider_source_and_nonrouting_configuration",
        "method_location": "four_strategic_items_in_initial_native_M_not_Phi",
        "method_update_rule": "native_ordinary_learning_only; no_separate_method_editor",
        "method_visibility": "same_retrieval_candidate_position; selection_is_native_not_forced",
        "initial_memory": "native_seven_coldstart_priors_plus_four_public_methods",
        "ordinary_reflection": "native_successful_trajectory_extraction; no_additional_reflection_model",
        "failed_learning": "native_skips_extraction; actual_failure_logs_visible_to_system_reflection",
        "system_reflection": {
            "boundary": "after_all_four_primary_learning_tasks_before_any_holdout",
            "inputs": "actual_primary_learning_logs_and_final_M_and_current_Phi_only",
            "same_opportunities_for": ["S0", "S1"],
            "proposal": "native_phi_only_analyze_generate_create; P_frozen; retention_allowed",
            "validation": "native_fast_interface_validation_in_discarded_copy; no_auto_fix",
            "comparison": "parent_and_one_candidate_each_rerun_four_learning_tasks",
            "finalists": "parent_and_candidate_each_rerun_three_prespecified_learning_tasks",
            "selection": "native_AutoEvolver_select_top_accuracy_then_total_model_tokens",
            "selection_metric_limit": "native_success_ranking_does_not_optimize_net_utility; report_both",
            "comparison_memory": "fresh_identical_final_primary_M_per_task; ingestion_disabled",
            "extra_experience_returned_to_primary_M": False,
            "S0_actor_receives_shadow_candidate_or_analysis": False,
            "S0_adoption": "retain_Phi0_even_if_shadow_selection_prefers_candidate",
            "S1_adoption": "selected_parent_or_candidate; no_forced_change",
            "later_learning_under_revised_phi": False,
            "discovery_actor_executions": "the_four_primary_learning_tasks; no_extra_discovery_replay",
            "malformed_candidate": "record_and_exclude; keep_parent; no_silent_retry",
            "technical_failure": "stop_and_preserve; never_treat_as_decision_failure",
        },
        "holdout": {"ingestion": False, "feedback_to_evolver": False,
                    "each_task": "fresh_process_and_full_final_M_copy",
                    "M0_memory": "initial_M_copy",
                    "ordering": "primary_holdout_after_revision; probes_use_independent_copies",
                    "after_inspection_changes": "new_exploratory_version_and_new_confirmatory_holdout"},
        "paired_probes": {"lineage": "S1", "P": "fixed", "M": "final_primary_learning_M",
                          "same_phi_repetitions_per_task": 2,
                          "phi0_repetitions_per_task": 2, "phit_repetitions_per_task": 2,
                          "same_phi_condition": "Phi0_with_final_S1_M",
                          "primary_holdout_reused_as_repeat": False,
                          "version_order": "ABBA_for_first_probe_BAAB_for_second",
                          "new_independent_samples": False,
                          "identical_phi": "record_retention_and_measure_execution_variability"},
        "scoring": {"deterministic_environment": True,
                    "report": ["completed", "reward", "cost", "utility", "time",
                               "violations", "invalid_calls", "precondition_failures", "actual_tool_trace", "model_tokens"],
                    "constraint_failure_independent_of_reward": True,
                    "oracle": "evaluator_only; never_actor_or_reflection_or_seed",
                    "failure_classes": ["decision_failure", "constraint_failure", "invalid_output",
                                        "no_action", "environment_tool_failure", "budget_interruption"],
                    "aggregation_unit": "independent_task_stream_or_lineage_not_API_call_or_replay"},
        "isolation": {"process": "new_for_every_primary_task_and_comparison_and_probe",
                      "state": "full_provider_storage_and_private_cache_copy; no_shared_writes",
                      "retrieval_phi": "unchanged_native_synthesis_without_complete_Phi",
                      "allow_extra_failure_ingestion": False,
                      "allow_tool_or_P_evolution": False,
                      "holdout_material_in_learning_directory": False},
        "stopping": {"unknown_usage": "stop", "pending_request": "stop",
                     "failed_network_or_model_request": "stop; no_automatic_retry",
                     "token_limit": "soft_stop_with_pre_request_reservation_and_post_response_settlement",
                     "request_limit": "hard_pre_network_cap",
                     "stage_limits": "must_be_approved_before_live_execution",
                     "unit_review_triggers": {
                         "actor_task": {"requests": 12, "actual_tokens": 51000},
                         "system_analysis_generation": {"requests": 8, "actual_tokens": 113000},
                         "interface_validation": {"requests": 4, "actual_tokens": 32288}},
                     "on_unit_trigger": "stop_after_inflight_response; retain_incomplete_lineage; no_extra_budget",
                     "price": "not_assumed; provider_bill_and_rates_needed_for_currency_cap"},
    }


def build_schedule(streams):
    """Pure schedule. No result-based task choice, task reordering, or API seed."""
    entries = []

    def add(stream_id, group, phase, task_id=None, **extra):
        entries.append({"event_id": f"event-{len(entries) + 1:04d}", "stream": stream_id,
                        "group": group, "phase": phase, "task_id": task_id, **extra})

    seen_ids = set()
    if not streams:
        raise ValueError("At least one independent task stream is required")
    for stream_index, (stream_id, stream) in enumerate(streams.items()):
        if not isinstance(stream_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", stream_id):
            raise ValueError("Task stream identifiers must be safe lowercase identifiers")
        if (not isinstance(stream, dict)
                or any(not isinstance(stream.get(key), (list, tuple)) for key in ("train_ids", "probe_ids"))):
            raise ValueError("Task ID splits must be explicit lists or tuples")
        train, probes = list(stream["train_ids"]), list(stream["probe_ids"])
        identifiers = train + probes
        if (len(train) != 4 or len(probes) != 2
                or any(not isinstance(item, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", item)
                       for item in identifiers)
                or len(set(identifiers)) != 6 or seen_ids.intersection(identifiers)):
            raise ValueError("Each stream requires four learning and two distinct heldout IDs")
        seen_ids.update(identifiers)
        finalists = [train[index] for index in (0, 2, 3)]
        # All four groups reach each learning checkpoint before the next one.
        order = GROUP_ORDERS[stream_index % len(GROUP_ORDERS)]
        for task_id in train:
            for group in order:
                add(stream_id, group, "primary_learning", task_id,
                    ingest=GROUPS[group]["ordinary_learning"], phi="initial",
                    memory="initial" if group == "M0" else "persistent_primary_M")
        for group in ("S0", "S1"):
            add(stream_id, group, "system_reflection", ingest=False)
            add(stream_id, group, "native_interface_validation", ingest=False, memory="discarded_copy")
        # The first approval option ends here, before any behavioral comparison.
        for group in ("S0", "S1"):
            for task_index, task_id in enumerate(train):
                for version in (("parent", "candidate") if task_index % 2 == 0 else ("candidate", "parent")):
                    add(stream_id, group, "candidate_comparison", task_id, ingest=False,
                        phi=version, memory="fresh_final_primary_M_copy")
            for task_index, task_id in enumerate(finalists):
                for version in (("candidate", "parent") if task_index % 2 == 0 else ("parent", "candidate")):
                    add(stream_id, group, "finalist_comparison", task_id, ingest=False,
                        phi=version, memory="fresh_final_primary_M_copy")
            add(stream_id, group, "native_selection", ingest=False,
                adoption="retain_initial" if group == "S0" else "native_winner")
        for task_id in probes:
            for group in order:
                add(stream_id, group, "primary_holdout", task_id, ingest=False,
                    phi="selected" if group == "S1" else "initial",
                    memory="initial" if group == "M0" else "fresh_final_primary_M_copy")
        for probe_index, task_id in enumerate(probes):
            for repetition in range(2):
                add(stream_id, "S1", "same_phi_baseline", task_id, ingest=False,
                    phi="initial", repetition=repetition, memory="fresh_final_primary_M_copy")
            versions = ("initial", "selected", "selected", "initial") if probe_index == 0 else (
                "selected", "initial", "initial", "selected")
            counts = Counter()
            for version in versions:
                add(stream_id, "S1", "paired_probe", task_id, ingest=False, phi=version,
                    repetition=counts[version], memory="fresh_final_primary_M_copy")
                counts[version] += 1
    return entries


def schedule_counts(schedule):
    phases = Counter(entry["phase"] for entry in schedule)
    actor_phases = {"primary_learning", "primary_holdout", "candidate_comparison",
                    "finalist_comparison", "same_phi_baseline", "paired_probe"}
    return {"events": len(schedule), "by_phase": dict(sorted(phases.items())),
            "actor_task_executions": sum(value for phase, value in phases.items() if phase in actor_phases),
            "system_lineages": phases["system_reflection"]}


def budget_estimate(schedule):
    """Observed-envelope extrapolation, never a permission or a hard token bound."""
    counts = schedule_counts(schedule)
    tasks, systems = counts["actor_task_executions"], counts["system_lineages"]
    task_unit = {"requests": 12, "input_tokens": 45000, "output_tokens": 6000}
    reflection_unit = {"requests": 8, "input_tokens": 76000, "output_tokens": 37000}
    validation_unit = {"requests": 4, "input_tokens": 20000, "output_tokens": 12288}
    totals = {key: tasks * task_unit[key] + systems * (reflection_unit[key] + validation_unit[key])
              for key in task_unit}
    totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return {"method": "planning_units_from_r01; not_worst_case_bound; validator_unit_unmeasured",
            "task_unit": task_unit, "reflection_unit": reflection_unit,
            "interface_validation_unit": validation_unit, "planning_envelope": totals,
            "proposed_limits": {"max_calls": 800 if tasks == 64 else 2400,
                                "max_tokens_soft_stop": 4000000 if tasks == 64 else 12000000},
            "first_stage_option": {"scope": "one_stream_primary_learning_system_proposals_and_interface_validation",
                                   "actor_task_executions": 16, "system_lineages": 2,
                                   "planning_requests": 216, "planning_tokens": 1106576,
                                   "proposed_max_calls": 216, "proposed_token_soft_stop": 1300000,
                                   "automatically_continue_to_comparisons": False},
            "approved_limits": None, "currency_estimate": None,
            "automatic_budget_expansion": False,
            "tail_risk": "one_admitted_response_may_exceed_reservation; unknown_usage_locks_ledger"}


def _source_hashes():
    # Hash all relevant adapter/vendor Python sources, excluding datasets,
    # credentials, model outputs and ignored bytecode. No environment is loaded.
    from .cli import FLASH, source_hash
    return {"adapter_python_sha256": source_hash(PROJECT / "philoagent"),
            "native_python_sha256": source_hash(FLASH),
            "seed_files": {name: sha((PROJECT / name).read_bytes()) for name in (
                "seeds/systems/dialectical-v1.json", "seeds/systems/common-methods.json")}}


def build_bundle(*, scope="pilot"):
    from .formal_memory import build_initial_memory, phi_seed
    from .formal_tasks import DESIGN_VERSION, STREAMS, Episode, export_evaluator_spec

    if scope not in {"pilot", "full"}:
        raise ValueError("scope must be pilot or full")
    streams = dict(STREAMS)
    if scope == "pilot":
        streams = dict(list(streams.items())[:1])
    schedule = build_schedule(streams)
    artifacts = {
        "protocol.json": canonical(protocol()),
        "schedule.json": canonical(schedule),
        "seeds/phi0.json": canonical({"schema_version": 1, "items": list(phi_seed())}),
        "seeds/initial_memory.json": canonical(build_initial_memory()),
    }
    for stream_id, stream in streams.items():
        artifacts[f"learning/{stream_id}.json"] = canonical(
            [Episode(task_id).public_task() for task_id in stream["train_ids"]])
        artifacts[f"evaluation_only/{stream_id}.json"] = canonical(
            {task_id: export_evaluator_spec(task_id) for task_id in (*stream["train_ids"], *stream["probe_ids"])})
    artifact_hashes = {name: sha(data) for name, data in sorted(artifacts.items())}
    manifest = {
        "schema_version": 1, "kind": "formal_matrix_review_proposal",
        "date": "2026-10-29", "timezone": "Asia/Hong_Kong", "scope": scope,
        "status": "pending_user_review", "formal_matrix_authorized": False,
        "live_model_calls_performed": False, "model_training_or_probe_results_observed": False,
        "live_executor_ready": False,
        "remaining_before_live": ["specific_protocol_and_budget_user_approval",
                                  "formal_native_validation_selection_live_adapter",
                                  "offline_transport_closure_of_full_formal_lifecycle"],
        "pilot_policy": "pilot_is_exploratory; do_not_pool_into_later_confirmatory_results",
        "scope_policy": "pilot_and_full_are_alternatives; after_pilot_full_is_exploratory_and_confirmation_needs_new_tasks",
        "task_generation": {"mode": "hand_authored_deterministic_versioned_fixtures",
                            "version": DESIGN_VERSION,
                            "random_seed": None, "model_sampling_seed": None,
                            "independence": "distinct_causal_structures; no_iid_population_claim"},
        "streams": {name: {"train_ids": list(value["train_ids"]), "probe_ids": list(value["probe_ids"])}
                    for name, value in streams.items()},
        "group_order": [list(order) for order in GROUP_ORDERS[:len(streams)]],
        "counts": schedule_counts(schedule), "budget": budget_estimate(schedule),
        "sources": _source_hashes(), "artifact_hashes": artifact_hashes,
        "review_reference": "research/formal-matrix-review-20261029.md",
        "inference_limit": f"{len(streams)}_designed_streams; repeats_and_comparison_runs_not_new_samples; not_iid",
    }
    manifest["proposal_sha256"] = sha(manifest)
    artifacts["manifest.json"] = canonical(manifest)
    return manifest, artifacts


def write_bundle(destination, *, scope="pilot"):
    """Conflict-check every file before writing; identical retries are read-only."""
    from .formal_memory import _safe_path
    destination = _safe_path(destination)
    manifest, artifacts = build_bundle(scope=scope)
    for relative, data in artifacts.items():
        path = _safe_path(destination / relative)
        if path.exists() and (not path.is_file() or path.read_bytes() != data):
            raise ValueError(f"Refusing changed review artifact: {path}")
    for relative, data in artifacts.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(data)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Concurrent review artifact conflict: {path}") from None
    return {"directory": str(destination), **manifest}


def verify_bundle(destination):
    """Validate hashes AND current protocol; a recomputed hash is not approval."""
    from .formal_memory import _safe_path, snapshot_manifest
    root = _safe_path(destination)
    snapshot_manifest(root)  # Also reject linked directories and special files.
    saved = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in saved.items() if key != "proposal_sha256"}
    if saved.get("proposal_sha256") != sha(unsigned):
        raise ValueError("Proposal manifest hash differs")
    expected, artifacts = build_bundle(scope=saved["scope"])
    if saved != expected:
        raise ValueError("Proposal differs from current sources, protocol or pending approval state")
    found = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if found != set(artifacts):
        raise ValueError("Review bundle file inventory differs")
    for relative, data in artifacts.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"Review artifact changed: {relative}")
    return {"verified": True, "directory": str(root), "proposal_sha256": expected["proposal_sha256"],
            "formal_matrix_authorized": False, "live_model_calls_performed": False,
            "live_executor_ready": False, "counts": expected["counts"], "budget": expected["budget"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Write pending review inputs; zero API calls")
    prepare.add_argument("destination", type=Path)
    prepare.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    verify = commands.add_parser("verify", help="Recheck a complete proposal offline")
    verify.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    try:
        result = (write_bundle(args.destination, scope=args.scope) if args.command == "prepare"
                  else verify_bundle(args.destination))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
