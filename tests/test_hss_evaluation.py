"""Quality differences must survive native selection of open-ended reports."""

import json
import hashlib

import pytest

from philoagent.hss_evaluation import (RUBRICS, judge_messages, native_quality_summary,
                                      normalize_grade_output, rubric, validate_grade)


def grade(domain, score):
    return validate_grade({"dimensions": {name: {"score": score, "reason": "Evidence supports this assessment."}
                                          for name in RUBRICS[domain]},
                           "overall_comment": "A completed report with assessable strengths and gaps."}, domain)


@pytest.mark.parametrize("domain", ["humanities", "social"])
def test_live_complete_judge_grade_with_missing_outer_brace_and_nested_comment(domain):
    # Minimized from the two corresponding failed formal-run judge outputs:
    # all dimension objects and the comment are complete; only their placement
    # and the final outer delimiter differ from the requested schema.
    document = {"dimensions": {name: {"score": 3.5, "reason": f"Evidence for {name}."}
                                for name in RUBRICS[domain]}}
    document["dimensions"]["overall_comment"] = "Complete overall assessment."
    raw = json.dumps(document, ensure_ascii=False)[:-1]
    result = validate_grade(raw, domain)
    assert result["quality_score"] == 70
    assert result["overall_comment"] == "Complete overall assessment."
    assert {name: item["reason"] for name, item in result["dimensions"].items()} == {
        name: f"Evidence for {name}." for name in RUBRICS[domain]}


def test_format_repair_preserves_every_value_and_records_raw_and_normalized_hashes():
    dimensions = {name: {"score": index + .25, "reason": f'依据 {index}: "引语" 与 {{括号}}。'}
                  for index, name in enumerate(RUBRICS["social"])}
    expected = {"dimensions": dimensions, "overall_comment": "完整评语。"}
    misplaced = {"dimensions": {**dimensions, "overall_comment": expected["overall_comment"]}}
    raw = "```json\n" + json.dumps(misplaced, ensure_ascii=False)[:-1] + "\n```"
    normalized, evidence = normalize_grade_output(raw, "social")
    assert normalized == expected
    assert evidence["normalization_version"] == 1
    assert evidence["repaired"]
    assert evidence["repairs"] == [
        {"operation": "remove_json_fence"},
        {"operation": "append_missing_object_closures", "count": 1},
        {"operation": "relocate_overall_comment", "from": "dimensions.overall_comment",
         "to": "overall_comment"}]
    assert evidence["raw_sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    canonical = json.dumps(expected, ensure_ascii=False, sort_keys=True, allow_nan=False)
    assert evidence["normalized_sha256"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_valid_grade_is_unchanged_and_normalization_does_not_mutate_input():
    document = grade("business", 4.2)
    normalized, evidence = normalize_grade_output(document, "business")
    assert normalized == document
    assert not evidence["repaired"]
    assert evidence["repairs"] == []
    normalized["dimensions"]["argument"]["reason"] = "Changed copy."
    assert document["dimensions"]["argument"]["reason"] != "Changed copy."


@pytest.mark.parametrize("duplicate", [
    '"overall_comment":"Conflicting assessment.",',
    '"dimensions":{},',
])
def test_duplicate_root_fields_are_ambiguous_even_when_one_complete_grade_exists(duplicate):
    raw = json.dumps(grade("humanities", 4))
    with pytest.raises(ValueError, match="duplicate"):
        normalize_grade_output("{" + duplicate + raw[1:], "humanities")


def test_duplicate_dimension_scores_are_rejected_instead_of_choosing_last():
    raw = json.dumps(grade("social", 3)).replace('"score": 3.0', '"score": 1.0, "score": 3.0', 1)
    with pytest.raises(ValueError, match="duplicate"):
        validate_grade(raw, "social")


def test_live_humanities_score_vector_ignores_only_auxiliary_placeholder():
    scores = dict(zip(RUBRICS["humanities"], (3.4, 4.2, 3.9, 3.7, 4.6)))
    original = {"dimensions": {name: {"score": score, "reason": f"Unchanged evidence for {name}."}
                                for name, score in scores.items()},
                "overall_comment": "A complete assessment with specific strengths and weaknesses."}
    original["dimensions"]["argument_placeholder"] = 0
    raw = json.dumps(original, ensure_ascii=False)
    normalized, evidence = normalize_grade_output(raw, "humanities")
    result = validate_grade(normalized, "humanities")
    assert result["quality_score"] == 78.6
    assert {name: item["score"] for name, item in result["dimensions"].items()} == scores
    assert {name: original["dimensions"][name] for name in scores} == normalized["dimensions"]
    assert evidence["raw_sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("bad", [True, -1, 6, "4", float("nan"), float("inf")])
def test_auxiliary_projection_never_relaxes_required_score_validation(bad):
    document = grade("humanities", 4)
    document["dimensions"]["argument_placeholder"] = 0
    document["dimensions"]["argument"]["score"] = bad
    with pytest.raises(ValueError):
        normalize_grade_output(document, "humanities")


@pytest.mark.parametrize("damage", ["missing_dimension", "truncated_string", "trailing_comma",
                                  "both_comments", "missing_reason", "surrounding_prose"])
def test_repair_does_not_invent_or_discard_grade_content(damage):
    document = grade("humanities", 4)
    if damage == "missing_dimension":
        del document["dimensions"]["communication"]
    elif damage == "both_comments":
        document["dimensions"]["overall_comment"] = "Conflicting nested assessment."
    elif damage == "missing_reason":
        del document["dimensions"]["argument"]["reason"]
    raw = json.dumps(document)
    if damage == "truncated_string":
        raw = raw[:raw.rfind('"')]
    elif damage == "trailing_comma":
        raw = raw[:-1] + ","
    elif damage == "surrounding_prose":
        raw = "Here is my grade:\n" + raw
    else:
        raw = raw[:-1]
    with pytest.raises(ValueError):
        normalize_grade_output(raw, "humanities")


def test_live_restarted_incomplete_judge_output_cannot_be_repaired():
    # The third live failure restarts the object inside dimensions and repeats
    # the first three dimensions, but never supplies the remaining two.
    fragment = {"dimensions": {name: {"score": 4.2, "reason": "Complete reason."}
                               for name in list(RUBRICS["humanities"])[:3]},
                "overall_comment": "A complete comment cannot replace missing dimensions."}
    raw = json.dumps(fragment)
    restarted = raw[:raw.index('}, "overall_comment"')] + "," + raw
    with pytest.raises(ValueError):
        normalize_grade_output(restarted, "humanities")


@pytest.mark.parametrize("domain", RUBRICS)
def test_fractional_quality_preserves_difference_between_nonzero_reports(domain):
    low, high = grade(domain, 1.25), grade(domain, 4.5)
    assert low["quality_score"] == 25
    assert high["quality_score"] == 90
    result = native_quality_summary([
        {"task_id": "one", "status": "completed", "grade": low, "total_model_tokens": 100},
        {"task_id": "two", "status": "completed", "grade": high, "total_model_tokens": 500}])
    assert result["success"]
    assert result["summary"]["quality_score"] == 57.5
    assert result["summary"]["accuracy"] == .575
    assert result["summary"]["tokens"]["total_tokens"] == 600


def test_actual_native_selection_prefers_quality_even_when_it_costs_more():
    from test_evolution_integration import AUTO
    native = object.__new__(AUTO.AutoEvolver)
    native.use_pareto_selection = False
    low = native_quality_summary([{"status": "completed", "grade": grade("business", 1), "total_model_tokens": 10}])
    high = native_quality_summary([{"status": "completed", "grade": grade("business", 4), "total_model_tokens": 9000}])
    assert native._select_top({"low": low, "high": high}, k=1) == ["high"]


def test_actual_native_selection_does_not_hide_failed_task_in_high_partial_mean():
    from test_evolution_integration import AUTO
    native = object.__new__(AUTO.AutoEvolver)
    native.use_pareto_selection = False
    partial = native_quality_summary([
        {"status": "completed", "grade": grade("business", 5), "total_model_tokens": 100},
        {"task_id": "failure", "status": "judge_failed", "total_model_tokens": 20}])
    complete = native_quality_summary([
        {"status": "completed", "grade": grade("business", 3), "total_model_tokens": 200}])
    assert native._select_top({"partial": partial, "complete": complete}, k=1) == ["complete"]


def test_judge_only_receives_task_material_and_anonymous_final_work():
    task = {"domain": "social", "question": "Explain the evidence and recommend a policy.",
            "materials": [{"id": "M1", "content": "Survey finding."}],
            "task_criteria": ["Consider sampling limitations."], "deliverable": "Report",
            "group": "SECRET_GROUP", "phi": "SECRET_PHI", "memory": "SECRET_MEMORY",
            "reflection": "SECRET_REFLECTION", "cost": "SECRET_COST"}
    messages = judge_messages(task, "Anonymous work cites M1.")
    text = json.dumps(messages)
    assert "M1" in text
    assert all(secret not in text for secret in ("SECRET_GROUP", "SECRET_PHI", "SECRET_MEMORY", "SECRET_REFLECTION", "SECRET_COST"))


@pytest.mark.parametrize("bad", [True, -1, 6, float("nan"), float("inf"), "4"])
def test_malformed_score_is_a_judge_error_not_a_low_quality_answer(bad):
    document = grade("humanities", 3)
    document["dimensions"]["interpretation"]["score"] = bad
    with pytest.raises(ValueError):
        validate_grade(document, "humanities")


def test_total_cannot_be_forged_and_invalid_batches_are_preserved():
    completed = {"status": "completed", "grade": grade("social", 3), "total_model_tokens": 100}
    completed["grade"]["quality_score"] = 100
    with pytest.raises(ValueError, match="Saved quality"):
        native_quality_summary([completed])
    completed["grade"] = grade("social", 3)
    result = native_quality_summary([completed, {"task_id": "broken", "status": "judge_failed", "total_model_tokens": 20}])
    assert not result["success"]
    assert result["summary"]["quality_score"] == 60
    assert result["summary"]["graded_tasks"] == 1
    assert result["summary"]["total_tasks"] == 2
    assert result["summary"]["failures"] == [{"task_id": "broken", "status": "judge_failed"}]


@pytest.mark.parametrize("usage", [None, True, -1, "100"])
def test_unknown_or_invalid_usage_cannot_be_ranked_as_free(usage):
    with pytest.raises(ValueError, match="unknown usage"):
        native_quality_summary([{"status": "completed", "grade": grade("business", 4),
                                 "total_model_tokens": usage}])
    with pytest.raises(ValueError, match="unknown usage"):
        native_quality_summary([{"status": "completed", "grade": grade("business", 4)}])
