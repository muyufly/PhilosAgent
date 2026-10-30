"""Preserve complete paid grades with unambiguous JSON layout variations."""

from copy import deepcopy
import json

import pytest

from philoagent.hss_evaluation import normalize_grade_output, rubric, validate_grade


def grade():
    return {"dimensions": {name: {"score": 4.2, "reason": "Evidence retained."}
                            for name in rubric("business")}, "overall_comment": "Complete assessment."}


def test_complete_flattened_grade_wraps_dimensions_without_changing_values():
    expected = grade()
    flattened = {**expected["dimensions"], "overall_comment": expected["overall_comment"]}
    normalized, evidence = normalize_grade_output(flattened, "business")
    assert normalized == expected and evidence["repaired"]


def test_identical_redundant_comment_has_one_canonical_location():
    expected = grade()
    duplicated = deepcopy(expected)
    duplicated["dimensions"]["overall_comment"] = duplicated["overall_comment"]
    normalized, evidence = normalize_grade_output(duplicated, "business")
    assert normalized == expected and evidence["repaired"]


def test_schema_comment_wins_and_auxiliary_comment_is_preserved_in_audit():
    expected = grade()
    duplicated = deepcopy(expected)
    duplicated["dimensions"]["overall_comment"] = "A conflicting assessment."
    raw = json.dumps(duplicated)
    normalized, evidence = normalize_grade_output(raw, "business")
    assert normalized == expected
    assert evidence["excluded_fields"] == {"dimensions.overall_comment": "A conflicting assessment."}
    assert validate_grade(normalized, "business")["quality_score"] == 84
    assert duplicated["dimensions"]["overall_comment"] == "A conflicting assessment."


def test_complete_rubric_projects_placeholder_without_altering_any_grade_content():
    expected = grade()
    original = deepcopy(expected)
    original["dimensions"]["argument_placeholder"] = 0
    raw = json.dumps(original, ensure_ascii=False)
    normalized, evidence = normalize_grade_output(raw, "business")
    assert normalized == expected
    assert validate_grade(normalized, "business")["quality_score"] == 84
    assert evidence["excluded_fields"] == {"dimensions.argument_placeholder": 0}
    assert original["dimensions"]["argument_placeholder"] == 0


@pytest.mark.parametrize("domain", ["humanities", "social", "business"])
def test_auxiliary_object_has_no_score_or_weight_and_is_preserved_in_audit(domain):
    expected = {"dimensions": {name: {"score": 3.5, "reason": f"Evidence: {name}."}
                               for name in rubric(domain)}, "overall_comment": "An unchanged assessment."}
    original = deepcopy(expected)
    extra = {"score": 0, "weight": 999999, "reason": "Not part of the rubric.",
             "nested": {"notes": ["Retain every value", 42, None]}}
    original["dimensions"]["auxiliary_evaluation"] = deepcopy(extra)
    normalized, evidence = normalize_grade_output(original, domain)
    assert normalized == expected
    assert validate_grade(normalized, domain)["quality_score"] == 70
    assert evidence["excluded_fields"] == {"dimensions.auxiliary_evaluation": extra}
    assert original["dimensions"]["auxiliary_evaluation"] == extra
    evidence["excluded_fields"]["dimensions.auxiliary_evaluation"]["score"] = 999
    assert all(item["score"] == 3.5 for item in normalized["dimensions"].values())


def test_missing_required_dimension_cannot_be_replaced_by_extra_or_alias():
    document = grade()
    document["dimensions"]["execution_alias"] = document["dimensions"].pop("execution")
    document["dimensions"]["placeholder"] = 0
    with pytest.raises(ValueError, match=r"missing=\['execution'\]; actual=.*execution_alias"):
        normalize_grade_output(document, "business")


def test_multiple_auxiliary_fields_are_retained_without_changing_complete_rubric():
    expected = grade()
    document = grade()
    document["dimensions"]["placeholder"] = 0
    document["dimensions"]["overall_comment"] = "A conflicting comment."
    normalized, evidence = normalize_grade_output(document, "business")
    assert normalized == expected
    assert evidence["excluded_fields"] == {"dimensions.placeholder": 0,
                                            "dimensions.overall_comment": "A conflicting comment."}


@pytest.mark.parametrize("damage", ["missing_dimension", "invalid_score", "missing_reason", "invalid_top_comment"])
def test_auxiliary_comment_cannot_repair_incomplete_or_invalid_rubric(damage):
    document = grade()
    document["dimensions"]["overall_comment"] = "A nested auxiliary assessment."
    if damage == "missing_dimension":
        document["dimensions"].pop("execution")
    elif damage == "invalid_score":
        document["dimensions"]["execution"]["score"] = 6
    elif damage == "missing_reason":
        document["dimensions"]["execution"].pop("reason")
    else:
        document["overall_comment"] = None
    with pytest.raises(ValueError):
        normalize_grade_output(document, "business")


def test_actual_paid_business_grade_can_be_parsed_without_changing_raw_or_scores():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "runs/hss-business-formal-20261029-r01/budget/attempts/013093.json"
    paid = json.loads(path.read_text(encoding="utf-8"))
    raw = paid["response"]["choices"][0]["message"]["content"]
    parsed = json.loads(raw)
    normalized, evidence = normalize_grade_output(raw, "business")
    assert normalized["dimensions"] == {name: parsed["dimensions"][name] for name in rubric("business")}
    assert normalized["overall_comment"] == parsed["overall_comment"]
    assert evidence["excluded_fields"] == {"dimensions.overall_comment": parsed["dimensions"]["overall_comment"]}
    assert validate_grade(normalized, "business")["quality_score"] == 91.3
    assert paid["response"]["choices"][0]["message"]["content"] == raw


@pytest.mark.parametrize("redundant", [False, True])
def test_nested_comment_layout_and_auxiliary_fields_are_normalized_independently(redundant):
    expected = grade()
    document = deepcopy(expected)
    document["dimensions"]["overall_comment"] = document["overall_comment"]
    document["dimensions"]["placeholder"] = 0
    if not redundant:
        document.pop("overall_comment")
    normalized, evidence = normalize_grade_output(document, "business")
    assert normalized == expected
    assert evidence["excluded_fields"] == {"dimensions.placeholder": 0}


def test_flattened_complete_core_can_preserve_auxiliary_values_without_scoring_them():
    expected = grade()
    original = {**expected["dimensions"], "overall_comment": expected["overall_comment"],
                "auxiliary": {"score": 100000, "note": "Not a dimension."}}
    normalized, evidence = normalize_grade_output(original, "business")
    assert normalized == expected
    assert evidence["excluded_fields"] == {"auxiliary": original["auxiliary"]}


@pytest.mark.parametrize("extra", ['NaN', 'Infinity', '-Infinity', '1e999', '{"nested":1e999}'])
def test_nonfinite_auxiliary_json_is_rejected_before_projection(extra):
    raw = json.dumps(grade()).replace('"dimensions": {', '"dimensions": {"auxiliary": ' + extra + ', ', 1)
    with pytest.raises(ValueError):
        normalize_grade_output(raw, "business")


def test_duplicate_auxiliary_json_keys_are_not_resolved_by_projection():
    raw = json.dumps(grade()).replace('"dimensions": {', '"dimensions": {"auxiliary":1,"auxiliary":0, ', 1)
    with pytest.raises(ValueError, match="duplicate JSON fields"):
        normalize_grade_output(raw, "business")
