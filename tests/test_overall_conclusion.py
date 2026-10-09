from backend.assessment_service import APPLICABLE_EVIDENCE_FOUND, APPLICABLE_EVIDENCE_NOT_FOUND, AssessmentService, INSUFFICIENT


def test_box_three_contains_only_box_one_and_box_two_findings() -> None:
    box_one = "Sector-level findings for Energy: Carbon pricing could increase compliance costs."
    box_two = f"{APPLICABLE_EVIDENCE_FOUND}: The borrower has installed methane-abatement equipment."

    conclusion = AssessmentService.overall_conclusion(box_one, box_two)

    assert box_one in conclusion
    assert box_two in conclusion
    assert "Overall E&S credit-risk conclusion" in conclusion
    assert "profitability" not in conclusion
    assert "capital expenditure" not in conclusion


def test_box_three_safely_abstains_when_both_inputs_are_insufficient() -> None:
    assert AssessmentService.overall_conclusion(
        INSUFFICIENT, f"{APPLICABLE_EVIDENCE_NOT_FOUND}: {INSUFFICIENT}"
    ) == INSUFFICIENT
