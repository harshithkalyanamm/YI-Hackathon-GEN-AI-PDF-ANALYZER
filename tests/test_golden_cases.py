import json
from pathlib import Path


def test_golden_cases_are_valid_and_include_safe_abstention() -> None:
    cases_path = Path(__file__).parents[1] / "evaluation" / "golden_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    assert len(cases) >= 5
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert any(not case["should_find_evidence"] for case in cases)
    assert all("sector" in case and "request" in case for case in cases)
