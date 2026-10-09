"""Small, offline golden-set evaluation for retrieval regressions."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .retriever import Retriever


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    document_id: str
    sector: str
    request: str
    expected_section: str | None
    should_find_evidence: bool


@dataclass(frozen=True)
class EvaluationReport:
    total_cases: int
    recall_at_k: float
    correct_abstention_rate: float
    wrong_sector_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_retriever(retriever: Retriever, cases: list[RetrievalCase], top_k: int = 5) -> EvaluationReport:
    """Measure core safety/quality signals without sending documents anywhere."""
    found_expected = 0
    correct_abstentions = 0
    wrong_sector = 0
    answerable = 0
    unanswerable = 0
    for case in cases:
        results = retriever.search(case.document_id, case.sector, case.request, top_k)
        wrong_sector += sum(chunk.sector not in {None, case.sector} for chunk in results)
        if case.should_find_evidence:
            answerable += 1
            found_expected += any(chunk.section == case.expected_section for chunk in results)
        else:
            unanswerable += 1
            correct_abstentions += not results
    return EvaluationReport(
        total_cases=len(cases),
        recall_at_k=found_expected / answerable if answerable else 1.0,
        correct_abstention_rate=correct_abstentions / unanswerable if unanswerable else 1.0,
        wrong_sector_rate=wrong_sector / max(1, len(cases) * top_k),
    )
