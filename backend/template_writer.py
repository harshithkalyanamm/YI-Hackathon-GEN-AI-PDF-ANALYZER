"""Safe deterministic output templates; no generative model is involved."""
from __future__ import annotations

from .evidence import EvidenceFinding

INSUFFICIENT = "Insufficient information in the provided documents to support a conclusion."


class TemplateWriter:
    @staticmethod
    def sector_trends(sector: str, findings: list[EvidenceFinding]) -> str:
        if not findings:
            return INSUFFICIENT
        return f"Sector-level findings for {sector}: " + " ".join(finding.statement for finding in findings[:4])

    @staticmethod
    def deal_risk(status: str, findings: list[EvidenceFinding]) -> str:
        if status == "NOT_APPLICABLE":
            return "NOT_APPLICABLE: The supplied assessment marks this topic as not applicable."
        if findings:
            return f"{status}: " + " ".join(finding.statement for finding in findings[:3])
        return f"{status}: {INSUFFICIENT}"

    @staticmethod
    def overall_conclusion(sector_trends: str, deal_risk_drivers: str) -> str:
        if sector_trends == INSUFFICIENT and INSUFFICIENT in deal_risk_drivers:
            return INSUFFICIENT
        return " ".join((
            "Overall E&S credit-risk conclusion based on validated findings:",
            f"Sector context: {sector_trends}",
            f"Deal context: {deal_risk_drivers}",
        ))
