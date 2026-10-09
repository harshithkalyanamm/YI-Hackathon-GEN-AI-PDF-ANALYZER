"""Controlled assessment services built only from validated evidence."""
from __future__ import annotations

from .document_store import DocumentStore
from .qa_engine import QueryEngine
from .template_writer import INSUFFICIENT, TemplateWriter

APPLICABLE_EVIDENCE_FOUND = "APPLICABLE_EVIDENCE_FOUND"
APPLICABLE_EVIDENCE_NOT_FOUND = "APPLICABLE_EVIDENCE_NOT_FOUND"
APPLICABILITY_UNKNOWN = "APPLICABILITY_UNKNOWN"
NOT_APPLICABLE = "NOT_APPLICABLE"


class AssessmentService:
    """Box 1 baseline: sector trends from one approved primary BES document only."""
    def __init__(self, store: DocumentStore, query_engine: QueryEngine) -> None:
        self.store, self.query_engine = store, query_engine

    def sector_trends(self, document_id: str, sector: str) -> str:
        metadata = self.store.metadata(document_id)
        if metadata.document_type != "PRIMARY_BES":
            raise ValueError("Sector trends require a document uploaded as PRIMARY_BES.")
        # The writer receives only findings already validated by the evidence layer.
        selected_findings = []
        for topic in ("sector risk", "bank risk", "climate transition risk", "nature transition risk"):
            selected_findings.extend(self.query_engine.retrieve_evidence(document_id, sector, topic))
        unique_findings = []
        seen = set()
        for finding in selected_findings:
            if finding.statement not in seen:
                unique_findings.append(finding)
                seen.add(finding.statement)
        return TemplateWriter.sector_trends(sector, unique_findings)

    def deal_risk_drivers(self, document_id: str, transaction_id: str, sector: str, request: str,
                          applicability_override: str | None = None) -> str:
        """Box 2 baseline restricted to exactly one current transaction document.

        This deliberately does not infer financial channels: every sentence in the
        response must be direct, transaction-scoped document evidence.
        """
        metadata = self.store.metadata(document_id)
        if metadata.document_type != "DEAL_DOCUMENT":
            raise ValueError("Deal risk assessment requires a document uploaded as DEAL_DOCUMENT.")
        if metadata.transaction_id != transaction_id:
            raise ValueError("The document does not belong to the requested transaction.")
        if applicability_override == NOT_APPLICABLE:
            return TemplateWriter.deal_risk(NOT_APPLICABLE, [])
        findings = self.query_engine.retrieve_evidence(document_id, sector, request)
        if findings:
            return TemplateWriter.deal_risk(APPLICABLE_EVIDENCE_FOUND, findings)
        if applicability_override == "APPLICABLE":
            return TemplateWriter.deal_risk(APPLICABLE_EVIDENCE_NOT_FOUND, [])
        # A missing mention does not establish that a topic applies or does not apply.
        return TemplateWriter.deal_risk(APPLICABILITY_UNKNOWN, [])

    @staticmethod
    def overall_conclusion(sector_trends: str, deal_risk_drivers: str) -> str:
        """Box 3: deterministic synthesis of validated Box 1 and Box 2 text only.

        No retriever is invoked here and no causal/financial conclusion is added.
        This preserves the required boundary: Box 3 cannot introduce a new fact.
        """
        return TemplateWriter.overall_conclusion(sector_trends, deal_risk_drivers)
