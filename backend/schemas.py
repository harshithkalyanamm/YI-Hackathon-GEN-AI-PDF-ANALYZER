from enum import Enum

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    GENERAL = "GENERAL"
    PRIMARY_BES = "PRIMARY_BES"
    DEAL_DOCUMENT = "DEAL_DOCUMENT"


class ApplicabilityOverride(str, Enum):
    """Optional analyst-supplied state; never guessed by the API."""
    NOT_APPLICABLE = "NOT_APPLICABLE"
    APPLICABLE = "APPLICABLE"


class UploadResponse(BaseModel):
    document_id: str
    status: str


class QueryRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=100)
    sector: str = Field(min_length=1, max_length=200)
    request: str = Field(min_length=1, max_length=500)


class QueryResponse(BaseModel):
    answer: str


class SectorTrendsRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=100)
    sector: str = Field(min_length=1, max_length=200)


class AssessmentResponse(BaseModel):
    answer: str


class DealRiskRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=100)
    transaction_id: str = Field(min_length=1, max_length=100)
    sector: str = Field(min_length=1, max_length=200)
    request: str = Field(min_length=1, max_length=500)
    applicability_override: ApplicabilityOverride | None = None


class OverallConclusionRequest(BaseModel):
    sector_trends: str = Field(min_length=1, max_length=6000)
    deal_risk_drivers: str = Field(min_length=1, max_length=6000)
