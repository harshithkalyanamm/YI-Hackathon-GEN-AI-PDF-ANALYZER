from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    status: str


class QueryRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=100)
    sector: str = Field(min_length=1, max_length=200)
    request: str = Field(min_length=1, max_length=500)


class QueryResponse(BaseModel):
    answer: str
