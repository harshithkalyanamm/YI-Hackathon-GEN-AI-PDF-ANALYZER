from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status

from .assessment_service import AssessmentService
from .config import Settings
from .document_store import LocalDocumentStore
from .model_manager import LocalDistilBERTEmbedder
from .pdf_processor import PDFProcessor
from .qa_engine import QueryEngine
from .retriever import LocalNumpyRetriever
from .schemas import AssessmentResponse, DealRiskRequest, DocumentType, OverallConclusionRequest, QueryRequest, QueryResponse, SectorTrendsRequest, UploadResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_engine(settings: Settings) -> QueryEngine:
    settings.ensure_directories()
    store = LocalDocumentStore(settings.uploads_dir, settings.indexes_dir)
    embedder = LocalDistilBERTEmbedder(settings.model_dir)
    return QueryEngine(
        PDFProcessor(settings.chunk_words, settings.chunk_overlap_words),
        LocalNumpyRetriever(store, embedder, settings.retrieval_strategy), store, settings.top_k,
    )


def create_app(settings: Settings | None = None, engine: QueryEngine | None = None) -> FastAPI:
    chosen_settings = settings or Settings.from_environment()
    query_engine = engine or build_engine(chosen_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load once at startup; this intentionally fails early when the approved local model was not provisioned.
        if engine is None:
            query_engine.retriever.embedder.load()
        yield

    app = FastAPI(title="Financial Risk AI", version="1.0.0", lifespan=lifespan)
    app.state.query_engine = query_engine
    app.state.assessment_service = AssessmentService(query_engine.store, query_engine)

    def get_engine() -> QueryEngine:
        return app.state.query_engine

    def get_assessment_service() -> AssessmentService:
        return app.state.assessment_service

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
    async def upload(file: UploadFile = File(...), active_engine: QueryEngine = Depends(get_engine),
                     document_type: DocumentType = Form(DocumentType.GENERAL),
                     transaction_id: str | None = Form(None)) -> UploadResponse:
        if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary:
            temp_path = Path(temporary.name)
            try:
                shutil.copyfileobj(file.file, temporary)
            finally:
                await file.close()
        try:
            document_id = active_engine.store.save_upload(
                temp_path, file.filename, document_type.value, transaction_id
            )
            active_engine.index_document(document_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            logger.exception("PDF processing failed")
            raise HTTPException(status_code=500, detail="The PDF could not be processed.")
        finally:
            temp_path.unlink(missing_ok=True)
        logger.info("PDF processed and indexed: %s", document_id)
        return UploadResponse(document_id=document_id, status="processed")

    @app.post("/query", response_model=QueryResponse)
    async def query(payload: QueryRequest, active_engine: QueryEngine = Depends(get_engine)) -> QueryResponse:
        logger.info("Query received for document id: %s", payload.document_id)
        answer = active_engine.answer(payload.document_id, payload.sector, payload.request)
        logger.info("Answer generated for document id: %s", payload.document_id)
        return QueryResponse(answer=answer)

    @app.post("/assessment/sector-trends", response_model=AssessmentResponse)
    async def sector_trends(payload: SectorTrendsRequest,
                            service: AssessmentService = Depends(get_assessment_service)) -> AssessmentResponse:
        try:
            return AssessmentResponse(answer=service.sector_trends(payload.document_id, payload.sector))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/assessment/deal-risk-drivers", response_model=AssessmentResponse)
    async def deal_risk_drivers(payload: DealRiskRequest,
                                service: AssessmentService = Depends(get_assessment_service)) -> AssessmentResponse:
        try:
            return AssessmentResponse(answer=service.deal_risk_drivers(
                payload.document_id, payload.transaction_id, payload.sector, payload.request,
                payload.applicability_override.value if payload.applicability_override else None,
            ))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/assessment/overall-conclusion", response_model=AssessmentResponse)
    async def overall_conclusion(payload: OverallConclusionRequest,
                                 service: AssessmentService = Depends(get_assessment_service)) -> AssessmentResponse:
        return AssessmentResponse(answer=service.overall_conclusion(
            payload.sector_trends, payload.deal_risk_drivers
        ))

    return app


app = create_app()
