flowchart LR
%% =========================
%% Client + API Entry
%% =========================
U[Client App / Analyst UI] -->|POST /upload| API
U -->|POST /query| API
U -->|POST /assessment/sector-trends| API
U -->|POST /assessment/deal-risk-drivers| API
U -->|POST /assessment/overall-conclusion| API
U -->|GET /health| API

    subgraph SVC[FastAPI Service Layer]
        API[backend/app.py]
        ENG[build_engine:\nPDFProcessor + LocalNumpyRetriever + LocalDocumentStore + QueryEngine]
        AS[AssessmentService]
    end
    API --> ENG
    API --> AS

    %% =========================
    %% Upload + Index Pipeline
    %% =========================
    API -->|/upload| SAVE[DocumentStore.save_upload]
    SAVE --> UP[(data/uploads/{document_id}.pdf)]
    SAVE --> META[(data/indexes/{document_id}/document.json)]

    SAVE --> IDX[QueryEngine.index_document]
    IDX --> PARSE[PDFProcessor.parse_document\n(Active: PyMuPDF)]
    DOCLING[Optional Parser: Docling] -. benchmark-gated future path .-> PARSE

    PARSE --> STRUCT[(structured_document.json)]
    PARSE --> CHUNK[chunk_structured_document\nword chunks + heading path + sector metadata]
    CHUNK --> EMB[LocalDistilBERTEmbedder.embed]
    EMB --> VEC[(vectors.npy)]
    CHUNK --> CJSON[(chunks.json)]

    %% =========================
    %% Query Flow
    %% =========================
    API -->|/query| Q[QueryEngine.answer]
    Q --> RET[LocalNumpyRetriever.search]
    RET --> STRAT{Strategy}
    STRAT -->|dense| D1[Cosine on local vectors]
    STRAT -->|bm25| D2[Term scoring]
    STRAT -->|hybrid| D3[0.65 dense + 0.35 bm25]

    D1 --> FILT[Hard sector boundary filter\n+ required term checks]
    D2 --> FILT
    D3 --> FILT

    FILT --> EV1[EvidenceExtractor.extract\nsentence-level findings]
    EV1 --> EV2[EvidenceExtractor.validate\nsame document + sector scope + page validity]
    EV2 --> OUTQ[Deterministic response\n(top findings joined)]
    OUTQ --> U

    %% =========================
    %% Assessment Flows
    %% =========================
    API -->|/assessment/sector-trends| ST[AssessmentService.sector_trends]
    ST --> G1{document_type == PRIMARY_BES?}
    G1 -->|No| E1[422 validation error]
    G1 -->|Yes| RT1[retrieve_evidence for:\nsector risk, bank risk,\nclimate transition risk, nature transition risk]
    RT1 --> T1[TemplateWriter.sector_trends]
    T1 --> U

    API -->|/assessment/deal-risk-drivers| DR[AssessmentService.deal_risk_drivers]
    DR --> G2{document_type == DEAL_DOCUMENT?}
    G2 -->|No| E2[422 validation error]
    G2 -->|Yes| G3{transaction_id matches?}
    G3 -->|No| E3[422 validation error]
    G3 -->|Yes| APPL{applicability_override}
    APPL -->|NOT_APPLICABLE| T2[TemplateWriter.deal_risk\nNOT_APPLICABLE]
    APPL -->|APPLICABLE or None| RT2[retrieve_evidence for request]
    RT2 --> T3[TemplateWriter.deal_risk\nAPPLICABLE_EVIDENCE_FOUND /\nAPPLICABLE_EVIDENCE_NOT_FOUND /\nAPPLICABILITY_UNKNOWN]
    T2 --> U
    T3 --> U

    API -->|/assessment/overall-conclusion| OC[AssessmentService.overall_conclusion]
    OC --> T4[TemplateWriter.overall_conclusion\nDeterministic synthesis only]
    T4 --> U

    %% =========================
    %% Ops / Validation Side Flows
    %% =========================
    PF[preflight.py] --> C1[check_transformer_model]
    PF --> C2[check_docling optional]
    B1[evaluate.py] --> EVR[evaluate_retriever]
    B2[benchmark_models.py] --> EVR




Create a professional system architecture diagram (16:9, dark tech theme) titled:
“AI-Powered ESG Insights Engine – System Architecture (Code-Accurate)”.

Style:
- Clean enterprise architecture style, high readability.
- Rounded boxes, directional arrows, clear section boundaries.
- Use color-coded layers:
    - Blue: API/Service Layer
    - Purple: Document Processing + Indexing
    - Teal: Retrieval + Evidence Validation
    - Orange: Assessment Logic + Governance
    - Gray: Storage and Artifacts
    - Green dotted path: Optional Docling future path
- Include arrow labels and small legends.

Must include these exact flows:
1) Client -> FastAPI endpoints:
   /upload, /query, /assessment/sector-trends, /assessment/deal-risk-drivers, /assessment/overall-conclusion, /health
2) Upload pipeline:
   save_upload -> PDF in uploads + metadata json -> parse_document (PyMuPDF active) ->
   chunk_structured_document -> local DistilBERT embedding -> vectors.npy + chunks.json + structured_document.json
3) Query pipeline:
   retriever.search -> strategy branch (dense / bm25 / hybrid) ->
   sector boundary filter + required-term filter ->
   evidence extract -> evidence validate ->
   deterministic final answer
4) Assessment pipelines:
    - sector_trends: enforce PRIMARY_BES, gather 4 topic evidence groups, template output
    - deal_risk_drivers: enforce DEAL_DOCUMENT + transaction match + applicability override states
    - overall_conclusion: deterministic synthesis only
5) Optional future path:
   Docling parser is optional and benchmark-gated, shown as dotted arrow feeding parsing stage.
6) Ops side panel:
   preflight checks: local model readiness + optional docling check
   evaluation scripts: retrieval regression + model benchmark

Add a footer note:
“Current parser: PyMuPDF. Docling integration is optional and not active by default.”