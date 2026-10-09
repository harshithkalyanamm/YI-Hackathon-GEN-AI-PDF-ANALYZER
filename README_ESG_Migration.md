# ESG Document Intelligence - Migration Specification

## Purpose

This document defines the migration of the current local-only financial PDF intelligence proof of concept into a document-grounded ESG Credit Risk Assessment engine capable of processing structured 50-100 page PDFs and transaction-specific supporting documents.

The target system supports three outputs:

1. **Sector E&S trends**
2. **Deal's main E&S risk drivers and mitigants**
3. **Conclusion on overall E&S credit risk**

The design intentionally avoids unnecessary RAG infrastructure. Components should be added only when they solve a measured requirement.

---

## Hard Requirements

### Confidentiality

All confidential processing must remain inside the approved local execution boundary.

This includes:

- source PDFs and uploaded documents
- extracted text and tables
- chunks and metadata
- embeddings and vector indexes
- prompts and user inputs
- generated outputs
- logs and temporary files containing document content

Do not introduce cloud LLMs, cloud embeddings, cloud OCR, remote reranking, hosted vector databases, telemetry containing confidential content, or automatic runtime model downloads.

All models must be loaded from explicitly configured, pre-approved local filesystem paths.

### Knowledge Boundary

All ESG facts, risks, opportunities, mitigants, financial impacts, recommendations, and conclusions must be grounded exclusively in supplied internal documents.

Do not use:

- web search
- external ESG databases
- external reports or news
- general pretrained model knowledge to fill missing facts
- unsupported regulatory knowledge

When evidence is unavailable, the system must abstain or report insufficient information instead of guessing.

---

# 1. Current vs Target Migration

## API Layer

**Current:** FastAPI.

**Target:** Keep FastAPI.

**Reason:** The current API structure is already suitable for local uploads, indexing, retrieval, health checks, and future assessment endpoints. There is no need to replace it.

## PDF Parsing

**Current:** PyMuPDF / `fitz.get_text()` producing primarily plain text.

**Target:** Local Docling structured parsing.

**Fallback:** PyMuPDF may remain as a lightweight fallback for simple documents if useful.

**Reason:** The target PDFs contain headings, nested sections, tables, heatmaps, lists, and page-specific evidence. Plain text extraction loses too much document structure.

## OCR

**Current:** No explicit OCR pipeline.

**Target:** Do not add OCR by default.

If a document has a usable text layer, process it directly. Only invoke an approved local OCR path when text extraction is insufficient or the PDF is scanned.

**Reason:** Running OCR on every PDF adds cost and complexity without benefit for digitally generated documents.

## Document Representation

**Current:** Plain extracted text.

**Target:** Structured document objects retaining:

- page
- heading hierarchy
- section
- paragraph
- list
- table
- table headers
- captions where available
- sector metadata
- source-document metadata

## Heading Detection

**Current:** Custom `_is_heading()` / `_is_generic_heading()` heuristics.

**Target:** Prefer parser-derived document hierarchy plus deterministic sector mapping.

**Fallback:** Retain heuristics only where structured parsing fails.

## Sector Recognition

**Current:** Sector inferred from text/headings.

**Target:** Deterministic mapping from transaction classification to the internal BES sector.

Example conceptual flow:

```text
NACE / SG Activity
        |
        v
Approved Mapping
        |
        v
Internal BES Sector
```

If a mapping cannot be established confidently, flag it for analyst review. Do not ask an LLM to freely guess.

## Chunking

**Current:** Fixed word chunks with word overlap.

**Target:** Structure-aware chunking with tokenizer-aware maximum sizes.

**Candidates to benchmark:**

- Docling HybridChunker
- custom heading/section-aware chunker
- current word chunker as the baseline

**Reason:** Fixed word windows may mix unrelated sections, risk categories, mitigants, and table material.

Prefer semantic boundaries first, token limits second.

A reasonable initial experimental range is a few hundred model tokens per narrative chunk, but chunk size must be measured against the chosen local retrieval model and the actual documents rather than hard-coded as a universal value.

## Table Handling

**Current:** Tables may be flattened into text.

**Target:** Treat tables as structured evidence.

Benchmark representations such as:

- Markdown-like table representation
- header-aware row chunks
- whole-table representation for small tables

Requirements:

- preserve column headers
- preserve row/header relationship
- preserve page
- preserve surrounding section/heading
- preserve source document
- avoid splitting a table in a way that loses column meaning

Do not assume any parser configuration handles every BES table correctly. Test against the actual tables and heatmaps.

## Embedding Framework

**Current:** `transformers + torch`.

**Target:** Keep `transformers + torch`.

**Explicitly exclude:** Sentence Transformers.

There is no requirement to add another embedding framework.

## Embedding Model

**Current:** Generic DistilBERT with manual mean pooling and L2 normalization.

**Target:** Replace with an internally approved retrieval-trained encoder loaded directly through `transformers` from a local path.

Benchmark two or three approved retrieval encoders rather than selecting a model by popularity alone.

Determine pooling and query/passage formatting from the selected model's approved configuration.

Do not assume generic mean pooling is correct for every encoder.

## Model Loading

**Current:** Local model directory.

**Target:** Keep and strengthen this approach.

Requirements:

- model path comes from configuration
- local files only
- no runtime download
- ideally enforce outbound network blocking during security testing
- fail fast if required model artifacts are absent

## Vector Storage

**Current:** `vectors.npy`.

**Target:** Keep `vectors.npy` initially.

It is simple, portable, inspectable, and adequate for the expected corpus size.

## Vector Index

**Current:** FAISS when available, NumPy fallback.

**Target:** Keep both.

Use exact FAISS `IndexFlatIP` initially with normalized embeddings.

Do not introduce ANN merely because it is common in large RAG deployments.

### Test later only if scale requires it

- FAISS HNSW
- FAISS IVF

A vector database is not required for a corpus containing hundreds or thousands of chunks.

## Vector Database

**Current:** None.

**Target:** None for the MVP.

Do not initially introduce:

- Pinecone
- Qdrant
- Milvus
- Weaviate
- pgvector
- Elasticsearch solely for vector search

Reconsider only if operational requirements become substantially larger and FAISS/local metadata genuinely become limiting.

## Lexical Retrieval

**Current:** Custom token normalization and lexical bonuses.

**Target:** Add or implement proper local BM25-style lexical retrieval.

Why it matters:

Internal terms such as the following may need strong exact-term behavior:

- NACE codes
- CCVI / vulnerability indicators
- CORESP
- CTRC
- SFT
- AMH thresholds
- policy identifiers
- exact financial or policy terminology

Dense retrieval and lexical retrieval solve different problems.

## Hybrid Retrieval

**Current:** Dense retrieval plus custom lexical bonus.

**Target:** Benchmark a proper hybrid pipeline:

```text
BM25 retrieval
     +
Dense FAISS retrieval
     |
     v
Result fusion
```

Candidate fusion methods:

- Reciprocal Rank Fusion (RRF)
- normalized weighted score fusion

Do not assume RRF must be used. Select it only if evaluation shows an improvement.

## Metadata

**Current:** Sector and section metadata.

**Target:** Expand chunk metadata to include at least:

- chunk ID
- document ID
- filename
- document type
- transaction ID where applicable
- page start/end
- mapped sector
- NACE/activity where available
- heading path
- element type
- evidence scope
- original text
- embedding text

## Metadata Filtering

**Current:** Some filtering during/after retrieval.

**Target:** Apply hard filters before candidate retrieval wherever possible.

Examples:

```text
Box 1:
source = Primary BES
sector = mapped BES sector
```

```text
Box 2:
source in {relevant BES sector context, current transaction documents}
transaction = current transaction only
```

Metadata filtering is a correctness and security boundary, not merely a relevance optimization.

## Reranking

**Current:** None.

**Target:** Not required in the first implementation.

**Test later:** An internally approved local cross-encoder reranker may be benchmarked after hybrid retrieval is working.

Only keep the reranker if it materially improves retrieval metrics enough to justify its RAM and latency.

## HyDE

**Target:** Explicitly exclude.

Reason: HyDE creates a hypothetical answer before retrieval. That hypothetical content may contain knowledge not present in supplied internal documents and may influence retrieval. This conflicts with the strict document-only knowledge boundary.

## Query Expansion

Do not use generic LLM-driven query expansion initially.

If controlled expansion is eventually needed, expansions must come only from an approved internal glossary/taxonomy/document-derived synonym mapping.

## Evidence Representation

**Current:** Retrieved chunks/sentences become the answer.

**Target:** Introduce a structured evidence layer before writing.

Example:

```json
{
  "finding_id": "F001",
  "finding_type": "RISK",
  "scope": "SECTOR",
  "category": "CLIMATE_TRANSITION",
  "statement": "Document-grounded finding",
  "financial_channels": [],
  "source": {
    "document_id": "BES_2026",
    "filename": "2026 - BES - Memo April 2026.pdf",
    "page": 31,
    "sector": "SHIPPING_CRUISE",
    "heading_path": [
      "Shipping & Cruise",
      "Climate",
      "Transition Risk"
    ]
  }
}
```

For transaction evidence:

```json
{
  "finding_id": "F102",
  "finding_type": "MITIGANT",
  "scope": "DEAL",
  "statement": "Document-supported deal-specific finding",
  "source": {
    "document_id": "TRANSACTION_001_DOC_02",
    "page": 17
  }
}
```

The `scope` field is critical.

A sector-level finding must never be presented as something implemented by a specific borrower/deal unless transaction-specific evidence supports that claim.

## Provenance

Every material finding should retain:

- document
- page
- section/heading
- relevant source text
- sector
- scope
- evidence type/classification

The frontend may show clean prose, but the backend must retain evidence for audit, verification, debugging, and future "View Evidence" functionality.

## Final Writing

**Current:** Top sentences are concatenated/extractively assembled.

**Target:** Structured evidence -> validated findings -> controlled synthesis.

Two final-writing modes should be supported architecturally:

### Baseline: deterministic templates

Implement this first.

Benefits:

- very low hallucination risk
- minimal RAM
- highly auditable
- no generative model dependency
- safe fallback if the local LLM is unavailable

### Enhanced: approved local generative LLM

Add only after retrieval/evidence/template generation is stable.

The local LLM must receive validated evidence, not arbitrary vector search results.

The LLM acts as an evidence-to-prose writer, not as an ESG knowledge source.

Possible local runtime to evaluate on constrained machines:

- llama.cpp with an approved locally staged quantized GGUF model

Another possibility is an approved causal language model through the existing `transformers + torch` stack.

Do not automatically download models at runtime.

---

# 2. Current Flow

```text
PDF Upload
    |
    v
PyMuPDF
    |
    v
Plain Text
    |
    v
Whitespace Normalization
    |
    v
Custom Heading Detection
    |
    v
Sector / Section Metadata
    |
    v
Fixed Word Chunks + Overlap
    |
    v
Generic DistilBERT
    |
    v
Manual Mean Pooling
    |
    v
L2-Normalized Vectors
    |
    +------> vectors.npy
    |
    +------> FAISS IndexFlatIP
                  |
                  v
               Query
                  |
                  v
        DistilBERT Query Vector
                  |
                  v
          Vector Retrieval
                  |
                  v
     Sector / Lexical Heuristics
                  |
                  v
              Top Chunks
                  |
                  v
      Sentence Token-Overlap Rank
                  |
                  v
          Extractive Answer
```

This is acceptable for a small 20-line demonstration PDF but is insufficient as the final architecture for long structured ESG documents.

---

# 3. New Target Flow

## Ingestion and indexing

```text
                 Documents
                     |
        +------------+------------+
        |                         |
     Main BES                 Deal PDFs
        |                         |
        +------------+------------+
                     |
                     v
              Local Docling
                     |
                     v
            Structured Document
                     |
        +------------+------------+
        |            |            |
      Text         Tables       Metadata
                                page
                                heading
                                hierarchy
        |            |            |
        +------------+------------+
                     |
                     v
            Structural Chunking
             + Token Limits
                     |
                     v
          Context-Enriched Chunks
                     |
                     v
       Approved Local Retrieval Encoder
         transformers + torch only
                     |
                     v
            Normalized Embeddings
                     |
             +-------+-------+
             |               |
          vectors.npy     FAISS Flat
```

## Retrieval

```text
Assessment Information Need
          |
          v
     Hard Metadata Filter
          |
          v
     Eligible Chunks Only
          |
     +----+----+
     |         |
     v         v
   BM25      Dense
             FAISS
     |         |
     +----+----+
          |
          v
   Candidate Fusion
 (only if evaluation helps)
          |
          v
    Top Candidates
          |
          v
 Optional Local Reranker
 (only if proven useful)
          |
          v
   Retrieved Evidence
          |
          v
 Evidence Extraction
          |
          v
 Scope / Source Validation
          |
          v
 Structured Findings
```

## Generation

```text
Structured Findings
       |
       v
Evidence Validator
       |
   +---+---+
   |       |
   v       v
Template  Optional Approved
Writer    Local LLM
   |       |
   +---+---+
       |
       v
Output/Claim Validation
       |
       v
Final Textbox Draft
       |
       v
Analyst Review
```

---

# 4. Three Business Pipelines

## Box 1 - Sector E&S Trends

### Source boundary

Use only the designated primary internal BES document.

Transaction-specific uploaded documents must not introduce facts into Box 1.

### Flow

```text
Transaction Classification
         |
         v
NACE / SG Activity
         |
         v
Deterministic Internal Sector Mapping
         |
         v
Mapped BES Sector
         |
         v
Hard Filter:
source = BES only
sector = mapped sector
         |
         v
Retrieve Relevant Sector Evidence
         |
         v
Structured Sector Findings
         |
         v
Box 1 Synthesis
```

### Rules

- Only use information contained in the primary approved BES document.
- Do not introduce general ESG knowledge.
- Do not use transaction uploads as Box 1 factual sources.
- Do not represent sector-level mitigants as borrower/deal actions.
- Preserve supporting document/page/section for each material statement.

---

## Box 2 - Deal's Main E&S Risk Drivers and Mitigants

### Sources

Box 2 may use:

1. relevant sector context from the approved BES document
2. structured transaction information
3. supporting documents uploaded for the current transaction

Do not access documents belonging to other transactions.

### Applicability engine

Before retrieval/generation, evaluate applicable subtopics.

Current known guidance topics include:

- vulnerability indicators / xxVI
- E&S Sector Policies
- portfolio alignment / ex-NZBA
- CTRC escalation and/or CORESP commitment breaches
- E&S vigilance list / controversies
- independent E&S expert report or ESIA
- physical risk assessment
- SFT alignment

Represent applicability using distinct states:

```text
NOT_APPLICABLE
APPLICABLE_EVIDENCE_FOUND
APPLICABLE_EVIDENCE_NOT_FOUND
APPLICABILITY_UNKNOWN
```

Do not collapse all unavailable information into "N/A".

### Deterministic business rules

Where a business rule is explicitly known, implement it as application logic rather than relying on an LLM to remember it.

Example from the supplied application guidance:

```text
If AMH > 50 MEUR and the applicable policy requires priority-criteria verification,
trigger that verification requirement deterministically.
```

All policy rules must come from approved internal guidance.

### Financial transmission channels

For a supported risk, assess only financial channels supported by documentary evidence.

Possible channels identified in the application guidance include:

- revenue
- costs
- assets
- liabilities
- cash flow
- CAPEX
- financial charges
- collateral value

Use this logic:

```text
E&S Risk
   |
   v
Deal Exposure
   |
   v
Documented Financial Channel?
   |
  / \
YES  NO
 |    |
 v    v
Use  Do Not Infer
```

Plausibility is not sufficient evidence.

---

## Box 3 - Conclusion on Overall E&S Credit Risk

Box 3 is a synthesis layer.

### Inputs

- validated Box 1 findings
- validated Box 2 findings
- associated documentary evidence

### Flow

```text
Validated Box 1 Findings
          \
           \
            +--> Forward-Looking Synthesis --> Box 3
           /
          /
Validated Box 2 Findings
```

### Rules

- Do not perform fresh external ESG research.
- Do not introduce a factual claim that was absent from validated Box 1/Box 2 evidence.
- If evidence is insufficient to support a conclusion, state the limitation rather than fabricating certainty.

---

# 5. Storage and Isolation

Recommended structure:

```text
data/
|
+-- knowledge/
|   +-- bes_2026/
|       +-- source/
|       +-- structured_document/
|       +-- chunks.json
|       +-- vectors.npy
|       +-- index.faiss
|
+-- transactions/
    +-- TRANSACTION_001/
    |   +-- documents/
    |   +-- structured_documents/
    |   +-- chunks.json
    |   +-- vectors.npy
    |   +-- index.faiss
    |
    +-- TRANSACTION_002/
        +-- ...
```

Hard retrieval boundaries:

```text
BOX 1
-> BES knowledge only

BOX 2
-> relevant BES sector context
   + current transaction only

BOX 3
-> validated evidence from Box 1 and Box 2
```

Never permit one transaction to retrieve another transaction's evidence.

---

# 6. Suggested Chunk Schema

```json
{
  "chunk_id": "BES_2026_0031_004",
  "document_id": "BES_2026",
  "document_type": "PRIMARY_BES",
  "transaction_id": null,
  "page_start": 31,
  "page_end": 31,
  "sector": "SHIPPING_CRUISE",
  "heading_path": [
    "Shipping & Cruise",
    "Climate",
    "Transition Risk"
  ],
  "element_type": "paragraph",
  "evidence_scope": "SECTOR",
  "text": "...",
  "embedding_text": "Shipping & Cruise\nClimate\nTransition Risk\n..."
}
```

Transaction document chunks should include the current `transaction_id` and generally carry `evidence_scope = DEAL` only when the underlying content genuinely supports a deal-level statement.

---

# 7. What Must Be Included

The MVP must include:

- local/offline execution
- structured PDF parsing
- page-level provenance
- heading/section hierarchy
- table-aware extraction
- deterministic NACE/activity -> internal sector mapping
- structure-aware/token-aware chunking
- approved local retrieval encoder through `transformers + torch`
- FAISS exact search
- NumPy fallback
- lexical/BM25-style retrieval
- metadata filtering
- source isolation
- per-transaction isolation
- evidence scope: sector vs deal/client as applicable
- structured evidence objects
- Box 2 applicability rules
- insufficient-evidence handling
- deterministic template writer
- analyst review
- evaluation dataset
- regression tests

---

# 8. What Should Be Benchmarked

Do not prematurely choose "best" technologies without an internal evaluation.

## Retrieval Encoder

Benchmark two or three internally approved retrieval-trained encoders using direct `transformers + torch`.

Evaluate:

- Recall@5
- Recall@10
- MRR
- wrong-sector retrieval rate
- wrong-document retrieval rate
- not-found accuracy
- RAM
- latency

## Chunking

Compare:

1. current word chunking
2. Docling HybridChunker
3. custom structure-aware chunker

Check especially:

- heading preservation
- page provenance
- table preservation
- sector boundaries
- retrieval accuracy

## Retrieval Strategy

Compare:

1. dense only
2. BM25 only
3. dense + BM25
4. dense + BM25 + RRF
5. dense + BM25 + alternative score fusion

Select based on measured results.

## Reranking

Only after baseline retrieval is working, compare:

```text
Hybrid Retrieval
```

against:

```text
Hybrid Retrieval
      |
      v
Approved Local Reranker
```

Do not add the reranker unless the measured gain justifies another model and its RAM/latency.

## Final Writing

Compare:

- template-only output
- template + approved small local LLM

The template-only path must remain the safe fallback.

---

# 9. What Should Be Excluded

Do not include the following in the initial architecture:

- Sentence Transformers
- Pinecone
- Qdrant
- Milvus
- Weaviate
- pgvector
- Elasticsearch solely for this use case
- FAISS HNSW initially
- FAISS IVF initially
- HyDE
- generic LLM-driven query expansion
- LangChain
- LlamaIndex
- agents
- knowledge graph
- cloud LLMs
- cloud embeddings
- cloud reranking
- cloud OCR
- automatic web lookup
- external ESG enrichment
- cross-transaction retrieval
- automatic runtime model downloads

These components should not be added merely because they are common in generic RAG systems.

---

# 10. Avoid Unnecessary Processing

Not every document needs every processing step.

## OCR decision

```text
PDF
 |
 v
Usable text layer?
 |
 +-- YES -> use text layer, no OCR
 |
 +-- NO  -> approved local OCR path
```

## Table decision

```text
Relevant tables present?
 |
 +-- YES -> structured table processing
 |
 +-- NO  -> standard text/section processing
```

## Applicability decision

```text
Box 2 subtopic applicable?
 |
 +-- YES -> retrieve supporting evidence
 |
 +-- NO  -> do not run unnecessary retrieval/inference
```

---

# 11. Existing File Migration Plan

## `backend/config.py`

Modify.

Add configuration for:

- local parser/model paths
- retrieval encoder path
- chunk/token parameters
- hybrid-search settings
- optional reranker path, disabled by default
- knowledge data root
- transaction data root
- strict offline flags

## `backend/document_store.py`

Modify.

Add separation between:

- approved global/internal knowledge
- transaction-specific documents/indexes

Enforce transaction-scoped storage and retrieval.

## `backend/model_manager.py`

Major modification.

Remove the assumption that the embedding model is generic DistilBERT.

Support:

- approved retrieval encoder
- direct `transformers + torch`
- local filesystem loading only
- model-specific pooling
- model-specific query/passage prefixing if required by the selected approved model
- normalized float32 embeddings when appropriate

Do not add Sentence Transformers.

## `backend/pdf_processor.py`

Major refactor/replacement.

Responsibilities should become:

- local Docling conversion
- page provenance
- heading hierarchy
- structured tables
- section-aware chunking
- token-aware limits
- structured chunk metadata

Keep PyMuPDF only as an optional fallback if desired.

## `backend/retriever.py`

Major modification.

Add:

- hard metadata filters
- dense FAISS retrieval
- local lexical/BM25 retrieval
- candidate fusion
- richer retrieval result objects
- evidence metadata
- optional reranker interface disabled by default

## `backend/qa_engine.py`

Refactor.

The final system is not generic PDF Q&A.

Move toward:

- evidence retrieval service
- evidence extraction/validation
- Box 1 orchestration
- Box 2 orchestration
- Box 3 orchestration

The generic query endpoint may remain for diagnostics if useful, but should not be the primary business abstraction.

## `backend/schemas.py`

Extend.

Add schemas for:

- transaction metadata
- sector mapping result
- chunk/evidence provenance
- applicability states
- evidence findings
- Box 1 response
- Box 2 response
- Box 3 response
- complete assessment response

## `backend/app.py`

Modify.

Potential endpoints may include:

```text
POST /documents/upload
POST /assessment/sector-trends
POST /assessment/risk-drivers
POST /assessment/conclusion
POST /assessment/full
GET  /health
```

Exact endpoint design may follow the existing application's integration needs.

## `demo.py`

Expand or replace.

The demo should test more realistic structure:

- multiple pages
- nested headings
- at least one table
- multiple sectors or sections
- one transaction document
- one unanswerable information request

## Tests

Expand significantly.

Add tests for:

- sector mapping
- Box 1 BES-only isolation
- transaction isolation
- cross-transaction leakage prevention
- page provenance
- heading hierarchy
- table structure
- exact-term retrieval
- semantic retrieval
- hybrid retrieval
- insufficient evidence
- not applicable vs missing evidence
- sector-vs-deal scope handling
- Box 3 no-new-facts rule
- offline model loading failure

---

# 12. Evaluation Dataset

Create a manually validated golden dataset before selecting retrieval components.

Start with approximately 30-50 test cases against the supplied BES document, then add transaction-specific cases.

Example:

```json
{
  "id": "T001",
  "task": "BOX1",
  "sector": "SHIPPING_CRUISE",
  "query": "internal test question",
  "expected_document": "BES_2026",
  "expected_pages": [31, 32],
  "expected_heading": "Physical Risk",
  "must_not_retrieve_sectors": [
    "Oil & Gas",
    "Automotive"
  ],
  "should_find_evidence": true
}
```

Include negative/unanswerable cases.

Metrics should include:

- Recall@5
- Recall@10
- MRR
- wrong-sector rate
- wrong-transaction rate
- source/page correctness
- table retrieval accuracy
- answerable evidence-found rate
- unanswerable correct-abstention rate

Technology choices should be made from this evaluation rather than external benchmark popularity alone.

---

# 13. Limited-RAM Laptop Target

Recommended development configuration:

```text
Parser:
Local Docling

OCR:
Only when required

Embedding Framework:
transformers + torch

Embedding Model:
Small approved retrieval-trained encoder

Vector Search:
FAISS IndexFlatIP

Fallback:
NumPy dot product

Lexical Retrieval:
Local BM25-style implementation

Reranker:
Initially disabled

Final Writing:
Deterministic templates first

Optional LLM:
Small approved quantized local model only if RAM permits
```

Pre-process the primary BES document once and persist its structured representation, chunks, embeddings, and FAISS index.

Do not re-run the parser over the entire BES document for every assessment query.

---

# 14. Higher-Resource Approved Environment

If organizational policy later permits processing within an approved higher-resource environment, keep the same logical architecture.

Potential differences:

- larger approved retrieval encoder can be benchmarked
- larger batch sizes
- optional local reranker becomes more practical
- larger approved local generative model can be used
- higher concurrency

Do not automatically replace FAISS with a vector database merely because more RAM is available.

The authorization boundary must be explicitly approved by organizational security/governance before confidential data leaves the office laptop.

---

# 15. Recommended Implementation Order

## Phase 1 - Document Structure

Replace plain-text-first processing with structured parsing.

Deliverables:

- Docling integration
- page provenance
- headings
- tables
- structured document output

## Phase 2 - Chunking

Implement and benchmark:

- current word chunks baseline
- Docling HybridChunker
- custom structural chunker if needed

## Phase 3 - Retrieval Encoder

Replace generic DistilBERT with approved retrieval-trained encoders through direct `transformers + torch`.

Benchmark candidates on the golden dataset.

## Phase 4 - Retrieval

Implement:

- hard metadata filtering
- FAISS dense retrieval
- lexical/BM25 retrieval
- optional fusion

## Phase 5 - Evidence Layer

Introduce:

- structured findings
- provenance
- evidence scope
- evidence validation
- insufficient-evidence behavior

## Phase 6 - Business Pipelines

Implement:

- Box 1 sector pipeline
- Box 2 applicability and deal-risk pipeline
- Box 3 synthesis pipeline

## Phase 7 - Template Generation

Create deterministic output generation for all three boxes.

The system should now be fully usable without a generative LLM.

## Phase 8 - Optional Local LLM

Only if better natural language is needed:

- benchmark approved local generative model/runtime
- feed only validated structured findings
- validate resulting claims against evidence
- retain templates as fallback

## Phase 9 - Optional Reranker

Only if retrieval evaluation shows a meaningful need.

## Phase 10 - Scale Optimizations

Only if measurements justify them:

- HNSW
- IVF
- local vector database

Do not perform Phase 10 prematurely.

---

# 16. Code Generation Instructions

When implementing this migration, follow these rules:

1. Do not rewrite components that do not need changing.
2. Preserve FastAPI, the DocumentStore abstraction, FAISS, NumPy fallback, Pydantic models, and existing testing patterns where practical.
3. Do not introduce Sentence Transformers.
4. Do not introduce a vector database for the MVP.
5. Do not introduce HNSW or IVF initially.
6. Do not introduce HyDE.
7. Do not introduce LangChain or LlamaIndex.
8. Do not introduce external/cloud APIs.
9. Do not add a local LLM until parsing, retrieval, evidence handling, and template output work correctly.
10. Load all models from explicitly configured local filesystem paths.
11. Do not download model artifacts at runtime.
12. Do not transmit confidential content outside the approved local execution boundary.
13. ESG facts must come exclusively from supplied internal documents.
14. Never supplement missing evidence with pretrained model knowledge.
15. Box 1 may retrieve only from the designated primary BES document.
16. Box 2 may retrieve only from the relevant BES sector context plus documents belonging to the current transaction.
17. Box 3 may synthesize validated Box 1 and Box 2 findings but may not introduce new factual claims.
18. Never convert sector-level evidence into deal-level evidence without transaction-specific support.
19. Preserve separate states for `NOT_APPLICABLE`, `APPLICABLE_EVIDENCE_FOUND`, `APPLICABLE_EVIDENCE_NOT_FOUND`, and `APPLICABILITY_UNKNOWN`.
20. Preserve document ID, filename, page, heading/section, evidence scope, and supporting text for every material finding.
21. Add components only when they solve an identified requirement or improve measured evaluation results.
22. Implement incrementally: parser -> chunking -> embedding -> retrieval -> evidence -> business pipelines -> templates -> optional generation.
23. Run the golden evaluation suite after each major migration stage and compare against the previous baseline.
24. Optimize for grounded accuracy, source traceability, safe abstention, zero sector/deal leakage, and laptop RAM usage before retrieval latency.
25. Keep business rules deterministic whenever the source guidance defines deterministic applicability logic.
26. Make failure safe: ambiguous sector mapping, insufficient evidence, unreadable documents, conflicting sources, and low-confidence extraction must be surfaced for analyst review rather than guessed.

---

# 17. Final Target Architecture

```text
                 ESG Assessment Input
                         |
                         v
                  Sector / NACE Data
                         |
                         v
               Deterministic Mapping
                         |
             +-----------+-----------+
             |                       |
             v                       v
          BES PDF                 Deal PDFs
             |                       |
             +-----------+-----------+
                         |
                         v
                  Local Docling
                         |
                         v
                Structured Documents
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
        Text           Tables        Metadata
                                      page
                                      heading
                                      hierarchy
                         |
                         v
                Structural Chunking
                         |
                         v
             Local Retrieval Encoder
             transformers + torch
                         |
                         v
                    FAISS Flat
                         |
              +----------+----------+
              |                     |
              v                     v
            BM25               Dense Retrieval
              |                     |
              +----------+----------+
                         |
                         v
                 Fusion If Useful
                         |
                         v
             Optional Local Reranker
              Only If Proven Useful
                         |
                         v
                 Retrieved Evidence
                         |
                         v
                Evidence Extraction
                         |
                         v
                 Scope Validation
               Sector / Client / Deal
                         |
                         v
                 Structured Findings
                         |
          +--------------+---------------+
          |              |               |
          v              v               v
        Box 1          Box 2           Box 3
       Sector       Deal Risks /      Overall
       Trends        Mitigants       Conclusion
          |              |               |
          +--------------+---------------+
                         |
                         v
              Deterministic Templates
                    Safe Baseline
                         |
                         v
          Optional Approved Local LLM
                  Enhanced Writing
                         |
                         v
                Evidence Validation
                         |
                         v
                    Analyst Review
```

---

# 18. Final Design Principle

The target system is not:

```text
Upload PDFs -> Vector DB -> LLM -> Answer
```

The target system is:

```text
Document Intelligence
        +
Controlled Sector Mapping
        +
Local Retrieval
        +
Deterministic Applicability Rules
        +
Document-Grounded Evidence Extraction
        +
Financial Transmission Analysis
        +
Source/Scope Validation
        +
Controlled Writing
        +
Human Review
```

Reliability, confidentiality, evidence traceability, and correct abstention take priority over architectural complexity.
