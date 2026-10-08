Yes. Below is a code-generation handoff specification you can paste into another model. I am keeping it deliberately lean: improve the parts that are weak for 50 to 100-page structured PDFs, but do not turn this into an unnecessarily complex enterprise RAG platform.

Two hard constraints remain throughout: all processing stays local, and all ESG facts must come only from supplied internal documents.

1. Current vs Target Architecture
   Component	Current implementation	Recommended migration	Alternatives to benchmark	Why change / keep?	PriorityAPI	FastAPI	Keep FastAPI	None needed	Already suitable for upload, processing and assessment endpoints.	Keep
   PDF parsing	PyMuPDF fitz.get_text()	Docling local processing	Keep PyMuPDF only as fallback/simple-document parser	Current extraction flattens document structure. Main PDFs contain headings, tables and structured sector sections.	High
   OCR	None / dependent on PDF text layer	Do not automatically add OCR	Local Docling-supported OCR only if scanned documents require it	OCR is unnecessary overhead for digitally generated PDFs. Enable only after detecting insufficient/no text layer.	Conditional
   Document representation	Plain text	Structured document representation	Structured internal JSON derived from parsed document	Preserve page, headings, sections, tables and hierarchy.	High
   Heading identification	Custom _is_heading() heuristics	Parser structure + controlled mapping	Keep heuristic fallback	Font/layout-based document parsing is preferable to guessing headings from line length/capitalization.	High
   Sector recognition	Extracted heading text	Deterministic NACE/activity → BES-sector mapping	Analyst-confirmed mapping when mapping confidence is insufficient	Sector is a hard information boundary, not merely a search hint.	Critical
   Chunking	Fixed word chunks with overlap	Structure-aware + tokenizer-aware chunks	Compare Docling HybridChunker vs custom section-aware chunker	Fixed windows may mix risks, mitigants, sections and tables. Docling's HybridChunker uses hierarchical structure plus tokenizer-aware splitting/merging.	High
   Table processing	Flattened through PDF text	Preserve tables separately with headers + page/section metadata	Benchmark Markdown vs header-aware row serialization	Relationships between rows, columns and headers matter. Test actual BES tables because table serialization behavior can vary by configuration/version.	Critical
   Embedding framework	transformers + torch	Keep transformers + torch	None initially	User explicitly does not want Sentence Transformers. No need for another framework.
   Embedding model	Generic DistilBERT + mean pooling	Replace with approved retrieval-trained encoder loaded locally	Benchmark 2 to 3 internally approved encoder models	Generic DistilBERT mean pooling is not optimized specifically for query-passage retrieval.	Critical
   Model loading	Local model directory	Keep, but enforce offline/local paths	Network-disabled runtime verification	Local-path loading via Transformers remains suitable.	Critical
   Vector storage	vectors.npy	Keep	FAISS-only persistence later	Simple, transparent, local, adequate at current scale.
   Vector indexing	FAISS when available	Keep FAISS	NumPy fallback	No vector DB required at current scale.
   FAISS index	Exact inner product	Keep IndexFlatIP initially	HNSW only if benchmark proves necessary	Flat indexes provide exact search and are the FAISS baseline. ANN introduces unnecessary accuracy/speed trade-offs at this scale.
   Vector DB	None	Do not add	Evaluate only if corpus/operational requirements become much larger	Qdrant/Milvus/etc. solve problems that the current corpus does not have.	Exclude now
   Lexical search	Custom token overlap/bonuses	Add proper BM25-style lexical retrieval	Benchmark custom lexical vs BM25	Essential for exact terms such as NACE codes, CCVI, CORESP, SFT, AMH and named policy terminology.
   Semantic search	DistilBERT + FAISS	Retrieval-trained encoder + FAISS	Benchmark retrieval encoders	Semantic similarity remains useful.
   Hybrid retrieval	Ad-hoc lexical bonus	BM25 + dense retrieval	Compare dense-only vs lexical-only vs hybrid	Exact identifiers and semantic concepts require different retrieval mechanisms.
   Fusion	Custom scoring	RRF is a candidate, benchmark it	Weighted normalized score fusion	Do not add RRF merely because it is common. Prove improvement using the evaluation set.
   Metadata	Sector + section	Expand substantially	N/A	Need document, page, sector, heading path, transaction, document type, evidence scope, element type.	Critical
   Metadata filtering	Sector filtering after retrieval	Hard filtering before retrieval wherever possible	N/A	Prevents irrelevant sectors/documents from entering candidate set.	Critical
   Reranker	None	Not required initially	Test approved local cross-encoder later	Retrieve-rerank architectures can improve relevance, but reranking adds another model and RAM/latency. Benchmarked later only.
   HyDE	None	Do not add	None	Hypothetical answers may inject unsupported model knowledge into retrieval. Violates project philosophy.	Exclude
   Generic query expansion	None	Do not add initially	Controlled glossary derived only from internal docs	External/model-generated expansion can contaminate document-only retrieval.	Exclude
   Evidence extraction	Top chunks → sentences	Add structured evidence objects	N/A	Retrieval and final writing must be separated.	Critical
   Provenance	Limited chunk metadata	Document + page + section + heading + scope	N/A	Required for auditability and analyst verification.	Critical
   Answering	Select/concatenate top sentences	Structured findings → controlled synthesis	Template-only vs local LLM	Three ESG textboxes require coherent synthesis rather than sentence concatenation.	Critical
   Final generation	Extractive	Templates baseline + optional approved local LLM	llama.cpp vs existing organization-approved inference runtime	Templates provide safe fallback; local LLM provides higher-quality prose.	Later
   LLM input	N/A	Validated structured evidence only	N/A	Do not feed arbitrary raw corpus directly into generator.	Critical
   Vector namespaces	Per document	BES knowledge namespace + per-transaction namespace	N/A	Prevent cross-deal leakage.	Critical
   Testing	Basic API/parser tests	Add retrieval + grounding evaluation suite	N/A	Must quantitatively compare migration choices.	Critical

A locally run retrieval system does not require a vector database. FAISS IndexFlatIP can perform exact inner-product retrieval and works for cosine-style retrieval when vectors are normalized. More complex structures like HNSW and IVF exist specifically to trade exactness/memory/search speed as scale grows.

2. What the current flow does

The existing system is approximately:

PDF Upload
│
▼
PyMuPDF
│
▼
Plain extracted text
│
▼
Whitespace normalization
│
▼
Custom heading detection
│
▼
Sector / section labels
│
▼
Fixed word chunks + overlap
│
▼
DistilBERT
│
▼
Manual mean pooling
│
▼
L2-normalized vectors
│
├────────► vectors.npy
│
└────────► FAISS IndexFlatIP
│
▼
User Query
│
▼
DistilBERT embedding
│
▼
Vector retrieval
│
▼
Sector / lexical rules
│
▼
Top chunks
│
▼
Sentence token overlap
│
▼
Top sentences
│
▼
Answer

This is fine for
20-line example PDF
simple headings
mainly narrative text
simple extractive question answering
small corpus

It starts breaking down when
50-100 pages
nested headings
tables
heatmaps
multiple sectors
multiple documents
similar ESG terminology across sectors
deal-specific evidence
page-level citations
conditional assessment rules
coherent final narratives

3. Target processing flow

I recommend the following.

                 DOCUMENT INPUT
                       │
          ┌────────────┴────────────┐
          │                         │
       Main BES                Deal Documents
          │                         │
          └────────────┬────────────┘
                       ▼
             LOCAL DOCUMENT PARSER
                    Docling
                       │
                       ▼
               Structured Document
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      Text           Tables        Metadata
                                  page / heading
                                  / hierarchy
        │              │              │
        └──────────────┼──────────────┘
                       ▼
             Structure-aware chunks
                    +
              token-aware limits
                       │
                       ▼
              Context enrichment
       heading + section + chunk content
                       │
                       ▼
         Approved LOCAL retrieval encoder
         using transformers + torch
                       │
                       ▼
             Normalized embeddings
                       │
                       ▼
                 FAISS Flat
                       │
                 vectors.npy
                       │
                    metadata


Docling's HybridChunker is specifically designed to start from document hierarchy and then make tokenizer-aware splitting/merging decisions. Its contextualization mechanism can also enrich chunks with structural context before embedding.

4. Target retrieval flow

Do not immediately jump from vectors to generation.

Use:

                    Retrieval Requirement
                             │
                             ▼
                       HARD FILTERS
                             │
                  ┌──────────┼──────────┐
                  │          │          │
                source     sector   transaction
                  │          │          │
                  └──────────┼──────────┘
                             ▼
                    Eligible chunks only
                             │
                ┌────────────┴─────────────┐
                │                          │
                ▼                          ▼
          Lexical retrieval          Dense retrieval
              BM25                       FAISS
                │                          │
                └────────────┬─────────────┘
                             ▼
                       Result fusion
                    RRF / weighted score
                       BENCHMARK BOTH
                             │
                             ▼
                     Top ~15-30 chunks
                             │
                             ▼
                  OPTIONAL local reranker
                only if benchmark improves
                       retrieval quality
                             │
                             ▼
                      Top ~5-10 pieces
                       of evidence
                             │
                             ▼
                    Evidence Extraction


The retrieve-then-rerank pattern is useful because a fast first-stage retriever identifies candidate passages and a more expensive joint query/passage model can rerank only that small set. But because reranking increases memory and latency, it should be included only if evaluation demonstrates a useful gain.

5. Structured Evidence Layer

This is one of the most important changes.

Do not directly pass vector-search results to the final writer.

Convert retrieved content into evidence objects first.

Example conceptual structure:

{
"finding_id": "F001",
"finding_type": "RISK",
"scope": "SECTOR",
"category": "CLIMATE_TRANSITION",

"statement": "Exact document-grounded finding",

"financial_channels": [],

"source": {
"document_id": "BES_2026",
"filename": "2026 - BES - Memo April 2026.pdf",
"page": 31,
"sector": "Shipping & Cruise",
"heading_path": [
"Shipping & Cruise",
"Climate",
"Transition Risk"
]
}
}


A transaction-specific finding could instead be:

{
"finding_id": "F102",
"finding_type": "MITIGANT",
"scope": "DEAL",

"statement": "Document-supported transaction-specific mitigant",

"source": {
"document_id": "TRANSACTION_004_DOC_02",
"page": 17
}
}


The distinction:

scope = SECTOR


versus:

scope = DEAL


is mandatory.

A sector finding must never silently become a borrower/deal fact.

6. Three separate generation pipelines

Do not implement one generic:

/query


as the final business interface.

The business workflow has three separate tasks.

BOX 1: Sector E&S Trends
Allowed source
PRIMARY BES DOCUMENT ONLY


Flow:

NACE / SG Activity
│
▼
Deterministic Sector Mapping
│
▼
Mapped BES Sector
│
▼
Filter:
source = BES
sector = mapped sector
│
▼
Retrieve / extract:
sector trends
risks
opportunities
other relevant BES information
│
▼
Structured Sector Findings
│
▼
BOX 1 synthesis


Hard rule:

BES sector statement
≠
deal/borrower statement


Additional transaction uploads must not influence Box 1.

7. BOX 2: Deal's main E&S risk drivers and mitigants

Allowed information:

Relevant BES sector context
+
Transaction metadata
+
Documents uploaded for this transaction


Before retrieval, run deterministic applicability rules.

                  Transaction
                       │
                       ▼
               Applicability Engine
                       │
       ┌───────────────┼────────────────┐
       ▼               ▼                ▼
Vulnerability      Sector-policy     Physical
indicators          checks            risk
│               │                │
Portfolio           CTRC /           ESIA /
alignment           CORESP          expert report
│               │                │
controversies       SFT              others
└───────────────┼────────────────┘
▼
Applicable information needs
│
▼
Evidence retrieval


The applicability engine must distinguish:

NOT_APPLICABLE

APPLICABLE_EVIDENCE_FOUND

APPLICABLE_EVIDENCE_NOT_FOUND

APPLICABILITY_UNKNOWN


Do not turn all missing data into "N/A".

8. Box 2 financial transmission

For each documentary risk:

E&S Risk Driver
│
▼
Transaction Exposure
│
▼
Possible Financial Channel
│
├── Revenue
├── Costs
├── Assets
├── Liabilities
├── Cash Flow
├── CAPEX
├── Financial Charges
└── Collateral Value
│
▼
Supporting documentary evidence?
│
/ \
YES  NO
│    │
▼    ▼
include  DO NOT INFER


Plausibility is not sufficient.

Example:

ESG risk found in document


does not permit the program to invent:

therefore CAPEX will increase


unless the supplied documents support that relationship.

9. BOX 3: Overall E&S Credit Risk Conclusion

Box 3 should require no new ESG research.

Input:

Validated Box 1 findings
+
Validated Box 2 findings
+
Document evidence
↓
Forward-looking synthesis
↓
BOX 3


Hard rule:

Box 3 may synthesize existing findings but cannot introduce a new factual claim.

If evidence is insufficient:

Insufficient information in the provided documents
to support a conclusion regarding [...]


is preferable to fabrication.

10. How vectors become final written solutions

The vector is only a retrieval mechanism.

Do not design:

vector
↓
LLM
↓
answer


Design:

Stored vectors
│
▼
Retrieve candidate chunks
│
▼
Evidence identification
│
▼
Scope classification
│
▼
Document/page verification
│
▼
Structured findings
│
▼
Evidence validator
│
▼
┌──┴─────────────────┐
│                    │
▼                    ▼
Template Writer     Local LLM Writer
Safe baseline       Optional enhanced
│                    │
└─────────┬──────────┘
▼
Output validation
│
▼
Final textbox

11. Final-writing options
    Approach	Laptop suitability	Natural prose	Hallucination exposure	RecommendationExisting extractive sentence concatenation	Excellent	Low	Very low	Replace as final product
    Deterministic templates	Excellent	Medium	Lowest	Required baseline
    Small local generative LLM	Good depending on model/RAM	High	Medium, requires validation	Benchmark later
    Larger local LLM	Poor/medium laptop	High	Medium	Higher-resource approved environment only
    Cloud LLM	Technically strong	High	Data leaves local boundary	Forbidden
    Baseline

Build deterministic templates first.

Example conceptual input:

{
"sector": "Mapped internal sector",
"risks": [
"document-supported risk A",
"document-supported risk B"
],
"opportunities": [
"document-supported opportunity A"
]
}


The template writer turns that structured evidence into controlled narrative.

This gives the project a working final-output mechanism without requiring an LLM at all.

12. Optional local LLM

If polished analyst-style prose is required later:

Validated structured findings
↓
Strict local generation prompt
↓
Approved local generative model
↓
Draft
↓
Claim/evidence validation
↓
Final

First runtime to evaluate for constrained laptop
llama.cpp
+
approved locally stored quantized GGUF model


Do not automatically download a model at runtime.

The model should ideally be staged as an approved artifact:

approved-models/
generator/
model.gguf


A second option is continuing with transformers + torch using an approved local causal language model, avoiding another Python ML framework.

Model selection must be benchmarked and security-approved rather than hard-coded into the architecture.

13. What MUST be implemented
    Capability	WhyStrict offline processing	Confidentiality requirement
    Main-BES vs transaction-document separation	Box 1 and Box 2 have different source boundaries
    Transaction namespaces	Prevent data leakage between deals
    Controlled sector mapping	Prevent retrieval from wrong sector
    Page-level provenance	Auditability
    Heading/section metadata	Accurate contextual retrieval
    Proper table preservation	BES and uploaded documents contain tables
    Structure-aware chunking	Prevent unrelated document sections being mixed
    Retrieval-trained embeddings	Improve semantic retrieval over generic DistilBERT
    Exact-term lexical retrieval	Capture internal acronyms, codes and threshold terms
    Hard metadata filtering	Reduce search space and enforce boundaries
    Structured evidence objects	Separate retrieval from writing
    Evidence scope	Prevent sector claims becoming deal claims
    Applicability logic	Required by Box 2 guidance
    Insufficient-evidence handling	Prevent hallucinations
    Deterministic template writer	Safe final-answer baseline
    Retrieval evaluation	Choose models/techniques based on evidence rather than popularity
    Analyst review	Final decision remains human
14. What should be benchmarked first

Do not blindly implement all options.

Embedding model

Benchmark 2 to 3 approved retrieval-trained encoder models using direct transformers + torch.

Do not use Sentence Transformers.

Measure:

Recall@5
Recall@10
MRR
wrong-sector retrieval
wrong-document retrieval
not-found accuracy

Chunking

Compare:

A. Current fixed word chunks

B. Docling HybridChunker

C. Custom:
heading hierarchy
+ paragraphs/tables
+ embedding-token maximum


Because the actual documents contain tables, explicitly test whether table headers and row/column relationships survive chunking. Current Docling documentation exposes table-header repetition for split tables, but actual serialization behavior should be tested with your specific Docling version and files.

Retrieval

Benchmark:

Dense only

vs

BM25 only

vs

Dense + BM25

vs

Dense + BM25 + RRF


Do not assume hybrid is better until the internal evaluation set proves it.

Reranker

Only after baseline retrieval exists:

Hybrid without reranker

vs

Hybrid + approved local reranker


If the accuracy improvement is small, do not add the reranker.

15. What should NOT be included initially
    Technology / technique	Decision	ReasonSentence Transformers	Exclude	User requirement
    Pinecone	Exclude	Remote/service architecture unnecessary and violates current local-only boundary
    Milvus	Exclude for MVP	Operational overhead unnecessary
    Qdrant	Exclude for MVP	Vector database unnecessary at current scale
    Weaviate	Exclude for MVP	Same
    PostgreSQL + pgvector	Exclude for MVP	Adds DB infrastructure without current need
    Elasticsearch	Exclude for MVP	Too heavy solely for this use case
    FAISS HNSW	Later only	Exact Flat search is sufficient initially
    FAISS IVF	Later only	Same
    HyDE	Exclude	Hypothetical model text contaminates strict document-only retrieval
    External query expansion	Exclude	Could introduce external knowledge
    LangChain	Exclude initially	Additional abstraction/dependency without an identified need
    LlamaIndex	Exclude initially	Same
    Agents	Exclude	Business flow is deterministic
    Knowledge graph	Exclude	Unnecessary complexity currently
    Cloud LLM	Exclude	Confidentiality boundary
    Cloud embeddings	Exclude	Confidentiality boundary
    Cloud OCR	Exclude	Confidentiality boundary
    Automatic web lookup	Exclude	Document-only knowledge requirement
    Cross-deal shared retrieval	Exclude	Leakage risk
16. One particularly important point: don't over-process every PDF

Not every uploaded document needs:

OCR
+
every table transformation
+
embedding of every element
+
LLM extraction


Use conditional processing.

PDF
│
▼
Does usable text exist?
│
├─ YES → Don't OCR
│
└─ NO  → Approved local OCR path


Likewise:

Does document contain relevant tables?
│
├─ YES → preserve/process tables
│
└─ NO  → normal structural text pipeline


And:

Is document relevant to current Box 2 applicability?
│
├─ YES → retrieve
│
└─ NO  → don't waste inference

17. Storage layout

Move from a mostly per-document model toward a knowledge/deal separation:

data/
│
├── knowledge/
│   └── bes_2026/
│       ├── source/
│       ├── structured_document/
│       ├── chunks.json
│       ├── vectors.npy
│       └── index.faiss
│
└── transactions/
│
├── TRANSACTION_001/
│   ├── documents/
│   ├── structured_documents/
│   ├── chunks.json
│   ├── vectors.npy
│   └── index.faiss
│
└── TRANSACTION_002/
└── ...


Hard search rule:

BOX 1
→ knowledge/bes_2026 ONLY

BOX 2
→ knowledge/bes_2026 relevant sector
+
current transaction ONLY

BOX 3
→ validated Box 1 + Box 2 evidence


Never allow:

TRANSACTION_001
↓
search
↓
TRANSACTION_002

18. Recommended new internal Chunk structure

The current chunk needs richer metadata.

Target concept:

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


For deal content:

{
"document_type": "DEAL_DOCUMENT",
"transaction_id": "TRANSACTION_001",
"evidence_scope": "DEAL"
}

19. Migration map by existing file
    Existing file	Action	Main changebackend/config.py	Modify	Add parser options, approved model paths, chunk/token parameters, retrieval settings
    backend/document_store.py	Modify	Separate approved knowledge indexes from transaction indexes
    backend/model_manager.py	Major modification	Remove generic DistilBERT assumption; support approved retrieval encoder + model-specific pooling
    backend/pdf_processor.py	Major replacement/refactor	Structured Docling parsing, headings, pages, tables, structured chunking
    backend/retriever.py	Major modification	Hard metadata filters, BM25/dense retrieval, optional fusion, richer result objects
    backend/qa_engine.py	Refactor	Replace generic Q&A orchestration with evidence-oriented retrieval and later box-specific services
    backend/schemas.py	Modify	Add transaction, evidence, applicability and 3-box response schemas
    backend/app.py	Modify	Add assessment orchestration/endpoints
    demo.py	Replace/extend	Demo should contain multiple pages, headings, tables and sector structure
    Tests	Expand significantly	Retrieval, sector isolation, transaction isolation, table provenance, grounding, insufficient evidence
    Dockerfile	Modify only if required	Include local model/parser artifacts without remote download at runtime
    FAISS/NumPy logic	Keep	No need to replace
20. Evaluation dataset is mandatory before selecting "best"

Create a manually validated internal test set.

Start with roughly 30 to 50 questions against the BES document and later add transaction-specific cases.

Example schema:

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


Include deliberately unanswerable questions.

Measure:

Recall@5
Recall@10
MRR

wrong-sector rate
wrong-transaction rate

answerable → evidence found
unanswerable → correctly withheld

table retrieval accuracy

source/page correctness


Then choose:

embedding model
chunking strategy
hybrid retrieval
RRF
reranker


based on results.

21. Laptop vs higher-resource environment
    Layer	Limited-RAM office laptop	Higher-resource approved environmentParser	Docling local	Docling local
    OCR	Only when required	Only when required
    Embedding	Smaller approved retrieval encoder	Benchmark larger approved encoder
    Embedding framework	transformers + torch	transformers + torch
    Vector search	FAISS Flat	FAISS Flat initially
    Lexical	Local BM25	Local BM25
    Vector DB	No	Still not required initially
    Reranker	Initially omit	Approved local reranker can be tested
    Final writing	Template baseline	Template + local LLM
    Local LLM	Small quantized model only if resources allow	Larger approved local model possible
    External APIs	No	No, unless security policy explicitly changes
    Knowledge sources	Supplied internal documents only	Same
22. Final recommended MVP

Don't build the maximum architecture immediately.

Stage 1
Docling
↓
Structured document
↓
Page / sector / heading / table metadata


Get parsing right first.

Stage 2
Structured chunks
↓
Approved retrieval encoder
↓
FAISS Flat

Stage 3
Metadata filter
+
Dense retrieval
+
local BM25

Stage 4
Retrieved evidence
↓
Structured evidence objects
↓
Source/page/scope validation

Stage 5
Box 1 pipeline
Box 2 applicability + evidence pipeline
Box 3 synthesis pipeline

Stage 6
Deterministic template writer


At this point you already have a functional system without a generative LLM.

Stage 7

Only if narrative quality requires it:

locally approved LLM
↓
strict evidence-only prompt
↓
post-generation claim validation

Stage 8

Only if evaluation demonstrates a retrieval problem:

Local reranker

Stage 9

Only if scale eventually demonstrates a search problem:

HNSW / IVF
or
local vector DB

23. Final target architecture
    ESG ASSESSMENT INPUT
    │
    ▼
    Sector / NACE Data
    │
    ▼
    Deterministic Mapping
    │
    ┌───────────┴───────────┐
    ▼                       ▼
    BES PDF                Deal PDFs
    │                       │
    └──────────┬────────────┘
    ▼
    LOCAL DOCLING
    │
    ▼
    Structured Documents
    │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
    Text          Tables       Metadata
    page/heading
    │
    ▼
    Structural Chunking
    │
    ▼
    Local Retrieval Encoder
    transformers + torch
    │
    ▼
    FAISS Flat
    │
    │
    ┌──────────┴──────────┐
    ▼                     ▼
    BM25              Dense retrieval
    │                     │
    └──────────┬──────────┘
    ▼
    Fusion if useful
    │
    ▼
    Optional Local Reranker
    ONLY if proven useful
    │
    ▼
    Retrieved Evidence
    │
    ▼
    Evidence Extraction
    │
    ▼
    Scope Validation
    sector/client/deal
    │
    ▼
    Structured Findings
    │
    ┌─────────────┼──────────────┐
    │             │              │
    ▼             ▼              ▼
    BOX 1          BOX 2          BOX 3
    Sector          Deal         Overall
    trends      risks/mitigants  conclusion
    │             │              │
    └─────────────┼──────────────┘
    ▼
    Deterministic Templates
    baseline
    │
    ▼
    Optional approved LOCAL LLM
    enhanced writing
    │
    ▼
    Evidence Validation
    │
    ▼
    Analyst Review

Instruction to the code-generation model

I'd finish the handoff with this:

IMPORTANT IMPLEMENTATION PRINCIPLES

1. Do not rewrite components that do not need changing.

2. Preserve FastAPI, the existing DocumentStore abstraction,
   FAISS, NumPy fallback, Pydantic and existing testing patterns.

3. Do not introduce Sentence Transformers.

4. Do not introduce a vector database.

5. Do not introduce HNSW or IVF initially.

6. Do not introduce HyDE.

7. Do not introduce LangChain or LlamaIndex.

8. Do not introduce external/cloud APIs.

9. Do not implement a local LLM until the parsing, retrieval,
   evidence and template pipelines are working.

10. All models must be loaded from explicitly configured local
    filesystem paths.

11. No model or package may download artifacts at runtime.

12. No confidential content may be transmitted outside the
    approved local environment.

13. ESG knowledge must come exclusively from the supplied
    internal documents.

14. Never supplement missing evidence with pretrained model
    knowledge.

15. Box 1 may retrieve ONLY from the approved main BES document.

16. Box 2 may retrieve from the relevant BES sector context and
    the CURRENT transaction's documents only.

17. Box 3 synthesizes validated Box 1 and Box 2 findings and
    introduces no new factual claims.

18. Sector-level evidence must never be represented as
    deal-level evidence without transaction-specific support.

19. Missing information, not-applicable information and unknown
    applicability must be represented as different states.

20. Every material finding must preserve:
    document ID,
    filename,
    page,
    heading/section,
    scope,
    extracted evidence.

21. Do not build features simply because they are common in
    generic RAG architectures. Add components only when they
    solve an identified requirement or improve measured
    evaluation results.

22. First implement the migration incrementally:
    parser -> chunking -> embeddings -> retrieval -> evidence ->
    three-box orchestration -> template generation.

23. After every migration stage, run the golden evaluation set
    and compare the new implementation against the previous
    baseline.

24. Optimize primarily for:
    grounded accuracy,
    zero cross-sector/deal leakage,
    document traceability,
    safe abstention,
    reasonable laptop RAM usage.

    Retrieval latency is secondary at the current corpus size.


This is the version I would give the code-generation model because it clearly separates what must change, what should stay, what should be benchmarked, and what is unnecessary, which should prevent the model from "helpfully" rebuilding your relatively focused ESG application into an oversized generic RAG platform.