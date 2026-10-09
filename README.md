# Financial Risk AI

Local-only financial document intelligence for confidential PDFs. It parses pages, headings and detected tables locally with PyMuPDF, writes a structured document representation, embeds structure-aware chunks using a locally stored DistilBERT model, and stores normalized vectors in a per-document NumPy (`vectors.npy`) index. Retrieval applies the requested sector as a hard filter before exact cosine similarity ranking, preventing cross-sector chunks from entering candidates. `/query` uses `sector + request` to retrieve evidence and produces an extractive answer from the PDF text only.

This is intentionally not a generative chatbot: no PDF contents, queries, chunks, or embeddings are sent to an external service. DistilBERT is compact enough to run inference on a normal CPU-only laptop; startup loads it once and later requests reuse the saved document index. The `DocumentStore` and `Retriever` interfaces isolate local storage/retrieval so Azure Blob Storage or another private retriever can be added without changing the FastAPI layer.

## Run on Windows

From this `financial-risk-ai` directory:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
python -m huggingface_hub download distilbert/distilbert-base-uncased --local-dir models\distilbert-base-uncased
$env:HF_HUB_OFFLINE="1"
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The model command is a one-time download of the public model files only; it does not upload or transmit any financial data. After it completes, the service loads only `models\distilbert-base-uncased` and works offline. If the command is unavailable, install the optional downloader once with `pip install huggingface_hub` and rerun it.

## API

Upload a PDF:

```powershell
curl.exe -X POST http://127.0.0.1:8000/upload -F "file=@C:\path\to\study.pdf"
```

It returns `{"document_id":"...","status":"processed"}`. Query it:

```powershell
curl.exe -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d '{"document_id":"YOUR_ID","sector":"Energy","request":"climate transition risk"}'
```

Every successful query response has exactly one public field: `{"answer":"..."}`. If there is no matching evidence, the answer is `The requested information could not be found in the uploaded document.`

`GET /health` returns `{"status":"ok"}`.

`POST /assessment/sector-trends` is the controlled Box 1 baseline. Upload its source with multipart field `document_type=PRIMARY_BES`, then send `{"document_id":"...", "sector":"Energy"}`. The service rejects general or deal documents and uses only validated sector-scoped evidence from that one primary BES document.

`POST /assessment/deal-risk-drivers` is the conservative Box 2 baseline. Upload a document with `document_type=DEAL_DOCUMENT` and its `transaction_id`; the request must supply the same transaction ID, sector, and requested topic. The service rejects cross-transaction access and returns only transaction-document evidence. It distinguishes `NOT_APPLICABLE`, `APPLICABLE_EVIDENCE_FOUND`, `APPLICABLE_EVIDENCE_NOT_FOUND`, and `APPLICABILITY_UNKNOWN`. A missing mention is `APPLICABILITY_UNKNOWN` unless an analyst sends the optional `applicability_override` value `APPLICABLE` or `NOT_APPLICABLE`.

`POST /assessment/overall-conclusion` is the Box 3 deterministic synthesizer. It accepts only the already-produced Box 1 and Box 2 text, performs no document retrieval, and prefixes those inputs without adding any claim. When both inputs are insufficient-evidence states, it returns the insufficient-information response directly.

## One-command demonstration

After starting the API, run this from the project directory:

```powershell
python demo.py
```

The script generates a harmless 20-line Energy-study PDF, uploads it, asks `Energy` + `bank risk`, and prints the real `/query` response. To inspect only the generated PDF without calling the API, run `python demo.py --create-only`.

## Tests

The tests generate a harmless sample PDF at runtime (no confidential fixture is stored in the repository) and cover extraction, indexing, sector-aware retrieval, registered upload/query endpoint handlers, response shape, and missing information.

Each index also stores `structured_document.json` and `chunks.json` privately beside `vectors.npy`. They retain page number, heading path, sector, section, element type (`paragraph` or `table`), and scope for audit-oriented work in later stages; the public `/query` response continues to return only `{"answer":"..."}`.

Internally, retrieved chunks are converted to validated evidence findings before an answer is assembled. Each finding retains its exact statement plus document, filename, page, heading path, sector, element type, and evidence scope. A finding is rejected if its document, page, sector, or sector scope does not match the query boundary.

## Retrieval strategy and evaluation

The default local strategy is `dense`: exact NumPy cosine similarity. A dependency-free BM25 implementation is also available for identifiers, acronyms, and named policy terms; set `FINANCIAL_RISK_RETRIEVAL_STRATEGY` to `dense`, `bm25`, or `hybrid` before starting the API. The sector filter runs before either score is calculated. The offline golden-set tests measure Recall@k, correct abstention for unanswerable questions, and wrong-sector retrieval rate; add approved internal cases before selecting BM25 or hybrid as the default.

Use the included non-confidential starter set as a format, then replace or extend it with manually validated approved internal cases:

```powershell
python evaluate.py --document-id YOUR_DOCUMENT_ID --cases evaluation\golden_cases.json
```

Run the same cases with `dense`, `bm25`, and `hybrid` before changing the default. Do not add a local reranker, a different encoder, or Docling until that comparison shows a measurable improvement on the approved evaluation set.

## Offline artifact preflight and encoder benchmark

The remaining model/parser stages intentionally require explicitly staged local artifacts. No command in this project downloads model files or sends document content outside the machine.

```powershell
python preflight.py
python benchmark_models.py --document-id YOUR_DOCUMENT_ID --model approved-a=C:\approved-models\encoder-a --model approved-b=C:\approved-models\encoder-b
```

The benchmark skips and reports any directory missing `config.json` or model weights. Compare Recall@k, abstention, and wrong-sector rate on approved cases before selecting a new encoder. `preflight.py` also reports whether optional Docling is installed; keep the tested PyMuPDF parser unless a local Docling benchmark on representative structured/table PDFs proves an improvement.

```powershell
$env:PYTHONPATH="."
pytest -q
```

## Container deployment

The included `Dockerfile` is appropriate once the approved local model directory has been provisioned into the build context or mounted securely at runtime. The core code has no Azure dependency and makes no hosted inference calls; deploy the same image inside a private Azure network with private storage and a mounted model/index volume.
