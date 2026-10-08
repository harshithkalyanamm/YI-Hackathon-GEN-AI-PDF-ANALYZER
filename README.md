# Financial Risk AI

Local-only financial document intelligence for confidential PDFs. It extracts text with PyMuPDF, detects sector/section boundaries, embeds chunks using a locally stored DistilBERT model, and stores the vectors in a per-document FAISS index. On a Windows host where the native FAISS DLL cannot load, it automatically uses a persisted NumPy cosine-similarity index instead; the API and document format remain unchanged. `/query` uses `sector + request` to retrieve evidence and produces an extractive answer from the PDF text only.

This is intentionally not a generative chatbot: no PDF contents, queries, chunks, or embeddings are sent to an external service. DistilBERT is compact enough to run inference on a normal CPU-only laptop; startup loads it once and later requests reuse the saved document index. The `DocumentStore` and `Retriever` interfaces isolate local storage/FAISS so Azure Blob Storage or Azure Search can be added without changing the FastAPI layer.

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

## One-command demonstration

After starting the API, run this from the project directory:

```powershell
python demo.py
```

The script generates a harmless 20-line Energy-study PDF, uploads it, asks `Energy` + `bank risk`, and prints the real `/query` response. To inspect only the generated PDF without calling the API, run `python demo.py --create-only`.

## Tests

The tests generate a harmless sample PDF at runtime (no confidential fixture is stored in the repository) and cover extraction, indexing, sector-aware retrieval, registered upload/query endpoint handlers, response shape, and missing information.

```powershell
$env:PYTHONPATH="."
pytest -q
```

## Container deployment

The included `Dockerfile` is appropriate once the approved local model directory has been provisioned into the build context or mounted securely at runtime. The core code has no Azure dependency and makes no hosted inference calls; deploy the same image inside a private Azure network with private storage and a mounted model/index volume.
