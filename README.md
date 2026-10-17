# Document QA System (RAG)

A compact retrieval-augmented generation project for asking questions about local PDF and text files. The application keeps document parsing, chunking, embeddings, and FAISS retrieval local. A Gemini API key is optional: without one, the interface still works as a retrieval-only evidence explorer.

## What it demonstrates

- Multi-file PDF and TXT ingestion
- Configurable overlapping word chunks
- Local `all-MiniLM-L6-v2` embeddings and an in-memory FAISS index
- Top-k semantic retrieval with visible L2 distance scores
- Grounded Gemini answers using the current Google Gen AI SDK
- A no-key mode that returns the closest passage and exposes every retrieved chunk

## Request flow

```text
PDF/TXT -> text extraction -> overlapping chunks -> local embeddings -> FAISS
                                                                    |
question -> local query embedding -> top-k evidence ----------------+
                                      |                         |
                                      +-> retrieval-only answer +-> Gemini answer (optional)
```

Documents and the vector index remain in the Streamlit process; they are not persisted by this project. When Gemini generation is enabled, only the question and retrieved text excerpts are sent to the configured Gemini model.

## Quick start

Prerequisites: Python 3.10+ and enough memory to load the sentence-transformer model.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload `examples/sample_document.txt` to try the project. Leave the API-key field empty for retrieval-only mode.

To enable generated answers, copy `.env.example` to `.env`, add a Gemini API key, and optionally change the model alias:

```env
GEMINI_API_KEY="your-key"
GEMINI_MODEL="gemini-flash-latest"
```

The model is configurable because model availability changes over time. The default alias follows the model naming shown in the [Google Gen AI SDK documentation](https://googleapis.github.io/python-genai/).

## Docker

Build and run from the repository root:

```bash
docker build -t document-qa .
docker run --rm -p 8501:8501 document-qa
```

For Gemini generation, pass the key at runtime rather than baking it into the image:

```bash
docker run --rm -p 8501:8501 -e GEMINI_API_KEY -e GEMINI_MODEL document-qa
```

## Tests

```bash
pytest -q
```

The unit tests use a deterministic fake embedding model and do not call Gemini or download a transformer model.

## Repository map

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface and session state |
| `rag_pipeline.py` | Chunking, FAISS retrieval, optional generation |
| `utils.py` | PDF and text extraction |
| `tests/` | Offline pipeline tests |
| `examples/` | A small document for the no-key demo |

## Current limitations

- The FAISS index is in memory and disappears when the process restarts.
- Word-count chunking can split tables, headings, or semantic units.
- Scanned PDFs need OCR; `pypdf` only extracts embedded text.
- L2 distance is displayed as a diagnostic, not a calibrated confidence score.
- The extractive fallback surfaces the closest passage; it does not synthesize an answer.
- API errors are shown to the user and are not retried automatically.

Never commit `.env` or API keys. Uploaded documents can contain sensitive information, so review the excerpts before enabling a hosted model.
