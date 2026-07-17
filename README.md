# Semantic Navigation

A semantic document search system with a FastAPI backend and a React reader UI. Documents are parsed into chunks, embedded with sentence-transformers, indexed with FAISS, and queried by meaning rather than exact keyword match.

**Pipeline:** Document → text extraction → chunking → embeddings → FAISS → search API

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** (frontend only)
- Git (to clone the repo)

Sample documents are included in `app/data/` (`pdf_example.pdf`, `txt_example.txt`, `md_example.md`, etc.).

## Quick start (backend + frontend)

Use this if you want the full reader UI: document preview, chunk navigation, text-to-speech, and semantic search.

### 1. Clone and set up the backend

```bash
git clone <your-repo-url>
cd semantic-nav

python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install Python dependencies and the spaCy language model:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Copy the environment file (offline mode works without API keys):

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

On first run, the embedding model (`all-MiniLM-L6-v2`) must be available locally. In **offline mode** (the default), set `SN_ALLOW_DOWNLOAD=true` for the first startup, or pre-populate the `models/` directory. In **online mode**, the model downloads automatically.

Start the API server **from the repository root** (`semantic-nav/`, where `app/` and `requirements.txt` live):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**First startup is slow.** The server auto-loads and indexes `pdf_example.pdf` on launch. Expect 1–3+ minutes the first time while the embedding model downloads (if needed), spaCy and sentence-transformers initialize, and chunks are embedded. Later startups are much faster if the model is already cached in `models/`.

The API will be available at http://127.0.0.1:8000 and interactive docs at http://127.0.0.1:8000/docs.

### 2. Set up the frontend

In a second terminal:

```bash
cd frontend

# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env

npm install
npm run dev
```

Open http://localhost:5173 in your browser. The frontend expects the API at `http://127.0.0.1:8000` (configured in `frontend/.env` as `VITE_API_BASE`).

### Reader features

- Load and switch between sample documents
- Navigate document chunks with prev/next controls (arrow keys)
- Browser text-to-speech reads chunk-by-chunk
- Semantic search jumps to and highlights matching text
- API debug drawer for inspecting backend responses

---

## Backend only

Use this if you only need the REST API (curl, Postman, or your own client) without the React UI.

### Install

Follow steps 1–4 from [Quick start](#quick-start-backend--frontend) above (venv, `pip install`, spaCy model, `.env`). You do not need Node.js.

### Start the server

From the repository root (`semantic-nav/`):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The first startup may take 1–3+ minutes while the default document is indexed (see [Quick start](#1-clone-and-set-up-the-backend) for model download notes).

### Basic usage

**Load a document** (indexes it for search; defaults to `pdf_example.pdf`):

```bash
curl "http://127.0.0.1:8000/load?document=pdf_example.pdf"
```

**Search** the loaded document:

```bash
curl "http://127.0.0.1:8000/search?q=what%20is%20calculus"
```

**Switch documents** at query time:

```bash
curl "http://127.0.0.1:8000/search?q=who%20is%20in%20the%20csv&document=csv_example.csv"
```

**Check current state:**

```bash
curl "http://127.0.0.1:8000/status"
```

**Get a document outline:**

```bash
curl "http://127.0.0.1:8000/outline?document=pdf_example.pdf"
```

**Get extracted text and chunks** (used by the reader):

```bash
curl "http://127.0.0.1:8000/preview/text?document=pdf_example.pdf"
```

### Supported document types

`pdf`, `txt`, `md`, `csv`, `docx` — place files in `app/data/` or pass the filename to the API.

### Search parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `q` | (required) | Search query |
| `document` | current document | Document filename in `app/data/` |
| `document_type` | inferred | Override type: `pdf`, `txt`, `md`, `csv`, `docx` |
| `chunker_type` | `semantic` | `semantic` or `fixed` |
| `k` | `3` | Number of results to retrieve |
| `confidence_threshold` | `0.0` | Minimum score (0–1 or 0–100) |

### Example search response

```json
{
  "query": "what is calculus",
  "results": [
    {
      "text": "...",
      "score": 0.82,
      "start": 1200,
      "end": 1450,
      "location": {
        "start": 1200,
        "end": 1450,
        "text": "...",
        "primary_page": 2
      }
    }
  ],
  "document": "pdf_example.pdf",
  "document_type": "pdf",
  "chunker": "semantic"
}
```

---

## API reference

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check |
| `GET /load?document=...` | Load and index a document |
| `GET /status` | Current document and mode |
| `GET /search?q=...` | Semantic search |
| `GET /outline` | Topic tree / document outline |
| `GET /preview/meta` | Preview metadata (page count, etc.) |
| `GET /preview/text` | Extracted text and chunks |
| `GET /preview/page?page=1` | PDF page as PNG |

Full interactive documentation: http://127.0.0.1:8000/docs

---

## Configuration

Copy `.env.example` to `.env`. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `SN_MODE` | `offline` | `offline` or `online` |
| `SN_ALLOW_DOWNLOAD` | `false` | In offline mode, set to `true` on first run to download the embedding model. Ignored in online mode (downloads are always allowed). |
| `GROQ_API_KEY` | — | **Online mode only.** Used for query rewriting and re-ranking. Also used for query splitting when `SN_QUERY_SPLITTER=groq`. |
| `SN_QUERY_SPLITTER` | `groq` | LLM provider for online query splitting: `groq`, `gemini`, `cerebras` |
| `SN_CORS_ORIGINS` | Vite dev URLs | Allowed frontend origins |

**Offline mode** (default): local embedding search only. No API keys needed. The embedding model must already exist in `models/`, or set `SN_ALLOW_DOWNLOAD=true` once to fetch it.

**Online mode** (`SN_MODE=online`): adds query rewriting, multi-query splitting, and result re-ranking via cloud LLMs. Requires `GROQ_API_KEY`. If using `gemini` or `cerebras` as the query splitter, the matching provider key is also required.

---

## Running tests

```bash
python -m unittest discover -s tests -v
```

---

## Tech stack

**Backend:** FastAPI, spaCy, sentence-transformers (all-MiniLM-L6-v2), FAISS, PyMuPDF, scikit-learn

**Frontend:** React 18, Vite 5
