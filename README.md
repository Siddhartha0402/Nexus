# Nexus

A personal RAG-powered AI research assistant that keeps you up to date with the latest arXiv papers. Ask questions in plain English and get cited, grounded answers — with a confidence score so you know when to trust the response.

---

## What it does

- **Ingests arXiv papers** across AI, ML, and NLP categories (cs.AI, cs.LG, cs.CL) and stores them as searchable vectors
- **Answers research questions** using retrieval-augmented generation — only citing what's actually in the knowledge base
- **Scores answer confidence** by computing cosine similarity between the GPT-4o response and the retrieved source chunks
- **Returns sources** with titles, dates, and arXiv links alongside every answer

---

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | GPT-4o (OpenAI) |
| Embeddings | `text-embedding-3-large` (OpenAI) |
| Vector store | Pinecone |
| Orchestration | LangChain |
| Paper ingestion | `arxiv` Python client |
| Frontend | Plain HTML/JS |

---

## Project structure

```
nexus/
├── backend/
│   ├── main.py          # FastAPI app, /chat and /ingest endpoints
│   ├── rag.py           # Retrieval, GPT-4o answer, confidence scoring
│   ├── ingest.py        # arXiv fetching logic
│   ├── vector_store.py  # Pinecone upsert and search
│   └── config.py        # Environment variable loading
├── frontend/
│   └── index.html       # Chat UI
├── requirements.txt
└── .env
```

---

## Local setup

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)
- A [Pinecone account](https://www.pinecone.io/) with an index created (dimension: `3072`, metric: `cosine`)

### 1. Clone and install

```bash
git clone https://github.com/your-username/nexus.git
cd nexus
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=nexus
```

### 3. Start the server

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`.

### 4. Open the frontend

Open `frontend/index.html` directly in your browser. It connects to the local API at port 8000.

---

## Ingesting papers

Nexus pulls the 30 most recent papers from `cs.AI`, `cs.LG`, and `cs.CL` on arXiv and stores their abstracts as embeddings in Pinecone.

**Via the API:**

```bash
curl -X POST http://localhost:8000/ingest
# {"papers_added": 87}
```

**Via the command line:**

```bash
python -m backend.ingest
```

Re-run this whenever you want to refresh the knowledge base with the latest papers. Duplicate titles are automatically deduplicated before upsert.

---

## API reference

### `POST /chat`

```json
{
  "message": "What are the latest advances in chain-of-thought prompting?",
  "history": []
}
```

**Response:**

```json
{
  "answer": "...",
  "sources": [
    { "title": "...", "url": "https://arxiv.org/abs/...", "date": "2025-05-20" }
  ],
  "confidence": 0.87
}
```

The `confidence` field is a cosine similarity score (0–1) between the answer and the best-matching source chunk. Values above ~0.75 generally indicate a well-grounded response.

### `POST /ingest`

Fetches and indexes the latest arXiv papers. Returns `{ "papers_added": N }`.
