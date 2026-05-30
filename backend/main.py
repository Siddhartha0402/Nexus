from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag import ask
from backend.ingest import fetch_papers, QUERIES
from backend.vector_store import upsert_documents

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: list = []


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        return ask(req.message, req.history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest():
    try:
        seen_titles = set()
        docs = []
        for query in QUERIES:
            for paper in fetch_papers(query):
                if paper["title"] not in seen_titles:
                    seen_titles.add(paper["title"])
                    docs.append(paper)
        upsert_documents(docs)
        return {"papers_added": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
