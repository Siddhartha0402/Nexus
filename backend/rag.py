import numpy as np
from openai import OpenAI
from backend.vector_store import search, _embed
from backend.config import OPENAI_API_KEY

_openai = OpenAI(api_key=OPENAI_API_KEY)
_CHAT_MODEL = "gpt-4o"
_EMBED_MODEL = "text-embedding-3-large"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a)
    vb = np.array(b)
    return float(np.dot(va / np.linalg.norm(va), vb / np.linalg.norm(vb)))


def ask(question: str, history: list = []) -> dict:
    chunks = search(question, k=5)

    context = "\n\n".join(
        f"[{i+1}] Title: {c['title']} | Date: {c['date']}\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    system_content = (
        "Answer only using the context below.\n"
        "Always cite the source title. If the answer is not in the context, say so clearly.\n\n"
        + context
    )

    messages = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    response = _openai.chat.completions.create(
        model=_CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    answer = response.choices[0].message.content

    answer_embedding = _embed([answer])[0]
    chunk_embeddings = _embed([c["text"] for c in chunks])

    confidence = max(
        _cosine_similarity(answer_embedding, emb) for emb in chunk_embeddings
    )

    seen_titles = set()
    sources = []
    for c in chunks:
        if c["title"] not in seen_titles:
            seen_titles.add(c["title"])
            sources.append({"title": c["title"], "url": c["url"], "date": c["date"]})

    return {"answer": answer, "sources": sources, "confidence": confidence}
