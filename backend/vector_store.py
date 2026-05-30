from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from backend.config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

_openai = OpenAI(api_key=OPENAI_API_KEY)
_EMBED_MODEL = "text-embedding-3-large"
_DIMENSION = 3072


def get_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(PINECONE_INDEX_NAME)


def _chunk(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def _embed(texts: list[str]) -> list[list[float]]:
    response = _openai.embeddings.create(model=_EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def upsert_documents(docs: list[dict]):
    index = get_index()
    vectors = []
    for doc in docs:
        chunks = _chunk(doc["text"])
        embeddings = _embed(chunks)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{doc['url']}#chunk{i}",
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "title": doc["title"],
                    "date": doc["date"],
                    "url": doc["url"],
                    "chunk_index": i,
                },
            })
    for i in range(0, len(vectors), 50):
        index.upsert(vectors=vectors[i : i + 50])
    print(f"Upserted {len(vectors)} vectors")


def search(query: str, k: int = 5) -> list[dict]:
    index = get_index()
    embedding = _embed([query])[0]
    results = index.query(vector=embedding, top_k=k, include_metadata=True)
    return [
        {
            "text": match.metadata["text"],
            "title": match.metadata["title"],
            "date": match.metadata["date"],
            "url": match.metadata["url"],
            "score": match.score,
        }
        for match in results.matches
    ]
