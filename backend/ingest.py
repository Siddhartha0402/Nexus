import arxiv
from backend.vector_store import upsert_documents

QUERIES = ["cs.AI", "cs.LG", "cs.CL"]
MAX_RESULTS = 30


def fetch_papers(query: str) -> list[dict]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    return [
        {
            "title": paper.title,
            "text": paper.summary,
            "date": str(paper.published.date()),
            "url": paper.entry_id,
        }
        for paper in client.results(search)
    ]


if __name__ == "__main__":
    seen_titles = set()
    docs = []

    for query in QUERIES:
        for paper in fetch_papers(query):
            if paper["title"] not in seen_titles:
                seen_titles.add(paper["title"])
                docs.append(paper)

    upsert_documents(docs)
    print(f"Ingested {len(docs)} papers into Pinecone")
