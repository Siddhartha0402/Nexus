import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

_missing = [
    name for name, val in [
        ("OPENAI_API_KEY", OPENAI_API_KEY),
        ("PINECONE_API_KEY", PINECONE_API_KEY),
        ("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME),
    ]
    if not val
]

if _missing:
    raise ValueError(f"Missing required environment variables: {', '.join(_missing)}")
