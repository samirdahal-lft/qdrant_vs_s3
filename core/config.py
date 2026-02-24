"""Shared configuration constants."""

import os

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "movies"

AWS_REGION = "us-east-1"
S3V_BUCKET_NAME = "movie-search-comparison"
S3V_INDEX_NAME = "movies"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")


def qdrant_id(string_id: str) -> int:
    """Convert string ID like 'mov_01' to integer for Qdrant (requires int or UUID)."""
    return int(string_id.split("_")[-1])
