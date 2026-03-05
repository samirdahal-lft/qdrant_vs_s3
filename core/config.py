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
    """Convert a string ID to an integer for Qdrant.

    Qdrant requires integer or UUID point identifiers. This function
    extracts the numeric suffix from IDs like ``'mov_01'``.

    Parameters
    ----------
    string_id : str
        String identifier with a numeric suffix (e.g. ``'mov_01'``).

    Returns
    -------
    int
        Extracted integer (e.g. ``1``).
    """
    return int(string_id.split("_")[-1])
