"""Embedding generation and caching using sentence-transformers."""

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import CACHE_DIR, EMBEDDING_MODEL

_model = None


def get_model() -> SentenceTransformer:
    """Return a singleton sentence-transformer model instance.

    Lazily loads ``EMBEDDING_MODEL`` on first call and caches it
    for subsequent invocations.

    Returns
    -------
    SentenceTransformer
        Pre-loaded embedding model.
    """
    global _model
    if _model is None:
        print(f"  Loading model: {EMBEDDING_MODEL}...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _cache_path(name: str) -> str:
    """Build the absolute path for a named embedding cache file.

    Parameters
    ----------
    name : str
        Logical cache name (used as the JSON filename stem).

    Returns
    -------
    str
        Absolute file path under ``CACHE_DIR``.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.json")


def generate_movie_embeddings(movies: list[dict]) -> dict[str, list[float]]:
    """Generate or load cached embeddings for a list of movies.

    Each movie is embedded as ``"{title}. {description}"``. Results are
    cached to disk as JSON so subsequent calls skip re-encoding.

    Parameters
    ----------
    movies : list[dict]
        Movie records; each must contain ``'id'``, ``'title'``, and
        ``'description'`` keys.

    Returns
    -------
    dict[str, list[float]]
        Mapping of ``movie_id`` to its embedding vector.
    """
    path = _cache_path("movie_embeddings")
    if os.path.exists(path):
        print("  Using cached movie embeddings.")
        with open(path) as f:
            return json.load(f)

    model = get_model()
    texts = [f"{m['title']}. {m['description']}" for m in movies]
    print(texts)
    print(f"  Generating embeddings for {len(texts)} movies...")
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    result = {m["id"]: vec.tolist() for m, vec in zip(movies, vectors)}
    print(result)

    with open(path, "w") as f:
        json.dump(result, f)
    print(f"  Cached to {path}")
    return result


def generate_query_embedding(text: str) -> list[float]:
    """Generate a single normalised query embedding.

    Parameters
    ----------
    text : str
        Natural-language query string.

    Returns
    -------
    list[float]
        Unit-length embedding vector.
    """
    model = get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()
