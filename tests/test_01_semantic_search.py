"""Test 01 : Semantic Search (top-K cosine) — Qdrant vs S3 Vectors.

Runs pure vector similarity search (no filters) against both backends
and compares result quality and latency.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from core.clients import get_clients
from core.config import QDRANT_COLLECTION, S3V_BUCKET_NAME, S3V_INDEX_NAME
from core.embeddings import generate_query_embedding

TOP_K = 5

QUERIES = [
    "space exploration and adventure movies",
    "romantic stories about love",
    "crime and gangster films",
]


@dataclass(frozen=True)
class SearchResult:
    """Immutable record returned by every engine search call.

    Attributes
    ----------
    title : str
        Movie title extracted from the payload / metadata.
    score : float
        Cosine-similarity (Qdrant) or distance (S3 Vectors) score.
    platform : str
        Identifier of the engine that produced this result.
    latency_ms : float
        Wall-clock time of the search call in milliseconds.
    metadata : Dict[str, Any]
        Full payload / metadata dictionary for the matched point.
    """

    title: str
    score: float
    platform: str
    latency_ms: float
    metadata: Dict[str, Any]


class VectorBenchmark(ABC):
    """Abstract base for vector-database search engines.

    Provides shared latency measurement and enforces a uniform
    ``search`` interface that each concrete engine must implement.

    Parameters
    ----------
    client : object
        Platform-specific SDK client (Qdrant or S3 Vectors).

    Attributes
    ----------
    client : object
        The injected SDK client used for every search call.
    """

    def __init__(self, client):
        self.client = client

    @abstractmethod
    def search(self, vector: List[float], limit: int) -> List[SearchResult]:
        """Execute a vector search and return ranked results.

        Parameters
        ----------
        vector : List[float]
            Query embedding vector.
        limit : int
            Maximum number of results to return.

        Returns
        -------
        List[SearchResult]
            Ranked results with score and metadata.
        """
        pass

    def get_latency(self, start_time: float) -> float:
        """Calculate elapsed milliseconds since *start_time*.

        Parameters
        ----------
        start_time : float
            Value returned by ``time.perf_counter()`` before the operation.

        Returns
        -------
        float
            Elapsed time in milliseconds.
        """
        return (time.perf_counter() - start_time) * 1000


class QdrantEngine(VectorBenchmark):
    """Qdrant implementation of semantic search (no filter)."""

    def search(self, vector: List[float], limit: int) -> List[SearchResult]:
        start = time.perf_counter()

        response = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=limit,
            with_payload=True,
        )

        return [
            SearchResult(
                title=p.payload["title"],
                score=p.score,
                platform="Qdrant",
                latency_ms=self.get_latency(start),
                metadata=p.payload,
            )
            for p in response.points
        ]


class S3VectorEngine(VectorBenchmark):
    """S3 Vectors implementation of semantic search (no filter)."""

    def search(self, vector: List[float], limit: int) -> List[SearchResult]:
        start = time.perf_counter()

        response = self.client.query_vectors(
            vectorBucketName=S3V_BUCKET_NAME,
            indexName=S3V_INDEX_NAME,
            queryVector={"float32": vector},
            topK=limit,
            returnDistance=True,
            returnMetadata=True,
        )

        return [
            SearchResult(
                title=v["metadata"]["title"],
                score=v["distance"],
                platform="S3 Vectors",
                latency_ms=self.get_latency(start),
                metadata=v["metadata"],
            )
            for v in response.get("vectors", [])
        ]


def report(test_name: str, result_groups: List[List[SearchResult]]) -> None:
    """Print a formatted benchmark report, one section per engine.

    Parameters
    ----------
    test_name : str
        Human-readable label displayed as the report header.
    result_groups : List[List[SearchResult]]
        One inner list per engine, each containing ranked results.
    """
    print("=" * 60)
    print(f"RUNNING: {test_name}")
    print("=" * 60)

    for results in result_groups:
        if not results:
            continue

        engine_meta = results[0]
        print(f"\n{engine_meta.platform} ({engine_meta.latency_ms:.0f}ms):")

        for r in results:
            meta_str = ", ".join(
                [f"{k}={v}" for k, v in r.metadata.items() if k != "title"]
            )
            print(f"  {r.title} — {meta_str}, score={r.score:.4f}")


def run() -> None:
    """Run semantic search benchmark across multiple queries.

    Iterates over ``QUERIES``, generates embeddings, and benchmarks
    both Qdrant and S3 Vectors for each query.
    """
    qc, sc = get_clients()
    engines = [QdrantEngine(qc), S3VectorEngine(sc)]

    for query in QUERIES:
        query_vector = generate_query_embedding(query)
        benchmark_data = [
            engine.search(query_vector, limit=TOP_K) for engine in engines
        ]
        report(f'TEST 01: Semantic Search — query="{query}"', benchmark_data)


if __name__ == "__main__":
    run()
