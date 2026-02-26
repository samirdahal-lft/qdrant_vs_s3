import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from qdrant_client import models

from core.clients import get_clients
from core.config import QDRANT_COLLECTION, S3V_BUCKET_NAME, S3V_INDEX_NAME
from core.embeddings import generate_query_embedding

"""
Test 05 : Filter — OR Logic (Drama OR Comedy).

Uses OR filter to match either genre and compares both
backends for correctness and latency.
"""


@dataclass(frozen=True)
class SearchResult:
    """Immutable record returned by every engine search call.

    Attributes
    ----------
    title : str
        Movie title.
    score : float
        Similarity score or distance.
    platform : str
        Engine identifier.
    latency_ms : float
        Wall-clock search time in milliseconds.
    metadata : Dict[str, Any]
        Full payload / metadata for the matched point.
    """

    title: str
    score: float
    platform: str
    latency_ms: float
    metadata: Dict[str, Any]


class VectorBenchmark(ABC):
    """Abstract base for vector-database search engines.

    Parameters
    ----------
    client : object
        Platform-specific SDK client.

    Attributes
    ----------
    client : object
        Injected SDK client.
    """

    def __init__(self, client):
        self.client = client

    @abstractmethod
    def search(self, vector: List[float], limit: int) -> List[SearchResult]:
        """Execute a filtered vector search.

        Parameters
        ----------
        vector : List[float]
            Query embedding vector.
        limit : int
            Maximum number of results.

        Returns
        -------
        List[SearchResult]
            Ranked results matching the filter.
        """
        pass

    def get_latency(self, start_time: float) -> float:
        """Calculate elapsed milliseconds since *start_time*.

        Parameters
        ----------
        start_time : float
            ``time.perf_counter()`` value before the operation.

        Returns
        -------
        float
            Elapsed time in milliseconds.
        """
        return (time.perf_counter() - start_time) * 1000


class QdrantEngine(VectorBenchmark):
    """Qdrant implementation — OR logic via ``should`` clause list."""

    def search(self, vector: List[float], limit: int) -> List[SearchResult]:
        start = time.perf_counter()

        # Test-specific Filter Logic
        query_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="genre", match=models.MatchValue(value="Drama")
                ),
                models.FieldCondition(
                    key="genre", match=models.MatchValue(value="Comedy")
                ),
            ]
        )

        response = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=limit,
            query_filter=query_filter,
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
    """S3 Vectors implementation — OR logic via ``$or`` operator."""

    def search(self, vector: List[float], limit: int) -> List[SearchResult]:
        start = time.perf_counter()

        # Test-specific Filter Logic
        query_filter = {"$or": [{"genre": "Drama"}, {"genre": "Comedy"}]}

        response = self.client.query_vectors(
            vectorBucketName=S3V_BUCKET_NAME,
            indexName=S3V_INDEX_NAME,
            queryVector={"float32": vector},
            topK=limit,
            filter=query_filter,
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


def report(test_name: str, result_groups: List[List[SearchResult]]):
    """Print a formatted benchmark report, one section per engine.

    Parameters
    ----------
    test_name : str
        Human-readable label for the report header.
    result_groups : List[List[SearchResult]]
        One inner list per engine containing ranked results.
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
            # Dynamically list metadata keys (excluding title) to keep output clean
            meta_str = ", ".join(
                [f"{k}={v}" for k, v in r.metadata.items() if k != "title"]
            )
            print(f"  {r.title} — {meta_str}, score={r.score:.4f}")


def run():
    """Run OR filter benchmark on both engines."""
    qc, sc = get_clients()
    query_vector = generate_query_embedding("entertaining feel-good movies")

    engines = [QdrantEngine(qc), S3VectorEngine(sc)]
    benchmark_data = [engine.search(query_vector, limit=5) for engine in engines]

    report("TEST 05: OR Logic (Drama OR Comedy)", benchmark_data)


if __name__ == "__main__":
    run()
