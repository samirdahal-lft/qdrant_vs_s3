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
    title: str
    score: float
    platform: str
    latency_ms: float
    metadata: Dict[str, Any]


class VectorBenchmark(ABC):
    """Encapsulates common behavior for all vector database tests."""

    def __init__(self, client):
        self.client = client

    @abstractmethod
    def search(self, vector: List[float], limit: int) -> List[SearchResult]:
        """Subclasses implement specific search logic here."""
        pass

    def get_latency(self, start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000


class QdrantEngine(VectorBenchmark):
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
    """Prints a formatted benchmark report separating results per engine."""
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
    """Entry point: runs semantic search benchmark across multiple queries."""
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
