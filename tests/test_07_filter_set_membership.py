import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any

from qdrant_client import models
from core.clients import get_clients
from core.config import QDRANT_COLLECTION, S3V_BUCKET_NAME, S3V_INDEX_NAME
from core.embeddings import generate_query_embedding


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
        pass

    def get_latency(self, start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000


class QdrantEngine(VectorBenchmark):
    def search(self, vector: List[float], limit: int) -> List[SearchResult]:
        start = time.perf_counter()

        target_genres = ["Action", "Thriller"]
        query_filter = models.Filter(
            should=[
                models.FieldCondition(key="genre", match=models.MatchValue(value=g))
                for g in target_genres
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
    def search(self, vector: List[float], limit: int) -> List[SearchResult]:
        start = time.perf_counter()

        # Set Membership in S3 Vectors uses the '$in' operator
        query_filter = {"genre": {"$in": ["Action", "Thriller"]}}

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


def run():
    qc, sc = get_clients()
    query_vector = generate_query_embedding("exciting intense movies")

    engines = [QdrantEngine(qc), S3VectorEngine(sc)]
    benchmark_data = [engine.search(query_vector, limit=5) for engine in engines]

    report(
        "TEST 07: Filter — Set Membership (genre IN [Action, Thriller])", benchmark_data
    )


if __name__ == "__main__":
    run()
