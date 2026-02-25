"""Test 11: Delete Vectors — Qdrant vs S3 Vectors."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from qdrant_client import models

from core.clients import get_clients
from core.config import QDRANT_COLLECTION, S3V_BUCKET_NAME, S3V_INDEX_NAME, qdrant_id


@dataclass(frozen=True)
class DeleteResult:
    movie_id: str
    count_before: int
    count_after: int
    platform: str
    latency_ms: float
    method_description: str


class VectorBenchmark(ABC):
    """Encapsulates common behavior for delete operations across platforms."""

    def __init__(self, client):
        self.client = client

    @abstractmethod
    def delete_vector(self, movie_id: str) -> DeleteResult:
        """Subclasses implement platform-specific vector deletion logic."""
        pass

    def get_latency(self, start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000


class QdrantEngine(VectorBenchmark):
    def _count(self) -> int:
        return self.client.get_collection(QDRANT_COLLECTION).points_count

    def delete_vector(self, movie_id: str) -> DeleteResult:
        before = self._count()
        start = time.perf_counter()

        self.client.delete(
            QDRANT_COLLECTION,
            points_selector=models.PointIdsList(points=[qdrant_id(movie_id)]),
        )

        return DeleteResult(
            movie_id=movie_id,
            count_before=before,
            count_after=self._count(),
            platform="Qdrant",
            latency_ms=self.get_latency(start),
            method_description="delete(points=[id])",
        )


class S3VectorEngine(VectorBenchmark):
    def _count(self) -> int:
        return len(
            self.client.list_vectors(
                vectorBucketName=S3V_BUCKET_NAME,
                indexName=S3V_INDEX_NAME,
                maxResults=1000,
            ).get("vectors", [])
        )

    def delete_vector(self, movie_id: str) -> DeleteResult:
        before = self._count()
        start = time.perf_counter()

        self.client.delete_vectors(
            vectorBucketName=S3V_BUCKET_NAME,
            indexName=S3V_INDEX_NAME,
            keys=[movie_id],
        )

        return DeleteResult(
            movie_id=movie_id,
            count_before=before,
            count_after=self._count(),
            platform="S3 Vectors",
            latency_ms=self.get_latency(start),
            method_description='delete_vectors(keys=[{"key": id}])',
        )


def report(test_name: str, results: List[DeleteResult]) -> None:
    """Prints a comparison report for deletion operations."""
    print("=" * 60)
    print(f"RUNNING: {test_name}")
    print("=" * 60)

    for res in results:
        if not res:
            continue
        print(
            f"\n{res.platform} ({res.latency_ms:.0f}ms): {res.count_before} → {res.count_after} vectors"
        )
        print(f"  Method: {res.method_description}")

    print("\n→ Both support batch delete. S3 Vectors max 500 per call.")


def run() -> None:
    """Entry point: runs vector deletion benchmark."""
    qc, sc = get_clients()
    target_id = "mov_50"  # Everything Everywhere All at Once

    engines = [QdrantEngine(qc), S3VectorEngine(sc)]
    results = [engine.delete_vector(target_id) for engine in engines]

    report("TEST 11: Delete Vectors", results)


if __name__ == "__main__":
    run()
