import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Union

from core.clients import get_clients
from core.config import QDRANT_COLLECTION, S3V_BUCKET_NAME, S3V_INDEX_NAME
from core.embeddings import generate_query_embedding


@dataclass(frozen=True)
class LimitResult:
    requested_k: int
    actual_k: int
    platform: str
    latency_ms: float
    error_message: str = ""
    is_success: bool = True


class LimitBenchmark(ABC):
    def __init__(self, client):
        self.client = client

    @abstractmethod
    def test_top_k(self, vector: List[float], k: int) -> LimitResult:
        pass

    def get_latency(self, start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000




class QdrantLimitEngine(LimitBenchmark):
    def test_top_k(self, vector: List[float], k: int) -> LimitResult:
        start = time.perf_counter()
        try:
            response = self.client.query_points(
                collection_name=QDRANT_COLLECTION, query=vector, limit=k
            )
            return LimitResult(
                requested_k=k,
                actual_k=len(response.points),
                platform="Qdrant",
                latency_ms=self.get_latency(start),
            )
        except Exception as e:
            return LimitResult(k, 0, "Qdrant", 0, str(e), False)


class S3LimitEngine(LimitBenchmark):
    def test_top_k(self, vector: List[float], k: int) -> LimitResult:
        start = time.perf_counter()
        try:
            response = self.client.query_vectors(
                vectorBucketName=S3V_BUCKET_NAME,
                indexName=S3V_INDEX_NAME,
                queryVector={"float32": vector},
                topK=k,
            )
            return LimitResult(
                requested_k=k,
                actual_k=len(response.get("vectors", [])),
                platform="S3 Vectors",
                latency_ms=self.get_latency(start),
            )
        except Exception as e:
            # Capturing the expected architectural limit rejection
            return LimitResult(
                k, 0, "S3 Vectors", self.get_latency(start), str(e), False
            )


def report_limit_test(results: List[LimitResult]):
    for res in results:
        status = "✓" if res.is_success else "✗ (REJECTED)"
        print(f"\n{res.platform}: topK={res.requested_k}")
        print(f"  Status:  {status}")
        if res.is_success:
            print(
                f"  Results: {res.actual_k} points retrieved in {res.latency_ms:.0f}ms"
            )
        else:
            print(f"  Error:   {res.error_message[:80]}...")


def run():
    qc, sc = get_clients()
    qvec = generate_query_embedding("popular movies")

    q_engine = QdrantLimitEngine(qc)
    s_engine = S3LimitEngine(sc)

    print("=" * 60)
    print("TEST 22: TopK Boundary Analysis (The 100-limit Wall)")
    print("=" * 60)

    print("\n--- Scenario A: Requesting 100 results ---")
    report_limit_test([q_engine.test_top_k(qvec, 100), s_engine.test_top_k(qvec, 100)])

    print("\n--- Scenario B: Requesting 101 results ---")
    report_limit_test([q_engine.test_top_k(qvec, 101), s_engine.test_top_k(qvec, 101)])

    print("\n--- Scenario C: Requesting 200 results ---")
    report_limit_test([q_engine.test_top_k(qvec, 200)])


if __name__ == "__main__":
    run()
