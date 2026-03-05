import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from core.clients import get_clients
from core.config import QDRANT_COLLECTION, S3V_BUCKET_NAME, S3V_INDEX_NAME
from core.embeddings import generate_query_embedding

"""
Test 18 : TopK Boundary Analysis (The 100-limit Wall).

Probes the maximum ``topK`` each backend allows and records
whether the request succeeds or is rejected.
"""


@dataclass(frozen=True)
class LimitResult:
    """Immutable record for a topK boundary probe.

    Attributes
    ----------
    requested_k : int
        Requested number of top results.
    actual_k : int
        Number of results actually returned.
    platform : str
        Engine identifier.
    latency_ms : float
        Wall-clock query time in milliseconds.
    error_message : str
        Error text when the request was rejected.
    is_success : bool
        ``True`` if the query completed without error.
    """

    requested_k: int
    actual_k: int
    platform: str
    latency_ms: float
    error_message: str = ""
    is_success: bool = True


class LimitBenchmark(ABC):
    """Abstract base for topK limit probing engines.

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
    def test_top_k(self, vector: List[float], k: int) -> LimitResult:
        """Attempt a search with the given topK value.

        Parameters
        ----------
        vector : List[float]
            Query embedding vector.
        k : int
            Requested topK.

        Returns
        -------
        LimitResult
            Outcome of the probe.
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


class QdrantLimitEngine(LimitBenchmark):
    """Qdrant implementation — no hard topK ceiling."""

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
    """S3 Vectors implementation — enforces a 100-result ceiling."""

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
    """Print topK probe outcomes.

    Parameters
    ----------
    results : List[LimitResult]
        One result per engine for the given K.
    """
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
    """Run topK boundary analysis on both engines."""
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
