"""Test 13 : Recommendation API (Positive / Negative Examples).

Uses Qdrant's native recommendation query with positive and
negative example IDs. S3 Vectors lacks this feature.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from qdrant_client import models

from core.clients import get_qdrant
from core.config import QDRANT_COLLECTION, qdrant_id


@dataclass(frozen=True)
class RecommendResult:
    """Immutable record for a recommendation hit.

    Attributes
    ----------
    title : str
        Movie title.
    genre : str
        Movie genre.
    year : int
        Release year.
    score : float
        Recommendation similarity score.
    platform : str
        Engine identifier.
    latency_ms : float
        Wall-clock search time in milliseconds.
    is_supported : bool
        ``False`` when the engine lacks a recommendation API.
    """

    title: str
    genre: str
    year: int
    score: float
    platform: str
    latency_ms: float
    is_supported: bool = True


class RecommendationBenchmark(ABC):
    """Abstract base for recommendation engines.

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
    def get_recommendations(
        self, positive_ids: List[str], negative_ids: List[str], limit: int
    ) -> List[RecommendResult]:
        """Return recommendations using positive/negative examples.

        Parameters
        ----------
        positive_ids : List[str]
            IDs of movies the user liked.
        negative_ids : List[str]
            IDs of movies the user disliked.
        limit : int
            Maximum number of results.

        Returns
        -------
        List[RecommendResult]
            Ranked recommendations.
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


class QdrantRecommendEngine(RecommendationBenchmark):
    """Qdrant implementation — ``RecommendQuery`` with average-vector strategy."""

    def get_recommendations(
        self, positive_ids: List[str], negative_ids: List[str], limit: int
    ) -> List[RecommendResult]:
        start = time.perf_counter()

        pos = [qdrant_id(pid) for pid in positive_ids]
        neg = [qdrant_id(nid) for nid in negative_ids]

        response = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=pos,
                    negative=neg,
                    strategy=models.RecommendStrategy.AVERAGE_VECTOR,
                )
            ),
            limit=limit,
            with_payload=True,
        )

        latency = self.get_latency(start)
        return [
            RecommendResult(
                title=p.payload["title"],
                genre=p.payload["genre"],
                year=p.payload["year"],
                score=p.score,
                platform="Qdrant",
                latency_ms=latency,
            )
            for p in response.points
        ]


class S3RecommendEngine(RecommendationBenchmark):
    """Null-object — S3 Vectors has no recommendation API."""

    def get_recommendations(self, *args, **kwargs) -> List[RecommendResult]:
        return [RecommendResult("", "", 0, 0.0, "S3 Vectors", 0.0, is_supported=False)]


def report_recommendations(
    title: str, results_list: List[List[RecommendResult]]
) -> None:
    """Print recommendation benchmark results.

    Parameters
    ----------
    title : str
        Human-readable scenario label.
    results_list : List[List[RecommendResult]]
        One inner list per engine.
    """
    print(f"\n--- {title} ---")

    for results in results_list:
        if not results:
            continue
        engine = results[0]

        if not engine.is_supported:
            print(f"{engine.platform}: Not supported (No native Recommendation API)")
            continue

        print(f"{engine.platform} ({engine.latency_ms:.0f}ms):")
        for i, res in enumerate(results, 1):
            print(f"  {i}. {res.title} ({res.genre}, {res.year}) score={res.score:.4f}")


def run() -> None:
    """Run recommendation benchmark (Qdrant only, two scenarios)."""
    qc = get_qdrant()

    engines: List[RecommendationBenchmark] = [
        QdrantRecommendEngine(qc),
        S3RecommendEngine(None),
    ]

    print("=" * 60)
    print("TEST 13: Recommendation API (Positive/Negative Examples)")
    print("=" * 60)

    results_1 = [e.get_recommendations(["mov_01"], [], 5) for e in engines]
    report_recommendations("Movies similar to Inception", results_1)

    # Scenario 2: Positive and Negative constraints
    results_2 = [
        e.get_recommendations(["mov_02", "mov_03"], ["mov_08"], 5) for e in engines
    ]
    report_recommendations(
        "Like Interstellar + Matrix, NOT like Forrest Gump", results_2
    )


if __name__ == "__main__":
    run()
