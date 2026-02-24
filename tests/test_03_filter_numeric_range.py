import time
from dataclasses import dataclass
from typing import List, Dict, Any

from qdrant_client import models
from core.clients import get_clients
from core.config import QDRANT_COLLECTION, S3V_BUCKET_NAME, S3V_INDEX_NAME
from core.embeddings import generate_query_embedding


@dataclass(frozen=True)
class SearchResult:
    """A unified data structure for search results across different platforms."""

    title: str
    year: int
    score: float
    platform: str
    latency_ms: float


class VectorSearchService:
    """Encapsulates logic for querying different vector databases."""

    def __init__(self):
        self.qdrant, self.s3_vectors = get_clients()

    def search_qdrant(self, vector: List[float], min_year: int) -> List[SearchResult]:
        start_time = time.perf_counter()

        response = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=5,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="year", range=models.Range(gte=min_year))
                ]
            ),
            with_payload=True,
        )

        latency = self._calculate_latency(start_time)
        return [
            SearchResult(
                title=p.payload["title"],
                year=p.payload["year"],
                score=p.score,
                platform="Qdrant",
                latency_ms=latency,
            )
            for p in response.points
        ]

    def search_s3_vectors(
        self, vector: List[float], min_year: int
    ) -> List[SearchResult]:
        start_time = time.perf_counter()

        response = self.s3_vectors.query_vectors(
            vectorBucketName=S3V_BUCKET_NAME,
            indexName=S3V_INDEX_NAME,
            queryVector={"float32": vector},
            topK=5,
            filter={"year": {"$gte": min_year}},
            returnDistance=True,
            returnMetadata=True,
        )

        latency = self._calculate_latency(start_time)
        return [
            SearchResult(
                title=v["metadata"]["title"],
                year=v["metadata"]["year"],
                score=v["distance"],
                platform="S3 Vectors",
                latency_ms=latency,
            )
            for v in response.get("vectors", [])
        ]

    @staticmethod
    def _calculate_latency(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000


def report_results(results: List[SearchResult]):
    """Handles the UI logic independently of the data retrieval."""
    if not results:
        print("No results found.")
        return

    platform_name = results[0].platform
    latency = results[0].latency_ms

    print(f"\n{platform_name} ({latency:.0f}ms):")
    for res in results:
        print(f"  {res.title} — year={res.year}, score={res.score:.4f}")


def run_benchmark():
    """Main execution flow following the 'Tell, Don't Ask' principle."""
    service = VectorSearchService()
    query_vector = generate_query_embedding("great modern movies")
    target_year = 2010

    qdrant_results = service.search_qdrant(query_vector, target_year)
    s3_results = service.search_s3_vectors(query_vector, target_year)

    report_results(qdrant_results)
    report_results(s3_results)


if __name__ == "__main__":
    run_benchmark()
