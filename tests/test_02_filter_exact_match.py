import time
from typing import List
from dataclasses import dataclass

from qdrant_client import models
from core.clients import get_clients
from core.config import QDRANT_COLLECTION, S3V_BUCKET_NAME, S3V_INDEX_NAME
from core.embeddings import generate_query_embedding

@dataclass
class SearchResult:
    title: str
    genre: str
    score: float
    platform: str
    latency_ms: float

class MovieSearchTester:
    """Encapsulates search logic for benchmarking different vector platforms."""
    
    def __init__(self):
        self.qdrant, self.s3_vectors = get_clients()

    def run_sci_fi_benchmark(self, query_text: str = "popular movies") -> List[SearchResult]:
        query_vector = generate_query_embedding(query_text)
        
        results = []
        results.extend(self._search_qdrant(query_vector))
        results.extend(self._search_s3_vectors(query_vector))
        return results

    def _search_qdrant(self, vector: List[float]) -> List[SearchResult]:
        start_time = time.perf_counter()
        
        response = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=5,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="genre", match=models.MatchValue(value="Sci-Fi"))]
            ),
            with_payload=True,
        )
        
        latency = self._calculate_latency(start_time)
        return [
            SearchResult(p.payload['title'], p.payload['genre'], p.score, "Qdrant", latency)
            for p in response.points
        ]

    def _search_s3_vectors(self, vector: List[float]) -> List[SearchResult]:
        start_time = time.perf_counter()
        
        response = self.s3_vectors.query_vectors(
            vectorBucketName=S3V_BUCKET_NAME,
            indexName=S3V_INDEX_NAME,
            queryVector={"float32": vector},
            topK=5,
            filter={"genre": "Sci-Fi"},
            returnDistance=True,
            returnMetadata=True,
        )
        
        latency = self._calculate_latency(start_time)
        return [
            SearchResult(v["metadata"]['title'], v["metadata"]['genre'], v["distance"], "S3 Vectors", latency)
            for v in response.get("vectors", [])
        ]

    @staticmethod
    def _calculate_latency(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000

def display_results(results: List[SearchResult]):
    """Separates UI/Logging logic from business logic."""
    print("=" * 60)
    print('TEST 02: Filter — Exact Match (genre = "Sci-Fi")')
    print("=" * 60)
    
    current_platform = ""
    for res in results:
        if res.platform != current_platform:
            print(f"\n{res.platform} ({res.latency_ms:.0f}ms):")
            current_platform = res.platform
        
        print(f"  {res.title} — genre={res.genre}, score={res.score:.4f}")

def main():
    tester = MovieSearchTester()
    results = tester.run_sci_fi_benchmark()
    display_results(results)

if __name__ == "__main__":
    main()