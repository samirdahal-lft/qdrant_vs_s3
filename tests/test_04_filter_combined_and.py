import time
from dataclasses import dataclass
from typing import List, Protocol

from qdrant_client import models
from core.clients import get_clients
from core.config import QDRANT_COLLECTION, S3V_BUCKET_NAME, S3V_INDEX_NAME
from core.embeddings import generate_query_embedding

@dataclass(frozen=True)
class MovieMetadata:
    title: str
    genre: str
    year: int
    rating: float
    description: str
    director: str
    language:str

@dataclass(frozen=True)
class SearchResult:
    metadata: MovieMetadata
    score: float
    platform: str
    latency_ms: float


class VectorStore(Protocol):
    def search(self, vector: List[float], limit: int) -> List[SearchResult]:
        ...

class QdrantStore:
    def __init__(self, client):
        self.client = client


    def search(self, vector: List[float], limit: int) -> List[SearchResult]:
        start = time.perf_counter()
        
        # Define Filter Logic
        filter_query = models.Filter(
            must=[
                models.FieldCondition(key="genre", match=models.MatchValue(value="Sci-Fi")),
                models.FieldCondition(key="year", range=models.Range(gte=2010)),
                models.FieldCondition(key="rating", range=models.Range(gt=7.5)),
            ]
        )

        response = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=limit,
            query_filter=filter_query,
            with_payload=True,
        )

        latency = (time.perf_counter() - start) * 1000
        return [
            SearchResult(
                metadata=MovieMetadata(**p.payload),
                score=p.score,
                platform="Qdrant",
                latency_ms=latency
            ) for p in response.points
        ]

class S3VectorStore:
    def __init__(self, client):
        self.client = client

    def search(self, vector: List[float], limit: int) -> List[SearchResult]:
        start = time.perf_counter()
        
        # Define Filter Logic
        filter_query = {
            "$and": [
                {"genre": "Sci-Fi"},
                {"year": {"$gte": 2010}},
                {"rating": {"$gt": 7.5}},
            ]
        }

        response = self.client.query_vectors(
            vectorBucketName=S3V_BUCKET_NAME,
            indexName=S3V_INDEX_NAME,
            queryVector={"float32": vector},
            topK=limit,
            filter=filter_query,
            returnDistance=True,
            returnMetadata=True,
        )

        latency = (time.perf_counter() - start) * 1000
        return [
            SearchResult(
                metadata=MovieMetadata(**v["metadata"]),
                score=v["distance"],
                platform="S3 Vectors",
                latency_ms=latency
            ) for v in response.get("vectors", [])
        ]


def print_benchmark_report(results: List[SearchResult]):
    if not results:
        return
    
    first = results[0]
    print(f"\n{first.platform} ({first.latency_ms:.0f}ms):")
    for r in results:
        m = r.metadata
        print(f"  {m.title} — genre={m.genre}, year={m.year}, rating={m.rating}")

def run_test_04():
    qc, sc = get_clients()
    query_vector = generate_query_embedding("best sci-fi movies")
    
    stores: List[VectorStore] = [QdrantStore(qc), S3VectorStore(sc)]

    print("=" * 60)
    print("TEST 04: Filter — Combined AND (Sci-Fi + year>=2010 + rating>7.5)")
    print("=" * 60)

    for store in stores:
        results = store.search(query_vector, limit=5)
        print_benchmark_report(results)

if __name__ == "__main__":
    run_test_04()