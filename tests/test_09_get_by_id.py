import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from core.clients import get_clients
from core.config import QDRANT_COLLECTION, S3V_BUCKET_NAME, S3V_INDEX_NAME, qdrant_id

@dataclass(frozen=True)
class RetrievalResult:
    """Unified record for ID-based retrieval."""
    id: str
    title: str
    platform: str
    latency_ms: float
    metadata: Dict[str, Any]

class VectorBenchmark(ABC):
    def __init__(self, client):
        self.client = client

    @abstractmethod
    def retrieve_by_id(self, point_id: str) -> Optional[RetrievalResult]:
        """Implementation for specific platform retrieval."""
        pass

    def get_latency(self, start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000

class QdrantEngine(VectorBenchmark):
    def retrieve_by_id(self, point_id: str) -> Optional[RetrievalResult]:
        start = time.perf_counter()
        
        response = self.client.retrieve(
            collection_name=QDRANT_COLLECTION, 
            ids=[qdrant_id(point_id)], 
            with_payload=True
        )
        
        if not response:
            return None
            
        point = response[0]
        return RetrievalResult(
            id=str(point.id),
            title=point.payload['title'],
            platform="Qdrant",
            latency_ms=self.get_latency(start),
            metadata=point.payload
        )

class S3VectorEngine(VectorBenchmark):
    def retrieve_by_id(self, point_id: str) -> Optional[RetrievalResult]:
        start = time.perf_counter()

        response = self.client.get_vectors(
            vectorBucketName=S3V_BUCKET_NAME,
            indexName=S3V_INDEX_NAME,
            keys=[point_id],
            returnMetadata=True,
        )

        vectors = response.get("vectors", [])
        if not vectors:
            return None

        vector = vectors[0]
        return RetrievalResult(
            id=vector['key'],
            title=vector['metadata']['title'],
            platform="S3 Vectors",
            latency_ms=self.get_latency(start),
            metadata=vector['metadata']
        )

def report(test_name: str, results: List[Optional[RetrievalResult]]):
    print("=" * 60)
    print(f"RUNNING: {test_name}")
    print("=" * 60)

    for res in results:
        if not res:
            continue
            
        print(f"\n{res.platform} ({res.latency_ms:.0f}ms):")
        print(f"  ID/Key: {res.id}")
        print(f"  Title:  {res.title}")
        
        # Display extra metadata fields dynamically
        details = [f"{k}: {v}" for k, v in res.metadata.items() if k != 'title']
        for detail in details:
            print(f"  {detail}")

def run():
    qc, sc = get_clients()
    target_id = "mov_01"  # Inception

    engines = [QdrantEngine(qc), S3VectorEngine(sc)]
    benchmark_data = [engine.retrieve_by_id(target_id) for engine in engines]

    report("TEST 09: Get Vectors by ID", benchmark_data)

if __name__ == "__main__":
    run()