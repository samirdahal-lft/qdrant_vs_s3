import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from core.clients import get_qdrant
from core.config import QDRANT_COLLECTION
from core.embeddings import generate_query_embedding

@dataclass(frozen=True)
class GroupedResult:
    group_id: str
    title: str
    score: float
    platform: str
    latency_ms: float
    is_supported: bool = True

class GroupingBenchmark(ABC):
    def __init__(self, client):
        self.client = client

    @abstractmethod
    def search_grouped(self, query_text: str, group_field: str, limit: int) -> List[GroupedResult]:
        """Subclasses implement platform-specific grouping or diversity logic."""
        pass

    def get_latency(self, start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000




class QdrantGroupingEngine(GroupingBenchmark):
    def search_grouped(self, query_text: str, group_field: str, limit: int) -> List[GroupedResult]:
        query_vector = generate_query_embedding(query_text)
        start = time.perf_counter()
        
        # Qdrant native server-side grouping logic
        response = self.client.query_points_groups(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            group_by=group_field,
            group_size=1,  # only 1 best hit per group
            limit=limit,
            with_payload=True,
        )
        
        latency = self.get_latency(start)
        results = []
        for group in response.groups:
            for hit in group.hits:
                results.append(GroupedResult(
                    group_id=str(group.id),
                    title=hit.payload['title'],
                    score=hit.score,
                    platform="Qdrant",
                    latency_ms=latency
                ))
        return results

class S3GroupingEngine(GroupingBenchmark):
    """Explicitly handles the lack of native server-side grouping."""
    def search_grouped(self, query_text: str, group_field: str, limit: int) -> List[GroupedResult]:
        return [GroupedResult("N/A", "N/A", 0.0, "S3 Vectors", 0.0, is_supported=False)]

def report_groups(test_name: str, result_groups: List[List[GroupedResult]]):
    print("=" * 60)
    print(f"RUNNING: {test_name}")
    print("=" * 60)

    for results in result_groups:
        if not results:
            continue
            
        engine = results[0]
        if not engine.is_supported:
            print(f"\n{engine.platform}:  Not supported")
            print("  Missing Feature: Server-side grouping (group_by).")
            print("  Manual Workaround: Client-side grouping required.")
            continue

        print(f"\n{engine.platform} ({engine.latency_ms:.0f}ms):")
        for res in results:
            print(f"  [{res.group_id}] {res.title} (score={res.score:.4f})")

def run():
    qc = get_qdrant()
    
    engines: List[GroupingBenchmark] = [
        QdrantGroupingEngine(qc), 
        S3GroupingEngine(None)
    ]

    # Conduct benchmark for diverse genre results
    benchmark_results = [
        engine.search_grouped("great movies of all time", group_field="genre", limit=10) 
        for engine in engines
    ]

    report_groups("TEST 17: Result Diversity via Grouping (1 per Genre)", benchmark_results)

if __name__ == "__main__":
    run()