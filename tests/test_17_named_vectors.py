import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from qdrant_client import QdrantClient, models
from core.config import EMBEDDING_DIM, QDRANT_URL, qdrant_id
from core.dataset import MOVIES
from core.embeddings import generate_query_embedding, get_model

COLLECTION = "movies_named"


@dataclass(frozen=True)
class NamedVectorResult:
    title: str
    score: float
    platform: str
    latency_ms: float
    using_vector: str
    is_supported: bool = True


class NamedVectorBenchmark(ABC):
    def __init__(self, client):
        self.client = client

    @abstractmethod
    def search_by_named_vector(
        self, query_text: str, vector_name: str, limit: int
    ) -> List[NamedVectorResult]:
        pass

    def get_latency(self, start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000


class QdrantNamedEngine(NamedVectorBenchmark):
    def __init__(self, client):
        super().__init__(client)
        self._setup_collection()

    def _setup_collection(self):
        """Initializes collection with multiple named vector configurations."""
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)

        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "title": models.VectorParams(
                    size=EMBEDDING_DIM, distance=models.Distance.COSINE
                ),
                "description": models.VectorParams(
                    size=EMBEDDING_DIM, distance=models.Distance.COSINE
                ),
            },
        )
        self._ingest_named_data()

    def _ingest_named_data(self):
        model = get_model()
        points = []
        for m in MOVIES[:20]:
            title_vec = model.encode(m["title"], normalize_embeddings=True).tolist()
            desc_vec = model.encode(
                m["description"], normalize_embeddings=True
            ).tolist()

            points.append(
                models.PointStruct(
                    id=qdrant_id(m["id"]),
                    vector={"title": title_vec, "description": desc_vec},
                    payload={"title": m["title"], "genre": m["genre"]},
                )
            )
        self.client.upsert(COLLECTION, points)

    def search_by_named_vector(
        self, query_text: str, vector_name: str, limit: int
    ) -> List[NamedVectorResult]:
        qvec = generate_query_embedding(query_text)
        start = time.perf_counter()

        results = self.client.query_points(
            COLLECTION,
            query=qvec,
            using=vector_name,
            limit=limit,
            with_payload=True,
        )

        latency = self.get_latency(start)
        return [
            NamedVectorResult(
                p.payload["title"], p.score, "Qdrant", latency, vector_name
            )
            for p in results.points
        ]


class S3NamedEngine(NamedVectorBenchmark):
    """Signals lack of multi-vector support per key."""

    def search_by_named_vector(
        self, query_text: str, vector_name: str, limit: int
    ) -> List[NamedVectorResult]:
        return [
            NamedVectorResult(
                "", 0.0, "S3 Vectors", 0.0, vector_name, is_supported=False
            )
        ]


def report_named_search(title: str, results_list: List[List[NamedVectorResult]]):
    print(f"\n--- {title} ---")
    for results in results_list:
        if not results:
            continue
        engine = results[0]

        if not engine.is_supported:
            print(f"{engine.platform}:Not supported")
            print("  Required: Named Vectors (multi-vector per point).")
            continue

        print(
            f"{engine.platform} (using '{engine.using_vector}' vector - {engine.latency_ms:.0f}ms):"
        )
        for i, res in enumerate(results, 1):
            print(f"  {i}. {res.title} (score={res.score:.4f})")


def run():
    qc = QdrantClient(url=QDRANT_URL)
    engines = [QdrantNamedEngine(qc), S3NamedEngine(None)]

    print("=" * 60)
    print("TEST 18: Named Vectors (Title vs Description Embeddings)")
    print("=" * 60)

    query1 = "The Matrix"
    results1 = [e.search_by_named_vector(query1, "title", 5) for e in engines]
    report_named_search(f"Search by TITLE: '{query1}'", results1)

    query2 = "a computer hacker discovers reality is simulated"
    results2 = [e.search_by_named_vector(query2, "description", 5) for e in engines]
    report_named_search(f"Search by DESCRIPTION: '{query2}'", results2)

    qc.delete_collection(COLLECTION)


if __name__ == "__main__":
    run()
