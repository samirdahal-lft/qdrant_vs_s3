"""Test 12 : Hybrid Search (Dense + Sparse / BM25 with RRF Fusion).

Demonstrates Qdrant-only hybrid search using dense and sparse
vectors fused via Reciprocal Rank Fusion (RRF).
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from core.config import EMBEDDING_DIM, QDRANT_URL, qdrant_id
from core.dataset import MOVIES
from core.embeddings import generate_movie_embeddings, generate_query_embedding


@dataclass(frozen=True)
class HybridSearchResult:
    """Immutable record for a hybrid-search hit.

    Attributes
    ----------
    title : str
        Movie title.
    score : float
        Fused similarity score.
    platform : str
        Engine identifier.
    latency_ms : float
        Wall-clock search time in milliseconds.
    is_supported : bool
        ``False`` when the engine lacks hybrid search.
    """

    title: str
    score: float
    platform: str
    latency_ms: float
    is_supported: bool = True


class HybridSearchBenchmark(ABC):
    """Abstract base for hybrid-search engines.

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
    def search_hybrid(self, query_text: str, limit: int) -> List[HybridSearchResult]:
        """Run a hybrid dense + sparse search.

        Parameters
        ----------
        query_text : str
            Natural-language query string.
        limit : int
            Maximum number of results.

        Returns
        -------
        List[HybridSearchResult]
            Fused ranked results.
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


class QdrantHybridEngine(HybridSearchBenchmark):
    """Qdrant implementation — dense + BM25 sparse with RRF fusion.

    Attributes
    ----------
    COLLECTION : str
        Dedicated hybrid-search collection name.
    sparse_model : SparseTextEmbedding
        BM25 sparse encoder.
    """

    COLLECTION = "movies_hybrid"

    def __init__(self, client):
        super().__init__(client)
        self.sparse_model = SparseTextEmbedding(model_name="qdrant/bm25")
        self._setup_collection()

    def _setup_collection(self):
        """Create the hybrid collection with dense + sparse vector configs."""
        if self.client.collection_exists(self.COLLECTION):
            self.client.delete_collection(self.COLLECTION)

        self.client.create_collection(
            collection_name=self.COLLECTION,
            vectors_config={
                "dense": models.VectorParams(
                    size=EMBEDDING_DIM, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        self._ingest_data()

    def _ingest_data(self):
        """Embed movies and upsert dense + sparse vectors."""
        dense_embeddings = generate_movie_embeddings(MOVIES)
        points = []
        for m in MOVIES:
            text = f"{m['title']} {m['description']} {m['genre']}"
            sparse_vec = list(self.sparse_model.embed([text]))[0]

            points.append(
                models.PointStruct(
                    id=qdrant_id(m["id"]),
                    vector={
                        "dense": dense_embeddings[m["id"]],
                        "bm25": models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload={"title": m["title"], "genre": m["genre"]},
                )
            )
        self.client.upsert(collection_name=self.COLLECTION, points=points)

    def search_hybrid(self, query_text: str, limit: int) -> List[HybridSearchResult]:
        dense_vec = generate_query_embedding(query_text)
        sparse_query_vec = list(self.sparse_model.embed([query_text]))[0]

        start = time.perf_counter()
        results = self.client.query_points(
            collection_name=self.COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=10),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query_vec.indices.tolist(),
                        values=sparse_query_vec.values.tolist(),
                    ),
                    using="bm25",
                    limit=10,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        latency = self.get_latency(start)
        return [
            HybridSearchResult(p.payload["title"], p.score, "Qdrant", latency)
            for p in results.points
        ]


class S3HybridEngine(HybridSearchBenchmark):
    """Null-object — S3 Vectors does not support hybrid search."""

    def search_hybrid(self, query_text: str, limit: int) -> List[HybridSearchResult]:
        return [HybridSearchResult("N/A", 0.0, "S3 Vectors", 0.0, is_supported=False)]


def report(query_text: str, result_groups: List[List[HybridSearchResult]]) -> None:
    """Print hybrid-search benchmark results.

    Parameters
    ----------
    query_text : str
        The natural-language query that was executed.
    result_groups : List[List[HybridSearchResult]]
        One inner list per engine.
    """
    print("=" * 60)
    print(f"TEST 12: Hybrid Search — Query: '{query_text}'")
    print("=" * 60)

    for results in result_groups:
        if not results:
            continue
        platform = results[0].platform

        if not results[0].is_supported:
            print(f"\n{platform}: Not supported")
            print("  Required features: Sparse Vectors, Fusion (RRF)")
            continue

        print(f"\n{platform} (RRF Fusion - {results[0].latency_ms:.0f}ms):")
        for i, res in enumerate(results, 1):
            print(f"  {i}. {res.title} (score={res.score:.4f})")


def run() -> None:
    """Run hybrid search benchmark (Qdrant only)."""
    qc = QdrantClient(url=QDRANT_URL)
    query = "space robots adventure"

    engines = [QdrantHybridEngine(qc), S3HybridEngine(None)]
    benchmark_results = [engine.search_hybrid(query, limit=5) for engine in engines]

    report(query, benchmark_results)
    qc.delete_collection(QdrantHybridEngine.COLLECTION)


if __name__ == "__main__":
    run()
