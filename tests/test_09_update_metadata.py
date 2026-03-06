"""Test 09 : Update Metadata — Qdrant vs S3 Vectors.

Compares partial-update (Qdrant ``set_payload``) with full
re-put (S3 Vectors ``put_vectors``) for a single field change.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

from core.clients import get_clients
from core.config import QDRANT_COLLECTION, S3V_BUCKET_NAME, S3V_INDEX_NAME, qdrant_id
from core.dataset import MOVIES
from core.embeddings import generate_movie_embeddings


@dataclass(frozen=True)
class UpdateResult:
    """Immutable record capturing one metadata-update operation.

    Attributes
    ----------
    movie_id : str
        Targeted movie identifier.
    title : str
        Human-readable movie title.
    old_value : Any
        Original field value before update.
    new_value : Any
        Requested new field value.
    verified_value : Any
        Value read back after update for verification.
    platform : str
        Engine identifier.
    latency_ms : float
        Wall-clock update time in milliseconds.
    method_description : str
        Short description of the API call used.
    """

    movie_id: str
    title: str
    old_value: Any
    new_value: Any
    verified_value: Any
    platform: str
    latency_ms: float
    method_description: str


class VectorBenchmark(ABC):
    """Abstract base for metadata-update engines.

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
    def update_metadata(
        self, movie_id: str, field: str, new_value: Any
    ) -> UpdateResult:
        """Perform a metadata field update.

        Parameters
        ----------
        movie_id : str
            Target movie identifier.
        field : str
            Metadata field name to modify.
        new_value : Any
            Desired new value for *field*.

        Returns
        -------
        UpdateResult
            Outcome including verified read-back.
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


class QdrantEngine(VectorBenchmark):
    """Qdrant implementation — partial update via ``set_payload()``."""

    def update_metadata(
        self, movie_id: str, field: str, new_value: Any
    ) -> UpdateResult:
        original = next(m for m in MOVIES if m["id"] == movie_id)
        start = time.perf_counter()

        # Qdrant supports partial field update — only send the changed field
        self.client.set_payload(
            QDRANT_COLLECTION,
            payload={field: new_value},
            points=[qdrant_id(movie_id)],
        )

        latency = self.get_latency(start)
        verified = self.client.retrieve(
            QDRANT_COLLECTION, ids=[qdrant_id(movie_id)], with_payload=True
        )

        return UpdateResult(
            movie_id=movie_id,
            title=original["title"],
            old_value=original[field],
            new_value=new_value,
            verified_value=verified[0].payload[field],
            platform="Qdrant",
            latency_ms=latency,
            method_description="set_payload() — partial update, send only changed field",
        )


class S3VectorEngine(VectorBenchmark):
    """S3 Vectors implementation — full re-put via ``put_vectors()``.

    Parameters
    ----------
    client : object
        S3 Vectors SDK client.
    embeddings : dict
        Pre-computed movie embeddings keyed by movie ID.

    Attributes
    ----------
    embeddings : dict
        Stored embeddings for re-upload.
    """

    def __init__(self, client, embeddings):
        super().__init__(client)
        self.embeddings = embeddings

    def update_metadata(
        self, movie_id: str, field: str, new_value: Any
    ) -> UpdateResult:
        original = next(m for m in MOVIES if m["id"] == movie_id)
        full_metadata = {
            k: original[k]
            for k in [
                "title",
                "description",
                "genre",
                "year",
                "rating",
                "director",
                "language",
            ]
        }
        full_metadata[field] = new_value

        start = time.perf_counter()

        # S3 Vectors requires re-putting the entire vector + all metadata
        self.client.put_vectors(
            vectorBucketName=S3V_BUCKET_NAME,
            indexName=S3V_INDEX_NAME,
            vectors=[
                {
                    "key": movie_id,
                    "data": {"float32": self.embeddings[movie_id]},
                    "metadata": full_metadata,
                }
            ],
        )

        latency = self.get_latency(start)
        verified = self.client.get_vectors(
            vectorBucketName=S3V_BUCKET_NAME,
            indexName=S3V_INDEX_NAME,
            keys=[movie_id],
            returnMetadata=True,
        )

        return UpdateResult(
            movie_id=movie_id,
            title=original["title"],
            old_value=original[field],
            new_value=new_value,
            verified_value=verified["vectors"][0]["metadata"][field],
            platform="S3 Vectors",
            latency_ms=latency,
            method_description="put_vectors() — must re-send vector + ALL metadata",
        )


def report(test_name: str, results: List[UpdateResult]) -> None:
    """Print a comparison report for metadata update operations.

    Parameters
    ----------
    test_name : str
        Human-readable label for the report header.
    results : List[UpdateResult]
        One result per engine.
    """
    print("=" * 60)
    print(f"RUNNING: {test_name}")
    print("=" * 60)

    for res in results:
        if not res:
            continue
        print(f"\n{res.platform} ({res.latency_ms:.0f}ms):")
        print(f"  Movie:  {res.title} (id={res.movie_id})")
        print(f"  Rating: {res.old_value} → verified={res.verified_value}")
        print(f"  Method: {res.method_description}")

    print("\n→ Key difference:")
    print("  Qdrant: partial update — just send the field you want to change")
    print("  S3 Vectors: no patch API — must re-put entire vector + metadata")


def run() -> None:
    """Run metadata update benchmark on both engines."""
    qc, sc = get_clients()
    embeddings = generate_movie_embeddings(MOVIES)

    engines = [QdrantEngine(qc), S3VectorEngine(sc, embeddings)]
    results = [engine.update_metadata("mov_05", "rating", 9.5) for engine in engines]

    report("TEST 09: Update Metadata (change rating of Shawshank Redemption)", results)


if __name__ == "__main__":
    run()
