import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from qdrant_client import models

from core.clients import get_qdrant
from core.config import QDRANT_COLLECTION
from core.embeddings import generate_query_embedding

"""
Test 14 : Full-Text Match Filter (Keyword Search in description).

Uses Qdrant’s text-index + ``MatchText`` filter to find movies
whose description contains a keyword. S3 Vectors lacks this.
"""


@dataclass(frozen=True)
class FullTextResult:
    """Immutable record for a full-text match hit.

    Attributes
    ----------
    title : str
        Movie title.
    keyword : str
        The keyword searched for.
    platform : str
        Engine identifier.
    latency_ms : float
        Wall-clock search time in milliseconds.
    is_supported : bool
        ``False`` when the engine lacks full-text filtering.
    """

    title: str
    keyword: str
    platform: str
    latency_ms: float
    is_supported: bool = True


class TextSearchBenchmark(ABC):
    """Abstract base for full-text search engines.

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
    def search_keyword_in_description(
        self, keyword: str, limit: int
    ) -> List[FullTextResult]:
        """Search for a keyword within the description field.

        Parameters
        ----------
        keyword : str
            Word to match inside movie descriptions.
        limit : int
            Maximum number of results.

        Returns
        -------
        List[FullTextResult]
            Matching results.
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


class QdrantTextEngine(TextSearchBenchmark):
    """Qdrant implementation — ``MatchText`` filter with text index."""

    def __init__(self, client):
        super().__init__(client)
        self._ensure_text_index()

    def _ensure_text_index(self):
        """Create a text payload index on the description field."""
        self.client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="description",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
            ),
        )

    def search_keyword_in_description(
        self, keyword: str, limit: int
    ) -> List[FullTextResult]:
        # Using a generic vector for the query as focus is on the filter
        qvec = generate_query_embedding("interesting movies")

        start = time.perf_counter()
        results = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=qvec,
            limit=limit,
            with_payload=True,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="description", match=models.MatchText(text=keyword)
                    )
                ]
            ),
        )

        latency = self.get_latency(start)
        return [
            FullTextResult(p.payload["title"], keyword, "Qdrant", latency)
            for p in results.points
        ]


class S3TextEngine(TextSearchBenchmark):
    """Null-object — S3 Vectors has no full-text filtering."""

    def search_keyword_in_description(
        self, keyword: str, limit: int
    ) -> List[FullTextResult]:
        return [FullTextResult("", keyword, "S3 Vectors", 0.0, is_supported=False)]


# --- 4. Reporting Layer ---


def report_search(results_groups: List[List[FullTextResult]]):
    """Print full-text search results per engine.

    Parameters
    ----------
    results_groups : List[List[FullTextResult]]
        One inner list per engine.
    """
    for results in results_groups:
        if not results:
            continue
        engine = results[0]

        if not engine.is_supported:
            print(f"\n{engine.platform}: Not supported")
            print("  Metadata filters are restricted to exact match or range.")
            continue

        print(
            f'\n{engine.platform} Keyword: "{engine.keyword}" ({engine.latency_ms:.0f}ms):'
        )
        for res in results:
            print(f"  {res.title} — found '{res.keyword}' in description")


# --- 5. Main Execution ---
def run():
    """Run full-text match benchmark across multiple keywords."""
    qc = get_qdrant()

    engines: List[TextSearchBenchmark] = [QdrantTextEngine(qc), S3TextEngine(None)]

    print("=" * 60)
    print("TEST 15: Full-Text Match Filter (Keyword Search)")
    print("=" * 60)

    keywords = ["robot", "war", "love"]

    for kw in keywords:
        results = [
            engine.search_keyword_in_description(kw, limit=5) for engine in engines
        ]
        report_search(results)


if __name__ == "__main__":
    run()
