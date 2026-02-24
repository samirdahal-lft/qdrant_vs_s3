import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from qdrant_client import QdrantClient, models
from core.config import EMBEDDING_DIM, QDRANT_URL, qdrant_id
from core.embeddings import generate_query_embedding

COLLECTION = "movies_geo"


@dataclass(frozen=True)
class GeoLocation:
    lat: float
    lon: float


@dataclass(frozen=True)
class GeoSearchResult:
    title: str
    lat: float
    lon: float
    platform: str
    latency_ms: float
    is_supported: bool = True


class GeoBenchmark(ABC):
    def __init__(self, client):
        self.client = client

    @abstractmethod
    def search_nearby(
        self, center: GeoLocation, radius_meters: float, limit: int
    ) -> List[GeoSearchResult]:
        pass

    def get_latency(self, start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000


# --- 2. Platform Implementations ---


class QdrantGeoEngine(GeoBenchmark):
    def __init__(self, client):
        super().__init__(client)
        self._setup_collection()

    def _setup_collection(self):
        """Standardizes the creation of geo-indexed collection."""
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)

        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM, distance=models.Distance.COSINE
            ),
        )
        self.client.create_payload_index(
            COLLECTION, "location", models.PayloadSchemaType.GEO
        )
        self._ingest_synthetic_data()

    def _ingest_synthetic_data(self):
        movies = [
            {
                "id": "geo_01",
                "title": "Inception (Paris)",
                "lat": 48.8566,
                "lon": 2.3522,
            },
            {
                "id": "geo_02",
                "title": "Amélie (Montmartre)",
                "lat": 48.8867,
                "lon": 2.3431,
            },
            {
                "id": "geo_03",
                "title": "Dark Knight (Chicago)",
                "lat": 41.8781,
                "lon": -87.6298,
            },
            {
                "id": "geo_07",
                "title": "Midnight in Paris (Paris)",
                "lat": 48.8606,
                "lon": 2.3376,
            },
        ]
        points = [
            models.PointStruct(
                id=qdrant_id(m["id"]),
                vector=generate_query_embedding(m["title"]),
                payload={
                    "title": m["title"],
                    "location": {"lat": m["lat"], "lon": m["lon"]},
                },
            )
            for m in movies
        ]
        self.client.upsert(COLLECTION, points)

    def search_nearby(
        self, center: GeoLocation, radius_meters: float, limit: int
    ) -> List[GeoSearchResult]:
        qvec = generate_query_embedding("movies filmed in France")
        start = time.perf_counter()

        results = self.client.query_points(
            COLLECTION,
            query=qvec,
            limit=limit,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="location",
                        geo_radius=models.GeoRadius(
                            center=models.GeoPoint(lat=center.lat, lon=center.lon),
                            radius=radius_meters,
                        ),
                    )
                ]
            ),
            with_payload=True,
        )

        latency = self.get_latency(start)
        return [
            GeoSearchResult(
                p.payload["title"],
                p.payload["location"]["lat"],
                p.payload["location"]["lon"],
                "Qdrant",
                latency,
            )
            for p in results.points
        ]


class S3GeoEngine(GeoBenchmark):
    """Explicitly handles the lack of geo-filtering support."""

    def search_nearby(
        self, center: GeoLocation, radius_meters: float, limit: int
    ) -> List[GeoSearchResult]:
        return [GeoSearchResult("", 0.0, 0.0, "S3 Vectors", 0.0, is_supported=False)]


# --- 3. Execution & Reporting ---


def run():
    qc = QdrantClient(url=QDRANT_URL)
    paris_center = GeoLocation(lat=48.8566, lon=2.3522)
    radius = 50000.0  # 50km

    engines = [QdrantGeoEngine(qc), S3GeoEngine(None)]

    print("=" * 60)
    print("TEST 16: Geo Filtering — Radius Search (50km of Paris)")
    print("=" * 60)

    for engine in engines:
        results = engine.search_nearby(paris_center, radius, limit=5)

        if not results[0].is_supported:
            print(f"\n{results[0].platform}: ❌ Not supported")
            print("  No native support for geo_radius or spatial indexing.")
            continue

        print(f"\n{results[0].platform} ({results[0].latency_ms:.0f}ms):")
        for r in results:
            print(f"  {r.title} — lat={r.lat}, lon={r.lon}")

    qc.delete_collection(COLLECTION)


if __name__ == "__main__":
    run()
