import time

from dotenv import load_dotenv

from core.config import (
    AWS_REGION,
    EMBEDDING_DIM,
    QDRANT_COLLECTION,
    QDRANT_URL,
    S3V_BUCKET_NAME,
    S3V_INDEX_NAME,
    qdrant_id,
)
from core.dataset import MOVIES
from core.embeddings import generate_movie_embeddings

load_dotenv()


def setup_qdrant():
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=QDRANT_URL, timeout=120)
    if client.collection_exists(collection_name=QDRANT_COLLECTION):
        client.delete_collection(collection_name=QDRANT_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIM, distance=models.Distance.COSINE
        ),
    )
    for f in ["genre", "director", "language"]:
        client.create_payload_index(
            QDRANT_COLLECTION, f, models.PayloadSchemaType.KEYWORD
        )
    for f in ["year", "rating"]:
        client.create_payload_index(
            QDRANT_COLLECTION, f, models.PayloadSchemaType.FLOAT
        )

    embeddings = generate_movie_embeddings(MOVIES)
    points = [
        models.PointStruct(
            id=qdrant_id(m["id"]),
            vector=embeddings[m["id"]],
            payload={
                k: m[k]
                for k in [
                    "title",
                    "description",
                    "genre",
                    "year",
                    "rating",
                    "director",
                    "language",
                ]
            },
        )
        for m in MOVIES
    ]
    client.upsert(QDRANT_COLLECTION, points)
    print(f"  Qdrant: {len(points)} movies loaded")
    return client, embeddings


def setup_s3vectors():
    import boto3

    client = boto3.client("s3vectors", region_name=AWS_REGION)

    try:
        client.create_vector_bucket(vectorBucketName=S3V_BUCKET_NAME)
    except client.exceptions.ConflictException:
        pass
    try:
        client.create_index(
            vectorBucketName=S3V_BUCKET_NAME,
            indexName=S3V_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            distanceMetric="cosine",
            dataType="float32",
            metadataConfiguration={"nonFilterableMetadataKeys": ["description"]},
        )
    except client.exceptions.ConflictException:
        pass

    embeddings = generate_movie_embeddings(MOVIES)
    vectors = [
        {
            "key": m["id"],
            "data": {"float32": embeddings[m["id"]]},
            "metadata": {
                k: m[k]
                for k in [
                    "title",
                    "description",
                    "genre",
                    "year",
                    "rating",
                    "director",
                    "language",
                ]
            },
        }
        for m in MOVIES
    ]
    client.put_vectors(
        vectorBucketName=S3V_BUCKET_NAME, indexName=S3V_INDEX_NAME, vectors=vectors
    )
    print(f"  S3 Vectors: {len(vectors)} movies loaded")
    return client, embeddings


def setup_both():
    """Returns (qdrant_client, s3v_client, embeddings)."""
    print("Setting up both platforms...")
    qc, emb = setup_qdrant()
    sc, _ = setup_s3vectors()
    time.sleep(2)
    print("  Ready.\n")
    return qc, sc, emb


def cleanup_qdrant():
    from qdrant_client import QdrantClient

    QdrantClient(url=QDRANT_URL).delete_collection(QDRANT_COLLECTION)
    print("Qdrant cleaned up.")


def cleanup_s3vectors():
    import boto3

    c = boto3.client("s3vectors", region_name=AWS_REGION)
    try:
        c.delete_vector_index(
            vectorBucketName=S3V_BUCKET_NAME, indexName=S3V_INDEX_NAME
        )
    except:
        pass
    try:
        c.delete_vector_bucket(vectorBucketName=S3V_BUCKET_NAME)
    except:
        pass
    print("S3 Vectors cleaned up.")


def cleanup_both():
    cleanup_qdrant()
    cleanup_s3vectors()


if __name__ == "__main__":
    setup_both()
