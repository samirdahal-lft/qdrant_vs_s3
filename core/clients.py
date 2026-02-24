import boto3
from qdrant_client import QdrantClient

from core.config import AWS_REGION, QDRANT_URL

qdrant_client = None
s3vector_client = None


def get_qdrant() -> QdrantClient:
    """Return a Qdrant client"""
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient(url=QDRANT_URL, timeout=120)
    return qdrant_client


def get_s3v():
    """Return an S3 Vectors boto3 client"""
    global s3vector_client
    if s3vector_client is None:
        s3vector_client = boto3.client("s3vectors", region_name=AWS_REGION)
    return s3vector_client


def get_clients():
    """Return (qdrant_client, s3v_client)"""
    return get_qdrant(), get_s3v()
