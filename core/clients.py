"""Singleton factory functions for Qdrant and S3 Vectors clients."""

import boto3
from botocore.client import BaseClient
from qdrant_client import QdrantClient

from core.config import AWS_REGION, QDRANT_URL

qdrant_client = None
s3vector_client = None


def get_qdrant() -> QdrantClient:
    """Return a singleton Qdrant client instance.

    Returns
    -------
    QdrantClient
        Connected client pointing to ``QDRANT_URL``.
    """
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient(url=QDRANT_URL, timeout=120)
    return qdrant_client


def get_s3v() -> BaseClient:
    """Return a singleton S3 Vectors boto3 client.

    Returns
    -------
    botocore.client.BaseClient
        Low-level ``s3vectors`` service client for ``AWS_REGION``.
    """
    global s3vector_client
    if s3vector_client is None:
        s3vector_client = boto3.client("s3vectors", region_name=AWS_REGION)
    return s3vector_client


def get_clients() -> tuple[QdrantClient, BaseClient]:
    """Return both vector-database clients as a tuple.

    Returns
    -------
    tuple[QdrantClient, botocore.client.BaseClient]
        ``(qdrant_client, s3v_client)``.
    """
    return get_qdrant(), get_s3v()
