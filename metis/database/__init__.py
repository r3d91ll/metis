"""
Database Module

Provides ArangoDB client interfaces with Unix socket support for high performance.
Includes both low-level HTTP/2 client and high-level client with gRPC-compatible API.
"""

from .client import ArangoHttp2Client, ArangoHttp2Config, ArangoHttpError
from .memory import (
    ArangoClient,
    ArangoClientConfig,
    ArangoClientError,
    CollectionDefinition,
    resolve_client_config,
)

__all__ = [
    # Low-level HTTP/2 client
    "ArangoHttp2Client",
    "ArangoHttp2Config",
    "ArangoHttpError",
    # High-level client
    "ArangoClient",
    "ArangoClientConfig",
    "ArangoClientError",
    "CollectionDefinition",
    "resolve_client_config",
]
