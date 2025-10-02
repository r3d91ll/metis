"""
Database Module

Provides ArangoDB client interfaces with Unix socket support for high performance.
Includes both low-level HTTP/2 client and higher-level memory client.
"""

from .client import ArangoHttp2Client, ArangoHttp2Config, ArangoHttpError
from .memory import (
    ArangoMemoryClient,
    ArangoMemoryClientConfig,
    CollectionDefinition,
    MemoryServiceError,
    resolve_memory_config,
)

__all__ = [
    "ArangoHttp2Client",
    "ArangoHttp2Config",
    "ArangoHttpError",
    "ArangoMemoryClient",
    "ArangoMemoryClientConfig",
    "CollectionDefinition",
    "MemoryServiceError",
    "resolve_memory_config",
]
