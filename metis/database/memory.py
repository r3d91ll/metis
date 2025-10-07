"""
ArangoDB High-Level Client
===========================

High-level ArangoDB client with Unix socket support and HTTP/2 transport.

Features:
- Dual-socket support: separate read-only and read-write connections
- HTTP/2 multiplexing for efficient database operations
- gRPC-compatible error handling for backward compatibility
- Bulk operations: efficient batch insert/update with NDJSON
- Collection and index management
- AQL query execution with cursor support

Security:
- Read-only socket blocks AQL DML mutations
- Read-write socket blocks admin endpoints
- Policy enforcement at proxy layer (see metis/database/proxies/)

Environment Variables:
- ARANGO_RO_SOCKET: Read-only socket path (default: /run/metis/readonly/arangod.sock)
- ARANGO_RW_SOCKET: Read-write socket path (default: /run/metis/readwrite/arangod.sock)
- ARANGO_USERNAME: Database username (default: root)
- ARANGO_PASSWORD: Database password (required unless ARANGO_SKIP_AUTH=1)
- ARANGO_HTTP_BASE_URL: Base URL for HTTP requests (default: http://localhost)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import grpc
from grpc import StatusCode

from .client import ArangoHttp2Client, ArangoHttp2Config, ArangoHttpError

DEFAULT_ARANGO_SOCKET = "/run/arangodb3/arangodb.sock"

HTTP_TO_GRPC_STATUS: dict[int, StatusCode] = {
    400: StatusCode.INVALID_ARGUMENT,
    401: StatusCode.UNAUTHENTICATED,
    403: StatusCode.PERMISSION_DENIED,
    404: StatusCode.NOT_FOUND,
    408: StatusCode.DEADLINE_EXCEEDED,
    409: StatusCode.ALREADY_EXISTS,
    412: StatusCode.FAILED_PRECONDITION,
    413: StatusCode.RESOURCE_EXHAUSTED,
    425: StatusCode.FAILED_PRECONDITION,
    429: StatusCode.RESOURCE_EXHAUSTED,
    500: StatusCode.INTERNAL,
    503: StatusCode.UNAVAILABLE,
}


class ArangoClientError(grpc.RpcError):
    """gRPC-compatible error raised by the ArangoDB client."""

    def __init__(self, message: str, *, status: StatusCode = StatusCode.UNKNOWN, details: dict[str, Any] | None = None) -> None:
        """
        Initialize an ArangoClientError with a message, gRPC status code, and optional structured details.
        
        Parameters:
            message (str): Human-readable error message.
            status (StatusCode): gRPC status code representing the error condition.
            details (dict[str, Any] | None): Additional structured error metadata; stored as an empty dict when not provided.
        """
        super().__init__(message)
        self._message = message
        self._status = status
        self._details = details or {}

    # grpc.RpcError API -------------------------------------------------
    def code(self) -> StatusCode:  # pragma: no cover - trivial delegation
        return self._status

    def details(self) -> str:  # pragma: no cover - trivial delegation
        return self._message

    def trailing_metadata(self):  # pragma: no cover - compatibility stub
        return None

    def debug_error_string(self) -> str:  # pragma: no cover - compatibility stub
        return self._message


@dataclass(slots=True)
class ArangoClientConfig:
    """Configuration values resolved for the ArangoDB client."""

    database: str
    username: str
    password: str
    base_url: str
    read_socket: str
    write_socket: str
    connect_timeout: float
    read_timeout: float
    write_timeout: float

    def build_read_config(self) -> ArangoHttp2Config:
        return ArangoHttp2Config(
            database=self.database,
            socket_path=self.read_socket,
            base_url=self.base_url,
            username=self.username,
            password=self.password,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            write_timeout=self.write_timeout,
        )

    def build_write_config(self) -> ArangoHttp2Config:
        return ArangoHttp2Config(
            database=self.database,
            socket_path=self.write_socket,
            base_url=self.base_url,
            username=self.username,
            password=self.password,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            write_timeout=self.write_timeout,
        )


@dataclass(slots=True)
class CollectionDefinition:
    """Collection metadata used when creating collections."""

    name: str
    type: str = "document"
    options: dict[str, Any] | None = None
    indexes: Sequence[dict[str, Any]] | None = None


def _http_status_to_grpc(status_code: int) -> StatusCode:
    return HTTP_TO_GRPC_STATUS.get(status_code, StatusCode.UNKNOWN)


def _parse_timeout(value: str | None, default: float) -> float:
    """
    Parse a timeout value from a string, falling back to a default when the value is None or invalid.
    
    Parameters:
        value (str | None): The timeout value to parse, typically in seconds as a string.
        default (float): The fallback timeout to use if `value` is None or cannot be parsed as a float.
    
    Returns:
        float: The parsed timeout in seconds, or `default` if parsing is not possible.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_client_config(
    *,
    database: str = "metis_db",
    username: str | None = None,
    password: str | None = None,
    socket_path: str | None = None,
    read_socket: str | None = None,
    write_socket: str | None = None,
    use_proxies: bool | None = None,
    base_url: str | None = None,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    write_timeout: float | None = None,
) -> ArangoMemoryClientConfig:
    """
    Resolve an ArangoClientConfig from explicit arguments and environment variables.
    
    Builds the final client configuration by combining provided parameters with environment
    variables and sensible defaults. Enforces authentication unless ARANGO_SKIP_AUTH is set;
    resolves HTTP base URL, read/write Unix socket paths (supporting separate read/write
    proxies or a single direct socket), and connect/read/write timeouts.
    
    Parameters:
        database: Database name to use (default "metis_db").
        username: Optional username; if omitted, uses ARANGO_USERNAME or "root".
        password: Optional password; if omitted and authentication is required, reads ARANGO_PASSWORD.
        socket_path: Single socket path to apply to both read and write if read_socket/write_socket are not provided.
        read_socket: Optional path for the read-only Unix domain socket.
        write_socket: Optional path for the read-write Unix domain socket.
        use_proxies: If True (or omitted), prefer separate read/write proxy sockets; if False, use a single direct socket.
        base_url: Optional HTTP base URL; if omitted, uses ARANGO_HTTP_BASE_URL or "http://localhost".
        connect_timeout: Optional connect timeout in seconds; if omitted, reads ARANGO_CONNECT_TIMEOUT or defaults to 5.0.
        read_timeout: Optional read timeout in seconds; if omitted, reads ARANGO_READ_TIMEOUT or defaults to 30.0.
        write_timeout: Optional write timeout in seconds; if omitted, reads ARANGO_WRITE_TIMEOUT or defaults to 30.0.
    
    Environment variables referenced:
        ARANGO_SKIP_AUTH, ARANGO_USERNAME, ARANGO_PASSWORD,
        ARANGO_HTTP_BASE_URL, ARANGO_RO_SOCKET, ARANGO_RW_SOCKET, ARANGO_SOCKET,
        ARANGO_CONNECT_TIMEOUT, ARANGO_READ_TIMEOUT, ARANGO_WRITE_TIMEOUT.
    
    Returns:
        An ArangoClientConfig populated with resolved credentials, base URL, socket paths, and timeouts.
    
    Raises:
        ValueError: If authentication is required but no password is provided (and ARANGO_SKIP_AUTH is not set).
    """

    env = os.environ

    # Username defaults to env or 'root'
    if username is None:
        username = env.get("ARANGO_USERNAME", "root")

    # Allow skipping auth entirely for local dev when server auth is disabled
    skip_auth = (env.get("ARANGO_SKIP_AUTH", "").strip().lower() in {"1", "true", "yes", "on"})

    if not skip_auth and password is None:
        password = env.get("ARANGO_PASSWORD")
        if password is None or password == "":
            raise ValueError(
                "ArangoDB password required (set ARANGO_PASSWORD) or export ARANGO_SKIP_AUTH=1 for dev"
            )

    if base_url is None:
        base_url = env.get("ARANGO_HTTP_BASE_URL", "http://localhost")

    # Allow explicit sockets to override environment
    if socket_path:
        read_socket = read_socket or socket_path
        write_socket = write_socket or socket_path

    env_ro = env.get("ARANGO_RO_SOCKET")
    env_rw = env.get("ARANGO_RW_SOCKET")
    env_direct = env.get("ARANGO_SOCKET")

    proxies_requested = True if use_proxies is None else use_proxies

    if proxies_requested:
        default_ro = "/run/metis/readonly/arangod.sock"
        default_rw = "/run/metis/readwrite/arangod.sock"

        read_socket = read_socket or env_ro or default_ro
        write_socket = write_socket or env_rw or default_rw

        # If only one socket could be resolved, mirror it so we still have a
        # functioning configuration instead of passing ``None`` into httpx.
        if read_socket is None:
            read_socket = write_socket or default_ro
        if write_socket is None:
            write_socket = read_socket or default_rw
    else:
        direct_socket = socket_path or env_direct or DEFAULT_ARANGO_SOCKET
        read_socket = read_socket or direct_socket
        write_socket = write_socket or direct_socket

    connect_timeout = connect_timeout if connect_timeout is not None else _parse_timeout(env.get("ARANGO_CONNECT_TIMEOUT"), 5.0)
    read_timeout = read_timeout if read_timeout is not None else _parse_timeout(env.get("ARANGO_READ_TIMEOUT"), 30.0)
    write_timeout = write_timeout if write_timeout is not None else _parse_timeout(env.get("ARANGO_WRITE_TIMEOUT"), 30.0)

    return ArangoClientConfig(
        database=database,
        username=username if not skip_auth else "",
        password=password if not skip_auth else "",
        base_url=base_url,
        read_socket=read_socket,
        write_socket=write_socket,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        write_timeout=write_timeout,
    )


class ArangoClient:
    """High-level ArangoDB client with Unix socket support and gRPC-compatible error handling."""

    def __init__(self, config: ArangoClientConfig) -> None:
        """
        Initialize the ArangoClient and create HTTP/2 clients for read and write according to the provided configuration.
        
        If the configured read and write sockets are the same, the same HTTP/2 client instance is reused for both directions; otherwise separate clients are created. The client's closed state is initialized to False.
        
        Parameters:
            config (ArangoClientConfig): Resolved client configuration used to construct read/write HTTP/2 transports.
        """
        self._config = config
        self._read_client = ArangoHttp2Client(config.build_read_config())
        if config.read_socket == config.write_socket:
            self._write_client = self._read_client
            self._shared_clients = True
        else:
            self._write_client = ArangoHttp2Client(config.build_write_config())
            self._shared_clients = False
        self._closed = False

    # Context management ------------------------------------------------
    def close(self) -> None:
        """
        Close the client's underlying HTTP clients and mark the ArangoClient as closed.
        
        Closes the read client and, if the write client is distinct, closes the write client as well. This operation is idempotent; calling it multiple times has no effect. After closing, the client must not be used for further requests.
        """
        if self._closed:
            return
        self._read_client.close()
        if not self._shared_clients:
            self._write_client.close()
        self._closed = True

    def __enter__(self) -> "ArangoClient":  # pragma: no cover - helper
        """
        Enter a context for the client and return the client instance.
        
        Returns:
            self: The same ArangoClient instance.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - helper
        """
        Exit the context manager and close the client.
        
        Called when leaving a with-block to ensure the client's resources are closed.
        """
        self.close()

    # Public API --------------------------------------------------------
    def execute_query(
        self,
        aql: str,
        bind_vars: dict[str, Any] | None = None,
        *,
        batch_size: int | None = None,
        full_count: bool = False,
    ) -> list[dict[str, Any]]:
        try:
            return self._read_client.query(
                aql,
                bind_vars=bind_vars,
                batch_size=batch_size or 1000,
                full_count=full_count,
            )
        except ArangoHttpError as exc:  # pragma: no cover - thin wrapper
            raise self._wrap_error(exc) from exc

    def bulk_insert(
        self,
        collection: str,
        documents: Iterable[dict[str, Any]],
        *,
        chunk_size: int = 1000,
    ) -> int:
        total_inserted = 0
        batch: list[dict[str, Any]] = []

        def flush() -> int:
            if not batch:
                return 0
            try:
                response = self._write_client.insert_documents(collection, batch, on_duplicate="ignore")
            except ArangoHttpError as exc:  # pragma: no cover - thin wrapper
                raise self._wrap_error(exc) from exc
            created = response.get("created")
            return int(created) if isinstance(created, int) else len(batch)

        for doc in documents:
            batch.append(doc)
            if len(batch) >= chunk_size:
                total_inserted += flush()
                batch.clear()

        if batch:
            total_inserted += flush()
            batch.clear()

        return total_inserted

    def bulk_import(
        self,
        collection: str,
        documents: Iterable[dict[str, Any]],
        *,
        chunk_size: int = 1000,
        on_duplicate: str = "ignore",
    ) -> int:
        """Bulk import with configurable duplicate policy.

        on_duplicate: one of 'ignore', 'update', 'replace', 'error'.
        """
        total_inserted = 0
        batch: list[dict[str, Any]] = []

        def flush() -> int:
            if not batch:
                return 0
            try:
                response = self._write_client.insert_documents(collection, batch, on_duplicate=on_duplicate)
            except ArangoHttpError as exc:  # pragma: no cover - thin wrapper
                raise self._wrap_error(exc) from exc
            # import API reports created/updated/ignored; treat any mutation as progress
            created = int(response.get("created", 0))
            updated = int(response.get("updated", 0))
            return created + updated

        for doc in documents:
            batch.append(doc)
            if len(batch) >= chunk_size:
                total_inserted += flush()
                batch.clear()

        if batch:
            total_inserted += flush()
            batch.clear()

        return total_inserted

    def get_document(self, collection: str, key: str) -> dict[str, Any]:
        try:
            return self._read_client.get_document(collection, key)
        except ArangoHttpError as exc:  # pragma: no cover - thin wrapper
            raise self._wrap_error(exc) from exc

    def execute_transaction(
        self,
        *,
        write: Sequence[str],
        read: Sequence[str] | None = None,
        exclusive: Sequence[str] | None = None,
        action: str,
        params: dict[str, Any] | None = None,
        wait_for_sync: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "collections": {
                "write": list(write),
            },
            "action": action,
            "params": params or {},
        }

        if read:
            payload["collections"]["read"] = list(read)
        if exclusive:
            payload["collections"]["exclusive"] = list(exclusive)
        if wait_for_sync is not None:
            payload["waitForSync"] = bool(wait_for_sync)

        path = f"/_db/{self._config.database}/_api/transaction"
        try:
            return self._write_client.request("POST", path, json=payload)
        except ArangoHttpError as exc:  # pragma: no cover - thin wrapper
            raise self._wrap_error(exc) from exc

    def drop_collections(self, names: Iterable[str], *, ignore_missing: bool = True) -> None:
        for name in names:
            path = f"/_db/{self._config.database}/_api/collection/{name}"
            try:
                self._write_client.request("DELETE", path)
            except ArangoHttpError as exc:
                if ignore_missing and exc.status_code == 404:
                    continue
                raise self._wrap_error(exc) from exc

    def create_collections(self, definitions: Iterable[CollectionDefinition]) -> None:
        created: list[str] = []
        try:
            for definition in definitions:
                options = dict(definition.options or {})
                collection_type = 3 if definition.type.lower() == "edge" else 2
                options.setdefault("type", collection_type)
                options.setdefault("name", definition.name)
                path = f"/_db/{self._config.database}/_api/collection"

                try:
                    self._write_client.request("POST", path, json=options)
                except ArangoHttpError as exc:
                    if exc.status_code != 409:
                        raise
                else:
                    created.append(definition.name)

                if definition.indexes:
                    for index in definition.indexes:
                        index_path = f"/_db/{self._config.database}/_api/index"
                        params = {"collection": definition.name}
                        try:
                            self._write_client.request("POST", index_path, json=index, params=params)
                        except ArangoHttpError as exc:
                            if exc.status_code != 409:
                                raise
        except ArangoHttpError as exc:
            self._rollback_created(created)
            raise self._wrap_error(exc) from exc
        except Exception:
            self._rollback_created(created)
            raise

    def _rollback_created(self, created: Sequence[str]) -> None:
        """
        Drop the specified collections in reverse creation order.
        
        This performs a best-effort cleanup: each collection name from `created` (interpreted as creation order) is deleted in reverse order and any errors raised while dropping a collection are ignored.
        
        Parameters:
            created (Sequence[str]): Collection names in creation order; the function will attempt to drop them in reverse order.
        """
        for name in reversed(created):
            path = f"/_db/{self._config.database}/_api/collection/{name}"
            try:
                self._write_client.request("DELETE", path)
            except ArangoHttpError:
                # Suppress rollback errors; best-effort cleanup
                continue

    # Internal helpers --------------------------------------------------
    def _wrap_error(self, error: ArangoHttpError) -> ArangoClientError:
        """
        Convert an ArangoHttpError into an ArangoClientError suitable for gRPC-style handling.
        
        Returns:
        	ArangoClientError: an error whose status is the gRPC StatusCode corresponding to the HTTP status and whose message and details are taken from the original ArangoHttpError.
        """
        status = _http_status_to_grpc(error.status_code)
        message = error.details.get("errorMessage") or error.details.get("message") or str(error)
        return ArangoClientError(message, status=status, details=error.details)


__all__ = [
    "ArangoClient",
    "ArangoClientConfig",
    "CollectionDefinition",
    "ArangoClientError",
    "resolve_client_config",
]