# ArangoDB Unix Socket Proxies

This directory contains Go-based security proxies for ArangoDB Unix socket access.

## Overview

The proxies provide a security layer between client applications and ArangoDB, enforcing access control policies at the socket level:

- **Read-Only Proxy** (`roproxy`): Allows only read operations and AQL queries without mutation keywords
- **Read-Write Proxy** (`rwproxy`): Allows document CRUD operations but blocks admin endpoints

## Architecture

```
Client Application → Unix Socket Proxy → ArangoDB Unix Socket
                    (security layer)
```

### Read-Only Proxy

**Default sockets**:
- Listen: `/run/metis/readonly/arangod.sock` (0660 permissions, group access)
- Upstream: `/run/arangodb3/arangodb.sock`

**Allowed operations**:
- GET, HEAD, OPTIONS (all paths)
- POST to `/_api/cursor` (AQL queries only, no mutation keywords)
- DELETE to `/_api/cursor/{id}` (cursor cleanup)

**Forbidden**:
- AQL mutation keywords: INSERT, UPDATE, REPLACE, REMOVE, UPSERT (detected using word-boundary matching)
- PUT to `/_api/cursor` (deprecated by ArangoDB; use POST instead)
- Non-GET methods on non-cursor endpoints

### Read-Write Proxy

**Default sockets**:
- Listen: `/run/metis/readwrite/arangod.sock` (0660 permissions, group access in dev; 0600 owner-only in prod)
- Upstream: `/run/arangodb3/arangodb.sock`

**Allowed operations**:
- All read-only operations
- POST, PUT, PATCH, DELETE to DB-scoped endpoints: `/_db/{name}/_api/{document|index|collection|import|cursor}`

**Forbidden**:
- Admin endpoints: `/_api/database`, `/_api/view`, `/_api/analyzer`
- Server-level operations outside DB scope

## Building

### Prerequisites

- Go 1.23.0 or later
- Make (optional, for convenience)

### Build Commands

```bash
# Build both proxies
make build

# Build individual proxies
make build-roproxy
make build-rwproxy

# Install to /usr/local/bin (requires sudo)
sudo make install

# Clean build artifacts
make clean
```

### Manual Build

```bash
# Read-only proxy
go build -o bin/roproxy ./cmd/roproxy

# Read-write proxy
go build -o bin/rwproxy ./cmd/rwproxy
```

## Running

### Standalone

```bash
# Read-only proxy
./bin/roproxy

# Read-write proxy
./bin/rwproxy
```

### With Environment Variables

```bash
# Override listen socket
LISTEN_SOCKET=/custom/path.sock ./bin/roproxy

# Override upstream socket
UPSTREAM_SOCKET=/var/run/arangodb.sock ./bin/rwproxy

# Set timeouts
PROXY_CLIENT_TIMEOUT_SECONDS=60 ./bin/roproxy
PROXY_DIAL_TIMEOUT_SECONDS=5 ./bin/rwproxy

# Set AQL body peek limit (read-only proxy)
AQL_PEEK_LIMIT_BYTES=262144 ./bin/roproxy
```

### Systemd Service

Example systemd unit files:

#### `/etc/systemd/system/metis-roproxy.service`

```ini
[Unit]
Description=Metis ArangoDB Read-Only Proxy
After=arangodb3.service
Requires=arangodb3.service

[Service]
Type=simple
User=metis
Group=arangodb
Environment="LISTEN_SOCKET=/run/metis/readonly/arangod.sock"
Environment="UPSTREAM_SOCKET=/run/arangodb3/arangodb.sock"
Environment="PROXY_CLIENT_TIMEOUT_SECONDS=120"
RuntimeDirectory=metis/readonly
RuntimeDirectoryMode=0750
ExecStart=/usr/local/bin/metis-roproxy
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

#### `/etc/systemd/system/metis-rwproxy.service`

```ini
[Unit]
Description=Metis ArangoDB Read-Write Proxy
After=arangodb3.service
Requires=arangodb3.service

[Service]
Type=simple
User=metis
Group=arangodb
Environment="LISTEN_SOCKET=/run/metis/readwrite/arangod.sock"
Environment="UPSTREAM_SOCKET=/run/arangodb3/arangodb.sock"
Environment="PROXY_CLIENT_TIMEOUT_SECONDS=120"
RuntimeDirectory=metis/readwrite
RuntimeDirectoryMode=0770
ExecStart=/usr/local/bin/metis-rwproxy
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable metis-roproxy metis-rwproxy
sudo systemctl start metis-roproxy metis-rwproxy
```

## Configuration

### Socket Permissions

- **Read-Only**: 0660 (group `hades` can connect)
- **Read-Write**: 0660 (dev), 0600 (production - owner only)

### Timeouts

- **Client Timeout**: Default 120s, set via `PROXY_CLIENT_TIMEOUT_SECONDS`
- **Dial Timeout**: Default 10s, set via `PROXY_DIAL_TIMEOUT_SECONDS`
- **Disable Timeout**: Set to `0` to disable

### AQL Peek Limit

The read-only proxy inspects POST request bodies to cursor endpoints to detect mutation keywords. By default, it peeks at the first 128 KiB of the body.

- Override via `AQL_PEEK_LIMIT_BYTES` environment variable
- Increase for large queries, decrease to save memory

## Security Considerations

1. **Socket Permissions**: Ensure socket files have appropriate permissions for your deployment
2. **Upstream Socket**: Protect the upstream ArangoDB socket with restrictive permissions
3. **Read-Only Enforcement**: The proxy performs keyword scanning but is not foolproof - defense in depth recommended
4. **Admin Endpoints**: Never expose admin endpoints through the RW proxy
5. **Group Access**: In production, consider 0600 permissions for RW proxy socket

## Performance

- **Overhead**: ~0.2ms p50 additional latency
- **HTTP/2**: Multiplexed connections reduce overhead
- **Keep-Alive**: Persistent upstream connections minimize dial latency

## Development

### Project Structure

```
proxies/
├── proxy_common.go     # Shared reverse proxy logic
├── ro_proxy.go         # Read-only policy enforcement
├── rw_proxy.go         # Read-write policy enforcement
├── cmd/
│   ├── roproxy/        # RO proxy entry point
│   └── rwproxy/        # RW proxy entry point
├── go.mod              # Go module definition
├── go.sum              # Dependency checksums
├── Makefile            # Build automation
└── README.md           # This file
```

### Testing

```bash
# Test read-only proxy
curl --unix-socket /run/metis/readonly/arangod.sock http://localhost/_api/version

# Test mutation blocking
curl --unix-socket /run/metis/readonly/arangod.sock \
  -X POST http://localhost/_db/test/_api/cursor \
  -d '{"query": "INSERT {} INTO docs"}' \
  # Should return 403 Forbidden

# Test read-write proxy
curl --unix-socket /run/metis/readwrite/arangod.sock \
  -X POST http://localhost/_db/test/_api/document/docs \
  -d '{"_key": "test", "data": "value"}'
```

## Troubleshooting

### Socket Permission Errors

```
failed to listen on /run/hades/readonly/arangod.sock: permission denied
```

Solution: Ensure parent directory exists and has correct permissions:

```bash
sudo mkdir -p /run/metis/readonly
sudo chown metis:arangodb /run/metis/readonly
sudo chmod 750 /run/metis/readonly
```

### Connection Refused

```
upstream error: dial unix /run/arangodb3/arangodb.sock: connect: connection refused
```

Solution: Verify ArangoDB is running and socket path is correct:

```bash
sudo systemctl status arangodb3
ls -la /run/arangodb3/arangodb.sock
```

### Timeout Errors

```
upstream error: context deadline exceeded
```

Solution: Increase timeout or check ArangoDB responsiveness:

```bash
PROXY_CLIENT_TIMEOUT_SECONDS=300 ./bin/roproxy
```

## License

Apache 2.0 - See LICENSE file for details
