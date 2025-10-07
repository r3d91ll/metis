# Unix Socket Permission Issue

## Problem
The arXiv import pipeline cannot connect to ArangoDB via Unix socket at `/run/arangodb3/arangodb.sock` due to permission denied errors.

## Current Status
- Socket exists: `/run/arangodb3/arangodb.sock`
- Permissions: `srwxr-xr-x 1 arangodb arangodb`
- User is in `arangodb` group: ✓
- Python socket connection: **FAILS with Permission Denied**

## Root Cause
Despite being in the `arangodb` group, Python's socket module cannot connect to the Unix socket. This suggests either:
1. Group membership not effective in current session (need to log out/in or use `newgrp`)
2. AppArmor/SELinux restrictions
3. Socket directory permissions issue
4. ArangoDB not listening on the socket (only TCP)

## Attempted Solutions
1. ✗ Direct socket connection via Python
2. ✗ Using `ARANGO_SKIP_AUTH=1`
3. ✗ Configuring `use_proxies=False`

## Current Workaround Needed
The metis library is designed for Unix sockets only (`ArangoHttp2Client` always uses `uds=` parameter). To proceed with the import, you need to either:

### Option 1: Fix Socket Permissions (Recommended)
```bash
# Try refreshing group membership
newgrp arangodb

# Or log out and back in to refresh groups

# Test socket access
python3 -c "import socket; s = socket.socket(socket.AF_UNIX); s.connect('/run/arangodb3/arangodb.sock'); print('OK')"
```

### Option 2: Modify Metis for TCP Support
Add TCP transport support to `metis/database/client.py`:

```python
# In ArangoHttp2Client.__init__
if config.socket_path:
    transport = httpx.HTTPTransport(uds=config.socket_path, ...)
else:
    # Use TCP
    transport = httpx.HTTPTransport(...)
```

### Option 3: Use Proxy Sockets
Set up the metis proxy sockets at `/run/metis/readonly/` and `/run/metis/readwrite/` with proper permissions.

## Next Steps
1. Determine why socket connection fails despite group membership
2. Either fix permissions OR add TCP support to metis
3. Resume arXiv import once database connection works

## Testing Socket Connection
```bash
# Test as current user
python3 -c "import socket; s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect('/run/arangodb3/arangodb.sock'); print('Connected')"

# Check effective groups
id
groups

# Test with newgrp
newgrp arangodb
python3 -c "import socket; s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect('/run/arangodb3/arangodb.sock'); print('Connected')"
```

## Pipeline Status
- ✓ arXiv ID parser implemented and tested (33/33 tests passing)
- ✓ Import pipeline implemented with Jina v4 embeddings
- ✓ Batch processing and streaming ready
- ✓ Database schema definition ready
- ✗ **BLOCKED**: Cannot connect to database via socket
- ✗ TCP fallback not supported by metis library

## Files Modified
- `experiments/arxiv_import/import_pipeline.py` - Main pipeline
- `experiments/arxiv_import/config/arxiv_import.yaml` - Configuration
- `experiments/arxiv_import/arxiv_parser.py` - ID parser
- `tests/test_arxiv_parser.py` - Parser tests
