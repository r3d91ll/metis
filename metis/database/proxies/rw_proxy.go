package proxies

import (
    "fmt"
    "log"
    "net"
    "net/http"
    "regexp"
)

// RunReadWriteProxy starts a Unix domain socket proxy that forwards read-write HTTP requests
// from the configured listen socket to the upstream socket with access control.
//
// It sets up the listening socket with appropriate permissions, removes any existing socket file,
// and logs the proxy operation. It returns an error if any step in preparation or serving fails.
func RunReadWriteProxy() error {
	listenSocket := getEnv("LISTEN_SOCKET", defaultRWListenSocket)
	upstreamSocket := getEnv("UPSTREAM_SOCKET", defaultUpstreamSocket)

	if err := ensureParentDir(listenSocket); err != nil {
		return fmt.Errorf("failed to prepare directory for %s: %w", listenSocket, err)
	}
	removeIfExists(listenSocket)

	proxy := newUnixReverseProxy(upstreamSocket, allowReadWrite)

	listener, err := net.Listen("unix", listenSocket)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", listenSocket, err)
	}
	ensureSocketMode(listenSocket, rwSocketPermissions)

	server := &http.Server{Handler: logRequests(proxy)}

	log.Printf("Read-write proxy listening on %s -> %s", listenSocket, upstreamSocket)
	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("proxy server error: %w", err)
	}
	return nil
}

var dbScopedRW = regexp.MustCompile(`^/_db/[^/]+/_api/(document|index|collection|import|cursor)(/|$)`) // DB-scoped only
var databaseAdminRW = regexp.MustCompile(`^/_api/database(/[^/]+)?$`) // allowReadWrite enforces read-write access rules for incoming proxy requests.
// It permits requests already allowed by allowReadOnly; allows database administration operations (POST to /_api/database to create a database and DELETE to /_api/database/{name} to drop a database); and permits mutation methods (POST, PUT, PATCH, DELETE) only for database-scoped endpoints. If a request is not permitted, it returns an error describing the disallowed method or path.

func allowReadWrite(r *http.Request, peek BodyPeeker) error {
    // Permit safe RO requests as-is (GET, HEAD, OPTIONS, and safe cursor operations)
    if err := allowReadOnly(r, peek); err == nil {
        return nil
    }

    path := r.URL.Path

    // Allow database management operations (create, drop, list databases)
    if databaseAdminRW.MatchString(path) {
        switch r.Method {
        case http.MethodPost:
            // POST /_api/database - create database
            return nil
        case http.MethodDelete:
            // DELETE /_api/database/{name} - drop database
            return nil
        }
        return fmt.Errorf("method %s not permitted on database admin endpoint %s", r.Method, path)
    }

    // For mutations (POST, PUT, PATCH, DELETE), enforce DB-scoped endpoints only
    if !dbScopedRW.MatchString(path) {
        // Explicitly forbid other admin endpoints like /_api/view, /_api/analyzer
        return fmt.Errorf("path not permitted: %s (DB-scoped endpoints only)", path)
    }

    switch r.Method {
    case http.MethodPost:
        // POST is allowed for cursor/import/document/collection/index under DB scope
        return nil
    case http.MethodPut, http.MethodPatch, http.MethodDelete:
        // Mutations to document/collection/index are allowed; cursor DELETE allowed for cursor close
        return nil
    }
    return fmt.Errorf("method %s not permitted on %s", r.Method, path)
}