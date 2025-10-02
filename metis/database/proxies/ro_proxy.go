package proxies

import (
    "encoding/json"
    "fmt"
    "log"
    "net"
    "net/http"
    "regexp"
    "strings"
    "strconv"
)

func RunReadOnlyProxy() error {
	listenSocket := getEnv("LISTEN_SOCKET", defaultROListenSocket)
	upstreamSocket := getEnv("UPSTREAM_SOCKET", defaultUpstreamSocket)

	if err := ensureParentDir(listenSocket); err != nil {
		return fmt.Errorf("failed to prepare directory for %s: %w", listenSocket, err)
	}
	removeIfExists(listenSocket)

	proxy := newUnixReverseProxy(upstreamSocket, allowReadOnly)

	listener, err := net.Listen("unix", listenSocket)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", listenSocket, err)
	}
	ensureSocketMode(listenSocket, roSocketPermissions)

	server := &http.Server{Handler: logRequests(proxy)}

	log.Printf("Read-only proxy listening on %s -> %s", listenSocket, upstreamSocket)
	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("proxy server error: %w", err)
	}
	return nil
}

func allowReadOnly(r *http.Request, peek BodyPeeker) error {
	switch r.Method {
	case http.MethodGet, http.MethodHead, http.MethodOptions:
		return nil
	case http.MethodPost:
		if isCursorPath(r.URL.Path) {
			// Enforce a configurable peek limit to avoid scanning arbitrarily large bodies.
			// Default 128 KiB; override via AQL_PEEK_LIMIT_BYTES.
			peekLimit := int64(128 * 1024)
			if v := getEnv("AQL_PEEK_LIMIT_BYTES", ""); v != "" {
				if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
					peekLimit = n
				}
			}
			body, err := peek(peekLimit)
			if err != nil {
				return err
			}
			var payload struct {
				Query string `json:"query"`
			}
			if err := json.Unmarshal(body, &payload); err == nil && payload.Query != "" {
				// Check for AQL mutation keywords using word boundaries
				// Only check for actual AQL DML verbs: INSERT, UPDATE, REPLACE, REMOVE, UPSERT
				upper := strings.ToUpper(payload.Query)
				forbiddenKeywords := []string{"INSERT", "UPDATE", "REPLACE", "REMOVE", "UPSERT"}
				for _, keyword := range forbiddenKeywords {
					// Use word boundary check to avoid false positives in strings/identifiers
					pattern := regexp.MustCompile(`\b` + keyword + `\b`)
					if pattern.MatchString(upper) {
						return fmt.Errorf("forbidden AQL mutation keyword %q detected", keyword)
					}
				}
				return nil
			}
			// Fallback: conservative scan of raw body with word boundaries
			upper := strings.ToUpper(string(body))
			forbiddenKeywords := []string{"INSERT", "UPDATE", "REPLACE", "REMOVE", "UPSERT"}
			for _, keyword := range forbiddenKeywords {
				pattern := regexp.MustCompile(`\b` + keyword + `\b`)
				if pattern.MatchString(upper) {
					return fmt.Errorf("forbidden AQL mutation keyword %q detected in request body", keyword)
				}
			}
			return nil
		}
	case http.MethodDelete:
		// DELETE is allowed for cursor cleanup (/_api/cursor/{id})
		if isCursorPath(r.URL.Path) {
			return nil
		}
	}
	return fmt.Errorf("method %s not permitted on %s", r.Method, r.URL.Path)
}
