# Makefile for ArangoDB Unix Socket Proxies

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
GOMOD=$(GOCMD) mod

# Binary names
ROPROXY_BINARY=bin/roproxy
RWPROXY_BINARY=bin/rwproxy

# Build flags
LDFLAGS=-ldflags "-s -w"
BUILD_FLAGS=-trimpath $(LDFLAGS)

.PHONY: all build build-roproxy build-rwproxy clean test install uninstall tidy

all: build

# Build both proxies
build: build-roproxy build-rwproxy

# Build read-only proxy
build-roproxy:
	@echo "Building read-only proxy..."
	@mkdir -p bin
	$(GOBUILD) $(BUILD_FLAGS) -o $(ROPROXY_BINARY) ./cmd/roproxy
	@echo "Built: $(ROPROXY_BINARY)"

# Build read-write proxy
build-rwproxy:
	@echo "Building read-write proxy..."
	@mkdir -p bin
	$(GOBUILD) $(BUILD_FLAGS) -o $(RWPROXY_BINARY) ./cmd/rwproxy
	@echo "Built: $(RWPROXY_BINARY)"

# Clean build artifacts
clean:
	@echo "Cleaning..."
	$(GOCLEAN)
	rm -rf bin/
	@echo "Clean complete"

# Run tests
test:
	@echo "Running tests..."
	$(GOTEST) -v ./...
	@echo "Note: Add test files under cmd/ and package root to increase coverage"

# Install binaries to /usr/local/bin
install: build
	@echo "Installing binaries to /usr/local/bin..."
	install -m 755 $(ROPROXY_BINARY) /usr/local/bin/roproxy
	install -m 755 $(RWPROXY_BINARY) /usr/local/bin/rwproxy
	@echo "Installation complete"

# Uninstall binaries
uninstall:
	@echo "Uninstalling binaries..."
	rm -f /usr/local/bin/roproxy
	rm -f /usr/local/bin/rwproxy
	@echo "Uninstall complete"

# Tidy dependencies
tidy:
	@echo "Tidying Go modules..."
	$(GOMOD) tidy
	@echo "Tidy complete"

# Download dependencies
deps:
	@echo "Downloading dependencies..."
	$(GOMOD) download
	@echo "Dependencies downloaded"

# Show help
help:
	@echo "Metis ArangoDB Proxy Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make build          - Build both proxies"
	@echo "  make build-roproxy  - Build read-only proxy"
	@echo "  make build-rwproxy  - Build read-write proxy"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make test           - Run tests"
	@echo "  make install        - Install binaries to /usr/local/bin (requires sudo)"
	@echo "  make uninstall      - Remove installed binaries (requires sudo)"
	@echo "  make tidy           - Tidy Go modules"
	@echo "  make deps           - Download dependencies"
	@echo "  make help           - Show this help message"
