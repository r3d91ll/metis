#!/bin/bash
# Deployment script for Metis ArangoDB Proxies
# Migrates from HADES to METIS infrastructure

set -e

echo "==================================================================="
echo "Metis ArangoDB Proxy Deployment Script"
echo "==================================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
    exit 1
fi

echo -e "${YELLOW}Phase 1: Creating metis user and group${NC}"
if ! getent group metis > /dev/null 2>&1; then
    groupadd -r metis
    echo "  ✓ Created metis group"
else
    echo "  → metis group already exists"
fi

if ! id -u metis > /dev/null 2>&1; then
    useradd -r -g metis -G arangodb -s /sbin/nologin -c "Metis service account" metis
    echo "  ✓ Created metis user"
else
    echo "  → metis user already exists"
    # Ensure metis is in arangodb group
    usermod -aG arangodb metis
    echo "  ✓ Added metis to arangodb group"
fi

echo ""
echo -e "${YELLOW}Phase 2: Verifying proxy binaries${NC}"
# Change to script directory (handles symlinks and arbitrary invocation paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || {
    echo -e "  ${RED}Error: Failed to determine script directory${NC}"
    exit 1
}
cd "$SCRIPT_DIR" || {
    echo -e "  ${RED}Error: Failed to cd to script directory: $SCRIPT_DIR${NC}"
    exit 1
}

# Check if binaries exist
if [ ! -f "bin/roproxy" ] || [ ! -f "bin/rwproxy" ]; then
    echo -e "  ${RED}Error: Proxy binaries not found in bin/ directory${NC}"
    echo "  Please build them first as your regular user:"
    echo "    cd $SCRIPT_DIR"
    echo "    make build"
    exit 1
fi

echo "  ✓ Proxy binaries found"
ls -lh bin/roproxy bin/rwproxy

echo ""
echo -e "${YELLOW}Phase 3: Installing binaries${NC}"
install -m 755 bin/roproxy /usr/local/bin/metis-roproxy
install -m 755 bin/rwproxy /usr/local/bin/metis-rwproxy
echo "  ✓ Installed /usr/local/bin/metis-roproxy"
echo "  ✓ Installed /usr/local/bin/metis-rwproxy"

echo ""
echo -e "${YELLOW}Phase 4: Installing systemd service files${NC}"
install -m 644 metis-roproxy.service /etc/systemd/system/metis-roproxy.service
install -m 644 metis-rwproxy.service /etc/systemd/system/metis-rwproxy.service
systemctl daemon-reload
echo "  ✓ Installed systemd service files"

echo ""
echo -e "${YELLOW}Phase 5: Enabling and starting services${NC}"
systemctl enable metis-roproxy.service metis-rwproxy.service
systemctl start metis-roproxy.service metis-rwproxy.service
echo "  ✓ Enabled and started metis-roproxy.service"
echo "  ✓ Enabled and started metis-rwproxy.service"

echo ""
echo -e "${YELLOW}Phase 6: Verifying new sockets${NC}"
sleep 2  # Give services time to create sockets
if [ -S /run/metis/readonly/arangod.sock ]; then
    ls -la /run/metis/readonly/arangod.sock
    echo "  ✓ Read-only socket created"
else
    echo -e "  ${RED}✗ Read-only socket not found${NC}"
    exit 1
fi

if [ -S /run/metis/readwrite/arangod.sock ]; then
    ls -la /run/metis/readwrite/arangod.sock
    echo "  ✓ Read-write socket created"
else
    echo -e "  ${RED}✗ Read-write socket not found${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Phase 7: Stopping old HADES services${NC}"
if systemctl is-active --quiet hades-roproxy.service; then
    systemctl stop hades-roproxy.service
    systemctl disable hades-roproxy.service
    echo "  ✓ Stopped and disabled hades-roproxy.service"
else
    echo "  → hades-roproxy.service not running"
fi

if systemctl is-active --quiet hades-rwproxy.service; then
    systemctl stop hades-rwproxy.service
    systemctl disable hades-rwproxy.service
    echo "  ✓ Stopped and disabled hades-rwproxy.service"
else
    echo "  → hades-rwproxy.service not running"
fi

echo ""
echo -e "${YELLOW}Phase 8: Cleaning up old HADES infrastructure${NC}"
rm -f /etc/systemd/system/hades-roproxy.service
rm -f /etc/systemd/system/hades-rwproxy.service
systemctl daemon-reload
echo "  ✓ Removed old service files"

# Keep old binaries as backup with .bak extension
if [ -f /usr/local/bin/hades-roproxy ]; then
    mv /usr/local/bin/hades-roproxy /usr/local/bin/hades-roproxy.bak
    echo "  ✓ Backed up hades-roproxy to hades-roproxy.bak"
fi

if [ -f /usr/local/bin/hades-rwproxy ]; then
    mv /usr/local/bin/hades-rwproxy /usr/local/bin/hades-rwproxy.bak
    echo "  ✓ Backed up hades-rwproxy to hades-rwproxy.bak"
fi

if [ -f /usr/local/bin/hades-arango-ro-proxy ]; then
    rm /usr/local/bin/hades-arango-ro-proxy
    echo "  ✓ Removed old hades-arango-ro-proxy"
fi

if [ -f /usr/local/bin/hades-arango-rw-proxy ]; then
    rm /usr/local/bin/hades-arango-rw-proxy
    echo "  ✓ Removed old hades-arango-rw-proxy"
fi

# Remove old socket directories
if [ -d /run/hades ]; then
    rm -rf /run/hades
    echo "  ✓ Removed /run/hades directory"
fi

echo ""
echo -e "${GREEN}==================================================================="
echo "Deployment Complete!"
echo "===================================================================${NC}"
echo ""
echo "New services:"
echo "  • metis-roproxy.service  - Read-only proxy"
echo "  • metis-rwproxy.service  - Read-write proxy"
echo ""
echo "Socket paths:"
echo "  • /run/metis/readonly/arangod.sock   (0660, metis:arangodb)"
echo "  • /run/metis/readwrite/arangod.sock  (0660, metis:arangodb)"
echo ""
echo "Check status with:"
echo "  systemctl status metis-roproxy.service"
echo "  systemctl status metis-rwproxy.service"
echo ""
echo "View logs with:"
echo "  journalctl -fu metis-roproxy.service"
echo "  journalctl -fu metis-rwproxy.service"
echo ""
echo -e "${YELLOW}Old HADES binaries backed up with .bak extension${NC}"
echo "Remove backups manually when confident: rm /usr/local/bin/hades-*proxy.bak"
echo ""
