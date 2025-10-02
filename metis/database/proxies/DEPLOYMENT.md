# Metis Proxy Deployment Instructions

## Status

✅ **Phase 1-3 Complete**: Code updated, systemd files created, binaries built

## What's Ready

- ✅ Updated code defaults (`/run/metis/` instead of `/run/hades/`)
- ✅ Updated README documentation
- ✅ Built new proxy binaries (`bin/roproxy`, `bin/rwproxy`)
- ✅ Created systemd service files (`metis-roproxy.service`, `metis-rwproxy.service`)
- ✅ Created deployment script (`deploy-metis-proxies.sh`)
- ✅ Metis user created

## Next Steps

### Option 1: Automated Deployment (Recommended)

**Note:** Binaries are already built! The script will verify and deploy them.

Run the deployment script as root from your repository root:

```bash
cd metis/database/proxies
sudo ./deploy-metis-proxies.sh
```

This script will:

1. ✓ Skip user creation (already done)
2. Verify binaries exist (already built in `bin/` directory)
3. Install binaries to `/usr/local/bin/metis-{roproxy,rwproxy}`
4. Install systemd service files
5. Enable and start services
6. Verify socket creation
7. Stop and disable old HADES services
8. Clean up old infrastructure
9. Backup old binaries with `.bak` extension

### Option 2: Manual Step-by-Step Deployment

If you prefer manual control, first set your repository root and navigate to the proxies directory:

```bash
# Set REPO_ROOT to your repository clone location
export REPO_ROOT=/path/to/your/repo  # Update this to your actual repo path
cd $REPO_ROOT/metis/database/proxies

# 1. Install binaries
sudo install -m 755 bin/roproxy /usr/local/bin/metis-roproxy
sudo install -m 755 bin/rwproxy /usr/local/bin/metis-rwproxy

# 2. Install systemd service files
sudo install -m 644 metis-roproxy.service /etc/systemd/system/
sudo install -m 644 metis-rwproxy.service /etc/systemd/system/
sudo systemctl daemon-reload

# 3. Enable and start new services
sudo systemctl enable metis-roproxy.service metis-rwproxy.service
sudo systemctl start metis-roproxy.service metis-rwproxy.service

# 4. Verify sockets were created
ls -la /run/metis/{readonly,readwrite}/arangod.sock

# 5. Stop old HADES services
sudo systemctl stop hades-roproxy.service hades-rwproxy.service
sudo systemctl disable hades-roproxy.service hades-rwproxy.service

# 6. Remove old service files
sudo rm /etc/systemd/system/hades-roproxy.service
sudo rm /etc/systemd/system/hades-rwproxy.service
sudo systemctl daemon-reload

# 7. Backup old binaries
sudo mv /usr/local/bin/hades-roproxy /usr/local/bin/hades-roproxy.bak
sudo mv /usr/local/bin/hades-rwproxy /usr/local/bin/hades-rwproxy.bak
sudo rm /usr/local/bin/hades-arango-{ro,rw}-proxy

# 8. Clean up old socket directories
sudo rm -rf /run/hades/
```

## Verification

After deployment, verify everything is working:

```bash
# Check service status
systemctl status metis-roproxy.service
systemctl status metis-rwproxy.service

# Check socket permissions
ls -la /run/metis/{readonly,readwrite}/arangod.sock
# Expected: srw-rw---- metis arangodb

# Test read-only proxy
curl --unix-socket /run/metis/readonly/arangod.sock http://localhost/_api/version

# Test read-write proxy
curl --unix-socket /run/metis/readwrite/arangod.sock http://localhost/_api/version

# View logs
journalctl -fu metis-roproxy.service
journalctl -fu metis-rwproxy.service
```

## Expected Results

### Sockets

```bash
$ ls -la /run/metis/{readonly,readwrite}/arangod.sock
srw-rw---- 1 metis arangodb 0 Oct  1 21:00 /run/metis/readonly/arangod.sock
srw-rw---- 1 metis arangodb 0 Oct  1 21:00 /run/metis/readwrite/arangod.sock
```

### Services

```bash
$ systemctl status metis-roproxy.service
● metis-roproxy.service - Metis ArangoDB Read-Only Proxy (UDS)
     Loaded: loaded (/etc/systemd/system/metis-roproxy.service; enabled)
     Active: active (running)

$ systemctl status metis-rwproxy.service
● metis-rwproxy.service - Metis ArangoDB Read-Write Proxy (UDS)
     Loaded: loaded (/etc/systemd/system/metis-rwproxy.service; enabled)
     Active: active (running)
```

## Rollback Plan

If you need to rollback:

```bash
# Stop new services
sudo systemctl stop metis-roproxy.service metis-rwproxy.service
sudo systemctl disable metis-roproxy.service metis-rwproxy.service

# Restore old binaries
sudo mv /usr/local/bin/hades-roproxy.bak /usr/local/bin/hades-roproxy
sudo mv /usr/local/bin/hades-rwproxy.bak /usr/local/bin/hades-rwproxy

# Restore old service files (you'll need to recreate them)
# Or restore from backup if you made one

# Start old services
sudo systemctl start hades-roproxy.service hades-rwproxy.service
```

## Key Differences from HADES

| Aspect | HADES (Old) | Metis (New) |
|--------|-------------|-------------|
| **User** | arangodb | metis |
| **Group** | hades | arangodb |
| **Binary names** | hades-{roproxy,rwproxy} | metis-{roproxy,rwproxy} |
| **Socket paths** | /run/hades/* | /run/metis/* |
| **RO permissions** | 0660 | 0660 |
| **RW permissions** | 0600 | 0660 |

## Troubleshooting

### Services won't start

```bash
# Check logs
journalctl -xe -u metis-roproxy.service
journalctl -xe -u metis-rwproxy.service

# Check metis user exists and is in arangodb group
id metis
# Expected: uid=XXX(metis) gid=XXX(metis) groups=XXX(metis),986(arangodb)
```

### Socket permissions wrong

```bash
# Fix manually if needed
sudo chown metis:arangodb /run/metis/readonly/arangod.sock
sudo chown metis:arangodb /run/metis/readwrite/arangod.sock
sudo chmod 0660 /run/metis/readonly/arangod.sock
sudo chmod 0660 /run/metis/readwrite/arangod.sock
```

### Can't connect to sockets

```bash
# Verify your user is in the arangodb group
groups
# If not: sudo usermod -aG arangodb $USER
# Then logout and login again
```

## Files Modified

### Code Changes

- `metis/database/proxies/proxy_common.go` - Updated socket paths
- `metis/database/proxies/README.md` - Updated documentation

### Files Created

- `metis/database/proxies/metis-roproxy.service` - Systemd service for RO proxy
- `metis/database/proxies/metis-rwproxy.service` - Systemd service for RW proxy
- `metis/database/proxies/deploy-metis-proxies.sh` - Automated deployment script
- `metis/database/proxies/bin/roproxy` - Built RO proxy binary
- `metis/database/proxies/bin/rwproxy` - Built RW proxy binary
- `metis/database/proxies/DEPLOYMENT.md` - This file

## Support

For issues, check:

1. Service logs: `journalctl -u metis-roproxy.service`
2. Binary version: `/usr/local/bin/metis-roproxy --version` (if supported)
3. ArangoDB status: `systemctl status arangodb3.service`
4. Upstream socket: `ls -la /run/arangodb3/arangodb.sock`
