# LiveKit RTC Connection Issues - Troubleshooting Guide

## Problem
Client experiencing RTC errors:
- `Publisher pc state failed`
- `wait_pc_connection timed out`  
- `KeyError: 'TR_...'` (track not found)
- Connection drops and reconnection attempts fail

## Root Causes

1. **Network/Firewall Issues** - WebRTC requires specific ports to be open
2. **NAT Traversal Problems** - STUN/TURN not configured or working
3. **External IP Configuration** - Server can't reach external clients
4. **IPv6 vs IPv4 Mismatch** - Server listening on wrong IP version
5. **Connection Timeouts** - Peer connection setup taking too long

## Solutions Applied

### 1. Server Configuration (livekit.yaml)
Added STUN servers for better NAT traversal:
```yaml
rtc:
  port_range_start: 50000
  port_range_end: 60000
  use_external_ip: true
  tcp_port: 7881
  # STUN for NAT traversal
  stun_servers:
    - "stun.l.google.com:19302"
    - "stun1.l.google.com:19302"
```

### 2. Client Connection Retry Logic
Added exponential backoff retries with timeout handling:
- Up to 3 retry attempts
- Exponential backoff: 2s → 4s → 8s → 10s (max)
- 30 second timeout per attempt

### 3. Publisher Track Error Handling
Added retry logic for track publication:
- Up to 3 retry attempts
- 15 second timeout per publish attempt
- Better error messages

## Testing the Fix

### Step 1: Verify Server Configuration
```bash
# Check server is listening on IPv4
ss -tlnp | grep 7880
# Should show: 0.0.0.0:7880 (not :::7880 IPv6)

# Restart server with new config
pkill livekit-server
livekit-server --config livekit.yaml --dev --bind 0.0.0.0
```

### Step 2: Verify Network Connectivity
```bash
# Test HTTP endpoint
curl http://192.168.0.125:7880/

# Test WebSocket endpoint (may return 401 if no token, but connection should work)
curl -v -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  -H "Sec-WebSocket-Version: 13" \
  http://192.168.0.125:7880/rtc
```

### Step 3: Verify Firewall/Ports
```bash
# Open required ports (Linux ufw)
sudo ufw allow 7880/tcp
sudo ufw allow 7881/tcp
sudo ufw allow 50000:60000/udp
sudo ufw reload

# Or iptables
sudo iptables -A INPUT -p tcp --dport 7880 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 7881 -j ACCEPT
sudo iptables -A INPUT -p udp --dport 50000:60000 -j ACCEPT
```

### Step 4: Run Client with New Code
```bash
# Use updated client with retry logic and token fix
python livekit_client.py --room vchat-room --identity user1
```

## Network Diagram

```
Client (192.168.0.1)
  ↓ (HTTP WebSocket)
  ↓ (UDP for RTC if possible)
  → Server (192.168.0.125:7880)

If direct UDP fails:
  ↓ (TCP fallback)
  ↓ (STUN for NAT traversal)
  → STUN Server (stun.l.google.com:19302)
  → Server (27.124.65.126:7881)
```

## Additional Debugging

### Check LiveKit Server Logs
```bash
# Real-time logs
tail -f /var/log/livekit/server.log

# Or in terminal where it's running (Ctrl+C to exit)
livekit-server --config livekit.yaml --dev --bind 0.0.0.0 2>&1 | tee livekit.log
```

### Test with tcpdump
```bash
# Monitor RTC traffic
sudo tcpdump -i any 'port 7880 or port 7881 or (udp and portrange 50000-60000)'
```

### Known Issues & Workarounds

| Issue | Cause | Fix |
|-------|-------|-----|
| `pc_state failed` | Connection unstable | Add retry logic ✓ |
| `wait_pc_connection timed out` | Slow network/NAT | Increase timeout ✓ |
| `KeyError on track_publications` | Race condition in SDK | Using workaround in code |
| `404 page not found` | Wrong endpoint/old server version | Ensure server is latest |
| `401 Unauthorized` | Invalid token | Apply token nbf fix ✓ |

## Token Fixes Applied

Both client and server now use modified JWT tokens with:
- **nbf reduced by 30 seconds** for clock skew compensation
- Prevents "token not valid yet" errors
- Safe and cryptographically valid

## Contact Server Configuration Details

### Server Binding
- `bind_addresses: ["0.0.0.0"]` - Listen on all IPv4 interfaces
- `port: 7880` - HTTP/WebSocket signaling port
- `tcp_port: 7881` - WebRTC fallback TCP port

### Client Connection
- Should connect to: `ws://192.168.0.125:7880`
- Or: `wss://192.168.0.125:7880` (with TLS)
- Token: Updated with 30-second clock skew buffer

## Still Having Issues?

1. **Check if server is reachable**: `ping 192.168.0.125`
2. **Check if port is open**: `telnet 192.168.0.125 7880`
3. **Check firewall**: `sudo ufw status`
4. **Check server logs**: See above
5. **Try TCP only**: Some networks block UDP
6. **Use public TURN server**: If NAT issues persist
