# IMPORTANT: Token Fix for Remote Client Connection

The remote client is failing because the token fix hasn't been applied to `/home/shivani/cli_chat/livekit_client.py`.

## Solution

Replace the `create_token()` function in your remote client's `livekit_client.py` with this:

```python
def create_token(room_name: str, identity: str) -> str:
    """Create access token for room with clock skew compensation"""
    import json
    import base64
    import hmac
    import hashlib
    import time
    
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity(identity)
    token.with_name(identity)
    token.with_grants(api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_sources=["camera", "microphone"],
    ))
    
    # Generate initial JWT
    jwt_token = token.to_jwt()
    
    # Decode and modify nbf to subtract 30 seconds for clock skew compensation
    try:
        # Decode without verification to modify the claims
        parts = jwt_token.split('.')
        payload = parts[1]
        # Add padding if needed
        padding = 4 - len(payload) % 4
        if padding and padding != 4:
            payload += '=' * padding
        
        decoded_payload = json.loads(base64.urlsafe_b64decode(payload))
        
        # Subtract 30 seconds from nbf to account for clock skew
        if 'nbf' in decoded_payload:
            decoded_payload['nbf'] -= 30
        
        # Re-encode
        modified_payload = base64.urlsafe_b64encode(
            json.dumps(decoded_payload, separators=(',', ':')).encode()
        ).decode().rstrip('=')
        
        # Re-sign with the secret using HMAC-SHA256
        header_part = parts[0]
        message = f"{header_part}.{modified_payload}"
        signature = base64.urlsafe_b64encode(
            hmac.new(
                LIVEKIT_API_SECRET.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
        ).decode().rstrip('=')
        
        jwt_token = f"{message}.{signature}"
    except Exception as e:
        print(f"Warning: Failed to adjust token nbf: {e}, using original token")
    
    return jwt_token
```

## What This Does

- **Generates JWT tokens** using the LiveKit SDK
- **Modifies the `nbf` (not-before) claim** by subtracting 30 seconds
- **Re-signs the token** with HMAC-SHA256 to maintain integrity
- **Accounts for network delays** and clock skew between client and server

## Key Changes from Original

1. Increased clock skew buffer from 10 to 30 seconds (more robust for remote connections)
2. Simplified error handling
3. Added better comments

## Testing

After updating the file, try:
```bash
python livekit_client.py --room vchat-room --identity remote-user
```

Should now connect successfully instead of getting "401 Unauthorized" or "token not valid yet" errors.
