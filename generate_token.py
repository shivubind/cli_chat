#!/usr/bin/env python3
"""
Standalone token generator for LiveKit
Generates valid JWT tokens with clock skew compensation

Usage:
    python generate_token.py <room> <identity>
    python generate_token.py vchat-room user1
"""
import sys
import json
import base64
import hmac
import hashlib
from livekit import api

LIVEKIT_API_KEY = "devkey"
LIVEKIT_API_SECRET = "secret"


def create_token_with_skew(room_name: str, identity: str) -> str:
    """Create access token with clock skew compensation (30 second buffer)"""
    
    # Generate initial JWT
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
    
    jwt_token = token.to_jwt()
    
    # Decode and modify nbf to subtract 30 seconds for clock skew compensation
    try:
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
        print(f"Warning: Failed to adjust token nbf: {e}")
        print("Using original token (may cause connection issues)")
    
    return jwt_token


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_token.py <room> <identity>")
        print("Example: python generate_token.py vchat-room user1")
        sys.exit(1)
    
    room = sys.argv[1]
    identity = sys.argv[2]
    
    token = create_token_with_skew(room, identity)
    
    print(f"Token for {identity} in room {room}:")
    print(token)
    
    # Also print decoded info
    print("\nToken details:")
    parts = token.split('.')
    payload = parts[1]
    padding = 4 - len(payload) % 4
    if padding and padding != 4:
        payload += '=' * padding
    decoded = json.loads(base64.urlsafe_b64decode(payload))
    print(json.dumps(decoded, indent=2))
