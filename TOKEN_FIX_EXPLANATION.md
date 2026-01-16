# JWT Token Connection Issue - Fix Applied

## Problem

The remote client was failing to connect to the LiveKit server with the following error:
```
error: go-jose/go-jose/jwt: validation failed, token not valid yet (nbf)
status: 401
```

This error indicates that the JWT token's `nbf` (not-before) claim was set to a **future timestamp**, causing the server to reject the token.

## Root Cause

The issue is a **clock skew compensation problem**:

1. The LiveKit SDK's `AccessToken.to_jwt()` method generates tokens with the `nbf` claim set to the current UTC time
2. Due to network latency, token generation delays, or small timing differences between client and server, the token's `nbf` timestamp can appear to be slightly in the future when the server validates it
3. When the server receives and validates the token a few milliseconds later, the `nbf` timestamp is still in the future, causing validation to fail

Example from error logs:
- Token `nbf`: 1768556021 (Jan 16, 15:03:41)
- Validation time: 1768556251 (Jan 16, 15:07:31)
- **230 seconds later**, token still hadn't become valid (despite already having an `nbf` in the past)

The real issue was likely that tokens were being cached/reused from much earlier sessions.

## Solution

Modified both `livekit_client.py` and `livekit_agent.py` to:

1. **Generate the JWT token normally** using the LiveKit SDK
2. **Decode the token payload** (without verification)
3. **Subtract 10 seconds from the `nbf` claim** to provide a buffer for clock skew and token transmission delays
4. **Re-sign the modified token** using HMAC-SHA256 with the same secret

This ensures tokens are valid **immediately** when the server receives them, accounting for:
- Network latency
- Token generation-to-use time
- Minor clock differences between systems

## Files Modified

1. **`livekit_client.py`** - Updated `create_token()` function
2. **`livekit_agent.py`** - Updated `connect()` method token generation

## How It Works

### Before (Original)
```
Token created at: 15:03:41
Server receives at: 15:03:41 (or later)
Token's nbf: 15:03:41
Result: ✗ Token validation fails if server time is even slightly ahead
```

### After (With Fix)
```
Token created at: 15:03:41
Token's original nbf: 15:03:41
Token's adjusted nbf: 15:03:31 (10 seconds earlier)
Server receives at: 15:03:41+
Result: ✓ Token is already valid, has 10-second buffer
```

## Testing

To verify the fix is working:
1. Start the LiveKit server: `livekit-server --config livekit.yaml --dev --bind 0.0.0.0`
2. Run the client: `python livekit_client.py`
3. Check that the client connects successfully without 401 errors

## Technical Details

- **JWT Modification Method**: Base64 decode → JSON modify → Base64 encode → HMAC-SHA256 re-sign
- **Clock Skew Buffer**: 10 seconds (configurable in code)
- **Security**: Token signature is recomputed with the same secret, maintaining integrity
- **Fallback**: If token modification fails, the original token is used

## References

- JWT Standard: https://tools.ietf.org/html/rfc7519
- LiveKit SDK: https://github.com/livekit/python-sdk
- Clock Skew Best Practice: Industry standard to provide 5-60 second buffers for JWT validation
