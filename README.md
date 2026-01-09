# VChat - WebRTC Audio Streaming

WebRTC audio caller with AI-powered server.

## Files

- `signaling_server.py` - Exchanges connection info between peers
- `caller.py` - Audio caller (initiates calls)
- `smart_server.py` - AI server with STT + LLM + TTS

## Install

**For caller only (minimal):**
```bash
pip install -r requirements-caller.txt
```

**For smart server (full AI features):**
```bash
pip install -r requirements.txt
```

On Linux, also install:
```bash
sudo apt-get install libavdevice-dev libavfilter-dev portaudio19-dev
```

## Usage

### Step 1: Start Signaling Server

```bash
python3 signaling_server.py
```

### Step 2: Run Call

```bash
# Terminal 2: Smart server (answers calls with AI)
python3 smart_server.py

# Terminal 3: Caller
python3 caller.py
```

**Remote server:**
```bash
python3 caller.py http://192.168.1.100:8080
```

## How It Works

```
Caller                Signaling Server            Smart Server
  |                         |                          |
  |--- Offer (SDP) -------->|                          |
  |                         |<------ Get Offer --------|
  |                         |                          |
  |                         |<------ Answer (SDP) -----|
  |<------ Get Answer ------|                          |
  |                         |                          |
  |<============ Direct P2P Audio Connection =========>|
  |                                                    |
  |  Your voice ────────────────────────────────────>  |
  |                                    STT → LLM → TTS |
  |  <──────────────────────────────── AI response     |
```

## Notes

- Uses PyAudio for low-latency mic capture and playback
- Smart server requires Ollama running locally for LLM
# cli_chat
