# VChat - AI Voice Assistant

Real-time voice chat with AI using LiveKit WebRTC.

## Architecture

```
┌─────────────────┐        ┌──────────────────┐        ┌─────────────────┐
│                 │        │                  │        │                 │
│  LiveKit Client │◄─────►│  LiveKit Server  │◄─────►│  LiveKit Agent  │
│   (Your Voice)  │  WebRTC│   (SFU Router)   │  WebRTC│   (AI Backend)  │
│                 │        │                  │        │                 │
└─────────────────┘        └──────────────────┘        └─────────────────┘
                                                              │
                                                              ▼
                                                    ┌─────────────────┐
                                                    │  STT (Whisper)  │
                                                    │  LLM (Ollama)   │
                                                    │  TTS (Kokoro)   │
                                                    │  Vision (Qwen)  │
                                                    │  Memory (Chroma)│
                                                    └─────────────────┘
```

## Features

- **LiveKit WebRTC** - Low-latency audio streaming
- **Speech-to-Text** - Local Whisper models
- **LLM** - Ollama for local AI inference
- **Text-to-Speech** - Kokoro for natural voice
- **Vision** - Say "what do you see" for camera-based image analysis
- **Memory** - ChromaDB for persistent conversation context
- **Barge-in** - Interrupt the AI mid-speech
- **YAML Config** - Easy customization via `config.yml`

## Files

```
VChat/
├── livekit_agent.py      # AI voice agent (server)
├── livekit_client.py     # Audio client
├── config.yml            # Configuration
├── requirements-server.txt   # Server dependencies
├── requirements-client.txt   # Client dependencies
├── env.example           # Environment template
└── README.md
```

## Quick Start

### 1. Install LiveKit Server

```bash
curl -sSL https://get.livekit.io | bash
```

### 2. Install Dependencies

**Server (AI Agent):**
```bash
pip install -r requirements-server.txt

# Linux system dependencies
sudo apt-get install libavdevice-dev libavfilter-dev portaudio19-dev
```

**Client (Audio only):**
```bash
pip install -r requirements-client.txt
```

### 3. Install Ollama Models

```bash
# Text model
ollama pull qwen2.5:0.5b

# Vision model (for "what do you see" feature)
ollama pull qwen3-vl:2b-instruct-bf16
```

### 4. Configure

```bash
cp env.example .env
# Edit .env with your LiveKit credentials
```

### 5. Run

```bash
# Terminal 1: LiveKit Server
livekit-server --dev

# Terminal 2: Ollama
ollama serve

# Terminal 3: AI Agent
python livekit_agent.py --room vchat-room

# Terminal 4: Client
python livekit_client.py --room vchat-room
```

## Configuration

Edit `config.yml` to customize:

### System Prompt
```yaml
system:
  prompt: |
    You are a helpful voice assistant. Keep responses SHORT.
  greeting: "Hello! How can I help?"
```

### Models
```yaml
llm:
  text_model: "qwen2.5:0.5b"      # Fast text model
  vision_model: "qwen3-vl:2b-instruct-bf16"  # Vision model
  
stt:
  model: "small"  # Whisper: tiny, base, small, medium, large

tts:
  voice: "am_puck"  # Kokoro voice
```

### Memory
```yaml
memory:
  enabled: true
  top_k: 5           # Memories to retrieve
  min_similarity: 0.35
```

### Vision Triggers
```yaml
system:
  vision_triggers:
    - "what do you see"
    - "describe what you see"
    - "look at this"
```

## Usage

### Voice Commands

| Say | Action |
|-----|--------|
| "What do you see?" | Captures camera and describes |
| "My name is X" | Saves to memory |
| "What's my name?" | Recalls from memory |
| (Speak while AI talks) | Interrupts AI (barge-in) |

### Command Line

```bash
# Agent with custom room
python livekit_agent.py --room my-room

# Agent with custom config
python livekit_agent.py --config custom.yml

# Client options
python livekit_client.py --room my-room --identity user1
```

## Troubleshooting

### No audio
```bash
# Test microphone
python -c "import pyaudio; p = pyaudio.PyAudio(); print(p.get_default_input_device_info())"
```

### Ollama not responding
```bash
# Check if running
curl http://localhost:11434/api/tags

# Restart if stuck
sudo systemctl restart ollama
```

### Vision not working
```bash
# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'No camera')"

# Check vision model
ollama run qwen3-vl:2b-instruct-bf16 "describe this image"
```

### Memory not finding info
```bash
# Check stored memories
python -c "
import chromadb
client = chromadb.PersistentClient(path='./memory_db')
col = client.get_or_create_collection('vchat_memory')
print(f'Memories: {col.count()}')
for doc in col.get()['documents'][:5]:
    print(f'  - {doc[:60]}')
"
```

## License

MIT
