# VChat - Voice Chat with AI

Real-time voice chat with AI-powered responses using LiveKit.

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
                                                    └─────────────────┘
```

## Features

- **LiveKit-based WebRTC** - Scalable, low-latency audio streaming
- **Voice Activity Detection** - Silero VAD for smart speech detection
- **Speech-to-Text** - Local Whisper models (no API keys needed)
- **LLM Processing** - Ollama for local AI inference
- **Text-to-Speech** - Kokoro for natural voice synthesis
- **Barge-in Support** - Interrupt the AI mid-speech

## Quick Start

### 1. Install LiveKit Server

```bash
# Linux/macOS
curl -sSL https://get.livekit.io | bash

# Or via Homebrew (macOS)
brew install livekit
```

### 2. Install Dependencies

```bash
# System dependencies (Linux)
sudo apt-get install libavdevice-dev libavfilter-dev portaudio19-dev

# Python dependencies
pip install -r requirements-livekit.txt
```

### 3. Configure Environment

```bash
cp env.example .env
# Edit .env with your settings
```

### 4. Start Services

```bash
# Terminal 1: Start LiveKit server (development mode)
livekit-server --dev

# Terminal 2: Start Ollama (if not already running)
ollama serve

# Terminal 3: Start the AI agent
python livekit_agent.py dev

# Terminal 4: Start the client
python livekit_client.py --room vchat-room
```

## Files

| File | Description |
|------|-------------|
| `livekit_agent.py` | AI voice agent with STT/LLM/TTS pipeline |
| `livekit_client.py` | Client for audio streaming to LiveKit |
| `requirements-livekit.txt` | Dependencies for LiveKit version |
| `env.example` | Environment configuration template |

### Legacy Files (WebRTC without LiveKit)

| File | Description |
|------|-------------|
| `signaling_server.py` | Simple SDP signaling server |
| `client.py` | Direct WebRTC audio client |
| `smart_server.py` | AI server (direct WebRTC) |
| `requirements.txt` | Dependencies for legacy version |

## Configuration

### Environment Variables

```bash
# LiveKit Server
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret

# AI Models
WHISPER_MODEL=small          # tiny, base, small, medium, large
OLLAMA_MODEL=qwen2.5:0.5b    # Any Ollama model
KOKORO_VOICE=af_heart        # Kokoro voice ID
```

### Model Options

**Whisper STT:**
- `tiny` - Fastest, lower accuracy
- `base` - Good balance
- `small` - Recommended
- `medium` - Better accuracy, slower
- `large` - Best accuracy, slowest

**Ollama LLM:**
- `qwen2.5:0.5b` - Fast, lightweight
- `llama3.2:1b` - Good quality
- `mistral:7b` - High quality

## Usage Examples

### Client Options

```bash
# Connect to default room
python livekit_client.py

# Specify room name
python livekit_client.py --room my-room

# Connect to remote server
python livekit_client.py --url wss://my-server.livekit.cloud
```

### Agent Options

```bash
# Development mode (auto-creates room)
python livekit_agent.py dev

# Production mode
python livekit_agent.py connect
```

## LiveKit Cloud Deployment

For production, use [LiveKit Cloud](https://cloud.livekit.io):

1. Create a project at cloud.livekit.io
2. Get your API key and secret
3. Update `.env`:
   ```bash
   LIVEKIT_URL=wss://your-project.livekit.cloud
   LIVEKIT_API_KEY=your-api-key
   LIVEKIT_API_SECRET=your-api-secret
   ```

## Troubleshooting

### No audio input
```bash
# Check if microphone is available
python -c "import pyaudio; p = pyaudio.PyAudio(); print(p.get_default_input_device_info())"
```

### Ollama connection error
```bash
# Ensure Ollama is running
ollama serve

# Pull the model
ollama pull qwen2.5:0.5b
```

### CUDA issues
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## Development

### Running Tests

```bash
# Test audio devices
python -c "import pyaudio; p = pyaudio.PyAudio(); print([p.get_device_info_by_index(i) for i in range(p.get_device_count())])"

# Test Whisper
python -c "import whisper; m = whisper.load_model('tiny'); print('Whisper OK')"

# Test Kokoro
python -c "from kokoro import KPipeline; p = KPipeline(lang_code='en-us'); print('Kokoro OK')"
```

## License

MIT
