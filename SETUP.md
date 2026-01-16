# VChat Setup Guide

Quick setup for running client and server separately (remote deployment).

---

## 🖥️ Server Setup (AI Agent)

The server runs the AI voice agent with:
- **STT**: Whisper (speech recognition)
- **LLM**: Ollama (text generation)
- **Vision**: Moondream (image analysis)
- **TTS**: Kokoro (text-to-speech)
- **Memory**: ChromaDB (conversation memory)

### Quick Setup

```bash
# Run setup script
./setup_server.sh

# Or manual setup:
python3 -m venv venv_server
source venv_server/bin/activate
pip install -r requirements-server.txt
```

### Start Server

**Terminal 1 - LiveKit Server:**
```bash
livekit-server --config livekit.yaml --dev
```

**Terminal 2 - AI Agent:**
```bash
source venv_server/bin/activate
cp env.server.example .env  # First time only
python livekit_agent.py
```

### Server Requirements
- Python 3.10+
- 8GB+ RAM (16GB recommended)
- NVIDIA GPU recommended (CPU works but slower)
- Ollama installed
- Open ports: 7880 (TCP), 50000-60000 (UDP)

---

## 📱 Client Setup (User)

The client captures audio/video and connects to the remote server.

### Quick Setup

```bash
# Run setup script
./setup_client.sh

# Or manual setup:
python3 -m venv venv_client
source venv_client/bin/activate
pip install -r requirements-client.txt
```

### Configure Remote Server

```bash
cp env.client.example .env

# Edit .env and set your server IP:
# LIVEKIT_URL=ws://YOUR_SERVER_IP:7880
```

### Start Client

```bash
source venv_client/bin/activate
python livekit_client.py
```

### Client Options

```bash
# Connect to specific room
python livekit_client.py --room my-room

# Connect to remote server
python livekit_client.py --url ws://192.168.1.100:7880

# Disable camera (audio only)
python livekit_client.py --no-video

# Select different camera
python livekit_client.py --camera 1
```

### Client Requirements
- Python 3.10+
- Microphone
- Camera (optional, for vision features)
- Network access to server

---

## 🌐 Network Configuration

### For Remote Access

1. **Server firewall** - Open these ports:
   ```bash
   sudo ufw allow 7880/tcp    # LiveKit signaling
   sudo ufw allow 7881/tcp    # LiveKit TCP fallback
   sudo ufw allow 50000:60000/udp  # WebRTC media
   ```

2. **Router** - Port forward if behind NAT:
   - 7880 TCP → Server IP
   - 50000-60000 UDP → Server IP

3. **Client config** - Update `.env`:
   ```
   LIVEKIT_URL=ws://PUBLIC_IP:7880
   ```

---

## 🔧 Troubleshooting

### Server Issues

**Ollama not using GPU:**
```bash
# Reinstall Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Check GPU detection
nvidia-smi
ollama ps  # Should show GPU %
```

**Memory (ChromaDB) not working:**
```bash
pip install --upgrade chromadb distro
```

### Client Issues

**No audio:**
```bash
# Test microphone
python -c "import pyaudio; p=pyaudio.PyAudio(); print(p.get_default_input_device_info())"
```

**No camera:**
```bash
# List cameras
ls /dev/video*

# Test camera
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**Can't connect to server:**
```bash
# Test connection
curl http://SERVER_IP:7880
```

---

## 📁 File Structure

```
cli_chat/
├── livekit_agent.py      # Server: AI agent
├── livekit_client.py     # Client: Audio/video
├── config.yml            # Server configuration
├── livekit.yaml          # LiveKit server config
├── requirements-server.txt
├── requirements-client.txt
├── env.server.example
├── env.client.example
├── setup_server.sh
├── setup_client.sh
└── memory_db/            # ChromaDB storage (server)
```

---

## 🎤 Voice Commands

Once connected, try these:

- **"Hello"** - Basic conversation
- **"What do you see?"** - AI describes camera view
- **"My name is [Name]"** - AI will remember
- **"What's my name?"** - Test memory

---

## 📞 Support

For issues, check:
1. Both server terminals for error messages
2. Client terminal for connection status
3. Firewall/port forwarding configuration

