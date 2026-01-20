#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# VChat Server Setup Script
# Sets up the AI voice agent server with STT, LLM, TTS, Vision, and Memory
# ═══════════════════════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  VChat Server Setup"
echo "═══════════════════════════════════════════════════════════════"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}[1/8] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

# Check NVIDIA GPU
echo -e "\n${YELLOW}[2/8] Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo -e "${GREEN}✓ GPU found: $GPU_NAME${NC}"
    HAS_GPU=true
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected. Will use CPU (slower)${NC}"
    HAS_GPU=false
fi

# Create virtual environment
echo -e "\n${YELLOW}[3/8] Setting up environment...${NC}"
if [ ! -d "venv_server" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_server
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate venv
source venv_server/bin/activate

# Install system dependencies
echo -e "\n${YELLOW}[4/8] Installing system dependencies...${NC}"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq ffmpeg libsndfile1 espeak-ng 2>/dev/null || true
    echo -e "${GREEN}✓ System dependencies installed${NC}"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install ffmpeg espeak 2>/dev/null || true
    echo -e "${GREEN}✓ System dependencies installed${NC}"
fi

# Install Python dependencies
echo -e "\n${YELLOW}[5/8] Installing Python packages (this may take a while)...${NC}"
pip install --upgrade pip -q

# Install PyTorch with CUDA if GPU available
if [ "$HAS_GPU" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 -q 2>/dev/null || \
    pip install torch torchaudio -q
else
    echo "Installing PyTorch (CPU)..."
    pip install torch torchaudio -q
fi

pip install -r requirements-server.txt -q
echo -e "${GREEN}✓ Python packages installed${NC}"

# Check/Install Ollama
echo -e "\n${YELLOW}[6/8] Setting up Ollama...${NC}"
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
    echo -e "${GREEN}✓ Ollama $OLLAMA_VERSION found${NC}"
else
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "${GREEN}✓ Ollama installed${NC}"
fi

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama server..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# Pull required models
echo -e "\n${YELLOW}[7/8] Downloading AI models...${NC}"
echo "Pulling text model (qwen2.5:0.5b)..."
ollama pull qwen2.5:0.5b -q 2>/dev/null || ollama pull qwen2.5:0.5b

echo "Pulling vision model (moondream)..."
ollama pull moondream -q 2>/dev/null || ollama pull moondream

echo -e "${GREEN}✓ AI models ready${NC}"

# Setup LiveKit Server
echo -e "\n${YELLOW}[8/8] Setting up LiveKit Server...${NC}"
if command -v livekit-server &> /dev/null; then
    echo -e "${GREEN}✓ LiveKit server found${NC}"
else
    echo "Installing LiveKit server..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -sSL https://get.livekit.io | bash
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install livekit
    fi
    echo -e "${GREEN}✓ LiveKit server installed${NC}"
fi

# Create server config if not exists
if [ ! -f ".env.server" ]; then
    cat > .env.server << 'EOF'
# VChat Server Configuration
# Copy this to .env and update values

# LiveKit Server
LIVEKIT_URL=ws://0.0.0.0:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Models
OLLAMA_MODEL=qwen2.5:0.5b
OLLAMA_VISION_MODEL=moondream
WHISPER_MODEL=small

# TTS Voice
KOKORO_VOICE=am_puck
EOF
    echo -e "${GREEN}✓ Created .env.server template${NC}"
fi

# Create LiveKit config
if [ ! -f "livekit.yaml" ]; then
    cat > livekit.yaml << 'EOF'
# LiveKit Server Configuration
port: 7880
rtc:
  port_range_start: 50000
  port_range_end: 60000
  use_external_ip: true
keys:
  devkey: secret
logging:
  level: info
EOF
    echo -e "${GREEN}✓ Created livekit.yaml${NC}"
fi

# Summary
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "${GREEN}  ✓ Server Setup Complete!${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo -e "${BLUE}To start the server:${NC}"
echo ""
echo "  1. Start LiveKit server (in terminal 1):"
echo "     livekit-server --config livekit.yaml --dev"
echo ""
echo "  2. Start AI agent (in terminal 2):"
echo "     source venv_server/bin/activate"
echo "     python livekit_agent.py"
echo ""
echo -e "${BLUE}For remote clients, update their .env with:${NC}"
echo "     LIVEKIT_URL=ws://YOUR_SERVER_IP:7880"
echo ""
echo -e "${BLUE}Installed components:${NC}"
echo "  • STT: Whisper (local)"
echo "  • LLM: Ollama + qwen2.5:0.5b"
echo "  • Vision: Ollama + moondream"
echo "  • TTS: Kokoro"
echo "  • Memory: ChromaDB"
echo ""

