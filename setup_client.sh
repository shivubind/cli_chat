#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# VChat Client Setup Script
# Sets up the client for remote voice/video chat with AI agent
# ═══════════════════════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  VChat Client Setup"
echo "═══════════════════════════════════════════════════════════════"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}[1/5] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

# Create virtual environment (optional)
echo -e "\n${YELLOW}[2/5] Setting up environment...${NC}"
if [ ! -d "venv_client" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_client
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate venv
source venv_client/bin/activate

# Install system dependencies (Linux)
echo -e "\n${YELLOW}[3/5] Checking system dependencies...${NC}"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Installing system audio/video libraries..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq portaudio19-dev python3-pyaudio libopencv-dev v4l-utils 2>/dev/null || true
    echo -e "${GREEN}✓ System dependencies installed${NC}"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing macOS dependencies..."
    brew install portaudio opencv 2>/dev/null || true
    echo -e "${GREEN}✓ System dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Please manually install: portaudio, opencv${NC}"
fi

# Install Python dependencies
echo -e "\n${YELLOW}[4/5] Installing Python packages...${NC}"
pip install --upgrade pip -q
pip install -r requirements-client.txt -q
echo -e "${GREEN}✓ Python packages installed${NC}"

# Create client config if not exists
echo -e "\n${YELLOW}[5/5] Setting up configuration...${NC}"
if [ ! -f ".env.client" ]; then
    cat > .env.client << 'EOF'
# VChat Client Configuration
# Copy this to .env and update values

# LiveKit Server (change to your remote server)
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret

# Room name (must match server)
LIVEKIT_ROOM=vchat-room

# Your identity
USER_IDENTITY=user
EOF
    echo -e "${GREEN}✓ Created .env.client template${NC}"
else
    echo -e "${GREEN}✓ .env.client exists${NC}"
fi

# Test camera
echo -e "\n${YELLOW}Testing camera...${NC}"
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print('✓ Camera working:', frame.shape[1], 'x', frame.shape[0])
    cap.release()
else:
    print('⚠ No camera detected (video will be disabled)')
" 2>/dev/null || echo "⚠ Camera test skipped"

# Test microphone
echo -e "\n${YELLOW}Testing microphone...${NC}"
python3 -c "
import pyaudio
p = pyaudio.PyAudio()
info = p.get_default_input_device_info()
print('✓ Microphone:', info['name'])
p.terminate()
" 2>/dev/null || echo "⚠ Microphone test failed"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "${GREEN}  ✓ Client Setup Complete!${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "To run the client:"
echo "  1. Copy .env.client to .env and update LIVEKIT_URL to server address"
echo "  2. Activate environment: source venv_client/bin/activate"
echo "  3. Run: python livekit_client.py"
echo ""
echo "Options:"
echo "  --room ROOM_NAME    Specify room name"
echo "  --url URL           Specify LiveKit server URL"
echo "  --no-video          Disable camera"
echo "  --camera N          Select camera device (0, 1, 2...)"
echo ""

