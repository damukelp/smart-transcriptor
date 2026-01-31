#!/bin/bash
# One-time setup script for a fresh RunPod instance
# Run this once after creating a new pod, then restart the pod.
#
# Prerequisites:
#   - RunPod pod with GPU (tested on RTX 2000 Ada / RTX 4090)
#   - Network volume mounted at /workspace
#   - Docker Start Command set to: bash /workspace/start.sh
#   - (Optional) Set HF_TOKEN environment variable in RunPod pod config

set -e

echo "============================================"
echo "  Smart Transcriptor - RunPod Setup"
echo "============================================"

# 1. Clone the repo
echo ""
echo "[1/5] Cloning smart-transcriptor..."
if [ -d /workspace/smart-transcriptor ]; then
  echo "  Already exists, pulling latest..."
  cd /workspace/smart-transcriptor && git pull
else
  cd /workspace
  git clone https://github.com/damukelp/smart-transcriptor.git
fi

# 2. Create Python venv and install dependencies
echo ""
echo "[2/5] Setting up Python virtual environment..."
if [ ! -d /workspace/venv ]; then
  python -m venv /workspace/venv
fi
source /workspace/venv/bin/activate
cd /workspace/smart-transcriptor
pip install uvicorn fastapi websockets faster-whisper httpx pydantic-settings
echo "  Python venv ready at /workspace/venv"

# 3. Install Ollama
echo ""
echo "[3/5] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# 4. Set up persistent Ollama models directory and pull model
echo ""
echo "[4/5] Pulling Ollama model..."
mkdir -p /workspace/ollama/models
export OLLAMA_HOST=0.0.0.0
export OLLAMA_MODELS=/workspace/ollama/models
ollama serve > /tmp/ollama_setup.log 2>&1 &
OLLAMA_PID=$!
sleep 5
ollama pull mistral:latest
kill $OLLAMA_PID 2>/dev/null

# 5. Install the startup script
echo ""
echo "[5/5] Installing startup script..."
cp /workspace/smart-transcriptor/runpod_start.sh /workspace/start.sh
chmod +x /workspace/start.sh

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Set Docker Start Command to: bash /workspace/start.sh"
echo "     (in RunPod UI > Edit Pod)"
echo "  2. (Optional) Set HF_TOKEN env var in pod config"
echo "  3. Restart the pod"
echo ""
echo "Exposed ports needed:"
echo "  - 8080  (Gateway)"
echo "  - 8082  (SLM)"
echo "  - 8083  (ASR)"
echo "  - 11434 (Ollama)"
echo ""
