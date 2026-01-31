#!/bin/bash
# RunPod auto-start script
# Copy this to /workspace/start.sh on your RunPod instance

echo "[startup] Starting services at $(date)"

# Start Ollama
export OLLAMA_HOST=0.0.0.0
ollama serve > /tmp/ollama.log 2>&1 &
echo "[startup] Ollama started (pid $!)"

# Wait for Ollama to be ready before starting dependent services
sleep 5

# Start smart-transcriptor services
cd /workspace/smart-transcriptor

ASR_MODEL_SIZE=large-v3 ASR_CHUNK_DURATION_S=3.0 ASR_HF_TOKEN="${HF_TOKEN}" \
  python -m uvicorn asr_service.main:app --host 0.0.0.0 --port 8083 > /tmp/asr.log 2>&1 &
echo "[startup] ASR started (pid $!)"

SLM_MODEL_NAME=mistral:latest \
  python -m uvicorn slm_service.main:app --host 0.0.0.0 --port 8082 > /tmp/slm.log 2>&1 &
echo "[startup] SLM started (pid $!)"

GATEWAY_ASR_WS_URL=ws://localhost:8083/stream \
  python -m uvicorn gateway.main:app --host 0.0.0.0 --port 8080 > /tmp/gateway.log 2>&1 &
echo "[startup] Gateway started (pid $!)"

echo "[startup] All services launched"
