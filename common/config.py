from pydantic_settings import BaseSettings


class GatewaySettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    asr_ws_url: str = "ws://asr:8001/stream"
    max_sessions: int = 10

    model_config = {"env_prefix": "GATEWAY_"}


class ASRSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8001
    model_size: str = "large-v3"
    device: str = "auto"
    compute_type: str = "auto"
    hf_token: str = ""
    chunk_duration_s: float = 3.0
    diarize_window_s: float = 15.0
    diarize_clustering_threshold: float = 0.55

    model_config = {"env_prefix": "ASR_"}


class SLMSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8002
    ollama_url: str = "http://localhost:11434"
    model_name: str = "phi3"
    temperature: float = 0.3
    max_tokens: int = 2048

    model_config = {"env_prefix": "SLM_"}
