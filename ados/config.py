"""
ADOS Config — Neo4j · Groq LLM · Grafana · LangGraph
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
for d in (DATA_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _load_env_file():
    """Reload .env file into os.environ (called on every get_settings())."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ[key.strip()] = value.strip()


class Neo4jSettings(BaseModel):
    uri: str = Field(default=os.getenv("NEO4J_URI", "bolt://localhost:7688"))
    user: str = Field(default=os.getenv("NEO4J_USER", "neo4j"))
    password: str = Field(default=os.getenv("NEO4J_PASSWORD", "ados_secret"))


class LLMSettings(BaseModel):
    provider: str = Field(default=os.getenv("LLM_PROVIDER", "groq"))
    model_name: str = Field(default=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"))
    api_key: str = Field(default=os.getenv("GROQ_API_KEY", ""))
    temperature: float = 0.1
    # Comma-separated fallback models (tried in order on rate-limit)
    fallback_models: str = Field(
        default=os.getenv(
            "LLM_FALLBACK_MODELS",
            "llama-3.1-8b-instant,gemma2-9b-it,llama3-8b-8192,mixtral-8x7b-32768",
        )
    )
    # Cache identical LLM calls for this many seconds (0 = disabled)
    cache_ttl_seconds: int = int(os.getenv("LLM_CACHE_TTL", "300"))


class GrafanaSettings(BaseModel):
    url: str = Field(default=os.getenv("GRAFANA_URL", "http://localhost:3001"))


class AppSettings(BaseModel):
    app_name: str = "ADOS"
    version: str = "2.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    grafana: GrafanaSettings = Field(default_factory=GrafanaSettings)
    csv_dir: str = str(PROJECT_ROOT)
    log_level: str = "INFO"


_settings: Optional[AppSettings] = None

def get_settings() -> AppSettings:
    """Get settings - always reloads .env to pick up API key changes."""
    global _settings
    # Reload .env file to pick up any changes (especially API keys)
    _load_env_file()
    # Always recreate to pick up environment changes
    _settings = AppSettings()
    return _settings

def reset_settings() -> AppSettings:
    """Force reload settings (picks up new .env values)."""
    global _settings
    _settings = None
    return get_settings()
