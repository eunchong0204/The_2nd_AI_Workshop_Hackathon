"""
Central configuration for the LLM ADR-classification stage.

Step 2 (drug-agnostic PT blocklist) and Step 3 (per-drug label comparison) both
import their model client from here, so a single env setting guarantees the SAME
model runs across both stages.

Every provider is reached through the OpenAI Python SDK.

  LLM_PROVIDER=ollama  -> local open-source model
  LLM_PROVIDER=openai  -> an OpenAI frontier model

"""



###########
# 1. Set-up
###########
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    """Read a .env value, tolerating inline comments in simple settings."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.split("#", 1)[0].strip()


##########
# 2. Paths
##########
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LABEL_CACHE_DIR = PROJECT_ROOT / "data" / "labels"

# Inputs produced by the R stage.
UNIQUE_PTS_CSV = PROCESSED_DIR / "unique_pts.csv"   # Step 2 input  (pt, n_pairs)
SIGNALS_CSV = PROCESSED_DIR / "signals.csv"         # Step 3 input  (prod_ai, pt, ...)

# Outputs produced here.
PT_TYPES_CSV = PROCESSED_DIR / "pt_types.csv"       # Step 2 (pt, pt_type, keep); Step 3 blocklist
ADR_RESULTS_CSV = PROCESSED_DIR / "adr_results.csv" # Step 3 output (all statuses)
SCORED_ADRS_CSV = PROCESSED_DIR / "scored_adrs.csv" # Step 4 output (per-drug ranking)


#######################
# 3. Provider selection
#######################
# Per-provider defaults; any can be overridden by env vars (LLM_MODEL, LLM_BASE_URL).
_PROVIDERS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1:8b",
        "api_key_env": None,          # Ollama ignores the key; a dummy is fine
    },
    "openai": {
        "base_url": None,             # SDK default (api.openai.com)
        "model": "gpt-4o-mini",       # override per run with LLM_MODEL (gpt-4.1, gpt-4o, ...)
        "api_key_env": "OPENAI_API_KEY",
    },
}

PROVIDER = (_env("LLM_PROVIDER", "ollama") or "ollama").lower()
if PROVIDER not in _PROVIDERS:
    raise ValueError(
        f"LLM_PROVIDER={PROVIDER!r} is not one of {list(_PROVIDERS)}."
    )

_cfg = _PROVIDERS[PROVIDER]
MODEL = _env("LLM_MODEL", _cfg["model"]) or _cfg["model"]
BASE_URL = _env("LLM_BASE_URL", _cfg["base_url"])

# Request-level knobs shared by every call.
TEMPERATURE = float(_env("LLM_TEMPERATURE", "0") or "0")   # 0 -> reproducible labels
MAX_RETRIES = int(_env("LLM_MAX_RETRIES", "4") or "4")
REQUEST_TIMEOUT = float(_env("LLM_TIMEOUT", "120") or "120")

# Embeddings (utils/embed.py) are decoupled from the chat provider on purpose.
EMBED_MODEL = _env("EMBED_MODEL", "text-embedding-3-large") or "text-embedding-3-large"

OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "").strip()


############
# 4. Client
############
def get_client() -> OpenAI:
    """Build an OpenAI-SDK client pointed at the configured provider."""
    if _cfg["api_key_env"]:
        api_key = os.getenv(_cfg["api_key_env"], "").strip()
        if not api_key:
            raise RuntimeError(
                f"LLM_PROVIDER={PROVIDER} needs {_cfg['api_key_env']} set in .env"
            )
    else:
        # Keyless local server (ollama); the SDK still requires a non-empty key.
        api_key = "ollama"

    kwargs = {"api_key": api_key, "timeout": REQUEST_TIMEOUT}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    return OpenAI(**kwargs)


def describe() -> str:
    return f"provider={PROVIDER} model={MODEL} base_url={BASE_URL or 'default'}"
