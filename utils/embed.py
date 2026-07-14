"""
OpenAI embedding client with an on-disk cache -- the embedding twin of utils/llm.py

Embeddings are decoupled from the chat provider on purpose: the chat model may be a
local ollama model (LLM_PROVIDER=ollama), but embeddings always go to OpenAI, so
OPENAI_API_KEY is required regardless. The model name comes from config.EMBED_MODEL.

Vectors are cached to data/embeddings/<EMBED_MODEL>.json keyed by (model, text), so
re-runs and notebook re-executions cost nothing and never re-hit the API.
"""



###########
# 1. Set-up
###########
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from openai import OpenAI

from . import config

_CACHE_PATH = config.PROJECT_ROOT / "data" / "embeddings" / f"{config.EMBED_MODEL}.json"


############
# 2. Helpers
############
def _client() -> OpenAI:
    """Embeddings always go to api.openai.com, never the configured chat provider."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY must be set in .env to compute embeddings.")
    return OpenAI(api_key=key)


def _key(text: str) -> str:
    """Cache key: the model matters, so switching EMBED_MODEL invalidates cleanly."""
    return hashlib.sha1(f"{config.EMBED_MODEL}\n{text}".encode()).hexdigest()


_CACHE: dict | None = None       # loaded once per process -- see _load_cache()


def _load_cache() -> dict:
    """Read the cache file ONCE, then hold it in memory.

    The file is ~750MB (3072-dim vectors as JSON) and takes ~12s to parse. Re-reading
    it on every embed() call made scoring I/O-bound rather than API-bound: Step 4 spent
    ~90s per drug loading and re-dumping this file, and almost no time embedding.
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = (json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
                  if _CACHE_PATH.exists() else {})
    return _CACHE


def _save_cache(cache: dict) -> None:
    """Write the cache back. Only called when embed() actually computed new vectors,
    so a fully-cached re-run does no writes at all."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


###############
# 3. Public API
###############
def embed(texts: list[str]) -> np.ndarray:
    """Embed texts with OpenAI, caching to disk so re-runs are free.

    Returns an (n, d) float array aligned to `texts`. Only texts missing from the
    cache are sent, in chunks of 256.
    """
    texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]
    cache = _load_cache()
    missing = [t for t in dict.fromkeys(texts) if _key(t) not in cache]
    if missing:
        client = _client()
        for i in range(0, len(missing), 256):
            chunk = missing[i : i + 256]
            resp = client.embeddings.create(model=config.EMBED_MODEL, input=chunk)
            for t, d in zip(chunk, resp.data):
                cache[_key(t)] = d.embedding
        _save_cache(cache)
    return np.asarray([cache[_key(t)] for t in texts], dtype=float)


def unit(mat: np.ndarray) -> np.ndarray:
    """Row-normalise to unit length, so a dot product == cosine similarity."""
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)
