"""
Thin JSON-in/JSON-out wrapper over the configured chat model.

Small open models (llama3.1:8b) don't always honour a JSON schema, so every
call: (1) asks for `response_format=json_object` when the provider supports it,
(2) still parses defensively -- stripping ```json fences and slicing to the
outermost braces -- and (3) retries with backoff on transport / parse errors.

Every prompt in this project returns a single JSON *object* (never a bare
array), because json_object mode requires an object at the top level. List
results are returned under a "results" key.
"""



###########
# 1. Set-up
###########
from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI

from . import config

_client: OpenAI | None = None


############
# 2. Helpers
############
def _get() -> OpenAI:
    global _client
    if _client is None:
        _client = config.get_client()
    return _client


def _extract_json(text: str) -> Any:
    """Parse JSON out of a model reply that may be wrapped in prose/fences."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} (object) or [...] (array) span.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from model output:\n{text[:500]}")


###############
# 3. Public API
###############
def chat_json(
    system: str,
    user: str,
    *,
    max_tokens: int = 4096,
    temperature: float | None = None,
) -> Any:
    """Send one system+user turn, return the parsed JSON object."""
    temperature = config.TEMPERATURE if temperature is None else temperature
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    last_err: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = _get().chat.completions.create(
                model=config.MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return _extract_json(resp.choices[0].message.content or "")
        except TypeError:
            # Some OpenAI-compatible servers reject the response_format kwarg.
            resp = _get().chat.completions.create(
                model=config.MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return _extract_json(resp.choices[0].message.content or "")
        except Exception as err:  # noqa: BLE001 - retry any transport/parse error
            last_err = err
            if attempt < config.MAX_RETRIES:
                time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"chat_json failed after {config.MAX_RETRIES} tries: {last_err}")
