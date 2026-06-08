"""Módulo 2 — Destilación semántica (backend-agnóstico).

Convierte el texto crudo de un bug en un JSON estructurado
(`fault_location` / `symptoms` / `capabilities`, §5.4 del PLAN) usando un LLM Gemma,
y lo serializa al texto destilado `[FL] ... [SY] ... [CP] ...` (§5.8) que consumen
DeBERTa (CBR) y SBERT (IBR).

Backends (config.distill_backend):
  - "lmstudio": servidor local OpenAI-compatible (sin límite, para el batch completo).
  - "google":   API Google AI vía google.genai (rápido para probar; ojo rate limits).

El prompt (system + few-shot) replica `scripts/test_api.py`. La destilación se
cachea en `artifacts/pilot/distillations.parquet` (clave `Bug Id`) y es idempotente.

Uso programático:
    client = make_client()
    out = distill_one(client, build_input(bug_row, first_comment))
"""

from __future__ import annotations

import json

import pandas as pd
from loguru import logger

from triager_omega.config import Settings, settings

# --------------------------------------------------------------------------- #
# Prompt (idéntico a scripts/test_api.py)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are an expert software engineer that reads bug reports and extracts structured information.

Return ONLY a raw JSON object — no markdown, no code blocks, no extra text.
The JSON must have exactly this structure:
{
  "fault_location": {
    "product": "<exact product name from input>",
    "component": "<exact component name from input>",
    "subsystem": "<inferred subsystem, max 5 words, or empty string>"
  },
  "symptoms": ["<short symptom 1, max 6 words>", "..."],
  "capabilities": ["<technical skill 1>", "..."]
}
Rules:
- fault_location.product and fault_location.component must be copied exactly from the input.
- symptoms: 1 to 5 items describing what the user observes.
- capabilities: 1 to 5 concrete technical skills a developer needs to fix this bug.
- If information is insufficient, use empty list [] or empty string "".
"""

FEW_SHOT_EXAMPLES = [
    {
        "input": """Summary: Bookmarks toolbar disappears after update
Product: Firefox
Component: Bookmarks & History
Severity: major
Priority: P2
Initial report:
After upgrading to Firefox 125, the bookmarks toolbar is gone. Checked View > Toolbars but it shows enabled. Restarting does not help. Profile is not corrupted (tested with new profile — same issue).""",
        "output": """{
  "fault_location": {
    "product": "Firefox",
    "component": "Bookmarks & History",
    "subsystem": "toolbar visibility state"
  },
  "symptoms": ["toolbar disappears after update", "toolbar shows as enabled in menu", "persists after restart"],
  "capabilities": ["Firefox toolbar UI", "profile migration", "XUL/CSS layout"]
}""",
    },
    {
        "input": """Summary: Email with large attachment causes Thunderbird to freeze
Product: Thunderbird
Component: Message Compose Window
Severity: major
Priority: P2
Initial report:
When attaching a file larger than 50MB and clicking Send, the compose window freezes indefinitely. CPU usage spikes to 100%. Only force-quitting resolves it. Tested on Windows 11 and Linux.""",
        "output": """{
  "fault_location": {
    "product": "Thunderbird",
    "component": "Message Compose Window",
    "subsystem": "large attachment send path"
  },
  "symptoms": ["UI freeze on send", "100% CPU spike", "requires force quit"],
  "capabilities": ["MIME encoding", "async IO", "attachment streaming", "cross-platform threading"]
}""",
    },
]


# --------------------------------------------------------------------------- #
# Construcción de la entrada al LLM (§5.3)
# --------------------------------------------------------------------------- #
def build_input(bug: pd.Series, first_comment: str | None, cfg: Settings = settings) -> str:
    """Concatenación estructurada: campos del bug + primer comentario truncado."""
    parts = [
        f"Summary: {bug.get('Summary', '')}",
        f"Product: {bug.get('Product', '')}",
        f"Component: {bug.get('Component', '')}",
        f"Severity: {bug.get('Severity', '')}",
        f"Priority: {bug.get('Priority', '')}",
    ]
    if first_comment:
        body = str(first_comment)[: cfg.distill_max_comment_chars]
        parts.append("Initial report:\n" + body)
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class _GoogleBackend:
    """Gemma vía google.genai. Gemma NO admite system_instruction → se pliega el
    system en el primer turno de usuario y el few-shot va con roles user/model."""

    def __init__(self, cfg: Settings):
        import os

        from dotenv import load_dotenv
        from google import genai

        load_dotenv()  # carga GOOGLE_API_KEY del .env a os.environ (como test_api.py)
        self._genai = genai
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = cfg.google_model
        self.max_tokens = cfg.distill_max_tokens

    def complete(self, user_input: str) -> str:
        from google.genai import types

        contents = []
        first = FEW_SHOT_EXAMPLES[0]
        contents.append(types.Content(
            role="user", parts=[types.Part(text=SYSTEM_PROMPT + "\n\n" + first["input"])]))
        contents.append(types.Content(role="model", parts=[types.Part(text=first["output"])]))
        for ex in FEW_SHOT_EXAMPLES[1:]:
            contents.append(types.Content(role="user", parts=[types.Part(text=ex["input"])]))
            contents.append(types.Content(role="model", parts=[types.Part(text=ex["output"])]))
        contents.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=self.max_tokens),
        )
        return (resp.text or "").strip()


def _chat_messages(user_input: str) -> list[dict]:
    """system + few-shot (user/assistant) + entrada. Para backends OpenAI-compatibles
    que sí soportan rol system (Ollama, LM Studio)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": user_input})
    return messages


class _OllamaBackend:
    """Gemma servido por Ollama (API OpenAI-compatible). `think=False` desactiva el
    razonamiento → el modelo emite el JSON directo, sin preámbulo (ver test_ollama.py)."""

    def __init__(self, cfg: Settings):
        from openai import OpenAI

        self.client = OpenAI(base_url=cfg.ollama_base_url, api_key="ollama")
        self.model = cfg.ollama_model
        self.max_tokens = cfg.distill_max_tokens
        self.think = cfg.ollama_think

    def complete(self, user_input: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=_chat_messages(user_input),
            temperature=0.0,
            max_tokens=self.max_tokens,
            extra_body={"think": self.think},
        )
        return (resp.choices[0].message.content or "").strip()


class _LMStudioBackend:
    """Gemma local servido por LM Studio (API OpenAI-compatible)."""

    def __init__(self, cfg: Settings):
        from openai import OpenAI

        self.client = OpenAI(base_url=cfg.lm_studio_base_url, api_key="lm-studio")
        self.model = cfg.lm_studio_model
        self.max_tokens = cfg.distill_max_tokens

    def complete(self, user_input: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=_chat_messages(user_input),
            temperature=0.0, max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


class _GrokBackend:
    """Grok vía API xAI (OpenAI-compatible). Modelo en cfg.grok_model. Lee GROK_API_KEY del .env."""

    def __init__(self, cfg: Settings):
        from openai import OpenAI
        from dotenv import load_dotenv
        import os

        load_dotenv()
        api_key = cfg.grok_api_key or os.environ.get("GROK_API_KEY", "")
        if not api_key:
            raise ValueError("GROK_API_KEY no encontrada en .env ni en config")
        self.client = OpenAI(base_url=cfg.grok_base_url, api_key=api_key)
        self.model = cfg.grok_model
        self.max_tokens = cfg.distill_max_tokens

    def complete(self, user_input: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=_chat_messages(user_input),
            temperature=0.0, max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


def make_client(cfg: Settings = settings):
    """Instancia el backend según config.distill_backend."""
    if cfg.distill_backend == "ollama":
        return _OllamaBackend(cfg)
    if cfg.distill_backend == "google":
        return _GoogleBackend(cfg)
    if cfg.distill_backend == "lmstudio":
        return _LMStudioBackend(cfg)
    if cfg.distill_backend == "grok":
        return _GrokBackend(cfg)
    raise ValueError(f"distill_backend desconocido: {cfg.distill_backend}")


# --------------------------------------------------------------------------- #
# Destilación de un bug (con validación + fallback, §5.5)
# --------------------------------------------------------------------------- #
def _valid(d: dict) -> bool:
    try:
        return (
            isinstance(d.get("fault_location"), dict)
            and "product" in d["fault_location"]
            and "component" in d["fault_location"]
            and isinstance(d.get("symptoms"), list)
            and isinstance(d.get("capabilities"), list)
        )
    except (TypeError, AttributeError):
        return False


def _fallback(bug: pd.Series) -> dict:
    """Si el LLM falla 2 veces: usar Product/Component crudos (§5.5)."""
    return {
        "fault_location": {
            "product": str(bug.get("Product", "")),
            "component": str(bug.get("Component", "")),
            "subsystem": "",
        },
        "symptoms": [s for s in [str(bug.get("Severity", "")).strip()] if s and s != "--"],
        "capabilities": [],
        "_fallback": True,
    }


def distill_one(client, user_input: str, bug: pd.Series, retries: int = 1) -> dict:
    """Llama al LLM, parsea y valida el JSON; reintenta y cae a fallback determinista."""
    for _ in range(retries + 1):
        try:
            raw = client.complete(user_input)
            # quitar fences markdown si el modelo los mete igual.
            if raw.startswith("```"):
                raw = raw.strip("`").split("\n", 1)[-1]
                if raw.lstrip().startswith("json"):
                    raw = raw.lstrip()[4:]
            d = json.loads(raw)
            if _valid(d):
                return d
        except (json.JSONDecodeError, Exception) as e:  # noqa: BLE001
            logger.debug("Bug {} reintento por error: {}", bug.get("Bug Id"), e)
    return _fallback(bug)


# --------------------------------------------------------------------------- #
# Serialización al texto destilado (§5.8)
# --------------------------------------------------------------------------- #
def to_distilled_text(d: dict) -> str:
    fl = d.get("fault_location", {}) or {}
    fl_str = " ".join(str(fl.get(k, "")) for k in ("product", "component", "subsystem")).strip()
    sy = " ".join(str(s) for s in (d.get("symptoms") or []))
    cp = " ".join(str(c) for c in (d.get("capabilities") or []))
    return f"[FL] {fl_str} [SY] {sy} [CP] {cp}".strip()
