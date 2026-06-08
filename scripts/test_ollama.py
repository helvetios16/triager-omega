"""Prueba de conexión con Ollama (gemma4:12b) via API OpenAI-compatible."""

import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = "gemma4:e4b-it-qat" # gemma4:e4b-it-qat , gemma4:12b

# Controla el "pensamiento" del modelo en Ollama: False lo desactiva,
# o usa "low"/"medium"/"high" para regular el nivel de razonamiento.
THINK = False

client = OpenAI(base_url=BASE_URL, api_key="ollama")

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
        "role": "user",
        "content": """Summary: Bookmarks toolbar disappears after update
Product: Firefox
Component: Bookmarks & History
Severity: major
Priority: P2
Initial report:
After upgrading to Firefox 125, the bookmarks toolbar is gone. Checked View > Toolbars but it shows enabled. Restarting does not help.""",
    },
    {
        "role": "assistant",
        "content": """{
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
        "role": "user",
        "content": """Summary: Email with large attachment causes Thunderbird to freeze
Product: Thunderbird
Component: Message Compose Window
Severity: major
Priority: P2
Initial report:
When attaching a file larger than 50MB and clicking Send, the compose window freezes. CPU usage spikes to 100%. Only force-quitting resolves it.""",
    },
    {
        "role": "assistant",
        "content": """{
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

BUG_INPUT = """Summary: Firefox crashes when restoring session with more than 200 tabs
Product: Firefox
Component: Session Restore
Severity: critical
Priority: P1
Initial report:
Reproducible 100% on macOS. Steps: open 200+ tabs, quit Firefox, reopen.
Crash signature: mozilla::SessionStore::Restore. Stack trace shows memory allocation failure in tab state deserialization."""


def get_model() -> str:
    """Verifica que gemma4:12b esté disponible en Ollama."""
    models = client.models.list()
    available = [m.id for m in models.data]
    if not available:
        raise RuntimeError("Ollama no tiene modelos. Ejecuta primero: ollama pull gemma4:12b")
    if not any(m == MODEL or m.startswith(MODEL) for m in available):
        raise RuntimeError(f"{MODEL} no está disponible. Ejecuta: ollama pull {MODEL}\nDisponibles: {available}")
    return MODEL


def distill(bug_text: str, model: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": bug_text})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=1500,
        extra_body={"think": THINK},
    )
    return json.loads(response.choices[0].message.content.strip())


def main():
    print("=" * 60)
    print("  TRIAGER-OMEGA — Prueba Ollama (gemma4:12b)")
    print("=" * 60)

    # --- Prueba 1: detectar modelo ---
    print("\nPRUEBA 1: Verificar modelo en Ollama")
    print("-" * 60)
    try:
        model = get_model()
        print(f"  ✓ Modelo activo: {model} (think={THINK})")
    except Exception as e:
        print(f"  ✗ {e}")
        return

    # --- Prueba 2: respuesta simple ---
    print("\nPRUEBA 2: Respuesta simple")
    print("-" * 60)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Responde en una sola línea: ¿qué es un bug report?"}],
        temperature=0.0,
        max_tokens=300,
        extra_body={"think": THINK},
    )
    print(f"  {response.choices[0].message.content.strip()}")

    # --- Prueba 3: destilación con schema exacto ---
    print("\nPRUEBA 3: Destilación con schema y few-shot")
    print("-" * 60)
    try:
        result = distill(BUG_INPUT, model)
        print("  ✓ JSON válido")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON inválido: {e}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("  Pruebas completadas")
    print("=" * 60)


if __name__ == "__main__":
    main()
