"""Prueba de conexión con Google AI API y el modelo Gemma."""

import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# --- Prueba 1: respuesta simple ---
print("=" * 60)
print("PRUEBA 1: Respuesta simple")
print("=" * 60)
response = client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    contents="Responde en una sola línea: ¿qué es un bug report?",
)
print(response.text)

# --- Prueba 2: destilación con schema exacto + few-shot ---
print("\n" + "=" * 60)
print("PRUEBA 2: Destilación con schema y few-shot")
print("=" * 60)

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

BUG_INPUT = """Summary: Firefox crashes when restoring session with more than 200 tabs
Product: Firefox
Component: Session Restore
Severity: critical
Priority: P1
Initial report:
Reproducible 100% on macOS. Steps: open 200+ tabs, quit Firefox, reopen.
Crash signature: mozilla::SessionStore::Restore. Stack trace shows memory allocation failure in tab state deserialization."""

# Gemma (a diferencia de Gemini) NO soporta system_instruction en la API de
# Google AI: si se pasa, el candidato vuelve vacío y response.text es None.
# Solución: plegar el system prompt en el primer turno de usuario y usar roles
# explícitos en el few-shot (los "output" van como rol "model").
contents = []
first = FEW_SHOT_EXAMPLES[0]
contents.append(
    types.Content(
        role="user",
        parts=[types.Part(text=SYSTEM_PROMPT + "\n\n" + first["input"])],
    )
)
contents.append(types.Content(role="model", parts=[types.Part(text=first["output"])]))
for ex in FEW_SHOT_EXAMPLES[1:]:
    contents.append(types.Content(role="user", parts=[types.Part(text=ex["input"])]))
    contents.append(types.Content(role="model", parts=[types.Part(text=ex["output"])]))
contents.append(types.Content(role="user", parts=[types.Part(text=BUG_INPUT)]))

response = client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    contents=contents,
    config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=512,
    ),
)

if response.text is None:
    print("  ✗ response.text es None — diagnóstico:")
    print(f"    prompt_feedback: {response.prompt_feedback}")
    for i, cand in enumerate(response.candidates or []):
        print(f"    candidate[{i}].finish_reason: {cand.finish_reason}")
        print(f"    candidate[{i}].content: {cand.content}")
    raise SystemExit(1)

raw = response.text.strip()
print("Respuesta cruda:")
print(raw)

print("\nValidación JSON:")
try:
    parsed = json.loads(raw)
    assert "fault_location" in parsed
    assert "symptoms" in parsed
    assert "capabilities" in parsed
    assert "product" in parsed["fault_location"]
    assert "component" in parsed["fault_location"]
    print("  ✓ JSON válido con schema correcto")
    print(f"  fault_location: {parsed['fault_location']}")
    print(f"  symptoms:       {parsed['symptoms']}")
    print(f"  capabilities:   {parsed['capabilities']}")
except (json.JSONDecodeError, AssertionError) as e:
    print(f"  ✗ Error: {e}")

# --- Prueba 3: modelos disponibles ---
print("\n" + "=" * 60)
print("PRUEBA 3: Modelos Gemma disponibles en tu cuenta")
print("=" * 60)
for model in client.models.list():
    if "gemma" in model.name.lower():
        print(f"  {model.name}")
