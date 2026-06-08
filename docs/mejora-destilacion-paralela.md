# Mejora pendiente: destilación paralela + structured outputs

## Contexto

`scripts/run_distillation.py` actualmente envía los bugs a Ollama de forma secuencial
(un bug → espera respuesta → siguiente bug). Con ~1200 bugs del piloto, esto es el
cuello de botella principal del Módulo 2.

---

## Qué hacer

### 1. Async concurrente en el cliente (mayor impacto, menor esfuerzo)

Ollama usa **continuous batching**: si llegan varios requests simultáneos, los agrupa
internamente y los procesa en paralelo. Para aprovechar esto:

- Configurar `OLLAMA_NUM_PARALLEL=4` en el servidor Ollama (variable de entorno).
- Reescribir el loop de destilación con `asyncio` + un semáforo que limite a 4 workers
  concurrentes (con 8GB VRAM y Gemma, más de 4 presiona la memoria).
- Estimado de mejora: **3-4x más rápido** sin cambiar de herramienta.

```python
# Esquema del cambio
import asyncio

sem = asyncio.Semaphore(4)

async def distill_one_async(client, bug_id, ...):
    async with sem:
        # llamada async a Ollama
        ...

await asyncio.gather(*[distill_one_async(...) for bug_id in todo])
```

### 2. JSON schema enforcement en Ollama (mejora de calidad)

Desde Ollama v0.5, el parámetro `format` acepta un JSON schema object. Ollama lo
convierte a GBNF grammar y enmascara tokens inválidos durante el sampling → el modelo
**siempre devuelve JSON válido**, sin necesidad de reintentos ni fallback.

Pasar el schema de `fault_location/symptoms/capabilities` directamente en cada request
eliminaría los `_fallback` del código actual.

---

## Lo que NO hacer (por ahora)

- **vLLM**: mayor throughput pero requiere pesos de Gemma en formato HuggingFace
  (safetensors), no el GGUF que usa Ollama actualmente. Implica re-descargar el modelo.
  Solo vale la pena si el async no alcanza.
- **llama.cpp server**: historial de bugs con `json_schema`, no recomendado como
  reemplazo directo.

---

## Referencias

Investigación realizada 2026-06-08 via deep-research (103 agentes, fuentes verificadas
adversarialmente). Fuentes principales:
- https://ollama.com/blog/structured-outputs
- https://www.glukhov.org/post/2025/05/how-ollama-handles-parallel-requests/
- https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html
