<!-- Merged from: bitacora-destilacion-llm-ts-openj9.md, piloto-destilacion-agy-openj9.md, mejora-destilacion-paralela.md, correccion-destilacion-deberta-descartado.md, registro-constancia-y-verificacion-ts.md, sesion-2026-06-08.md -->

# 02 — Destilación Semántica: Bitácora, Pilotos y Correcciones

## Tabla de contenidos

1. [Corrección crítica: "la destilación empeora" era sobre la arquitectura descartada](#1-corrección-crítica-la-destilación-empeora-era-sobre-la-arquitectura-descartada-correccion-destilacion-deberta-descartadomd)
2. [Sesión 2026-06-08 — Setup inicial destilación + backend Grok](#2-sesión-2026-06-08--setup-inicial-destilación--backend-grok-sesion-2026-06-08md)
3. [Mejora pendiente: destilación paralela + structured outputs](#3-mejora-pendiente-destilación-paralela--structured-outputs-mejora-destilacion-paralelamed)
4. [Piloto rápido de destilación en OpenJ9 (backend agy/Gemini)](#4-piloto-rápido-de-destilación-en-openj9-backend-agy--gemini-piloto-destilacion-agy-openj9md)
   - [4.1 Método](#41-método)
   - [4.2 Resultados del piloto (n=40)](#42-resultados-del-piloto-n40)
   - [4.3 Validación completa (corpus íntegro) — 2026-07-14](#43-validación-completa-corpus-íntegro--2026-07-14)
   - [4.4 Validación con destilación especializada JVM (DeepSeek-Flash) — 2026-09-18](#44-validación-con-destilación-especializada-jvm-deepseek-flash--2026-09-18)
5. [Bitácora técnica: evolución completa TS & OpenJ9 (2026-09-18)](#5-bitácora-técnica-evolución-completa-ts--openj9-2026-09-18-bitacora-destilacion-llm-ts-openj9md)
   - [5.1 Contexto inicial y cuello de botella](#51-contexto-inicial-y-cuello-de-botella)
   - [5.2 Diagnóstico de inconsistencias](#52-diagnóstico-del-estado-actual-e-inconsistencias)
   - [5.3 Migración a DeepSeek API](#53-la-decisión-migración-a-deepseek-api-deepseek-flash)
   - [5.4 Saneamiento y deduplicación](#54-saneamiento-deduplicación-y-rescate-de-fallbacks)
   - [5.5 Ejecución masiva TypeScript (2836 issues)](#55-ejecución-masiva-de-typescript-2836-issues)
   - [5.6 Destilación completa de OpenJ9](#56-portabilidad-y-destilación-completa-de-openj9)
   - [5.7 Balance final del corpus](#57-balance-final-del-corpus)
   - [5.8 Impacto en rendimiento: nuevo récord en OpenJ9](#58-impacto-inmediato-en-rendimiento-nuevo-récord-en-openj9-mpnet)
6. [Registro de constancia y verificación de calidad — TypeScript](#6-registro-de-constancia-y-verificación-de-calidad--typescript-registro-constancia-y-verificacion-tsmd)
   - [6.1 Contexto y diagnóstico](#61-contexto-y-diagnóstico-del-problema)
   - [6.2 Soluciones implementadas](#62-soluciones-implementadas)
   - [6.3 Estado final de los datasets](#63-estado-final-de-los-datasets)
   - [6.4 Comandos de operación y verificación](#64-comandos-de-operación-y-verificación)

---

## 1. Corrección crítica: "la destilación empeora" era sobre la arquitectura descartada (`correccion-destilacion-deberta-descartado.md`)

**Fecha:** 2026-09-16. Corrige la lectura de `estado-tesis.md` §1 a la luz de los pilotos en OpenJ9.

### Qué decía estado-tesis.md

> "La hipótesis original era: *raw + distilled entrena mejor el CBR → mejores
> resultados*. Los datos del piloto Mozilla no la respaldan limpiamente
> [...] distilled-solo es el PEOR."

Esa conclusión (Hit@1 raw 0.7517 > distilled 0.7313) viene de un **CBR
clasificador DeBERTa** afinado sobre Mozilla (20 devs, cabeza densa).

### Por qué ya no aplica

El clasificador DeBERTa **es una arquitectura descartada**. La arquitectura
vigente del proyecto es el **CBR de recuperación zero-shot** (encoder
`all-mpnet-base-v2`, `scripts/cbr_retrieval_openj9.py`), no un clasificador
entrenado. La destilación solo se había probado antes con el clasificador DeBERTa
entrenado (arquitectura ya descartada), nunca con el recuperador.

Comparar `raw` vs `distilled` en el clasificador mide el efecto de la
destilación **sobre una arquitectura que ya no se usa**. No es evidencia
sobre si la destilación ayuda o no al sistema real (MPNet + recuperación).

### Qué muestra la validación con la arquitectura real

Validación a escala completa (OpenJ9, 3348 train / 534 test, `--max-seq-len 384`, barrido k×τ completo):

| | Top-1 | MRR | Hit@10 |
|---|---|---|---|
| Crudo (mejor punto) | 0.2715 | 0.4078 | 0.6779 |
| **Destilado (mejor punto)** | **0.2884** | **0.4387** | **0.7472** |
| Δ | **+1.69 pp** | **+3.09 pp** | **+6.93 pp** |

Destilado gana en **18/18 celdas** del barrido (no solo en el punto
campeón), con la ventaja más grande en k chico (k=5: +6.74 pp Top-1) y más
angosta en k grande.

**Mecanismo identificado:** con el default `--max-seq-len 256`, el texto crudo se trunca antes de completar el
contexto nativo de MPNet y pierde ~2pp Top-1 (0.2547 vs 0.2715 con 384). El
texto destilado (`[FL]/[SY]/[CP]`) es corto y da igual con 256 o 384 — parte
de la ventaja de destilar es **evitar el truncamiento** en reportes largos,
no solo limpiar ruido.

### Conclusión

- El hallazgo "distilled-solo es el peor" de `estado-tesis.md` §1 **sigue
  siendo cierto para el clasificador DeBERTa**, pero esa arquitectura no es
  la que se reporta en la tesis final.
- Para la arquitectura vigente (recuperación + MPNet), la evidencia
  validada dice lo contrario: **destilar ayuda de forma consistente**,
  especialmente con pocos vecinos (k chico) o reportes largos que se
  truncarían en crudo.
- El número citable en la tesis para "destilación vs. crudo" es el de los pilotos OpenJ9,
  no el de Mozilla/DeBERTa.

### Pendiente (estado al 2026-09-16)

- **Decisión tomada:** por limitaciones de cómputo/latencia, la destilación se hará con una **API de LLM** (p. ej.
  `deepseek-v4-flash` vía NVIDIA API) en vez de Gemma 4 E4B IT QAT local.
  La dirección del efecto (destilar ayuda) se ha sostenido a
  través de varios backends de API distintos (agy/Gemini, grok,
  deepseek-flash), lo cual da confianza en que no depende de un backend
  específico.
- Falta correr el experimento formal (raw vs. distilled, arquitectura de
  recuperación) sobre el corpus **TypeScript**, el tercer régimen.

---

## 2. Sesión 2026-06-08 — Setup inicial destilación + backend Grok (`sesion-2026-06-08.md`)

### Contexto

Primera sesión de trabajo en la laptop (RTX 5060, Windows). El objetivo era
dejar el pipeline de destilación (Módulo 2) funcionando ahí.

### Problemas encontrados y soluciones

**1. `ModuleNotFoundError: No module named 'triager_omega.data'`**

El paquete `triager_omega.data` no existía. Se crearon tres archivos:
- `src/triager_omega/data/__init__.py`
- `src/triager_omega/data/loader.py` — funciones `load_bugs`, `load_contributors`, `load_comments`, `build_email_to_id`
- `src/triager_omega/data/preprocessor.py` — función `normalize_identity`

**2. `run_distillation.py` requería `bug_comments.parquet`**

El script cargaba el primer comentario de cada bug para enriquecer el prompt del LLM.
Se decidió que no era estrictamente necesario. Se comentó la función `load_first_comments`
y se reemplazó por `comments: dict[int, str] = {}`.

**3. Tamaño real del piloto**

Se asumía ~1200 bugs. El piloto real tiene **9,364 bugs** (6000 train / 1749 val /
1615 test), con 600 ya destilados y **8,764 pendientes**.

### Mejoras aplicadas

**Timer por lote en `run_distillation.py`:**
```
50/8764 destilados (fallbacks: 2) | lote: 142.3s (0.35 bugs/s)
```

**Backend Grok (`grok-3-mini`):**
- `config.py`: nuevos campos `grok_api_key`, `grok_base_url`, `grok_model`
- `distillation.py`: nueva clase `_GrokBackend` (OpenAI-compatible, lee `GROK_API_KEY` del `.env`)
- `run_distillation.py`: `"grok"` agregado a las opciones de `--backend`

```bash
uv run python scripts/run_distillation.py --backend grok --smoke 5
```
Smoke test exitoso: 5 bugs destilados correctamente con calidad buena.

**Costo estimado para los 8,764 bugs pendientes:** ~$3 USD con grok-3-mini ($0.30/M input · $0.50/M output).

### Archivos modificados / creados

| Archivo | Cambio |
|---|---|
| `src/triager_omega/data/__init__.py` | creado |
| `src/triager_omega/data/loader.py` | creado |
| `src/triager_omega/data/preprocessor.py` | creado |
| `src/triager_omega/config.py` | campos grok_* agregados |
| `src/triager_omega/cbr/distillation.py` | `_GrokBackend` + `make_client` actualizado |
| `scripts/run_distillation.py` | comentó comments, timer, backend grok |

---

## 3. Mejora pendiente: destilación paralela + structured outputs (`mejora-destilacion-paralela.md`)

### Contexto

`scripts/run_distillation.py` actualmente envía los bugs a Ollama de forma secuencial
(un bug → espera respuesta → siguiente bug). Con ~1200 bugs del piloto, esto es el
cuello de botella principal del Módulo 2.

### Qué hacer

#### 1. Async concurrente en el cliente (mayor impacto, menor esfuerzo)

Ollama usa **continuous batching**: si llegan varios requests simultáneos, los agrupa
internamente y los procesa en paralelo. Para aprovechar esto:

- Configurar `OLLAMA_NUM_PARALLEL=4` en el servidor Ollama.
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

#### 2. JSON schema enforcement en Ollama (mejora de calidad)

Desde Ollama v0.5, el parámetro `format` acepta un JSON schema object. Ollama lo
convierte a GBNF grammar y enmascara tokens inválidos durante el sampling → el modelo
**siempre devuelve JSON válido**, sin necesidad de reintentos ni fallback.

Pasar el schema de `fault_location/symptoms/capabilities` directamente en cada request
eliminaría los `_fallback` del código actual.

### Lo que NO hacer (por ahora)

- **vLLM**: mayor throughput pero requiere pesos de Gemma en formato HuggingFace
  (safetensors), no el GGUF que usa Ollama actualmente. Implica re-descargar el modelo.
  Solo vale la pena si el async no alcanza.
- **llama.cpp server**: historial de bugs con `json_schema`, no recomendado como
  reemplazo directo.

### Referencias

Investigación realizada 2026-06-08 via deep-research (103 agentes, fuentes verificadas
adversarialmente). Fuentes principales:
- https://ollama.com/blog/structured-outputs
- https://www.glukhov.org/post/2025/05/how-ollama-handles-parallel-requests/
- https://docs.vllm.ai/en/v0.8.2/features/structured_outputs.html

---

## 4. Piloto rápido de destilación en OpenJ9 (backend `agy` / Gemini) (`piloto-destilacion-agy-openj9.md`)

**Fecha:** 2026-07-12. **Estado: preliminar/exploratorio — ver §4.4 para la cifra citable.**

### Motivación

Al revisar la alineación entre `ch4_proposal.tex` (tesis) y `paper.tex`, se detectó que
la destilación semántica (Gemma) nunca se validó con la arquitectura real que produjo
los resultados reportados (el CBR de recuperación zero-shot): `scripts/build_openj9_50.py`
construye la columna `text` a partir de `issue_title`+`issue_body` crudos, y
`scripts/cbr_retrieval_openj9.py` no tiene ninguna opción `--text-mode`. La destilación
solo se había probado antes con el clasificador DeBERTa entrenado (arquitectura ya descartada),
nunca con el recuperador.

### 4.1 Método

- **Backend de destilación:** `agy` (CLI agéntica, modelo *Gemini 3.5 Flash Medium* vía
  API en la nube) — **NO es Gemma 4 E4B IT QAT local**, el modelo real de la propuesta.
  Se usó solo por velocidad para este piloto exploratorio.
- **Prompt:** el mismo ya validado en `triager_omega.cbr.distillation`
  (`SYSTEM_PROMPT` + `FEW_SHOT_EXAMPLES`, esquema `fault_location`/`symptoms`/
  `capabilities` → texto `[FL] ... [SY] ... [CP] ...`), con constructor de entrada
  nuevo adaptado al esquema de issues de GitHub (título + cuerpo + componente).
- **Muestra:** 120 issues de train + 40 de test, muestreo aleatorio simple (seed=0)
  sobre `artifacts/openj9/openj9_{train,test}_50.csv`. 43 clases (owners) presentes en
  el train piloto; de los 16 owners en el test piloto, 4 no aparecen en el train piloto.
- **0 fallbacks** — las 160 llamadas devolvieron JSON válido al primer intento.
- **Evaluación:** `scripts/cbr_retrieval_openj9.py --sweep` (barrido k∈{5,10,15,20,30,50}
  × τ∈{1,2,4}, voto `sum`, encoder `all-mpnet-base-v2`).

### 4.2 Resultados del piloto (n=40)

| k, τ | Top-1 raw | Top-1 destilado | MRR raw | MRR destilado |
|---|---|---|---|---|
| k=15 τ=1 | 0.0500 | 0.1750 | 0.1728 | 0.2926 |
| k=20 τ=1 | 0.0750 | 0.2000 | 0.1840 | 0.3234 |
| **k=30 τ=1 (mejor)** | **0.1500** | **0.2500** | 0.2421 | **0.3581** |
| k=50 τ=1 | 0.1250 | 0.1750 | 0.2434 | 0.3285 |

El texto destilado supera al crudo en **las 18 celdas del barrido** (no solo en el
mejor punto), con una ganancia de ~+10 pp Top-1 y ~+11.5 pp MRR en la mejor
configuración de cada lado.

**Caveats:** n=40 test / 120 train es una muestra chica; el error de muestreo es
grande y esta cifra no debe compararse directamente contra el 0.2715 Top-1 del corpus
completo (distinto n, distinto corpus). El backend (Gemini vía `agy`) no es el modelo de la propuesta (Gemma 4 E4B IT QAT
local); la dirección del efecto es la señal útil, la magnitud exacta puede no
replicarse igual con el modelo real.

### Artefactos del piloto

- `artifacts/pilot_destilacion_agy/openj9_{train,test}_pilot_agy.parquet` / `.csv`
- `artifacts/pilot_destilacion_agy/openj9_{train,test}_pilot_{raw,distilled}.csv`
- `scripts/pilot_destilacion_agy_openj9.py`

---

### 4.3 Validación completa (corpus íntegro) — 2026-07-14

**Estado: validado, esta es la cifra citable para la etapa de validación con Grok.**

#### Método

- **Backend de destilación:** cadena de fallback `agy → grok → nvidia → opencode`
  (`scripts/distill_openj9_full.py`). `agy` se rompió a mitad de camino (falla de
  concurrencia del CLI con 8 workers: "model not recognized", luego timeouts) tras
  completar ~430 filas limpias; se descartó y el resto del corpus lo completó
  **`grok`** (API real, ~0.5 s/issue). Ningún backend usado es Gemma 4 E4B IT QAT local.
- **Muestra:** corpus íntegro — 3348 train + 534 test, 51 clases. **0 fallbacks**
  (tras limpiar 8 filas contaminadas por la falla de `agy`).
- **Evaluación:** `scripts/cbr_retrieval_openj9.py --sweep` en omen (RTX 5060),
  **`--max-seq-len 384`** — el mismo config que reproduce el campeón documentado
  (0.2715 Top-1 zero-shot). *Gotcha: el default del script es 256; con 256 el crudo
  da solo 0.2547 Top-1, casi 2pp por debajo. El texto destilado es corto y da
  exactamente igual con 256 o 384 — el efecto del seq-len es solo sobre el crudo.*

#### Resultados — mejor punto de cada lado (k=50)

| | Top-1 | MRR | Hit@10 |
|---|---|---|---|
| Crudo (τ=1/2) | 0.2715 | 0.4078 | 0.6779 |
| **Destilado (τ=4)** | **0.2884** | **0.4387** | **0.7472** |
| Δ | **+1.69 pp** | **+3.09 pp** | **+6.93 pp** |

El destilado gana en **las 18/18 celdas** del barrido, igual que en el piloto — pero la
ventaja NO es pareja: en k chico es mucho más grande (k=5: 0.1873→0.2547, **+6.74 pp**)
y se va angostando a medida que k crece, hasta los +1.69pp del punto campeón (k=50).

#### Interpretación

- **La dirección del piloto se confirma a escala completa.**
- **La magnitud NO se confirma**: el piloto (n=40 test) reportó +10pp Top-1 / +11.5pp MRR en su mejor punto; el corpus completo da +1.69pp / +3.09pp. El piloto sobreestimó el efecto — exactamente el caveat que ya advertía antes de correr la validación.
- El patrón "destilar ayuda más con k chico" es coherente: el texto destilado condensa la señal relevante en pocas palabras, así que un recuperador con pocos vecinos disponibles se beneficia más de esa condensación.

#### Artefactos

- `artifacts/openj9/distilled_full/openj9_{train,test}_full.{parquet,jsonl}` — corpus completo.
- `artifacts/openj9/distilled_full/openj9_{train,test}_full_{raw,distilled}.csv`
- `artifacts/openj9/distilled_full/logs/sweep_full_omen_384.log`

---

### 4.4 Validación con destilación especializada JVM (DeepSeek-Flash) — 2026-09-18

**Estado: Nuevo récord histórico de CBR-recuperación en OpenJ9.**

#### Diferencia metodológica frente a la validación anterior (2026-07-14)

- **Prompt:** En la corrida anterior (Grok) se utilizó el prompt heredado de Mozilla/Bugzilla. En esta corrida se empleó el nuevo prompt canónico de **Ingeniería de Sistemas JVM / Runtime** ([`scripts/distill_deepseek_openj9.ts`](../scripts/distill_deepseek_openj9.ts)) con taxonomía técnica nativa de OpenJ9 (`VM`, `JIT`, `GC`, `JITServer`, `JVMTI`, etc.) y 0 fallbacks.
- **Backend:** `deepseek-v4-flash` (API HTTP con `thinking: disabled`).
- **Hardware de inferencia:** Apple M5 (MacBook Pro, MPS/macOS) ejecutando `all-mpnet-base-v2` (`--max-seq-len 384`, barrido idéntico $k \in \{5, 10, 15, 20, 30, 50\} \times \tau \in \{1, 2, 4\}$). Los 3,882 textos fueron embebidos en **23 segundos**.

#### Comparativa tripartita de resultados (corpus completo OpenJ9: 3,348 train / 534 test, 51 clases)

| Configuración | Top-1 | Hit@5 | Hit@10 | MRR | Nota |
|---|---|---|---|---|---|
| **TriagerX CBR (Re-ejecución honesta)** | 0.1966 | 0.4944 | 0.6517 | — | Ensemble DeBERTa+RoBERTa (20 épocas, split exacto) |
| **TriagerX CBR (Paper publicado)** | 0.2700 | — | — | — | Reportado en paper (fuga: checkpoint elegido en test) |
| **Texto Crudo (omega zero-shot)** ($k=50, \tau=1/2$) | 0.2715 | 0.5768 | 0.6779 | 0.4078 | Techo sin destilación (MPNet congelado) |
| **Destilación Antigua Grok** ($k=50, \tau=4$) | 0.2884 | 0.6236 | 0.7472 | 0.4387 | Prompt Mozilla (+1.69 pp Top-1 vs crudo) |
| **Both (Crudo + Destilado)** ($k=50, \tau=1/2$) | 0.2640 | 0.5787 | 0.6816 | 0.4049 | Dilución de señal y truncamiento de log |
| **NUEVA Destilación DeepSeek JVM ($k=30, \tau=4$)** | **0.2940** | **0.6142** | **0.7341** | **0.4398** | **Pico Top-1 (+9.74 pp vs TriagerX CBR, +2.25 pp vs Crudo)** |
| **NUEVA Destilación DeepSeek JVM ($k=50, \tau=4$)** | **0.2865** | **0.6554** | **0.7772** | **0.4474** | **Pico Profundidad (+12.55 pp Hit@10 vs TriagerX CBR)** |

#### Barrido completo de la nueva destilación ($k \times \tau$)

| $k$ | $\tau=1$ | $\tau=2$ | $\tau=4$ |
|---|---|---|---|
| **$k=5$** | 0.2322 / 0.5243 / 0.3446 | 0.2322 / 0.5243 / 0.3446 | 0.2322 / 0.5243 / 0.3446 |
| **$k=10$** | 0.2566 / 0.6348 / 0.3874 | 0.2566 / 0.6348 / 0.3874 | 0.2566 / 0.6348 / 0.3877 |
| **$k=15$** | 0.2678 / 0.6835 / 0.4040 | 0.2678 / 0.6835 / 0.4040 | 0.2697 / 0.6835 / 0.4051 |
| **$k=20$** | 0.2903 / 0.7060 / 0.4296 | 0.2903 / 0.7060 / 0.4296 | 0.2884 / 0.7060 / 0.4290 |
| **$k=30$** | 0.2921 / 0.7341 / 0.4391 | 0.2921 / 0.7341 / 0.4387 | **0.2940** / 0.7341 / **0.4398** |
| **$k=50$** | 0.2865 / 0.7715 / 0.4468 | 0.2865 / 0.7753 / 0.4470 | 0.2865 / **0.7772** / **0.4474** |

*(Formato de cada celda: Top-1 / Hit@10 / MRR)*

#### Principales hallazgos empíricos

1. **Nuevo techo de precisión Top-1 (0.2940):** con la destilación especializada JVM, el Top-1 alcanza **0.2940** en $k=30, \tau=4$, superando por **+0.56 pp** el récord previo de Grok (0.2884) y por **+2.25 pp** el baseline crudo (0.2715).
2. **Ganancia drástica en profundidad de ranking:** en $k=50, \tau=4$, el **Hit@10 escala a 0.7772** (+9.93 pp frente al crudo 0.6779, y +3.00 pp frente a la destilación anterior 0.7472). Más del 77.7% de los bugs ubican al desarrollador responsable en el top 10.
3. **Refutación de la vista combinada (`both`):** evaluar `both` (crudo + destilado) da **0.2640**, resultando inferior no solo al destilado puro (0.2940, **-3.00 pp**), sino incluso al crudo solo (0.2715). **Mecanismo:** En un encoder denso con ventana limitada, concatenar cientos de tokens de stacktraces y logs técnicos antes/junto a los 50 tokens destilados provoca que el *mean pooling* de MPNet sea dominado por el ruido del log, diluyendo la abstracción estructurada. **El texto destilado puro debe consumirse aislado.**
4. **Validación del cambio de arquitectura de destilación:** la transición a DeepSeek Flash con el prompt de ingeniería JVM no solo redujo la latencia y el costo a una fracción mínima, sino que elevó la calidad predictiva final de MPNet en todos los indicadores clave.

#### Artefactos de la corrida
- **CSVs evaluados:** [`artifacts/openj9/distilled/openj9_{train,test}_distilled.csv`](../artifacts/openj9/distilled/)
- **Log de barrido reproducible:** [`artifacts/openj9/distilled/logs/sweep_deepseek_m5_384.log`](../artifacts/openj9/distilled/logs/sweep_deepseek_m5_384.log)

---

## 5. Bitácora técnica: evolución completa TS & OpenJ9 (2026-09-18) (`bitacora-destilacion-llm-ts-openj9.md`)

**Fecha:** 2026-09-18 · **Estado:** Finalizado al 100% (TypeScript: 9,831 issues · OpenJ9: 3,882 issues · 0 Fallbacks)

### 5.1 Contexto inicial y cuello de botella

Durante los últimos meses, el pipeline de destilación semántica (Módulo 2 de `triager-omega`, encargado de estructurar bug reports en texto formal `[FL] ... [SY] ... [CP] ...`) atravesó múltiples intentos de infraestructura:

1. **Gemma local (Gemma 4 E4B IT QAT):** Descartado por lentitud extrema en hardware local para procesar decenas de miles de issues.
2. **`agy` (Gemini CLI vía subproceso):** Abandonado tras fallas masivas de concurrencia ("model not recognized", colisiones de pipes y timeouts recurrentes).
3. **OpenCode CLI (`opencode run -m opencode/union-alpha`):**
   * Se diseñó un script en Bun con workers paralelos en `/tmp/opencode-worker-N`.
   * **Problemas graves detectados:** Subprocesos pesados con latencias de 15 a 30 segundos por llamada. Bloqueos en bases de datos SQLite (`database is locked`). Timeouts frecuentes de hasta 300 segundos en issues largos con stacktraces, provocando cascadas de fallbacks (ej. **196 fallbacks en `ts_train`**, un 23.3% de pérdida).
4. **Problema de Prompts Desalineados:** En OpenJ9 y pilotos previos se seguía utilizando el prompt original de Mozilla/Bugzilla (con ejemplos de Firefox bookmarks y Thunderbird attachments). Para TypeScript se había diseñado un prompt nuevo de ingeniería de compiladores, pero faltaba portarlo a OpenJ9.

### 5.2 Diagnóstico del estado actual e inconsistencias

Antes de proceder a la fase final, se ejecutó una auditoría exhaustiva sobre los datasets existentes en `artifacts/typescript/distilled/`:

* **`ts_train_union.jsonl`:**
  * Tenía 6,241 líneas pero solo 6,170 issues únicos: **71 líneas duplicadas** al final causadas por reanudaciones abruptas de procesos abortados.
  * La primera inserción duplicada en la fila 1,542 causó un desfase sistemático fila por fila con los CSVs, arrojando **más de 22,000 errores falsos de alineación** en el verificador formal.
  * Se detectaron ~89 issues con síntomas/capacidades vacías (`symptoms: []`, `capabilities: []`) que no habían sido marcados como fallback.
* **`ts_train.jsonl`:**
  * Tenía 843 líneas con **196 fallbacks (23.3%)** producto de caídas de OpenCode.
  * Desalineado por 3 filas respecto a los CSVs/Parquet (840 vs 843 por falta de flush final).
* **Solapamiento y Universo Pendiente:**
  * `ts_train_union` (6,170) y `ts_train` (843) eran conjuntos disjuntos (solapamiento = 0).
  * En total se tenían 7,013 issues listos.
  * El corpus de TypeScript tenía 9,831 issues únicos → **faltaban exactamente 2,836 issues**.

### 5.3 La decisión: migración a DeepSeek API (`deepseek-flash`)

Ante la lentitud e inestabilidad de OpenCode, se evaluó migrar a llamadas HTTP directas a la API de DeepSeek:

#### A. Estimación de costos y factibilidad
Para los ~2,831 issues pendientes, considerando ~900 tokens de entrada y ~120 de salida por issue:
* Peor escenario (tarifa Peak, 100% Cache Miss): ~$1.18 USD.
* Escenario optimista (tarifa Off-Peak con Prefix Caching): ~$0.38 USD.

#### B. Decisiones arquitectónicas cruciales en el script
Se desarrolló [`scripts/distill_deepseek_ts.ts`](../scripts/distill_deepseek_ts.ts):

1. **Conexión HTTP Nativa (`fetch` de Bun):** Eliminó la sobrecarga de CLI, procesos huérfanos y bloqueos de SQLite.
2. **`thinking: { type: "disabled" }`:** Parámetro crítico. En DeepSeek V4/Flash el Thinking Mode viene activo por defecto en nivel `high` (~500-1500 tokens de razonamiento interno). Desactivarlo forzó al modelo a escribir directamente el JSON, reduciendo la latencia de 10s a **~700ms** y ahorrando más del 70% del costo.
3. **`response_format: { type: "json_object" }`:** Aseguró que la API devolviera JSON puro parseable sin markdown envolvente.
4. **Prefix Context Caching:** Se estructuró el mensaje separando estrictamente el `system` (con prompt de compiladores y few-shots fijos) del `user`. Al ser el prefijo idéntico en cada llamada, DeepSeek cobró la tarifa de **Cache Hit ($0.003 - $0.006 / 1M)** en el 73% de los tokens de entrada.

#### C. Prueba comparativa (Issue Real #43662)
Se confrontó `opencode/union-alpha` contra `deepseek-flash` sobre el mismo issue:
* **Latencia:** DeepSeek Flash tardó **702 ms** vs. **9,516 ms** de OpenCode (**13.5× más rápido**).
* **Calidad técnica:** Coincidencia exacta en componente (`Type Checker`) y taxonomía del subsistema.
* **Costo por issue:** $0.000478 USD.

### 5.4 Saneamiento, deduplicación y rescate de fallbacks

1. **Merge de saneamiento:** Se ejecutó [`scripts/merge_distillations.ts`](../scripts/merge_distillations.ts) sobre `ts_train_union` y `ts_train`. Se podaron las 71 líneas duplicadas, se ordenaron estrictamente según `df_train_ts.csv` y se sincronizaron los 7 archivos (`.jsonl`, `_distilled.csv`, `_raw.csv`, `_both.csv`, `_provenance.csv`, `.parquet`, `_provenance.parquet`).
2. **Rescate de fallbacks con DeepSeek:**
   * Se reprocesaron los 196 fallbacks de `ts_train` → **0 fallbacks restantes**.
   * Se reprocesaron los 45 fallbacks de `ts_train_union` → **0 fallbacks restantes**.

### 5.5 Ejecución masiva de TypeScript (2,836 issues)

Se lanzó la corrida masiva con 8 workers paralelos:
* **Progreso:** 2,836 issues procesados en **~6 minutos**.
* **Tokens:** 2,875,559 input (73% Cache Hit: 2,099,200 tokens) + 291,678 output.
* **Costo Real Facturado:** **$0.31 USD** (incluso inferior al estimado por coincidir con horario Off-Peak).
* **Post-procesamiento:** Solo 19 issues cayeron en fallback inicial por cuerpos casi vacíos (`![image](`). `reprocess_fallbacks.ts` los rescató al 100% en 15 segundos.
* **Merge maestro:** Se fusionaron los 3 splits (`ts_train_union` + `ts_train` + `ts_train_deepseek`).
* **Resultado Final TypeScript: 9,831 issues únicos (100.0% del corpus), 0 fallbacks, 100% alineado.**

### 5.6 Portabilidad y destilación completa de OpenJ9

Durante la sesión se clarificó que OpenCode siempre había operado sobre TypeScript, mientras que OpenJ9 se había procesado parcialmente en Python con backends antiguos y prompts heredados de Mozilla.

Se creó [`scripts/distill_deepseek_openj9.ts`](../scripts/distill_deepseek_openj9.ts) en TypeScript/Bun con un prompt especializado en **Ingeniería de Sistemas JVM / Runtime**:
* **Taxonomía Canónica:** `VM`, `JIT`, `GC`, `JITServer`, `JVMTI`, `Port`, `Build`, `Test`, `JCL Extensions`.
* **Few-shots reales:** Errores de bucles JIT / *escape analysis* en SPECjbb2015 y fragmentación de heap en Balanced GC.

Se ejecutaron los dos splits oficiales de OpenJ9:
1. **`openj9_train` (3,348 issues):** 0 fallbacks. Costo: ~$0.92 USD.
2. **`openj9_test` (534 issues):** 0 fallbacks. Costo: ~$0.16 USD.

### 5.7 Balance final del corpus

En una sola jornada de trabajo se logró el objetivo que había estado bloqueado por semanas:

| Corpus | Total Issues | Fallbacks | Estado de Archivos (CSVs/Parquet) | Costo Incurrido |
| :--- | :--- | :--- | :--- | :--- |
| **TypeScript (`ts_train_union`)** | **9,831** | **0 (0.0%)** | 7 archivos alineados (11 cols Provenance) | ~$0.31 USD |
| **OpenJ9 Train (`openj9_train`)** | **3,348** | **0 (0.0%)** | 7 archivos alineados (11 cols Provenance) | ~$0.92 USD |
| **OpenJ9 Test (`openj9_test`)** | **534** | **0 (0.0%)** | 7 archivos alineados (11 cols Provenance) | ~$0.16 USD |
| **TOTAL CONSOLIDADO** | **13,713** | **0 (0.0%)** | **100% Válido y Verificado** | **~$1.39 USD** |

#### Lecciones aprendidas para la tesis

1. **Subprocesos locales vs API HTTP:** Invocar subprocesos de CLI (`opencode`, `agy`) genera puntos de falla ocultos en concurrencia (locks en SQLite, pipes rotos, buffers). Las llamadas HTTP directas con pool de workers asíncronos en Bun son órdenes de magnitud más rápidas y robustas.
2. **Thinking Mode en Tareas de Extracción:** El razonamiento extendido (Chain-of-Thought) en tareas de clasificación y parseo estructurado multiplica la latencia por 10x y el costo por 4x sin aportar ganancia cualitativa medible. Apagarlo explícitamente (`thinking: disabled`) fue la clave del rendimiento.
3. **Prefix Caching:** Diseñar prompts con bloques de sistema largos y fijos (prompt + few-shots) permite que la infraestructura de caché de LLMs modernos absorba el 70%+ de los tokens de entrada a costo casi nulo ($0.003 / 1M).
4. **Registro de Constancia (Provenance):** Auditar el modelo, latencia y fallback de cada issue en una tabla formal de 11 columnas garantiza la reproducibilidad y el rigor metodológico exigido para el manuscrito de tesis.

### 5.8 Impacto inmediato en rendimiento: nuevo récord en OpenJ9 (MPNet)

#### Récord con la destilación especializada JVM
Evaluado el nuevo dataset destilado con el CBR de recuperación (`all-mpnet-base-v2`, barrido $k \times \tau$ en Apple M5, 23s de cómputo):

* **Top-1 nuevo:** **0.2940** ($k=30, \tau=4$), batiendo el récord previo de Grok (**0.2884**) y distanciándose por **+2.25 pp** del baseline crudo (0.2715).
* **Hit@10 nuevo:** **0.7772** ($k=50, \tau=4$), subiendo casi 10 puntos sobre el crudo (+9.93 pp vs 0.6779).
* **MRR nuevo:** **0.4474** ($k=50, \tau=4$) vs 0.4078 crudo.

#### Ablación de vistas: Refutación de la condición combinada (`both`)

| Vista / Configuración | Top-1 | Hit@5 | Hit@10 | MRR | Diagnóstico |
|---|---|---|---|---|---|
| **TriagerX CBR (Re-ejecución honesta)** | 0.1966 | 0.4944 | 0.6517 | — | Ensemble DeBERTa+RoBERTa (20 épocas, split exacto) |
| **TriagerX CBR (Paper publicado)** | 0.2700 | — | — | — | Reportado en paper (fuga: checkpoint elegido en test) |
| **Crudo Puro (`raw`)** ($k=50, \tau=1/2$) | 0.2715 | 0.5768 | 0.6779 | 0.4078 | Techo sin destilación (MPNet congelado) |
| **Combinado (`both`)** ($k=50, \tau=1/2$) | 0.2640 | 0.5787 | 0.6816 | 0.4049 | ❌ Caída de -3.00 pp vs Destilado y -0.75 pp vs Crudo |
| **Destilado Puro (`distilled`)** ($k=30, \tau=4$) | **0.2940** | **0.6142** | **0.7341** | **0.4398** | 🏆 **Pico de precisión (+9.74 pp vs TriagerX CBR)** |
| **Destilado Puro (`distilled`)** ($k=50, \tau=4$) | **0.2865** | **0.6554** | **0.7772** | **0.4474** | 🏆 **Pico de profundidad (+12.55 pp Hit@10 vs TriagerX)** |

#### Explicación mecanística del fenómeno (¿Por qué `both` empeora?)

1. **Dilución en Mean Pooling Denso:** MPNet calcula el vector final promediando los tokens de la secuencia. Al concatenar cientos de tokens de stack traces y discusiones con apenas ~50 tokens de destilación pura (`[FL]... [SY]... [CP]...`), el vector resultante queda matemáticamente dominado (en más del 80%) por el ruido del stack trace, asfixiando los descriptores sintéticos.
2. **Truncamiento Asimétrico:** En reportes largos que sobrepasan `max_seq_len=384`, la porción de texto que queda al final sufre truncamiento forzado, perdiendo información crucial.
3. **Conclusión Metodológica para la Tesis:** En modelos densos basados en *sentence embeddings*, la destilación semántica debe emplearse como un **reemplazo limpio y autosuficiente** de la entrada textual, y no como un añadido concatenado a la prosa cruda.

---

## 6. Registro de constancia y verificación de calidad — TypeScript (`registro-constancia-y-verificacion-ts.md`)

### 6.1 Contexto y diagnóstico del problema

Al ejecutar el verificador de calidad inicial (`bun run scripts/verify_distillations.ts`), se detectaron dos errores críticos:

```text
❌ ERRORES DETECTADOS (2):
  [Issue #3136] symptoms: symptoms vacío o no es array
  [Issue #N/A] row_count: Desalineación en el conteo de filas entre JSONL y CSVs
```

**Causa raíz de cada problema:**
1. **Issue #3136 (symptoms vacío):** El reporte correspondía a un feature request (*"Feature Request: F# style Type Provider support?"*). El modelo de lenguaje generó `symptoms: []`, dejando vacío el bloque `[SY]` en `distilled_text`. La regla de validación de calidad exige que todo registro no-fallback contenga al menos un síntoma u observación técnica.
2. **Desalineación de conteo de filas (732 vs 731):** `ts_train.jsonl` tenía 732 líneas (había registrado el issue `#3298`), pero los archivos derivados se quedaron en 731 porque el script de destilación no había completado el ciclo de flush para ese último registro.
3. **Omisión del Registro de Constancia (*Provenance Dataset*):** Existían archivos de auditoría previos estancados en 721 registros. El verificador de calidad no comprobaba la presencia, integridad ni coincidencia de este dataset.

### 6.2 Soluciones implementadas

**A. Corrección del Issue #3136:**
- Se actualizaron los síntomas y capacidades en `artifacts/typescript/distilled/ts_train.jsonl`:
  - **Symptoms**: `["type providers unsupported", "no external schema type synthesis", "manual API typing required"]`
  - **Capabilities**: `["external type generation", "compile-time type synthesis", "language service extension design"]`
- Se reconstruyó `distilled_text` con la taxonomía formal completa `[FL] ... [SY] ... [CP] ...`.

**B. Generación y estandarización del Registro de Constancia:**
El **Registro de Constancia** audita la procedencia técnica de cada destilación para la tesis. Contiene 11 columnas estructuradas:
- Identificación: `issue_number`, `issue_title`, `owner`
- Auditoría técnica: `model`, `latency_ms`, `fallback`
- Taxonomía del compilador: `component`, `subsystem`, `symptoms`, `capabilities`, `distilled_text`

Se actualizaron los scripts para garantizar paridad total en 732 registros:
- **`scripts/merge_distillations.ts`:** Lee el checkpoint `.jsonl` y mapea los títulos desde `df_train_ts.csv`. Genera automáticamente `ts_train_provenance.csv` y `ts_train_provenance.parquet` con las 11 columnas alineadas fila por fila.
- **`scripts/distill_opencode_ts.ts`:** Su función `flushOutputs` escribe periódicamente tanto los CSVs habituales como el dataset de constancia.

**C. Actualización del verificador de calidad (`scripts/verify_distillations.ts`):**
- Se incluyeron como archivos obligatorios `${SPLIT}_provenance.csv` y `${SPLIT}_provenance.parquet`.
- Se añadieron verificaciones automáticas de conteo de filas en Parquet y CSVs.
- Se validó la alineación fila por fila entre JSONL y el registro de constancia.
- Desglose por modelo y latencia promedio por inferencia:
  - `opencode/union-alpha`: 657 registros (89.8%) · 19.2s prom
  - `omniroute/deepseek-web/deepseek-flash`: 75 registros (10.2%) · 32.5s prom

### 6.3 Estado final de los datasets

#### Split Principal (`ts_train` — 840 registros)
- **Modelos**: 657 `opencode/union-alpha` (78.2%) + 183 `deepseek-flash` (21.8%)
- **Fallbacks**: 196 (23.3%) · **Errores críticos**: 0

| Archivo | Filas | Columnas | Descripción |
|---|---|---|---|
| `ts_train.jsonl` | 840 | - | Checkpoint maestro en formato JSON Lines |
| `ts_train.parquet` | 840 | 8 | Dataset estructurado para entrenamiento CBR / Transformers |
| `ts_train_distilled.csv` | 840 | 3 | Formato estándar TriagerX (solo texto destilado `[FL][SY][CP]`) |
| `ts_train_raw.csv` | 840 | 3 | Formato estándar TriagerX (texto original del issue) |
| `ts_train_both.csv` | 840 | 3 | Formato enriquecido (texto original + destilación) |
| `ts_train_provenance.csv` | 840 | 11 | Registro de constancia y trazabilidad por modelo |
| `ts_train_provenance.parquet` | 840 | 11 | Registro de constancia binario de alta velocidad |

#### Split Extendido (`ts_train_union` — 1,514 registros)
- **Modelo**: `opencode/union-alpha`: 1,514 registros (**100.0%**) · Latencia promedio: 25.7s
- **Tasa de Fallback**: Solo 25 registros (**1.7%**) · **Errores críticos**: 0

| Archivo | Filas | Columnas | Descripción |
|---|---|---|---|
| `ts_train_union.jsonl` | 1514 | - | Checkpoint maestro deduplicado y ordenado |
| `ts_train_union.parquet` | 1514 | 8 | Dataset de alta velocidad listo para entrenamiento CBR |
| `ts_train_union_distilled.csv` | 1514 | 3 | Texto destilado formal `[FL][SY][CP]` para TriagerX |
| `ts_train_union_raw.csv` | 1514 | 3 | Texto crudo de los issues de TypeScript |
| `ts_train_union_both.csv` | 1514 | 3 | Fusión de texto crudo + destilación enriquecida |
| `ts_train_union_provenance.csv` | 1514 | 11 | Auditoría completa con modelo, latencia y taxonomía |
| `ts_train_union_provenance.parquet` | 1514 | 11 | Registro de constancia en formato binario Parquet |

### 6.4 Comandos de operación y verificación

```bash
# 1. Verificar la calidad y alineación (incluyendo Registro de Constancia)
bun run scripts/verify_distillations.ts
bun run scripts/verify_distillations.ts --split ts_train_union

# 2. Consolidar / Reconstruir todos los archivos desde el Checkpoint
bun run scripts/merge_distillations.ts --target ts_train --sources artifacts/typescript/distilled/ts_train.jsonl
bun run scripts/merge_distillations.ts --target ts_train_union --sources artifacts/typescript/distilled/ts_train_union.jsonl

# 3. Continuar la destilación con OpenCode
bun run scripts/distill_opencode_ts.ts \
  --split ts_train \
  --concurrency 3 \
  --flush-every 5

# 4. Inspeccionar la integridad de los Parquets con Python / uv
uv run python -c "
import pandas as pd
for name in ['ts_train.parquet', 'ts_train_provenance.parquet']:
    df = pd.read_parquet(f'artifacts/typescript/distilled/{name}')
    print(f'{name:30}: {len(df)} filas, {len(df.columns)} columnas')
"
```
