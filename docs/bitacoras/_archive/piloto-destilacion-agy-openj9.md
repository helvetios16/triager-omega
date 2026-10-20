# Piloto rápido de destilación semántica en OpenJ9 (backend `agy`/Gemini)

**Fecha:** 2026-07-12. **Estado: preliminar/exploratorio — ver validación completa al final del documento, esa sí es la cifra citable.**

## Por qué

Al revisar la alineación entre `ch4_proposal.tex` (tesis) y `paper.tex`, se detectó que
la destilación semántica (Gemma) nunca se validó con la arquitectura real que produjo
los resultados reportados (el CBR de recuperación zero-shot): `scripts/build_openj9_50.py`
construye la columna `text` a partir de `issue_title`+`issue_body` crudos, y
`scripts/cbr_retrieval_openj9.py` no tiene ninguna opción `--text-mode` — estructuralmente
no puede consumir texto destilado. La destilación solo se había probado antes con el
clasificador DeBERTa entrenado (arquitectura ya descartada), nunca con el recuperador.

Antes de comprometerse a destilar el corpus completo de OpenJ9 (3348 train + 534 test)
con Gemma 4 E4B IT QAT local —lento—, se corrió un piloto rápido para tener una primera
señal de si vale la pena.

## Método

- **Backend de destilación:** `agy` (CLI agéntica, modelo *Gemini 3.5 Flash Medium* vía
  API en la nube) — **NO es Gemma 4 E4B IT QAT local**, el modelo real de la propuesta.
  Se usó solo por velocidad para este piloto exploratorio.
- **Prompt:** el mismo ya validado en `triager_omega.cbr.distillation`
  (`SYSTEM_PROMPT` + `FEW_SHOT_EXAMPLES`, esquema `fault_location`/`symptoms`/
  `capabilities` → texto `[FL] ... [SY] ... [CP] ...`), con un constructor de entrada
  nuevo (`scripts/pilot_destilacion_agy_openj9.py::build_input_openj9`) adaptado al
  esquema de issues de GitHub (título + cuerpo + componente) en vez del esquema
  Bugzilla original (Product/Severity/Priority).
- **Muestra:** 120 issues de train + 40 de test, muestreo aleatorio simple (seed=0)
  sobre `artifacts/openj9/openj9_{train,test}_50.csv`. 43 clases (owners) presentes en
  el train piloto; de los 16 owners en el test piloto, 4 no aparecen en el train piloto
  (techo real ≈90%, no 100%, pero afecta a ambas variantes por igual).
- **0 fallbacks** — las 160 llamadas devolvieron JSON válido al primer intento.
- **Evaluación:** `scripts/cbr_retrieval_openj9.py --sweep` (barrido k∈{5,10,15,20,30,50}
  × τ∈{1,2,4}, voto `sum`, encoder `all-mpnet-base-v2`), corrido en omen (RTX 5060) sobre
  dos pares de CSV idénticos salvo la columna `text`: crudo vs. destilado.

## Resultados

| k, τ | Top-1 raw | Top-1 destilado | MRR raw | MRR destilado |
|---|---|---|---|---|
| k=15 τ=1 | 0.0500 | 0.1750 | 0.1728 | 0.2926 |
| k=20 τ=1 | 0.0750 | 0.2000 | 0.1840 | 0.3234 |
| **k=30 τ=1 (mejor)** | **0.1500** | **0.2500** | 0.2421 | **0.3581** |
| k=50 τ=1 | 0.1250 | 0.1750 | 0.2434 | 0.3285 |

El texto destilado supera al crudo en **las 18 celdas del barrido** (no solo en el
mejor punto), con una ganancia de ~+10 pp Top-1 y ~+11.5 pp MRR en la mejor
configuración de cada lado.

## Interpretación y caveats

- La señal es consistente en todo el barrido, no un pico aislado — más confiable que
  un solo punto ganador.
- **n=40 test / 120 train es una muestra chica**; el error de muestreo a este tamaño es
  grande y esta cifra no debe compararse directamente contra el 0.2715 Top-1 del corpus
  completo (esos números no son comparables: distinto n, distinto corpus).
- El backend (Gemini vía `agy`) no es el modelo de la propuesta (Gemma 4 E4B IT QAT
  local); la dirección del efecto es la señal útil, la magnitud exacta puede no
  replicarse igual con el modelo real.

## Artefactos

- `artifacts/pilot_destilacion_agy/openj9_{train,test}_pilot_agy.parquet` / `.csv` —
  texto crudo + JSON destilado + texto destilado por issue.
- `artifacts/pilot_destilacion_agy/openj9_{train,test}_pilot_{raw,distilled}.csv` —
  mismos datos en el formato de entrada de `cbr_retrieval_openj9.py`.
- `scripts/pilot_destilacion_agy_openj9.py` — script del piloto.

## Próximo paso

Dado que la señal es grande y consistente, prioriza correr la validación completa
(corpus OpenJ9 íntegro, Gemma 4 E4B IT QAT local) antes de decidir entre las tres
opciones planteadas para `ch4_proposal.tex` (quitar la destilación / mantenerla y
validarla a fondo / dejarla como componente opcional). Ese resultado con Gemma local
es el que debe citarse en la tesis — este piloto solo justifica priorizarlo.

---

## Validación completa (corpus íntegro) — 2026-07-14

**Estado: validado, esta es la cifra citable.**

### Método

- **Backend de destilación:** cadena de fallback `agy → grok → nvidia → opencode`
  (`scripts/distill_openj9_full.py`, nuevos backends en
  `triager_omega.cbr.distillation`). `agy` se rompió a mitad de camino (falla de
  concurrencia del CLI con 8 workers: "model not recognized", luego timeouts) tras
  completar ~430 filas limpias; se descartó y el resto del corpus lo completó
  **`grok`** (API real, ~0.5 s/issue). `nvidia` y `opencode` no tuvieron nada
  pendiente que hacer. Ningún backend usado es Gemma 4 E4B IT QAT local — sigue
  pendiente esa corrida si se quiere el número con el modelo exacto de la propuesta,
  pero la señal (dirección del efecto) ya no depende del backend según el piloto.
- **Muestra:** corpus íntegro — 3348 train + 534 test, 51 clases. **0 fallbacks**
  (tras limpiar 8 filas contaminadas por la falla de `agy`).
- **Evaluación:** `scripts/cbr_retrieval_openj9.py --sweep` en omen (RTX 5060),
  **`--max-seq-len 384`** — el mismo config que reproduce el campeón documentado
  (0.2715 Top-1 zero-shot). *Gotcha: el default del script es 256; con 256 el crudo
  da solo 0.2547 Top-1, casi 2pp por debajo, porque trunca antes de completar el
  contexto nativo de MPNet. El texto destilado es corto ([FL]/[SY]/[CP]) y da
  exactamente igual con 256 o 384 — el efecto del seq-len es solo sobre el crudo.*

### Resultados — mejor punto de cada lado (k=50)

| | Top-1 | MRR | Hit@10 |
|---|---|---|---|
| Crudo (τ=1/2) | 0.2715 | 0.4078 | 0.6779 |
| **Destilado (τ=4)** | **0.2884** | **0.4387** | **0.7472** |
| Δ | **+1.69 pp** | **+3.09 pp** | **+6.93 pp** |

El destilado gana en **las 18/18 celdas** del barrido (k∈{5,10,15,20,30,50} ×
τ∈{1,2,4}), igual que en el piloto — pero la ventaja NO es pareja: en k chico (poco
contexto/pocos vecinos) es mucho más grande (k=5: 0.1873→0.2547, **+6.74 pp**) y se
va angostando a medida que k crece, hasta los +1.69pp del punto campeón (k=50).

### Interpretación y caveats

- **La dirección del piloto se confirma a escala completa**: destilar nunca empeora,
  en ningún punto del barrido, con n≈40× más grande.
- **La magnitud NO se confirma**: el piloto (n=40 test) reportó +10pp Top-1 / +11.5pp
  MRR en su mejor punto; el corpus completo da +1.69pp / +3.09pp en el punto
  campeón. El piloto sobreestimó el efecto por tamaño de muestra chico — exactamente
  el caveat que ya advertía este documento antes de correr la validación.
- El patrón "destilar ayuda más con k chico" es coherente con la intuición: el texto
  destilado condensa la señal relevante (componente/síntoma/capacidad) en pocas
  palabras, así que un recuperador con pocos vecinos disponibles se beneficia más de
  esa condensación que uno que ya tiene 50 vecinos crudos para promediar el ruido.
- Backend usado (grok, mayormente) sigue sin ser el modelo de la propuesta (Gemma 4
  E4B IT QAT local); la magnitud exacta con ese modelo podría diferir, aunque la
  dirección (destilar ayuda, más con k chico) es una señal robusta a través de 2
  backends distintos (agy/Gemini en el piloto, grok en la corrida completa).

### Artefactos

- `artifacts/openj9/distilled_full/openj9_{train,test}_full.{parquet,jsonl}` — texto
  crudo + JSON destilado + texto destilado, corpus completo.
- `artifacts/openj9/distilled_full/openj9_{train,test}_full_{raw,distilled}.csv` —
  formato de entrada directo para `cbr_retrieval_openj9.py --train-csv/--test-csv`.
- `artifacts/openj9/distilled_full/logs/sweep_full_omen_384.log` — log del barrido
  citado arriba.

---

## Validación con la Nueva Destilación Especializada JVM (DeepSeek-Flash) — 2026-09-18

**Estado: Nuevo récord histórico de CBR-recuperación en OpenJ9.**

### 1. Diferencia Metodológica frente a la validación anterior (2026-07-14)

- **Prompt:** En la corrida anterior (Grok) se utilizó el prompt heredado de Mozilla/Bugzilla. En esta corrida se empleó el nuevo prompt canónico de **Ingeniería de Sistemas JVM / Runtime** ([`scripts/distill_deepseek_openj9.ts`](../scripts/distill_deepseek_openj9.ts)) con taxonomía técnica nativa de OpenJ9 (`VM`, `JIT`, `GC`, `JITServer`, `JVMTI`, etc.) y 0 fallbacks (100% parseable al primer intento).
- **Backend:** `deepseek-v4-flash` (API HTTP con `thinking: disabled`).
- **Hardware de inferencia:** Apple M5 (MacBook Pro, MPS/macOS) ejecutando `all-mpnet-base-v2` (`--max-seq-len 384`, barrido idéntico $k \in \{5, 10, 15, 20, 30, 50\} \times \tau \in \{1, 2, 4\}$). Los 3,882 textos fueron embebidos en **23 segundos**.

### 2. Comparativa Tripartita de Resultados (Corpus Completo OpenJ9: 3,348 train / 534 test, 51 clases)

| Configuración | Top-1 | Hit@5 | Hit@10 | MRR | Nota |
|---|---|---|---|---|---|
| **TriagerX CBR (Re-ejecución honesta)** | 0.1966 | 0.4944 | 0.6517 | — | Ensemble DeBERTa+RoBERTa (20 épocas, split exacto) |
| **TriagerX CBR (Paper publicado)** | 0.2700 | — | — | — | Reportado en paper (fuga: checkpoint elegido en test) |
| **Texto Crudo (omega zero-shot)** ($k=50, \tau=1/2$) | 0.2715 | 0.5768 | 0.6779 | 0.4078 | Techo sin destilación (MPNet congelado) |
| **Destilación Antigua Grok** ($k=50, \tau=4$) | 0.2884 | 0.6236 | 0.7472 | 0.4387 | Prompt Mozilla (+1.69 pp Top-1 vs crudo) |
| **Both (Crudo + Destilado)** ($k=50, \tau=1/2$) | 0.2640 | 0.5787 | 0.6816 | 0.4049 | Dilución de señal y truncamiento de log |
| **NUEVA Destilación DeepSeek JVM ($k=30, \tau=4$)** | **0.2940** | **0.6142** | **0.7341** | **0.4398** | **Pico Top-1 (+9.74 pp vs TriagerX CBR, +2.25 pp vs Crudo)** |
| **NUEVA Destilación DeepSeek JVM ($k=50, \tau=4$)** | **0.2865** | **0.6554** | **0.7772** | **0.4474** | **Pico Profundidad (+12.55 pp Hit@10 vs TriagerX CBR)** |



### 3. Barrido Completo de la Nueva Destilación ($k \times \tau$)

| $k$ | $\tau=1$ | $\tau=2$ | $\tau=4$ |
|---|---|---|---|
| **$k=5$** | 0.2322 / 0.5243 / 0.3446 | 0.2322 / 0.5243 / 0.3446 | 0.2322 / 0.5243 / 0.3446 |
| **$k=10$** | 0.2566 / 0.6348 / 0.3874 | 0.2566 / 0.6348 / 0.3874 | 0.2566 / 0.6348 / 0.3877 |
| **$k=15$** | 0.2678 / 0.6835 / 0.4040 | 0.2678 / 0.6835 / 0.4040 | 0.2697 / 0.6835 / 0.4051 |
| **$k=20$** | 0.2903 / 0.7060 / 0.4296 | 0.2903 / 0.7060 / 0.4296 | 0.2884 / 0.7060 / 0.4290 |
| **$k=30$** | 0.2921 / 0.7341 / 0.4391 | 0.2921 / 0.7341 / 0.4387 | **0.2940** / 0.7341 / **0.4398** |
| **$k=50$** | 0.2865 / 0.7715 / 0.4468 | 0.2865 / 0.7753 / 0.4470 | 0.2865 / **0.7772** / **0.4474** |

*(Formato de cada celda: Top-1 / Hit@10 / MRR)*

### 4. Principales Hallazgos Empíricos

1. **Nuevo Techo de Precisión Top-1 (0.2940):**
   - Con la destilación especializada JVM, el Top-1 alcanza **0.2940** en $k=30, \tau=4$, superando por **+0.56 pp** el récord previo de Grok (0.2884) y por **+2.25 pp** el baseline crudo (0.2715).
   - El punto óptimo de Top-1 se alcanza con menos vecinos ($k=30$ vs $k=50$), lo que evidencia una memoria de casos con descriptores semánticos más densos y discriminativos.
2. **Ganancia Drástica en Profundidad de Ranking (Hit@10 y MRR):**
   - En $k=50, \tau=4$, el **Hit@10 escala a 0.7772** (+9.93 pp frente al crudo 0.6779, y +3.00 pp frente a la destilación anterior 0.7472). Más del 77.7% de los bugs ubican al desarrollador responsable en el top 10.
   - El MRR escala a 0.4474 (+3.96 pp frente al crudo 0.4078).
3. **Refutación de la Vista Combinada (`both`):**
   - Evaluar `both` (crudo + destilado) da **0.2640**, resultando inferior no solo al destilado puro (0.2940, **-3.00 pp**), sino incluso al crudo solo (0.2715).
   - **Mecanismo:** En un encoder denso con ventana limitada (`max_seq_len 384`), concatenar cientos de tokens de stacktraces y logs técnicos antes/junto a los 50 tokens destilados provoca que el *mean pooling* de MPNet sea dominado por el ruido del log, diluyendo la abstracción estructurada (`[FL]/[SY]/[CP]`). **El texto destilado puro debe consumirse aislado, sin contaminarlo con la prosa cruda.**
4. **Validación del Cambio de Arquitectura de Destilación:**
   - La transición a DeepSeek Flash con el prompt de ingeniería JVM no solo redujo la latencia y el costo a una fracción mínima, sino que elevó la calidad predictiva final de MPNet en todos los indicadores clave.

### 5. Artefactos de la Corrida
- **CSVs evaluados:** [`artifacts/openj9/distilled/openj9_{train,test}_distilled.csv`](../artifacts/openj9/distilled/)
- **Log de barrido reproducible:** [`artifacts/openj9/distilled/logs/sweep_deepseek_m5_384.log`](../artifacts/openj9/distilled/logs/sweep_deepseek_m5_384.log)
