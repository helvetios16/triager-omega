# PLAN.md — Triager-Omega

> Sistema de triaje automático de bugs basado en una arquitectura híbrida de cinco módulos: destilación semántica con LLM local, clasificación basada en contenido (DeBERTa), clasificación basada en **interacciones tipadas** (SBERT + Interaction Points commit/review/assignment/discussion + decaimiento temporal, alimentadas por Bugzilla y minería de gecko-dev) y agregación aditiva `FS=NPS+W_f·NIS` con filtro de candidatos activos. Diseño alineado con la implementación real de TriagerX (ver [[triagerx-repo-local]]).

---

## Tabla de contenidos

1. [Visión general y objetivos](#1-visión-general-y-objetivos)
2. [Arquitectura del sistema](#2-arquitectura-del-sistema)
3. [Datos de entrada](#3-datos-de-entrada)
4. [Módulo 1 — Preprocesamiento y balanceo](#4-módulo-1--preprocesamiento-y-balanceo)
5. [Módulo 2 — Destilación semántica (Gemma 4 E4B)](#5-módulo-2--destilación-semántica-gemma-4-e4b)
6. [Módulo 3 — Clasificador CBR (DeBERTa)](#6-módulo-3--clasificador-cbr-deberta)
7. [Módulo 4 — Clasificador IBR (SBERT + decaimiento)](#7-módulo-4--clasificador-ibr-sbert--decaimiento)
8. [Módulo 5 — Agregador WRA + filtro de candidatos](#8-módulo-5--agregador-wra--filtro-de-candidatos)
9. [Diseño de prompts few-shot para Gemma](#9-diseño-de-prompts-few-shot-para-gemma)
10. [Hiperparámetros y valores por defecto](#10-hiperparámetros-y-valores-por-defecto)
11. [Métricas y protocolo de evaluación](#11-métricas-y-protocolo-de-evaluación)
12. [Stack tecnológico](#12-stack-tecnológico)
13. [Estructura del proyecto](#13-estructura-del-proyecto)
14. [Plan de implementación por fases](#14-plan-de-implementación-por-fases)
15. [Riesgos y mitigaciones](#15-riesgos-y-mitigaciones)
16. [Glosario](#16-glosario)

---

## 1. Visión general y objetivos

### 1.1 Problema

Cuando se reporta un nuevo bug en un proyecto grande como Mozilla, el triaje manual (decidir qué desarrollador debe atenderlo) es costoso, lento y propenso a errores. **Triager-Omega** busca automatizar esta tarea: dado un reporte de bug nuevo, recomendar los **Top-K desarrolladores** más probables de resolverlo.

### 1.2 Hipótesis de diseño

1. La asignación efectiva de un bug combina **señales de contenido** (¿qué dice el reporte?) y **señales de interacción histórica** (¿quién ha trabajado en bugs similares y qué tan recientemente?).
2. Un LLM local puede **destilar y normalizar** el texto crudo de un reporte para reducir ruido y facilitar la clasificación.
3. El problema sufre una **cola larga severa** (algunos desarrolladores tienen miles de bugs, la mayoría apenas decenas). Esto obliga a (a) restringir el espacio de candidatos a desarrolladores activos y (b) ponderar el muestreo durante el entrenamiento.

### 1.3 Objetivos medibles

| Objetivo | Meta inicial |
|---|---|
| Hit@1 | ≥ 0.25 |
| Hit@5 | ≥ 0.55 |
| Hit@10 | ≥ 0.70 |
| MRR | ≥ 0.35 |
| Latencia inferencia (end-to-end, por bug) | ≤ 5 s en Apple M5 |
| Memoria pico en inferencia | ≤ 16 GB |

Estas metas son orientativas y se ajustarán tras la línea base.

---

## 2. Arquitectura del sistema

### 2.1 Diagrama lógico (alto nivel)

```
                ┌───────────────────────────────────┐
                │   Nuevo Bug Report (entrada)      │
                │   Summary, Product, Component,    │
                │   Severity, Priority, 1er comment │
                └──────────────┬────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │  Módulo 2: Destilación Semántica (Gemma)     │
        │  → {fault_location, symptoms, capabilities}  │
        └──────────────┬───────────────────────────────┘
                       │
           ┌───────────┴────────────┐
           ▼                        ▼
 ┌──────────────────┐     ┌────────────────────────────────┐
 │ Módulo 3: CBR    │     │ Módulo 4: IBR                  │
 │ DeBERTa fine-    │     │ SBERT + interacciones tipadas  │
 │ tuned → NPS      │     │ (commit/review/assign/discus.) │
 │                  │     │ × IP × decaimiento → NIS       │
 └────────┬─────────┘     └───────────────┬────────────────┘
          │                               │
          └──────────────┬────────────────┘
                         ▼
        ┌──────────────────────────────────────────┐
        │  Módulo 5: WRA + filtro de candidatos    │
        │  Solo devs en directorio activo (Mod. 1) │
        │  FS = NPS + W_f · NIS                     │
        └──────────────┬───────────────────────────┘
                       ▼
              Top-K recomendaciones
```

### 2.2 Flujo de entrenamiento vs. flujo de inferencia

| Fase | Módulo 1 | Módulo 2 | Módulo 3 | Módulo 4 | Módulo 5 |
|---|---|---|---|---|---|
| **Entrenamiento** | construye directorio activo + muestreo ponderado | destila todo el corpus offline | fine-tuning DeBERTa sobre destilados | indexa embeddings SBERT + construye Interaction Table (Bugzilla + gecko-dev) | calibra `W_f` en validación |
| **Inferencia** | aplica filtro de candidatos | destila bug nuevo | predice P(dev) → NPS | recupera bugs similares + agrega interacciones tipadas → NIS | `FS = NPS + W_f·NIS` y filtra |

---

## 3. Datos de entrada

### 3.1 Origen

Mozilla BugsRepo (Bugzilla), curado previamente y entregado en formato **Parquet** dentro de `data/`. El usuario coloca los archivos manualmente; no se descargan.

### 3.2 Archivos

| Archivo | Filas | Columnas | Rol |
|---|---|---|---|
| `bug_metadata.parquet` | 225 196 | 46 | Tabla principal de bugs. PK: `Bug Id`. Label: `Assigned To` (email). |
| `contributors.parquet` | 50 345 | 14 | Perfil de contribuidores. PK: `Contributor Id`. Nombre: `User Name`. Métrica clave: `Assigned To and Fixed`. |
| `bug_comments.parquet` | ~1 011 057 | 8 | Comentarios. FK: `Bug Id`, `Author Id` (float→int). Flag: `Bug Report` (True = comentario inicial). |

**Fuente derivada (no entregada, se genera):**

| Artefacto | Origen | Rol |
|---|---|---|
| `artifacts/repo_interactions.parquet` | minería de **gecko-dev** (`repo_miner.py`, `git log`) | Interacciones `commit`/`review` para el IBR (señal de código, la más fuerte). Cols: `bug_id, Contributor Id, kind, timestamp, commit_hash`. Ver [[msr-repo-interactions]]. |

### 3.3 Campos relevantes

**Bug Meta Data**
- `Bug Id` (PK)
- `Summary` (texto corto)
- `Product`, `Component` (taxonomía organizacional)
- `Priority`, `Severity` (etiquetas categóricas)
- `Assigned To` → **etiqueta objetivo**
- `Creation Time`, `Last Change Time`

**Contribution Information**
- `Contributor Id`, `User Name`
- `Assigned To and Fixed` → umbral de actividad (≥ 20)
- `Last Activity`

**Bug Report Comments**
- `Comment Id`, `Bug Id`, `Author Id`, `Creator`, `Text`, `Time`
- `Bug Report` (bool): si True, es el primer comentario / cuerpo del reporte

### 3.4 Etiqueta objetivo

`Assigned To` (Bug Meta Data) son **emails**. Se normaliza al `Contributor Id` entero (clave canónica del sistema) mediante un **puente vía la tabla de comentarios**: `comments.Creator` (email) ↔ `comments.Author Id` (id). El join directo contra `contributors.User Name` NO funciona (no son emails). Cobertura del puente: ~99 % de los asignados reales.

### 3.5 Splits

- **Temporal split** (recomendado para evitar fuga): ordenar por `Creation Time` y partir en 70 / 15 / 15 (train / val / test).
- Justificación: el triaje real es prospectivo; un split aleatorio inflaría artificialmente las métricas porque podría usar bugs futuros para predecir pasados.

---

## 4. Módulo 1 — Preprocesamiento y balanceo

### 4.1 Directorio de candidatos válidos

**Definición**: subconjunto de desarrolladores considerados "activos y competentes" para recibir asignaciones.

**Criterio de inclusión** (decisión tras EDA: umbral **≥ 50**):
- `Assigned To and Fixed ≥ 50` en el dataset de Contribución → **538 devs**, cobertura 92 % de bugs con asignado real. (Parametrizable en `config.py`; ≥20→661 devs/94 %, ≥10→772/95 %.)
- (Opcional, evaluable) `Last Activity` dentro de un horizonte temporal (ej. últimos 24 meses respecto al último bug del split de train).

Se materializa como un `set[ContributorId]` y se serializa en disco (`artifacts/active_candidates.json`).

### 4.2 Muestreo ponderado para entrenamiento

Para mitigar la cola larga: cada bug `b` con desarrollador asignado `d` recibe un peso de muestreo

```
w(b) = 1 / freq_train(d)
```

donde `freq_train(d)` es el número de bugs asignados a `d` dentro del split de entrenamiento. Estos pesos se pasan al `DataLoader` vía `WeightedRandomSampler` de PyTorch.

**Rationale**: en lugar de truncar a los devs prolíficos (perdería información) o sobre-muestrear los raros con réplicas exactas (riesgo de overfit), el muestreo ponderado da a cada época una vista balanceada estocásticamente.

### 4.3 Filtrado de bugs

Se descartan bugs:
- Sin `Assigned To` o con `Assigned To` no en el directorio de candidatos.
- Sin `Summary` o con `Summary` vacío.
- Cuyo `Assigned To` mapea a usuarios automáticos (ej. `nobody@mozilla.org`).

### 4.4 Limpieza de texto (mínima)

- Eliminar firmas, stacktraces excesivamente largos (mantener primeras N=30 líneas).
- Normalizar URLs a `<URL>` para reducir vocabulario.
- Conservar contenido semántico; no se aplica stemming ni stopword removal (los transformers se encargan).

### 4.5 Salidas

- `artifacts/active_candidates.json` — lista de `Contributor Id` elegibles.
- `artifacts/label_encoder.json` — mapeo `Contributor Id ↔ class_idx` para DeBERTa.
- `artifacts/sample_weights_train.npy` — pesos para el sampler.

---

## 5. Módulo 2 — Destilación semántica (Gemma 4 E4B)

### 5.1 Motivación

Los reportes de bug son ruidosos, heterogéneos y mezclan información de UI, logs, opinión y reproducción. Una **destilación estructurada** previa:
- Reduce la longitud efectiva → cabe mejor en DeBERTa (max 512 tokens).
- Normaliza vocabulario heterogéneo en categorías estables.
- Proporciona una señal interpretable y auditable.

### 5.2 Servidor local

- **LM Studio** sirviendo Gemma 4 E4B en `http://localhost:1234/v1`.
- Cliente: librería `openai` Python con `base_url` apuntando al servidor local.
- Modelo: `gemma-4-e4b` (nombre exacto se ajusta al cargado en LM Studio).
- Modo: `chat.completions.create`, `temperature=0.0`, `response_format={"type":"json_object"}` (si soportado; sino post-validación).

### 5.3 Entrada al LLM

Concatenación estructurada:

```
Summary: <Summary>
Product: <Product>
Component: <Component>
Severity: <Severity>
Priority: <Priority>
Initial report:
<primer comentario donde Bug Report == True, truncado a 1500 chars>
```

### 5.4 Salida esperada (JSON)

```json
{
  "fault_location": {
    "product": "Firefox",
    "component": "Bookmarks & History",
    "subsystem": "places database sync"
  },
  "symptoms": ["crash on startup", "data loss after restart"],
  "capabilities": ["sqlite", "places architecture", "asynchronous IO"]
}
```

### 5.5 Validación y fallback

- Parsear JSON; si falla, intentar 1 reintento con prompt reforzado.
- Si vuelve a fallar, fallback determinista: usar `Product`/`Component` como `fault_location`, `Severity` como `symptoms`, y cadena vacía para `capabilities`.
- Log de fallas para análisis posterior.

### 5.6 Caché

Toda destilación se cachea en `artifacts/distillations.parquet` con clave `Bug Id`. La destilación es **idempotente**: una vez generada, no se recomputa.

### 5.7 Pre-procesamiento por lotes

Para 225k bugs, ejecutar en lote offline. Asumiendo 1-2 s por bug, son ~60-120 horas en una sola máquina; paralelizable con varias instancias de LM Studio o quedando como tarea overnight de fase 2.

### 5.8 Concatenación para módulos posteriores

El texto destilado que pasa a DeBERTa y SBERT se construye así:

```
[FL] {product} {component} {subsystem} [SY] {symptom1} {symptom2} ... [CP] {capability1} {capability2} ...
```

Los marcadores `[FL]`, `[SY]`, `[CP]` ayudan al modelo a distinguir secciones.

> **Nota (decisión):** el destilado **se concatena al texto crudo** (Summary + primer comentario truncado), no lo reemplaza. La destilación es *lossy* (descarta logs, stack traces, nombres de archivo) y esos tokens raros son señal discriminativa entre devs; usar solo el destilado empobrecería el dataset para un clasificador. El crudo aporta el detalle fino; el destilado, la estructura normalizada y el puente bug↔skill del campo `capabilities`. Las dos vistas juntas son ≥ que cualquiera por separado. Se evalúa crudo-vs-destilado-vs-ambos en §11.2.4.

### 5.9 Augmentación multi-vista para la cola larga (experimental)

Técnica para enriquecer las clases raras (devs con pocos bugs), donde el muestreo ponderado (§4.2) solo **repite** los mismos textos y arriesga overfitting al string exacto. La multi-vista genera, **solo para devs con `freq_train(d) < U_aug`** (umbral, p.ej. 10), `M` destilaciones alternativas del mismo bug (con `temperature>0` o pidiendo reformulación), produciendo varias vistas con el mismo significado y distinto fraseo → DeBERTa aprende el concepto, no el texto literal.

- Restringida a la cola: NO se augmentan devs frecuentes (gasto LLM innecesario).
- Hereda la clave de caché de §5.6 con sufijo de vista (`Bug Id` + `view_idx`).
- Riesgo: el LLM puede alucinar `capabilities` falsas → el filtro de calidad (§4.3) y la validación post-hoc (§9.5) deben aplicarse también a las vistas augmentadas.
- **No reemplaza** la frecuencia inversa: se apila sobre ella. La comparación de ambas (y del submuestreo) es la ablación de §11.3.

---

## 6. Módulo 3 — Clasificador CBR (DeBERTa)

### 6.1 Modelo base

`microsoft/deberta-v3-base` (HuggingFace). Razones:
- DeBERTa v3 supera a BERT/RoBERTa en clasificación con tamaño similar (~184M params).
- Tokenizer SentencePiece maneja bien código y términos técnicos.
- Compatible con MPS (Apple Silicon).

### 6.2 Arquitectura

```
Texto destilado
    ↓ tokenizer (max_length=512, truncation, padding)
DeBERTa-v3-base
    ↓ pooler_output ([CLS] hidden state)
Dropout(p=0.1)
    ↓
Linear(hidden_size → num_classes)
    ↓
Softmax (en inferencia)
```

`num_classes` = tamaño del directorio activo (Módulo 1), estimado en orden de 1-3 k tras filtrado.

### 6.3 Loss

Cross-entropy estándar. Se considera `label_smoothing=0.05` para regularizar.

### 6.4 Entrenamiento

| Item | Valor por defecto |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| LR scheduler | Linear warmup (10%) + linear decay |
| Batch size | 16 (limitado por MPS) |
| Gradient accumulation | 2 |
| Epochs | 3-5 |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Mixed precision | bf16 (MPS) |
| Sampler | `WeightedRandomSampler` del Módulo 1 |
| Early stopping | sobre `val Hit@5`, paciencia 1 |

### 6.5 Salida en inferencia

Vector de probabilidades `p_CBR ∈ ℝ^{num_classes}`. Se preserva como diccionario `{Contributor Id: prob}` para Módulo 5.

### 6.6 Persistencia

- `artifacts/cbr_model/` con pesos, tokenizer y `label_encoder.json`.
- Versionado por timestamp.

---

## 7. Módulo 4 — Clasificador IBR (SBERT + interacciones tipadas + decaimiento)

> **Actualización (alineado con la implementación real de TriagerX).** El IBR ya **no** se alimenta solo de comentarios. Recupera bugs similares por SBERT y suma señal de **interacciones tipadas** de los desarrolladores sobre esos bugs, donde cada **tipo** de interacción pesa distinto (un *Interaction Point*). La discusión (comentar) es la señal **más débil**; las contribuciones de código (commit / review) son la **más fuerte**. La versión anterior (solo comentarios) infrautilizaba la señal: se quedaba solo con `discussion`.

### 7.1 Motivación

CBR mira solo texto. IBR explota **señal de interacción histórica**: quién ha *trabajado* en bugs parecidos y qué tan recientemente, distinguiendo el **tipo** de trabajo. Resolver/parchear un bug (commit) pesa más que solo comentarlo (discussion).

### 7.2 Tabla de interacciones (Interaction Table)

El IBR consume una tabla larga `(bug_id, Contributor Id, kind, timestamp)` construida desde **tres fuentes**, porque los parquets de Bugzilla por sí solos solo dan `assignment` y `discussion`; la señal de código (`commit`/`review`) se recupera minando git:

**Esquema de 3 pesos (fiel a TriagerX).** TriagerX usa 3 Interaction Points: `contribution` (commit+PR), `direct_assignment`, `discussion`. Replicamos eso: `commit` y `review` siguen siendo *kinds* distintos en los datos (no hay que re-minar), pero **comparten el mismo peso `ip_contribution`**, igual que TriagerX mapea `pull_request`+`commits` a `contribution_score`. Valores = `triagerx_config.yaml`.

| `kind` | Fuente | Peso (config) | IP |
|---|---|---|---|
| `commit` | Minería de **gecko-dev** (`repo_miner.py` → `artifacts/repo_interactions.parquet`); autor del commit que referencia `Bug <id>` | `ip_contribution` | **1.5** |
| `review` | gecko-dev: nicks en `r=`/`a=` del mensaje de commit | `ip_contribution` | **1.5** |
| `assignment` | Bugzilla `Assigned To` (y, si se incorpora, el historial de cambios de `assigned_to` vía la API REST de Bugzilla) | `ip_assignment` | 0.5 |
| `discussion` | `bug_comments.parquet`: autor de cada comentario (limpio: sin bots, dedup — `scripts/build_discussion_interactions.py`) | `ip_discussion` | 0.1 |

**Resolución de identidad** (a `Contributor Id` entero, clave canónica):
- `commit` trae **email** → puente email→id vía la tabla de comentarios (`Creator`↔`Author Id`). Ver [[identity-bridge-via-comments]].
- `review` trae **`:nick`** → puente vía `contributors.User Name` (formato `(:nick)`). Ver [[msr-repo-interactions]].
- `assignment`/`discussion` ya vienen con id o email resoluble por el mismo puente.

### 7.3 Modelo de embeddings

`sentence-transformers/all-mpnet-base-v2` (768 dim) — **el modelo exacto que usa TriagerX** (`config.sbert_model`). Sin fine-tuning: SBERT off-the-shelf da buena similitud de issues.

### 7.4 Índice de bugs históricos

- Embeber el texto destilado (Módulo 2) de **cada bug del split de entrenamiento**.
- Indexar con FAISS (`IndexFlatIP` sobre vectores L2-normalizados ≡ coseno).
- Persistir en `artifacts/ibr_faiss.index` + `artifacts/ibr_bug_ids.npy`.

### 7.5 Recuperación

Para un bug nuevo:
1. Embeber su texto destilado.
2. Recuperar los **Top-k** bugs más similares (`config.ibr_top_k_retrieve = 20`, valor del paper), filtrando por umbral `s_j ≥ τ` (`config.ibr_tau = 0.6`, valor de TriagerX).
3. Para cada bug similar con similitud `s_j`, obtener sus filas de la Interaction Table (§7.2).

### 7.6 Puntuación: similitud × Interaction Point × decaimiento

Réplica de `_get_historical_contributors` / Algoritmo 1 de TriagerX. Para cada interacción `(dev, kind, t)` sobre un bug similar `I_j`:

```
Δt = (t_now − t)            # días; t_now = Creation Time del bug nuevo
IS[dev] += s_j · IP[kind] · exp(−λ · Δt)
```

- `λ = 0.01` (1/día; vida media ≈ 69 d), `config.ibr_lambda`.
- `IP[kind]` de la tabla §7.2: `commit` y `review` → `config.ip_contribution` (1.5); `assignment` → `config.ip_assignment` (0.5); `discussion` → `config.ip_discussion` (0.1). Réplica de `_get_contribution_point` de TriagerX (3 pesos).
- Solo cuentan `dev` en el directorio activo (Módulo 1) e interacciones con `t < t_now` (**anti-fuga temporal**).

### 7.7 Normalización (NIS — Normalized Interaction Score)

Min-max sobre los devs con `IS > 0` (Ecuación 7 de TriagerX), solo si existe al menos uno:

```
NIS[dev] = (IS[dev] − min(IS)) / (max(IS) − min(IS))
```

Si ningún dev interactuó con bugs similares, `NIS` queda en 0 para todos (el IBR no aporta y el bug se decide solo por CBR).

### 7.8 Salida

Diccionario `{Contributor Id: NIS}` cubriendo solo a desarrolladores con al menos una interacción (de cualquier tipo) en los Top-k bugs similares. Lo consume el Módulo 5.

---

## 8. Módulo 5 — Agregador WRA + filtro de candidatos

> **Actualización.** La agregación pasa de una **combinación convexa** `α·CBR+(1−α)·IBR` a la fórmula **aditiva** de TriagerX `FS = NPS + W_f·NIS`. Diferencia clave: el IBR ya no *diluye* al CBR para todos los devs; solo **bonifica** a quienes tienen historial de interacción relevante, dejando intacto el ranking de contenido para el resto.

### 8.1 Candidate-constrained decoding

El CBR (Módulo 3) ya opera sobre el directorio activo (sus clases = devs activos). El IBR (Módulo 4) ignora interacciones de devs fuera del directorio. Cualquier dev fuera del directorio → score 0 en ambas modalidades.

### 8.2 Scores normalizados de entrada

- `NPS` (Normalized Prediction Score): `p_CBR` normalizado (el CBR ya entrega softmax sobre devs activos). Cubre **todos** los devs activos.
- `NIS` (Normalized Interaction Score): salida del Módulo 4 (§7.7), min-max en `[0,1]`. Cubre **solo** devs con interacción en bugs similares; el resto = 0.

### 8.3 Weighted Ranking Aggregation (fórmula de TriagerX)

```
FS(dev) = NPS(dev) + W_f · NIS(dev)            # Ecuación 8
```

- `W_f ∈ (0,1)` se sintoniza por grid search en validación (`config.ibr_w_f`, default **0.7** = `similarity_prediction_weight` de TriagerX).
- Interpretación: repos con interacciones densas (tipo OpenJ9) toleran `W_f` alto; repos con interacciones esparsas se benefician de `W_f` bajo. Mozilla, con minería de gecko-dev, debería tolerar `W_f` medio-alto.
- Aditivo, no convexo: un dev sin historial relevante conserva su `NPS` puro; uno con historial fuerte recibe un empujón proporcional a `W_f·NIS`.

### 8.4 Manejo de devs ausentes en IBR

Si un dev aparece en CBR pero no en IBR, su `NIS = 0` y `FS = NPS` (sin penalización). Como el CBR cubre todos los devs activos, no hay devs ausentes en NPS.

### 8.5 Top-K

Ordenar por `FS` desc y devolver los primeros K. Para evaluación K ∈ {1, 3, 5, 10}; operativo K=5.

### 8.6 Explicabilidad (opcional, fase 4)

Para cada recomendación, exponer:
- Contribución CBR vs IBR.
- Los 3 bugs históricos más similares que llevaron a la recomendación IBR.
- La destilación que alimentó CBR.

---

## 9. Diseño de prompts few-shot para Gemma

### 9.1 Estructura general

```
[SYSTEM]
You are an expert software engineer who distills bug reports into structured JSON.
Always return valid JSON with exactly three keys: "fault_location", "symptoms", "capabilities".
Do not include explanations.

[USER — ejemplo 1]
<entrada estructurada>

[ASSISTANT — ejemplo 1]
<JSON>

... (3-5 ejemplos) ...

[USER — bug real]
<entrada estructurada>
```

### 9.2 Criterios para elegir few-shot examples

- Diversidad por `Product` (al menos uno de Firefox, Thunderbird, Core, DevTools).
- Diversidad por `Severity` (critical, major, minor).
- Diversidad por tipo de síntoma (crash, regression, perf, UI glitch, data corruption).
- Longitud media (~150-300 palabras de cuerpo).
- Etiqueta histórica conocida (para validar que el JSON tiene sentido).

### 9.3 Ejemplo (esqueleto)

**Entrada**:
```
Summary: Firefox crashes when restoring session with > 200 tabs
Product: Firefox
Component: Session Restore
Severity: critical
Priority: P1
Initial report:
Reproducible 100% on macOS 14. Steps: open 200+ tabs, quit, reopen. Crash signature
mozilla::SessionStore::Restore. Stacktrace shows ...
```

**Salida**:
```json
{
  "fault_location": {
    "product": "Firefox",
    "component": "Session Restore",
    "subsystem": "tab state deserialization"
  },
  "symptoms": ["crash on restore", "high tab count trigger", "reproducible 100%"],
  "capabilities": ["session restore internals", "C++ crash debugging", "macOS platform"]
}
```

### 9.4 Reglas para el modelo (en el SYSTEM)

1. `fault_location.product` debe ser exactamente el valor de `Product` del input.
2. `fault_location.component` debe ser exactamente el valor de `Component`.
3. `fault_location.subsystem` debe inferirse del texto; máx 5 palabras.
4. `symptoms`: lista de 1-5 elementos cortos (≤ 6 palabras cada uno).
5. `capabilities`: lista de 1-5 habilidades técnicas concretas.
6. Si no hay información suficiente para un campo, usar lista vacía o string vacío en subsystem.

### 9.5 Validación post-hoc

- Schema check con `pydantic` o `jsonschema`.
- Si `fault_location.product != input.Product` → corregir automáticamente al valor de input.
- Si listas vacías en ambas, marcar bug como "degradado" y excluir del fine-tuning.

---

## 10. Hiperparámetros y valores por defecto

### 10.1 Tabla maestra

| Módulo | Hiperparámetro | Valor por defecto | Rango exploración |
|---|---|---|---|
| 1 | Umbral `Assigned To and Fixed` | 20 | {10, 20, 50} |
| 1 | Horizonte `Last Activity` | 24 meses | {12, 24, 36, ∞} |
| 2 | Modelo LLM | gemma-4-e4b | — |
| 2 | Temperature | 0.0 | — |
| 2 | Max output tokens | 512 | — |
| 2 | Few-shot examples | 4 | {2, 4, 6} |
| 3 | Modelo base | deberta-v3-base | — |
| 3 | Learning rate | 2e-5 | {1e-5, 2e-5, 5e-5} |
| 3 | Batch size | 16 | {8, 16, 32} |
| 3 | Epochs | 4 | {3, 4, 5} |
| 3 | Max seq length | 512 | — |
| 3 | Label smoothing | 0.05 | {0.0, 0.05, 0.1} |
| 3 | Dropout cabeza | 0.1 | {0.0, 0.1, 0.2} |
| 4 | Modelo SBERT | all-mpnet-base-v2 | {MiniLM, mpnet} |
| 4 | Top-k similares (`ibr_top_k_retrieve`) | 20 | {20, 50, 100} |
| 4 | Umbral similitud τ (`ibr_tau`) | 0.6 | [0, 2] paso 0.1 (TriagerX) |
| 4 | λ decaimiento (1/día, `ibr_lambda`) | 0.01 | {0.005, 0.01, 0.02, 0.05} |
| 4 | IP contribution commit+review (`ip_contribution`) | 1.5 | [0, 2] paso 0.1 |
| 4 | IP assignment (`ip_assignment`) | 0.5 | [0, 2] paso 0.1 |
| 4 | IP discussion (`ip_discussion`) | 0.1 | [0, 2] paso 0.1 |
| 5 | `W_f` (peso del IBR en `FS=NPS+W_f·NIS`) | 0.7 | (0, 1) paso 0.1 (TriagerX) |
| 5 | K (Top-K operativo) | 5 | {1, 3, 5, 10} eval |

### 10.2 Estrategia de tuning

1. Línea base con valores por defecto.
2. Tuning de DeBERTa (lr, epochs).
3. Tuning de IBR (τ, λ, Top-k e Interaction Points) — grid search estilo TriagerX (Tabla II del paper).
4. Tuning de `W_f` en validación con DeBERTa e IBR ya congelados.

---

## 11. Métricas y protocolo de evaluación

### 11.1 Métricas

- **Hit@K (Accuracy@K)**: fracción de bugs en test cuyo `Assigned To` real aparece en el Top-K recomendado. K ∈ {1, 3, 5, 10}.
- **MRR**: `1/|D_test| · Σ 1/rank(true_dev)`; rango 0..1. Si el dev real no está en Top-10, contribuye 0.
- **Latencia**: tiempo medio y p95 de inferencia end-to-end (destilación + CBR + IBR + agregación).
- **Memoria**: pico de RAM/VRAM/MPS durante inferencia.
- **Cobertura del directorio activo**: % de bugs en test cuyo `Assigned To` real pertenece al directorio (techo superior de Hit@K).

### 11.2 Comparativas

Reportar Hit@K y MRR para:
1. Baseline: solo CBR.
2. Baseline: solo IBR.
3. Sistema completo (CBR + IBR + agregación `FS=NPS+W_f·NIS`).
4. Variantes de input para CBR (Módulo 2): **crudo solo** vs **destilado solo** vs **crudo+destilado (dos vistas, §5.8)**. Verifica que concatenar no degrada frente al crudo y mide cuánto aporta el destilado.

### 11.3 Ablations

- Sin filtro de candidatos.
- Sin decaimiento temporal en IBR (λ=0).
- Sin muestreo ponderado.
- **Estrategias de balanceo de la cola larga** (comparativa controlada, mismo modelo/seed/split, métrica clave = Hit@K **segmentado por frecuencia del dev**, sobre todo en el bucket cola):
  - (a) **Baseline** sin balanceo (sampler uniforme).
  - (b) **Frecuencia inversa** — `WeightedRandomSampler`, peso `1/freq_train(d)` (§4.2). *Estado: ya implementado en `scripts/train_cbr_quick.py`.*
  - (c) **Multi-vista** — augmentación por destilación para `freq_train(d) < U_aug` (§5.9), con sampler uniforme.
  - (d) **Submuestreo** (undersampling) — recortar bugs de los devs frecuentes a un tope `C_max` para aplanar la distribución.
  - (e) **Combinada** — multi-vista (c) **+** frecuencia inversa (b), para confirmar que se apilan.

  Reportar Hit@1/5/10 y MRR globales y por bucket de frecuencia (cola / cuerpo / cabeza). Hipótesis: (b) sube la cola pero overfittea su texto; (c) da diversidad real; (e) ≥ todas.
- **IBR solo-comentarios** (la versión anterior del plan): desactivar `commit`/`review`/`assignment`, dejar solo `discussion`. Mide cuánto aporta la señal de código frente al IBR original.
- **Por tipo de interacción** (3 tipos, esquema TriagerX): quitar `contribution` (commit+review, sin minería gecko-dev), quitar `assignment`, quitar `discussion` — aísla la contribución de cada fuente poniendo su IP a 0.
- **¿Influye `assignment`? (ablación dedicada)**: correr el IBR **con** (`ip_assignment=0.5`) y **sin** (`ip_assignment=0`) y comparar Hit@K/MRR. Motivación: `assignment` sale del campo `Assigned To`, que **es la misma etiqueta que predice el CBR**, así que podría ser redundante con él (ver §8). Hipótesis: aporta poco sobre el CBR, mucho menos que `contribution`. **Referencia TriagerX**: su IBR usa assignment (`direct_assignment` + `last_assignment`) con peso 0.5 (vs contribution 1.5); su grid recorre `[0, 2]` paso 0.1 incluyendo el 0, así que ya prueba implícitamente "sin assignment". Decisión a tomar con el resultado: mantener `ip_assignment=0.5` o llevarlo a 0.
- **Agregación aditiva vs convexa**: `FS=NPS+W_f·NIS` (TriagerX) frente al viejo `α·CBR+(1−α)·IBR`.

### 11.4 Reportes

Salida en `artifacts/eval/<timestamp>/`:
- `metrics.json`
- `confusion_topk.csv`
- `latency.json`
- `per_product_breakdown.csv` (Hit@K segmentado por `Product`).

---

## 12. Stack tecnológico

| Capa | Herramienta | Notas |
|---|---|---|
| Gestor de entorno | `uv` | `pyproject.toml`, lockfile reproducible |
| Datos | `pandas`, `pyarrow` | lectura de parquet |
| LLM local | LM Studio + Gemma 4 E4B | API en `http://localhost:1234/v1` |
| Cliente LLM | `openai` (Python) | `base_url` overrideado |
| Fine-tuning | `transformers`, `torch` | DeBERTa-v3-base |
| Embeddings | `sentence-transformers` | MiniLM / MPNet |
| Índice vectorial | `faiss-cpu` | suficiente para 200k vectores |
| Métricas | `scikit-learn`, custom | Hit@K, MRR manual |
| Aceleración | `torch` + MPS | Apple M5 Silicon |
| Logging | `loguru` | runs reproducibles |
| Config | `pydantic-settings` | jerárquica |
| Tests | `pytest` | unit + integration |

### 12.1 Variables de entorno

```
DISTILL_BACKEND=ollama                       # ollama | lmstudio | google
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=gemma4:e4b-it-qat               # think=False → JSON directo sin preámbulo
GOOGLE_API_KEY=...                           # solo si DISTILL_BACKEND=google
TORCH_DEVICE=mps
DATA_DIR=./data
ARTIFACTS_DIR=./artifacts
```

---

## 13. Estructura del proyecto

```
triager-omega/
├── data/                              # parquet (manual)
│   ├── Bug_meta_data_curado.parquet
│   ├── Contribution_information_dataset_curado.parquet
│   └── Bug_Report_Comments_curado.parquet
├── artifacts/                         # outputs generados
│   ├── active_candidates.json
│   ├── label_encoder.json             # 450 devs (Módulo 1)
│   ├── splits.parquet                 # 450 devs, 64k bugs
│   ├── sample_weights_train.npy
│   ├── repo_interactions.parquet      # commit/review (gecko-dev) — IBR
│   ├── discussion_interactions.parquet# discussion limpio (sin bots, dedup) — IBR
│   ├── pilot/                         # FASE CBR (piloto top-20 devs)
│   │   ├── splits.parquet             #   subset piloto
│   │   ├── label_encoder.json         #   20 devs
│   │   ├── distillations.parquet      #   ETAPA 1: salida de la destilación (Bug Id, distilled_text)
│   │   └── cbr_model/                 #   ETAPA 3: DeBERTa entrenado (pesos + tokenizer + metrics.json)
│   ├── balance_experiment/            # resultados ablación cola larga
│   └── eval/
├── src/
│   └── triager_omega/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── preprocessor.py
│       │   └── repo_miner.py
│       ├── cbr/                       # FASE CBR (Módulos 2-3)
│       │   ├── __init__.py
│       │   ├── pilot.py               #   selecciona el subset piloto → artifacts/pilot/
│       │   ├── distillation.py        #   ETAPA 1: cliente backend-agnóstico (ollama/lmstudio/google) + prompt + cache
│       │   └── train.py               #   ETAPA 2-3: DeBERTa sobre crudo+destilado → guarda cbr_model/
│       ├── modules/                   # (pendiente) ibr.py, aggregator.py
│       └── pipeline.py                # (pendiente) orquestador end-to-end
├── scripts/
│   ├── eda.py                         # estadísticas del dataset
│   ├── run_distillation.py           # corre la ETAPA 1 (batch reanudable, --smoke N)
│   ├── train_cbr_quick.py            # entrenamiento exploratorio (referencia para cbr/train.py)
│   ├── balance_experiment.py         # ablación de 5 estrategias de cola larga (§11.3)
│   ├── build_discussion_interactions.py  # limpia comments → discussion_interactions.parquet
│   ├── inspect_bug.py                # audita la relación bug ↔ 3 fuentes del IBR
│   └── test_{api,ollama}.py          # pruebas de conexión a los backends LLM
├── tests/
├── pyproject.toml
├── CLAUDE.md
├── PLAN.md
└── README.md
```

### 13.1 Responsabilidades por archivo

| Archivo | Contenido |
|---|---|
| `config.py` | dataclass/pydantic con todas las rutas y hparams |
| `data/loader.py` | funciones `load_bugs()`, `load_contributors()`, `load_comments()` |
| `data/preprocessor.py` | directorio de candidatos, splits temporales, sample weights |
| `data/repo_miner.py` | minería de gecko-dev (`git log`) → `repo_interactions.parquet`; parsea `Bug <id>` y `r=`/`a=`; puente de identidad a `Contributor Id` |
| `cbr/pilot.py` | selecciona el subset piloto (top-20 devs, cap 300/100) → `artifacts/pilot/` |
| `cbr/distillation.py` | **ETAPA 1**: cliente backend-agnóstico (ollama/lmstudio/google), prompt+few-shot, validación+fallback, `to_distilled_text` `[FL][SY][CP]` |
| `cbr/train.py` | **ETAPA 2-3**: DeBERTa sobre texto crudo+destilado (dos vistas), sampler ponderado, `adam_eps=1e-4` (MPS), Hit@K/MRR → guarda `cbr_model/` |
| `modules/ibr.py` *(pendiente)* | index FAISS, retrieval Top-k, Interaction Table (Bugzilla+MSR), scoring `s·IP·decay` → NIS |
| `modules/aggregator.py` *(pendiente)* | fusión `FS=NPS+W_f·NIS` y filtro de candidatos |
| `pipeline.py` *(pendiente)* | clase `TriagerPipeline` orquestadora |

**Flujo de archivos de la fase CBR:** `distillation` → `artifacts/pilot/distillations.parquet` → `train` (lee el parquet + une crudo) → `artifacts/pilot/cbr_model/`. La destilación se corre **una vez**; el entrenamiento puede repetirse leyendo el parquet sin volver a llamar al LLM.

---

## 14. Plan de implementación por fases

### Fase 1 — Datos y andamiaje (semana 1)

**Objetivos**: dejar listo el pipeline de datos.

- [ ] Configurar `uv` y `pyproject.toml` con dependencias.
- [ ] Implementar `data/loader.py` con lectura perezosa por columnas.
- [ ] Script `scripts/eda.py`: distribución de `Assigned To`, longitud de summaries, cobertura de `Bug Report=True` en comentarios. Salida a consola.
- [ ] Implementar `data/preprocessor.py`: directorio activo, splits temporales, sample weights.
- [ ] Persistir `active_candidates.json`, `label_encoder.json`.
- [ ] Tests unitarios de carga y filtrado.

**Entregable**: script CLI que produce todos los artefactos de Módulo 1.

### Fase 2 — Módulos individuales (semanas 2-4)

> **NOTA — Estrategia piloto.** La fase CBR se implementa primero como **piloto a escala TriagerX** (top-20 devs, cap 300/dev; `cbr/pilot.py` → `artifacts/pilot/`) para validar el diseño completo antes de correr sobre los 450 devs / 64k bugs. Decisión y dimensionamiento: ver [[cbr-pilot-destilacion]].

**Semana 2 — Módulo 2 (Destilación)** — *implementado en `src/triager_omega/cbr/distillation.py`*
- [x] Cliente **backend-agnóstico** (`ollama` default / `lmstudio` / `google`) con `openai`/`genai`.
- [x] Builder de prompt con few-shot (reusa `scripts/test_api.py`/`test_ollama.py`); `think=False` en Ollama → JSON directo.
- [x] Validación JSON + fallback determinista.
- [x] Caché en parquet idempotente y reanudable (`scripts/run_distillation.py`, `--smoke N`).
- [x] `cbr/pilot.py`: subset piloto (9.364 bugs). Smoke test OK (calidad de `capabilities` validada).
- [ ] Lanzar destilación batch del piloto con Ollama → `artifacts/pilot/distillations.parquet` (~4 h).

**Semana 3 — Módulo 3 (CBR / DeBERTa)** — *implementado en `src/triager_omega/cbr/train.py`*
- [x] Dataset PyTorch sobre texto crudo+destilado (dos vistas §5.8).
- [x] Modelo DeBERTa con cabeza lineal (base `train_cbr_quick.py`; `adam_eps=1e-4` para MPS).
- [x] Loop de entrenamiento con `WeightedRandomSampler` + métricas Hit@K/MRR.
- [x] Guardado del modelo entrenado → `artifacts/pilot/cbr_model/` (pesos + tokenizer + `metrics.json`).
- [ ] Correr el entrenamiento (tras el batch de destilación) y leer métricas.
- [ ] Inferencia batched (`predict()` → NPS) para el agregador.
- [x] **Experimento de balanceo de cola larga (§11.3):** `scripts/balance_experiment.py` — 5 variantes con mismo seed+split, Hit@K/MRR segmentados por bucket. *(Surrogate v1: clasificador TF-IDF+lineal y multi-vista por EDA, hasta tener DeBERTa+destilación. Hallazgo: el Hit@5 global es plano (~0.74) entre todas → enmascara la cola; segmentado, `combined` casi cuadruplica el Hit@10 de la cola (0.055→0.242) a costa de ~5pp en la cabeza.)* Pendiente: re-correr con backend DeBERTa (Módulo 3) y vistas LLM reales (Módulo 2).
- [ ] **Experimento de input CBR (§11.2.4):** entrenar crudo-solo vs destilado-solo vs crudo+destilado y comparar. *Ya soportado:* `cbr/train.py --text-mode {raw,distilled,both}`.

**Semana 4 — Módulo 4 (IBR / SBERT + interacciones tipadas)**
- [x] `data/repo_miner.py`: minería gecko-dev → `repo_interactions.parquet` (commit/review) con puente de identidad. *(ya implementado)*
- [ ] Construir la **Interaction Table** unificada: unir Bugzilla (`assignment` vía `Assigned To`, `discussion` vía `bug_comments`) + MSR (`commit`/`review`).
  - [x] Señal `discussion`: `scripts/build_discussion_interactions.py` → `discussion_interactions.parquet` (145.798 interacciones; filtra bots por dominio y dedup del 58% de filas repetidas).
  - [ ] Señal `assignment` desde `Assigned To` (+ history si se quiere timestamp del cambio).
  - [ ] Concat de las 3 fuentes → `interaction_table.parquet`.
- [ ] Embeddings batch del corpus train (SBERT mpnet) + índice FAISS.
- [ ] Función de retrieval Top-k con umbral τ.
- [ ] Scoring `IS += s_j·IP[kind]·exp(−λΔt)` + normalización NIS (min-max).
- [ ] Tests: λ=0 vs λ=0.01; ablation solo-`discussion` vs todos los tipos; anti-fuga temporal (`t<t_now`).
- [ ] **Ablación `assignment` on/off (§11.3):** correr el IBR con `ip_assignment=0.5` (valor TriagerX) vs `ip_assignment=0`, mismos splits/seed, y comparar Hit@K/MRR. Objetivo: medir si `assignment` aporta sobre el CBR o es redundante (es el mismo campo `Assigned To` que la etiqueta). TriagerX lo usa con peso 0.5 y su grid incluye 0 — replicar esa lógica. Salida: decidir el valor de `ip_assignment` (mantener 0.5 o llevar a 0).

### Fase 3 — Integración (semana 5)

- [ ] Módulo 5: agregador `FS=NPS+W_f·NIS` + filtro.
- [ ] `pipeline.py`: clase end-to-end con métodos `fit()`, `predict(bug_dict)`, `predict_topk()`.
- [ ] Integración smoke test sobre 100 bugs de validación.
- [ ] CLI de inferencia (`python -m triager_omega.pipeline predict --bug-id ...`).

### Fase 4 — Evaluación y tuning (semana 6)

- [ ] `evaluate.py`: Hit@K, MRR, latencia, memoria.
- [ ] Línea base con defaults.
- [ ] Grid de `W_f` (peso del IBR en `FS=NPS+W_f·NIS`).
- [ ] Tuning de IBR: τ, λ, Top-k e Interaction Points (commit/review/assignment/discussion).
- [ ] Ablations (incl. solo-`discussion`, por tipo de interacción, aditiva vs convexa).
- [ ] Reporte final en `artifacts/eval/`.
- [ ] Script `scripts/error_analysis.py`: análisis de fallos (qué clases se confunden, en qué `Product`). Salida a consola + CSV.

### Fase 5 (opcional) — Mejoras

- Explicabilidad expuesta en API.
- Re-entrenamiento incremental al llegar nuevos bugs.
- Servidor HTTP (FastAPI) para integración con Bugzilla.

---

## 15. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Destilación lenta (225k bugs × 1-2 s) | Alta | Medio | paralelizar instancias de LM Studio; priorizar bugs en train; cache obligatoria |
| Gemma produce JSON inválido frecuente | Media | Alto | validación + reintento + fallback determinista; ajustar few-shot |
| Cola larga sigue dominando tras muestreo | Media | Alto | combinar con focal loss; explorar umbral de candidatos más alto |
| DeBERTa no cabe en MPS con bs=16 | Media | Medio | gradient accumulation; bs=8; checkpoint activations |
| Bug "nobody@mozilla.org" como label real | Alta | Alto | excluir en preprocesamiento; documentar en README |
| Fuga temporal en IBR | Media | Medio | filtrar **toda** interacción (commit/review/assign/discussion) con `timestamp < bug.CreationTime` |
| Identidad commit/review no resuelve a `Contributor Id` | Media | Medio | doble puente (email vía comments, `:nick` vía `User Name`); loguear y descartar no resolubles |
| Minería gecko-dev desalineada con rango de bugs | Baja | Medio | `repo_mine_since` alineado a `Creation Time`; validar cobertura de `Bug <id>` parseados |
| Cobertura del directorio < 80% en test | Media | Medio | ajustar umbral; reportar como techo |
| Latencia de inferencia > 5 s | Media | Bajo | precomputar destilación; batch SBERT |
| MPS con bf16 inestable | Baja | Medio | fallback a fp32; medir impacto |

---

## 16. Glosario

- **CBR**: Content-Based Recommender. Recomienda basándose en el contenido del bug.
- **IBR**: Interaction-Based Recommender. Recomienda según el historial de interacciones **tipadas** (commit, review, assignment, discussion) en bugs similares, ponderadas por tipo y recencia.
- **Interaction Point (IP)**: peso asignado a cada **tipo** de interacción. Esquema de 3 pesos como TriagerX: `contribution` (commit+review) = 1.5 > `assignment` = 0.5 > `discussion` = 0.1. Refleja cuánta señal de pertenencia aporta cada acción.
- **NPS / NIS / FS**: Normalized Prediction Score (salida CBR), Normalized Interaction Score (salida IBR) y Final Score `FS = NPS + W_f·NIS` (agregación de TriagerX, Ec. 8).
- **WRA**: Weighted Ranking Aggregation. Aquí, la fusión aditiva `FS = NPS + W_f·NIS` (no una combinación convexa).
- **Interaction Table**: tabla larga `(bug_id, Contributor Id, kind, timestamp)` que alimenta al IBR, construida desde Bugzilla (assignment/discussion) + minería de gecko-dev (commit/review).
- **Destilación**: extracción estructurada (JSON) de información clave de un texto crudo, hecha por un LLM.
- **Directorio activo**: subconjunto de desarrolladores elegibles como recomendación, filtrado por actividad mínima.
- **Decaimiento temporal**: ponderación exponencial decreciente con la edad del evento.
- **Hit@K**: métrica binaria de éxito si el valor real está en el Top-K predicho.
- **MRR**: Mean Reciprocal Rank, promedio del recíproco del rango del valor real.
- **Candidate-constrained decoding**: restringir el espacio de predicción solo al directorio activo.
- **Temporal split**: partición train/val/test ordenada por tiempo, simulando despliegue real.

---

*Documento vivo: actualizar tras cada decisión arquitectónica relevante.*
