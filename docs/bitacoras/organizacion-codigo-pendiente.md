# Organización pendiente del código

> Revisión arquitectónica del código activo realizada antes de implementar cambios.
>
> **Estado:** diagnóstico histórico previo a la refactorización. Varias propuestas ya están implementadas; consultar el [estado actualizado y los pendientes](pendientes-tras-refactorizacion.md).
>
> **Principio rector:** DeBERTa y los clasificadores supervisados ya no son parte de la línea activa. Se preservan solo como trazabilidad en `scripts/legacy/`. La arquitectura vigente es destilación especializada → CBR por recuperación MPNet, con BM25/RRF cuando el dominio lo justifique e IBR únicamente si añade señal complementaria.

---

## 1. Contexto y avances ya realizados

Los refactors recientes ya avanzaron en la dirección correcta:

- Se movieron el entrenamiento y predictor DeBERTa a `scripts/legacy/`.
- Se eliminó `src/triager_omega/cbr/train.py` y prototipos/shell scripts obsoletos.
- Se centralizaron `vote_scores`, BM25, RRF y métricas de ranking.
- Se extrajo parcialmente un motor compartido de DeepSeek en `scripts/distill/deepseek_engine.ts`.

Sin embargo, aún no hay una frontera clara entre:

1. **núcleo reutilizable** (`src/triager_omega/`),
2. **adaptadores de corpus** (Mozilla, OpenJ9, TypeScript), y
3. **scripts de benchmark o ejecución experimental**.

La meta no es una reescritura grande: es que el código activo sea fácil de entender, reproducir y extender —sobre todo para completar Mozilla con el pipeline moderno— sin duplicar lógica ni reintroducir la línea DeBERTa.

---

## 2. Diagnóstico principal

### 2.1 El núcleo CBR y los benchmarks se solapan

Existe un recuperador genérico en:

- `src/triager_omega/retrieval/retriever.py`

Pero distintos benchmarks vuelven a implementar gran parte del flujo:

- `scripts/benchmarks/openj9/cbr_retrieval_openj9.py`
- `scripts/benchmarks/mozilla/cbr_retrieval_pilot.py`
- `scripts/benchmarks/typescript/eval_ts_sweep.py`
- `scripts/benchmarks/openj9/eval_openj9_full.py`

En diversas combinaciones, todos:

1. cargan train/test;
2. construyen el `label_encoder`;
3. embeben train/test;
4. calculan coseno con `emb_test @ emb_train.T`;
5. aplican BM25/RRF opcional;
6. votan por el resolvedor;
7. calculan métricas;
8. ejecutan barridos sobre `k`, `tau` y/o estrategia de voto.

#### Duplicaciones concretas

- Las funciones `label_encoder(...)` locales de OpenJ9 y TypeScript solo envuelven `build_label_encoder(...)`.
- El encoding con SentenceTransformers y `batch_size=64` aparece repetido en varios scripts.
- Los barridos y el formateo de resultados se implementan distinto por corpus.
- OpenJ9 consume directamente utilidades de `retrieval`; Mozilla conserva un flujo particular; TypeScript es casi autocontenido.

#### Dirección propuesta

Usar una única orquestación activa:

```text
dataset adapter
    ↓
CbrRetriever.fit(...)
    ↓
CbrRetriever.score(...)
    ↓
evaluate / sweep
```

Los scripts deben ser adaptadores del corpus y de la corrida:

- cargar los datos;
- elegir la vista textual;
- construir una configuración explícita;
- llamar al núcleo;
- persistir y mostrar los resultados.

No deberían volver a construir embeddings, label encoders, votos o métricas.

---

### 2.2 `InteractionRecommender` mezcla un motor genérico con el piloto Mozilla

`src/triager_omega/modules/ibr.py` hoy mezcla dos tipos de responsabilidad.

#### Lógica genérica valiosa

- recuperación de vecinos;
- scoring `similitud · Interaction Point · decay`;
- filtro temporal anti-fuga;
- normalización NIS;
- ranking por desarrollador.

#### Lógica acoplada al piloto Mozilla

- `load_pilot_text`;
- rutas de `pilot/splits.parquet`;
- columnas `Bug Id`, `contributor_id`, `creation_time`;
- unión Bugzilla/gecko-dev;
- persistencia en `pilot/ibr_embeddings.npy`.

Como resultado, OpenJ9 utiliza el objeto inyectándole estado privado:

```python
ibr._active = ...
ibr._train_bug_ids = ...
ibr._train_embeddings = ...
ibr._interactions = ...
```

Esto ocurre en:

- `scripts/benchmarks/openj9/eval_openj9_ibr.py`;
- `scripts/benchmarks/openj9/eval_openj9_full.py`.

Además, `_interaction_table(...)` está duplicada en ambos scripts.

#### Dirección propuesta

Separar motor y adaptadores:

```text
interactions/
├── scorer.py       # IP, decay, normalización NIS
├── index.py        # índice de casos + embeddings + interacciones
└── datasets/
    ├── mozilla.py  # Bugzilla + gecko-dev → tabla canónica
    └── openj9.py   # GitHub timelines → tabla canónica
```

Una API conceptual sería:

```python
index = InteractionIndex.fit(
    case_ids=train_ids,
    texts=train_texts,
    interactions=interaction_table,
    candidates=active_developers,
    config=ibr_config,
)

scores = index.score(query_texts, query_times)
```

Así ningún benchmark necesita tocar atributos privados.

---

### 2.3 El agregador conserva una interfaz no activa

`src/triager_omega/modules/aggregator.py` tiene tres partes:

1. `normalize_rows()` — útil y genérica.
2. `rank_metrics()` — útil y genérica.
3. `TriagerAggregator` + CLI — acoplados al piloto y al CBR clasificador histórico.

El método `_build()` lanza deliberadamente `NotImplementedError`, por lo que la CLI del módulo no es utilizable. La clase además depende de `self.cbr.text_mode` y de artefactos de un flujo DeBERTa ya descartado.

#### Dirección propuesta

- Conservar `normalize_rows` y `rank_metrics` en una capa de evaluación/fusión.
- Extraer una función pura, por ejemplo `linear_fuse(nps, nis, weight, gate=None)`.
- Mantener RRF de fusión solo si sigue como experimento activo.
- Mover `TriagerAggregator` a histórico o sustituirlo por una API agnóstica al tipo de CBR.
- No exponer una CLI activa que termina en error.

---

### 2.4 Inconsistencia real de rutas Mozilla

La configuración define:

```text
artifacts/mozilla/repo_interactions.parquet
artifacts/mozilla/discussion_interactions.parquet
```

Sin embargo, dos generadores escriben en la raíz de `artifacts/`:

| Código | Ruta escrita actualmente |
|---|---|
| `src/triager_omega/data/repo_miner.py` | `artifacts/repo_interactions.parquet` |
| `scripts/benchmarks/mozilla/build_discussion_interactions.py` | `artifacts/discussion_interactions.parquet` |
| `src/triager_omega/modules/ibr.py` | lee `artifacts/mozilla/...` |

También `scripts/benchmarks/mozilla/inspect_bug.py` lee rutas raíz en lugar de las rutas configuradas.

Esto es prioritario: una regeneración puede producir archivos que el IBR no consume, mientras el pipeline utiliza artefactos anteriores dentro de `artifacts/mozilla/`.

#### Regla propuesta

> Ningún script debe construir manualmente una ruta de artefacto de corpus. Debe usar una ruta canónica proporcionada por configuración o por un objeto `CorpusPaths`.

---

### 2.5 `config.py` concentra demasiadas responsabilidades

`src/triager_omega/config.py` reúne a la vez:

- rutas globales;
- rutas de Mozilla y OpenJ9;
- parámetros IBR específicos de Mozilla;
- backend y credenciales de destilación;
- opciones del piloto;
- ubicación local de TriagerX;
- detección de dispositivo;
- valores de experimentos históricos.

Además, `openj9_dir` está definido dos veces. Algunos scripts mutan el singleton global en tiempo de ejecución, por ejemplo:

```python
settings.distill_backend = args.backend
```

#### Dirección propuesta

Separar conceptos, sin requerir necesariamente una nueva dependencia:

```text
Settings de entorno
├── project_root
├── data_dir
├── artifacts_dir
├── dispositivo
└── credenciales / URL de API

CorpusSpec
├── mozilla
├── openj9
└── typescript

ExperimentConfig
├── retrieval: modelo, k, tau, vote, RRF
├── interaction: IPs, lambda, top-k, canal
├── fusion: weight, gate, modo
└── input: raw / distilled / both
```

`dataclass` inmutables y parsers compartidos bastan inicialmente.

---

### 2.6 Los argumentos CLI están repetidos y son ambiguos

Actualmente hay parámetros homónimos con significados distintos:

| Concepto | Variantes actuales |
|---|---|
| vecinos CBR | `--top-k`, `--cbr-topk` |
| vecinos IBR | `--top-k` |
| parámetro CBR | `--tau`, `--cbr-tau` |
| umbral IBR | `--tau` |
| longitud CBR | `--max-seq-len`, `--cbr-maxlen` |
| clasificador histórico | `--max-length` |
| IPs IBR | `--ip-c`, `--ip-a`, `--ip-d` |
| dispositivo | `--cpu`, autodetección o MPS hardcodeado |

El caso más cargado es `scripts/benchmarks/openj9/eval_openj9_full.py`: mezcla CBR histórico, CBR recuperación, IBR, BM25 para IBR, RRF, gating, hardware y rutas de dataset.

#### Dirección propuesta

Crear grupos reutilizables de argumentos:

```python
add_dataset_args(parser)
add_retrieval_args(parser)
add_interaction_args(parser)
add_fusion_args(parser)
add_runtime_args(parser)
```

Cuando CBR e IBR coexistan, usar nombres explícitos:

```text
--cbr-k
--cbr-tau
--ibr-k
--ibr-threshold
--ibr-decay
```

Tras consolidar las APIs internas se puede ofrecer una CLI con subcomandos:

```bash
triager benchmark openj9 cbr ...
triager benchmark openj9 full ...
triager benchmark mozilla cbr ...
triager distill mozilla ...
```

No es conveniente comenzar por la CLI: primero se consolidan las APIs.

---

### 2.7 La destilación DeepSeek está consolidada solo parcialmente

El archivo `scripts/distill/deepseek_engine.ts` centraliza correctamente:

- tipos de destilación;
- validación;
- serialización `[FL] [SY] [CP]`;
- extracción de JSON;
- cliente HTTP con reintento;
- uso/costo;
- circuit breaker.

Pero `distill_deepseek_ts.ts` y `distill_deepseek_openj9.ts` aún duplican mucho código:

- parsing de CLI;
- `CheckpointRecord`;
- carga de checkpoints;
- `flushOutputs`;
- CSV/Parquet/provenance;
- workers;
- circuit breaker;
- contabilidad;
- llamada HTTP;
- fallback;
- reintentos.

Los runners importan tipos y validadores del engine, pero todavía no usan plenamente `callDeepSeekWithRetry()` ni `CircuitBreaker()` como fuente única de comportamiento.

#### Dirección propuesta

Antes de crear el runner moderno de Mozilla, completar la consolidación:

```text
deepseek_engine.ts
├── cliente HTTP / retry / usage
├── pool concurrente
├── checkpoint idempotente
├── serializers CSV / Parquet / provenance
└── runner genérico

corpora
├── typescript.ts  # prompt + mapeo de columnas + fallback
├── openj9.ts      # prompt + mapeo de columnas + fallback
└── mozilla.ts     # próximo prompt especializado
```

La intención es evitar copiar por tercera vez el mismo pipeline para Mozilla.

---

### 2.8 Elementos que ya son históricos deben verse como históricos

DeBERTa no es parte de la línea activa. Algunos elementos aún generan confusión:

- `scripts/legacy/train_deberta_supervised.py` y predictor: correctamente archivados.
- `scripts/legacy/train_openj9_cbr.py`: correctamente archivado.
- `scripts/benchmarks/mozilla/balance_experiment.py`: usa TF-IDF + clasificador lineal como surrogate y todavía habla de reemplazarlo por DeBERTa; ya no encaja con la dirección actual.
- `scripts/benchmarks/openj9/eval_openj9_full.py --cbr-mode classifier`: conserva una rama de ejecución histórica.
- `TriagerAggregator`: conserva supuestos del clasificador/piloto anterior.

No se propone borrar trazabilidad, sino separarla visual y operativamente:

```text
scripts/
├── benchmarks/        # pipeline vigente de recuperación
├── distill/           # pipeline vigente de destilación
├── tools/
└── legacy/            # clasificación, réplicas históricas y surrogates
```

`balance_experiment.py` debería archivarse o rediseñarse específicamente para la arquitectura de recuperación.

---

## 3. Estructura objetivo orientativa

Sin hacer una reescritura total, una estructura gradual podría converger hacia:

```text
src/triager_omega/
├── config/
│   ├── settings.py            # entorno, paths, credenciales
│   ├── corpora.py             # rutas y contratos por corpus
│   └── experiments.py         # dataclasses de CBR/IBR/fusión
├── data/
│   ├── common.py              # carga, identidad, splits
│   ├── mozilla.py             # adaptador Bugzilla
│   └── openj9.py              # adaptador GitHub/OpenJ9
├── retrieval/
│   ├── retriever.py           # CbrRetriever: API única
│   ├── bm25.py
│   ├── fusion.py
│   ├── voting.py
│   └── evaluation.py          # métricas y barridos
├── interactions/
│   ├── index.py               # índice genérico de casos/interacciones
│   ├── scoring.py             # IP, decay, NIS
│   └── fusion.py              # NPS + Wf·NIS + gating
├── distillation/
│   └── ...
└── cli/
    └── ...
```

Principio importante:

> `src/` debe contener lógica reutilizable y no asumir un piloto o un corpus; `scripts/` deben cargar/adaptar datos y ejecutar experimentos.

---

## 4. Plan recomendado de refactorización

### Fase 1 — Corregir fronteras y riesgos sin cambiar resultados

1. Unificar rutas Mozilla de `repo_interactions` y `discussion_interactions`.
2. Eliminar la propiedad `openj9_dir` duplicada.
3. Dejar de exponer la CLI rota de `aggregator.py`.
4. Mover o marcar como histórico el experimento de balanceo basado en clasificación.
5. Eliminar imports muertos; por ejemplo, `rerank_cross_encoder` se importa en `retriever.py` pero no se usa desde esa clase.
6. Normalizar comentarios/docstrings que aún mencionan `artifacts/pilot/` en la raíz en vez de `artifacts/mozilla/pilot/`.

### Fase 2 — Consolidar CBR de recuperación

1. Hacer que OpenJ9, TypeScript y Mozilla consuman `CbrRetriever`.
2. Extraer helpers de carga de datasets/tablas canónicas.
3. Unificar el contrato de los casos:

   ```text
   case_id | text | owner | created_at | split
   ```

4. Centralizar barridos y persistencia de resultados estructurados.

### Fase 3 — Desacoplar IBR del piloto Mozilla

1. Convertir `InteractionRecommender` en un motor independiente de corpus.
2. Extraer adaptadores Mozilla/OpenJ9.
3. Eliminar la inyección de atributos privados desde benchmarks.
4. Compartir la construcción de tabla de interacciones y evaluación IBR.

### Fase 4 — Consolidar destilación antes de Mozilla moderno

1. Hacer que `deepseek_engine.ts` sea la implementación común efectiva.
2. Reducir los runners por corpus a prompt + mapeo de columnas + fallback.
3. Definir un único formato canónico de checkpoint, parquet y provenance.
4. Crear Mozilla especializado sobre ese runner, no como un tercer pipeline copiado.

### Fase 5 — Añadir tests antes de mover más lógica

No hay suite automatizada visible. Antes de cambios grandes, fijar pruebas puras para:

- `vote_scores`;
- `rrf_fuse`;
- `rank_metrics`;
- normalización NIS;
- anti-fuga temporal;
- gating;
- construcción/validación de destilados;
- resolución de rutas por corpus.

Esto permite reorganizar sin modificar las métricas experimentales que sostienen la tesis.

---

## 5. Orden práctico sugerido

El inicio de menor riesgo y mayor impacto es:

1. **Fase 1** completa;
2. definir el contrato canónico de dataset/configuración;
3. consolidar CBR;
4. desacoplar IBR;
5. consolidar el runner DeepSeek y, solo entonces, crear Mozilla moderno.

No se recomienda comenzar con una CLI grande ni con una reescritura total. Primero se corrigen rutas, se separa lo histórico y se fija una API interna única para recuperación e interacción.
