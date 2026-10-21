# Catálogo de Scripts — Triager-Omega

Organización funcional de los scripts del proyecto por propósito y dataset.

---

## 1. `scripts/distill/` — Destilación Semántica LLM

Generación, validación y consolidación de la taxonomía diagnóstica `[FL][SY][CP]` sobre los reportes de error:

- **`deepseek_engine.ts`**: Motor central compartido de destilación (cliente HTTP con retry, parser JSON y validación taxonómica).
- **`distill_deepseek_ts.ts`**: Runner especializado para TypeScript con prompts de compilador.
- **`distill_deepseek_openj9.ts`**: Runner especializado para OpenJ9 con prompts de arquitectura JVM.
- **`distill_opencode_ts.ts`**: Pipeline de destilación alternativo con modelos OpenCode.
- **`distill_openj9_full.py`**: Batch runner en Python para destilación OpenJ9.
- **`run_distillation.py`**: Generador de destilación para Mozilla Bugzilla (Ollama / LM Studio / Google).
- **`merge_distillations.ts`**: Unifica checkpoints y salidas parciales de destilación en parquet/csv.
- **`reprocess_fallbacks.ts`**: Reprocesa casos que cayeron en fallback durante la destilación.
- **`verify_distillations.ts`**: Chequea la integridad, formato de etiquetas y completitud del corpus destilado.
- **`inspect_distillations.py`**: Visor rápido en consola de muestras destiladas y fallbacks.
- **`compare_prompts.ts`**: Comparativa cualitativa head-to-head entre diferentes prompts del sistema.

---

## 2. `scripts/benchmarks/` — Evaluación Experimental por Corpus

### `openj9/` (IBM / Eclipse OpenJ9)
- **`build_openj9_50.py`**: Filtra los 50 desarrolladores más activos y genera splits train/test.
- **`cbr_retrieval_openj9.py`**: Recuperador CBR zero-shot (MPNet densa, BM25, RRF y voting idf/sum/rank).
- **`eval_openj9_full.py`**: Evaluación integral del sistema (CBR + IBR con agregación adaptativa y gates).
- **`eval_openj9_ibr.py`**: Evaluación aislada del módulo IBR sobre las líneas de tiempo minadas.
- **`mine_openj9_timelines.py`**: Minería de eventos en GitHub (commits, reviews, issues) para OpenJ9.

### `typescript/` (Microsoft TypeScript)
- **`eval_ts_sweep.py`**: Barrido de hiperparámetros ($k \in \{5..50\}$, $\tau$, normalización de voto) sobre embeddings densos e híbridos.

### `mozilla/` (Mozilla Bugzilla)
- **`cbr_retrieval_pilot.py`**: Evaluación del CBR zero-shot sobre el piloto de 20 desarrolladores activos.
- **`build_discussion_interactions.py`**: Procesa comentarios de bugs para construir la matriz de interacción social.
- **`balance_experiment.py`**: Experimento histórico/surrogate de cola larga con TF-IDF y clasificación lineal; no forma parte del CBR de recuperación activo.
- **`inspect_bug.py`**: Auditoría de la relación entre un bug y sus 3 fuentes de interacción.
- **`eda.py`**: Estadísticas descriptivas del corpus curado de Mozilla.

---

## 3. `scripts/tools/` — Utilidades de Conectividad y Diagnóstico

- **`test_api.py`**: Verificación de credenciales y conectividad con APIs externas (Google GenAI).
- **`test_ollama.py`**: Verificación de conectividad con servidores Ollama locales o remotos.

---

## 4. `scripts/legacy/` — Archivo Histórico

Código de aproximaciones descartadas preservado por trazabilidad:

- **`train_deberta_supervised.py`** y **`predict_deberta_supervised.py`**: Pipeline de entrenamiento supervisado e inferencia de DeBERTa-v3-base (sustituido por CBR de recuperación).
- **`train_openj9_cbr.py`**: Script de fine-tuning sobre OpenJ9.
- **`triagerx_compat/`**: Implementación original de referencia del paper TriagerX para replicación empírica.
