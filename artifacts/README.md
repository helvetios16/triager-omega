# Directorio de Artefactos — Triager-Omega

Estructura de artefactos experimentales generados, particionados por corpus de evaluación.

---

## 1. `artifacts/mozilla/` — Corpus Mozilla Bugzilla
- `splits.parquet`: División temporal (train, val, test) para el directorio de 450 desarrolladores activos.
- `active_candidates.json`: Lista curada de los 450 desarrolladores activos de Mozilla.
- `label_encoder.json`: Mapeo de identidad (Contributor Id ↔ class_idx).
- `sample_weights_train.npy`: Ponderaciones para mitigar el desbalance de clases.
- `repo_interactions.parquet`: Interacciones de commits y revisiones minadas de `gecko-dev.git`.
- `discussion_interactions.parquet`: Interacciones de comentarios depuradas (sin bots, deduplicadas).
- `balance_experiment/`: Resultados de ablación sobre estrategias de cola larga.
- `pilot/`:
  - `splits.parquet`: Subconjunto de evaluación para los 20 desarrolladores más activos.
  - `label_encoder.json`: Mapeo para los 20 desarrolladores del piloto.
  - `distillations.parquet`: Destilaciones semánticas estructuradas con formato `[FL][SY][CP]`.
  - `ibr_embeddings.npy` y `ibr_bug_ids.npy`: Matriz de representaciones vectoriales densas para el índice IBR.

---

## 2. `artifacts/openj9/` — Benchmark IBM / Eclipse OpenJ9
- `openj9_{train,val,test,traininner}_50.csv`: Splits temporales para los 50 desarrolladores con mayor volumen de resolución.
- `openj9_interactions.parquet`: Matriz de eventos (issue, dev, tipo, timestamp) minada de la API de GitHub.
- `openj9_issue_meta.parquet`: Marcas de tiempo de creación para evitar fuga de información temporal.
- `distilled/`: Salida de destilación semántica generada con DeepSeek Flash.
- `distilled_full/`: Corpus completo destilado.
- `raw/`: Caché local de timelines JSON minados desde GitHub.
- `legacy_pilots/`: Experimentos y destilaciones preliminares anteriores.

---

## 3. `artifacts/typescript/` — Benchmark Microsoft TypeScript
- `distilled/`: Checkpoints y archivos consolidados de destilación semántica (`ts_train_union.jsonl`, `ts_train.jsonl`, `ts_test.jsonl`).
- `embeddings/`: Embeddings precalculados con `all-mpnet-base-v2` (`ts_train_raw.npy`, `ts_test_raw.npy`, `ts_train_distilled.npy`, `ts_test_distilled.npy`).
- `ts_cbr_evaluation_results.json`: Métricas registradas del barrido exhaustivo de CBR denso e híbrido.

---

## 4. Limpieza y Purga de Checkpoints Históricos
- Los pesos entrenados (`model.safetensors`, 1.44 GB) del clasificador supervisado DeBERTa-v3-base fueron depurados de disco, al haber sido formalmente reemplazados por el CBR de recuperación zero-shot e híbrido MPNet+BM25.
