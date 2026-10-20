# Registro de Constancia y Verificación de Calidad — Destilación TypeScript

Documento de registro técnico sobre la corrección de inconsistencias en el split de entrenamiento `ts_train`, la integración formal del **Dataset de Registro de Constancia (*Provenance Dataset*)**, y la suite de comandos de verificación y sincronización.

---

## 1. Contexto y Diagnóstico del Problema

Al ejecutar el verificador de calidad inicial (`bun run scripts/verify_distillations.ts`), se detectaron dos errores críticos que detenían el pipeline de validación:

```text
❌ ERRORES DETECTADOS (2):
  [Issue #3136] symptoms: symptoms vacío o no es array
  [Issue #N/A] row_count: Desalineación en el conteo de filas entre JSONL y CSVs
```

### Causa raíz de cada problema:
1. **Issue #3136 (symptoms vacío)**:
   - El reporte correspondía a un feature request (*"Feature Request: F# style Type Provider support?"*).
   - El modelo de lenguaje generó `symptoms: []`, dejando vacío el bloque `[SY]` en `distilled_text`. La regla de validación de calidad de `triager-omega` exige que todo registro no-fallback contenga al menos un síntoma u observación técnica.
2. **Desalineación de conteo de filas (732 vs 731)**:
   - `ts_train.jsonl` tenía 732 líneas (había registrado el issue `#3298`), pero los archivos derivados (`_distilled.csv`, `_raw.csv`, `_both.csv` y `.parquet`) se quedaron en 731 porque el script de destilación no había completado el ciclo de flush para ese último registro.
3. **Omisión del Registro de Constancia (*Provenance Dataset*)**:
   - Existían archivos de auditoría previos (`ts_train_provenance.csv` y `ts_train_provenance.parquet`) estancados en 721 registros.
   - El verificador de calidad no comprobaba la presencia, integridad ni coincidencia de este dataset frente a la base de datos de entrenamiento.

---

## 2. Soluciones Implementadas

### A. Corrección del Issue #3136
- Se actualizaron los síntomas y capacidades en `artifacts/typescript/distilled/ts_train.jsonl`:
  - **Symptoms**: `["type providers unsupported", "no external schema type synthesis", "manual API typing required"]`
  - **Capabilities**: `["external type generation", "compile-time type synthesis", "language service extension design"]`
- Se reconstruyó `distilled_text` con la taxonomía formal completa `[FL] ... [SY] ... [CP] ...`.

### B. Generación y Estandarización del Registro de Constancia
El **Registro de Constancia** audita la procedencia técnica de cada destilación para la tesis. Contiene 11 columnas estructuradas:
- Identificación: `issue_number`, `issue_title`, `owner`
- Auditoría técnica: `model`, `latency_ms`, `fallback`
- Taxonomía del compilador: `component`, `subsystem`, `symptoms`, `capabilities`, `distilled_text`

Se actualizaron los scripts para garantizar paridad total en 732 registros:
- **`scripts/merge_distillations.ts`**:
  - Lee el checkpoint `.jsonl` y mapea los títulos desde `df_train_ts.csv`.
  - Genera automáticamente `ts_train_provenance.csv` y `ts_train_provenance.parquet` con las 11 columnas alineadas fila por fila.
  - Previene que al compilar `ts_train` se mezclen accidentalmente archivos de otros experimentos como `ts_train_union.jsonl`.
- **`scripts/distill_opencode_ts.ts`**:
  - Ahora su función `flushOutputs` escribe periódicamente tanto los CSVs habituales como el dataset de constancia (`_provenance.csv` y `_provenance.parquet`).

### C. Actualización del Verificador de Calidad (`scripts/verify_distillations.ts`)
- Se incluyeron como archivos obligatorios `${SPLIT}_provenance.csv` y `${SPLIT}_provenance.parquet`.
- Se añadieron verificaciones automáticas de conteo de filas en Parquet y CSVs mediante `uv run python`.
- Se validó la alineación fila por fila entre JSONL y el registro de constancia (coincidencia de `issue_number`, `owner`, latencia válida $\ge 0$ y modelo no nulo).
- Se agregó al reporte final el desglose por modelo y latencia promedio por inferencia:
  - `opencode/union-alpha`: 657 registros (89.8%) · 19.2s prom
  - `omniroute/deepseek-web/deepseek-flash`: 75 registros (10.2%) · 32.5s prom

---

## 3. Estado Final de los Datasets

### A. Split Principal (`ts_train` — 840 registros)
Todos los archivos en `artifacts/typescript/distilled/` cuentan con **840 registros alineados al 100%**:
- **Modelos**: 657 `opencode/union-alpha` (78.2%) + 183 `deepseek-flash` (21.8%)
- **Fallbacks**: 196 (23.3%)
- **Errores críticos**: 0

| Archivo | Filas | Columnas | Descripción |
|---|---|---|---|
| `ts_train.jsonl` | 840 | - | Checkpoint maestro en formato JSON Lines |
| `ts_train.parquet` | 840 | 8 | Dataset estructurado para entrenamiento CBR / Transformers |
| `ts_train_distilled.csv` | 840 | 3 | Formato estándar TriagerX (solo texto destilado `[FL][SY][CP]`) |
| `ts_train_raw.csv` | 840 | 3 | Formato estándar TriagerX (texto original del issue) |
| `ts_train_both.csv` | 840 | 3 | Formato enriquecido (texto original + destilación) |
| `ts_train_provenance.csv` | 840 | 11 | Registro de constancia y trazabilidad por modelo |
| `ts_train_provenance.parquet` | 840 | 11 | Registro de constancia binario de alta velocidad |

### B. Split Extendido (`ts_train_union` — 1,514 registros)
Destilación masiva procesada al 100% mediante el LLM `opencode/union-alpha`:
- **Modelo**: `opencode/union-alpha`: 1,514 registros (**100.0%**) · Latencia promedio: 25.7s
- **Tasa de Fallback**: Solo 25 registros (**1.7%**) — excelente retención de razonamiento estructurado
- **Errores críticos**: **0** (Validación exitosa al 100% de la taxonomía del compilador)

| Archivo | Filas | Columnas | Descripción |
|---|---|---|---|
| `ts_train_union.jsonl` | 1514 | - | Checkpoint maestro deduplicado y ordenado |
| `ts_train_union.parquet` | 1514 | 8 | Dataset de alta velocidad listo para entrenamiento CBR |
| `ts_train_union_distilled.csv` | 1514 | 3 | Texto destilado formal `[FL][SY][CP]` para TriagerX |
| `ts_train_union_raw.csv` | 1514 | 3 | Texto crudo de los issues de TypeScript |
| `ts_train_union_both.csv` | 1514 | 3 | Fusión de texto crudo + destilación enriquecida |
| `ts_train_union_provenance.csv` | 1514 | 11 | Auditoría completa con modelo, latencia y taxonomía |
| `ts_train_union_provenance.parquet` | 1514 | 11 | Registro de constancia en formato binario Parquet |


---

## 4. Comandos de Operación y Verificación

### 1. Verificar la calidad y alineación (incluyendo Registro de Constancia)
```bash
# Verificación del split principal ts_train
bun run scripts/verify_distillations.ts

# Verificación de un split específico (ej. ts_train_union)
bun run scripts/verify_distillations.ts --split ts_train_union
```

### 2. Consolidar / Reconstruir todos los archivos desde el Checkpoint
Si se añaden nuevos issues al `.jsonl` y se desea regenerar los CSVs, Parquets y el Registro de Constancia:
```bash
# Consolidar ts_train
bun run scripts/merge_distillations.ts --target ts_train --sources artifacts/typescript/distilled/ts_train.jsonl

# Consolidar ts_train_union
bun run scripts/merge_distillations.ts --target ts_train_union --sources artifacts/typescript/distilled/ts_train_union.jsonl
```

### 3. Continuar la destilación con OpenCode
Para continuar destilando con workers concurrentes y auto-flush sincronizado a Parquet y Constancia:
```bash
bun run scripts/distill_opencode_ts.ts \
  --split ts_train \
  --concurrency 3 \
  --flush-every 5
```

### 4. Inspeccionar la integridad de los Parquets con Python / uv
```bash
uv run python -c "
import pandas as pd
for name in ['ts_train.parquet', 'ts_train_provenance.parquet']:
    df = pd.read_parquet(f'artifacts/typescript/distilled/{name}')
    print(f'{name:30}: {len(df)} filas, {len(df.columns)} columnas')
"
```
