# Entrenar el CBR (DeBERTa) en la laptop con GPU (RTX 5060)

El entrenamiento de DeBERTa **no es una API remota** como Ollama: es un trabajo de
cómputo que necesita el **código + los datos** en la máquina donde corre, y el modelo
se genera ahí. Como la laptop tiene **GPU CUDA**, conviene correrlo en ella (más rápido
y sin el bug del `nan` de MPS de la Mac).

> `data/` y `artifacts/` están **gitignored** → git lleva solo el código. Los datos y el
> modelo entrenado se mueven aparte (ver §"Copiar archivos").

---

## Opción 1 (recomendada): TODO en la laptop

La laptop ya tiene **Ollama + CUDA**, así que haz la fase CBR completa ahí. La Mac queda
solo para desarrollo. Una sola transferencia al final (el modelo), en una dirección.

### 1. Código (por git)
```bash
# en la laptop:
git clone <tu-repo> triager-omega
cd triager-omega
uv sync
```

### 2. Datos de entrada (copiar a la laptop, no van en git)
- `data/bug_metadata.parquet`   (~21 MB, para el texto crudo)
- `artifacts/pilot/`            (splits.parquet + label_encoder.json)

Ver §"Copiar archivos" para el cómo.

### 3. Destilar y entrenar en la laptop (Ollama local, sin red)
```bash
# Ollama corriendo local en la laptop:
uv run python scripts/run_distillation.py                  # crea artifacts/pilot/distillations.parquet
uv run python -m triager_omega.cbr.train --adam-eps 1e-8   # CUDA: 1e-8 normal (el 1e-4 era parche de MPS)
```

### 4. El modelo sale en la laptop
`artifacts/pilot/cbr_model/` (pesos + tokenizer + `metrics.json`). Lo usas ahí, o lo
copias a la Mac si lo necesitas (ver §"Copiar archivos").

---

## Opción 2 (híbrida): destilar desde la Mac, entrenar en la laptop

Útil si prefieres orquestar la destilación desde la Mac (Ollama remoto, ver
`docs/ollama-remoto.md`).

1. La Mac destila → `artifacts/pilot/distillations.parquet` queda en la **Mac**.
2. Copiar a la laptop: `data/bug_metadata.parquet` + `artifacts/pilot/` (ya con el
   `distillations.parquet`).
3. Entrenar en la laptop:
   ```bash
   uv run python -m triager_omega.cbr.train --adam-eps 1e-8
   ```
4. Copiar `artifacts/pilot/cbr_model/` de vuelta a la Mac.

---

## Copiar archivos entre máquinas (no-código)

Ambas en la misma red. Ejemplos con la IP de la laptop `192.168.1.50` y usuario `user`:

```bash
# Mac → laptop (subir los datos de entrada):
scp -r artifacts/pilot user@192.168.1.50:~/triager-omega/artifacts/
scp data/bug_metadata.parquet user@192.168.1.50:~/triager-omega/data/

# laptop → Mac (bajar el modelo entrenado, ~700 MB-1.5 GB):
scp -r user@192.168.1.50:~/triager-omega/artifacts/pilot/cbr_model artifacts/pilot/
```

Alternativas a `scp`: `rsync -av` (mejor para reanudar), una carpeta compartida en red,
o un USB.

---

## Notas

- **`--adam-eps 1e-8` en la laptop.** El `1e-4` del default es el parche para el `nan` de
  DeBERTa-v3 en **MPS** (Apple Silicon). En CUDA usa el `1e-8` estándar.
- **VRAM 8 GB.** `deberta-v3-base` (~184M params) entra de sobra para entrenar con
  `--batch-size 16 --max-length 256`. Si te quedas corto de VRAM, baja el batch o la
  longitud (`--batch-size 8`).
- **El modelo se regenera.** No hace falta versionar `cbr_model/`: con la
  `distillations.parquet` puedes re-entrenar las veces que quieras. Lo único caro de
  reproducir es la destilación (las llamadas al LLM), por eso ese parquet sí conviene
  conservarlo/copiarlo.
- **Flujo de archivos** (recordatorio): `distillations.parquet` ─► `cbr/train.py` ─►
  `cbr_model/`. Ver el PLAN §13.1.
