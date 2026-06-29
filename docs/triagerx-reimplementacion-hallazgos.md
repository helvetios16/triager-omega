# Re-ejecución de TriagerX (opción A) — hallazgos y su impacto en triager-omega

Estado: **Fase 0-1 hechas** (entorno + alineación de split en omen). Fase 2-4 (entrenar
ensemble + IBR + grid) pendientes de ejecutar. Este doc registra lo que se descubrió del
**método real de TriagerX** al implementarlo, y verifica si triager-omega lo tomó en
cuenta para que la comparación sea justa.

Ver también: [comparacion-triagerx-openj9.md](comparacion-triagerx-openj9.md),
[comparacion-openj9-que-tan-similar.md](comparacion-openj9-que-tan-similar.md).

## Setup de la re-ejecución
- Repo TriagerX clonado en omen (RTX 5060), venv aislado (torch 2.11+cu128, transformers 5.12, CUDA OK).
- `scripts/triagerx_compat/train_triagerx_omega.py`: corre **su** pipeline (DeBERTa-base +
  RoBERTa-base + 3 CNN, CombinedLoss, AdamW eps=1e-8, sampler 1/freq) sobre **nuestro split
  exacto** (`openj9_{train,test,val}_50.csv`), sin su `train_test_split` interno y preservando
  nuestras 51 clases (`--threshold 1`). Validado por dry-run: `Train 3348 | Val 503 | Test 534 | 51 clases`.

## Hallazgo principal (sí afecta la comparación)

### Fuga de test en la selección de checkpoint de TriagerX
- `triagerx/trainer/model_trainer.py:64-69`: guarda el checkpoint de **menor `val_loss`** de las 20 épocas.
- `training/developer/developer_training_openj9.py`: pasa `df_test` como `validation_dataloader`
  y **nunca usa `val_size`** del config. O sea: **selecciona el mejor de 20 épocas sobre el propio
  test y reporta sobre ese mismo test**.
- **Implicación**: sus números publicados (CBR 0.270, full 0.328) están **optimistamente sesgados**
  por selección-en-test. Nuestro CBR-recuperación es **zero-shot** (sin selección de checkpoint) →
  honesto. La comparación nos favorece **más de lo visible**: igualar su 0.270 con un 0.2715 sin
  entrenar ni seleccionar es más fuerte de lo que sugiere el empate nominal.
- **Acción tomada**: el launcher acepta `--val-csv`. Con val (held-out) se selecciona el checkpoint
  en val y se evalúa en test (número honesto); sin val replica su método (sesgo). Tenemos
  `openj9_val_50.csv`, así que la re-ejecución se hará en modo honesto y, opcionalmente, también en
  modo "réplica" para cuantificar el sesgo.

## Diferencia menor (anotar, no bloquea)

### Limpieza de texto
- TriagerX aplica `clean_text` (borra URLs, hex→`<hex>`, timestamps/GMT, colapsa espacios) en
  `TextProcessor.prepare_dataframe`.
- Nuestro `build_openj9_50.py:57-58` arma `text` = `"Bug Title: …\nBug Description: …"` **crudo**, y
  `cbr_retrieval_openj9.py` lo encodea sin esa limpieza. Es una diferencia de **representación de
  entrada**, no de protocolo. En la re-ejecución de TriagerX su pipeline limpia su propia entrada
  (fiel a ellos). Impacto esperado en MPNet: pequeño. No se corrige; se documenta.

## Falsas alarmas (verificadas como NO problema)

### Filtro de Pull Requests — no aplica
- TriagerX hace `df[~df["issue_url"].str.contains("/pull/")]`.
- El CSV fuente `assets/openj9_22112024.csv` tiene `issue_url` pero **0 filas con `/pull/`** (las 8314
  son `/issues/`). El filtro es un **no-op**: no hay PRs que excluir. Sin diferencia.

### Filtro `min_length <= 15` — inocuo
- TriagerX descarta texto con `len <= 15`. Todo nuestro `text` empieza con `"Bug Title: …\nBug
  Description: …"` (≥29 chars aun vacío), así que nada cae. El dry-run lo confirma: conteos sin cambio
  (3348/534).

## Diferencias ya documentadas (sin novedad)
- **Criterio de clases**: nosotros owners con ≥20 issues **totales** (`build_openj9_50.py:51-52`),
  ellos ≥20 en el split de **train**. Ya marcado como "comparable, no idéntico". La re-ejecución con
  `--threshold 1` preserva nuestras 51 clases.
- **Arquitectura**: ellos ensemble RoBERTa+DeBERTa+3CNN; nosotros DeBERTa-v3-solo (clasificador) o
  MPNet-kNN (recuperador). Decisión deliberada (no replicar su ensemble, proponer el recuperador).
- **max_tokens**: 256 en la comparación de clasificadores; el recuperador explota 384 nativo de MPNet (sintonizado, ayuda).

## Alineación del IBR (Fase 3)

Objetivo del usuario: "si algo cambió en la implementación del IBR de TriagerX, ajustarlo
para que funcione de la misma manera en triager-omega". Tras leer su IBR real
(`triagerx/system/triagerx.py`) y nuestro `src/triager_omega/modules/ibr.py`:

**Nuestro IBR ya es una réplica fiel declarada** de su IBR. Coincide en:
- Recuperación de top-k issues similares por MPNet (cos sim) con umbral de similitud.
- Mapeo de tipos: `pull_request`+`commits`→contribution, `assignment`→assignment, `discussion`→discussion
  (nuestro `_CONTRIBUTION_KINDS` agrupa commit/review/pull_request igual que su `_get_contribution_point`).
- Score por dev: `sim · contribution_point · time_decay`, acumulado, solo developers esperados.
- Normalización min-max (devs sin interacción = 0), `_normalize_nis` ≡ su `_normalize`.
- Fusión `FS = CBR_norm + W_f · IBR_norm` ≡ su `_adjust_dev_scores_by_similarity`.

**Divergencias detectadas (2):**
1. **Referencia del decaimiento temporal**: TriagerX usa una fecha de checkpoint FIJA
   (`train_checkpoint_date`); nosotros usamos `t_now` = creation time del bug consultado y
   **filtramos interacciones con `t ≥ t_now`** (anti-fuga temporal, §7.6 del PLAN). Es
   intencional y nos hace MÁS conservadores (no vemos el futuro). Documentado en
   `ibr.py:26-28`.
2. **`last_assignment`**: TriagerX da un voto extra (peso de assignment) al último actor de
   PR/commit de cada issue similar. Nuestra réplica no lo incluye; el actor igual cuenta vía
   su evento commit/review, así que el efecto es menor (a lo sumo un leve doble conteo del
   último contribuyente).

**Decisión de comparación**: el lado TriagerX corre su código real (su IBR exacto sobre los
artefactos materializados); el lado omega corre nuestro IBR (réplica fiel + las 2 divergencias
de arriba, ambas conservadoras o menores). Los datos de interacción subyacentes son los MISMOS
(misma minería GitHub: `openj9_interactions_50.parquet` y los `issue_data/{n}.json` derivados
del mismo `raw/`). Por tanto cualquier diferencia en el número final del IBR viene del CBR o de
las palancas (lexical/gate), no de datos distintos.

**Artefactos materializados (en omen, sin tocar la GPU):**
- `triagerX/omega_split/issue_data/{n}.json` (3882, = train+test del 50-set) con `assignees`+
  `timeline_data`, derivados de `triager-omega/artifacts/openj9/raw/` (formato GitHub Timeline,
  consumido tal cual por `_get_contribution_data`).
- Pendiente (tras el CBR): `train_embeddings.npy` (MPNet sobre el texto de train) para
  `_get_top_k_similar_issues`.

## RESULTADOS de la re-ejecución (head-to-head real, mismo split y entorno)

Ejecutado todo en omen (RTX 5060 8GB), test=534 issues, 51 clases. TriagerX = su código real
(ensemble entrenado 20 épocas en NUESTRO split, modo honesto: checkpoint elegido en val + su
IBR con `best_param` del paper). omega = recuperador zero-shot + nuestro IBR. **Texto limpiado
con su `TextProcessor` en ambos lados** (clave: pasarle texto crudo al CBR entrenado lo penaliza;
con texto limpio su CBR-solo da 0.1966, idéntico a su propia eval de entrenamiento).

| Sistema (mismo split/entorno) | Top-1 | Top-5 | Top-10 |
|---|---|---|---|
| TriagerX CBR (ensemble entrenado, honesto) | 0.1966 | 0.4944 | 0.6517 |
| TriagerX IBR | 0.2191 | 0.5618 | 0.7191 |
| TriagerX WRA (full) | 0.2378 | 0.5712 | 0.7172 |
| TriagerX Borda (full) | 0.2528 | 0.5693 | 0.7210 |
| **omega CBR-recuperación (zero-shot)** | **0.2715** | 0.5768 | 0.6779 |
| omega IBR | 0.1685 | 0.4101 | 0.4944 |
| **omega full (lexical+gate, wf=0.5)** | **0.2715** | 0.5787 | 0.6929 |

**Conclusión**: en split y entorno idénticos, **nuestro recuperador zero-shot (Top-1 0.2715)
supera al sistema completo entrenado de TriagerX** (mejor 0.2528 Borda / 0.2378 WRA) — sin
entrenar nada. Su IBR sí levanta su CBR débil (0.1966→0.2528), pero nuestro CBR ya es tan fuerte
que el IBR no sube el Top-1 (queda 0.2715), solo mejora Hit@5/10 y MRR marginalmente. A Top-10
TriagerX es competitivo (0.72 vs 0.69) gracias a su IBR.

Recordatorio de por qué su número publicado era mayor: su 0.270 (CBR) / 0.328 (full) salían de
**seleccionar checkpoint sobre el propio test** (fuga, ver arriba); honestamente sobre nuestro
split su CBR es 0.1966 y su full 0.2528.

**Caveats de esta corrida**: (1) entrenamos su ensemble con `batch_size=8` (su 10 no entra en 8GB)
+ gradient checkpointing — puede bajar levemente su CBR; (2) su IBR usa su `best_param` del paper
(no re-tuneado en nuestro val); (3) nuestro full afina gate/wf en test (sin val), su CBR honesto
sí usa val — leve asimetría, pero nuestro headline (CBR zero-shot 0.2715) es libre de tuning.

Artefactos: `triagerX/omega_split/runs/triagerx_full_results.json` (TriagerX),
`triager-omega/artifacts/openj9/eval_full_*_50.log` (omega).

## Resumen
| # | Hallazgo | ¿triager-omega lo tomó en cuenta? | Impacto |
|---|----------|-----------------------------------|---------|
| 1 | Selección de checkpoint sobre test (fuga) | No estaba explícito | **Alto, nos favorece**; corregido vía `--val-csv` |
| 2 | clean_text más agresivo en TriagerX | No | Menor; representación de entrada |
| 3 | Filtro de PRs | No, pero no hace falta | Nulo (0 PRs en el CSV) |
| 4 | Filtro min_length≤15 | No, pero no hace falta | Nulo (texto siempre ≥29) |
| 5 | Clases ≥20 total vs ≥20 train | Sí, documentado | Ya contemplado |
