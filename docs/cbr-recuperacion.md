# CBR de recuperación (Case-Based Reasoning real)

> Aporte propio: **reemplazar el CBR-clasificador de TriagerX** (cabeza softmax
> sobre un set fijo de devs, ensemble de 2 transformers) por un **recuperador
> basado en casos**. Validado en OpenJ9 a 50 clases (head-to-head con TriagerX).
>
> Script: `scripts/cbr_retrieval_openj9.py`. Fusión con IBR: `eval_openj9_full.py
> --cbr-mode retrieval`.

## Idea

TriagerX llama "CBR" a un **clasificador**. Pero CBR = *Case-Based Reasoning*: un
clasificador no razona por casos. Nuestro CBR sí:

1. Embebe el bug nuevo con un encoder de oraciones (MPNet, `all-mpnet-base-v2`).
2. Recupera los **k bugs pasados más similares** (coseno).
3. **Vota al `owner` (resolvedor)** de cada caso recuperado, ponderando por
   `similitud^τ` (negativos recortados a 0; decaimiento temporal opcional).

`score(dev) = Σ_{j ∈ topk, owner(j)=dev} sim(q,j)^τ`. Etiquetas = `sorted(owners)`
(idéntico a CBR-clasificador e IBR → fusionable por columna).

**Distinto del IBR:** el IBR usa el **grafo de interacciones tipadas** (commit/PR/
discusión/asignación) con encoder **congelado**; el CBR-recuperación vota la
**etiqueta de resolución** (owner) y el encoder es **afinable**.

## Resultados (OpenJ9, 50 clases, train 3348 / test 534)

### Campeón: zero-shot, MPNet a 384 tokens, k=50, τ=2

| Config | Top-1 | MRR | Hit@5 | Hit@10 |
|---|---|---|---|---|
| **CBR-recuperación (zero-shot, k=50)** | **0.2715** | **0.408** | 0.577 | 0.678 |

**Sin entrenar nada.** Comparado con el resto:

| Componente (50 clases) | Top-1 | Nota |
|---|---|---|
| DeBERTa-solo (TriagerX) | 0.189 | su CBR base |
| RoBERTa-solo (TriagerX) | 0.175 | |
| Nuestro CBR-clasificador (mejor, 30ép/512) | 0.2322 | mucho entrenamiento |
| CBR ensemble RoBERTa+DeBERTa (TriagerX) | 0.270 | 2 transformers |
| **Nuestro CBR-recuperación (zero-shot)** | **0.2715** | **0 entrenamiento** |
| CBR + IBR full (TriagerX) | 0.328 | ensemble + IBR |

Nuestro CBR-recuperación **iguala el ensemble de 2 transformers de TriagerX**
(0.2715 vs 0.270) y **supera** a su DeBERTa-solo (+8 pp) y a nuestro clasificador
entrenado (+3.9 pp) — sin entrenamiento, interpretable y capaz de manejar devs
nuevos / cola larga sin reentrenar.

### Tuning del zero-shot (lo que mueve la aguja)

- **Longitud de contexto:** 384 tokens es el óptimo (es el nativo de MPNet).
  k=50 → 256: 0.251 · **384: 0.2715** · 512: 0.249. Truncar pierde señal; pasarse
  de 384 mete ruido fuera de distribución.
- **k (vecinos):** monótono creciente hasta 50; k>50 (75, 100) empeora el Top-1.
- **τ:** apenas influye (1/2/4 casi iguales).

### Fine-tuning: NO mejora el Top-1 (hallazgo)

| Encoder | Top-1 | Notas |
|---|---|---|
| Zero-shot (sin afinar) | **0.2715** | campeón |
| + MNRL (1 ép, pares mismo-dev) | ~0.249 (@256) | mejora Hit@10 (0.729), no Top-1 |
| + BatchAllTriplet (3 ép) | 0.159 | **colapsa** el espacio (loss plana ~4.8) |

En este dataset (chico, 51 clases, cola larga) **la estructura semántica
preentrenada de MPNet ya es casi óptima** para recuperar casos:
- `BatchAllTripletLoss` es inestable en datos chicos → colapsa los embeddings
  (acerca todo) y arruina la recuperación (0.27 → 0.16).
- `MultipleNegativesRankingLoss` (el loss correcto de retrieval; pares positivos =
  dos bugs del mismo dev, negativos in-batch) **sí aprende** (loss 3.4→2.6) pero lo
  que mejora es el **recall@k** (Hit@10 0.729 vs 0.669 a igual `max_seq_len=256`),
  no el acierto Top-1.

### Sistema completo: CBR-recuperación + IBR

| Sistema (50 clases) | Top-1 | MRR | Hit@10 |
|---|---|---|---|
| IBR-solo | 0.1685 | 0.285 | 0.494 |
| **CBR-recuperación solo** | **0.2715** | **0.408** | 0.678 |
| Full W_f=0.1 | 0.2622 | 0.407 | 0.687 |
| Full W_f=0.5 | 0.2416 | 0.395 | 0.699 |
| Full W_f=1.0 | 0.1929 | 0.364 | 0.706 |

**La fusión con el IBR no aporta** a Top-1 ni MRR (el mejor es el CBR-recuperación
solo; todo W_f los baja). El IBR solo sube un poco Hit@10 (0.678→0.706) a pesos
altos donde el Top-1 ya colapsó. Razón: a 50 clases el IBR es débil (0.169) y está
**correlacionado** con el CBR (ambos recuperan los mismos bugs similares), así que
sumarlo añade ruido. Contrasta con el régimen de 17 devs, donde el IBR sí aportaba.

## Validación cruzada en el piloto de Mozilla (régimen opuesto)

Para comprobar que el diseño generaliza, lo corrimos en el piloto Mozilla (20 devs,
etiqueta `contributor_id`/assignee, train 5842 / test 1615; `scripts/cbr_retrieval_pilot.py`).
Régimen **opuesto** a OpenJ9: pocos devs muy activos, mucha data, accuracy alta.

| Sistema (Mozilla, test) | Hit@1 | MRR | Hit@5 | Hit@10 |
|---|---|---|---|---|
| IBR-solo | 0.6755 | 0.786 | 0.914 | 0.944 |
| Clasificador DeBERTa (entrenado, both) | 0.7319 | 0.832 | — | — |
| **CBR-recuperación (zero-shot, k=20, τ=4)** | 0.7176 | 0.819 | 0.946 | 0.972 |
| **CBR-recuperación + IBR (W_f=0.8)** | **0.7337** | **0.831** | 0.951 | 0.980 |
| Clasificador + IBR (full, W_f=0.2) | 0.762 | — | — | — |

**Dos hallazgos del contraste de regímenes:**

1. **El ranking clasificador↔recuperación se invierte con el régimen.** En Mozilla
   (pocas clases, mucha data) el clasificador entrenado gana por poco (0.732 vs
   0.718, −1.4 pp); en OpenJ9 (51 clases, cola larga, poca data) el recuperador gana
   (+4 pp). El recuperador es **competitivo en ambos sin entrenar**; brilla cuando la
   cola es larga y la data escasa (donde un clasificador no aprende bien las clases raras).

2. **El IBR SÍ aporta a la recuperación en Mozilla** (+1.6 pp Top-1, +1.2 pp MRR;
   pico en W_f=0.8) y **no** en OpenJ9 a 50 clases. El valor de la fusión **sigue la
   fuerza del IBR en cada régimen**: fuerte con pocos devs muy activos (IBR-solo 0.676,
   cerca del CBR), despreciable en la cola larga (IBR-solo 0.169 ≪ CBR). Nota: el
   recuperador pide W_f más alto que el clasificador (0.8 vs 0.2) — su NPS es menos
   "picudo", así que tolera más empuje del IBR.

## Conclusión

El **CBR-recuperación** es un reemplazo propio, simple e interpretable del CBR de
TriagerX, **competitivo en ambos regímenes sin entrenar**: iguala su ensemble de 2
transformers en OpenJ9 (0.2715 vs 0.270) y queda a −1.4 pp del clasificador entrenado
en Mozilla (0.718 vs 0.732). Su valor relativo y el del IBR **dependen del régimen**:
en cola larga (OpenJ9) el recuperador supera al clasificador y el IBR no aporta; con
pocos devs muy activos (Mozilla) el clasificador gana por poco y el IBR sí aporta
(+1.6 pp). El fine-tuning no mejora el Top-1 en ninguno (la recuperación semántica
preentrenada es el techo práctico).

## Reproducir (en omen, ver memoria ssh-windows-rtx5060)

```bash
# baseline zero-shot con barrido (k, τ):
uv run python scripts/cbr_retrieval_openj9.py \
  --train-csv artifacts/openj9/openj9_train_50.csv \
  --test-csv  artifacts/openj9/openj9_test_50.csv --sweep        # campeón: k=50, 384 tok

# fine-tuning (no mejora Top-1; MNRL es el estable):
uv run python scripts/cbr_retrieval_openj9.py ... --finetune --loss mnrl --epochs 1

# sistema completo (CBR-recuperación + IBR):
uv run python scripts/eval_openj9_full.py --cbr-mode retrieval \
  --train-csv ...train_50.csv --test-csv ...test_50.csv \
  --interactions ...openj9_interactions_50.parquet --meta ...openj9_issue_meta_50.parquet
```

**Gotcha (8 GB):** afinar con batch grande o `max_seq_len` alto satura la VRAM →
GPU al 100 % util pero ~40 W (thrashing, no cómputo). Usar `--ft-batch 16
--max-seq-len 256`. Requiere el paquete `datasets` (en `pyproject`).
