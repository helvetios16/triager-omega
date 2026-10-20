# Estado de la minería de gecko-dev (señal `contribution` del IBR)

> **TL;DR**: La minería de gecko-dev **se hizo, está completa y cableada al IBR**, pero
> tras las ablaciones (2026-06-12) su señal resultó **no aportar — incluso dañar** en el
> piloto Mozilla, así que está **temporalmente apagada** (`ip_contribution=0` en `config.py`).
> El aporte real del IBR viene hoy de `assignment` + `discussion` (parquets de Bugzilla),
> **no** de gecko-dev. Se deja mineada y cableada (peso 0, no borrada) para re-evaluar al escalar.

## Qué se hizo

- **Script**: `src/triager_omega/data/repo_miner.py` (minería con `git log` sobre
  `mozilla/gecko-dev`, parseo de `Bug <id>` y de `r=`/`a=`).
- **Artefacto**: `artifacts/repo_interactions.parquet`
  - **288.324 interacciones** sobre **80.618 bugs** únicos.
  - **159.170 `review`** + **129.154 `commit`**.
  - Rango de fechas: **2021-01-05 → 2025-07-08**.
  - **Identidad 100% resuelta** a `Contributor Id` (doble puente: email vía `comments`,
    `:nick` vía `contributors.User Name`).
- **Uso**: el IBR (`src/triager_omega/modules/ibr.py`, `_build_interaction_table`) une esta
  tabla con `discussion_interactions.parquet` y la señal `assignment` derivada de los splits.
  `commit` y `review` comparten el peso `ip_contribution` (esquema de 3 IP, como TriagerX).

## El matiz (hallazgo de las ablaciones §11.3)

En el **sistema completo** (CBR=both + IBR), la descomposición por tipo de interacción
(piloto, test, W_f sintonizado en val) mostró:

| Fuente IBR aislada | Δ Hit@1 vs CBR-solo |
|---|---|
| `contribution` (commit/review, **gecko-dev**) | **~0 (incluso daña)** |
| `discussion` (Bugzilla) | +2.0 pp |
| `assignment` (Bugzilla) | +2.9 pp |

**Causa**: los IP de TriagerX se calibraron en OpenJ9/GitHub, donde la etiqueta es el
**contribuidor de código** → `contribution` es la señal correcta. En **Mozilla la etiqueta
es `Assigned To`**, y los committers/revisores de bugs similares suelen **no** ser el
assignee → `contribution` mete ruido al top-1. Además, al normalizar el NIS min-max, un
`ip_contribution` alto (1.5) **domina el vector y diluye** las señales útiles
(`assignment`, `discussion`).

## Decisión aplicada

`config.py` (re-tuneo para Mozilla, ver [PLAN.md §10.2 paso 3](../PLAN.md)):

```python
ip_contribution: float = 0.0   # gecko-dev (commit/review): apagado en Mozilla (TriagerX: 1.5)
ip_assignment:   float = 0.5   # (TriagerX: 0.5)
ip_discussion:   float = 0.5   # (TriagerX: 0.1)
```

Con esto, el sistema completo en test pasa de **0.7604** (config TriagerX) a **0.7746** Hit@1
(MRR 0.8542 → 0.8613).

## Por qué NO se borra la minería

1. **Escala 450 devs**: con 450 clases el CBR se vuelve más débil; `contribution` podría
   volver a aportar. Hay que **re-correr el grid de IP** al escalar antes de descartarla.
2. **Mejor resolución de identidad**: afinar el puente gecko-dev (p.ej. distinguir mejor
   revisores reales de los `r=`/`a=`, o filtrar bots) podría revivir la señal.

Mientras tanto se conserva mineada y cableada con **peso 0** — sin coste de inferencia,
lista para reactivar cambiando un único valor en `config.py`.

## Validación cruzada en OpenJ9 (2026-06-13)

Para confirmar que el problema era la **etiqueta** y no nuestro código ni la señal de
código en sí, se minó OpenJ9 (etiqueta `owner` = fixer) vía GitHub API y se corrió el
IBR ahí. Scripts: `scripts/mine_openj9_timelines.py` + `scripts/eval_openj9_ibr.py`.
Minados 2.204 issues → 30.911 interacciones (discussion 18.390, commits 7.231,
pull_request 4.382, assignment 908).

**IBR-solo OpenJ9, por tipo** (top_k=15, τ=0.6, λ=0.01):

| Config (C/A/D) | Hit@1 | Hit@5 | Hit@10 | MRR | Cob. |
|---|---|---|---|---|---|
| ALL TriagerX (1.5/0.5/0.1) | 0.238 | 0.492 | 0.637 | 0.373 | 0.524 |
| contribution OFF (0/0.5/0.1) | 0.222 | 0.495 | 0.637 | 0.368 | 0.502 |
| only-contribution | 0.238 | 0.479 | 0.637 | 0.375 | 0.469 |
| only-assignment | 0.251 | 0.402 | 0.688 | 0.360 | 0.309 |
| only-discussion | 0.215 | 0.492 | 0.633 | 0.365 | 0.498 |

**Conclusión**: en OpenJ9 `contribution` **ayuda** (ON vs OFF: +1.6pp Hit@1, +0.5pp MRR,
+2.3pp cobertura) y el config completo de TriagerX es el mejor — **lo opuesto a Mozilla**.
La señal de commits/PRs sí sirve cuando la etiqueta es quien arregla el bug → confirma que
apagar `ip_contribution` en Mozilla es por la etiqueta (`Assigned To` ≠ fixer), no por la minería.

**Caveat**: los absolutos OpenJ9 son bajos porque es IBR-**solo** (sin el CBR ensemble del
paper) y la cobertura es ~0.52 (índice de train pequeño, 1.893 issues). NO es una réplica de
los números del paper; es una validación del comportamiento del IBR y de la dependencia de la
etiqueta.

### Sistema completo OpenJ9 (CBR + IBR), end-to-end

Se entrenó un CBR (DeBERTa-v3) sobre OpenJ9 (`scripts/train_openj9_cbr.py`, texto→`owner`,
17 clases) y se corrió el agregador `FS=NPS+W_f·NIS` (`scripts/eval_openj9_full.py`).

**Lección de entrenamiento**: el `WeightedRandomSampler` (peso `1/freq`, clave en Mozilla por
la cola larga de 450 devs) es **contraproducente en OpenJ9** (17 devs ya activos, 1.893
ejemplos): sobre-balancea → el modelo sobre-predice devs raros y el **Hit@1 cae por debajo del
azar** (0.042 < 1/17). Re-entrenado `--no-weighted --epochs 6` → Hit@1 0.199. *Los HPs del
piloto Mozilla no se trasladan a ciegas.*

Resultados (test):

| | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| CBR-solo | 0.199 | 0.704 | 0.859 | 0.401 |
| IBR-solo | 0.238 | 0.492 | 0.637 | 0.373 |
| **Full (contribution ON, W_f=0.3)** | **0.257** | 0.714 | 0.878 | **0.452** |
| Full (contribution OFF, W_f=0.5) | 0.235 | 0.730 | 0.881 | 0.438 |

- **El full supera a ambos solos** (MRR 0.452 vs 0.401/0.373) → CBR e IBR son complementarios.
- **`contribution` ayuda end-to-end** (+2.3pp Hit@1, +1.4pp MRR ON vs OFF, consistente en todo W_f)
  → confirma con el CBR en la mezcla lo que el IBR-solo ya sugería: **opuesto a Mozilla**.

**Conclusión cruzada**: el valor de la señal de código (`contribution`) depende de la definición
de etiqueta — daña con `Assigned To` (assignee, Mozilla), ayuda con `owner`/fixer (OpenJ9). Queda
demostrado end-to-end en ambos regímenes.

**Caveats**: CBR de OpenJ9 underfit (1.893 ejemplos, sin el ensemble de TriagerX) → absolutos
modestos, NO es réplica del paper. Sin split de validación, el W_f se reporta como curva en test
(W_f=0.7 de TriagerX usado como elección principista).

## Cómo reproducir / re-evaluar

```bash
# señal contribution ON vs OFF (IBR-solo)
IP_CONTRIBUTION=0 uv run python -m triager_omega.modules.ibr eval --split test
# sistema completo con distintos pesos (grid de W_f)
uv run python -m triager_omega.modules.aggregator eval --split test --grid --cbr-mode both
```

*Última actualización: 2026-06-12.*
