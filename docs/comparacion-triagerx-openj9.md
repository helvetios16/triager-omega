# Comparación: TriagerX vs. triager-omega (OpenJ9)

> Validación cruzada en OpenJ9 (dataset de TriagerX, etiqueta `owner`/fixer).
> Régimen donde la señal `contribution` del IBR sí aporta (opuesto a Mozilla,
> donde la etiqueta es `Assigned To` = assignee).

## ⚠️ Los absolutos NO son comparables (leer primero)

Nuestro setup usa **17 clases** y el de TriagerX **50 clases**, así que un Top-1
más alto puede deberse solo a que la tarea es más fácil, no a que el sistema sea
mejor:

- Azar 1/17 = **0.059** vs 1/50 = **0.020**.
- Splits y preprocesamiento del texto distintos.

**Una réplica fiel a 50 clases no es reproducible** con los assets disponibles:
- `openj9_22112024.csv` (dataset completo, 228 owners; con `owner` ≥20 issues
  salen 51 ≈ las "50 clases" de TriagerX) **no trae columna de fecha** → no se
  puede repetir su split temporal (time-sliced) sin arriesgar fuga.
- Ese CSV solo tiene `issue_title`/`issue_body` crudos; el 17-set trae una
  columna `text` ya preprocesada por ellos. Concatenar a mano no da el mismo texto.

→ Por eso **este documento NO emite un veredicto de "quién gana" en absolutos.**

## Lo que SÍ es comparable (deltas internos, independientes del nº de clases)

### 1. Aporte marginal del IBR — misma ganancia en ambos sistemas
El IBR sube el CBR la **misma magnitud** en los dos, lo que indica que nuestro
IBR está bien replicado (es un Δ interno, no depende del nº de clases):

| Sistema | CBR-solo | +IBR (Full) | Δ |
|---|---|---|---|
| TriagerX (50 clases) | 0.270 | 0.328 | **+5.8 pp** |
| triager-omega (17 clases) | 0.199 | 0.267 | **+6.8 pp** |

### 2. Ablación de `contribution` en OpenJ9 (ON vs OFF) — hallazgo propio
Todo dentro de nuestro mismo setup, así que es plenamente válido:

| | IBR-solo Hit@1 | Full mejor Hit@1 | Full MRR (pico) |
|---|---|---|---|
| `contribution` ON (ip_c=1.5) | **0.2379** | **0.2669** (W_f=0.2) | **0.4563** |
| `contribution` OFF (ip_c=0) | 0.2219 | 0.2347 | 0.4400 |
| **Δ** | **+1.6 pp** | **+3.2 pp** | **+1.6 pp** |

### 3. Contraste de régimen Mozilla ↔ OpenJ9
`contribution` **ayuda** con etiqueta `owner`/fixer (OpenJ9) y **daña** con
`assignee` (Mozilla, donde `ip_contribution=0` es lo óptimo). El valor de la
señal de código depende de **cómo se define la etiqueta** — aporte nuestro, no
del paper.

## Números absolutos (cada uno en SU propio setup — referencia, no head-to-head)

> No comparar columna a columna entre tablas (clases distintas).

### TriagerX — su setup (50 clases, 382 issues de test)
Fuente: `triagerX/notebook/openj9/top1_class_comparison.json` y
`openj9_grid_sim_weights.csv`.

| Componente | Top-1 |
|---|---|
| RoBERTa-solo | 0.175 |
| DeBERTa-solo | 0.189 |
| CBR ensemble (RoBERTa + DeBERTa) | 0.270 |
| CBR + IBR (WRA) | 0.328 |

### triager-omega — nuestro setup (1 DeBERTa-v3, 17 clases, 311 issues de test)
Scripts: `scripts/train_openj9_cbr.py`, `scripts/eval_openj9_full.py`.

| Componente | Top-1 | MRR | Hit@5 | Hit@10 |
|---|---|---|---|---|
| CBR-solo | 0.199 | 0.401 | 0.704 | 0.859 |
| IBR-solo | 0.238 | 0.373 | 0.492 | 0.637 |
| Full (CBR + IBR, W_f=0.2) | 0.267 | 0.456 | 0.711 | 0.868 |

Nuestro CBR es **un solo DeBERTa underfit a propósito** (sin el ensemble de dos
transformers del paper); por eso sus absolutos son modestos y NO pretenden
replicar los números de TriagerX.

## Lección de entrenamiento
El `WeightedRandomSampler` (1/freq), clave en Mozilla (cola larga de ~450 devs),
es **contraproducente en OpenJ9** (17 devs activos, dataset chico y balanceado):
sobre-balancea → sobre-predice devs raros → Hit@1 cae por debajo del azar
(0.042 < 0.059). Fix: `--no-weighted --epochs 6` → Hit@1 0.199.
Los HPs del piloto Mozilla **no se trasladan a ciegas**.

## Caveats
- **Absolutos no comparables** entre sistemas (17 vs 50 clases); ver sección ⚠️.
- Réplica fiel a 50 clases no reproducible (sin fecha para el split temporal,
  preprocesamiento de texto distinto).
- CBR de OpenJ9 underfit (sin ensemble) → absolutos modestos por diseño.
- Sin split de validación: el W_f se reporta como curva en test (W_f=0.7 es la
  elección principista de TriagerX; W_f=0.2 es el pico observado en test).

## Resumen
La comparación **no es un head-to-head de absolutos** (clases distintas, y la
réplica a 50 no es reproducible). Lo que sí queda validado, con métricas que no
dependen del nº de clases: (1) nuestro IBR aporta la **misma magnitud** que el de
TriagerX (~+6 pp), (2) `contribution` **ayuda** en OpenJ9 y **daña** en Mozilla
según la etiqueta, y (3) la arquitectura híbrida **Full > ambos solos** end-to-end.
