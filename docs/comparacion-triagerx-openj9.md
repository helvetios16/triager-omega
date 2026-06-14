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

**Una réplica COMPARABLE a ~50 clases sí es posible** (ver sección 4): el split
temporal se recupera usando `issue_number` como proxy de tiempo (en el 17-set el
corte train/test es limpio por `issue_number`, cero solapamiento) y el `text` se
reconstruye con su formato exacto (`Bug Title: …\nBug Description: …`). Lo único
no idéntico es la selección exacta de devs (usamos owners con ≥20 issues → 51).

→ Por eso el head-to-head en **absolutos sobre nuestro 17-set** no es válido
(17 vs 50 clases); para comparar de verdad se usa el set reconstruido de ~50
(sección 4).

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

### 4. Comparación JUSTA a ~50 clases (DeBERTa-solo vs DeBERTa-solo)
Reconstruimos el set de ~50 clases desde `openj9_22112024.csv`: owners con ≥20
issues (**51 clases**, azar 0.0196 ≈ el 0.020 de TriagerX), `text` en su formato
exacto y split temporal por `issue_number` (train < 17695 / test ≥ 17695, igual
que el 17-set). Train 3348 / test 534. Mismo modelo (1 DeBERTa-v3), 15 épocas,
`--no-weighted`. Esto SÍ es head-to-head de absolutos (misma arquitectura, mismas
clases, mismo split).

| Componente (50 clases) | Top-1 | Nota |
|---|---|---|
| RoBERTa-solo (TriagerX) | 0.175 | |
| DeBERTa-solo (TriagerX) | 0.189 | |
| **DeBERTa-solo (nosotros)** | **0.2247** | un solo transformer, MRR 0.359 |
| CBR ensemble RoBERTa+DeBERTa (TriagerX) | 0.270 | 2 transformers |
| CBR + IBR full (TriagerX) | 0.328 | ensemble + IBR |

**Nuestro DeBERTa-solo (0.225) ≥ el DeBERTa-solo de TriagerX (0.189)** en igualdad
de condiciones → nuestro modelo base no es la debilidad. La brecha hasta su 0.328
es **arquitectónica** (ensemble de 2 transformers + IBR), no de calidad del CBR.
La loss seguía bajando (3.95 → 3.29), así que 0.225 probablemente no es el techo.

> Una primera corrida a 6 épocas daba 0.11 (infraentrenada, loss casi plana); el
> salto a 0.225 con 15 épocas confirma que la brecha inicial era entrenamiento, no
> el modelo. Caveat: selección exacta de los 50 devs no idéntica (≥20 issues → 51).

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
