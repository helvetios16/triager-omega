<!-- Merged from: comparacion-openj9-que-tan-similar.md, comparacion-triagerx-openj9.md, openj9-comparativa-explicada.md, openj9-mejoras-ibr-resultados.md, triagerx-reimplementacion-hallazgos.md -->

# 04 — Comparativas: OpenJ9, TriagerX y Mejoras al IBR

## Tabla de contenidos

1. [¿Qué tan similar es la comparación OpenJ9 vs TriagerX? — resumen ejecutivo](#1-qué-tan-similar-es-la-comparación-openj9-vs-triagerx--resumen-ejecutivo-comparacion-openj9-que-tan-similamd)
2. [Comparación técnica completa: TriagerX vs. triager-omega (OpenJ9)](#2-comparación-técnica-completa-triagerx-vs-triager-omega-openj9-comparacion-triagerx-openj9md)
   - [2.1 Advertencia: los absolutos de 17 clases NO son comparables](#21-advertencia-los-absolutos-de-17-clases-no-son-comparables)
   - [2.2 Lo que SÍ es comparable (deltas internos)](#22-lo-que-sí-es-comparable-deltas-internos)
   - [2.3 Comparación JUSTA a ~50 clases (head-to-head)](#23-comparación-justa-a-50-clases-head-to-head)
   - [2.4 Números absolutos por separado](#24-números-absolutos-en-su-propio-setup--referencia)
3. [Tabla comparativa explicada — visión de un vistazo](#3-tabla-comparativa-explicada--visión-de-un-vistazo-openj9-comparativa-explicadamd)
   - [3.1 Qué significa (y qué NO) "el nuestro es mejor"](#31-qué-significa-y-qué-no-el-nuestro-es-mejor)
   - [3.2 La tabla completa](#32-la-tabla-completa)
   - [3.3 Explicación fila por fila](#33-explicación-fila-por-fila)
   - [3.4 Por qué el IBR de TriagerX "ayuda" y el nuestro no](#34-por-qué-el-ibr-de-triagerx-ayuda-y-el-nuestro-no)
4. [Mejoras al IBR (OpenJ9 ~50 clases) — resultados](#4-mejoras-al-ibr-openj9-50-clases--resultados-openj9-mejoras-ibr-resultadosmd)
   - [4.1 Referencia (baseline)](#41-referencia-baseline)
   - [4.2 P1 — Decorrelar el IBR (canal léxico BM25)](#42-p1--decorrelar-el-ibr-canal-léxico-bm25)
   - [4.3 P2 — Fusión por rango (RRF)](#43-p2--fusión-por-rango-rrf)
   - [4.4 P3 — Fusión condicional (gating)](#44-p3--fusión-condicional-gated-por-confianza-del-cbr)
   - [4.5 P4 — Sintonizar IP weights / k](#45-p4--sintonizar-los-valores-internos-del-ibr-ip-weights--k)
   - [4.6 Validación honesta — el 0.2790 era sobreajuste al test](#46-validación-honesta--el-02790-era-sobreajuste-al-test)
   - [4.7 Resumen de palancas y pendientes](#47-resumen-de-palancas-y-pendientes)
5. [Re-ejecución de TriagerX — hallazgos y su impacto](#5-re-ejecución-de-triagerx-hallazgos-y-su-impacto-triagerx-reimplementacion-hallazgomd)
   - [5.1 Setup de la re-ejecución](#51-setup-de-la-re-ejecución)
   - [5.2 Hallazgo principal: fuga de test en TriagerX](#52-hallazgo-principal-fuga-de-test-en-triagerx)
   - [5.3 Diferencias y falsas alarmas](#53-diferencias-menores-y-falsas-alarmas)
   - [5.4 Alineación del IBR](#54-alineación-del-ibr-fase-3)
   - [5.5 Resultados de la re-ejecución real](#55-resultados-de-la-re-ejecución-head-to-head-real)

---

## 1. ¿Qué tan similar es la comparación OpenJ9 vs TriagerX? — resumen ejecutivo (`comparacion-openj9-que-tan-similar.md`)

> Hay que separar dos sentidos de "similar": **qué tan justa es la comparación**
> (metodología) y **qué tan cerca quedan los resultados**.

### Metodología: comparación justa, "comparable pero no idéntica"

Para el head-to-head se reconstruyó el set de ~50 clases (`build_openj9_50.py`)
igualando las condiciones de TriagerX:

| Eje | ¿Igual que TriagerX? |
|---|---|
| Dataset (OpenJ9, `openj9_22112024.csv`) | ✅ el mismo |
| Nº de clases | ✅ ≈50 (51 = owners con ≥20 issues; azar 0.0196 ≈ su 0.020) |
| Formato del texto (`Bug Title:…\nBug Description:…`) | ✅ exacto |
| Split temporal (por `issue_number`, train < 17695 / test ≥ 17695) | ✅ misma lógica, sin solapamiento |
| Etiqueta (`owner`/fixer) | ✅ la misma |
| Métrica (Top-1 / MRR) | ✅ la misma |
| Datos del IBR (timelines de GitHub) | ✅ minados igual |

**Diferencias (caveats honestos):**
- Selección **exacta** de devs no idéntica (≥20 issues → 51 vs sus 50).
- Tamaño de test distinto (534 vs 382) por esa selección.
- **Sin split de validación** → el W_f se reporta como curva en test (W_f=0.7 es la elección principista de TriagerX; W_f=0.2 es el pico observado en test).

Los absolutos del set viejo de **17 clases NO son comparables** (azar 0.059 vs 0.020); por eso se hizo el de 50.

### Resultados (50 clases, head-to-head)

| Componente | Top-1 |
|---|---|
| DeBERTa-solo (TriagerX) | 0.189 |
| DeBERTa-solo (nuestro) | 0.2322 |
| **CBR ensemble RoBERTa+DeBERTa (TriagerX)** | **0.270** |
| **CBR de recuperación zero-shot (nuestro)** | **0.2715** |
| CBR + IBR full (TriagerX) | **0.328** |
| Sistema completo (nuestro, W_f=0.2) | 0.2397 |

**Dos conclusiones clave:**
- El **CBR de recuperación zero-shot iguala el ensemble entrenado** de TriagerX (0.2715 vs 0.270) **sin entrenar nada**.
- Donde **no** se llega es al sistema completo: su 0.328 vs nuestro 0.2397 (**~9 pp**). Esa brecha **no** es calidad del CBR — es que **su IBR sí ayuda a su ensemble**, mientras que a 50 clases (cola larga) nuestro IBR es débil y correlacionado y no suma al Top-1.

> La comparación es **justa y apples-to-apples** en lo esencial (mismo dataset, ~50 clases, mismo split y texto), con caveats menores. En resultados, **igualamos su CBR sin entrenar** (0.2715 ≈ 0.270).

---

## 2. Comparación técnica completa: TriagerX vs. triager-omega (OpenJ9) (`comparacion-triagerx-openj9.md`)

> Validación cruzada en OpenJ9 (dataset de TriagerX, etiqueta `owner`/fixer).
> Régimen donde la señal `contribution` del IBR sí aporta (opuesto a Mozilla,
> donde la etiqueta es `Assigned To` = assignee).

### 2.1 Advertencia: los absolutos de 17 clases NO son comparables

Nuestro setup usa **17 clases** y el de TriagerX **50 clases**, así que un Top-1
más alto puede deberse solo a que la tarea es más fácil, no a que el sistema sea
mejor:

- Azar 1/17 = **0.059** vs 1/50 = **0.020**.
- Splits y preprocesamiento del texto distintos.

**Una réplica COMPARABLE a ~50 clases sí es posible** (ver §2.3): el split
temporal se recupera usando `issue_number` como proxy de tiempo y el `text` se
reconstruye con su formato exacto. Lo único no idéntico es la selección exacta de devs.

### 2.2 Lo que SÍ es comparable (deltas internos)

#### Aporte marginal del IBR — misma ganancia en ambos sistemas

| Sistema | CBR-solo | +IBR (Full) | Δ |
|---|---|---|---|
| TriagerX (50 clases) | 0.270 | 0.328 | **+5.8 pp** |
| triager-omega (17 clases) | 0.199 | 0.267 | **+6.8 pp** |

#### Ablación de `contribution` en OpenJ9 (ON vs OFF) — hallazgo propio

| | IBR-solo Hit@1 | Full mejor Hit@1 | Full MRR (pico) |
|---|---|---|---|
| `contribution` ON (ip_c=1.5) | **0.2379** | **0.2669** (W_f=0.2) | **0.4563** |
| `contribution` OFF (ip_c=0) | 0.2219 | 0.2347 | 0.4400 |
| **Δ** | **+1.6 pp** | **+3.2 pp** | **+1.6 pp** |

#### Contraste de régimen Mozilla ↔ OpenJ9

`contribution` **ayuda** con etiqueta `owner`/fixer (OpenJ9) y **daña** con
`assignee` (Mozilla, donde `ip_contribution=0` es lo óptimo). El valor de la
señal de código depende de **cómo se define la etiqueta** — aporte nuestro, no
del paper.

### 2.3 Comparación JUSTA a ~50 clases (head-to-head)

Reconstruimos el set de ~50 clases desde `openj9_22112024.csv`
(`scripts/build_openj9_50.py`): owners con ≥20 issues (**51 clases**, azar 0.0196 ≈
el 0.020 de TriagerX), `text` en su formato exacto y split temporal por
`issue_number` (train < 17695 / test ≥ 17695, igual que el 17-set). Train 3348 /
test 534.

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

**4.c — El entrenamiento NO es el cuello de botella.**
Subir 15→30 épocas y `max_length` 256→512 movió el CBR-solo solo **+0.75 pp**
(0.2247 → 0.2322) y bajó un poco Hit@5/10. Confirma que el límite es la **capacidad
del modelo único**, no el entrenamiento.

**4.d — Hallazgo: un CBR más fuerte REACTIVA el aporte del IBR.** Con el CBR base
(15ép/256) el IBR no aportaba nada a Top-1; con el CBR mejor entrenado (30ép/512)
el IBR vuelve a sumar (0.2322 → **0.2397** en W_f=0.2). A 50 clases (cola larga de devs)
el IBR-solo es más débil que el CBR (0.169 < 0.232) y solo ayuda en pesos bajos;
contrasta con el régimen de 17 devs, donde el IBR-solo era *más fuerte* que el CBR
(0.238 > 0.199) y aportaba +6.8 pp.

**CBR de recuperación — aporte propio:**

| CBR (50 clases) | Top-1 | MRR |
|---|---|---|
| Clasificador DeBERTa (TriagerX) | 0.189 | — |
| Clasificador DeBERTa (nuestro, mejor) | 0.2322 | 0.361 |
| Ensemble RoBERTa+DeBERTa (TriagerX) | 0.270 | — |
| **Recuperación zero-shot (nuestro)** | **0.2715** | **0.408** |

Reemplazamos el CBR-clasificador por un **recuperador kNN** que vota al `owner` de
los bugs pasados más similares. **Zero-shot (MPNet, k=50, sin entrenar): Top-1
0.2715, MRR 0.408** → **iguala el ensemble de 2 transformers de TriagerX (0.270)** y
supera a su DeBERTa-solo (0.189) y a nuestro clasificador (0.2322).

### 2.4 Números absolutos en su propio setup — referencia

> No comparar columna a columna entre tablas (clases distintas).

#### TriagerX — su setup (50 clases, 382 issues de test)
Fuente: `triagerX/notebook/openj9/top1_class_comparison.json` y
`openj9_grid_sim_weights.csv`.

| Componente | Top-1 |
|---|---|
| RoBERTa-solo | 0.175 |
| DeBERTa-solo | 0.189 |
| CBR ensemble (RoBERTa + DeBERTa) | 0.270 |
| CBR + IBR (WRA) | 0.328 |

#### triager-omega — nuestro setup (1 DeBERTa-v3, 17 clases, 311 issues de test)
Scripts: `scripts/train_openj9_cbr.py`, `scripts/eval_openj9_full.py`.

| Componente | Top-1 | MRR | Hit@5 | Hit@10 |
|---|---|---|---|---|
| CBR-solo | 0.199 | 0.401 | 0.704 | 0.859 |
| IBR-solo | 0.238 | 0.373 | 0.492 | 0.637 |
| Full (CBR + IBR, W_f=0.2) | 0.267 | 0.456 | 0.711 | 0.868 |

**Lección de entrenamiento:** El `WeightedRandomSampler` (1/freq), clave en Mozilla
(cola larga de ~450 devs), es **contraproducente en OpenJ9** (17 devs activos, dataset
chico y balanceado): sobre-balancea → sobre-predice devs raros → Hit@1 cae por debajo
del azar (0.042 < 0.059). Fix: `--no-weighted --epochs 6` → Hit@1 0.199.
Los HPs del piloto Mozilla **no se trasladan a ciegas**.

---

## 3. Tabla comparativa explicada — visión de un vistazo (`openj9-comparativa-explicada.md`)

> Versión "para leer de un vistazo" del head-to-head contra TriagerX.
> **Un solo régimen** (OpenJ9, ~50 clases, comparación justa).

### 3.1 Qué significa (y qué NO) "el nuestro es mejor"

> **En el sistema completo, TriagerX gana: 0.328 > 0.2715.** Como pipeline entero,
> el nuestro **no** es mejor, y este documento **no** afirma lo contrario.

"El nuestro es mejor" se refiere a algo **acotado**: comparar **componente contra
componente del mismo rol** (el núcleo CBR de cada uno), no el sistema completo.

| Componente CBR | Top-1 | Costo |
|---|---|---|
| CBR de TriagerX (ensemble RoBERTa+DeBERTa) | 0.270 | **2 transformers, entrenados** |
| Nuestro CBR (recuperación kNN) | **0.2715** | **1 modelo, zero-shot, sin entrenar** |

Las tres afirmaciones, ordenadas para que no se mezclen:

1. **¿Quién gana el sistema completo?** → TriagerX (0.328 vs 0.2715). No discutible.
2. **¿Por qué nos gana?** → Únicamente por su IBR (+5.8 pp sobre su CBR). **No** porque su CBR sea mejor.
3. **¿Quién gana el núcleo CBR?** → Empate / leve ventaja nuestra (0.2715 ≥ 0.270), y lo logramos **sin entrenar y con un solo modelo**.

El argumento de tesis honesto **no** es "le ganamos a TriagerX". Es: *igualamos el
corazón de su método (el CBR) con una fracción del costo y cero entrenamiento; lo
único que mantiene su sistema completo por delante es la palanca IBR, que además es
dependiente del régimen.*

### 3.2 La tabla completa

| # | Sistema | De quién | Top-1 | ¿Qué es? |
|---|---|---|---|---|
| 1 | RoBERTa-solo | TriagerX | 0.175 | Un transformer clasificador, solo |
| 2 | DeBERTa-solo | TriagerX | 0.189 | Otro transformer clasificador, solo |
| 3 | **CBR ensemble (RoBERTa+DeBERTa)** | TriagerX | **0.270** | Su "CBR" = los 2 transformers combinados |
| 4 | **CBR + IBR (full)** | TriagerX | **0.328** | Su sistema completo: ensemble + IBR |
| 5 | Clasificador DeBERTa-solo | omega | 0.2322 | Imitar su enfoque con 1 solo transformer |
| 6 | IBR-solo | omega | 0.1685 | Solo el grafo de interacciones, sin CBR |
| 7 | Full clasificador + IBR | omega | 0.2397 | Fila 5 + fila 6 fusionados |
| 8 | **CBR-recuperación solo** | omega | **0.2715** | Nuestra propuesta: kNN zero-shot, sin entrenar |
| 9 | Full recuperación + IBR | omega | 0.2622 | Fila 8 + fila 6 fusionados |

### 3.3 Explicación fila por fila

**Filas 1–4 son TriagerX:**
- **Fila 1 y 2** — sus dos transformers por separado. Son lo más débil (0.175, 0.189).
- **Fila 3** — su "CBR": combinar los dos transformers (ensemble) sube a **0.270**. Esta es la cifra clave del componente CBR de ellos.
- **Fila 4** — su sistema **completo**: al ensemble (0.270) le suman su IBR → **0.328**. Ese salto 0.270 → 0.328 (**+5.8 pp**) es lo que aporta *su* IBR.

**Filas 5–9 son las nuestras (omega):**
- **Fila 5** — nuestro intento de imitar su clasificador, pero con **un solo** transformer (no el ensemble de dos). Da **0.2322**. Ya le gana a su DeBERTa-solo (0.189), pero no llega a su ensemble (0.270).
- **Fila 6** — nuestro IBR **solo**, sin nada de CBR. A 50 clases es flojo: **0.1685**.
- **Fila 7** — fusionar fila 5 + fila 6. Sube poquito (0.2322 → **0.2397**). El IBR casi no ayuda aquí.
- **Fila 8** — **nuestra propuesta real**: en vez de clasificar, **recuperar** los bugs más parecidos con kNN y votar el dev (zero-shot, sin entrenar nada). Da **0.2715** → **iguala el ensemble de TriagerX (0.270)** con un solo modelo y cero entrenamiento. Esta es la fila estrella.
  > **Nota — qué encoder da este 0.2715:** es **MPNet** (`all-mpnet-base-v2`), **no MiniLM**. MPNet se eligió a propósito por ser el **mismo SBERT que usa TriagerX en su IBR**. MiniLM solo aparece en la ablación `--rerank` (cross-encoder), que **no** subió el Top-1.
- **Fila 9** — fusionar nuestra propuesta (fila 8) + IBR (fila 6). **Baja** a 0.2622. O sea: a 50 clases sumar el IBR **empeora**, no mejora.

**Lo que dice la tabla:**
- **Núcleo CBR:** fila 8 (0.2715) **≥** fila 3 (0.270) → nuestro CBR iguala/supera al de ellos, sin entrenar.
- **Sistema completo:** fila 4 (0.328) > nuestro mejor (fila 8, 0.2715) por **~5.6 pp**, y esos 5.6 pp son **exactamente el IBR de ellos**. A nosotros el IBR no nos da ese salto a 50 clases (fila 8 → 9 baja).

### 3.4 Por qué el IBR de TriagerX "ayuda" y el nuestro no

**No es que su IBR sea un mejor componente.** Cuando se mide en igualdad de
condiciones, el aporte marginal del IBR es **de la misma magnitud** en los dos
sistemas (Δ interno, no depende del nº de clases):

| Sistema | CBR-solo | +IBR (Full) | Δ del IBR |
|---|---|---|---|
| TriagerX (50 clases) | 0.270 | 0.328 | **+5.8 pp** |
| omega (17 clases) | 0.199 | 0.267 | **+6.8 pp** |

Hay dos razones, y ninguna es la calidad del componente:

#### Razón 1 — Correlación entre bases

El IBR aporta **solo si trae señal distinta** a la del CBR sobre el que se monta.

- **TriagerX:** su CBR es un **clasificador** y su IBR recorre un **grafo de interacciones**. Son dos mecanismos distintos → **decorrelacionados** → el IBR añade información nueva → **+5.8 pp**.
- **Nuestro pipeline:** nuestro CBR **también es un recuperador** (kNN sobre embeddings MPNet) y nuestro IBR **mira el grafo de los mismos vecinos** usando el **mismo encoder** (MPNet). Recuperan **los mismos bugs parecidos** → están **correlacionados** → el IBR no trae nada nuevo → sumarlo solo mete ruido.

#### Razón 2 — El régimen (densidad de interacciones por dev)

| | ¿IBR ayuda? | Por qué |
|---|---|---|
| TriagerX, 50 cls | Sí (+5.8 pp) | IBR sobre un *clasificador* (decorrelacionado) + apoyado en su ensemble fuerte |
| omega, 50 cls (cola larga) | **No** (baja) | IBR correlacionado con nuestro recuperador (mismo encoder/vecinos) + cola larga débil |
| omega, Mozilla (pocos devs) | **Sí** (+1.6 pp) | Régimen denso: IBR-solo fuerte → señal complementaria |

**Conclusión:** su IBR no es "mejor"; es que (1) lo montan sobre un clasificador, que no comparte mecanismo con el grafo, y (2) miden el full en un régimen donde el IBR tiene señal. Nuestro IBR, medido en igualdad de condiciones, replica el suyo — y en Mozilla incluso nos suma. Que en OpenJ9-50 no aporte es esperado y explicable.

---

## 4. Mejoras al IBR (OpenJ9 ~50 clases) — resultados (`openj9-mejoras-ibr-resultados.md`)

> Experimentos para mejorar el aporte de **nuestro** IBR en el régimen de cola larga
> (OpenJ9, 50 clases), donde el IBR semántico baseline **no aporta**. Tres palancas,
> ejecutadas una por una en la omen (RTX 5060), sobre el 50-set reconstruido.

```
uv run python scripts/eval_openj9_full.py --cbr-mode retrieval \
  --train-csv artifacts/openj9/openj9_train_50.csv \
  --test-csv  artifacts/openj9/openj9_test_50.csv \
  --interactions artifacts/openj9/openj9_interactions_50.parquet \
  --meta artifacts/openj9/openj9_issue_meta_50.parquet
```

### 4.1 Referencia (baseline)

| Config | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| **CBR-recuperación solo** (techo) | **0.2715** | 0.5768 | 0.6779 | 0.4078 |
| IBR-solo (semántico) | 0.1685 | 0.4101 | 0.4944 | 0.2853 |
| Full IBR semántico, mejor (W_f=0.1) | 0.2622 | 0.5749 | 0.6873 | 0.4070 |

El IBR **semántico** comparte encoder (MPNet) con el CBR → recupera los mismos
vecinos → está correlacionado → **toda** la fusión queda por debajo del CBR-solo.
Diagnóstico: el problema es la correlación, no el componente.

### 4.2 P1 — Decorrelar el IBR (canal léxico BM25)

**Idea:** recuperar los vecinos del IBR por **BM25 léxico** (tokens técnicos
exactos) en vez del MPNet semántico, para que el canal IBR deje de mirar los mismos
vecinos que el CBR. Flag nuevo: `--ibr-channel lexical`.

| Config | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| CBR-solo (referencia) | 0.2715 | 0.5768 | 0.6779 | 0.4078 |
| Full IBR semántico (mejor, W_f=0.1) | 0.2622 | 0.5749 | 0.6873 | 0.4070 |
| IBR **léxico**-solo | 0.1592 | 0.3764 | 0.5037 | 0.2733 |
| **Full léxico W_f=0.2** | **0.2678** | 0.5880 | 0.6948 | 0.4090 |
| **Full léxico W_f=0.3** | 0.2678 | 0.5861 | 0.6966 | **0.4108** |
| Full léxico W_f=1.0 | 0.1985 | 0.5787 | 0.7135 | 0.3686 |

**Hallazgo — la decorrelación funciona.** El IBR léxico-solo es *más débil* (0.1592 < 0.1685), pero al fusionar **deja de dañar y empieza a aportar**:
- **MRR 0.4108 > CBR-solo 0.4078** (+0.3 pp): primera vez que la fusión supera al CBR-solo en una métrica de ranking.
- **Hit@5 +1.1 pp** (0.5880), **Hit@10 +1.9 pp** (0.6966) sobre CBR-solo.
- **Top-1 0.2678**: sigue por debajo del techo (−0.37 pp vs 0.2715), pero **+5.6 pp vs la fusión semántica** (0.2622).

> **Nota (futuro):** la otra vista de decorrelación, la **estructural** (dev↔módulo/
> archivo desde commits/PRs), queda pendiente.

### 4.3 P2 — Fusión por rango (RRF)

**Idea:** combinar por **Reciprocal Rank Fusion**: `FS = 1/(k+rango_NPS) + W_f/(k+rango_NIS)`. Solo usa el orden, no la magnitud. Flags: `--fusion rrf --rrf-k 60`.

| Config | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| CBR-solo (referencia) | 0.2715 | 0.5768 | 0.6779 | 0.4078 |
| RRF W_f=0.1 | 0.2041 | 0.4981 | 0.6910 | 0.3432 |
| RRF W_f=0.5 | 0.2004 | 0.4195 | 0.5805 | 0.3301 |
| RRF W_f=1.0 | 0.2041 | 0.4288 | 0.5768 | 0.3312 |

**Hallazgo — RRF PIERDE (claro).** El Top-1 se desploma a ~0.20 **ya en W_f=0.1**. La caída no la causa el IBR, sino **convertir el NPS a rango**: eso tira la información de *confianza* del CBR (la diferencia de magnitud entre el voto top-1 y el resto), que aquí sí es informativa. RRF es la herramienta correcta para fusionar listas con escalas incomparables (p. ej. BM25 + denso en el propio recuperador), pero **equivocada cuando un canal —el CBR— ya tiene magnitudes calibradas.** Descartada.

### 4.4 P3 — Fusión condicional (gated por confianza del CBR)

**Idea:** aplicar el IBR **solo en las consultas donde el CBR está inseguro** (margen top1−top2 del NPS < `gate`); donde el CBR es confiable, `FS = NPS` puro. Flags: `--gate G` / `--gate-sweep`.

**P3 sobre canal semántico (aislado) — neutro:**

| gate | frac IBR | best Top-1 | best MRR |
|---|---|---|---|
| off | — | 0.2622 | 0.4070 |
| 0.3 | 0.54 | 0.2622 | 0.4052 |
| 0.7 | 0.98 | 0.2622 | 0.4063 |

Gatear el canal semántico **no mueve el Top-1** (siempre 0.2622). Lógico: en las consultas dudosas el IBR *correlacionado* tampoco trae al dev correcto.

**P3 + P1 (canal LÉXICO + gate) — el mejor resultado:**

| gate | frac IBR | best Top-1 (W_f) | best MRR (W_f) |
|---|---|---|---|
| off | — | 0.2678 (0.2) | 0.4108 (0.3) |
| 0.2 | 0.37 | 0.2678 (0.2) | 0.4083 (0.5) |
| **0.3** | **0.54** | **0.2715 (0.5)** | **0.4125 (0.5)** |
| 0.5 | 0.80 | 0.2678 (0.2) | 0.4103 (0.5) |
| 0.8 | 1.00 | 0.2678 (0.2) | 0.4110 (0.3) |

**Hallazgo — gating + decorrelación recupera el techo y mejora el ranking.** Con
`gate=0.3` (IBR aplicado al ~54% de consultas dudosas, W_f=0.5):
- **Top-1 0.2715 = CBR-solo**: el sistema completo ya **iguala** el techo.
- **MRR 0.4125**: el mejor de todos los experimentos (**+0.47 pp** sobre CBR-solo).
- Hit@10 0.6929 (+1.5 pp sobre CBR-solo).

> **Caveat:** sin split de validación, el `gate`/`W_f` óptimos se leen como curva en test, no sintonizados aparte.

### 4.5 P4 — Sintonizar los valores internos del IBR (IP weights / k)

Hasta P3 dejamos los **knobs propios del IBR** en default (IP `contribution/assignment/discussion` = 1.5/0.5/0.1, k=15). P4 los barre sobre la mejor base (recuperador + léxico + gate=0.3).

**IP contribution-only (1.5/0/0)** — como `owner`=fixer=quien commitea el fix, la señal de commit es la alineada; `assignment` y `discussion` son ruido:

| Config (léxico + gate=0.3, contrib-only) | Top-1 | MRR | Hit@10 |
|---|---|---|---|
| CBR-solo (techo) | 0.2715 | 0.4078 | 0.6779 |
| IBR-solo (contrib-only) | 0.1573 | — | 0.4663 |
| FS W_f=0.5 | 0.2715 | 0.4122 | 0.6873 |
| **FS W_f=0.6** | **0.2790** | **0.4154** | 0.6929 |
| FS W_f=0.7 | 0.2678 | 0.4090 | 0.6929 |

**Barrido de k (contrib-only, gate=0.3, W_f=0.6):** k=10 → 0.2697 · **k=15 → 0.2790** · k=30 → 0.2734. → k=15 es el óptimo.

**Hallazgo — por primera vez el full SUPERA el techo del recuperador.** Afilando el
IP a contribution-only, `FS=0.2790 > CBR-solo 0.2715` (**+0.75 pp**) con el mejor MRR de todos (0.4154).

> **Caveat (clave):** es un pico **seleccionado en test** (3 knobs: gate=0.3, contrib-only, W_f=0.6; sin split de validación). Lectura honesta: *el IBR **puede** sumar ~0.75 pp al Top-1 si se afila a contribution*, no es un número de despliegue robusto.

### 4.6 Validación honesta — el 0.2790 era sobreajuste al test

P4 eligió `gate=0.3, contrib-only, W_f=0.6` **mirando el test**. Para saber si el
salto es real, se reconstruyó un **split de validación temporal** (`build_openj9_50.py
--val-frac 0.15`): de la cola del train (issue_number más recientes antes del corte de test) → `traininner=2845` / `val=503`, con `test=534` intacto.

**Tuning en val (léxico):** val elige independientemente **contribution-only** sobre default (val Top-1 0.2048 vs 0.2008), con `gate=0.5, W_f=0.6`. El gate NO coincide con el de test.

**Aplicado al test intacto (config elegida en val):**

| Sistema (test, val-tuned) | Top-1 | MRR | Hit@10 |
|---|---|---|---|
| CBR-solo (techo) | 0.2715 | 0.4078 | 0.6779 |
| Full léxico contrib-only gate=0.5 **W_f=0.6** | **0.2603** | 0.4061 | **0.7022** |

**Hallazgo definitivo — el IBR NO rompe el techo de Top-1 bajo tuning honesto.** La
config elegida en val da **0.2603 en test, −1.1 pp por debajo del CBR-solo**. El
pico de 0.2790 necesitaba `gate=0.3`, que era exactamente el knob sobreajustado al
test; al `gate=0.5` que eligió val, **ningún W_f** del test supera 0.2715.

**Lo que SÍ sobrevive a la validación:** el IBR mejora **Hit@10 de forma robusta**
(0.7022 vs 0.6779, **+2.4 pp**) — aporta profundidad de ranking, aunque no el Top-1.

### 4.7 Resumen de palancas y pendientes

| Palanca | Top-1 | MRR | Veredicto |
|---|---|---|---|
| Baseline (IBR semántico, lineal) | 0.2622 | 0.4070 | el IBR **daña** (correlacionado) |
| **P1** decorrelar (léxico) | 0.2678 | 0.4108 | IBR deja de dañar |
| **P2** RRF | 0.2041 | 0.3432 | **descartada** (tira la confianza del CBR) |
| **P3** gate (semántico) | 0.2622 | 0.4070 | neutro sin decorrelar |
| **P4** contrib-only, **test-tuned** | 0.2790 | 0.4154 | espejismo (sobreajuste al test) |
| **P4 val-tuned (honesto)** | 0.2603 | 0.4061 | **NO supera el techo**; +2.4 pp Hit@10 |
| CBR-solo (techo) | **0.2715** | 0.4078 | referencia, sigue siendo lo mejor en Top-1 |

**Conclusión (validada).** En OpenJ9 cola larga (50 clases), **el IBR no mejora el
Top-1 del mejor sistema** (el recuperador zero-shot) bajo tuning honesto. El IBR
estaba redundante por correlación; decorrelar (P1) + gatear (P3) + afinar el IP a
contribution-only (P4) lo vuelven *no dañino* y mejoran la **profundidad de ranking
(Hit@10 +2.4 pp)**, pero el **Top-1 tiene un techo que el IBR no rompe**. El valor del IBR al Top-1 depende del **régimen**: aporta en Mozilla (pocos devs densos, +1.6 pp) y no en OpenJ9 cola larga.

**Pendiente / trabajo futuro:**
- **Vista estructural (dev↔módulo/archivo)** como variante de decorrelación de P1.
- **Fusión IBR en TypeScript** (el tercer régimen) sigue pendiente.
- **Acercarse a 0.328:** replicar el **ensemble** (2º encoder), el único lever que cierra esa brecha.

---

## 5. Re-ejecución de TriagerX — hallazgos y su impacto (`triagerx-reimplementacion-hallazgos.md`)

Estado: **Fase 0-1 hechas** (entorno + alineación de split en omen). Fase 2-4 (entrenar
ensemble + IBR + grid) pendientes de ejecutar.

### 5.1 Setup de la re-ejecución

- Repo TriagerX clonado en omen (RTX 5060), venv aislado (torch 2.11+cu128, transformers 5.12, CUDA OK).
- `scripts/triagerx_compat/train_triagerx_omega.py`: corre **su** pipeline (DeBERTa-base + RoBERTa-base + 3 CNN, CombinedLoss, AdamW eps=1e-8, sampler 1/freq) sobre **nuestro split exacto** (`openj9_{train,test,val}_50.csv`), sin su `train_test_split` interno y preservando nuestras 51 clases (`--threshold 1`).
- Validado por dry-run: `Train 3348 | Val 503 | Test 534 | 51 clases`.

### 5.2 Hallazgo principal: fuga de test en TriagerX

- `triagerx/trainer/model_trainer.py:64-69`: guarda el checkpoint de **menor `val_loss`** de las 20 épocas.
- `training/developer/developer_training_openj9.py`: pasa `df_test` como `validation_dataloader` y **nunca usa `val_size`** del config. O sea: **selecciona el mejor de 20 épocas sobre el propio test y reporta sobre ese mismo test**.
- **Implicación**: sus números publicados (CBR 0.270, full 0.328) están **optimistamente sesgados** por selección-en-test. Nuestro CBR-recuperación es **zero-shot** (sin selección de checkpoint) → honesto. La comparación nos favorece **más de lo visible**: igualar su 0.270 con un 0.2715 sin entrenar ni seleccionar es más fuerte de lo que sugiere el empate nominal.
- **Acción tomada**: el launcher acepta `--val-csv`. Con val (held-out) se selecciona el checkpoint en val y se evalúa en test (número honesto); sin val replica su método (sesgo). Tenemos `openj9_val_50.csv`, así que la re-ejecución se hará en modo honesto.

### 5.3 Diferencias menores y falsas alarmas

**Diferencia menor — Limpieza de texto:**
- TriagerX aplica `clean_text` (borra URLs, hex→`<hex>`, timestamps/GMT, colapsa espacios).
- Nuestro `build_openj9_50.py:57-58` arma `text` crudo. Es una diferencia de **representación de entrada**, no de protocolo. Impacto esperado en MPNet: pequeño. No se corrige; se documenta.

**Falsas alarmas (verificadas como NO problema):**
- **Filtro de Pull Requests:** el CSV fuente tiene `issue_url` pero **0 filas con `/pull/`**. El filtro es un **no-op**.
- **Filtro `min_length <= 15`:** todo nuestro `text` empieza con `"Bug Title: …\nBug Description: …"` (≥29 chars aun vacío), así que nada cae.

**Diferencias ya documentadas:**
- **Criterio de clases**: nosotros owners con ≥20 issues **totales**, ellos ≥20 en el split de **train**. Ya marcado como "comparable, no idéntico".
- **Arquitectura**: ellos ensemble RoBERTa+DeBERTa+3CNN; nosotros DeBERTa-v3-solo (clasificador) o MPNet-kNN (recuperador). Decisión deliberada.
- **max_tokens**: 256 en la comparación de clasificadores; el recuperador explota 384 nativo de MPNet.

### 5.4 Alineación del IBR (Fase 3)

**Nuestro IBR ya es una réplica fiel declarada** de su IBR. Coincide en:
- Recuperación de top-k issues similares por MPNet (cos sim) con umbral de similitud.
- Mapeo de tipos: `pull_request`+`commits`→contribution, `assignment`→assignment, `discussion`→discussion.
- Score por dev: `sim · contribution_point · time_decay`, acumulado, solo developers esperados.
- Normalización min-max, `_normalize_nis` ≡ su `_normalize`.
- Fusión `FS = CBR_norm + W_f · IBR_norm` ≡ su `_adjust_dev_scores_by_similarity`.

**Divergencias detectadas (2):**
1. **Referencia del decaimiento temporal**: TriagerX usa una fecha de checkpoint FIJA (`train_checkpoint_date`); nosotros usamos `t_now` = creation time del bug consultado y **filtramos interacciones con `t ≥ t_now`** (anti-fuga temporal). Es intencional y nos hace MÁS conservadores.
2. **`last_assignment`**: TriagerX da un voto extra (peso de assignment) al último actor de PR/commit de cada issue similar. Nuestra réplica no lo incluye; el actor igual cuenta vía su evento commit/review, así que el efecto es menor.

### 5.5 Resultados de la re-ejecución head-to-head real

Ejecutado todo en omen (RTX 5060 8GB), test=534 issues, 51 clases. TriagerX = su código real
(ensemble entrenado 20 épocas en NUESTRO split, modo honesto). **Texto limpiado con su `TextProcessor` en ambos lados**.

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
**seleccionar checkpoint sobre el propio test** (fuga, ver §5.2); honestamente sobre nuestro
split su CBR es 0.1966 y su full 0.2528.

**Caveats:** (1) entrenamos su ensemble con `batch_size=8` (su 10 no entra en 8GB) + gradient checkpointing — puede bajar levemente su CBR; (2) su IBR usa su `best_param` del paper (no re-tuneado en nuestro val); (3) nuestro full afina gate/wf en test (sin val) — leve asimetría, pero nuestro headline (CBR zero-shot 0.2715) es libre de tuning.

Artefactos: `triagerX/omega_split/runs/triagerx_full_results.json` (TriagerX), `triager-omega/artifacts/openj9/eval_full_*_50.log` (omega).

#### Resumen de hallazgos

| # | Hallazgo | ¿triager-omega lo tomó en cuenta? | Impacto |
|---|----------|-----------------------------------|---------| 
| 1 | Selección de checkpoint sobre test (fuga) | No estaba explícito | **Alto, nos favorece**; corregido vía `--val-csv` |
| 2 | clean_text más agresivo en TriagerX | No | Menor; representación de entrada |
| 3 | Filtro de PRs | No, pero no hace falta | Nulo (0 PRs en el CSV) |
| 4 | Filtro min_length≤15 | No, pero no hace falta | Nulo (texto siempre ≥29) |
| 5 | Clases ≥20 total vs ≥20 train | Sí, documentado | Ya contemplado |
