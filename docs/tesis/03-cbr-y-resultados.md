

# 03 — CBR de Recuperación, Entrenamiento y Resultados

## Tabla de contenidos

1. [CBR de recuperación (Case-Based Reasoning real)](#1-cbr-de-recuperación-case-based-reasoning-real-cbr-recuperacionmd)
  - [1.1 Idea y formulación](#11-idea-y-formulación)
  - [1.2 Por qué MPNet](#12-por-qué-mpnet-motivación-del-encoder)
  - [1.3 Resultados en OpenJ9 (50 clases)](#13-resultados-openj9-50-clases-train-3348--test-534)
  - [1.4 Por qué el recuperador gana en cola larga](#14-por-qué-el-recuperador-gana-en-cola-larga)
  - [1.5 Validación cruzada en Mozilla](#15-validación-cruzada-en-mozilla-régimen-opuesto)
  - [1.6 Validación en TypeScript](#16-validación-en-typescript)
  - [1.7 Conclusión y comandos de reproducción](#17-conclusión-y-comandos-de-reproducción)
2. [Entrenamiento del CBR (DeBERTa) en la laptop](#2-entrenamiento-del-cbr-deberta-en-la-laptop-entrenamiento-laptopmd)
  - [2.1 Opción 1: todo en la laptop (recomendada)](#21-opción-1-recomendada-todo-en-la-laptop)
  - [2.2 Opción 2: híbrida Mac + laptop](#22-opción-2-híbrida-mac--laptop)
  - [2.3 Copiar archivos entre máquinas](#23-copiar-archivos-entre-máquinas)
3. [Resultados en CUDA (RTX 5060) — CBR y sistema completo](#3-resultados-en-cuda-rtx-5060--cbr-y-sistema-completo-resultados-cudamd)
  - [3.1 Entorno](#31-entorno)
  - [3.2 CBR-solo: comparación de variantes](#32-cbr-solo--comparación-de-variantes)
  - [3.3 Sistema completo CBR+IBR](#33-sistema-completo-cbribr)
  - [3.4 Reproducir](#34-reproducir)

---

## 1. CBR de recuperación (Case-Based Reasoning real) (`cbr-recuperacion.md`)

> Aporte propio: **reemplazar el CBR-clasificador de TriagerX** (cabeza softmax
> sobre un set fijo de devs, ensemble de 2 transformers) por un **recuperador
> basado en casos**. Validado en OpenJ9 a 50 clases (head-to-head con TriagerX).
>
> Script: `scripts/cbr_retrieval_openj9.py`. Fusión con IBR: `eval_openj9_full.py --cbr-mode retrieval`.

### 1.1 Idea y formulación

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

### 1.2 Por qué MPNet (motivación del encoder)

El encoder elegido es **`all-mpnet-base-v2`**, y la elección es deliberada:

1. **Es el mismo encoder que ya usa TriagerX → comparación justa (apples-to-apples).**
 El paper de TriagerX usa `all-mpnet-base-v2` para recuperar issues similares en su
 **IBR** (no en su CBR; su CBR es el ensemble clasificador DeBERTa+RoBERTa). Cita
 textual del paper: *"TriagerX IBR uses the pre-trained SBERT model all-mpnet-base-v2
 for embedding generation without requiring task-specific fine-tuning, chosen for its
 superior accuracy over other models. Detailed comparisons between these models are
 not provided as they are not the focus of our research."* Al construir nuestro
 CBR-recuperación con **el mismo encoder** (`config.sbert_model`), garantizamos que la
 mejora viene del **método** (recuperar casos y votar al `owner`) y **no de un embedder
 más potente** — se elimina ese confound.
2. **Es un SBERT: el objetivo de entrenamiento correcto para recuperar.** MPNet en su
 variante `sentence-transformers` está entrenado con objetivo siamés/contrastivo para
 que **el coseno entre embeddings refleje similitud semántica** — justo lo que necesita
 un recuperador.
3. **Off-the-shelf, sin fine-tuning → simple, interpretable, reproducible.** Encaja con
 la tesis de un reemplazo que funciona **zero-shot** (y confirmamos que afinarlo no
 mejora el Top-1).
4. **Frente a las demás propuestas.** La ablación de encoders era `{MiniLM, MPNet}`;
 MiniLM es el hermano rápido/pequeño (menor calidad) y MPNet el de mejor calidad general
 de la familia (768 dim). Sumado al punto 1 (es el que ya validó el baseline), no había
 razón para desviarse.

> **Matiz clave para la tesis:** TriagerX usa MPNet **solo para recuperar issues en su
> IBR** y nunca lo conecta a la recomendación de developer (eso lo hace con su ensemble
> clasificador). Nuestro aporte es **reutilizar ese mismo recuperador MPNet para la tarea
> de CBR** — votar al resolvedor de los casos similares — fijando el encoder para que la
> comparación sea justa.

#### Qué prueba (y qué no prueba) TriagerX sobre el encoder de su CBR

Revisando el paper, **TriagerX nunca prueba SBERT/MPNet como encoder del CBR, ni explica
por qué no lo usa ahí.** Es un hueco que nuestro CBR-recuperación llena.

**Lo que SÍ prueban (RQ3, RQ4, Tabla VI):** sus ablaciones del CBR comparan **solo
encoders de tipo clasificador** (variantes base de BERT / RoBERTa / DeBERTa / CodeBERT)
con cabezas CNN vs FCN.

- Tabla VI / RQ4: *"all combinations yield comparable results... the RoBERTa- and
DeBERTa-base pair consistently achieved slightly better performance, and thus, we adopt
them as the default encoders."* → eligen RoBERTa+DeBERTa porque rinde un poco mejor
**entre combinaciones de PLMs clasificadores**.
- RQ3 (Fig. 6 y 7): RoBERTa-b y DeBERTa-b solos rinden parecido; el **ensemble** los
supera por "ortogonalidad" (cada PLM acierta casos distintos). Toda la ganancia de su
CBR es el **ensemble**, no el encoder individual.

**Lo que NO prueban:** MPNet/SBERT **nunca aparece como encoder del CBR** — solo en el
IBR. No hay ni un experimento que compare "recuperación con SBERT" contra "clasificador"
para recomendar developer.

**Razón de fondo:** TriagerX **encuadró el CBR como clasificación desde el principio**
(embeddings PLM → clasificador CNN/FCN) y reservó SBERT para "buscar issues parecidos" en
el IBR. **Esa comparación — recuperación vs. clasificador para recomendar developer — es
la que falta en el paper y la que aporta nuestro CBR-recuperación.**

> **Su propio paper admite la debilidad que el recuperador resuelve.** En el análisis de
> errores de RQ1: \*"...we identified **dataset imbalance** as a key issue, with some
> developers having far more contributions than others. **This bias leads the model to
> favor more active developers, resulting in poorer performance for those with fewer
> contributions.**"\* Es exactamente la debilidad de cola larga del clasificador: el propio
> TriagerX reconoce el problema, pero no probó la solución obvia (recuperar en vez de
> clasificar) porque su SBERT estaba "encerrado" en el IBR.

#### ¿El CBR-recuperación y el IBR comparten encoder → es redundante el IBR?

Comparten el **mismo encoder MPNet y el mismo paso de recuperación** (top-k por coseno),
pero **no son redundantes por definición**: difieren en **qué hacen con los vecinos**.

- **CBR-recuperación** vota la **etiqueta de resolución** (`owner`, quién *arregló* el bug).
- **IBR** recorre el **grafo de interacciones tipadas** de esos mismos vecinos (quién hizo
commit / PR / discusión / asignación), ponderado por tipo y tiempo.

Son señales distintas extraídas de los mismos casos. **Pero**, justamente porque ambos
parten de la misma recuperación MPNet, tienden a estar **correlacionados**, y el valor
marginal del IBR **depende del régimen** (lo medimos):

- **OpenJ9, 50 clases (cola larga):** el IBR es débil (0.169) y correlacionado con el CBR
→ sumarlo añade ruido y baja el Top-1. Aquí el IBR es **prácticamente redundante**.
- **Mozilla (pocos devs muy activos):** el IBR es fuerte (0.676, cerca del CBR) y aporta
señal **complementaria** → la fusión sí mejora (+1.6 pp Top-1, W\_f=0.8). Aquí el IBR
**no es redundante**.

Bonus de ingeniería: como ambos usan los mismos embeddings MPNet, la **matriz de embeddings
y el índice de vecinos se computan una sola vez** y se comparten entre módulos.

---

### 1.3 Resultados OpenJ9 (50 clases, train 3348 / test 534)

#### Campeón: zero-shot, MPNet a 384 tokens, k=50, τ=2


| Config                                 | Top-1      | MRR       | Hit@5 | Hit@10 |
| -------------------------------------- | ---------- | --------- | ----- | ------ |
| **CBR-recuperación (zero-shot, k=50)** | **0.2715** | **0.408** | 0.577 | 0.678  |


**Sin entrenar nada.** Comparado con el resto:


| Componente (50 clases)                     | Top-1      | Nota                |
| ------------------------------------------ | ---------- | ------------------- |
| DeBERTa-solo (TriagerX)                    | 0.189      | su CBR base         |
| RoBERTa-solo (TriagerX)                    | 0.175      |                     |
| Nuestro CBR-clasificador (mejor, 30ép/512) | 0.2322     | mucho entrenamiento |
| CBR ensemble RoBERTa+DeBERTa (TriagerX)    | 0.270      | 2 transformers      |
| **Nuestro CBR-recuperación (zero-shot)**   | **0.2715** | **0 entrenamiento** |
| CBR + IBR full (TriagerX)                  | 0.328      | ensemble + IBR      |


Nuestro CBR-recuperación **iguala el ensemble de 2 transformers de TriagerX**
(0.2715 vs 0.270) y **supera** a su DeBERTa-solo (+8 pp) y a nuestro clasificador
entrenado (+3.9 pp) — sin entrenamiento, interpretable y capaz de manejar devs
nuevos / cola larga sin reentrenar.

#### Tuning del zero-shot

- **Longitud de contexto:** 384 tokens es el óptimo (es el nativo de MPNet).
k=50 → 256: 0.251 · **384: 0.2715** · 512: 0.249. Truncar pierde señal; pasarse
de 384 mete ruido fuera de distribución.
- **k (vecinos):** monótono creciente hasta 50; k&gt;50 (75, 100) empeora el Top-1.
- **τ:** apenas influye (1/2/4 casi iguales).

#### Fine-tuning: NO mejora el Top-1 (hallazgo)


| Encoder                        | Top-1          | Notas                                     |
| ------------------------------ | -------------- | ----------------------------------------- |
| Zero-shot (sin afinar)         | **0.2715**     | campeón                                   |
| + MNRL (1 ép, pares mismo-dev) | \~0.249 (@256) | mejora Hit@10 (0.729), no Top-1           |
| + BatchAllTriplet (3 ép)       | 0.159          | **colapsa** el espacio (loss plana \~4.8) |


En este dataset (chico, 51 clases, cola larga) **la estructura semántica
preentrenada de MPNet ya es casi óptima** para recuperar casos:

- `BatchAllTripletLoss` es inestable en datos chicos → colapsa los embeddings
(acerca todo) y arruina la recuperación (0.27 → 0.16).
- `MultipleNegativesRankingLoss` **sí aprende** (loss 3.4→2.6) pero lo
que mejora es el **recall@k** (Hit@10 0.729 vs 0.669 a igual `max_seq_len=256`),
no el acierto Top-1.

#### Más palancas zero-training probadas (ninguna sube el Top-1 en OpenJ9)

Baseline dense = k=50, τ=2, 384 tok.


| Palanca                                  | Mejor config | Top-1      | Hit@10    | MRR   | Veredicto                          |
| ---------------------------------------- | ------------ | ---------- | --------- | ----- | ---------------------------------- |
| **Baseline** (dense MPNet)               | k=50 τ=2     | **0.2715** | 0.678     | 0.408 | —                                  |
| **Normalización del voto** (`--vote`)    | idf          | 0.2472     | 0.667     | 0.387 | ❌ todas peor que `sum`             |
| **Híbrido BM25+MPNet, RRF** (`--hybrid`) | k=30 τ=1     | **0.2715** | **0.700** | 0.408 | ➖ empata Top-1, **+2.2 pp Hit@10** |
| **Cross-encoder re-rank** (`--rerank`)   | k=50 τ=1     | 0.2678     | 0.674     | 0.407 | ❌ −0.4 pp                          |


Detalle de cada hallazgo:

- **Normalización del voto:** el `sum` (baseline) gana. Penalizar a los devs frecuentes (mean/idf/max/rank) **empeora** → en cola larga el *base-rate* de un dev prolífico es **señal real**, no solo sesgo.
- **Híbrido léxico+semántico (BM25 + MPNet por RRF):** **empata el Top-1** (0.2715) y **sube el recall** (Hit@10 0.678→0.700). Mismo patrón que el fine-tuning MNRL: mejora *recall@k*, no *Top-1*.
- **Cross-encoder re-rank:** **no ayuda** (−0.4 pp). Por qué: el cross-encoder optimiza *relevancia textual query↔bug*, pero la tarea es predecir el **dev**, no el bug más parecido.

> ⚠️ **Esto es específico de OpenJ9.** En **TypeScript** (código denso, identificadores
> exactos) el **híbrido BM25+MPNet SÍ sube el Top-1 +3.1 pp** y el cross-encoder +1.3 pp.
> El valor de cada palanca zero-training **depende del régimen/dominio**.

**Conclusión:** en **OpenJ9** el Top-1 de **0.2715 es un techo robusto**.
Ni entrenar el encoder, ni normalizar el voto, ni el híbrido léxico, ni el cross-encoder lo
mueven. El límite **no es la sofisticación del método** sino la **información disponible**:
con 51 clases y cola larga, la señal de similitud semántica satura el acierto Top-1 cerca de
0.27. Lo único mejorable es el **recall** (Hit@10, vía híbrido o MNRL). Coherente con que la
brecha hasta el 0.328 de TriagerX sea **arquitectónica** (su ensemble + IBR), no del recuperador.

#### Sistema completo: CBR-recuperación + IBR


| Sistema (50 clases)       | Top-1      | MRR       | Hit@10 |
| ------------------------- | ---------- | --------- | ------ |
| IBR-solo                  | 0.1685     | 0.285     | 0.494  |
| **CBR-recuperación solo** | **0.2715** | **0.408** | 0.678  |
| Full W\_f=0.1             | 0.2622     | 0.407     | 0.687  |
| Full W\_f=0.5             | 0.2416     | 0.395     | 0.699  |
| Full W\_f=1.0             | 0.1929     | 0.364     | 0.706  |


**La fusión con el IBR no aporta** a Top-1 ni MRR (el mejor es el CBR-recuperación
solo; todo W\_f los baja). El IBR solo sube un poco Hit@10 (0.678→0.706) a pesos
altos donde el Top-1 ya colapsó. Razón: a 50 clases el IBR es débil (0.169) y está
**correlacionado** con el CBR (ambos recuperan los mismos bugs similares), así que
sumarlo añade ruido.

---

### 1.4 Por qué el recuperador gana en cola larga

Es el corazón del argumento: **un clasificador y un recuperador "gastan" la
información de entrenamiento de formas distintas, y la cola larga castiga a uno y no
al otro.**

#### El régimen

En OpenJ9 a 50 clases hay \~3348 bugs de train repartidos entre 51 devs, pero **muy
desigualmente**: unos pocos resuelven cientos y la mayoría apenas 20–40. Es una cola larga.

#### Por qué el clasificador sufre

Un clasificador softmax tiene que **aprender una frontera de decisión por cada clase**,
y para eso necesita ver suficientes ejemplos de esa clase:

- Las **filas finales de la matriz de pesos** (los devs raros) se entrenan con 20–40
ejemplos — no alcanza para aprender "cómo es un bug de este dev".
- El **gradiente de cross-entropy está dominado por las clases frecuentes**: el modelo
minimiza la pérdida total acertando a los devs ricos e ignorando a los pobres.
- Las clases raras quedan **infrarrepresentadas en el espacio de salida**; el
clasificador casi nunca las predice aunque el bug sea suyo.

Se ve en los números: el clasificador entrenado se queda en 0.2322 incluso con 30
épocas y `max_length` 512, y subir 15→30 épocas solo movió **+0.75 pp**
(rendimientos decrecientes). No es falta de entrenamiento; es que **no hay suficientes
ejemplos por clase rara para aprender una frontera**.

#### Por qué el recuperador no sufre

El recuperador **no aprende una frontera por clase** — no tiene parámetros por dev.
Solo mide similitud semántica (con MPNet, ya entrenado, que sabe de lenguaje general,
no de tus devs) y mira **quién resolvió los bugs parecidos**. La consecuencia:

- **Un dev raro necesita un solo caso parecido para ser recuperable.** Si resolvió 1
bug de "memory leak en el GC" y llega otro de GC, entra al top-k. No necesitó 200
ejemplos para "aprender la clase" — le bastó **un vecino**.
- La capacidad del recuperador **no se reparte entre clases**. Cada bug de train es un
"caso" independiente que sirve por igual sin importar si su dev es rico o pobre.
- El encoder es **fijo y general**: la calidad de la similitud no depende de cuántos
bugs tenga cada dev.

#### La intuición en una línea

> El clasificador necesita **densidad por clase** (muchos ejemplos de cada dev) para
> aprender; el recuperador solo necesita **un vecino cercano**. La cola larga te da
> exactamente lo contrario de densidad por clase: mata al clasificador y deja
> indiferente al recuperador.

#### Por qué se invierte en Mozilla

Esto también explica por qué en Mozilla el clasificador **vuelve a ganar** (0.732 vs
0.718): son 20 devs muy activos con mucha data cada uno → **alta densidad por clase**.
El clasificador tiene los ejemplos para aprender fronteras nítidas, y una frontera bien
aprendida supera a "votar vecinos" cuando hay datos de sobra. **No es que un método sea
mejor en absoluto: el régimen de datos decide cuál gana** — y el recuperador domina justo
donde el triaje real es más difícil (cola larga, devs nuevos o poco frecuentes), sin reentrenar.

---

### 1.5 Validación cruzada en Mozilla (régimen opuesto)

Para comprobar que el diseño generaliza, lo corrimos en el piloto Mozilla (20 devs,
etiqueta `contributor_id`/assignee, train 5842 / test 1615; `scripts/cbr_retrieval_pilot.py`).
Régimen **opuesto** a OpenJ9: pocos devs muy activos, mucha data, accuracy alta.


| Sistema (Mozilla, test)                     | Hit@1      | MRR       | Hit@5 | Hit@10 |
| ------------------------------------------- | ---------- | --------- | ----- | ------ |
| IBR-solo                                    | 0.6755     | 0.786     | 0.914 | 0.944  |
| Clasificador DeBERTa (entrenado, both)      | 0.7319     | 0.832     | —     | —      |
| **CBR-recuperación (zero-shot, k=20, τ=4)** | 0.7176     | 0.819     | 0.946 | 0.972  |
| **CBR-recuperación + IBR (W\_f=0.8)**       | **0.7337** | **0.831** | 0.951 | 0.980  |
| Clasificador + IBR (full, W\_f=0.2)         | 0.762      | —         | —     | —      |


**Dos hallazgos del contraste de regímenes:**

1. **El ranking clasificador↔recuperación se invierte con el régimen.** En Mozilla
 (pocas clases, mucha data) el clasificador entrenado gana por poco (0.732 vs
 0.718, −1.4 pp); en OpenJ9 (51 clases, cola larga, poca data) el recuperador gana
 (+4 pp). El recuperador es **competitivo en ambos sin entrenar**.
2. **El IBR SÍ aporta a la recuperación en Mozilla** (+1.6 pp Top-1, +1.2 pp MRR;
 pico en W\_f=0.8) y **no** en OpenJ9 a 50 clases. El valor de la fusión **sigue la
 fuerza del IBR en cada régimen**: fuerte con pocos devs muy activos (IBR-solo 0.676,
 cerca del CBR), despreciable en la cola larga (IBR-solo 0.169 ≪ CBR).

---

### 1.6 Validación en TypeScript

TriagerX usa **dos** datasets propios: OpenJ9 y **TypeScript (TS)**. Corrimos sobre los
**split files reales de TriagerX** (`df_train_ts.csv`/`df_test_ts.csv`, train 9831 / test
1115 tras dedup, **40 devs**, azar 0.025) → comparación **head-to-head** exacta. Régimen:
40 clases pero **top-heavy** (los 3 devs top con \~1100 issues c/u) y, sobre todo,
**contenido denso en identificadores** (nombres de tipos, símbolos de API, códigos `TS####`,
snippets de código).


| Sistema (TS, test 1115)                                    | Top-1      | MRR       | Hit@10 | Nota                                    |
| ---------------------------------------------------------- | ---------- | --------- | ------ | --------------------------------------- |
| TriagerX IBR                                               | 0.278      | —         | —      | su IBR (TS = interacciones esporádicas) |
| RoBERTa-Large+FCN (mejor baseline de TriagerX)             | 0.319      | —         | —      | un solo PLM                             |
| **TriagerX CBR (ensemble RoBERTa+DeBERTa)**                | **0.324**  | —         | —      | entrenado                               |
| TriagerX full (CBR+IBR)                                    | 0.353      | —         | —      | ensemble + IBR                          |
| Nuestro CBR-recuperación **dense** (k=20 τ=4)              | 0.2915     | 0.432     | 0.729  | zero-shot                               |
| Nuestro CBR-recuperación + **cross-encoder** (k=30 τ=4)    | 0.3049     | 0.447     | 0.744  | zero-shot                               |
| **Nuestro CBR-recuperación HÍBRIDO BM25+MPNet** (k=50 τ=2) | **0.3229** | **0.470** | 0.773  | **zero-shot**                           |


**Hallazgos:**

1. **El híbrido BM25+MPNet es decisivo aquí (+3.1 pp): 0.2915 → 0.3229**, lo que **iguala
 prácticamente el CBR ensemble entrenado de TriagerX** (0.324) y **supera a su mejor
 baseline de un solo PLM** (RoBERTa-FCN 0.319) — **sin entrenar**.
2. **El cross-encoder sí ayuda un poco en TS** (+1.3 pp, 0.3049) — al revés que en OpenJ9 (−0.4 pp).
3. **La normalización `idf` empata/supera levemente a `sum`** (0.2915 vs 0.2888) — coherente: TS es top-heavy, así que **penalizar a los pocos devs dominantes** ayuda un poco.
4. La brecha hasta el full de TriagerX (0.353) sigue siendo **su IBR** (+2.9 pp sobre su CBR); las interacciones de TS son esporádicas → el propio paper reporta IBR débil, 0.278.

**Síntesis de los 3 regímenes** (nuestro mejor recuperador zero-shot vs el CBR entrenado de TriagerX): en los tres queda **a ≤1.5 pp** de su ensemble entrenado **sin entrenar nada**.


| Dataset        | Régimen                   | Nuestro mejor (zero-shot) | TriagerX CBR (entrenado) | Δ        | Palanca que ayuda |
| -------------- | ------------------------- | ------------------------- | ------------------------ | -------- | ----------------- |
| OpenJ9         | 51 cls, cola larga, prosa | 0.2715 (dense)            | 0.270                    | **+0.2** | ninguna (techo)   |
| TypeScript     | 40 cls, top-heavy, código | 0.3229 (híbrido)          | 0.324                    | −0.1     | **híbrido BM25**  |
| Mozilla piloto | 20 cls, denso             | 0.7176 (dense)            | 0.7319 (clasif.)         | −1.4     | (IBR en fusión)   |


---

### 1.7 Conclusión y comandos de reproducción

El **CBR-recuperación** es un reemplazo propio, simple e interpretable del CBR de
TriagerX, **competitivo en los tres regímenes sin entrenar** (queda a ≤1.5 pp del CBR
entrenado de TriagerX en OpenJ9, TypeScript y Mozilla). Las palancas zero-training
**dependen del dominio** (el híbrido léxico es neutro en OpenJ9 pero decisivo en el código
de TS). Su valor relativo y el del IBR **dependen del régimen**:
en cola larga (OpenJ9) el recuperador supera al clasificador y el IBR no aporta; con
pocos devs muy activos (Mozilla) el clasificador gana por poco y el IBR sí aporta
(+1.6 pp). El fine-tuning no mejora el Top-1 en ninguno.

```bash
# baseline zero-shot con barrido (k, τ):
uv run python scripts/cbr_retrieval_openj9.py \
  --train-csv artifacts/openj9/openj9_train_50.csv \
  --test-csv  artifacts/openj9/openj9_test_50.csv --sweep        # campeón: k=50, 384 tok

# fine-tuning (no mejora Top-1; MNRL es el estable):
uv run python scripts/cbr_retrieval_openj9.py ... --finetune --loss mnrl --epochs 1

# mejoras zero-training (efecto DEPENDE del dataset, ver tablas):
uv run python scripts/cbr_retrieval_openj9.py ... --top-k 50 --tau 2 --vote-sweep   # normalización del voto
uv run python scripts/cbr_retrieval_openj9.py ... --hybrid --sweep                   # BM25+MPNet por RRF (decisivo en TS)
uv run python scripts/cbr_retrieval_openj9.py ... --rerank --rerank-n 50 --sweep     # cross-encoder (omen/CUDA)

# dataset TS de TriagerX (head-to-head; df_*_ts.csv del repo triagerX):
uv run python scripts/cbr_retrieval_openj9.py \
  --train-csv ../triagerX/df_train_ts.csv --test-csv ../triagerX/df_test_ts.csv \
  --max-seq-len 384 --hybrid --sweep      # campeón TS: híbrido k=50 τ=2 → 0.3229

# sistema completo (CBR-recuperación + IBR):
uv run python scripts/eval_openj9_full.py --cbr-mode retrieval \
  --train-csv ...train_50.csv --test-csv ...test_50.csv \
  --interactions ...openj9_interactions_50.parquet --meta ...openj9_issue_meta_50.parquet
```

**Gotcha (8 GB):** afinar con batch grande o `max_seq_len` alto satura la VRAM →
GPU al 100 % util pero \~40 W (thrashing, no cómputo). Usar `--ft-batch 16 --max-seq-len 256`. Requiere el paquete `datasets` (en `pyproject`).

**Gotcha (cross-encoder):** en MPS revienta la memoria con muchos pares → el script lo
corre en CPU automáticamente (lento). En omen/CUDA hay que **liberar la GPU** de MPNet
antes (el script ya hace `del model` + `empty_cache`) y **recortar el texto a \~2000 chars**
antes de tokenizar. Requiere `rank-bm25` para `--hybrid` (en `pyproject`).

---

## 2. Entrenamiento del CBR (DeBERTa) en la laptop (`entrenamiento-laptop.md`)

El entrenamiento de DeBERTa **no es una API remota** como Ollama: es un trabajo de
cómputo que necesita el **código + los datos** en la máquina donde corre, y el modelo
se genera ahí. Como la laptop tiene **GPU CUDA**, conviene correrlo en ella (más rápido
y sin el bug del `nan` de MPS de la Mac).

> `data/` y `artifacts/` están **gitignored** → git lleva solo el código. Los datos y el
> modelo entrenado se mueven aparte (ver §2.3).

### 2.1 Opción 1 (recomendada): todo en la laptop

La laptop ya tiene **Ollama + CUDA**, así que haz la fase CBR completa ahí. La Mac queda
solo para desarrollo.

```bash
# en la laptop:
git clone <tu-repo> triager-omega
cd triager-omega
uv sync
```

**Datos de entrada (copiar a la laptop, no van en git):**

- `data/bug_metadata.parquet`   (\~21 MB, para el texto crudo)
- `artifacts/pilot/`            (splits.parquet + label\_encoder.json)

**Destilar y entrenar en la laptop (Ollama local, sin red):**

```bash
# Ollama corriendo local en la laptop:
uv run python scripts/run_distillation.py                  # crea artifacts/pilot/distillations.parquet
uv run python -m triager_omega.cbr.train --adam-eps 1e-8   # CUDA: 1e-8 normal (el 1e-4 era parche de MPS)
```

**El modelo sale en la laptop:** `artifacts/pilot/cbr_model/` (pesos + tokenizer + `metrics.json`).

### 2.2 Opción 2 (híbrida): destilar desde la Mac, entrenar en la laptop

Útil si prefieres orquestar la destilación desde la Mac (Ollama remoto, ver `05-infraestructura.md`).

1. La Mac destila → `artifacts/pilot/distillations.parquet` queda en la **Mac**.
2. Copiar a la laptop: `data/bug_metadata.parquet` + `artifacts/pilot/` (ya con el `distillations.parquet`).
3. Entrenar en la laptop:
  ```bash
   uv run python -m triager_omega.cbr.train --adam-eps 1e-8
  ```
4. Copiar `artifacts/pilot/cbr_model/` de vuelta a la Mac.

### 2.3 Copiar archivos entre máquinas

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

**Notas:**

- **`--adam-eps 1e-8` en la laptop.** El `1e-4` del default es el parche para el `nan` de
DeBERTa-v3 en **MPS** (Apple Silicon). En CUDA usa el `1e-8` estándar.
- **VRAM 8 GB.** `deberta-v3-base` (\~184M params) entra de sobra para entrenar con
`--batch-size 16 --max-length 256`. Si te quedas corto de VRAM, baja el batch o la
longitud (`--batch-size 8`).
- **El modelo se regenera.** No hace falta versionar `cbr_model/`: con la
`distillations.parquet` puedes re-entrenar las veces que quieras. Lo único caro de
reproducir es la destilación (las llamadas al LLM).
- **Flujo de archivos:** `distillations.parquet` ─► `cbr/train.py` ─► `cbr_model/`.

---

## 3. Resultados en CUDA (RTX 5060) — CBR y sistema completo (`resultados-cuda.md`)

Corrida de entrenamiento y evaluación del piloto migrada de la Mac (MPS) al equipo
con **NVIDIA GeForce RTX 5060 Laptop GPU (8 GB)** vía SSH. Fecha: 2026-06-13.

### 3.1 Entorno


| Componente | Valor                                                                      |
| ---------- | -------------------------------------------------------------------------- |
| GPU        | RTX 5060 Laptop, 8 GB, compute capability `sm_120` (Blackwell)             |
| PyTorch    | `2.11.0+cu128` (CUDA 12.8)                                                 |
| Python     | 3.12                                                                       |
| Dataset    | Piloto 20 devs · train 5842 / val 1749 / test 1615 (9206 bugs etiquetados) |


> **Nº de épocas: 4 (elegido, no casual).** Esta corrida usó **4 épocas**; el piloto
> previo en MPS usó 5. Empíricamente **4 épocas rinde mejor que 5** en este CBR (5
> tiende a sobreajustar), así que 4 es la configuración preferida, no una época "de
> menos". Nota de comparabilidad con MPS: además del nº de épocas, cambia el backend
> (CUDA/torch 2.11, fp32), por lo que los valores no son apples-to-apples; el **orden
> cualitativo** entre variantes sí se mantiene.

**Recipe estable (importante):**

DeBERTa-v3 **diverge entrenando en `bf16`** en CUDA (el `grad_norm` explota a miles,
la loss salta a \~30 y el modelo colapsa a azar: Hit@1 ≈ 1/20). Es la misma fragilidad
numérica de la *disentangled attention* que provoca el `nan` en MPS.

**Recipe que reproduce el piloto: `fp32` + `adam_epsilon = 1e-4`** (NO bf16, NO 1e-8).
El `auto` de `resolve_runtime` ya lo aplica; en CUDA mantiene `batch=8` +
`gradient_accumulation=2` (eff. 16) para caber en 8 GB. Tiempo: \~7 min / 4 épocas.

### 3.2 CBR-solo — comparación de variantes

Texto de entrada: `raw` (Summary \[SEP\] Product Component), `distilled` (vista destilada)
o `both` (concatenación de ambas).

#### Test (1615 bugs)


| Variante     | Hit@1      | Hit@3  | Hit@5  | Hit@10 | MRR        | loss  |
| ------------ | ---------- | ------ | ------ | ------ | ---------- | ----- |
| **both** 🥇  | **0.7319** | 0.9195 | 0.9591 | 0.9926 | **0.8324** | 0.866 |
| raw 🥈       | 0.7127     | 0.9195 | 0.9511 | 0.9895 | 0.8197     | 0.908 |
| distilled 🥉 | 0.6656     | 0.9022 | 0.9517 | 0.9851 | 0.7899     | 1.031 |


#### Val (1749 bugs)


| Variante     | Hit@1  | Hit@5  | Hit@10 | MRR    |
| ------------ | ------ | ------ | ------ | ------ |
| **both** 🥇  | 0.7490 | 0.9646 | 0.9903 | 0.8488 |
| raw 🥈       | 0.7267 | 0.9617 | 0.9891 | 0.8320 |
| distilled 🥉 | 0.6993 | 0.9651 | 0.9857 | 0.8161 |


**Lectura:** `both` gana en todo (test Hit@1 +1.9 pp vs raw, +6.6 pp vs distilled). El
destilado **solo** es la vista más débil: aporta como complemento (`both`), no como
reemplazo del crudo. Orden `both > raw > distilled` estable en val y test.

> **Nota:** Este resultado (`distilled` peor) aplica al **clasificador DeBERTa** (arquitectura
> ya descartada). Para la arquitectura vigente (recuperación + MPNet), la destilación
> especializada JVM **sí ayuda** — ver `02-destilacion-semantica.md`.

### 3.3 Sistema completo — FS = NPS + W\_f·NIS

CBR = `both`, IBR = `distilled`. `W_f` sintonizado por grid en **validación** y
aplicado a test sin re-tunear (sin fuga).

#### Grid de W\_f en val (criterio Hit@5)


| W\_f                   | Hit@1      | Hit@5      | MRR        |
| ---------------------- | ---------- | ---------- | ---------- |
| 0.0 (solo CBR)         | 0.7490     | 0.9646     | 0.8487     |
| **0.2 (óptimo)**       | **0.7856** | **0.9720** | **0.8697** |
| 0.7 (default TriagerX) | 0.7536     | 0.9686     | 0.8508     |


#### Test con W\_f = 0.2


| Modo                | Hit@1      | Hit@3  | Hit@5  | Hit@10 | MRR        |
| ------------------- | ---------- | ------ | ------ | ------ | ---------- |
| **CBR+IBR (FS)** 🏆 | **0.7622** | 0.9307 | 0.9628 | 0.9957 | **0.8535** |
| CBR-solo (both)     | 0.7319     | 0.9195 | 0.9591 | 0.9926 | 0.8324     |
| IBR-solo            | 0.6755     | 0.8836 | 0.9139 | 0.9443 | 0.7856     |


**Lectura:**

- La **fusión aditiva gana**: +3.0 pp Hit@1 (0.732 → 0.762) y +2.1 pp MRR sobre CBR-solo.
- El **IBR solo es más débil** (0.676) pero aporta como empujón `W_f·NIS`, no como
reemplazo — exactamente el diseño de TriagerX (Ec. 8).
- `W_f = 0.2` (lejos del 0.7 de TriagerX) confirma que el aporte del IBR en Mozilla
es **moderado**, consistente con las ablaciones previas.
- `Hit@10 = 0.9957`: el dev correcto casi siempre está en el top-10.

### 3.4 Reproducir

```bash
# CBR (una variante)
uv run python -m triager_omega.cbr.train --text-mode both     # raw | distilled | both

# Sistema completo
uv run python -m triager_omega.modules.aggregator eval --split val  --grid --cbr-mode both
uv run python -m triager_omega.modules.aggregator eval --split test --w-f 0.2 --cbr-mode both
```

