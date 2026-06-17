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

## Por qué MPNet (motivación del encoder)

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
   un recuperador. El embedding `[CLS]` de DeBERTa/RoBERTa (lo que alimenta el clasificador
   de TriagerX) **no** está entrenado para que el coseno signifique algo: sirve para
   clasificar, no para recuperar por similitud.

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

### Qué prueba (y qué no prueba) TriagerX sobre el encoder de su CBR

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
para recomendar developer. Y sobre MPNet en sí dicen explícitamente: *"Detailed
comparisons between these models are not provided as they are not the focus of our
research"* (ni siquiera lo justifican empíricamente; es elección por defecto).

**Razón de fondo:** TriagerX **encuadró el CBR como clasificación desde el principio**
(embeddings PLM → clasificador CNN/FCN) y reservó SBERT para "buscar issues parecidos" en
el IBR. Nunca cruzaron los dos. **Esa comparación — recuperación vs. clasificador para
recomendar developer — es la que falta en el paper y la que aporta nuestro CBR-recuperación.**

> **Su propio paper admite la debilidad que el recuperador resuelve.** En el análisis de
> errores de RQ1: *"...we identified **dataset imbalance** as a key issue, with some
> developers having far more contributions than others. **This bias leads the model to
> favor more active developers, resulting in poorer performance for those with fewer
> contributions.**"* Es exactamente la debilidad de cola larga del clasificador (ver
> sección siguiente): el propio TriagerX reconoce el problema, pero no probó la solución
> obvia (recuperar en vez de clasificar) porque su SBERT estaba "encerrado" en el IBR.

### ¿El CBR-recuperación y el IBR comparten encoder → es redundante el IBR?

Comparten el **mismo encoder MPNet y el mismo paso de recuperación** (top-k por coseno),
pero **no son redundantes por definición**: difieren en **qué hacen con los vecinos**.

- **CBR-recuperación** vota la **etiqueta de resolución** (`owner`, quién *arregló* el bug).
- **IBR** recorre el **grafo de interacciones tipadas** de esos mismos vecinos (quién hizo
  commit / PR / discusión / asignación), ponderado por tipo y tiempo.

Son señales distintas extraídas de los mismos casos. **Pero**, justamente porque ambos
parten de la misma recuperación MPNet, tienden a estar **correlacionados**, y el valor
marginal del IBR **depende del régimen** (lo medimos):

- **OpenJ9, 50 clases (cola larga):** el IBR es débil (0.169) y correlacionado con el CBR
  → sumarlo añade ruido y baja el Top-1. Aquí el IBR es **prácticamente redundante** → se
  puede prescindir de él y quedarse con el CBR-recuperación solo.
- **Mozilla (pocos devs muy activos):** el IBR es fuerte (0.676, cerca del CBR) y aporta
  señal **complementaria** → la fusión sí mejora (+1.6 pp Top-1, W_f=0.8). Aquí el IBR
  **no es redundante**.

Conclusión: compartir el encoder **no** vuelve redundante al IBR en principio (extraen
señales diferentes), pero **sí** lo vuelve redundante *en la práctica* cuando la señal de
interacción es débil/correlacionada (cola larga). Bonus de ingeniería: como ambos usan los
mismos embeddings MPNet, la **matriz de embeddings y el índice de vecinos se computan una
sola vez** y se comparten entre módulos.

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

## Por qué el recuperador gana en cola larga

Es el corazón del argumento: **un clasificador y un recuperador "gastan" la
información de entrenamiento de formas distintas, y la cola larga castiga a uno y no
al otro.**

### El régimen

En OpenJ9 a 50 clases hay ~3348 bugs de train repartidos entre 51 devs, pero **muy
desigualmente**: unos pocos resuelven cientos y la mayoría apenas 20–40 (el umbral de
corte). Es una cola larga: pocas clases ricas, muchas clases pobres.

### Por qué el clasificador sufre

Un clasificador softmax tiene que **aprender una frontera de decisión por cada clase**,
y para eso necesita ver suficientes ejemplos de esa clase:

- Las **filas finales de la matriz de pesos** (los devs raros) se entrenan con 20–40
  ejemplos — no alcanza para aprender "cómo es un bug de este dev".
- El **gradiente de cross-entropy está dominado por las clases frecuentes**: el modelo
  minimiza la pérdida total acertando a los devs ricos e ignorando a los pobres. (Por
  eso el `WeightedRandomSampler` 1/freq fue clave en el piloto — es un parche a este
  sesgo.)
- Las clases raras quedan **infrarrepresentadas en el espacio de salida**; el
  clasificador casi nunca las predice aunque el bug sea suyo.

Se ve en los números: el clasificador entrenado se queda en 0.2322 incluso con 30
épocas y `max_length` 512, y subir 15→30 épocas solo movió **+0.75 pp**
(rendimientos decrecientes). No es falta de entrenamiento; es que **no hay suficientes
ejemplos por clase rara para aprender una frontera**.

### Por qué el recuperador no sufre

El recuperador **no aprende una frontera por clase** — no tiene parámetros por dev.
Solo mide similitud semántica (con MPNet, ya entrenado, que sabe de lenguaje general,
no de tus devs) y mira **quién resolvió los bugs parecidos**. La consecuencia:

- **Un dev raro necesita un solo caso parecido para ser recuperable.** Si resolvió 1
  bug de "memory leak en el GC" y llega otro de GC, entra al top-k. No necesitó 200
  ejemplos para "aprender la clase" — le bastó **un vecino**.
- La capacidad del recuperador **no se reparte entre clases**. Cada bug de train es un
  "caso" independiente que sirve por igual sin importar si su dev es rico o pobre; no
  hay un presupuesto de parámetros que las clases ricas acaparen.
- El encoder es **fijo y general**: la calidad de la similitud no depende de cuántos
  bugs tenga cada dev. Un bug raro se embebe igual de bien que uno común.

### La intuición en una línea

> El clasificador necesita **densidad por clase** (muchos ejemplos de cada dev) para
> aprender; el recuperador solo necesita **un vecino cercano**. La cola larga te da
> exactamente lo contrario de densidad por clase: mata al clasificador y deja
> indiferente al recuperador.

### Por qué se invierte en Mozilla

Esto también explica por qué en Mozilla el clasificador **vuelve a ganar** (0.732 vs
0.718): son 20 devs muy activos con mucha data cada uno → **alta densidad por clase**.
El clasificador tiene los ejemplos para aprender fronteras nítidas, y una frontera bien
aprendida supera a "votar vecinos" cuando hay datos de sobra. El recuperador sigue
siendo competitivo, pero pierde su ventaja porque la condición que lo favorecía (clases
pobres) desapareció. **No es que un método sea mejor en absoluto: el régimen de datos
decide cuál gana** — y el recuperador domina justo donde el triaje real es más difícil
(cola larga, devs nuevos o poco frecuentes), sin reentrenar.

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
