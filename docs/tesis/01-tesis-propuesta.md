<!-- Merged from: estado-tesis.md, ajustes-a-la-propuesta.md, diseno-corrida-oficial.md, presentacion-guion-10min.md -->

# 01 — Tesis: Estado, Propuesta y Corrida Oficial

## Tabla de contenidos

1. [Estado de la tesis — qué sostiene la evidencia](#1-estado-de-la-tesis--qué-sostiene-la-evidencia-estado-tesismd)
   - [1.1 La destilación (Módulo 2) NO es la contribución estrella](#11-la-destilación-módulo-2-no-es-la-contribución-estrella)
   - [1.2 Lo que SÍ quedó fuerte](#12-lo-que-sí-quedó-fuerte)
   - [1.3 Ranking de hallazgos por fuerza de evidencia](#13-ranking-de-hallazgos-por-fuerza-de-evidencia)
   - [1.4 Relato propuesto de la tesis](#14-relato-propuesto-de-la-tesis)
2. [Ajustes a la propuesta — evidencia y reencuadre](#2-ajustes-a-la-propuesta--evidencia-y-reencuadre-ajustes-a-la-propuestamd)
   - [2.1 Evidencia que motivó el cambio](#21-evidencia-que-motivó-el-cambio)
   - [2.2 Reencuadre del objetivo](#22-reencuadre-del-objetivo)
   - [2.3 Qué se conserva de la propuesta original](#23-qué-se-conserva-de-la-propuesta-original)
   - [2.4 Cifras de soporte](#24-cifras-de-soporte)
3. [Diseño de la corrida oficial (versión publicable)](#3-diseño-de-la-corrida-oficial-versión-publicable-diseno-corrida-oficialmd)
   - [3.1 Preguntas de investigación](#31-preguntas-de-investigación)
   - [3.2 Datasets](#32-datasets)
   - [3.3 Protocolo experimental](#33-protocolo-experimental)
   - [3.4 Condiciones a comparar](#34-condiciones-a-comparar)
   - [3.5 Métricas y rigor estadístico](#35-métricas-y-rigor-estadístico)
   - [3.6 Análisis, amenazas y reproducibilidad](#36-análisis-amenazas-y-reproducibilidad)
   - [3.7 Venue y resumen ejecutivo](#37-venue-y-resumen-ejecutivo)
4. [Guion de presentación (10 min) + banco de preguntas](#4-guion-de-presentación-10-min--banco-de-preguntas-presentacion-guion-10minmd)
   - [4.1 Guion](#41-guion-10-minutos)
   - [4.2 Banco de preguntas y respuestas](#42-banco-de-preguntas-y-respuestas)
   - [4.3 Chuleta](#43-chuleta)

---

## 1. Estado de la tesis — qué sostiene la evidencia (`estado-tesis.md`)

> **Corrección (2026-09-16):** §1.1 de abajo mide la destilación sobre el
> **clasificador DeBERTa, arquitectura ya descartada**. Con la arquitectura
> vigente (recuperación zero-shot + MPNet), la destilación sí ayuda de forma
> consistente — ver la sección §1 de `02-destilacion-semantica.md`
> (correccion-destilacion-deberta-descartado).

Documento de encuadre. Resume qué hipótesis del proyecto quedaron respaldadas
por los datos del piloto y cuáles no, y propone el relato defendible de la tesis.

Contexto: todo es **escala piloto** (Mozilla 20 devs / OpenJ9 17 devs) con un
**CBR deliberadamente recortado** (un solo DeBERTa, sin el ensemble de dos
transformers del paper TriagerX). Comparar absolutos contra TriagerX mide
"piloto vs. sistema entrenado a fondo", no "qué diseño es mejor".

---

### 1.1 La destilación (Módulo 2) NO es la contribución estrella

La hipótesis original era: *"raw + distilled entrena mejor el CBR → mejores
resultados"*. Los datos del piloto Mozilla (20 devs, train 5842 / val 1749 /
test 1615) no la respaldan limpiamente.

**CBR-solo en test** (Hit@1 / MRR):

| Modo | Hit@1 | MRR |
|---|---|---|
| **raw** | **0.7517** | 0.8440 |
| distilled | 0.7313 | 0.8347 ← **el peor** |
| both | 0.7492 | **0.8473** |

- **distilled-solo es el PEOR** → la destilación *por sí sola* empeora.
- **both NO le gana a raw en Hit@1** (0.7492 < 0.7517); solo gana en MRR por
  3 milésimas. En el CBR puro la destilación **no aporta**.

**Sistema completo CBR+IBR en test** (W_f óptimo en val por MRR):

| Sistema | Hit@1 | MRR |
|---|---|---|
| **both + IBR** (W_f=0.1) | **0.7604** | **0.8542** ← ganador |
| raw + IBR (W_f=0.4) | 0.7430 | 0.8473 |
| distilled + IBR (W_f=0.2) | 0.7412 | 0.8415 |

**Lectura honesta:** el aporte de la destilación es **marginal e indirecto** —
emerge solo en el sistema completo y es de menos de un punto. No es un fracaso
(la decisión de *concatenar* crudo+destilado en vez de *reemplazar* queda
confirmada: reemplazar perjudica), pero **no es el hallazgo central** que la
idea prometía.

#### Hipótesis no probada que podría rescatarla: la cola larga
Con 20 devs (la cabeza) el texto crudo ya es rico, así que destilar no agrega
señal. Con 450 devs y reportes pobres/escasos, la destilación podría rescatar a
los devs raros. **El piloto no prueba este escenario.** Trabajo futuro: medir
distilled vs both *solo en el subconjunto cola-larga* si se decide escalar.

---

### 1.2 Lo que SÍ quedó fuerte

1. **Arquitectura híbrida + IBR.** Full > CBR-solo y > IBR-solo en *ambos*
   datasets. El IBR aporta **+5.8 pp Top-1 idéntico** al que aporta en TriagerX
   (0.270→0.328 ellos; 0.199→0.257 nosotros) → réplica fiel.
2. **Dependencia de la etiqueta (hallazgo propio, no del paper).** La señal
   `contribution` del IBR **ayuda con etiqueta `fixer`/`owner`** (OpenJ9) y
   **daña con `assignee`** (Mozilla). Demostrado end-to-end en dos regímenes.

---

### 1.3 Ranking de hallazgos por fuerza de evidencia

| Hallazgo | Fuerza | Rol en la tesis |
|---|---|---|
| Arquitectura híbrida + IBR replicado | Fuerte | **Pilar** |
| Dependencia de la etiqueta (assignee vs fixer) | Fuerte / novedoso | **Pilar / aporte original** |
| Destilación (concatenar > reemplazar) | Débil / marginal | **Ablación medida**, no titular |

---

### 1.4 Relato propuesto de la tesis

**No reescribir la tesis alrededor de la destilación.** Recolocar el peso:

- **Tesis = sistema híbrido de triaje (contenido + interacción) sin grafos**,
  validado a escala piloto en dos repos de regímenes opuestos (Mozilla/Bugzilla
  con assignee; OpenJ9/GitHub con fixer).
- **Aporte 1 (metodológico):** replicación fiel del IBR de TriagerX y de la
  arquitectura híbrida, con ganancia cuantificada y reproducible.
- **Aporte 2 (original):** el valor de la señal de código depende de la
  **definición de la etiqueta** — resultado que matiza/extiende a TriagerX.
- **Aporte 3 (ablación):** la destilación LLM como *aumento concatenado* (no
  reemplazo) da mejora pequeña pero consistente en el sistema completo; se
  hipotetiza mayor efecto en la cola larga (trabajo futuro).
- **Encuadre de los absolutos:** piloto vs. sistema full; la brecha con TriagerX
  es casi enteramente el CBR ensemble (2 transformers) que no entrenamos. No es
  una derrota de diseño.

#### Decisión pendiente del autor
¿La destilación era la contribución **principal** a defender, o **una de
varias**? De eso depende si basta el encuadre de arriba (B-de-varias) o si hay
que invertir en escalar para probar la hipótesis de cola larga (A-principal).

---

## 2. Ajustes a la propuesta — evidencia y reencuadre (`ajustes-a-la-propuesta.md`)

> Sección lista para integrar en el documento de tesis (apartado de metodología o al
> inicio de resultados). Explica, con evidencia del piloto, por qué la propuesta se
> refinó de *mejorar el clasificador de contenido* a *reemplazarlo por recuperación de
> casos*, y por qué el objetivo pasó de *superar* a *competir con menos + caracterizar
> regímenes*. Todas las cifras son Top-1 (Hit@1) en test.

La propuesta inicial planteaba **replicar y mejorar el componente de contenido (CBR)**
del sistema de referencia (TriagerX) mediante un **clasificador transformer afinado**,
fusionado con un recomendador por interacción (IBR). Los experimentos piloto
condujeron a **refinar la propuesta** hacia un **CBR basado en recuperación de casos
(Case-Based Reasoning)** y a **reencuadrar el objetivo**. El cambio no responde a una
preferencia, sino a una **cadena de hallazgos empíricos** que se detalla a
continuación. Reportar este ajuste de forma explícita es parte del método: el piloto
existe precisamente para validar el diseño antes de la corrida a escala.

### 2.1 Evidencia que motivó el cambio

1. **El modelo base no era la debilidad.** En condiciones comparables (un solo
   transformer, mismas clases y partición), nuestro clasificador DeBERTa **igualó o
   superó** al modelo base de la referencia (0.225–0.232 frente a 0.189). Esto descarta
   la hipótesis de que el bajo rendimiento provenía del modelo de contenido.

2. **El entrenamiento no era el cuello de botella.** Aumentar el entrenamiento
   (de 15 a 30 épocas y la longitud de contexto de 256 a 512 tokens) mejoró el Top-1
   apenas **+0.75 pp**. La mejora por entrenamiento mostró **rendimientos decrecientes**:
   seguir por esa vía no era productivo.

3. **La brecha restante era arquitectónica, no de calidad.** La diferencia hasta el
   sistema completo de la referencia se explica por su **ensemble de dos transformers**
   (su CBR de dos modelos ya alcanza 0.270, más que un único transformer por bien
   entrenado que esté). Replicar ese ensemble habría sido reproducir su arquitectura
   sin aportar una propuesta propia.

4. **Decisión de diseño.** En lugar de replicar un ensemble costoso, se planteó si un
   **paradigma distinto de CBR —la recuperación de casos— podía alcanzar un
   rendimiento comparable de forma más simple e interpretable.** Conceptualmente, el
   término CBR (*Case-Based Reasoning*) describe razonar por casos similares, no
   clasificar; el sistema de referencia, pese al nombre, emplea un clasificador.

5. **Validación del ajuste.** El CBR de recuperación (recuperar los casos pasados más
   similares y votar por su resolvedor, con un codificador **congelado, sin
   entrenamiento**) **igualó al ensemble de dos transformers** (0.2715 frente a 0.270)
   y superó tanto al modelo base de la referencia (0.189) como a nuestro propio
   clasificador entrenado (0.232). El cambio quedó así respaldado por evidencia.

### 2.2 Reencuadre del objetivo

Como consecuencia, el objetivo se ajustó de **"superar al sistema de referencia"** a
**"alcanzar un rendimiento competitivo con una arquitectura más simple e interpretable,
y caracterizar en qué régimen de datos aporta cada señal (contenido frente a
interacción)"**. Justificación: superar en términos absolutos exigiría replicar el
ensemble de la referencia —sin novedad propia—, mientras que el eje de **simplicidad
(cero entrenamiento), interpretabilidad y análisis de regímenes** constituye una
contribución original. Este reencuadre convierte hallazgos como "el ajuste fino no
mejora el Top-1" o "la señal de interacción solo aporta en ciertos regímenes" de
aparentes limitaciones en **resultados del estudio**.

### 2.3 Qué se conserva de la propuesta original

El esqueleto del sistema **no cambia**: sigue siendo híbrido (contenido + interacción)
con fusión aditiva `FS = NPS + W_f·NIS`, el IBR con interacciones tipadas, la
partición temporal y el directorio activo de candidatos. **Solo se reemplaza el
componente de contenido** (clasificador → recuperación de casos). Todo el desarrollo
del piloto (datos, scripts, infraestructura, hallazgos) se reutiliza.

> «La propuesta inicial planteaba replicar y mejorar el componente de contenido (CBR)
> de TriagerX mediante un clasificador afinado. Los experimentos piloto mostraron que
> (i) nuestro clasificador igualaba al modelo base de la referencia, pero (ii) su
> mejora por entrenamiento era marginal y la diferencia con el sistema completo era
> atribuible a su ensemble de dos transformers. A partir de esta evidencia, la
> propuesta se refinó hacia un CBR basado en recuperación de casos, que alcanza un
> rendimiento comparable sin entrenamiento y permite analizar en qué régimen aporta
> cada señal. En consecuencia, el objetivo se reencuadró de superar la referencia a
> lograr resultados competitivos con una arquitectura más simple e interpretable y
> caracterizar la dependencia del régimen. Este documento reporta la propuesta
> refinada y su validación.»

### 2.4 Cifras de soporte

| Hallazgo | Cifra (Top-1) |
|---|---|
| Clasificador nuestro (1 transformer) vs. base de la referencia | 0.225–0.232 vs. 0.189 |
| Mejora por más entrenamiento (15→30 ép, 256→512 tok) | +0.75 pp |
| Ensemble de 2 transformers (referencia) | 0.270 |
| **CBR de recuperación (nuestro, sin entrenar)** | **0.2715** |
| Sistema completo de la referencia (ensemble + interacción) | 0.328 |

---

## 3. Diseño de la corrida oficial (versión publicable) (`diseno-corrida-oficial.md`)

> Estado: el trabajo actual es un **piloto / proof-of-concept** que validó el diseño
> en 2 regímenes (OpenJ9 cola larga, Mozilla pocos devs). Este documento fija las
> decisiones de la **corrida oficial** para que el resultado salga directo en formato
> de **estudio empírico** publicable (no "método nuevo").
>
> Regla de oro: el aporte no es "mi CBR gana", es **"caracterizar cuándo basta la
> señal de contenido y cuándo aporta la de interacción en triaging"**. La recuperación
> kNN es un baseline conocido → como método no pasa revisión; como estudio a escala, sí.

### 3.1 Preguntas de investigación

- **RQ1 — Contenido sin entrenar.** ¿Un CBR de recuperación (kNN sobre embeddings
  congelados + voto al resolvedor) iguala a un clasificador entrenado / a un ensemble
  de transformers en triaging? ¿En qué condiciones?
- **RQ2 — Régimen.** ¿Cómo depende el rendimiento relativo (recuperación vs.
  clasificador) de las propiedades del proyecto (nº de devs, tamaño, longitud de cola,
  densidad de datos)?
- **RQ3 — Señal de interacción.** ¿Cuándo el IBR aporta sobre el CBR y cuándo no?
  ¿El aporte se predice por la fuerza del IBR-solo / por el régimen?
- **RQ4 — Fine-tuning.** ¿Afinar el encoder (contrastivo) mejora la recuperación, o
  la estructura preentrenada ya es el techo? ¿Depende del tamaño de datos?
- **RQ5 — Costo.** ¿Cuál es el trade-off rendimiento/costo (entrenamiento, latencia,
  memoria) entre recuperación, clasificador y ensemble?

Hipótesis del piloto a confirmar: H1 recuperación ≈ ensemble en cola larga; H2 el
ranking se invierte con el régimen; H3 el IBR aporta solo cuando IBR-solo es fuerte;
H4 el fine-tuning no mejora Top-1 en datos chicos.

### 3.2 Datasets

- **≥ 5–8 proyectos** de distinto régimen (no 2). Candidatos: Eclipse, Mozilla Core
  + Firefox, OpenJ9, GCC, LLVM, VS Code, Kubernetes, o los benchmarks de triaging ya
  publicados (Bugzilla/GitHub). Reusar los splits de papers previos donde existan
  (comparabilidad).
- Cubrir el **eje de régimen** a propósito: proyectos con pocos devs muy activos **y**
  proyectos de cola larga (cientos de devs), para responder RQ2 con variación real.
- Por proyecto reportar: nº bugs, nº devs (clases), % cola (devs con < N bugs),
  azar (1/clases), periodo temporal.
- Definir formalmente la **etiqueta** (assignee vs. owner/fixer) y ser consistente;
  documentar que el régimen de etiqueta cambia qué señal aporta (hallazgo del piloto).

### 3.3 Protocolo experimental

- **Split temporal estricto** por fecha (no aleatorio): train = pasado, test = futuro.
  Validación temporal intermedia para tunear (k, τ, W_f) **sin tocar test**.
- **Directorio activo (candidate-constrained):** definir el conjunto de devs candidatos
  por proyecto (p. ej. activos en ventana reciente) y evaluar dentro de él.
- **Sin fuga:** índice de recuperación = solo train; el bug de consulta nunca se
  recupera a sí mismo.
- Congelar versiones (modelos, librerías) y semillas.

### 3.4 Condiciones a comparar

(Mismas clases, mismo split, por proyecto)

| Familia | Condición |
|---|---|
| Baselines | azar · frecuencia (most-active dev) · TF-IDF + kNN (baseline clásico) |
| Contenido | **CBR-recuperación zero-shot** (nuestro foco) · CBR-recuperación afinado (MNRL) · clasificador 1 transformer · ensemble 2 transformers (réplica TriagerX) |
| Interacción | IBR-solo (interacciones tipadas) |
| Fusión | CBR + IBR (barrido W_f en validación) para cada CBR |

Incluir TF-IDF+kNN es clave: deja claro qué aporta el embedding semántico sobre el
baseline de similitud clásico (refuerza el related work y la honestidad).

### 3.5 Métricas y rigor estadístico

**Métricas de calidad:** Hit@{1,3,5,10}, MRR, y MAP. Reportar por proyecto y agregado.

**Costo (primera clase, RQ5):** tiempo de entrenamiento (recuperación = 0), latencia
de inferencia por bug, memoria/VRAM, tamaño del modelo. Tabla rendimiento-vs-costo.

**Rigor estadístico (sin esto no hay paper):**
- **Intervalos de confianza por bootstrap** sobre el test (p. ej. 1000 remuestreos)
  en toda métrica reportada.
- **Tests de significancia** entre condiciones (Wilcoxon pareado sobre bugs, o
  bootstrap de la diferencia); corrección por comparaciones múltiples.
- **Varias semillas** donde haya aleatoriedad (fine-tuning, sampler) → media ± std.
- Tamaño de efecto, no solo p-valor.

### 3.6 Análisis, amenazas y reproducibilidad

**Análisis (lo que convierte resultados en contribución):**
- **Caracterización del régimen (RQ2):** correlacionar el delta (recuperación −
  clasificador) y el aporte del IBR con métricas del proyecto (nº devs, % cola,
  densidad). Idealmente un gráfico delta vs. régimen que muestre la inversión.
- **Análisis de errores:** dónde falla cada enfoque (devs raros, bugs cortos, etc.).
- **Interpretabilidad (ventaja del CBR de casos):** ejemplos "se recomienda a X
  porque este bug se parece a #1234, #998 que X resolvió" — TriagerX no ofrece esto.
- **Ablaciones:** encoder (MPNet vs. otros), k, τ, decaimiento temporal, tipos de
  interacción (contribution/assignment/discussion).

**Amenazas a la validez (sección obligatoria):**
- Externa: selección de proyectos; generaliza solo a OSS con historial.
- Constructo: etiqueta = quien arregló ≠ "mejor" triador; ruido de identidad.
- Interna: tuning en validación temporal; posible fuga por re-aperturas.
- Conclusión: dependencia de un único encoder/familia si no se ablaciona.

**Reproducibilidad:**
- Liberar scripts (ya existen: `cbr_retrieval_*`, `eval_openj9_full --cbr-mode
  retrieval`, mineros), splits, semillas, versiones y un README de reproducción.
- Paquete de réplica (datos derivados o instrucciones de minería).

### 3.7 Venue y resumen ejecutivo

- **Workshop / short** (alcanzable pronto): RQ1+RQ3 con IC, 3-4 proyectos.
- **Journal medio (IST / JSS) o replication track** (objetivo real): las 5 RQ,
  5-8 proyectos, estadística completa, costo, análisis de errores e interpretabilidad.
- Checklist mínimo antes de enviar: IC en todas las tablas ✔ · ≥5 proyectos ✔ ·
  baseline TF-IDF+kNN ✔ · tabla de costo ✔ · gráfico de régimen ✔ · amenazas ✔.

**Qué se reutiliza del piloto (no se tira nada):**
- Scripts: `scripts/cbr_retrieval_openj9.py`, `scripts/cbr_retrieval_pilot.py`,
  `scripts/eval_openj9_full.py --cbr-mode retrieval`, mineros de timelines.
- Hallazgos del piloto = hipótesis preregistradas de la corrida oficial.
- Infra de cómputo (omen RTX 5060) para el clasificador/ensemble y el fine-tuning;
  la recuperación zero-shot corre hasta en CPU/MPS.

**Resumen ejecutivo:** La corrida oficial **no cambia el método** del piloto; cambia **escala + estadística +
encuadre**. Si se entra a ella con (a) 5-8 proyectos de regímenes variados, (b)
bootstrap/IC y tests desde la primera tabla, (c) baseline TF-IDF+kNN, (d) tabla de
costo y (e) la narrativa de estudio empírico ("cuándo contenido, cuándo interacción"),
el resultado es publicable. Como método nuevo, no — y eso no se arregla escalando.

---

## 4. Guion de presentación (10 min) + banco de preguntas (`presentacion-guion-10min.md`)

> Defensa de **propuesta de tesis** — *triager-omega*: un *framework* híbrido de triaje de
> incidencias que combina contenido e interacciones históricas, con un **recomendador de
> contenido por recuperación** (CBR) que no se entrena.
> Alineado diapositiva por diapositiva con `oficuo/src/presentacion.tex`.
> Ritmo objetivo ~130 palabras/min.

### 4.1 Guion (10 minutos)

#### [0:00 – 0:45] Apertura

> [Diapositiva: Portada — título, autor, asesor]

Buenos días. Mi nombre es Sebastián Mendoza y presento mi propuesta de tesis:
*Modelos de Aprendizaje Profundo para el Triaje de Incidencias de Software*, asesorada
por el profesor Edgar Sarmiento. En los próximos diez minutos voy a plantear el
**problema**, el **estado del arte**, los **datos**, la **propuesta de sistema** y la
**estrategia de evaluación** con que pienso validarla.

---

#### [0:45 – 2:00] El problema del triaje

> [Diapositiva: El problema del triaje de incidencias]

El **triaje** consiste en asignar cada reporte de incidencia al desarrollador adecuado
para resolverlo. Hoy se hace en gran medida de forma **manual**: es costoso y depende
del juicio de unos pocos expertos. Una consecuencia conocida es el **rebote** o *bug
tossing* —cadenas de reasignación—; alrededor de una cuarta parte de los reportes se
reasigna al menos una vez.

Y los datos imponen tres desafíos que marcan toda la propuesta:

- La **cola larga**: unos pocos desarrolladores resuelven la mayoría de los bugs, y la
  gran masa restante tiene muy poco historial.
- El **arranque en frío**: desarrolladores nuevos o con casi ningún caso previo.
- El **texto ruidoso**: prosa mezclada con código, trazas y *logs*.

Estos tres puntos son los que un sistema de triaje realista tiene que enfrentar.

---

#### [2:00 – 3:00] Motivación y propuesta

> [Diapositiva: Motivación y propuesta — dos columnas]

El enfoque habitual trata el triaje como **clasificación supervisada** sobre el conjunto
de desarrolladores. El problema práctico es que, cada vez que cambia el equipo —alguien
entra o se va—, hay que **reentrenar** el modelo.

Mi propuesta cambia el marco: tratar el triaje como **recuperación de casos similares**
—paradigma *Case-Based Reasoning*, CBR— **sin entrenar** el recomendador de contenido.
Sobre esa base:

- Combino una señal de **contenido** con una de **interacciones históricas**.
- Uso un modelo de lenguaje solo como **preprocesamiento** del texto, no como
  clasificador.
- La adaptación al proyecto se hace **actualizando la base de casos**, no los parámetros
  del modelo.

---

#### [3:00 – 3:45] Objetivos

> [Diapositiva: Objetivos]

El objetivo general es proponer un *framework* híbrido que integre tres piezas: la
**destilación semántica** del reporte con un LLM, la **recuperación** de reportes
históricos similares con codificadores **sin ajuste fino**, y el análisis de
**interacciones históricas** —todo con una arquitectura simple e interpretable—, y además
**caracterizar el régimen de datos** en que aporta cada señal.

De ahí salen los objetivos específicos: destilar el reporte en frases clave con Gemma 4;
recuperar casos con un codificador congelado y votación de vecinos; mitigar la cola larga
con voto ponderado y un directorio activo; modelar interacciones tipadas y fusionarlas con
*gating*; validar las recomendaciones con decodificación restringida; y caracterizar el
contraste entre etiqueta de *assignee* y de *fixer*.

---

#### [3:45 – 5:00] Estado del arte

> [Diapositiva: Tres paradigmas en la literatura + TriagerX]

La literatura se organiza en tres paradigmas: **aprendizaje profundo y modelos de
lenguaje**, que aportan comprensión semántica del texto; **grafos y bases de
conocimiento**, que modelan las interacciones entre desarrolladores, reportes y
componentes; y **optimización estocástica y ML clásico**.

Mi referencia metodológica es **TriagerX**: combina un clasificador de contenido —DeBERTa
más RoBERTa— con un módulo de interacciones basado en SBERT y una fusión ponderada. Es el
diseño más completo y validado industrialmente, así que lo adopto como punto de
comparación.

No las leeré una a una; el mensaje es que el campo está dominado por modelos cada vez más
pesados y específicos, y que casi todos comparten dos limitaciones: **dependen de
reentrenamiento** y **rinden mal en la cola larga**. Ahí es donde mi propuesta busca
diferenciarse.

---

#### [5:00 – 5:45] Conjuntos de datos

> [Diapositiva: Conjuntos de datos — tabla]

Para que la evaluación sea sólida, uso **tres conjuntos** que cubren dos plataformas y dos
regímenes de etiqueta:

- **BugsRepo (Mozilla)**, de Bugzilla, etiquetado por *Assigned To* —régimen *assignee*—,
  con partición temporal y restricción a un directorio de desarrolladores activos.
- **OpenJ9**, de GitHub, etiquetado por *owner* —régimen *fixer*—; es el dataset que
  permite la comparación directa con la referencia.
- **TypeScript**, también de GitHub y régimen *fixer*, con reportes muy densos en código.

Esa diversidad —dos plataformas, dos definiciones de etiqueta— es deliberada: quiero ver
**cuándo** aporta cada señal, no solo un promedio.

---

#### [5:45 – 6:30] Visión general y arquitectura

> [Diapositiva: Visión general del sistema]

El sistema es una **tubería de cuatro etapas**, del reporte a la lista de candidatos:

1. **Destilación semántica**: reescribir y limpiar el texto con un LLM.
2. **Recomendador de contenido (CBR)**: recuperar reportes similares, produciendo una
   puntuación NPS.
3. **Recomendador por interacciones (IBR)**: participación histórica tipada, produciendo
   una puntuación NIS.
4. **Fusión con compuerta** y restricción al directorio activo.

La idea central, y lo que distingue la propuesta: **el recomendador de contenido no se
entrena**; usa un codificador congelado y una base de casos actualizable.

---

#### [6:30 – 7:10] Destilación semántica

Primera etapa. El texto crudo de un bug mezcla prosa, código y trazas; eso **degrada al
codificador** y obliga a truncar. Para limpiarlo uso un modelo de instrucción —**Gemma 4
E4B IT**, cuantizado y ejecutado localmente— que reescribe el reporte aislando tres cosas:
los **síntomas**, la **ubicación** en el sistema y los **requerimientos** asociados.

Lo importante conceptualmente: es una transformación de los **datos**, no una compresión
del modelo. Reduce ruido y truncamiento antes de recuperar.

---

#### [7:10 – 7:50] Recomendador de contenido por recuperación

Segunda etapa, el corazón de la propuesta. Uso un codificador de oraciones **congelado**
—`all-mpnet-base-v2`— y mido similitud por coseno. Recupero los **k reportes históricos
más similares** —un kNN— y agrego el voto de quienes los resolvieron, normalizado, para
obtener la puntuación **NPS**.

Dos refinamientos pensados para la cola larga: una **agregación robusta** del voto
—normalizar por frecuencia, voto máximo, por rango— y, opcionalmente, un **canal léxico
BM25** que rescata los *tokens* técnicos exactos que la similitud semántica difumina.

---

#### [7:50 – 8:30] Recomendador por interacciones históricas

Tercera etapa. Sobre esos mismos incidentes similares, ahora miro **cómo** participó cada
candidato. Las interacciones son **tipadas**, con un peso —un **Punto de Interacción**— por
tipo: *commit*, *review*, *assignment* y *discussion*. Se agregan ponderadas por similitud
y se normalizan en la puntuación **NIS**.

Como opciones para mejorar la señal: recuperar los vecinos por **BM25**, para decorrelar
este módulo del de contenido, y un **decaimiento temporal** que favorezca la actividad
reciente.

---

#### [8:30 – 9:10] Fusión de señales con compuerta

Cuarta etapa: cómo combino ambas señales. La fusión es **aditiva**:
**FS = NPS + W_f · NIS**. Sumo **puntuaciones, no rangos**, justamente para preservar la
**confianza calibrada** del recomendador de contenido.

Y le añado una **compuerta** o *gating*: el IBR interviene **solo cuando el contenido
duda** —cuando el margen entre el primer y el segundo candidato es estrecho—; si el
contenido está confiado, se respeta su decisión. Finalmente, toda recomendación se
restringe a un **directorio de desarrolladores activos**: garantiza viabilidad, contiene
la cola larga y evita salidas inviables.

---

#### [9:10 – 9:50] Estrategia de evaluación

Para validar el *framework* propongo: métricas de *ranking* —**Hit@K** y **MRR**—, que es
como se usa un recomendador en la práctica; **partición temporal**, entrenando con el
pasado y evaluando con el futuro para evitar fuga de información; y, sobre todo, una
**caracterización por régimen de la etiqueta**. La hipótesis es que con *fixer* la señal de
contribución —los *commits*— está **alineada** y el IBR aporta, mientras que con *assignee*
esa señal puede **desalinearse**. Caracterizar ese contraste es parte central del aporte.

---

#### [9:50 – 10:00] Cierre

En resumen, propongo un triaje híbrido **simple, interpretable y sin reentrenar el
recomendador de contenido**, que enfrenta de cara la cola larga y el arranque en frío, y
que se validará en tres regímenes de datos distintos. Muchas gracias; quedo atento a sus
preguntas.

---

### 4.2 Banco de preguntas y respuestas

> Ordenadas de más probables a más técnicas. Cada una con respuesta corta lista para decir.
> Adaptadas a una defensa de **propuesta** (qué se hará y por qué), no de resultados finales.

#### Sobre el encuadre

**P: ¿Por qué recuperación de casos en vez de clasificación, que es lo habitual?**
R: Por dos razones. Una **práctica**: un clasificador hay que reentrenarlo cada vez que
cambia el equipo; un recuperador se adapta actualizando la base de casos. Y una **de
régimen de datos**: el clasificador necesita muchos ejemplos por desarrollador para
aprender una frontera, y en la cola larga no los hay; al recuperador le basta **un solo
caso parecido** para que un dev raro sea recomendable. Por eso espero que domine justo
donde el triaje es más difícil.

**P: ¿Por qué adoptar TriagerX como referencia?**
R: Porque es el diseño híbrido más completo y validado industrialmente: combina contenido
(DeBERTa+RoBERTa) e interacciones (SBERT) con fusión ponderada. Me da un punto de
comparación serio y una arquitectura de dos señales sobre la que construir mi alternativa
sin entrenamiento.

#### Sobre la destilación

**P: ¿Por qué Gemma 4 y no afinar un modelo propio?**
R: Porque la destilación aquí es **preprocesamiento de datos**, no clasificación: solo
necesito un buen modelo de instrucción que limpie el reporte. Gemma 4 E4B IT cuantizado
corre **localmente**, sin coste de API ni de entrenamiento, y eso encaja con la filosofía
de un sistema accesible y reproducible.

**P: ¿No se pierde información al reescribir el reporte con un LLM?**
R: Es un riesgo real y por eso lo trato como hipótesis a **medir**, no como verdad dada. La
destilación aísla síntomas, ubicación y requerimientos para reducir ruido y truncamiento;
contemplo evaluar también el texto crudo y la **concatenación** crudo+destilado, para
cuantificar qué aporta realmente cada variante.

#### Sobre el recomendador de contenido

**P: ¿Por qué un encoder congelado y no fine-tuning?**
R: Es la apuesta central de la propuesta: comprobar hasta dónde llega la representación
**preentrenada** sin ajuste fino, que es el escenario más barato y reproducible. Si el
fine-tuning aportara, sería un *extra*; pero el objetivo es demostrar que un kNN sobre
buenos embeddings es competitivo **sin** ese coste.

**P: ¿Qué es MPNet y por qué ese encoder?**
R: `all-mpnet-base-v2`, de la familia *sentence-transformers*, entrenado con objetivo
contrastivo para que el **coseno refleje similitud semántica** —justo lo que pide un
recuperador—. Además es **el mismo encoder que usa TriagerX** en su módulo de
interacciones, así que la comparación atribuye cualquier diferencia al **método**, no al
embedder.

**P: ¿Para qué el canal léxico BM25 si ya hay similitud semántica?**
R: Porque la semántica difumina los *tokens* técnicos exactos —nombres de función,
identificadores, códigos de error—. BM25 los recupera literalmente. Espero que sea
especialmente útil en dominios densos en código como TypeScript; es una palanca cuyo valor
depende del dominio, y eso forma parte de lo que quiero caracterizar.

#### Sobre las interacciones y la fusión

**P: ¿En qué se diferencia el IBR del CBR, si ambos recuperan vecinos?**
R: Comparten el encoder y el paso de recuperación, pero **difieren en qué hacen con los
vecinos**. El CBR vota la **etiqueta de resolución** —quién arregló el bug—. El IBR recorre
el **grafo de interacciones tipadas** de esos vecinos —commit, review, assignment,
discussion—, ponderado por tipo y por tiempo. Bonus de ingeniería: como comparten
embeddings, el índice se computa **una sola vez**.

**P: ¿Por qué sumar puntuaciones y no fusionar por rango?**
R: Porque sumar **NPS + W_f·NIS** preserva la **confianza calibrada** del recomendador de
contenido: cuánto más seguro está, más pesa. Fusionar por rango tira esa información. La
fusión aditiva mantiene la señal cuando el contenido está confiado y deja entrar al IBR
cuando no.

**P: ¿Qué hace exactamente la compuerta (*gating*)?**
R: El IBR interviene **solo cuando el contenido duda** —margen estrecho entre el primer y
el segundo candidato—. Si el CBR está confiado, se respeta su decisión. Así el IBR aporta
donde hace falta y no mete ruido donde el contenido ya acierta.

**P: ¿Por qué restringir a un "directorio de desarrolladores activos"?**
R: Por tres motivos: **viabilidad** —no recomendar a quien ya no está—, **contención de la
cola larga**, y evitar **salidas inviables**. Es una restricción realista de cómo se usaría
el sistema en producción.

#### Sobre los datos y la evaluación

**P: ¿Por qué tres datasets y no uno?**
R: Porque la pregunta de investigación no es solo "¿funciona?", sino "**¿cuándo** funciona
cada señal?". Los tres cubren dos plataformas (Bugzilla y GitHub) y dos regímenes de
etiqueta (*assignee* en Mozilla, *fixer* en OpenJ9 y TypeScript). Eso me deja
**caracterizar el régimen**, que es uno de los objetivos.

**P: ¿Qué significa el contraste *assignee* vs *fixer* y por qué importa?**
R: Es **qué** etiqueta predigo. *Fixer* es quien arregló el bug —en OpenJ9, quien commitea
el fix—, así que la señal de commits del IBR está **alineada**. *Assignee* es a quién se le
asignó, y ahí la señal de código puede **desalinearse**. La hipótesis es que la utilidad
del IBR **depende de esa definición**, y verificarlo es parte del aporte.

**P: ¿Por qué partición temporal y no validación cruzada aleatoria?**
R: Porque el triaje es intrínsecamente temporal: en producción siempre se predice el
**futuro** con el **pasado**. Una partición aleatoria filtraría información del futuro al
entrenamiento e inflaría las métricas. La partición temporal evita esa fuga y refleja el
uso real.

**P: ¿Por qué Hit@K y MRR y no solo exactitud Top-1?**
R: Porque un recomendador de triaje se usa como **lista de sugerencias**, no como única
respuesta. Hit@K mide si el dev correcto está entre los K primeros y MRR premia ponerlo
arriba; ambas reflejan mejor la utilidad práctica que un Top-1 estricto.

#### Preguntas "trampa" / de fondo

**P: ¿No es esto solo kNN de toda la vida?**
R: El mecanismo es kNN, y esa **simplicidad es parte del argumento**. La contribución no es
inventar kNN, sino **demostrar dónde y por qué** la recuperación sobre buenos embeddings,
bien sintonizada y combinada con interacciones tipadas, compite con ensembles entrenados
—un experimento que la literatura, centrada en clasificación, no había hecho.

**P: ¿Para qué la decodificación restringida por candidatos?**
R: Es la etapa de **validación** de la salida: restringe la generación al conjunto de
desarrolladores **válidos** del directorio, de modo que el sistema no proponga nombres
inexistentes o fuera del equipo. Garantiza recomendaciones siempre **accionables**.

**P: ¿Cuál es la mayor limitación de la propuesta?**
R: Que es una validación a **escala piloto** y en un **entorno deliberadamente limitado**
—una laptop, zero-shot—. Lo asumo como una **fortaleza de encuadre**: es el escenario
realista y accesible para la mayoría de los equipos; demostrar que funciona ahí es más
valioso que demostrarlo en un clúster, y escalar solo puede mejorarlo.

---

### 4.3 Chuleta

| Pieza | Detalle |
|---|---|
| Etapas del pipeline | Destilación → CBR (NPS) → IBR (NIS) → Fusión con compuerta |
| Destilación | Gemma 4 E4B IT, cuantizado, local; síntomas/ubicación/requerimientos |
| Encoder de contenido | `all-mpnet-base-v2`, congelado, coseno, kNN |
| Cola larga | voto robusto (frec./máx./rango) + canal léxico BM25 + directorio activo |
| Interacciones (IBR) | tipadas: commit, review, assignment, discussion (Puntos de Interacción) |
| Fusión | FS = NPS + W_f · NIS (suma puntuaciones, no rangos) + *gating* por margen |
| Validación de salida | decodificación restringida a candidatos del directorio |
| Datasets / regímenes | Mozilla (*assignee*), OpenJ9 (*fixer*), TypeScript (*fixer*) |
| Evaluación | Hit@K, MRR, partición temporal, caracterización por régimen |
| Referencia | TriagerX (DeBERTa+RoBERTa + SBERT + fusión ponderada) |
