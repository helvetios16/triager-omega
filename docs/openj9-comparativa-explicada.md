# OpenJ9 (~50 clases): tabla comparativa explicada

> Versión "para leer de un vistazo" del head-to-head contra TriagerX.
> **Un solo régimen** (OpenJ9, ~50 clases, comparación justa) para no mezclar
> números de Mozilla/TypeScript. Detalle técnico completo en
> `docs/comparacion-triagerx-openj9.md` y `docs/cbr-recuperacion.md`.

## ⚠️ Qué significa (y qué NO significa) "el nuestro es mejor" — leer primero

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
2. **¿Por qué nos gana?** → Únicamente por su IBR (+5.8 pp sobre su CBR). **No**
   porque su CBR sea mejor.
3. **¿Quién gana el núcleo CBR?** → Empate / leve ventaja nuestra (0.2715 ≥ 0.270),
   y lo logramos **sin entrenar y con un solo modelo**.

El argumento de tesis honesto **no** es "le ganamos a TriagerX". Es: *igualamos el
corazón de su método (el CBR) con una fracción del costo y cero entrenamiento; lo
único que mantiene su sistema completo por delante es la palanca IBR, que además es
dependiente del régimen.*

## La tabla (todo junto, un solo régimen)

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

## Explicación fila por fila

**Filas 1–4 son TriagerX (la referencia que queremos batir):**

- **Fila 1 y 2** — sus dos transformers por separado, cada uno clasificando solo.
  Son lo más débil (0.175, 0.189).
- **Fila 3** — lo que ellos llaman su "CBR": combinar los dos transformers
  (ensemble). Combinarlos sube a **0.270**. Esta es la cifra clave del componente
  CBR de ellos.
- **Fila 4** — su sistema **completo**: al ensemble (0.270) le suman su IBR (grafo
  de interacciones) → **0.328**. Ese salto 0.270 → 0.328 (**+5.8 pp**) es lo que
  aporta *su* IBR.

**Filas 5–9 son las nuestras (omega):**

- **Fila 5** — nuestro intento de imitar su clasificador, pero con **un solo**
  transformer (no el ensemble de dos). Da **0.2322**. Ya le gana a su DeBERTa-solo
  (0.189), pero no llega a su ensemble (0.270) porque nos falta el segundo
  transformer.
- **Fila 6** — nuestro IBR **solo**, sin nada de CBR. A 50 clases es flojo:
  **0.1685**.
- **Fila 7** — fusionar fila 5 + fila 6. Sube poquito (0.2322 → **0.2397**). El IBR
  casi no ayuda aquí.
- **Fila 8** — **nuestra propuesta real**: en vez de clasificar, **recuperar** los
  bugs más parecidos con kNN y votar el dev (zero-shot, sin entrenar nada). Da
  **0.2715** → **iguala el ensemble de TriagerX (0.270)** con un solo modelo y cero
  entrenamiento. Esta es la fila estrella.

  > **Nota — qué encoder da este 0.2715:** es **MPNet** (`all-mpnet-base-v2`), **no
  > MiniLM**. MPNet se eligió a propósito por ser el **mismo SBERT que usa TriagerX
  > en su IBR** (comparación justa). MiniLM solo aparece en la ablación `--rerank`
  > (cross-encoder `ms-marco-MiniLM`), que **no** subió el Top-1 (0.2715 es el techo).
  > Resumen: **encoder del CBR = MPNet; MiniLM = solo el reranker descartado.**
- **Fila 9** — fusionar nuestra propuesta (fila 8) + IBR (fila 6). **Baja** a
  0.2622. O sea: a 50 clases sumar el IBR **empeora**, no mejora.

## Lo que dice la tabla, en una frase

- **Núcleo CBR:** fila 8 (0.2715) **≥** fila 3 (0.270) → nuestro CBR iguala/supera
  al de ellos, sin entrenar.
- **Sistema completo:** fila 4 (0.328) > nuestro mejor (fila 8, 0.2715) por
  **~5.6 pp**, y esos 5.6 pp son **exactamente el IBR de ellos** (el salto fila
  3 → 4). A nosotros el IBR no nos da ese salto a 50 clases (fila 8 → 9 baja).

Ganamos el componente; la única ventaja del full de TriagerX es su palanca IBR —
que en este régimen (cola larga) a nosotros no nos aplica.

## ¿Por qué el IBR de TriagerX "ayuda" y el nuestro no? (el tema clave)

**No es que su IBR sea un mejor componente.** Cuando se mide en igualdad de
condiciones, el aporte marginal del IBR es **de la misma magnitud** en los dos
sistemas (es un Δ interno, no depende del nº de clases):

| Sistema | CBR-solo | +IBR (Full) | Δ del IBR |
|---|---|---|---|
| TriagerX (50 clases) | 0.270 | 0.328 | **+5.8 pp** |
| omega (17 clases) | 0.199 | 0.267 | **+6.8 pp** |

→ Nuestro IBR está **bien replicado**: cuando lo ponemos sobre un *clasificador*,
suma lo mismo que el suyo. Entonces la pregunta no es "¿por qué su IBR es mejor?"
sino **"¿por qué sumarlo nos ayuda en unos casos y en otros no?"**. Hay dos
razones, y ninguna es la calidad del componente:

### Razón 1 — Sobre qué base se apoya el IBR (correlación)

El IBR aporta **solo si trae señal distinta** a la del CBR sobre el que se monta.

- **TriagerX:** su CBR es un **clasificador** (ensemble de transformers) y su IBR
  recorre un **grafo de interacciones**. Son dos mecanismos distintos →
  **decorrelacionados** → el IBR añade información nueva → **+5.8 pp**.
- **Nuestro pipeline nuevo:** nuestro CBR **también es un recuperador** (kNN sobre
  embeddings MPNet) y nuestro IBR **mira el grafo de los mismos vecinos** usando el
  **mismo encoder** (MPNet). Recuperan **los mismos bugs parecidos** → están
  **correlacionados** → el IBR no trae nada nuevo → sumarlo solo mete ruido
  (fila 8 → 9 baja). Es una consecuencia de *nuestra* decisión de arquitectura
  (hacer el CBR un recuperador), no de que su IBR sea superior.

### Razón 2 — El régimen (densidad de interacciones por dev)

El IBR necesita **historial de interacciones por dev** para tener señal:

- **OpenJ9 a 50 clases (cola larga):** muchos devs con pocas interacciones → el
  IBR-solo es débil (**0.1685**, fila 6) → poco que aportar.
- **OpenJ9 a 17 devs / Mozilla (pocos devs muy activos):** el IBR-solo es fuerte
  (0.238 en el 17-set; **0.676** en Mozilla) → ahí sí aporta. En Mozilla, fusionar
  nuestro recuperador + IBR **sube +1.6 pp** (0.718 → 0.7337, W_f=0.8).

### Resumen del tema IBR

| | ¿IBR ayuda? | Por qué |
|---|---|---|
| TriagerX, 50 cls | Sí (+5.8 pp) | IBR sobre un *clasificador* (decorrelacionado) + apoyado en su ensemble fuerte |
| omega, 50 cls (cola larga) | **No** (baja) | IBR correlacionado con nuestro recuperador (mismo encoder/vecinos) + cola larga débil |
| omega, Mozilla (pocos devs) | **Sí** (+1.6 pp) | Régimen denso: IBR-solo fuerte → señal complementaria |

**Conclusión:** su IBR no es "mejor"; es que (1) lo montan sobre un clasificador,
que no comparte mecanismo con el grafo, y (2) miden el full en un régimen donde el
IBR tiene señal. Nuestro IBR, medido en igualdad de condiciones, replica el suyo —
y en Mozilla incluso nos suma. Que en OpenJ9-50 no aporte es esperado y explicable,
no una debilidad del componente.
