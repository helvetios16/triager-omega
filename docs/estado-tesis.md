# Estado de la tesis — qué sostiene la evidencia (2026-06-13)

Documento de encuadre. Resume qué hipótesis del proyecto quedaron respaldadas
por los datos del piloto y cuáles no, y propone el relato defendible de la tesis.

Contexto: todo es **escala piloto** (Mozilla 20 devs / OpenJ9 17 devs) con un
**CBR deliberadamente recortado** (un solo DeBERTa, sin el ensemble de dos
transformers del paper TriagerX). Comparar absolutos contra TriagerX mide
"piloto vs. sistema entrenado a fondo", no "qué diseño es mejor". Ver
[comparacion-triagerx-openj9.md](comparacion-triagerx-openj9.md).

---

## 1. La destilación (Módulo 2) NO es la contribución estrella

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
(la decisión §5.8 de *concatenar* crudo+destilado en vez de *reemplazar* queda
confirmada: reemplazar perjudica), pero **no es el hallazgo central** que la
idea prometía.

### Hipótesis no probada que podría rescatarla: la cola larga
Con 20 devs (la cabeza) el texto crudo ya es rico, así que destilar no agrega
señal. Con 450 devs y reportes pobres/escasos, la destilación podría rescatar a
los devs raros. **El piloto no prueba este escenario.** Trabajo futuro: medir
distilled vs both *solo en el subconjunto cola-larga* si se decide escalar.

---

## 2. Lo que SÍ quedó fuerte

1. **Arquitectura híbrida + IBR.** Full > CBR-solo y > IBR-solo en *ambos*
   datasets. El IBR aporta **+5.8 pp Top-1 idéntico** al que aporta en TriagerX
   (0.270→0.328 ellos; 0.199→0.257 nosotros) → réplica fiel.
2. **Dependencia de la etiqueta (hallazgo propio, no del paper).** La señal
   `contribution` del IBR **ayuda con etiqueta `fixer`/`owner`** (OpenJ9) y
   **daña con `assignee`** (Mozilla). Demostrado end-to-end en dos regímenes.

---

## 3. Ranking de hallazgos por fuerza de evidencia

| Hallazgo | Fuerza | Rol en la tesis |
|---|---|---|
| Arquitectura híbrida + IBR replicado | Fuerte | **Pilar** |
| Dependencia de la etiqueta (assignee vs fixer) | Fuerte / novedoso | **Pilar / aporte original** |
| Destilación (concatenar > reemplazar) | Débil / marginal | **Ablación medida**, no titular |

---

## 4. Qué sería de la tesis — relato propuesto

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

### Decisión pendiente del autor
¿La destilación era la contribución **principal** a defender, o **una de
varias**? De eso depende si basta el encuadre de arriba (B-de-varias) o si hay
que invertir en escalar para probar la hipótesis de cola larga (A-principal).
