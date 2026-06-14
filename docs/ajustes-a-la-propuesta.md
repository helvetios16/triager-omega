# Ajustes a la propuesta

> Sección lista para integrar en el documento de tesis (apartado de metodología o al
> inicio de resultados). Explica, con evidencia del piloto, por qué la propuesta se
> refinó de *mejorar el clasificador de contenido* a *reemplazarlo por recuperación de
> casos*, y por qué el objetivo pasó de *superar* a *competir con menos + caracterizar
> regímenes*. Todas las cifras son Top-1 (Hit@1) en test.

## Resumen del ajuste

La propuesta inicial planteaba **replicar y mejorar el componente de contenido (CBR)**
del sistema de referencia (TriagerX) mediante un **clasificador transformer afinado**,
fusionado con un recomendador por interacción (IBR). Los experimentos piloto
condujeron a **refinar la propuesta** hacia un **CBR basado en recuperación de casos
(Case-Based Reasoning)** y a **reencuadrar el objetivo**. El cambio no responde a una
preferencia, sino a una **cadena de hallazgos empíricos** que se detalla a
continuación. Reportar este ajuste de forma explícita es parte del método: el piloto
existe precisamente para validar el diseño antes de la corrida a escala.

## Evidencia que motivó el cambio

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

## Reencuadre del objetivo

Como consecuencia, el objetivo se ajustó de **"superar al sistema de referencia"** a
**"alcanzar un rendimiento competitivo con una arquitectura más simple e interpretable,
y caracterizar en qué régimen de datos aporta cada señal (contenido frente a
interacción)"**. Justificación: superar en términos absolutos exigiría replicar el
ensemble de la referencia —sin novedad propia—, mientras que el eje de **simplicidad
(cero entrenamiento), interpretabilidad y análisis de regímenes** constituye una
contribución original. Este reencuadre convierte hallazgos como "el ajuste fino no
mejora el Top-1" o "la señal de interacción solo aporta en ciertos regímenes" de
aparentes limitaciones en **resultados del estudio**.

## Qué se conserva de la propuesta original

El esqueleto del sistema **no cambia**: sigue siendo híbrido (contenido + interacción)
con fusión aditiva `FS = NPS + W_f·NIS`, el IBR con interacciones tipadas, la
partición temporal y el directorio activo de candidatos. **Solo se reemplaza el
componente de contenido** (clasificador → recuperación de casos). Todo el desarrollo
del piloto (datos, scripts, infraestructura, hallazgos) se reutiliza.

## Redacción breve (para pegar directo)

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

## Detalle de soporte (cifras)

| Hallazgo | Cifra (Top-1) |
|---|---|
| Clasificador nuestro (1 transformer) vs. base de la referencia | 0.225–0.232 vs. 0.189 |
| Mejora por más entrenamiento (15→30 ép, 256→512 tok) | +0.75 pp |
| Ensemble de 2 transformers (referencia) | 0.270 |
| **CBR de recuperación (nuestro, sin entrenar)** | **0.2715** |
| Sistema completo de la referencia (ensemble + interacción) | 0.328 |

Fuentes: `docs/comparacion-triagerx-openj9.md`, `docs/cbr-recuperacion.md`.
