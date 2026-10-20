# Guion de presentación (10 min) + banco de preguntas

> Defensa de **propuesta de tesis** — *triager-omega*: un *framework* híbrido de triaje de
> incidencias que combina contenido e interacciones históricas, con un **recomendador de
> contenido por recuperación** (CBR) que no se entrena.
> Alineado diapositiva por diapositiva con `oficuo/src/presentacion.tex`.
> Ritmo objetivo ~130 palabras/min. Marcas de tiempo y de diapositiva entre [ ].

---

## GUION (≈10 minutos)

### [0:00 – 0:45] Apertura

> [Diapositiva: Portada — título, autor, asesor]

Buenos días. Mi nombre es Sebastián Mendoza y presento mi propuesta de tesis:
*Modelos de Aprendizaje Profundo para el Triaje de Incidencias de Software*, asesorada
por el profesor Edgar Sarmiento. En los próximos diez minutos voy a plantear el
**problema**, el **estado del arte**, los **datos**, la **propuesta de sistema** y la
**estrategia de evaluación** con que pienso validarla.

---

### [0:45 – 2:00] El problema del triaje

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

### [2:00 – 3:00] Motivación y propuesta

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

### [3:00 – 3:45] Objetivos

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

### [3:45 – 5:00] Estado del arte

> [Diapositiva: Tres paradigmas en la literatura + TriagerX]

La literatura se organiza en tres paradigmas: **aprendizaje profundo y modelos de
lenguaje**, que aportan comprensión semántica del texto; **grafos y bases de
conocimiento**, que modelan las interacciones entre desarrolladores, reportes y
componentes; y **optimización estocástica y ML clásico**.

Mi referencia metodológica es **TriagerX**: combina un clasificador de contenido —DeBERTa
más RoBERTa— con un módulo de interacciones basado en SBERT y una fusión ponderada. Es el
diseño más completo y validado industrialmente, así que lo adopto como punto de
comparación.

> [Diapositivas: Trabajos relacionados I y II — tablas]

Estas dos tablas resumen el panorama. La primera reúne los trabajos de **aprendizaje
profundo y modelos de lenguaje**; la segunda, los de **grafos** y **aprendizaje clásico**.
No las leeré una a una; el mensaje es que el campo está dominado por modelos cada vez más
pesados y específicos, y que casi todos comparten dos limitaciones: **dependen de
reentrenamiento** y **rinden mal en la cola larga**. Ahí es donde mi propuesta busca
diferenciarse.

---

### [5:00 – 5:45] Conjuntos de datos

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

### [5:45 – 6:30] Visión general y arquitectura

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

> [Diapositiva: Arquitectura del sistema — diagrama del pipeline]

Aquí está el diagrama completo: se ve cómo el reporte entra, se destila, alimenta en
paralelo a los dos recomendadores, y sus señales se fusionan antes de filtrar por el
directorio activo.

---

### [6:30 – 7:10] Destilación semántica

> [Diapositiva: Destilación semántica]

Primera etapa. El texto crudo de un bug mezcla prosa, código y trazas; eso **degrada al
codificador** y obliga a truncar. Para limpiarlo uso un modelo de instrucción —**Gemma 4
E4B IT**, cuantizado y ejecutado localmente— que reescribe el reporte aislando tres cosas:
los **síntomas**, la **ubicación** en el sistema y los **requerimientos** asociados.

Lo importante conceptualmente: es una transformación de los **datos**, no una compresión
del modelo. Reduce ruido y truncamiento antes de recuperar.

---

### [7:10 – 7:50] Recomendador de contenido por recuperación

> [Diapositiva: Recomendador de contenido por recuperación]

Segunda etapa, el corazón de la propuesta. Uso un codificador de oraciones **congelado**
—`all-mpnet-base-v2`— y mido similitud por coseno. Recupero los **k reportes históricos
más similares** —un kNN— y agrego el voto de quienes los resolvieron, normalizado, para
obtener la puntuación **NPS**.

Dos refinamientos pensados para la cola larga: una **agregación robusta** del voto
—normalizar por frecuencia, voto máximo, por rango— y, opcionalmente, un **canal léxico
BM25** que rescata los *tokens* técnicos exactos que la similitud semántica difumina.

---

### [7:50 – 8:30] Recomendador por interacciones históricas

> [Diapositiva: Recomendador por interacciones históricas]

Tercera etapa. Sobre esos mismos incidentes similares, ahora miro **cómo** participó cada
candidato. Las interacciones son **tipadas**, con un peso —un **Punto de Interacción**— por
tipo: *commit*, *review*, *assignment* y *discussion*. Se agregan ponderadas por similitud
y se normalizan en la puntuación **NIS**.

Como opciones para mejorar la señal: recuperar los vecinos por **BM25**, para decorrelar
este módulo del de contenido, y un **decaimiento temporal** que favorezca la actividad
reciente.

---

### [8:30 – 9:10] Fusión de señales con compuerta

> [Diapositiva: Fusión de señales con compuerta]

Cuarta etapa: cómo combino ambas señales. La fusión es **aditiva**:
**FS = NPS + W_f · NIS**. Sumo **puntuaciones, no rangos**, justamente para preservar la
**confianza calibrada** del recomendador de contenido.

Y le añado una **compuerta** o *gating*: el IBR interviene **solo cuando el contenido
duda** —cuando el margen entre el primer y el segundo candidato es estrecho—; si el
contenido está confiado, se respeta su decisión. Finalmente, toda recomendación se
restringe a un **directorio de desarrolladores activos**: garantiza viabilidad, contiene
la cola larga y evita salidas inviables.

---

### [9:10 – 9:50] Estrategia de evaluación

> [Diapositiva: Estrategia de evaluación]

Para validar el *framework* propongo: métricas de *ranking* —**Hit@K** y **MRR**—, que es
como se usa un recomendador en la práctica; **partición temporal**, entrenando con el
pasado y evaluando con el futuro para evitar fuga de información; y, sobre todo, una
**caracterización por régimen de la etiqueta**. La hipótesis es que con *fixer* la señal de
contribución —los *commits*— está **alineada** y el IBR aporta, mientras que con *assignee*
esa señal puede **desalinearse**. Caracterizar ese contraste es parte central del aporte.

---

### [9:50 – 10:00] Cierre

> [Diapositiva: Gracias]

En resumen, propongo un triaje híbrido **simple, interpretable y sin reentrenar el
recomendador de contenido**, que enfrenta de cara la cola larga y el arranque en frío, y
que se validará en tres regímenes de datos distintos. Muchas gracias; quedo atento a sus
preguntas.

---

## BANCO DE PREGUNTAS Y RESPUESTAS

> Ordenadas de más probables a más técnicas. Cada una con respuesta corta lista para decir.
> Adaptadas a una defensa de **propuesta** (qué se hará y por qué), no de resultados finales.

### Sobre el encuadre

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

### Sobre la destilación

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

### Sobre el recomendador de contenido

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

### Sobre las interacciones y la fusión

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

### Sobre los datos y la evaluación

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

### Preguntas "trampa" / de fondo

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

## CHULETA (para tener a la vista)

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
