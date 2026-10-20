# Índice de Documentación — Triager-Omega

Guía de navegación del corpus de documentación del proyecto, organizado por propósito y vigencia.

---

## 1. Tesis y Propuesta Formal (`docs/tesis/`)

Documentos consolidados y capítulos que fundamentan la memoria / propuesta de tesis:

- [**01 — Tesis: Estado, Propuesta y Corrida Oficial**](file:///Users/sebastian/Documents/Python/triager-omega/docs/tesis/01-tesis-propuesta.md): Encuadre formal, preguntas de investigación, diseño experimental para publicación.
- [**02 — Destilación Semántica**](file:///Users/sebastian/Documents/Python/triager-omega/docs/tesis/02-destilacion-semantica.md): Fundamento y evolución de la destilación LLM con taxonomía diagnóstica `[FL][SY][CP]`.
- [**03 — CBR de Recuperación, Entrenamiento y Resultados**](file:///Users/sebastian/Documents/Python/triager-omega/docs/tesis/03-cbr-y-resultados.md): Fundamentos del Case-Based Reasoning real (recuperación zero-shot vs clasificación densa) y validaciones.
- [**04 — Comparativas: OpenJ9, TriagerX y Mejoras al IBR**](file:///Users/sebastian/Documents/Python/triager-omega/docs/tesis/04-comparativas.md): Análisis comparativo head-to-head contra los baselines de la literatura.
- [**05 — Infraestructura y Setup**](file:///Users/sebastian/Documents/Python/triager-omega/docs/tesis/05-infraestructura.md): Configuración de hardware (Apple Silicon MPS, CUDA), backends de inferencia y minería.
- [**Guion de Presentación (10 min)**](file:///Users/sebastian/Documents/Python/triager-omega/docs/tesis/presentacion-guion-10min.md): Guion de defensa oral y banco de preguntas/respuestas.
- [**Pipelines Interactivos**](file:///Users/sebastian/Documents/Python/triager-omega/docs/tesis/pipelines.html) y [**Resumen del Proyecto**](file:///Users/sebastian/Documents/Python/triager-omega/docs/tesis/resumen-proyecto.html): Vistas HTML de diagramas y flujos.

---

## 2. Resultados Experimentales y Tareas en Curso (`docs/experimental/`)

Reportes de experimentos específicos, barridos de hiperparámetros y hojas de trabajo activas:

- [**Resultados CBR en TypeScript**](file:///Users/sebastian/Documents/Python/triager-omega/docs/experimental/typescript-cbr-resultados.md): Récord experimental (Hit@1 0.3354 vs TriagerX 0.3240).
- [**Resultados y Mejoras IBR en OpenJ9**](file:///Users/sebastian/Documents/Python/triager-omega/docs/experimental/openj9-mejoras-ibr-resultados.md): Retuning de interaction points y mitigaciones en OpenJ9.
- [**Explicación Comparativa OpenJ9**](file:///Users/sebastian/Documents/Python/triager-omega/docs/experimental/openj9-comparativa-explicada.md): Análisis cualitativo de predicciones.
- [**CBR de Recuperación (Notas de diseño)**](file:///Users/sebastian/Documents/Python/triager-omega/docs/experimental/cbr-recuperacion.md): Formulación matemática del recuperador y agregación por voto.
- [**Resultados de Entrenamiento CUDA**](file:///Users/sebastian/Documents/Python/triager-omega/docs/experimental/resultados-cuda.md): Métricas de corrida en GPU NVIDIA (RTX 5060).
- [**Tarea Pendiente: Evaluación Mozilla**](file:///Users/sebastian/Documents/Python/triager-omega/docs/experimental/pendiente-evaluacion-mozilla.md): Hoja de ruta para completar la evaluación de destilación + CBR zero-shot en el corpus Bugzilla.

---

## 3. Bitácoras y Notas Históricas (`docs/bitacoras/`)

Registro de notas de trabajo activas y bitácoras de laboratorio. Las 15 notas previas de investigación fueron completamente consolidadas e integradas en los capítulos formales de [`docs/tesis/`](file:///Users/sebastian/Documents/Python/triager-omega/docs/tesis), y sus borradores fuente se preservan en `docs/bitacoras/_archive/`.

---

## 4. Schemas y Datasets (`docs/schemas/`)

- [**BUG_REPO_SCHEMA.md**](file:///Users/sebastian/Documents/Python/triager-omega/docs/schemas/BUG_REPO_SCHEMA.md): Schema detallado de los datasets de Mozilla Bugzilla (`Bug_meta_data`, `Contribution_information`, `Bug_Report_Comments`).
- [**DESCRIPCION_DATASETS.md**](file:///Users/sebastian/Documents/Python/triager-omega/docs/schemas/DESCRIPCION_DATASETS.md): Guía de columnas, tipos y volumen de datos curados.
