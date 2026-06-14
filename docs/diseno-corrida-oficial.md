# Diseño de la corrida oficial (versión publicable)

> Estado: el trabajo actual es un **piloto / proof-of-concept** que validó el diseño
> en 2 regímenes (OpenJ9 cola larga, Mozilla pocos devs). Este documento fija las
> decisiones de la **corrida oficial** para que el resultado salga directo en formato
> de **estudio empírico** publicable (no "método nuevo").
>
> Regla de oro: el aporte no es "mi CBR gana", es **"caracterizar cuándo basta la
> señal de contenido y cuándo aporta la de interacción en triaging"**. La recuperación
> kNN es un baseline conocido → como método no pasa revisión; como estudio a escala, sí.

## 1. Preguntas de investigación (fijar antes de correr)

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

## 2. Datasets (lo que falta para escala)

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

## 3. Protocolo experimental (fijar y no tocar)

- **Split temporal estricto** por fecha (no aleatorio): train = pasado, test = futuro.
  Validación temporal intermedia para tunear (k, τ, W_f) **sin tocar test**.
- **Directorio activo (candidate-constrained):** definir el conjunto de devs candidatos
  por proyecto (p. ej. activos en ventana reciente) y evaluar dentro de él.
- **Sin fuga:** índice de recuperación = solo train; el bug de consulta nunca se
  recupera a sí mismo.
- Congelar versiones (modelos, librerías) y semillas.

## 4. Condiciones a comparar (mismas clases, mismo split, por proyecto)

| Familia | Condición |
|---|---|
| Baselines | azar · frecuencia (most-active dev) · TF-IDF + kNN (baseline clásico) |
| Contenido | **CBR-recuperación zero-shot** (nuestro foco) · CBR-recuperación afinado (MNRL) · clasificador 1 transformer · ensemble 2 transformers (réplica TriagerX) |
| Interacción | IBR-solo (interacciones tipadas) |
| Fusión | CBR + IBR (barrido W_f en validación) para cada CBR |

Incluir TF-IDF+kNN es clave: deja claro qué aporta el embedding semántico sobre el
baseline de similitud clásico (refuerza el related work y la honestidad).

## 5. Métricas

- **Calidad:** Hit@{1,3,5,10}, MRR, y MAP. Reportar por proyecto y agregado.
- **Costo (primera clase, RQ5):** tiempo de entrenamiento (recuperación = 0), latencia
  de inferencia por bug, memoria/VRAM, tamaño del modelo. Tabla rendimiento-vs-costo.

## 6. Rigor estadístico (sin esto no hay paper)

- **Intervalos de confianza por bootstrap** sobre el test (p. ej. 1000 remuestreos)
  en toda métrica reportada.
- **Tests de significancia** entre condiciones (Wilcoxon pareado sobre bugs, o
  bootstrap de la diferencia); corrección por comparaciones múltiples.
- **Varias semillas** donde haya aleatoriedad (fine-tuning, sampler) → media ± std.
- Tamaño de efecto, no solo p-valor.

## 7. Análisis (lo que convierte resultados en contribución)

- **Caracterización del régimen (RQ2):** correlacionar el delta (recuperación −
  clasificador) y el aporte del IBR con métricas del proyecto (nº devs, % cola,
  densidad). Idealmente un gráfico delta vs. régimen que muestre la inversión.
- **Análisis de errores:** dónde falla cada enfoque (devs raros, bugs cortos, etc.).
- **Interpretabilidad (ventaja del CBR de casos):** ejemplos "se recomienda a X
  porque este bug se parece a #1234, #998 que X resolvió" — TriagerX no ofrece esto.
- **Ablaciones:** encoder (MPNet vs. otros), k, τ, decaimiento temporal, tipos de
  interacción (contribution/assignment/discussion).

## 8. Amenazas a la validez (sección obligatoria)

- Externa: selección de proyectos; generaliza solo a OSS con historial.
- Constructo: etiqueta = quien arregló ≠ "mejor" triador; ruido de identidad.
- Interna: tuning en validación temporal; posible fuga por re-aperturas.
- Conclusión: dependencia de un único encoder/familia si no se ablaciona.

## 9. Reproducibilidad

- Liberar scripts (ya existen: `cbr_retrieval_*`, `eval_openj9_full --cbr-mode
  retrieval`, mineros), splits, semillas, versiones y un README de reproducción.
- Paquete de réplica (datos derivados o instrucciones de minería).

## 10. Venue y checklist

- **Workshop / short** (alcanzable pronto): RQ1+RQ3 con IC, 3-4 proyectos.
- **Journal medio (IST / JSS) o replication track** (objetivo real): las 5 RQ,
  5-8 proyectos, estadística completa, costo, análisis de errores e interpretabilidad.
- Checklist mínimo antes de enviar: IC en todas las tablas ✔ · ≥5 proyectos ✔ ·
  baseline TF-IDF+kNN ✔ · tabla de costo ✔ · gráfico de régimen ✔ · amenazas ✔.

## 11. Qué se reutiliza del piloto (no se tira nada)

- Scripts: `scripts/cbr_retrieval_openj9.py`, `scripts/cbr_retrieval_pilot.py`,
  `scripts/eval_openj9_full.py --cbr-mode retrieval`, mineros de timelines.
- Hallazgos del piloto = hipótesis preregistradas de la corrida oficial.
- Infra de cómputo (omen RTX 5060) para el clasificador/ensemble y el fine-tuning;
  la recuperación zero-shot corre hasta en CPU/MPS.
- Docs: `docs/cbr-recuperacion.md`, `docs/comparacion-triagerx-openj9.md`,
  `docs/pipelines.html`.

## Resumen ejecutivo

La corrida oficial **no cambia el método** del piloto; cambia **escala + estadística +
encuadre**. Si se entra a ella con (a) 5-8 proyectos de regímenes variados, (b)
bootstrap/IC y tests desde la primera tabla, (c) baseline TF-IDF+kNN, (d) tabla de
costo y (e) la narrativa de estudio empírico ("cuándo contenido, cuándo interacción"),
el resultado es publicable. Como método nuevo, no — y eso no se arregla escalando.
