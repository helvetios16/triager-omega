# Descripción de Datasets — Bug Repo Dataset

> Schema de referencia: [`BUG_REPO_SCHEMA.md`](BUG_REPO_SCHEMA.md)  
> Archivos de trabajo: `data_parquet/*_curado.parquet` (resultado del pipeline de 9 pasos documentado en `PROCESO.md`)

---

## Datasets curados

### 1. Bug Meta Data
**Archivo:** `data_parquet/Bug_meta_data_curado.parquet`  
**Volumen:** 225.196 filas · 46 columnas (seleccionadas de 299)  
**Clave primaria:** `Bug Id`

Contiene la ficha técnica de cada reporte de error registrado en el repositorio Mozilla Bugzilla. Cada fila es un bug único. Concentra toda la información estructural del bug: quién lo reportó, a quién está asignado, en qué producto y componente ocurre, su estado de resolución, prioridad y severidad, y métricas de actividad comunitaria (votos, comentarios, flags).

Columnas clave (ver schema §1 para la lista completa):

| Columna | Tipo | Descripción |
|---|---|---|
| `Bug Id` | Int | PK — identificador único del bug |
| `Product` / `Component` | String | Dónde ocurre el error |
| `Priority` / `Severity` | Categorical | Urgencia e impacto |
| `Bug Status` / `Resolution` | Categorical | Estado actual y resultado |
| `Creator` | String | Quien reportó el bug |
| `Contributor Id` | List | IDs de colaboradores involucrados (FK → Contributor Information) |
| `Comment Count` | Int | Cantidad de comentarios (proxy de actividad) |
| `Votes` | Int | Votos de la comunidad |
| `Blocks` / `Depends On` | List | Relaciones de bloqueo entre bugs |
| `Creation Time` / `Last Change Time` | Datetime | Granularidad temporal |

---

### 2. Contributor Information
**Archivo:** `data_parquet/Contribution_information_dataset_curado.parquet`  
**Volumen:** 50.345 filas · 14 columnas  
**Clave primaria:** `Contributor Id`

Perfil de actividad acumulada de cada colaborador en el repositorio. Cada fila es un usuario único. Las columnas representan métricas de contribución: cuántos bugs reportó, cuántos comentarios hizo, cuántos parches envió y revisó, cuántas veces actuó como QA Contact. Permite clasificar a los colaboradores por experiencia (casual, regular, core).

Columnas clave (ver schema §2 para la lista completa):

| Columna | Tipo | Descripción |
|---|---|---|
| `Contributor Id` | Int | PK — identificador único del colaborador |
| `User Name` | String | Nombre de usuario |
| `Bugs Filed` | Int | Total de bugs reportados (proxy de experiencia) |
| `Comments Made` | Int | Total de comentarios realizados |
| `Patches Submitted` / `Patches Reviewed` | Int | Actividad de código |
| `Assigned To` / `Assigned To and Fixed` | Int | Bugs asignados y resueltos |
| `Permissions` | Categorical | Nivel de acceso en el sistema |
| `Last Activity` | Datetime | Fecha de última interacción |

---

### 3. Bug Report Comments
**Archivo:** `data_parquet/Bug_Report_Comments_curado.parquet`  
**Volumen:** ~1.011.057 filas · 8 columnas  
**Clave primaria:** `Comment Id`

Unión de las tres partes del dataset de comentarios (`Part_1/2/3`). Cada fila es un comentario individual en un bug report. Incluye el texto completo del comentario, quién lo escribió, cuándo, y a qué bug pertenece. Es el dataset más voluminoso y el único que contiene texto libre (`Text`), haciendo de base para tareas NLP.

Columnas clave (ver schema §3 para la lista completa):

| Columna | Tipo | Descripción |
|---|---|---|
| `Comment Id` | Int | PK — identificador único del comentario |
| `Bug Id` | Int | FK → Bug Meta Data |
| `Author Id` | Int | FK → Contributor Information (`Contributor Id`) |
| `Creator` | String | Email/usuario del autor |
| `Text` | String | Contenido textual del comentario |
| `Time` | Datetime | Fecha y hora de publicación |
| `Bug Report` | Bool | True si el comentario es el reporte inicial del bug |

---

### 4. Good Bug Reports
**Archivo:** `data_parquet/Good_Bug_reports_curado.parquet`  
**Volumen:** 12.614 filas · 61 columnas  
**Clave primaria:** heredada de `Comment Id`

Subconjunto filtrado de Bug Report Comments que corresponde a reportes de bugs "buenos": comentarios iniciales (`Bug Report = True`) que incluyen pasos para reproducir el error (*Steps to Reproduce*). Está denormalizado — cada fila combina los campos del comentario (schema §3) con los campos del bug asociado (schema §1, sufijo `_y` donde hay conflicto de nombre). Incluye columnas adicionales propias del join: `Contains Steps To Reproduce`, `Is Private`, `Reactions`, `Raw Text`, `Attachment Id`.

Es el dataset sobre el que se aplica la etiqueta de calidad (`Total Score`) al hacer join con Good CTQRS Filtered.

---

### 5. Good CTQRS Filtered
**Archivo:** `data_parquet/Good_CTQRS_filtered_curado.parquet`  
**Volumen:** 10.351 filas · 2 columnas  
**Clave de join:** `text` (texto exacto del comentario)

Dataset de evaluación de calidad. Cada fila es un texto de bug report evaluado según la rúbrica CTQRS, con su puntaje total (`Total Score`, rango 13–17). Es el único dataset con etiqueta supervisada del proyecto.

| Columna | Tipo | Descripción |
|---|---|---|
| `text` | String | Texto del comentario (FK por contenido → Good Bug Reports.`Text`) |
| `Total Score` | Int | Puntaje de calidad CTQRS (13 = peor, 17 = mejor) |

Distribución de clases (desbalanceada): Score 14 domina con 35.9%; ratio máximo/mínimo = 12.2x.

---

## Relaciones entre datasets

```
Contributor Information (1)
        |
        | Contributor Id
        |
        ↓ (N)
Bug Meta Data (1) ──────────── Bug Id ──────────── (N) Bug Report Comments
        |                                                       |
        |                                                       | Bug Report = True
        |                                                       | + Steps To Reproduce
        └───────────── join denormalizado ──────────────────── ↓
                                                      Good Bug Reports (N)
                                                               |
                                                               | text (exact match)
                                                               |
                                                               ↓ (1)
                                                     Good CTQRS Filtered
                                                       (etiqueta = Total Score)
```

### Claves de relación

| Relación | Tipo | Clave |
|---|---|---|
| Bug Meta Data → Bug Report Comments | 1:N | `Bug Id` |
| Contributor Information → Bug Meta Data | 1:N | `Contributor Id` (lista en Bug Meta) |
| Contributor Information → Bug Report Comments | 1:N | `Author Id` = `Contributor Id` |
| Bug Report Comments → Good Bug Reports | filtro | `Bug Report = True` + `Contains Steps To Reproduce` |
| Good Bug Reports → Good CTQRS Filtered | 1:1 (parcial) | `Text` exacto (99.9% match rate; 6.138 textos duplicados sin scores contradictorios) |

### Advertencias de join

- **Join por texto exacto** (Good Bug Reports ↔ CTQRS): sensible a diferencias de espacios o encoding. No existe ID numérico compartido.
- **`Contributor Id` en Bug Meta Data** es una lista (un bug puede tener múltiples colaboradores): requiere `explode()` antes de hacer join.
- **Solo el 1% del repositorio tiene etiqueta CTQRS** (10.351 textos etiquetados sobre ~1M comentarios totales).
