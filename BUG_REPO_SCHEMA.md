# Estructura Maestra: Bug Repo Dataset

Este documento define la estructura de los tres conjuntos de datos principales del proyecto.

## 1. Bug Report Meta Data
Información técnica principal de los reportes de error.
- **Bug Id**: Identificador numérico único (PK).
- **Summary**: Descripción breve del error.
- **Priority**: Nivel de prioridad (P1, P2, etc.).
- **Severity**: Nivel de severidad (crítica, mayor, menor).
- **Creator**: Nombre de usuario o email del creador.
- **Creator Detail**: Objeto con información detallada del creador.
- **Bug Status**: Estado actual (NEW, IN PROGRESS, RESOLVED).
- **Product**: Nombre del producto asociado.
- **Component**: Módulo específico donde se encontró el error.
- **Resolution**: Estado de resolución ('FIXED', 'INVALID').
- **Assigned To**: Usuario asignado al error.
- **Assigned To Detail**: Objeto con información detallada del asignado.
- **Contributor Email**: Lista de correos de los colaboradores involucrados.
- **Contributor Id**: Lista de IDs de los colaboradores.
- **Votes**: Número de votos del error.
- **QA Contact**: Persona de QA responsable.
- **QA Whiteboard**: Notas de seguimiento de QA.
- **URL**: Enlace relevante al reporte.
- **Whiteboard**: Notas adicionales.
- **Platform**: Plataforma de hardware.
- **Operating System (op sys)**: Sistema operativo.
- **Version**: Versión del producto.
- **Creation Time**: Fecha y hora de creación.
- **Last Change Time**: Fecha y hora del último cambio.
- **Is Open**: Indica si el error está abierto.
- **Is Confirmed**: Indica si el error ha sido confirmado.
- **Is CC Accessible**: Acceso de la lista CC.
- **Is Creator Accessible**: Acceso del creador.
- **Type**: Tipo de error ('defect', 'enhancement', etc.).
- **Duplicate Of (dupeof)**: ID del bug original (si es duplicado).
- **Duplicates**: Lista de IDs que duplican a este bug.
- **Blocks**: IDs de errores bloqueados por este bug.
- **Depends On**: IDs de errores de los que depende este bug.
- **Regressions**: Errores introducidos por este bug.
- **Regressed By**: Errores que causaron este bug.
- **Comment Count**: Cantidad de comentarios.
- **Flags**: Lista de banderas asignadas.
- **Keywords**: Palabras clave asociadas.
- **Classification**: Categoría de clasificación.
- **Target Milestone**: Hito u objetivo para la corrección.
- **Performance Impact**: Impacto en el rendimiento.
- **Status**: Estado actual simplificado.
- **Rank**: Rango de prioridad.
- **Rank in Product**: Rango dentro del producto.
- **Groups**: Grupos con acceso al error.
- **Ally Review Project Flag**: Bandera de revisión de accesibilidad del proyecto.
- **Cab Review**: Indica si requiere revisión del Change Advisory Board.
- **Accessibility Review**: Indica si requiere revisión de accesibilidad.

## 2. Contributor Information Dataset
Actividad y perfil de los colaboradores.
- **Contributor Id**: Identificador único del colaborador (PK).
- **User Name**: Nombre del colaborador.
- **Created On**: Fecha de creación en el sistema.
- **Last Activity**: Fecha de la última actividad.
- **Commented On**: Número de comentarios realizados.
- **Permissions**: Permisos asignados.
- **Bugs Filed**: Cantidad de errores reportados.
- **Comments Made**: Total de comentarios en reportes.
- **Assigned To**: Errores asignados al colaborador.
- **Assigned To and Fixed**: Errores asignados y resueltos por él.
- **Patches Submitted**: Parches enviados.
- **Patches Reviewed**: Parches revisados.
- **QA Contact**: Casos donde actúa como contacto de QA.
- **Bugs Poked**: Veces que ha interactuado con errores.

## 3. Bug Report Comments
Detalle de los comentarios realizados en los reportes.
- **Comment Id**: Identificador único del comentario (PK).
- **Bug Report**: Indica si el comentario es sobre un reporte de error.
- **Time**: Fecha y hora de publicación.
- **Creator**: Usuario o email del autor del comentario.
- **Author Id**: ID único del autor.
- **Tags**: Etiquetas asociadas al comentario.
- **Bug Id**: ID del error al que pertenece el comentario.
- **Text**: Contenido textual del comentario.
