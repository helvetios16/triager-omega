# Estado y pendientes tras la refactorización

Este documento resume el estado al terminar la consolidación compartida y la prueba real de CSV → runner → dry-run de TypeScript y OpenJ9.

El [diagnóstico original](organizacion-codigo-pendiente.md) se conserva como antecedente: sus afirmaciones sobre código duplicado o ausencia de tests describen el estado previo, no necesariamente el actual.

## 1. Qué quedó implementado

- [x] Pruebas de caracterización de normalización, métricas, votación, BM25, CBR e IBR.
- [x] Rutas Mozilla/piloto deterministas, lecturas legacy explícitas y escrituras canónicas.
- [x] Recuperador `CbrRetriever` compartido por los cuatro benchmarks migrados: OpenJ9 standalone y full, TypeScript y Mozilla piloto.
- [x] Motor IBR reutilizable mediante `IbrConfig`, `InteractionIndex` e `IbrScorer`, con adaptadores públicos para los consumidores activos.
- [x] Funciones compartidas de fusión, normalización y gating.
- [x] Archivo de `TriagerAggregator` histórico y eliminación de su CLI/export activo roto; marcado del surrogate de balanceo como histórico.
- [x] Infraestructura DeepSeek compartida: transporte/reintentos, workers/breaker, persistencia, reanudación, validación/fallback y contabilidad.
- [x] Runners TypeScript/OpenJ9 con `CONFIG` y `run(config, dependencies)`; la CLI existente delega en esa ruta.
- [x] Pruebas de checkpoints, reanudación, duplicados por corpus, tokens/costos, fallbacks, concurrencia y breaker.
- [x] Carga de CSV real en ambos runners y dry-run sin llamadas HTTP ni escritura de resultados, sin simular el parser ni inyectar filas.

Se conservaron las diferencias de los corpus y los comportamientos caracterizados. Esta refactorización no equivale a una corrección metodológica ni a una reproducción de los experimentos de la tesis.

## 2. Pendientes de cierre técnico

### Prioridad 1 — Dependencias TypeScript reproducibles

- [ ] Declarar las dependencias TypeScript/Bun del proyecto y fijar sus versiones mediante un manifiesto y lockfile adecuados.
- [ ] Comprobar la ejecución desde un entorno limpio, sin depender de la caché personal de Bun.
- [ ] Documentar la forma de ejecutar los runners y las pruebas.

**Hallazgo confirmado:** `csv-parse/sync` existe en la caché local de Bun y funciona en la ejecución normal. El import directo dentro de `bun test` y el build sin externalizar la dependencia no lo resolvieron en el entorno revisado. Eso no significa que las destilaciones anteriores no pudieran ejecutarse.

Las pruebas de integración lanzan un proceso normal de Bun con `--prefer-offline` para recorrer el cargador real. Actualmente requieren que el paquete esté disponible en el entorno. No se instaló ninguna dependencia durante esta revisión.

### Prioridad 2 — Completar validaciones de herramientas y persistencia

- [ ] Establecer y ejecutar una comprobación completa de tipos TypeScript. Bun test/build no sustituyen un typecheck.
- [ ] Validar la conversión real a Parquet en un directorio temporal, con datos sintéticos y sin API.
- [ ] Decidir por separado si se mantiene o se cambia el fallo silencioso de conversión a Parquet. Se preservó por compatibilidad; no debe modificarse incidentalmente.

Las pruebas existentes ejercitan checkpoints JSONL, CSV y provenance con archivos temporales reales. La conversión a Parquet se inyecta/simula en las pruebas: no se debe dar por validada la integración real.

## 3. Validación experimental pendiente

- [ ] Ejecutar los benchmarks migrados con los datos, modelos y configuraciones originales, cuando se autorice el uso de esos recursos.
- [ ] Comparar resultados con las referencias anteriores: métricas, orden de candidatos, splits, defaults y artefactos relevantes.
- [ ] Registrar las configuraciones y resultados de esa comparación, distinguiendo equivalencia de implementación y cambios metodológicos.
- [ ] Si se requiere validar DeepSeek en vivo, acordar antes corpus, muestra, modelo, presupuesto y destino de los artefactos.

**No realizado:** llamadas reales a DeepSeek, descargas de modelos, regeneración de artefactos experimentales ni reproducción de las métricas de la tesis. Las pruebas sintéticas/offline no demuestran por sí solas equivalencia experimental completa.

## 4. Siguiente funcionalidad: Mozilla moderno

El runner moderno de destilación Mozilla **todavía no está creado**.

- [ ] Definir las columnas de entrada, identidad de caso, splits y rutas de salida.
- [ ] Definir/revisar el prompt especializado Bugzilla/Mozilla, ejemplos si corresponden y fallback.
- [ ] Crear el adaptador sobre la infraestructura DeepSeek común, sin copiar un tercer pipeline de HTTP, workers o persistencia.
- [ ] Añadir pruebas de mapeo, prompt, fallback, reanudación y dry-run con CSV real.
- [ ] Acordar después cualquier corrida con API y su presupuesto.
- [ ] Integrar los destilados modernos en la evaluación CBR y, si aporta señal complementaria, IBR.

La configuración debe seguir siendo directa en código mediante `CONFIG` y funciones invocables. No hace falta introducir un nuevo framework de CLI.

## 5. Decisiones metodológicas separadas

Estos asuntos no se corrigieron durante la reorganización, para evitar cambios experimentales ocultos:

- [ ] Definir explícitamente la política temporal de IBR: con `t_now=None`, el comportamiento caracterizado permite interacciones futuras sin decaimiento.
- [ ] Definir qué hacer con fechas `NaT`, que pueden propagar `NaN`.
- [ ] Revisar el uso de `creation_time` como proxy temporal de asignación y qué garantía anti-fuga se necesita realmente.
- [ ] Evaluar cualquier política nueva en un experimento separado, conservando una referencia del comportamiento anterior.

No debe afirmarse que la refactorización garantiza ausencia de fuga temporal.

## 6. Mejoras opcionales, no bloqueantes

- [ ] Evaluar contratos canónicos adicionales para tablas de casos/interacciones, si reducen duplicación real entre adaptadores.
- [ ] Compartir más lógica de barridos y persistencia de resultados de benchmarks cuando compense, preservando las diferencias por corpus.
- [ ] Revisar comentarios y documentación histórica que todavía puedan confundirse con instrucciones vigentes.

No es necesario materializar toda la estructura de carpetas orientativa del diagnóstico original. El objetivo es compartir lógica útil, no hacer una reescritura completa.

## 7. Evidencia de la última validación

| Comprobación | Resultado |
| --- | --- |
| `bun test tests/*.test.ts` | 40 pruebas pasan |
| `uv run --with pytest pytest tests/ -q` | 67 pruebas pasan; avisos de deprecación SWIG |
| `git diff --check` | Sin errores |
| Prompts y ejemplos few-shot TypeScript/OpenJ9 frente a HEAD original | Sin cambios |
| CSV real → runner → dry-run, ambos corpus | Correcto; sin HTTP ni archivos de resultados |
| Build Bun con `csv-parse/sync` externo | Correcto; no constituye validación completa de dependencias ni tipos |

Pruebas de integración CSV: `tests/deepseek_csv_dry_run.test.ts`.

Los cambios permanecen sin commit ni staging. Antes de preparar un commit, revisar también los archivos fuente y tests no rastreados; no añadir indiscriminadamente artefactos ni cambios ajenos.

## Orden recomendado para continuar

1. Fijar dependencias TypeScript/Bun y validar tipos/conversión real a Parquet.
2. Cerrar la comparación experimental autorizada de los runners/benchmarks existentes, o dejar explícitamente registrada como pendiente.
3. Implementar el adaptador Mozilla moderno sobre la infraestructura común.
4. Tratar las correcciones temporales/metodológicas como trabajo independiente.
