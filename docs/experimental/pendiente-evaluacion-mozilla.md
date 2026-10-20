# Tarea Pendiente: Evaluación de Mozilla / BugsRepo con Nueva Destilación y CBR de Recuperación

**Estado:** Pendiente / Por hacer  
**Fecha de registro:** 2026-10-20  
**Prioridad:** Alta (para completar la trilogía de evaluación experimental en la tesis)

---

## 1. Contexto y Justificación

* En **OpenJ9** y **TypeScript**, la nueva destilación semántica (DeepSeek Flash con taxonomía formal `[FL][SY][CP]`) y la recuperación zero-shot (densa e híbrida BM25+MPNet) lograron superar los techos de rendimiento del estado del arte (TriagerX publicado y re-ejecución honesta).
* En **Mozilla / BugsRepo**, la destilación solo se había probado en la arquitectura antigua del **clasificador supervisado DeBERTa** (arquitectura ya descartada), y el recuperador CBR zero-shot únicamente se corrió sobre **texto crudo** (alcanzando Hit@1 de 0.7176 y 0.7337 con IBR).
* **Todavía no se ha probado la nueva destilación semántica especializada ni el recuperador híbrido sobre el corpus de Mozilla.**

---

## 2. Objetivos de la Tarea

1. **Diseñar el prompt especializado de ingeniería para Mozilla / Bugzilla:**
   - Adaptar la taxonomía diagnóstica a subsistemas de Mozilla (Gecko layout, SpiderMonkey JS engine, DOM, IPC, Networking, Graphics/WebRender).
   - Mantener el formato estándar de salida estructurada `[FL] ... [SY] ... [CP] ...`.
2. **Generar la destilación del corpus:**
   - Procesar los splits de train (5,842 issues) y test (1,615 issues) mediante el script basado en Bun y la API de DeepSeek Flash (`thinking: { type: "disabled" }`, cache context).
   - Generar los datasets correspondientes en `artifacts/mozilla/distilled/` (`mozilla_{train,test}_distilled.csv`, `provenance`, etc.).
3. **Evaluación experimental sistemática:**
   - Ejecutar el barrido de hiperparámetros ($k \in \{5, 10, 15, 20, 30, 50\}$, $\tau \in \{1.0, 2.0, 4.0\}$).
   - Comparar:
     - `raw_dense` vs `distilled_dense`.
     - `distilled_hybrid` (BM25 + MPNet) vs `raw_hybrid`.
     - Impacto de la normalización del voto (`sum` vs `idf`).
4. **Fusión con IBR bajo texto destilado:**
   - Medir si la destilación semántica potencia la complementariedad con el IBR en Mozilla (donde el IBR crudo ya aportaba +1.6 pp).
   - Analizar el impacto en Top-1, Top-5, Top-10 y MRR.

---

## 3. Scripts y Recursos Relacionados
- Script de referencia para destilación: `scripts/distill_deepseek_openj9.ts`, `scripts/distill_deepseek_ts.ts`.
- Script de evaluación CBR en Mozilla: `scripts/cbr_retrieval_pilot.py`.
- CSVs fuente: `data/mozilla_pilot_train.csv`, `data/mozilla_pilot_test.csv` (o equivalentes en `data/`).
