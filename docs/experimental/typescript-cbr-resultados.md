# Resultados de Evaluación Experimental: CBR de Recuperación en TypeScript

**Dispositivo:** Apple Silicon (`mps`) | **Encoder:** `all-mpnet-base-v2` (`max_seq_len=384`)
**Dataset:** Train 9,831 issues · Test 1,115 issues · 40 clases

## Tabla Comparativa Head-to-Head

| Configuración | Top-1 (Hit@1) | Hit@5 | Hit@10 | MRR | Hiperparámetros |
|---|---|---|---|---|---|
| **TriagerX CBR (Paper publicado)** | 0.3240 | — | — | — | Ensemble 2 Transformers + 3 CNN (Entrenado) |
| **TriagerX RoBERTa-FCN** | 0.3190 | — | — | — | Clasificador denso individual |
| **raw_dense** | **0.3058** | 0.6099 | 0.7372 | 0.4471 | k=50, τ=4.0, vote=idf |
| **raw_hybrid** | **0.3354** | 0.6511 | 0.7740 | 0.4851 | k=50, τ=2.0, vote=idf |
| **distilled_dense** | **0.3247** | 0.5740 | 0.6888 | 0.4438 | k=50, τ=0.5, vote=idf |
| **distilled_hybrid** | **0.3354** | 0.6036 | 0.7309 | 0.4664 | k=50, τ=1.0, vote=idf |
| **both_dense** | **0.3022** | 0.6027 | 0.7274 | 0.4425 | k=50, τ=2.0, vote=idf |

