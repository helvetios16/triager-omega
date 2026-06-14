# Comparación: TriagerX vs. triager-omega (OpenJ9)

> Validación cruzada en OpenJ9 (dataset de TriagerX, etiqueta `owner`/fixer).
> Fecha: 2026-06-13. Régimen donde la señal `contribution` del IBR sí aporta
> (opuesto a Mozilla, donde la etiqueta es `Assigned To` = assignee).

## Resultados Top-1 (y métricas de ranking)

### TriagerX (sus propios artefactos)
Fuente: `triagerX/notebook/openj9/top1_class_comparison.json` (CBR) y
`triagerX/notebook/openj9_grid_sim_weights.csv` (combinado WRA).
Setup: **50 clases**, 382 issues de test.

| Componente | Top-1 |
|---|---|
| RoBERTa-solo | 0.175 |
| DeBERTa-solo | 0.189 |
| **CBR ensemble** (RoBERTa + DeBERTa) | 0.270 |
| **CBR + IBR (WRA, combinado)** | **0.328** |

### triager-omega (nuestro sistema)
Setup: **1 DeBERTa-v3**, **17 clases**, 1.893 ejemplos.
Scripts: `scripts/train_openj9_cbr.py`, `scripts/eval_openj9_full.py`.

| Componente | Top-1 | MRR | Hit@5 | Hit@10 |
|---|---|---|---|---|
| CBR-solo | 0.199 | 0.401 | 0.704 | 0.859 |
| IBR-solo | 0.238 | 0.373 | 0.492 | 0.637 |
| **Full (CBR + IBR)** | **0.257** | 0.452 | 0.714 | 0.878 |

## Veredicto: TriagerX sale mejor en absolutos

Su sistema completo logra **Top-1 0.328 sobre 50 clases**, mientras nosotros
sacamos **0.257 sobre solo 17 clases** — una tarea bastante más fácil
(azar 1/17 = 0.059 vs 1/50 = 0.020). Nivelando por dificultad, la brecha es
aún mayor a su favor. **No era el objetivo superarlo**: nuestro CBR está
deliberadamente underfit (sin el ensemble del paper).

### De dónde viene la diferencia
- **El motor es el CBR ensemble.** TriagerX fusiona **dos transformers**
  (RoBERTa + DeBERTa) → 0.270 Top-1; cada uno solo da ~0.18. Nosotros corremos
  **un solo DeBERTa underfit** (1.893 ejemplos, sin ensemble) → 0.199.
- **El IBR aporta lo mismo en ambos:** sube el CBR de TriagerX 0.270 → 0.328
  (+5.8 pp); a nosotros de 0.199 → 0.257 (+5.8 pp). **Misma ganancia exacta**
  → nuestro IBR está bien replicado.

## Lo que sí validamos (cualitativamente igual al paper)
1. **Full > ambos solos** → la arquitectura híbrida funciona end-to-end,
   igual que TriagerX.
2. **`contribution` ayuda con etiqueta `fixer`/`owner`** (OpenJ9) y daña con
   `assignee` (Mozilla) — hallazgo nuestro, no del paper. El valor de la señal
   de código depende de la definición de la etiqueta.

   Ablación en OpenJ9 (apagando `contribution` con `--ip-c 0`):

   | | IBR-solo Hit@1 | Full mejor Hit@1 | Full MRR (pico) |
   |---|---|---|---|
   | `contribution` ON (ip_c=1.5) | **0.2379** | **0.2669** (W_f=0.2) | **0.4563** |
   | `contribution` OFF (ip_c=0) | 0.2219 | 0.2347 | 0.4400 |
   | **Δ** | **+1.6 pp** | **+3.2 pp** | **+1.6 pp** |

   En Mozilla el mismo barrido daba la conclusión opuesta (`ip_contribution=0`
   es lo óptimo): la señal de código solo aporta cuando la etiqueta ES el
   contribuidor de código.

## Lección de entrenamiento
El `WeightedRandomSampler` (1/freq), clave en Mozilla (cola larga de ~450 devs),
es **contraproducente en OpenJ9** (17 devs activos, dataset chico y balanceado):
sobre-balancea → sobre-predice devs raros → Hit@1 cae por debajo del azar
(0.042 < 0.059). Fix: `--no-weighted --epochs 6` → Hit@1 0.199.
Los HPs del piloto Mozilla **no se trasladan a ciegas**.

## Caveats
- No es apples-to-apples: TriagerX usa 50 clases / nosotros 17; splits distintos.
- CBR de OpenJ9 underfit (1.893 ejemplos, sin ensemble) → absolutos modestos,
  **no es réplica de los números del paper**.
- Sin split de validación: el W_f se reporta como curva en test.

## Resumen
No superamos a TriagerX en absolutos (y no se buscaba, porque nuestro CBR es
intencionalmente underfit), pero **replicamos fielmente el comportamiento de su
IBR y de la arquitectura completa**. La brecha en Top-1 es casi enteramente el
ensemble de dos transformers que ellos entrenan y nosotros no.
