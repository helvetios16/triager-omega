# ¿Qué tan similar es la comparación OpenJ9 (pipeline nuevo) vs TriagerX?

> Resumen para tesis/defensa. Detalle completo en `comparacion-triagerx-openj9.md`
> y `cbr-recuperacion.md`.

Hay que separar dos sentidos de "similar": **qué tan justa es la comparación**
(metodología) y **qué tan cerca quedan los resultados**.

## 1. Metodología: comparación justa, "comparable pero no idéntica"

Para el head-to-head se reconstruyó el set de ~50 clases (`build_openj9_50.py`)
igualando las condiciones de TriagerX:

| Eje | ¿Igual que TriagerX? |
|---|---|
| Dataset (OpenJ9, `openj9_22112024.csv`) | ✅ el mismo |
| Nº de clases | ✅ ≈50 (51 = owners con ≥20 issues; azar 0.0196 ≈ su 0.020) |
| Formato del texto (`Bug Title:…\nBug Description:…`) | ✅ exacto |
| Split temporal (por `issue_number`, train < 17695 / test ≥ 17695) | ✅ misma lógica, sin solapamiento |
| Etiqueta (`owner`/fixer) | ✅ la misma |
| Métrica (Top-1 / MRR) | ✅ la misma |
| Datos del IBR (timelines de GitHub) | ✅ minados igual |

**Diferencias (caveats honestos):**
- Selección **exacta** de devs no idéntica (≥20 issues → 51 vs sus 50).
- Tamaño de test distinto (534 vs 382) por esa selección.
- **Sin split de validación** → el W_f se reporta como curva en test (W_f=0.7 es la
  elección principista de TriagerX; W_f=0.2 es el pico observado en test).

→ Justa en los ejes que importan, con diferencias menores documentadas. Los
absolutos del set viejo de **17 clases NO son comparables** (azar 0.059 vs 0.020);
por eso se hizo el de 50.

## 2. Resultados (50 clases, head-to-head)

| Componente | Top-1 |
|---|---|
| DeBERTa-solo (TriagerX) | 0.189 |
| DeBERTa-solo (nuestro) | 0.2322 |
| **CBR ensemble RoBERTa+DeBERTa (TriagerX)** | **0.270** |
| **CBR de recuperación zero-shot (nuestro)** | **0.2715** |
| CBR + IBR full (TriagerX) | **0.328** |
| Sistema completo (nuestro, W_f=0.2) | 0.2397 |

**Dos conclusiones clave:**
- El **CBR de recuperación zero-shot iguala el ensemble entrenado** de TriagerX
  (0.2715 vs 0.270) **sin entrenar nada**. Resultado fuerte y plenamente comparable.
- Donde **no** se llega es al sistema completo: su 0.328 vs nuestro 0.2397 (**~9 pp**).
  Esa brecha **no** es calidad del CBR (nuestro DeBERTa-solo 0.2322 ya supera al suyo
  0.189) — es que **su IBR sí ayuda a su ensemble**, mientras que a 50 clases
  (cola larga) nuestro IBR es débil y correlacionado y no suma al Top-1.

## En una frase

La comparación es **justa y apples-to-apples** en lo esencial (mismo dataset,
~50 clases, mismo split y texto), con caveats menores. En resultados, **igualamos
su CBR sin entrenar** (0.2715 ≈ 0.270); la diferencia restante hasta 0.328 viene de
que su ensemble+IBR rinde más en este régimen, no de una desventaja del método base.
