# Corrección: "la destilación empeora" era un hallazgo de la arquitectura descartada

**Fecha:** 2026-09-16. Corrige la lectura de §1 de
[estado-tesis.md](estado-tesis.md) a la luz de
[piloto-destilacion-agy-openj9.md](piloto-destilacion-agy-openj9.md).

## Qué decía estado-tesis.md

> "La hipótesis original era: *raw + distilled entrena mejor el CBR → mejores
> resultados*. Los datos del piloto Mozilla no la respaldan limpiamente
> [...] distilled-solo es el PEOR."

Esa conclusión (Hit@1 raw 0.7517 > distilled 0.7313) viene de un **CBR
clasificador DeBERTa** afinado sobre Mozilla (20 devs, cabeza densa).

## Por qué ya no aplica

El clasificador DeBERTa **es una arquitectura descartada**. La arquitectura
vigente del proyecto es el **CBR de recuperación zero-shot** (encoder
`all-mpnet-base-v2`, `scripts/cbr_retrieval_openj9.py`), no un clasificador
entrenado. `piloto-destilacion-agy-openj9.md` lo dice explícitamente: *"La
destilación solo se había probado antes con el clasificador DeBERTa
entrenado (arquitectura ya descartada), nunca con el recuperador."*

Comparar `raw` vs `distilled` en el clasificador mide el efecto de la
destilación **sobre una arquitectura que ya no se usa**. No es evidencia
sobre si la destilación ayuda o no al sistema real (MPNet + recuperación).

## Qué muestra la validación con la arquitectura real

`piloto-destilacion-agy-openj9.md`, validación a escala completa (OpenJ9,
3348 train / 534 test, `--max-seq-len 384`, barrido k×τ completo):

| | Top-1 | MRR | Hit@10 |
|---|---|---|---|
| Crudo (mejor punto) | 0.2715 | 0.4078 | 0.6779 |
| **Destilado (mejor punto)** | **0.2884** | **0.4387** | **0.7472** |
| Δ | **+1.69 pp** | **+3.09 pp** | **+6.93 pp** |

Destilado gana en **18/18 celdas** del barrido (no solo en el punto
campeón), con la ventaja más grande en k chico (k=5: +6.74 pp Top-1) y más
angosta en k grande.

**Mecanismo identificado (no es solo "menos ruido semántico"):** con el
default `--max-seq-len 256`, el texto crudo se trunca antes de completar el
contexto nativo de MPNet y pierde ~2pp Top-1 (0.2547 vs 0.2715 con 384). El
texto destilado (`[FL]/[SY]/[CP]`) es corto y da igual con 256 o 384 — parte
de la ventaja de destilar es **evitar el truncamiento** en reportes largos,
no solo limpiar ruido.

## Conclusión

- El hallazgo "distilled-solo es el peor" de `estado-tesis.md` §1 **sigue
  siendo cierto para el clasificador DeBERTa**, pero esa arquitectura no es
  la que se reporta en la tesis final.
- Para la arquitectura vigente (recuperación + MPNet), la evidencia
  validada dice lo contrario: **destilar ayuda de forma consistente**,
  especialmente con pocos vecinos (k chico) o reportes largos que se
  truncarían en crudo.
- `estado-tesis.md` debería marcarse como referido a una arquitectura
  descartada en lo que toca a destilación, y el número citable en la tesis
  para "destilación vs. crudo" es el de `piloto-destilacion-agy-openj9.md`,
  no el de Mozilla/DeBERTa.

## Pendiente

- **Decisión tomada:** por limitaciones (de cómputo/latencia para destilar
  corpus completos), la destilación se hará con una **API de LLM** (p. ej.
  `deepseek-v4-flash` vía NVIDIA API) en vez de Gemma 4 E4B IT QAT local.
  Ya no está pendiente correr Gemma local — el backend de la propuesta
  cambió. La dirección del efecto (destilar ayuda) se ha sostenido a
  través de varios backends de API distintos (agy/Gemini, grok,
  deepseek-flash), lo cual da confianza en que no depende de un backend
  específico.
- `bun-nvidia` (el pipeline TypeScript de este repo hermano) es un
  **sandbox de pruebas** para ese backend de API — no es todavía la corrida
  oficial de validación.
- Falta correr el experimento formal (raw vs. distilled, arquitectura de
  recuperación) sobre el corpus **TypeScript**, el tercer régimen — sigue
  pendiente según `openj9-mejoras-ibr-resultados.md`.
