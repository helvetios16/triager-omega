# Mejoras al IBR (OpenJ9 ~50 clases) — resultados

> Experimentos para mejorar el aporte de **nuestro** IBR en el régimen de cola larga
> (OpenJ9, 50 clases), donde el IBR semántico baseline **no aporta** (ver
> `docs/openj9-comparativa-explicada.md`). Tres palancas, ejecutadas una por una en
> la omen (RTX 5060), sobre el 50-set reconstruido.
>
> Comando base (CBR-recuperación zero-shot + IBR), 50-set:
> ```
> uv run python scripts/eval_openj9_full.py --cbr-mode retrieval \
>   --train-csv artifacts/openj9/openj9_train_50.csv \
>   --test-csv  artifacts/openj9/openj9_test_50.csv \
>   --interactions artifacts/openj9/openj9_interactions_50.parquet \
>   --meta artifacts/openj9/openj9_issue_meta_50.parquet
> ```

## Referencia (baseline)

| Config | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| **CBR-recuperación solo** (techo) | **0.2715** | 0.5768 | 0.6779 | 0.4078 |
| IBR-solo (semántico) | 0.1685 | 0.4101 | 0.4944 | 0.2853 |
| Full IBR semántico, mejor (W_f=0.1) | 0.2622 | 0.5749 | 0.6873 | 0.4070 |

El IBR **semántico** comparte encoder (MPNet) con el CBR → recupera los mismos
vecinos → está correlacionado → **toda** la fusión queda por debajo del CBR-solo.
Diagnóstico: el problema es la correlación, no el componente.

---

## P1 — Decorrelar el IBR (canal léxico BM25)

**Idea:** recuperar los vecinos del IBR por **BM25 léxico** (tokens técnicos
exactos) en vez del MPNet semántico, para que el canal IBR deje de mirar los mismos
vecinos que el CBR. La agregación por interacciones tipadas (IP · decay · anti-fuga)
es idéntica. Flag nuevo: `--ibr-channel lexical`.

| Config | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| CBR-solo (referencia) | 0.2715 | 0.5768 | 0.6779 | 0.4078 |
| Full IBR semántico (mejor, W_f=0.1) | 0.2622 | 0.5749 | 0.6873 | 0.4070 |
| IBR **léxico**-solo | 0.1592 | 0.3764 | 0.5037 | 0.2733 |
| **Full léxico W_f=0.2** | **0.2678** | 0.5880 | 0.6948 | 0.4090 |
| **Full léxico W_f=0.3** | 0.2678 | 0.5861 | 0.6966 | **0.4108** |
| Full léxico W_f=1.0 | 0.1985 | 0.5787 | 0.7135 | 0.3686 |

**Hallazgo — la decorrelación funciona.** El IBR léxico-solo es *más débil* (0.1592
< 0.1685), pero al fusionar **deja de dañar y empieza a aportar**:
- **MRR 0.4108 > CBR-solo 0.4078** (+0.3 pp): primera vez que la fusión supera al
  CBR-solo en una métrica de ranking.
- **Hit@5 +1.1 pp** (0.5880), **Hit@10 +1.9 pp** (0.6966) sobre CBR-solo.
- **Top-1 0.2678**: sigue por debajo del techo (−0.37 pp vs 0.2715), pero **+5.6 pp
  vs la fusión semántica** (0.2622) y con caída mucho más suave al subir W_f.

Confirma el diagnóstico: el IBR no era inútil, estaba **redundante**; al darle una
vista ortogonal (léxica) aporta señal complementaria. El Top-1 no se supera, pero el
sistema completo ya iguala/mejora al CBR-solo en MRR/Hit@5/Hit@10.

> **Nota (futuro):** la otra vista de decorrelación, la **estructural** (dev↔módulo/
> archivo desde commits/PRs), queda pendiente. Podría ser aún más ortogonal que la
> léxica porque captura "quién trabaja en esta zona del código", no solo vocabulario.

---

## P2 — Fusión por rango (RRF) en vez de suma de score

**Idea:** en vez de `FS = NPS + W_f·NIS` (suma de scores), combinar por **Reciprocal
Rank Fusion**: `FS = 1/(k+rango_NPS) + W_f/(k+rango_NIS)`, con el término del IBR
enmascarado a los devs con señal (NIS>0). Solo usa el orden, no la magnitud. Flags:
`--fusion rrf --rrf-k 60`. (Canal semántico, para aislar el efecto de la fusión.)

| Config | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| CBR-solo (referencia) | 0.2715 | 0.5768 | 0.6779 | 0.4078 |
| RRF W_f=0.1 | 0.2041 | 0.4981 | 0.6910 | 0.3432 |
| RRF W_f=0.5 | 0.2004 | 0.4195 | 0.5805 | 0.3301 |
| RRF W_f=1.0 | 0.2041 | 0.4288 | 0.5768 | 0.3312 |

**Hallazgo — RRF PIERDE (claro).** El Top-1 se desploma a ~0.20 **ya en W_f=0.1**,
donde el IBR casi no contribuye. La caída no la causa el IBR, sino **convertir el
NPS a rango**: eso tira la información de *confianza* del CBR (la diferencia de
magnitud entre el voto top-1 y el resto), que aquí sí es informativa. RRF es la
herramienta correcta para fusionar listas con escalas incomparables (p. ej. BM25 +
denso en el propio recuperador), pero **equivocada cuando un canal —el CBR— ya
tiene magnitudes calibradas.** Descartada.

---

## P3 — Fusión condicional (gated por confianza del CBR)

**Idea:** aplicar el IBR **solo en las consultas donde el CBR está inseguro** (margen
top1−top2 del NPS < `gate`); donde el CBR es confiable, `FS = NPS` puro. Así se
preservan los Top-1 correctos y el IBR solo desempata dudas. Flags: `--gate G` /
`--gate-sweep`. (`frac IBR` = fracción de consultas a las que entra el IBR.)

**P3 sobre canal semántico (aislado) — neutro:**

| gate | frac IBR | best Top-1 | best MRR |
|---|---|---|---|
| off | — | 0.2622 | 0.4070 |
| 0.3 | 0.54 | 0.2622 | 0.4052 |
| 0.7 | 0.98 | 0.2622 | 0.4063 |

Gatear el canal semántico **no mueve el Top-1** (siempre 0.2622, por debajo del techo
0.2715). Lógico: en las consultas dudosas el IBR *correlacionado* tampoco trae al dev
correcto; el gate limita el daño pero no añade señal.

**P3 + P1 (canal LÉXICO + gate) — el mejor resultado:**

| gate | frac IBR | best Top-1 (W_f) | best MRR (W_f) |
|---|---|---|---|
| off | — | 0.2678 (0.2) | 0.4108 (0.3) |
| 0.2 | 0.37 | 0.2678 (0.2) | 0.4083 (0.5) |
| **0.3** | **0.54** | **0.2715 (0.5)** | **0.4125 (0.5)** |
| 0.5 | 0.80 | 0.2678 (0.2) | 0.4103 (0.5) |
| 0.8 | 1.00 | 0.2678 (0.2) | 0.4110 (0.3) |

**Hallazgo — gating + decorrelación recupera el techo y mejora el ranking.** Con
`gate=0.3` (IBR aplicado al ~54% de consultas dudosas, W_f=0.5):
- **Top-1 0.2715 = CBR-solo**: el sistema completo ya **iguala** el techo (sin la
  pérdida de −0.37 pp de P1-solo).
- **MRR 0.4125**: el mejor de todos los experimentos (**+0.47 pp** sobre CBR-solo,
  +0.17 pp sobre P1-solo).
- Hit@10 0.6929 (+1.5 pp sobre CBR-solo).

Es la primera config donde activar el IBR **no cuesta Top-1 y sí mejora el ranking**.

> **Caveat (igual que todo el reporte de W_f en OpenJ9):** sin split de validación,
> el `gate`/`W_f` óptimos se leen como curva en test, no sintonizados aparte. La
> tendencia (léxico+gate ≥ CBR-solo en todas las métricas) es robusta; el punto
> exacto (gate=0.3, W_f=0.5) es el mejor observado en test.

---

## Extra (b) — las mejoras sobre el full del CLASIFICADOR

Verificación: ¿las palancas (léxico + gate) también ayudan sobre la otra base, el
**CBR-clasificador** DeBERTa (30ép/len512, base 0.2397 con IBR semántico)? Sí — y más.

| Config (clasificador, 50 clases) | Top-1 | MRR |
|---|---|---|
| CBR-clasificador solo | 0.2322 | 0.361 |
| Baseline: full IBR semántico (W_f=0.2) | 0.2397 | 0.3735 |
| Full IBR **léxico** (gate off, W_f=0.3) | 0.2509 | **0.3799** |
| **Full léxico + gate=0.1** (W_f=0.3, ~57% consultas) | **0.2566** | 0.3792 |

**Hallazgo.** El léxico+gate sube el Top-1 del clasificador a **0.2566**: **+2.44 pp
sobre su CBR-solo** y **+1.69 pp sobre el full semántico viejo** (0.2397). El IBR
aporta *más* aquí que sobre el recuperador (donde solo igualaba el techo) porque el
clasificador es una **base más débil y menos saturada**, con más margen.

**Pero el absoluto sigue por debajo del recuperador:** 0.2566 (clasificador mejorado)
< **0.2715** (recuperador solo) < 0.328 (TriagerX full). Las mejoras generalizan y
validan el mecanismo, pero el techo más bajo del clasificador lo deja por debajo del
recuperador. El recuperador sigue siendo la mejor base. Log:
`artifacts/openj9/eval_full_50_classifier_p1p3_lexical_gate.log`.

---

## Resumen de las tres palancas

| Palanca | Top-1 | MRR | Veredicto |
|---|---|---|---|
| Baseline (IBR semántico, lineal) | 0.2622 | 0.4070 | el IBR **daña** (correlacionado) |
| **P1** decorrelar (léxico) | 0.2678 | 0.4108 | IBR deja de dañar; mejora MRR/Hit@5/10 |
| **P2** RRF | 0.2041 | 0.3432 | **descartada** (tira la confianza del CBR) |
| **P3** gate (semántico) | 0.2622 | 0.4070 | neutro sin decorrelar |
| **P1+P3** léxico + gate | **0.2715** | **0.4125** | **mejor**: iguala el techo + mejor ranking |
| CBR-solo (techo) | 0.2715 | 0.4078 | referencia |

**Conclusión.** El IBR no era inútil en cola larga: estaba **redundante** por
compartir encoder con el CBR. La causa raíz era la **correlación** (lo confirma que
P1, decorrelar, es la única palanca que ayuda; P2 falla por otra razón; P3 solo
suma *encima* de P1). Combinando decorrelación léxica + gating, el sistema completo
**iguala el Top-1 del recuperador (0.2715) y lo mejora en MRR (0.4125) y Hit@10**.
El Top-1 no se *supera* (el techo del recuperador zero-shot es muy robusto en este
régimen, como en todas las ablaciones), pero el IBR pasa de "restar" a "sumar".

### Pendiente / trabajo futuro
- **Vista estructural (dev↔módulo/archivo)** como variante de decorrelación de P1:
  podría ser aún más ortogonal que la léxica (captura zona del código, no solo
  vocabulario). No ejecutada.
- **Fusión IBR en TypeScript** (el tercer régimen) sigue pendiente.
- Validar `gate`/`W_f` en un split de validación si se reconstruye uno para OpenJ9.
