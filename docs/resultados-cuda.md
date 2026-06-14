# Resultados en CUDA (RTX 5060) — CBR y sistema completo CBR+IBR

Corrida de entrenamiento y evaluación del piloto migrada de la Mac (MPS) al equipo
con **NVIDIA GeForce RTX 5060 Laptop GPU (8 GB)** vía SSH. Fecha: 2026-06-13.

## Entorno

| Componente | Valor |
|---|---|
| GPU | RTX 5060 Laptop, 8 GB, compute capability `sm_120` (Blackwell) |
| PyTorch | `2.11.0+cu128` (CUDA 12.8) |
| Python | 3.12 |
| Dataset | Piloto 20 devs · train 5842 / val 1749 / test 1615 (9206 bugs etiquetados) |

> ⚠️ **Comparabilidad con el piloto previo en MPS.** Esta corrida usó **4 épocas**
> (default actual de `train.py`); el piloto anterior en MPS se entrenó con **5 épocas**.
> Por tanto las pequeñas diferencias frente al ~0.749 Hit@1 reportado en MPS no son
> apples-to-apples: pesan tanto el cambio de backend (CUDA/torch 2.11, fp32) como
> la época de menos. El **orden cualitativo** entre variantes sí se mantiene.

## Recipe estable (importante)

DeBERTa-v3 **diverge entrenando en `bf16`** en CUDA (el `grad_norm` explota a miles,
la loss salta a ~30 y el modelo colapsa a azar: Hit@1 ≈ 1/20). Es la misma fragilidad
numérica de la *disentangled attention* que provoca el `nan` en MPS.

**Recipe que reproduce el piloto: `fp32` + `adam_epsilon = 1e-4`** (NO bf16, NO 1e-8).
El `auto` de `resolve_runtime` ya lo aplica; en CUDA mantiene `batch=8` +
`gradient_accumulation=2` (eff. 16) para caber en 8 GB. Tiempo: ~7 min / 4 épocas.

## CBR-solo — comparación de variantes (§11.2.4)

Texto de entrada: `raw` (Summary [SEP] Product Component), `distilled` (vista destilada)
o `both` (concatenación de ambas).

### Test (1615 bugs)

| Variante | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | loss |
|---|---|---|---|---|---|---|
| **both** 🥇 | **0.7319** | 0.9195 | 0.9591 | 0.9926 | **0.8324** | 0.866 |
| raw 🥈 | 0.7127 | 0.9195 | 0.9511 | 0.9895 | 0.8197 | 0.908 |
| distilled 🥉 | 0.6656 | 0.9022 | 0.9517 | 0.9851 | 0.7899 | 1.031 |

### Val (1749 bugs)

| Variante | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| **both** 🥇 | 0.7490 | 0.9646 | 0.9903 | 0.8488 |
| raw 🥈 | 0.7267 | 0.9617 | 0.9891 | 0.8320 |
| distilled 🥉 | 0.6993 | 0.9651 | 0.9857 | 0.8161 |

**Lectura:** `both` gana en todo (test Hit@1 +1.9 pp vs raw, +6.6 pp vs distilled). El
destilado **solo** es la vista más débil: aporta como complemento (`both`), no como
reemplazo del crudo. Orden `both > raw > distilled` estable en val y test.

## Sistema completo — FS = NPS + W_f·NIS (Módulo 5)

CBR = `both`, IBR = `distilled`. `W_f` sintonizado por grid en **validación** y
aplicado a test sin re-tunear (sin fuga).

### Grid de W_f en val (criterio Hit@5)

| W_f | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| 0.0 (solo CBR) | 0.7490 | 0.9646 | 0.8487 |
| **0.2 (óptimo)** | **0.7856** | **0.9720** | **0.8697** |
| 0.7 (default TriagerX) | 0.7536 | 0.9686 | 0.8508 |

### Test con W_f = 0.2

| Modo | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|
| **CBR+IBR (FS)** 🏆 | **0.7622** | 0.9307 | 0.9628 | 0.9957 | **0.8535** |
| CBR-solo (both) | 0.7319 | 0.9195 | 0.9591 | 0.9926 | 0.8324 |
| IBR-solo | 0.6755 | 0.8836 | 0.9139 | 0.9443 | 0.7856 |

**Lectura:**
- La **fusión aditiva gana**: +3.0 pp Hit@1 (0.732 → 0.762) y +2.1 pp MRR sobre CBR-solo.
- El **IBR solo es más débil** (0.676) pero aporta como empujón `W_f·NIS`, no como
  reemplazo — exactamente el diseño de TriagerX (Ec. 8).
- `W_f = 0.2` (lejos del 0.7 de TriagerX) confirma que el aporte del IBR en Mozilla
  es **moderado**, consistente con las ablaciones previas.
- `Hit@10 = 0.9957`: el dev correcto casi siempre está en el top-10.

## Reproducir

```bash
# CBR (una variante)
uv run python -m triager_omega.cbr.train --text-mode both     # raw | distilled | both

# Sistema completo
uv run python -m triager_omega.modules.aggregator eval --split val  --grid --cbr-mode both
uv run python -m triager_omega.modules.aggregator eval --split test --w-f 0.2 --cbr-mode both
```
