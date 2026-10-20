<!-- Merged from: ollama-remoto.md, gecko-dev-mining-status.md -->

# 05 — Infraestructura y Setup

## Tabla de contenidos

1. [Ollama remoto: destilar desde la Mac usando la GPU de la laptop (RTX 5060)](#1-ollama-remoto-destilar-desde-la-mac-usando-la-gpu-de-la-laptop-rtx-5060-ollama-remotomed)
   - [1.1 Arquitectura](#11-arquitectura)
   - [1.2 Configuración del servidor (laptop RTX 5060)](#12-en-la-laptop-rtx-5060)
   - [1.3 Configuración del cliente (Mac)](#13-en-la-mac-cliente)
   - [1.4 Notas y troubleshooting](#14-notas-y-troubleshooting)
2. [Estado de la minería de gecko-dev (señal `contribution` del IBR)](#2-estado-de-la-minería-de-gecko-dev-señal-contribution-del-ibr-gecko-dev-mining-statusmd)
   - [2.1 Qué se hizo](#21-qué-se-hizo)
   - [2.2 El matiz: la señal de código daña en Mozilla](#22-el-matiz-hallazgo-de-las-ablaciones)
   - [2.3 Decisión aplicada](#23-decisión-aplicada)
   - [2.4 Por qué NO se borra la minería](#24-por-qué-no-se-borra-la-minería)
   - [2.5 Validación cruzada en OpenJ9](#25-validación-cruzada-en-openj9-2026-06-13)
   - [2.6 Sistema completo OpenJ9 (CBR + IBR) end-to-end](#26-sistema-completo-openj9-cbr--ibr-end-to-end)
   - [2.7 Cómo reproducir / re-evaluar](#27-cómo-reproducir--re-evaluar)

---

## 1. Ollama remoto: destilar desde la Mac usando la GPU de la laptop (RTX 5060) (`ollama-remoto.md`)

Cómo correr el LLM de destilación (Gemma) en la **laptop con RTX 5060 8GB** y que la
**Mac** le envíe los bugs y reciba el JSON destilado. No requiere cambios de código:
solo configurar la red y el `.env`.

### 1.1 Arquitectura

```
┌─ Laptop RTX 5060 (servidor) ─┐         ┌─ Mac (cliente) ─────────────┐
│  Ollama + gemma4:e4b-it-qat  │◄───LAN──│  scripts/run_distillation.py│
│  la GPU hace la inferencia   │  HTTP   │  manda el texto del bug,    │
│  escucha en :11434           │────────►│  recibe el JSON destilado   │
└──────────────────────────────┘         └─────────────────────────────┘
```

La Mac orquesta (lee parquets, arma el prompt, cachea); la laptop solo corre el modelo.
Por la red solo viajan unos KB de texto por bug → hasta WiFi basta.

---

### 1.2 En la laptop (RTX 5060)

#### 1.2.1 Instalar Ollama y el modelo

```bash
# instalar Ollama desde https://ollama.com/download
ollama pull gemma4:e4b-it-qat
```
> **Usar `e4b-it-qat`, NO el `12b`.** El `e4b` (4B eficiente) cuantizado ocupa ~3-4 GB
> y entra cómodo en 8 GB. El `12b` necesita ~8-9 GB y se queda sin margen.

#### 1.2.2 Hacer que Ollama escuche en la red (no solo en localhost)

**Windows:**
```powershell
setx OLLAMA_HOST "0.0.0.0:11434"
# cerrar y reabrir Ollama (o reiniciar el servicio) para que tome la variable
```

**Linux:**
```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
# (o añadir Environment="OLLAMA_HOST=0.0.0.0:11434" al servicio systemd)
```

#### 1.2.3 Abrir el puerto 11434 en el firewall

**Windows (PowerShell como admin):**
```powershell
New-NetFirewallRule -DisplayName "Ollama" -Direction Inbound -LocalPort 11434 -Protocol TCP -Action Allow
```

**Linux (ufw):**
```bash
sudo ufw allow 11434/tcp
```

#### 1.2.4 Anotar la IP local de la laptop

```bash
# Windows:
ipconfig          # buscar "Dirección IPv4", ej. 192.168.1.50
# Linux/Mac:
ip addr           # o: hostname -I
```

---

### 1.3 En la Mac (cliente)

#### 1.3.1 Apuntar el `.env` a la laptop

Editar `.env`: comentar la línea LOCAL y descomentar la REMOTO con la IP real:
```
DISTILL_BACKEND=ollama
OLLAMA_MODEL=gemma4:e4b-it-qat
# OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_BASE_URL=http://192.168.1.50:11434/v1   # <-- IP de la laptop
```

#### 1.3.2 Verificar conexión

```bash
curl http://192.168.1.50:11434/v1/models        # debe listar gemma4:e4b-it-qat
```

#### 1.3.3 Probar y correr la destilación

```bash
uv run python scripts/run_distillation.py --smoke 5   # prueba 5 bugs, imprime
uv run python scripts/run_distillation.py             # batch completo (reanudable)
```

---

### 1.4 Notas y troubleshooting

**Notas:**
- **Ambas máquinas en la misma red.** La Mac y la laptop deben estar en la misma LAN/WiFi.
- **Velocidad esperada:** la RTX 5060 ≈ 0.3-0.8 s/bug vs ~1.5 s en MPS → el batch de
  9.364 bugs baja de ~4 h a **~1-2 h**.
- **Seguridad:** `0.0.0.0` expone Ollama a toda la red local **sin contraseña**. Está
  bien en red de casa; NO lo hagas en una red pública/compartida.
- **Reanudable:** si se corta la conexión, `run_distillation.py` retoma desde el cache
  (`artifacts/pilot/distillations.parquet`) al volver a ejecutarlo.
- **Bonus — entrenamiento:** esa misma laptop con CUDA puede entrenar el DeBERTa
  (`cbr/train.py`) sin el bug del `nan` de MPS y más rápido. Para eso habría que correr
  el repo allí (no es remoto como Ollama); ver `03-cbr-y-resultados.md` §2.

**Troubleshooting:**

| Síntoma | Causa probable | Arreglo |
|---|---|---|
| `curl` desde la Mac da timeout | firewall cerrado o `OLLAMA_HOST` no aplicado | revisar §1.2.2 y §1.2.3; reiniciar Ollama |
| `curl localhost` funciona en la laptop pero no desde la Mac | Ollama sigue en `127.0.0.1` | confirmar `OLLAMA_HOST=0.0.0.0:11434` y reinicio |
| `model not found` | falta el pull | `ollama pull gemma4:e4b-it-qat` en la laptop |
| Respuestas vacías / muy lentas | modelo no entra en VRAM (¿usaste 12b?) | volver a `e4b-it-qat` |

---

## 2. Estado de la minería de gecko-dev (señal `contribution` del IBR) (`gecko-dev-mining-status.md`)

> **TL;DR**: La minería de gecko-dev **se hizo, está completa y cableada al IBR**, pero
> tras las ablaciones (2026-06-12) su señal resultó **no aportar — incluso dañar** en el
> piloto Mozilla, así que está **temporalmente apagada** (`ip_contribution=0` en `config.py`).
> El aporte real del IBR viene hoy de `assignment` + `discussion` (parquets de Bugzilla),
> **no** de gecko-dev. Se deja mineada y cableada (peso 0, no borrada) para re-evaluar al escalar.

### 2.1 Qué se hizo

- **Script**: `src/triager_omega/data/repo_miner.py` (minería con `git log` sobre `mozilla/gecko-dev`, parseo de `Bug <id>` y de `r=`/`a=`).
- **Artefacto**: `artifacts/repo_interactions.parquet`
  - **288.324 interacciones** sobre **80.618 bugs** únicos.
  - **159.170 `review`** + **129.154 `commit`**.
  - Rango de fechas: **2021-01-05 → 2025-07-08**.
  - **Identidad 100% resuelta** a `Contributor Id` (doble puente: email vía `comments`, `:nick` vía `contributors.User Name`).
- **Uso**: el IBR (`src/triager_omega/modules/ibr.py`, `_build_interaction_table`) une esta tabla con `discussion_interactions.parquet` y la señal `assignment` derivada de los splits. `commit` y `review` comparten el peso `ip_contribution` (esquema de 3 IP, como TriagerX).

### 2.2 El matiz: hallazgo de las ablaciones

En el **sistema completo** (CBR=both + IBR), la descomposición por tipo de interacción
(piloto, test, W_f sintonizado en val) mostró:

| Fuente IBR aislada | Δ Hit@1 vs CBR-solo |
|---|---|
| `contribution` (commit/review, **gecko-dev**) | **~0 (incluso daña)** |
| `discussion` (Bugzilla) | +2.0 pp |
| `assignment` (Bugzilla) | +2.9 pp |

**Causa**: los IP de TriagerX se calibraron en OpenJ9/GitHub, donde la etiqueta es el
**contribuidor de código** → `contribution` es la señal correcta. En **Mozilla la etiqueta
es `Assigned To`**, y los committers/revisores de bugs similares suelen **no** ser el
assignee → `contribution` mete ruido al top-1. Además, al normalizar el NIS min-max, un
`ip_contribution` alto (1.5) **domina el vector y diluye** las señales útiles
(`assignment`, `discussion`).

### 2.3 Decisión aplicada

`config.py` (re-tuneo para Mozilla, ver `PLAN.md §10.2 paso 3`):

```python
ip_contribution: float = 0.0   # gecko-dev (commit/review): apagado en Mozilla (TriagerX: 1.5)
ip_assignment:   float = 0.5   # (TriagerX: 0.5)
ip_discussion:   float = 0.5   # (TriagerX: 0.1)
```

Con esto, el sistema completo en test pasa de **0.7604** (config TriagerX) a **0.7746** Hit@1 (MRR 0.8542 → 0.8613).

### 2.4 Por qué NO se borra la minería

1. **Escala 450 devs**: con 450 clases el CBR se vuelve más débil; `contribution` podría
   volver a aportar. Hay que **re-correr el grid de IP** al escalar antes de descartarla.
2. **Mejor resolución de identidad**: afinar el puente gecko-dev (p.ej. distinguir mejor
   revisores reales de los `r=`/`a=`, o filtrar bots) podría revivir la señal.

Mientras tanto se conserva mineada y cableada con **peso 0** — sin coste de inferencia, lista para reactivar cambiando un único valor en `config.py`.

### 2.5 Validación cruzada en OpenJ9 (2026-06-13)

Para confirmar que el problema era la **etiqueta** y no nuestro código ni la señal de
código en sí, se minó OpenJ9 (etiqueta `owner` = fixer) vía GitHub API y se corrió el
IBR ahí. Scripts: `scripts/mine_openj9_timelines.py` + `scripts/eval_openj9_ibr.py`.
Minados 2.204 issues → 30.911 interacciones (discussion 18.390, commits 7.231,
pull_request 4.382, assignment 908).

**IBR-solo OpenJ9, por tipo** (top_k=15, τ=0.6, λ=0.01):

| Config (C/A/D) | Hit@1 | Hit@5 | Hit@10 | MRR | Cob. |
|---|---|---|---|---|---|
| ALL TriagerX (1.5/0.5/0.1) | 0.238 | 0.492 | 0.637 | 0.373 | 0.524 |
| contribution OFF (0/0.5/0.1) | 0.222 | 0.495 | 0.637 | 0.368 | 0.502 |
| only-contribution | 0.238 | 0.479 | 0.637 | 0.375 | 0.469 |
| only-assignment | 0.251 | 0.402 | 0.688 | 0.360 | 0.309 |
| only-discussion | 0.215 | 0.492 | 0.633 | 0.365 | 0.498 |

**Conclusión**: en OpenJ9 `contribution` **ayuda** (ON vs OFF: +1.6pp Hit@1, +0.5pp MRR,
+2.3pp cobertura) y el config completo de TriagerX es el mejor — **lo opuesto a Mozilla**.
La señal de commits/PRs sí sirve cuando la etiqueta es quien arregla el bug → confirma que
apagar `ip_contribution` en Mozilla es por la etiqueta (`Assigned To` ≠ fixer), no por la minería.

**Caveat**: los absolutos OpenJ9 son bajos porque es IBR-**solo** (sin el CBR ensemble del
paper) y la cobertura es ~0.52 (índice de train pequeño, 1.893 issues). NO es una réplica de
los números del paper; es una validación del comportamiento del IBR y de la dependencia de la
etiqueta.

### 2.6 Sistema completo OpenJ9 (CBR + IBR) end-to-end

Se entrenó un CBR (DeBERTa-v3) sobre OpenJ9 (`scripts/train_openj9_cbr.py`, texto→`owner`,
17 clases) y se corrió el agregador `FS=NPS+W_f·NIS` (`scripts/eval_openj9_full.py`).

**Lección de entrenamiento**: el `WeightedRandomSampler` (peso `1/freq`, clave en Mozilla por
la cola larga de 450 devs) es **contraproducente en OpenJ9** (17 devs ya activos, 1.893
ejemplos): sobre-balancea → el modelo sobre-predice devs raros y el **Hit@1 cae por debajo del
azar** (0.042 < 1/17). Re-entrenado `--no-weighted --epochs 6` → Hit@1 0.199. *Los HPs del
piloto Mozilla no se trasladan a ciegas.*

Resultados (test):

| | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| CBR-solo | 0.199 | 0.704 | 0.859 | 0.401 |
| IBR-solo | 0.238 | 0.492 | 0.637 | 0.373 |
| **Full (contribution ON, W_f=0.3)** | **0.257** | 0.714 | 0.878 | **0.452** |
| Full (contribution OFF, W_f=0.5) | 0.235 | 0.730 | 0.881 | 0.438 |

- **El full supera a ambos solos** (MRR 0.452 vs 0.401/0.373) → CBR e IBR son complementarios.
- **`contribution` ayuda end-to-end** (+2.3pp Hit@1, +1.4pp MRR ON vs OFF, consistente en todo W_f).

**Conclusión cruzada**: el valor de la señal de código (`contribution`) depende de la definición
de etiqueta — daña con `Assigned To` (assignee, Mozilla), ayuda con `owner`/fixer (OpenJ9). Queda
demostrado end-to-end en ambos regímenes.

**Caveats**: CBR de OpenJ9 underfit (1.893 ejemplos, sin el ensemble de TriagerX) → absolutos
modestos, NO es réplica del paper. Sin split de validación, el W_f se reporta como curva en test.

### 2.7 Cómo reproducir / re-evaluar

```bash
# señal contribution ON vs OFF (IBR-solo)
IP_CONTRIBUTION=0 uv run python -m triager_omega.modules.ibr eval --split test

# sistema completo con distintos pesos (grid de W_f)
uv run python -m triager_omega.modules.aggregator eval --split test --grid --cbr-mode both
```

*Última actualización: 2026-06-12.*
