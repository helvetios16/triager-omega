# Ollama remoto: destilar desde la Mac usando la GPU de la laptop (RTX 5060)

Cómo correr el LLM de destilación (Gemma) en la **laptop con RTX 5060 8GB** y que la
**Mac** le envíe los bugs y reciba el JSON destilado. No requiere cambios de código:
solo configurar la red y el `.env`.

## Arquitectura

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

## 1. En la laptop (RTX 5060)

### 1.1 Instalar Ollama y el modelo
```bash
# instalar Ollama desde https://ollama.com/download
ollama pull gemma4:e4b-it-qat
```
> **Usar `e4b-it-qat`, NO el `12b`.** El `e4b` (4B eficiente) cuantizado ocupa ~3-4 GB
> y entra cómodo en 8 GB. El `12b` necesita ~8-9 GB y se queda sin margen.

### 1.2 Hacer que Ollama escuche en la red (no solo en localhost)

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

### 1.3 Abrir el puerto 11434 en el firewall

**Windows (PowerShell como admin):**
```powershell
New-NetFirewallRule -DisplayName "Ollama" -Direction Inbound -LocalPort 11434 -Protocol TCP -Action Allow
```

**Linux (ufw):**
```bash
sudo ufw allow 11434/tcp
```

### 1.4 Anotar la IP local de la laptop
```bash
# Windows:
ipconfig          # buscar "Dirección IPv4", ej. 192.168.1.50
# Linux/Mac:
ip addr           # o: hostname -I
```

---

## 2. En la Mac (cliente)

### 2.1 Apuntar el `.env` a la laptop
Editar `.env`: comentar la línea LOCAL y descomentar la REMOTO con la IP real:
```
DISTILL_BACKEND=ollama
OLLAMA_MODEL=gemma4:e4b-it-qat
# OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_BASE_URL=http://192.168.1.50:11434/v1   # <-- IP de la laptop
```

### 2.2 Verificar conexión
```bash
curl http://192.168.1.50:11434/v1/models        # debe listar gemma4:e4b-it-qat
```

### 2.3 Probar y correr la destilación
```bash
uv run python scripts/run_distillation.py --smoke 5   # prueba 5 bugs, imprime
uv run python scripts/run_distillation.py             # batch completo (reanudable)
```

---

## Notas

- **Ambas máquinas en la misma red.** La Mac y la laptop deben estar en la misma LAN/WiFi.
- **Velocidad esperada:** la RTX 5060 ≈ 0.3-0.8 s/bug vs ~1.5 s en MPS → el batch de
  9.364 bugs baja de ~4 h a **~1-2 h**.
- **Seguridad:** `0.0.0.0` expone Ollama a toda la red local **sin contraseña**. Está
  bien en red de casa; NO lo hagas en una red pública/compartida.
- **Reanudable:** si se corta la conexión, `run_distillation.py` retoma desde el cache
  (`artifacts/pilot/distillations.parquet`) al volver a ejecutarlo.
- **Bonus — entrenamiento:** esa misma laptop con CUDA puede entrenar el DeBERTa
  (`cbr/train.py`) sin el bug del `nan` de MPS y más rápido. Para eso habría que correr
  el repo allí (no es remoto como Ollama); queda como opción futura.

## Troubleshooting

| Síntoma | Causa probable | Arreglo |
|---|---|---|
| `curl` desde la Mac da timeout | firewall cerrado o `OLLAMA_HOST` no aplicado | revisar §1.2 y §1.3; reiniciar Ollama |
| `curl localhost` funciona en la laptop pero no desde la Mac | Ollama sigue en `127.0.0.1` | confirmar `OLLAMA_HOST=0.0.0.0:11434` y reinicio |
| `model not found` | falta el pull | `ollama pull gemma4:e4b-it-qat` en la laptop |
| Respuestas vacías / muy lentas | modelo no entra en VRAM (¿usaste 12b?) | volver a `e4b-it-qat` |
