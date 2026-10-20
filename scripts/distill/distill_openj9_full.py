"""Destilación semántica del corpus COMPLETO de OpenJ9 (3348 train + 534 test),
usando un backend en la nube OpenAI-compatible (default: NVIDIA NIM / deepseek-v4-flash)
en vez de Gemma 4 E4B IT QAT local (lento) o `agy` (subprocess, sin control de rate limit).

Justificación (ver docs/piloto-destilacion-agy-openj9.md): el piloto rápido (120 train +
40 test, backend agy/Gemini) mostró que el texto destilado supera al crudo en las 18
celdas de un barrido (k, τ) de `cbr_retrieval_openj9.py`, con +10pp Top-1 / +11.5pp MRR
en el mejor punto. Esta corrida valida esa señal sobre el corpus íntegro.

Reusa el prompt (system + few-shot) y los backends OpenAI-compatibles ya definidos en
`triager_omega.cbr.distillation` (mismo módulo que usa el pipeline principal de Bugzilla).
Solo cambia el constructor de entrada (issues de GitHub: título+cuerpo, sin comentarios).

Uso:
    uv run python scripts/distill_openj9_full.py --backend nvidia --workers 10
    uv run python scripts/distill_openj9_full.py --splits test  # solo test, para probar rápido
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger

from triager_omega.cbr.distillation import _valid, make_client, to_distilled_text
from triager_omega.config import settings


class RateLimiter:
    """Pacer de intervalo fijo: espacia el INICIO de cada llamada cada 60/max_per_minute
    segundos, sin importar cuántos workers concurrentes la estén pidiendo. Un token
    bucket con ventana de 60s deja pasar ráfagas (hasta max_per_minute casi simultáneas)
    que chocan con el límite de concurrencia del servidor (visto en la práctica: 429/503
    "Worker local total request limit reached" con el token bucket)."""

    def __init__(self, max_per_minute: int):
        self.interval = 60.0 / max_per_minute
        self.lock = threading.Lock()
        self.next_slot = 0.0

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_slot)
            self.next_slot = start + self.interval
        sleep_for = start - now
        if sleep_for > 0:
            time.sleep(sleep_for)


def _clean(value, default: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return str(value)


def build_input_openj9(row: pd.Series) -> str:
    return "\n".join([
        f"Summary: {_clean(row.get('issue_title'))}",
        "Product: OpenJ9",
        f"Component: {_clean(row.get('component'))}",
        "Initial report:",
        _clean(row.get("issue_body"))[:1500],
    ])


def local_fallback(row: pd.Series) -> dict:
    return {
        "fault_location": {"product": "OpenJ9", "component": _clean(row.get("component")), "subsystem": ""},
        "symptoms": [],
        "capabilities": [],
        "_fallback": True,
    }


def parse_json_output(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def distill_row(client, row: pd.Series, limiter: RateLimiter, retries: int = 4) -> dict:
    """Llama al backend con reintentos + backoff exponencial (rate limits / timeouts
    transitorios). Cae al fallback determinista solo tras agotar los reintentos."""
    user_input = build_input_openj9(row)
    for attempt in range(retries + 1):
        try:
            limiter.acquire()
            raw = client.complete(user_input)
            d = parse_json_output(raw)
            if _valid(d):
                return d
        except Exception as e:  # noqa: BLE001
            logger.warning("issue {} intento {} falló: {}", row["issue_number"], attempt, e)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))
    return local_fallback(row)


def _load_checkpoint(ckpt_path: Path) -> dict[int, dict]:
    done: dict[int, dict] = {}
    if ckpt_path.exists():
        with ckpt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                done[int(row["issue_number"])] = row
    return done


def _dump_outputs(ckpt_path: Path, out_dir: Path, split_name: str, tag: str) -> pd.DataFrame:
    """Reconstruye parquet + raw/distilled csv desde el checkpoint jsonl (idempotente,
    se puede llamar en cualquier momento, incluso a mitad de la corrida)."""
    all_rows = list(_load_checkpoint(ckpt_path).values())
    out = pd.DataFrame(all_rows)
    out.to_parquet(out_dir / f"openj9_{split_name}_{tag}.parquet", index=False)

    raw_cols = out[["issue_number", "owner", "text"]]
    raw_cols.to_csv(out_dir / f"openj9_{split_name}_{tag}_raw.csv", index=False)
    dist_cols = out[["issue_number", "owner", "distilled_text"]].rename(columns={"distilled_text": "text"})
    dist_cols.to_csv(out_dir / f"openj9_{split_name}_{tag}_distilled.csv", index=False)
    return out


def run_split(client, df: pd.DataFrame, split_name: str, out_dir: Path, workers: int, tag: str,
              limiter: RateLimiter, save_every: int = 50) -> pd.DataFrame:
    if df.empty:
        logger.info("[{}] split vacío, se omite", split_name)
        return df

    ckpt_path = out_dir / f"openj9_{split_name}_{tag}.jsonl"
    done_rows = _load_checkpoint(ckpt_path)
    if done_rows:
        logger.info("[{}] checkpoint encontrado: {} filas ya hechas, se reanuda", split_name, len(done_rows))
    pending = df[~df["issue_number"].isin(done_rows.keys())]
    logger.info("[{}] {} pendientes de {} totales", split_name, len(pending), len(df))

    t0 = time.time()
    done_lock = threading.Lock()
    ckpt_file = ckpt_path.open("a", encoding="utf-8")
    completed = 0

    def _work(row: pd.Series) -> None:
        nonlocal completed
        d = distill_row(client, row, limiter)
        record = {
            "issue_number": int(row["issue_number"]),
            "owner": row["owner"],
            "component": row.get("component", ""),
            "text": row["text"],
            "distilled_json": json.dumps(d, ensure_ascii=False),
            "distilled_text": to_distilled_text(d),
            "fallback": bool(d.get("_fallback", False)),
        }
        with done_lock:
            ckpt_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt_file.flush()
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                logger.info("[{}] {}/{} ({:.0f}s acumulado, {:.1f}s/issue)",
                             split_name, completed, len(pending), time.time() - t0,
                             (time.time() - t0) / completed)
            if completed % save_every == 0:
                _dump_outputs(ckpt_path, out_dir, split_name, tag)
                logger.info("[{}] snapshot intermedio guardado ({} filas)", split_name, completed + len(done_rows))

    try:
        if not pending.empty:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_work, row) for _, row in pending.iterrows()]
                for f in as_completed(futures):
                    f.result()
    finally:
        ckpt_file.close()

    out = _dump_outputs(ckpt_path, out_dir, split_name, tag)
    n_fb = int(out["fallback"].sum())
    logger.info("[{}] guardado final: {} filas, {} fallback ({:.0%})", split_name, len(out), n_fb, n_fb / max(len(out), 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="nvidia",
                     choices=["nvidia", "grok", "google", "ollama", "lmstudio", "agy", "opencode"])
    ap.add_argument("--out-dir", default="artifacts/openj9/distilled_full")
    ap.add_argument("--workers", type=int, default=15,
                     help="llamadas concurrentes; el --max-rpm es el que realmente limita el throughput")
    ap.add_argument("--max-rpm", type=int, default=38,
                     help="tope de solicitudes/minuto (NVIDIA free tier: 40; se deja margen de seguridad)")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--splits", default="train,test", help="subconjunto de {train,test} separado por comas")
    ap.add_argument("--save-every", type=int, default=50, help="cada cuántas filas completadas volcar snapshot parquet/csv")
    args = ap.parse_args()

    settings.distill_backend = args.backend
    client = make_client(settings)
    limiter = RateLimiter(args.max_rpm)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split_name in splits:
        df = pd.read_csv(f"artifacts/openj9/openj9_{split_name}_50.csv")
        run_split(client, df, split_name, out_dir, workers=args.workers, tag=args.tag,
                  limiter=limiter, save_every=args.save_every)


if __name__ == "__main__":
    main()
