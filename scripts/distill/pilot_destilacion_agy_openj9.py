"""Piloto RÁPIDO y PRELIMINAR de destilación semántica sobre OpenJ9, usando `agy`
(CLI agéntica con modelos en la nube: Gemini/Claude/GPT-OSS) como proxy de velocidad.

IMPORTANTE — esto NO es el modelo de la propuesta (Gemma 4 E4B IT QAT local). Es un
piloto exploratorio para tener una primera señal de si destilar el texto ayuda a la
recuperación, antes de comprometerse a la corrida completa y más lenta con el LLM
local sobre todo el corpus. Resultados de aquí no deben citarse como validación final.

Reusa el prompt (system + few-shot) ya validado en
`triager_omega.cbr.distillation`, solo cambia el backend (agy en vez de
ollama/google/lmstudio/grok) y el constructor de entrada (issues de GitHub en vez de
reportes de Bugzilla).

Uso:
    uv run python scripts/pilot_destilacion_agy_openj9.py --n-train 120 --n-test 40 --seed 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger

from triager_omega.cbr.distillation import (
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    _valid,
    to_distilled_text,
)

AGY_MODEL = "Gemini 3.5 Flash (Medium)"  # default; sobreescribible con --model


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


def build_prompt(user_input: str) -> str:
    turns = [SYSTEM_PROMPT.strip()]
    for ex in FEW_SHOT_EXAMPLES:
        turns.append("### Ejemplo\nInput:\n" + ex["input"] + "\nOutput:\n" + ex["output"])
    turns.append("### Ahora tú — responde SOLO con el JSON, nada más\nInput:\n" + user_input + "\nOutput:")
    return "\n\n".join(turns)


def call_agy(prompt: str, timeout_s: int = 60) -> str:
    result = subprocess.run(
        ["agy", "-p", prompt, "--model", AGY_MODEL, "--print-timeout", f"{timeout_s}s"],
        capture_output=True, text=True, timeout=timeout_s + 20,
    )
    if result.returncode != 0:
        raise RuntimeError(f"agy exit {result.returncode}: {result.stderr[:300]}")
    return result.stdout.strip()


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


def local_fallback(row: pd.Series) -> dict:
    return {
        "fault_location": {"product": "OpenJ9", "component": _clean(row.get("component")), "subsystem": ""},
        "symptoms": [],
        "capabilities": [],
        "_fallback": True,
    }


def distill_row(row: pd.Series, retries: int = 3) -> dict:
    prompt = build_prompt(build_input_openj9(row))
    for attempt in range(retries + 1):
        try:
            raw = call_agy(prompt)
            d = parse_json_output(raw)
            if _valid(d):
                return d
        except Exception as e:  # noqa: BLE001
            logger.warning("issue {} intento {} falló: {}", row["issue_number"], attempt, e)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 20))
    return local_fallback(row)


def _load_checkpoint(ckpt_path: Path) -> dict[int, dict]:
    """Lee el JSONL de checkpoint (si existe) → {issue_number: fila} de lo ya hecho."""
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
    """Reconstruye parquet+csv desde el checkpoint jsonl (idempotente, se puede llamar
    en cualquier momento, incluso a mitad de la corrida — para no perder avance)."""
    all_rows = list(_load_checkpoint(ckpt_path).values())
    out = pd.DataFrame(all_rows)
    out.to_parquet(out_dir / f"openj9_{split_name}_{tag}_agy.parquet", index=False)
    out.to_csv(out_dir / f"openj9_{split_name}_{tag}_agy.csv", index=False)
    return out


def run_split(df: pd.DataFrame, split_name: str, out_dir: Path, workers: int = 8, tag: str = "pilot",
              save_every: int = 25) -> pd.DataFrame:
    if df.empty:
        logger.info("[{}] split vacío, se omite", split_name)
        return df

    ckpt_path = out_dir / f"openj9_{split_name}_{tag}_agy.jsonl"
    done_rows = _load_checkpoint(ckpt_path)
    # las filas que cayeron a fallback NO cuentan como "hechas" — se reintentan (bajo
    # concurrencia más baja evitan el mismo problema: 78.5% fallback con workers=8
    # resultó ser sobrecarga de agy, no un problema del prompt/modelo; ver commit).
    good_done = {k: v for k, v in done_rows.items() if not v.get("fallback")}
    n_retry = len(done_rows) - len(good_done)
    if good_done:
        logger.info("[{}] checkpoint encontrado: {} filas OK, {} fallback a reintentar",
                     split_name, len(good_done), n_retry)
    pending = df[~df["issue_number"].isin(good_done.keys())]
    logger.info("[{}] {} pendientes de {} totales", split_name, len(pending), len(df))

    t0 = time.time()
    done_lock = threading.Lock()
    ckpt_file = ckpt_path.open("a", encoding="utf-8")
    completed = 0

    def _work(row: pd.Series) -> None:
        nonlocal completed
        d = distill_row(row)
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
            logger.info("[{}] {}/{} issue={} ({:.0f}s acumulado)",
                         split_name, completed, len(pending), row["issue_number"], time.time() - t0)
            if completed % save_every == 0:
                _dump_outputs(ckpt_path, out_dir, split_name, tag)
                logger.info("[{}] snapshot intermedio guardado", split_name)

    try:
        if not pending.empty:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_work, row) for _, row in pending.iterrows()]
                for f in as_completed(futures):
                    f.result()  # propaga excepciones no capturadas dentro de _work, si las hubiera
    finally:
        ckpt_file.close()

    out = _dump_outputs(ckpt_path, out_dir, split_name, tag)
    n_fb = int(out["fallback"].sum())
    logger.info("[{}] guardado final: {} filas, {} fallback ({:.0%})", split_name, len(out), n_fb, n_fb / max(len(out), 1))
    return out


def main() -> None:
    global AGY_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="artifacts/pilot_destilacion_agy")
    ap.add_argument("--workers", type=int, default=8, help="llamadas concurrentes a agy")
    ap.add_argument("--tag", default="pilot", help="sufijo de los archivos de salida (p.ej. 'full')")
    ap.add_argument("--model", default=AGY_MODEL, help="modelo de agy (ver `agy models`)")
    ap.add_argument("--save-every", type=int, default=25, help="cada cuántas filas completadas volcar snapshot parquet/csv")
    args = ap.parse_args()
    AGY_MODEL = args.model

    train = pd.read_csv("artifacts/openj9/openj9_train_50.csv")
    test = pd.read_csv("artifacts/openj9/openj9_test_50.csv")
    train_s = train.sample(n=min(args.n_train, len(train)), random_state=args.seed).reset_index(drop=True)
    test_s = test.sample(n=min(args.n_test, len(test)), random_state=args.seed).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_split(train_s, "train", out_dir, workers=args.workers, tag=args.tag, save_every=args.save_every)
    run_split(test_s, "test", out_dir, workers=args.workers, tag=args.tag, save_every=args.save_every)


if __name__ == "__main__":
    main()
