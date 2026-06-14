"""Minería de interacciones de OpenJ9 desde la GitHub API (para validar el IBR).

Los CSV de TriagerX (`openj9_{train,test}_17.csv`) traen texto + etiqueta (`owner`)
pero NO las interacciones (PRs/commits/comentarios) que consume el IBR. Este script
las reconstruye llamando a la GitHub API por cada issue del subset de 17 devs y las
escribe como una tabla larga, en el mismo esquema que `repo_interactions.parquet`.

Mapeo de eventos → `kind` (idéntico a `_get_contribution_data` de TriagerX y a los
3 Interaction Points de nuestro IBR):

  | Evento del timeline                                   | kind          | IP            |
  |-------------------------------------------------------|---------------|---------------|
  | `cross-referenced` con source.issue.pull_request      | pull_request  | ip_contribution
  | `referenced` con commit_id                            | commits       | ip_contribution
  | `commented`                                           | discussion    | ip_discussion
  | issue.assignees (estado final, fechado en created_at) | assignment    | ip_assignment

Resolución de identidad TRIVIAL: el `actor.login` del timeline ya es el mismo espacio
que la etiqueta `owner` (login de GitHub) — sin el doble puente email/`:nick` de gecko-dev.

Salidas (en artifacts/openj9/):
  - openj9_interactions.parquet  (issue_number, dev, kind, timestamp)
  - openj9_issue_meta.parquet    (issue_number, created_at)  → t_now de las queries

Idempotente y reanudable: cachea la respuesta cruda por issue en
`artifacts/openj9/raw/{n}.json`; al re-correr no vuelve a bajar lo ya cacheado.

Requiere un token de GitHub (lectura pública basta) en la env var GITHUB_TOKEN
(o en .env). Sin token el límite anónimo (60 req/h) hace inviable bajar ~2.200 issues.

Uso:
    GITHUB_TOKEN=ghp_xxx uv run python scripts/mine_openj9_timelines.py
    GITHUB_TOKEN=ghp_xxx uv run python scripts/mine_openj9_timelines.py --smoke 20
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd
import requests
from loguru import logger

from triager_omega.config import settings


def _issue_numbers(cfg=settings) -> list[int]:
    """Issues únicos del subset de 17 devs (train + test)."""
    nums: set[int] = set()
    for csv in (cfg.openj9_train_csv, cfg.openj9_test_csv):
        if not csv.exists():
            raise FileNotFoundError(f"No existe {csv}. ¿Está el repo TriagerX en {cfg.triagerx_repo}?")
        nums.update(int(n) for n in pd.read_csv(csv)["issue_number"].dropna().unique())
    return sorted(nums)


class _GitHub:
    """Cliente mínimo de la GitHub API con manejo de rate-limit y paginación."""

    def __init__(self, repo: str, token: str, cfg=settings):
        self.repo = repo
        self.base = cfg.github_api_base
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Authorization": f"Bearer {token}",
        })

    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        last_exc = None
        for attempt in range(8):
            try:
                r = self.session.get(url, params=params, timeout=30)
            except requests.exceptions.RequestException as e:
                # cortes de red transitorios (RemoteDisconnected, timeouts, DNS...).
                last_exc = e
                wait = min(60, 2 ** attempt)
                logger.warning("Error de red ({}); reintento en {}s...", type(e).__name__, wait)
                time.sleep(wait)
                continue
            # rate-limit: si quedan 0 llamadas, esperar hasta el reset.
            if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(5, reset - int(time.time()) + 2)
                logger.warning("Rate-limit agotado; esperando {}s...", wait)
                time.sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt))
                continue
            return r
        if last_exc is not None:
            raise last_exc
        return r  # última respuesta (probablemente error); el caller decide

    def issue(self, n: int) -> dict | None:
        r = self._get(f"{self.base}/repos/{self.repo}/issues/{n}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def timeline(self, n: int) -> list[dict]:
        events, page = [], 1
        while True:
            r = self._get(
                f"{self.base}/repos/{self.repo}/issues/{n}/timeline",
                params={"per_page": 100, "page": page},
            )
            r.raise_for_status()
            batch = r.json()
            events.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return events


def _fetch_raw(gh: _GitHub, n: int, raw_dir) -> dict | None:
    """Baja (o lee de caché) el issue + timeline crudos de un issue."""
    cache = raw_dir / f"{n}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    issue = gh.issue(n)
    if issue is None:
        logger.warning("Issue {} no encontrado (404); se omite.", n)
        return None
    raw = {
        "issue_number": n,
        "created_at": issue.get("created_at"),
        "assignees": [a["login"] for a in (issue.get("assignees") or []) if a.get("login")],
        "timeline": gh.timeline(n),
    }
    cache.write_text(json.dumps(raw), encoding="utf-8")
    return raw


def _to_rows(raw: dict) -> tuple[list[dict], dict]:
    """Convierte el crudo en filas de interacción + meta del issue."""
    n = raw["issue_number"]
    created = raw.get("created_at")
    rows: list[dict] = []

    # assignment: estado final de assignees, fechado en la creación del issue.
    for login in raw.get("assignees", []):
        rows.append({"issue_number": n, "dev": login, "kind": "assignment", "timestamp": created})

    for ev in raw.get("timeline", []):
        etype = ev.get("event")
        actor = (ev.get("actor") or {}).get("login")
        ts = ev.get("created_at")
        if not actor:
            continue
        if etype == "cross-referenced" and (ev.get("source", {}).get("issue", {}) or {}).get("pull_request"):
            kind = "pull_request"
        elif etype == "referenced" and (ev.get("commit_id") or ev.get("commit_url")):
            kind = "commits"
        elif etype == "commented":
            kind = "discussion"
        else:
            continue
        rows.append({"issue_number": n, "dev": actor, "kind": kind, "timestamp": ts})

    return rows, {"issue_number": n, "created_at": created}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", type=int, default=None, help="solo los primeros N issues (prueba)")
    args = p.parse_args()

    cfg = settings
    # config.py (pydantic-settings) ya lee GITHUB_TOKEN del .env; env var explícita tiene prioridad.
    token = os.environ.get("GITHUB_TOKEN", "") or cfg.github_token
    if not token:
        raise SystemExit(
            "Falta GITHUB_TOKEN. Ponlo en .env (GITHUB_TOKEN=ghp_...) o expórtalo en el entorno. "
            "Token de lectura pública basta."
        )

    raw_dir = cfg.openj9_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    issues = _issue_numbers(cfg)
    if args.smoke:
        issues = issues[: args.smoke]
    logger.info("Minando {} issues de {}...", len(issues), cfg.openj9_gh_repo)

    gh = _GitHub(cfg.openj9_gh_repo, token, cfg)
    all_rows, meta_rows, failed = [], [], []
    for i, n in enumerate(issues, 1):
        try:
            raw = _fetch_raw(gh, n, raw_dir)
        except Exception as e:  # noqa: BLE001 - un issue no debe tumbar la corrida
            logger.error("Issue {} falló tras reintentos ({}); se omite.", n, type(e).__name__)
            failed.append(n)
            continue
        if raw is None:
            continue
        rows, meta = _to_rows(raw)
        all_rows.extend(rows)
        meta_rows.append(meta)
        if i % 100 == 0:
            logger.info("  {}/{} issues ({} interacciones)", i, len(issues), len(all_rows))
    if failed:
        logger.warning("{} issues omitidos por error: {}", len(failed), failed[:20])

    inter = pd.DataFrame(all_rows)
    inter["timestamp"] = pd.to_datetime(inter["timestamp"], utc=True, errors="coerce")
    meta = pd.DataFrame(meta_rows)
    meta["created_at"] = pd.to_datetime(meta["created_at"], utc=True, errors="coerce")

    inter.to_parquet(cfg.openj9_interactions_path, index=False)
    meta.to_parquet(cfg.openj9_issue_meta_path, index=False)

    logger.success("Escrito {} ({} interacciones) y {} ({} issues)",
                   cfg.openj9_interactions_path, len(inter), cfg.openj9_issue_meta_path, len(meta))
    logger.info("kinds: {}", inter["kind"].value_counts().to_dict())


if __name__ == "__main__":
    main()
