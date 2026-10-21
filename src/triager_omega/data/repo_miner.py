"""Minería de repositorio (MSR) para enriquecer el IBR con interacciones de código.

El IBR del paper TriagerX (Ec. 6) pondera distintos *tipos* de interacción mediante
una tabla de Interaction Points. Los parquets de Bugzilla solo aportan `assignment`
(Assigned To) y `discussion` (comentarios); falta la señal más fuerte del paper:
`commit`/`review`. Este módulo la recupera minando el historial de git de
``gecko-dev`` (mirror de mozilla-central).

Flujo:
    1. ``mine_raw_commits``  — git log -> DataFrame de commits (metadatos, sin diffs).
    2. ``extract_interactions`` — parsea "Bug <id>" y "r=<nick>" -> tabla larga
       (bug_id, raw_actor, kind, timestamp).
    3. ``bridge_to_contributor_id`` — mapea actor (email/nick) -> Contributor Id.
    4. ``build_interaction_table`` — orquesta y persiste el artefacto que consume ibr.py.

Convención de Mozilla en mensajes de commit:
    "Bug 1819876 - Do the thing in places. r=emilio,markh"
    "Backed out changeset abc for bug 1700000 ..."
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd
from loguru import logger

from triager_omega.config import Settings, settings
from triager_omega.data.loader import build_email_to_id, load_contributors

# --- Regex de parseo ---
# Bug IDs: 3-7 dígitos tras "bug" (mozilla usa hasta 7). Captura múltiples por commit.
_BUG_RE = re.compile(r"\bbug[\s#:]+(\d{3,7})\b", re.IGNORECASE)
# Revisores/aprobadores: "r=foo,bar.baz" "r?foo" "a=release". Captura lista de nicks.
_REVIEW_RE = re.compile(r"\b[ra][=?]([A-Za-z0-9][\w.+#-]*(?:,[A-Za-z0-9][\w.+#-]*)*)")
# Nick(s) de Mozilla embebidos en contributors.User Name: "(:emilio)", "[:emk]".
_NICK_RE = re.compile(r":([A-Za-z0-9][\w.-]*)")
# Separadores ASCII para un git log parseable aunque el body sea multilínea.
_FS = "\x1f"  # entre campos
_RS = "\x1e"  # entre commits


def mine_raw_commits(
    repo_path: Path | None = None,
    since: str = "2021-01-01",
    branch: str = "master",
    cfg: Settings = settings,
) -> pd.DataFrame:
    """Ejecuta ``git log`` sobre el repo bare y devuelve un DataFrame de commits.

    Columnas: ``hash, author_name, author_email, author_date (UTC), subject, body``.
    No se piden diffs ni árboles: solo metadatos, por lo que es rápido y funciona
    sobre un clon ``--filter=tree:0``.
    """
    repo_path = repo_path or (cfg.artifacts_dir.parent / ".repos" / "gecko-dev.git")
    if not repo_path.exists():
        raise FileNotFoundError(
            f"Repo bare no encontrado en {repo_path}. Clónalo primero (ver README MSR)."
        )

    fmt = _FS.join(["%H", "%an", "%ae", "%aI", "%s", "%b"]) + _RS
    cmd = [
        "git", "-C", str(repo_path), "log", branch,
        "--no-merges", f"--since={since}", f"--pretty=format:{fmt}",
    ]
    logger.info("git log sobre {} desde {} (rama {})", repo_path.name, since, branch)
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

    rows = []
    for rec in out.split(_RS):
        rec = rec.strip("\n")
        if not rec:
            continue
        parts = rec.split(_FS)
        if len(parts) < 6:
            continue
        h, an, ae, adate, subject, body = parts[:6]
        rows.append((h, an, ae, adate, subject, body))

    df = pd.DataFrame(rows, columns=["hash", "author_name", "author_email", "author_date", "subject", "body"])
    df["author_date"] = pd.to_datetime(df["author_date"], errors="coerce", utc=True)
    logger.info("Commits minados: {}", len(df))
    return df


def extract_interactions(commits: pd.DataFrame) -> pd.DataFrame:
    """Parsea cada commit a filas (bug_id, raw_actor, kind, timestamp).

    - ``kind='commit'``: el autor del commit, por cada bug referenciado.
    - ``kind='review'``: cada nick en ``r=``/``a=``, por cada bug referenciado.
    """
    records = []
    for h, an, ae, adate, subject, body in commits.itertuples(index=False, name=None):
        text = f"{subject}\n{body}"
        bug_ids = {int(m) for m in _BUG_RE.findall(text)}
        if not bug_ids:
            continue  # commit sin bug vinculado: inútil para el IBR

        reviewers: set[str] = set()
        for grp in _REVIEW_RE.findall(text):
            reviewers.update(n.strip().lower() for n in grp.split(",") if n.strip())
        reviewers.discard("")  # por si acaso

        for bug_id in bug_ids:
            records.append((bug_id, ae.strip().lower(), "commit", adate, h))
            for nick in reviewers:
                records.append((bug_id, nick, "review", adate, h))

    out = pd.DataFrame(records, columns=["bug_id", "raw_actor", "kind", "timestamp", "commit_hash"])
    logger.info(
        "Interacciones extraídas: {} (commit={}, review={}) sobre {} bugs únicos",
        len(out),
        (out["kind"] == "commit").sum(),
        (out["kind"] == "review").sum(),
        out["bug_id"].nunique(),
    )
    return out


def bridge_to_contributor_id(
    interactions: pd.DataFrame, cfg: Settings = settings
) -> pd.DataFrame:
    """Mapea ``raw_actor`` -> ``Contributor Id`` (clave canónica del sistema).

    Dos rutas, en orden:
      1. **email** (autores de commit): puente email->id vía tabla de comentarios
         (mismo mecanismo que usa el label `Assigned To`).
      2. **nick** (revisores r=): match contra ``contributors.User Name`` (case-insensitive).

    Las filas sin match quedan con ``Contributor Id`` nulo; se reporta cobertura
    por tipo y se descartan al final (el IBR solo puntúa devs del directorio activo).
    """
    email_to_id = build_email_to_id(cfg=cfg)
    email_map = {str(k).lower(): int(v) for k, v in email_to_id.items()}

    contribs = load_contributors(columns=["Contributor Id", "User Name"], cfg=cfg)
    contribs = contribs.dropna(subset=["User Name"])
    # Mozilla embebe el/los nick(s) de revisor en User Name: "Emilio (:emilio)",
    # "Masatoshi [:emk]", "BugBot [:suhaib / :marco]". Extraemos cada :nick -> id.
    nick_map: dict[str, int] = {}
    full_name_map: dict[str, int] = {}
    for name, cid in zip(contribs["User Name"], contribs["Contributor Id"]):
        full_name_map.setdefault(str(name).lower(), int(cid))
        for nick in _NICK_RE.findall(str(name)):
            nick_map.setdefault(nick.lower(), int(cid))

    def resolve(actor: str, kind: str) -> int | float:
        # commits traen email -> probar email primero; reviewers traen :nick.
        if kind == "commit":
            return email_map.get(actor, nick_map.get(actor, float("nan")))
        return nick_map.get(actor, full_name_map.get(actor, float("nan")))

    df = interactions.copy()
    df["Contributor Id"] = [resolve(a, k) for a, k in zip(df["raw_actor"], df["kind"])]

    for kind in ("commit", "review"):
        sub = df[df["kind"] == kind]
        cov = sub["Contributor Id"].notna().mean() if len(sub) else 0.0
        logger.info("Cobertura puente {}: {:.1%} ({} filas)", kind, cov, len(sub))

    resolved = df.dropna(subset=["Contributor Id"]).copy()
    resolved["Contributor Id"] = resolved["Contributor Id"].astype("int64")
    logger.info("Interacciones con identidad resuelta: {} / {}", len(resolved), len(df))
    return resolved


def build_interaction_table(
    repo_path: Path | None = None,
    since: str = "2021-01-01",
    cfg: Settings = settings,
) -> pd.DataFrame:
    """Pipeline completo: mina, extrae, puentea y persiste el artefacto MSR.

    Salida: ``artifacts/mozilla/repo_interactions.parquet`` con columnas
    ``bug_id, Contributor Id, kind, timestamp, commit_hash``. Lo consume ibr.py
    para alimentar la tabla de Interaction Points con los tipos `commit`/`review`.
    """
    commits = mine_raw_commits(repo_path=repo_path, since=since, cfg=cfg)
    interactions = extract_interactions(commits)
    resolved = bridge_to_contributor_id(interactions, cfg=cfg)

    out_path = cfg.repo_interactions_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resolved.to_parquet(out_path, index=False)
    logger.success("Artefacto MSR guardado en {} ({} filas)", out_path, len(resolved))
    return resolved


if __name__ == "__main__":
    build_interaction_table()
