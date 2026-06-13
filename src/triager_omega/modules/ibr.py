"""Módulo 4 — IBR (Interaction-Based Recommender) estilo TriagerX.

Réplica fiel de `triagerx/system/triagerx.py` (métodos `_get_top_k_similar_issues`,
`_get_historical_contributors`, `_get_contribution_point`, `_calculate_time_decay`,
`_normalize`) adaptada a nuestros datos:

  - **Recuperación.** TriagerX usa similitud coseno *brute-force* con
    `sentence_transformers.util.cos_sim` sobre una matriz de embeddings (NO usa
    FAISS). Lo replicamos: embebemos el texto (destilado, §7.4) de cada bug de
    train del piloto, guardamos la matriz y, en consulta, hacemos topk del coseno
    filtrado por el umbral τ (`ibr_tau`).

  - **Interacciones.** TriagerX lee un JSON por issue con `timeline_data`. Nosotros
    consumimos tres tablas largas ya construidas y las unimos en una Interaction
    Table `(bug_id, dev, kind, timestamp)`:
      · commit / review → `artifacts/repo_interactions.parquet` (minería gecko-dev)
      · discussion      → `artifacts/discussion_interactions.parquet`
      · assignment      → derivada de los splits (`Assigned To` + `creation_time`)

  - **Interaction Points (3 pesos, idéntico a `_get_contribution_point`):**
      commit/review → `ip_contribution` (1.5)   [TriagerX: pull_request/commits]
      assignment    → `ip_assignment`   (0.5)   [TriagerX: direct/last_assignment]
      discussion    → `ip_discussion`   (0.1)

  - **Decaimiento temporal** `exp(-λ·Δt)`. *Divergencia deliberada con TriagerX*:
    TriagerX mide Δt contra una fecha de checkpoint fija (`train_checkpoint_date`);
    nosotros usamos `t_now = Creation Time` del bug consultado y **filtramos
    interacciones con `t ≥ t_now`** (anti-fuga temporal, §7.6 del PLAN).

  - **NIS** (min-max, Ec. 7 de TriagerX) sobre el vector completo de devs activos:
    los que no interactuaron entran como 0 → min=0 → `NIS = IS / max(IS)`. La
    agregación `FS = NPS + W_f·NIS` (Ec. 8) vive en el Módulo 5 (aggregator).

CLI:
    uv run python -m triager_omega.modules.ibr build          # embebe + guarda índice
    uv run python -m triager_omega.modules.ibr eval           # IBR-solo Hit@K/MRR en test
    uv run python -m triager_omega.modules.ibr eval --split val --limit 300
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from loguru import logger

from triager_omega.config import Settings, settings
from triager_omega.data import loader

# Tipos de interacción que comparten el peso `ip_contribution` (commit + review),
# igual que TriagerX agrupa pull_request + commits en contribution_score.
_CONTRIBUTION_KINDS = frozenset({"commit", "review", "pull_request", "commits"})


# --------------------------------------------------------------------------- #
# Texto del bug (misma construcción que cbr/train.build_dataframe, §5.8)
# --------------------------------------------------------------------------- #
def combine_text(df: pd.DataFrame, mode: str) -> pd.Series:
    """Vista de texto según `mode` ('raw'|'distilled'|'both'), §5.8/§11.2.4.

    Requiere que `df` ya tenga las columnas 'raw' y 'distilled_text'. En modo
    'distilled' cae al crudo cuando falta el destilado (no pierde la fila)."""
    if mode == "raw":
        return df["raw"]
    if mode == "distilled":
        return df["distilled_text"].where(df["distilled_text"].str.strip() != "", df["raw"])
    return df["raw"] + " " + df["distilled_text"]  # both


def load_pilot_text(text_mode: str = "distilled", cfg: Settings = settings):
    """Une splits piloto + texto crudo + destilado. Devuelve (df, label_encoder).

    El df trae siempre 'raw' y 'distilled_text' (para derivar cualquier vista con
    `combine_text`) y 'text' = la vista pedida en `text_mode`. El IBR embebe por
    defecto el destilado (§7.4); el CBR usa la vista con la que se entrenó.
    """
    label_encoder: dict[str, int] = json.loads(
        cfg.pilot_label_encoder_path.read_text(encoding="utf-8")
    )
    splits = pd.read_parquet(cfg.pilot_splits_path).drop_duplicates(subset="Bug Id")

    bugs = loader.load_bugs(columns=["Bug Id", "Summary", "Product", "Component"])
    bugs = bugs.drop_duplicates(subset="Bug Id")
    df = splits.merge(bugs, on="Bug Id", how="left")

    for col in ("Summary", "Product", "Component"):
        df[col] = df[col].fillna("").astype(str)
    df["raw"] = df["Summary"] + " [SEP] " + df["Product"] + " " + df["Component"]

    if cfg.distillations_path.exists():
        dist = pd.read_parquet(cfg.distillations_path)[["Bug Id", "distilled_text"]]
        df = df.merge(dist.drop_duplicates("Bug Id"), on="Bug Id", how="left")
    else:
        df["distilled_text"] = ""
    df["distilled_text"] = df["distilled_text"].fillna("")

    df["text"] = combine_text(df, text_mode)
    return df, label_encoder


# --------------------------------------------------------------------------- #
# Recomendador
# --------------------------------------------------------------------------- #
class InteractionRecommender:
    """IBR: recupera bugs similares por SBERT y suma interacciones tipadas → NIS."""

    def __init__(self, text_mode: str = "distilled", cfg: Settings = settings):
        self.cfg = cfg
        self.text_mode = text_mode
        self._model = None
        self._train_bug_ids: np.ndarray | None = None      # [N] Bug Id por fila
        self._train_embeddings: torch.Tensor | None = None  # [N, D] L2-normalizado
        self._interactions: dict[int, list[tuple[int, str, pd.Timestamp]]] = {}
        self._active: set[int] = set()                      # directorio de candidatos

    # ---- modelo SBERT (carga perezosa) ----
    def _sbert(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Cargando SBERT: {}", self.cfg.sbert_model)
            self._model = SentenceTransformer(self.cfg.sbert_model, device=self.cfg.torch_device)
        return self._model

    # ---- Interaction Table: une las 3 fuentes y la indexa por bug_id ----
    def _build_interaction_table(self, bug_ids: set[int]) -> dict:
        """`(bug_id → [(dev_id, kind, timestamp)])`, solo para los `bug_ids` del índice."""
        frames = []

        repo = pd.read_parquet(self.cfg.repo_interactions_path)[
            ["bug_id", "Contributor Id", "kind", "timestamp"]
        ].rename(columns={"Contributor Id": "dev"})
        frames.append(repo)

        disc = pd.read_parquet(self.cfg.discussion_interactions_path)[
            ["bug_id", "Contributor Id", "kind", "timestamp"]
        ].rename(columns={"Contributor Id": "dev"})
        frames.append(disc)

        # assignment: el `Assigned To` de cada bug, fechado en su Creation Time.
        spl = pd.read_parquet(self.cfg.splits_path)[["Bug Id", "contributor_id", "creation_time"]]
        spl = spl.rename(columns={"Bug Id": "bug_id", "contributor_id": "dev", "creation_time": "timestamp"})
        spl["kind"] = "assignment"
        frames.append(spl[["bug_id", "dev", "kind", "timestamp"]])

        long = pd.concat(frames, ignore_index=True)
        long = long[long["bug_id"].isin(bug_ids)].copy()
        long["dev"] = long["dev"].astype("int64")
        long["bug_id"] = long["bug_id"].astype("int64")

        table: dict[int, list] = defaultdict(list)
        for bug_id, dev, kind, ts in long[["bug_id", "dev", "kind", "timestamp"]].itertuples(
            index=False, name=None
        ):
            table[int(bug_id)].append((int(dev), kind, ts))
        logger.info(
            "Interaction Table: {} interacciones sobre {} bugs de train",
            len(long), len(table),
        )
        return dict(table)

    # ---- fit: embebe train + construye la tabla ----
    def fit(self, save: bool = True) -> "InteractionRecommender":
        df, label_encoder = load_pilot_text(self.text_mode, self.cfg)
        self._active = {int(d) for d in label_encoder}  # devs del piloto = expected_developers

        train = df[df["split"] == "train"].drop_duplicates("Bug Id")
        self._train_bug_ids = train["Bug Id"].to_numpy()
        texts = train["text"].tolist()
        logger.info("Embebiendo {} bugs de train del piloto...", len(texts))

        emb = self._sbert().encode(
            texts, batch_size=64, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True,
        )
        self._train_embeddings = torch.as_tensor(emb, dtype=torch.float32)
        self._interactions = self._build_interaction_table(set(int(b) for b in self._train_bug_ids))

        if save:
            self.cfg.pilot_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.cfg.ibr_embeddings_path, emb)
            np.save(self.cfg.ibr_bug_ids_path, self._train_bug_ids)
            logger.success("Índice IBR guardado en {}", self.cfg.ibr_embeddings_path)
        return self

    # ---- load: restaura embeddings persistidos y reconstruye la tabla ----
    def load(self) -> "InteractionRecommender":
        if not self.cfg.ibr_embeddings_path.exists():
            raise FileNotFoundError(
                f"No existe {self.cfg.ibr_embeddings_path}. Corre primero "
                "`python -m triager_omega.modules.ibr build`."
            )
        _, label_encoder = load_pilot_text(self.text_mode, self.cfg)
        self._active = {int(d) for d in label_encoder}
        emb = np.load(self.cfg.ibr_embeddings_path)
        self._train_embeddings = torch.as_tensor(emb, dtype=torch.float32)
        self._train_bug_ids = np.load(self.cfg.ibr_bug_ids_path)
        self._interactions = self._build_interaction_table(set(int(b) for b in self._train_bug_ids))
        logger.info("Índice IBR cargado: {} bugs.", len(self._train_bug_ids))
        return self

    # ---- TriagerX: _get_contribution_point ----
    def _ip(self, kind: str) -> float:
        if kind in _CONTRIBUTION_KINDS:
            return self.cfg.ip_contribution
        if kind == "assignment":
            return self.cfg.ip_assignment
        return self.cfg.ip_discussion  # discussion

    # ---- TriagerX: _calculate_time_decay (con t_now por consulta) ----
    def _decay(self, ts: pd.Timestamp, t_now: pd.Timestamp | None) -> float:
        if t_now is None:
            return 1.0
        days = (t_now - ts).days
        if days < 0:  # interacción futura: ya filtrada, pero defensa extra
            return 0.0
        return math.exp(-self.cfg.ibr_lambda * days)

    # ---- scoring desde un embedding ya calculado ----
    def _score(self, query_emb: torch.Tensor, t_now: pd.Timestamp | None) -> dict[int, float]:
        from sentence_transformers import util

        cos = util.cos_sim(query_emb, self._train_embeddings)[0]  # [N]
        k = min(self.cfg.ibr_top_k_retrieve, cos.shape[0])
        vals, idxs = torch.topk(cos, k)
        vals = vals.cpu().numpy()
        idxs = idxs.cpu().numpy()

        # IS[dev] += s_j · IP[kind] · exp(-λ·Δt)   (Algoritmo 1 de TriagerX)
        interaction_score: dict[int, float] = {}
        for sim, idx in zip(vals, idxs):
            if sim < self.cfg.ibr_tau:
                continue
            bug_id = int(self._train_bug_ids[idx])
            for dev, kind, ts in self._interactions.get(bug_id, ()):  # noqa: B007
                if dev not in self._active:
                    continue
                if t_now is not None and ts >= t_now:  # anti-fuga temporal (§7.6)
                    continue
                interaction_score[dev] = interaction_score.get(dev, 0.0) + (
                    float(sim) * self._ip(kind) * self._decay(ts, t_now)
                )
        return self._normalize_nis(interaction_score)

    # ---- NIS (Ec. 7): min-max sobre el vector completo de devs activos ----
    def _normalize_nis(self, interaction_score: dict[int, float]) -> dict[int, float]:
        """Réplica de `_normalize` de TriagerX: los devs sin interacción entran como
        0 (min del vector) → `NIS = IS / max(IS)`. Devuelve NIS para *todos* los
        devs activos (0 para los que no interactuaron)."""
        nis = {dev: 0.0 for dev in self._active}
        if not interaction_score:
            return nis
        mx = max(interaction_score.values())
        if mx <= 0:
            return nis
        for dev, score in interaction_score.items():
            nis[dev] = score / mx
        return nis

    # ---- API pública ----
    def predict(self, query_text: str, t_now: pd.Timestamp | None) -> dict[int, float]:
        """Devuelve `{Contributor Id: NIS}` para el bug consultado."""
        emb = self._sbert().encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        return self._score(torch.as_tensor(emb, dtype=torch.float32), t_now)

    # ---- evaluación IBR-solo (baseline §11.2.2) ----
    def evaluate(self, split: str = "test", ks=(1, 3, 5, 10), limit: int | None = None) -> dict:
        df, _ = load_pilot_text(self.text_mode, self.cfg)
        part = df[df["split"] == split].drop_duplicates("Bug Id").reset_index(drop=True)
        if limit:
            part = part.head(limit)
        logger.info("Evaluando IBR-solo en split={} ({} bugs)...", split, len(part))

        # Embebe todas las queries de una sola pasada (mucho más rápido).
        q_emb = self._sbert().encode(
            part["text"].tolist(), batch_size=64, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True,
        )
        q_emb = torch.as_tensor(q_emb, dtype=torch.float32)

        ranks, signaled = [], []
        for i, row in enumerate(part.itertuples()):
            nis = self._score(q_emb[i : i + 1], row.creation_time)
            true = int(row.contributor_id)
            # rank por NIS desc (desempate determinista por dev id).
            order = sorted(nis.items(), key=lambda kv: (-kv[1], kv[0]))
            rank = next((r for r, (d, _) in enumerate(order) if d == true), len(order))
            ranks.append(rank)
            signaled.append(nis.get(true, 0.0) > 0.0)

        ranks_arr = np.asarray(ranks)
        metrics = {f"hit@{k}": float((ranks_arr < k).mean()) for k in ks}
        metrics["mrr"] = float((1.0 / (ranks_arr + 1)).mean())
        # cobertura: % de bugs donde el IBR le dio señal (NIS>0) al dev real.
        # (techo del IBR-solo: sin señal, el rank del dev real cae entre los empates en 0.)
        metrics["coverage"] = float(np.mean(signaled))
        return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["build", "eval"], help="build: embebe índice · eval: Hit@K/MRR IBR-solo")
    p.add_argument("--text-mode", choices=["raw", "distilled", "both"], default="distilled")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--limit", type=int, default=None, help="evalúa solo los primeros N bugs")
    args = p.parse_args()

    ibr = InteractionRecommender(text_mode=args.text_mode)
    if args.cmd == "build":
        ibr.fit(save=True)
        return

    ibr.load()
    metrics = ibr.evaluate(split=args.split, limit=args.limit)
    logger.success("== IBR-solo | split={} | text_mode={} ==", args.split, args.text_mode)
    for k, v in metrics.items():
        logger.info("  {} = {:.4f}", k, v)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
