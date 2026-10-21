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
    usamos `t_now = Creation Time` del bug consultado y omitimos interacciones
    `t ≥ t_now`. Assignment usa `creation_time` como proxy temporal, así que esto
    no constituye una garantía absoluta contra toda fuga (ver §7.6 del PLAN).

  - **NIS** (Ec. 7 de TriagerX) sobre el vector completo de candidatos activos:
    los que no interactuaron entran como 0. La implementación divide IS por su
    máximo positivo (div-max; no min-max). La agregación `FS = NPS + W_f·NIS`
    (Ec. 8) vive en el Módulo 5 (aggregator).

CLI:
    uv run python -m triager_omega.modules.ibr build          # embebe + guarda índice
    uv run python -m triager_omega.modules.ibr eval           # IBR-solo Hit@K/MRR en test
    uv run python -m triager_omega.modules.ibr eval --split val --limit 300
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Sequence, Set as AbstractSet

import numpy as np
import pandas as pd
import torch
from loguru import logger

from triager_omega.config import Settings, settings
from triager_omega.data import loader

# Tipos de interacción que comparten el peso `ip_contribution` (commit + review),
# igual que TriagerX agrupa pull_request + commits en contribution_score.
_CONTRIBUTION_KINDS = frozenset({"commit", "review", "pull_request", "commits"})


@dataclass(frozen=True)
class IbrConfig:
    # Canonical Settings defaults: 20 similar cases and cosine threshold 0.6.
    top_k: int = 20
    tau: float = 0.6
    decay_lambda: float = 0.01
    ip_contribution: float = 1.5
    ip_assignment: float = 0.5
    ip_discussion: float = 0.1


@dataclass
class InteractionIndex:
    """Corpus-independent IBR data; ``case_ids`` align with similarity columns."""
    case_ids: np.ndarray
    active_candidates: Sequence | AbstractSet  # ordered sequence preferred; sets remain accepted for compatibility
    interactions: dict


def interaction_table_from_rows(rows, case_ids=None) -> dict:
    """Construye tabla genérica desde filas (case_id, dev, kind, timestamp)."""
    allowed = None if case_ids is None else set(case_ids)
    table = defaultdict(list)
    for case_id, dev, kind, timestamp in rows:
        if allowed is None or case_id in allowed:
            table[case_id].append((dev, kind, timestamp))
    return dict(table)


class IbrScorer:
    """Scoring puro sobre similitudes ya calculadas, sin encoder ni corpus."""
    def __init__(self, config: IbrConfig, index: InteractionIndex):
        self.config, self.index = config, index

    def interaction_point(self, kind: str) -> float:
        if kind in _CONTRIBUTION_KINDS:
            return self.config.ip_contribution
        if kind == "assignment":
            return self.config.ip_assignment
        return self.config.ip_discussion

    def decay(self, ts, t_now) -> float:
        if t_now is None:
            return 1.0
        days = (t_now - ts).days
        if days < 0:
            return 0.0
        return math.exp(-self.config.decay_lambda * days)

    def normalize(self, values: dict) -> dict:
        nis = {dev: 0.0 for dev in self.index.active_candidates}
        if not values:
            return nis
        maximum = max(values.values())
        if maximum <= 0:
            return nis
        for dev, score in values.items():
            nis[dev] = score / maximum
        return nis

    def score(self, similarities, t_now=None, *, policy="semantic") -> dict:
        """Score un query. semantic conserva torch.topk/umbral; lexical argpartition/>0."""
        sims = np.asarray(similarities)
        if sims.ndim != 1:
            raise ValueError(f"similarities debe ser un vector 1D; recibido ndim={sims.ndim}.")
        if len(sims) != len(self.index.case_ids):
            raise ValueError(
                f"similarities y case_ids deben tener igual longitud; "
                f"recibido {len(sims)} y {len(self.index.case_ids)}."
            )
        k = min(self.config.top_k, len(sims))
        if k == 0:
            return self.normalize({})
        if policy == "semantic":
            vals, indices = torch.topk(torch.as_tensor(sims), k)
            selected = zip(vals.cpu().numpy(), indices.cpu().numpy())
        elif policy == "lexical":
            indices = np.argpartition(-sims, kth=k - 1)[:k]
            selected = ((sims[i], i) for i in indices)
        else:
            raise ValueError("policy debe ser 'semantic' o 'lexical'")
        values = {}
        for similarity, idx in selected:
            similarity = float(similarity)
            if policy == "semantic" and similarity < self.config.tau:
                continue
            if policy == "lexical" and similarity <= 0:
                continue
            for dev, kind, ts in self.index.interactions.get(self.index.case_ids[int(idx)], ()):
                if dev not in self.index.active_candidates:
                    continue
                if t_now is not None and ts >= t_now:
                    continue
                values[dev] = values.get(dev, 0.0) + similarity * self.interaction_point(kind) * self.decay(ts, t_now)
        return self.normalize(values)


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
        cfg.pilot_read_path("label_encoder.json").read_text(encoding="utf-8")
    )
    splits = pd.read_parquet(cfg.pilot_read_path("splits.parquet")).drop_duplicates(subset="Bug Id")

    bugs = loader.load_bugs(columns=["Bug Id", "Summary", "Product", "Component"])
    bugs = bugs.drop_duplicates(subset="Bug Id")
    df = splits.merge(bugs, on="Bug Id", how="left")

    for col in ("Summary", "Product", "Component"):
        df[col] = df[col].fillna("").astype(str)
    df["raw"] = df["Summary"] + " [SEP] " + df["Product"] + " " + df["Component"]

    distillations_path = cfg.pilot_read_path("distillations.parquet")
    if distillations_path.exists():
        dist = pd.read_parquet(distillations_path)[["Bug Id", "distilled_text"]]
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

        repo = pd.read_parquet(self.cfg.mozilla_read_path("repo_interactions.parquet"))[
            ["bug_id", "Contributor Id", "kind", "timestamp"]
        ].rename(columns={"Contributor Id": "dev"})
        frames.append(repo)

        disc = pd.read_parquet(self.cfg.mozilla_read_path("discussion_interactions.parquet"))[
            ["bug_id", "Contributor Id", "kind", "timestamp"]
        ].rename(columns={"Contributor Id": "dev"})
        frames.append(disc)

        # assignment: el `Assigned To` de cada bug, fechado en su Creation Time.
        spl = pd.read_parquet(self.cfg.mozilla_read_path("splits.parquet"))[["Bug Id", "contributor_id", "creation_time"]]
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
        embeddings_path = self.cfg.pilot_read_path("ibr_embeddings.npy")
        bug_ids_path = self.cfg.pilot_read_path("ibr_bug_ids.npy")
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"No existe {embeddings_path}. Corre primero "
                "`python -m triager_omega.modules.ibr build`."
            )
        _, label_encoder = load_pilot_text(self.text_mode, self.cfg)
        self._active = {int(d) for d in label_encoder}
        emb = np.load(embeddings_path)
        self._train_embeddings = torch.as_tensor(emb, dtype=torch.float32)
        self._train_bug_ids = np.load(bug_ids_path)
        self._interactions = self._build_interaction_table(set(int(b) for b in self._train_bug_ids))
        logger.info("Índice IBR cargado: {} bugs.", len(self._train_bug_ids))
        return self

    def _scorer(self) -> IbrScorer:
        config = IbrConfig(self.cfg.ibr_top_k_retrieve, self.cfg.ibr_tau, self.cfg.ibr_lambda,
                           self.cfg.ip_contribution, self.cfg.ip_assignment, self.cfg.ip_discussion)
        return IbrScorer(config, InteractionIndex(self._train_bug_ids, self._active, self._interactions))

    def set_index(self, index: InteractionIndex) -> "InteractionRecommender":
        self._train_bug_ids = np.asarray(index.case_ids)
        self._active = index.active_candidates
        self._interactions = index.interactions
        return self

    def score_similarities(self, similarities, t_now=None, *, policy="semantic") -> dict:
        return self._scorer().score(similarities, t_now, policy=policy)

    def cosine_similarities(self, query_embedding):
        """Calcula la fila de similitudes con la API cos_sim de Sentence-Transformers."""
        from sentence_transformers import util
        return util.cos_sim(query_embedding, self._train_embeddings)[0]

    def score_embedding(self, query_embedding, t_now=None) -> dict:
        similarities = self.cosine_similarities(query_embedding)
        return self.score_similarities(similarities.cpu().numpy(), t_now)

    def encode_texts(self, texts, *, batch_size=64, show_progress_bar=False):
        """Codifica usando el encoder IBR (independiente del encoder CBR)."""
        return self._sbert().encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=show_progress_bar)

    # ---- TriagerX: _get_contribution_point ----
    def _ip(self, kind: str) -> float:
        return self._scorer().interaction_point(kind)

    # ---- TriagerX: _calculate_time_decay (con t_now por consulta) ----
    def _decay(self, ts: pd.Timestamp, t_now: pd.Timestamp | None) -> float:
        return self._scorer().decay(ts, t_now)

    # ---- scoring desde un embedding ya calculado ----
    def _score(self, query_emb: torch.Tensor, t_now: pd.Timestamp | None) -> dict[int, float]:
        cos = self.cosine_similarities(query_emb)
        return self.score_similarities(cos.cpu().numpy(), t_now)

    # ---- NIS (Ec. 7): división por el máximo positivo sobre devs activos ----
    def _normalize_nis(self, interaction_score: dict[int, float]) -> dict[int, float]:
        """Normaliza IS por su máximo positivo y devuelve todos los candidatos activos."""
        return self._scorer().normalize(interaction_score)

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
