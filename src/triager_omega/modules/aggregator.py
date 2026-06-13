"""Módulo 5 — Agregador WRA: FS = NPS + W_f·NIS (réplica de TriagerX, Ec. 8).

Fusiona el CBR (NPS, Módulo 3) y el IBR (NIS, Módulo 4) con la fórmula **aditiva**
de TriagerX (`_adjust_dev_scores_by_similarity` / `_aggregate_rankings`):

    FS(dev) = NPS(dev) + W_f · NIS(dev)

- Aditiva, no convexa: un dev sin historial conserva su NPS puro; uno con historial
  recibe un empujón `W_f·NIS`. NPS y NIS están en la misma escala [0,1] (min-max),
  así que la suma es coherente.
- `W_f` = `ibr_w_f` (0.7, = `similarity_prediction_weight` de TriagerX). Se sintoniza
  por grid search en validación (§8.3, §10.2 paso 4); `--grid` lo recorre.
- Candidate-constrained: ambas modalidades operan sobre el directorio activo (las 20
  clases del piloto), así que no hay devs fuera del espacio.

Todo se alinea por `class_idx` del `label_encoder` (mismo orden en CBR e IBR), así que
la fusión es una suma de matrices `[N, num_classes]`.

CLI:
    uv run python -m triager_omega.modules.aggregator eval                 # full system en test
    uv run python -m triager_omega.modules.aggregator eval --grid          # barre W_f
    uv run python -m triager_omega.modules.aggregator eval --split val --grid
    uv run python -m triager_omega.modules.aggregator eval --w-f 0.7 --nps-norm minmax
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from loguru import logger

from triager_omega.cbr.predict import CbrPredictor
from triager_omega.config import Settings, settings
from triager_omega.modules.ibr import InteractionRecommender, combine_text, load_pilot_text


def rank_metrics(scores: np.ndarray, true_idx: np.ndarray, ks=(1, 3, 5, 10)) -> dict:
    """Hit@K y MRR a partir de la matriz de scores `[N, C]` y la clase verdadera.

    Desempate estable (kind='stable'): a igual score gana la clase de menor índice."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = (order == true_idx[:, None]).argmax(axis=1)
    metrics = {f"hit@{k}": float((ranks < k).mean()) for k in ks}
    metrics["mrr"] = float((1.0 / (ranks + 1)).mean())
    return metrics


class TriagerAggregator:
    """Combina NPS (CBR) y NIS (IBR) en FS = NPS + W_f·NIS y rankea Top-K."""

    def __init__(
        self,
        cbr: CbrPredictor,
        ibr: InteractionRecommender,
        w_f: float | None = None,
        nps_norm: str = "minmax",
        cfg: Settings = settings,
    ):
        self.cbr = cbr
        self.ibr = ibr
        self.cfg = cfg
        self.w_f = cfg.ibr_w_f if w_f is None else w_f
        self.nps_norm = nps_norm
        self.label_encoder: dict[str, int] = json.loads(
            cfg.pilot_label_encoder_path.read_text(encoding="utf-8")
        )
        self.num_classes = len(self.label_encoder)

    # ---- NIS del IBR → matriz [N, C] alineada por class_idx ----
    def _nis_matrix(self, ibr_texts: list[str], t_nows) -> np.ndarray:
        q_emb = self.ibr._sbert().encode(
            ibr_texts, batch_size=64, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True,
        )
        q_emb = torch.as_tensor(q_emb, dtype=torch.float32)
        nis_mat = np.zeros((len(ibr_texts), self.num_classes), dtype=np.float32)
        for i, t_now in enumerate(t_nows):
            nis = self.ibr._score(q_emb[i : i + 1], t_now)  # {dev_id: nis}
            for dev_str, idx in self.label_encoder.items():
                nis_mat[i, idx] = nis.get(int(dev_str), 0.0)
        return nis_mat

    def _matrices(self, split: str, limit: int | None = None):
        """Devuelve (nps_mat, nis_mat, true_idx) para el split pedido."""
        df, _ = load_pilot_text(self.cbr.text_mode, self.cfg)
        part = df[df["split"] == split].drop_duplicates("Bug Id").reset_index(drop=True)
        if limit:
            part = part.head(limit)
        logger.info("Agregador | split={} | {} bugs", split, len(part))

        cbr_texts = combine_text(part, self.cbr.text_mode).tolist()
        ibr_texts = combine_text(part, self.ibr.text_mode).tolist()

        logger.info("NPS (CBR, norm={})...", self.nps_norm)
        nps_mat = self.cbr.nps_matrix(cbr_texts, how=self.nps_norm)
        logger.info("NIS (IBR)...")
        nis_mat = self._nis_matrix(ibr_texts, list(part["creation_time"]))

        true_idx = part["contributor_id"].astype(str).map(self.label_encoder).to_numpy()
        return nps_mat, nis_mat, true_idx.astype(int)

    # ---- evaluación del sistema completo ----
    def evaluate(self, split: str = "test", ks=(1, 3, 5, 10), limit: int | None = None) -> dict:
        nps_mat, nis_mat, true_idx = self._matrices(split, limit)
        fs = nps_mat + self.w_f * nis_mat
        metrics = rank_metrics(fs, true_idx, ks)
        metrics["w_f"] = self.w_f
        # referencias de las modalidades aisladas (mismo conjunto), para comparar.
        metrics["_cbr_only"] = rank_metrics(nps_mat, true_idx, ks)   # FS con W_f=0
        metrics["_ibr_only"] = rank_metrics(nis_mat, true_idx, ks)
        return metrics

    def grid_search_wf(self, split: str = "val", ks=(1, 3, 5, 10),
                       select="hit@5", grid=None, limit: int | None = None) -> dict:
        """Barre W_f sobre `grid` (default 0.0..1.0 paso 0.1) y elige el mejor `select`.

        Las matrices NPS/NIS se calculan una sola vez; la fusión por W_f es trivial."""
        if grid is None:
            grid = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.1)]
        nps_mat, nis_mat, true_idx = self._matrices(split, limit)

        table = []
        for wf in grid:
            m = rank_metrics(nps_mat + wf * nis_mat, true_idx, ks)
            m["w_f"] = wf
            table.append(m)
        best = max(table, key=lambda m: m[select])
        return {"select": select, "best": best, "grid": table}


def _build(cbr_mode: str, ibr_mode: str, w_f: float | None, nps_norm: str) -> TriagerAggregator:
    cbr = CbrPredictor(text_mode=cbr_mode)
    ibr = InteractionRecommender(text_mode=ibr_mode).load()
    return TriagerAggregator(cbr, ibr, w_f=w_f, nps_norm=nps_norm)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["eval"], default="eval", nargs="?")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--cbr-mode", choices=["raw", "distilled", "both"], default="both",
                   help="vista con la que se entrenó el CBR (carpeta cbr_model_<modo>)")
    p.add_argument("--ibr-mode", choices=["raw", "distilled", "both"], default="distilled",
                   help="vista con la que se construyó el índice IBR")
    p.add_argument("--w-f", type=float, default=None, help="peso del IBR (default: ibr_w_f=0.7)")
    p.add_argument("--nps-norm", choices=["minmax", "softmax"], default="minmax")
    p.add_argument("--grid", action="store_true", help="barre W_f y reporta el mejor por Hit@5")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    agg = _build(args.cbr_mode, args.ibr_mode, args.w_f, args.nps_norm)

    if args.grid:
        res = agg.grid_search_wf(split=args.split, limit=args.limit)
        logger.success("== Grid W_f | split={} | criterio={} ==", args.split, res["select"])
        for m in res["grid"]:
            logger.info("  W_f={:.1f} | hit@1={hit@1:.4f} hit@5={hit@5:.4f} hit@10={hit@10:.4f} mrr={mrr:.4f}",
                        m["w_f"], **m)
        logger.success("Mejor: W_f={:.1f} → {}", res["best"]["w_f"],
                       {k: round(v, 4) for k, v in res["best"].items() if k != "w_f"})
        print(json.dumps(res, indent=2))
        return

    metrics = agg.evaluate(split=args.split, limit=args.limit)
    logger.success("== Sistema completo (FS=NPS+W_f·NIS) | split={} | W_f={} ==",
                   args.split, metrics["w_f"])
    for k in ("hit@1", "hit@3", "hit@5", "hit@10", "mrr"):
        logger.info("  FS  {} = {:.4f}", k, metrics[k])
    logger.info("  (CBR-solo: {})", {k: round(v, 4) for k, v in metrics["_cbr_only"].items()})
    logger.info("  (IBR-solo: {})", {k: round(v, 4) for k, v in metrics["_ibr_only"].items()})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
