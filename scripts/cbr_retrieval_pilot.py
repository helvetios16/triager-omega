"""CBR de RECUPERACIÓN sobre el piloto de Mozilla (validación cruzada del diseño).

Mismo recuperador que `cbr_retrieval_openj9.py` (kNN sobre bugs pasados → voto al
resolvedor), pero sobre el piloto Mozilla (etiqueta = `contributor_id`/assignee, 20
devs, texto raw/distilled/both) en vez de OpenJ9. Sirve para ver si el CBR-recuperación
generaliza a un régimen distinto (pocos devs muy activos, accuracy alta ~0.73) y si
ahí la fusión con el IBR SÍ aporta (en OpenJ9 a 50 clases no aportaba).

Reusa el contrato de datos del piloto (`load_pilot_text`/`combine_text`), el motor del
IBR (`InteractionRecommender`) y `rank_metrics`/`vote_scores` ya existentes.

Uso:
    # barrido del CBR-recuperación zero-shot (k, τ) en test:
    uv run python scripts/cbr_retrieval_pilot.py --text-mode both --sweep
    # sistema completo CBR-recuperación + IBR (barre W_f):
    uv run python scripts/cbr_retrieval_pilot.py --text-mode both --fuse-ibr --top-k 50
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from cbr_retrieval_openj9 import vote_scores
from triager_omega.cbr.predict import normalize_rows
from triager_omega.config import settings
from triager_omega.modules.aggregator import rank_metrics
from triager_omega.modules.ibr import InteractionRecommender, combine_text, load_pilot_text


def _fmt(m: dict) -> str:
    return (f"Hit@1={m['hit@1']:.4f} Hit@3={m['hit@3']:.4f} "
            f"Hit@5={m['hit@5']:.4f} Hit@10={m['hit@10']:.4f} MRR={m['mrr']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--text-mode", choices=["raw", "distilled", "both"], default="both",
                   help="vista del texto para el CBR (igual que el clasificador del piloto)")
    p.add_argument("--ibr-mode", choices=["raw", "distilled", "both"], default="distilled")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--max-seq-len", type=int, default=384)
    p.add_argument("--sweep", action="store_true", help="barre (k, τ) sobre los mismos embeddings")
    p.add_argument("--fuse-ibr", action="store_true", help="fusiona con el IBR y barre W_f")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    cfg = settings
    df, le = load_pilot_text(args.text_mode, cfg)
    C = len(le)
    train = df[df["split"] == "train"].drop_duplicates("Bug Id").reset_index(drop=True)
    test = df[df["split"] == args.split].drop_duplicates("Bug Id").reset_index(drop=True)
    train_cls = train["contributor_id"].astype(str).map(le).to_numpy().astype(int)
    true_idx = test["contributor_id"].astype(str).map(le).to_numpy().astype(int)
    logger.info("CBR-recuperación Mozilla | text={} | train={} {}={} clases={}",
                args.text_mode, len(train), args.split, len(test), C)

    device = "cpu" if args.cpu else cfg.torch_device
    enc = SentenceTransformer(args.model, device=device)
    enc.max_seq_length = args.max_seq_len
    embed = lambda txts: enc.encode(txts, batch_size=64, convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False)
    logger.info("Embebiendo train + {} con {} (maxlen={})...", args.split, args.model, args.max_seq_len)
    emb_tr = embed(combine_text(train, args.text_mode).tolist())
    emb_te = embed(combine_text(test, args.text_mode).tolist())
    sims = emb_te @ emb_tr.T  # coseno [Nte, Ntr]

    if args.sweep:
        logger.success("== Barrido CBR-recuperación Mozilla (zero-shot) | split={} ==", args.split)
        for k in (5, 10, 20, 30, 50, 75, 100):
            for tau in (1.0, 2.0, 4.0):
                m = rank_metrics(vote_scores(sims, train_cls, C, k, tau), true_idx)
                logger.info("  k={:>3} τ={:.0f} : {}", k, tau, _fmt(m))
        return

    nps_raw = vote_scores(sims, train_cls, C, args.top_k, args.tau)
    logger.success("== CBR-recuperación Mozilla | split={} k={} τ={} ==",
                   args.split, args.top_k, args.tau)
    logger.info("  CBR-solo : {}", _fmt(rank_metrics(nps_raw, true_idx)))

    if not args.fuse_ibr:
        return

    # ---------- fusión con el IBR (FS = NPS + W_f·NIS) ----------
    nps = normalize_rows(nps_raw, how="minmax")
    logger.info("IBR ({}): cargando índice y scoring NIS...", args.ibr_mode)
    ibr = InteractionRecommender(text_mode=args.ibr_mode).load()
    q = ibr._sbert().encode(combine_text(test, args.ibr_mode).tolist(), batch_size=64,
                            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    q = torch.as_tensor(q, dtype=torch.float32)
    nis = np.zeros((len(test), C), dtype=np.float32)
    for i, t_now in enumerate(test["creation_time"]):
        d = ibr._score(q[i : i + 1], t_now)  # {dev_id(int): nis}
        for dev_str, idx in le.items():
            nis[i, idx] = d.get(int(dev_str), 0.0)

    logger.success("== Sistema completo CBR-recuperación + IBR | split={} ==", args.split)
    logger.info("  CBR-solo : {}", _fmt(rank_metrics(nps, true_idx)))
    logger.info("  IBR-solo : {}", _fmt(rank_metrics(nis, true_idx)))
    for wf in [round(x, 1) for x in np.arange(0.1, 1.0001, 0.1)]:
        logger.info("  FS W_f={:.1f}: {}", wf, _fmt(rank_metrics(nps + wf * nis, true_idx)))


if __name__ == "__main__":
    main()
