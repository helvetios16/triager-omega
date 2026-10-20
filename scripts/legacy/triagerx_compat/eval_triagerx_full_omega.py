"""Evalúa el SISTEMA COMPLETO de TriagerX (CBR ensemble + IBR, fusión WRA y Borda)
sobre NUESTRO split de OpenJ9, reusando su código real (triagerx/system/triagerx.py).

Produce Top-1/3/5/10/20 para: CBR-solo (dev model), IBR-solo (similar_devs),
WRA (combined_ranking) y Borda — la tabla head-to-head de su paper.

Notas de cableado:
- Los componentes NO afectan el ranking de devs (get_recommendation los predice pero
  _aggregate_rankings solo usa CBR+IBR), así que se pasa el dev model como modelo de
  componente dummy y se ignora su salida.
- El IBR baja los logins a minúscula (`_get_historical_contributors`), así que el
  developer_id_map y expected_developers se construyen en minúscula, y el match top-k
  se hace en minúscula. El ORDEN de clases (sorted owners) replica el del entrenamiento
  para alinear el índice de salida del CBR.
- train_checkpoint_date = máx. created_at de los issues de train (cutoff temporal).

Uso (omen):
  python eval_triagerx_full_omega.py --triagerx-root <...> \
     --train-csv ... --test-csv ... --weights <...>.pt \
     --issues-path <...>\\issue_data --raw <...>\\artifacts\\openj9\\raw \
     --emb <...>\\train_embeddings.npy --out <...>\\triagerx_full_results.json
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

tqdm.pandas()

# Mejores hiperparámetros del IBR de TriagerX (assets/best_param.txt, OpenJ9)
BEST = dict(
    similarity_prediction_weight=1.0,
    time_decay_factor=0.01,
    direct_assignment_score=1.0,
    contribution_score=0.5,
    discussion_score=0.5,
    similarity_threshold=0.45,
)
K_RANK = 15
TOPKS = [1, 3, 5, 10, 20]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--triagerx-root", required=True)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--weights", required=True, help="checkpoint .pt del dev model entrenado")
    p.add_argument("--issues-path", required=True, help="dir issue_data/{n}.json")
    p.add_argument("--raw", required=True, help="dir raw/{n}.json (para created_at del checkpoint)")
    p.add_argument("--emb", required=True, help="ruta train_embeddings.npy (se genera si falta)")
    p.add_argument("--out", required=True, help="JSON de salida con la tabla")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=0, help="si >0, evalúa solo N test (smoke)")
    return p.parse_args()


def clean_split(path, TextProcessor):
    """Construye `text` con el MISMO clean_text de TriagerX con que se entrenó el modelo
    (si no, pasarle texto crudo al CBR lo penaliza injustamente)."""
    df = pd.read_csv(path)
    if "issue_url" not in df.columns:
        df["issue_url"] = df["issue_number"].apply(
            lambda n: f"https://github.com/eclipse-openj9/openj9/issues/{n}")
    if "labels" not in df.columns:
        df["labels"] = df["component"] if "component" in df.columns else ""
    df = df.rename(columns={"issue_body": "description"})
    df = TextProcessor.prepare_dataframe(
        df, use_special_tokens=False, use_summary=False,
        use_description=True, component_training=False)
    return df.sort_values("issue_number").reset_index(drop=True)


def checkpoint_date(raw_dir, train_ids):
    """Máx created_at entre los issues de train (cutoff temporal del entrenamiento)."""
    best = None
    for n in train_ids:
        f = os.path.join(raw_dir, f"{n}.json")
        if not os.path.exists(f):
            continue
        try:
            ca = json.load(open(f, encoding="utf-8")).get("created_at")
            if ca:
                d = datetime.strptime(ca[:19], "%Y-%m-%dT%H:%M:%S")
                if best is None or d > best:
                    best = d
        except Exception:
            continue
    return best or datetime(2024, 11, 22)


def main():
    args = parse_args()
    sys.path.insert(0, args.triagerx_root)
    from triagerx.model.module_factory import ModelFactory
    from triagerx.system.triagerx import TriagerX
    from triagerx.dataset.text_processor import TextProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df_train = clean_split(args.train_csv, TextProcessor)
    df_test = clean_split(args.test_csv, TextProcessor)
    if args.limit:
        df_test = df_test.head(args.limit)

    # MISMO orden de clases que el entrenamiento: sorted(train owners)
    train_owners = sorted(set(df_train["owner"]))
    lbl2idx = {o: i for i, o in enumerate(train_owners)}
    df_train["owner_id"] = df_train["owner"].map(lbl2idx)
    idx2lbl = {i: o for o, i in lbl2idx.items()}

    # maps en MINÚSCULA para el IBR (logins del timeline van en minúscula)
    dev_id_map = {o.lower(): i for o, i in lbl2idx.items()}
    expected_devs = set(dev_id_map.keys())

    logger.info(f"clases={len(train_owners)} | train={len(df_train)} | test={len(df_test)} | device={device}")

    base_models = ["microsoft/deberta-base", "roberta-base"]
    dev_model = ModelFactory.get_model(
        model_key="triagerx", output_size=len(train_owners), unfrozen_layers=3,
        num_classifiers=3, base_models=base_models, dropout=0.2,
        max_tokens=args.max_tokens, label_map=idx2lbl,
    )
    dev_model.load_state_dict(torch.load(args.weights, map_location=device))
    logger.info("dev model cargado.")

    sim_model = SentenceTransformer("all-mpnet-base-v2", device=device)
    if not os.path.exists(args.emb):
        logger.info("Generando train_embeddings.npy (MPNet sobre train)...")
        enc = sim_model.encode(df_train["text"].tolist(), show_progress_bar=True)
        np.save(args.emb, enc)
        logger.info(f"Guardado {args.emb} shape={np.array(enc).shape}")

    ckpt_date = checkpoint_date(args.raw, df_train["issue_number"].tolist())
    logger.info(f"train_checkpoint_date={ckpt_date}")

    trx = TriagerX(
        component_prediction_model=dev_model,   # dummy: componentes no afectan el ranking de devs
        developer_prediction_model=dev_model,
        similarity_model=sim_model,
        train_data=df_train,
        train_embeddings=args.emb,
        issues_path=args.issues_path,
        developer_id_map=dev_id_map,
        component_id_map=lbl2idx,                # dummy {nombre: id}, salida ignorada
        expected_developers=expected_devs,
        device=device,
        train_checkpoint_date=ckpt_date,
        # similarity_threshold NO es de __init__ (va en get_recommendation)
        **{k: v for k, v in BEST.items() if k != "similarity_threshold"},
    )

    n_dev = len(train_owners)
    hits = {m: {k: 0 for k in TOPKS} for m in ["cbr", "ibr", "wra", "borda"]}
    for i in tqdm(range(len(df_test)), desc="eval test"):
        actual = str(df_test.iloc[i]["owner"]).lower()
        rec = trx.get_recommendation(
            df_test.iloc[i]["text"], k_comp=2, k_dev=n_dev, k_rank=K_RANK,
            similarity_threshold=BEST["similarity_threshold"],
        )
        ranks = {
            "cbr": [d.lower() for d in rec["predicted_developers"]],
            "ibr": [d.lower() for d in rec["similar_devs"]],
            "wra": [d.lower() for d in rec["combined_ranking"]],
            "borda": [d.lower() for d in rec["borda_ranking"]],
        }
        for m, r in ranks.items():
            for k in TOPKS:
                if actual in r[:k]:
                    hits[m][k] += 1

    n = len(df_test)
    table = {m: {f"top{k}": round(hits[m][k] / n, 4) for k in TOPKS} for m in hits}
    logger.info("RESULTADOS TriagerX full (nuestro split):")
    for m in ["cbr", "ibr", "wra", "borda"]:
        logger.info(f"  {m.upper():6s} " + " ".join(f"@{k}={table[m][f'top{k}']}" for k in TOPKS))
    json.dump({"n_test": n, "n_classes": n_dev, "k_rank": K_RANK, "best_param": BEST, "table": table},
              open(args.out, "w"), indent=2)
    logger.info(f"Guardado {args.out}")


if __name__ == "__main__":
    main()
