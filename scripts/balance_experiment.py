"""Experimento de balanceo de la cola larga para el CBR (PLAN §11.3 / §14 Semana 3).

Compara, con MISMO modelo + seed + split, cinco estrategias de balanceo y mide
Hit@K y MRR SEGMENTADOS POR BUCKET DE FRECUENCIA del desarrollador (cola / cuerpo
/ cabeza). El Hit@K global enmascara la cola porque la cabeza lo domina; lo que
decide si una estrategia "arregla la cola" es el Hit@K del bucket cola.

Estrategias (cada una es una TRANSFORMACIÓN del set de entrenamiento, agnóstica al
modelo):
  baseline      — set tal cual, sampler uniforme.
  inverse_freq  — sample_weight = 1/freq_train(d)  (análogo a WeightedRandomSampler, §4.2).
  multi_view    — augmentación de la cola (freq<U_AUG): N_VIEWS vistas extra por bug (§5.9).
  undersample   — recorta cada dev a C_MAX bugs para aplanar la distribución.
  combined      — multi_view + inverse_freq (confirma que se apilan).

NOTAS DE SURROGATE (para correr HOY, sin esperar a los módulos 2/3):
  * Clasificador: TF-IDF + clasificador lineal (sample_weight nativo). El arnés es
    agnóstico al modelo; en producción se cambia el backend por DeBERTa (Módulo 3)
    sin tocar estrategias ni métricas. Las estrategias mueven datos, no el modelo,
    así que su comparación relativa es informativa con cualquier clasificador.
  * Texto: crudo "Summary [SEP] Product Component" (la destilación del Módulo 2 aún
    no está corrida; cuando lo esté, basta cambiar build_base_frame()).
  * multi_view: hasta tener la destilación LLM de Gemma, las vistas se generan por
    random token deletion + shuffle (EDA). Misma mecánica que la multi-vista real:
    varias vistas del mismo bug rompen la memorización del string exacto en la cola.

Ejecutar:
    uv run python scripts/balance_experiment.py                  # full train (64k)
    uv run python scripts/balance_experiment.py --max-train 20000  # rápido
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from triager_omega.config import settings
from triager_omega.data import loader

# Cortes de bucket por frecuencia de bugs del dev EN TRAIN.
BUCKET_EDGES = {"cola": (0, 10), "cuerpo": (10, 100), "cabeza": (100, np.inf)}
K_VALUES = (1, 3, 5, 10)


# --------------------------------------------------------------------------- #
# Datos base
# --------------------------------------------------------------------------- #
def build_base_frame() -> pd.DataFrame:
    """Une splits + texto del bug y mapea contributor_id -> class_idx.

    Devuelve un df con: text, label, split, freq_train, bucket.
    """
    label_encoder: dict[str, int] = json.loads(
        settings.label_encoder_path.read_text(encoding="utf-8")
    )
    splits = pd.read_parquet(settings.splits_path)
    bugs = loader.load_bugs(columns=["Bug Id", "Summary", "Product", "Component"])
    df = splits.merge(bugs, on="Bug Id", how="left")

    for col in ("Summary", "Product", "Component"):
        df[col] = df[col].fillna("").astype(str)
    # Texto crudo (sin destilación, Módulo 2 aún no corrido).
    df["text"] = df["Summary"] + " [SEP] " + df["Product"] + " " + df["Component"]

    df["label"] = df["contributor_id"].astype(str).map(label_encoder)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # Frecuencia de cada dev EN TRAIN (define el bucket; clave del experimento).
    freq = df[df["split"] == "train"]["label"].value_counts()
    df["freq_train"] = df["label"].map(freq).fillna(0).astype(int)
    df["bucket"] = df["freq_train"].apply(bucket_of)
    return df


def bucket_of(freq: int) -> str:
    for name, (lo, hi) in BUCKET_EDGES.items():
        if lo < freq <= hi:
            return name
    return "cola"  # freq==0 (no debería pasar en train) cae a cola


# --------------------------------------------------------------------------- #
# Augmentación surrogate (EDA: random deletion + shuffle) para la cola
# --------------------------------------------------------------------------- #
def augment_text(text: str, rng: np.random.Generator, drop: float = 0.3) -> str:
    """Genera UNA vista alternativa: borra ~drop de los tokens y baraja el resto.

    Surrogate de la multi-vista LLM (§5.9). En bag-of-words/TF-IDF con ngramas,
    borrar tokens y barajar produce un vector distinto del mismo bug → el modelo
    no puede memorizar el string exacto de los pocos bugs de un dev de cola.
    """
    tokens = text.split()
    if len(tokens) <= 4:
        return text
    keep = [t for t in tokens if rng.random() > drop]
    if len(keep) < 3:
        keep = tokens[:]
    rng.shuffle(keep)
    return " ".join(keep)


# --------------------------------------------------------------------------- #
# Estrategias: cada una devuelve (texts, labels, sample_weight|None)
# --------------------------------------------------------------------------- #
def strat_baseline(train: pd.DataFrame, *_):
    return train["text"].tolist(), train["label"].to_numpy(), None


def strat_inverse_freq(train: pd.DataFrame, *_):
    # Peso = 1/freq de la clase dentro del train (cola pesa más). Normalizado a media 1.
    freq = train["label"].value_counts()
    w = np.array(1.0 / train["label"].map(freq), dtype=np.float64)
    w *= len(w) / w.sum()
    return train["text"].tolist(), train["label"].to_numpy(), w


def strat_multi_view(train: pd.DataFrame, args, rng: np.random.Generator, weights=False):
    # Para devs de cola (freq<U_AUG) añade N_VIEWS vistas augmentadas por bug.
    freq = train["label"].value_counts()
    texts = train["text"].tolist()
    labels = train["label"].tolist()
    tail = train[train["label"].map(freq) < args.u_aug]
    for txt, lab in zip(tail["text"], tail["label"]):
        for _ in range(args.n_views):
            texts.append(augment_text(txt, rng))
            labels.append(lab)
    labels = np.asarray(labels)
    if not weights:
        return texts, labels, None
    # combined: tras augmentar, aplica además peso inverso a la frecuencia AUMENTADA.
    lab_s = pd.Series(labels)
    new_freq = lab_s.value_counts()
    w = np.array(1.0 / lab_s.map(new_freq), dtype=np.float64)
    w *= len(w) / w.sum()
    return texts, labels, w


def strat_undersample(train: pd.DataFrame, args, rng: np.random.Generator):
    # Recorta cada dev a C_MAX bugs (aplana la cabeza). Mantiene todas las clases.
    idx = (
        train.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), args.c_max), random_state=args.seed))
        .index
    )
    sub = train.loc[idx]
    return sub["text"].tolist(), sub["label"].to_numpy(), None


def strat_combined(train: pd.DataFrame, args, rng: np.random.Generator):
    return strat_multi_view(train, args, rng, weights=True)


STRATEGIES = {
    "baseline": strat_baseline,
    "inverse_freq": strat_inverse_freq,
    "multi_view": strat_multi_view,
    "undersample": strat_undersample,
    "combined": strat_combined,
}


# --------------------------------------------------------------------------- #
# Métricas Hit@K / MRR segmentadas por bucket
# --------------------------------------------------------------------------- #
def rank_of_true(proba: np.ndarray, classes: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Rango 0-based del label verdadero en el ranking por probabilidad.

    Si el label no está entre las clases entrenadas, devuelve un rango enorme (miss).
    """
    cls_to_col = {c: i for i, c in enumerate(classes)}
    cols = np.array([cls_to_col.get(y, -1) for y in y_true])
    ranks = np.full(len(y_true), 10**9, dtype=np.int64)
    seen = cols >= 0
    true_p = proba[np.arange(len(y_true))[seen], cols[seen]]
    # rango = nº de clases con probabilidad estrictamente mayor que la verdadera.
    ranks[seen] = (proba[seen] > true_p[:, None]).sum(axis=1)
    return ranks


def metrics_from_ranks(ranks: np.ndarray) -> dict[str, float]:
    out = {f"hit@{k}": float((ranks < k).mean()) for k in K_VALUES}
    out["mrr"] = float((1.0 / (ranks + 1)).mean())
    out["n"] = int(len(ranks))
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-train", type=int, default=0, help="submuestrea train (0 = todo)")
    p.add_argument("--max-features", type=int, default=40000)
    p.add_argument("--u-aug", type=int, default=10, help="umbral de cola para multi_view")
    p.add_argument("--n-views", type=int, default=3, help="vistas extra por bug de cola")
    p.add_argument("--c-max", type=int, default=200, help="tope de bugs/dev en undersample")
    p.add_argument("--seed", type=int, default=settings.seed)
    p.add_argument(
        "--strategies",
        default="baseline,inverse_freq,multi_view,undersample,combined",
        help="lista separada por comas",
    )
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    df = build_base_frame()
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"].copy()
    if args.max_train and len(train) > args.max_train:
        train = train.sample(n=args.max_train, random_state=args.seed)
    logger.info("train={} test={} clases={}", len(train), len(test), df["label"].nunique())
    logger.info(
        "buckets test (por freq del dev en train): {}",
        test["bucket"].value_counts().to_dict(),
    )

    # Vectorizador TF-IDF: se AJUSTA UNA SOLA VEZ sobre el train baseline para que
    # el espacio de features sea idéntico entre estrategias → comparación justa.
    logger.info("Ajustando TF-IDF (max_features={})...", args.max_features)
    vec = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2), min_df=2)
    vec.fit(train["text"].tolist())
    X_test = vec.transform(test["text"].tolist())
    y_test = test["label"].to_numpy()
    test_bucket = test["bucket"].to_numpy()

    rows = []
    for name in args.strategies.split(","):
        name = name.strip()
        if name not in STRATEGIES:
            logger.warning("estrategia desconocida: {}", name)
            continue
        logger.info("== Estrategia: {} ==", name)
        texts, labels, weights = STRATEGIES[name](train, args, rng)
        Xtr = vec.transform(texts)
        clf = SGDClassifier(
            loss="log_loss", alpha=1e-5, max_iter=20, tol=1e-3, random_state=args.seed
        )
        clf.fit(Xtr, labels, sample_weight=weights)
        proba = clf.predict_proba(X_test)
        ranks = rank_of_true(proba, clf.classes_, y_test)

        # Global + por bucket.
        for seg in ("global", "cola", "cuerpo", "cabeza"):
            mask = slice(None) if seg == "global" else (test_bucket == seg)
            m = metrics_from_ranks(ranks[mask])
            rows.append({"strategy": name, "segment": seg, **m})
            logger.info(
                "  [{:>7}] n={:5d}  Hit@1={:.3f} Hit@5={:.3f} Hit@10={:.3f} MRR={:.3f}",
                seg, m["n"], m["hit@1"], m["hit@5"], m["hit@10"], m["mrr"],
            )

    res = pd.DataFrame(rows)
    out_dir = settings.artifacts_dir / "balance_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    res.to_csv(csv_path, index=False)

    # Tabla pivote legible: Hit@5 y MRR por estrategia × bucket (lo que decide todo).
    print("\n================ Hit@5 por bucket (lo que importa) ================")
    pivot5 = res.pivot(index="strategy", columns="segment", values="hit@5")
    pivot5 = pivot5[["cola", "cuerpo", "cabeza", "global"]]
    print(pivot5.round(3).to_string())
    print("\n================ MRR por bucket ================")
    pivotm = res.pivot(index="strategy", columns="segment", values="mrr")
    pivotm = pivotm[["cola", "cuerpo", "cabeza", "global"]]
    print(pivotm.round(3).to_string())
    print(f"\nResultados completos -> {csv_path}")


if __name__ == "__main__":
    main()
