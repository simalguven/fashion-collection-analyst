"""Measure how well "find similar" retrieves products of the same type.

A neighbour counts as correct when it shares the query's catalogue label. Labels
are used only for scoring, because retrieval itself sees only image vectors.

Usage: python scripts/evaluate_similarity.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from catalog_vision import embeddings, quality
from catalog_vision.config import ARTIFACT_DIR, BACKBONES, IMAGE_DIR, REPORT_DIR

K_VALUES = (1, 5, 10)
LABELS = ("articleType", "subCategory")
BOOTSTRAP_ROUNDS = 2000
MIN_TYPE_SIZE = 10
RNG = np.random.default_rng(0)


def load_development() -> tuple[pd.DataFrame, dict]:
    items = pd.read_parquet(ARTIFACT_DIR / "items.parquet")
    return items, {b: np.load(ARTIFACT_DIR / f"embeddings_{b}.npy") for b in BACKBONES}


def load_test() -> tuple[pd.DataFrame, dict]:
    """Load the unseen sample, embedding it the first time."""
    raw = pd.read_parquet(ARTIFACT_DIR / "test_raw.parquet")
    _, flags = quality.check_images(raw)
    items = raw[~flags["drop"].to_numpy()].dropna(subset=["articleType"]).reset_index(drop=True)
    vectors = {}
    for backbone in BACKBONES:
        path = ARTIFACT_DIR / f"test_embeddings_{backbone}.npy"
        if not path.exists():
            np.save(path, embeddings.embed_images([IMAGE_DIR / p for p in items["path"]], backbone))
        vectors[backbone] = np.load(path)
    return items, vectors


def nearest(vectors: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of each item's top-k neighbours, excluding itself."""
    scores = vectors @ vectors.T
    np.fill_diagonal(scores, -np.inf)
    return np.argsort(-scores, axis=1)[:, :k]


def hit_matrix(vectors: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    return labels[nearest(vectors, k)] == labels[:, None]


def per_query_scores(hits: np.ndarray) -> dict:
    scores = {}
    for k in K_VALUES:
        scores[f"recall@{k}"] = hits[:, :k].any(axis=1).astype(float)
        scores[f"precision@{k}"] = hits[:, :k].mean(axis=1)
    return scores


def random_baseline(labels: np.ndarray) -> dict:
    """Expected score if neighbours were picked at random."""
    counts = pd.Series(labels).map(pd.Series(labels).value_counts()).to_numpy()
    others = len(labels) - 1
    positives = counts - 1
    baseline = {}
    for k in K_VALUES:
        baseline[f"recall@{k}"] = round(float(np.mean(1 - hypergeom.pmf(0, others, positives, k))), 3)
        baseline[f"precision@{k}"] = round(float(np.mean(positives / others)), 3)
    return baseline


def bootstrap_interval(values: np.ndarray) -> list[float]:
    picks = RNG.integers(0, len(values), (BOOTSTRAP_ROUNDS, len(values)))
    means = values[picks].mean(axis=1)
    return [round(float(q), 3) for q in np.percentile(means, [2.5, 97.5])]


def evaluate(items: pd.DataFrame, vectors: dict, label: str) -> dict:
    labels = items[label].to_numpy(dtype=object)
    counts = items[label].map(items[label].value_counts()).to_numpy()
    usable = counts > 1  # A query with no other item of its type cannot succeed.

    scores = {b: per_query_scores(hit_matrix(v, labels, max(K_VALUES))[usable]) for b, v in vectors.items()}
    result = {"queries": int(usable.sum()), "random": random_baseline(labels[usable])}
    for backbone, metrics in scores.items():
        result[backbone] = {
            name: {"mean": round(float(v.mean()), 3), "ci_95": bootstrap_interval(v)}
            for name, v in metrics.items()
        }
    result["dinov2_minus_clip"] = {
        name: {
            "mean": round(float((scores["dinov2"][name] - scores["clip"][name]).mean()), 3),
            "ci_95": bootstrap_interval(scores["dinov2"][name] - scores["clip"][name]),
        }
        for name in scores["dinov2"]
    }
    return result


def per_type_precision(items: pd.DataFrame, vectors: np.ndarray, k: int = 5) -> pd.DataFrame:
    labels = items["articleType"].to_numpy(dtype=object)
    hits = hit_matrix(vectors, labels, k)
    frame = pd.DataFrame({"type": labels, "precision": hits.mean(axis=1)})
    summary = frame.groupby("type")["precision"].agg(["mean", "size"])
    return summary[summary["size"] >= MIN_TYPE_SIZE].sort_values("mean").round(3)


def common_confusions(items: pd.DataFrame, vectors: np.ndarray, k: int = 5, top: int = 10) -> list[dict]:
    labels = items["articleType"].to_numpy(dtype=object)
    retrieved = labels[nearest(vectors, k)]
    pairs = pd.DataFrame(
        {"query": np.repeat(labels, k), "retrieved": retrieved.flatten()}
    ).query("query != retrieved")
    counts = pairs.value_counts().head(top)
    return [{"query": q, "retrieved": r, "count": int(n)} for (q, r), n in counts.items()]


def main() -> None:
    dev_items, dev_vectors = load_development()
    test_items, test_vectors = load_test()
    types = per_type_precision(dev_items, dev_vectors["dinov2"])
    report = {
        "note": "A retrieved item is correct when its catalogue label matches the query's.",
        "development": {label: evaluate(dev_items, dev_vectors, label) for label in LABELS},
        "unseen_sample": {label: evaluate(test_items, test_vectors, label) for label in LABELS},
        "dinov2_precision_at_5_worst_types": types.head(8).reset_index().to_dict("records"),
        "dinov2_precision_at_5_best_types": types.tail(5).reset_index().to_dict("records"),
        "dinov2_top_confusions": common_confusions(dev_items, dev_vectors["dinov2"]),
    }
    (REPORT_DIR / "similarity_eval.json").write_text(json.dumps(report, indent=2) + "\n")

    for split in ("development", "unseen_sample"):
        for label in LABELS:
            block = report[split][label]
            print(f"\n{split} / {label} / {block['queries']} queries")
            for metric in block["random"]:
                diff = block["dinov2_minus_clip"][metric]
                print(f"  {metric:13} dinov2 {block['dinov2'][metric]['mean']:.3f}  "
                      f"clip {block['clip'][metric]['mean']:.3f}  random {block['random'][metric]:.3f}  "
                      f"diff {diff['mean']:+.3f} {diff['ci_95']}")
    print("\nworst", report["dinov2_precision_at_5_worst_types"])
    print("best", report["dinov2_precision_at_5_best_types"])
    print("confusions", report["dinov2_top_confusions"])


if __name__ == "__main__":
    main()
