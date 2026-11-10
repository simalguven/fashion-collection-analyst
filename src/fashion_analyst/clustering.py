"""Style clustering and a label-based comparison of embedding backbones."""

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .config import K_RANGE, SEED, SILHOUETTE_TOLERANCE


MIXED_CLUSTER_THRESHOLD = 0.3


def fit_kmeans(embeddings: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(embeddings)


def scan_k(embeddings: np.ndarray, labels: pd.Series, k_values=K_RANGE) -> pd.DataFrame:
    """Score each k by silhouette (unsupervised) and by agreement with article types."""
    rows = []
    for k in k_values:
        assignment = fit_kmeans(embeddings, k)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(embeddings, assignment, metric="cosine"),
                "nmi_article_type": normalized_mutual_info_score(labels, assignment),
                "ari_article_type": adjusted_rand_score(labels, assignment),
            }
        )
    return pd.DataFrame(rows).round(4)


def choose_k(scan: pd.DataFrame) -> int:
    """Pick the smallest k whose silhouette is close to the best one.

    Silhouette flattens out at higher k, so preferring the smallest near-best
    k keeps the number of clusters small enough for a person to review.
    """
    threshold = scan["silhouette"].max() * (1 - SILHOUETTE_TOLERANCE)
    return int(scan.loc[scan["silhouette"] >= threshold, "k"].min())


def project_2d(embeddings: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, metric="cosine", random_state=SEED)
    return reducer.fit_transform(embeddings)


def name_clusters(items: pd.DataFrame) -> dict[int, str]:
    """Name each cluster after the article types that dominate it."""
    names = {}
    for cluster, group in items.groupby("cluster"):
        top = group["articleType"].value_counts(normalize=True)
        if top.iloc[0] < MIXED_CLUSTER_THRESHOLD:
            names[int(cluster)] = "Mixed: " + ", ".join(top.index[:3])
            continue
        parts = [top.index[0]]
        if len(top) > 1 and top.iloc[1] >= 0.2:
            parts.append(top.index[1])
        names[int(cluster)] = " & ".join(parts)
    return _disambiguate(names, items)


def _disambiguate(names: dict[int, str], items: pd.DataFrame) -> dict[int, str]:
    """Give clusters that share a name a suffix from what most sets them apart."""
    counts = pd.Series(names).value_counts()
    result = dict(names)
    for name in counts[counts > 1].index:
        members = [c for c, n in names.items() if n == name]
        for column in ("usage", "gender", "subCategory"):
            suffixes = {c: items.loc[items["cluster"] == c, column].mode().iloc[0] for c in members}
            if len(set(suffixes.values())) == len(members):
                result.update({c: f"{name} ({suffixes[c]})" for c in members})
                break
        else:
            result.update({c: f"{name} (group {i + 1})" for i, c in enumerate(members)})
    return result
