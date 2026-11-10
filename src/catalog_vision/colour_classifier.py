"""Colour family classifier on image embeddings, trained on catalogue colour labels.

Chosen in scripts/compare_colour_methods.py: CLIP and DINOv2 vectors side by side,
scaled, then logistic regression. It scored 67% on unseen products, against 52%
for the OpenCV method, which is kept for exact shades.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_BACKBONES = ("clip", "dinov2")
REGULARISATION_C = 0.01
CV_FOLDS = 5
SEED = 0


def make_classifier(c: float = REGULARISATION_C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def features(vectors: dict) -> np.ndarray:
    return np.hstack([vectors[backbone] for backbone in FEATURE_BACKBONES])


def _is_labelled(labels: np.ndarray) -> np.ndarray:
    return np.array([label is not None and label == label for label in labels])


def predict_families(
    matrix: np.ndarray,
    labels: np.ndarray,
    extra_matrix: np.ndarray | None = None,
    extra_labels: np.ndarray | None = None,
    c: float = REGULARISATION_C,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a colour family and its probability for every product.

    Labelled products get out-of-fold predictions, so no product is classified by
    a model that saw its own label. Products without a usable label are
    classified by a model trained on all labelled products. Optional extra
    products, outside this sample, are added to every training fold.
    """
    labels = np.asarray(labels, dtype=object)
    labelled = _is_labelled(labels)
    if extra_matrix is None:
        extra_matrix, extra_labels = matrix[:0], np.array([], dtype=str)
    extra_labels = np.asarray(extra_labels).astype(str)

    families = np.empty(len(labels), dtype=object)
    confidence = np.zeros(len(labels))
    rows = np.flatnonzero(labelled)
    y = labels[labelled].astype(str)
    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    for train, held_out in folds.split(rows, y):
        model = make_classifier(c).fit(np.vstack([extra_matrix, matrix[rows[train]]]),
                                       np.concatenate([extra_labels, y[train]]))
        probabilities = model.predict_proba(matrix[rows[held_out]])
        families[rows[held_out]] = model.classes_[probabilities.argmax(axis=1)]
        confidence[rows[held_out]] = probabilities.max(axis=1)

    if (~labelled).any():
        model = make_classifier(c).fit(np.vstack([extra_matrix, matrix[labelled]]),
                                       np.concatenate([extra_labels, y]))
        probabilities = model.predict_proba(matrix[~labelled])
        families[~labelled] = model.classes_[probabilities.argmax(axis=1)]
        confidence[~labelled] = probabilities.max(axis=1)
    return families, confidence.round(3)
