"""Compare colour methods on development data, then once on an unseen test set.

Every choice (prompt wording, classifier settings, overall winner) uses only the
development products. Run without flags while developing, and with --final once.

Usage: python scripts/compare_colour_methods.py [--final]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from fashion_analyst import colour_classifier, colours, data, embeddings, quality
from fashion_analyst.config import ARTIFACT_DIR, BACKBONES, IMAGE_DIR, REPORT_DIR

TEST_START = 2000
TEST_SIZE = 1000
CV_FOLDS = 5
C_VALUES = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 1.0)
TEMPLATES = {
    "generic": "a photo of a {colour} fashion product",
    "article_type": "a photo of a {colour} {article}",
}
FAMILIES = sorted(set(colours.COLOUR_FAMILIES.values()))
SINGLE_WORDS = {family: [family.lower()] for family in FAMILIES} | {"Neutral": ["beige"]}
ALL_SHADES = defaultdict(list)
for shade, shade_family in colours.COLOUR_FAMILIES.items():
    ALL_SHADES[shade_family].append(shade.lower())
WORD_LISTS = {"family_name": SINGLE_WORDS, "all_shades": dict(ALL_SHADES)}


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for an accuracy measured on `total` products."""
    share = correct / total
    centre = (share + z**2 / (2 * total)) / (1 + z**2 / total)
    margin = z * np.sqrt(share * (1 - share) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    return round(float(centre - margin), 3), round(float(centre + margin), 3)


def score(truth: np.ndarray, predicted: np.ndarray) -> dict:
    correct = int((truth == predicted).sum())
    low, high = wilson_interval(correct, len(truth))
    return {
        "accuracy": round(correct / len(truth), 3),
        "ci_95": [low, high],
        "macro_f1": round(float(f1_score(truth, predicted, labels=FAMILIES, average="macro",
                                         zero_division=0)), 3),
        "products": int(len(truth)),
    }


def load_development() -> tuple[pd.DataFrame, dict]:
    items = pd.read_parquet(ARTIFACT_DIR / "items.parquet")
    vectors = {b: np.load(ARTIFACT_DIR / f"embeddings_{b}.npy") for b in BACKBONES}
    return select_scored(items, vectors)


def select_scored(items: pd.DataFrame, vectors: dict) -> tuple[pd.DataFrame, dict]:
    """Keep products whose catalogue colour maps to a family."""
    items = items.copy()
    items["family"] = items["baseColour"].map(colours.family)
    keep = items["family"].notna().to_numpy()
    return items[keep].reset_index(drop=True), {b: v[keep] for b, v in vectors.items()}


def prepare_test_set(dev_ids: set[int]) -> tuple[pd.DataFrame, dict]:
    """Build the unseen sample from a later slice of the same shuffle."""
    catalogue = data.load_catalogue()
    raw = data.materialise_sample(catalogue, size=TEST_SIZE, start=TEST_START, raw_name="test_raw.parquet")
    _, flags = quality.check_images(raw)
    items = raw[~flags["drop"].to_numpy()].dropna(subset=["articleType"]).reset_index(drop=True)
    overlap = set(items["id"].astype(int)) & dev_ids
    if overlap:
        raise RuntimeError(f"{len(overlap)} test products also appear in development data")

    records = [
        colours.item_dominant_colour(colours.load_rgb(IMAGE_DIR / path), is_apparel=department == "Apparel")
        for path, department in zip(items["path"], items["masterCategory"])
    ]
    items = pd.concat([items, pd.DataFrame(records)], axis=1)
    paths = [IMAGE_DIR / p for p in items["path"]]
    vectors = {b: embeddings.embed_images(paths, b) for b in BACKBONES}
    return select_scored(items, vectors)


class ZeroShot:
    """Classify colour by comparing CLIP image vectors with colour descriptions."""

    def __init__(self) -> None:
        self.encoder = embeddings.ClipTextEncoder()
        self.cache = {}

    def _text(self, text: str) -> np.ndarray:
        if text not in self.cache:
            self.cache[text] = self.encoder.encode(text)
        return self.cache[text]

    def _family_matrix(self, template: str, words: dict, article: str) -> np.ndarray:
        rows = []
        for family in FAMILIES:
            vectors = [self._text(template.format(colour=w, article=article)) for w in words[family]]
            mean = np.mean(vectors, axis=0)
            rows.append(mean / np.linalg.norm(mean))
        return np.stack(rows)

    def predict(self, items: pd.DataFrame, clip: np.ndarray, template: str, words: dict) -> np.ndarray:
        predicted = np.empty(len(items), dtype=object)
        for article, index in items.groupby("articleType").indices.items():
            matrix = self._family_matrix(template, words, article.lower())
            predicted[index] = np.array(FAMILIES, dtype=object)[np.argmax(clip[index] @ matrix.T, axis=1)]
        return predicted


def feature_sets(vectors: dict) -> dict:
    return {**vectors, "clip+dinov2": np.hstack([vectors["clip"], vectors["dinov2"]])}


def make_classifier(c: float):
    return colour_classifier.make_classifier(c)


def tune_zero_shot(zero_shot: ZeroShot, items: pd.DataFrame, clip: np.ndarray) -> tuple[dict, dict]:
    truth = items["family"].to_numpy()
    results = {}
    for template_name, template in TEMPLATES.items():
        for words_name, words in WORD_LISTS.items():
            key = f"{template_name}/{words_name}"
            results[key] = score(truth, zero_shot.predict(items, clip, template, words))
            print(f"  zero-shot {key}: {results[key]['accuracy']}", flush=True)
    best = max(results, key=lambda k: results[k]["accuracy"])
    template_name, words_name = best.split("/")
    return {"key": best, "template": TEMPLATES[template_name], "words": WORD_LISTS[words_name]}, results


def tune_classifier(items: pd.DataFrame, vectors: dict) -> tuple[dict, dict]:
    truth = items["family"].to_numpy()
    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=0)
    results = {}
    for name, matrix in feature_sets(vectors).items():
        for c in C_VALUES:
            predicted = cross_val_predict(make_classifier(c), matrix, truth, cv=folds)
            results[f"{name}/C={c}"] = score(truth, predicted)
            print(f"  classifier {name} C={c}: {results[f'{name}/C={c}']['accuracy']}", flush=True)
    best = max(results, key=lambda k: results[k]["accuracy"])
    name, c = best.split("/C=")
    return {"key": best, "features": name, "c": float(c)}, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="score the unseen test set")
    args = parser.parse_args()

    dev, dev_vectors = load_development()
    truth = dev["family"].to_numpy()
    print(f"Development products: {len(dev)}", flush=True)

    opencv_dev = dev["colour_name"].map(colours.family).to_numpy()
    zero_shot = ZeroShot()
    best_zero, zero_results = tune_zero_shot(zero_shot, dev, dev_vectors["clip"])
    best_classifier, classifier_results = tune_classifier(dev, dev_vectors)

    dev_summary = {
        "opencv": score(truth, opencv_dev),
        "always_most_common": score(truth, np.full(len(truth), Counter(truth).most_common(1)[0][0], dtype=object)),
        f"clip_zero_shot ({best_zero['key']})": zero_results[best_zero["key"]],
        f"classifier ({best_classifier['key']})": classifier_results[best_classifier["key"]],
    }
    report = {
        "development_note": "OpenCV was tuned on this data, and classifier scores are 5-fold cross-validated.",
        "development": dev_summary,
        "zero_shot_variants": zero_results,
        "classifier_variants": classifier_results,
    }
    print(json.dumps(dev_summary, indent=2), flush=True)

    if args.final:
        test, test_vectors = prepare_test_set(set(dev["id"].astype(int)))
        test_truth = test["family"].to_numpy()
        model = make_classifier(best_classifier["c"]).fit(feature_sets(dev_vectors)[best_classifier["features"]], truth)
        predictions = {
            "opencv": test["colour_name"].map(colours.family).to_numpy(),
            "always_most_common (from development)": np.full(
                len(test_truth), Counter(truth).most_common(1)[0][0], dtype=object),
            "clip_zero_shot": zero_shot.predict(test, test_vectors["clip"], best_zero["template"], best_zero["words"]),
            "classifier": model.predict(feature_sets(test_vectors)[best_classifier["features"]]),
        }
        report["test"] = {name: score(test_truth, p) for name, p in predictions.items()}
        report["test_by_department"] = {
            name: {
                department: round(float((p[index] == test_truth[index]).mean()), 3)
                for department, index in test.groupby("masterCategory").indices.items()
                if len(index) >= 30
            }
            for name, p in predictions.items()
        }
        confusions = Counter(zip(test_truth, predictions["classifier"]))
        report["classifier_top_confusions"] = [
            {"label": t, "predicted": p, "count": n} for (t, p), n in confusions.most_common(30) if t != p
        ][:10]
        print(json.dumps({k: report[k] for k in ("test", "test_by_department")}, indent=2), flush=True)

    (REPORT_DIR / "colour_method_comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
