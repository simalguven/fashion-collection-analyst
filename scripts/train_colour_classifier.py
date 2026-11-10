"""Train the colour classifier on more labelled products, then score it on unseen ones.

Training uses the pipeline sample plus a further slice of the catalogue. The
regularisation strength is chosen by cross-validation on training products
only. The unseen products from compare_colour_methods.py are scored at several
training sizes with that fixed setting. Training images that are exact or near
duplicates of an unseen image are removed first.

Run after scripts/run_pipeline.py, then run the pipeline again to use the result.

Usage: python scripts/train_colour_classifier.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from compare_colour_methods import TEST_SIZE, TEST_START, score
from catalog_vision import colour_classifier, colours, data, embeddings, quality
from catalog_vision.config import ARTIFACT_DIR, BACKBONES, IMAGE_DIR, REPORT_DIR

EXTRA_START = TEST_START + TEST_SIZE
EXTRA_SIZE = 6000
C_VALUES = (0.003, 0.01, 0.03, 0.1)
EXTRA_STEPS = (0, 2000, 4000, EXTRA_SIZE)


def filtered(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the pipeline's quality filter and keep the image fingerprints."""
    _, flags = quality.check_images(raw)
    keep = ~flags["drop"].to_numpy() & raw["articleType"].notna().to_numpy()
    items = raw[keep].reset_index(drop=True)
    return items, flags[keep].reset_index(drop=True)[["pixel_md5", "dhash"]]


def cached_vectors(items: pd.DataFrame, prefix: str) -> dict:
    vectors = {}
    for backbone in BACKBONES:
        path = ARTIFACT_DIR / f"{prefix}_embeddings_{backbone}.npy"
        if not path.exists() or len(np.load(path)) != len(items):
            print(f"Embedding {len(items)} images with {backbone}", flush=True)
            np.save(path, embeddings.embed_images([IMAGE_DIR / p for p in items["path"]], backbone))
        vectors[backbone] = np.load(path)
    return vectors


def load_split(raw_name: str, prefix: str, **sample) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    raw_path = ARTIFACT_DIR / raw_name
    if raw_path.exists():
        raw = pd.read_parquet(raw_path)
    else:
        print(f"Downloading {sample['size']} products", flush=True)
        raw = data.materialise_sample(data.load_catalogue(), raw_name=raw_name, **sample)
    items, prints = filtered(raw)
    return items, prints, colour_classifier.features(cached_vectors(items, prefix))


def main() -> None:
    test, test_prints, test_x = load_split("test_raw.parquet", "test", size=TEST_SIZE, start=TEST_START)

    # The pipeline sample: its embeddings already exist, and its rows follow sample_raw.
    sample_items = pd.read_parquet(ARTIFACT_DIR / "items.parquet")
    _, sample_flags = quality.check_images(pd.read_parquet(ARTIFACT_DIR / "sample_raw.parquet"))
    sample_prints = sample_flags.set_index("id").loc[sample_items["id"], ["pixel_md5", "dhash"]]
    sample_x = colour_classifier.features(
        {b: np.load(ARTIFACT_DIR / f"embeddings_{b}.npy") for b in BACKBONES})

    extra, extra_prints, extra_x = load_split(
        "colour_train_raw.parquet", "colour_train", size=EXTRA_SIZE, start=EXTRA_START)

    seen_ids = set(sample_items["id"].astype(int)) | set(test["id"].astype(int))
    if seen_ids & set(extra["id"].astype(int)):
        raise RuntimeError("Training slice overlaps the sample or the unseen products")

    test_md5, test_dhash = set(test_prints["pixel_md5"]), set(test_prints["dhash"])

    def usable(items: pd.DataFrame, prints: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
        family = items["baseColour"].map(colours.family).to_numpy(dtype=object)
        duplicate = prints["pixel_md5"].isin(test_md5).to_numpy() | prints["dhash"].isin(test_dhash).to_numpy()
        keep = pd.notna(family) & ~duplicate
        return keep, family, int((pd.notna(family) & duplicate).sum())

    sample_keep, sample_family, sample_dupes = usable(sample_items, sample_prints)
    extra_keep, extra_family, extra_dupes = usable(extra, extra_prints)
    test_family = test["baseColour"].map(colours.family).to_numpy(dtype=object)
    test_scored = pd.notna(test_family)

    train_x = np.vstack([sample_x[sample_keep], extra_x[extra_keep]])
    train_y = np.concatenate([sample_family[sample_keep], extra_family[extra_keep]]).astype(str)
    n_sample = int(sample_keep.sum())
    print(f"Training products: {n_sample} from the sample + {int(extra_keep.sum())} extra. "
          f"Removed as duplicates of unseen images: {sample_dupes + extra_dupes}", flush=True)

    folds = StratifiedKFold(5, shuffle=True, random_state=colour_classifier.SEED)
    tuning = {}
    for c in C_VALUES:
        tuning[c] = round(float(cross_val_score(colour_classifier.make_classifier(c), train_x, train_y,
                                                cv=folds, n_jobs=-1).mean()), 3)
        print(f"  C={c}: cross-validated accuracy {tuning[c]}", flush=True)
    best_c = max(tuning, key=tuning.get)

    def unseen_score(n_extra: int) -> tuple[dict, np.ndarray]:
        rows = n_sample + min(n_extra, len(train_y) - n_sample)
        model = colour_classifier.make_classifier(best_c).fit(train_x[:rows], train_y[:rows])
        predicted = model.predict(test_x[test_scored])
        return {"training_products": rows, **score(test_family[test_scored], predicted)}, predicted

    curve = []
    for n_extra in EXTRA_STEPS:
        result, predicted = unseen_score(n_extra)
        curve.append(result)
        print(f"  {result['training_products']} training products: {result['accuracy']} on unseen", flush=True)

    truth = test_family[test_scored]
    departments = test.loc[test_scored, "masterCategory"].to_numpy()
    report = {
        "note": "C chosen by 5-fold cross-validation on training products only; unseen products "
                "scored with that fixed C at each training size.",
        "training_products": int(len(train_y)),
        "removed_as_duplicates_of_unseen": sample_dupes + extra_dupes,
        "cross_validated_accuracy_by_c": {str(c): a for c, a in tuning.items()},
        "chosen_c": best_c,
        "unseen_accuracy_by_training_size": curve,
        "unseen_accuracy_by_department": {
            d: round(float((predicted[departments == d] == truth[departments == d]).mean()), 3)
            for d in np.unique(departments) if (departments == d).sum() >= 30
        },
        "top_confusions": [
            {"label": t, "predicted": p, "count": n}
            for (t, p), n in Counter(zip(truth, predicted)).most_common(30) if t != p
        ][:10],
    }
    (REPORT_DIR / "colour_training.json").write_text(json.dumps(report, indent=2) + "\n")

    # Everything the pipeline needs to train on the extra products.
    np.save(ARTIFACT_DIR / "colour_extra_features.npy", extra_x[extra_keep])
    np.save(ARTIFACT_DIR / "colour_extra_labels.npy", extra_family[extra_keep].astype(str))
    (ARTIFACT_DIR / "colour_classifier.json").write_text(json.dumps({"c": best_c}) + "\n")
    print(json.dumps({k: report[k] for k in ("chosen_c", "unseen_accuracy_by_training_size")}, indent=2))


if __name__ == "__main__":
    main()
