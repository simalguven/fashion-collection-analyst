"""Build every artifact the notebook, the agent and the evals depend on.

Usage: python scripts/run_pipeline.py [--sample-size 2000]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fashion_analyst import clustering, colour_classifier, colours, data, embeddings, plots, quality
from fashion_analyst.config import (
    ARTIFACT_DIR,
    BACKBONES,
    CLUSTER_BACKBONE,
    FIGURE_DIR,
    IMAGE_DIR,
    REPORT_DIR,
    SAMPLE_SIZE,
    ensure_dirs,
)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def run_quality(sample_size: int) -> pd.DataFrame:
    catalogue = data.load_catalogue()
    metadata_report = quality.check_metadata(data.catalogue_metadata(catalogue))
    sample = data.materialise_sample(catalogue, size=sample_size)
    image_report, flags = quality.check_images(sample)
    write_json(REPORT_DIR / "data_quality.json", {"metadata": metadata_report, "images": image_report})

    clean = sample[~flags["drop"].to_numpy()].dropna(subset=["articleType"]).reset_index(drop=True)
    print(f"Quality: kept {len(clean)} of {len(sample)} sampled items")
    return clean


def compare_backbones(items: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    paths = [IMAGE_DIR / p for p in items["path"]]
    vectors, scans = {}, []
    for backbone in BACKBONES:
        print(f"Embedding with {backbone}")
        vectors[backbone] = embeddings.embed_images(paths, backbone)
        scan = clustering.scan_k(vectors[backbone], items["articleType"])
        scan.insert(0, "backbone", backbone)
        scans.append(scan)
    return vectors, pd.concat(scans, ignore_index=True)


def score_colour_extraction(items: pd.DataFrame, whole_image_names: list[str]) -> dict:
    """Compare predicted colour families with labels and two simple baselines.

    Only odd rows are scored: even rows were used to tune the OpenCV method.
    Classifier predictions are out-of-fold, but its settings were chosen on this
    sample; the unseen-product score is in reports/colour_method_comparison.json.
    """
    truth = items["baseColour"].map(colours.family)
    scored = truth.notna() & (items.index.to_series() % 2 == 1)
    masked = items["colour_name"].map(colours.family)[scored]
    classifier = items["colour_family"][scored]
    whole = pd.Series(whole_image_names, index=items.index).map(colours.family)[scored]
    truth = truth[scored]

    def by_department(predicted: pd.Series) -> dict:
        return {
            department: round(float((predicted[index] == truth[index]).mean()), 3)
            for department, index in truth.groupby(items.loc[scored, "masterCategory"]).groups.items()
            if len(index) >= 30
        }

    return {
        "items_scored": int(scored.sum()),
        "items_with_unmapped_label": int(items["baseColour"].map(colours.family).isna().sum()),
        "accuracy_classifier_out_of_fold": round(float((classifier == truth).mean()), 3),
        "accuracy_with_background_removal": round(float((masked == truth).mean()), 3),
        "accuracy_whole_image": round(float((whole == truth).mean()), 3),
        "accuracy_always_majority_family": round(float(truth.value_counts(normalize=True).iloc[0]), 3),
        "classifier_accuracy_by_department": by_department(classifier),
        "opencv_accuracy_by_department": by_department(masked),
    }


def classify_colours(items: pd.DataFrame, vectors: dict) -> pd.DataFrame:
    """Predict colour families, adding the extra training products when they exist."""
    labels = items["baseColour"].map(colours.family).to_numpy(dtype=object)
    extra = {}
    settings = ARTIFACT_DIR / "colour_classifier.json"
    if settings.exists():
        extra = {"extra_matrix": np.load(ARTIFACT_DIR / "colour_extra_features.npy"),
                 "extra_labels": np.load(ARTIFACT_DIR / "colour_extra_labels.npy"),
                 "c": json.loads(settings.read_text())["c"]}
        print(f"Colour classifier: adding {len(extra['extra_labels'])} extra training products")
    else:
        print("Colour classifier: sample only (run scripts/train_colour_classifier.py for more)")
    families, confidence = colour_classifier.predict_families(
        colour_classifier.features(vectors), labels, **extra)
    return items.assign(colour_family=families, colour_confidence=confidence)


def extract_colours(items: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    records, whole_names = [], []
    for path, department in zip(items["path"], items["masterCategory"]):
        image = colours.load_rgb(IMAGE_DIR / path)
        records.append(colours.item_dominant_colour(image, is_apparel=department == "Apparel"))
        centres, _ = colours._kmeans_lab(colours._rgb_to_lab(image), 3)
        whole_names.append(colours.nearest_colour_name(centres[0]))
    return pd.concat([items, pd.DataFrame(records)], axis=1), whole_names


def summarise_clusters(items: pd.DataFrame) -> list[dict]:
    names = clustering.name_clusters(items)
    summaries = []
    for cluster, group in items.groupby("cluster"):
        def top(column: str, n: int = 3) -> dict:
            return group[column].value_counts(normalize=True).head(n).round(3).to_dict()

        summaries.append(
            {
                "cluster_id": int(cluster),
                "name": names[int(cluster)],
                "size": int(len(group)),
                "top_article_types": top("articleType"),
                "gender": top("gender"),
                "usage": top("usage"),
                "season": top("season"),
                "palette": colours.build_palette(group["colour_family"], group["colour_name"],
                                                 group["colour_hex"]),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()
    ensure_dirs()

    items = run_quality(args.sample_size)
    vectors, scan = compare_backbones(items)
    scan.to_csv(REPORT_DIR / "backbone_comparison.csv", index=False)

    chosen = vectors[CLUSTER_BACKBONE]
    k = clustering.choose_k(scan[scan["backbone"] == CLUSTER_BACKBONE])
    print(f"Clustering with {CLUSTER_BACKBONE}, k={k}")
    items["cluster"] = clustering.fit_kmeans(chosen, k)
    projection = clustering.project_2d(chosen)
    items["umap_x"], items["umap_y"] = projection[:, 0], projection[:, 1]

    items, whole_names = extract_colours(items)
    items = classify_colours(items, vectors)
    clusters = summarise_clusters(items)
    colour_scores = score_colour_extraction(items, whole_names)

    items.to_parquet(ARTIFACT_DIR / "items.parquet")
    for backbone, matrix in vectors.items():
        np.save(ARTIFACT_DIR / f"embeddings_{backbone}.npy", matrix)
    write_json(ARTIFACT_DIR / "clusters.json", clusters)
    write_json(REPORT_DIR / "pipeline_metrics.json",
               {"backbone": CLUSTER_BACKBONE, "k": k, "items": int(len(items)),
                "colour_extraction": colour_scores})

    plots.plot_style_map(items, FIGURE_DIR / "style_map.png")
    plots.plot_cluster_gallery(clusters, items, chosen, FIGURE_DIR / "cluster_gallery.png")
    print(json.dumps(colour_scores, indent=2))


if __name__ == "__main__":
    main()
