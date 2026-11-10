"""Draw the README results figure from the saved evaluation reports.

Usage: python scripts/plot_results.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashion_analyst.config import FIGURE_DIR, REPORT_DIR
from fashion_analyst.plots import plot_results


def load(name: str) -> dict:
    return json.loads((REPORT_DIR / name).read_text())


def main() -> None:
    methods = load("colour_method_comparison.json")["test"]
    training = load("colour_training.json")["unseen_accuracy_by_training_size"]
    similar = load("similarity_eval.json")["unseen_sample"]["articleType"]

    colour = {
        "title": "Colour family correct",
        "bars": [
            (f"Classifier, {training[-1]['training_products']:,} training products",
             training[-1]["accuracy"], True),
            (f"Classifier, {training[0]['training_products']:,} training products",
             training[0]["accuracy"], False),
            ("CLIP zero-shot", methods["clip_zero_shot"]["accuracy"], False),
            ("OpenCV pixel colours",methods["opencv"]["accuracy"], False),
            ("Always the most common", methods["always_most_common (from development)"]["accuracy"], False),
        ],
    }
    find_similar = {
        "title": "\"Find similar\": top match is the same type",
        "bars": [
            ("DINOv2", similar["dinov2"]["recall@1"]["mean"], True),
            ("CLIP", similar["clip"]["recall@1"]["mean"], False),
            ("Random", similar["random"]["recall@1"], False),
        ],
    }
    plot_results([colour, find_similar], FIGURE_DIR / "results.png")


if __name__ == "__main__":
    main()
