"""Download the catalogue and materialise a reproducible image sample on disk."""

import pandas as pd
from datasets import load_dataset

from .config import ARTIFACT_DIR, DATASET, IMAGE_DIR, SAMPLE_SIZE, SEED


def load_catalogue():
    return load_dataset(DATASET, split="train")


def catalogue_metadata(dataset) -> pd.DataFrame:
    """Metadata for the full catalogue, used for the data quality report."""
    return dataset.remove_columns("image").to_pandas()


def materialise_sample(dataset, size: int = SAMPLE_SIZE, seed: int = SEED) -> pd.DataFrame:
    """Write a random sample of images to disk and return its metadata.

    Images that cannot be decoded are skipped and recorded with
    ``decodable=False`` so that the quality report can count them.
    """
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    subset = dataset.shuffle(seed=seed).select(range(size))
    rows = []
    for index in range(size):
        try:
            row = subset[index]
            image = row.pop("image").convert("RGB")
            path = IMAGE_DIR / f"{row['id']}.jpg"
            image.save(path, quality=95)
            row["path"] = str(path.relative_to(IMAGE_DIR))
            row["decodable"] = True
        except (OSError, ValueError):
            row = {"id": None, "path": None, "decodable": False}
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(ARTIFACT_DIR / "sample_raw.parquet")
    return frame
