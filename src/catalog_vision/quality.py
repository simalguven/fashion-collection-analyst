"""Basic data quality checks on catalogue metadata and on sampled images."""

import hashlib

import cv2
import numpy as np
import pandas as pd

from .config import IMAGE_DIR

LABEL_COLUMNS = ["articleType", "baseColour", "masterCategory", "season", "usage", "year"]
BLANK_STD_THRESHOLD = 3.0
NEAR_DUPLICATE_BITS = 0  # dHash distance: 0 means visually indistinguishable at 9x8.


def check_metadata(meta: pd.DataFrame) -> dict:
    counts = meta["articleType"].value_counts()
    year = meta["year"].dropna()
    return {
        "rows": int(len(meta)),
        "missing_by_column": {c: int(meta[c].isna().sum()) for c in LABEL_COLUMNS},
        "duplicate_ids": int(meta["id"].duplicated().sum()),
        "duplicate_display_names": int(meta["productDisplayName"].duplicated().sum()),
        "article_types": int(counts.size),
        "article_types_under_10_items": int((counts < 10).sum()),
        "top_5_article_type_share": round(float(counts.head(5).sum() / len(meta)), 3),
        "year_min": int(year.min()),
        "year_max": int(year.max()),
        "share_from_2011_2013": round(float(year.between(2011, 2013).mean()), 3),
        "master_category_share": meta["masterCategory"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
    }


def _dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def check_images(sample: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Inspect each image; return a summary and the sample with issue flags."""
    records = []
    for row in sample.itertuples():
        record = {"id": row.id, "unreadable": True}
        image = cv2.imread(str(IMAGE_DIR / row.path)) if row.decodable else None
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            record.update(
                unreadable=False,
                width=image.shape[1],
                height=image.shape[0],
                blank=float(gray.std()) < BLANK_STD_THRESHOLD,
                pixel_md5=hashlib.md5(image.tobytes()).hexdigest(),
                dhash=_dhash(gray),
            )
        records.append(record)

    flags = pd.DataFrame(records)
    flags["exact_duplicate"] = flags["pixel_md5"].notna() & flags["pixel_md5"].duplicated()
    flags["near_duplicate"] = flags["dhash"].notna() & flags["dhash"].duplicated()
    flags["blank"] = flags["blank"].fillna(False).astype(bool)
    flags["drop"] = flags["unreadable"] | flags["blank"] | flags["exact_duplicate"]

    sizes = flags.dropna(subset=["width"]).groupby(["width", "height"]).size()
    summary = {
        "sampled": int(len(flags)),
        "unreadable": int(flags["unreadable"].sum()),
        "blank": int(flags["blank"].sum()),
        "exact_duplicates": int(flags["exact_duplicate"].sum()),
        "near_duplicates": int(flags["near_duplicate"].sum()),
        "dropped": int(flags["drop"].sum()),
        "most_common_size": "x".join(str(int(v)) for v in sizes.idxmax()),
        "share_most_common_size": round(float(sizes.max() / sizes.sum()), 3),
    }
    return summary, flags
