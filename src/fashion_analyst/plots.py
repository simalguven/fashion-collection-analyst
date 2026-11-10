"""Figures for the README and the notebook."""

import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from .config import IMAGE_DIR

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
CATEGORY_COLOURS = {"Apparel": "#2a78d6", "Accessories": "#eb6834", "Footwear": "#1baf7a"}
OTHER_COLOUR = "#b5b3ad"
STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT,
    "font.size": 10,
}


def plot_style_map(items: pd.DataFrame, path) -> None:
    """2D map of the collection, coloured by department, with cluster numbers."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 6.5))
        other = ~items["masterCategory"].isin(CATEGORY_COLOURS)
        ax.scatter(items.loc[other, "umap_x"], items.loc[other, "umap_y"], s=9,
                   c=OTHER_COLOUR, label="Other", linewidths=0)
        for category, colour in CATEGORY_COLOURS.items():
            subset = items[items["masterCategory"] == category]
            ax.scatter(subset["umap_x"], subset["umap_y"], s=9, c=colour,
                       label=category, linewidths=0)
        for cluster, group in items.groupby("cluster"):
            ax.annotate(str(cluster), (group["umap_x"].median(), group["umap_y"].median()),
                        ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT,
                        bbox={"boxstyle": "circle,pad=0.25", "fc": SURFACE, "ec": MUTED, "lw": 0.6})
        ax.set_title("The collection groups into distinct style families", loc="left",
                     fontsize=14, fontweight="bold", pad=24)
        ax.text(0, 1.015, "Each dot is a product, closer dots look more alike. "
                        "Numbers mark the style clusters.",
                transform=ax.transAxes, color=MUTED, va="bottom")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4,
                  markerscale=2)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_cluster_gallery(clusters: list[dict], items: pd.DataFrame,
                         embeddings: np.ndarray, path, examples: int = 6) -> None:
    """One row per cluster: name, the most typical products and the colour palette."""
    with plt.rc_context(STYLE):
        rows = len(clusters)
        height = 1.15 * rows + 0.5
        fig = plt.figure(figsize=(11, height))
        grid = fig.add_gridspec(rows, examples + 2, top=1 - 0.5 / height, bottom=0.01,
                                width_ratios=[2.6] + [0.62] * examples + [3], wspace=0.06)
        for row, cluster in enumerate(clusters):
            member_index = np.flatnonzero(items["cluster"].to_numpy() == cluster["cluster_id"])
            centre = embeddings[member_index].mean(axis=0)
            closest = member_index[np.argsort(-embeddings[member_index] @ centre)[:examples]]

            label = fig.add_subplot(grid[row, 0])
            label.axis("off")
            title = textwrap.fill(f"{cluster['cluster_id']}. {cluster['name']}", 26)
            label.text(0, 0.62, title, fontweight="bold", va="center")
            label.text(0, 0.12, f"{cluster['size']} items", color=MUTED, va="center")

            for column, position in enumerate(closest):
                ax = fig.add_subplot(grid[row, column + 1])
                ax.imshow(Image.open(IMAGE_DIR / items["path"].iloc[position]))
                ax.axis("off")

            bar = fig.add_subplot(grid[row, examples + 1])
            left = 0.0
            for swatch in cluster["palette"]:
                bar.add_patch(Rectangle((left, 0.3), swatch["share"] - 0.008, 0.4,
                                        fc=swatch["hex"], ec="none"))
                if swatch["share"] >= 0.14:
                    bar.text(left, 0.2, swatch["name"], fontsize=8, color=MUTED, va="top")
                left += swatch["share"]
            bar.set_xlim(0, 1)
            bar.set_ylim(0, 1)
            bar.axis("off")
        fig.suptitle("Style clusters, typical products and dominant colours",
                     x=0.125, y=0.998, ha="left", fontsize=14, fontweight="bold")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
