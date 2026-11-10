"""Paths and run settings shared by the pipeline, the agent and the evals."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
ARTIFACT_DIR = DATA_DIR / "artifacts"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

DATASET = "ashraq/fashion-product-images-small"
SAMPLE_SIZE = 2000
SEED = 42

BACKBONES = {
    "clip": "openai/clip-vit-base-patch32",
    "dinov2": "facebook/dinov2-small",
}
# DINOv2 scored higher on clustering; CLIP is kept because it also embeds text.
CLUSTER_BACKBONE = "dinov2"
SILHOUETTE_TOLERANCE = 0.05

K_RANGE = range(6, 21, 2)
DEFAULT_AGENT_MODEL = "groq:qwen/qwen3.8-27b"


def ensure_dirs() -> None:
    for path in (IMAGE_DIR, ARTIFACT_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)
