"""Image and text embeddings from Hugging Face vision models."""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

from .config import BACKBONES


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _as_tensor(output) -> torch.Tensor:
    """Newer transformers versions wrap CLIP feature outputs in a dataclass."""
    return output if isinstance(output, torch.Tensor) else output.pooler_output


def _normalise(features: torch.Tensor) -> np.ndarray:
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()


@torch.inference_mode()
def embed_images(paths: list, backbone: str, batch_size: int = 32) -> np.ndarray:
    """Return L2-normalised embeddings, one row per image path."""
    device = get_device()
    name = BACKBONES[backbone]
    if backbone == "clip":
        model = CLIPModel.from_pretrained(name).to(device).eval()
        processor = CLIPProcessor.from_pretrained(name)
    else:
        model = AutoModel.from_pretrained(name).to(device).eval()
        processor = AutoImageProcessor.from_pretrained(name)

    batches = []
    for start in range(0, len(paths), batch_size):
        images = [Image.open(p).convert("RGB") for p in paths[start : start + batch_size]]
        inputs = processor(images=images, return_tensors="pt").to(device)
        if backbone == "clip":
            features = _as_tensor(model.get_image_features(**inputs))
        else:
            features = model(**inputs).last_hidden_state[:, 0]  # CLS token
        batches.append(_normalise(features))
    return np.concatenate(batches)


class ClipTextEncoder:
    """Encodes free-text queries into the same space as the CLIP image embeddings."""

    def __init__(self) -> None:
        name = BACKBONES["clip"]
        self.model = CLIPModel.from_pretrained(name).eval()
        self.processor = CLIPProcessor.from_pretrained(name)

    @torch.inference_mode()
    def encode(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        return _normalise(_as_tensor(self.model.get_text_features(**inputs)))[0]
