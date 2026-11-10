"""Dominant colour extraction with OpenCV: background removal plus LAB k-means."""

import cv2
import numpy as np
import pandas as pd

SEED = 42
WHITE_THRESHOLD = 235
MIN_FOREGROUND_SHARE = 0.03
ACHROMATIC_CHROMA = 10.0
LIGHTNESS_WEIGHT = 0.5
MIN_NON_SKIN_SHARE = 0.25
ACHROMATIC_NAMES = ("Black", "Grey", "Silver", "White")

# Reference RGB values for the catalogue's main colour vocabulary.
REFERENCE_COLOURS = {
    "Black": (20, 20, 20),
    "Grey": (128, 128, 128),
    "Silver": (192, 192, 192),
    "White": (245, 245, 245),
    "Beige": (222, 203, 164),
    "Brown": (101, 67, 33),
    "Tan": (170, 125, 80),
    "Red": (200, 30, 40),
    "Maroon": (110, 20, 40),
    "Pink": (235, 130, 170),
    "Orange": (240, 120, 30),
    "Yellow": (245, 210, 50),
    "Gold": (195, 160, 70),
    "Green": (40, 140, 70),
    "Olive": (100, 105, 45),
    "Teal": (20, 125, 130),
    "Blue": (45, 95, 190),
    "Navy Blue": (25, 35, 80),
    "Purple": (105, 50, 145),
}

# Coarser groups used to score against labels, since exact shades are subjective.
COLOUR_FAMILIES = {
    "Black": "Black", "Charcoal": "Black",
    "Grey": "Grey", "Grey Melange": "Grey", "Silver": "Grey", "Steel": "Grey", "Metallic": "Grey",
    "White": "White", "Off White": "White", "Cream": "White",
    "Beige": "Neutral", "Skin": "Neutral", "Nude": "Neutral", "Khaki": "Neutral",
    "Brown": "Brown", "Tan": "Brown", "Coffee Brown": "Brown", "Mushroom Brown": "Brown",
    "Taupe": "Brown", "Bronze": "Brown", "Copper": "Brown", "Rust": "Brown",
    "Red": "Red", "Maroon": "Red", "Burgundy": "Red",
    "Pink": "Pink", "Rose": "Pink", "Peach": "Pink", "Magenta": "Pink",
    "Orange": "Orange",
    "Yellow": "Yellow", "Gold": "Yellow", "Mustard": "Yellow",
    "Green": "Green", "Olive": "Green", "Sea Green": "Green", "Fluorescent Green": "Green",
    "Lime Green": "Green",
    "Blue": "Blue", "Navy Blue": "Blue", "Teal": "Blue", "Turquoise Blue": "Blue",
    "Purple": "Purple", "Lavender": "Purple", "Mauve": "Purple",
}

_REFERENCE_NAMES = list(REFERENCE_COLOURS)
_ACHROMATIC = np.array([name in ACHROMATIC_NAMES for name in _REFERENCE_NAMES])


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) uint8 RGB array to true CIELAB units.

    OpenCV's 8-bit LAB stretches lightness to 0-255, which would let lightness
    outweigh hue in every distance computed below.
    """
    pixels = rgb.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return lab * [100 / 255, 1, 1] - [0, 128, 128]


_REFERENCE_LAB = _rgb_to_lab(np.array(list(REFERENCE_COLOURS.values())))


def foreground_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Mark product pixels by flood-filling the near-white border inwards.

    Filling from the edges keeps white garments, whose pixels are near-white
    but not connected to the background through a near-white path.
    """
    height, width = image_rgb.shape[:2]
    near_white = (image_rgb.min(axis=2) >= WHITE_THRESHOLD).astype(np.uint8)
    # A one-pixel frame connects every border pixel into a single component.
    framed = cv2.copyMakeBorder(near_white, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    _, components = cv2.connectedComponents(framed, connectivity=4)
    background = components[1:-1, 1:-1] == components[0, 0]
    foreground = ~background
    if foreground.mean() < MIN_FOREGROUND_SHARE:
        return np.ones((height, width), dtype=bool)
    return foreground


def _kmeans_lab(lab: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    lab = lab.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(SEED)
    k = min(k, len(lab))
    _, labels, centres = cv2.kmeans(lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.flatten(), minlength=k) / len(lab)
    order = np.argsort(-shares)
    return centres[order], shares[order]


def nearest_colour_name(lab: np.ndarray, allow_silver: bool = True) -> str:
    """Name a LAB colour, treating near-neutral colours as a separate decision.

    Silver is a metallic finish, so light shaded fabric should not receive it.
    """
    achromatic = bool(np.hypot(lab[1], lab[2]) < ACHROMATIC_CHROMA)
    difference = (_REFERENCE_LAB - lab) * [LIGHTNESS_WEIGHT, 1, 1]
    distances = np.linalg.norm(difference, axis=1)
    distances[_ACHROMATIC != achromatic] = np.inf
    if not allow_silver:
        distances[_REFERENCE_NAMES.index("Silver")] = np.inf
    return _REFERENCE_NAMES[int(np.argmin(distances))]


def lab_to_hex(lab: np.ndarray) -> str:
    opencv_lab = (np.asarray(lab) + [0, 128, 128]) * [255 / 100, 1, 1]
    pixel = np.clip(opencv_lab, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    r, g, b = cv2.cvtColor(pixel, cv2.COLOR_LAB2RGB).flatten()
    return f"#{r:02x}{g:02x}{b:02x}"


def load_rgb(path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def _skin_like(lab: np.ndarray) -> np.ndarray:
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    return (lab[:, 0] > 35) & (lab[:, 0] < 88) & (chroma > 10) & (chroma < 50) & (hue > 30) & (hue < 80)


def foreground_lab(image_rgb: np.ndarray, drop_skin: bool = False) -> np.ndarray:
    """Return product pixels, optionally removing skin tones from on-model shots.

    Skin removal is skipped when it would leave too little of the product,
    which keeps beige and tan garments intact.
    """
    lab = _rgb_to_lab(image_rgb[foreground_mask(image_rgb)])
    if drop_skin:
        keep = ~_skin_like(lab)
        if keep.mean() >= MIN_NON_SKIN_SHARE:
            return lab[keep]
    return lab


def item_dominant_colour(image_rgb: np.ndarray, k: int = 3, is_apparel: bool = False) -> dict:
    """Return the main colour of one product; apparel gets on-model handling."""
    centres, shares = _kmeans_lab(foreground_lab(image_rgb, drop_skin=is_apparel), k)
    return {
        "colour_name": nearest_colour_name(centres[0], allow_silver=not is_apparel),
        "colour_hex": lab_to_hex(centres[0]),
        "colour_share": round(float(shares[0]), 3),
    }


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))


def build_palette(families: pd.Series, shade_names: pd.Series, hexes: pd.Series,
                  top: int = 5) -> list[dict]:
    """Summarise a group of products by their colour families.

    Shares count products per family. Each swatch is the mean measured shade of
    the products whose OpenCV shade falls in the same family, or a reference
    colour when none does. Averaging pixels across products would turn every
    swatch grey.
    """
    shares = families.value_counts(normalize=True).head(top)
    shade_families = shade_names.map(family)
    palette = []
    for name, share in shares.items():
        agrees = (families == name) & (shade_families == name)
        if agrees.any():
            rgb = np.array([_hex_to_rgb(h) for h in hexes[agrees]])
            swatch = lab_to_hex(_rgb_to_lab(rgb).mean(axis=0))
        else:
            swatch = "#{:02x}{:02x}{:02x}".format(*REFERENCE_COLOURS[FAMILY_REFERENCE[name]])
        palette.append({"name": name, "hex": swatch, "share": round(float(share), 3)})
    return palette


def family(colour_name) -> str | None:
    return COLOUR_FAMILIES.get(colour_name)


FAMILIES = sorted(set(COLOUR_FAMILIES.values()))
# The reference shade drawn for a family when no measured shade is available.
FAMILY_REFERENCE = {name: name for name in FAMILIES} | {"Neutral": "Beige"}
