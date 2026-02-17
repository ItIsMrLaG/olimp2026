import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path

JPEG_QUALITY = 95


@dataclass(frozen=True)
class RecolorConfig:
    """Parameters for a single foliage hue-shift operation."""

    src_hue_ranges: list[tuple[int, int]]
    dst_hue_center: int
    dst_hue_spread: int
    sat_min: int
    val_min: int
    sat_factor: float
    val_factor: float


AUTUMN_TO_SUMMER = RecolorConfig(
    src_hue_ranges=[(0, 35), (170, 180)],
    dst_hue_center=52,
    dst_hue_spread=10,
    sat_min=35,
    val_min=50,
    sat_factor=0.60,
    val_factor=0.90,
)

SUMMER_TO_AUTUMN = RecolorConfig(
    src_hue_ranges=[(30, 85)],
    dst_hue_center=22,
    dst_hue_spread=10,
    sat_min=30,
    val_min=40,
    sat_factor=1.35,
    val_factor=1.08,
)


# --- Low-level helpers ---


def _hue_in_range(h: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Boolean mask for pixels whose hue falls in [lo, hi], with wrap-around."""
    if lo <= hi:
        return (h >= lo) & (h <= hi)
    return (h >= lo) | (h <= hi)


def _normalize_to_unit(h: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Map hue values from [lo, hi] (possibly wrapping) to [0..1]."""
    if lo <= hi:
        return (h - lo) / max(hi - lo, 1)

    span = max((180 - lo) + hi, 1)
    out = h.copy()
    out[out >= lo] -= lo
    out[out < lo] += (180 - lo)
    return out / span


# --- Mask building ---


def _build_raw_mask(hsv: np.ndarray, cfg: RecolorConfig) -> np.ndarray:
    """Binary uint8 mask: pixels matching hue ranges with sufficient S and V."""
    h, s, v = cv2.split(hsv)
    mask = np.zeros(h.shape, dtype=np.uint8)

    for lo, hi in cfg.src_hue_ranges:
        mask[_hue_in_range(h, lo, hi)] = 255

    mask[(s < cfg.sat_min) | (v < cfg.val_min)] = 0
    return mask


def _refine_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological cleanup + Gaussian blur -> soft float32 [0..1] mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    soft = mask.astype(np.float32) / 255.0
    return cv2.GaussianBlur(soft, (21, 21), 0)


# --- Pixel transforms ---


def _remap_hue(
    h_channel: np.ndarray,
    mask: np.ndarray,
    cfg: RecolorConfig,
) -> np.ndarray:
    """Linearly map masked pixel hues from src ranges to dst_center +/- dst_spread."""
    h = h_channel.astype(np.float32).copy()
    active = mask > 0.01

    if not np.any(active):
        return h_channel

    for lo, hi in cfg.src_hue_ranges:
        in_range = active & _hue_in_range(h, lo, hi)
        t = _normalize_to_unit(h[in_range], lo, hi)
        new_hue = cfg.dst_hue_center - cfg.dst_hue_spread + t * (2 * cfg.dst_hue_spread)
        h[in_range] = np.clip(new_hue, 0, 179)

    return h.astype(np.uint8)


def _apply_sv_factors(
    hsv: np.ndarray,
    mask: np.ndarray,
    cfg: RecolorConfig,
) -> np.ndarray:
    """Scale S and V channels within masked region via alpha blend."""
    out = hsv.copy().astype(np.float32)

    for ch_idx, factor in [(1, cfg.sat_factor), (2, cfg.val_factor)]:
        ch = out[:, :, ch_idx]
        out[:, :, ch_idx] = ch + (ch * factor - ch) * mask

    return np.clip(out, 0, 255).astype(np.uint8)


def _alpha_blend(
    original: np.ndarray,
    converted: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Blend converted image with original using soft mask."""
    m = mask[:, :, np.newaxis]
    blended = original.astype(np.float32) + \
              (converted.astype(np.float32) - original.astype(np.float32)) * m
    return np.clip(blended, 0, 255).astype(np.uint8)


# --- Public API ---


def recolor_foliage(img_bgr: np.ndarray, cfg: RecolorConfig) -> np.ndarray:
    """Recolor foliage in a BGR image according to the given config."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    raw_mask = _build_raw_mask(hsv, cfg)
    mask = _refine_mask(raw_mask)

    hsv_new = hsv.copy()
    hsv_new[:, :, 0] = _remap_hue(hsv[:, :, 0], mask, cfg)
    hsv_new = _apply_sv_factors(hsv_new, mask, cfg)

    converted = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
    return _alpha_blend(img_bgr, converted, mask)


# --- I/O ---


def process_tasks(
    tasks: list[tuple[str, str, RecolorConfig]],
    base: Path = Path("."),
) -> None:
    """Load each source image, apply recolor, save result."""
    for src_name, dst_name, cfg in tasks:
        img = cv2.imread(str(base / src_name))
        if img is None:
            print(f"[!] Failed to load {src_name}")
            continue

        result = recolor_foliage(img, cfg)
        cv2.imwrite(str(base / dst_name), result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        print(f"[ok] {dst_name} saved ({result.shape[1]}x{result.shape[0]})")


if __name__ == "__main__":
    process_tasks([
        ("Photo1.jpg", "Summer.jpg", AUTUMN_TO_SUMMER),
        ("Photo2.jpg", "Autumn.jpg", SUMMER_TO_AUTUMN),
    ])
