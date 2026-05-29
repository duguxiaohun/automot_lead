"""Image loading helpers for local Qwen3-VL tests."""

from __future__ import annotations

import pathlib
from typing import Any, List, Optional, Tuple

from .prompt_pipeline import SCENARIO_LABELS

try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def auto_detect_scenario_from_route(route_dir: str) -> Optional[str]:
    parts = pathlib.Path(route_dir).resolve().parts
    for p in reversed(parts):
        if p in SCENARIO_LABELS:
            return p
    return None


def ensure_hwc_uint8(img: Any) -> Any:
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"ensure_hwc_uint8 requires numpy: {e}")

    arr = img
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise ValueError(f"RGB frame ndim invalid: {arr.ndim}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def build_synthetic_images(
    num_frames: int = 4,
    height: int = 384,
    width: int = 1152,
) -> List[Any]:
    if not _HAS_PIL:
        raise RuntimeError("PIL is required for synthetic image generation")
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"synthetic image generation requires numpy: {e}")

    images: List[Any] = []
    stripe_w = max(width // 3, 1)
    for i in range(num_frames):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, :stripe_w, :] = (210, 70, 70)
        rgb[:, stripe_w:2 * stripe_w, :] = (70, 170, 90)
        rgb[:, 2 * stripe_w:, :] = (70, 110, 220)
        rgb[:40, :180, :] = min(255, 30 + i * 40)
        images.append(_PILImage.fromarray(rgb, mode="RGB"))
    return images


def build_synthetic_raw_and_model_input(
    num_frames: int = 4,
    raw_size: Tuple[int, int] = (1920, 1080),
    model_input_size: Tuple[int, int] = (1152, 384),
) -> Tuple[List[Any], List[Any]]:
    raw_w, raw_h = raw_size
    mi_w, mi_h = model_input_size
    raw_imgs = build_synthetic_images(num_frames=num_frames, height=raw_h, width=raw_w)
    model_input_imgs = [
        img.resize((mi_w, mi_h), resample=_PILImage.BILINEAR) for img in raw_imgs
    ]
    return raw_imgs, model_input_imgs


def load_lead_rgb_clip(
    route_dir: str,
    anchor: int = 12,
    rgb_frame_step: int = 1,
    rgb_frame_count: int = 4,
) -> Tuple[List[Any], List[Any]]:
    try:
        import cv2
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"real LEAD RGB loading requires opencv-python + numpy: {e}")
    if not _HAS_PIL:
        raise RuntimeError("PIL is required for real RGB loading")

    route = pathlib.Path(route_dir)
    rgb_dir = route / "rgb"
    if not rgb_dir.exists():
        raise FileNotFoundError(f"missing rgb directory: {rgb_dir}")
    rgb_files = sorted(rgb_dir.glob("*.jpg"))
    if not rgb_files:
        raise FileNotFoundError(f"no .jpg files under {rgb_dir}")

    total = len(rgb_files)
    if anchor < 0 or anchor >= total:
        raise ValueError(f"anchor={anchor} out of range [0, {total - 1}]")
    rgb_frame_step = max(1, rgb_frame_step)
    rgb_frame_count = max(1, rgb_frame_count)

    desc = [max(anchor - i * rgb_frame_step, 0) for i in range(rgb_frame_count)]
    asc = list(reversed(desc))
    print(f"[load] route={route} total_frames={total} anchor={anchor}")
    print(f"[load] sampled rgb indices (asc): {asc}")

    rgb_ndarray_list: List[Any] = []
    for idx in asc:
        rgb_path = rgb_dir / f"{idx:04d}.jpg"
        if not rgb_path.exists():
            rgb_path = rgb_files[idx]
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2 failed to read image: {rgb_path}")
        rgb_ndarray_list.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    rgb_clip = np.stack(rgb_ndarray_list, axis=0)
    raw_imgs: List[Any] = []
    for t in range(rgb_clip.shape[0]):
        rgb_hwc = ensure_hwc_uint8(rgb_clip[t])
        raw_imgs.append(_PILImage.fromarray(rgb_hwc, mode="RGB"))
    return raw_imgs, list(raw_imgs)

