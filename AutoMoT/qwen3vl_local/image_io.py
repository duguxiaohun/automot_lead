"""Image loading helpers for local Qwen3-VL tests.

这里统一处理“图片从哪里来”和“以什么格式交给 processor”：

- 有真实 route_dir 时，读取 LEAD route 下的 rgb/*.jpg。
- route_dir 为空时，生成简单合成图，只验证推理链路是否能跑通。
- 返回值统一是 PIL RGB 图片列表，因为 HuggingFace Qwen processor 可以直接消费 PIL。
"""

from __future__ import annotations

import pathlib
from typing import Any, List, Optional, Tuple

from .prompt_pipeline import SCENARIO_LABELS

try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except Exception:
    # PIL 是读取/生成图片的硬依赖，但放在 try 里可以让 import 阶段先通过，
    # 真正调用图片函数时再抛出更明确的错误。
    _HAS_PIL = False


def auto_detect_scenario_from_route(route_dir: str) -> Optional[str]:
    """从 LEAD route 路径里自动识别 scenario 名称。

    LEAD 数据通常组织成 data/<Scenario>/<route_name>/，所以倒序检查路径分量，
    找到第一个落在 SCENARIO_LABELS 里的名字即可。
    """

    parts = pathlib.Path(route_dir).resolve().parts
    for p in reversed(parts):
        if p in SCENARIO_LABELS:
            return p
    return None


def ensure_hwc_uint8(img: Any) -> Any:
    """把输入图像规范成 HWC uint8 RGB ndarray。

    LEAD/调试代码里可能拿到 CHW、float [0,1]、RGBA、单通道等格式。
    Qwen processor 最稳妥的输入是 PIL RGB，因此这里先统一成 HWC uint8。
    """

    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"ensure_hwc_uint8 requires numpy: {e}")

    arr = img
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        # CHW -> HWC。常见于深度学习张量转 numpy 后直接传进来。
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise ValueError(f"RGB frame ndim invalid: {arr.ndim}")
    if arr.dtype != np.uint8:
        # float 图按 [0,1] 处理并转成 0..255。
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    if arr.shape[2] > 3:
        # 丢掉 alpha 或其它额外通道，保留 RGB。
        arr = arr[:, :, :3]
    if arr.shape[2] == 1:
        # 灰度图复制成 3 通道，保持 processor 输入一致。
        arr = np.repeat(arr, 3, axis=2)
    return arr


def build_synthetic_images(
    num_frames: int = 4,
    height: int = 384,
    width: int = 1152,
) -> List[Any]:
    """生成简单三色合成图。

    这些图没有驾驶语义，只用于在没有 LEAD 数据时测试模型加载、图片 token 注入、
    prefill/decode 和输出落盘链路。三条色带模拟 left/front/right 三视角拼接。
    """

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
        # 三段颜色分别占据左/中/右区域，便于肉眼确认图像没有被转置或裁错。
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
    """同时生成原始尺寸图和模型输入尺寸图。

    raw_imgs 用来模拟原始传感器/日志图片；model_input_imgs 是实际送给 Qwen 的尺寸。
    真实 LEAD 路径目前两者相同，合成路径保留这个双返回值是为了和真实路径接口一致。
    """

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
    """从 LEAD route 读取一段 RGB clip。

    anchor 表示当前帧索引；函数按 rgb_frame_step 往前采样 rgb_frame_count 帧，
    最后再反转成 oldest -> newest 的时间顺序，符合 prompt 中“最近观测序列”的描述。
    """

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

    # 先按当前帧向历史回看生成降序索引，再反转成模型看到的时间顺序。
    desc = [max(anchor - i * rgb_frame_step, 0) for i in range(rgb_frame_count)]
    asc = list(reversed(desc))
    print(f"[load] route={route} total_frames={total} anchor={anchor}")
    print(f"[load] sampled rgb indices (asc): {asc}")

    rgb_ndarray_list: List[Any] = []
    for idx in asc:
        rgb_path = rgb_dir / f"{idx:04d}.jpg"
        if not rgb_path.exists():
            # 有些数据集可能不是严格 0000.jpg 命名，退回 sorted 列表索引。
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

