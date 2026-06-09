"""VAE / patch-unpatch 重建诊断脚本。

本脚本只做评估和可视化，不训练任何参数：

- v1：image -> VAE.encode -> VAE.decode -> image
- v2：image -> VAE.encode -> patch -> unpatch -> VAE.decode -> image

默认用法（运行目录建议为 AutoMoT/）：

    # 小样本 eval：默认完整 dump 每条样本的 compare.png。
    python vae_standalone/vae_reconstruct.py \
        --version v1 \
        --save-root checkpoints/patch_unpatch_v1 \
        --max-samples 100

    # v2 小样本 eval：VAE + patch/unpatch，默认读取 best patch/unpatch 权重。
    python vae_standalone/vae_reconstruct.py \
        --version v2 \
        --save-root checkpoints/patch_unpatch_v1 \
        --max-samples 100

    # 大样本 eval：--max-samples 0 表示全量候选图，默认不 dump 图像，主要看本地 JSON 和 TensorBoard。
    python vae_standalone/vae_reconstruct.py \
        --version v2 \
        --save-root checkpoints/patch_unpatch_v1

    # 大样本 eval：最多随机抽 1000 张；如果候选图不足 1000 张，则自动使用全量候选图。
    python vae_standalone/vae_reconstruct.py \
        --version v2 \
        --save-root checkpoints/patch_unpatch_v1 \
        --max-samples 1000

默认输入：

- route_dir 参考其它 runner 的远程 LEAD 数据路径：
  /datashare/IOL4SGH/data/data/Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46
- 不传 --jsonl / --image-path / --image-paths 时，自动从 route_dir/rgb/*.jpg 全随机抽样。
- 默认不固定随机 seed；每次前几张可视化样本都会重新随机。需要复现实验时再显式传 --seed。
- --max-samples 1..100 是小样本 eval，默认完整 dump compare.png。
- --max-samples 0 或 >100 是大样本 eval，默认只写统计和 TB；--max-samples 大于候选图数量时自动使用全量候选图。

默认权重：

- v1/v2 VAE：vae_standalone/config/vae_only.yaml + vae_standalone/weights/vae_only.safetensors
- v2 patch/unpatch：checkpoints/patch_unpatch_v1/latest/weights/patch_unpatch_best.safetensors

默认输出：

- checkpoints/patch_unpatch_v1/vae_reconstruct_eval/<v1|v2>/run_<YYYYmmdd_HHMMSS_ffffff>/
- metrics_summary.json：本次综合 loss / 指标
- metrics_per_image.jsonl：每张图的 loss / 指标
- loss_doc.json：每个 loss / metric 字段的含义
- visual_cases/：小样本 eval 的逐 case PNG 对比，每个 case 含 compare.png
- tb/：TensorBoard scalar/image；可用 ``bash qwen3vl_local/sft/tb_serve.sh checkpoints/patch_unpatch_v1`` 查看
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import pathlib
import random
import sys
from dataclasses import asdict, dataclass
from typing import Any, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[1]
_PROJECT_ROOT = _THIS_FILE.parents[2]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.goalgen.dit import Patchify, Unpatchify  # noqa: E402


_DEFAULT_ROUTE_DIR = pathlib.Path(
    "/datashare/IOL4SGH/data/data/Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46"
)
_DEFAULT_PATCH_OUTPUT_ROOT = _AUTOMOT_ROOT / "checkpoints" / "patch_unpatch_v1"
_VAE_STANDALONE_DIR = _AUTOMOT_ROOT / "vae_standalone"
_DEFAULT_PATCH_BEST = _DEFAULT_PATCH_OUTPUT_ROOT / "latest" / "weights" / "patch_unpatch_best.safetensors"
_NO_RUN_SUBDIR_PATCH_BEST = _DEFAULT_PATCH_OUTPUT_ROOT / "weights" / "patch_unpatch_best.safetensors"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_DEFAULT_SMALL_DUMP_MAX_SAMPLES = 100

try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: E402

    _TB_AVAILABLE = True
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False


@dataclass
class ReconMetrics:
    image_path: str
    pipeline: str
    input_shape: List[int]
    latent_shape: List[int]
    vae_pixel_mse: float
    vae_pixel_l1: float
    vae_psnr: float
    patch_latent_mse: float | None = None
    patch_latent_l1: float | None = None
    patch_pixel_mse: float | None = None
    patch_pixel_l1: float | None = None
    patch_psnr: float | None = None


METRIC_DOC: dict[str, str] = {
    "vae_pixel_mse": "VAE-only 重建图像 vs 原图的像素 MSE；输入/输出都在 [-1,1] 归一化空间，越小越好。",
    "vae_pixel_l1": "VAE-only 重建图像 vs 原图的像素 L1；输入/输出都在 [-1,1] 归一化空间，越小越好。",
    "vae_psnr": "VAE-only 重建图像 vs 原图的 PSNR(dB)；越大越好，可理解为 VAE 本身的重建上限参考。",
    "patch_latent_mse": "v2 专用：patch -> unpatch 后 latent vs 原始 VAE latent 的 MSE；越小表示 patch/unpatch 越接近恒等映射。",
    "patch_latent_l1": "v2 专用：patch -> unpatch 后 latent vs 原始 VAE latent 的 L1；越小越好。",
    "patch_pixel_mse": "v2 专用：patch/unpatch latent 再经 VAE decode 后图像 vs 原图的像素 MSE；越小越好。",
    "patch_pixel_l1": "v2 专用：patch/unpatch latent 再经 VAE decode 后图像 vs 原图的像素 L1；越小越好。",
    "patch_psnr": "v2 专用：patch/unpatch latent 再经 VAE decode 后图像 vs 原图的 PSNR(dB)；越大越好。",
}


class PatchUnpatchAutoencoder(torch.nn.Module):
    """保持和 DiTMoT.patch/unpatch 完全相同 state_dict key 的小包装。"""

    def __init__(self, latent_channels: int, hidden_dim: int, patch_size: int):
        super().__init__()
        self.patch = Patchify(latent_channels, hidden_dim, patch_size)
        self.unpatch = Unpatchify(hidden_dim, latent_channels, patch_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        tokens, grid = self.patch(z)
        return self.unpatch(tokens, grid)


def load_model(model: torch.nn.Module, device: str) -> None:
    model.to(device)


def unload_model(model: torch.nn.Module) -> None:
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前环境 torch.cuda.is_available()=False")
    return device


def preprocess_one_image(file_path: str, target_height: int, target_width: int, do_center_crop: bool) -> torch.Tensor:
    image = Image.open(file_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    ori_w, ori_h = image.size
    print(f"输入图像: {file_path} | 原始尺寸: {ori_w}x{ori_h}")

    if do_center_crop:
        if ori_w / ori_h > target_width / target_height:
            tmp_w = int(target_width / target_height * ori_h)
            left = (ori_w - tmp_w) // 2
            right = (ori_w + tmp_w) // 2
            image = image.crop((left, 0, right, ori_h))
        elif ori_w / ori_h < target_width / target_height:
            tmp_h = int(target_height / target_width * ori_w)
            top = (ori_h - tmp_h) // 2
            bottom = (ori_h + tmp_h) // 2
            image = image.crop((0, top, ori_w, bottom))

    image = image.resize((target_width, target_height), resample=Image.LANCZOS)
    return pil_to_normalized_tensor(image)


def pil_to_unit_tensor(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def pil_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    return pil_to_unit_tensor(image) * 2.0 - 1.0


def infer_target_size(image_path: str, do_center_crop: bool, height: int | None, width: int | None) -> tuple[int, int]:
    with Image.open(image_path) as image:
        inferred_width, inferred_height = image.size

    if do_center_crop:
        target_height = height if height is not None else 384
        target_width = width if width is not None else 1152
    else:
        target_height = height if height is not None else inferred_height
        target_width = width if width is not None else inferred_width

    if target_height % 64 != 0 or target_width % 64 != 0:
        raise ValueError(f"height/width 必须是 64 的倍数，got {target_height}x{target_width}")
    return target_height, target_width


def load_image_tensor(path: str, args: argparse.Namespace, device: str) -> torch.Tensor:
    h, w = infer_target_size(path, do_center_crop=(not args.no_crop), height=args.height, width=args.width)
    x = preprocess_one_image(path, h, w, do_center_crop=(not args.no_crop)).unsqueeze(0)
    return x.to(device)


@torch.no_grad()
def encode_first_stage(
    first_stage_model,
    x: torch.Tensor,
    scale_factor: float,
    n_samples_per_round: int,
    autocast_enabled: bool,
) -> torch.Tensor:
    n_samples = _batch_split_size(n_samples_per_round, x.shape[0])
    n_rounds = math.ceil(x.shape[0] / n_samples)
    all_out = []

    with torch.autocast("cuda", enabled=autocast_enabled):
        for n in range(n_rounds):
            current_x = x[n * n_samples: (n + 1) * n_samples]
            all_out.append(first_stage_model.encode(current_x))

    return torch.cat(all_out, dim=0) * scale_factor


@torch.no_grad()
def decode_first_stage(
    first_stage_model,
    z: torch.Tensor,
    scale_factor: float,
    n_samples_per_round: int,
    autocast_enabled: bool,
    overlap: int = 0,
) -> torch.Tensor:
    z = z / scale_factor
    n_samples = _batch_split_size(n_samples_per_round, z.shape[0])
    all_out = []

    with torch.autocast("cuda", enabled=autocast_enabled):
        if overlap < n_samples and overlap > 0:
            previous_z = z[:overlap]
            for current_z in z[overlap:].split(n_samples - overlap, dim=0):
                kwargs = _decode_kwargs(first_stage_model, current_z.shape[0] + overlap)
                context_z = torch.cat((previous_z, current_z), dim=0)
                previous_z = current_z[-overlap:]
                out = first_stage_model.decode(context_z, **kwargs)
                if not all_out:
                    all_out.append(out)
                else:
                    all_out[-1][-overlap:] = (all_out[-1][-overlap:] + out[:overlap]) / 2
                    all_out.append(out[overlap:])
        else:
            for current_z in z.split(n_samples, dim=0):
                kwargs = _decode_kwargs(first_stage_model, current_z.shape[0])
                all_out.append(first_stage_model.decode(current_z, **kwargs))

    return torch.cat(all_out, dim=0)


def pixel_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> tuple[float, float, float]:
    mse = F.mse_loss(reconstructed, original).item()
    l1 = F.l1_loss(reconstructed, original).item()
    psnr = 20 * np.log10(2.0) - 10 * np.log10(mse) if mse > 0 else float("inf")
    return float(mse), float(l1), float(psnr)


def latent_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> tuple[float, float]:
    return float(F.mse_loss(reconstructed, original).item()), float(F.l1_loss(reconstructed, original).item())


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = ((tensor.detach().float().cpu().clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
    arr = (tensor.numpy() * 255).astype(np.uint8)
    return Image.fromarray(np.transpose(arr, (1, 2, 0)))


def contact_sheet(items: Sequence[tuple[str, Image.Image]], path: pathlib.Path) -> None:
    """把单个 case 的几张对比图拼成一张横向预览图。"""

    if not items:
        return
    thumbs: List[Image.Image] = []
    cell_w = 360
    label_h = 28
    for label, image in items:
        im = image.copy()
        im.thumbnail((cell_w, 240), Image.LANCZOS)
        canvas = Image.new("RGB", (cell_w, im.height + label_h), "white")
        canvas.paste(im, ((cell_w - im.width) // 2, label_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), label, fill=(0, 0, 0))
        thumbs.append(canvas)

    cols = min(len(thumbs), 3)
    rows = [thumbs[i: i + cols] for i in range(0, len(thumbs), cols)]
    sheet_w = cell_w * cols
    sheet_h = sum(max(im.height for im in row) for row in rows if row)
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    y = 0
    for row in rows:
        if not row:
            continue
        row_h = max(im.height for im in row)
        for x_idx, im in enumerate(row):
            sheet.paste(im, (x_idx * cell_w, y))
        y += row_h
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def load_patch_model(args: argparse.Namespace, device: str) -> PatchUnpatchAutoencoder:
    from safetensors.torch import load_file

    weights = pathlib.Path(args.patch_weights).expanduser()
    if not weights.exists():
        raise FileNotFoundError(
            f"patch/unpatch 权重不存在: {weights}\n"
            f"v2 默认会读取 train_patch_unpatch.py 约定的 best 权重：{_DEFAULT_PATCH_BEST}\n"
            "如果你还没有训练 patch/unpatch，请先跑 train_patch_unpatch.py。"
        )

    model = PatchUnpatchAutoencoder(
        latent_channels=args.latent_channels,
        hidden_dim=args.patch_hidden_dim,
        patch_size=args.patch_size,
    ).to(device=device, dtype=torch.float32).eval()
    sd = load_file(str(weights.resolve()))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    required = {"patch.proj.weight", "patch.proj.bias", "unpatch.proj.weight", "unpatch.proj.bias"}
    missing_required = sorted(required - set(sd.keys()))
    if missing_required:
        raise ValueError(f"patch/unpatch 权重缺少必要 key: {missing_required}")
    if unexpected:
        print(f"[patch] 忽略权重中的额外 key: {unexpected}")
    if missing:
        missing_non_required = sorted(set(missing) - required)
        if missing_non_required:
            print(f"[patch] 非必要 key 未加载: {missing_non_required}")
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[patch] 已加载 {weights.resolve()}")
    return model


def load_vae_model(config_path: str, weights_path: str):
    """按 vae_standalone 原用法加载 VAE；依赖延迟到真正 eval 时再 import。"""

    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from vwm.util import instantiate_from_config

    cfg = OmegaConf.load(str(config_path))
    first_stage_model = instantiate_from_config(cfg.first_stage_config)
    sd = load_file(str(weights_path))
    missing, unexpected = first_stage_model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[vae] 未加载到的 key 数量: {len(missing)}")
    if unexpected:
        print(f"[vae] 权重中的额外 key 数量: {len(unexpected)}")
    first_stage_model = first_stage_model.eval()
    for p in first_stage_model.parameters():
        p.requires_grad_(False)
    return cfg, first_stage_model


def _batch_split_size(n_samples_per_round: int | None, batch_size: int) -> int:
    return int(n_samples_per_round) if n_samples_per_round else int(batch_size)


def _decode_kwargs(first_stage_model, timesteps: int) -> dict[str, int]:
    from vwm.modules.autoencoding.temporal_ae import VideoDecoder

    if isinstance(first_stage_model.decoder, VideoDecoder):
        return {"timesteps": int(timesteps)}
    return {}


def resolve_pipeline(args: argparse.Namespace) -> str:
    if args.pipeline != "auto":
        return args.pipeline
    if args.version == "v1":
        return "vae"
    if args.version == "v2":
        return "patch"
    raise ValueError(f"未知 version: {args.version}")


def resolve_default_paths(args: argparse.Namespace, pipeline: str) -> None:
    """把空参数补成项目默认路径，让用户只选 v1/v2 也能跑。"""

    cfg_path, vae_weights_path = default_vae_paths()
    if not args.config:
        args.config = str(cfg_path)
    if not args.weights:
        args.weights = str(vae_weights_path)
    if pipeline == "patch" and not args.patch_weights:
        args.patch_weights = str(resolve_default_patch_weights())


def default_vae_paths() -> tuple[pathlib.Path, pathlib.Path]:
    """与 qwen3vl_local.goalgen.vae.default_vae_paths 保持同一默认路径。"""

    return (
        _VAE_STANDALONE_DIR / "config" / "vae_only.yaml",
        _VAE_STANDALONE_DIR / "weights" / "vae_only.safetensors",
    )


def resolve_default_patch_weights() -> pathlib.Path:
    """优先用 latest/best；latest 不存在时兜底找最近一次 run 的 best。"""

    for candidate in (_DEFAULT_PATCH_BEST, _NO_RUN_SUBDIR_PATCH_BEST):
        if candidate.exists():
            return candidate
    run_candidates = [
        p for p in _DEFAULT_PATCH_OUTPUT_ROOT.glob("run_*/weights/patch_unpatch_best.safetensors")
        if p.is_file()
    ]
    if run_candidates:
        return max(run_candidates, key=lambda p: p.stat().st_mtime)
    return _DEFAULT_PATCH_BEST


def make_run_output_dir(args: argparse.Namespace) -> pathlib.Path:
    """输出统一挂到 patch_unpatch_v1 下，并按版本与时间分目录。"""

    base = pathlib.Path(args.save_root).expanduser() if args.save_root else _DEFAULT_PATCH_OUTPUT_ROOT
    run_tag = args.run_tag or _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return base / "vae_reconstruct_eval" / args.version / f"run_{run_tag}"


def make_rng(seed: int | None) -> tuple[random.Random, int]:
    """默认用系统随机数；显式 --seed 时才复现同一批样本。"""

    resolved_seed = int(seed) if seed is not None else int.from_bytes(os.urandom(8), "big")
    return random.Random(resolved_seed), resolved_seed


def collect_paths_from_jsonl(jsonl_path: str) -> List[str]:
    rows: List[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    paths: List[str] = []
    for row in rows:
        for p in row.get("history_rgb_paths", []) or []:
            paths.append(str(p))
        for key in ("current_rgb_path", "target_rgb_path", "rgb_path"):
            if row.get(key):
                paths.append(str(row[key]))
    return paths


def collect_paths_from_route(route_dir: str) -> List[str]:
    route = pathlib.Path(route_dir).expanduser()
    rgb_dir = route / "rgb"
    if rgb_dir.exists():
        paths = sorted(str(p) for p in rgb_dir.glob("*.jpg"))
        if paths:
            return paths

    if not route.exists():
        raise FileNotFoundError(
            f"默认 route_dir 不存在: {route}\n"
            "请在远程数据机上运行，或传 --route-dir / --jsonl / --image-path 指定输入。"
        )

    paths = sorted(
        str(p)
        for p in route.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES and "rgb" in str(p).lower()
    )
    if not paths:
        paths = sorted(str(p) for p in route.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"route_dir 下没有可用图像: {route}")
    return paths


def resolve_sample_limit(args: argparse.Namespace) -> int:
    """解析 --max-samples；0 表示全量。"""

    return max(0, int(args.max_samples))


def resolve_full_dump(args: argparse.Namespace, sample_limit: int) -> bool:
    """固定规则：小样本 dump，大样本只写 JSON/TB。"""

    return 0 < sample_limit <= _DEFAULT_SMALL_DUMP_MAX_SAMPLES


def collect_image_paths(args: argparse.Namespace, rng: random.Random, sample_limit: int) -> List[str]:
    paths: List[str] = []
    if args.jsonl:
        paths.extend(collect_paths_from_jsonl(args.jsonl))
    if args.image_paths:
        paths.extend(args.image_paths)
    if args.image_path:
        paths.append(args.image_path)
    if not paths:
        route_dir = args.route_dir or str(_DEFAULT_ROUTE_DIR)
        paths.extend(collect_paths_from_route(route_dir))

    deduped: List[str] = []
    seen = set()
    for p in paths:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    if not deduped:
        raise ValueError("没有收集到任何图像；请检查 --route-dir / --jsonl / --image-path。")
    for p in deduped:
        if not os.path.exists(p):
            raise FileNotFoundError(f"图像不存在: {p}")

    rng.shuffle(deduped)
    if sample_limit > 0:
        deduped = deduped[: min(sample_limit, len(deduped))]
    return deduped


@torch.no_grad()
def process_image(
    image_path: str,
    first_stage_model,
    patch_model: PatchUnpatchAutoencoder | None,
    cfg,
    args: argparse.Namespace,
    device: str,
    pipeline: str,
) -> tuple[ReconMetrics, dict[str, Image.Image]]:
    x = load_image_tensor(image_path, args, device)
    autocast_enabled = device.startswith("cuda") and (not bool(cfg.disable_first_stage_autocast))
    n_round = int(cfg.en_and_decode_n_samples_a_time)
    scale_factor = float(cfg.scale_factor)

    z = encode_first_stage(first_stage_model, x, scale_factor, n_round, autocast_enabled)
    rec_vae = decode_first_stage(first_stage_model, z, scale_factor, n_round, autocast_enabled, overlap=0).clamp(-1, 1)

    vae_mse, vae_l1, vae_psnr = pixel_metrics(x, rec_vae)
    metrics = ReconMetrics(
        image_path=image_path,
        pipeline=pipeline,
        input_shape=list(x.shape),
        latent_shape=list(z.shape),
        vae_pixel_mse=vae_mse,
        vae_pixel_l1=vae_l1,
        vae_psnr=vae_psnr,
    )

    visuals = {
        "00_original": tensor_to_pil(x),
        "01_vae_recon": tensor_to_pil(rec_vae),
    }

    if pipeline == "patch":
        if patch_model is None:
            raise RuntimeError("pipeline=patch but patch_model is None")
        z_patch = patch_model(z.float()).to(dtype=z.dtype)
        patch_mse, patch_l1 = latent_metrics(z, z_patch)
        rec_patch = decode_first_stage(first_stage_model, z_patch, scale_factor, n_round, autocast_enabled, overlap=0).clamp(-1, 1)
        patch_pixel_mse, patch_pixel_l1, patch_psnr = pixel_metrics(x, rec_patch)
        metrics.patch_latent_mse = patch_mse
        metrics.patch_latent_l1 = patch_l1
        metrics.patch_pixel_mse = patch_pixel_mse
        metrics.patch_pixel_l1 = patch_pixel_l1
        metrics.patch_psnr = patch_psnr
        visuals = {
            "00_original": tensor_to_pil(x),
            "01_vae_recon": tensor_to_pil(rec_vae),
            "02_patch_recon": tensor_to_pil(rec_patch),
        }

    return metrics, visuals


def write_visual_case(output_dir: pathlib.Path, idx: int, image_path: str, metrics: ReconMetrics, visuals: dict[str, Image.Image]) -> None:
    stem = pathlib.Path(image_path).stem
    case_dir = output_dir / "visual_cases" / f"{idx:04d}_{stem}"
    case_dir.mkdir(parents=True, exist_ok=True)
    items: List[tuple[str, Image.Image]] = []
    for name, image in visuals.items():
        out_path = case_dir / f"{name}.png"
        image.save(out_path)
        items.append((name, image))
    contact_sheet(items, case_dir / "compare.png")
    (case_dir / "metrics.json").write_text(json.dumps(asdict(metrics), ensure_ascii=False, indent=2), encoding="utf-8")


def aggregate_metrics(metrics: Sequence[ReconMetrics]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = [
        "vae_pixel_mse",
        "vae_pixel_l1",
        "vae_psnr",
        "patch_latent_mse",
        "patch_latent_l1",
        "patch_pixel_mse",
        "patch_pixel_l1",
        "patch_psnr",
    ]
    out: dict[str, float] = {}
    for key in keys:
        vals = [getattr(m, key) for m in metrics if getattr(m, key) is not None]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[f"{key}/count"] = float(arr.size)
            out[f"{key}/mean"] = float(np.mean(vals))
            out[f"{key}/median"] = float(np.median(vals))
            out[f"{key}/std"] = float(np.std(arr))
            out[f"{key}/min"] = float(np.min(arr))
            out[f"{key}/max"] = float(np.max(arr))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="VAE / patch-unpatch 重建诊断：随机抽图、统计 loss、写 TensorBoard 和 PNG 对比。")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1",
                        help="自动链路规则：v1 默认 VAE-only，v2 默认 VAE+patch。")
    parser.add_argument("--pipeline", choices=["auto", "vae", "patch"], default="auto")
    parser.add_argument("--config", type=str, default="", help="VAE 配置路径；默认使用 vae_standalone/config/vae_only.yaml")
    parser.add_argument("--weights", type=str, default="", help="VAE 权重路径；默认使用 vae_standalone/weights/vae_only.safetensors")
    parser.add_argument("--patch-weights", type=str, default="",
                        help=f"v2 patch/unpatch 权重；默认 {_DEFAULT_PATCH_BEST}")
    parser.add_argument("--patch-hidden-dim", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--route-dir", type=str, default=str(_DEFAULT_ROUTE_DIR),
                        help="不传图像/jsonl 时从该 LEAD route 的 rgb/*.jpg 随机抽样。")
    parser.add_argument("--jsonl", type=str, default="", help="GoalGen jsonl；会读取 history/current/target RGB 路径")
    parser.add_argument("--image-path", type=str, default=None, help="单图模式输入")
    parser.add_argument("--image-paths", nargs="+", default=None, help="多图模式输入")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="随机评估多少张图；0 表示使用全部候选图；大于候选图数量时自动退化为全量。默认 <=100 视作小样本 dump，>100 视作大样本只写 JSON/TB。")
    parser.add_argument("--height", type=int, default=None, help="目标高度（64倍数；no_crop 时可省略）")
    parser.add_argument("--width", type=int, default=None, help="目标宽度（64倍数；no_crop 时可省略）")
    parser.add_argument("--no-crop", action="store_true", help="不做中心裁剪")
    parser.add_argument("--crop", action="store_false", dest="no_crop", help="启用中心裁剪")
    parser.set_defaults(no_crop=True)
    parser.add_argument("--device", type=str, default="auto", help="auto / cuda / cpu")
    parser.add_argument("--save-root", type=str, default="",
                        help=f"eval 输出 base；默认 {_DEFAULT_PATCH_OUTPUT_ROOT}")
    parser.add_argument("--run-tag", type=str, default="", help="输出 run 名后缀；默认当前时间 YYYYmmdd_HHMMSS_ffffff。")
    parser.add_argument("--tb", action="store_true", default=True, help="写 TensorBoard scalar/image")
    parser.add_argument("--no-tb", dest="tb", action="store_false")
    parser.add_argument("--seed", type=int, default=None,
                        help="默认不固定 seed，每次重新随机；传整数可复现同一批样本。")
    parser.add_argument("--log-every", type=int, default=20, help="每处理多少张图打印一次进度。")
    args = parser.parse_args()

    device = resolve_device(args.device)
    pipeline = resolve_pipeline(args)
    resolve_default_paths(args, pipeline)
    sample_limit = resolve_sample_limit(args)
    full_dump_enabled = resolve_full_dump(args, sample_limit)
    output_dir = make_run_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng, resolved_seed = make_rng(args.seed)

    cfg, first_stage_model = load_vae_model(args.config, args.weights)

    patch_model = load_patch_model(args, device) if pipeline == "patch" else None
    image_paths = collect_image_paths(args, rng, sample_limit)
    print(
        f"[run] version={args.version} pipeline={pipeline} images={len(image_paths)} "
        f"device={device} seed={resolved_seed} full_dump={full_dump_enabled}"
    )
    print(f"[paths] vae_config={pathlib.Path(args.config).resolve()}")
    print(f"[paths] vae_weights={pathlib.Path(args.weights).resolve()}")
    if pipeline == "patch":
        print(f"[paths] patch_weights={pathlib.Path(args.patch_weights).resolve()}")
    print(f"[paths] output_dir={output_dir.resolve()}")

    writer = None
    tb_dir: pathlib.Path | None = None
    if args.tb:
        if _TB_AVAILABLE:
            tb_dir = output_dir / "tb"
            writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"[tb] SummaryWriter -> {tb_dir}")
        else:
            print("[tb] SummaryWriter 不可用，跳过 TB")

    if full_dump_enabled:
        dump_indices = set(range(len(image_paths)))
    else:
        dump_indices = set()
    all_metrics: List[ReconMetrics] = []
    perline_path = output_dir / "metrics_per_image.jsonl"

    load_model(first_stage_model, device)
    try:
        with perline_path.open("w", encoding="utf-8") as f:
            for idx, image_path in enumerate(image_paths):
                metrics, visuals = process_image(
                    image_path=image_path,
                    first_stage_model=first_stage_model,
                    patch_model=patch_model,
                    cfg=cfg,
                    args=args,
                    device=device,
                    pipeline=pipeline,
                )
                all_metrics.append(metrics)
                f.write(json.dumps(asdict(metrics), ensure_ascii=False) + "\n")

                if writer is not None:
                    writer.add_scalar("loss/vae_pixel_mse", metrics.vae_pixel_mse, idx)
                    writer.add_scalar("loss/vae_pixel_l1", metrics.vae_pixel_l1, idx)
                    writer.add_scalar("metric/vae_psnr", metrics.vae_psnr, idx)
                    if metrics.patch_latent_mse is not None:
                        writer.add_scalar("loss/patch_latent_mse", metrics.patch_latent_mse, idx)
                        writer.add_scalar("loss/patch_latent_l1", metrics.patch_latent_l1, idx)
                        writer.add_scalar("loss/patch_pixel_mse", metrics.patch_pixel_mse, idx)
                        writer.add_scalar("loss/patch_pixel_l1", metrics.patch_pixel_l1, idx)
                        writer.add_scalar("metric/patch_psnr", metrics.patch_psnr, idx)

                if idx in dump_indices:
                    write_visual_case(output_dir, idx, image_path, metrics, visuals)
                    if writer is not None:
                        grid_path = output_dir / "visual_cases" / f"{idx:04d}_{pathlib.Path(image_path).stem}" / "compare.png"
                        grid = pil_to_unit_tensor(Image.open(grid_path))
                        writer.add_image(f"samples/{idx:04d}_{pathlib.Path(image_path).stem}", grid, 0)

                if (idx + 1) % max(1, args.log_every) == 0:
                    print(f"[progress] {idx + 1}/{len(image_paths)}")

        summary = {
            "_metric_doc": METRIC_DOC,
            "config": vars(args),
            "resolved_pipeline": pipeline,
            "resolved_seed": resolved_seed,
            "sample_limit": sample_limit,
            "full_dump": full_dump_enabled,
            "dump_count": len(dump_indices),
            "num_images": len(all_metrics),
            "output_dir": str(output_dir.resolve()),
        }
        summary["overall"] = aggregate_metrics(all_metrics)
        summary["aggregate"] = summary["overall"]
        (output_dir / "loss_doc.json").write_text(json.dumps(METRIC_DOC, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        if writer is not None:
            for key, value in summary["overall"].items():
                writer.add_scalar(f"summary/{key}", value, 0)
    finally:
        unload_model(first_stage_model)
        if writer is not None:
            writer.close()

    print(f"[done] 逐图 loss={perline_path}")
    print(f"[done] 汇总 loss={output_dir / 'metrics_summary.json'}")
    print(f"[done] loss 字段说明={output_dir / 'loss_doc.json'}")
    if full_dump_enabled:
        print(f"[done] case 图像目录={output_dir / 'visual_cases'}（含 compare.png）")
    if tb_dir is not None:
        print(f"[done] TensorBoard 目录={output_dir / 'tb'}")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"[done] outputs -> {output_dir.resolve()}")


if __name__ == "__main__":
    main()
