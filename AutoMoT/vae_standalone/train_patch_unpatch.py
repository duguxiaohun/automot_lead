"""端到端训练 patch / unpatch：image → VAE.encode → patch → unpatch → VAE.decode → image'。

设计：
- VAE encoder + decoder 全程冻结（FrozenVAE.__init__ 已经 requires_grad=False）；
  encoder 走 no_grad 节省显存，decoder 必须保留梯度链让 pixel loss 能反传到 unpatch / patch。
- 唯一可训练的两个模块是 Patchify 和 Unpatchify，直接 import 自
  ``qwen3vl_local.goalgen.dit``。保证 state_dict key 与 DiTMoT 里 ``self.patch`` /
  ``self.unpatch`` 一一对应（``patch.proj.weight`` / ``unpatch.proj.weight`` 等），
  训练产物可以被 DiTMoT.load_patch_unpatch 直接吃下。
- 数据复用 GoalGen v1 的 jsonl：每条样本读 history + current + target 共 6 张 RGB；
  patch/unpatch 不区分时序，把 6 帧当 batch=6 处理。
- 主 loss = pixel MSE（image_hat vs image）；--lambda-latent > 0 时可叠加 latent MSE
  作为辅助监督，默认 0 表示用户问需求时确认的"端到端图像重建"目标。

运行目录：AutoMoT/

运行（单卡）：默认强制挑 1 张最空闲 GPU**覆盖**已有的 CUDA_VISIBLE_DEVICES。
确实要手动锁定 GPU 时 `PATCH_UNPATCH_RESPECT_CUDA_VISIBLE_DEVICES=1` 关掉自动覆盖。::

    python vae_standalone/train_patch_unpatch.py \
        --train-jsonl checkpoints/goalgen_v1_data/train.jsonl \
        --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
        --output-dir checkpoints/patch_unpatch_v1

DDP：torchrun 启动后每个 worker 已经持有独立 CUDA 上下文，**.py 内部无法再换卡**
（CUDA driver 已初始化，修改 CUDA_VISIBLE_DEVICES 无效），必须由 launcher 提前
挑好。下面这段一行 shell 自动挑 4 张最闲 GPU 再交给 torchrun::

    export CUDA_VISIBLE_DEVICES=$(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n | head -n 4 | awk -F',' '{print $1}' | paste -sd, -
    )
    torchrun --standalone --nproc_per_node=4 \
        vae_standalone/train_patch_unpatch.py \
        --train-jsonl checkpoints/goalgen_v1_data/train.jsonl \
        --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
        --output-dir checkpoints/patch_unpatch_v1

训练产物（顶层）::

    OUTPUT_DIR/
      weights/
        patch_unpatch_latest.safetensors    # 每个 epoch 末覆盖
        patch_unpatch_epoch{NNN}.safetensors# 每个 epoch 末单独存
        patch_unpatch_best.safetensors      # val/pixel_mse 历史最小
        patch_unpatch_best.json             # {"step": ..., "val_pixel_mse": ...}
      tb/                                   # TensorBoard events
      invocations/                          # 每次启动 rank0 写一份
                                            # <ts>_<host>_pid<pid>.txt（sys.argv +
                                            # 关键 env + git_commit），事后追溯
                                            # "这版 ckpt 是哪条命令训出来的"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import subprocess
import sys
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image


_THIS_FILE = pathlib.Path(__file__).resolve()
# vae_standalone/train_patch_unpatch.py -> AutoMoT -> automot_lead
_AUTOMOT_ROOT = _THIS_FILE.parents[1]
_PROJECT_ROOT = _THIS_FILE.parents[2]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# 这两个模块在 DiTMoT 里就是 self.patch / self.unpatch。直接复用，state_dict key
# 自然兼容；如果换成自己重写一份，未来 DiT 改 patch 卷积参数就两边漂移。
from qwen3vl_local.goalgen.dit import Patchify, Unpatchify  # noqa: E402
from qwen3vl_local.goalgen.vae import FrozenVAE, default_vae_paths  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: E402
    _TB_AVAILABLE = True
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False


# --------------------------------------------------------------------------- #
# 模型：把 Patchify + Unpatchify 串成一个最小自编码器壳
# --------------------------------------------------------------------------- #


class PatchUnpatchAutoencoder(torch.nn.Module):
    """latent z → tokens → latent z_hat 的可逆 patch 嵌入。

    state_dict 命名约定（**与 DiTMoT 兼容的关键**）::

        patch.proj.weight   / patch.proj.bias
        unpatch.proj.weight / unpatch.proj.bias

    这套 key 跟 DiTMoT 里 ``self.patch`` / ``self.unpatch`` 的 state_dict 完全一致；
    DiTMoT.load_patch_unpatch 直接 filter 这四个 key 喂给 ``self.load_state_dict``
    即可，无需任何 rename 或 strip 前缀。
    """

    def __init__(self, latent_channels: int, hidden_dim: int, patch_size: int):
        super().__init__()
        self.patch = Patchify(latent_channels, hidden_dim, patch_size)
        self.unpatch = Unpatchify(hidden_dim, latent_channels, patch_size)
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        tokens, grid = self.patch(z)
        return self.unpatch(tokens, grid)


# --------------------------------------------------------------------------- #
# DDP / 数据小工具（保持单文件、零外部依赖）
# --------------------------------------------------------------------------- #


def pick_idle_gpus(want_count: int) -> str:
    """用 nvidia-smi 按显存占用、GPU 利用率从低到高挑卡。"""

    if want_count <= 0:
        return ""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""

    rows: List[tuple[int, int, int]] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpu_idx = int(parts[0])
            mem_used = int(parts[1])
            util = int(parts[2])
        except ValueError:
            continue
        rows.append((mem_used, util, gpu_idx))
    rows.sort()
    return ",".join(str(gpu_idx) for _, _, gpu_idx in rows[:want_count])


def auto_configure_cuda_visible_devices(want_count: int) -> str:
    """挑 want_count 张最空闲 GPU 并 export CUDA_VISIBLE_DEVICES。

    单卡（want_count=1）：**默认覆盖**外部已设的 CUDA_VISIBLE_DEVICES，强制挑最闲。
        显式 ``PATCH_UNPATCH_RESPECT_CUDA_VISIBLE_DEVICES=1`` opt-out。
    DDP（want_count>1）：在 .py 里改 CUDA_VISIBLE_DEVICES 无效（torchrun 启动每个
        worker 时 CUDA driver 已初始化，visible set 固化）。所以这里只在"外部完全没
        设 CUDA_VISIBLE_DEVICES"的兜底情况下 pick；建议用户在 torchrun 前 export 好。
    """

    respect = os.environ.get("PATCH_UNPATCH_RESPECT_CUDA_VISIBLE_DEVICES", "0") == "1"

    if respect and "CUDA_VISIBLE_DEVICES" in os.environ:
        return f"preset-respect:{os.environ['CUDA_VISIBLE_DEVICES']}"

    if want_count > 1 and "CUDA_VISIBLE_DEVICES" in os.environ:
        # DDP 场景下尊重 launcher 已挑好的，不在 .py 里覆盖（CUDA 已初始化改不动）。
        return f"ddp-preset:{os.environ['CUDA_VISIBLE_DEVICES']}"

    selected = pick_idle_gpus(want_count)
    if selected:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        picked = len([x for x in selected.split(",") if x.strip()])
        if picked < want_count:
            return f"auto-partial:{selected}"
        # 标记 source 类型：单卡覆盖 vs 单卡新挑 vs DDP 兜底
        prefix = "auto-override" if want_count == 1 and "CUDA_VISIBLE_DEVICES" in os.environ else "auto"
        return f"{prefix}:{selected}"
    return "auto-unavailable"


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    gpu_source = auto_configure_cuda_visible_devices(world_size)
    os.environ["PATCH_UNPATCH_GPU_SOURCE"] = gpu_source
    if world_size > 1:
        if torch.cuda.is_available() and torch.cuda.device_count() < world_size:
            raise RuntimeError(
                f"DDP 需要 {world_size} 张可见 GPU，但当前只看到 {torch.cuda.device_count()} 张；"
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
            )
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def _dump_invocation(output_dir: pathlib.Path, rank: int = 0) -> None:
    """把 sys.argv + 关键 env vars + 元信息写到 ``output_dir/invocations/<ts>_<host>_pid<pid>.txt``。

    只 rank0 写；失败不阻塞训练（缺 git / IO 错误等都吞掉只打印一行警告）。
    事后想"这版 ckpt 是哪条命令跑的"直接 cat 就够，不用回翻 shell history。
    """

    if rank != 0:
        return
    try:
        import datetime as _dt
        import platform as _platform
        import shlex as _shlex
        import socket as _socket
        import subprocess as _subprocess

        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        host = _socket.gethostname()
        inv_dir = output_dir / "invocations"
        inv_dir.mkdir(parents=True, exist_ok=True)
        out_path = inv_dir / f"{ts}_{host}_pid{os.getpid()}.txt"

        env_keys = (
            "CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "RANK", "LOCAL_RANK",
            "MASTER_ADDR", "MASTER_PORT", "NCCL_DEBUG", "NCCL_P2P_LEVEL",
            "PYTORCH_CUDA_ALLOC_CONF",
            "GOALGEN_COMPILE_DIT", "GOALGEN_CUDNN_BENCHMARK",
            "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        )
        try:
            git = _subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(pathlib.Path(__file__).resolve().parent),
                capture_output=True, text=True, timeout=5,
            )
            git_commit = git.stdout.strip() if git.returncode == 0 else "<unavailable>"
        except Exception:
            git_commit = "<unavailable>"

        lines = [
            f"# saved at {ts}",
            f"# hostname = {host}",
            f"# pid = {os.getpid()}",
            f"# python = {sys.version.split()[0]}",
            f"# torch = {getattr(torch, '__version__', '<unknown>')}",
            f"# platform = {_platform.platform()}",
            f"# git_commit = {git_commit}",
            "",
            "# ---- selected env vars ----",
            *[f"{k}={os.environ.get(k, '<unset>')}" for k in env_keys],
            "",
            "# ---- sys.argv (one per line) ----",
            *sys.argv,
            "",
            "# ---- shell replay ----",
            " ".join(_shlex.quote(a) for a in sys.argv),
        ]
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[invocation] saved -> {out_path}")
    except Exception as exc:
        print(f"[invocation] 保存失败（不阻塞）：{exc}")


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rgb(path: str) -> Image.Image:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"RGB 不存在: {p}")
    img = Image.open(p)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def collect_image_paths(sample: Dict[str, Any]) -> List[str]:
    """history(默认 4) + current(1) + target(1) = 6 张 RGB。

    current_rgb_path 字段在 goalgen v1 jsonl 里始终存在；若上游格式调整缺字段，
    跳过它（仍然有 history + target 可用），保留对旧数据集的兼容。
    """

    paths: List[str] = list(sample["history_rgb_paths"])
    cur = sample.get("current_rgb_path")
    if cur:
        paths.append(cur)
    paths.append(sample["target_rgb_path"])
    return paths


# --------------------------------------------------------------------------- #
# VAE 前向：encode 走 no_grad，decode 必须留梯度
# --------------------------------------------------------------------------- #


def vae_encode_no_grad(vae: FrozenVAE, images: List[Image.Image]) -> torch.Tensor:
    """与 FrozenVAE.encode 等价，但显式 no_grad 节省显存。

    返回 scaled latent（乘过 scale_factor），**不**应用 latent_stats 归一化。
    patch/unpatch 学的是"对 scaled latent 的可逆映射"，归一化只是线性缩放，加不
    加都不影响重构能力；不加可以让训练时和 DiT 端"未启用 latent stats"路径完全
    对齐 dtype/数值范围，省一道 stats 加载逻辑。
    """

    with torch.no_grad():
        x = vae.pil_to_tensor(images)
        with torch.autocast("cuda", enabled=(not vae.cfg.disable_autocast and vae.device.type == "cuda")):
            z = vae.model.encode(x)
        return z * vae.cfg.scale_factor


def vae_decode_with_grad(vae: FrozenVAE, z: torch.Tensor) -> torch.Tensor:
    """FrozenVAE.decode 是 @torch.no_grad() 装饰的，这里复制等价逻辑但保留梯度。

    VAE 参数本身是 requires_grad=False（FrozenVAE 已冻结），所以 backward 不会
    更新 VAE；但梯度链必须穿过 decode 才能让 image-MSE 反传到 unpatch / patch。
    """

    # dtype 对齐到 vae.dtype（默认 fp32）：decode 的第一层 Conv2d 在 fp32 权重上，
    # z 若是 bf16 直接进去会 dtype mismatch；.to 是可微的不会断梯度，与 bf16
    # 训练 patch/unpatch 共存无问题。
    z_in = z.to(device=vae.device, dtype=vae.dtype) / vae.cfg.scale_factor
    # 与 FrozenVAE.decode 同样的 VideoDecoder 适配；不放到模块顶层 import 是为了
    # 避免 sys.path 未注入时静态导入失败。
    from vwm.modules.autoencoding.temporal_ae import VideoDecoder  # noqa: E402
    if isinstance(vae.model.decoder, VideoDecoder):
        kwargs: Dict[str, Any] = {"timesteps": z_in.shape[0]}
    else:
        kwargs = {}
    with torch.autocast("cuda", enabled=(not vae.cfg.disable_autocast and vae.device.type == "cuda")):
        return vae.model.decode(z_in, **kwargs)


# --------------------------------------------------------------------------- #
# 保存 / 调度器
# --------------------------------------------------------------------------- #


def save_weights(model: torch.nn.Module, path: pathlib.Path) -> None:
    """把 patch + unpatch 的 state_dict 保存为 safetensors。

    DDP 包装后的 .module 才是真模型；不解包会写入 ``module.patch.proj.weight``
    这种带前缀的 key，DiTMoT 加载时会 missing。
    """

    from safetensors.torch import save_file  # noqa: E402
    module = model.module if hasattr(model, "module") else model
    sd = {k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path))


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# 验证
# --------------------------------------------------------------------------- #


@torch.no_grad()
def run_val_pass(
    model: torch.nn.Module,
    vae: FrozenVAE,
    val_samples: List[Dict[str, Any]],
    dtype: torch.dtype,
    max_samples: int,
) -> Dict[str, float]:
    """跑验证集计算 pixel_mse / latent_mse / PSNR(dB)。

    PSNR 按"输入归一到 [-1,1]"的动态范围 2 算：psnr = -10 log10(mse / 4)。
    数据范围跟 FrozenVAE.pil_to_tensor 内部归一化保持一致。
    """

    model.eval()
    pixel_mses: List[float] = []
    latent_mses: List[float] = []
    psnrs: List[float] = []
    take = min(max_samples, len(val_samples)) if max_samples > 0 else len(val_samples)
    for sample in val_samples[:take]:
        images = [load_rgb(p) for p in collect_image_paths(sample)]
        z = vae_encode_no_grad(vae, images)
        z_hat = model(z.to(dtype=dtype))
        x = vae.pil_to_tensor(images)
        x_hat = vae_decode_with_grad(vae, z_hat).clamp(-1.0, 1.0)
        pixel_mse = float(F.mse_loss(x_hat.float(), x.float()).item())
        latent_mse = float(F.mse_loss(z_hat.float(), z.float()).item())
        psnr = -10.0 * math.log10(max(pixel_mse / 4.0, 1e-12))
        pixel_mses.append(pixel_mse)
        latent_mses.append(latent_mse)
        psnrs.append(psnr)
    model.train()
    n = max(1, len(pixel_mses))
    return {
        "val/pixel_mse": sum(pixel_mses) / n,
        "val/latent_mse": sum(latent_mses) / n,
        "val/psnr_db": sum(psnrs) / n,
    }


@torch.no_grad()
def log_image_samples(
    writer: Any,
    model: torch.nn.Module,
    vae: FrozenVAE,
    val_samples: List[Dict[str, Any]],
    dtype: torch.dtype,
    step: int,
    num_samples: int,
) -> None:
    """落几条 val 样本的 (原图, 重建图) 对到 TensorBoard。"""

    if writer is None or not val_samples or num_samples <= 0:
        return
    model.eval()
    take = min(num_samples, len(val_samples))
    grid: List[torch.Tensor] = []
    for sample in val_samples[:take]:
        # 这里每条样本只用 target 一帧做可视化，足够直观；history 全用上图像太多
        # 浪费 TB 渲染时间。
        img = load_rgb(sample["target_rgb_path"])
        z = vae_encode_no_grad(vae, [img]).to(dtype=dtype)
        z_hat = model(z)
        x = vae.pil_to_tensor([img]).clamp(-1.0, 1.0)
        x_hat = vae_decode_with_grad(vae, z_hat).clamp(-1.0, 1.0)
        # 归一化到 [0,1] 给 TB add_images；并排堆 [原图, 重建] 两张。
        x01 = ((x + 1.0) / 2.0).float().cpu()
        x_hat01 = ((x_hat + 1.0) / 2.0).float().cpu()
        grid.append(x01[0])
        grid.append(x_hat01[0])
    if grid:
        writer.add_images("samples/orig_vs_recon", torch.stack(grid, dim=0), step, dataformats="NCHW")
    model.train()


# --------------------------------------------------------------------------- #
# 主训练循环
# --------------------------------------------------------------------------- #


def train(args: argparse.Namespace) -> None:
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("训练 patch/unpatch 需要 CUDA")
    if is_rank0(rank):
        print(
            "[gpu] "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
            f"source={os.environ.get('PATCH_UNPATCH_GPU_SOURCE', '<unknown>')} "
            f"world_size={world_size} local_rank={local_rank} device={device}"
        )

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    output_dir = pathlib.Path(args.output_dir)
    _dump_invocation(output_dir, rank=rank)
    weights_dir = output_dir / "weights"
    if is_rank0(rank):
        weights_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(pathlib.Path(args.train_jsonl))
    if not samples:
        raise RuntimeError(f"训练 jsonl 为空: {args.train_jsonl}")

    val_samples: List[Dict[str, Any]] = []
    if is_rank0(rank) and args.val_jsonl:
        vp = pathlib.Path(args.val_jsonl)
        if vp.exists():
            val_samples = load_jsonl(vp)[: max(0, args.val_max_samples)]
            print(f"[data] 验证样本 {len(val_samples)} 来源 {vp}")
        else:
            print(f"[data] 警告：验证 jsonl 不存在 {vp}，跳过验证")

    if is_rank0(rank):
        print(f"[data] 训练样本 {len(samples)} world_size={world_size} dtype={args.dtype}")

    writer = None
    if is_rank0(rank) and args.tb and _TB_AVAILABLE:
        tb_dir = output_dir / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"[tb] SummaryWriter -> {tb_dir}")
    elif is_rank0(rank) and args.tb and not _TB_AVAILABLE:
        print("[tb] 警告：SummaryWriter 不可用，TB 关闭")

    vae_cfg_path, vae_weights_path = default_vae_paths()
    vae = FrozenVAE.load(
        config_path=vae_cfg_path,
        weights_path=vae_weights_path,
        device=str(device),
        dtype=args.vae_dtype,
    )
    vae.model.eval()
    # FrozenVAE.__init__ 已 requires_grad=False，这里再做一道防御性 freeze：
    # 如果未来有人改 vae.py 把"参数解冻"作为开关，本脚本仍然只训 patch/unpatch。
    for pp in vae.model.parameters():
        pp.requires_grad_(False)

    model = PatchUnpatchAutoencoder(
        latent_channels=args.latent_channels,
        hidden_dim=args.hidden_dim,
        patch_size=args.patch_size,
    ).to(device=device, dtype=dtype)

    if world_size > 1:
        # 这个壳里没有"条件分支只走部分参数"的情况，所以 find_unused_parameters=False
        # 既能跑又略省 hook 开销；与 train_v1.py（DiT 带 null KV 分支）不同。
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    usable = (len(samples) // world_size) * world_size
    if usable <= 0:
        raise RuntimeError(f"数据集太小 {len(samples)} 不足 world_size={world_size}")
    steps_per_epoch = max(1, math.ceil((usable / world_size) / args.grad_accum_steps))
    total_steps = max(1, steps_per_epoch * args.num_epochs)
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_ratio)

    if is_rank0(rank):
        n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] 可训练参数 {n_train_params} (patch+unpatch)；总 step={total_steps}")

    global_step = 0
    running_pixel = 0.0
    running_latent = 0.0
    running_micro = 0
    best_pixel = float("inf")

    module = model.module if hasattr(model, "module") else model

    try:
        for epoch in range(args.num_epochs):
            order = list(range(len(samples)))
            # 同 train_v1.py 思路：所有进程用同一 seed 产生相同 order，再按 rank 步长切片，
            # 保证不重叠不遗漏。
            random.Random(args.seed + epoch).shuffle(order)
            order = order[:usable]
            shard = order[rank::world_size]

            for local_idx, sample_idx in enumerate(shard):
                sample = samples[sample_idx]
                image_paths = collect_image_paths(sample)
                n_frames = len(image_paths)

                # 逐帧 forward+backward：把"单条 sample 当 batch=N"拆成 N 个 sub-step，
                # 一次只让 1 帧的 VAE decode activation 驻留在显存，**显存峰值 ÷ N**。
                # 数学等价：F.mse_loss 默认 reduction='mean'；一次 forward N 帧的 loss
                #   = mean over (N * C * H * W) 个元素
                # 拆开后 N 次 sub-step 累 (1/N) 的 grad，sum 之后
                #   = (1/N) * sum_i mean over (1 * C * H * W) 个元素
                # 两者完全相等（mean over N 帧的整体 = N 帧 per-frame mean 的平均）。
                # 这是 train_patch_unpatch.py 在 H20 95GB 上从 OOM 到能跑的关键改动。
                sample_pixel_mse = 0.0
                sample_latent_mse = 0.0
                for img_path in image_paths:
                    img = load_rgb(img_path)
                    # encode 单张 → [1, C, h, w]；no_grad 释放 encoder activation。
                    z_i = vae_encode_no_grad(vae, [img]).to(dtype=dtype)
                    # 前向：z → patch → unpatch → z_hat
                    z_hat_i = model(z_i)
                    latent_mse_i = F.mse_loss(z_hat_i, z_i.to(dtype=z_hat_i.dtype))
                    # pixel 重建：z_hat → VAE.decode → x_hat；与原图 x 比 MSE。
                    # 这是主目标：让 patch/unpatch 在 latent 空间的微小损失，
                    # 不会被解码放大成可见图像差异。
                    x_i = vae.pil_to_tensor([img])
                    x_hat_i = vae_decode_with_grad(vae, z_hat_i).clamp(-1.0, 1.0)
                    pixel_mse_i = F.mse_loss(x_hat_i, x_i.to(dtype=x_hat_i.dtype))
                    # 1/n_frames 让 N 帧梯度求和等价于一次 forward 整 batch 的 mean。
                    loss_i = (args.lambda_pixel * pixel_mse_i + args.lambda_latent * latent_mse_i) / n_frames
                    # 再除 grad_accum_steps 把 N micro 的梯度规模和"无累积"对齐。
                    (loss_i / args.grad_accum_steps).backward()
                    sample_pixel_mse += float(pixel_mse_i.detach().item())
                    sample_latent_mse += float(latent_mse_i.detach().item())

                # 累计单条 sample 的 N 帧平均（与原代码一次 forward 整 batch 的
                # pixel_mse 数值含义一致），供 logging 用。
                running_pixel += sample_pixel_mse / n_frames
                running_latent += sample_latent_mse / n_frames
                running_micro += 1

                # 边界判定：当前 micro 是 accum 组里最后一条，或 shard 已扫完。
                micro_pos = (local_idx % args.grad_accum_steps) + 1
                shard_remaining = len(shard) - local_idx - 1
                is_last_of_group = (micro_pos == args.grad_accum_steps) or (shard_remaining == 0)

                if is_last_of_group:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in module.parameters() if p.requires_grad],
                            args.max_grad_norm,
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if is_rank0(rank) and global_step % args.logging_steps == 0:
                        n = max(1, running_micro)
                        avg_pixel = running_pixel / n
                        avg_latent = running_latent / n
                        lr = optimizer.param_groups[0]["lr"]
                        print(
                            f"[step {global_step}/{total_steps}] "
                            f"pixel_mse={avg_pixel:.4f} latent_mse={avg_latent:.4f} lr={lr:.2e}"
                        )
                        if writer is not None:
                            writer.add_scalar("train/pixel_mse", avg_pixel, global_step)
                            writer.add_scalar("train/latent_mse", avg_latent, global_step)
                            writer.add_scalar("train/lr", lr, global_step)
                        running_pixel = 0.0
                        running_latent = 0.0
                        running_micro = 0

                    # 验证 + best 跟踪（只 rank0）。
                    if (
                        is_rank0(rank)
                        and args.val_steps > 0
                        and val_samples
                        and global_step % args.val_steps == 0
                    ):
                        metrics = run_val_pass(module, vae, val_samples, dtype, args.val_max_samples)
                        print(f"[val step {global_step}] {metrics}")
                        if writer is not None:
                            for k, v in metrics.items():
                                writer.add_scalar(k, v, global_step)
                            log_image_samples(
                                writer, module, vae, val_samples, dtype, global_step,
                                num_samples=args.image_log_samples,
                            )
                        val_pixel = metrics["val/pixel_mse"]
                        if math.isfinite(val_pixel) and val_pixel < best_pixel:
                            best_pixel = val_pixel
                            save_weights(model, weights_dir / "patch_unpatch_best.safetensors")
                            (weights_dir / "patch_unpatch_best.json").write_text(
                                json.dumps(
                                    {"step": global_step, "val_pixel_mse": float(val_pixel)},
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                            print(f"[best] 更新 pixel_mse={best_pixel:.6f} @ step {global_step}")

                    if global_step >= total_steps:
                        break

            # epoch 末统一保存 latest + epoch-N。
            if is_rank0(rank):
                save_weights(model, weights_dir / f"patch_unpatch_epoch{epoch:03d}.safetensors")
                save_weights(model, weights_dir / "patch_unpatch_latest.safetensors")
                print(f"[epoch {epoch}] 保存 latest + epoch{epoch:03d} -> {weights_dir}")

            if global_step >= total_steps:
                break

        # check 模式（max_train_steps 很小）下，循环外再兜底写一次 latest。
        if is_rank0(rank):
            save_weights(model, weights_dir / "patch_unpatch_latest.safetensors")
            print(f"[final] step={global_step} 已写 patch_unpatch_latest.safetensors")

    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train patch/unpatch as a latent autoencoder for GoalGen DiT.")
    p.add_argument("--train-jsonl", required=True, help="GoalGen v1 build_dataset 输出的 train.jsonl")
    p.add_argument("--val-jsonl", default="", help="可选；非空且文件存在时按 --val-steps 评估")
    p.add_argument("--output-dir", required=True, help="权重 + TB 输出根目录")

    # 必须和 DiTMoTConfig 默认值一致；改这里时 DiT 端要同步。
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=768)
    p.add_argument("--patch-size", type=int, default=2)

    # epoch 数：831k 样本 × 5 帧 ≈ 4.1M frame-step / epoch；4 GPU × accum 2 =
    # 等效 batch 8 → ~520k optimizer step / epoch。patch/unpatch 是俩小卷积/线性
    # 层，1 epoch 已经远远过收敛；2 epoch 留一点 cosine decay 尾段做精修。
    # 想要更多就显式 --num-epochs 3 之类，没必要默认就 5（会浪费几天 wall-time）。
    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-train-steps", type=int, default=0,
                   help="0 = 按 num_epochs；>0 时直接夹到该 step 数，常用于 check。")
    p.add_argument("--seed", type=int, default=20260601)

    p.add_argument("--lambda-pixel", type=float, default=1.0,
                   help="image MSE 主 loss 权重；默认 1.0 对应用户需求的端到端图像重建。")
    p.add_argument("--lambda-latent", type=float, default=0.0,
                   help="latent MSE 辅助权重；默认 0 表示纯像素监督。设 0.1 可作为正则。")

    p.add_argument("--vae-dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                   help="VAE 权重精度；保留默认 fp32 与 vae_only.yaml 的 disable_autocast=true 一致。")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"],
                   help="patch/unpatch 的训练精度。")

    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--val-steps", type=int, default=200, help="每多少 optimizer step 跑一次 val；0 关闭。")
    p.add_argument("--val-max-samples", type=int, default=32)
    p.add_argument("--image-log-samples", type=int, default=4,
                   help="每次验证落几条 (原图, 重建) TB 图像；0 关闭。")
    p.add_argument("--tb", action="store_true", default=True)
    p.add_argument("--no-tb", dest="tb", action="store_false")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
