"""使用 build_dataset.py 生成的 jsonl 训练 GoalGen v1 DiT-MoT。

这个训练入口刻意保持小而直白：
- Qwen3-VL-Instruct 全程冻结，只用于 teacher-forced 预填充。
- VAE 全程冻结，只用于历史帧 / 目标帧潜变量编码。
- DiT-MoT 是唯一可训练模块。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import shutil
import sys
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image


_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from qwen3vl_local.engine import LocalQwen3VLInstructEngine  # noqa: E402
from qwen3vl_local.goalgen.dit import (  # noqa: E402
    DiTMoT,
    DiTMoTConfig,
)
from qwen3vl_local.goalgen.flow import (  # noqa: E402
    DiTEMA,
    euler_sample_cfg,
    flow_matching_loss,
    sample_flow_batch,
)
from qwen3vl_local.goalgen.qwen_kv import teacher_forced_prefill  # noqa: E402
from qwen3vl_local.goalgen.vae import FrozenVAE, default_vae_paths  # noqa: E402
from qwen3vl_local.prompt_pipeline import DrivingMemory  # noqa: E402

# 延迟到运行时再 import：训练机一定有 tb（pytorch 自带），但本地静态检查时可能没装；
# 即便 TensorBoard 导入失败也不应该挂掉整个训练器，留 `--no-tb` 作为兜底。
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: E402
    _TB_AVAILABLE = True
except Exception:  # pragma: no cover - 运行时缺包才会进
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False


def setup_distributed() -> tuple[int, int, int]:
    # 走 torchrun / accelerate 启动时这三个变量由 launcher 注入；直接读 env 而不是
    # argparse 是为了让单卡 / DDP 共用一份脚本——单卡时三个值都默认 0/1，下面
    # if world_size > 1 分支不触发，调用方无需写两套代码路径。
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        # nccl 是 GPU-GPU all-reduce 后端；gloo 在多 GPU 上慢一个数量级，DiT
        # 这种 bf16 大量同步的工作量必须用 nccl。
        dist.init_process_group(backend="nccl")
        # 必须在 init_process_group 之后立刻 set_device，否则后续 torch.cuda.xxx
        # 默认落到 cuda:0，多个进程会抢同一张卡然后挂死。
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    # is_available + is_initialized 双重保护：单卡跑（没 init）也调用这个函数也安全，
    # 不会丢出 "Default process group not initialized" 异常。
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
        raise FileNotFoundError(f"RGB 图像不存在：{p}")
    img = Image.open(p)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def memory_from_sample(sample: Dict[str, Any]) -> DrivingMemory:
    memory = sample["memory"]
    return DrivingMemory(
        scenario=memory["scenario"],
        scenario_label=memory.get("scenario_label", memory["scenario"]),
        event_sequence=tuple(memory["event_sequence"]),
        status=memory["status"],
        subgoal=memory["subgoal"],
        completed_events=list(memory.get("completed_events", [])),
    )


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def build_dit(args: argparse.Namespace) -> DiTMoT:
    """构造 DiT-MoT（v2 架构）。

    v2 起 DiT 的 (n_heads, head_dim) 必须严格等于 Qwen 的 (num_key_value_heads, head_dim)，
    所以**不再需要 language_kv_input_dim 这一字段**。默认 hidden_dim=1024 / n_heads=8 /
    head_dim=128 已经对齐 Qwen3-VL-4B-Instruct；想接其它 Qwen 时调 --hidden-dim /
    --n-heads 保持二者乘积等于 Qwen K/V 总维度即可。

    实际 K/V 形状是否真匹配在 DiTMoT.forward 第一个 step 内做严格断言（pooled_kv[0] 形状），
    出错时报错路径直接，不需要额外的 probe。
    """

    cfg = DiTMoTConfig(
        latent_channels=4,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        num_layers=args.num_layers,
        cond_dim=args.cond_dim,
        max_history_frames=args.max_history_frames,
    )
    return DiTMoT(cfg)


def freeze_module(module: torch.nn.Module) -> None:
    """显式关掉所有 requires_grad 并切到 eval；用于 Qwen / VAE 这种冻结大模型。

    虽然 optimizer 不传它们的参数 + 前向走 no_grad 已经足够安全，这里仍然加一道
    保险：如果未来有 agent 不小心在 Qwen 上挂 hook / 拿 .parameters() 喂 optimizer，
    会立刻发现是被冻结的。
    """

    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


# --------------------------------------------------------------------------- #
# Muon optimizer：对 2D 权重矩阵走 Newton-Schulz 正交化的 momentum。
# --------------------------------------------------------------------------- #


@torch.no_grad()
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """5 步 Newton-Schulz 迭代，把 G 投到与 G 同左奇异空间的"半正交"矩阵。

    数学含义：返回值 ≈ U @ V.T（G = U Σ V.T 的 SVD 简化形式），相当于把
    G 的奇异值统统拉到 ~1，只保留方向。这正是 Muon 的核心 trick——SGD-momentum
    输出做正交化后，每个奇异方向都用相同步长更新，避开了 AdamW 那种"看着 m/v
    自适应、其实在不同方向 LR 漂移"的问题。

    Implementation notes：
    - 系数 (a, b, c) = (3.4445, -4.7750, 2.0315) 来自 Keller Jordan 的 NanoGPT
      speedrun 实现；这组系数比经典 (3, -3, 1) 在小步数下收敛更快。
    - 计算用 bfloat16 做：Newton-Schulz 对量级不敏感，bf16 已足够；省一半显存。
    - 横长矩阵（行 > 列）先转置：减少中间 X @ X.T 的内存，最后再转回来。
    """

    assert G.ndim == 2, f"Muon Newton-Schulz 仅适用 2D 矩阵，got {G.shape}"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    # 归一化：除以 Frobenius 范数让 X 的奇异值都 ≤ 1，是 Newton-Schulz 收敛区间的前提。
    # +1e-7 防御零梯度（warmup 早期 + 大量稀疏 grad 时会出现）。
    X = X / (X.norm() + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        # 经典 5 步形式：A = X X.T；B = b A + c A^2；X ← a X + B X
        # 等价于多项式 p(X X.T) X，把奇异值 σ 推向 1（p(σ²) σ ≈ 1）。
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer：2D 权重矩阵专用，对 SGD-momentum 输出做 Newton-Schulz 正交化。

    用法（与 AdamW 配套）：
        muon_params  = [p for n, p in dit.named_parameters() if p.ndim == 2 and p.requires_grad]
        other_params = [p for n, p in dit.named_parameters() if p.ndim != 2 and p.requires_grad]
        muon  = Muon(muon_params, lr=0.02, momentum=0.95, weight_decay=0.0)
        adamw = AdamW(other_params, lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)
        # train loop:
        muon.step(); adamw.step()
        muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)

    Caveats：
    - **只接受 2D 张量**：Conv2d 权重（4D）、Embedding（2D 但语义不同）、norm 的 1D weight
      应该交给 AdamW，不要塞进 Muon。本实现对 ndim≠2 直接抛错。
    - LR 通常比 AdamW 大 5-10×（典型 0.01-0.05），weight_decay=0 起步；不要照搬 AdamW 配方。
    - 单卡正确；DDP 下每个 rank 自己跑 Newton-Schulz 在 grad sync 后是等价的（grad 已 all-reduce），
      不需要额外的同步逻辑。
    - 配合 `torch.compile` 可能在 ns 迭代上失败：本实现走 ``@torch.no_grad`` + 手动 op，
      compile 的 dynamo trace 不必进 Muon.step，所以兼容性 OK。

    参考：Keller Jordan, "Muon: An optimizer for hidden layers in neural networks",
    https://github.com/KellerJordan/Muon
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError(f"Muon lr 必须 > 0，got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Muon momentum ∈ [0,1)，got {momentum}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    # 严格校验：Muon 设计上只跑 2D；4D Conv 或 1D norm 进来肯定是 param group
                    # 分组出错，直接报错而不是默默跳过（默默跳过会让人误以为 Muon 在更新这些参数）。
                    raise RuntimeError(
                        f"Muon 仅接受 2D 张量，但收到 ndim={p.ndim} shape={tuple(p.shape)}。"
                        "请检查 param group 分组：把 Conv2d / Embedding / 1D 参数交给 AdamW。"
                    )
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                # 经典 momentum：buf = momentum * buf + g
                buf.mul_(momentum).add_(g)
                # Nesterov 在 buf 基础上再前瞻一步：update = g + momentum * buf
                # 收敛更快、过冲更轻；Keller Jordan 实现默认开启。
                update = g.add(buf, alpha=momentum) if nesterov else buf

                # Newton-Schulz 正交化：把 update 推到与之对齐的"半正交"矩阵。
                ortho = _zeropower_via_newtonschulz5(update, steps=ns_steps)

                # 缩放：保证不同 shape 的层有可比的 update 量级。
                # 高个矩阵（rows > cols）的"半正交"自带较大量级，需要 sqrt(rows/cols) 缩放。
                rows, cols = p.shape
                scale = max(1.0, rows / cols) ** 0.5

                # 解耦 weight decay：仿 AdamW，wd 直接乘 p，与 grad 路径独立；
                # 默认 weight_decay=0（对 attention 权重通常不加 wd）。
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)
                p.data.add_(ortho, alpha=-lr * scale)
        return loss


def split_dit_params_for_muon(
    dit_module: torch.nn.Module,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """把 DiT 参数分成 (muon_2d, adamw_other) 两组。

    分组规则：
    - 2D 权重（attention q/k/v/o、MLP gate/up/down、AdaLN modulation Linear、
      t_mlp 的 Linear、unpatch.proj 的 Linear）-> Muon
    - 其它（Conv2d patch.proj、所有 4D 张量、所有 1D weight 与 bias、embeddings、
      pos_embed_table、null_lang_k/v）-> AdamW
    - requires_grad=False 的参数（patch/unpatch 冻结时）跳过

    返回的两组合并起来应该等于 dit_module.parameters() 里所有可训参数，无遗漏。
    """

    muon_params: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []
    for name, p in dit_module.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            muon_params.append(p)
        else:
            other_params.append(p)
    return muon_params, other_params


class _DualOptimizer:
    """轻量包装：把 Muon + AdamW 当一个 optimizer 用。

    设计目标：让训练 loop 里 `optimizer.step()` / `optimizer.zero_grad()` /
    `optimizer.state_dict()` 都不用改动；LambdaLR scheduler 也能直接走，
    因为 ``.param_groups`` 暴露的是两边 group 的合集，每个 group 都有自己的 base_lr。

    不继承 ``torch.optim.Optimizer``：那要求传入 params 走 super().__init__，
    而我们这里两边各有独立的 Optimizer 子实例，套继承反而别扭。

    save_checkpoint 走 state_dict() 时返回 ``{"muon": ..., "adamw": ...}``，
    load 时按相同结构反向加载。
    """

    def __init__(self, muon: torch.optim.Optimizer, adamw: torch.optim.Optimizer):
        self.muon = muon
        self.adamw = adamw

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        # 顺序：先 Muon 再 AdamW；LambdaLR 会按这个顺序逐 group 缩放 LR，与 base_lr
        # 在 Optimizer 子实例里的设置一致，无需特殊处理。
        return self.muon.param_groups + self.adamw.param_groups

    def step(self, closure=None) -> None:
        # 注意：两边 step 顺序不影响最终结果（各自独立的参数子集）。
        self.muon.step(closure=None)
        self.adamw.step(closure=None)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        # 旧 ckpt（v1 单 AdamW）没有这个分组结构；提示用户重新训练，不做兼容性 hack。
        if "muon" not in sd or "adamw" not in sd:
            raise RuntimeError(
                "optimizer state_dict 缺少 muon/adamw 键；这通常是想 resume v1 单 AdamW ckpt。"
                "v2 架构改动后参数 shape 与 v1 不兼容，必须从头训练。"
            )
        self.muon.load_state_dict(sd["muon"])
        self.adamw.load_state_dict(sd["adamw"])


def cosine_velocity(v_pred: torch.Tensor, v_target: torch.Tensor) -> float:
    # .detach() 切计算图防误传梯度；.float() 把 bf16 / fp16 升回 fp32 再算余弦，
    # 否则低精度下 sum_x2 容易溢出/下溢，dot 值会被截断到 0 或 inf。
    pred = v_pred.detach().float().flatten(1)
    target = v_target.detach().float().flatten(1)
    # flatten(1) 把 [B, C, H, W] 摊成 [B, C*H*W]，沿 dim=1 算余弦相似度，再 batch 平均。
    # 这个指标比损失更直观：训练健康时 cos 应该从 ~0 单调升到 ~0.5+，损失反映得没这么明显。
    return float(F.cosine_similarity(pred, target, dim=1).mean().item())


def _latent_stats_path(args: argparse.Namespace) -> pathlib.Path:
    if args.latent_stats_path:
        return pathlib.Path(args.latent_stats_path)
    return pathlib.Path(args.train_jsonl).parent / "latent_stats.json"


@torch.no_grad()
def _compute_latent_stats(
    vae: FrozenVAE,
    samples: List[Dict[str, Any]],
    max_samples: int,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    """Compute per-channel mean/std on raw scaled VAE latents.

    We scan history frames and target frames from the first N jsonl rows. This
    keeps the cache cheap while covering both conditioning latents and target
    latents with the same distribution used by DiT.

    多卡时按 ``samples[rank::world_size]`` 跳取分片（与训练 loop 的分片方式一致：
    均匀分散对 NFS 缓存更友好），各 rank 独立累计本地 sum/sumsq/count，最后通过
    ``dist.all_reduce(SUM)`` 跨进程汇总。这跟单卡跑 take 个 sample 数学等价，
    wall-time ÷ world_size，杜绝了"rank0 串行算 1000 次 VAE encode，其它 rank 干等"。
    """

    take = len(samples) if max_samples <= 0 else min(max_samples, len(samples))
    # 用 device 上的 float64 张量做累加，方便 all_reduce 直接归约。
    sum_c = torch.zeros(4, dtype=torch.float64, device=vae.device)
    sumsq_c = torch.zeros(4, dtype=torch.float64, device=vae.device)
    pixel_count_t = torch.zeros(1, dtype=torch.float64, device=vae.device)
    count_t = torch.zeros(1, dtype=torch.float64, device=vae.device)

    # rank::world_size 跳取：每个 sample_idx 只被一个 rank 处理；4 卡时 rank0 拿
    # [0,4,8,...]，rank1 拿 [1,5,9,...]，对相邻 run 的 NFS 缓存命中模式更友好。
    shard_indices = list(range(take))[rank::world_size]
    for idx in shard_indices:
        sample = samples[idx]
        paths = list(sample["history_rgb_paths"]) + [sample["target_rgb_path"]]
        images = [load_rgb(p) for p in paths]
        z = vae.encode_raw(images).detach().double()
        sum_c += z.sum(dim=(0, 2, 3))
        sumsq_c += z.pow(2).sum(dim=(0, 2, 3))
        pixel_count_t += float(z.shape[0] * z.shape[2] * z.shape[3])
        count_t += float(len(images))

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        # 跨 rank 汇总各自分片的统计量。SUM 后正好 = 全 take 的总和，
        # 再算 mean/std 与单卡数学等价。pixel_count 和 count 也走 SUM。
        dist.all_reduce(sum_c, op=dist.ReduceOp.SUM)
        dist.all_reduce(sumsq_c, op=dist.ReduceOp.SUM)
        dist.all_reduce(pixel_count_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)

    pixel_count = int(pixel_count_t.item())
    if pixel_count <= 0:
        raise RuntimeError("latent stats 统计失败：没有可用 latent")

    mean = sum_c / pixel_count
    var = (sumsq_c / pixel_count) - mean.pow(2)
    std = var.clamp_min(1e-12).sqrt()
    return {
        "mean": [float(x) for x in mean.detach().cpu()],
        "std": [float(x) for x in std.detach().cpu()],
        "num_jsonl_samples": take,
        "num_images": int(count_t.item()),
        "space": "scaled_vae_latent",
    }


def _load_or_compute_latent_stats(
    vae: FrozenVAE,
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> Dict[str, Any]:
    stats_path = _latent_stats_path(args)

    # cache 命中判断在 rank0 做，再 broadcast 给其它 rank。
    # 避免 rank0 觉得没命中（在算），rank1 觉得命中（去 load 半成品 / 空文件）。
    cache_hit = False
    if is_rank0(rank):
        cache_hit = stats_path.exists() and not args.recompute_latent_stats
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        flag = torch.tensor([1 if cache_hit else 0], dtype=torch.int32, device=vae.device)
        dist.broadcast(flag, src=0)
        cache_hit = bool(flag.item())

    if cache_hit:
        # 所有 rank 都直接读 cache（json 文件几 KB，比 broadcast tensor 简单）。
        with stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)
        if is_rank0(rank):
            print(f"[latent_stats] loaded {stats_path}")
    else:
        if is_rank0(rank):
            print(
                f"[latent_stats] computing from first {args.latent_stats_max_samples} "
                f"samples (sharded across {world_size} ranks) -> {stats_path}"
            )
        # 所有 rank 一起跑分片计算 + all_reduce 汇总；wall-time ÷ world_size。
        stats = _compute_latent_stats(
            vae, samples, args.latent_stats_max_samples,
            rank=rank, world_size=world_size,
        )
        if is_rank0(rank):
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with stats_path.open("w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"[latent_stats] mean={stats['mean']} std={stats['std']}")
        # 等 rank0 写完文件再让其它 rank 离开，避免后续步骤竞争未落盘的文件。
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()

    vae.load_latent_stats_dict(stats)
    return stats


def _make_z_init_from_prior(
    z_history: torch.Tensor,
    shape: tuple[int, int, int, int],
    device: torch.device,
    dtype: torch.dtype,
    alpha: float,
    sigma: float,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    z_prior = z_history[:, -1].to(device=device, dtype=dtype)
    return alpha * z_prior + sigma * noise


# --------------------------------------------------------------------------- #
# TensorBoard 辅助函数
# --------------------------------------------------------------------------- #


def _decode_latent_to_image(vae: FrozenVAE, z: torch.Tensor) -> torch.Tensor:
    """Latent → [0,1] 范围 RGB 张量 [B, 3, H, W]，给 tb writer.add_image / add_images 用。

    VAE 解码默认输出 [-1,1]（与训练输入归一化一致），tb 渲染要 [0,1]，所以这里 +1 /2。
    clamp 防止偶发数值出 [-1,1] 把渲染搞糊。
    """

    # 显式转到 vae 自己的 (device, dtype) 作为防御层：
    # 训练 dit_dtype=bf16 而 vae_dtype=fp32 时不转直接喂 vae.decode 会撞 dtype mismatch。
    # FrozenVAE.decode 内部也做了同样的 cast（vae.py），这里再做一次防止未来有人重构
    # 把 vae 内部那道保险删掉时下游悄悄崩，符合"训练主路径绝不能被 image sample 拖崩"的原则。
    z = z.to(device=vae.device, dtype=vae.dtype)
    decoded = vae.decode(z)
    decoded = decoded.clamp(-1.0, 1.0)
    return ((decoded + 1.0) / 2.0).float().cpu()


@torch.no_grad()
def _run_val_pass(
    engine: LocalQwen3VLInstructEngine,
    vae: FrozenVAE,
    dit_module: torch.nn.Module,
    val_samples: List[Dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    dit_dtype: torch.dtype,
) -> Dict[str, float]:
    """跑一小撮验证样本，只算前向损失 / cos，不做反传 / Euler 采样。

    设计取舍：
    - 只在 0 号进程调，所以传 dit_module（DDP 解包后的裸模型），无需跨卡归约；
    - 每条样本走完整 prefill + encode 流程，慢但语义忠实；
    - 取 val_max_samples 上限避免每次验证时间失控。
    """

    dit_module.eval()
    losses: List[float] = []
    cosines: List[float] = []
    for sample in val_samples:
        history_images = [load_rgb(p) for p in sample["history_rgb_paths"]]
        target_img = load_rgb(sample["target_rgb_path"])
        memory = memory_from_sample(sample)

        prefill = teacher_forced_prefill(
            engine=engine,
            memory=memory,
            images=history_images,
            num_segments=args.num_layers,
            kv_segment_mode=args.qwen_kv_segment_mode,
        )
        pooled_kv = [
            (k.to(device=device, dtype=dit_dtype), v.to(device=device, dtype=dit_dtype))
            for k, v in prefill.pooled_kv
        ]
        z_history = vae.encode(history_images).to(dtype=dit_dtype).unsqueeze(0)
        z1 = vae.encode([target_img]).to(dtype=dit_dtype)
        batch = sample_flow_batch(
            z1=z1,
            z_prior=z_history[:, -1],
            alpha=args.z0_prior_alpha,
            sigma=args.z0_prior_sigma,
            t_sampler=args.t_sampler,
            t_logit_mean=args.t_logit_mean,
            t_logit_std=args.t_logit_std,
        )
        v_pred = dit_module(batch.z_t, z_history, batch.t, pooled_kv)
        losses.append(float(flow_matching_loss(v_pred, batch.v_target).item()))
        cosines.append(cosine_velocity(v_pred, batch.v_target))

    dit_module.train()
    n = max(1, len(losses))
    return {"val/loss": sum(losses) / n, "val/cos": sum(cosines) / n}


@torch.no_grad()
def _log_image_samples(
    writer: Any,
    engine: LocalQwen3VLInstructEngine,
    vae: FrozenVAE,
    dit_module: torch.nn.Module,
    val_samples: List[Dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    dit_dtype: torch.dtype,
    step: int,
) -> None:
    """每 image-log-every 步写一组预测 / 真值图像并排到 TensorBoard。

    image-log-samples 条样本各跑一次 Euler 采样（用 image-log-euler-steps 步数），
    再通过 VAE 解码得到预测 RGB；与真值关键帧直接读盘后归一化对齐做并排比较。
    完整数据流和 eval 一致，所以这里看到的图就是模型当前的"生成能力快照"。
    """

    if writer is None or not val_samples:
        return
    dit_module.eval()
    pred_imgs: List[torch.Tensor] = []
    gt_imgs: List[torch.Tensor] = []
    take = min(args.image_log_samples, len(val_samples))
    for sample in val_samples[:take]:
        history_images = [load_rgb(p) for p in sample["history_rgb_paths"]]
        target_img = load_rgb(sample["target_rgb_path"])
        memory = memory_from_sample(sample)

        prefill = teacher_forced_prefill(
            engine=engine,
            memory=memory,
            images=history_images,
            num_segments=args.num_layers,
            kv_segment_mode=args.qwen_kv_segment_mode,
        )
        pooled_kv = [
            (k.to(device=device, dtype=dit_dtype), v.to(device=device, dtype=dit_dtype))
            for k, v in prefill.pooled_kv
        ]
        z_history = vae.encode(history_images).to(dtype=dit_dtype).unsqueeze(0)
        z1_gt = vae.encode([target_img]).to(dtype=dit_dtype)

        # euler 起点 z0 用固定 seed 让"同一 step 同一样本"复现相同图像，
        # 跨 step 跨样本随机性仍然存在，便于观察模型而不是观察噪声。
        gen = torch.Generator(device=device).manual_seed(args.seed + step + 1)
        z_init = _make_z_init_from_prior(
            z_history=z_history,
            shape=tuple(z1_gt.shape),
            device=device,
            dtype=dit_dtype,
            alpha=args.z0_prior_alpha,
            sigma=args.z0_prior_sigma,
            generator=gen,
        )
        z1_pred = euler_sample_cfg(
            dit=dit_module,
            z_history=z_history,
            pooled_kv=pooled_kv,
            shape=tuple(z1_gt.shape),
            device=device,
            dtype=dit_dtype,
            num_steps=args.image_log_euler_steps,
            cfg_scale=args.cfg_scale,
            z_init=z_init,
        )
        pred_imgs.append(_decode_latent_to_image(vae, z1_pred)[0])
        gt_imgs.append(_decode_latent_to_image(vae, z1_gt)[0])

    if pred_imgs:
        # 交错排：pred_0, gt_0, pred_1, gt_1, ... 直接靠 TensorBoard 行列布局对比；
        # 这种布局比 [all_pred, all_gt] 二段式更便于人眼"对一对一"比对。
        interleaved = []
        for p, g in zip(pred_imgs, gt_imgs):
            interleaved.append(p)
            interleaved.append(g)
        grid = torch.stack(interleaved, dim=0)
        writer.add_images("samples/pred_vs_gt", grid, step, dataformats="NCHW")

    dit_module.train()


def save_checkpoint(
    output_dir: pathlib.Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    ema: DiTEMA,
    latent_stats: Dict[str, Any],
    step: int,
    args: argparse.Namespace,
    name_prefix: str = "checkpoint-",
) -> None:
    """保存训练 ckpt。

    name_prefix 控制目录名前缀：
    - 默认 "checkpoint-"：epoch 末快照，参与 --keep-recent-checkpoints 滚动。
    - "step-checkpoint-"：训练中按 --step-save-every 触发的 step 快照，参与
      --keep-recent-step-checkpoints 滚动，与 epoch ckpt 互不淘汰。
    两套命名共用同一份序列化逻辑，避免双重维护；eval.py 的 _infer_ckpt_step
    会同时识别两种前缀。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    # step 用 :06d 0-padding 让目录名按字典序自然递增，方便 `ls | sort` 拿最新 ckpt；
    # 如果不 pad，"checkpoint-9" 会排在 "checkpoint-10" 后面。
    target = output_dir / f"{name_prefix}{step:06d}"
    target.mkdir(parents=True, exist_ok=True)
    # 兼容 DDP 包过的模型：DDP 会把真模型放在 .module 下；裸模型直接用自己。
    # 不解包就会 save 进 "module.xxx" 前缀的 state_dict，再加载到单卡时 key 对不上。
    module = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "step": step,
            "dit_state_dict": module.state_dict(),
            # optimizer / scheduler state 用于 resume 训练（AdamW 的 m/v、cosine 进度）；
            # 不存就只能从头训。仅落最新一个 step，磁盘成本可控。
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            # dit_config 直接 dump 配置 dataclass：恢复时不依赖代码默认值漂移，
            # 例如以后改了默认 hidden_dim，旧 ckpt 仍能按存档的配置正确重建模型。
            "dit_config": asdict(module.cfg),
            "ema_state_dict": ema.state_dict(),
            "ema_decay": ema.decay,
            "latent_stats": latent_stats,
            "args": vars(args),
        },
        target / "goalgen_v1.pt",
    )
    # latest.pt 是"轻量版本"：只存权重 + 配置，不存优化器 / 调度器，
    # 给下游推理 / 评测用，避免每次都拖一份几百 MB 的 AdamW 状态。
    latest = output_dir / "latest.pt"
    torch.save(
        {
            "step": step,
            "dit_state_dict": module.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "ema_decay": ema.decay,
            "dit_config": asdict(module.cfg),
            "latent_stats": latent_stats,
            "args": vars(args),
        },
        latest,
    )


def _prune_old_checkpoints(
    output_dir: pathlib.Path,
    keep: int,
    name_prefix: str = "checkpoint-",
) -> None:
    """保留最新 keep 个 {name_prefix}XXXXXX/，更老的整体删除。

    best.pt / latest.pt 都在 OUTPUT_DIR 顶层，不在 checkpoint-* 模式里，所以
    不会被这个 prune 误删；即使 best 对应的 step 早已被淘汰，best.pt 文件仍然存在。

    name_prefix:
    - "checkpoint-"      ：epoch 末池（受 --keep-recent-checkpoints 控制）。
    - "step-checkpoint-" ：step 池（受 --keep-recent-step-checkpoints 控制）。
    glob 用 f"{name_prefix}*"，所以两个池互不污染：epoch prune 不会动到 step ckpt，
    反之亦然。注意 step-checkpoint-* 也匹配 checkpoint-*，所以 epoch 池的 glob
    必须显式排除 step- 前缀，下面做了过滤。
    """

    if keep <= 0:
        return
    raw = [p for p in output_dir.glob(f"{name_prefix}*") if p.is_dir()]
    if name_prefix == "checkpoint-":
        # "checkpoint-*" 也会匹配 "step-checkpoint-*"，显式过滤掉 step 池；否则
        # epoch prune 会误删 step ckpt（用户明确要求两池独立）。
        raw = [p for p in raw if not p.name.startswith("step-checkpoint-")]
    ckpts = sorted(raw)
    if len(ckpts) <= keep:
        return
    for old in ckpts[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def _update_best_checkpoint(
    output_dir: pathlib.Path,
    val_loss: float,
    best_loss: float,
    step: int,
) -> Tuple[float, bool]:
    """若 val_loss 比 best_loss 更优，把当前 latest.pt 拷贝成 best.pt，并写 best.json 元信息。

    返回 (新 best_loss, 是否更新)。NaN / inf val_loss 视为无效，跳过。
    通过 copy 而不是 symlink：保证最新 N 个 ckpt 被滚动淘汰后 best.pt 仍能独立存活。
    """

    if not math.isfinite(val_loss) or val_loss >= best_loss:
        return best_loss, False
    src = output_dir / "latest.pt"
    if not src.exists():
        return best_loss, False
    dst = output_dir / "best.pt"
    shutil.copyfile(str(src), str(dst))
    meta = output_dir / "best.json"
    meta.write_text(
        json.dumps({"step": step, "val_loss": float(val_loss)}, indent=2),
        encoding="utf-8",
    )
    return val_loss, True


def _cosine_warmup_lambda(total_steps: int, warmup_ratio: float):
    """共享的 LR lambda：warmup 线性 + 之后 cosine 衰减到 0。

    返回的乘子对所有 param_group 同样作用——Muon 和 AdamW 各自的 base_lr 由
    optimizer 本身设置，这里只产生统一的进度因子。
    """

    # max(1, ...) 防止 total_steps=0 或 warmup_ratio=0 时 warmup_steps 变 0，导致下面
    # 除零；warmup 至少跑一步在工程上无害，可避免 check 模式 total_steps=2 时崩溃。
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # 线性 warmup：从 1/warmup 缓慢爬到 1.0；step+1 起步，避免第一步 lr=0。
            # 用 LambdaLR 的好处是这里返回的是"对 base_lr 的乘子"，所以无需手算 lr。
            return float(step + 1) / float(warmup_steps)
        # progress ∈ [0, 1]；分母用 max(1, ...) 防 total_steps == warmup_steps 时除零。
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # min(1.0, progress) 是兜底：optimizer 可能多走几步（例如 save 时多 step 一下），
        # 不夹断的话 cos(>π) 会让 lr 反向爬上去，破坏收敛末期的稳定。
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return lr_lambda


class _DualScheduler:
    """同时驱动 Muon 与 AdamW 两个 LambdaLR；接口与单 LambdaLR 等价。

    每个底层 LambdaLR 都按自己 optimizer 的 base_lr 缩放同一份 lambda 因子；
    所以 `get_last_lr()` 默认返回两边的 LR 拼起来，方便日志区分。
    """

    def __init__(self, optimizer: _DualOptimizer, lr_lambda):
        self.muon_sched = torch.optim.lr_scheduler.LambdaLR(optimizer.muon, lr_lambda)
        self.adamw_sched = torch.optim.lr_scheduler.LambdaLR(optimizer.adamw, lr_lambda)

    def step(self) -> None:
        self.muon_sched.step()
        self.adamw_sched.step()

    def get_last_lr(self) -> List[float]:
        # 顺序：muon 在前、adamw 在后；与 _DualOptimizer.param_groups 顺序一致。
        # 训练日志只读 [0] 拿 Muon LR；想看 AdamW LR 翻列表后段或单独 print。
        return self.muon_sched.get_last_lr() + self.adamw_sched.get_last_lr()

    def state_dict(self) -> Dict[str, Any]:
        return {"muon": self.muon_sched.state_dict(), "adamw": self.adamw_sched.state_dict()}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if "muon" not in sd or "adamw" not in sd:
            raise RuntimeError(
                "scheduler state_dict 缺少 muon/adamw 键；v1 单 scheduler ckpt 不兼容 v2。"
            )
        self.muon_sched.load_state_dict(sd["muon"])
        self.adamw_sched.load_state_dict(sd["adamw"])


def make_scheduler(
    optimizer,
    total_steps: int,
    warmup_ratio: float,
):
    """构造调度器：单 optimizer 走 LambdaLR；_DualOptimizer 走 _DualScheduler。"""

    lr_lambda = _cosine_warmup_lambda(total_steps, warmup_ratio)
    if isinstance(optimizer, _DualOptimizer):
        return _DualScheduler(optimizer, lr_lambda)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args: argparse.Namespace) -> None:
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GoalGen 训练需要 CUDA；本地只适合跑数据构建，训练请放到远端机器。")

    # cuDNN benchmark：第一次见到 conv shape 时会同时探测多个 algorithm，每个都申请
    # workspace，**瞬时显存峰值可能高出稳态 10-30GB**。VAE 走 conv3d + 大 spatial 时
    # 尤其严重（实测在 H20 95GB 上能把 [latent_stats] 阶段直接 OOM）。
    # 默认改为关闭；如果你的 GPU 显存有大余量、想拿那 5-10% 速度，导出
    # GOALGEN_CUDNN_BENCHMARK=1 显式启用。
    if os.environ.get("GOALGEN_CUDNN_BENCHMARK", "0") == "1":
        torch.backends.cudnn.benchmark = True
        if is_rank0(rank):
            print("[cudnn] benchmark=True 启用（瞬时 workspace 峰值更高，注意 OOM 风险）")
    else:
        torch.backends.cudnn.benchmark = False

    # TF32 matmul：bf16/fp32 路径上启用 TensorCore 的 TF32 加速，精度损失可忽略，
    # 不影响显存峰值，无 OOM 风险，保留默认开启。
    torch.set_float32_matmul_precision("high")

    output_dir = pathlib.Path(args.output_dir)
    _dump_invocation(output_dir, rank=rank)
    samples = load_jsonl(pathlib.Path(args.train_jsonl))
    if not samples:
        raise RuntimeError(f"empty train jsonl: {args.train_jsonl}")

    # 验证集只在 0 号进程用（验证 + 图像样例仅 0 号进程跑），其它进程留空省 IO。
    val_samples: List[Dict[str, Any]] = []
    if is_rank0(rank) and args.val_jsonl:
        val_path = pathlib.Path(args.val_jsonl)
        if val_path.exists():
            val_samples = load_jsonl(val_path)[: max(0, args.val_max_samples)]
            print(f"[data] 验证样本={len(val_samples)}（上限={args.val_max_samples}）来源={val_path}")
        else:
            print(f"[data] 警告：验证 jsonl 不存在 ({val_path})，跳过验证/样例记录")

    if is_rank0(rank):
        print(f"[data] 训练样本={len(samples)} world_size={world_size}")

    # TensorBoard：只在 0 号进程起 writer，避免多进程写同一目录冲突。
    # 写到 output_dir/tb（与 ckpt 同根，按用户 5.1 选项）。
    writer = None
    if is_rank0(rank) and args.tb and _TB_AVAILABLE:
        tb_dir = output_dir / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"[tb] SummaryWriter -> {tb_dir}")
    elif is_rank0(rank) and args.tb and not _TB_AVAILABLE:
        print("[tb] 警告：SummaryWriter 导入失败，TensorBoard 关闭")

    engine = LocalQwen3VLInstructEngine(
        checkpoint_dir=pathlib.Path(args.checkpoint_dir).resolve(),
        device=str(device),
        torch_dtype=args.qwen_dtype,
        max_gen_tokens=0,
        temperature=0.0,
        do_sample=False,
        save_cache=False,
        cache_system_prompt=False,
    )
    engine.load()
    # 可选：挂上 SFT v1 训出来的 LoRA 适配器，让 GoalGen 预填充用"微调后的语言编码"。
    # merge=True 把 LoRA 权重合并进基础矩阵；之后 self.model 上无 PEFT 包装，KV 提取
    # / 段切分等下游代码完全无感知。--no-qwen-adapter-merge 可关闭合并保留 PeftModel。
    if args.qwen_adapter_dir:
        engine.attach_lora_adapter(args.qwen_adapter_dir, merge=args.qwen_adapter_merge)
    # 显式冻结 Qwen 全部参数。即便 optimizer 没传 Qwen 参数 + prefill 在 no_grad
    # 上下文里，这道保险能防止未来误用 .parameters() 把 Qwen 喂进 optimizer。
    # 这里在 attach 之后冻一次：merged 模型有自己的新参数视图，也要标 requires_grad=False。
    freeze_module(engine.model)

    vae_cfg, vae_weights = default_vae_paths()
    vae = FrozenVAE.load(
        config_path=vae_cfg,
        weights_path=vae_weights,
        device=str(device),
        dtype=args.vae_dtype,
    )
    # FrozenVAE.__init__ 已经 requires_grad_(False)，这里再确认 eval 一次。
    vae.model.eval()
    latent_stats = _load_or_compute_latent_stats(vae, samples, args, rank, world_size)

    # v2 起 DiT (n_heads, head_dim) 必须严格匹配 Qwen (n_kv_heads, head_dim)，
    # 不再需要运行时 probe；实际不匹配时 DiTMoT.forward 会在第一步 step 内抛清晰错误。
    # 把校验留在前向是为了让"换 Qwen 模型却忘改 --hidden-dim/--n-heads"的低级问题
    # 立刻被捕获，而不是在训练若干小时后在 attention 层 SDPA 内崩。
    dit_dtype = dtype_from_name(args.dit_dtype)
    dit = build_dit(args).to(device=device, dtype=dit_dtype)
    if is_rank0(rank):
        print(
            f"[build] DiT v2 cfg: hidden={args.hidden_dim} n_heads={args.n_heads} "
            f"head_dim={args.hidden_dim // args.n_heads} patch={args.patch_size} layers={args.num_layers}"
        )

    # 可选：加载 vae_standalone/train_patch_unpatch.py 训出来的 patch/unpatch 权重。
    # 默认 freeze=True，patch/unpatch 不再更新；optimizer 下面会按 requires_grad
    # 过滤参数，所以冻结后既省 AdamW state 也省反传计算。不提供路径就保持现状
    # （patch/unpatch 与 DiT 一起随机初始化联合训练）。
    if args.patch_unpatch_weights:
        info = dit.load_patch_unpatch(
            args.patch_unpatch_weights,
            freeze=not args.patch_unpatch_unfreeze,
        )
        if is_rank0(rank):
            print(f"[patch_unpatch] 加载 {args.patch_unpatch_weights}: {info}")
    elif is_rank0(rank):
        print("[patch_unpatch] 未提供权重 -> patch/unpatch 跟随 DiT 随机初始化训练")

    # warm start：在 patch/unpatch 加载之后、EMA 实例化之前先把 ckpt 的 dit_state_dict
    # copy 进当前 dit。次序很重要——先 patch_unpatch、再 warm-start dit、再建 EMA，
    # 这样 EMA 初始 shadow 自动等于 ckpt 里 dit 的权重，再被随后的 ema_state_dict 覆盖。
    # ckpt 里如果有 _orig_mod. / module. 前缀（DDP 或 compile wrap 后 save 的痕迹），
    # save_checkpoint 已 unwrap module.，compile 也用 default mode 不带前缀，所以
    # 这里直接 strict=True，让前缀异常显式抛错而不是吞掉。
    warm_ckpt_payload = None
    if args.init_from_ckpt:
        warm_ckpt_path = pathlib.Path(args.init_from_ckpt)
        if not warm_ckpt_path.is_file():
            raise FileNotFoundError(f"--init-from-ckpt 指向的文件不存在：{warm_ckpt_path}")
        # map_location="cpu"：先放 CPU，再 load_state_dict 时 PyTorch 自动 copy_ 到
        # dit 当前 device + dtype；避免 ckpt 的 device 与本进程 device 不一致导致额外搬运。
        warm_ckpt_payload = torch.load(warm_ckpt_path, map_location="cpu", weights_only=False)
        if "dit_state_dict" not in warm_ckpt_payload:
            raise KeyError(
                f"--init-from-ckpt {warm_ckpt_path} 缺少 dit_state_dict 字段；"
                f"该路径应指向 latest.pt / best.pt 这种轻量 ckpt。"
            )
        dit.load_state_dict(warm_ckpt_payload["dit_state_dict"], strict=True)
        if is_rank0(rank):
            src_step = warm_ckpt_payload.get("step", "?")
            print(
                f"[warm_start] DiT 权重已从 {warm_ckpt_path} 载入（src step={src_step}, strict=True）"
            )

    # 可选 gradient checkpointing：显存省 ~40%，wall-clock 多 ~30%。
    # 默认开（patch=4 后 token 数本就不多，启用 ckpt 几乎不影响速度但能塞更大 batch）；
    # 想压低单步耗时关 `--no-grad-ckpt` 即可。
    if args.grad_ckpt:
        dit.enable_gradient_checkpointing(True)
        if is_rank0(rank):
            print("[ckpt] gradient checkpointing 启用（per-block，use_reentrant=False）")

    # EMA 在 patch/unpatch 加载完之后初始化：让 EMA 起步快照 = 预训过的权重，
    # 而不是 build_dit 给的随机值；否则训练初期 EMA 推理图像会失真很久。
    ema = DiTEMA(dit, decay=args.ema_decay)
    # warm start 第二步：把 ckpt 里的 ema_state_dict 覆盖到刚建好的 EMA shadow。
    # 不接 EMA 时（warm_ckpt_payload 没这个 key 或用户只想 warm-start DiT）跳过——
    # 此时 EMA shadow = 当前 dit 权重，与 patch_unpatch 老语义一致。
    if warm_ckpt_payload is not None and "ema_state_dict" in warm_ckpt_payload:
        ema.load_state_dict(warm_ckpt_payload["ema_state_dict"], strict=True)
        if is_rank0(rank):
            print(
                f"[warm_start] EMA shadow 已从 {args.init_from_ckpt} 载入（strict=True）"
            )
    # 释放 ckpt 引用，避免 DDP wrap / optimizer 构建期间多占一份 host 内存。
    del warm_ckpt_payload
    if world_size > 1:
        # find_unused_parameters=True：DiT 的 null_lang_k / null_lang_v（24 个 Parameter）
        # 仅在 force_uncond=True 的 micro-step 被使用；cfg_drop_prob=0.1 时大多数
        # iteration 不会触发，DDP 默认模式会因为"上一次 reduction 未完成所有 param"而 raise。
        # 打开后 DDP 会每次 forward 遍历参数图找未使用 param，开销极小（<1%）。
        dit = torch.nn.parallel.DistributedDataParallel(
            dit, device_ids=[local_rank], find_unused_parameters=True
        )
        # 解包真模型供 EMA / 验证 / 图像样例旁路 DDP+compile 调用。
        dit_module = dit.module
    else:
        dit_module = dit

    # 可选 torch.compile(dit)：v2 起默认**开启**（patch=4 后 token 数砍到 1/4，
    # compile 的固定 overhead 比 v1 划算很多）。`--no-compile` 关掉作为退路。
    # - 只 compile DiT；Qwen3-VL 走 HF DynamicCache + Python 控制流不友好。
    # - mode="default" 用 Inductor 优化 attention/linear；fullgraph=False 容忍少量
    #   Python 分支（如 force_uncond），不强求一次性 graph 化。
    # - dynamic=True：pooled_kv 的 seq_len 跨 sample 会变，避免反复重 trace。
    # - 失败时回退原模型，不阻塞训练。
    if args.compile_dit:
        try:
            dit = torch.compile(dit, mode="default", fullgraph=False, dynamic=True)
            if is_rank0(rank):
                print("[compile] torch.compile(dit) 启用 (mode=default, dynamic=True)")
        except Exception as exc:
            if is_rank0(rank):
                print(f"[compile] torch.compile 失败，回退原模型：{exc}")

    # v2 双 optimizer：Muon 跑 2D 权重矩阵，AdamW 跑其它（Conv2d patch.proj、norm 1D weight、
    # embedding、null_lang_k/v 等）。Muon 在大型 attention 模型上比单 AdamW 收敛 1.5-2× 更快，
    # 但仅对 2D 矩阵有效——所以 1D / 4D 参数仍走 AdamW。
    # 注意分组取自 `dit_module`（DDP 解包后的真模型）：DDP wrap 不改参数引用，
    # 但 named_parameters() 名字会多出 "module." 前缀，分组逻辑只看 ndim 所以不受影响。
    muon_params, adamw_params = split_dit_params_for_muon(dit_module)
    if is_rank0(rank):
        n_muon = sum(p.numel() for p in muon_params)
        n_adamw = sum(p.numel() for p in adamw_params)
        print(
            f"[optim] Muon 接管 2D 权重: {len(muon_params)} 张, {n_muon/1e6:.2f}M 参数; "
            f"AdamW 接管其它: {len(adamw_params)} 张, {n_adamw/1e6:.2f}M 参数"
        )
    muon_optimizer = Muon(
        muon_params,
        lr=args.muon_lr,
        momentum=args.muon_momentum,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0,  # 2D 矩阵的 Muon 一般不挂 wd；wd 留给 AdamW 那条路径
    )
    adamw_optimizer = torch.optim.AdamW(
        adamw_params,
        lr=args.learning_rate,
        # betas=(0.9, 0.95) 是 DiT / 大型 diffusion 模型的常见配方；第二阶矩衰减比 Adam
        # 默认 0.999 快，对 latent flow matching 这种损失曲线较平的目标更稳。
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    # 用一个轻量包装把两个 optimizer 当成一个用：scheduler / step / zero_grad / state_dict
    # 都自动作用到两者。保留双 optimizer 各自独立调度的 LR 由 LambdaLR 自动按 base_lr 缩放。
    optimizer = _DualOptimizer(muon=muon_optimizer, adamw=adamw_optimizer)
    # 把样本数夹到 world_size 整除，让每个进程拿到等长分片；不夹断会出现"某进程多
    # backward 一次"，DDP all-reduce 等不到对应张量进而挂死。
    usable_per_epoch = (len(samples) // world_size) * world_size
    if usable_per_epoch <= 0:
        raise RuntimeError(f"数据集太小，不足以支撑 world_size={world_size}：当前只有 {len(samples)} 条样本")
    # 每个进程单个 epoch 的优化器步数 = 分片样本数 / grad_accum_steps，向上取整。
    # ceil 而不是 floor 是为了让"最后不满一个 accum 组也能 step 一次"，否则尾部样本
    # 算完梯度却不更新，浪费前向。
    steps_per_epoch = max(1, math.ceil((usable_per_epoch / world_size) / args.grad_accum_steps))
    total_steps = max(1, steps_per_epoch * args.num_epochs)
    if args.max_train_steps > 0:
        # CLI 给了硬上限就 clip；常用于 check 模式只跑两步快速验证而无需改 num_epochs。
        total_steps = min(total_steps, args.max_train_steps)
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_ratio)

    global_step = 0
    running_loss = 0.0
    running_cos = 0.0
    running_micro = 0
    running_kv_seq_len = 0  # 累计 Qwen prefill seq_len，做 logging 间均值
    last_grad_norm = 0.0    # 记录 clip 前 grad_norm，做 logging 间最近值
    accum = 0
    # best 跟踪：跨 epoch 保留 val/loss 历史最小值。无 val_samples 时永远不更新，
    # 也就不会产生 best.pt，eval.py 默认会 fallback 到 latest.pt。
    best_loss = float("inf")

    # dit_module 已在 DDP wrap 时定义（如果有 world_size>1 走 dit.module，否则裸 dit）；
    # 这里只复述用途：验证 / 图像样例用 dit_module 直接前向，旁路 DDP + torch.compile，
    # 既省跨卡归约，又避免 module.training=True 期间 dropout 等差异影响诊断。

    try:
        for epoch in range(args.num_epochs):
            order = list(range(len(samples)))
            # 每 epoch 用 seed+epoch 重洗：保证不同 epoch 见到的顺序不同（防止周期性过拟合），
            # 又保证同一份 seed + 同一台机器复现完全一致——所有进程用相同 seed 算出相同 order，
            # 才能保证下面 order[rank::world_size] 切出的分片互不重叠且无遗漏。
            random.Random(args.seed + epoch).shuffle(order)
            # 砍到 usable_per_epoch（world_size 整除）：避免尾部样本造成进程间分片长度差 1，
            # 那会让 DDP 在最后一个 step 卡死等不到对应张量的 all-reduce。
            order = order[:usable_per_epoch]
            # 步长 world_size 跳取：进程 0 拿 [0, W, 2W, ...]，进程 1 拿 [1, W+1, ...]，
            # 每个 sample_idx 只会被一个进程处理；比按连续块切分对 NFS 缓存更友好
            # （相邻进程的样本来自相邻 run 的概率低，分散读取反而能让磁盘并行加载）。
            shard = order[rank::world_size]

            for local_idx, sample_idx in enumerate(shard):
                sample = samples[sample_idx]
                # 同一张历史图片要喂 Qwen（作为 vision token）也要喂 VAE（作为 z_history），
                # 一次 load_rgb 复用避免读两遍盘；列表顺序与 jsonl 里 "history_rgb_paths" 一致，
                # 即"旧 → 新"，下面 VAE encode 和 DiT frame_embed 都依赖这个顺序。
                history_images = [load_rgb(p) for p in sample["history_rgb_paths"]]
                target_img = load_rgb(sample["target_rgb_path"])
                memory = memory_from_sample(sample)

                # ---- 计算 grad accum 的本地 micro 位置 ----
                # micro_pos: 在当前 accum 组内是第几条（1-based），用来判断要不要触发 optimizer.step
                # micro_start / micro_end: 当前 accum 组覆盖的 shard 索引范围；
                # 用 min(..., len(shard)) 处理尾部不满一个 accum 组的情况：尾巴的 group_size
                # 可能 < grad_accum_steps，此时 loss / micro_group_size 才能保持梯度尺度等价。
                micro_pos = (local_idx % args.grad_accum_steps) + 1
                micro_start = local_idx - (micro_pos - 1)
                micro_end = min(micro_start + args.grad_accum_steps, len(shard))
                micro_group_size = micro_end - micro_start
                will_step = micro_pos == micro_group_size

                # Qwen prefill：把 history 图像 + teacher-forced STATUS/SUBGOAL 真值塞进 Qwen，
                # 拿出 36 层 past_key_values 切 12 段。num_segments 必须 = DiT 层数，否则
                # DiT.forward 会在 zip(blocks, pooled_kv) 时静默错位（旧版会沉默，新版会抛错）。
                prefill = teacher_forced_prefill(
                    engine=engine,
                    memory=memory,
                    images=history_images,
                    num_segments=args.num_layers,
                    kv_segment_mode=args.qwen_kv_segment_mode,
                )
                # KV 来自 Qwen，dtype 可能是 bf16 / fp16；DiT 内部走 dit_dtype（默认 bf16）。
                # 显式 .to 强制对齐：不齐时 SDPA 会在 attention 内部 raise dtype mismatch，
                # 错误堆栈在 C++ 端不好定位，所以这里前向之前就把语言 KV 搬到目标 device + dtype。
                pooled_kv = [
                    (k.to(device=device, dtype=dit_dtype), v.to(device=device, dtype=dit_dtype))
                    for k, v in prefill.pooled_kv
                ]

                # VAE 默认 fp32 输出（vae_only.yaml 关了 autocast），.to(dit_dtype) 才能和 DiT 对齐。
                # .unsqueeze(0)：vae.encode 返回 [F, 4, 48, 144]（F 是历史帧数），加 batch 维变 [1, F, ...]，
                # DiT 的前向期望 [B, F, C, H, W]，这里 B=1（每进程 batch）。
                z_history = vae.encode(history_images).to(dtype=dit_dtype).unsqueeze(0)
                # 目标帧只有一张，encode 返回 [1, 4, 48, 144] 直接当作 z1（[B, C, H, W]），不需要再加维。
                z1 = vae.encode([target_img]).to(dtype=dit_dtype)
                # 在 z1 上采 z0 / t / 计算 z_t、v_target。z0 / t 留给 flow.py 内部默认采样，
                # 这里不传是为了让每条样本独立采，跟其他样本的随机性解耦。
                batch = sample_flow_batch(
                    z1=z1,
                    z_prior=z_history[:, -1],
                    alpha=args.z0_prior_alpha,
                    sigma=args.z0_prior_sigma,
                    t_sampler=args.t_sampler,
                    t_logit_mean=args.t_logit_mean,
                    t_logit_std=args.t_logit_std,
                )

                force_uncond = random.random() < args.cfg_drop_prob
                v_pred = dit(batch.z_t, z_history, batch.t, pooled_kv, force_uncond=force_uncond)
                loss_raw = flow_matching_loss(v_pred, batch.v_target)
                # 除以 micro_group_size：grad accum 时把 N 个 micro 的梯度加起来等价于 batch=N，
                # 但 PyTorch loss.backward() 是累加而不是平均，所以这里手动均一化，否则梯度尺度
                # 会随 grad_accum_steps 漂；尾部不满组时用动态 micro_group_size 保持等价。
                loss = loss_raw / micro_group_size

                # DDP 优化：grad accum 期间的 micro-step 不做 all-reduce，仅累积本地梯度；
                # 只在最后一个 micro-step 让 backward 触发 all-reduce 一次。
                # 单卡或非 grad-accum 边界以外用 nullcontext，行为与原版一致。
                if world_size > 1 and not will_step:
                    # dit.no_sync() 是 DDP 提供的 contextmanager：把 reduce hook 暂时摘掉，
                    # 反传仍照常累积本地 .grad 但不触发跨进程同步；总跨卡归约次数
                    # 从 N×grad_accum 降到 N，对 bandwidth 有显著收益。
                    sync_ctx = dit.no_sync()
                else:
                    sync_ctx = nullcontext()
                with sync_ctx:
                    loss.backward()

                # 累计统计量。用 detach + item() 解开计算图避免误持有 v_pred 引用；
                # 写成 += float(...) 比 .item() 直接更直观，且避免在低概率下触发 GPU sync。
                running_loss += float(loss_raw.detach().item())
                running_cos += cosine_velocity(v_pred, batch.v_target)
                running_micro += 1
                # prefill.seq_len 是这条样本 Qwen prefill 后的 token 数；监控这一项可以
                # 早期发现"prompt 不知不觉变长"导致 KV/显存膨胀，也能确认数据构建器没漏帧。
                running_kv_seq_len += int(prefill.seq_len)
                accum += 1

                if will_step:
                    if args.max_grad_norm > 0:
                        # clip_grad_norm_ 必须在 step 之前调；flow matching 初期 v_target 数值可能
                        # 很大（z1 - z0 在 latent 尺度上方差 ~2），不裁剪偶发会让 2e-4 学习率也炸。
                        # 返回值是 clip 前的全局梯度范数，写进 tb 用来诊断"训练是否在快炸"。
                        last_grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(dit.parameters(), args.max_grad_norm)
                        )
                    optimizer.step()
                    scheduler.step()
                    ema.update(dit_module)
                    # set_to_none=True 比 zero_(0) 快：让 .grad = None，下次 backward 第一次写
                    # 直接分配新张量，省一次"已分配张量清零"的 kernel。
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if is_rank0(rank) and global_step % args.logging_steps == 0:
                        denom = max(1, running_micro)
                        avg_loss = running_loss / denom
                        avg_cos = running_cos / denom
                        avg_kv_seq = running_kv_seq_len / denom
                        # _DualScheduler.get_last_lr() 返回 [muon_lr, ..., adamw_lr, ...]；
                        # 取首尾分别作 Muon / AdamW 代表 LR 写日志。单 scheduler 时两者一致。
                        last_lrs = scheduler.get_last_lr()
                        muon_lr_now = last_lrs[0]
                        adamw_lr_now = last_lrs[-1]
                        print(
                            f"[train] epoch={epoch} step={global_step}/{total_steps} "
                            f"loss={avg_loss:.6f} cos={avg_cos:.4f} "
                            f"grad_norm={last_grad_norm:.3f} kv_seq={avg_kv_seq:.0f} "
                            f"muon_lr={muon_lr_now:.3e} adamw_lr={adamw_lr_now:.3e}"
                        )
                        if writer is not None:
                            # 标量分组：train/* 用于训练曲线；diag/* 用于诊断指标。
                            # 这种命名让 tb 左侧 tag 树自动分组，不会被几十个指标淹没。
                            writer.add_scalar("train/loss", avg_loss, global_step)
                            writer.add_scalar("train/cos", avg_cos, global_step)
                            writer.add_scalar("train/muon_lr", muon_lr_now, global_step)
                            writer.add_scalar("train/adamw_lr", adamw_lr_now, global_step)
                            writer.add_scalar("diag/grad_norm", last_grad_norm, global_step)
                            writer.add_scalar("diag/kv_seq_len", avg_kv_seq, global_step)
                        running_loss = 0.0
                        running_cos = 0.0
                        running_micro = 0
                        running_kv_seq_len = 0

                    # 验证评估：每 val_steps 跑一次验证子集（仅 0 号进程，使用解包后的 dit_module）。
                    # 失败不阻断训练——验证集脏数据 / 单条显存溢出都先记录再继续。
                    if (
                        is_rank0(rank)
                        and writer is not None
                        and val_samples
                        and args.val_steps > 0
                        and global_step % args.val_steps == 0
                    ):
                        try:
                            with ema.apply_to(dit_module):
                                metrics = _run_val_pass(
                                    engine, vae, dit_module, val_samples,
                                    args, device, dit_dtype,
                                )
                            for tag, value in metrics.items():
                                writer.add_scalar(tag, value, global_step)
                            print(
                                f"[val] step={global_step} loss={metrics['val/loss']:.6f} "
                                f"cos={metrics['val/cos']:.4f}"
                            )
                        except Exception as e:
                            print(f"[val] 警告：验证前向出错，跳过：{e}")

                    # 图像样例：每 image_log_every 步生成一组预测 vs 真值落 TensorBoard，
                    # 是判断"损失在降但生成质量是否真的在改善"的关键可视化。
                    # 由于含一次 euler_sample（默认 32 步），频率不宜过高。
                    if (
                        is_rank0(rank)
                        and writer is not None
                        and val_samples
                        and args.image_log_every > 0
                        and global_step % args.image_log_every == 0
                    ):
                        try:
                            with ema.apply_to(dit_module):
                                _log_image_samples(
                                    writer, engine, vae, dit_module, val_samples,
                                    args, device, dit_dtype, global_step,
                                )
                        except Exception as e:
                            print(f"[image] 警告：图像样例生成失败，跳过：{e}")

                    # ---- step 级 ckpt：用户场景"数据量大、几天才跑完 1 epoch"，
                    # 单靠 epoch 末 save 拿不到中间产物。每 --step-save-every 步写一份
                    # step-checkpoint-NNNNNN/ 到独立池，--keep-recent-step-checkpoints
                    # 控制滚动数量（默认 3，即最近 30k 步）。命名前缀与 epoch ckpt 分开，
                    # 两个池互不淘汰对方；epoch 末逻辑完全不变。
                    if (
                        is_rank0(rank)
                        and args.step_save_every > 0
                        and global_step > 0
                        and global_step % args.step_save_every == 0
                    ):
                        save_checkpoint(
                            output_dir, dit, optimizer, scheduler, ema, latent_stats,
                            global_step, args,
                            name_prefix="step-checkpoint-",
                        )
                        _prune_old_checkpoints(
                            output_dir,
                            keep=args.keep_recent_step_checkpoints,
                            name_prefix="step-checkpoint-",
                        )
                        print(
                            f"[ckpt] step ckpt 已写 step-checkpoint-{global_step:06d}/ "
                            f"(keep={args.keep_recent_step_checkpoints})"
                        )

                    if global_step >= total_steps:
                        break
            if global_step >= total_steps:
                # max_train_steps 截断时跳出 epoch 循环；不在此 epoch 末做 save，
                # 完整保存交给循环外的 fallback 分支统一处理。
                break

            # ---- epoch 末：跑一次 val → 写 epoch ckpt → 滚动淘汰 → 比较 best ----
            # 与 step 内 val 完全独立：step 内 val 受 args.val_steps 控制（默认 500 step
            # 一次，是训练监控信号）；epoch 末 val 是 best 跟踪的判定信号，每个 epoch 末必跑一次。
            if is_rank0(rank):
                epoch_val_loss = math.nan
                if val_samples:
                    try:
                        with ema.apply_to(dit_module):
                            metrics = _run_val_pass(
                                engine, vae, dit_module, val_samples,
                                args, device, dit_dtype,
                            )
                        epoch_val_loss = float(metrics["val/loss"])
                        if writer is not None:
                            # epoch_end/ 子前缀避免和 step 内 val/* 标量混在同一曲线上。
                            for tag, value in metrics.items():
                                writer.add_scalar(f"epoch_end/{tag}", value, epoch + 1)
                        print(
                            f"[val][epoch={epoch+1}] loss={epoch_val_loss:.6f} "
                            f"cos={metrics.get('val/cos', float('nan')):.4f}"
                        )
                    except Exception as e:
                        print(f"[val] epoch={epoch+1} 验证失败，跳过 best 比较：{e}")

                save_checkpoint(
                    output_dir, dit, optimizer, scheduler, ema, latent_stats,
                    global_step, args,
                )
                _prune_old_checkpoints(output_dir, keep=args.keep_recent_checkpoints)
                best_loss, updated = _update_best_checkpoint(
                    output_dir, epoch_val_loss, best_loss, global_step,
                )
                if updated:
                    print(
                        f"[best] new best val/loss={best_loss:.6f} @ step={global_step} -> best.pt"
                    )

        # 兜底 save：epoch 中途因 max_train_steps 截断时，循环里不会写 ckpt；
        # 这里补一刀，附带滚动淘汰，让 OUTPUT_DIR 里至少有最新一份可用 ckpt。
        # 正常跑完 num_epochs 时，最后 epoch 末已经 save 过，本分支跳过避免重复写盘。
        if is_rank0(rank):
            tail_ckpt = output_dir / f"checkpoint-{global_step:06d}"
            if not tail_ckpt.exists():
                save_checkpoint(
                    output_dir, dit, optimizer, scheduler, ema, latent_stats,
                    global_step, args,
                )
                _prune_old_checkpoints(output_dir, keep=args.keep_recent_checkpoints)
            print(f"[done] GoalGen DiT 已保存到 {output_dir}")
    finally:
        # writer 必须在 cleanup_distributed 之前 close，避免 tb 写最后一笔时进程组已挂。
        if writer is not None:
            writer.close()
        cleanup_distributed()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="训练 GoalGen v1 DiT-MoT")
    p.add_argument("--train-jsonl", default="checkpoints/goalgen_v1_data/train.jsonl")
    p.add_argument("--checkpoint-dir", default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--output-dir", default="checkpoints/goalgen_v1_dit")
    p.add_argument("--qwen-dtype", choices=["bfloat16", "float16", "float32", "auto"], default="bfloat16")
    p.add_argument("--vae-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--dit-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    # LoRA / PEFT 适配器：空字符串表示不挂；否则在 engine.load() 之后 attach。
    # merge 默认开启，把 LoRA 合进基础矩阵，省预填充推理时间。
    p.add_argument("--qwen-adapter-dir", type=str, default="",
                   help="可选 LoRA / PEFT 适配器目录；为空则跑基础 Qwen。")
    p.add_argument("--qwen-adapter-merge", action="store_true", default=True,
                   help="挂适配器后立即 merge_and_unload；默认开。")
    p.add_argument("--no-qwen-adapter-merge", dest="qwen_adapter_merge", action="store_false",
                   help="保留 PeftModel 包装不合并（调试 LoRA 自身行为用）。")

    # v2 默认架构：patch=4 / hidden=1024 / n_heads=8 / head_dim=128
    # 与 Qwen3-VL-4B-Instruct 的 (num_key_value_heads=8, head_dim=128) 严格对齐，
    # 这样语言 K/V 直接接入 DiT attention，省掉 lang_k_proj/v_proj 跨维线性。
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=1024)
    # 可选 patch/unpatch 预训权重（来自 vae_standalone/train_patch_unpatch.py）。
    # 给路径就加载并默认冻结；不给就维持原行为（随机初始化、跟 DiT 一起训练）。
    # v2 注意：必须用 hidden=1024 / patch=4 默认重训的 safetensors，旧版 hidden=768 不兼容。
    p.add_argument("--patch-unpatch-weights", type=str, default="",
                   help='可选 patch_unpatch_*.safetensors 路径；非空时调用 '
                        'DiTMoT.load_patch_unpatch 加载并冻结。v2 起需 hidden=1024/patch=4 训出。')
    p.add_argument("--patch-unpatch-unfreeze", action="store_true", default=False,
                   help="加载 patch/unpatch 权重后仍允许联合更新（默认加载即冻结）。")
    # warm start：从 latest.pt / best.pt schema 的轻量 ckpt 里读 dit_state_dict +
    # ema_state_dict，作为本次训练的初始权重 / EMA shadow。**不**加载 optimizer /
    # scheduler / global_step——这是 warm start（接近 fine-tune），不是 resume。
    # 典型用法：v2 从 v1 best.pt 续训，让 DiT 起点已经是 v1 训好的参数。
    # strict=True：架构不一致直接报错，避免 hidden_dim / n_heads 改动后静默载入错权重。
    p.add_argument("--init-from-ckpt", type=str, default="",
                   help="可选 latest.pt / best.pt 路径；非空时只加载 dit_state_dict + "
                        "ema_state_dict 做 warm start（不接 optimizer/scheduler/step）。"
                        "strict=True 校验，架构不匹配立即抛错。")
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--cond-dim", type=int, default=256)
    p.add_argument("--max-history-frames", type=int, default=8,
                   help="DiT 可接收的最大历史潜变量帧数；数据构建器默认 4 帧。")
    p.add_argument("--qwen-kv-segment-mode",
                   choices=["concat_layers", "select_last", "mean"],
                   default="select_last",
                   help="select_last 每段只取最后一层 Qwen KV，省显存（默认）；"
                        "concat_layers 把 3 层 token 维拼起来（重，消融用）；mean 为旧版层平均。")

    # num_epochs 默认 2：与 train.sh 的 NUM_EPOCHS:-2 保持一致；831k 样本 /
    # 4 GPU / GRAD_ACC=4 ≈ 52k step/epoch，DiT 从零训通常 100-200k step 才稳定收敛。
    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    # AdamW 走 1D/4D 参数（norm weight、embeddings、Conv2d patch.proj、null_lang_k/v）
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    # Muon 走 2D 矩阵（attention/MLP/AdaLN linear）。Muon LR 通常 5-10× 大于 AdamW。
    p.add_argument("--muon-lr", type=float, default=2e-3,
                   help="Muon optimizer 学习率（仅作用于 2D 权重矩阵）；通常比 AdamW 大 5-10×。")
    p.add_argument("--muon-momentum", type=float, default=0.95,
                   help="Muon momentum；与 nesterov=True 配合，0.95 是 Keller Jordan 默认。")
    # v2 默认开启：torch.compile + gradient checkpointing
    p.add_argument("--compile-dit", action="store_true", default=True,
                   help="torch.compile(DiT) 启用（默认开）。")
    p.add_argument("--no-compile", dest="compile_dit", action="store_false",
                   help="关闭 torch.compile（首次 step 编译慢时可临时关掉）。")
    p.add_argument("--grad-ckpt", action="store_true", default=True,
                   help="per-block gradient checkpointing 启用（默认开）。")
    p.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false",
                   help="关闭 gradient checkpointing（追求 wall-clock 时用，但显存上涨）。")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--keep-recent-checkpoints", type=int, default=3,
                   help="checkpoint-XXXXXX/ 滚动保留数量；更老的会被整体删除。"
                        "best.pt 在顶层独立保存，不受此 keep 影响。")
    # ---- step-level ckpt（独立池，epoch 行为不动） ----
    # 用户场景：数据量上来后一个 epoch 几天都跑不完，光靠 epoch 末 save 拿不到中间
    # 产物。每 --step-save-every 步额外写一份 step-checkpoint-NNNNNN/，独立 keep 池
    # （默认 3，即最近 30k 步）；与 epoch 池不互相淘汰。eval.py / probe.py 的
    # _infer_ckpt_step / _default_run_tag 已同步识别 step-checkpoint- 前缀。
    p.add_argument("--step-save-every", type=int, default=10000,
                   help="每多少优化器步额外写一份 step-checkpoint-NNNNNN/；0 关闭 step 保存。")
    p.add_argument("--keep-recent-step-checkpoints", type=int, default=3,
                   help="step-checkpoint-XXXXXX/ 滚动保留数量；默认 3，配合 step-save-every=10000 即最近 30k 步。")
    p.add_argument("--max-train-steps", type=int, default=0,
                   help="0 表示按 num_epochs 跑完；正整数表示限制优化器更新步数。")
    p.add_argument("--seed", type=int, default=20260529)
    p.add_argument("--t-sampler", choices=["uniform", "logit_normal"], default="logit_normal")
    p.add_argument("--t-logit-mean", type=float, default=0.0)
    p.add_argument("--t-logit-std", type=float, default=1.0)
    p.add_argument("--z0-prior-alpha", type=float, default=1.0)
    p.add_argument("--z0-prior-sigma", type=float, default=1.0)
    p.add_argument("--cfg-drop-prob", type=float, default=0.1)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--latent-stats-path", type=str, default="")
    p.add_argument("--latent-stats-max-samples", type=int, default=1000)
    p.add_argument("--recompute-latent-stats", action="store_true", default=False)

    # ------ TensorBoard / 验证 / 图像样例 ------
    p.add_argument("--tb", action="store_true", default=True,
                   help="0 号进程写 TensorBoard 到 output_dir/tb；--no-tb 关掉。")
    p.add_argument("--no-tb", dest="tb", action="store_false",
                   help="完全关闭 TensorBoard 写入（仅保留 stdout 日志）。")
    p.add_argument("--val-jsonl", type=str, default="",
                   help="验证 jsonl 路径；非空时按 --val-steps 间隔跑 val/loss + val/cos 并落 TensorBoard。")
    p.add_argument("--val-steps", type=int, default=500,
                   help="每多少优化器步跑一次验证；0 关闭。")
    p.add_argument("--val-max-samples", type=int, default=64,
                   help="验证集每次最多取多少条样本，防止验证时间失控。")
    p.add_argument("--image-log-every", type=int, default=500,
                   help="每多少优化器步落一组预测 vs 真值图像；0 关闭。")
    p.add_argument("--image-log-samples", type=int, default=4,
                   help="每次落几张图（也是 Euler 采样调用次数，越多越慢）。")
    p.add_argument("--image-log-euler-steps", type=int, default=32,
                   help="图像样例使用的 Euler 步数；32 在 rectified flow 下通常够用。")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
