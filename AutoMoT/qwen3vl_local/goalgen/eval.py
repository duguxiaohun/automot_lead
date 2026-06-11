"""GoalGen v1/v2 离线评测：在 val.jsonl 上跑 DiT、解码图像、报告 5 个核心指标 +
小样本完整 dump（输入图像/输入文本/输出图像/指标全部本地落盘）。

数据流（与 train 完全同构，只是不反传）：
  history RGB -> 冻结 Qwen 预填充 -> 分段 KV
  history RGB -> 冻结 VAE -> z_history
  target keyframe RGB -> 冻结 VAE -> z1（真值）
  z0 ~ N(0, I)
  z1_pred = 从 z0 通过 Euler 采样得到
  pred RGB = VAE 解码 z1_pred

5 个核心指标（含义见 summary.json["_metric_doc"]）：
  (a) latent_mse:   MSE(z1_pred, z1_gt)     —— 越小越好；与训练损失同口径
  (b) latent_cos:   cosine(z1_pred, z1_gt)  —— 越接近 1 越好（方向相似）
  (c) pixel_l1:     解码 RGB 的 L1            —— 越小越好；地板 = VAE 重建误差
  (d) psnr:         解码 RGB 的 PSNR (dB)   —— 越大越好；地板 = VAE 重建 PSNR
  (e) velocity_cos: 5 个固定 t 上 v_pred vs v_target cosine 平均 —— 越接近 1 越好

输出布局（必填 --save-root，与 train.sh 同根）：
  <save_root>/eval/eval_v1_summary.json     聚合指标 + _metric_doc 说明
  <save_root>/eval/eval_v1_perline.jsonl    每条样本一行（含 5 指标 + png 路径）
  <save_root>/eval/cases/<NNNNN>__<scenario>__<run>__anchor<N>/   小样本完整 dump
      inputs/system_prompt.txt              teacher-forced system prompt 全文
      inputs/user_prompt.txt                teacher-forced user prompt 全文
      inputs/memory.json                    DrivingMemory（scenario / status / subgoal）
      inputs/history_00.jpg ... 03.jpg      history RGB，**复制**到本地
      inputs/target_raw.jpg                 真值 keyframe 原图（VAE 输入）
      outputs/pred.png                      DiT 采样 + VAE 解码（最关心的输出）
      outputs/target_vae_recon.png          真值经 VAE encode→decode，作生成质量天花板
      outputs/compare.png                   target_raw | pred | target_vae_recon 横拼图
      metrics.json                          单 case 5 指标 + _metric_doc
      step.json                             完整元信息（dit_ckpt / qwen_adapter / seed）
      summary.md                            一页可读，顶部直接引用 compare.png
  <save_root>/eval_tb/<run_tag>/            TB scalar + image grid（步骤二 TB 入口）

完整 dump 触发条件：
  默认 --max-samples > 0 时开启（小样本 spot-check）；--full-dump / --no-full-dump 显式覆盖。

多卡分片（H）：
  脚本读 RANK / WORLD_SIZE / LOCAL_RANK；torchrun 启动自动分片，rank0 聚合 perline + TB。
  完整 dump 由各 rank 各自写自己分片的 case 目录（互不冲突）。

典型用法（远程，在 AutoMoT/ 目录下）：

```bash
# 小样本完整 dump（推荐：可直接拿到本地人工 review pred.png vs target_raw.jpg）
GPU_IDS=0 python qwen3vl_local/goalgen/eval.py \
  --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
  --dit-checkpoint checkpoints/goalgen_v1_dit/latest.pt \
  --save-root checkpoints/goalgen_v1_dit \
  --max-samples 100

# 全集跑指标 + TB（不 dump 详情）
GPU_IDS=0 python qwen3vl_local/goalgen/eval.py \
  --dit-checkpoint checkpoints/goalgen_v1_dit/latest.pt \
  --save-root checkpoints/goalgen_v1_dit

# 多卡分片跑全集
GPU_IDS=0,1,2,3 torchrun --standalone --nproc_per_node=4 qwen3vl_local/goalgen/eval.py \
  --dit-checkpoint checkpoints/goalgen_v1_dit/latest.pt \
  --save-root checkpoints/goalgen_v1_dit
```
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple


_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _normalize_gpu_ids(value: str) -> str:
    ids = [part.strip() for part in str(value).split(",") if part.strip()]
    return ",".join(ids)


def _count_gpu_ids(value: str) -> int:
    normalized = _normalize_gpu_ids(value)
    if not normalized:
        return 0
    return len(normalized.split(","))


def _pick_idle_gpus(n: int = 1) -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[1]), int(parts[2]), parts[0]))
        except ValueError:
            continue
    rows.sort(key=lambda x: (x[0], x[1], int(x[2]) if x[2].isdigit() else 9999))
    return ",".join(row[2] for row in rows[:n])


# DDP race 兜底：torchrun 多 worker 且外部未预设 CVD 时，只让 rank0 跑 nvidia-smi 挑卡
# → atomic 写共享文件，其它 rank 阻塞读，避免每 worker 各自挑卡导致 set_device 撞同一张卡。
_GPU_PICK_IMPORT_TIME = time.time()
_GPU_PICK_WAIT_TIMEOUT_S = 60.0
_GPU_PICK_STALE_TOLERANCE_S = 30.0
_GPU_PICK_LOCK_PREFIX = "goalgen_eval_cvd"


def _share_cvd_via_file_for_ddp(want_count: int) -> str:
    """rank0 挑卡 → 共享文件 → 其它 rank 读；锁文件按 MASTER_ADDR+MASTER_PORT 命名隔离
    不同 run，非 rank0 用 mtime >= 本进程 import 时刻 - 容差 拒绝上一轮残留旧文件。"""
    rank = int(os.environ.get("RANK", "0"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    lock_path = pathlib.Path(tempfile.gettempdir()) / f"{_GPU_PICK_LOCK_PREFIX}_{master_addr}_{master_port}.txt"
    min_mtime = _GPU_PICK_IMPORT_TIME - _GPU_PICK_STALE_TOLERANCE_S
    if rank == 0:
        selected = _pick_idle_gpus(want_count)
        if not selected:
            return ""
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        tmp_path = lock_path.with_suffix(f".tmp_{os.getpid()}")
        tmp_path.write_text(selected, encoding="utf-8")
        os.replace(tmp_path, lock_path)
        return selected
    deadline = time.time() + _GPU_PICK_WAIT_TIMEOUT_S
    while True:
        try:
            mtime = lock_path.stat().st_mtime
        except FileNotFoundError:
            mtime = -1.0
        if mtime >= min_mtime:
            break
        if time.time() > deadline:
            raise RuntimeError(
                f"rank {rank} timed out waiting {_GPU_PICK_WAIT_TIMEOUT_S:.0f}s for "
                f"rank0 to publish CUDA_VISIBLE_DEVICES at {lock_path}"
            )
        time.sleep(0.05)
    return lock_path.read_text(encoding="utf-8").strip()


def _maybe_set_idle_gpu_mask() -> None:
    """GoalGen eval 的 GPU 规则。

    优先级：
    1. ``GPU_IDS=...``：显式指定物理卡号；DDP 卡数从 GPU_IDS 数量推断/校验。
    2. 都不指定：单进程自动挑 1 张空闲卡；DDP 按 WORLD_SIZE 自动挑 N 张空闲卡。
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    pinned = _normalize_gpu_ids(os.environ.get("GPU_IDS", ""))
    if pinned:
        picked = _count_gpu_ids(pinned)
        if world_size > 1 and picked < world_size:
            raise RuntimeError(
                f"GPU_IDS={pinned} 只给了 {picked} 张卡，但 torchrun WORLD_SIZE={world_size}。"
                "请让 GPU_IDS 数量 >= nproc_per_node。"
            )
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = pinned
        os.environ["GOALGEN_LOCAL_CUDA_INDEX"] = "0"
        if rank == 0:
            print(
                f"[gpu] using explicit GPU_IDS={pinned}; world_size={world_size}; "
                f"previous CUDA_VISIBLE_DEVICES={previous or '<unset>'}"
            )
        return
    if world_size > 1:
        selected = _share_cvd_via_file_for_ddp(world_size)
    else:
        selected = _pick_idle_gpus(1)
    if selected:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        os.environ["GOALGEN_LOCAL_CUDA_INDEX"] = "0"
        print(
            f"[gpu] auto selected idle CUDA_VISIBLE_DEVICES={selected}; "
            f"world_size={world_size}"
        )


_maybe_set_idle_gpu_mask()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

# TensorBoard 可选依赖；缺包就静默关闭 TB 写入，不让 eval 崩。
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: E402
    _TB_AVAILABLE = True
except Exception:
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False

from qwen3vl_local.engine import LocalQwen3VLInstructEngine  # noqa: E402
from qwen3vl_local.goalgen.dit import (  # noqa: E402
    DiTMoT,
    DiTMoTConfig,
)
from qwen3vl_local.goalgen.flow import euler_sample_cfg, sample_flow_batch  # noqa: E402
from qwen3vl_local.goalgen.prompt import (  # noqa: E402
    build_teacher_system_prompt,
    build_teacher_user_prompt,
    describe_image_inputs,
)
from qwen3vl_local.goalgen.qwen_kv import teacher_forced_prefill  # noqa: E402
from qwen3vl_local.goalgen.vae import FrozenVAE, default_vae_paths  # noqa: E402
from qwen3vl_local.prompt_pipeline import DrivingMemory  # noqa: E402


# --------------------------------------------------------------------------- #
# I/O helpers (与 train 等价，复制而非 import 是为了让 eval 文件可独立运行)
# --------------------------------------------------------------------------- #


def _dump_invocation(output_dir: pathlib.Path, rank: int = 0) -> None:
    """把 sys.argv + 关键 env vars + 元信息写到 ``output_dir/invocations/<ts>_<host>_pid<pid>.txt``。

    只 rank0 写；失败不阻塞 eval（缺 git / IO 错误等都吞掉只打印一行警告）。
    事后想"这版 eval 是哪条命令跑的"直接 cat 就够，不用回翻 shell history。
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


# --------------------------------------------------------------------------- #
# DiT 加载 — 与 runner 完全同口径：优先用 ckpt dit_config 反建模型
# --------------------------------------------------------------------------- #


def build_dit_from_ckpt(
    ckpt_path: pathlib.Path,
    pooled_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> DiTMoT:
    """读取 ckpt → 反建 DiTMoTConfig → 实例化 → strict load。

    当前共享架构的 num_layers 仍从 pooled_kv 推；不再有 language_kv_input_dim 字段，
    pooled_kv 的 (n_heads=8, head_dim=128) 必须与 DiT cfg 严格一致，
    DiTMoT.forward 内部会做硬断言。
    """

    payload = torch.load(ckpt_path, map_location=device)
    saved_cfg_dict = payload.get("dit_config") if isinstance(payload, dict) else None
    saved_args_dict = payload.get("args") if isinstance(payload, dict) else None

    runtime_kwargs = dict(
        num_layers=len(pooled_kv),
    )

    # ---- Qwen 适配器一致性校验（与 runner 同口径）----
    # 评测用错适配器会让 KV 分布偏移，指标完全不可比；形状一致时 strict load 不报错。
    if saved_args_dict is not None:
        def _resolve_adapter(s: str) -> str:
            return str(pathlib.Path(s).resolve()) if s else ""

        saved_adapter = _resolve_adapter(saved_args_dict.get("qwen_adapter_dir", "") or "")
        current_adapter = _resolve_adapter(args.qwen_adapter_dir or "")
        saved_merge = bool(saved_args_dict.get("qwen_adapter_merge", True))
        current_merge = bool(args.qwen_adapter_merge)

        if saved_adapter != current_adapter:
            msg = (
                f"DiT 训练时 qwen_adapter_dir='{saved_adapter or '<base>'}'，"
                f"当前 eval qwen_adapter_dir='{current_adapter or '<base>'}'，不一致会让 KV 分布"
                f"漂移，eval 指标不可比。"
                f" 解决：把 --qwen-adapter-dir 改成训练时同款；"
                f"故意做消融时传 --allow-qwen-adapter-mismatch。"
            )
            if not args.allow_qwen_adapter_mismatch:
                raise RuntimeError(msg)
            print(f"[dit] WARN: {msg}")
        elif saved_adapter and saved_merge != current_merge:
            print(
                f"[dit] 提示：qwen_adapter_merge 训练={saved_merge} 评测={current_merge}（数学等价，仅浮点精度差异）"
            )
        else:
            print(
                f"[dit] qwen_adapter 一致性检查通过："
                f"adapter='{current_adapter or '<base>'}' merge={current_merge}"
            )

    if saved_cfg_dict is not None:
        # 形状预检：训练时与运行时的 num_layers 不一致 → strict load 必炸 attention 投影。
        # 提前断言给出双数字 + 原因。当前架构不再有 language_kv_input_dim，KV 形状由 DiTMoT.forward
        # 内部断言（n_heads × head_dim 必须 == Qwen K/V 形状）。
        saved_layers = saved_cfg_dict.get("num_layers")
        runtime_layers = runtime_kwargs["num_layers"]
        if saved_layers is not None and saved_layers != runtime_layers:
            raise RuntimeError(
                f"DiT ckpt num_layers={saved_layers}（训练时）≠ 运行时 KV 段数={runtime_layers}。"
                f" 解决：eval 与训练保持同样的 --num-layers。"
            )

        merged = dict(saved_cfg_dict)
        # 兼容性：早期架构 ckpt 里有 language_kv_input_dim 字段，DiTMoTConfig 已不接受 -> 弹掉。
        merged.pop("language_kv_input_dim", None)
        merged.update(runtime_kwargs)
        merged.setdefault("latent_channels", 4)
        cfg = DiTMoTConfig(**merged)
        print(f"[dit] 已从检查点里的 dit_config 重建配置")
    else:
        # 旧 ckpt 兼容路径：靠 CLI 默认值 + 运行时维度凑齐。
        cfg = DiTMoTConfig(
            latent_channels=4,
            patch_size=args.patch_size,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            mlp_ratio=args.mlp_ratio,
            cond_dim=args.cond_dim,
            max_history_frames=args.max_history_frames,
            **runtime_kwargs,
        )
        print("[dit] 警告：检查点无 dit_config，回退命令行默认值")

    # v2 强校验：DiT (n_heads, head_dim) 必须严格 = Qwen pooled K/V 的形状。
    # 这里在 instantiate 之前提前检查，错误堆栈更清晰。
    if pooled_kv:
        k0, _ = pooled_kv[0]
        kv_n_heads = int(k0.shape[1])
        kv_head_dim = int(k0.shape[3])
        if cfg.n_heads != kv_n_heads or (cfg.hidden_dim // cfg.n_heads) != kv_head_dim:
            raise RuntimeError(
                f"DiT cfg (n_heads={cfg.n_heads}, head_dim={cfg.hidden_dim // cfg.n_heads}) "
                f"与运行时 Qwen K/V (n_kv_heads={kv_n_heads}, head_dim={kv_head_dim}) 不匹配；"
                "当前共享架构要求严格相同。请确认 Qwen 模型 / DiT cfg 一致（默认 8×128）。"
            )

    model = DiTMoT(cfg).to(device=device, dtype=dtype)
    if isinstance(payload, dict) and args.use_ema and payload.get("ema_state_dict") is not None:
        if payload.get("dit_state_dict") is not None:
            # EMA only tracks trainable parameters. Frozen external patch/unpatch
            # weights therefore live only in dit_state_dict; use raw weights as
            # the complete base, then overlay EMA shadows for trainable params.
            state_dict = dict(payload["dit_state_dict"])
            state_dict.update(payload["ema_state_dict"])
            print(f"[dit] using EMA weights over full DiT base for eval (decay={payload.get('ema_decay', 'unknown')})")
        else:
            state_dict = payload["ema_state_dict"]
            print(f"[dit] using EMA weights for eval (decay={payload.get('ema_decay', 'unknown')})")
    else:
        state_dict = payload.get("dit_state_dict", payload) if isinstance(payload, dict) else payload
        if isinstance(payload, dict) and args.use_ema:
            print("[dit] WARN: checkpoint has no ema_state_dict; using raw DiT weights")
    model.load_state_dict(state_dict, strict=True)
    patch_info = model.restore_patch_unpatch_from_config()
    model.eval()
    print(f"[dit] 已加载检查点：{ckpt_path}")
    print(f"[patch_unpatch] {patch_info}")
    print(
        f"[dit] hidden={cfg.hidden_dim} heads={cfg.n_heads} head_dim={cfg.hidden_dim // cfg.n_heads} "
        f"layers={cfg.num_layers} patch={cfg.patch_size}"
    )
    return model


def _probe_language_kv(
    engine: LocalQwen3VLInstructEngine,
    sample: Dict[str, Any],
    num_segments: int,
    kv_segment_mode: str,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """用 val.jsonl 第一条样本跑一次 prefill，拿到 segmented KV 形状（推 DiT 几个超参用）。"""
    history_images = [load_rgb(p) for p in sample["history_rgb_paths"]]
    memory = memory_from_sample(sample)
    return teacher_forced_prefill(
        engine=engine,
        memory=memory,
        images=history_images,
        num_segments=num_segments,
        kv_segment_mode=kv_segment_mode,
    ).pooled_kv


def _load_vae_latent_stats_from_ckpt(vae: FrozenVAE, ckpt_path: pathlib.Path) -> None:
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict) and payload.get("latent_stats"):
        vae.load_latent_stats_dict(payload["latent_stats"])
        print(f"[vae] loaded latent stats from {ckpt_path}")
    else:
        print("[vae] WARN: checkpoint has no latent_stats; VAE latents are not channel-normalized")


def _make_z_init_from_prior(
    z_history: torch.Tensor,
    shape: Tuple[int, int, int, int],
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
# 四个指标
# --------------------------------------------------------------------------- #


def latent_mse(z_pred: torch.Tensor, z_gt: torch.Tensor) -> float:
    # 全局 MSE，不分通道；与训练损失同口径（只是 v_target 换成 z）。
    return float((z_pred.float() - z_gt.float()).pow(2).mean().item())


def latent_cosine(z_pred: torch.Tensor, z_gt: torch.Tensor) -> float:
    # 把 [B, C, H, W] flatten 成 [B, C*H*W] 再算 cosine；升 fp32 避免 bf16 精度损失。
    p = z_pred.float().flatten(1)
    g = z_gt.float().flatten(1)
    return float(F.cosine_similarity(p, g, dim=1).mean().item())


def pixel_l1_psnr(rgb_pred: torch.Tensor, rgb_gt: torch.Tensor) -> Tuple[float, float]:
    """rgb_* 是 [-1, 1] 范围 [B, 3, H, W]。先 L1，再换 PSNR。

    PSNR 公式：20 * log10(max_signal / sqrt(mse))。我们的信号范围是 [-1, 1]，
    所以 max_signal = 2.0。常见做法。
    """
    pred = rgb_pred.float()
    gt = rgb_gt.float()
    l1 = float((pred - gt).abs().mean().item())
    mse = float((pred - gt).pow(2).mean().item())
    if mse <= 0.0:
        psnr = float("inf")
    else:
        psnr = 20.0 * math.log10(2.0 / math.sqrt(mse))
    return l1, psnr


def velocity_cosine_multi_t(
    dit: torch.nn.Module,
    z_history: torch.Tensor,
    pooled_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    z1: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    z0_prior_alpha: float = 0.0,    # 默认纯噪声起点，与 train 默认对齐
    z0_prior_sigma: float = 1.0,
    t_grid: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> float:
    """在 5 个固定 t 各采一次 v_pred / v_target，算余弦相似度的平均值。

    单点 t 噪声大，多点平均更接近"模型在整段轨迹上的 velocity 质量"，跟训练时的
    train/cos 同口径但稳定。固定 t 而非随机，让 eval 之间可比较。
    """
    cosines: List[float] = []
    for t_val in t_grid:
        # 固定 z0：所有 t 用同一份 z0 让 v_target = z1 - z0 也固定，对比维度只剩 t 与 v_pred。
        t = torch.full((z1.shape[0],), t_val, device=device, dtype=dtype)
        batch = sample_flow_batch(
            z1=z1,
            z_prior=z_history[:, -1],
            alpha=z0_prior_alpha,
            sigma=z0_prior_sigma,
            t=t,
        )
        v_pred = dit(batch.z_t, z_history, batch.t, pooled_kv)
        p = v_pred.float().flatten(1)
        g = batch.v_target.float().flatten(1)
        cosines.append(float(F.cosine_similarity(p, g, dim=1).mean().item()))
    return sum(cosines) / len(cosines)


# --------------------------------------------------------------------------- #
# 主 eval 循环
# --------------------------------------------------------------------------- #


def _save_rgb_png(rgb_tensor: torch.Tensor, path: pathlib.Path) -> None:
    """rgb_tensor: [3, H, W]，范围 [-1, 1]。直接落盘 PNG，方便人眼对比。

    不依赖 torchvision —— eval 机器上可能没装；改用 PIL：[-1,1] -> [0,255] uint8。
    """
    tensor = rgb_tensor.clamp(-1.0, 1.0)
    tensor = ((tensor + 1.0) / 2.0 * 255.0).round().clamp(0, 255)
    arr = tensor.to(torch.uint8).cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(arr, mode="RGB").save(path)


def _maybe_dump_pair(
    sample_idx: int,
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    samples_dir: pathlib.Path,
    image_dump_count: int,
) -> Dict[str, str]:
    """前 image_dump_count 条样本落 pred / gt 两张 PNG，返回路径字典（写进 perline jsonl）。"""
    if sample_idx >= image_dump_count:
        return {}
    samples_dir.mkdir(parents=True, exist_ok=True)
    pred_path = samples_dir / f"{sample_idx:05d}_pred.png"
    gt_path = samples_dir / f"{sample_idx:05d}_gt.png"
    _save_rgb_png(pred_rgb[0], pred_path)
    _save_rgb_png(gt_rgb[0], gt_path)
    return {"decoded_png_path": str(pred_path), "gt_png_path": str(gt_path)}


# --------------------------------------------------------------------------- #
# 完整 dump：把单条样本的 inputs/outputs/summary 全写到一个 case 目录
# --------------------------------------------------------------------------- #


def _copy_image(src: str, dst: pathlib.Path) -> bool:
    """复制 RGB 图像到 case 目录（不 symlink，方便远端跑完拉到本地）。源不存在返回 False。"""
    src_path = pathlib.Path(src)
    if not src_path.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dst)
    return True


def _save_compare_png(
    rgb_pred: torch.Tensor,
    rgb_gt_vae: torch.Tensor,
    target_raw_path: Optional[str],
    out_path: pathlib.Path,
) -> None:
    """横向拼 [target_raw | pred | target_vae_recon] 三联图。

    - target_raw：真值原图（VAE 输入前的样子），从 sample["target_rgb_path"] 读。
    - pred：DiT 采样 + VAE 解码（模型生成）。
    - target_vae_recon：真值经 VAE encode→decode（生成质量天花板）。

    三张图高度统一为 pred 的高度（VAE 解码出来通常 256/512 这个量级），
    target_raw 按等比例 resize。如果 target_raw 读不到，只拼后两张。
    """
    # pred / vae_recon 都是 [3, H, W] in [-1, 1]
    def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
        t = t.clamp(-1.0, 1.0)
        arr = ((t + 1.0) / 2.0 * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy().transpose(1, 2, 0)
        return Image.fromarray(arr, mode="RGB")

    pred_img = _tensor_to_pil(rgb_pred)
    recon_img = _tensor_to_pil(rgb_gt_vae)
    H = pred_img.height

    parts: List[Image.Image] = []
    if target_raw_path and pathlib.Path(target_raw_path).exists():
        raw = Image.open(target_raw_path).convert("RGB")
        # 按 pred 高度等比例缩放，保留三视角拼接的横向条带感
        new_w = max(1, int(raw.width * H / max(raw.height, 1)))
        parts.append(raw.resize((new_w, H), Image.BILINEAR))
    parts.append(pred_img)
    parts.append(recon_img)

    total_w = sum(p.width for p in parts)
    canvas = Image.new("RGB", (total_w, H), (0, 0, 0))
    x = 0
    for p in parts:
        canvas.paste(p, (x, 0))
        x += p.width
    canvas.save(out_path)


def _render_goalgen_summary_md(
    case_dir_name: str,
    sample: Dict[str, Any],
    sample_idx: int,
    memory: DrivingMemory,
    system_prompt: str,
    user_prompt: str,
    metrics: Dict[str, float],
    saved_history: List[str],
    has_target_raw: bool,
    args: argparse.Namespace,
) -> str:
    """一页 markdown：顶部就是 compare.png（target_raw | pred | recon 横拼），
    一眼就能看出"模型生成的子目标图像 vs 真值长啥样"——这就是用户最关心的可视化。
    """
    lines: List[str] = [
        f"# Case: {sample.get('scenario')}/{sample.get('run_id')} anchor={sample.get('anchor')}",
        "",
        f"- val.jsonl sample_idx: **{sample_idx}**",
        f"- target_frame: {sample.get('target_frame')}",
        f"- dit_checkpoint: `{args.dit_checkpoint}`",
        f"- qwen_adapter_dir: `{args.qwen_adapter_dir or '<base>'}`",
        f"- euler_steps: {args.euler_steps}, seed: {args.seed + sample_idx}",
        "",
        "## 最关心的可视化：target_raw | pred | target_vae_recon",
        "",
        "![compare](outputs/compare.png)",
        "",
        "- **target_raw**：真值 keyframe 原图（VAE 输入前的样子）",
        "- **pred**：DiT 采样 + VAE 解码（模型生成的子目标图像）",
        "- **target_vae_recon**：真值经 VAE encode→decode；生成质量的天花板（pred 不会比这个清）",
        "",
        "## Metrics",
        "| metric | value | 说明 |",
        "|---|---|---|",
        f"| latent_mse   | {metrics['latent_mse']:.6f} | MSE(z1_pred, z1_gt)；与训练损失同口径，越小越好 |",
        f"| latent_cos   | {metrics['latent_cos']:.4f} | cosine(z1_pred, z1_gt)；越接近 1 越好 |",
        f"| pixel_l1     | {metrics['pixel_l1']:.4f} | 解码 RGB L1；越小越好 |",
        f"| psnr         | {metrics['psnr']:.2f} | 解码 RGB PSNR (dB)；越大越好 |",
        f"| velocity_cos | {metrics['velocity_cos']:.4f} | 5 个 t 上 v_pred vs v_target cosine 平均；训练健康度同口径 |",
        "",
        "## Memory (driving state)",
        "```json",
        json.dumps(asdict(memory), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Input history (oldest → newest)",
    ]
    src_history = sample.get("history_rgb_paths", [])
    for k, fname in enumerate(saved_history):
        src = src_history[k] if k < len(src_history) else ""
        lines.append(f"- ![h{k}](inputs/{fname}) `inputs/{fname}` ← src `{src}`")
    if has_target_raw:
        lines.append("")
        lines.append("## Target raw")
        lines.append(f"- ![target_raw](inputs/target_raw.jpg) `inputs/target_raw.jpg` ← src `{sample.get('target_rgb_path', '')}`")
    lines.append("")
    lines.append("## Teacher-forced system prompt")
    lines.append("```")
    lines.append(system_prompt)
    lines.append("```")
    lines.append("")
    lines.append("## Teacher-forced user prompt")
    lines.append("```")
    lines.append(user_prompt)
    lines.append("```")
    lines.append("")
    return "\n".join(lines) + "\n"


def dump_goalgen_case(
    case_dir: pathlib.Path,
    sample: Dict[str, Any],
    sample_idx: int,
    memory: DrivingMemory,
    rgb_pred: torch.Tensor,
    rgb_gt_vae: torch.Tensor,
    metrics: Dict[str, float],
    patch_unpatch: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """把一条样本完整 dump 到 <case_dir>/{inputs, outputs, metrics.json, step.json, summary.md}。

    与 SFT eval 的 dump_case 同口径（inputs/outputs 二分 + 顶层 summary.md），
    区别是 GoalGen 的"输出"是图像而不是文本，所以 outputs/ 下放 pred.png /
    target_vae_recon.png / compare.png 三张 PNG。
    """
    inputs_dir = case_dir / "inputs"
    outputs_dir = case_dir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1) inputs：teacher-forced 输入文本（system + user）+ memory + history RGB + target_raw。
    system_prompt = build_teacher_system_prompt()
    user_prompt = build_teacher_user_prompt(
        memory,
        image_description=describe_image_inputs(len(sample.get("history_rgb_paths", []))),
    )
    (inputs_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    (inputs_dir / "user_prompt.txt").write_text(user_prompt, encoding="utf-8")
    (inputs_dir / "memory.json").write_text(
        json.dumps(asdict(memory), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    saved_history: List[str] = []
    for k, src in enumerate(sample.get("history_rgb_paths", [])):
        fname = f"history_{k:02d}.jpg"
        if _copy_image(src, inputs_dir / fname):
            saved_history.append(fname)
        else:
            print(f"[dump][warn] sample_idx={sample_idx} history src 不存在：{src}")
    target_raw_path = sample.get("target_rgb_path")
    has_target_raw = bool(target_raw_path and _copy_image(target_raw_path, inputs_dir / "target_raw.jpg"))
    if target_raw_path and not has_target_raw:
        print(f"[dump][warn] sample_idx={sample_idx} target_raw 不存在：{target_raw_path}")

    # 2) outputs：pred.png（模型生成）+ target_vae_recon.png（VAE 天花板）+ compare.png（横拼三联图）。
    _save_rgb_png(rgb_pred[0], outputs_dir / "pred.png")
    _save_rgb_png(rgb_gt_vae[0], outputs_dir / "target_vae_recon.png")
    _save_compare_png(
        rgb_pred=rgb_pred[0],
        rgb_gt_vae=rgb_gt_vae[0],
        target_raw_path=str(inputs_dir / "target_raw.jpg") if has_target_raw else None,
        out_path=outputs_dir / "compare.png",
    )

    # 3) metrics.json：单 case 5 指标 + _metric_doc（指标含义跟 summary.json 顶层同）。
    metric_doc = {
        "latent_mse": "MSE(z1_pred, z1_gt)；与训练损失同口径，越小越好",
        "latent_cos": "cosine(z1_pred, z1_gt)；越接近 1 越好（方向相似）",
        "pixel_l1": "解码 RGB [-1,1] L1；越小越好；地板 = VAE 重建误差",
        "psnr": "解码 RGB PSNR (dB)；越大越好；地板 = VAE 重建 PSNR",
        "velocity_cos": "5 个固定 t 上 v_pred vs v_target cosine 平均；越接近 1 越好",
    }
    (case_dir / "metrics.json").write_text(
        json.dumps({"_metric_doc": metric_doc, **metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 4) step.json：完整元信息（dit_ckpt / qwen_adapter / euler_steps / seed），可回溯。
    step = {
        "sample_idx": sample_idx,
        "scenario": sample.get("scenario"),
        "run_id": sample.get("run_id"),
        "anchor": sample.get("anchor"),
        "target_frame": sample.get("target_frame"),
        "history_rgb_paths_src": sample.get("history_rgb_paths", []),
        "history_files_local": saved_history,
        "target_rgb_path_src": target_raw_path or "",
        "target_raw_local": "inputs/target_raw.jpg" if has_target_raw else None,
        "metrics": metrics,
        "dit_checkpoint": args.dit_checkpoint,
        "patch_unpatch": patch_unpatch,
        "qwen_adapter_dir": args.qwen_adapter_dir or "",
        "qwen_adapter_merge": bool(args.qwen_adapter_merge),
        "euler_steps": args.euler_steps,
        "seed": args.seed + sample_idx,
    }
    (case_dir / "step.json").write_text(
        json.dumps(step, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 5) summary.md：一页可读，顶部就是 compare.png（用户最关心的可视化）。
    md = _render_goalgen_summary_md(
        case_dir_name=case_dir.name,
        sample=sample,
        sample_idx=sample_idx,
        memory=memory,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        metrics=metrics,
        saved_history=saved_history,
        has_target_raw=has_target_raw,
        args=args,
    )
    (case_dir / "summary.md").write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------- #
# 分布式 + 路径 helper（H）
# --------------------------------------------------------------------------- #

def setup_distributed() -> Tuple[int, int, int]:
    """与 train.py 同口径：torchrun 注入 RANK / WORLD_SIZE / LOCAL_RANK。

    单卡 = world_size=1，dist init 不触发，所有 if rank0 分支恒进。
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def all_gather_records(records: List[Dict[str, Any]], world_size: int) -> List[Dict[str, Any]]:
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return records
    bucket: List[Optional[List[Dict[str, Any]]]] = [None] * world_size  # type: ignore[type-var]
    dist.all_gather_object(bucket, records)
    merged: List[Dict[str, Any]] = []
    for shard in bucket:
        if shard:
            merged.extend(shard)
    merged.sort(key=lambda r: r.get("sample_idx", 0))
    return merged


def _resolve_eval_paths(args: argparse.Namespace) -> Dict[str, pathlib.Path]:
    """所有 eval 产物在 <save_root>/eval/ 与 <save_root>/eval_tb/<run_tag>/ 之下。

    --save-root 必填（main 里 argparse required=True 强制）。
    """
    root = pathlib.Path(args.save_root)
    run_tag = (args.run_tag or "").strip() or _default_run_tag(args)
    eval_dir = root / "eval"
    return {
        "eval_dir": eval_dir,
        "tb_dir": root / "eval_tb" / run_tag,
        "samples_dir": eval_dir / "samples",
        "cases_dir": eval_dir / "cases",
        "perline_jsonl": eval_dir / "eval_v1_perline.jsonl",
        "summary_json": eval_dir / "eval_v1_summary.json",
    }


def _default_run_tag(args: argparse.Namespace) -> str:
    """根据 dit_checkpoint 路径推 TB run 名。

    checkpoint-000200/goalgen_v1.pt      → ckpt200
    step-checkpoint-090000/goalgen_v1.pt → stepckpt90000
    latest.pt → latest
    其它 → 文件名（去 .pt 后缀）
    """
    p = pathlib.Path(args.dit_checkpoint)
    parent_name = p.parent.name
    if parent_name.startswith("step-checkpoint-"):
        # 注意：step-checkpoint- 也以 "checkpoint-" 子串出现，必须先于下面的 elif
        # 命中（startswith 优先匹配更长的前缀）。
        try:
            return "stepckpt" + str(int(parent_name.split("-", 2)[2]))
        except (ValueError, IndexError):
            pass
    elif parent_name.startswith("checkpoint-"):
        try:
            return "ckpt" + str(int(parent_name.split("-", 1)[1]))
        except (ValueError, IndexError):
            pass
    return p.stem or "latest"


@torch.no_grad()
def eval_loop(args: argparse.Namespace) -> None:
    rank, local_rank, world_size = setup_distributed()
    paths = _resolve_eval_paths(args)
    out_dir = paths["eval_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    from qwen3vl_local.run_log import install_output_log
    install_output_log(out_dir, rank=rank)
    _dump_invocation(pathlib.Path(args.save_root), rank=rank)

    # device：多卡时 pin 到 LOCAL_RANK；单卡在 import 阶段已把 GPU_IDS/自动选卡
    # 映射成进程内 cuda:0。
    if world_size > 1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_cuda_index = int(os.environ.get("GOALGEN_LOCAL_CUDA_INDEX", "0"))
        device = torch.device(f"cuda:{local_cuda_index}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GoalGen eval 需要 CUDA；离线机器只能跑数据校验，不能跑 eval。")

    samples_dir = paths["samples_dir"]
    if is_rank0(rank):
        run_tag = args.run_tag.strip() if args.run_tag else _default_run_tag(args)
        print(f"[eval] world_size={world_size} rank={rank} eval_dir={out_dir}")
        print(f"[eval] tb_dir={paths['tb_dir']} (run_tag={run_tag})")

    samples = load_jsonl(pathlib.Path(args.val_jsonl))
    if not samples:
        raise RuntimeError(f"验证 jsonl 为空：{args.val_jsonl}")
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if is_rank0(rank):
        print(f"[data] 验证样本={len(samples)} 来源={args.val_jsonl}")

    # ---- 完整 dump 模式判定（与 SFT eval_sft_v1 同口径）----
    # 默认：--max-samples > 0 → 开；跑全集 → 关。
    # 显式 --full-dump / --no-full-dump 可覆盖默认。
    if args.full_dump is None:
        full_dump_enabled = args.max_samples > 0
    else:
        full_dump_enabled = bool(args.full_dump)
    dump_limit = args.full_dump_limit if args.full_dump_limit > 0 else len(samples)
    cases_dir = paths["cases_dir"]
    if full_dump_enabled and is_rank0(rank):
        cases_dir.mkdir(parents=True, exist_ok=True)
        print(f"[dump] 完整 dump 启用 → cases_dir={cases_dir}")
        print(f"[dump] dump 数量上限 = {dump_limit}（每个 rank 写自己分片的 case 目录，互不冲突）")
    dump_count_local = 0  # 本 rank 已经 dump 的样本数

    # 1) 起 engine / vae。
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
    # eval 必须用与训练时同款 Qwen 编码：训练挂了 LoRA，eval 不挂会导致 KV 分布漂移，
    # 指标对比毫无意义。所以这里也走同样的 attach + merge 路径。
    if args.qwen_adapter_dir:
        engine.attach_lora_adapter(args.qwen_adapter_dir, merge=args.qwen_adapter_merge)

    vae_cfg_path, vae_weights = default_vae_paths()
    vae = FrozenVAE.load(
        config_path=vae_cfg_path,
        weights_path=vae_weights,
        device=str(device),
        dtype=args.vae_dtype,
    )
    _load_vae_latent_stats_from_ckpt(vae, pathlib.Path(args.dit_checkpoint).resolve())

    # 2) 用第一条样本探一次 KV，反推 DiT shape 并加载 ckpt。
    dit_dtype = dtype_from_name(args.dit_dtype)
    print("[probe] 正在用第一条样本的分段 KV 推断 DiT 形状 ...")
    probe_pooled = _probe_language_kv(engine, samples[0], args.num_layers, args.qwen_kv_segment_mode)
    dit = build_dit_from_ckpt(
        ckpt_path=pathlib.Path(args.dit_checkpoint).resolve(),
        pooled_kv=probe_pooled,
        args=args,
        device=device,
        dtype=dit_dtype,
    )

    # 3) per-line + 汇总。
    perline_path = paths["perline_jsonl"]
    summary_path = paths["summary_json"]
    by_scenario: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    all_metrics: List[Dict[str, float]] = []
    # 分布式时本地 rank 只累积自己的分片到 perline_rows，最后 all_gather 由 rank0 落盘。
    perline_rows: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        # rank 分片：每条样本只被一个 rank 处理。步长 world_size 对 NFS 缓存更友好。
        if world_size > 1 and (idx % world_size) != rank:
            continue
        # 占位 with 块的缩进保持原样（下方 unindent 到 for 同级）
        if True:
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

            # ---- 生成预测 latent（euler）----
            # 固定 seed：让同一条样本在同一 ckpt 上 eval 多次得到相同结果，便于复现。
            gen = torch.Generator(device=device).manual_seed(args.seed + idx)
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
                dit=dit,
                z_history=z_history,
                pooled_kv=pooled_kv,
                shape=tuple(z1_gt.shape),
                device=device,
                dtype=dit_dtype,
                num_steps=args.euler_steps,
                cfg_scale=args.cfg_scale,
                z_init=z_init,
            )

            # ---- 四个指标 ----
            m_mse = latent_mse(z1_pred, z1_gt)
            m_cos = latent_cosine(z1_pred, z1_gt)
            # 显式转到 vae 的 (device, dtype)：与 train._decode_latent_to_image 同口径
            # 的防御层，避免未来 vae.py 内部 cast 被删时这里悄悄 dtype mismatch。
            z1_pred_for_vae = z1_pred.to(device=vae.device, dtype=vae.dtype)
            z1_gt_for_vae = z1_gt.to(device=vae.device, dtype=vae.dtype)
            rgb_pred = vae.decode(z1_pred_for_vae).clamp(-1.0, 1.0)
            rgb_gt = vae.decode(z1_gt_for_vae).clamp(-1.0, 1.0)
            m_l1, m_psnr = pixel_l1_psnr(rgb_pred, rgb_gt)
            m_vcos = velocity_cosine_multi_t(
                dit,
                z_history,
                pooled_kv,
                z1_gt,
                device,
                dit_dtype,
                z0_prior_alpha=args.z0_prior_alpha,
                z0_prior_sigma=args.z0_prior_sigma,
            )

            png_paths = _maybe_dump_pair(idx, rgb_pred, rgb_gt, samples_dir, args.image_dump_count)

            # ---- 完整 dump：每条样本一个 case 目录（在 rank 分片内顺序写）----
            if full_dump_enabled and dump_count_local < dump_limit:
                metrics_one = {
                    "latent_mse": m_mse,
                    "latent_cos": m_cos,
                    "pixel_l1": m_l1,
                    "psnr": m_psnr,
                    "velocity_cos": m_vcos,
                }
                case_name = (
                    f"{idx:05d}__{sample.get('scenario', 'unknown')}"
                    f"__{sample.get('run_id', 'norun')}"
                    f"__anchor{sample.get('anchor', 'na')}"
                )
                try:
                    dump_goalgen_case(
                        case_dir=cases_dir / case_name,
                        sample=sample,
                        sample_idx=idx,
                        memory=memory,
                        rgb_pred=rgb_pred,
                        rgb_gt_vae=rgb_gt,
                        metrics=metrics_one,
                        patch_unpatch=dit.patch_unpatch_metadata(args.dit_checkpoint),
                        args=args,
                    )
                    dump_count_local += 1
                except Exception as dump_err:
                    # dump 失败不影响主指标；只 warn。
                    print(f"[dump][warn] sample_idx={idx} dump 失败：{dump_err}")

            row: Dict[str, Any] = {
                "sample_idx": idx,
                "scenario": sample.get("scenario"),
                "run_id": sample.get("run_id"),
                "anchor": sample.get("anchor"),
                "status": sample.get("status"),
                "subgoal": sample.get("subgoal"),
                "target_frame": sample.get("target_frame"),
                "latent_mse": m_mse,
                "latent_cos": m_cos,
                "pixel_l1": m_l1,
                "psnr": m_psnr,
                "velocity_cos": m_vcos,
            }
            row.update(png_paths)
            perline_rows.append(row)

            all_metrics.append({
                "latent_mse": m_mse,
                "latent_cos": m_cos,
                "pixel_l1": m_l1,
                "psnr": m_psnr,
                "velocity_cos": m_vcos,
            })
            by_scenario[sample.get("scenario", "<unknown>")].append(all_metrics[-1])

            if (idx + 1) % args.log_every == 0 or idx == len(samples) - 1:
                if is_rank0(rank):
                    print(
                        f"[eval][rank{rank}] {idx + 1}/{len(samples)} "
                        f"latent_mse={m_mse:.6f} latent_cos={m_cos:.4f} "
                        f"pixel_l1={m_l1:.4f} psnr={m_psnr:.2f} v_cos={m_vcos:.4f}"
                    )

    # ---- 跨 rank 聚合（H）----
    if world_size > 1:
        dist.barrier()
    perline_rows = all_gather_records(perline_rows, world_size)

    if not is_rank0(rank):
        cleanup_distributed()
        return

    # 由 rank0 用合并后的 perline 重算 by_scenario / overall（替代分片视角的本地 all_metrics）
    by_scenario = defaultdict(list)
    all_metrics = []
    for row in perline_rows:
        m = {
            "latent_mse": row["latent_mse"],
            "latent_cos": row["latent_cos"],
            "pixel_l1": row["pixel_l1"],
            "psnr": row["psnr"],
            "velocity_cos": row["velocity_cos"],
        }
        all_metrics.append(m)
        by_scenario[row.get("scenario", "<unknown>")].append(m)

    # 4) 汇总（按 scenario 拆分 + 整体）。
    def _agg(metrics: List[Dict[str, float]]) -> Dict[str, float]:
        if not metrics:
            return {}
        keys = list(metrics[0].keys())
        out: Dict[str, float] = {}
        for k in keys:
            vals = [m[k] for m in metrics]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            out[f"{k}_mean"] = mean
            out[f"{k}_std"] = math.sqrt(var)
        out["count"] = len(metrics)
        return out

    metric_doc = {
        "latent_mse": "MSE(z1_pred, z1_gt)；与训练损失同口径，越小越好",
        "latent_cos": "cosine(z1_pred, z1_gt)；越接近 1 越好（方向相似）",
        "pixel_l1": "解码 RGB [-1,1] L1；越小越好；地板 = VAE 重建误差",
        "psnr": "解码 RGB PSNR (dB)；越大越好；地板 = VAE 重建 PSNR",
        "velocity_cos": "5 个固定 t 上 v_pred vs v_target cosine 平均；越接近 1 越好",
    }
    summary = {
        "_metric_doc": metric_doc,
        "config": vars(args),
        "patch_unpatch": dit.patch_unpatch_metadata(args.dit_checkpoint),
        "overall": _agg(all_metrics),
        "by_scenario": {s: _agg(ms) for s, ms in sorted(by_scenario.items())},
        "world_size": world_size,
    }

    # ---- 写 perline + summary ----
    perline_path.parent.mkdir(parents=True, exist_ok=True)
    with perline_path.open("w", encoding="utf-8") as fout:
        for row in perline_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] 逐行结果={perline_path}")
    print(f"[done] 汇总结果={summary_path}")
    print(f"[done] 样例 PNG 目录={samples_dir}（前 {args.image_dump_count} 条 pred/gt 分开 PNG）")
    if full_dump_enabled:
        print(f"[done] rank0 完整 dump 目录={cases_dir}（rank0 本地写 {dump_count_local} 条 case，含 compare.png + 输入图文）")
    overall = summary["overall"]
    if overall:
        print(
            f"[overall] latent_mse={overall['latent_mse_mean']:.6f} "
            f"latent_cos={overall['latent_cos_mean']:.4f} "
            f"pixel_l1={overall['pixel_l1_mean']:.4f} "
            f"psnr={overall['psnr_mean']:.2f} "
            f"velocity_cos={overall['velocity_cos_mean']:.4f}"
        )

    # ---- TensorBoard 写入（G）----
    # 与 SFT eval 同口径：scalar 用 ckpt step 作为 global_step（latest.pt 退到 0），
    # 让"同一 DiT 在不同训练步 ckpt 的 eval"形成横向曲线。
    # image：把前 image_dump_count 条样本的 pred 和 gt 都写到 TB，方便人眼对比；
    # 多次 eval 同一 ckpt 会重叠，建议用 --run-tag 区分（如 ckpt500 / final）。
    tb_dir = paths["tb_dir"]
    if (not args.no_tb) and _TB_AVAILABLE and overall:
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        try:
            step = _infer_ckpt_step(args.dit_checkpoint)
            for k in ("latent_mse", "latent_cos", "pixel_l1", "psnr", "velocity_cos"):
                writer.add_scalar(f"eval/{k}", overall[f"{k}_mean"], step)
                writer.add_scalar(f"eval/{k}_std", overall[f"{k}_std"], step)
            # by_scenario 拆开写：横向看哪个场景拉低整体指标。
            for sc, agg in summary["by_scenario"].items():
                for k in ("latent_mse", "latent_cos", "pixel_l1", "psnr", "velocity_cos"):
                    if f"{k}_mean" in agg:
                        writer.add_scalar(f"eval_by_scenario/{sc}/{k}", agg[f"{k}_mean"], step)
            # 图像：复用 samples_dir 里已落盘的 PNG（rank 分片各自写过，文件已就绪）；
            # 取前 min(image_dump_count, 8) 条做交错排（pred, gt, pred, gt ...）写入 TB。
            try:
                _write_tb_image_grid(writer, perline_rows, step, max_pairs=8)
            except Exception as e:
                print(f"[tb][warn] image grid 写入失败：{e}")
            print(f"[tb] eval scalars + image grid written to {tb_dir}")
        finally:
            writer.close()
    elif args.no_tb:
        print("[tb] 已通过 --no-tb 关闭 TB 写入。")
    elif not _TB_AVAILABLE:
        print("[tb] 警告：SummaryWriter 不可用，跳过 TB 写入。")

    cleanup_distributed()


def _infer_ckpt_step(ckpt_path: str) -> int:
    """从 dit_checkpoint 路径推 step。

    checkpoint-NNNNNN/goalgen_v1.pt      → NNNNNN  （epoch 池）
    step-checkpoint-NNNNNN/goalgen_v1.pt → NNNNNN  （step 池）
    其它 → 0
    """
    parent = pathlib.Path(ckpt_path).parent.name
    # 必须先匹配更长的前缀；否则 step-checkpoint-NNN 会落到 startswith("checkpoint-")
    # 的分支去 split("-", 1)[1] → 拿到 "checkpoint-NNN"，int() 会抛错退到 0。
    if parent.startswith("step-checkpoint-"):
        try:
            return int(parent.split("-", 2)[2])
        except (ValueError, IndexError):
            return 0
    if parent.startswith("checkpoint-"):
        try:
            return int(parent.split("-", 1)[1])
        except (ValueError, IndexError):
            return 0
    return 0


def _write_tb_image_grid(
    writer: Any,
    perline_rows: List[Dict[str, Any]],
    step: int,
    max_pairs: int = 8,
) -> None:
    """把前 N 条样本的 pred/gt PNG 读回来交错排写到 TB Image 面板。

    交错布局（pred_0, gt_0, pred_1, gt_1, ...）与训练 image_samples 同口径，让人
    一眼能"一对一"对照预测和真值；超过 max_pairs 时截断（image 写太多 TB 会卡）。
    """
    import numpy as np  # 局部 import：TB 写图才需要，不污染顶层导入。
    pairs: List[Tuple[str, str]] = []
    for row in perline_rows:
        pred = row.get("decoded_png_path")
        gt = row.get("gt_png_path")
        if pred and gt and pathlib.Path(pred).exists() and pathlib.Path(gt).exists():
            pairs.append((pred, gt))
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        return
    interleaved = []
    for pred, gt in pairs:
        for path in (pred, gt):
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img).astype("float32") / 255.0  # [H, W, 3] in [0,1]
            arr = arr.transpose(2, 0, 1)  # → [3, H, W] for TB add_images NCHW
            interleaved.append(torch.from_numpy(arr))
    grid = torch.stack(interleaved, dim=0)
    writer.add_images("eval/pred_vs_gt", grid, step, dataformats="NCHW")


def _resolve_default_dit_checkpoint(save_root_hint: Optional[str] = None) -> str:
    """根据 --save-root 推 base 目录，再按"latest 子目录 > 老顶层"顺序找 ckpt。

    base 推导（让默认 ckpt 跟着 --save-root 自动切换到对应 v1/v2 训练产物，
    用户不必再手动同步两个路径）：
    - save_root_hint=None → 兼容老调用：base = checkpoints/goalgen_v1_dit
    - save_root_hint 末尾是 "latest" 或 "run_XXX" → base = parent（用户传 symlink
      或具体 run 时，应在 base 层探测，而不是再在 run 子目录里嵌套找 latest）
    - 其它情况 → base = save_root_hint 自身（假设传的是 base 顶层）

    探测顺序（base 已确定后）：
    1. <base>/latest/best.pt        ← 新布局首选：train.sh 维护的 latest symlink 下的 best
    2. <base>/latest/latest.pt      ← 训练末尾快照（无 val_jsonl 时 best.pt 不存在）
    3. <base>/best.pt               ← 老顶层布局（NO_RUN_SUBDIR=1 或 v2 之前的训练产物）
    4. <base>/latest.pt             ← 同上，老顶层兜底

    都没有时仍返回 (1) 的路径——让 ckpt 加载阶段抛 FileNotFoundError，比在这里
    偷偷往下走更容易排查。
    """

    if save_root_hint:
        hint = pathlib.Path(save_root_hint)
        if hint.name == "latest" or hint.name.startswith("run_"):
            base = hint.parent
        else:
            base = hint
    else:
        base = _AUTOMOT_ROOT / "checkpoints" / "goalgen_v1_dit"

    for candidate in (
        base / "latest" / "best.pt",
        base / "latest" / "latest.pt",
        base / "best.pt",
        base / "latest.pt",
    ):
        if candidate.exists():
            return str(candidate)
    return str(base / "latest" / "best.pt")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="在 val.jsonl 上评测 GoalGen v1/v2 共用 DiT")
    p.add_argument("--val-jsonl", default="checkpoints/goalgen_v1_data/val.jsonl")
    p.add_argument("--dit-checkpoint", default="",
                   help="DiT ckpt 路径。**留空时由 main() 根据 --save-root 自动推**："
                        "<base>/latest/best.pt > <base>/latest/latest.pt > <base>/best.pt > <base>/latest.pt。"
                        "其中 base 是 --save-root 推出来的训练根目录（v1/v2 自动跟随）。"
                        "训练若启用 val_jsonl + epoch save，best.pt = val/loss 最小的那次轻量权重。"
                        "想绑定具体历史 run 直接传 <base>/run_YYYYmmdd_HHMMSS/best.pt。")
    p.add_argument("--checkpoint-dir", default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--save-root", type=str, required=True,
                   help="统一保存根目录（必填，通常与 train.sh OUTPUT_DIR 相同）。"
                        "eval 产物落到 <root>/eval/，TB 落到 <root>/eval_tb/<run_tag>/。")
    p.add_argument("--run-tag", type=str, default="",
                   help="TB run 子目录名，默认从 --dit-checkpoint 推导（ckpt200 / latest 等）。")
    p.add_argument("--no-tb", action="store_true",
                   help="完全关闭 TB 写入（仅保留 stdout + json + perline）。"
                        "GoalGen 一侧 TB 默认开（步骤二 TB 入口）。")
    # ---- 完整 dump 开关（用户最关心的"小样本完整保存"路径）----
    p.add_argument("--full-dump", dest="full_dump",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="是否每条样本完整 dump（inputs/outputs/compare.png/summary.md）。"
                        "默认行为：--max-samples > 0 时开，跑全集（max-samples=0）时关。"
                        "可显式 --full-dump / --no-full-dump 覆盖。")
    p.add_argument("--full-dump-limit", type=int, default=0,
                   help="最多 dump 多少条样本（防止误开后铺满磁盘）。"
                        "0 = 不限（受 --max-samples 限制）。")

    p.add_argument("--qwen-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--vae-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--dit-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    # LoRA / PEFT 适配器（与 train 同口径）：评测必须用与训练同款 Qwen 编码。
    p.add_argument("--qwen-adapter-dir", type=str, default="",
                   help="可选 LoRA / PEFT 适配器目录；为空则跑基础 Qwen。"
                        " 训练若用了适配器，评测也必须传同一个目录。")
    p.add_argument("--qwen-adapter-merge", action="store_true", default=True)
    p.add_argument("--no-qwen-adapter-merge", dest="qwen_adapter_merge", action="store_false")
    p.add_argument("--allow-qwen-adapter-mismatch", action="store_true", default=False,
                   help="允许 DiT ckpt 训练时的 qwen_adapter_dir 与当前 CLI 不一致；"
                        " 仅消融实验使用；默认抛错，防止 KV 分布漂移导致指标不可比。")

    # DiT 几何参数：仅在 ckpt 没存 dit_config 时使用（旧 ckpt 兼容）。
    # 当前共享默认与 train.py 同步：patch=4 / hidden=1024 / n_heads=8。
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--cond-dim", type=int, default=256)
    p.add_argument("--max-history-frames", type=int, default=8)
    p.add_argument("--qwen-kv-segment-mode",
                   choices=["concat_layers", "select_last", "mean"],
                   default="select_last")

    p.add_argument("--max-samples", type=int, default=0,
                   help="0 表示跑完整验证集；正整数会截断。")
    p.add_argument("--euler-steps", type=int, default=32,
                   help="生成 z1_pred 的 Euler 步数；rectified flow 下 32 通常足够。")
    p.add_argument("--cfg-scale", type=float, default=2.0)
    # 与 train 默认对齐：alpha=0.0 用纯噪声起点。设回 1.0 仅做 image-to-image ablation。
    p.add_argument("--z0-prior-alpha", type=float, default=0.0)
    p.add_argument("--z0-prior-sigma", type=float, default=1.0)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--image-dump-count", type=int, default=32,
                   help="前 N 条样本同时落预测 / 真值 PNG，方便人眼对比。")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260530)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    # --dit-checkpoint 留空时由 resolver 推：基于 --save-root 找到对应 base
    # 下的 latest/best.pt（v1/v2 自动跟随训练产物，无需用户两处同步路径）。
    if not args.dit_checkpoint:
        args.dit_checkpoint = _resolve_default_dit_checkpoint(args.save_root)
        print(f"[ckpt] --dit-checkpoint 未指定，自动解析 = {args.dit_checkpoint}")
    eval_loop(args)


if __name__ == "__main__":
    main()
