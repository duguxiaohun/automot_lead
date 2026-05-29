"""Train GoalGen v1 DiT-MoT on jsonl produced by build_dataset_v1.py.

This trainer intentionally stays small and explicit:
- Qwen3-VL-Instruct is frozen and only used for teacher-forced prefill.
- VAE is frozen and only used for history/target latent encoding.
- DiT-MoT is the only trainable module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Dict, List

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
    language_kv_input_dim_from_pooled,
)
from qwen3vl_local.goalgen.flow import flow_matching_loss, sample_flow_batch  # noqa: E402
from qwen3vl_local.goalgen.qwen_kv import teacher_forced_prefill  # noqa: E402
from qwen3vl_local.goalgen.vae import FrozenVAE, default_vae_paths  # noqa: E402
from qwen3vl_local.prompt_pipeline import DrivingMemory  # noqa: E402


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rgb(path: str) -> Image.Image:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"RGB image not found: {p}")
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


def build_dit(args: argparse.Namespace, language_kv_input_dim: int) -> DiTMoT:
    """构造 DiT-MoT。

    language_kv_input_dim 由调用方根据实际 segmented KV 推出，避免硬编码 1024 在换
    base 模型时直接撞 attention shape mismatch。
    """

    cfg = DiTMoTConfig(
        latent_channels=4,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        num_layers=args.num_layers,
        cond_dim=args.cond_dim,
        language_kv_input_dim=language_kv_input_dim,
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


def _probe_language_kv_dim(
    engine: LocalQwen3VLInstructEngine,
    samples: List[Dict[str, Any]],
    num_segments: int,
    kv_segment_mode: str,
) -> int:
    """跑第一条样本 prefill 拿真实 segmented KV，反推 language_kv_input_dim。

    这避免了把 `n_kv_heads * head_dim`（Qwen3-VL-4B-Instruct = 8*128 = 1024）
    硬编码到 CLI，使得换 base 模型时不需要手动改参数；同时让 build_dit 的输入
    维度永远与下游 forward 一致。
    """

    sample = samples[0]
    history_images = [load_rgb(p) for p in sample["history_rgb_paths"]]
    memory = memory_from_sample(sample)
    probe = teacher_forced_prefill(
        engine=engine,
        memory=memory,
        images=history_images,
        num_segments=num_segments,
        kv_segment_mode=kv_segment_mode,
    )
    return language_kv_input_dim_from_pooled(probe.pooled_kv)


def cosine_velocity(v_pred: torch.Tensor, v_target: torch.Tensor) -> float:
    pred = v_pred.detach().float().flatten(1)
    target = v_target.detach().float().flatten(1)
    return float(F.cosine_similarity(pred, target, dim=1).mean().item())


def save_checkpoint(
    output_dir: pathlib.Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"checkpoint-{step:06d}"
    target.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "step": step,
            "dit_state_dict": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "dit_config": asdict(module.cfg),
            "args": vars(args),
        },
        target / "goalgen_v1.pt",
    )
    latest = output_dir / "latest.pt"
    torch.save(
        {
            "step": step,
            "dit_state_dict": module.state_dict(),
            "dit_config": asdict(module.cfg),
            "args": vars(args),
        },
        latest,
    )


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


def train(args: argparse.Namespace) -> None:
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GoalGen training expects CUDA; use dataset builder locally, train remotely.")

    output_dir = pathlib.Path(args.output_dir)
    samples = load_jsonl(pathlib.Path(args.train_jsonl))
    if not samples:
        raise RuntimeError(f"empty train jsonl: {args.train_jsonl}")

    if is_rank0(rank):
        print(f"[data] train={len(samples)} world_size={world_size}")

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
    # 显式冻结 Qwen 全部参数。即便 optimizer 没传 Qwen 参数 + prefill 在 no_grad
    # 上下文里，这道保险能防止未来误用 .parameters() 把 Qwen 喂进 optimizer。
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

    # 解析 language_kv_input_dim：默认 auto -> 用第一条样本做一次 dummy prefill 推维度。
    # 给 int 时按 CLI 值走（用户明确知道 base 模型的 n_kv_heads * head_dim 想跳过 probe 时用）。
    if isinstance(args.language_kv_input_dim, str) and args.language_kv_input_dim.lower() == "auto":
        if is_rank0(rank):
            print("[probe] inferring language_kv_input_dim from first sample's segmented KV ...")
        language_kv_dim = _probe_language_kv_dim(
            engine,
            samples,
            args.num_layers,
            args.qwen_kv_segment_mode,
        )
        if is_rank0(rank):
            print(f"[probe] language_kv_input_dim={language_kv_dim}")
    else:
        language_kv_dim = int(args.language_kv_input_dim)

    dit_dtype = dtype_from_name(args.dit_dtype)
    dit = build_dit(args, language_kv_input_dim=language_kv_dim).to(device=device, dtype=dit_dtype)
    if world_size > 1:
        dit = torch.nn.parallel.DistributedDataParallel(dit, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        dit.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    usable_per_epoch = (len(samples) // world_size) * world_size
    if usable_per_epoch <= 0:
        raise RuntimeError(f"dataset too small for world_size={world_size}: {len(samples)} samples")
    steps_per_epoch = max(1, math.ceil((usable_per_epoch / world_size) / args.grad_accum_steps))
    total_steps = max(1, steps_per_epoch * args.num_epochs)
    if args.max_train_steps > 0:
        total_steps = min(total_steps, args.max_train_steps)
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_ratio)

    global_step = 0
    running_loss = 0.0
    running_cos = 0.0
    running_micro = 0
    accum = 0

    try:
        for epoch in range(args.num_epochs):
            order = list(range(len(samples)))
            random.Random(args.seed + epoch).shuffle(order)
            order = order[:usable_per_epoch]
            shard = order[rank::world_size]

            for local_idx, sample_idx in enumerate(shard):
                sample = samples[sample_idx]
                history_images = [load_rgb(p) for p in sample["history_rgb_paths"]]
                target_img = load_rgb(sample["target_rgb_path"])
                memory = memory_from_sample(sample)

                micro_pos = (local_idx % args.grad_accum_steps) + 1
                micro_start = local_idx - (micro_pos - 1)
                micro_end = min(micro_start + args.grad_accum_steps, len(shard))
                micro_group_size = micro_end - micro_start
                will_step = micro_pos == micro_group_size

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
                batch = sample_flow_batch(z1=z1)

                v_pred = dit(batch.z_t, z_history, batch.t, pooled_kv)
                loss_raw = flow_matching_loss(v_pred, batch.v_target)
                loss = loss_raw / micro_group_size

                # DDP 优化：grad accum 期间的 micro-step 不做 all-reduce，仅累积本地梯度；
                # 只在最后一个 micro-step 让 backward 触发 all-reduce 一次。
                # 单卡或非 grad-accum 边界以外用 nullcontext，行为与原版一致。
                if world_size > 1 and not will_step:
                    sync_ctx = dit.no_sync()
                else:
                    sync_ctx = nullcontext()
                with sync_ctx:
                    loss.backward()

                running_loss += float(loss_raw.detach().item())
                running_cos += cosine_velocity(v_pred, batch.v_target)
                running_micro += 1
                accum += 1

                if will_step:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(dit.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if is_rank0(rank) and global_step % args.logging_steps == 0:
                        denom = max(1, running_micro)
                        print(
                            f"[train] epoch={epoch} step={global_step}/{total_steps} "
                            f"loss={running_loss / denom:.6f} "
                            f"cos={running_cos / denom:.4f} "
                            f"lr={scheduler.get_last_lr()[0]:.3e}"
                        )
                        running_loss = 0.0
                        running_cos = 0.0
                        running_micro = 0

                    if is_rank0(rank) and args.save_steps > 0 and global_step % args.save_steps == 0:
                        save_checkpoint(output_dir, dit, optimizer, scheduler, global_step, args)

                    if global_step >= total_steps:
                        break
            if global_step >= total_steps:
                break

        if is_rank0(rank):
            save_checkpoint(output_dir, dit, optimizer, scheduler, global_step, args)
            print(f"[done] saved GoalGen DiT under {output_dir}")
    finally:
        cleanup_distributed()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train GoalGen v1 DiT-MoT")
    p.add_argument("--train-jsonl", default="checkpoints/goalgen_v1_data/train.jsonl")
    p.add_argument("--checkpoint-dir", default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--output-dir", default="checkpoints/goalgen_v1_dit")
    p.add_argument("--qwen-dtype", choices=["bfloat16", "float16", "float32", "auto"], default="bfloat16")
    p.add_argument("--vae-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--dit-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")

    p.add_argument("--patch-size", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--cond-dim", type=int, default=256)
    p.add_argument("--max-history-frames", type=int, default=8,
                   help="DiT 可接收的最大历史 latent 帧数；builder 默认 4 帧。")
    p.add_argument("--qwen-kv-segment-mode",
                   choices=["concat_layers", "select_last", "mean"],
                   default="select_last",
                   help="select_last 每段只取最后一层 Qwen KV，省显存（默认）；"
                        "concat_layers 把 3 层 token 维拼起来（重，ablation 用）；mean 为旧版层平均。")
    # str 类型 + "auto" 默认值：trainer 启动后先跑一条 sample 探测真实 KV 维度。
    # 想固定到具体整数（跳过 probe）也支持，传 `--language-kv-input-dim 1024` 即可。
    p.add_argument("--language-kv-input-dim", type=str, default="auto",
                   help='"auto" 时从首条样本 segmented KV 推导；或传具体整数（如 1024）跳过 probe。')

    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.02)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--max-train-steps", type=int, default=0,
                   help="0 means run num_epochs; positive value caps optimizer steps.")
    p.add_argument("--seed", type=int, default=20260529)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
