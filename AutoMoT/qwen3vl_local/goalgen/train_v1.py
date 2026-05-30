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
from qwen3vl_local.goalgen.flow import euler_sample, flow_matching_loss, sample_flow_batch  # noqa: E402
from qwen3vl_local.goalgen.qwen_kv import teacher_forced_prefill  # noqa: E402
from qwen3vl_local.goalgen.vae import FrozenVAE, default_vae_paths  # noqa: E402
from qwen3vl_local.prompt_pipeline import DrivingMemory  # noqa: E402

# 延迟到运行时再 import：训练机一定有 tb（pytorch 自带），但本地静态检查时可能没装；
# 即便 tb import 失败也不应该挂掉整个 trainer，留 `--no-tb` 作为兜底。
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
        # 默认落到 cuda:0，多个 rank 会抢同一张卡然后挂死。
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    # is_available + is_initialized 双重保护：单卡跑（没 init）也调用这个函数也安全，
    # 不会丢出 "Default process group not initialized" 异常。
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
    # .detach() 切计算图防误传梯度；.float() 把 bf16 / fp16 升回 fp32 再算余弦，
    # 否则低精度下 sum_x2 容易溢出/下溢，dot 值会被截断到 0 或 inf。
    pred = v_pred.detach().float().flatten(1)
    target = v_target.detach().float().flatten(1)
    # flatten(1) 把 [B, C, H, W] 摊成 [B, C*H*W]，沿 dim=1 算余弦相似度，再 batch 平均。
    # 这个指标比 loss 更直观：训练健康时 cos 应该从 ~0 单调升到 ~0.5+，loss 反映得没这么明显。
    return float(F.cosine_similarity(pred, target, dim=1).mean().item())


# --------------------------------------------------------------------------- #
# TensorBoard helpers
# --------------------------------------------------------------------------- #


def _decode_latent_to_image(vae: FrozenVAE, z: torch.Tensor) -> torch.Tensor:
    """Latent → [0,1] 范围 RGB 张量 [B, 3, H, W]，给 tb writer.add_image / add_images 用。

    VAE 解码默认输出 [-1,1]（与训练输入归一化一致），tb 渲染要 [0,1]，所以这里 +1 /2。
    clamp 防止偶发数值出 [-1,1] 把渲染搞糊。
    """

    # 显式 cast 到 vae 自己的 (device, dtype) 作为 defensive layer：
    # 训练 dit_dtype=bf16 而 vae_dtype=fp32 时不 cast 直接喂 vae.decode 会撞 dtype mismatch。
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
    """跑一小撮 val 样本只算 forward loss / cos，不做 backward / euler sample。

    设计取舍：
    - 只在 rank 0 调，所以传 dit_module（DDP 解包后的裸模型），无需 all-reduce；
    - 每条样本走完整 prefill + encode 流程，慢但语义忠实；
    - 取 val_max_samples 上限避免每次 val 时间失控。
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
        batch = sample_flow_batch(z1=z1)
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
    """每 image-log-every step 写一组 pred vs gt 图像并排到 tb。

    image-log-samples 条样本各跑一次 euler_sample（用 image-log-euler-steps 步数），
    再 VAE.decode 得到 pred RGB；与 GT keyframe 直接读盘后归一化对齐做并排比较。
    完整数据流和 eval_v1 一致，所以这里看到的图就是模型当前的"生成能力快照"。
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
        z_init = torch.randn(z1_gt.shape, device=device, dtype=dit_dtype, generator=gen)
        z1_pred = euler_sample(
            velocity_fn=lambda z, t: dit_module(z, z_history, t, pooled_kv),
            shape=tuple(z1_gt.shape),
            device=device,
            dtype=dit_dtype,
            num_steps=args.image_log_euler_steps,
            z_init=z_init,
        )
        pred_imgs.append(_decode_latent_to_image(vae, z1_pred)[0])
        gt_imgs.append(_decode_latent_to_image(vae, z1_gt)[0])

    if pred_imgs:
        # 交错排：pred_0, gt_0, pred_1, gt_1, ... 直接靠 tb 行列布局对比；
        # 这种 layout 比 [all_pred, all_gt] 二段式更便于人眼"对一对一"比对。
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
    step: int,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # step 用 :06d 0-padding 让目录名按字典序自然递增，方便 `ls | sort` 拿最新 ckpt；
    # 如果不 pad，"checkpoint-9" 会排在 "checkpoint-10" 后面。
    target = output_dir / f"checkpoint-{step:06d}"
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
            "args": vars(args),
        },
        target / "goalgen_v1.pt",
    )
    # latest.pt 是"轻量版本"：只存 weights + config，不存 optimizer / scheduler，
    # 给下游 inference / eval 用，避免每次都拖一份几百 MB 的 AdamW state。
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

    # val 集只在 rank 0 用（val + image sample 仅 rank 0 跑），其它 rank 留空省 IO。
    val_samples: List[Dict[str, Any]] = []
    if is_rank0(rank) and args.val_jsonl:
        val_path = pathlib.Path(args.val_jsonl)
        if val_path.exists():
            val_samples = load_jsonl(val_path)[: max(0, args.val_max_samples)]
            print(f"[data] val={len(val_samples)} (cap={args.val_max_samples}) source={val_path}")
        else:
            print(f"[data] WARN: val jsonl 不存在 ({val_path})，跳过 val/sample 记录")

    if is_rank0(rank):
        print(f"[data] train={len(samples)} world_size={world_size}")

    # TensorBoard：只在 rank 0 起 writer，避免多 rank 写同一目录冲突。
    # 写到 output_dir/tb（与 ckpt 同根，按用户 5.1 选项）。
    writer = None
    if is_rank0(rank) and args.tb and _TB_AVAILABLE:
        tb_dir = output_dir / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"[tb] SummaryWriter -> {tb_dir}")
    elif is_rank0(rank) and args.tb and not _TB_AVAILABLE:
        print("[tb] WARN: SummaryWriter import 失败，tb 关闭")

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
    # 可选：挂上 SFT v1 训出来的 LoRA adapter，让 GoalGen prefill 用"微调后的语言编码"。
    # merge=True 把 LoRA 权重合并进 base 矩阵；之后 self.model 上无 PEFT 包装，KV 提取
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
        # 这里只传 dit.parameters()：Qwen / VAE 上面已 freeze_module 关掉 grad，但 optimizer
        # 看到 requires_grad=False 仍会保留它们的 state（占显存）。显式只传 DiT 参数能
        # 把 AdamW 的 m/v state 也限制在 DiT 上，省一份 Qwen 量级的优化器内存。
        dit.parameters(),
        lr=args.learning_rate,
        # betas=(0.9, 0.95) 是 DiT / 大型 diffusion 模型的常见配方；第二阶矩衰减比 Adam
        # 默认 0.999 快，对 latent flow matching 这种损失曲线较平的目标更稳。
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    # 把样本数夹到 world_size 整除，让每个 rank 拿到等长 shard；不夹断会出现"某 rank 多
    # backward 一次"，DDP all-reduce 等不到对应张量进而挂死。
    usable_per_epoch = (len(samples) // world_size) * world_size
    if usable_per_epoch <= 0:
        raise RuntimeError(f"dataset too small for world_size={world_size}: {len(samples)} samples")
    # 每 rank 单 epoch 的 optimizer step 数 = shard 样本数 / grad_accum_steps，向上取整。
    # ceil 而不是 floor 是为了让"最后不满一个 accum 组也能 step 一次"，否则尾部样本
    # 算完梯度却不更新，浪费 forward。
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

    # DDP 解包：val / image sample 用 dit_module 直接 forward，不经过 DDP hooks，
    # 既省 all-reduce 又避免 module.training=True 期间 dropout 等差异影响诊断。
    dit_module = dit.module if hasattr(dit, "module") else dit

    try:
        for epoch in range(args.num_epochs):
            order = list(range(len(samples)))
            # 每 epoch 用 seed+epoch 重洗：保证不同 epoch 见到的顺序不同（防止周期性过拟合），
            # 又保证同一份 seed + 同一台机器复现完全一致——所有 rank 用相同 seed 算出相同 order，
            # 才能保证下面 order[rank::world_size] 切出的 shard 互不重叠且无遗漏。
            random.Random(args.seed + epoch).shuffle(order)
            # 砍到 usable_per_epoch（world_size 整除）：避免尾部样本造成 rank 间 shard 长度差 1，
            # 那会让 DDP 在最后一个 step 卡死等不到对应张量的 all-reduce。
            order = order[:usable_per_epoch]
            # 步长 world_size 跳取：rank 0 拿 [0, W, 2W, ...]，rank 1 拿 [1, W+1, ...]，
            # 每个 sample_idx 只会被一个 rank 处理；比 chunk-by-chunk 切分对 NFS 缓存更友好
            # （相邻 rank 的样本来自相邻 run 的概率低，scattered read 反而能让磁盘并行加载）。
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
                # DiT.forward 会在 zip(blocks, pooled_kv) 时静默错位（旧版会 silent，新版会 raise）。
                prefill = teacher_forced_prefill(
                    engine=engine,
                    memory=memory,
                    images=history_images,
                    num_segments=args.num_layers,
                    kv_segment_mode=args.qwen_kv_segment_mode,
                )
                # KV 来自 Qwen，dtype 可能是 bf16 / fp16；DiT 内部走 dit_dtype（默认 bf16）。
                # 显式 .to 强制对齐：不齐时 SDPA 会在 attention 内部 raise dtype mismatch，
                # 错误堆栈在 C++ 端不好定位，所以这里 forward 之前就把语言 KV 搬到目标 device + dtype。
                pooled_kv = [
                    (k.to(device=device, dtype=dit_dtype), v.to(device=device, dtype=dit_dtype))
                    for k, v in prefill.pooled_kv
                ]

                # VAE 默认 fp32 输出（vae_only.yaml 关了 autocast），.to(dit_dtype) 才能和 DiT 对齐。
                # .unsqueeze(0)：vae.encode 返回 [F, 4, 48, 144]（F 是历史帧数），加 batch 维变 [1, F, ...]，
                # DiT 的 forward 期望 [B, F, C, H, W]，这里 B=1（per-rank batch）。
                z_history = vae.encode(history_images).to(dtype=dit_dtype).unsqueeze(0)
                # 目标帧只有一张，encode 返回 [1, 4, 48, 144] 直接当作 z1（[B, C, H, W]），不需要再加维。
                z1 = vae.encode([target_img]).to(dtype=dit_dtype)
                # 在 z1 上采 z0 / t / 计算 z_t、v_target。z0 / t 留给 flow.py 内部默认采样，
                # 这里不传是为了让每条样本独立采，跟其他样本的随机性解耦。
                batch = sample_flow_batch(z1=z1)

                v_pred = dit(batch.z_t, z_history, batch.t, pooled_kv)
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
                    # backward 仍照常累积本地 .grad 但不触发跨 rank 同步；总 all-reduce 次数
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
                # 早期发现"prompt 不知不觉变长"导致 KV/显存膨胀，也能确认 builder 没漏帧。
                running_kv_seq_len += int(prefill.seq_len)
                accum += 1

                if will_step:
                    if args.max_grad_norm > 0:
                        # clip_grad_norm_ 必须在 step 之前调；flow matching 初期 v_target 数值可能
                        # 很大（z1 - z0 在 latent 尺度上方差 ~2），不裁剪偶发会让 lr=1e-4 也炸。
                        # 返回值是 clip 前的全局梯度范数，写进 tb 用来诊断"训练是否在快炸"。
                        last_grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(dit.parameters(), args.max_grad_norm)
                        )
                    optimizer.step()
                    scheduler.step()
                    # set_to_none=True 比 zero_(0) 快：让 .grad = None，下次 backward 第一次写
                    # 直接分配新张量，省一次"已分配张量清零"的 kernel。
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if is_rank0(rank) and global_step % args.logging_steps == 0:
                        denom = max(1, running_micro)
                        avg_loss = running_loss / denom
                        avg_cos = running_cos / denom
                        avg_kv_seq = running_kv_seq_len / denom
                        cur_lr = scheduler.get_last_lr()[0]
                        print(
                            f"[train] epoch={epoch} step={global_step}/{total_steps} "
                            f"loss={avg_loss:.6f} cos={avg_cos:.4f} "
                            f"grad_norm={last_grad_norm:.3f} kv_seq={avg_kv_seq:.0f} "
                            f"lr={cur_lr:.3e}"
                        )
                        if writer is not None:
                            # 标量分组：train/* 用于训练曲线；diag/* 用于诊断指标。
                            # 这种命名让 tb 左侧 tag 树自动分组，不会被几十个指标淹没。
                            writer.add_scalar("train/loss", avg_loss, global_step)
                            writer.add_scalar("train/cos", avg_cos, global_step)
                            writer.add_scalar("train/lr", cur_lr, global_step)
                            writer.add_scalar("diag/grad_norm", last_grad_norm, global_step)
                            writer.add_scalar("diag/kv_seq_len", avg_kv_seq, global_step)
                        running_loss = 0.0
                        running_cos = 0.0
                        running_micro = 0
                        running_kv_seq_len = 0

                    # val 评估：每 val_steps 跑一次 val 子集（仅 rank 0，使用解包后的 dit_module）。
                    # 失败不阻断训练——val 集脏数据 / 单条 OOM 都先 log 再继续。
                    if (
                        is_rank0(rank)
                        and writer is not None
                        and val_samples
                        and args.val_steps > 0
                        and global_step % args.val_steps == 0
                    ):
                        try:
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
                            print(f"[val] WARN: val pass 出错，跳过：{e}")

                    # 图像 sample：每 image_log_every 步生成一组 pred vs gt 落 tb，
                    # 是判断"loss 在降但生成质量是否真的在改善"的关键可视化。
                    # 由于含一次 euler_sample（默认 32 步），频率不宜过高。
                    if (
                        is_rank0(rank)
                        and writer is not None
                        and val_samples
                        and args.image_log_every > 0
                        and global_step % args.image_log_every == 0
                    ):
                        try:
                            _log_image_samples(
                                writer, engine, vae, dit_module, val_samples,
                                args, device, dit_dtype, global_step,
                            )
                        except Exception as e:
                            print(f"[image] WARN: image sample 失败，跳过：{e}")

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
        # writer 必须在 cleanup_distributed 之前 close，避免 tb 写最后一笔时进程组已挂。
        if writer is not None:
            writer.close()
        cleanup_distributed()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train GoalGen v1 DiT-MoT")
    p.add_argument("--train-jsonl", default="checkpoints/goalgen_v1_data/train.jsonl")
    p.add_argument("--checkpoint-dir", default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--output-dir", default="checkpoints/goalgen_v1_dit")
    p.add_argument("--qwen-dtype", choices=["bfloat16", "float16", "float32", "auto"], default="bfloat16")
    p.add_argument("--vae-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--dit-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    # LoRA / PEFT adapter：空字符串表示不挂；否则在 engine.load() 之后 attach。
    # merge 默认开启把 LoRA 合进 base 矩阵，省 prefill 推理时间。
    p.add_argument("--qwen-adapter-dir", type=str, default="",
                   help="可选 LoRA / PEFT adapter 目录；为空则跑 base Qwen。")
    p.add_argument("--qwen-adapter-merge", action="store_true", default=True,
                   help="挂 adapter 后立即 merge_and_unload；默认开。")
    p.add_argument("--no-qwen-adapter-merge", dest="qwen_adapter_merge", action="store_false",
                   help="保留 PeftModel 包装不合并（debug LoRA 自身行为用）。")

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

    # ------ TensorBoard / val / image sample ------
    p.add_argument("--tb", action="store_true", default=True,
                   help="rank 0 写 TensorBoard 到 output_dir/tb；--no-tb 关掉。")
    p.add_argument("--no-tb", dest="tb", action="store_false",
                   help="完全关闭 tb 写入（仅保留 stdout 日志）。")
    p.add_argument("--val-jsonl", type=str, default="",
                   help="val jsonl 路径；非空时按 --val-steps 间隔跑 val/loss + val/cos 并落 tb。")
    p.add_argument("--val-steps", type=int, default=500,
                   help="每多少 optimizer step 跑一次 val；0 关闭。")
    p.add_argument("--val-max-samples", type=int, default=64,
                   help="val 集每次最多取多少条样本，防止 val 时间失控。")
    p.add_argument("--image-log-every", type=int, default=500,
                   help="每多少 optimizer step 落一组 pred vs gt 图像；0 关闭。")
    p.add_argument("--image-log-samples", type=int, default=4,
                   help="每次落几张图（也是 euler_sample 调用次数，越多越慢）。")
    p.add_argument("--image-log-euler-steps", type=int, default=32,
                   help="image sample 用的 Euler 步数；32 在 rectified flow 下通常够用。")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
