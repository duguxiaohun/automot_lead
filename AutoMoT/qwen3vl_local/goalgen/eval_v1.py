"""GoalGen v1 离线 eval：在 val.jsonl 上跑 DiT、解码图像、报告四个核心指标。

数据流（与 train_v1 完全同构，只是不反传）：
  history RGB -> frozen Qwen prefill -> segmented KV
  history RGB -> frozen VAE -> z_history
  target keyframe RGB -> frozen VAE -> z1 (GT)
  z0 ~ N(0, I)
  z1_pred = euler_sample(velocity_fn=DiT(.|z_history, KV), num_steps=K) from z0
  pred RGB = VAE.decode(z1_pred)

四个指标（与 5.3 约定一致）：

  (a) latent_mse:   MSE(z1_pred, z1_gt)     —— 与训练 loss 同构，最直接
  (b) latent_cos:   cosine(z1_pred, z1_gt)  —— 向量方向上的相似度，对尺度不敏感
  (c) pixel_l1 / psnr:  VAE.decode 后 [-1,1] RGB 的 L1 + 等价 PSNR
                        —— 量"生成的子目标图像"质量；地板是 VAE 重建误差
  (d) velocity_cos: cosine(v_pred, v_target)，t 在 {0.1, 0.3, 0.5, 0.7, 0.9} 各采 1 次平均
                    —— 训练健康性诊断，跟训练时的 train/cos 同口径

落盘：
  - <out>/eval_v1_summary.json    汇总（mean/std/by_scenario）
  - <out>/eval_v1_perline.jsonl   每条样本一行：scenario / run_id / anchor / status /
                                  subgoal / 四个指标 / 可选 decoded_png_path / gt_png_path
  - <out>/samples/<idx>_pred.png  仅前 --image-dump-count 条
  - <out>/samples/<idx>_gt.png

典型用法（远程，AutoMoT/ 下）：

```bash
python qwen3vl_local/goalgen/eval_v1.py \
  --val-jsonl checkpoints/goalgen_v1_data/val.jsonl \
  --dit-checkpoint checkpoints/goalgen_v1_dit/latest.pt \
  --out-dir eval_json/goalgen_v1 \
  --max-samples 200 \
  --euler-steps 32 \
  --image-dump-count 32
```
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import torch
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
from qwen3vl_local.goalgen.flow import euler_sample, sample_flow_batch  # noqa: E402
from qwen3vl_local.goalgen.qwen_kv import teacher_forced_prefill  # noqa: E402
from qwen3vl_local.goalgen.vae import FrozenVAE, default_vae_paths  # noqa: E402
from qwen3vl_local.prompt_pipeline import DrivingMemory  # noqa: E402


# --------------------------------------------------------------------------- #
# I/O helpers (与 train_v1 等价，复制而非 import 是为了让 eval 文件可独立运行)
# --------------------------------------------------------------------------- #


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

    num_layers 与 language_kv_input_dim 永远取自当前 pooled_kv，这两条是"DiT 形状必须
    与运行时 Qwen KV 对齐"的硬约束；其它字段优先 ckpt 存档。
    """

    payload = torch.load(ckpt_path, map_location=device)
    saved_cfg_dict = payload.get("dit_config") if isinstance(payload, dict) else None

    runtime_kwargs = dict(
        num_layers=len(pooled_kv),
        language_kv_input_dim=language_kv_input_dim_from_pooled(pooled_kv),
    )

    if saved_cfg_dict is not None:
        # 形状预检与 runner 同口径：训练时与运行时的 num_layers / language_kv_input_dim
        # 不一致 → strict load 必炸 attention 投影。提前断言给出双数字 + 原因。
        saved_layers = saved_cfg_dict.get("num_layers")
        saved_lang_dim = saved_cfg_dict.get("language_kv_input_dim")
        runtime_layers = runtime_kwargs["num_layers"]
        runtime_lang_dim = runtime_kwargs["language_kv_input_dim"]
        if saved_layers is not None and saved_layers != runtime_layers:
            raise RuntimeError(
                f"DiT ckpt num_layers={saved_layers}（训练时）≠ 运行时 KV 段数={runtime_layers}。"
                f" 解决：eval 与训练保持同样的 --num-layers。"
            )
        if saved_lang_dim is not None and saved_lang_dim != runtime_lang_dim:
            raise RuntimeError(
                f"DiT ckpt language_kv_input_dim={saved_lang_dim}（训练时 Qwen n_kv*head_dim）"
                f" ≠ 当前 Qwen 推出来的 {runtime_lang_dim}。 解决：用与训练时同款 Qwen 权重。"
            )

        merged = dict(saved_cfg_dict)
        merged.update(runtime_kwargs)
        merged.setdefault("latent_channels", 4)
        cfg = DiTMoTConfig(**merged)
        print(f"[dit] config rebuilt from ckpt dit_config")
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
        print("[dit] WARN: ckpt 无 dit_config，回退 CLI 默认值")

    model = DiTMoT(cfg).to(device=device, dtype=dtype)
    state_dict = payload.get("dit_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"[dit] loaded checkpoint: {ckpt_path}")
    print(
        f"[dit] hidden={cfg.hidden_dim} heads={cfg.n_heads} layers={cfg.num_layers} "
        f"patch={cfg.patch_size} lang_kv_in={cfg.language_kv_input_dim}"
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


# --------------------------------------------------------------------------- #
# 四个指标
# --------------------------------------------------------------------------- #


def latent_mse(z_pred: torch.Tensor, z_gt: torch.Tensor) -> float:
    # 全局 MSE，不分通道；与训练 loss 同口径（只是 v_target 换成 z）。
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
    t_grid: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> float:
    """在 5 个固定 t 各采一次 v_pred / v_target，算余弦相似度的平均值。

    单点 t 噪声大，多点平均更接近"模型在整段轨迹上的 velocity 质量"，跟训练时的
    train/cos 同口径但稳定。固定 t 而非随机，让 eval 之间可比较。
    """
    cosines: List[float] = []
    for t_val in t_grid:
        # 固定 z0：所有 t 用同一份 z0 让 v_target = z1 - z0 也固定，对比维度只剩 t 与 v_pred。
        z0 = torch.randn_like(z1)
        t = torch.full((z1.shape[0],), t_val, device=device, dtype=dtype)
        z_t = (1.0 - t_val) * z0 + t_val * z1
        v_target = z1 - z0
        v_pred = dit(z_t, z_history, t, pooled_kv)
        p = v_pred.float().flatten(1)
        g = v_target.float().flatten(1)
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


@torch.no_grad()
def eval_loop(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GoalGen eval 需要 CUDA；离线机器只能跑数据校验，不能跑 eval。")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"

    samples = load_jsonl(pathlib.Path(args.val_jsonl))
    if not samples:
        raise RuntimeError(f"empty val jsonl: {args.val_jsonl}")
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"[data] val={len(samples)} source={args.val_jsonl}")

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

    # 2) 用第一条样本探一次 KV，反推 DiT shape 并加载 ckpt。
    dit_dtype = dtype_from_name(args.dit_dtype)
    print("[probe] inferring DiT shape from first sample's segmented KV ...")
    probe_pooled = _probe_language_kv(engine, samples[0], args.num_layers, args.qwen_kv_segment_mode)
    dit = build_dit_from_ckpt(
        ckpt_path=pathlib.Path(args.dit_checkpoint).resolve(),
        pooled_kv=probe_pooled,
        args=args,
        device=device,
        dtype=dit_dtype,
    )

    # 3) per-line + 汇总。
    perline_path = out_dir / "eval_v1_perline.jsonl"
    summary_path = out_dir / "eval_v1_summary.json"
    by_scenario: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    all_metrics: List[Dict[str, float]] = []

    with perline_path.open("w", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples):
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
            z_init = torch.randn(z1_gt.shape, device=device, dtype=dit_dtype, generator=gen)
            z1_pred = euler_sample(
                velocity_fn=lambda z, t: dit(z, z_history, t, pooled_kv),
                shape=tuple(z1_gt.shape),
                device=device,
                dtype=dit_dtype,
                num_steps=args.euler_steps,
                z_init=z_init,
            )

            # ---- 四个指标 ----
            m_mse = latent_mse(z1_pred, z1_gt)
            m_cos = latent_cosine(z1_pred, z1_gt)
            # 显式 cast 到 vae (device, dtype)：与 train_v1._decode_latent_to_image 同口径
            # 的 defensive layer，避免未来 vae.py 内部 cast 被删时这里悄悄 dtype mismatch。
            z1_pred_for_vae = z1_pred.to(device=vae.device, dtype=vae.dtype)
            z1_gt_for_vae = z1_gt.to(device=vae.device, dtype=vae.dtype)
            rgb_pred = vae.decode(z1_pred_for_vae).clamp(-1.0, 1.0)
            rgb_gt = vae.decode(z1_gt_for_vae).clamp(-1.0, 1.0)
            m_l1, m_psnr = pixel_l1_psnr(rgb_pred, rgb_gt)
            m_vcos = velocity_cosine_multi_t(dit, z_history, pooled_kv, z1_gt, device, dit_dtype)

            png_paths = _maybe_dump_pair(idx, rgb_pred, rgb_gt, samples_dir, args.image_dump_count)

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
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            all_metrics.append({
                "latent_mse": m_mse,
                "latent_cos": m_cos,
                "pixel_l1": m_l1,
                "psnr": m_psnr,
                "velocity_cos": m_vcos,
            })
            by_scenario[sample.get("scenario", "<unknown>")].append(all_metrics[-1])

            if (idx + 1) % args.log_every == 0 or idx == len(samples) - 1:
                print(
                    f"[eval] {idx + 1}/{len(samples)} "
                    f"latent_mse={m_mse:.6f} latent_cos={m_cos:.4f} "
                    f"pixel_l1={m_l1:.4f} psnr={m_psnr:.2f} v_cos={m_vcos:.4f}"
                )

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

    summary = {
        "config": vars(args),
        "overall": _agg(all_metrics),
        "by_scenario": {s: _agg(ms) for s, ms in sorted(by_scenario.items())},
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] perline={perline_path}")
    print(f"[done] summary={summary_path}")
    print(f"[done] samples png dir={samples_dir}（前 {args.image_dump_count} 条）")
    overall = summary["overall"]
    if overall:
        print(
            f"[overall] latent_mse={overall['latent_mse_mean']:.6f} "
            f"latent_cos={overall['latent_cos_mean']:.4f} "
            f"pixel_l1={overall['pixel_l1_mean']:.4f} "
            f"psnr={overall['psnr_mean']:.2f} "
            f"velocity_cos={overall['velocity_cos_mean']:.4f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate GoalGen v1 DiT on val.jsonl")
    p.add_argument("--val-jsonl", default="checkpoints/goalgen_v1_data/val.jsonl")
    p.add_argument("--dit-checkpoint", default="checkpoints/goalgen_v1_dit/latest.pt")
    p.add_argument("--checkpoint-dir", default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--out-dir", default="eval_json/goalgen_v1")

    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--qwen-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--vae-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--dit-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    # LoRA / PEFT adapter（与 train_v1 同口径）：eval 必须用与训练同款 Qwen 编码。
    p.add_argument("--qwen-adapter-dir", type=str, default="",
                   help="可选 LoRA / PEFT adapter 目录；为空则跑 base Qwen。"
                        " 训练若用了 adapter，eval 也必须传同一个目录。")
    p.add_argument("--qwen-adapter-merge", action="store_true", default=True)
    p.add_argument("--no-qwen-adapter-merge", dest="qwen_adapter_merge", action="store_false")

    # DiT 几何参数：仅在 ckpt 没存 dit_config 时使用（旧 ckpt 兼容）。
    p.add_argument("--patch-size", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--cond-dim", type=int, default=256)
    p.add_argument("--max-history-frames", type=int, default=8)
    p.add_argument("--qwen-kv-segment-mode",
                   choices=["concat_layers", "select_last", "mean"],
                   default="select_last")

    p.add_argument("--max-samples", type=int, default=0,
                   help="0 表示跑完整 val；正整数会截断。")
    p.add_argument("--euler-steps", type=int, default=32,
                   help="生成 z1_pred 的 Euler 步数；rectified flow 下 32 足够。")
    p.add_argument("--image-dump-count", type=int, default=32,
                   help="前 N 条样本同时落 pred / gt PNG，方便人眼对比。")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260530)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    eval_loop(args)


if __name__ == "__main__":
    main()
