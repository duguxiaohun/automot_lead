"""GoalGen v1/v2 case-level probe — 随机选 N 个场景的样本，dump 输入+生成+真值+诊断。

eval.py 给的是聚合视角（mean/std/by_scenario）；probe 给的是单条样本视角：
- 历史 RGB 全图（symlink）
- VAE 重建的 history reference（rank0 一次 encode→decode，看 VAE 自身重建损失）
- 真值 keyframe RGB + 真值经 VAE encode→decode 的 reference
- DiT 在固定 seed 下 Euler 采样的 z1_pred + 解码后的 pred RGB
- per-step euler 轨迹（每个时间步 t 上的 v_pred vs v_target cosine 序列）
- memory.json（DrivingMemory dump，scenario / event_sequence / status / subgoal / completed_events）
- metrics.json（latent_mse / latent_cos / pixel_l1 / psnr / velocity_cos）
- meta.json（dit_checkpoint / qwen_adapter / euler_steps / 推理耗时）
- overview.md（一页 markdown 把上述全部串起来）

输出布局（与 train.sh 同根 — OUTPUT_DIR 平铺）：
  <save_root>/eval_cases/<scenario>__<run_id>__<anchor>/
    input_history/00.jpg ... 03.jpg
    target_raw.jpg          (真值 keyframe 原图)
    target_vae_recon.png    (真值 VAE encode→decode；看 VAE 自身重建上限)
    pred.png                (DiT 采样 + VAE 解码)
    euler_trace.json        ({"t": [...], "v_cos": [...]} per-step）
    memory.json
    metrics.json
    meta.json
    overview.md

不接 torchrun（与 probe_sft_v1.py 同理：per-sample 顺序写文件更直观）。
默认自动挑 1 张空闲 GPU，并覆盖外层残留的 `CUDA_VISIBLE_DEVICES`。

典型用法（**从 AutoMoT/ 目录运行**）：

```bash
python qwen3vl_local/goalgen/probe.py \
  --dit-checkpoint checkpoints/goalgen_v1_dit/latest.pt \
  --save-root checkpoints/goalgen_v1_dit \
  --num-per-scenario 4 --seed 0

# 同 seed 跑两个不同 ckpt + 不同 case-suffix，目录不互相覆盖，便于人工对比
python qwen3vl_local/goalgen/probe.py \
  --dit-checkpoint checkpoints/goalgen_v1_dit/checkpoint-000500/goalgen_v1.pt \
  --save-root checkpoints/goalgen_v1_dit \
  --num-per-scenario 4 --seed 0 --case-suffix "_ckpt500"
```
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
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


def _cli_has(name: str) -> bool:
    return any(item == name or item.startswith(name + "=") for item in sys.argv[1:])


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


def _maybe_set_idle_gpu_mask() -> None:
    """probe 默认自动挑 1 张空闲 GPU；显式 --gpu N 时不覆盖 CUDA_VISIBLE_DEVICES。"""
    if _cli_has("--gpu"):
        return
    selected = _pick_idle_gpus(1)
    if selected:
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        print(
            f"[gpu] auto selected idle CUDA_VISIBLE_DEVICES={selected}; "
            f"process uses cuda:0; previous={previous or '<unset>'}"
        )


_maybe_set_idle_gpu_mask()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from qwen3vl_local.engine import LocalQwen3VLInstructEngine  # noqa: E402
from qwen3vl_local.goalgen.qwen_kv import teacher_forced_prefill  # noqa: E402
from qwen3vl_local.goalgen.vae import FrozenVAE, default_vae_paths  # noqa: E402
from qwen3vl_local.prompt_pipeline import DrivingMemory  # noqa: E402

# 直接复用 eval 里已经写好的 ckpt 反向构建 DiT、probe KV、score helper，
# 不重复实现。
from qwen3vl_local.goalgen.eval import (  # noqa: E402
    build_dit_from_ckpt,
    dtype_from_name,
    latent_cosine,
    latent_mse,
    load_jsonl,
    load_rgb,
    memory_from_sample,
    pixel_l1_psnr,
    velocity_cosine_multi_t,
    _probe_language_kv,
    _load_vae_latent_stats_from_ckpt,
    _make_z_init_from_prior,
    _save_rgb_png,
)


# --------------------------------------------------------------------------- #
# 样本挑选
# --------------------------------------------------------------------------- #

def select_samples(
    samples: List[Dict[str, Any]],
    scenarios: Optional[List[str]],
    num_per_scenario: int,
    seed: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    by_scenario: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, s in enumerate(samples):
        sc = s.get("scenario", "<unknown>")
        if scenarios and sc not in scenarios:
            continue
        by_scenario[sc].append((idx, s))
    if not by_scenario:
        raise RuntimeError(f"未找到匹配场景的样本：{scenarios}")
    rng = random.Random(seed)
    picked: List[Tuple[int, Dict[str, Any]]] = []
    for sc in sorted(by_scenario.keys()):
        bucket = by_scenario[sc]
        rng.shuffle(bucket)
        picked.extend(bucket[:num_per_scenario])
    return picked


def link_or_copy(src: str, dst: pathlib.Path) -> None:
    src_path = pathlib.Path(src)
    if not src_path.exists():
        print(f"[probe][warn] 源图不存在，跳过：{src_path}")
        return
    if dst.exists():
        dst.unlink()
    try:
        os.symlink(src_path.resolve(), dst)
    except (OSError, NotImplementedError):
        shutil.copyfile(src_path, dst)


# --------------------------------------------------------------------------- #
# 带轨迹记录的 Euler 采样
# --------------------------------------------------------------------------- #

@torch.no_grad()
def euler_sample_with_trace(
    dit: torch.nn.Module,
    z_history: torch.Tensor,
    pooled_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    z1_gt: torch.Tensor,
    z_init: torch.Tensor,
    num_steps: int,
    cfg_scale: float,
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """跑 Euler 采样的同时记录每一步的 v_cos 与 z_t L2 距离 GT 的轨迹。

    与 flow.euler_sample 数值等价；不直接复用是因为这里要在内层每一步抽出 v_pred
    与 v_target 算 cosine。
    """
    device = z_init.device
    dtype = z_init.dtype
    z_t = z_init.clone()
    dt = 1.0 / num_steps
    trace: Dict[str, List[float]] = {"t": [], "v_cos_vs_gt_direction": [], "z_l2_to_gt": []}
    for step in range(num_steps):
        t_val = step * dt
        t = torch.full((z_t.shape[0],), t_val, device=device, dtype=dtype)
        v_cond = dit(z_t, z_history, t, pooled_kv, force_uncond=False)
        v_uncond = dit(z_t, z_history, t, pooled_kv, force_uncond=True)
        v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        # 拿"真值方向" v_target_gt = z1_gt - z_init 与当前 v_pred 算 cosine（只是诊断，
        # 不参与生成）。flow matching 中 v_target = z1 - z0 是常量，所以同一个样本
        # 32 步的 v_cos_vs_gt 都用同一个分母 reference，曲线漂移=模型走偏。
        v_ref = (z1_gt - z_init).float().flatten(1)
        v_p = v_pred.float().flatten(1)
        cos = float(F.cosine_similarity(v_p, v_ref, dim=1).mean().item())
        l2 = float((z_t.float() - z1_gt.float()).flatten(1).norm(dim=1).mean().item())
        trace["t"].append(t_val)
        trace["v_cos_vs_gt_direction"].append(cos)
        trace["z_l2_to_gt"].append(l2)
        z_t = z_t + v_pred * dt
    return z_t, trace


# --------------------------------------------------------------------------- #
# Overview markdown
# --------------------------------------------------------------------------- #

def render_overview_md(
    case_dir: pathlib.Path,
    sample: Dict[str, Any],
    memory: DrivingMemory,
    metrics: Dict[str, float],
    trace: Dict[str, List[float]],
    meta: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append(f"# Case: {sample.get('scenario')}/{sample.get('run_id')} anchor={sample.get('anchor')}")
    lines.append("")
    lines.append(f"- val.jsonl sample_idx: **{meta.get('sample_idx')}**")
    lines.append(f"- target_frame: {sample.get('target_frame')}")
    lines.append(f"- dit_checkpoint: `{meta.get('dit_checkpoint')}`")
    lines.append(f"- qwen_adapter_dir: `{meta.get('qwen_adapter_dir') or '<base>'}`")
    lines.append(f"- euler_steps: {meta.get('euler_steps')}, seed: {meta.get('seed')}")
    lines.append(f"- inference elapsed: {meta.get('elapsed_sec'):.3f}s")
    lines.append("")

    lines.append("## Memory (driving state)")
    lines.append("```json")
    lines.append(json.dumps(asdict(memory), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Input history")
    for k, p in enumerate(sample.get("history_rgb_paths", [])):
        lines.append(f"- `input_history/{k:02d}.jpg` ← `{p}`")
    lines.append("")
    lines.append("## Files")
    lines.append("- `target_raw.jpg` ← `" + str(sample.get("target_rgb_path")) + "` (真值原图)")
    lines.append("- `target_vae_recon.png` (真值经 VAE encode→decode；做生成质量的天花板对比)")
    lines.append("- `pred.png` (DiT Euler 采样 + VAE 解码出的预测子目标)")
    lines.append("")

    lines.append("## Metrics")
    lines.append("| metric | value | 说明 |")
    lines.append("|---|---|---|")
    lines.append(f"| latent_mse | {metrics['latent_mse']:.6f} | MSE(z1_pred, z1_gt)，与训练损失同口径 |")
    lines.append(f"| latent_cos | {metrics['latent_cos']:.4f} | cosine 越接近 1 越准 |")
    lines.append(f"| pixel_l1   | {metrics['pixel_l1']:.4f} | 解码 RGB L1 |")
    lines.append(f"| psnr       | {metrics['psnr']:.2f} | 越高越好；地板 = VAE 重建 PSNR |")
    lines.append(f"| velocity_cos | {metrics['velocity_cos']:.4f} | 5 个 t 上平均，训练健康度同口径 |")
    lines.append("")

    if trace.get("t"):
        lines.append("## Euler trace (per-step 诊断)")
        lines.append("`v_cos_vs_gt_direction` 是 v_pred 与真值方向 (z1_gt - z_init) 的 cosine；越接近 1 说明模型每步都在朝目标方向走。")
        lines.append("`z_l2_to_gt` 是当前 z_t 到 z1_gt 的 L2 距离；理想轨迹应单调下降。")
        lines.append("```")
        lines.append("step  t      v_cos    z_l2")
        for k in range(0, len(trace["t"]), max(1, len(trace["t"]) // 8)):
            lines.append(
                f"{k:4d}  {trace['t'][k]:.3f}  {trace['v_cos_vs_gt_direction'][k]:+.4f}  {trace['z_l2_to_gt'][k]:.2f}"
            )
        lines.append("```")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def _resolve_default_dit_checkpoint(save_root_hint: Optional[str] = None) -> str:
    """根据 --save-root 推 base 目录，再按"latest 子目录 > 老顶层"顺序找 ckpt。

    与 eval.py 同名函数保持完全一致的语义，**不 import 复用**以避免触发
    eval 的模块级副作用（GPU mask 检测、依赖加载等）。维护时两边同步改。

    base 推导：
    - save_root_hint=None → base = checkpoints/goalgen_v1_dit（老兼容）
    - save_root_hint 末尾 "latest" 或 "run_XXX" → base = parent（用户绑定 symlink/具体 run）
    - 其它 → base = save_root_hint（base 顶层）

    探测顺序：<base>/latest/best.pt > <base>/latest/latest.pt >
              <base>/best.pt > <base>/latest.pt；都缺时返回 (1) 让加载阶段抛错。
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


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="GoalGen v1/v2 case-level probe（随机场景 dump）")
    parser.add_argument("--val-jsonl", default="checkpoints/goalgen_v1_data/val.jsonl")
    parser.add_argument("--dit-checkpoint", default="",
                        help="DiT ckpt 路径。**留空时由 main() 根据 --save-root 自动推**："
                             "<base>/latest/best.pt > <base>/latest/latest.pt > "
                             "<base>/best.pt > <base>/latest.pt（v1/v2 自动跟随 save-root）。"
                             "想绑定具体历史 run 直接传 <base>/run_YYYYmmdd_HHMMSS/best.pt。")
    parser.add_argument("--checkpoint-dir", default="checkpoints/Qwen3-VL-4B-Instruct")
    parser.add_argument("--save-root", default="checkpoints/goalgen_v1_dit",
                        help="case dump 落到 <save-root>/eval_cases/<scenario>__<run>__<anchor>/")
    parser.add_argument("--case-suffix", default="",
                        help="给 case 目录名加后缀，便于多 ckpt 同 seed 并排对比。")
    parser.add_argument("--scenarios", default="",
                        help="逗号分隔过滤；空 = 全场景。")
    parser.add_argument("--num-per-scenario", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--qwen-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--vae-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--dit-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--qwen-adapter-dir", default="",
                        help="可选 LoRA / PEFT 适配器目录；必须与训练 DiT 时一致。")
    parser.add_argument("--qwen-adapter-merge", action="store_true", default=True)
    parser.add_argument("--no-qwen-adapter-merge", dest="qwen_adapter_merge", action="store_false")
    parser.add_argument("--allow-qwen-adapter-mismatch", action="store_true", default=False)
    parser.add_argument("--euler-steps", type=int, default=32)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--z0-prior-alpha", type=float, default=1.0)
    parser.add_argument("--z0-prior-sigma", type=float, default=1.0)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    # 当前共享默认对齐 Qwen3-VL-4B-Instruct 的 (n_kv_heads=8, head_dim=128)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--max-history-frames", type=int, default=8)
    parser.add_argument("--qwen-kv-segment-mode",
                        choices=["concat_layers", "select_last", "mean"],
                        default="select_last")
    args = parser.parse_args()
    # --dit-checkpoint 留空时根据 --save-root 自动推（v1/v2 自动跟随训练产物）。
    if not args.dit_checkpoint:
        args.dit_checkpoint = _resolve_default_dit_checkpoint(args.save_root)
        print(f"[ckpt] --dit-checkpoint 未指定，自动解析 = {args.dit_checkpoint}")

    samples = load_jsonl(pathlib.Path(args.val_jsonl))
    scenarios_filter = [s.strip() for s in args.scenarios.split(",") if s.strip()] or None
    picked = select_samples(samples, scenarios_filter, args.num_per_scenario, args.seed)
    print(f"[probe] selected {len(picked)} samples from {len(samples)} total")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GoalGen probe 需要 CUDA。")

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

    dit_dtype = dtype_from_name(args.dit_dtype)
    probe_pooled = _probe_language_kv(engine, samples[0], args.num_layers, args.qwen_kv_segment_mode)
    dit = build_dit_from_ckpt(
        ckpt_path=pathlib.Path(args.dit_checkpoint).resolve(),
        pooled_kv=probe_pooled,
        args=args,
        device=device,
        dtype=dit_dtype,
    )

    case_root = pathlib.Path(args.save_root) / "eval_cases"
    case_root.mkdir(parents=True, exist_ok=True)
    index_records: List[Dict[str, Any]] = []

    for sample_idx, sample in picked:
        scenario = sample.get("scenario", "unknown")
        run_id = sample.get("run_id", "norun")
        anchor = sample.get("anchor", "noanchor")
        case_name = f"{scenario}__{run_id}__{anchor}{args.case_suffix}"
        case_dir = case_root / case_name
        (case_dir / "input_history").mkdir(parents=True, exist_ok=True)

        # 1) symlink 历史 + target raw
        for k, p in enumerate(sample.get("history_rgb_paths", [])):
            link_or_copy(p, case_dir / "input_history" / f"{k:02d}.jpg")
        if sample.get("target_rgb_path"):
            link_or_copy(sample["target_rgb_path"], case_dir / "target_raw.jpg")

        # 2) 推理
        history_images = [load_rgb(p) for p in sample["history_rgb_paths"]]
        target_img = load_rgb(sample["target_rgb_path"])
        memory = memory_from_sample(sample)

        t0 = time.time()
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

        gen = torch.Generator(device=device).manual_seed(args.seed + sample_idx)
        z_init = _make_z_init_from_prior(
            z_history=z_history,
            shape=tuple(z1_gt.shape),
            device=device,
            dtype=dit_dtype,
            alpha=args.z0_prior_alpha,
            sigma=args.z0_prior_sigma,
            generator=gen,
        )
        z1_pred, trace = euler_sample_with_trace(
            dit,
            z_history,
            pooled_kv,
            z1_gt,
            z_init,
            num_steps=args.euler_steps,
            cfg_scale=args.cfg_scale,
        )

        m_mse = latent_mse(z1_pred, z1_gt)
        m_cos = latent_cosine(z1_pred, z1_gt)
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
        elapsed = time.time() - t0

        # 3) PNG dump
        _save_rgb_png(rgb_pred[0], case_dir / "pred.png")
        _save_rgb_png(rgb_gt[0], case_dir / "target_vae_recon.png")

        # 4) 写 metrics / euler trace / memory / meta
        metrics = {
            "latent_mse": m_mse,
            "latent_cos": m_cos,
            "pixel_l1": m_l1,
            "psnr": m_psnr,
            "velocity_cos": m_vcos,
        }
        with (case_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with (case_dir / "euler_trace.json").open("w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        with (case_dir / "memory.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(memory), f, ensure_ascii=False, indent=2)

        meta = {
            "sample_idx": sample_idx,
            "scenario": scenario,
            "run_id": run_id,
            "anchor": anchor,
            "target_frame": sample.get("target_frame"),
            "dit_checkpoint": args.dit_checkpoint,
            "patch_unpatch": dit.patch_unpatch_metadata(args.dit_checkpoint),
            "qwen_adapter_dir": args.qwen_adapter_dir,
            "qwen_adapter_merge": args.qwen_adapter_merge,
            "euler_steps": args.euler_steps,
            "cfg_scale": args.cfg_scale,
            "z0_prior_alpha": args.z0_prior_alpha,
            "z0_prior_sigma": args.z0_prior_sigma,
            "use_ema": args.use_ema,
            "seed": args.seed,
            "case_suffix": args.case_suffix,
            "elapsed_sec": elapsed,
        }
        with (case_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        overview = render_overview_md(case_dir, sample, memory, metrics, trace, meta)
        (case_dir / "overview.md").write_text(overview, encoding="utf-8")

        index_records.append({
            "sample_idx": sample_idx,
            "case_dir": str(case_dir),
            "scenario": scenario,
            "run_id": run_id,
            "anchor": anchor,
            "latent_mse": m_mse,
            "latent_cos": m_cos,
            "pixel_l1": m_l1,
            "psnr": m_psnr,
            "velocity_cos": m_vcos,
            "elapsed_sec": elapsed,
        })
        with (case_root / f"_index{args.case_suffix}.jsonl").open("w", encoding="utf-8") as f:
            for r in index_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[probe] done {scenario}/{run_id}/anchor={anchor} → {case_dir}  "
              f"v_cos={m_vcos:.3f} psnr={m_psnr:.2f}")

    print(f"\n[probe] all {len(index_records)} cases dumped under {case_root}")


if __name__ == "__main__":
    main()
