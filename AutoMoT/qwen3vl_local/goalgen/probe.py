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
GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --dit-checkpoint checkpoints/goalgen_v1_dit/latest.pt \
  --save-root checkpoints/goalgen_v1_dit \
  --num-per-scenario 4 --seed 0

# 同 seed 跑两个不同 ckpt + 不同 case-suffix，目录不互相覆盖，便于人工对比
GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
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
from dataclasses import asdict, dataclass, replace
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
    """probe 单进程入口的 GPU 规则。

    优先级：
    1. ``GPU_IDS=...``：用户显式指定物理卡号，直接写入 CUDA_VISIBLE_DEVICES。
    2. 都不指定：用 nvidia-smi 自动挑 1 张最空闲物理卡，并覆盖外层残留 CVD。
    """
    pinned = _normalize_gpu_ids(os.environ.get("GPU_IDS", ""))
    if pinned:
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = pinned
        os.environ["GOALGEN_LOCAL_CUDA_INDEX"] = "0"
        print(
            f"[gpu] using explicit GPU_IDS={pinned}; process uses cuda:0; "
            f"previous CUDA_VISIBLE_DEVICES={previous or '<unset>'}"
        )
        return
    selected = _pick_idle_gpus(1)
    if selected:
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        os.environ["GOALGEN_LOCAL_CUDA_INDEX"] = "0"
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
from qwen3vl_local.prompt_pipeline import (  # noqa: E402
    DrivingMemory,
    EVENT_DESCRIPTIONS,
    SCENARIO_EVENT_SEQUENCES,
    SCENARIO_LABELS,
    get_full_sequence,
)

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


def _parse_csv_tokens(text: str) -> List[str]:
    """解析逗号分隔 token 列表，保持输入顺序并去重。"""

    seen = set()
    out: List[str] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _safe_name(text: str) -> str:
    """把事件名转成文件夹安全的短名字。"""

    keep = []
    for ch in str(text):
        keep.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(keep).strip("_") or "empty"


DEFAULT_COUNTERFACTUAL_CONFIG: Dict[str, Dict[str, List[str]]] = {
    "NonSignalizedJunctionLeftTurn": {
        "swap_in_subgoals": ["assert_priority", "turn_on_green", "brake_at_light"],
    },
    "NonSignalizedJunctionRightTurn": {
        "swap_in_subgoals": ["yield_and_turn", "brake_at_light", "assert_priority"],
    },
    "SignalizedJunctionLeftTurn": {
        "swap_in_subgoals": ["assert_priority", "turn_on_green", "brake_at_light"],
    },
    "SignalizedJunctionRightTurn": {
        "swap_in_subgoals": ["yield_and_turn", "brake_at_light", "assert_priority"],
    },
    "PriorityAtJunction": {
        "swap_in_subgoals": ["brake_at_light", "yield_and_turn", "wait_or_turn_on_green"],
    },
    "EnterActorFlow": {
        "swap_in_subgoals": ["yield_and_turn", "max_brake_or_min_gap", "passing_hazard"],
    },
    "HazardAtSideLane": {
        "swap_in_subgoals": ["gap_accept_merge", "yield_and_turn", "max_brake_or_min_gap"],
    },
    "PedestrianCrossing": {
        "swap_in_subgoals": ["assert_priority", "proceed_resume", "turn_on_green"],
    },
    "Accident": {
        "swap_in_subgoals": ["assert_priority", "proceed_resume", "gap_accept_merge"],
    },
}


@dataclass
class CounterfactualRequest:
    subgoal: str
    target_scenario: str = ""
    target_status: str = ""
    source: str = "cli"


@dataclass
class CounterfactualVariant:
    tag: str
    memory: DrivingMemory
    mode: str
    requested_subgoal: str
    is_truth: bool = False
    is_noop: bool = False
    prompt_consistency: str = "consistent"
    warning: str = ""


def _all_legal_pairs() -> Dict[Tuple[str, str], List[Tuple[str, int]]]:
    pairs: Dict[Tuple[str, str], List[Tuple[str, int]]] = defaultdict(list)
    for scenario in sorted(SCENARIO_EVENT_SEQUENCES):
        seq = get_full_sequence(scenario)
        for idx in range(len(seq) - 1):
            pairs[(seq[idx], seq[idx + 1])].append((scenario, idx))
    return pairs


LEGAL_EVENT_PAIRS = _all_legal_pairs()


def _load_counterfactual_config(path_or_default: str) -> Dict[str, Any]:
    if not path_or_default:
        return {}
    if path_or_default.lower() == "default":
        return DEFAULT_COUNTERFACTUAL_CONFIG
    path = pathlib.Path(path_or_default)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _requests_from_config_entry(entry: Any, source: str) -> List[CounterfactualRequest]:
    if not entry:
        return []
    if isinstance(entry, list):
        raw_items = entry
    elif isinstance(entry, dict):
        raw_items = entry.get("swap_in_subgoals", [])
    else:
        raise TypeError(f"不支持的 counterfactual config entry 类型：{type(entry)}")

    out: List[CounterfactualRequest] = []
    for item in raw_items:
        if isinstance(item, str):
            out.append(CounterfactualRequest(subgoal=item, source=source))
        elif isinstance(item, dict):
            subgoal = str(item.get("subgoal") or item.get("event") or "").strip()
            if not subgoal:
                raise ValueError(f"counterfactual config item 缺少 subgoal/event: {item}")
            out.append(CounterfactualRequest(
                subgoal=subgoal,
                target_scenario=str(item.get("scenario", "") or ""),
                target_status=str(item.get("status", "") or ""),
                source=source,
            ))
        else:
            raise TypeError(f"不支持的 counterfactual config item 类型：{type(item)}")
    return out


def _counterfactual_requests_for_sample(
    scenario: str,
    config: Dict[str, Any],
    cli_subgoals: List[str],
) -> Tuple[List[CounterfactualRequest], str]:
    if config and scenario in config:
        return _requests_from_config_entry(config[scenario], f"config:{scenario}"), "config"
    return [CounterfactualRequest(subgoal=s, source="cli") for s in cli_subgoals], "cli"


def _find_predecessor_for_subgoal(
    request: CounterfactualRequest,
    original_scenario: str,
) -> Optional[Tuple[str, int]]:
    """为 scenario_swap 找一个包含目标 SUBGOAL 的合法前驱。"""

    candidates: List[Tuple[str, int]] = []
    for (_status, subgoal), owners in LEGAL_EVENT_PAIRS.items():
        if subgoal == request.subgoal:
            candidates.extend(owners)
    if not candidates:
        return None
    if request.target_scenario:
        for scenario, idx in candidates:
            if scenario == request.target_scenario:
                return scenario, idx
    if request.target_status:
        for scenario, idx in candidates:
            seq = get_full_sequence(scenario)
            if seq[idx] == request.target_status:
                return scenario, idx
    for scenario, idx in candidates:
        if scenario == original_scenario:
            return scenario, idx
    return sorted(candidates, key=lambda x: (x[0], x[1]))[0]


def _build_scenario_swap_memory(
    memory: DrivingMemory,
    request: CounterfactualRequest,
) -> Tuple[Optional[DrivingMemory], str]:
    found = _find_predecessor_for_subgoal(request, memory.scenario)
    if found is None:
        return None, f"subgoal {request.subgoal!r} 不存在于任何场景状态机，跳过"
    scenario, idx = found
    seq = get_full_sequence(scenario)
    if request.target_status and seq[idx] != request.target_status:
        return None, (
            f"subgoal {request.subgoal!r} 在 scenario={scenario!r} 的前驱是 "
            f"{seq[idx]!r}，不等于指定 status={request.target_status!r}"
        )
    return DrivingMemory(
        scenario=scenario,
        scenario_label=SCENARIO_LABELS.get(scenario, scenario),
        event_sequence=seq,
        status=seq[idx],
        subgoal=seq[idx + 1],
        completed_events=list(seq[:idx + 1]),
    ), ""


def _build_subgoal_only_memory(
    memory: DrivingMemory,
    request: CounterfactualRequest,
) -> Tuple[Optional[DrivingMemory], str, str]:
    pair = (memory.status, request.subgoal)
    legal_somewhere = pair in LEGAL_EVENT_PAIRS
    if not legal_somewhere:
        return None, "invalid_pair", (
            f"(STATUS={memory.status!r}, SUBGOAL={request.subgoal!r}) "
            "不是任何场景状态机里的合法相邻转移，subgoal_only 跳过"
        )
    consistency = "consistent" if request.subgoal in memory.event_sequence else "sequence_mismatch"
    warning = ""
    if consistency != "consistent":
        owners = ",".join(s for s, _idx in LEGAL_EVENT_PAIRS[pair][:3])
        warning = (
            f"subgoal_only 保留原 scenario/event_sequence，但该 SUBGOAL 不在原序列中；"
            f"同 STATUS 合法 pair 出现在: {owners}"
        )
    return replace(memory, subgoal=request.subgoal), consistency, warning


def _make_counterfactual_variants(
    memory: DrivingMemory,
    requests: List[CounterfactualRequest],
    mode: str,
) -> Tuple[List[CounterfactualVariant], List[Dict[str, Any]]]:
    """构造 truth/noop/CF variant；noop 永远保留，用来估计数值 floor。"""

    variants: List[CounterfactualVariant] = [
        CounterfactualVariant(
            tag="truth",
            memory=memory,
            mode=mode,
            requested_subgoal=memory.subgoal,
            is_truth=True,
        ),
        CounterfactualVariant(
            tag=f"noop_{_safe_name(memory.subgoal)}",
            memory=replace(memory),
            mode=mode,
            requested_subgoal=memory.subgoal,
            is_noop=True,
        ),
    ]
    skipped: List[Dict[str, Any]] = []
    seen = {("truth", memory.subgoal), ("noop", memory.subgoal)}
    cf_idx = 1
    for request in requests:
        if request.subgoal == memory.subgoal:
            skipped.append({
                "subgoal": request.subgoal,
                "mode": mode,
                "reason": "same_as_truth_subgoal; covered by noop control",
            })
            continue
        if request.subgoal not in EVENT_DESCRIPTIONS:
            skipped.append({
                "subgoal": request.subgoal,
                "mode": mode,
                "reason": "unknown_event_token",
            })
            continue
        if ("cf", request.subgoal, request.target_scenario, request.target_status) in seen:
            continue
        if mode == "scenario_swap":
            cf_memory, warning = _build_scenario_swap_memory(memory, request)
            consistency = "consistent"
        elif mode == "subgoal_only":
            cf_memory, consistency, warning = _build_subgoal_only_memory(memory, request)
        else:
            raise ValueError(f"未知 counterfactual mode: {mode}")
        if cf_memory is None:
            skipped.append({
                "subgoal": request.subgoal,
                "mode": mode,
                "reason": warning,
            })
            print(f"[probe][cf][skip] mode={mode} subgoal={request.subgoal}: {warning}")
            continue
        tag = f"cf_{cf_idx:02d}_{_safe_name(request.subgoal)}"
        variants.append(CounterfactualVariant(
            tag=tag,
            memory=cf_memory,
            mode=mode,
            requested_subgoal=request.subgoal,
            prompt_consistency=consistency,
            warning=warning,
        ))
        seen.add(("cf", request.subgoal, request.target_scenario, request.target_status))
        cf_idx += 1
        if warning:
            print(f"[probe][cf][warn] mode={mode} tag={tag}: {warning}")
    return variants, skipped


def _save_counterfactual_grid(
    out: pathlib.Path,
    items: List[Tuple[str, pathlib.Path]],
) -> Optional[pathlib.Path]:
    """把 target / truth / counterfactual pred 拼成一张横向 compare 图。"""

    loaded: List[Tuple[str, Image.Image]] = []
    for label, path in items:
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        loaded.append((label, img))
    if not loaded:
        return None

    width = max(img.width for _, img in loaded)
    height = max(img.height for _, img in loaded)
    label_h = 34
    grid = Image.new("RGB", (width * len(loaded), height + label_h), "white")
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(grid)
    except Exception:
        draw = None
    for idx, (label, img) in enumerate(loaded):
        x = idx * width
        grid.paste(img, (x, label_h))
        if draw is not None:
            draw.text((x + 8, 8), label[:48], fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    return out


def _counterfactual_experiment_doc(original_subgoal: str, requested_subgoals: List[str]) -> Dict[str, Any]:
    """返回写入 meta/summary 的实验说明，方便离线翻记录时不丢语义。"""

    return {
        "name": "goalgen_counterfactual_subgoal",
        "question": (
            "Does GoalGen change the generated subgoal keyframe when the "
            "teacher-forced SUBGOAL text is manually replaced?"
        ),
        "intervention": (
            "Keep the same history RGB, scenario, STATUS, target image, Euler seed "
            "and z_init; only replace the SUBGOAL token and its semantic description "
            "inside the Qwen teacher-forced prompt."
        ),
        "baseline": f"truth SUBGOAL = {original_subgoal}",
        "requested_counterfactual_subgoals": requested_subgoals,
        "primary_artifact": "counterfactual_compare_<mode>.png",
        "how_to_read": (
            "The first column is the original target frame, the second column is "
            "the prediction under the truth SUBGOAL, and the following columns are "
            "predictions under manually injected counterfactual SUBGOALs. For "
            "counterfactual branches, worse metrics against the original target are "
            "expected and are not automatically failures; the key evidence is whether "
            "the image changes in a semantically interpretable direction."
        ),
    }


def _parse_float_csv(text: str, fallback: float) -> List[float]:
    vals: List[float] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        vals.append(float(item))
    return vals or [float(fallback)]


def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {"mean": mean, "std": var ** 0.5, "count": len(values)}


def _variant_delta_summary(
    variant_records: List[Dict[str, Any]],
    eps: float = 1e-9,
) -> Dict[str, Any]:
    noop_runs: List[Dict[str, Any]] = []
    for record in variant_records:
        if record.get("is_noop"):
            noop_runs.extend(record.get("runs", []))
    noop_latents = [
        float(run["delta_latent_mse_vs_truth_pred"])
        for run in noop_runs
        if run.get("delta_latent_mse_vs_truth_pred") is not None
    ]
    noop_pixels = [
        float(run["delta_pixel_l1_vs_truth_pred"])
        for run in noop_runs
        if run.get("delta_pixel_l1_vs_truth_pred") is not None
    ]
    noop_floor = {
        "latent_mse": _mean_std(noop_latents),
        "pixel_l1": _mean_std(noop_pixels),
    }
    latent_floor = max(noop_floor["latent_mse"]["mean"], eps)
    pixel_floor = max(noop_floor["pixel_l1"]["mean"], eps)

    noop_by_cfg: Dict[float, Dict[str, Dict[str, float]]] = {}
    cfg_values = sorted({
        float(run["cfg_scale"])
        for record in variant_records
        for run in record.get("runs", [])
        if "cfg_scale" in run
    })
    for cfg in cfg_values:
        cfg_noop_latents = [
            float(run["delta_latent_mse_vs_truth_pred"])
            for run in noop_runs
            if run.get("delta_latent_mse_vs_truth_pred") is not None
            and float(run.get("cfg_scale", cfg)) == cfg
        ]
        cfg_noop_pixels = [
            float(run["delta_pixel_l1_vs_truth_pred"])
            for run in noop_runs
            if run.get("delta_pixel_l1_vs_truth_pred") is not None
            and float(run.get("cfg_scale", cfg)) == cfg
        ]
        noop_by_cfg[cfg] = {
            "latent_mse": _mean_std(cfg_noop_latents),
            "pixel_l1": _mean_std(cfg_noop_pixels),
        }

    per_cf: List[Dict[str, Any]] = []
    for record in variant_records:
        if record.get("is_truth") or record.get("is_noop"):
            continue
        latent_vals = [
            float(run["delta_latent_mse_vs_truth_pred"])
            for run in record.get("runs", [])
            if run.get("delta_latent_mse_vs_truth_pred") is not None
        ]
        pixel_vals = [
            float(run["delta_pixel_l1_vs_truth_pred"])
            for run in record.get("runs", [])
            if run.get("delta_pixel_l1_vs_truth_pred") is not None
        ]
        latent_stats = _mean_std(latent_vals)
        pixel_stats = _mean_std(pixel_vals)
        per_cfg: List[Dict[str, Any]] = []
        for cfg in cfg_values:
            cfg_runs = [run for run in record.get("runs", []) if float(run.get("cfg_scale", cfg)) == cfg]
            cfg_latent_vals = [
                float(run["delta_latent_mse_vs_truth_pred"])
                for run in cfg_runs
                if run.get("delta_latent_mse_vs_truth_pred") is not None
            ]
            cfg_pixel_vals = [
                float(run["delta_pixel_l1_vs_truth_pred"])
                for run in cfg_runs
                if run.get("delta_pixel_l1_vs_truth_pred") is not None
            ]
            cfg_latent_stats = _mean_std(cfg_latent_vals)
            cfg_pixel_stats = _mean_std(cfg_pixel_vals)
            cfg_latent_floor = max(noop_by_cfg.get(cfg, {}).get("latent_mse", {}).get("mean", 0.0), eps)
            cfg_pixel_floor = max(noop_by_cfg.get(cfg, {}).get("pixel_l1", {}).get("mean", 0.0), eps)
            per_cfg.append({
                "cfg_scale": cfg,
                "delta_latent_mse_vs_truth_pred": cfg_latent_stats,
                "delta_pixel_l1_vs_truth_pred": cfg_pixel_stats,
                "ratio_over_noop_floor_latent": cfg_latent_stats["mean"] / cfg_latent_floor,
                "ratio_over_noop_floor_pixel": cfg_pixel_stats["mean"] / cfg_pixel_floor,
            })
        per_cf.append({
            "tag": record.get("tag"),
            "subgoal": record.get("subgoal"),
            "mode": record.get("mode"),
            "prompt_consistency": record.get("prompt_consistency"),
            "delta_latent_mse_vs_truth_pred": latent_stats,
            "delta_pixel_l1_vs_truth_pred": pixel_stats,
            "ratio_over_noop_floor_latent": latent_stats["mean"] / latent_floor,
            "ratio_over_noop_floor_pixel": pixel_stats["mean"] / pixel_floor,
            "per_cfg": per_cfg,
        })

    return {
        "noop_floor": noop_floor,
        "noop_floor_by_cfg": {
            str(cfg): stats for cfg, stats in noop_by_cfg.items()
        },
        "per_cf": per_cf,
        "verdict_hint": (
            "ratio_over_noop_floor > 5 suggests the generated image is likely "
            "sensitive to the injected SUBGOAL; inspect counterfactual_compare*.png "
            "for semantic direction."
        ),
    }


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
    if meta.get("counterfactual_compare"):
        lines.append(f"- counterfactual_compare: `{meta.get('counterfactual_compare')}`")
    lines.append("")

    experiment = meta.get("experiment")
    if experiment:
        lines.append("## Experiment")
        lines.append(f"- name: `{experiment.get('name')}`")
        lines.append(f"- question: {experiment.get('question')}")
        lines.append(f"- intervention: {experiment.get('intervention')}")
        lines.append(f"- baseline: {experiment.get('baseline')}")
        lines.append(f"- how_to_read: {experiment.get('how_to_read')}")
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
    if meta.get("counterfactual_compare"):
        lines.append("- `counterfactual_compare_<mode>.png` (同一 history / 同一组 z_init，按 mode 替换 prompt 语义的并排图)")
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
    parser.add_argument("--counterfactual-subgoals", default="",
                        help="逗号分隔的人工 SUBGOAL token。非空时，同一 history / 同一 z_init "
                             "会额外按这些 SUBGOAL 生成 counterfactual pred，输出到 "
                             "case_dir/counterfactual/ 并生成 counterfactual_compare_<mode>.png。")
    parser.add_argument("--counterfactual-mode",
                        choices=["scenario_swap", "subgoal_only", "both"],
                        default="scenario_swap",
                        help="counterfactual prompt 构造方式：scenario_swap 会同时替换 "
                             "scenario/status/event_sequence 保持 prompt 自洽；subgoal_only "
                             "只替换 SUBGOAL；both 在同一 case 下保存两套结果。")
    parser.add_argument("--counterfactual-config", default="",
                        help="per-scenario 干预配置 JSON；传 default 使用内置标准实验表。"
                             "若当前 scenario 命中配置，优先用配置；否则回退 --counterfactual-subgoals。")
    parser.add_argument("--cfg-scale-sweep", default="",
                        help="逗号分隔 CFG 扫描值，例如 0.0,1.0,2.0,4.0；空则只用 --cfg-scale。")
    parser.add_argument("--counterfactual-seed-replicates", type=int, default=1,
                        help="每个 (case, variant, cfg) 用多少个 z_init seed 重复采样；默认 1。")
    parser.add_argument("--num-per-scenario", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
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
    # 与 train 默认对齐：alpha=0.0 用纯噪声起点。设回 1.0 仅做 image-to-image ablation。
    parser.add_argument("--z0-prior-alpha", type=float, default=0.0)
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

    case_root = pathlib.Path(args.save_root) / "eval_cases"
    case_root.mkdir(parents=True, exist_ok=True)
    from qwen3vl_local.run_log import install_output_log
    install_output_log(case_root)

    samples = load_jsonl(pathlib.Path(args.val_jsonl))
    scenarios_filter = [s.strip() for s in args.scenarios.split(",") if s.strip()] or None
    counterfactual_subgoals = _parse_csv_tokens(args.counterfactual_subgoals)
    counterfactual_config = _load_counterfactual_config(args.counterfactual_config)
    counterfactual_modes = (
        ["scenario_swap", "subgoal_only"]
        if args.counterfactual_mode == "both"
        else [args.counterfactual_mode]
    )
    cfg_scale_values = _parse_float_csv(args.cfg_scale_sweep, args.cfg_scale)
    seed_replicates = max(1, int(args.counterfactual_seed_replicates))
    picked = select_samples(samples, scenarios_filter, args.num_per_scenario, args.seed)
    print(f"[probe] selected {len(picked)} samples from {len(samples)} total")
    if counterfactual_subgoals:
        print(f"[probe] counterfactual SUBGOAL variants: {counterfactual_subgoals}")
    if counterfactual_config:
        src = "built-in default" if args.counterfactual_config.lower() == "default" else args.counterfactual_config
        print(f"[probe] counterfactual config enabled: {src}")
    if counterfactual_subgoals or counterfactual_config:
        print(
            f"[probe] counterfactual modes={counterfactual_modes} "
            f"cfg_sweep={cfg_scale_values} seed_replicates={seed_replicates}"
        )

    local_cuda_index = int(os.environ.get("GOALGEN_LOCAL_CUDA_INDEX", "0"))
    device = torch.device(f"cuda:{local_cuda_index}" if torch.cuda.is_available() else "cpu")
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

        z_history = vae.encode(history_images).to(dtype=dit_dtype).unsqueeze(0)
        z1_gt = vae.encode([target_img]).to(dtype=dit_dtype)
        z1_gt_for_vae = z1_gt.to(device=vae.device, dtype=vae.dtype)
        rgb_gt = vae.decode(z1_gt_for_vae).clamp(-1.0, 1.0)

        cf_requests, cf_request_source = _counterfactual_requests_for_sample(
            scenario,
            counterfactual_config,
            counterfactual_subgoals,
        )
        dump_counterfactual = bool(cf_requests)
        all_mode_summaries: Dict[str, Any] = {}
        all_mode_metric_summaries: Dict[str, Any] = {}
        skipped_variants: List[Dict[str, Any]] = []
        truth_metrics: Optional[Dict[str, float]] = None
        truth_trace: Optional[Dict[str, List[float]]] = None
        truth_elapsed = 0.0
        primary_compare = ""

        z_inits: List[Tuple[int, torch.Tensor]] = []
        for rep_idx in range(seed_replicates):
            replicate_seed = int(args.seed + sample_idx * 1009 + rep_idx)
            gen = torch.Generator(device=device).manual_seed(replicate_seed)
            z_init = _make_z_init_from_prior(
                z_history=z_history,
                shape=tuple(z1_gt.shape),
                device=device,
                dtype=dit_dtype,
                alpha=args.z0_prior_alpha,
                sigma=args.z0_prior_sigma,
                generator=gen,
            )
            z_inits.append((replicate_seed, z_init))

        active_modes = counterfactual_modes if dump_counterfactual else ["scenario_swap"]
        for mode in active_modes:
            if dump_counterfactual:
                variants, skipped = _make_counterfactual_variants(memory, cf_requests, mode)
            else:
                variants = [
                    CounterfactualVariant(
                        tag="truth",
                        memory=memory,
                        mode=mode,
                        requested_subgoal=memory.subgoal,
                        is_truth=True,
                    )
                ]
                skipped = []
            skipped_variants.extend(skipped)
            variant_records: List[Dict[str, Any]] = []
            truth_preds: Dict[Tuple[float, int], Tuple[torch.Tensor, torch.Tensor]] = {}

            for variant in variants:
                t0_prefill = time.time()
                prefill = teacher_forced_prefill(
                    engine=engine,
                    memory=variant.memory,
                    images=history_images,
                    num_segments=args.num_layers,
                    kv_segment_mode=args.qwen_kv_segment_mode,
                )
                pooled_kv = [
                    (k.to(device=device, dtype=dit_dtype), v.to(device=device, dtype=dit_dtype))
                    for k, v in prefill.pooled_kv
                ]
                prefill_elapsed = time.time() - t0_prefill
                runs: List[Dict[str, Any]] = []

                for cfg_scale_value in cfg_scale_values:
                    for rep_idx, (replicate_seed, z_init) in enumerate(z_inits):
                        t0 = time.time()
                        z1_pred, trace = euler_sample_with_trace(
                            dit,
                            z_history,
                            pooled_kv,
                            z1_gt,
                            z_init,
                            num_steps=args.euler_steps,
                            cfg_scale=cfg_scale_value,
                        )

                        m_mse = latent_mse(z1_pred, z1_gt)
                        m_cos = latent_cosine(z1_pred, z1_gt)
                        z1_pred_for_vae = z1_pred.to(device=vae.device, dtype=vae.dtype)
                        rgb_pred = vae.decode(z1_pred_for_vae).clamp(-1.0, 1.0)
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
                        elapsed_variant = time.time() - t0
                        variant_metrics = {
                            "latent_mse": m_mse,
                            "latent_cos": m_cos,
                            "pixel_l1": m_l1,
                            "psnr": m_psnr,
                            "velocity_cos": m_vcos,
                        }

                        pred_delta_mse = None
                        pred_delta_pixel_l1 = None
                        pred_delta_psnr = None
                        key = (float(cfg_scale_value), rep_idx)
                        if variant.is_truth:
                            truth_preds[key] = (z1_pred.detach().clone(), rgb_pred.detach().clone())
                            if (
                                mode == active_modes[0]
                                and float(cfg_scale_value) == float(cfg_scale_values[0])
                                and rep_idx == 0
                            ):
                                truth_metrics = variant_metrics
                                truth_trace = trace
                                truth_elapsed = elapsed_variant + prefill_elapsed
                                _save_rgb_png(rgb_pred[0], case_dir / "pred.png")
                        elif key in truth_preds:
                            truth_z, truth_rgb = truth_preds[key]
                            pred_delta_mse = float(F.mse_loss(z1_pred.float(), truth_z.float()).item())
                            pred_delta_pixel_l1, pred_delta_psnr = pixel_l1_psnr(rgb_pred, truth_rgb)

                        pred_path = ""
                        if dump_counterfactual:
                            cfg_tag = f"cfg_{str(cfg_scale_value).replace('.', 'p').replace('-', 'm')}"
                            run_dir = (
                                case_dir / "counterfactual" / mode / variant.tag /
                                cfg_tag / f"seed_{rep_idx:02d}"
                            )
                            run_dir.mkdir(parents=True, exist_ok=True)
                            pred_path = str(run_dir / "pred.png")
                            _save_rgb_png(rgb_pred[0], run_dir / "pred.png")
                            with (run_dir / "metrics_vs_original_target.json").open("w", encoding="utf-8") as f:
                                json.dump(variant_metrics, f, ensure_ascii=False, indent=2)

                        runs.append({
                            "cfg_scale": float(cfg_scale_value),
                            "replicate_idx": rep_idx,
                            "seed": replicate_seed,
                            "pred_path": pred_path,
                            "metrics_vs_original_target": variant_metrics,
                            "delta_latent_mse_vs_truth_pred": pred_delta_mse,
                            "delta_pixel_l1_vs_truth_pred": pred_delta_pixel_l1,
                            "delta_psnr_vs_truth_pred": pred_delta_psnr,
                            "elapsed_sec": elapsed_variant,
                        })

                variant_dir = case_dir / "counterfactual" / mode / variant.tag
                chat_text_path = ""
                memory_path = ""
                if dump_counterfactual:
                    variant_dir.mkdir(parents=True, exist_ok=True)
                    chat_text_path = str(variant_dir / "chat_text.txt")
                    memory_path = str(variant_dir / "memory.json")
                    (variant_dir / "chat_text.txt").write_text(prefill.chat_text, encoding="utf-8")
                    with (variant_dir / "memory.json").open("w", encoding="utf-8") as f:
                        json.dump(asdict(variant.memory), f, ensure_ascii=False, indent=2)

                variant_records.append({
                    "tag": variant.tag,
                    "mode": mode,
                    "subgoal": variant.memory.subgoal,
                    "requested_subgoal": variant.requested_subgoal,
                    "scenario": variant.memory.scenario,
                    "status": variant.memory.status,
                    "is_truth": variant.is_truth,
                    "is_noop": variant.is_noop,
                    "prompt_consistency": variant.prompt_consistency,
                    "warning": variant.warning,
                    "chat_text_path": chat_text_path,
                    "memory_path": memory_path,
                    "prefill_elapsed_sec": prefill_elapsed,
                    "runs": runs,
                })

            if dump_counterfactual:
                first_cfg = float(cfg_scale_values[0])
                grid_items = [("target_raw", case_dir / "target_raw.jpg")]
                for record in variant_records:
                    first_run = next(
                        (
                            run for run in record.get("runs", [])
                            if run["cfg_scale"] == first_cfg and run["replicate_idx"] == 0
                        ),
                        None,
                    )
                    if not first_run or not first_run.get("pred_path"):
                        continue
                    label = ("truth:" if record["is_truth"] else "noop:" if record["is_noop"] else "cf:")
                    label += str(record["subgoal"])
                    grid_items.append((label, pathlib.Path(first_run["pred_path"])))
                mode_compare = case_dir / f"counterfactual_compare_{mode}.png"
                grid_path = _save_counterfactual_grid(mode_compare, grid_items)
                if mode == active_modes[0]:
                    primary_compare = str(grid_path) if grid_path else ""
                metric_summary = _variant_delta_summary(variant_records)
                all_mode_metric_summaries[mode] = metric_summary
                all_mode_summaries[mode] = {
                    "compare_png": str(grid_path) if grid_path else "",
                    "variants": variant_records,
                    "metrics_summary": metric_summary,
                }

        if truth_metrics is None or truth_trace is None:
            raise RuntimeError("内部错误：counterfactual variants 缺少 truth baseline")

        metrics = truth_metrics
        trace = truth_trace
        elapsed = truth_elapsed

        # 3) PNG dump
        _save_rgb_png(rgb_gt[0], case_dir / "target_vae_recon.png")

        experiment_doc = (
            _counterfactual_experiment_doc(memory.subgoal, [r.subgoal for r in cf_requests])
            if dump_counterfactual
            else None
        )
        if experiment_doc:
            experiment_doc.update({
                "mode": args.counterfactual_mode,
                "active_modes": active_modes,
                "request_source": cf_request_source,
                "cfg_scale_sweep": cfg_scale_values,
                "seed_replicates": seed_replicates,
            })
        if dump_counterfactual:
            cf_summary = {
                "experiment": experiment_doc,
                "note": (
                    "All modes share the same history RGB, z_history, target RGB and "
                    "replicate z_init seeds. scenario_swap keeps the prompt internally "
                    "consistent by swapping scenario/status/event_sequence; subgoal_only "
                    "keeps the original scenario/status and only edits SUBGOAL."
                ),
                "original_subgoal": memory.subgoal,
                "request_source": cf_request_source,
                "counterfactual_requests": [asdict(r) for r in cf_requests],
                "skipped_variants": skipped_variants,
                "cfg_scale_sweep": cfg_scale_values,
                "seed_replicates": seed_replicates,
                "primary_compare_png": primary_compare,
                "modes": all_mode_summaries,
            }
            with (case_dir / "counterfactual_summary.json").open("w", encoding="utf-8") as f:
                json.dump(cf_summary, f, ensure_ascii=False, indent=2)
            metrics_summary_payload = {
                "experiment": experiment_doc,
                "modes": all_mode_metric_summaries,
                "verdict_hint": (
                    "ratio_over_noop_floor > 5 suggests SUBGOAL likely influences output. "
                    "Use both latent and pixel ratios, then inspect images."
                ),
            }
            with (case_dir / "counterfactual_metrics_summary.json").open("w", encoding="utf-8") as f:
                json.dump(metrics_summary_payload, f, ensure_ascii=False, indent=2)

        # 4) 写 metrics / euler trace / memory / meta
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
            "counterfactual_subgoals": counterfactual_subgoals,
            "counterfactual_mode": args.counterfactual_mode,
            "counterfactual_config": args.counterfactual_config,
            "cfg_scale_sweep": cfg_scale_values,
            "counterfactual_seed_replicates": seed_replicates,
            "counterfactual_summary": str(case_dir / "counterfactual_summary.json") if dump_counterfactual else "",
            "counterfactual_metrics_summary": str(case_dir / "counterfactual_metrics_summary.json") if dump_counterfactual else "",
            "counterfactual_compare": primary_compare,
            "experiment": experiment_doc,
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
            "latent_mse": metrics["latent_mse"],
            "latent_cos": metrics["latent_cos"],
            "pixel_l1": metrics["pixel_l1"],
            "psnr": metrics["psnr"],
            "velocity_cos": metrics["velocity_cos"],
            "counterfactual_compare": primary_compare,
            "counterfactual_summary": str(case_dir / "counterfactual_summary.json") if dump_counterfactual else "",
            "elapsed_sec": elapsed,
        })
        with (case_root / f"_index{args.case_suffix}.jsonl").open("w", encoding="utf-8") as f:
            for r in index_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[probe] done {scenario}/{run_id}/anchor={anchor} → {case_dir}  "
              f"v_cos={metrics['velocity_cos']:.3f} psnr={metrics['psnr']:.2f}")

    print(f"\n[probe] all {len(index_records)} cases dumped under {case_root}")


if __name__ == "__main__":
    main()
