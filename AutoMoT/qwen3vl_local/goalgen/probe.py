"""GoalGen v1/v2 case-level probe — 随机选 N 个场景的样本，dump 输入+生成+真值+诊断。

普通 probe 视角（无 CF）：
- 历史 RGB（symlink）
- 真值 keyframe 原图 + VAE encode→decode 的 reference（生成天花板）
- DiT 在固定 seed 下 Euler 采样的 z1_pred + 解码 pred RGB
- per-step euler 轨迹（v_pred vs v_target_gt 的 cosine 序列、z_t 到 GT 的 L2）
- memory.json / metrics.json / meta.json / overview.md

Counterfactual 视角（开启 --counterfactual-subgoals / --counterfactual-config）：
- cf_overview_<mode>.png   一张拼图：行=variant（truth + 各 CF），列=cfg sweep
- cf_summary.json          单一精简 JSON，含 noise_floor + per_cf delta + ratio + verdict
- cf_report.md             人类可读 markdown，能直接拷给别人看
- _cf_index<suffix>.jsonl  case_root 级一行一 case 的 max-ratio / verdict roll-up

默认不再产生大量 per-(cfg, seed) 子目录文件。想保留 chat_text.txt / memory.json
/ 各 seed 单独的 pred.png + metrics.json，加 --cf-verbose-artifacts。

Floor 概念（重要）：
- floor = truth 自己跨 z_init seed 的 pairwise 差异（pixel_l1 / latent_mse）
- 物理含义："换一个 seed 跑同 prompt 会变多少" → 采样噪声地板
- 需要 --counterfactual-seed-replicates >= 2 才能算 floor；否则 ratio 字段为 null
  并在 stdout 打印一条提示

不接 torchrun（per-sample 顺序写文件更直观）。
默认自动挑 1 张空闲 GPU；GPU_IDS=N 强制指定。

典型用法（**从 AutoMoT/ 目录运行**）：

```bash
GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --save-root checkpoints/goalgen_v1_dit \
  --num-per-scenario 4 --seed 0

# 默认推荐 counterfactual：scenario_swap + 内置 per-scenario config + 3 seed floor
GPU_IDS=0 python qwen3vl_local/goalgen/probe.py \
  --save-root checkpoints/goalgen_v1_dit \
  --scenarios NonSignalizedJunctionLeftTurn,PriorityAtJunction,HazardAtSideLane \
  --num-per-scenario 1 --seed 7 \
  --counterfactual-mode scenario_swap \
  --counterfactual-config default \
  --counterfactual-seed-replicates 3
```
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Optional, Set, Tuple

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
from PIL import Image, ImageDraw  # noqa: E402

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

# 直接复用 eval 里已经写好的 ckpt 反向构建 DiT、probe KV、score helper。
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


# =========================================================================== #
# 1. Per-scenario CF 配置：覆盖 RUN.md §7.1 推荐表的全部场景。
# =========================================================================== #
#
# 设计原则：
# - 每个 scenario 给 3 个跨语义簇的 CF SUBGOAL；
# - 一定包含一个"通行/抢行"类 (assert_priority / turn_on_green / proceed_resume)
#   和一个"刹停/避让"类 (brake_at_light / max_brake_or_min_gap / yield_and_turn)，
#   保证 CF 与 truth 语义方向反差明显；
# - 未在表里出现的 scenario，命中失败时打 warn 提醒用户扩 config。

DEFAULT_COUNTERFACTUAL_CONFIG: Dict[str, Dict[str, List[str]]] = {
    # ---- 左转 ----
    "NonSignalizedJunctionLeftTurn":          {"swap_in_subgoals": ["assert_priority", "turn_on_green", "brake_at_light"]},
    "SignalizedJunctionLeftTurn":             {"swap_in_subgoals": ["assert_priority", "turn_on_green", "brake_at_light"]},
    "NonSignalizedJunctionLeftTurnEnterFlow": {"swap_in_subgoals": ["assert_priority", "turn_on_green", "brake_at_light"]},
    "SignalizedJunctionLeftTurnEnterFlow":    {"swap_in_subgoals": ["assert_priority", "turn_on_green", "brake_at_light"]},
    # ---- 右转 ----
    "NonSignalizedJunctionRightTurn":         {"swap_in_subgoals": ["yield_and_turn", "brake_at_light", "assert_priority"]},
    "SignalizedJunctionRightTurn":            {"swap_in_subgoals": ["yield_and_turn", "brake_at_light", "assert_priority"]},
    # ---- 路口直行 / 优先权 ----
    "PriorityAtJunction":                     {"swap_in_subgoals": ["brake_at_light", "yield_and_turn", "wait_or_turn_on_green"]},
    "CrossJunctionDefectTrafficLight":        {"swap_in_subgoals": ["assert_priority", "brake_at_light", "yield_and_turn"]},
    "T_Junction":                             {"swap_in_subgoals": ["assert_priority", "brake_at_light", "turn_on_green"]},
    # ---- 汇入 / 慢车流 ----
    "EnterActorFlow":                         {"swap_in_subgoals": ["yield_and_turn", "max_brake_or_min_gap", "passing_hazard"]},
    "EnterActorFlowV2":                       {"swap_in_subgoals": ["yield_and_turn", "max_brake_or_min_gap", "passing_hazard"]},
    "InterurbanActorFlow":                    {"swap_in_subgoals": ["yield_and_turn", "max_brake_or_min_gap", "passing_hazard"]},
    "InterurbanAdvancedActorFlow":            {"swap_in_subgoals": ["yield_and_turn", "max_brake_or_min_gap", "passing_hazard"]},
    "MergerIntoSlowTraffic":                  {"swap_in_subgoals": ["assert_priority", "max_brake_or_min_gap", "yield_and_turn"]},
    "MergerIntoSlowTrafficV2":                {"swap_in_subgoals": ["assert_priority", "max_brake_or_min_gap", "yield_and_turn"]},
    # ---- 避障 / 障碍物 ----
    "HazardAtSideLane":                       {"swap_in_subgoals": ["gap_accept_merge", "yield_and_turn", "max_brake_or_min_gap"]},
    "HazardAtSideLaneTwoWays":                {"swap_in_subgoals": ["gap_accept_merge", "yield_and_turn", "max_brake_or_min_gap"]},
    "ConstructionObstacle":                   {"swap_in_subgoals": ["assert_priority", "proceed_resume", "gap_accept_merge"]},
    "ConstructionObstacleTwoWays":            {"swap_in_subgoals": ["assert_priority", "proceed_resume", "gap_accept_merge"]},
    "ParkedObstacle":                         {"swap_in_subgoals": ["assert_priority", "proceed_resume", "gap_accept_merge"]},
    "ParkedObstacleTwoWays":                  {"swap_in_subgoals": ["assert_priority", "proceed_resume", "gap_accept_merge"]},
    "BlockedIntersection":                    {"swap_in_subgoals": ["assert_priority", "yield_and_turn", "turn_on_green"]},
    "InvadingTurn":                           {"swap_in_subgoals": ["assert_priority", "proceed_resume", "yield_and_turn"]},
    "VehicleOpensDoorTwoWays":                {"swap_in_subgoals": ["assert_priority", "proceed_resume", "gap_accept_merge"]},
    # ---- 行人 / 动态物 ----
    "PedestrianCrossing":                     {"swap_in_subgoals": ["assert_priority", "proceed_resume", "turn_on_green"]},
    "DynamicObjectCrossing":                  {"swap_in_subgoals": ["assert_priority", "proceed_resume", "turn_on_green"]},
    "ParkingCrossingPedestrian":              {"swap_in_subgoals": ["assert_priority", "proceed_resume", "turn_on_green"]},
    "CrossingBicycleFlow":                    {"swap_in_subgoals": ["assert_priority", "proceed_resume", "turn_on_green"]},
    "VehicleTurningRoute":                    {"swap_in_subgoals": ["assert_priority", "proceed_resume", "turn_on_green"]},
    "VehicleTurningRoutePedestrian":          {"swap_in_subgoals": ["assert_priority", "proceed_resume", "turn_on_green"]},
    # ---- 事故 / 对向 ----
    "Accident":                               {"swap_in_subgoals": ["assert_priority", "proceed_resume", "gap_accept_merge"]},
    "AccidentTwoWays":                        {"swap_in_subgoals": ["assert_priority", "proceed_resume", "gap_accept_merge"]},
    "OppositeVehicleRunningRedLight":         {"swap_in_subgoals": ["assert_priority", "proceed_resume", "yield_and_turn"]},
    "OppositeVehicleTakingPriority":          {"swap_in_subgoals": ["assert_priority", "proceed_resume", "yield_and_turn"]},
    # ---- 跟车 / 高速 / cut-in ----
    "HardBreakRoute":                         {"swap_in_subgoals": ["assert_priority", "proceed_resume", "gap_accept_merge"]},
    "HighwayCutIn":                           {"swap_in_subgoals": ["assert_priority", "proceed_resume", "gap_accept_merge"]},
    "HighwayExit":                            {"swap_in_subgoals": ["assert_priority", "proceed_resume", "yield_and_turn"]},
    "StaticCutIn":                            {"swap_in_subgoals": ["assert_priority", "proceed_resume", "gap_accept_merge"]},
    "ParkingCutIn":                           {"swap_in_subgoals": ["assert_priority", "proceed_resume", "gap_accept_merge"]},
    # ---- 红灯 / 其它 ----
    "RedLightWithoutLeadVehicle":             {"swap_in_subgoals": ["assert_priority", "proceed_resume", "turn_on_green"]},
    "ControlLoss":                            {"swap_in_subgoals": ["assert_priority", "proceed_resume", "gap_accept_merge"]},
    "ParkingExit":                            {"swap_in_subgoals": ["assert_priority", "proceed_resume", "gap_accept_merge"]},
}


# =========================================================================== #
# 2. dataclass：CF 请求 / 单 variant 描述
# =========================================================================== #


@dataclass
class CFRequest:
    """从 config / CLI 解析出的一条 CF 请求。"""

    subgoal: str
    target_scenario: str = ""
    target_status: str = ""
    source: str = "cli"


@dataclass
class CFVariant:
    """一个待跑的变体；不含 noop，floor 改由 truth 自身跨 seed pairwise 提供。"""

    tag: str
    memory: DrivingMemory
    mode: str
    requested_subgoal: str
    request_source: str = ""
    is_truth: bool = False
    prompt_consistency: str = "consistent"
    warning: str = ""


# =========================================================================== #
# 3. 样本挑选 / 文件链接 / 小工具
# =========================================================================== #


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


def _parse_float_csv(text: str, fallback: float) -> List[float]:
    """解析逗号分隔的 float 列表；空串回退到 fallback。"""

    vals: List[float] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        vals.append(float(item))
    return vals or [float(fallback)]


def _safe_name(text: str) -> str:
    """把事件名转成文件夹安全的短名字。"""

    keep = []
    for ch in str(text):
        keep.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(keep).strip("_") or "empty"


def _format_cfg_tag(cfg_scale: float) -> str:
    return f"cfg_{str(cfg_scale).replace('.', 'p').replace('-', 'm')}"


# =========================================================================== #
# 4. 合法 (STATUS, SUBGOAL) pair 索引
# =========================================================================== #


def _all_legal_pairs() -> Dict[Tuple[str, str], List[Tuple[str, int]]]:
    pairs: Dict[Tuple[str, str], List[Tuple[str, int]]] = defaultdict(list)
    for scenario in sorted(SCENARIO_EVENT_SEQUENCES):
        seq = get_full_sequence(scenario)
        for idx in range(len(seq) - 1):
            pairs[(seq[idx], seq[idx + 1])].append((scenario, idx))
    return pairs


LEGAL_EVENT_PAIRS = _all_legal_pairs()

SUBGOAL_ONLY_AUTO_PRIORITY = [
    # status=initial 时常见的跨语义入口，优先让自动候选覆盖不同场景簇。
    "hazard_detect",
    "pedestrian_detect",
    "obstacle_detect",
    "flow_approach",
    "cutin_onset",
    "junction_approach",
    # status=junction_approach 等中间态常见动作。
    "assert_priority",
    "turn_on_green",
    "brake_at_light",
    "yield_and_turn",
    "wait_or_turn_on_green",
    "yield_and_enter_flow",
]


def _auto_subgoal_only_requests(
    memory: DrivingMemory,
    seen_subgoals: Set[str],
    limit: int = 3,
) -> List[CFRequest]:
    """为 subgoal_only 兜底生成当前 STATUS 下的合法 SUBGOAL 候选。

    默认 config 是按 scenario 设计的跨语义候选；当随机 case 还处在 initial
    等早期 STATUS 时，那些候选可能全部不是合法相邻 pair。这里只在没有任何
    有效 variant 时补齐，不改变用户显式传入且已命中的配置行为。
    """

    candidates: List[str] = []
    for (status, subgoal), _owners in LEGAL_EVENT_PAIRS.items():
        if status != memory.status:
            continue
        if subgoal in seen_subgoals or subgoal not in EVENT_DESCRIPTIONS:
            continue
        candidates.append(subgoal)

    priority = {name: idx for idx, name in enumerate(SUBGOAL_ONLY_AUTO_PRIORITY)}
    candidates = sorted(
        set(candidates),
        key=lambda name: (priority.get(name, 10_000), name),
    )
    return [
        CFRequest(subgoal=subgoal, source=f"auto_subgoal_only:{memory.status}")
        for subgoal in candidates[:limit]
    ]


# =========================================================================== #
# 5. CF 请求加载（config / CLI）
# =========================================================================== #


def _load_counterfactual_config(path_or_default: str) -> Dict[str, Any]:
    if not path_or_default:
        return {}
    if path_or_default.lower() == "default":
        return DEFAULT_COUNTERFACTUAL_CONFIG
    path = pathlib.Path(path_or_default)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _requests_from_config_entry(entry: Any, source: str) -> List[CFRequest]:
    if not entry:
        return []
    if isinstance(entry, list):
        raw_items = entry
    elif isinstance(entry, dict):
        raw_items = entry.get("swap_in_subgoals", [])
    else:
        raise TypeError(f"不支持的 counterfactual config entry 类型：{type(entry)}")

    out: List[CFRequest] = []
    for item in raw_items:
        if isinstance(item, str):
            out.append(CFRequest(subgoal=item, source=source))
        elif isinstance(item, dict):
            subgoal = str(item.get("subgoal") or item.get("event") or "").strip()
            if not subgoal:
                raise ValueError(f"counterfactual config item 缺少 subgoal/event: {item}")
            out.append(CFRequest(
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
    config_source_label: str,
) -> Tuple[List[CFRequest], str, str]:
    """返回 (requests, request_source, warn_message)。

    优先级：config 命中 > CLI fallback。命中失败时给出明确 warn，避免静默无 CF。
    """

    if config and scenario in config:
        return (
            _requests_from_config_entry(config[scenario], f"config:{scenario}"),
            "config",
            "",
        )
    if cli_subgoals:
        warn = ""
        if config:
            warn = (
                f"scenario={scenario!r} 不在 counterfactual config 中"
                f"（source={config_source_label}），回退到 --counterfactual-subgoals"
            )
        return [CFRequest(subgoal=s, source="cli") for s in cli_subgoals], "cli", warn
    warn = ""
    if config:
        warn = (
            f"scenario={scenario!r} 不在 counterfactual config 中"
            f"（source={config_source_label}），且 --counterfactual-subgoals 为空，"
            f"该 case 跳过 CF；请把该 scenario 加入 config 或显式传 CF 列表"
        )
    return [], "none", warn


# =========================================================================== #
# 6. 反查 scenario_swap 的前驱 / 构造 CF variant
# =========================================================================== #


def _find_predecessor_for_subgoal(
    request: CFRequest,
    original_scenario: str,
) -> Optional[Tuple[str, int]]:
    """为 scenario_swap 找一个包含目标 SUBGOAL 的合法前驱 (scenario, idx)。"""

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
    request: CFRequest,
) -> Tuple[Optional[DrivingMemory], str]:
    found = _find_predecessor_for_subgoal(request, memory.scenario)
    if found is None:
        return None, f"subgoal {request.subgoal!r} 不存在于任何场景状态机"
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
    request: CFRequest,
) -> Tuple[Optional[DrivingMemory], str, str]:
    """subgoal_only：保留原 scenario/STATUS，只换 SUBGOAL。

    要求 (原 STATUS, CF SUBGOAL) 至少在某个场景中合法相邻；否则跳过——
    这种 pair 在状态机里不存在，喂给模型纯属噪声。
    """

    pair = (memory.status, request.subgoal)
    legal_somewhere = pair in LEGAL_EVENT_PAIRS
    if not legal_somewhere:
        return None, "invalid_pair", (
            f"(STATUS={memory.status!r}, SUBGOAL={request.subgoal!r}) "
            "不是任何场景状态机里的合法相邻转移"
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
    requests: List[CFRequest],
    mode: str,
) -> Tuple[List[CFVariant], List[Dict[str, Any]]]:
    """构造 truth + CF variant；不再含 noop，floor 改由 truth 跨 seed pairwise。"""

    variants: List[CFVariant] = [
        CFVariant(
            tag="truth",
            memory=memory,
            mode=mode,
            requested_subgoal=memory.subgoal,
            is_truth=True,
        )
    ]
    skipped: List[Dict[str, Any]] = []
    seen_subgoals = {memory.subgoal}
    cf_idx = 1

    def _try_add_request(request: CFRequest) -> bool:
        nonlocal cf_idx
        if request.subgoal == memory.subgoal:
            skipped.append({
                "subgoal": request.subgoal,
                "mode": mode,
                "reason": "same_as_truth_subgoal",
            })
            return False
        if request.subgoal not in EVENT_DESCRIPTIONS:
            skipped.append({
                "subgoal": request.subgoal,
                "mode": mode,
                "reason": "unknown_event_token",
            })
            return False
        if request.subgoal in seen_subgoals:
            return False

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
            return False

        tag = f"cf_{cf_idx:02d}_{_safe_name(request.subgoal)}"
        variants.append(CFVariant(
            tag=tag,
            memory=cf_memory,
            mode=mode,
            requested_subgoal=request.subgoal,
            request_source=request.source,
            prompt_consistency=consistency,
            warning=warning,
        ))
        seen_subgoals.add(request.subgoal)
        cf_idx += 1
        if warning:
            print(f"[probe][cf][warn] mode={mode} tag={tag}: {warning}")
        return True

    for request in requests:
        _try_add_request(request)

    if mode == "subgoal_only" and len(variants) == 1:
        auto_requests = _auto_subgoal_only_requests(memory, seen_subgoals, limit=3)
        if auto_requests:
            print(
                f"[probe][cf][auto] mode=subgoal_only status={memory.status!r}: "
                f"原请求没有有效 variant，自动补合法 SUBGOAL="
                f"{[r.subgoal for r in auto_requests]}"
            )
            for request in auto_requests:
                _try_add_request(request)
        else:
            print(
                f"[probe][cf][auto] mode=subgoal_only status={memory.status!r}: "
                "没有可用的其它合法 SUBGOAL，仍只保留 truth"
            )
    return variants, skipped


# =========================================================================== #
# 7. Pairwise floor / per-CF delta 聚合
# =========================================================================== #


def _pairwise_floor(
    truth_items: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Optional[Dict[str, float]]:
    """truth 跨 seed pairwise 差异 → 采样噪声 floor。

    truth_items: list of (z1_pred, rgb_pred)，每个 seed 一条；同 cfg 下 N 条。
    返回 None 表示 seed_replicates < 2 → 没法算 floor。
    """

    if len(truth_items) < 2:
        return None
    pl_vals: List[float] = []
    lm_vals: List[float] = []
    for i in range(len(truth_items)):
        for j in range(i + 1, len(truth_items)):
            zi, ri = truth_items[i]
            zj, rj = truth_items[j]
            lm_vals.append(float(F.mse_loss(zi.float(), zj.float()).item()))
            l1, _ = pixel_l1_psnr(ri, rj)
            pl_vals.append(l1)
    return {
        "latent_mse_mean": sum(lm_vals) / len(lm_vals),
        "pixel_l1_mean":   sum(pl_vals) / len(pl_vals),
        "n_pairs":         len(pl_vals),
    }


def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {"mean": mean, "std": math.sqrt(var), "count": len(values)}


def _verdict_from_ratio(ratio: Optional[float]) -> str:
    """把 ratio_over_floor 翻译成一句人话。

    阈值：
    - None / floor 不可用 → "insufficient_seeds"
    - ratio < 2  → "near_floor"（基本是采样噪声）
    - 2 ≤ ratio < 5 → "weak_response"（模型有反应但弱）
    - 5 ≤ ratio < 15 → "responsive"（明显响应 SUBGOAL）
    - ratio ≥ 15 → "highly_responsive"
    """

    if ratio is None:
        return "insufficient_seeds"
    if ratio < 2.0:
        return "near_floor"
    if ratio < 5.0:
        return "weak_response"
    if ratio < 15.0:
        return "responsive"
    return "highly_responsive"


# =========================================================================== #
# 8. 图像合成：一张 cf_overview_<mode>.png
# =========================================================================== #


def _load_cell(path: pathlib.Path) -> Optional[Image.Image]:
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")


_ASCII_FALLBACK = {
    "↓": "v",      # ↓ down arrow
    "↑": "^",      # ↑ up arrow
    "→": "->",     # →
    "←": "<-",     # ←
    "Δ": "d",      # Δ delta
    "⚠": "!",      # ⚠ warning
    "≤": "<=",     # ≤
    "≥": ">=",     # ≥
    "±": "+-",     # ±
    "—": "--",     # — em dash
    "–": "-",      # – en dash
    "·": ".",      # · middle dot
    """: '"', """: '"',
    "'": "'", "'": "'",
}


def _to_ascii_safe(text: str) -> str:
    """PIL 默认 bitmap font 只支持 latin-1，遇到 Unicode 字符会抛
    `UnicodeEncodeError` 并中断整个 grid 渲染（之前 cf_overview 只画到 truth
    行的 bug 就是这么来的）。在 draw.text 前把已知特殊字符替换成 ASCII 同义符；
    剩下的非 latin-1 字符统一用 ``_`` 占位，宁可丢字也不要丢图。"""

    out_chars: List[str] = []
    for ch in str(text):
        if ch in _ASCII_FALLBACK:
            out_chars.append(_ASCII_FALLBACK[ch])
            continue
        try:
            ch.encode("latin-1")
            out_chars.append(ch)
        except UnicodeEncodeError:
            out_chars.append("_")
    return "".join(out_chars)


def _draw_multiline(
    draw: Optional[ImageDraw.ImageDraw],
    xy: Tuple[int, int],
    lines: List[str],
    fill: Tuple[int, int, int] = (0, 0, 0),
    line_h: int = 14,
) -> None:
    if draw is None:
        return
    x, y = xy
    for i, line in enumerate(lines):
        safe = _to_ascii_safe(line)
        try:
            draw.text((x, y + i * line_h), safe, fill=fill)
        except Exception as exc:  # noqa: BLE001 — 兜底，绝不让画文字异常中断 grid
            print(f"[probe][cf][warn] draw.text 失败，跳过该行：{exc}; line={safe!r}")


def _cf_overview_label_lines(record: Dict[str, Any], truth_scenario: str) -> List[str]:
    """生成 cf_overview 左侧标签。

    overview PNG 使用 PIL 默认字体，中文会被降级成问号；图片里只放 ASCII 摘要，
    完整 warning 仍保存在 cf_summary.json / cf_report.md。
    """

    label = ("truth" if record["is_truth"] else record["tag"]) + f": {record['subgoal']}"
    lines = [label]
    if record.get("request_source"):
        lines.append(f"source={record['request_source']}")
    if record["scenario"] and record["scenario"] != truth_scenario:
        lines.append(f"scenario={record['scenario']}")
    consistency = record.get("prompt_consistency")
    if consistency and consistency != "consistent":
        lines.append(f"prompt={consistency}")
    if record.get("warning"):
        lines.append("detail=cf_report.md")
    return lines


def _compose_cf_overview(
    out_path: pathlib.Path,
    target_path: pathlib.Path,
    rows: List[Dict[str, Any]],
    col_headers: List[str],
    case_header: str,
) -> Optional[pathlib.Path]:
    """画一张 cf_overview_<mode>.png。

    布局：
      [HEADER 一行：case 名 + truth subgoal]
      [target_raw |  CFG=col0 col label | CFG=col1 ... ]
      [row0 label | row0 cell0 | row0 cell1 | ...]
      [row1 label | ...]
      ...

    rows: 每项形如
      {"label": "truth: yield_and_turn",
       "annot_lines": ["truth", "scenario=..."],
       "cells": [{"img_path": Path, "annot": ["dpix=...", "r=...x", "responsive"]}, ...]}
    col_headers: 每个 CFG 的列头文字（None 时不画列头行）
    """

    target_img = _load_cell(target_path)
    if target_img is None:
        return None
    # 选 cell 尺寸：取 target 高度 / 4 作为 cell 高，等比缩放
    cell_h = max(96, target_img.height // 2)
    target_disp_w = int(target_img.width * (cell_h / target_img.height))
    cell_w_default = target_disp_w

    # 收集所有 cell 图，统一尺寸
    cell_w = cell_w_default
    for row in rows:
        for cell in row.get("cells", []):
            img = _load_cell(pathlib.Path(cell["img_path"])) if cell.get("img_path") else None
            cell["_img"] = img
            if img is not None:
                w = int(img.width * (cell_h / img.height))
                cell_w = max(cell_w, w)

    target_resized = target_img.resize((cell_w, cell_h), Image.BICUBIC)

    header_h = 32
    col_header_h = 24
    annot_h = 36           # 每个 cell 顶部留 2 行 annot
    row_label_w = 220
    margin = 16

    n_cols = len(col_headers)
    n_rows = len(rows)

    canvas_w = margin * 2 + row_label_w + (n_cols + 1) * cell_w  # +1 for target column
    canvas_h = (
        margin
        + header_h
        + col_header_h
        + n_rows * (annot_h + cell_h)
        + margin
    )

    img = Image.new("RGB", (canvas_w, canvas_h), "white")
    try:
        draw = ImageDraw.Draw(img)
    except Exception:
        draw = None

    # ---- header ----
    _draw_multiline(draw, (margin, margin), [case_header], fill=(0, 0, 0), line_h=14)

    # ---- column headers ----
    col_header_y = margin + header_h
    # 左上"target / row label" 区域
    _draw_multiline(
        draw,
        (margin, col_header_y),
        ["row label", "target ->"],
        line_h=12,
    )
    # target 列
    target_x = margin + row_label_w
    _draw_multiline(draw, (target_x + 4, col_header_y), ["target_raw"], line_h=12)
    # CFG 列
    for ci, header in enumerate(col_headers):
        x = margin + row_label_w + (ci + 1) * cell_w
        _draw_multiline(draw, (x + 4, col_header_y), [header], line_h=12)

    # ---- rows ----
    row_y = col_header_y + col_header_h
    for ri, row in enumerate(rows):
        # row label（左侧多行说明）
        _draw_multiline(
            draw,
            (margin, row_y + 4),
            row.get("annot_lines", [row.get("label", "")]),
            line_h=12,
        )
        # target 列：只在第一行画 target_raw
        if ri == 0:
            img.paste(target_resized, (target_x, row_y + annot_h))
        # 每个 CFG cell
        for ci, cell in enumerate(row.get("cells", [])):
            x = margin + row_label_w + (ci + 1) * cell_w
            # 顶部 annot
            _draw_multiline(
                draw,
                (x + 4, row_y),
                cell.get("annot", []),
                line_h=12,
            )
            cell_img = cell.get("_img")
            if cell_img is None:
                # 画一个灰底占位
                if draw is not None:
                    draw.rectangle(
                        [x, row_y + annot_h, x + cell_w, row_y + annot_h + cell_h],
                        fill=(230, 230, 230),
                    )
                    draw.text((x + 8, row_y + annot_h + 8), "n/a", fill=(80, 80, 80))
            else:
                resized = cell_img.resize((cell_w, cell_h), Image.BICUBIC)
                img.paste(resized, (x, row_y + annot_h))
        row_y += annot_h + cell_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


# =========================================================================== #
# 9. 带 Euler 轨迹的采样
# =========================================================================== #


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
    """跑 Euler 采样并记录每一步 v_pred 与真值方向的 cosine、z_t 到 GT 的 L2。"""

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
        v_ref = (z1_gt - z_init).float().flatten(1)
        v_p = v_pred.float().flatten(1)
        cos = float(F.cosine_similarity(v_p, v_ref, dim=1).mean().item())
        l2 = float((z_t.float() - z1_gt.float()).flatten(1).norm(dim=1).mean().item())
        trace["t"].append(t_val)
        trace["v_cos_vs_gt_direction"].append(cos)
        trace["z_l2_to_gt"].append(l2)
        z_t = z_t + v_pred * dt
    return z_t, trace


# =========================================================================== #
# 10. Overview markdown（普通 probe + 简要 CF 引用）
# =========================================================================== #


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
    lines.append(f"- euler_steps: {meta.get('euler_steps')}, baseline seed: {meta.get('seed')}")
    lines.append(f"- inference elapsed (truth, baseline): {meta.get('elapsed_sec'):.3f}s")
    lines.append("")

    lines.append("## Truth metrics (vs GT keyframe)")
    lines.append("| metric | value | 说明 |")
    lines.append("|---|---|---|")
    lines.append(f"| latent_mse | {metrics['latent_mse']:.6f} | MSE(z1_pred, z1_gt) |")
    lines.append(f"| latent_cos | {metrics['latent_cos']:.4f} | cosine 越接近 1 越好 |")
    lines.append(f"| pixel_l1   | {metrics['pixel_l1']:.4f} | 解码 RGB L1 |")
    lines.append(f"| psnr       | {metrics['psnr']:.2f} | 越高越好；地板 = VAE 重建 PSNR |")
    lines.append(f"| velocity_cos | {metrics['velocity_cos']:.4f} | 5 个 t 上平均 |")
    lines.append("")

    if meta.get("cf_overview"):
        lines.append("## Counterfactual experiment")
        lines.append(f"- overview image: `{meta.get('cf_overview')}`")
        if meta.get("cf_overview_extra"):
            for extra in meta["cf_overview_extra"]:
                lines.append(f"  - extra: `{extra}`")
        lines.append(f"- summary: `{meta.get('cf_summary')}`")
        lines.append(f"- report:  `{meta.get('cf_report')}`")
        lines.append(f"- modes:   {meta.get('cf_active_modes')}")
        lines.append(f"- cfg sweep: {meta.get('cf_cfg_sweep')}")
        lines.append(f"- seed replicates: {meta.get('cf_seed_replicates')}")
        cf_max_ratio = meta.get("cf_max_ratio")
        ratio_str = f"{cf_max_ratio:.2f}x" if isinstance(cf_max_ratio, (int, float)) else "n/a"
        lines.append(f"- max verdict (over all CF / mode / cfg): **{meta.get('cf_max_verdict')}** "
                     f"@ tag=`{meta.get('cf_max_tag') or 'n/a'}`, ratio={ratio_str}")
        lines.append("")

    if trace.get("t"):
        lines.append("## Euler trace (truth, baseline)")
        lines.append("`v_cos_vs_gt_direction` 越接近 1 越好；`z_l2_to_gt` 应随 t 单调下降。")
        lines.append("```")
        lines.append("step  t      v_cos    z_l2")
        for k in range(0, len(trace["t"]), max(1, len(trace["t"]) // 8)):
            lines.append(
                f"{k:4d}  {trace['t'][k]:.3f}  {trace['v_cos_vs_gt_direction'][k]:+.4f}  {trace['z_l2_to_gt'][k]:.2f}"
            )
        lines.append("```")
    return "\n".join(lines) + "\n"


# =========================================================================== #
# 11. CF report markdown
# =========================================================================== #


def _render_cf_report_md(
    case_header: str,
    mode_summaries: Dict[str, Any],
    case_meta: Dict[str, Any],
) -> str:
    """人类可读的 CF 报告，单一文件，能直接拷给别人看。

    case_meta 提供整次 case 的全局上下文（baseline_cfg / cfg_sweep / seed_replicates / 等），
    让读 report 的人不用同时打开 cf_summary.json 也能知道 floor 是基于哪个 cfg、几个 seed 算的。
    """

    lines: List[str] = []
    lines.append(f"# Counterfactual report — {case_header}")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- truth_subgoal: `{case_meta.get('truth_subgoal')}`")
    lines.append(f"- modes:           `{case_meta.get('active_modes')}`")
    lines.append(f"- cfg sweep:       `{case_meta.get('cfg_scale_sweep')}`  (baseline = first value)")
    lines.append(f"- seed replicates: `{case_meta.get('seed_replicates')}`")
    lines.append(f"- request source:  `{case_meta.get('request_source')}`")
    lines.append("")
    lines.append(
        "**实验问题**：把 Qwen teacher-forced prompt 里的 SUBGOAL 换成别的 token 后，"
        "GoalGen 解码出的子目标图像会不会跟着语义变化？"
    )
    lines.append("")
    lines.append(
        "**floor**：truth 自己在不同 z_init seed 下的两两 pairwise 差异；"
        "等价于「换 seed 跑同 prompt」会变多少 → 采样噪声地板。"
        "需要 seed_replicates ≥ 2 才能算；为 1 时 floor / ratio 为 null。"
    )
    lines.append("")
    lines.append(
        "**verdict 阈值（ratio = Δ_CF / floor）**：`<2` near_floor / `[2,5)` weak / "
        "`[5,15)` responsive / `≥15` highly_responsive。"
    )
    lines.append("")

    for mode, info in mode_summaries.items():
        lines.append(f"## mode = `{mode}`")
        lines.append("")
        lines.append(f"- compare image: `{info.get('overview_png')}`")
        if info.get("extra_overview_pngs"):
            for extra in info["extra_overview_pngs"]:
                lines.append(f"  - extra: `{extra}`")
        floor = info.get("noise_floor")
        if floor:
            lines.append(
                f"- noise floor (baseline cfg): "
                f"pixel_l1={floor.get('pixel_l1_mean', 0.0):.4f}, "
                f"latent_mse={floor.get('latent_mse_mean', 0.0):.6f}, "
                f"n_pairs={floor.get('n_pairs', 0)}"
            )
        else:
            lines.append(
                "- noise floor: **n/a**（seed_replicates < 2；ratio 字段会是 null）"
            )
        lines.append("")

        per_cf = info.get("per_cf", [])
        if not per_cf:
            lines.append("_no CF variants (config 未命中且 CLI 为空，或全部被 skip)_")
            lines.append("")
            continue
        lines.append("| tag | subgoal | source | scenario(CF) | Δpixel_l1 | Δlatent_mse | ratio_pix | verdict |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cf in per_cf:
            ratio_pix = cf.get("ratio_over_floor_pixel")
            ratio_str = f"{ratio_pix:.2f}x" if ratio_pix is not None else "n/a"
            lines.append(
                "| `{tag}` | `{sub}` | `{src}` | `{sc}` | {dp:.4f} | {dl:.6f} | {r} | **{v}** |".format(
                    tag=cf.get("tag"),
                    sub=cf.get("subgoal"),
                    src=cf.get("request_source") or "",
                    sc=cf.get("scenario") or "(unchanged)",
                    dp=cf.get("delta_pixel_l1_mean", 0.0),
                    dl=cf.get("delta_latent_mse_mean", 0.0),
                    r=ratio_str,
                    v=cf.get("verdict", "n/a"),
                )
            )
        skipped = info.get("skipped", [])
        if skipped:
            lines.append("")
            lines.append("**Skipped CF requests:**")
            for s in skipped:
                lines.append(f"- `{s.get('subgoal')}` — {s.get('reason')}")
        lines.append("")

    lines.append("## 如何读")
    lines.append("")
    lines.append(
        "1. 直接看 `cf_overview_<mode>.png`：第一行是 truth，下面每行是一个 CF；"
        "如果有 CFG sweep，列数就是 CFG 个数。每个 cell 上方标了 Δpixel_l1 和 ratio。"
    )
    lines.append(
        "2. verdict=`responsive` 或更强说明模型对 SUBGOAL 有显著反应；"
        "`near_floor` 说明模型基本无视 SUBGOAL，生成只跟 history+target_pose 走。"
    )
    lines.append(
        "3. CF 列对原始 target 的传统指标会变差，这正常——目标已经被人为换了。"
        "判断「听不听 SUBGOAL」只看 ratio 和并排图像的语义方向。"
    )
    return "\n".join(lines) + "\n"


# =========================================================================== #
# 12. 主流程
# =========================================================================== #


def _resolve_default_dit_checkpoint(save_root_hint: Optional[str] = None) -> str:
    """根据 --save-root 推 base 目录，再按"latest 子目录 > 老顶层"顺序找 ckpt。"""

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
                        help="DiT ckpt 路径；空 = 按 --save-root 自动解析 latest/best.pt。")
    parser.add_argument("--checkpoint-dir", default="checkpoints/Qwen3-VL-4B-Instruct")
    parser.add_argument("--save-root", default="checkpoints/goalgen_v1_dit",
                        help="case dump 落到 <save-root>/eval_cases/<scenario>__<run>__<anchor>/")
    parser.add_argument("--case-suffix", default="",
                        help="给 case 目录名加后缀，便于多 ckpt 同 seed 并排对比。")
    parser.add_argument("--scenarios", default="",
                        help="逗号分隔过滤；空 = 全场景。")
    # ---- counterfactual ----
    parser.add_argument("--counterfactual-subgoals", default="",
                        help="逗号分隔的 CF SUBGOAL token；config 未命中时回退用这个。")
    parser.add_argument("--counterfactual-mode",
                        choices=["scenario_swap", "subgoal_only", "both"],
                        default="scenario_swap",
                        help="scenario_swap 同时换 scenario/status/event_sequence 保持 prompt 自洽（推荐）；"
                             "subgoal_only 只换 SUBGOAL；both 同 case 下保存两套。")
    parser.add_argument("--counterfactual-config", default="",
                        help="per-scenario CF 配置；传 default 用内置；传路径加载 JSON。")
    parser.add_argument("--cfg-scale-sweep", default="",
                        help="逗号分隔 CFG 扫描，例如 0.0,1.0,2.0,4.0；空则只用 --cfg-scale。")
    parser.add_argument("--counterfactual-seed-replicates", type=int, default=1,
                        help=">=2 时启用 floor / ratio；默认 1 时只画图不算 ratio。")
    parser.add_argument("--cf-verbose-artifacts", action="store_true", default=False,
                        help="开启后额外落 chat_text.txt / memory.json / 各 seed pred.png + metrics.json。")
    # ---- 其它 ----
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
    parser.add_argument("--z0-prior-alpha", type=float, default=0.0)
    parser.add_argument("--z0-prior-sigma", type=float, default=1.0)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
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

    if not args.dit_checkpoint:
        args.dit_checkpoint = _resolve_default_dit_checkpoint(args.save_root)
        print(f"[ckpt] --dit-checkpoint 未指定，自动解析 = {args.dit_checkpoint}")

    case_root = pathlib.Path(args.save_root) / "eval_cases"
    case_root.mkdir(parents=True, exist_ok=True)
    from qwen3vl_local.run_log import install_output_log
    install_output_log(case_root)

    samples = load_jsonl(pathlib.Path(args.val_jsonl))
    scenarios_filter = [s.strip() for s in args.scenarios.split(",") if s.strip()] or None
    cli_subgoals = _parse_csv_tokens(args.counterfactual_subgoals)
    cf_config = _load_counterfactual_config(args.counterfactual_config)
    config_source_label = (
        "built-in default"
        if args.counterfactual_config.lower() == "default"
        else (args.counterfactual_config or "<empty>")
    )
    cf_modes = (
        ["scenario_swap", "subgoal_only"]
        if args.counterfactual_mode == "both"
        else [args.counterfactual_mode]
    )
    cfg_scale_values = _parse_float_csv(args.cfg_scale_sweep, args.cfg_scale)
    seed_replicates = max(1, int(args.counterfactual_seed_replicates))
    picked = select_samples(samples, scenarios_filter, args.num_per_scenario, args.seed)
    print(f"[probe] selected {len(picked)} samples from {len(samples)} total")
    cf_enabled = bool(cli_subgoals or cf_config)
    if cf_enabled:
        print(f"[probe] counterfactual modes={cf_modes} cfg_sweep={cfg_scale_values} "
              f"seed_replicates={seed_replicates} config={config_source_label} "
              f"cli_fallback={cli_subgoals or '<empty>'}")
        if seed_replicates < 2:
            print("[probe][cf] seed_replicates<2: 不计算 floor / ratio；"
                  "想要量化结论请加 --counterfactual-seed-replicates 2 或以上。")

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
    cf_index_records: List[Dict[str, Any]] = []

    for sample_idx, sample in picked:
        scenario = sample.get("scenario", "unknown")
        run_id = sample.get("run_id", "norun")
        anchor = sample.get("anchor", "noanchor")
        case_name = f"{scenario}__{run_id}__{anchor}{args.case_suffix}"
        case_dir = case_root / case_name
        (case_dir / "input_history").mkdir(parents=True, exist_ok=True)

        # ---- 1) symlink 历史 + target raw ----
        for k, p in enumerate(sample.get("history_rgb_paths", [])):
            link_or_copy(p, case_dir / "input_history" / f"{k:02d}.jpg")
        if sample.get("target_rgb_path"):
            link_or_copy(sample["target_rgb_path"], case_dir / "target_raw.jpg")

        # ---- 2) 公共编码 ----
        history_images = [load_rgb(p) for p in sample["history_rgb_paths"]]
        target_img = load_rgb(sample["target_rgb_path"])
        memory = memory_from_sample(sample)

        z_history = vae.encode(history_images).to(dtype=dit_dtype).unsqueeze(0)
        z1_gt = vae.encode([target_img]).to(dtype=dit_dtype)
        z1_gt_for_vae = z1_gt.to(device=vae.device, dtype=vae.dtype)
        rgb_gt = vae.decode(z1_gt_for_vae).clamp(-1.0, 1.0)

        # ---- 3) 准备 CF 请求 + z_init seeds ----
        if cf_enabled:
            cf_requests, cf_request_source, cf_warn = _counterfactual_requests_for_sample(
                scenario, cf_config, cli_subgoals, config_source_label,
            )
            if cf_warn:
                print(f"[probe][cf][warn] {cf_warn}")
        else:
            cf_requests, cf_request_source, cf_warn = [], "none", ""

        dump_cf = bool(cf_requests)
        active_modes = cf_modes if dump_cf else [cf_modes[0]]

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

        # ---- 4) 跑所有 variant × cfg × seed ----
        truth_metrics: Optional[Dict[str, float]] = None
        truth_trace: Optional[Dict[str, List[float]]] = None
        truth_elapsed = 0.0

        # 每个 mode 一份汇总（用于 cf_summary.json + cf_report.md）
        mode_payloads: Dict[str, Dict[str, Any]] = {}
        case_max_ratio: Optional[float] = None
        case_max_ratio_tag: str = ""
        case_max_verdict: str = "insufficient_seeds"

        baseline_cfg = float(cfg_scale_values[0])

        for mode in active_modes:
            if dump_cf:
                variants, skipped = _make_counterfactual_variants(memory, cf_requests, mode)
            else:
                variants = [CFVariant(tag="truth", memory=memory, mode=mode,
                                       requested_subgoal=memory.subgoal, is_truth=True)]
                skipped = []

            # 收集每个 variant 的预测结果：variant_records[tag] = {...}
            variant_records: Dict[str, Dict[str, Any]] = {}
            # truth_preds_by_cfg[cfg] = [(z1_pred, rgb_pred), ...] 跨 seed
            truth_preds_by_cfg: Dict[float, List[Tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)

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
                # variant_dir 在 verbose 模式下用
                variant_dir = case_dir / "counterfactual" / mode / variant.tag

                for cfg_scale_value in cfg_scale_values:
                    for rep_idx, (replicate_seed, z_init) in enumerate(z_inits):
                        t0 = time.time()
                        z1_pred, trace = euler_sample_with_trace(
                            dit, z_history, pooled_kv, z1_gt, z_init,
                            num_steps=args.euler_steps,
                            cfg_scale=cfg_scale_value,
                        )
                        m_mse = latent_mse(z1_pred, z1_gt)
                        m_cos = latent_cosine(z1_pred, z1_gt)
                        z1_pred_for_vae = z1_pred.to(device=vae.device, dtype=vae.dtype)
                        rgb_pred = vae.decode(z1_pred_for_vae).clamp(-1.0, 1.0)
                        m_l1, m_psnr = pixel_l1_psnr(rgb_pred, rgb_gt)
                        m_vcos = velocity_cosine_multi_t(
                            dit, z_history, pooled_kv, z1_gt,
                            device, dit_dtype,
                            z0_prior_alpha=args.z0_prior_alpha,
                            z0_prior_sigma=args.z0_prior_sigma,
                        )
                        elapsed = time.time() - t0

                        metrics_vs_gt = {
                            "latent_mse": m_mse, "latent_cos": m_cos,
                            "pixel_l1": m_l1, "psnr": m_psnr,
                            "velocity_cos": m_vcos,
                        }

                        # truth：缓存 pred 给 floor / 给 grid；首组也作 case-level truth baseline
                        is_baseline = (
                            variant.is_truth
                            and mode == active_modes[0]
                            and float(cfg_scale_value) == baseline_cfg
                            and rep_idx == 0
                        )
                        if variant.is_truth:
                            truth_preds_by_cfg[float(cfg_scale_value)].append(
                                (z1_pred.detach().clone(), rgb_pred.detach().clone())
                            )

                        # 决定预测图落盘位置：
                        # - verbose：全部落 counterfactual/<mode>/<tag>/cfg_*/seed_*/pred.png
                        # - 否则：truth 的 baseline (mode/cfg/seed=首组) 仍写 case_dir/pred.png；
                        #         其它只在内存里留着拼 cf_overview，最后用临时文件
                        pred_path: Optional[pathlib.Path] = None
                        if is_baseline:
                            _save_rgb_png(rgb_pred[0], case_dir / "pred.png")
                            pred_path = case_dir / "pred.png"
                            truth_metrics = metrics_vs_gt
                            truth_trace = trace
                            truth_elapsed = elapsed + prefill_elapsed

                        if args.cf_verbose_artifacts and dump_cf:
                            cfg_tag = _format_cfg_tag(float(cfg_scale_value))
                            run_dir = variant_dir / cfg_tag / f"seed_{rep_idx:02d}"
                            run_dir.mkdir(parents=True, exist_ok=True)
                            run_pred = run_dir / "pred.png"
                            _save_rgb_png(rgb_pred[0], run_pred)
                            with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
                                json.dump(metrics_vs_gt, f, ensure_ascii=False, indent=2)
                            if pred_path is None:
                                pred_path = run_pred

                        # 拼 cf_overview 的图：每个 (variant, cfg) 只拿 rep_idx=0 即可
                        # 非 verbose 时给 truth/baseline 之外的 cell 在 tmp 子目录写一份
                        if pred_path is None and dump_cf and rep_idx == 0:
                            cfg_tag = _format_cfg_tag(float(cfg_scale_value))
                            tmp_dir = case_dir / ".cf_tmp" / mode / variant.tag / cfg_tag
                            tmp_dir.mkdir(parents=True, exist_ok=True)
                            pred_path = tmp_dir / "pred.png"
                            _save_rgb_png(rgb_pred[0], pred_path)

                        runs.append({
                            "cfg_scale": float(cfg_scale_value),
                            "replicate_idx": rep_idx,
                            "seed": replicate_seed,
                            "metrics_vs_gt": metrics_vs_gt,
                            "pred_path": str(pred_path) if pred_path is not None else "",
                            "elapsed_sec": elapsed,
                            # delta vs truth_pred 用 seed-matched 同 (cfg, rep_idx) 的 truth pred
                            "z1_pred": z1_pred.detach(),
                            "rgb_pred": rgb_pred.detach(),
                        })

                # 可选 verbose 写 chat_text + memory
                if args.cf_verbose_artifacts and dump_cf:
                    variant_dir.mkdir(parents=True, exist_ok=True)
                    (variant_dir / "chat_text.txt").write_text(prefill.chat_text, encoding="utf-8")
                    with (variant_dir / "memory.json").open("w", encoding="utf-8") as f:
                        json.dump(asdict(variant.memory), f, ensure_ascii=False, indent=2)

                variant_records[variant.tag] = {
                    "tag": variant.tag,
                    "mode": mode,
                    "subgoal": variant.memory.subgoal,
                    "requested_subgoal": variant.requested_subgoal,
                    "request_source": variant.request_source,
                    "scenario": variant.memory.scenario,
                    "status": variant.memory.status,
                    "is_truth": variant.is_truth,
                    "prompt_consistency": variant.prompt_consistency,
                    "warning": variant.warning,
                    "prefill_elapsed_sec": prefill_elapsed,
                    "runs": runs,
                }

            # ---- 计算 floor (per cfg) + per-CF delta ----
            floor_by_cfg: Dict[float, Optional[Dict[str, float]]] = {}
            for cfg_v in cfg_scale_values:
                floor_by_cfg[float(cfg_v)] = _pairwise_floor(truth_preds_by_cfg.get(float(cfg_v), []))

            baseline_floor = floor_by_cfg.get(baseline_cfg)
            per_cf_summaries: List[Dict[str, Any]] = []
            for tag, record in variant_records.items():
                if record["is_truth"]:
                    continue
                # 同 (cfg, rep_idx) 与 truth 的 z/rgb 对比
                latent_vals: List[float] = []
                pixel_vals: List[float] = []
                per_cfg_summary: List[Dict[str, Any]] = []
                latent_vals_by_cfg: Dict[float, List[float]] = defaultdict(list)
                pixel_vals_by_cfg: Dict[float, List[float]] = defaultdict(list)

                truth_runs = variant_records["truth"]["runs"]
                truth_lookup = {
                    (run["cfg_scale"], run["replicate_idx"]): (run["z1_pred"], run["rgb_pred"])
                    for run in truth_runs
                }
                for run in record["runs"]:
                    key = (run["cfg_scale"], run["replicate_idx"])
                    if key not in truth_lookup:
                        continue
                    tz, tr = truth_lookup[key]
                    lm = float(F.mse_loss(run["z1_pred"].float(), tz.float()).item())
                    pl, _ = pixel_l1_psnr(run["rgb_pred"], tr)
                    latent_vals.append(lm)
                    pixel_vals.append(pl)
                    latent_vals_by_cfg[float(run["cfg_scale"])].append(lm)
                    pixel_vals_by_cfg[float(run["cfg_scale"])].append(pl)

                latent_stats = _mean_std(latent_vals)
                pixel_stats = _mean_std(pixel_vals)

                # baseline ratio：用 baseline_cfg 下的 delta / floor
                baseline_pix_vals = pixel_vals_by_cfg.get(baseline_cfg, [])
                baseline_lat_vals = latent_vals_by_cfg.get(baseline_cfg, [])
                baseline_pix_mean = (sum(baseline_pix_vals) / len(baseline_pix_vals)
                                     if baseline_pix_vals else None)
                baseline_lat_mean = (sum(baseline_lat_vals) / len(baseline_lat_vals)
                                     if baseline_lat_vals else None)
                if baseline_floor is not None and baseline_pix_mean is not None:
                    ratio_pix = baseline_pix_mean / max(baseline_floor["pixel_l1_mean"], 1e-9)
                else:
                    ratio_pix = None
                if baseline_floor is not None and baseline_lat_mean is not None:
                    ratio_lat = baseline_lat_mean / max(baseline_floor["latent_mse_mean"], 1e-9)
                else:
                    ratio_lat = None

                # per-cfg breakdown
                for cfg_v in cfg_scale_values:
                    cfg_v_f = float(cfg_v)
                    pix_vals = pixel_vals_by_cfg.get(cfg_v_f, [])
                    lat_vals = latent_vals_by_cfg.get(cfg_v_f, [])
                    pix_mean = sum(pix_vals) / len(pix_vals) if pix_vals else None
                    lat_mean = sum(lat_vals) / len(lat_vals) if lat_vals else None
                    f_cfg = floor_by_cfg.get(cfg_v_f)
                    if f_cfg is not None and pix_mean is not None:
                        r_pix_cfg = pix_mean / max(f_cfg["pixel_l1_mean"], 1e-9)
                    else:
                        r_pix_cfg = None
                    per_cfg_summary.append({
                        "cfg_scale": cfg_v_f,
                        "delta_pixel_l1_mean": pix_mean,
                        "delta_latent_mse_mean": lat_mean,
                        "ratio_over_floor_pixel": r_pix_cfg,
                        "verdict": _verdict_from_ratio(r_pix_cfg),
                    })

                verdict = _verdict_from_ratio(ratio_pix)
                if ratio_pix is not None:
                    if case_max_ratio is None or ratio_pix > case_max_ratio:
                        case_max_ratio = ratio_pix
                        case_max_ratio_tag = f"{mode}/{tag}"
                        case_max_verdict = verdict

                per_cf_summaries.append({
                    "tag": tag,
                    "subgoal": record["subgoal"],
                    "requested_subgoal": record["requested_subgoal"],
                    "request_source": record["request_source"],
                    "scenario": record["scenario"] if record["scenario"] != memory.scenario else "",
                    "status": record["status"],
                    "prompt_consistency": record["prompt_consistency"],
                    "warning": record["warning"],
                    "delta_pixel_l1_mean": pixel_stats["mean"],
                    "delta_pixel_l1_std":  pixel_stats["std"],
                    "delta_latent_mse_mean": latent_stats["mean"],
                    "delta_latent_mse_std":  latent_stats["std"],
                    "ratio_over_floor_pixel":  ratio_pix,
                    "ratio_over_floor_latent": ratio_lat,
                    "verdict": verdict,
                    "per_cfg": per_cfg_summary,
                })

            # ---- 画 cf_overview_<mode>.png（默认只画 1 张：行=variant, 列=cfg sweep）----
            overview_png: Optional[pathlib.Path] = None
            extra_overview: List[str] = []
            if dump_cf:
                col_headers = [f"CFG={c:g}" for c in cfg_scale_values]
                rows_for_grid: List[Dict[str, Any]] = []
                for tag, record in variant_records.items():
                    label_lines = _cf_overview_label_lines(record, memory.scenario)
                    cells: List[Dict[str, Any]] = []
                    cf_match = next(
                        (cf for cf in per_cf_summaries if cf["tag"] == tag),
                        None,
                    )
                    for cfg_v in cfg_scale_values:
                        cfg_v_f = float(cfg_v)
                        # 找 rep_idx=0 那张 pred.png
                        run_match = next(
                            (run for run in record["runs"]
                             if run["cfg_scale"] == cfg_v_f and run["replicate_idx"] == 0),
                            None,
                        )
                        img_path = pathlib.Path(run_match["pred_path"]) if run_match and run_match.get("pred_path") else None
                        annot = []
                        if record["is_truth"]:
                            f_cfg = floor_by_cfg.get(cfg_v_f)
                            if f_cfg is not None:
                                annot.append(f"floor pix={f_cfg['pixel_l1_mean']:.3f}")
                        elif cf_match:
                            per_cfg = next(
                                (p for p in cf_match["per_cfg"] if p["cfg_scale"] == cfg_v_f),
                                None,
                            )
                            if per_cfg and per_cfg["delta_pixel_l1_mean"] is not None:
                                annot.append(f"dpix={per_cfg['delta_pixel_l1_mean']:.3f}")
                                if per_cfg["ratio_over_floor_pixel"] is not None:
                                    annot.append(f"r={per_cfg['ratio_over_floor_pixel']:.1f}x")
                                annot.append(per_cfg["verdict"])
                        cells.append({"img_path": img_path, "annot": annot})
                    rows_for_grid.append({
                        "annot_lines": label_lines,
                        "cells": cells,
                    })
                overview_png = _compose_cf_overview(
                    case_dir / f"cf_overview_{mode}.png",
                    case_dir / "target_raw.jpg",
                    rows_for_grid,
                    col_headers,
                    case_header=f"{scenario} / {run_id} / anchor={anchor} -- mode={mode}",
                )

            mode_payloads[mode] = {
                "overview_png": str(overview_png) if overview_png else "",
                "extra_overview_pngs": extra_overview,
                "noise_floor": baseline_floor,
                "noise_floor_by_cfg": {str(k): v for k, v in floor_by_cfg.items()},
                "per_cf": per_cf_summaries,
                "skipped": skipped,
                "variants_raw": [
                    {
                        "tag": rec["tag"],
                        "is_truth": rec["is_truth"],
                        "subgoal": rec["subgoal"],
                        "requested_subgoal": rec["requested_subgoal"],
                        "request_source": rec["request_source"],
                        "scenario": rec["scenario"],
                        "status": rec["status"],
                        "warning": rec["warning"],
                        "prompt_consistency": rec["prompt_consistency"],
                        "runs": [
                            # 非 verbose 模式下 .cf_tmp/ 已删，pred_path 失效；
                            # truth baseline 仍指向 case_dir/pred.png；其它清空避免误导。
                            {
                                k: v for k, v in run.items()
                                if k not in {"z1_pred", "rgb_pred"}
                                and not (
                                    k == "pred_path"
                                    and not args.cf_verbose_artifacts
                                    and ".cf_tmp" in str(v)
                                )
                            }
                            for run in rec["runs"]
                        ],
                    }
                    for rec in variant_records.values()
                ],
            }

        # 清理 tmp pred 目录（非 verbose 模式才会有）
        if not args.cf_verbose_artifacts:
            tmp_root = case_dir / ".cf_tmp"
            if tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)
                # 但 cf_overview 已经合成完，tmp pred 不再需要

        if truth_metrics is None or truth_trace is None:
            raise RuntimeError("内部错误：truth baseline 缺失")
        metrics = truth_metrics
        trace = truth_trace
        elapsed = truth_elapsed

        # ---- 5) 公共 dump ----
        _save_rgb_png(rgb_gt[0], case_dir / "target_vae_recon.png")
        with (case_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with (case_dir / "euler_trace.json").open("w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        with (case_dir / "memory.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(memory), f, ensure_ascii=False, indent=2)

        # ---- 6) CF 产物：cf_summary.json + cf_report.md ----
        cf_summary_path = ""
        cf_report_path = ""
        primary_overview = ""
        if dump_cf:
            cf_summary = {
                "experiment": {
                    "name": "goalgen_counterfactual_subgoal",
                    "truth_subgoal": memory.subgoal,
                    "scenario": memory.scenario,
                    "modes": active_modes,
                    "cfg_scale_sweep": cfg_scale_values,
                    "seed_replicates": seed_replicates,
                    "request_source": cf_request_source,
                    "verbose_artifacts": args.cf_verbose_artifacts,
                },
                "modes": mode_payloads,
                "case_max_ratio_over_floor_pixel": case_max_ratio,
                "case_max_ratio_tag": case_max_ratio_tag,
                "case_max_verdict": case_max_verdict,
                "verdict_legend": {
                    "near_floor":        "ratio < 2 — basically sampling noise",
                    "weak_response":     "2 ≤ ratio < 5 — model reacts a bit",
                    "responsive":        "5 ≤ ratio < 15 — clearly follows SUBGOAL",
                    "highly_responsive": "ratio ≥ 15 — strong SUBGOAL conditioning",
                    "insufficient_seeds": "seed_replicates < 2; floor not computed",
                },
            }
            cf_summary_path = str(case_dir / "cf_summary.json")
            with open(cf_summary_path, "w", encoding="utf-8") as f:
                json.dump(cf_summary, f, ensure_ascii=False, indent=2)
            cf_report_path = str(case_dir / "cf_report.md")
            with open(cf_report_path, "w", encoding="utf-8") as f:
                f.write(_render_cf_report_md(
                    case_header=f"{scenario} / {run_id} / anchor={anchor}",
                    mode_summaries=mode_payloads,
                    case_meta={
                        "truth_subgoal":   memory.subgoal,
                        "active_modes":    active_modes,
                        "cfg_scale_sweep": cfg_scale_values,
                        "seed_replicates": seed_replicates,
                        "request_source":  cf_request_source,
                    },
                ))
            # primary overview 用第一个 mode 的 png
            first_mode = active_modes[0]
            primary_overview = mode_payloads[first_mode].get("overview_png", "")

        # ---- 7) meta.json + overview.md ----
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
            "cf_overview": primary_overview,
            "cf_overview_extra": [
                mode_payloads[m]["overview_png"] for m in active_modes[1:]
                if dump_cf and mode_payloads[m].get("overview_png")
            ],
            "cf_summary": cf_summary_path,
            "cf_report":  cf_report_path,
            "cf_active_modes": active_modes if dump_cf else [],
            "cf_cfg_sweep": cfg_scale_values if dump_cf else [],
            "cf_seed_replicates": seed_replicates if dump_cf else 0,
            "cf_max_ratio": case_max_ratio,
            "cf_max_tag":   case_max_ratio_tag,
            "cf_max_verdict": case_max_verdict if dump_cf else "",
            "elapsed_sec": elapsed,
        }
        with (case_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        overview = render_overview_md(case_dir, sample, memory, metrics, trace, meta)
        (case_dir / "overview.md").write_text(overview, encoding="utf-8")

        # ---- 8) case_root 级 index ----
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
            "cf_overview": primary_overview,
            "cf_summary": cf_summary_path,
            "elapsed_sec": elapsed,
        })
        with (case_root / f"_index{args.case_suffix}.jsonl").open("w", encoding="utf-8") as f:
            for r in index_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if dump_cf:
            cf_index_records.append({
                "case_dir": str(case_dir),
                "scenario": scenario,
                "run_id": run_id,
                "anchor": anchor,
                "truth_subgoal": memory.subgoal,
                "modes": active_modes,
                "max_ratio_over_floor_pixel": case_max_ratio,
                "max_ratio_tag": case_max_ratio_tag,
                "max_verdict": case_max_verdict,
                "cf_overview": primary_overview,
                "cf_summary": cf_summary_path,
                "cf_report":   cf_report_path,
            })
            with (case_root / f"_cf_index{args.case_suffix}.jsonl").open("w", encoding="utf-8") as f:
                for r in cf_index_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[probe] done {scenario}/{run_id}/anchor={anchor} → {case_dir}  "
              f"v_cos={metrics['velocity_cos']:.3f} psnr={metrics['psnr']:.2f} "
              f"cf_max_verdict={case_max_verdict if dump_cf else '-'}")

    print(f"\n[probe] all {len(index_records)} cases dumped under {case_root}")
    if cf_index_records:
        print(f"[probe] CF verdict roll-up: _cf_index{args.case_suffix}.jsonl ({len(cf_index_records)} cases)")


if __name__ == "__main__":
    main()
