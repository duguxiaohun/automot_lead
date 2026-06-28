"""SFT v4 的 prompt、Memory 状态机与输出解析工具。

v4 把 v2 / v3 的"42 选 1"场景题改成**三层级联选择题**：

- **layer-1 ROAD_STRUCTURE**：6 桶视觉道路结构（JUNCTION / HIGHWAY_MERGE /
  ROADSIDE_HAZARD / PARKING_AREA / VRU_CROSSING / OPEN_ROAD_DYNAMICS）；
- **layer-2 SCENE**：当前 layer-1 桶下的具体 scenario（3~15 个候选）；
- **layer-3 STATUS / SUBGOAL**：当前 scene 事件序列中的离散状态/子目标。

学生每帧内循环 3 步，分别在三层各选一个；高层决策错时直接跳过低层
（`should_trigger_step2` / `should_trigger_step3`），避免训练信号与候选表错配。
设计细节见 [SFT_V4_PLAN.md §12](SFT_V4_PLAN.md)（D21~D27 决策）。

本文件只负责"文本协议"和"状态转移"，不读图片、不加载模型；train / eval / probe /
collect / learn / inspect_teacher 都从这里 import，禁止有第二份隐式实现。
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import re
from typing import Dict, List, Optional, Tuple

from qwen3vl_local.prompt_pipeline import (
    EVENT_DESCRIPTIONS,
    SCENARIO_LABELS,
    get_full_sequence as _raw_get_full_sequence,
)


DATASET_VERSION = "sft_v4_sequence"

# Teacher generation guard:
# - max only prevents runaway generation; it is not a word/sentence limit.
# - teacher steps use fresh dialogs and full context, so analysis is not capped to a tiny answer.
TEACHER_MAX_NEW_TOKENS_STEP1 = 384
TEACHER_MAX_NEW_TOKENS_STEP2 = 384
TEACHER_MAX_NEW_TOKENS_STEP3 = 384

# 7 项 loss 默认权重（§12.5）：分析段共 0.2 × 3 = 0.6，离散标签共 1.0 × 4 = 4.0。
DEFAULT_W_ANALYSIS = 0.2
DEFAULT_W_ROAD_STRUCTURE = 1.0  # L_RS1, D25 拍板 = 1.0
DEFAULT_W_SCENE = 1.0
DEFAULT_W_STATUS = 1.0
DEFAULT_W_SUBGOAL = 1.0

# D27 拍板：默认初始正确率从 0.5 提到 0.7，缓解 step3 触发率被 layer-1 拖累。
DEFAULT_P_INIT_CORRECT = 0.7
DEFAULT_SKIP_CORRECTION_SCENE_NOISE_PROB = 0.15

# ---------------------------------------------------------------------------
# 统一 system prompt：只定角色 + 通用证据原则。
# 任务、关注点、输出格式、词数目标全部下沉 user prompt，system 不再重复。
# 教师与学生共用同一段；label 写不写由 user prompt 控制。
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V4 = """\
You are an autonomous driving agent. Use the 4 stitched RGB frames as visual context: oldest to newest, left/front/right views.

Keep the believed label by default; change or advance it only when clear visible evidence supports it. If the evidence is weak, distant, foggy, or occluded, describe it as "not contradicted" rather than "confirmed". Never mention answers, ground truth, or reference labels."""

# ---------------------------------------------------------------------------
# Step system prompt getter + 向后兼容别名
# ---------------------------------------------------------------------------

# 所有 step 共用同一段 system prompt（职责已拆到 user prompt）。
_SYSTEM_PROMPT = SYSTEM_PROMPT_V4


def get_step_system_prompt(step_tag: str) -> str:
    """Return the system prompt for a given step (unified for all steps)."""

    return _SYSTEM_PROMPT


# 向后兼容：旧代码直接 import SYSTEM_PROMPT_STEP1/2/3 仍可用。
SYSTEM_PROMPT_STEP1 = SYSTEM_PROMPT_V4
SYSTEM_PROMPT_STEP2 = SYSTEM_PROMPT_V4
SYSTEM_PROMPT_STEP3 = SYSTEM_PROMPT_V4


# ---------------------------------------------------------------------------
# Layer-1 道路结构定义（D21 拍板：6 桶）
# ---------------------------------------------------------------------------

ROAD_STRUCTURE_LABELS: Dict[str, str] = {
    "JUNCTION": "Intersection, crossroad, turn, or traffic-light junction",
    "HIGHWAY_MERGE": "High-speed or multi-lane road with merge, exit, or cut-in flow",
    "ROADSIDE_HAZARD": "Normal road with blocked lane, accident, construction, or parked obstacle",
    "PARKING_AREA": "Parking lot, parking exit, door opening, or low-speed parking interaction",
    "VRU_CROSSING": "Pedestrian, cyclist, or small moving actor crossing ego path",
    "OPEN_ROAD_DYNAMICS": "Open road with lead vehicle, braking, control loss, or invading turn",
}

# Scene alias 规则（D28）：部分 CARLA scenario 的 V2 后缀只表示 benchmark
# 版本差异，视觉语义与事件序列完全相同。v4 的学生任务只监督可见/可推理的
# canonical scene，不要求区分这种不可视觉验证的版本名。
SCENE_CANONICAL_ALIASES: Dict[str, str] = {
    "EnterActorFlowV2": "EnterActorFlow",
    "MergerIntoSlowTrafficV2": "MergerIntoSlowTraffic",
}


def canonicalize_scene(scene: Optional[str]) -> Optional[str]:
    """Return the v4 canonical scene label, preserving unknown/None inputs."""

    if not scene:
        return scene
    return SCENE_CANONICAL_ALIASES.get(str(scene), str(scene))


def get_full_sequence(scene: str) -> Tuple[str, ...]:
    """Return the event sequence for the canonical scene label."""

    canonical = canonicalize_scene(scene)
    if canonical is None:
        raise KeyError("scene is None")
    return tuple(_raw_get_full_sequence(canonical))


# 42 个 raw scene 到 layer-1 桶的映射（D21 桶分配）。其中 V2 alias 会在
# layer-2 候选表和监督 label 中折叠到 canonical scene。
# 桶规模：JUNCTION=15, HIGHWAY_MERGE=9, ROADSIDE_HAZARD=8, PARKING_AREA=4,
#         VRU_CROSSING=3, OPEN_ROAD_DYNAMICS=3 → 总 42。
SCENE_TO_ROAD_STRUCTURE: Dict[str, str] = {
    # JUNCTION (15) ----------------------------------------------------------
    "BlockedIntersection":                    "JUNCTION",
    "CrossJunctionDefectTrafficLight":        "JUNCTION",
    "NonSignalizedJunctionLeftTurn":          "JUNCTION",
    "NonSignalizedJunctionLeftTurnEnterFlow": "JUNCTION",
    "NonSignalizedJunctionRightTurn":         "JUNCTION",
    "OppositeVehicleRunningRedLight":         "JUNCTION",
    "OppositeVehicleTakingPriority":          "JUNCTION",
    "PriorityAtJunction":                     "JUNCTION",
    "RedLightWithoutLeadVehicle":             "JUNCTION",
    "SignalizedJunctionLeftTurn":             "JUNCTION",
    "SignalizedJunctionLeftTurnEnterFlow":    "JUNCTION",
    "SignalizedJunctionRightTurn":            "JUNCTION",
    "T_Junction":                             "JUNCTION",
    "VehicleTurningRoute":                    "JUNCTION",
    "VehicleTurningRoutePedestrian":          "JUNCTION",
    # HIGHWAY_MERGE (9) ------------------------------------------------------
    "EnterActorFlow":                         "HIGHWAY_MERGE",
    "EnterActorFlowV2":                       "HIGHWAY_MERGE",
    "HighwayCutIn":                           "HIGHWAY_MERGE",
    "HighwayExit":                            "HIGHWAY_MERGE",
    "InterurbanActorFlow":                    "HIGHWAY_MERGE",
    "InterurbanAdvancedActorFlow":            "HIGHWAY_MERGE",
    "MergerIntoSlowTraffic":                  "HIGHWAY_MERGE",
    "MergerIntoSlowTrafficV2":                "HIGHWAY_MERGE",
    "StaticCutIn":                            "HIGHWAY_MERGE",
    # ROADSIDE_HAZARD (8) ----------------------------------------------------
    "Accident":                               "ROADSIDE_HAZARD",
    "AccidentTwoWays":                        "ROADSIDE_HAZARD",
    "ConstructionObstacle":                   "ROADSIDE_HAZARD",
    "ConstructionObstacleTwoWays":            "ROADSIDE_HAZARD",
    "HazardAtSideLane":                       "ROADSIDE_HAZARD",
    "HazardAtSideLaneTwoWays":                "ROADSIDE_HAZARD",
    "ParkedObstacle":                         "ROADSIDE_HAZARD",
    "ParkedObstacleTwoWays":                  "ROADSIDE_HAZARD",
    # PARKING_AREA (4) -------------------------------------------------------
    "ParkingCrossingPedestrian":              "PARKING_AREA",
    "ParkingCutIn":                           "PARKING_AREA",
    "ParkingExit":                            "PARKING_AREA",
    "VehicleOpensDoorTwoWays":                "PARKING_AREA",
    # VRU_CROSSING (3) -------------------------------------------------------
    "CrossingBicycleFlow":                    "VRU_CROSSING",
    "DynamicObjectCrossing":                  "VRU_CROSSING",
    "PedestrianCrossing":                     "VRU_CROSSING",
    # OPEN_ROAD_DYNAMICS (3) -------------------------------------------------
    "ControlLoss":                            "OPEN_ROAD_DYNAMICS",
    "HardBreakRoute":                         "OPEN_ROAD_DYNAMICS",
    "InvadingTurn":                           "OPEN_ROAD_DYNAMICS",
}

# 完整性校验：每个 SCENARIO_LABELS 里的 scene 必须有 road structure 映射。
# 这是 import-time guard——如果 prompt_pipeline.py 加了新 scene 而忘了在这里
# 分桶，import sft_v4.prompts 就会立刻报错而不是在训练中段崩。
_missing_scenes = sorted(set(SCENARIO_LABELS) - set(SCENE_TO_ROAD_STRUCTURE))
if _missing_scenes:
    raise RuntimeError(
        f"SCENE_TO_ROAD_STRUCTURE missing entries for {_missing_scenes!r}; "
        "update prompts.py SCENE_TO_ROAD_STRUCTURE when adding scenarios to SCENARIO_LABELS."
    )

for _alias, _canonical in SCENE_CANONICAL_ALIASES.items():
    if _alias not in SCENARIO_LABELS:
        raise RuntimeError(f"SCENE_CANONICAL_ALIASES alias {_alias!r} is not in SCENARIO_LABELS")
    if _canonical not in SCENARIO_LABELS:
        raise RuntimeError(f"SCENE_CANONICAL_ALIASES target {_canonical!r} is not in SCENARIO_LABELS")
    if SCENE_TO_ROAD_STRUCTURE.get(_alias) != SCENE_TO_ROAD_STRUCTURE.get(_canonical):
        raise RuntimeError(f"scene alias {_alias!r}->{_canonical!r} crosses road-structure buckets")
    if tuple(_raw_get_full_sequence(_alias)) != tuple(_raw_get_full_sequence(_canonical)):
        raise RuntimeError(f"scene alias {_alias!r}->{_canonical!r} has a different event sequence")

CANONICAL_SCENARIO_LABELS: Dict[str, str] = {
    name: desc
    for name, desc in SCENARIO_LABELS.items()
    if canonicalize_scene(name) == name
}

# 反向索引：每个桶下的 layer-2 canonical 候选列表（排序，便于 SCENE_CHOICES 渲染稳定）。
ROAD_STRUCTURE_TO_SCENES: Dict[str, List[str]] = {}
for _scene, _rs in SCENE_TO_ROAD_STRUCTURE.items():
    _canonical_scene = canonicalize_scene(_scene)
    if _canonical_scene != _scene:
        continue
    ROAD_STRUCTURE_TO_SCENES.setdefault(_rs, []).append(_scene)
for _rs in ROAD_STRUCTURE_TO_SCENES:
    ROAD_STRUCTURE_TO_SCENES[_rs].sort()


def get_road_structure(scene: str) -> str:
    """返回某 scene 对应的 layer-1 道路结构 token，缺失时直接 raise。"""

    canonical = canonicalize_scene(scene)
    if canonical not in SCENE_TO_ROAD_STRUCTURE:
        raise KeyError(
            f"scene {scene!r} has no road-structure mapping; "
            "update SCENE_TO_ROAD_STRUCTURE in prompts.py."
        )
    return SCENE_TO_ROAD_STRUCTURE[canonical]


def first_scene_in_bucket(road_structure: str) -> str:
    """返回某 layer-1 桶的 canonical 第一个 scene。

    用途：学生 step1 翻转 layer-1 时，scene 必须 reset 到新桶第一项；与
    scene flip → status/subgoal reset 同构。
    """

    scenes = ROAD_STRUCTURE_TO_SCENES.get(road_structure, [])
    if not scenes:
        raise KeyError(f"road_structure {road_structure!r} has no scenes in bucket")
    return scenes[0]


# ---------------------------------------------------------------------------
# 事件序列辅助（旧 API 保留）
# ---------------------------------------------------------------------------


def initial_event(scene: str) -> str:
    """返回某场景 canonical init 状态 token，从 get_full_sequence 取以保持同步。"""

    seq = get_full_sequence(scene)
    return seq[0]


def first_subgoal(scene: str) -> str:
    """返回某场景 init 之后的第一个子目标 token；序列退化时回退到 seq[0]。"""

    seq = get_full_sequence(scene)
    return seq[1] if len(seq) > 1 else seq[0]


def _label_with_optional_desc(name: str, descriptions: Dict[str, str], covered_names: set[str]) -> str:
    """Render ``name`` alone when a nearby choices block already explains it."""

    if name in covered_names:
        return name
    desc = descriptions.get(name)
    return f"{name} ({desc})" if desc else name


# ---------------------------------------------------------------------------
# Memory dataclass（三层文本状态机）
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Memory:
    """学生跨帧维护的纯文本三层 memory。

    字段顺序：road_structure (layer-1) → scene (layer-2) → status/subgoal
    (layer-3) → ego_to_goal_xy。**注意 dataclass 字段顺序与 __init__ 关键字一致**；
    若你新增字段请记得放到末尾并加默认值，避免破坏既有 ``Memory(...)`` 构造调用。
    """

    scene: str
    status: str
    subgoal: str
    ego_to_goal_x: float
    ego_to_goal_y: float
    road_structure: str = "JUNCTION"  # 默认值仅用于错误恢复，正常路径必须显式传

    def __post_init__(self) -> None:
        """Normalize v4 scene aliases whenever memory enters the state machine."""

        canonical = canonicalize_scene(self.scene)
        if canonical in SCENE_TO_ROAD_STRUCTURE:
            self.scene = str(canonical)

    def copy(self) -> "Memory":
        """深拷贝 memory（dataclass 全是不可变标量，等价于浅拷贝即可）。"""

        return Memory(
            scene=self.scene,
            status=self.status,
            subgoal=self.subgoal,
            ego_to_goal_x=self.ego_to_goal_x,
            ego_to_goal_y=self.ego_to_goal_y,
            road_structure=self.road_structure,
        )

    def format_text(self) -> str:
        """按固定 prompt 协议渲染 ``[MEMORY]`` 文本块。

        三层均带自然语言描述，让模型在不知道 memory 是否正确的情况下也能理解
        "我现在相信的道路结构 / 场景 / 状态 / 子目标"对应什么交通语义。
        """

        rs_desc = ROAD_STRUCTURE_LABELS.get(self.road_structure, self.road_structure)
        scene_desc = CANONICAL_SCENARIO_LABELS.get(self.scene, self.scene)
        status_desc = EVENT_DESCRIPTIONS.get(self.status, self.status)
        subgoal_desc = EVENT_DESCRIPTIONS.get(self.subgoal, self.subgoal)
        return (
            "[MEMORY]\n"
            f"BELIEVED_ROAD_STRUCTURE={self.road_structure} ({rs_desc})\n"
            f"BELIEVED_SCENE={self.scene} ({scene_desc})\n"
            f"BELIEVED_STATUS={self.status} ({status_desc})\n"
            f"BELIEVED_SUBGOAL={self.subgoal} ({subgoal_desc})\n"
            f"EGO_TO_GOAL_XY=({self.ego_to_goal_x:+.1f}, {self.ego_to_goal_y:+.1f}) m\n"
            "[/MEMORY]"
        )

    def format_step1_student_text(self) -> str:
        """Render the student-visible road-only memory for step1.

        Step1 should decide only the layer-1 road structure. Keeping scene/status/subgoal
        out of this prompt prevents lower-layer guesses from acting as accidental hints
        before step2/step3 have earned their trigger gates.
        """

        rs_desc = ROAD_STRUCTURE_LABELS.get(self.road_structure, self.road_structure)
        return (
            "[STEP1_ROAD_MEMORY]\n"
            f"BELIEVED_ROAD_STRUCTURE={self.road_structure} ({rs_desc})\n"
            f"EGO_TO_GOAL_XY=({self.ego_to_goal_x:+.1f}, {self.ego_to_goal_y:+.1f}) m\n"
            "[/STEP1_ROAD_MEMORY]"
        )

    def format_step1_road_text(self, gt_road_structure: str) -> str:
        """Render the road-only context used by the step1 teacher prompt."""

        return (
            "[STEP1_ROAD_CONTEXT]\n"
            f"BELIEVED_ROAD_STRUCTURE={self.road_structure}\n"
            f"EGO_TO_GOAL_XY=({self.ego_to_goal_x:+.1f}, {self.ego_to_goal_y:+.1f}) m\n"
            f"ANSWER_ROAD_STRUCTURE={gt_road_structure}\n"
            "[/STEP1_ROAD_CONTEXT]"
        )

    def format_step2_scene_text(self, gt_road_structure: str, gt_scene: str) -> str:
        """Render the independent scene context used by the step2 teacher prompt."""

        gt_scene = str(canonicalize_scene(gt_scene))
        believed_scene = str(canonicalize_scene(self.scene))
        gt_rs_desc = ROAD_STRUCTURE_LABELS.get(gt_road_structure, gt_road_structure)
        scene_choices = set(ROAD_STRUCTURE_TO_SCENES.get(gt_road_structure, []))
        return (
            "[STEP2_SCENE_CONTEXT]\n"
            f"ANSWER_ROAD_STRUCTURE={gt_road_structure} ({gt_rs_desc})\n"
            f"BELIEVED_SCENE={_label_with_optional_desc(believed_scene, CANONICAL_SCENARIO_LABELS, scene_choices)}\n"
            f"ANSWER_SCENE={_label_with_optional_desc(gt_scene, CANONICAL_SCENARIO_LABELS, scene_choices)}\n"
            f"EGO_TO_GOAL_XY=({self.ego_to_goal_x:+.1f}, {self.ego_to_goal_y:+.1f}) m\n"
            "[/STEP2_SCENE_CONTEXT]"
        )

    def format_step3_event_text(
        self,
        gt_road_structure: str,
        gt_scene: str,
        gt_status: str,
        gt_subgoal: str,
    ) -> str:
        """Render the independent status/subgoal context used by the step3 teacher prompt."""

        gt_scene = str(canonicalize_scene(gt_scene))
        gt_rs_desc = ROAD_STRUCTURE_LABELS.get(gt_road_structure, gt_road_structure)
        gt_scene_desc = CANONICAL_SCENARIO_LABELS.get(gt_scene, gt_scene)
        event_choices = set(get_full_sequence(gt_scene))
        return (
            "[STEP3_EVENT_CONTEXT]\n"
            f"ANSWER_ROAD_STRUCTURE={gt_road_structure} ({gt_rs_desc})\n"
            f"ANSWER_SCENE={gt_scene} ({gt_scene_desc})\n"
            f"BELIEVED_STATUS={_label_with_optional_desc(self.status, EVENT_DESCRIPTIONS, event_choices)}\n"
            f"BELIEVED_SUBGOAL={_label_with_optional_desc(self.subgoal, EVENT_DESCRIPTIONS, event_choices)}\n"
            f"ANSWER_STATUS={_label_with_optional_desc(gt_status, EVENT_DESCRIPTIONS, event_choices)}\n"
            f"ANSWER_SUBGOAL={_label_with_optional_desc(gt_subgoal, EVENT_DESCRIPTIONS, event_choices)}\n"
            f"EGO_TO_GOAL_XY=({self.ego_to_goal_x:+.1f}, {self.ego_to_goal_y:+.1f}) m\n"
            "[/STEP3_EVENT_CONTEXT]"
        )


# ---------------------------------------------------------------------------
# Memory 初始化 / Phase B 弱纠偏 / 噪声扰动
# ---------------------------------------------------------------------------


def init_memory(
    *,
    run_id: str,
    sub_scenario_id: str,
    ego_to_goal_x: float,
    ego_to_goal_y: float,
    gt_scene: str | None = None,
    p_init_correct: float = DEFAULT_P_INIT_CORRECT,
    seed_salt: str = "",
) -> Memory:
    """三层联合初始化（D22 + D27）。

    采样路径：
    - 概率 p：``road_structure = gt 桶``；scene 在该桶内按 50% = gt（其余从同桶
      其它项均匀挑）；
    - 概率 1-p：``road_structure`` 从其它 5 桶里均匀挑；scene 从该错桶 layer-2
      均匀挑。**memory 内部始终自洽**——不会出现 "BELIEVED road_structure=JUNCTION
      但 BELIEVED scene=Accident" 这种矛盾配置。

    ``p_init_correct`` 默认 0.7（D27）。``p_init_correct=0.0`` 退回 v3 必错口径
    （但 layer-1 一定错时 scene 仍在该错桶内随机），可作对照实验入口。
    """

    gt_scene = canonicalize_scene(gt_scene)
    p = min(1.0, max(0.0, float(p_init_correct)))
    # seed 只由数据身份 + collector/policy salt 决定，与全局 random 状态无关。
    # 同一配置重跑结果可复现；不同 collector / snapshot 版本能产生不同扰动。
    seed_src = f"{run_id}::{sub_scenario_id}::{seed_salt}".encode("utf-8")
    seed = int(hashlib.sha256(seed_src).hexdigest(), 16) % (2**31)
    rng = random.Random(seed)

    gt_rs: Optional[str] = None
    if gt_scene is not None and gt_scene in SCENE_TO_ROAD_STRUCTURE:
        gt_rs = SCENE_TO_ROAD_STRUCTURE[gt_scene]

    if gt_rs is not None and rng.random() < p:
        # layer-1 正确分支：scene 在 GT 桶内 50% = gt，其它 50% 同桶随机错。
        road_structure = gt_rs
        bucket_scenes = ROAD_STRUCTURE_TO_SCENES[road_structure]
        same_bucket_other = [s for s in bucket_scenes if s != gt_scene]
        if not same_bucket_other or rng.random() < 0.5:
            scene = gt_scene if gt_scene in bucket_scenes else rng.choice(bucket_scenes)
        else:
            scene = rng.choice(same_bucket_other)
    else:
        # layer-1 错误分支（或没传 gt_scene）：从其它桶均匀选，scene 从该错桶内挑。
        all_buckets = sorted(ROAD_STRUCTURE_TO_SCENES.keys())
        if gt_rs is not None and len(all_buckets) > 1:
            wrong_buckets = [b for b in all_buckets if b != gt_rs]
            road_structure = rng.choice(wrong_buckets)
        else:
            road_structure = rng.choice(all_buckets)
        bucket_scenes = ROAD_STRUCTURE_TO_SCENES[road_structure]
        scene = rng.choice(bucket_scenes)

    assert scene is not None
    return Memory(
        scene=scene,
        status=initial_event(scene),
        subgoal=first_subgoal(scene),
        ego_to_goal_x=ego_to_goal_x,
        ego_to_goal_y=ego_to_goal_y,
        road_structure=road_structure,
    )


def force_memory_to_gt_chain(
    memory: Memory,
    *,
    gt_road_structure: str,
    gt_scene: str,
) -> Memory:
    """Phase B 帧首三层弱纠偏（D23 + D27 配套）。

    顺序：先拉回 ``road_structure``，再拉回 ``scene``；status/subgoal 在没换 scene
    时保留（与 v3 D2 决议同步）。每层 = GT 时全 no-op，保持上一帧推进过的状态。

    机制目的：Phase B 期间把上层 memory 拉回稳定链路；若本帧确实发生
    road/scene reset，调用方会只训练被纠正的上层，下一帧再继续下钻，避免刚
    reset 的 status/subgoal 立刻进入 step3。
    """

    gt_scene = str(canonicalize_scene(gt_scene))
    mem = memory.copy()
    # Layer-1：道路结构层。错就先 reset 整条链，因为 scene 跨桶后必须刷新。
    if mem.road_structure != gt_road_structure:
        mem.road_structure = gt_road_structure
        mem.scene = first_scene_in_bucket(gt_road_structure)
        mem.status = initial_event(mem.scene)
        mem.subgoal = first_subgoal(mem.scene)
    # Layer-2：场景层。可能与 layer-1 reset 后的 first scene 不一样，仍需对齐。
    if mem.scene != gt_scene:
        mem.scene = gt_scene
        mem.status = initial_event(gt_scene)
        mem.subgoal = first_subgoal(gt_scene)
    return mem


def force_memory_to_gt_scene(memory: Memory, *, gt_scene: str) -> Memory:
    """兼容旧 API：从 gt_scene 推 gt_road_structure，然后委托给 force_memory_to_gt_chain。

    保留这个名字让 v3-style 调用方（如旧 collector code path）在迁移过程中
    不至于立刻报 ImportError；新代码应直接调 ``force_memory_to_gt_chain``。
    """

    gt_scene = str(canonicalize_scene(gt_scene))
    gt_rs = SCENE_TO_ROAD_STRUCTURE.get(gt_scene)
    if gt_rs is None:
        return memory.copy()
    return force_memory_to_gt_chain(memory, gt_road_structure=gt_rs, gt_scene=gt_scene)


def inject_phase_b_noise(
    memory: Memory,
    *,
    gt_scene: str,
    rng: random.Random,
    prob: float,
) -> Tuple[Memory, bool]:
    """D23 拍板：Phase B 噪声**仅扰 scene**，且限制在当前 road_structure 桶内。

    保持 memory 内部自洽（layer-1 / scene 始终同桶），同时让学生在 Phase B 后半段
    仍能见到"突然错记忆 → 重新纠正"样本。如果当前桶只有 1 个 scene、或全等于
    gt_scene，no-op 返回。
    """

    gt_scene = str(canonicalize_scene(gt_scene))
    p = min(1.0, max(0.0, float(prob)))
    if p <= 0.0 or rng.random() >= p:
        return memory.copy(), False
    bucket = ROAD_STRUCTURE_TO_SCENES.get(memory.road_structure, [])
    candidates = [s for s in bucket if s != gt_scene]
    if not candidates:
        return memory.copy(), False
    scene = rng.choice(candidates)
    mem = memory.copy()
    mem.scene = scene
    mem.status = initial_event(scene)
    mem.subgoal = first_subgoal(scene)
    return mem, True


def correct_memory_after_step1_skip(
    memory: Memory,
    *,
    gt_scene: str,
    rng: random.Random,
    scene_noise_prob: float = DEFAULT_SKIP_CORRECTION_SCENE_NOISE_PROB,
) -> Tuple[Memory, bool]:
    """Repair memory before the next inner loop after a step1 skip.

    Callers set the trigger only when the previous frame failed layer-1 and
    skipped step2/step3. The repair pulls ROAD_STRUCTURE back to the GT bucket,
    sets SCENE to GT with high probability, optionally perturbs SCENE to a
    non-GT candidate inside the same bucket, and resets STATUS/SUBGOAL to that
    chosen scene's init chain. EGO_TO_GOAL_XY is preserved because frame-end
    prefetch has already updated it for the current frame.
    """

    gt_scene = str(canonicalize_scene(gt_scene))
    gt_rs = get_road_structure(gt_scene)
    p = min(1.0, max(0.0, float(scene_noise_prob)))
    bucket = ROAD_STRUCTURE_TO_SCENES.get(gt_rs, [])
    candidates = [s for s in bucket if s != gt_scene]
    scene = gt_scene
    noisy = False
    if candidates and p > 0.0 and rng.random() < p:
        scene = rng.choice(candidates)
        noisy = True
    mem = memory.copy()
    mem.road_structure = gt_rs
    mem.scene = scene
    mem.status = initial_event(scene)
    mem.subgoal = first_subgoal(scene)
    return mem, noisy


# ---------------------------------------------------------------------------
# 三层触发链（D24）
# ---------------------------------------------------------------------------


def should_trigger_step2(
    *,
    memory_road_structure_before_step1: str,
    memory_road_structure_after_step1: str,
    gt_road_structure: str,
    road_structure_reset_this_frame: bool = False,
) -> bool:
    """step2 触发条件：layer-1 已稳定正确，而不是刚被纠正。

    只有 step1 前后 ``memory.road_structure`` 都等于 GT 桶，且本帧没有脚本层
    road reset，才进 step2。若 layer-1 在本帧刚被纠正，则本帧只监督 step1，
    下一帧再在稳定的正确 bucket 上训练 scene。
    """

    return (
        not road_structure_reset_this_frame
        and memory_road_structure_before_step1 == gt_road_structure
        and memory_road_structure_after_step1 == gt_road_structure
    )


def should_trigger_step3(
    *,
    memory_scene_before_step2: str,
    memory_scene_after_step2: str,
    gt_scene: str,
    scene_reset_this_frame: bool = False,
) -> bool:
    """step3 触发条件：layer-2 已稳定正确，而不是刚被纠正。

    只有 step2 前后 ``memory.scene`` 都等于 GT scene，且本帧没有脚本层 scene/event
    reset，才进 step3。若 scene 在本帧刚被纠正，status/subgoal 会随 scene 重置到
    init，本帧不继续训练 step3，避免把刚 reset 的下层状态立刻拿去对齐当前帧事件。
    """

    memory_scene_before_step2 = str(canonicalize_scene(memory_scene_before_step2))
    memory_scene_after_step2 = str(canonicalize_scene(memory_scene_after_step2))
    gt_scene = str(canonicalize_scene(gt_scene))
    return (
        not scene_reset_this_frame
        and memory_scene_before_step2 == gt_scene
        and memory_scene_after_step2 == gt_scene
    )


# ---------------------------------------------------------------------------
# Memory 更新规则（学生输出 → 下一帧 memory）
# ---------------------------------------------------------------------------


def update_memory_after_step1(
    memory: Memory,
    *,
    student_road_structure: Optional[str],
) -> Memory:
    """根据学生 step1 的 ``ROAD_STRUCTURE`` 输出更新 memory。

    - 非法输出（None / 不在 ROAD_STRUCTURE_LABELS 里）→ memory 保持不变；
    - 输出 = 当前 layer-1：no-op；
    - 输出 ≠ 当前 layer-1：layer-1 翻转，**强制 reset 整条链**（scene = 新桶的
      第一个 scene，status/subgoal 重置）。
    """

    mem = memory.copy()
    if not validate_road_structure(student_road_structure):
        return mem
    assert student_road_structure is not None
    if student_road_structure != mem.road_structure:
        mem.road_structure = student_road_structure
        mem.scene = first_scene_in_bucket(student_road_structure)
        mem.status = initial_event(mem.scene)
        mem.subgoal = first_subgoal(mem.scene)
    return mem


def update_memory_after_step2(
    memory: Memory,
    *,
    student_scene: Optional[str],
) -> Memory:
    """根据学生 step2 的 ``SCENE`` 输出更新 memory.scene。

    - 非法 scene（不在 SCENARIO_LABELS 里）→ no-op；
    - 跨桶 scene（与当前 memory.road_structure 不同桶）→ no-op，因为正常路径下
      学生看到的 SCENE_CHOICES 只来自当前桶，跨桶输出视为格式错误；
    - 输出 = 当前 scene：no-op；
    - 同桶不同 scene：scene 翻转，status/subgoal 重置。
    """

    mem = memory.copy()
    student_scene = canonicalize_scene(student_scene)
    if not validate_scene(student_scene):
        return mem
    assert student_scene is not None
    if SCENE_TO_ROAD_STRUCTURE.get(student_scene) != mem.road_structure:
        # 跨桶 scene 不允许：学生只该看到当前桶内的候选，跨桶输出 = 格式错误。
        return mem
    if student_scene != mem.scene:
        mem.scene = student_scene
        mem.status = initial_event(student_scene)
        mem.subgoal = first_subgoal(student_scene)
    return mem


def update_memory_after_step3(
    memory: Memory,
    *,
    student_status: Optional[str],
    student_subgoal: Optional[str],
) -> Memory:
    """根据学生 step3 输出更新 status/subgoal。

    只有当前 memory.scene 事件序列中的合法 event 才会生效。非法输出不写入 memory，
    避免一次格式错误污染后续整段 episode。
    """

    mem = memory.copy()
    if validate_event(mem.scene, student_status):
        assert student_status is not None
        mem.status = student_status
    if validate_event(mem.scene, student_subgoal):
        assert student_subgoal is not None
        mem.subgoal = student_subgoal
    return mem


# ---------------------------------------------------------------------------
# 选项块渲染
# ---------------------------------------------------------------------------


def road_structure_choices_block() -> str:
    """渲染 6 项 ROAD_STRUCTURE 选择表，供 step1 学生/老师 prompt 使用。

    顺序按字母排序，让 prompt 跨调用稳定（与 SCENE_CHOICES 同款约定）。
    """

    lines = ["[ROAD_STRUCTURE_CHOICES]"]
    for name in sorted(ROAD_STRUCTURE_LABELS):
        lines.append(f"- {name}: {ROAD_STRUCTURE_LABELS[name]}")
    lines.append("[/ROAD_STRUCTURE_CHOICES]")
    return "\n".join(lines)


def scene_choices_block_for(road_structure: str, *, heading: str = "BELIEVED_ROAD_STRUCTURE") -> str:
    """渲染指定 layer-1 桶下的 SCENE_CHOICES（v4 step2 专用）。

    SCENE_CHOICES 头部明确标注当前 layer-1 桶，帮助模型理解候选已经被收窄。
    未知 road_structure 时退回全量 SCENARIO_LABELS（防御性，正常路径不会触发）。
    """

    scenes = ROAD_STRUCTURE_TO_SCENES.get(road_structure, sorted(CANONICAL_SCENARIO_LABELS))
    lines = [f"[SCENE_CHOICES] under {heading} = {road_structure}"]
    for name in scenes:
        lines.append(f"- {name}: {CANONICAL_SCENARIO_LABELS[name]}")
    lines.append("[/SCENE_CHOICES]")
    return "\n".join(lines)


def scenario_choices_block() -> str:
    """渲染全量 canonical SCENE_CHOICES（保留作 v2/v3 兼容 / 调试用，v4 不调用）。"""

    lines = ["[SCENE_CHOICES]"]
    for name in sorted(CANONICAL_SCENARIO_LABELS):
        lines.append(f"- {name}: {CANONICAL_SCENARIO_LABELS[name]}")
    lines.append("[/SCENE_CHOICES]")
    return "\n".join(lines)


def _event_sequence_block(scene: str) -> str:
    """渲染某个 scene 对应的事件序列和事件描述（step3 专用）。"""

    seq = get_full_sequence(scene)
    lines = [f"EVENT_SEQUENCE: {' -> '.join(seq)}"]
    for event in seq:
        lines.append(f"- {event}: {EVENT_DESCRIPTIONS.get(event, event)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt protocol layer (system contract per step)
# ---------------------------------------------------------------------------
#
# 设计要点：
# - 三个 step 各有独立 system prompt（SYSTEM_PROMPT_STEP1/2/3），每个只暴露
#   自己层次的关注点、任务描述、证据规则和 4 行分析格式，不交叉注入
#   ROAD_STRUCTURE / SCENE / STATUS-SUBGOAL 噪音。
# - user message 只保留 step 级的 Task / Constraint / Then write 或 teacher
#   的 VERDICT / GT_HINT / 不写 label。
# - Memory Judgment 行的 opener 与 verdict 必须严格一致：KEEP→"Kept because"，
#   CHANGE→"Corrected because"，ADVANCE→"Advanced because"。教师 raw 输出会
#   经 `_enforce_memory_judgment_opener` 再次校正，避免训练时出现 "Kept because
#   …" 但 label 已变成另一个 scene 的 token-level 矛盾。

_VERDICT_OPENERS: Dict[str, str] = {
    "KEEP": "Kept because",
    "CHANGE": "Corrected because",
    "ADVANCE": "Advanced because",
}


def _memory_judgment_opener_for(verdict: str) -> str:
    """Return the canonical Memory Judgment opener for a teacher verdict.

    未知 verdict 默认按 KEEP 处理，保持向后兼容；正常路径 caller 会传
    "KEEP" / "CHANGE" / "ADVANCE" 三者之一。
    """

    key = (verdict or "").strip().upper()
    return _VERDICT_OPENERS.get(key, _VERDICT_OPENERS["KEEP"])


def step1_teacher_verdict(memory: Memory, gt_road_structure: str) -> str:
    """Return the scripted teacher verdict for ROAD_STRUCTURE supervision."""

    return "KEEP" if memory.road_structure == gt_road_structure else "CHANGE"


def step2_teacher_verdict(memory: Memory, gt_scene: str) -> str:
    """Return the scripted teacher verdict for SCENE supervision."""

    gt_scene = str(canonicalize_scene(gt_scene))
    return "KEEP" if memory.scene == gt_scene else "CHANGE"


def step3_teacher_verdict(memory: Memory, gt_scene: str, gt_status: str, gt_subgoal: str) -> str:
    """Return KEEP / ADVANCE / CHANGE for STATUS+SUBGOAL supervision.

    ``ADVANCE`` is reserved for normal forward progress inside the scene event
    sequence. ``CHANGE`` is used for off-sequence or backward corrections.
    """

    if memory.status == gt_status and memory.subgoal == gt_subgoal:
        return "KEEP"
    seq = list(get_full_sequence(str(canonicalize_scene(gt_scene))))
    try:
        mem_status_i = seq.index(memory.status)
        mem_subgoal_i = seq.index(memory.subgoal)
        gt_status_i = seq.index(gt_status)
        gt_subgoal_i = seq.index(gt_subgoal)
    except ValueError:
        return "CHANGE"
    moves = (gt_status_i - mem_status_i, gt_subgoal_i - mem_subgoal_i)
    if all(delta >= 0 for delta in moves) and any(delta > 0 for delta in moves):
        return "ADVANCE"
    return "CHANGE"


_MEMORY_JUDGMENT_LINE_RE = re.compile(
    r"^(?P<prefix>\s*Memory\s+Judgment\s*:\s*)(?P<body>.*)$",
    re.IGNORECASE,
)
_OPENER_PREFIX_RE = re.compile(
    r"^(?:Kept|Corrected|Advanced)\s+because\b[\s,.:;\-]*",
    re.IGNORECASE,
)


def _enforce_memory_judgment_opener(text: str, opener: str) -> str:
    """Rewrite the Memory Judgment line so it begins with the expected opener.

    教师 raw 文本偶尔会写 "Kept because …" 而后面追加的 label 实际改变了
    （CHANGE verdict）。此 sanitizer 只调整 opener 词组，保留教师写的理由，
    保证最终 supervised target 内 "opener 词 + label" token-level 自洽。

    - 找到第一行 "Memory Judgment:" → 把开头三选一替换为期望 opener。
    - 找不到则在末尾补一行最小 fallback，避免 4 行结构缺失。
    """

    if not text:
        return f"Memory Judgment: {opener} the believed memory remains consistent with the visible evidence."
    lines = text.splitlines()
    for idx, raw in enumerate(lines):
        m = _MEMORY_JUDGMENT_LINE_RE.match(raw)
        if not m:
            continue
        body = _OPENER_PREFIX_RE.sub("", m.group("body").strip()).strip()
        if body:
            # 避免 "Kept because The vehicle …" 这种重复大写；把首字母下放为小写。
            body = body[0].lower() + body[1:] if body[0].isalpha() else body
            rewritten = f"{m.group('prefix')}{opener} {body}"
        else:
            rewritten = f"{m.group('prefix')}{opener} the believed memory remains consistent with the visible evidence."
        if not rewritten.rstrip().endswith((".", "!", "?")):
            rewritten = rewritten.rstrip() + "."
        lines[idx] = rewritten
        return "\n".join(lines)
    # 没有 Memory Judgment 行：补一行（保持 4 行结构最低限度可用）。
    appended = f"Memory Judgment: {opener} the believed memory remains consistent with the visible evidence."
    return ("\n".join(lines) + "\n" + appended).strip()


# 弱证据检测：当 Memory Judgment 行的 body 包含否定信号（"no evidence supports"、
# "not contradicted"、"not supported" 等）但 opener 是 Corrected/Advanced 时，
# 说明教师模型认为证据不足但被 verdict 强行推向 GT。这种情况下正文与 opener 语义
# 矛盾（"Corrected because no evidence supports X"），需要将 body 改写为
# 诚实的弱证据表述，让学生学到"修正可以是弱依据的合理猜测"而不是"编造证据"。
_WEAK_EVIDENCE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bno\s+(?:visible\s+)?evidence\s+(?:supports?|confirms?)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+(?:strongly\s+)?supported\b", re.IGNORECASE),
    re.compile(r"\bdoes\s+not\s+support\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+support\b", re.IGNORECASE),                # 漏检: "... cues do not support ..."
    re.compile(r"\bunsupported\b", re.IGNORECASE),                       # 漏检: "... making X unsupported ..."
    re.compile(r"\bnot\s+contradicted\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:\w+\s+){0,2}cues?\b", re.IGNORECASE),         # 漏检: "no visual cues", "no interaction cues"
    re.compile(r"\bno\s+(?:clear\s+)?(?:visual\s+)?evidence\b", re.IGNORECASE),
    re.compile(r"\blacks?\s+(?:any|clear|visible|definitive)\s+(?:evidence|indicator|cue)\b", re.IGNORECASE),  # 漏检: "lacks any indicator"
)


def _honest_weak_evidence_judgment(text: str, *, verdict: str, slot: str) -> str:
    """Rewrite Memory Judgment body when it contains weak-evidence signals.

    仅对 CHANGE / ADVANCE verdict 生效。当 Memory Judgment 行的 body 包含
    否定性证据信号（如 "no evidence supports"、"not contradicted"）时，将
    body 替换为诚实的弱证据表述，避免 "Corrected because no evidence supports X"
    这种语义矛盾。KEEP verdict 的 body 通常是自洽的，无需改写。
    """

    verdict_upper = (verdict or "").strip().upper()
    if verdict_upper not in ("CHANGE", "ADVANCE"):
        return text
    if not text:
        return text

    lines = text.splitlines()
    for idx, raw in enumerate(lines):
        m = _MEMORY_JUDGMENT_LINE_RE.match(raw)
        if not m:
            continue
        body = m.group("body").strip()
        # 先剥掉 opener 词组，只检查 body 实质内容
        body_content = _OPENER_PREFIX_RE.sub("", body).strip()
        # 检测 _enforce 的通用 fallback "remains consistent"——它对 CHANGE/ADVANCE
        # 是语义矛盾的（"Corrected because ... remains consistent"），需要改写。
        weak_detected = any(p.search(body_content) for p in _WEAK_EVIDENCE_PATTERNS)
        if not weak_detected and "remains consistent" not in body_content.lower():
            # body 不含弱证据信号，也不含矛盾 fallback，保持原样
            return text
        # 检测到弱证据矛盾 → 改写 body
        if verdict_upper == "CHANGE":
            new_body = (
                f"the believed {slot} is not strongly supported "
                "and the correction is a plausible alternative without strong visual confirmation."
            )
        else:  # ADVANCE
            new_body = (
                "the advance is not strongly confirmed by visible evidence "
                "but is a plausible next step without clear contradiction."
            )
        prefix = m.group("prefix")
        opener = _memory_judgment_opener_for(verdict)
        rewritten = f"{prefix}{opener} {new_body}"
        if not rewritten.rstrip().endswith((".", "!", "?")):
            rewritten = rewritten.rstrip() + "."
        lines[idx] = rewritten
        return "\n".join(lines)

    # 没有 Memory Judgment 行，不做额外处理（_enforce 已补 fallback）
    return text


# Per-step spec: (task, focus, slot_phrase, options_name)。
# 学生 / 教师 step 块共用同一 Task / Constraint 骨架，step2/3 同模板只差槽位；
# 4 行分析格式 + label 顺序 + evidence policy 已下沉各 step 的 system prompt，
# 这里只留 step 级 Task / Constraint(focus + 通用兜底句) / Then write。
_STEP_SPEC: Dict[str, Tuple[str, str, str, str]] = {
    "STEP1": (
        "Decide ROAD_STRUCTURE from ROAD_STRUCTURE_CHOICES.",
        "Judge by visible road layout only; do not infer merging, braking, cut-in, or active flow from a single lead vehicle; a single lead vehicle alone never proves HIGHWAY_MERGE",
        "ROAD_STRUCTURE",
        "ROAD_STRUCTURE_CHOICES",
    ),
    "STEP2": (
        "Decide SCENE from SCENE_CHOICES.",
        "Rely on visible interaction cues; do not infer hidden merging, yielding, lane change, turn, stop, cut-in, or active flow unless visible across frames; do not infer a distant vehicle is stationary or slow-moving without motion evidence",
        "SCENE",
        "SCENE_CHOICES",
    ),
    "STEP3": (
        "Decide STATUS and SUBGOAL from EVENT_OPTIONS.",
        "Prefer keeping STATUS; advance only on a clear visible phase change; SUBGOAL may lead STATUS, and when STATUS is kept but SUBGOAL is ahead, describe SUBGOAL as the retained next objective",
        "STATUS/SUBGOAL",
        "EVENT_OPTIONS",
    ),
}

# VERDICT → 推理方向动词。强制教师分析正文与脚本 verdict 同向，避免
# "正文替旧 label 说话、opener 却被改成 Corrected" 这类轻微矛盾。
_VERDICT_VERB: Dict[str, str] = {
    "KEEP": "keep",
    "ADVANCE": "advance",
    "CHANGE": "replace",
}


def _step_constraint_sentence(step_tag: str) -> str:
    """Render the shared Constraint line: step-specific focus + 通用兜底句。"""

    _, focus, slot, options = _STEP_SPEC[step_tag]
    return (
        f"{focus}. Keep the believed {slot} if no listed option is clearly supported, "
        f"and never emit an option outside {options}."
    )


def _render_student_block(*, step_tag: str, label_instruction: str) -> str:
    """Compose the student turn body for one step: Task / Constraint / Write.

    4 行分析格式 + 词数目标 + "先分析再写 label" 的顺序全部合入一行 Write 指令，
    不再依赖 system prompt。system 只定角色和通用原则，user 定任务和格式。
    """

    task, _, _, _ = _STEP_SPEC[step_tag]
    return (
        f"[{step_tag}]\n"
        f"Task: {task}\n"
        f"Constraint: {_step_constraint_sentence(step_tag)}\n"
        "Write four analysis lines (Scene Description / Critical Object Description / "
        "Reasoning on Intent / Memory Judgment), aim for 100-150 words, "
        f"then write: {label_instruction}"
    )


def _render_teacher_block(*, step_tag: str, verdict: str, gt_hint: str) -> str:
    """Compose the privileged teacher turn body for one step.

    与学生同骨架（同 Task / Constraint），仅多 verdict / 推理方向一致性指令 /
    opener 强制 / GT-aware hint，并显式禁止写 label 行。4 行格式 + 词数目标
    合入 no-label 指令行，不再依赖 system prompt。
    """

    task, _, slot, _ = _STEP_SPEC[step_tag]
    opener = _memory_judgment_opener_for(verdict)
    verb = _VERDICT_VERB.get((verdict or "").strip().upper(), "keep")
    return (
        f"[{step_tag}_TEACHER]\n"
        f"VERDICT: {verdict}\n"
        f"Your reasoning must be consistent with this verdict: argue to {verb} the believed {slot}. "
        f'The Memory Judgment line MUST start with "{opener}".\n'
        f"{gt_hint}\n"
        f"Task: {task}\n"
        f"Constraint: {_step_constraint_sentence(step_tag)}\n"
        "Write only the four analysis lines (Scene Description / Critical Object Description / "
        "Reasoning on Intent / Memory Judgment), aim for 100-150 words; "
        "do not write any label lines."
    )


def build_step1_user_prompt(image_count: int, memory: Optional[Memory] = None) -> str:
    """学生 step1 prompt（road-only memory + 6 桶 ROAD_STRUCTURE 选择）。

    新口径（生产路径，``memory`` 必传）：
      - 只读 layer-1 road-only memory，不提前暴露 scene/status/subgoal；
      - user 仅包含 memory + ROAD_STRUCTURE_CHOICES + [STEP1] task/constraint
        + label 行；通用 evidence policy / 4 行格式由各 step system prompt 兜底；
      - 一行 ``ROAD_STRUCTURE: <name>``。

    兼容兜底：``memory=None`` 时退回最小形态（仅视觉描述、不读 memory、不出标签），
    仅供 test_kv_reuse 等单元入口使用，生产路径必须传 memory。
    """

    if memory is None:
        return (
            f"[STEP1]\n{image_count} images are ordered oldest to newest; the last image is now.\n"
            "Describe the road layout and nearby actors."
        )
    return (
        f"{memory.format_step1_student_text()}\n\n"
        f"{road_structure_choices_block()}\n\n"
        f"({image_count} stitched frames are ordered oldest to newest; the last frame is now.)\n\n"
        f"{_render_student_block(step_tag='STEP1', label_instruction='ROAD_STRUCTURE: <name>')}"
    )


def build_step1_teacher_prompt(memory: Memory, gt_road_structure: str) -> str:
    """老师 step1 prompt：road-only KEEP/CHANGE analysis for layer-1."""

    verdict = step1_teacher_verdict(memory, gt_road_structure)
    gt_rs_desc = ROAD_STRUCTURE_LABELS.get(gt_road_structure, gt_road_structure)

    if verdict == "KEEP":
        gt_hint = (
            "Explain whether the believed road structure is directly supported or merely "
            "not contradicted; do not invent unseen cues for the label."
        )
    else:
        gt_hint = (
            "If visible road-layout evidence contradicts the believed structure, name the "
            "clearest contradictory cue; otherwise note that the believed label is not "
            f"strongly supported and the correction to {gt_road_structure}: {gt_rs_desc} "
            "is a plausible alternative without strong visual confirmation. "
            "Do not invent unseen cues."
        )

    return (
        f"{memory.format_step1_road_text(gt_road_structure)}\n\n"
        f"{road_structure_choices_block()}\n\n"
        f"{_render_teacher_block(step_tag='STEP1', verdict=verdict, gt_hint=gt_hint)}"
    )


def build_step1_teacher_target(
    analysis: str,
    gt_road_structure: str,
    *,
    verdict: str = "KEEP",
) -> str:
    """Build the supervised step1 target from teacher analysis plus the GT label.

    ``verdict`` 控制 Memory Judgment 行 opener 与监督 label 的一致性。caller 应
    显式传 "KEEP" 或 "CHANGE"；省略时按 KEEP 兜底（向后兼容旧测试桩）。
    """

    opener = _memory_judgment_opener_for(verdict)
    cleaned = _clean_teacher_analysis(analysis)
    if not cleaned:
        cleaned = _fallback_teacher_analysis("road_structure", verdict=verdict)
    cleaned = _enforce_memory_judgment_opener(cleaned, opener)
    cleaned = _honest_weak_evidence_judgment(cleaned, verdict=verdict, slot="ROAD_STRUCTURE")
    return f"{cleaned}\nROAD_STRUCTURE: {gt_road_structure}".strip()


# ---------------------------------------------------------------------------
# Step 2：在 layer-1 桶下选 scene
# ---------------------------------------------------------------------------


def build_step2_student_prompt(memory: Memory) -> str:
    """学生 step2 prompt（SCENE_CHOICES 收窄到 ``memory.road_structure`` 桶）。"""

    return (
        f"{memory.format_text()}\n\n"
        f"{scene_choices_block_for(memory.road_structure)}\n\n"
        f"{_render_student_block(step_tag='STEP2', label_instruction='SCENE: <scenario_name>')}"
    )


def build_step2_teacher_prompt(memory: Memory, gt_road_structure: str, gt_scene: str) -> str:
    """老师 step2 prompt：independent KEEP/CHANGE scene analysis."""

    gt_scene = str(canonicalize_scene(gt_scene))
    verdict = step2_teacher_verdict(memory, gt_scene)
    gt_scene_desc = CANONICAL_SCENARIO_LABELS.get(gt_scene, gt_scene)

    if verdict == "KEEP":
        gt_hint = (
            "Explain whether the believed scene is directly supported or merely not contradicted; "
            "do not invent unseen cues for the label."
        )
    else:
        gt_hint = (
            "If visible interaction cues contradict the believed scene, name the clearest one; "
            "otherwise note that the believed scene is not strongly supported and the correction "
            f"to {gt_scene}: {gt_scene_desc} is a plausible alternative without strong visual "
            "confirmation. Do not invent unseen cues."
        )

    return (
        f"{memory.format_step2_scene_text(gt_road_structure, gt_scene)}\n\n"
        f"{scene_choices_block_for(gt_road_structure, heading='ANSWER_ROAD_STRUCTURE')}\n\n"
        f"{_render_teacher_block(step_tag='STEP2', verdict=verdict, gt_hint=gt_hint)}"
    )


def build_step2_teacher_target(
    analysis: str,
    gt_scene: str,
    *,
    verdict: str = "KEEP",
) -> str:
    """把 teacher step2 分析与 GT scene 拼成 student 的 teacher-forced target。"""

    gt_scene = str(canonicalize_scene(gt_scene))
    opener = _memory_judgment_opener_for(verdict)
    cleaned = _clean_teacher_analysis(analysis)
    if not cleaned:
        cleaned = _fallback_teacher_analysis("scene", verdict=verdict)
    cleaned = _enforce_memory_judgment_opener(cleaned, opener)
    cleaned = _honest_weak_evidence_judgment(cleaned, verdict=verdict, slot="SCENE")
    return f"{cleaned}\nSCENE: {gt_scene}".strip()


# ---------------------------------------------------------------------------
# Step 3：在 scene 事件序列中选 status / subgoal
# ---------------------------------------------------------------------------


def build_step3_student_prompt(memory: Memory) -> str:
    """学生 step3 prompt：EVENT_OPTIONS 按 memory.scene 事件序列裁剪。"""

    # label_instruction 含真实换行，必须先放普通字符串字面量再传入 f-string，
    # 不能塞进 f-string 表达式（Python<3.12 禁止表达式内出现反斜杠）。
    label_instruction = "STATUS: <event_name>\nSUBGOAL: <event_name>"
    return (
        f"{memory.format_text()}\n\n"
        f"[EVENT_OPTIONS]\n{_event_sequence_block(memory.scene)}\n[/EVENT_OPTIONS]\n\n"
        f"{_render_student_block(step_tag='STEP3', label_instruction=label_instruction)}"
    )


def build_step3_teacher_prompt(
    memory: Memory,
    gt_road_structure: str,
    gt_scene: str,
    gt_status: str,
    gt_subgoal: str,
) -> str:
    """老师 step3 prompt：independent KEEP/CHANGE status/subgoal analysis."""

    gt_scene = str(canonicalize_scene(gt_scene))
    verdict = step3_teacher_verdict(memory, gt_scene, gt_status, gt_subgoal)

    memory_status_desc = EVENT_DESCRIPTIONS.get(memory.status, memory.status)
    memory_subgoal_desc = EVENT_DESCRIPTIONS.get(memory.subgoal, memory.subgoal)
    gt_status_desc = EVENT_DESCRIPTIONS.get(gt_status, gt_status)
    gt_subgoal_desc = EVENT_DESCRIPTIONS.get(gt_subgoal, gt_subgoal)

    if verdict == "KEEP":
        gt_hint = (
            f"Explain why current phase '{memory_status_desc}' and next objective "
            f"'{memory_subgoal_desc}' can both be kept without advancing STATUS prematurely."
        )
    elif verdict == "ADVANCE":
        gt_hint = (
            "If visible temporal-progress cues support advancing, name them; "
            "otherwise note that the advance from current "
            f"'{memory_status_desc}' / next '{memory_subgoal_desc}' toward current "
            f"'{gt_status_desc}' / next '{gt_subgoal_desc}' is not strongly confirmed "
            "by visible evidence. Do not invent hidden intent."
        )
    else:
        gt_hint = (
            "If visible temporal-progress cues contradict current "
            f"'{memory_status_desc}' / next '{memory_subgoal_desc}', name the clearest one; "
            "otherwise note that the current phase is not strongly supported and the correction "
            f"to current '{gt_status_desc}' / next '{gt_subgoal_desc}' is a plausible "
            "alternative without strong visual confirmation. Do not invent hidden intent."
        )

    return (
        f"{memory.format_step3_event_text(gt_road_structure, gt_scene, gt_status, gt_subgoal)}\n\n"
        f"[EVENT_OPTIONS]\n{_event_sequence_block(gt_scene)}\n[/EVENT_OPTIONS]\n\n"
        f"{_render_teacher_block(step_tag='STEP3', verdict=verdict, gt_hint=gt_hint)}"
    )


def build_step3_teacher_target(
    analysis: str,
    gt_status: str,
    gt_subgoal: str,
    *,
    verdict: str = "KEEP",
) -> str:
    """把 teacher step3 分析与 GT status/subgoal 拼成监督 target。"""

    opener = _memory_judgment_opener_for(verdict)
    cleaned = _clean_teacher_analysis(analysis)
    if not cleaned:
        cleaned = _fallback_teacher_analysis("event", verdict=verdict)
    cleaned = _enforce_memory_judgment_opener(cleaned, opener)
    cleaned = _honest_weak_evidence_judgment(cleaned, verdict=verdict, slot="STATUS/SUBGOAL")
    return f"{cleaned}\nSTATUS: {gt_status}\nSUBGOAL: {gt_subgoal}".strip()


# ---------------------------------------------------------------------------
# 输出解析 / 校验
# ---------------------------------------------------------------------------

_ROAD_STRUCTURE_RE = re.compile(r"^\s*ROAD_STRUCTURE\s*:\s*([^\s]+)", re.IGNORECASE | re.MULTILINE)
_SCENE_RE = re.compile(r"^\s*SCENE\s*:\s*([^\s]+)", re.IGNORECASE | re.MULTILINE)
_STATUS_RE = re.compile(r"^\s*STATUS\s*:\s*([^\s]+)", re.IGNORECASE | re.MULTILINE)
_SUBGOAL_RE = re.compile(r"^\s*SUBGOAL\s*:\s*([^\s]+)", re.IGNORECASE | re.MULTILINE)

# 用来把老师万一漏写的整行标签从分析文本里剥掉，避免污染监督 target。
_LABEL_LINE_RE = re.compile(
    r"^\s*(?:ROAD_STRUCTURE|SCENE|STATUS|SUBGOAL)\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)

# 教师在 raw 输出里偶尔写 "Corrected because: ..." 或 "Kept because: ..."
# 这种不带 "Memory Judgment:" 前缀的 orphaned opener 行。它们不应进入
# student target——脚本在后面会重新构建正确的 Memory Judgment 行。
_ORPHANED_OPENER_LINE_RE = re.compile(
    r"^\s*(?:Kept|Corrected|Advanced)\s+because:.*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_label_lines(text: str) -> str:
    """删除分析文本中的 label 行和 orphaned opener 行。"""

    if not text:
        return ""
    cleaned = _ORPHANED_OPENER_LINE_RE.sub("", text)
    return _LABEL_LINE_RE.sub("", cleaned).strip()


_PROMPT_MARKER_LINE_RE = re.compile(
    r"(?im)^\s*\[/?(?:STEP\d(?:_TEACHER)?|MEMORY|STEP1_ROAD_MEMORY|STEP1_ROAD_CONTEXT|STEP2_SCENE_CONTEXT|STEP3_EVENT_CONTEXT|ROAD_STRUCTURE_CHOICES|SCENE_CHOICES|EVENT_OPTIONS)[^\n]*\n?"
)
_PROMPT_BLOCK_RE = re.compile(
    r"(?ims)^\s*\[(?:MEMORY|STEP1_ROAD_MEMORY|STEP1_ROAD_CONTEXT|STEP2_SCENE_CONTEXT|STEP3_EVENT_CONTEXT|ROAD_STRUCTURE_CHOICES|SCENE_CHOICES|EVENT_OPTIONS)[^\n]*\n"
    r".*?"
    r"^\s*\[/(?:MEMORY|STEP1_ROAD_MEMORY|STEP1_ROAD_CONTEXT|STEP2_SCENE_CONTEXT|STEP3_EVENT_CONTEXT|ROAD_STRUCTURE_CHOICES|SCENE_CHOICES|EVENT_OPTIONS)\]\s*\n?"
)

_PRIVATE_FIELD_REPLACEMENTS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bGROUND_TRUTH_ROAD_STRUCTURE\b", re.IGNORECASE), "the corrected road structure"),
    (re.compile(r"\bGROUND_TRUTH_SCENE\b", re.IGNORECASE), "the corrected scene"),
    (re.compile(r"\bGROUND_TRUTH_STATUS\b", re.IGNORECASE), "the corrected status"),
    (re.compile(r"\bGROUND_TRUTH_SUBGOAL\b", re.IGNORECASE), "the corrected subgoal"),
    (re.compile(r"\bANSWER_ROAD_STRUCTURE\b", re.IGNORECASE), "the corrected road structure"),
    (re.compile(r"\bANSWER_SCENE\b", re.IGNORECASE), "the corrected scene"),
    (re.compile(r"\bANSWER_STATUS\b", re.IGNORECASE), "the corrected status"),
    (re.compile(r"\bANSWER_SUBGOAL\b", re.IGNORECASE), "the corrected subgoal"),
    (re.compile(r"\bREFERENCE_ROAD_STRUCTURE\b", re.IGNORECASE), "the corrected road structure"),
    (re.compile(r"\bREFERENCE_SCENE\b", re.IGNORECASE), "the corrected scene"),
    (re.compile(r"\bREFERENCE_STATUS\b", re.IGNORECASE), "the corrected status"),
    (re.compile(r"\bREFERENCE_SUBGOAL\b", re.IGNORECASE), "the corrected subgoal"),
    (re.compile(r"\bground\s*truth\b", re.IGNORECASE), "corrected label"),
    (re.compile(r"\breference\s+label\b", re.IGNORECASE), "corrected label"),
    (re.compile(r"\banswer\s+label\b", re.IGNORECASE), "corrected label"),
    (re.compile(r"\bBELIEVED_ROAD_STRUCTURE\b", re.IGNORECASE), "the believed road structure"),
    (re.compile(r"\bBELIEVED_SCENE\b", re.IGNORECASE), "the believed scene"),
    (re.compile(r"\bBELIEVED_STATUS\b", re.IGNORECASE), "the believed status"),
    (re.compile(r"\bBELIEVED_SUBGOAL\b", re.IGNORECASE), "the believed subgoal"),
    (re.compile(r"\bBELIEF_SUBGOAL\b", re.IGNORECASE), "the believed subgoal"),
)


def _clean_private_field_names(text: str) -> str:
    """Remove teacher-private answer-field wording from student-facing analysis."""

    cleaned = text
    for pattern, replacement in _PRIVATE_FIELD_REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def _fallback_teacher_analysis(kind: str, *, verdict: str = "KEEP") -> str:
    """Build a minimal four-line fallback when teacher raw text is empty.

    新 4 行 heading：Scene Description / Critical Object Description /
    Reasoning on Intent / Memory Judgment。``verdict`` 决定 Memory Judgment
    opener，使后续脚本追加的 GT label 与 opener 自洽。
    """

    opener = _memory_judgment_opener_for(verdict)
    if verdict.upper() == "CHANGE":
        memory_line = {
            "road_structure": "the believed road structure is not strongly supported and the correction is a plausible alternative without strong visual confirmation.",
            "scene": "the believed scene is not strongly supported and the correction is a plausible alternative without strong visual confirmation.",
            "event": "the believed status or subgoal is not strongly supported and the correction is a plausible alternative without strong visual confirmation.",
        }.get(kind, "the believed label is not strongly supported and the correction is a plausible alternative without strong visual confirmation.")
    elif verdict.upper() == "ADVANCE":
        memory_line = {
            "road_structure": "the advance is not strongly confirmed by visible evidence but is a plausible next step without clear contradiction.",
            "scene": "the advance is not strongly confirmed by visible evidence but is a plausible next step without clear contradiction.",
            "event": "the advance is not strongly confirmed by visible evidence but is a plausible next step without clear contradiction.",
        }.get(kind, "the advance is not strongly confirmed by visible evidence but is a plausible next step without clear contradiction.")
    else:
        memory_line = {
            "road_structure": "no reliable teacher analysis was available to justify changing the believed road structure.",
            "scene": "no reliable teacher analysis was available to justify changing the believed scene.",
            "event": "no reliable teacher analysis was available to justify changing the believed status or subgoal.",
        }.get(kind, "no reliable teacher analysis was available to justify changing the remembered state.")
    return "\n".join(
        [
            "Scene Description: The latest frames show the current road layout and traffic context.",
            "Critical Object Description: No reliable specific critical object can be confirmed from the available cues.",
            "Reasoning on Intent: The ego decision is treated conservatively given the limited visible evidence.",
            f"Memory Judgment: {opener} {memory_line}",
        ]
    )


def _clean_teacher_analysis(text: str) -> str:
    """Light cleanup before appending scripted labels.

    v4 是分 step 监督学习，最终结构化标签仍由脚本追加；analysis 文本必须保持学生
    可学视角，不能把老师私有 answer / ground-truth 字段逐字带进 target。

    1. 剥掉所有 ``ROAD_STRUCTURE:`` / ``SCENE:`` / ``STATUS:`` / ``SUBGOAL:`` 整行，
       防止脚本在末尾追加的 GT 标签被 parser 取错（parser 是 ``re.search`` 取第一个）；
    2. 先剥掉完整 ``[MEMORY]`` / ``[STEP*_CONTEXT]`` / choice/options 块，再剥
       ``[STEPx]`` 等残留 prompt marker 行，避免老师把 prompt 模板抄进 target。
    3. 把 ``GROUND_TRUTH_*`` / ``ANSWER_*`` / ``REFERENCE_*`` 这类私有字段名改成
       ``the corrected ...`` 口径；把 ``BELIEVED_*`` / ``BELIEF_SUBGOAL`` 字段名改成
       自然语言 believed 口径。

    其余文字原样保留；剥完后空字符串才上调 ``_fallback_teacher_analysis``。
    """

    if not text:
        return ""
    cleaned = _strip_label_lines(text)
    cleaned = _PROMPT_BLOCK_RE.sub("", cleaned)
    cleaned = _PROMPT_MARKER_LINE_RE.sub("", cleaned)
    cleaned = _clean_private_field_names(cleaned)
    return cleaned.strip()


def parse_output(text: str) -> Dict[str, Optional[str]]:
    """从自由生成文本中解析 4 类标签值。

    解析器只取行首 ``KEY: value``，避免分析段落里偶然提到某个字段名时误判。
    返回值可能为 ``None``，调用方必须再用 ``validate_road_structure`` /
    ``validate_scene`` / ``validate_event`` 做合法性检查。
    """

    rs = _ROAD_STRUCTURE_RE.search(text or "")
    scene = _SCENE_RE.search(text or "")
    status = _STATUS_RE.search(text or "")
    subgoal = _SUBGOAL_RE.search(text or "")
    return {
        "road_structure": rs.group(1).strip() if rs else None,
        "scene": scene.group(1).strip() if scene else None,
        "status": status.group(1).strip() if status else None,
        "subgoal": subgoal.group(1).strip() if subgoal else None,
    }


def analysis_before_choices(text: str) -> str:
    """返回第一条标签行之前的分析文本。"""

    cuts = [
        m.start()
        for m in (
            _ROAD_STRUCTURE_RE.search(text),
            _SCENE_RE.search(text),
            _STATUS_RE.search(text),
            _SUBGOAL_RE.search(text),
        )
        if m
    ]
    return text[: min(cuts)].strip() if cuts else text.strip()


def target_spans_road_structure(assistant_text: str) -> Dict[str, Tuple[int, int]]:
    """返回 step1 中需要打高权重 CE 的 ROAD_STRUCTURE 值字符 span。"""

    m = _ROAD_STRUCTURE_RE.search(assistant_text)
    return {"road_structure": (m.start(1), m.end(1))} if m else {}


def target_spans_scene(assistant_text: str) -> Dict[str, Tuple[int, int]]:
    """返回 step2 中需要打高权重 CE 的 scene 值字符 span。"""

    m = _SCENE_RE.search(assistant_text)
    return {"scene": (m.start(1), m.end(1))} if m else {}


def target_spans_status(assistant_text: str) -> Dict[str, Tuple[int, int]]:
    """返回 step3 中需要打高权重 CE 的 status/subgoal 值字符 span。"""

    spans: Dict[str, Tuple[int, int]] = {}
    for key, regex in (("status", _STATUS_RE), ("subgoal", _SUBGOAL_RE)):
        m = regex.search(assistant_text)
        if m:
            spans[key] = (m.start(1), m.end(1))
    return spans


def _variants(name: str) -> List[str]:
    """Legacy helper kept for old imports."""

    vals = {name, name.replace("_", " "), name.replace("_", "")}
    return [x.lower() for x in vals if x]


def check_gt_leak_road_structure(analysis_text: str, gt_road_structure: str) -> bool:
    """Legacy no-op."""

    return False


def check_gt_leak_scene(analysis_text: str, gt_scene: str) -> bool:
    """Legacy no-op."""

    return False


def check_gt_leak_status_subgoal(analysis_text: str, gt_status: str, gt_subgoal: str) -> bool:
    """Legacy no-op."""

    return False


def validate_road_structure(rs: Optional[str]) -> bool:
    """检查 layer-1 是否来自 ROAD_STRUCTURE_LABELS 白名单。"""

    return bool(rs and rs in ROAD_STRUCTURE_LABELS)


def validate_scene(scene: Optional[str]) -> bool:
    """检查 scene 是否来自 v4 canonical 场景白名单或可折叠 alias。"""

    canonical = canonicalize_scene(scene)
    return bool(canonical and canonical in CANONICAL_SCENARIO_LABELS)


def validate_event(scene: str, event: Optional[str]) -> bool:
    """检查 event 是否属于当前 scene 的合法事件序列。"""

    if not event:
        return False
    try:
        return event in get_full_sequence(str(canonicalize_scene(scene)))
    except Exception:
        return False


def next_event(scene: str, status: str) -> str:
    """返回某个 status 后面的下一事件；非法 status 时回退到第一子目标。"""

    seq = get_full_sequence(str(canonicalize_scene(scene)))
    try:
        idx = seq.index(status)
    except ValueError:
        return seq[1] if len(seq) > 1 else seq[0]
    return seq[idx + 1] if idx + 1 < len(seq) else seq[-1]
