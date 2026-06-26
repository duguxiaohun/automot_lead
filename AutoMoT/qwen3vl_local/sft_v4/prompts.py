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
    get_full_sequence,
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

SYSTEM_PROMPT_V4 = """\
You are an autonomous driving agent.
Use the 4 stitched RGB frames as visual context: oldest to newest, left/front/right views.
Focus on traffic lights and signs, nearby vehicles, pedestrians, obstacles,
lane markings, road structure, and the key factors affecting ego's decision."""


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

# 42 个 scene 到 layer-1 桶的 1:1 映射（D21 桶分配）。
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

# 反向索引：每个桶下的 layer-2 候选列表（排序，便于 SCENE_CHOICES 渲染稳定）。
ROAD_STRUCTURE_TO_SCENES: Dict[str, List[str]] = {}
for _scene, _rs in SCENE_TO_ROAD_STRUCTURE.items():
    ROAD_STRUCTURE_TO_SCENES.setdefault(_rs, []).append(_scene)
for _rs in ROAD_STRUCTURE_TO_SCENES:
    ROAD_STRUCTURE_TO_SCENES[_rs].sort()


def get_road_structure(scene: str) -> str:
    """返回某 scene 对应的 layer-1 道路结构 token，缺失时直接 raise。"""

    if scene not in SCENE_TO_ROAD_STRUCTURE:
        raise KeyError(
            f"scene {scene!r} has no road-structure mapping; "
            "update SCENE_TO_ROAD_STRUCTURE in prompts.py."
        )
    return SCENE_TO_ROAD_STRUCTURE[scene]


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
        scene_desc = SCENARIO_LABELS.get(self.scene, self.scene)
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

        gt_rs_desc = ROAD_STRUCTURE_LABELS.get(gt_road_structure, gt_road_structure)
        scene_choices = set(ROAD_STRUCTURE_TO_SCENES.get(gt_road_structure, []))
        return (
            "[STEP2_SCENE_CONTEXT]\n"
            f"ANSWER_ROAD_STRUCTURE={gt_road_structure} ({gt_rs_desc})\n"
            f"BELIEVED_SCENE={_label_with_optional_desc(self.scene, SCENARIO_LABELS, scene_choices)}\n"
            f"ANSWER_SCENE={_label_with_optional_desc(gt_scene, SCENARIO_LABELS, scene_choices)}\n"
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

        gt_rs_desc = ROAD_STRUCTURE_LABELS.get(gt_road_structure, gt_road_structure)
        gt_scene_desc = SCENARIO_LABELS.get(gt_scene, gt_scene)
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

    机制目的：Phase B 期间保证 ``memory.road_structure`` 始终 = GT 桶 → step1
    永远命中 → step2/step3 触发率回到与 v3 / 现有 v4 持平，弥补 step1 加入触发
    链后 step3 触发率被腰斩的问题。
    """

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
    memory_road_structure_after_step1: str,
    gt_road_structure: str,
) -> bool:
    """step2 触发条件（D24 选项 A）。

    只有 step1 后 ``memory.road_structure == gt_road_structure`` 才进 step2。
    错 layer-1 → 跳 step2 + step3，仅监督 step1（L_A1 + L_RS1）。
    """

    return memory_road_structure_after_step1 == gt_road_structure


def should_trigger_step3(
    *,
    memory_scene_after_step2: str,
    gt_scene: str,
) -> bool:
    """step3 触发条件（与 v3 D8 / v4 现有口径一致）。

    只有 step2 后 ``memory.scene == gt_scene`` 才进 step3。前提是 step2 已跑过
    （由调用方保证）；本 helper 只判 scene 层。
    """

    return memory_scene_after_step2 == gt_scene


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

    scenes = ROAD_STRUCTURE_TO_SCENES.get(road_structure, sorted(SCENARIO_LABELS))
    lines = [f"[SCENE_CHOICES] under {heading} = {road_structure}"]
    for name in scenes:
        lines.append(f"- {name}: {SCENARIO_LABELS[name]}")
    lines.append("[/SCENE_CHOICES]")
    return "\n".join(lines)


def scenario_choices_block() -> str:
    """渲染全量 42 项 SCENE_CHOICES（保留作 v2/v3 兼容 / 调试用，v4 不调用）。"""

    lines = ["[SCENE_CHOICES]"]
    for name in sorted(SCENARIO_LABELS):
        lines.append(f"- {name}: {SCENARIO_LABELS[name]}")
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
# Step 1：纯视觉描述 + ROAD_STRUCTURE 选择
# ---------------------------------------------------------------------------


def _teacher_structured_analysis_instructions(memory_judgment: str) -> str:
    """Shared concise visual-analysis skeleton for all teacher steps.

    Teacher can see the answer fields, but the supervised analysis must read
    like a student-facing explanation rather than a private answer key.
    """

    return (
        "Write for the student perspective: use 'believed ...' and 'corrected ...' phrasing.\n"
        "Do not mention answer, ground truth, reference labels, or private field names.\n"
        "Use visible cues; if cues are unclear, note uncertainty briefly and do not invent unseen actors.\n"
        "Write the analysis in this order, one heading per line, each line non-empty:\n"
        "Scene Description: ...\n"
        "Critical Object Description: ...\n"
        "Reasoning on Intent: ...\n"
        f"Memory Judgment: {memory_judgment}\n"
        "Plain text only -- no markdown headings, bullets, numbered lists, JSON, or code blocks."
    )


def build_step1_user_prompt(image_count: int, memory: Optional[Memory] = None) -> str:
    """学生 step1 prompt（D26）。

    新口径（生产路径，``memory`` 必传）：
      - 2 句"不引用 memory"的视觉描述（保留 v3 纯视觉接地任务）；
      - 1 句对 believed road structure 的 KEEP/CHANGE 论证；
      - 一行 ``ROAD_STRUCTURE: <name>``。

    兼容兜底：``memory=None`` 时退回 v3 形态（仅视觉描述、不读 memory、不出标签）。
    仅供 test_kv_reuse 等单元入口使用，生产路径必须传 memory。
    """

    if memory is None:
        return (
            f"[STEP1]\n{image_count} images are ordered oldest to newest; the last image is now.\n"
            "Describe the road layout and nearby actors."
        )
    return (
        f"{memory.format_text()}\n\n"
        f"{road_structure_choices_block()}\n\n"
        f"[STEP1]\n"
        f"{image_count} images are ordered oldest to newest; the last image is now.\n"
        "Describe the visible road layout / actors / signals. "
        "Explain whether the believed road structure fits or should change. "
        "Then copy exactly one option name:\n"
        "ROAD_STRUCTURE: <name>"
    )


def build_step1_teacher_prompt(memory: Memory, gt_road_structure: str) -> str:
    """老师 step1 prompt：road-only KEEP/CHANGE analysis for layer-1."""

    verdict = "KEEP" if memory.road_structure == gt_road_structure else "CHANGE"
    gt_rs_desc = ROAD_STRUCTURE_LABELS.get(gt_road_structure, gt_road_structure)

    if verdict == "KEEP":
        verdict_line = "VERDICT: KEEP -- the believed road structure is correct."
        focus_line = "Explain which visible road-layout cues make the believed road structure fit."
        task_line = _teacher_structured_analysis_instructions(
            "explain why the believed road structure fits and which road-layout cues support it."
        )
    else:
        verdict_line = "VERDICT: CHANGE -- the believed road structure is wrong."
        focus_line = (
            f"Explain why the believed road structure does not fit, then guide toward {gt_road_structure}: {gt_rs_desc}."
        )
        task_line = _teacher_structured_analysis_instructions(
            "explain what does not fit the believed road structure and what visible cues support the correction."
        )

    return (
        f"{memory.format_step1_road_text(gt_road_structure)}\n\n"
        f"{road_structure_choices_block()}\n\n"
        "[STEP1_TEACHER]\n"
        f"{verdict_line}\n"
        f"{focus_line}\n"
        f"{task_line}"
    )


def build_step1_teacher_target(analysis: str, gt_road_structure: str) -> str:
    """Build the supervised step1 target from teacher analysis plus the GT label."""

    cleaned = _clean_teacher_analysis(analysis)
    if not cleaned:
        cleaned = _fallback_teacher_analysis("road_structure")
    return f"{cleaned}\nROAD_STRUCTURE: {gt_road_structure}".strip()


# ---------------------------------------------------------------------------
# Step 2：在 layer-1 桶下选 scene
# ---------------------------------------------------------------------------


def build_step2_student_prompt(memory: Memory) -> str:
    """学生 step2 prompt。

    SCENE_CHOICES 收窄到 ``memory.road_structure`` 这一桶（D21 + D24）。
    """

    return (
        f"{memory.format_text()}\n\n"
        f"{scene_choices_block_for(memory.road_structure)}\n\n"
        "[STEP2]\n"
        "Use the narrowed choices. Explain keep/correct for the believed scene. "
        "Then copy exactly one option name:\n"
        "SCENE: <scenario_name>"
    )


def build_step2_teacher_prompt(memory: Memory, gt_road_structure: str, gt_scene: str) -> str:
    """老师 step2 prompt：independent KEEP/CHANGE scene analysis."""

    verdict = "KEEP" if memory.scene == gt_scene else "CHANGE"
    gt_scene_desc = SCENARIO_LABELS.get(gt_scene, gt_scene)

    if verdict == "KEEP":
        verdict_line = "VERDICT: KEEP -- the believed scene is correct."
        focus_line = "Explain why the current situation matches the believed scene."
        task_line = _teacher_structured_analysis_instructions(
            "explain why the believed scene fits under the current road structure."
        )
    else:
        verdict_line = "VERDICT: CHANGE -- the believed scene is wrong."
        focus_line = (
            f"Explain why the believed scene does not fit, then guide toward {gt_scene}: {gt_scene_desc}."
        )
        task_line = _teacher_structured_analysis_instructions(
            "explain what does not fit the believed scene and what visible cues support the correction."
        )

    return (
        f"{memory.format_step2_scene_text(gt_road_structure, gt_scene)}\n\n"
        f"{scene_choices_block_for(gt_road_structure, heading='ANSWER_ROAD_STRUCTURE')}\n\n"
        "[STEP2_TEACHER]\n"
        f"{verdict_line}\n"
        f"{focus_line}\n"
        f"{task_line}"
    )


def build_step2_teacher_target(analysis: str, gt_scene: str) -> str:
    """把 teacher step2 分析与 GT scene 拼成 student 的 teacher-forced target。"""

    cleaned = _clean_teacher_analysis(analysis)
    if not cleaned:
        cleaned = _fallback_teacher_analysis("scene")
    return f"{cleaned}\nSCENE: {gt_scene}".strip()


# ---------------------------------------------------------------------------
# Step 3：在 scene 事件序列中选 status / subgoal
# ---------------------------------------------------------------------------


def build_step3_student_prompt(memory: Memory) -> str:
    """学生 step3 prompt（除 layout 不再用 60-token 表述外，与 v3/v4 旧版同义）。"""

    return (
        f"{memory.format_text()}\n\n"
        f"[EVENT_OPTIONS]\n{_event_sequence_block(memory.scene)}\n[/EVENT_OPTIONS]\n\n"
        "[STEP3]\n"
        "Use the event options. Explain keep/correct for current phase. "
        "Then copy exactly two event names:\n"
        "STATUS: <event_name>\n"
        "SUBGOAL: <event_name>"
    )


def build_step3_teacher_prompt(
    memory: Memory,
    gt_road_structure: str,
    gt_scene: str,
    gt_status: str,
    gt_subgoal: str,
) -> str:
    """老师 step3 prompt：independent KEEP/CHANGE status/subgoal analysis."""

    status_keep = memory.status == gt_status
    subgoal_keep = memory.subgoal == gt_subgoal
    if status_keep and subgoal_keep:
        verdict = "KEEP"
        verdict_line = "VERDICT: KEEP -- believed status and subgoal are correct."
    else:
        verdict = "CHANGE"
        verdict_line = "VERDICT: CHANGE -- believed status/subgoal should be corrected."

    memory_status_desc = EVENT_DESCRIPTIONS.get(memory.status, memory.status)
    memory_subgoal_desc = EVENT_DESCRIPTIONS.get(memory.subgoal, memory.subgoal)
    gt_status_desc = EVENT_DESCRIPTIONS.get(gt_status, gt_status)
    gt_subgoal_desc = EVENT_DESCRIPTIONS.get(gt_subgoal, gt_subgoal)

    if verdict == "KEEP":
        focus_line = (
            f"Explain why the current phase '{memory_status_desc}' and next intent '{memory_subgoal_desc}' fit."
        )
        task_line = _teacher_structured_analysis_instructions(
            "explain why the believed status and believed subgoal fit under the current road and scene."
        )
    else:
        focus_line = (
            f"Reject current '{memory_status_desc}' / next '{memory_subgoal_desc}', "
            f"and guide toward current '{gt_status_desc}' / next '{gt_subgoal_desc}'."
        )
        task_line = _teacher_structured_analysis_instructions(
            "explain why the believed status/subgoal are wrong and what visible cues support the correction."
        )

    return (
        f"{memory.format_step3_event_text(gt_road_structure, gt_scene, gt_status, gt_subgoal)}\n\n"
        f"[EVENT_OPTIONS]\n{_event_sequence_block(gt_scene)}\n[/EVENT_OPTIONS]\n\n"
        "[STEP3_TEACHER]\n"
        f"{verdict_line}\n"
        f"{focus_line}\n"
        f"{task_line}"
    )


def build_step3_teacher_target(analysis: str, gt_status: str, gt_subgoal: str) -> str:
    """把 teacher step3 分析与 GT status/subgoal 拼成监督 target。"""

    cleaned = _clean_teacher_analysis(analysis)
    if not cleaned:
        cleaned = _fallback_teacher_analysis("event")
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


def _strip_label_lines(text: str) -> str:
    """删除分析文本中的所有 ``ROAD_STRUCTURE:`` / ``SCENE:`` / ``STATUS:`` / ``SUBGOAL:`` 行。"""

    if not text:
        return ""
    return _LABEL_LINE_RE.sub("", text).strip()


_PROMPT_MARKER_LINE_RE = re.compile(
    r"(?im)^\s*\[/?(?:STEP\d(?:_TEACHER)?|MEMORY|STEP1_ROAD_CONTEXT|ROAD_STRUCTURE_CHOICES|SCENE_CHOICES|EVENT_OPTIONS)[^\n]*\n?"
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


def _fallback_teacher_analysis(kind: str) -> str:
    """Build a minimal four-line fallback when teacher raw text is empty."""

    memory_line = {
        "road_structure": "Use road-layout evidence to judge whether the believed road structure should change.",
        "scene": "Use the road bucket and visual evidence to judge whether the believed scene should change.",
        "event": "Use the event sequence and visual evidence to judge whether the believed status or subgoal should change.",
    }.get(kind, "Use the visual evidence to judge whether the remembered state should change.")
    return "\n".join(
        [
            "Scene Description: The latest frames show the current road layout and traffic context.",
            "Critical Object Description: No reliable critical object is described by the teacher.",
            "Reasoning on Intent: Ego should use nearby actors, signals, and lane structure to choose a safe intent.",
            f"Memory Judgment: {memory_line}",
        ]
    )


def _clean_teacher_analysis(text: str) -> str:
    """Light cleanup before appending scripted labels.

    v4 是分 step 监督学习，最终结构化标签仍由脚本追加；analysis 文本必须保持学生
    可学视角，不能把老师私有 answer / ground-truth 字段逐字带进 target。

    1. 剥掉所有 ``ROAD_STRUCTURE:`` / ``SCENE:`` / ``STATUS:`` / ``SUBGOAL:`` 整行，
       防止脚本在末尾追加的 GT 标签被 parser 取错（parser 是 ``re.search`` 取第一个）；
    2. 剥掉 ``[STEPx]`` / ``[MEMORY]`` 等 prompt marker 行，避免老师把 prompt 模板抄进 target。
    3. 把 ``GROUND_TRUTH_*`` / ``ANSWER_*`` / ``REFERENCE_*`` 这类私有字段名改成
       ``the corrected ...`` 口径；把 ``BELIEVED_*`` / ``BELIEF_SUBGOAL`` 字段名改成
       自然语言 believed 口径。

    其余文字原样保留；剥完后空字符串才上调 ``_fallback_teacher_analysis``。
    """

    if not text:
        return ""
    cleaned = _strip_label_lines(text)
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
    """检查 scene 是否来自全局场景白名单。"""

    return bool(scene and scene in SCENARIO_LABELS)


def validate_event(scene: str, event: Optional[str]) -> bool:
    """检查 event 是否属于当前 scene 的合法事件序列。"""

    if not event:
        return False
    try:
        return event in get_full_sequence(scene)
    except Exception:
        return False


def next_event(scene: str, status: str) -> str:
    """返回某个 status 后面的下一事件；非法 status 时回退到第一子目标。"""

    seq = get_full_sequence(scene)
    try:
        idx = seq.index(status)
    except ValueError:
        return seq[1] if len(seq) > 1 else seq[0]
    return seq[idx + 1] if idx + 1 < len(seq) else seq[-1]
