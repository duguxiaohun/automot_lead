"""SFT v4 老师输出抽检脚本。

目的：随机采样若干条 episode 的若干帧，把"喂给老师的私有 prompt + 图像"、
"学生真实可见 prompt"、"adapter-enabled 学生初始自由输出"和"老师生成的监督目标"
分块打印出来，便于用户人工评估老师推理、学生 prompt 暴露面和初始策略行为是否合理。

老师路径与 collector 完全一致：``load_model_with_lora`` 拿到 PEFT bundle 后，全程
``model.disable_adapter()`` 绕开 LoRA delta，等价于纯 frozen base
Qwen3-VL-4B-Instruct，再叠加 ``torch.no_grad()`` 与 train.py 同款 KV cache 生成路径
不启用强制最少生成长度。

每帧默认覆盖 4 种常规 memory 起点；``scene_change_cross_rs`` 保留为显式 stress 模式：

| 模式                      | memory.road_structure | memory.scene | step1 | step2 | step3 |
|---------------------------|-----------------------|--------------|-------|-------|-------|
| ``all_keep``              | = GT                  | = GT         | KEEP  | KEEP  | KEEP  |
| ``rs_change``             | 非 GT 桶              | 错桶首个 scene | CHANGE| SKIP  | SKIP  |
| ``scene_change_same_rs``  | = GT                  | 同桶非 GT     | KEEP  | CHANGE| SKIP  |
| ``event_change``          | = GT                  | = GT         | KEEP  | KEEP  | CHANGE|
| ``scene_change_cross_rs`` | = GT                  | 跨桶非 GT     | KEEP  | CHANGE| SKIP  |

产物写到 ``--out-dir``：

- ``teacher_report.md``    人类可读，按 role / step 分块、含 token 统计与图像路径
- ``teacher_report.jsonl`` 机器可读，每行一帧 + 一种 memory 配置

运行示例（远端 AutoMoT/ 当前目录）：

    python qwen3vl_local/sft_v4/inspect_teacher.py \
        --train-jsonl checkpoints/sft_v4_data/train.jsonl \
        --model-dir checkpoints/Qwen3-VL-4B-Instruct \
        --out-dir checkpoints/sft_v4_inspect/run_$(date +%Y%m%d_%H%M%S) \
        --num-episodes 3 --frames-per-episode 4

GPU 选址沿用 sft_v2 项目约定（详见 CLAUDE.md / AGENTS.md §5）：

- **默认**：脚本顶部在 import torch 前先调用 ``_maybe_set_idle_gpu_mask``，
  自动用 ``nvidia-smi`` 挑 1 张最空闲 GPU 并设置 ``CUDA_VISIBLE_DEVICES``，
  同时**覆盖**调用方继承下来的 mask（避免误用上一次运行残留的 GPU 设置）。
- **显式 pin**：在命令前加 ``GPU_IDS=0`` （或 ``GPU_IDS=2`` 之类），
  自动选址会跳过、改用给定的物理卡号。
- **CPU 调试**：``--device cpu``，会绕过 GPU 选址（auto 才会触发自动挑卡）。
- **禁止**：命令里手写 ``export CUDA_VISIBLE_DEVICES=...``，与项目约定冲突。

脚本本身只用 1 张 GPU（单进程推理），不走 torchrun，也不进 DDP。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time
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

# 必须在 import torch 前完成自动选址：transformers / torch 在 import 阶段会
# 锁定可见 GPU 列表，import 之后再改 CUDA_VISIBLE_DEVICES 不再生效。
# 该 helper 内部：
#   1. 如果命令行带了 ``--device <cuda:id|cpu>``，尊重用户选择直接返回；
#   2. 否则如果环境变量带了 ``GPU_IDS=0`` / ``GPU_IDS=0,1``，按 GPU_IDS pin；
#   3. 否则用 ``nvidia-smi`` 找出最空闲 GPU 数 1 张，写入 CUDA_VISIBLE_DEVICES。
# 这与训练/eval/probe 入口完全同步，保证抽检脚本不会和其它进程抢同一张卡。
from qwen3vl_local.sft_v4.prompts import (  # noqa: E402
    CANONICAL_SCENARIO_LABELS,
    ROAD_STRUCTURE_TO_SCENES,
    SCENE_TO_ROAD_STRUCTURE,
    SYSTEM_PROMPT_V4,
    SYSTEM_PROMPT_STEP1,
    SYSTEM_PROMPT_STEP2,
    SYSTEM_PROMPT_STEP3,
    get_step_system_prompt,
    TEACHER_MAX_NEW_TOKENS_STEP1,
    TEACHER_MAX_NEW_TOKENS_STEP2,
    TEACHER_MAX_NEW_TOKENS_STEP3,
    Memory,
    build_step1_user_prompt,
    build_step1_teacher_prompt,
    build_step1_teacher_target,
    build_step2_student_prompt,
    build_step2_teacher_prompt,
    build_step2_teacher_target,
    build_step3_student_prompt,
    build_step3_teacher_prompt,
    build_step3_teacher_target,
    first_subgoal,
    first_scene_in_bucket,
    get_road_structure,
    get_full_sequence,
    initial_event,
    should_trigger_step2,
    should_trigger_step3,
    step1_teacher_verdict,
    step2_teacher_verdict,
    step3_teacher_verdict,
    update_memory_after_step1,
    update_memory_after_step2,
    update_memory_after_step3,
    parse_output,
)
torch = None  # type: ignore[assignment]


VERDICT_SOURCE_NOTE = (
    "The verdict fields are scripted expectations inferred from the fixed "
    "memory-vs-GT comparison for this inspect mode; they are not parsed from "
    "the teacher raw text."
)
INSPECT_MODE_ORDER: Tuple[str, ...] = (
    "all_keep",
    "rs_change",
    "scene_change_same_rs",
    "event_change",
    "scene_change_cross_rs",
)
DEFAULT_INSPECT_MODE_ORDER: Tuple[str, ...] = (
    "all_keep",
    "rs_change",
    "scene_change_same_rs",
    "event_change",
)
INSPECT_MODE_RANK = {name: idx for idx, name in enumerate(INSPECT_MODE_ORDER)}
INSPECT_MODE_SET = set(INSPECT_MODE_ORDER)


def _load_runtime_dependencies(device_arg: str) -> None:
    """Load torch/model helpers after prompt-only self-checks.

    GPU selection must still happen before importing torch. Keeping this as a
    lazy step prevents prompt contract checks from failing on machines where
    torch cannot initialize, while preserving the remote inspect runtime order.
    """

    global torch
    global load_model_with_lora
    global EpisodeDataset, EpisodeRow
    global _append_user_turn, _build_messages_with_images, _build_rgb_paths
    global _clone_kv_state, _gt_status_subgoal, _is_phase_a, _load_goal_xy
    global _load_images, _student_generate_kv, _student_start_state
    global _teacher_generate_kv, _teacher_start_state

    from qwen3vl_local.sft_v2.eval import _maybe_set_idle_gpu_mask  # noqa: WPS433

    if device_arg != "cpu":
        _maybe_set_idle_gpu_mask()

    import torch as _torch  # noqa: WPS433
    from qwen3vl_local.sft_v2.train import load_model_with_lora as _load_model_with_lora  # noqa: WPS433
    from qwen3vl_local.sft_v4.train import (  # noqa: WPS433
        EpisodeDataset as _EpisodeDataset,
        EpisodeRow as _EpisodeRow,
        _append_user_turn as _runtime_append_user_turn,
        _build_messages_with_images as _runtime_build_messages_with_images,
        _build_rgb_paths as _runtime_build_rgb_paths,
        _clone_kv_state as _runtime_clone_kv_state,
        _gt_status_subgoal as _runtime_gt_status_subgoal,
        _is_phase_a as _runtime_is_phase_a,
        _load_goal_xy as _runtime_load_goal_xy,
        _load_images as _runtime_load_images,
        _student_generate_kv as _runtime_student_generate_kv,
        _student_start_state as _runtime_student_start_state,
        _teacher_generate_kv as _runtime_teacher_generate_kv,
        _teacher_start_state as _runtime_teacher_start_state,
    )

    torch = _torch
    load_model_with_lora = _load_model_with_lora
    EpisodeDataset = _EpisodeDataset
    EpisodeRow = _EpisodeRow
    _append_user_turn = _runtime_append_user_turn
    _build_messages_with_images = _runtime_build_messages_with_images
    _build_rgb_paths = _runtime_build_rgb_paths
    _clone_kv_state = _runtime_clone_kv_state
    _gt_status_subgoal = _runtime_gt_status_subgoal
    _is_phase_a = _runtime_is_phase_a
    _load_goal_xy = _runtime_load_goal_xy
    _load_images = _runtime_load_images
    _student_generate_kv = _runtime_student_generate_kv
    _student_start_state = _runtime_student_start_state
    _teacher_generate_kv = _runtime_teacher_generate_kv
    _teacher_start_state = _runtime_teacher_start_state


def _assert_prompt_contracts() -> None:
    """启动前验证 inspect_teacher 依赖的 v4 prompt / trigger 契约。

    这不是训练测试，只是防止后续改 prompt 时把抽检脚本的含义悄悄改掉：
    step1 teacher prompt 只读 road-only context，不再喂完整 MEMORY；
    step1 KEEP 只表示 memory 不更新，并用 supported / not contradicted / corrected
    三段证据规则约束 road-structure 分析；
    step2/step3 trigger 必须分别等价于 layer-1 / layer-2 在本帧前后都稳定命中 GT。
    """

    gt_scene = sorted(CANONICAL_SCENARIO_LABELS)[0]
    gt_rs = get_road_structure(gt_scene)
    wrong_rs = next(rs for rs in sorted(ROAD_STRUCTURE_TO_SCENES) if rs != gt_rs)
    mem = Memory(
        road_structure=gt_rs,
        scene=gt_scene,
        status=initial_event(gt_scene),
        subgoal=first_subgoal(gt_scene),
        ego_to_goal_x=0.0,
        ego_to_goal_y=0.0,
    )
    step1_prompt = build_step1_teacher_prompt(mem, gt_rs)
    if "[STEP1_ROAD_CONTEXT]" not in step1_prompt or "[ROAD_STRUCTURE_CHOICES]" not in step1_prompt:
        raise AssertionError("step1 teacher prompt must include STEP1_ROAD_CONTEXT and ROAD_STRUCTURE_CHOICES")
    step1_student = build_step1_user_prompt(4, mem)
    if "[STEP1_ROAD_MEMORY]" not in step1_student:
        raise AssertionError("step1 student prompt must include STEP1_ROAD_MEMORY")
    # 统一 system prompt 契约：只含角色 + 通用证据原则，不含任何 step 专属域、
    # 输出格式或词数目标（全部下沉 user prompt）。三步共用同一段。
    _sys_required = (
        "You are an autonomous driving agent",
        "Use the 4 stitched RGB frames as visual context",
        "Keep the believed label by default",
        "not contradicted",
    )
    for _marker in _sys_required:
        if _marker not in SYSTEM_PROMPT_V4:
            raise AssertionError(f"SYSTEM_PROMPT_V4 must encode the shared contract: {_marker}")
    # 系统提示词必须干净：不含 task 声明、输出格式、词数目标、label 指令。
    _sys_forbidden = (
        "Your task is",
        "Output the four analysis lines",
        "Scene Description:",
        "aim for 100-150 words",
        "Then write",
        "ROAD_STRUCTURE",
        "SCENE",
        "STATUS",
        "SUBGOAL",
        "road-layout",
        "interaction cue",
        "temporal-progress",
        "ANSWER_",
        "GROUND_TRUTH_",
        "REFERENCE_",
    )
    for _token in _sys_forbidden:
        if _token in SYSTEM_PROMPT_V4.replace("Scene Description", ""):
            raise AssertionError(f"SYSTEM_PROMPT_V4 must not leak step-specific domain or format: found {_token!r}")
    # 三个别名必须指向同一段（向后兼容）。
    if SYSTEM_PROMPT_STEP1 is not SYSTEM_PROMPT_V4 or SYSTEM_PROMPT_STEP2 is not SYSTEM_PROMPT_V4 or SYSTEM_PROMPT_STEP3 is not SYSTEM_PROMPT_V4:
        raise AssertionError("SYSTEM_PROMPT_STEP1/2/3 must be aliases of SYSTEM_PROMPT_V4")
    if "VERDICT:" not in step1_prompt or 'Memory Judgment line MUST start with' not in step1_prompt:
        raise AssertionError("step1 teacher prompt must include the VERDICT + opener directive")
    if "Your reasoning must be consistent with this verdict" not in step1_prompt:
        raise AssertionError("step1 teacher prompt must enforce verdict-consistent reasoning")
    if "a single lead vehicle alone never proves HIGHWAY_MERGE" not in step1_student:
        raise AssertionError("step1 student prompt must reject lead vehicles as standalone merge evidence")
    if "a single lead vehicle alone never proves HIGHWAY_MERGE" not in step1_prompt:
        raise AssertionError("step1 teacher prompt must reject lead vehicles as standalone merge evidence")
    forbidden_step1_fields = ("BELIEVED_SCENE", "BELIEVED_STATUS", "BELIEVED_SUBGOAL", "ANSWER_")
    if any(token in step1_student for token in forbidden_step1_fields):
        raise AssertionError("step1 student prompt must be road-only and must not contain private answer fields")
    step2_student = build_step2_student_prompt(mem)
    step3_student = build_step3_student_prompt(mem)
    step2_teacher = build_step2_teacher_prompt(mem, gt_rs, gt_scene)
    step3_teacher = build_step3_teacher_prompt(mem, gt_rs, gt_scene, mem.status, mem.subgoal)
    for student_prompt in (step1_student, step2_student, step3_student):
        forbidden_private = ("ANSWER_", "GROUND_TRUTH_", "REFERENCE_")
        if any(token in student_prompt for token in forbidden_private):
            raise AssertionError("student-facing prompts must not contain teacher-private answer fields")
        if "Write four analysis lines (Scene Description" not in student_prompt:
            raise AssertionError("student-facing prompts must include the Write four analysis lines format instruction")
    expected_student_labels = {
        "step1": (step1_student, ("ROAD_STRUCTURE: <name>",)),
        "step2": (step2_student, ("SCENE: <scenario_name>",)),
        "step3": (step3_student, ("STATUS: <event_name>", "SUBGOAL: <event_name>")),
    }
    for name, (prompt, labels) in expected_student_labels.items():
        for label in labels:
            if label not in prompt:
                raise AssertionError(f"{name} student prompt missing label placeholder: {label}")
    for teacher_name, teacher_prompt in (
        ("step1", step1_prompt),
        ("step2", step2_teacher),
        ("step3", step3_teacher),
    ):
        if "Then write the label line(s) yourself" in teacher_prompt:
            raise AssertionError(f"{teacher_name} teacher prompt must not ask the teacher to write labels")
        if "do not write any label lines" not in teacher_prompt:
            raise AssertionError(f"{teacher_name} teacher prompt must explicitly forbid teacher label lines")
        forbidden_label_placeholders = (
            "ROAD_STRUCTURE: <name>",
            "SCENE: <scenario_name>",
            "STATUS: <event_name>",
            "SUBGOAL: <event_name>",
        )
        if any(label in teacher_prompt for label in forbidden_label_placeholders):
            raise AssertionError(f"{teacher_name} teacher prompt must not include label placeholders")
    if "do not infer hidden merging, yielding, lane change" not in step2_student:
        raise AssertionError("step2 student prompt must contain the anti-hallucination constraint for scene judgment")
    if "stationary or slow-moving" not in step2_teacher:
        raise AssertionError("step2 teacher prompt must reject weak speed-state inference")
    if "retained next objective" not in step3_teacher:
        raise AssertionError("step3 teacher prompt must distinguish STATUS from SUBGOAL")
    if not should_trigger_step2(
        memory_road_structure_before_step1=gt_rs,
        memory_road_structure_after_step1=gt_rs,
        gt_road_structure=gt_rs,
    ):
        raise AssertionError("should_trigger_step2 must fire when layer-1 is stably GT")
    if should_trigger_step2(
        memory_road_structure_before_step1=wrong_rs,
        memory_road_structure_after_step1=gt_rs,
        gt_road_structure=gt_rs,
    ):
        raise AssertionError("should_trigger_step2 must skip when layer-1 was just corrected")
    if should_trigger_step2(
        memory_road_structure_before_step1=wrong_rs,
        memory_road_structure_after_step1=wrong_rs,
        gt_road_structure=gt_rs,
    ):
        raise AssertionError("should_trigger_step2 must skip when layer-1 differs from GT")
    if not should_trigger_step3(
        memory_scene_before_step2=gt_scene,
        memory_scene_after_step2=gt_scene,
        gt_scene=gt_scene,
    ):
        raise AssertionError("should_trigger_step3 must fire when layer-2 is stably GT")
    other_scene = next(scene for scene in sorted(CANONICAL_SCENARIO_LABELS) if scene != gt_scene)
    if should_trigger_step3(
        memory_scene_before_step2=other_scene,
        memory_scene_after_step2=gt_scene,
        gt_scene=gt_scene,
    ):
        raise AssertionError("should_trigger_step3 must skip when layer-2 was just corrected")
    if should_trigger_step3(
        memory_scene_before_step2=other_scene,
        memory_scene_after_step2=other_scene,
        gt_scene=gt_scene,
    ):
        raise AssertionError("should_trigger_step3 must skip when layer-2 differs from GT")


# --------------------------- memory 构造 ---------------------------


def _other_event(scene: str, exclude: str) -> str:
    """在某个 scene 的事件序列中挑一个 ≠ exclude 的合法 event。

    用于 ``event_change`` 模式：故意把 memory.status/subgoal 设为合法但与 GT 不同的
    event，让老师必须主张 CHANGE。如果该 scene 事件序列退化到只有 1 个，则原样返回。
    """

    seq = get_full_sequence(scene)
    candidates = [e for e in seq if e != exclude]
    return candidates[0] if candidates else seq[0]


def _build_inspect_memories(
    *,
    ep: EpisodeRow,
    frame: int,
    rng: random.Random,
    goal_x: float,
    goal_y: float,
) -> List[Tuple[str, Memory]]:
    """对单帧构造全部可选 memory 起点，对应分层状态机分支。

    返回 ``(mode_name, Memory)`` 列表。``mode_name`` 用于报告里标注当前帧测的是哪条
    分支，方便用户筛选阅读。
    """

    gt_scene = ep.gt_scene
    gt_rs = get_road_structure(gt_scene)
    gt_status, gt_subgoal = _gt_status_subgoal(ep, frame)

    mem_all_keep = Memory(
        road_structure=gt_rs,
        scene=gt_scene,
        status=gt_status,
        subgoal=gt_subgoal,
        ego_to_goal_x=goal_x,
        ego_to_goal_y=goal_y,
    )

    wrong_rs_candidates = [rs for rs in sorted(ROAD_STRUCTURE_TO_SCENES) if rs != gt_rs]
    wrong_rs = rng.choice(wrong_rs_candidates) if wrong_rs_candidates else gt_rs
    wrong_rs_scene = first_scene_in_bucket(wrong_rs)
    mem_rs_change = Memory(
        road_structure=wrong_rs,
        scene=wrong_rs_scene,
        status=initial_event(wrong_rs_scene),
        subgoal=first_subgoal(wrong_rs_scene),
        ego_to_goal_x=goal_x,
        ego_to_goal_y=goal_y,
    )

    same_bucket_scenes = [s for s in ROAD_STRUCTURE_TO_SCENES.get(gt_rs, []) if s != gt_scene]
    same_rs_scene = rng.choice(same_bucket_scenes) if same_bucket_scenes else gt_scene
    mem_scene_change_same_rs = Memory(
        road_structure=gt_rs,
        scene=same_rs_scene,
        status=initial_event(same_rs_scene),
        subgoal=first_subgoal(same_rs_scene),
        ego_to_goal_x=goal_x,
        ego_to_goal_y=goal_y,
    )

    wrong_status = _other_event(gt_scene, gt_status)
    wrong_subgoal = _other_event(gt_scene, gt_subgoal)
    if wrong_status == gt_status and wrong_subgoal == gt_subgoal:
        # 事件序列只有 1 个时 wrong=gt，这种 scene 几乎不会出现，但为了语义清晰兜底：
        # 直接退化成 init 状态，至少与 gt 不同的概率最大。
        wrong_status = initial_event(gt_scene)
        wrong_subgoal = first_subgoal(gt_scene)
    mem_event_change = Memory(
        road_structure=gt_rs,
        scene=gt_scene,
        status=wrong_status,
        subgoal=wrong_subgoal,
        ego_to_goal_x=goal_x,
        ego_to_goal_y=goal_y,
    )

    cross_bucket_scenes = [
        s for s in sorted(CANONICAL_SCENARIO_LABELS)
        if s != gt_scene and SCENE_TO_ROAD_STRUCTURE.get(s) != gt_rs
    ]
    cross_rs_scene = rng.choice(cross_bucket_scenes) if cross_bucket_scenes else same_rs_scene
    mem_scene_change_cross_rs = Memory(
        road_structure=gt_rs,
        scene=cross_rs_scene,
        status=initial_event(cross_rs_scene),
        subgoal=first_subgoal(cross_rs_scene),
        ego_to_goal_x=goal_x,
        ego_to_goal_y=goal_y,
    )

    return [
        ("all_keep", mem_all_keep),
        ("rs_change", mem_rs_change),
        ("scene_change_same_rs", mem_scene_change_same_rs),
        ("event_change", mem_event_change),
        ("scene_change_cross_rs", mem_scene_change_cross_rs),
    ]


# --------------------------- 老师 rollout（单帧 × 单 memory）---------------------------


def _count_tokens(bundle: Any, text: str) -> int:
    """估算 token 数（仅用于报告显示，不参与训练 loss）。"""

    if not text:
        return 0
    ids = bundle.tokenizer(text, add_special_tokens=False)["input_ids"]
    return int(len(ids))


def _memory_as_dict(memory: Memory) -> Dict[str, Any]:
    """Render Memory as a compact JSON-friendly dict for reports."""

    return {
        "road_structure": memory.road_structure,
        "scene": memory.scene,
        "status": memory.status,
        "subgoal": memory.subgoal,
        "ego_to_goal_xy": [float(memory.ego_to_goal_x), float(memory.ego_to_goal_y)],
    }


def _student_output_record(bundle: Any, text: str) -> Dict[str, Any]:
    """Build a compact report record for a student free-generation output."""

    parsed = parse_output(text or "")
    return {
        "raw": text or "",
        "token_count": _count_tokens(bundle, text or ""),
        "parsed": parsed,
        "note": (
            "Adapter-enabled student free generation from the displayed student-facing prompt; "
            "this text is diagnostic only and is not the supervised target."
        ),
    }


def _empty_student_output() -> Dict[str, Any]:
    """Return the report shape used when student free generation is disabled/skipped."""

    return {
        "raw": "",
        "token_count": 0,
        "parsed": {"road_structure": None, "scene": None, "status": None, "subgoal": None},
        "note": "Student free generation was skipped for this step.",
    }


def _run_teacher_for_frame(
    *,
    bundle: Any,
    ep: EpisodeRow,
    frame: int,
    memory: Memory,
    mode_name: str,
    image_paths: List[str],
    generate_student_output: bool,
) -> Dict[str, Any]:
    """在单帧、单 memory 配置下跑老师完整三步，返回结构化结果。

    复用 ``train.py`` 的 KV cache 生成函数与新版老师 prompt，不启用强制最少生成长度。
    每个 step 都新建独立的 teacher 对话并重新吃图——脚本只关心老师"如果此刻进入这个
    memory 状态会怎么说"，不让 step1 的生成污染 step2/step3。
    """

    images = _load_images(image_paths)
    gt_road_structure = get_road_structure(ep.gt_scene)
    gt_status, gt_subgoal = _gt_status_subgoal(ep, frame)

    step1_user = build_step1_teacher_prompt(memory, gt_road_structure)
    step2_teacher_user = ""

    step1_msgs = _build_messages_with_images(user_text=step1_user, images=images, system_prompt=get_step_system_prompt("STEP1"))
    teacher_model = bundle.unwrap()

    raw_step1 = ""
    raw_step2 = ""
    raw_step3 = ""
    step3_user_prompt = ""
    step2_fired = False
    step3_fired = False
    step1_after_target = update_memory_after_step1(
        memory,
        student_road_structure=gt_road_structure,
    )
    step2_after_target = step1_after_target.copy()
    step3_after_target = step2_after_target.copy()

    teacher_was_training = bool(teacher_model.training)
    teacher_model.eval()
    try:
        with teacher_model.disable_adapter():
            # ---- step1：ROAD_STRUCTURE KEEP/CHANGE 推理，吃图 ----
            step1_state = _teacher_start_state(bundle, step1_msgs)
            raw_step1, _step1_state_after = _teacher_generate_kv(
                bundle,
                _clone_kv_state(step1_state),
                TEACHER_MAX_NEW_TOKENS_STEP1,
            )

            # ---- step2：场景 KEEP/CHANGE 推理 ----
            # inspect_teacher 是固定初始 memory 的分支探针：这里故意不 parse teacher
            # 输出、也不调用 update_memory_after_step1。step2/3 是否触发只由当前
            # mode 构造出的 memory-vs-GT 关系决定，便于稳定观察 5 条状态机路径。
            step2_fired = should_trigger_step2(
                memory_road_structure_before_step1=memory.road_structure,
                memory_road_structure_after_step1=step1_after_target.road_structure,
                gt_road_structure=gt_road_structure,
            )
            if step2_fired:
                step2_teacher_user = build_step2_teacher_prompt(step1_after_target, gt_road_structure, ep.gt_scene)
                step2_msgs = _build_messages_with_images(user_text=step2_teacher_user, images=images, system_prompt=get_step_system_prompt("STEP2"))
                step2_state = _teacher_start_state(bundle, step2_msgs)
                raw_step2, _step2_state_after = _teacher_generate_kv(
                    bundle,
                    _clone_kv_state(step2_state),
                    TEACHER_MAX_NEW_TOKENS_STEP2,
                )

                # ---- step3：触发口径与训练一致（step2 前后 scene 都稳定为 GT） ----
                step2_after_target = update_memory_after_step2(step1_after_target, student_scene=ep.gt_scene)
                step3_fired = should_trigger_step3(
                    memory_scene_before_step2=step1_after_target.scene,
                    memory_scene_after_step2=step2_after_target.scene,
                    gt_scene=ep.gt_scene,
                )
                if step3_fired:
                    step3_user_prompt = build_step3_teacher_prompt(
                        step2_after_target,
                        gt_road_structure,
                        ep.gt_scene,
                        gt_status,
                        gt_subgoal,
                    )
                    step3_msgs = _build_messages_with_images(user_text=step3_user_prompt, images=images, system_prompt=get_step_system_prompt("STEP3"))
                    step3_state = _teacher_start_state(bundle, step3_msgs)
                    raw_step3, _ = _teacher_generate_kv(
                        bundle,
                        _clone_kv_state(step3_state),
                        TEACHER_MAX_NEW_TOKENS_STEP3,
                    )
    finally:
        if teacher_was_training:
            teacher_model.train()

    step1_verdict = step1_teacher_verdict(memory, gt_road_structure)
    step2_verdict = step2_teacher_verdict(step1_after_target, ep.gt_scene) if step2_fired else "SKIPPED"
    step3_verdict = (
        step3_teacher_verdict(step2_after_target, ep.gt_scene, gt_status, gt_subgoal)
        if step3_fired
        else "SKIPPED"
    )
    target1 = build_step1_teacher_target(raw_step1, gt_road_structure, verdict=step1_verdict)
    target2 = (
        build_step2_teacher_target(raw_step2, ep.gt_scene, verdict=step2_verdict)
        if step2_fired
        else ""
    )
    target3 = (
        build_step3_teacher_target(raw_step3, gt_status, gt_subgoal, verdict=step3_verdict)
        if step3_fired
        else ""
    )

    step1_student_prompt = build_step1_user_prompt(len(images), memory)
    step2_student_prompt = build_step2_student_prompt(step1_after_target) if step2_fired else ""
    if not step2_fired:
        step2_after_target = step1_after_target.copy()
    step3_student_prompt = build_step3_student_prompt(step2_after_target) if step3_fired else ""
    step3_after_target = (
        update_memory_after_step3(step2_after_target, student_status=gt_status, student_subgoal=gt_subgoal)
        if step3_fired
        else step2_after_target.copy()
    )

    student_step1 = _empty_student_output()
    student_step2 = _empty_student_output()
    student_step3 = _empty_student_output()
    if generate_student_output:
        student_was_training = bool(teacher_model.training)
        teacher_model.eval()
        try:
            # The student branch intentionally keeps the adapter enabled. This mirrors the
            # collector's policy rollout path, but does not parse/update memory for this
            # fixed-branch teacher inspection report.
            step1_student_msgs = _build_messages_with_images(user_text=step1_student_prompt, images=images, system_prompt=get_step_system_prompt("STEP1"))
            student_step1_state = _student_start_state(bundle, step1_student_msgs)
            raw_student_step1, student_step1_state = _student_generate_kv(
                bundle,
                student_step1_state,
                TEACHER_MAX_NEW_TOKENS_STEP1,
            )
            student_step1 = _student_output_record(bundle, raw_student_step1)
            if step2_fired and step2_student_prompt:
                student_step2_state = _append_user_turn(bundle, student_step1_state, step2_student_prompt)
                raw_student_step2, student_step2_state = _student_generate_kv(
                    bundle,
                    student_step2_state,
                    TEACHER_MAX_NEW_TOKENS_STEP2,
                )
                student_step2 = _student_output_record(bundle, raw_student_step2)
                if step3_fired and step3_student_prompt:
                    student_step3_state = _append_user_turn(bundle, student_step2_state, step3_student_prompt)
                    raw_student_step3, _ = _student_generate_kv(
                        bundle,
                        student_step3_state,
                        TEACHER_MAX_NEW_TOKENS_STEP3,
                    )
                    student_step3 = _student_output_record(bundle, raw_student_step3)
        finally:
            if student_was_training:
                teacher_model.train()

    verdict_step1 = step1_verdict
    verdict_step2 = step2_verdict
    verdict_step3 = step3_verdict

    return {
        "run_id": ep.run_id,
        "scenario": ep.scenario,
        "frame_idx": int(frame),
        "phase": "A" if _is_phase_a(ep, frame) else "B",
        "mode": mode_name,
        "mode_order_index": int(INSPECT_MODE_RANK.get(mode_name, len(INSPECT_MODE_ORDER))),
        "image_paths": list(image_paths),
        "memory": _memory_as_dict(memory),
        "gt": {
            "road_structure": gt_road_structure,
            "scene": ep.gt_scene,
            "status": gt_status,
            "subgoal": gt_subgoal,
        },
        "step1": {
            "verdict": verdict_step1,
            "verdict_source": "scripted_memory_vs_gt",
            "verdict_note": VERDICT_SOURCE_NOTE,
            "uses_memory_block": "[MEMORY]" in step1_user,
            "system_prompt": get_step_system_prompt("STEP1"),
            "user_prompt": step1_user,
            "teacher_raw": raw_step1,
            "token_count": _count_tokens(bundle, raw_step1),
            "supervised_target": target1,
            "student_prompt": step1_student_prompt,
            "student_initial_output": student_step1,
            "student_memory_before": _memory_as_dict(memory),
            "student_memory_after_supervised_label": _memory_as_dict(step1_after_target),
            "student_update_note": (
                "If the student emits the supervised ROAD_STRUCTURE label, "
                "a changed road bucket resets scene/status/subgoal to the first scene in that bucket."
            ),
        },
        "step2": {
            "fired": bool(step2_fired),
            "verdict": verdict_step2,
            "verdict_source": "scripted_memory_vs_gt",
            "verdict_note": VERDICT_SOURCE_NOTE,
            "system_prompt": get_step_system_prompt("STEP2"),
            "user_prompt": step2_teacher_user,
            "teacher_raw": raw_step2,
            "token_count": _count_tokens(bundle, raw_step2),
            "supervised_target": target2,
            "student_prompt": step2_student_prompt,
            "student_initial_output": student_step2,
            "student_memory_before": _memory_as_dict(memory),
            "student_memory_after_supervised_label": _memory_as_dict(step2_after_target),
            "student_update_note": (
                "If the student emits the supervised SCENE label inside the current bucket, "
                "status/subgoal reset to that scene's init chain."
            ) if step2_fired else "Step2 is not part of this fixed inspect branch.",
        },
        "step3": {
            "fired": bool(step3_fired),
            "verdict": verdict_step3,
            "verdict_source": "scripted_memory_vs_gt",
            "verdict_note": VERDICT_SOURCE_NOTE,
            "system_prompt": get_step_system_prompt("STEP3"),
            "user_prompt": step3_user_prompt,
            "teacher_raw": raw_step3,
            "token_count": _count_tokens(bundle, raw_step3),
            "supervised_target": target3,
            "student_prompt": step3_student_prompt,
            "student_initial_output": student_step3,
            "student_memory_before": _memory_as_dict(memory),
            "student_memory_after_supervised_label": _memory_as_dict(step3_after_target),
            "student_update_note": (
                "If the student emits valid STATUS/SUBGOAL events for the current scene, "
                "only those event fields are updated."
            ) if step3_fired else "Step3 is not part of this fixed inspect branch.",
        },
    }


# --------------------------- 帧采样 ---------------------------


def _pick_episodes(ds: EpisodeDataset, *, num: int, rng: random.Random) -> List[EpisodeRow]:
    """从数据集中随机抽样 num 条 episode，frame 范围合法者优先。"""

    valid = [r for r in ds.rows if r.frame_end >= r.frame_start]
    if not valid:
        raise RuntimeError("dataset has no valid episode with frame_end >= frame_start")
    if num >= len(valid):
        return list(valid)
    return rng.sample(valid, num)


def _pick_frames(ep: EpisodeRow, *, num: int, rng: random.Random) -> List[int]:
    """在 episode 的合法 frame 范围内均匀采样 num 帧。

    采用"等步长 + 小抖动"采样而不是均匀随机，以便覆盖 Phase A 早期 / Phase A 末尾 /
    Phase B 中段 / Phase B 末尾 等不同时间窗口，避免抽样集中在某一段。
    """

    lo, hi = int(ep.frame_start), int(ep.frame_end)
    if hi < lo:
        return []
    span = hi - lo + 1
    if num >= span:
        return list(range(lo, hi + 1))
    step = max(1, span // num)
    base = [lo + i * step for i in range(num)]
    jitter = [min(hi, max(lo, b + rng.randint(-1, 1))) for b in base]
    return sorted(set(jitter))


# --------------------------- 报告写出 ---------------------------


def _format_md_block(role: str, body: str) -> str:
    """渲染单个 role 块为 Markdown，使用代码围栏避免提示词里特殊字符串干扰渲染。"""

    fence = "```"
    return f"**[ROLE = {role}]**\n{fence}\n{body.rstrip()}\n{fence}\n"


def _format_student_view(step: Dict[str, Any]) -> str:
    """Render the student-facing prompt/target and supervised memory transition."""

    if not step.get("student_prompt"):
        return ""
    before = json.dumps(step.get("student_memory_before", {}), ensure_ascii=False)
    after = json.dumps(step.get("student_memory_after_supervised_label", {}), ensure_ascii=False)
    student_initial = step.get("student_initial_output") or _empty_student_output()
    parsed = json.dumps(student_initial.get("parsed", {}), ensure_ascii=False)
    student_raw = (
        f"token_count={student_initial.get('token_count', 0)}\n"
        f"parsed={parsed}\n"
        f"note={student_initial.get('note', '')}\n\n"
        f"{student_initial.get('raw', '').rstrip()}"
    ).rstrip()
    transition = (
        f"memory_before={before}\n"
        f"memory_after_if_supervised_label={after}\n"
        f"note={step.get('student_update_note', '')}"
    )
    return (
        _format_md_block("STUDENT-FACING user prompt (no teacher-private answer fields)", step["student_prompt"])
        + _format_md_block("STUDENT INITIAL OUTPUT (adapter enabled, diagnostic only)", student_raw)
        + _format_md_block(
            "STUDENT SUPERVISED TARGET (teacher analysis cleaned + scripted label)",
            step.get("supervised_target", ""),
        )
        + _format_md_block("STUDENT MEMORY TRANSITION if supervised label is emitted", transition)
    )


def _write_markdown(report_rows: List[Dict[str, Any]], out_path: pathlib.Path) -> None:
    """把所有帧 × 模式的老师输出写成易读的 Markdown。

    布局：每条 episode 一个 H1，每帧一个 H2，每个 memory 模式一个 H3；模式内部按
    step1/step2/step3 拆 H4，并按 system / user / teacher-assistant / appended-target
    四个 role 分别用代码围栏展示。
    """

    by_episode: Dict[str, List[Dict[str, Any]]] = {}
    for row in report_rows:
        by_episode.setdefault(row["run_id"], []).append(row)

    lines: List[str] = []
    lines.append("# SFT v4 Teacher Inspection Report\n")
    lines.append(f"_Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}._\n")
    lines.append(
        "Each frame is replayed under selected memory configurations to exercise the "
        "ROAD_STRUCTURE / SCENE / EVENT KEEP and CHANGE branches. Token counts "
        "below come from the same tokenizer the trainer uses.\n"
    )
    lines.append(
        f"\nNote: {VERDICT_SOURCE_NOTE} Step 1 teacher turns use a road-only "
        "[STEP1_ROAD_CONTEXT] block instead of the full [MEMORY] block.\n"
    )
    lines.append(
        "Shared evidence rule: keep believed memory by default, change only when clear "
        "visible evidence contradicts it, and describe weak evidence as not contradicted rather "
        "than directly confirmed. The public target uses four lines: Scene Description, "
        "Critical Object Description, Reasoning on Intent, and Memory Judgment. A single distant "
        "lead vehicle alone is not enough to prove merging, cut-in, braking, or active flow.\n"
    )
    lines.append(
        "Mode sections are ordered by the state-machine path. Default modes are "
        f"{', '.join(DEFAULT_INSPECT_MODE_ORDER)}; scene_change_cross_rs is stress-only.\n"
    )
    lines.append(
        "Block naming convention: TEACHER-PRIVATE prompts may contain ANSWER_* fields "
        "and are never shown to the student; STUDENT-FACING prompts are the actual "
        "adapter-enabled policy prompts and should contain no private answer fields. "
        "STUDENT INITIAL OUTPUT is free generation for diagnosis, not the supervised target.\n"
    )

    for run_id, rows in by_episode.items():
        rows.sort(
            key=lambda r: (
                r["frame_idx"],
                int(r.get("mode_order_index", INSPECT_MODE_RANK.get(r["mode"], len(INSPECT_MODE_ORDER)))),
                r["mode"],
            )
        )
        first = rows[0]
        lines.append(f"\n# Episode `{run_id}`\n")
        lines.append(
            f"- scenario: `{first['scenario']}`\n"
            f"- gt_road_structure: `{first['gt']['road_structure']}`\n"
            f"- gt_scene: `{first['gt']['scene']}`\n"
        )

        last_frame: Optional[int] = None
        for row in rows:
            frame_idx = row["frame_idx"]
            if last_frame != frame_idx:
                lines.append(f"\n## Frame {frame_idx} (phase {row['phase']})\n")
                lines.append("Image paths fed to the teacher (oldest → newest):\n")
                for p in row["image_paths"]:
                    lines.append(f"- `{p}`\n")
                lines.append(
                    f"\nGT for this frame: road_structure=`{row['gt']['road_structure']}`, "
                    f"scene=`{row['gt']['scene']}`, "
                    f"status=`{row['gt']['status']}`, subgoal=`{row['gt']['subgoal']}`.\n"
                )
                last_frame = frame_idx

            lines.append(f"\n### Mode `{row['mode']}`\n")
            mem = row["memory"]
            lines.append(
                f"Memory fed into teacher: road_structure=`{mem['road_structure']}`, "
                f"scene=`{mem['scene']}`, "
                f"status=`{mem['status']}`, subgoal=`{mem['subgoal']}`, "
                f"ego_to_goal_xy=`{mem['ego_to_goal_xy']}`.\n"
            )

            # ---- step1 ----
            s1 = row["step1"]
            lines.append(
                f"\n#### Step 1 — road-structure verdict `{s1['verdict']}` "
                f"(source={s1.get('verdict_source', 'scripted_memory_vs_gt')}, "
                f"uses_memory={s1.get('uses_memory_block', False)}, "
                f"teacher tokens: {s1['token_count']})\n"
            )
            lines.append(_format_md_block("TEACHER-PRIVATE system prompt", s1["system_prompt"]))
            lines.append(_format_md_block("TEACHER-PRIVATE user prompt (with 4 stitched RGB images attached)", s1["user_prompt"]))
            lines.append(_format_md_block("TEACHER-PRIVATE assistant raw output", s1["teacher_raw"]))
            lines.append(_format_student_view(s1))

            # ---- step2 ----
            s2 = row["step2"]
            if s2.get("fired", True):
                lines.append(
                    f"\n#### Step 2 — scene verdict `{s2['verdict']}` "
                    f"(source={s2.get('verdict_source', 'scripted_memory_vs_gt')}, "
                    f"teacher tokens: {s2['token_count']})\n"
                )
                lines.append(_format_md_block("TEACHER-PRIVATE user prompt (fresh dialog with 4 images)", s2["user_prompt"]))
                lines.append(_format_md_block("TEACHER-PRIVATE assistant raw output", s2["teacher_raw"]))
                lines.append(_format_student_view(s2))
            else:
                lines.append(
                    "\n#### Step 2 — skipped (road_structure was not stably GT before/after step1, "
                    "so the trainer would not fire step2 here).\n"
                )

            # ---- step3 ----
            s3 = row["step3"]
            if s3["fired"]:
                lines.append(
                    f"\n#### Step 3 — status/subgoal verdict `{s3['verdict']}` "
                    f"(source={s3.get('verdict_source', 'scripted_memory_vs_gt')}, "
                    f"teacher tokens: {s3['token_count']})\n"
                )
                lines.append(_format_md_block("TEACHER-PRIVATE user prompt (fresh dialog with 4 images)", s3["user_prompt"]))
                lines.append(_format_md_block("TEACHER-PRIVATE assistant raw output", s3["teacher_raw"]))
                lines.append(_format_student_view(s3))
            else:
                lines.append(
                    "\n#### Step 3 — skipped (step2 did not fire or scene was not stably GT before/after step2, "
                    "so the trainer would not fire step3 here).\n"
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")


def _write_jsonl(report_rows: List[Dict[str, Any]], out_path: pathlib.Path) -> None:
    """把同一份数据写一份 jsonl，方便后续脚本/工具消费。"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in report_rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


# --------------------------- main ---------------------------


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    所有路径默认贴合白名单内训练入口的产物布局：episode jsonl 在
    ``checkpoints/sft_v4_data/train.jsonl``，base 模型在
    ``checkpoints/Qwen3-VL-4B-Instruct``。
    """

    p = argparse.ArgumentParser(description="SFT v4 teacher prompt/response inspection")
    p.add_argument("--train-jsonl", type=str, default="checkpoints/sft_v4_data/train.jsonl")
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--out-dir", type=str, default="checkpoints/sft_v4_inspect/latest")
    p.add_argument("--num-episodes", type=int, default=3, help="随机抽样的 episode 数")
    p.add_argument("--frames-per-episode", type=int, default=4, help="每条 episode 内抽样的帧数")
    p.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_INSPECT_MODE_ORDER),
        help="逗号分隔的 memory 模式列表；默认常规 4 种，scene_change_cross_rs 需显式指定",
    )
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--device", type=str, default="auto", help="cuda:0 / cpu / auto；auto 时由 _maybe_set_idle_gpu_mask 选址")
    p.add_argument(
        "--student-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在报告中额外跑 adapter-enabled student free generation；默认开启，可用 --no-student-output 加速",
    )
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-vision-scope", type=str, default="off")
    p.add_argument("--strict-vision-scope", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    """决定本进程使用的 torch.device。

    入参 ``device_arg`` 来自 ``--device``，默认 ``auto``。这里**只决定 torch.device
    名字**——物理 GPU 已经在 import torch 前被 ``_maybe_set_idle_gpu_mask`` 通过
    ``CUDA_VISIBLE_DEVICES`` 限定到一张可见卡。所以 auto 模式下返回 ``cuda:0``
    指的就是"那张被自动挑中的物理卡"，不会和别的进程抢卡。

    返回:
        - ``--device cpu``：torch.device('cpu')；
        - ``--device cuda:N``：torch.device('cuda:N')，N 是相对 CUDA_VISIBLE_DEVICES
          的本地编号；
        - ``--device auto``（默认）：有 CUDA 走 cuda:0，否则回退 cpu。
    """

    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def main() -> None:
    """脚本入口。

    1. 加载 episode jsonl 并随机挑 episode + frame；
    2. 加载 base Qwen + (一次性的、未训练的) LoRA wrapper；
    3. 对每帧每种 memory 模式跑老师完整 step1/2/3，记录所有 prompt/output；
    4. 写 ``teacher_report.md`` / ``teacher_report.jsonl``。
    """

    args = parse_args()
    _assert_prompt_contracts()
    _load_runtime_dependencies(args.device)
    rng = random.Random(args.seed)

    print(f"[inspect] loading episodes from {args.train_jsonl}", flush=True)
    ds = EpisodeDataset(pathlib.Path(args.train_jsonl))
    if len(ds.rows) == 0:
        raise ValueError(f"empty train jsonl: {args.train_jsonl}")

    selected_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not selected_modes:
        raise ValueError("--modes produced an empty mode list")
    bad = [m for m in selected_modes if m not in INSPECT_MODE_SET]
    if bad:
        raise ValueError(f"unknown --modes entries: {bad}; valid={list(INSPECT_MODE_ORDER)}")

    device = _resolve_device(args.device)
    print(f"[inspect] loading model from {args.model_dir} on {device}", flush=True)
    bundle = load_model_with_lora(
        pathlib.Path(args.model_dir),
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_vision_scope=args.lora_vision_scope,
        strict_vision_scope=bool(args.strict_vision_scope),
        gradient_checkpointing=False,
    )

    episodes = _pick_episodes(ds, num=int(args.num_episodes), rng=rng)
    print(
        f"[inspect] picked {len(episodes)} episodes; "
        f"frames/episode={args.frames_per_episode}; modes={selected_modes}",
        flush=True,
    )

    report_rows: List[Dict[str, Any]] = []
    for ep in episodes:
        run_dir = pathlib.Path(ep.run_dir)
        frames = _pick_frames(ep, num=int(args.frames_per_episode), rng=rng)
        if not frames:
            print(f"[inspect][warn] episode {ep.run_id} has no valid frames", flush=True)
            continue
        for frame in frames:
            try:
                goal_x, goal_y = _load_goal_xy(run_dir, frame)
            except Exception as exc:
                print(f"[inspect][warn] load_goal_xy failed run={ep.run_id} frame={frame}: {exc}", flush=True)
                continue
            try:
                image_paths = _build_rgb_paths(run_dir, frame)
            except Exception as exc:
                print(f"[inspect][warn] rgb path build failed run={ep.run_id} frame={frame}: {exc}", flush=True)
                continue

            mem_configs = _build_inspect_memories(
                ep=ep,
                frame=frame,
                rng=rng,
                goal_x=goal_x,
                goal_y=goal_y,
            )
            for mode_name, memory in mem_configs:
                if mode_name not in selected_modes:
                    continue
                start = time.time()
                try:
                    record = _run_teacher_for_frame(
                        bundle=bundle,
                        ep=ep,
                        frame=frame,
                        memory=memory,
                        mode_name=mode_name,
                        image_paths=image_paths,
                        generate_student_output=bool(args.student_output),
                    )
                except Exception as exc:
                    print(
                        f"[inspect][error] teacher rollout failed run={ep.run_id} frame={frame} mode={mode_name}: {exc}",
                        flush=True,
                    )
                    continue
                elapsed = time.time() - start
                record["elapsed_sec"] = float(elapsed)
                report_rows.append(record)
                s2_tokens = record["step2"]["token_count"]
                s3_tokens = record["step3"]["token_count"] if record["step3"]["fired"] else "-"
                print(
                    f"[inspect] run={ep.run_id} frame={frame} mode={mode_name} "
                    f"step1_tok={record['step1']['token_count']} step2_tok={s2_tokens} "
                    f"step3_tok={s3_tokens} ({elapsed:.1f}s)",
                    flush=True,
                )

    if not report_rows:
        print("[inspect] no records produced; check warnings above", flush=True)
        return

    out_dir = pathlib.Path(args.out_dir)
    md_path = out_dir / "teacher_report.md"
    jsonl_path = out_dir / "teacher_report.jsonl"
    _write_markdown(report_rows, md_path)
    _write_jsonl(report_rows, jsonl_path)
    print(f"[inspect] wrote {md_path}", flush=True)
    print(f"[inspect] wrote {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
