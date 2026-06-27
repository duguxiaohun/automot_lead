"""SFT v4 off-policy collector。

collector 负责最慢的 rollout：teacher/student generate、memory 自更新、Phase B 噪声扰动。
它不进入 DDP，也不做 backward；产物是一条 episode 一个 trajectory jsonl，写入
``replay/ready`` 供 learner 随机抽样。

运行方式：

``launch_offpolicy.sh`` 会在 collector GPU 上启动多个本脚本进程，每个进程通过
``--collector-id`` 区分日志和 trajectory 文件名。collector 启动后先等待 learner
发布 ``latest_lora/v_0``，之后按 ``refresh_every_*`` 周期加载更新的 LoRA snapshot。

重要边界：

- collector 只做 ``torch.no_grad`` 推理，不产生 optimizer / scheduler 状态；
- teacher 分支通过 ``disable_adapter`` 使用 base Qwen，student 分支使用当前 LoRA；
- trajectory 里保存 teacher target 和 student raw output，不保存 KV cache 或图像本体；
- 任何 collector 掉线只减少生产速度，不会影响 learner DDP process group。
"""

from __future__ import annotations

import argparse
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

import torch

from qwen3vl_local.sft_v4 import replay
from qwen3vl_local.sft_v4.prompts import (
    DEFAULT_SKIP_CORRECTION_SCENE_NOISE_PROB,
    TEACHER_MAX_NEW_TOKENS_STEP1,
    TEACHER_MAX_NEW_TOKENS_STEP2,
    TEACHER_MAX_NEW_TOKENS_STEP3,
    build_step1_teacher_prompt,
    build_step1_teacher_target,
    build_step1_user_prompt,
    build_step2_student_prompt,
    build_step2_teacher_prompt,
    build_step2_teacher_target,
    build_step3_student_prompt,
    build_step3_teacher_prompt,
    build_step3_teacher_target,
    correct_memory_after_step1_skip,
    force_memory_to_gt_chain,
    get_road_structure,
    init_memory,
    inject_phase_b_noise,
    parse_output,
    should_trigger_step2,
    should_trigger_step3,
    update_memory_after_step1,
    update_memory_after_step2,
    update_memory_after_step3,
    validate_event,
    _fallback_teacher_analysis,
)
from qwen3vl_local.sft_v4.train import (
    EpisodeDataset,
    EpisodeRow,
    _analysis_before_labels,
    _build_messages_with_images,
    _build_rgb_paths,
    _gt_status_subgoal,
    _is_phase_a,
    _load_goal_xy,
    _load_images,
    _prefetch_goal_xy_for_next_frame,
    _student_generate_kv,
    _student_start_state,
    _teacher_generate_kv,
    _teacher_start_state,
    _append_user_turn,
)
from qwen3vl_local.sft_v2.train import load_model_with_lora


def _memory_to_dict(memory: Any) -> Dict[str, Any]:
    """把 Memory dataclass 转为可写入 jsonl 的稳定结构。

    trajectory 是跨进程/跨版本交换格式，不能直接 pickle dataclass；显式展开字段后，
    learner 可以在完全独立的 Python 进程里用 ``_memory_from_record`` 还原。

    v4 schema_v2：增加 ``road_structure``（layer-1）字段；旧 v1 trajectory 不带，
    learner 加载时按 ``replay.SCHEMA`` 校验，v1 traj 会被 reject 而不是默认填值。
    """

    return {
        "road_structure": str(getattr(memory, "road_structure", "JUNCTION")),
        "scene": str(memory.scene),
        "status": str(memory.status),
        "subgoal": str(memory.subgoal),
        "ego_to_goal_xy": [float(memory.ego_to_goal_x), float(memory.ego_to_goal_y)],
    }


def _load_adapter_state_if_present(bundle: Any, adapter_dir: pathlib.Path) -> bool:
    """把 learner 发布的 adapter snapshot 加载进当前 collector 模型。

    PEFT 的 ``save_pretrained`` 可能保存 safetensors，也可能保存 bin；两个路径都兼容。
    函数返回 bool 而不是抛错，是为了允许 collector 在 learner 刚更新 pointer、目录还在
    原子切换边界时下次再试。
    """

    adapter_dir = pathlib.Path(adapter_dir)
    if not adapter_dir.exists():
        return False
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if not safetensors_path.exists() and not bin_path.exists():
        return False
    from peft import set_peft_model_state_dict

    if safetensors_path.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_path), device=str(bundle.device))
    else:
        state = torch.load(str(bin_path), map_location=bundle.device)
    set_peft_model_state_dict(bundle.unwrap(), state)
    return True


def _current_snapshot_dir(latest_lora_dir: pathlib.Path) -> Tuple[int, Optional[pathlib.Path]]:
    """读取 latest_lora/current_version.txt，返回版本号和目录。

    ``current_version.txt`` 是 learner rank0 发布 snapshot 的唯一指针。collector 不扫描
    全部 ``v_*`` 目录，避免刚写到一半的临时目录被误加载。
    """

    pointer = pathlib.Path(latest_lora_dir) / "current_version.txt"
    if not pointer.exists():
        return -1, None
    text = pointer.read_text(encoding="utf-8").strip()
    if not text:
        return -1, None
    try:
        version = int(text)
    except ValueError:
        return -1, None
    return version, pathlib.Path(latest_lora_dir) / f"v_{version}"


def _maybe_refresh_snapshot(bundle: Any, latest_lora_dir: pathlib.Path, loaded_version: int) -> int:
    """如果 learner 发布了新 snapshot，则加载它并返回新版本号。

    ``loaded_version`` 是本 collector 已加载的策略版本。只有 pointer 变大才加载，避免
    每条 episode 重复读同一份 adapter 文件。
    """

    version, path = _current_snapshot_dir(latest_lora_dir)
    if path is None or version <= loaded_version:
        return loaded_version
    if _load_adapter_state_if_present(bundle, path):
        print(f"[collect] loaded snapshot version={version} path={path}", flush=True)
        return version
    return loaded_version


def _inject_phase_b_noise(memory: Any, *, gt_scene: str, rng: random.Random, prob: float) -> Tuple[Any, bool]:
    """Phase B 弱纠偏后按概率注入随机非 GT scene。

    Phase B 正常会把 scene 拉回 GT，训练“对的别改”。这里再用小概率把 scene 扰到非 GT，
    让模型在后半段也能见到“突然错记忆 → 重新纠正”的样本。status/subgoal 跟着新 scene
    重置，保持 memory 内部一致。
    """

    return inject_phase_b_noise(memory, gt_scene=gt_scene, rng=rng, prob=prob)


def collect_episode(
    bundle: Any,
    ep: EpisodeRow,
    *,
    collector_id: str,
    policy_version: int,
    p_init_correct: float,
    phase_b_noise_prob: float,
    skip_correction_scene_noise_prob: float,
    outer_stride: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """执行一条 episode rollout，并返回 trajectory records。

    输出 records 的第 0 行是 header，后面每一行是一帧。collector 在这里完成所有自由
    生成和 memory 推进，learner 后续只按这些记录复现 prompt state 并计算 CE loss。

    参数说明：

    - ``policy_version``：当前 collector 使用的 LoRA snapshot 版本，写入 header 供审计；
    - ``p_init_correct``：Phase A 初始 road_structure 命中 GT 桶的概率，默认 0.7；
    - ``phase_b_noise_prob``：Phase B 弱纠偏后注入随机非 GT scene 的概率；
    - ``skip_correction_scene_noise_prob``：上一帧跳过 step2/3 后，下一帧帧首纠偏时
      同桶非 GT scene 小扰动概率；
    - ``outer_stride``：调试用跳帧，生产默认 1，不改变 episode index 本身。
    """

    run_dir = pathlib.Path(ep.run_dir)
    stride = max(1, int(outer_stride))
    rng = random.Random(f"{seed}:{collector_id}:{ep.run_id}:{policy_version}")
    gx, gy = _load_goal_xy(run_dir, ep.frame_start)
    # Phase A 初始 memory 是 v4 的关键变化：同一 episode 在不同 collector / policy
    # version 下会得到可复现但不同的扰动，提升 replay 多样性。
    memory = init_memory(
        run_id=ep.run_id,
        sub_scenario_id=f"{ep.run_id}:{ep.anchors[1]}",
        ego_to_goal_x=gx,
        ego_to_goal_y=gy,
        gt_scene=ep.gt_scene,
        p_init_correct=p_init_correct,
        seed_salt=f"{collector_id}:{policy_version}",
    )

    records: List[Dict[str, Any]] = [
        {
            "schema": replay.SCHEMA,
            "kind": "header",
            "collector_id": collector_id,
            "policy_version": int(policy_version),
            # replay_stats / FIFO 驱逐都用这个时间，而不是 ready 文件 mtime。
            "created_at": time.time(),
            "frame_count": 0,
            "episode": {
                "run_id": ep.run_id,
                "scenario": ep.scenario,
                "anchors": [int(x) for x in ep.anchors],
                "delta": int(ep.delta),
                "frame_range": [int(ep.frame_start), int(ep.frame_end)],
                "gt_scene": ep.gt_scene,
                "gt_event_sequence": list(ep.gt_event_sequence),
                "run_dir": ep.run_dir,
            },
        }
    ]

    gt_road_structure = get_road_structure(ep.gt_scene)
    first_frame = True
    need_skip_correction = False
    for frame in range(ep.frame_start, ep.frame_end + 1, stride):
        phase_a = _is_phase_a(ep, frame)
        noise_injected = False
        skip_correction_applied = False
        skip_correction_scene_noisy = False
        road_structure_reset_this_frame = False
        scene_reset_this_frame = False
        if not phase_a:
            # Phase B 帧首三层弱纠偏（D23 + D27 配套）：先 layer-1 回 GT 桶，
            # 再 scene 回 GT；noise 只在 scene 这一层按 PHASE_B_NOISE_PROB 注入。
            before_force_rs = memory.road_structure
            before_force_scene = memory.scene
            memory = force_memory_to_gt_chain(
                memory, gt_road_structure=gt_road_structure, gt_scene=ep.gt_scene
            )
            road_structure_reset_this_frame = (
                road_structure_reset_this_frame
                or memory.road_structure != before_force_rs
            )
            scene_reset_this_frame = (
                scene_reset_this_frame
                or memory.scene != before_force_scene
            )
            memory, noise_injected = _inject_phase_b_noise(
                memory,
                gt_scene=ep.gt_scene,
                rng=rng,
                prob=phase_b_noise_prob,
            )
            scene_reset_this_frame = scene_reset_this_frame or bool(noise_injected)
        if need_skip_correction:
            # Previous frame failed layer-1 and skipped step2/3. Repair once
            # before this frame's inner loop so downstream choices are fetched
            # from the corrected bucket; STATUS/SUBGOAL restart at init.
            memory, skip_correction_scene_noisy = correct_memory_after_step1_skip(
                memory,
                gt_scene=ep.gt_scene,
                rng=rng,
                scene_noise_prob=skip_correction_scene_noise_prob,
            )
            road_structure_reset_this_frame = True
            scene_reset_this_frame = True
            skip_correction_applied = True
            need_skip_correction = False
        memory_before = memory.copy()
        image_paths = _build_rgb_paths(run_dir, frame)
        images = _load_images(image_paths)
        gt_status, gt_subgoal = _gt_status_subgoal(ep, frame)

        # ============ Step 1：视觉描述 + ROAD_STRUCTURE 判定 ============
        # 与 train/eval/probe/learner 共用 prompts.py：step1 学生只读 road-only
        # memory，step2/3 才读完整 [MEMORY]。
        step1_student_user = build_step1_user_prompt(len(images), memory=memory_before)
        step1_teacher_user_text = build_step1_teacher_prompt(memory_before, gt_road_structure)
        step1_msgs_student = _build_messages_with_images(user_text=step1_student_user, images=images)
        step1_msgs_teacher = _build_messages_with_images(user_text=step1_teacher_user_text, images=images)

        # Teacher 分支：disable_adapter = 纯 frozen base Qwen + no_repeat_ngram + 软 min。
        model = bundle.unwrap()
        was_training = bool(model.training)
        model.eval()
        with model.disable_adapter():
            teacher_step1_prompt_state = _teacher_start_state(bundle, step1_msgs_teacher)
            raw_teacher_step1, teacher_step1_state = _teacher_generate_kv(
                bundle,
                teacher_step1_prompt_state,
                TEACHER_MAX_NEW_TOKENS_STEP1,
            )
            raw_teacher_step1 = raw_teacher_step1 or _fallback_teacher_analysis("road_structure")
        if was_training:
            model.train()

        # Student 分支：当前 LoRA snapshot 自由生成，用于 memory 推进 + learner 重放。
        bundle.model.eval()
        student_step1_prompt_state = _student_start_state(bundle, step1_msgs_student)
        student_step1, student_step1_state = _student_generate_kv(
            bundle, student_step1_prompt_state, TEACHER_MAX_NEW_TOKENS_STEP1
        )
        student_step1 = student_step1 or raw_teacher_step1

        # ============ Step 1 后处理：更新 memory.road_structure ============
        analysis1 = _analysis_before_labels(raw_teacher_step1)
        target1 = build_step1_teacher_target(analysis1, gt_road_structure)
        pred1 = parse_output(student_step1)
        old_rs = memory.road_structure
        memory = update_memory_after_step1(memory, student_road_structure=pred1.get("road_structure"))
        rs_flip = memory.road_structure != old_rs
        memory_after_step1 = memory.copy()

        # ============ 触发门 1：road bucket 前后都稳定为 GT 才进 step2 ============
        step2_ran = should_trigger_step2(
            memory_road_structure_before_step1=old_rs,
            memory_road_structure_after_step1=memory.road_structure,
            gt_road_structure=gt_road_structure,
            road_structure_reset_this_frame=road_structure_reset_this_frame,
        )
        analysis2 = ""
        target2 = ""
        scene_flip = False
        memory_after_step2 = memory.copy()
        raw_teacher_step2 = ""
        raw_student_step2 = ""
        student_step2_state = None
        if step2_ran:
            # Step 2 的时序是 v4 最重要的不变量之一：
            #
            # 1. 先让 student step1 自由生成 ROAD_STRUCTURE；
            # 2. 用这个输出更新 memory.road_structure，并在翻桶时 reset 下游 scene/status/subgoal；
            # 3. 只有 step1 前后 road_structure 都已稳定命中 GT 桶，才构造 step2 prompt；
            #    若本帧刚纠正 layer-1，则下一帧再训练 scene。
            #
            # 不能把 step2_teacher_user / step2_student_user 提前放到 step1 前面构造。
            # 否则会出现“step1 已经把桶从旧值纠正到 GT，但 step2 仍列旧桶
            # SCENE_CHOICES”的隐性错配；trajectory 看起来合法，learner 却会学到
            # target 不在候选表里的坏样本。
            step2_teacher_user = build_step2_teacher_prompt(memory, gt_road_structure, ep.gt_scene)
            step2_student_user = build_step2_student_prompt(memory)
            step2_msgs_teacher = _build_messages_with_images(user_text=step2_teacher_user, images=images)

            was_training = bool(model.training)
            model.eval()
            with model.disable_adapter():
                teacher_step2_prompt_state = _teacher_start_state(bundle, step2_msgs_teacher)
                raw_teacher_step2, _teacher_step2_state = _teacher_generate_kv(
                    bundle,
                    teacher_step2_prompt_state,
                    TEACHER_MAX_NEW_TOKENS_STEP2,
                )
            if was_training:
                model.train()

            bundle.model.eval()
            student_step2_prompt_state = _append_user_turn(bundle, student_step1_state, step2_student_user)
            raw_student_step2, student_step2_state = _student_generate_kv(
                bundle, student_step2_prompt_state, TEACHER_MAX_NEW_TOKENS_STEP2
            )

            analysis2 = _analysis_before_labels(raw_teacher_step2)
            target2 = build_step2_teacher_target(analysis2, ep.gt_scene)
            pred2 = parse_output(raw_student_step2)
            old_scene = memory.scene
            memory = update_memory_after_step2(memory, student_scene=pred2.get("scene"))
            scene_flip = memory.scene != old_scene
            memory_after_step2 = memory.copy()

        # ============ 触发门 2：scene 前后都稳定为 GT 才进 step3 ============
        step3_ran = (
            step2_ran
            and should_trigger_step3(
                memory_scene_before_step2=old_scene,
                memory_scene_after_step2=memory.scene,
                gt_scene=ep.gt_scene,
                scene_reset_this_frame=scene_reset_this_frame,
            )
        )
        raw_teacher_step3 = ""
        raw_student_step3 = ""
        target3 = ""
        if step3_ran:
            assert student_step2_state is not None
            step3_teacher_user = build_step3_teacher_prompt(
                memory,
                gt_road_structure,
                ep.gt_scene,
                gt_status,
                gt_subgoal,
            )
            step3_msgs_teacher = _build_messages_with_images(user_text=step3_teacher_user, images=images)
            was_training = bool(model.training)
            model.eval()
            with model.disable_adapter():
                teacher_step3_prompt_state = _teacher_start_state(bundle, step3_msgs_teacher)
                raw_teacher_step3, _ = _teacher_generate_kv(
                    bundle,
                    teacher_step3_prompt_state,
                    TEACHER_MAX_NEW_TOKENS_STEP3,
                )
            if was_training:
                model.train()
            analysis3 = _analysis_before_labels(raw_teacher_step3)
            target3 = build_step3_teacher_target(analysis3, gt_status, gt_subgoal)

            # student step3 输出决定帧末 memory 的 status/subgoal。非法 event 不写入 memory，
            # 这样 learner 看到的下一帧 memory 与 collector 真实 rollout 完全一致。
            step3_student_user = build_step3_student_prompt(memory)
            student_step3_prompt_state = _append_user_turn(bundle, student_step2_state, step3_student_user)
            raw_student_step3, _ = _student_generate_kv(
                bundle, student_step3_prompt_state, TEACHER_MAX_NEW_TOKENS_STEP3
            )
            pred3 = parse_output(raw_student_step3)
            pred_status = pred3.get("status") if validate_event(memory.scene, pred3.get("status")) else None
            pred_subgoal = pred3.get("subgoal") if validate_event(memory.scene, pred3.get("subgoal")) else None
            memory = update_memory_after_step3(
                memory,
                student_status=pred_status,
                student_subgoal=pred_subgoal,
            )

        teacher_step2_raw_value: Optional[str] = raw_teacher_step2 if step2_ran else None
        teacher_step2_target_value: Optional[str] = target2 if step2_ran else None
        student_step2_value: Optional[str] = raw_student_step2 if step2_ran else None
        teacher_step3_raw_value: Optional[str] = raw_teacher_step3 if step3_ran else None
        teacher_step3_target_value: Optional[str] = target3 if step3_ran else None
        student_step3_value: Optional[str] = raw_student_step3 if step3_ran else None
        frame_record = {
            "kind": "frame",
            "frame_idx": int(frame),
            "phase": "A" if phase_a else "B",
            "image_paths": image_paths,
            # PLAN §0.5：trajectory schema_v2 显式字段（v1 → v2 bump 后旧 trajectory
            # 会被 replay 校验拒收）。所有 memory 快照都含 road_structure。
            "memory_before_frame": _memory_to_dict(memory_before),
            # learner 重放 step2 时必须使用这份快照，而不是 frame 初始 memory。
            # 这份快照记录了 student step1 之后的真实 bucket，是 v2 schema 相比 v1
            # 新增的关键字段。
            "memory_after_step1": _memory_to_dict(memory_after_step1),
            "init_was_correct": (
                memory_before.scene == ep.gt_scene
                and memory_before.road_structure == gt_road_structure
            ) if first_frame else None,
            "noise_injected": bool(noise_injected),
            "skip_correction_applied": bool(skip_correction_applied),
            "skip_correction_scene_noisy": bool(skip_correction_scene_noisy),
            # 三步老师/学生文本：未触发的步骤填 None 而非空串，方便 learner 区分。
            "teacher_step1_text": raw_teacher_step1,
            "teacher_step1_target": target1,
            "teacher_step2_raw": teacher_step2_raw_value,
            "teacher_step2_target": teacher_step2_target_value,
            "teacher_step3_raw": teacher_step3_raw_value,
            "teacher_step3_target": teacher_step3_target_value,
            "student_step1_raw": student_step1,
            "student_step2_raw": student_step2_value,
            "student_step3_raw": student_step3_value,
            "step2_fired": bool(step2_ran),
            "step3_fired": bool(step3_ran),
            "memory_before": _memory_to_dict(memory_before),
            "memory_after_step2": _memory_to_dict(memory_after_step2),
            "memory_after_frame": _memory_to_dict(memory),
            "gt": {
                "road_structure": gt_road_structure,
                "scene": ep.gt_scene,
                "status": gt_status,
                "subgoal": gt_subgoal,
            },
            "student_outputs": {
                "step1": student_step1,
                "step2": raw_student_step2 if step2_ran else None,
                "step3": raw_student_step3 if step3_ran else None,
            },
            "teacher_targets": {
                "step1": target1,
                "step2": target2 if step2_ran else None,
                "step3": target3 if step3_ran else None,
            },
            "teacher_raw_outputs": {
                "step1": raw_teacher_step1,
                "step2": raw_teacher_step2 if step2_ran else None,
                "step3": raw_teacher_step3 if step3_ran else None,
            },
            "flags": {
                # flags 全部是训练审计字段：learner 不依赖它们决定 prompt 文本，但会用来
                # 统计 step1/step2/step3 触发率、翻转率，定位行为偏移。
                "step2_ran": bool(step2_ran),
                "step3_ran": bool(step3_ran),
                "rs_flip": bool(rs_flip),
                "scene_flip": bool(scene_flip),
                "phase_a": bool(phase_a),
                "noise_injected": bool(noise_injected),
                "skip_correction_applied": bool(skip_correction_applied),
                "skip_correction_scene_noisy": bool(skip_correction_scene_noisy),
            },
        }
        frame_record["step2_ran"] = bool(step2_ran)
        frame_record["step3_ran"] = bool(step3_ran)
        records.append(frame_record)
        need_skip_correction = memory.road_structure != gt_road_structure
        first_frame = False
        # 帧末预取下一帧 goal 坐标，保证下一轮 memory_before 已经是下一帧视角。
        _prefetch_goal_xy_for_next_frame(memory, run_dir, frame + stride, ep.frame_end)

    records[0]["frame_count"] = len(records) - 1
    replay.validate_trajectory(records)
    return records


def parse_args() -> argparse.Namespace:
    """解析 collector 参数。

    大部分参数由 ``launch_offpolicy.sh`` 传入；手工单进程调试时至少需要指定
    ``--train-jsonl``、``--model-dir``、``--replay-dir`` 和 ``--latest-lora-dir``。
    """

    p = argparse.ArgumentParser(description="Collect SFT v4 off-policy trajectories")
    p.add_argument("--train-jsonl", type=str, default="checkpoints/sft_v4_data/train.jsonl")
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--replay-dir", type=str, default="checkpoints/sft_v4_lora/latest/replay")
    p.add_argument("--latest-lora-dir", type=str, default="checkpoints/sft_v4_lora/latest/latest_lora")
    p.add_argument("--collector-id", type=str, default="collector0")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--outer-stride", type=int, default=1)
    p.add_argument("--replay-capacity", type=int, default=256)
    p.add_argument("--p-init-correct", type=float, default=0.7)
    p.add_argument("--phase-b-noise-prob", type=float, default=0.15)
    p.add_argument(
        "--skip-correction-scene-noise-prob",
        type=float,
        default=DEFAULT_SKIP_CORRECTION_SCENE_NOISE_PROB,
    )
    p.add_argument("--refresh-every-eps", type=int, default=4)
    p.add_argument("--refresh-every-sec", type=float, default=60.0)
    p.add_argument("--allow-random-policy", action="store_true")
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-vision-scope", type=str, default="off")
    p.add_argument("--strict-vision-scope", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stop-file", type=str, default="")
    return p.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    """collector 设备选择；launcher 会用 CUDA_VISIBLE_DEVICES 限定到单卡。

    launcher 让每个 collector 只看到一张物理卡，因此默认 ``cuda:0`` 就是该进程对应
    的 collector GPU；手工调试可传 ``--device cpu``。
    """

    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def main() -> None:
    """collector 主循环。

    生命周期：

    1. 初始化 replay 目录和 Qwen+LoRA 模型；
    2. 等待 learner 发布初始 snapshot，除非显式 ``--allow-random-policy``；
    3. 循环抢 episode、刷新 snapshot、rollout、写 trajectory；
    4. 观察到 STOP 哨兵后，在当前 episode 完成边界退出。
    """

    args = parse_args()
    replay_dir = pathlib.Path(args.replay_dir)
    replay.ensure_replay_dirs(replay_dir)
    stop_file = pathlib.Path(args.stop_file) if args.stop_file else pathlib.Path(args.replay_dir).parent / "STOP"
    ds = EpisodeDataset(pathlib.Path(args.train_jsonl))
    if len(ds.rows) <= 0:
        raise ValueError(f"empty train jsonl: {args.train_jsonl}")

    device = _resolve_device(args.device)
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
    loaded_version = -1
    latest_lora_dir = pathlib.Path(args.latest_lora_dir)
    while loaded_version < 0:
        # 生产路径要求先有 v_0，否则 collector 会用随机 LoRA 采集脏数据。
        loaded_version = _maybe_refresh_snapshot(bundle, latest_lora_dir, loaded_version)
        if loaded_version >= 0 or args.allow_random_policy:
            break
        if stop_file.exists():
            print(f"[collect] stop file observed before first snapshot: {stop_file}", flush=True)
            return
        print(f"[collect] waiting for initial snapshot under {latest_lora_dir}", flush=True)
        time.sleep(5.0)

    collected = 0
    last_refresh = time.time()
    print(
        f"[collect] id={args.collector_id} episodes={len(ds.rows)} replay={replay_dir} "
        f"device={device} loaded_version={loaded_version}",
        flush=True,
    )
    while True:
        if stop_file.exists():
            print(f"[collect] stop file observed: {stop_file}", flush=True)
            break
        if args.max_episodes > 0 and collected >= args.max_episodes:
            break
        now = time.time()
        if (
            collected == 0
            or (args.refresh_every_eps > 0 and collected % args.refresh_every_eps == 0)
            or (args.refresh_every_sec > 0 and now - last_refresh >= args.refresh_every_sec)
        ):
            # reload 是 episode 边界操作，避免单条 trajectory 内前后帧来自不同策略。
            loaded_version = _maybe_refresh_snapshot(bundle, latest_lora_dir, loaded_version)
            last_refresh = now

        # claim_episode_index 是全局 counter；多个 collector 同时抢也不会拿到同一个 value。
        idx = replay.claim_episode_index(replay_dir, total_episodes=len(ds.rows))
        ep = ds.rows[idx]
        start = time.time()
        try:
            records = collect_episode(
                bundle,
                ep,
                collector_id=args.collector_id,
                policy_version=max(loaded_version, 0),
                p_init_correct=args.p_init_correct,
                phase_b_noise_prob=args.phase_b_noise_prob,
                skip_correction_scene_noise_prob=args.skip_correction_scene_noise_prob,
                outer_stride=args.outer_stride,
                seed=args.seed,
            )
            out = replay.write_trajectory(
                replay_dir,
                records,
                collector_id=args.collector_id,
                run_id=ep.run_id,
                capacity=args.replay_capacity,
            )
            collected += 1
            elapsed = time.time() - start
            print(
                f"[collect] id={args.collector_id} n={collected} idx={idx} "
                f"run={ep.run_id} frames={len(records) - 1} version={max(loaded_version, 0)} "
                f"elapsed={elapsed:.1f}s -> {out.name}",
                flush=True,
            )
        except Exception as exc:
            replay.move_failed(replay_dir, None, reason=f"{args.collector_id} idx={idx} run={ep.run_id}: {exc}")
            print(f"[collect][error] id={args.collector_id} idx={idx} run={ep.run_id}: {exc}", file=sys.stderr, flush=True)
            time.sleep(1.0)


if __name__ == "__main__":
    main()
