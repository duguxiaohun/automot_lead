"""SFT v4 off-policy collector。

collector 负责最慢的 rollout：teacher/student generate、memory 自更新、Phase B 噪声扰动。
它不进入 DDP，也不做 backward；产物是一条 episode 一个 trajectory jsonl，写入
``replay/ready`` 供 learner 随机抽样。
"""

from __future__ import annotations

import argparse
import json
import math
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
    SCENARIO_LABELS,
    TEACHER_MAX_NEW_TOKENS_STEP1,
    TEACHER_MAX_NEW_TOKENS_STEP2,
    TEACHER_MAX_NEW_TOKENS_STEP3,
    build_step1_user_prompt,
    build_step2_student_prompt,
    build_step2_teacher_prompt,
    build_step2_teacher_target,
    build_step3_student_prompt,
    build_step3_teacher_prompt,
    build_step3_teacher_target,
    check_gt_leak_scene,
    check_gt_leak_status_subgoal,
    first_subgoal,
    force_memory_to_gt_scene,
    init_memory,
    initial_event,
    parse_output,
    should_trigger_step3,
    update_memory_after_step2,
    update_memory_after_step3,
    validate_event,
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
    """把 Memory dataclass 转为可写入 jsonl 的稳定结构。"""

    return {
        "scene": str(memory.scene),
        "status": str(memory.status),
        "subgoal": str(memory.subgoal),
        "ego_to_goal_xy": [float(memory.ego_to_goal_x), float(memory.ego_to_goal_y)],
    }


def _load_adapter_state_if_present(bundle: Any, adapter_dir: pathlib.Path) -> bool:
    """把 learner 发布的 adapter snapshot 加载进当前 collector 模型。"""

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
    """读取 latest_lora/current_version.txt，返回版本号和目录。"""

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
    """如果 learner 发布了新 snapshot，则加载它并返回新版本号。"""

    version, path = _current_snapshot_dir(latest_lora_dir)
    if path is None or version <= loaded_version:
        return loaded_version
    if _load_adapter_state_if_present(bundle, path):
        print(f"[collect] loaded snapshot version={version} path={path}", flush=True)
        return version
    return loaded_version


def _inject_phase_b_noise(memory: Any, *, gt_scene: str, rng: random.Random, prob: float) -> Tuple[Any, bool]:
    """Phase B 弱纠偏后按概率注入随机非 GT scene。"""

    p = min(1.0, max(0.0, float(prob)))
    if p <= 0.0 or rng.random() >= p:
        return memory, False
    candidates = [s for s in sorted(SCENARIO_LABELS) if s != gt_scene]
    if not candidates:
        return memory, False
    scene = rng.choice(candidates)
    mem = memory.copy()
    mem.scene = scene
    mem.status = initial_event(scene)
    mem.subgoal = first_subgoal(scene)
    return mem, True


def collect_episode(
    bundle: Any,
    ep: EpisodeRow,
    *,
    collector_id: str,
    policy_version: int,
    p_init_correct: float,
    phase_b_noise_prob: float,
    outer_stride: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """执行一条 episode rollout，并返回 trajectory records。"""

    run_dir = pathlib.Path(ep.run_dir)
    stride = max(1, int(outer_stride))
    rng = random.Random(f"{seed}:{collector_id}:{ep.run_id}:{policy_version}")
    gx, gy = _load_goal_xy(run_dir, ep.frame_start)
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

    for frame in range(ep.frame_start, ep.frame_end + 1, stride):
        phase_a = _is_phase_a(ep, frame)
        noise_injected = False
        if not phase_a:
            memory = force_memory_to_gt_scene(memory, gt_scene=ep.gt_scene)
            memory, noise_injected = _inject_phase_b_noise(
                memory,
                gt_scene=ep.gt_scene,
                rng=rng,
                prob=phase_b_noise_prob,
            )
        memory_before = memory.copy()
        image_paths = _build_rgb_paths(run_dir, frame)
        images = _load_images(image_paths)
        gt_status, gt_subgoal = _gt_status_subgoal(ep, frame)

        step1_user = build_step1_user_prompt(len(images))
        step1_msgs = _build_messages_with_images(user_text=step1_user, images=images)
        step2_teacher_user = build_step2_teacher_prompt(memory_before, ep.gt_scene)
        step2_student_user = build_step2_student_prompt(memory_before)

        model = bundle.unwrap()
        was_training = bool(model.training)
        model.eval()
        with model.disable_adapter():
            teacher_step1_prompt_state = _teacher_start_state(bundle, step1_msgs)
            teacher_step1, teacher_step1_state = _teacher_generate_kv(
                bundle, teacher_step1_prompt_state, TEACHER_MAX_NEW_TOKENS_STEP1
            )
            teacher_step1 = teacher_step1 or "I observe the current driving scene from the images."
            teacher_step2_prompt_state = _append_user_turn(bundle, teacher_step1_state, step2_teacher_user)
            raw_teacher_step2, teacher_step2_state = _teacher_generate_kv(
                bundle, teacher_step2_prompt_state, TEACHER_MAX_NEW_TOKENS_STEP2
            )
        if was_training:
            model.train()

        bundle.model.eval()
        student_step1_prompt_state = _student_start_state(bundle, step1_msgs)
        student_step1, student_step1_state = _student_generate_kv(
            bundle, student_step1_prompt_state, TEACHER_MAX_NEW_TOKENS_STEP1
        )
        student_step1 = student_step1 or teacher_step1
        student_step2_prompt_state = _append_user_turn(bundle, student_step1_state, step2_student_user)
        raw_student_step2, student_step2_state = _student_generate_kv(
            bundle, student_step2_prompt_state, TEACHER_MAX_NEW_TOKENS_STEP2
        )

        analysis2 = _analysis_before_labels(raw_teacher_step2)
        leak2 = check_gt_leak_scene(analysis2, ep.gt_scene)
        target2 = build_step2_teacher_target(analysis2, ep.gt_scene)
        pred2 = parse_output(raw_student_step2)
        old_scene = memory.scene
        memory = update_memory_after_step2(memory, student_scene=pred2.get("scene"))
        scene_flip = memory.scene != old_scene
        memory_after_step2 = memory.copy()

        step3_ran = should_trigger_step3(memory_scene_after_step2=memory.scene, gt_scene=ep.gt_scene)
        raw_teacher_step3 = ""
        raw_student_step3 = ""
        target3 = ""
        leak3 = False
        if step3_ran:
            step3_teacher_user = build_step3_teacher_prompt(memory, gt_status, gt_subgoal)
            was_training = bool(model.training)
            model.eval()
            with model.disable_adapter():
                teacher_step3_prompt_state = _append_user_turn(bundle, teacher_step2_state, step3_teacher_user)
                raw_teacher_step3, _ = _teacher_generate_kv(
                    bundle, teacher_step3_prompt_state, TEACHER_MAX_NEW_TOKENS_STEP3
                )
            if was_training:
                model.train()
            analysis3 = _analysis_before_labels(raw_teacher_step3)
            leak3 = check_gt_leak_status_subgoal(analysis3, gt_status, gt_subgoal)
            target3 = build_step3_teacher_target(analysis3, gt_status, gt_subgoal)

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

        frame_record = {
            "kind": "frame",
            "frame_idx": int(frame),
            "phase": "A" if phase_a else "B",
            "image_paths": image_paths,
            "memory_before": _memory_to_dict(memory_before),
            "memory_after_step2": _memory_to_dict(memory_after_step2),
            "memory_after_frame": _memory_to_dict(memory),
            "gt": {
                "scene": ep.gt_scene,
                "status": gt_status,
                "subgoal": gt_subgoal,
            },
            "student_outputs": {
                "step1": student_step1,
                "step2": raw_student_step2,
                "step3": raw_student_step3,
            },
            "teacher_targets": {
                "step1": teacher_step1,
                "step2": target2,
                "step3": target3,
            },
            "flags": {
                "step3_ran": bool(step3_ran),
                "scene_flip": bool(scene_flip),
                "leak2": bool(leak2),
                "leak3": bool(leak3),
                "phase_a": bool(phase_a),
                "noise_injected": bool(noise_injected),
            },
        }
        frame_record["step3_ran"] = bool(step3_ran)
        records.append(frame_record)
        _prefetch_goal_xy_for_next_frame(memory, run_dir, frame + stride, ep.frame_end)

    records[0]["frame_count"] = len(records) - 1
    replay.validate_trajectory(records)
    return records


def parse_args() -> argparse.Namespace:
    """解析 collector 参数。"""

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
    p.add_argument("--p-init-correct", type=float, default=0.5)
    p.add_argument("--phase-b-noise-prob", type=float, default=0.15)
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
    """collector 设备选择；launcher 会用 CUDA_VISIBLE_DEVICES 限定到单卡。"""

    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def main() -> None:
    """collector 主循环。"""

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
            loaded_version = _maybe_refresh_snapshot(bundle, latest_lora_dir, loaded_version)
            last_refresh = now

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
