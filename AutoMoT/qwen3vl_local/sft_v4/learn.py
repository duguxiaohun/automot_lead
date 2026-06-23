"""SFT v4 off-policy learner。

learner 是唯一进入 DDP 的角色：两个 rank 从 replay 随机抽 trajectory，只做
teacher-forced loss + backward，不再现场 generate。collector 慢时 learner 只 sleep
等 ready 文件，不发起 NCCL collective；一旦两个 rank 都拿到有效 trajectory，每步
都执行一次同构 backward，DDP 梯度同步保持 lockstep。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import shutil
import sys
import time
from datetime import timedelta
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
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_AVAILABLE = True
except Exception:
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False

from qwen3vl_local.sft_v2.train import _is_vision_module_name, load_model_with_lora, make_scheduler
from qwen3vl_local.sft_v4 import replay
from qwen3vl_local.sft_v4.prompts import (
    build_step1_user_prompt,
    build_step2_student_prompt,
    build_step3_student_prompt,
    check_gt_leak_scene,
    check_gt_leak_status_subgoal,
    target_spans_scene,
    target_spans_status,
)
from qwen3vl_local.sft_v4.train import (
    _analysis_before_labels,
    _append_text,
    _append_user_turn,
    _assistant_loss_from_state,
    _build_messages_with_images,
    _clone_kv_state,
    _load_images,
    _param_norm,
    _safe_ratio,
    _student_start_state,
    _to_float,
)


def setup_distributed() -> Tuple[int, int, int]:
    """初始化 learner-only DDP process group。"""

    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, timeout=timedelta(hours=2))
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """关闭 DDP process group。"""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    """rank0 负责日志、TB、snapshot 和 checkpoint。"""

    return rank == 0


def _sync_bool(value: bool, device: torch.device, *, op: Any = None) -> bool:
    """跨 learner ranks 同步 bool；默认 MIN 语义用于“所有 rank 都 ready”。"""

    if not (dist.is_available() and dist.is_initialized()):
        return bool(value)
    tensor = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
    dist.all_reduce(tensor, op=op or dist.ReduceOp.MIN)
    return bool(int(tensor.item()))


def _memory_from_record(payload: Dict[str, Any]) -> Any:
    """把 trajectory 中的 memory dict 还原成 prompts.Memory。"""

    from qwen3vl_local.sft_v4.prompts import Memory

    xy = payload.get("ego_to_goal_xy") or [0.0, 0.0]
    return Memory(
        scene=str(payload["scene"]),
        status=str(payload["status"]),
        subgoal=str(payload["subgoal"]),
        ego_to_goal_x=float(xy[0]),
        ego_to_goal_y=float(xy[1]),
    )


def _append_student_raw(bundle: Any, state: Any, text: str) -> Any:
    """把 collector 当时的 student raw 输出写入 KV，用于复现后续 user turn。"""

    raw = text.strip() if text else ""
    if not raw:
        raw = "I observe the current driving scene from the images."
    with torch.no_grad():
        next_state, _ = _append_text(bundle, state, raw)
    return next_state


def trajectory_loss(bundle: Any, records: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, float]]:
    """对一条 off-policy trajectory 计算 teacher-forced loss。

    这里完全跳过 generate：step2/step3 的 prompt state 通过 collector 写入的
    ``student_outputs`` 复现，梯度只来自 teacher target 的 token CE。
    """

    total: Optional[torch.Tensor] = None
    stats: Dict[str, float] = {
        "frames": 0.0,
        "step3": 0.0,
        "phase_a": 0.0,
        "noise": 0.0,
        "leak2": 0.0,
        "leak3": 0.0,
        "loss_a1": 0.0,
        "loss_a2": 0.0,
        "loss_a3": 0.0,
        "loss_s2": 0.0,
        "loss_s3_status": 0.0,
        "loss_s3_subgoal": 0.0,
    }
    zero_ref: Optional[torch.Tensor] = None
    for frame in replay.iter_frame_records(records):
        images = _load_images([str(p) for p in frame["image_paths"]])
        memory_before = _memory_from_record(frame["memory_before"])
        targets = frame["teacher_targets"]
        student_outputs = frame.get("student_outputs") or {}
        flags = frame.get("flags") or {}

        step1_user = build_step1_user_prompt(len(images))
        messages = _build_messages_with_images(user_text=step1_user, images=images)
        step1_prompt_state = _student_start_state(bundle, messages)
        step1_parts = _assistant_loss_from_state(
            bundle,
            _clone_kv_state(step1_prompt_state),
            str(targets["step1"]),
            lambda _text: {},
            analysis_enabled=True,
        )
        zero_ref = step1_parts["analysis"] * 0.0

        student_step1_state = _append_student_raw(bundle, _clone_kv_state(step1_prompt_state), str(student_outputs.get("step1", "")))
        step2_prompt_state = _append_user_turn(bundle, student_step1_state, build_step2_student_prompt(memory_before))
        target2 = str(targets["step2"])
        analysis2 = _analysis_before_labels(target2)
        leak2 = bool(flags.get("leak2", check_gt_leak_scene(analysis2, str((frame.get("gt") or {}).get("scene", "")))))
        step2_parts = _assistant_loss_from_state(
            bundle,
            _clone_kv_state(step2_prompt_state),
            target2,
            target_spans_scene,
            analysis_enabled=not leak2,
        )

        a3 = zero_ref
        s3_status = zero_ref
        s3_subgoal = zero_ref
        leak3 = bool(flags.get("leak3", False))
        if bool(frame.get("step3_ran", flags.get("step3_ran", False))):
            student_step2_state = _append_student_raw(
                bundle,
                _clone_kv_state(step2_prompt_state),
                str(student_outputs.get("step2", "")),
            )
            memory_after_step2 = _memory_from_record(frame["memory_after_step2"])
            step3_prompt_state = _append_user_turn(bundle, student_step2_state, build_step3_student_prompt(memory_after_step2))
            target3 = str(targets.get("step3", ""))
            gt = frame.get("gt") or {}
            analysis3 = _analysis_before_labels(target3)
            leak3 = bool(flags.get("leak3", check_gt_leak_status_subgoal(analysis3, str(gt.get("status", "")), str(gt.get("subgoal", "")))))
            step3_parts = _assistant_loss_from_state(
                bundle,
                _clone_kv_state(step3_prompt_state),
                target3,
                target_spans_status,
                analysis_enabled=not leak3,
            )
            a3 = step3_parts["analysis"]
            s3_status = step3_parts["status"]
            s3_subgoal = step3_parts["subgoal"]

        loss = (
            float(args.w_a1) * step1_parts["analysis"]
            + float(args.w_a2) * step2_parts["analysis"]
            + float(args.w_s2) * step2_parts["scene"]
            + float(args.w_a3) * a3
            + float(args.w_s3_status) * s3_status
            + float(args.w_s3_subgoal) * s3_subgoal
        )
        total = loss if total is None else total + loss
        stats["frames"] += 1.0
        stats["step3"] += float(bool(frame.get("step3_ran", flags.get("step3_ran", False))))
        stats["phase_a"] += float(bool(flags.get("phase_a", frame.get("phase") == "A")))
        stats["noise"] += float(bool(flags.get("noise_injected", False)))
        stats["leak2"] += float(leak2)
        stats["leak3"] += float(leak3)
        stats["loss_a1"] += _to_float(step1_parts["analysis"])
        stats["loss_a2"] += _to_float(step2_parts["analysis"])
        stats["loss_a3"] += _to_float(a3)
        stats["loss_s2"] += _to_float(step2_parts["scene"])
        stats["loss_s3_status"] += _to_float(s3_status)
        stats["loss_s3_subgoal"] += _to_float(s3_subgoal)

    if total is None or zero_ref is None or stats["frames"] <= 0:
        raise ValueError("trajectory produced no trainable frames")
    return total / max(stats["frames"], 1.0), stats


def sample_valid_trajectory(replay_dir: pathlib.Path, rng: random.Random, attempts: int = 8) -> Tuple[Optional[pathlib.Path], Optional[List[Dict[str, Any]]]]:
    """随机抽一条可读 trajectory；坏文件会移到 failed 并继续尝试。"""

    for _ in range(max(1, attempts)):
        path = replay.sample_ready_file(replay_dir, rng)
        if path is None:
            return None, None
        try:
            return path, replay.read_trajectory(path)
        except Exception as exc:
            failed_dir = replay.ensure_replay_dirs(replay_dir)["failed"]
            target = failed_dir / f"bad_{path.name}"
            try:
                os.replace(path, target)
            except OSError:
                pass
            print(f"[learn][warn] bad trajectory moved aside: {path} reason={exc}", file=sys.stderr, flush=True)
    return None, None


def _trainable_groups(bundle: Any, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[nn.Parameter], List[nn.Parameter]]:
    """按语言/视觉 LoRA 参数分组，复用 v2/v3 的 LR 和裁剪策略。"""

    language_params: List[nn.Parameter] = []
    vision_params: List[nn.Parameter] = []
    for name, param in bundle.unwrap().named_parameters():
        if not param.requires_grad:
            continue
        if _is_vision_module_name(name):
            vision_params.append(param)
        else:
            language_params.append(param)
    groups: List[Dict[str, Any]] = []
    if language_params:
        groups.append({"params": language_params, "lr": args.learning_rate, "name": "language"})
    if vision_params:
        groups.append({"params": vision_params, "lr": args.learning_rate * float(args.vision_lr_scale), "name": "vision"})
    if not groups:
        raise RuntimeError("no trainable LoRA params found")
    return groups, language_params, vision_params


def _write_adapter_metadata(path: pathlib.Path, bundle: Any, args: argparse.Namespace, *, step: int, kind: str) -> None:
    """写 v4 off-policy adapter 自描述配置。"""

    targets = list(bundle.lora_target_modules)
    payload = {
        "schema_version": 1,
        "route": "sft_v4_offpolicy",
        "kind": kind,
        "step": int(step),
        "base_model_dir": str(args.model_dir),
        "base_model_mutated": False,
        "distributed_train": {
            "mode": "off_policy_actor_learner",
            "learner_world_size": int(args.learner_world_size),
            "collector_processes": int(args.collector_processes),
            "replay_sampling": "rank_local_random_choice",
            "snapshot_every_steps": int(args.snapshot_every_steps),
        },
        "lora_vision_scope": bundle.lora_vision_scope,
        "target_modules": targets,
        "loss_weights": {
            "a1": float(args.w_a1),
            "a2": float(args.w_a2),
            "a3": float(args.w_a3),
            "s2": float(args.w_s2),
            "s3_status": float(args.w_s3_status),
            "s3_subgoal": float(args.w_s3_subgoal),
        },
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "sft_v4_adapter_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def publish_snapshot(bundle: Any, args: argparse.Namespace, *, step: int) -> pathlib.Path:
    """rank0 发布给 collectors 使用的 LoRA snapshot，并原子更新 current_version.txt。"""

    latest = pathlib.Path(args.output_dir) / "latest_lora"
    latest.mkdir(parents=True, exist_ok=True)
    target = latest / f"v_{int(step)}"
    tmp = latest / f".tmp_v_{int(step)}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    bundle.unwrap().save_pretrained(str(tmp))
    _write_adapter_metadata(tmp, bundle, args, step=step, kind="snapshot")
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    pointer = latest / "current_version.txt"
    pointer_tmp = latest / "current_version.txt.tmp"
    pointer_tmp.write_text(str(int(step)), encoding="utf-8")
    os.replace(pointer_tmp, pointer)
    versions = sorted(
        [p for p in latest.glob("v_*") if p.is_dir() and p.name[2:].isdigit()],
        key=lambda p: int(p.name[2:]),
    )
    keep = max(3, int(args.keep_snapshots))
    for old in versions[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    return target


def save_checkpoint(bundle: Any, optimizer: torch.optim.Optimizer, scheduler: Any, args: argparse.Namespace, *, step: int) -> pathlib.Path:
    """rank0 保存可恢复训练 checkpoint。"""

    ckpt = pathlib.Path(args.output_dir) / f"checkpoint-{int(step)}"
    tmp = pathlib.Path(args.output_dir) / f".tmp_checkpoint_{int(step)}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    bundle.unwrap().save_pretrained(str(tmp))
    _write_adapter_metadata(tmp, bundle, args, step=step, kind="checkpoint")
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": int(step)}, tmp / "trainer_state.pt")
    if ckpt.exists():
        shutil.rmtree(ckpt, ignore_errors=True)
    tmp.rename(ckpt)
    checkpoints = sorted(
        [p for p in pathlib.Path(args.output_dir).glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    keep = max(1, int(args.save_total_limit))
    for old in checkpoints[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    return ckpt


def parse_args() -> argparse.Namespace:
    """解析 learner 参数。"""

    p = argparse.ArgumentParser(description="Train SFT v4 learner from off-policy replay")
    p.add_argument("--model-dir", type=str, default="checkpoints/Qwen3-VL-4B-Instruct")
    p.add_argument("--replay-dir", type=str, default="checkpoints/sft_v4_lora/latest/replay")
    p.add_argument("--output-dir", type=str, default="checkpoints/sft_v4_lora/latest")
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--wait-replay-sec", type=float, default=5.0)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-vision-scope", type=str, default="off")
    p.add_argument("--strict-vision-scope", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vision-lr-scale", type=float, default=0.1)
    p.add_argument("--language-clip-norm", type=float, default=1.0)
    p.add_argument("--vision-clip-norm", type=float, default=0.3)
    p.add_argument("--w-a1", type=float, default=0.2)
    p.add_argument("--w-a2", type=float, default=0.2)
    p.add_argument("--w-a3", type=float, default=0.2)
    p.add_argument("--w-s2", type=float, default=1.0)
    p.add_argument("--w-s3-status", type=float, default=1.0)
    p.add_argument("--w-s3-subgoal", type=float, default=1.0)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--snapshot-every-steps", type=int, default=1000)
    p.add_argument("--save-steps", type=int, default=5000)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--keep-snapshots", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument("--check", action="store_true")
    p.add_argument("--learner-world-size", type=int, default=2)
    p.add_argument("--collector-processes", type=int, default=6)
    return p.parse_args()


def main() -> None:
    """learner DDP 主入口。"""

    args = parse_args()
    if int(args.max_steps) <= 0:
        raise ValueError("--max-steps must be positive for off-policy learner")
    rank, world_size, local_rank = setup_distributed()
    if args.check:
        args.max_steps = min(int(args.max_steps), 2)
        args.snapshot_every_steps = 1
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if world_size != int(args.learner_world_size) and is_rank0(rank):
        print(f"[learn][warn] actual world_size={world_size}, configured learner_world_size={args.learner_world_size}", flush=True)
    output_dir = pathlib.Path(args.output_dir)
    replay_dir = pathlib.Path(args.replay_dir)
    if is_rank0(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        replay.ensure_replay_dirs(replay_dir)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

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
    groups, language_params, vision_params = _trainable_groups(bundle, args)
    has_vision_lora = bool(vision_params)
    if has_vision_lora and is_rank0(rank):
        print(
            "[learn][warn] vision LoRA is enabled, but v4 off-policy learner keeps image prefill under no_grad; "
            "DDP will use find_unused_parameters=True and visual gradients may stay zero.",
            flush=True,
        )
    if dist.is_available() and dist.is_initialized():
        for p in language_params + vision_params:
            dist.broadcast(p.data, src=0)
        bundle.model = DDP(
            bundle.model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=has_vision_lora,
            broadcast_buffers=False,
        )
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=float(args.weight_decay))
    scheduler = make_scheduler(optimizer, int(args.max_steps), int(int(args.max_steps) * float(args.warmup_ratio)))
    tb = SummaryWriter(log_dir=str(output_dir / "tb")) if (is_rank0(rank) and _TB_AVAILABLE) else None
    rng = random.Random(int(args.seed) + rank * 9973)
    if is_rank0(rank):
        publish_snapshot(bundle, args, step=0)
        print(f"[learn] output={output_dir} replay={replay_dir} world_size={world_size}", flush=True)

    global_step = 0
    start = time.time()
    while global_step < int(args.max_steps):
        path, records = sample_valid_trajectory(replay_dir, rng)
        local_ready = records is not None
        all_ready = _sync_bool(local_ready, device, op=dist.ReduceOp.MIN if dist.is_available() and dist.is_initialized() else None)
        if not all_ready:
            if is_rank0(rank):
                st = replay.replay_stats(replay_dir)
                print(f"[learn] waiting replay ready={st.ready_count} pending={st.pending_count}", flush=True)
            time.sleep(float(args.wait_replay_sec))
            continue
        assert records is not None
        bundle.model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, stats = trajectory_loss(bundle, records, args)
        loss.backward()
        lang_norm = torch.nn.utils.clip_grad_norm_(language_params, float(args.language_clip_norm)) if language_params else torch.tensor(0.0, device=device)
        vis_norm = torch.nn.utils.clip_grad_norm_(vision_params, float(args.vision_clip_norm)) if vision_params else torch.tensor(0.0, device=device)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if is_rank0(rank) and (global_step == 1 or global_step % max(1, args.logging_steps) == 0):
            frames = max(stats["frames"], 1.0)
            st = replay.replay_stats(replay_dir)
            lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else 0.0
            print(
                f"[learn] step={global_step}/{args.max_steps} loss={_to_float(loss):.4f} "
                f"frames={int(frames)} step3={_safe_ratio(stats['step3'], frames):.3f} "
                f"noise={_safe_ratio(stats['noise'], frames):.3f} replay={st.ready_count} "
                f"age={st.avg_age_minutes:.1f}m lr={lr:.2e} elapsed={(time.time() - start) / 60.0:.1f}m",
                flush=True,
            )
            if tb is not None:
                tb.add_scalar("train/loss_total", _to_float(loss), global_step)
                tb.add_scalar("train/lr", float(lr), global_step)
                tb.add_scalar("train/grad_norm/language", float(lang_norm), global_step)
                tb.add_scalar("train/grad_norm/vision", float(vis_norm), global_step)
                tb.add_scalar("train/replay/size", float(st.ready_count), global_step)
                tb.add_scalar("train/replay/avg_age_minutes", float(st.avg_age_minutes), global_step)
                tb.add_scalar("train/step3_trigger_rate", _safe_ratio(stats["step3"], frames), global_step)
                tb.add_scalar("train/phase_b_noise_rate", _safe_ratio(stats["noise"], frames), global_step)
                for key in ("a1", "a2", "a3", "s2", "s3_status", "s3_subgoal"):
                    tb.add_scalar(f"train/loss/{key}", stats[f"loss_{key}"] / frames, global_step)

        if is_rank0(rank) and int(args.snapshot_every_steps) > 0 and global_step % int(args.snapshot_every_steps) == 0:
            snap = publish_snapshot(bundle, args, step=global_step)
            if tb is not None:
                tb.add_scalar("train/lora_snapshot/version", float(global_step), global_step)
            print(f"[snapshot] {snap}", flush=True)
        if is_rank0(rank) and int(args.save_steps) > 0 and global_step % int(args.save_steps) == 0:
            ckpt = save_checkpoint(bundle, optimizer, scheduler, args, step=global_step)
            print(f"[checkpoint] {ckpt}", flush=True)

    if is_rank0(rank):
        final_dir = output_dir / "final"
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        bundle.unwrap().save_pretrained(str(final_dir))
        _write_adapter_metadata(final_dir, bundle, args, step=global_step, kind="final")
        publish_snapshot(bundle, args, step=global_step)
        (output_dir / "STOP").write_text("done\n", encoding="utf-8")
        if tb is not None:
            tb.flush()
            tb.close()
        print(f"[done] final={final_dir} stop={output_dir / 'STOP'}", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
