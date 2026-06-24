"""SFT v4 off-policy learner。

learner 是唯一进入 DDP 的角色：两个 rank 从 replay 随机抽 trajectory，只做
teacher-forced loss + backward，不再现场 generate。collector 慢时 learner 只 sleep
等 ready 文件，不发起 NCCL collective；一旦两个 rank 都拿到有效 trajectory，每步
都执行一次同构 backward，DDP 梯度同步保持 lockstep。

运行方式：

``launch_offpolicy.sh`` 用 ``torchrun --nproc_per_node=2`` 启动本脚本。每个 rank
各自随机抽一条 trajectory，构成 effective batch=2；rank0 负责 TensorBoard、checkpoint
和给 collector 使用的 LoRA snapshot。

训练语义：

- collector 已经完成 teacher/student generate，learner 绝不调用 generate；
- learner 重新读取图像做 prefill，但只对 assistant target token 计算 CE；
- student raw output 只用来复现 collector 当时的 KV 对话上下文；
- 两个 DDP rank 每个 step 都只 backward 一次，因此 collective 顺序固定。
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
from contextlib import nullcontext
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
    check_gt_leak_road_structure,
    check_gt_leak_scene,
    check_gt_leak_status_subgoal,
    target_spans_road_structure,
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
    _safe_ratio,
    _student_start_state,
    _to_float,
)


def setup_distributed() -> Tuple[int, int, int]:
    """初始化 learner-only DDP process group。

    collector 完全不调用这个函数。若用户单进程调试 ``learn.py``，环境里没有 ``RANK``，
    则退化为 rank0/world_size=1，不创建 process group。
    """

    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_id = None
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_id = torch.device(f"cuda:{local_rank}")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_kwargs: Dict[str, Any] = {"backend": backend, "timeout": timedelta(hours=2)}
    if device_id is not None:
        init_kwargs["device_id"] = device_id
    try:
        dist.init_process_group(**init_kwargs)
    except TypeError:
        init_kwargs.pop("device_id", None)
        dist.init_process_group(**init_kwargs)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """关闭 DDP process group。

    只在 learner 进程里调用；collector 没有 process group，也就没有清理动作。
    """

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    """rank0 负责日志、TB、snapshot 和 checkpoint。"""

    return rank == 0


def _sync_bool(value: bool, device: torch.device, *, op: Any = None) -> bool:
    """跨 learner ranks 同步 bool；默认 MIN 语义用于“所有 rank 都 ready”。

    replay 为空时不能让某个 rank 先 backward、另一个 rank sleep，否则 DDP collective
    数会不匹配。这里用一个 1 元素 tensor 做 CPU/GPU 侧的轻量 allreduce，只有所有 rank
    都拿到有效 trajectory 才进入真正的 loss/backward。
    """

    if not (dist.is_available() and dist.is_initialized()):
        return bool(value)
    tensor = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
    dist.all_reduce(tensor, op=op or dist.ReduceOp.MIN)
    return bool(int(tensor.item()))


def _memory_from_record(payload: Dict[str, Any]) -> Any:
    """把 trajectory 中的 memory dict 还原成 prompts.Memory。

    collector 写盘时把 dataclass 展平成普通 dict；learner 在构造 step1/2/3 prompt 前
    需要还原成 ``Memory``，这样可以继续复用 prompts.py 里的格式化逻辑。

    v2 schema 强制要求 ``road_structure`` 字段——v1 旧 trajectory 已被
    ``replay.validate_trajectory`` 拒收，所以这里直接当 KeyError 处理而不是兜底默认值，
    避免静默吃掉格式错误。
    """

    from qwen3vl_local.sft_v4.prompts import Memory

    xy = payload.get("ego_to_goal_xy") or [0.0, 0.0]
    return Memory(
        scene=str(payload["scene"]),
        status=str(payload["status"]),
        subgoal=str(payload["subgoal"]),
        ego_to_goal_x=float(xy[0]),
        ego_to_goal_y=float(xy[1]),
        road_structure=str(payload["road_structure"]),
    )


def _frame_teacher_targets(frame: Dict[str, Any]) -> Dict[str, str]:
    """统一读取 trajectory 里的 teacher target（v2 schema）。

    v2 trajectory：``teacher_step1_target`` 永远存在；``teacher_step2_target`` /
    ``teacher_step3_target`` 在对应触发位为 False 时为 None。这里把 None 转空串以便
    下游分支判断。
    """

    nested = frame.get("teacher_targets") or {}
    return {
        "step1": str(nested.get("step1") or frame.get("teacher_step1_target") or frame.get("teacher_step1_text") or ""),
        "step2": str(nested.get("step2") or frame.get("teacher_step2_target") or ""),
        "step3": str(nested.get("step3") or frame.get("teacher_step3_target") or ""),
    }


def _frame_student_outputs(frame: Dict[str, Any]) -> Dict[str, str]:
    """统一读取 collector 存下的 student raw 输出，兼容新旧 trajectory 字段。"""

    nested = frame.get("student_outputs") or {}
    return {
        "step1": str(nested.get("step1") or frame.get("student_step1_raw") or ""),
        "step2": str(nested.get("step2") or frame.get("student_step2_raw") or ""),
        "step3": str(nested.get("step3") or frame.get("student_step3_raw") or ""),
    }


def _frame_step3_fired(frame: Dict[str, Any]) -> bool:
    """统一读取 step3 触发标志。"""

    flags = frame.get("flags") or {}
    return bool(frame.get("step3_fired", frame.get("step3_ran", flags.get("step3_ran", False))))


def _frame_step2_fired(frame: Dict[str, Any]) -> bool:
    """统一读取 step2 触发标志（v2 新增）。

    v2 trajectory：step1 命中 layer-1 才会触发 step2；未触发时所有 step2 字段为 None。
    旧 trajectory 无此字段——但旧 traj 已经被 replay 拒收，这里默认 True 仅作防御。
    """

    flags = frame.get("flags") or {}
    return bool(frame.get("step2_fired", frame.get("step2_ran", flags.get("step2_ran", True))))


def _append_student_raw(bundle: Any, state: Any, text: str) -> Any:
    """把 collector 当时的 student raw 输出写入 KV，用于复现后续 user turn。

    learner 不关心这段 student raw 的 loss；它只是对话历史的一部分。举例：
    step3 prompt 必须接在 collector 当时的 student step2 输出后面，否则 KV 上下文就和
    trajectory 采集时不一致。
    """

    raw = text.strip() if text else ""
    if not raw:
        raw = "I observe the current driving scene from the images."
    with torch.no_grad():
        next_state, _ = _append_text(bundle, state, raw)
    return next_state


def trajectory_loss(bundle: Any, records: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute teacher-forced loss for a small off-policy record list.

    生产 learner 通过 ``trajectory_backward`` 每次只喂一帧并立刻 backward，避免整条
    episode 的计算图一直挂在显存里。这里完全跳过 generate：step2/step3 的 prompt state
    通过 collector 写入的 ``student_outputs`` 复现，梯度只来自 teacher target 的 token CE。
    """

    # Keep total as a Tensor so the caller can backprop to LoRA params.
    total: Optional[torch.Tensor] = None
    stats: Dict[str, float] = {
        "frames": 0.0,
        "step2": 0.0,
        "step3": 0.0,
        "phase_a": 0.0,
        "noise": 0.0,
        "rs_flip": 0.0,
        "scene_flip": 0.0,
        "leak1": 0.0,
        "leak2": 0.0,
        "leak3": 0.0,
        "loss_a1": 0.0,
        "loss_rs1": 0.0,
        "loss_a2": 0.0,
        "loss_a3": 0.0,
        "loss_s2": 0.0,
        "loss_s3_status": 0.0,
        "loss_s3_subgoal": 0.0,
    }
    zero_ref: Optional[torch.Tensor] = None
    for frame in replay.iter_frame_records(records):
        # 图像不存入 replay，按路径重读。prefill 在 train.py helper 内默认 no_grad，
        # 训练梯度只来自后续 assistant target 的 token CE。
        images = _load_images([str(p) for p in frame["image_paths"]])
        memory_payload = frame.get("memory_before") or frame.get("memory_before_frame")
        if not memory_payload:
            raise ValueError("trajectory frame missing memory_before/memory_before_frame")
        memory_before = _memory_from_record(memory_payload)
        targets = _frame_teacher_targets(frame)
        student_outputs = _frame_student_outputs(frame)
        flags = frame.get("flags") or {}
        gt = frame.get("gt") or {}
        gt_road_structure = str(gt.get("road_structure", ""))

        # ============ Step 1：分析 + ROAD_STRUCTURE 标签（每帧都跑）============
        step1_user = build_step1_user_prompt(len(images), memory=memory_before)
        messages = _build_messages_with_images(user_text=step1_user, images=images)
        step1_prompt_state = _student_start_state(bundle, messages)
        target1 = targets["step1"]
        analysis1 = _analysis_before_labels(target1)
        leak1 = bool(flags.get("leak1", check_gt_leak_road_structure(analysis1, gt_road_structure)))
        # step1 现在有 ROAD_STRUCTURE 标签 → target_spans_road_structure 给标签段独立 mask。
        step1_parts = _assistant_loss_from_state(
            bundle,
            _clone_kv_state(step1_prompt_state),
            target1,
            target_spans_road_structure,
            analysis_enabled=not leak1,
        )
        zero_ref = step1_parts["analysis"] * 0.0

        # ============ 触发门 1：step2_fired=False 时跳过 step2 + step3 ============
        a2 = zero_ref
        s2 = zero_ref
        leak2 = False
        step2_fired = _frame_step2_fired(frame)
        if step2_fired:
            # 用 collector 当时的 student step1 原文推 KV，再追加 step2 user prompt，
            # 保证 step2 teacher-forced target 的前文与采集时一致。step2 的候选表
            # 必须来自 collector 在 step1 后写入的 memory_after_step1，而不是帧首
            # memory_before；否则 learner 会重放出与 collector 不同的 SCENE_CHOICES。
            #
            # 注意：learner 这里绝不重新 parse student_step1_raw 来“现算”memory。
            # collector 才是 rollout 真相来源；learner 只做 teacher-forced loss。
            # 这样即便 parse 规则以后调整，已经采集好的 trajectory 也不会在重放时
            # 悄悄改变状态机语义。
            student_step1_state = _append_student_raw(
                bundle, _clone_kv_state(step1_prompt_state), str(student_outputs.get("step1", ""))
            )
            memory_after_step1_payload = frame.get("memory_after_step1")
            if not memory_after_step1_payload:
                raise ValueError("v2 trajectory frame missing memory_after_step1 for step2 replay")
            memory_after_step1 = _memory_from_record(memory_after_step1_payload)
            step2_prompt_state = _append_user_turn(
                bundle, student_step1_state, build_step2_student_prompt(memory_after_step1)
            )
            target2 = targets["step2"]
            analysis2 = _analysis_before_labels(target2)
            leak2 = bool(flags.get("leak2", check_gt_leak_scene(analysis2, str(gt.get("scene", "")))))
            step2_parts = _assistant_loss_from_state(
                bundle,
                _clone_kv_state(step2_prompt_state),
                target2,
                target_spans_scene,
                analysis_enabled=not leak2,
            )
            a2 = step2_parts["analysis"]
            s2 = step2_parts["scene"]
        else:
            step2_prompt_state = None  # step3 也不会触发，这里不需要构造

        # ============ 触发门 2：step3 ============
        a3 = zero_ref
        s3_status = zero_ref
        s3_subgoal = zero_ref
        leak3 = bool(flags.get("leak3", False))
        if step2_fired and _frame_step3_fired(frame):
            # step3 的 prompt 必须接在 collector 当时的 student step2 输出之后；
            # 再用 trajectory 里记录的 memory_after_step2 格式化 user prompt。
            student_step2_state = _append_student_raw(
                bundle,
                _clone_kv_state(step2_prompt_state),
                str(student_outputs.get("step2", "")),
            )
            memory_after_step2 = _memory_from_record(frame["memory_after_step2"])
            step3_prompt_state = _append_user_turn(
                bundle, student_step2_state, build_step3_student_prompt(memory_after_step2)
            )
            target3 = targets.get("step3", "")
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

        # 7 项加权（PLAN §12.5）；L_RS1 与 L_A1 每帧都计，L_A2/L_SC 在 step2_fired 时计，
        # L_A3/L_ST/L_SG 在 step3_fired 时计。未触发的项保持 zero_ref，不会污染梯度。
        loss = (
            float(args.w_a1) * step1_parts["analysis"]
            + float(args.w_rs1) * step1_parts["road_structure"]
            + float(args.w_a2) * a2
            + float(args.w_s2) * s2
            + float(args.w_a3) * a3
            + float(args.w_s3_status) * s3_status
            + float(args.w_s3_subgoal) * s3_subgoal
        )
        total = loss if total is None else total + loss
        stats["frames"] += 1.0
        stats["step2"] += float(step2_fired)
        stats["step3"] += float(_frame_step3_fired(frame))
        stats["phase_a"] += float(bool(flags.get("phase_a", frame.get("phase") == "A")))
        stats["noise"] += float(bool(flags.get("noise_injected", False)))
        stats["rs_flip"] += float(bool(flags.get("rs_flip", False)))
        stats["scene_flip"] += float(bool(flags.get("scene_flip", frame.get("scene_flip", False))))
        stats["leak1"] += float(leak1)
        stats["leak2"] += float(leak2)
        stats["leak3"] += float(leak3)
        stats["loss_a1"] += _to_float(step1_parts["analysis"])
        stats["loss_rs1"] += _to_float(step1_parts["road_structure"])
        stats["loss_a2"] += _to_float(a2)
        stats["loss_a3"] += _to_float(a3)
        stats["loss_s2"] += _to_float(s2)
        stats["loss_s3_status"] += _to_float(s3_status)
        stats["loss_s3_subgoal"] += _to_float(s3_subgoal)

    if total is None or zero_ref is None or stats["frames"] <= 0:
        raise ValueError("trajectory produced no trainable frames")
    return total / max(stats["frames"], 1.0), stats


def _average_trainable_grads(params: List[nn.Parameter]) -> None:
    """Mean-reduce LoRA gradients once per learner step.

    Frame counts can differ across ranks. Per-frame backward therefore runs
    under DDP ``no_sync()``, and this fixed-order pass is the only gradient
    collective in the step.
    """

    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = float(dist.get_world_size())
    for param in params:
        if not param.requires_grad:
            continue
        has_grad = torch.tensor([1 if param.grad is not None else 0], dtype=torch.int32, device=param.device)
        dist.all_reduce(has_grad, op=dist.ReduceOp.MAX)
        if int(has_grad.item()) == 0:
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def _merge_stats(dst: Dict[str, float], src: Dict[str, float]) -> None:
    """Accumulate scalar frame stats in-place."""

    for key, value in src.items():
        dst[key] = float(dst.get(key, 0.0)) + float(value)


def trajectory_backward(
    bundle: Any,
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    trainable_params: List[nn.Parameter],
) -> Tuple[float, Dict[str, float]]:
    """Backprop one trajectory with per-frame micro-backward.

    The old path accumulated all frame graphs and called backward once at the
    trajectory end, which can OOM on long episodes. This keeps the optimizer
    step at one trajectory per rank, but frees each frame graph immediately.
    DDP ranks may sample trajectories with different frame counts, so automatic
    DDP gradient sync is disabled for the frame loop and LoRA grads are reduced
    once afterward in a fixed parameter order.
    """

    frames = list(replay.iter_frame_records(records))
    local_frames = len(frames)
    if local_frames <= 0:
        raise ValueError("trajectory produced no trainable frames")

    stats: Dict[str, float] = {}
    loss_total = 0.0
    denom = float(max(local_frames, 1))
    sync_context = bundle.model.no_sync() if dist.is_available() and dist.is_initialized() and hasattr(bundle.model, "no_sync") else nullcontext()
    with sync_context:
        for frame in frames:
            frame_loss, frame_stats = trajectory_loss(bundle, [frame], args)
            (frame_loss / denom).backward()
            loss_total += _to_float(frame_loss)
            _merge_stats(stats, frame_stats)
            del frame_loss
    _average_trainable_grads(trainable_params)
    return loss_total / denom, stats


def sample_valid_trajectory(replay_dir: pathlib.Path, rng: random.Random, attempts: int = 8) -> Tuple[Optional[pathlib.Path], Optional[List[Dict[str, Any]]]]:
    """随机抽一条可读 trajectory；坏文件会移到 failed 并继续尝试。

    文件可能在 learner 列表后被 FIFO 驱逐，或由旧版本 collector 写坏。这里把异常吞掉
    并重抽，避免一个坏样本打断整个 DDP 训练。
    """

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
    """按语言/视觉 LoRA 参数分组，复用 v2/v3 的 LR 和裁剪策略。

    默认视觉 LoRA 关闭；如果用户显式开启，视觉参数会单独使用较小 LR 和 clip norm。
    v4 learner 的图像 prefill 默认 no_grad，因此视觉 LoRA 主要是兼容接口，不建议生产打开。
    """

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


def _param_norm(params: List[nn.Parameter]) -> float:
    """计算一组参数的 L2 范数，用于 TensorBoard 和视觉 LoRA 熔断。"""

    if not params:
        return 0.0
    with torch.no_grad():
        return math.sqrt(sum(float(p.detach().float().norm().item()) ** 2 for p in params))


def _sync_max_float(value: float, device: torch.device) -> float:
    """跨 learner ranks 取最大值；单进程时直接返回。"""

    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _validate_args(args: argparse.Namespace) -> None:
    """校验视觉 LoRA 保险参数，保持与 v2 的安全边界一致。"""

    if float(args.vision_lr_scale) < 0.0:
        raise ValueError("--vision-lr-scale must be >= 0")
    if float(args.max_vision_lr_scale) <= 0.0:
        raise ValueError("--max-vision-lr-scale must be > 0")
    if float(args.language_clip_norm) <= 0.0:
        raise ValueError("--language-clip-norm must be > 0")
    if float(args.vision_clip_norm) <= 0.0:
        raise ValueError("--vision-clip-norm must be > 0")
    if float(args.vision_guard_grad_norm_max) <= 0.0:
        raise ValueError("--vision-guard-grad-norm-max must be > 0")
    if float(args.vision_guard_param_norm_max) <= 0.0:
        raise ValueError("--vision-guard-param-norm-max must be > 0")
    if int(args.vision_guard_patience) <= 0:
        raise ValueError("--vision-guard-patience must be > 0")
    if float(args.startup_replay_timeout_sec) < 0.0:
        raise ValueError("--startup-replay-timeout-sec must be >= 0")
    scope = (args.lora_vision_scope or "off").lower()
    if scope != "off" and float(args.vision_lr_scale) == 0.0:
        raise ValueError("--vision-lr-scale=0 with visual LoRA enabled would freeze visual adapters")
    if scope != "off" and float(args.vision_lr_scale) > float(args.max_vision_lr_scale):
        raise ValueError(
            "--vision-lr-scale exceeds --max-vision-lr-scale under visual LoRA. "
            "Lower vision LR scale or explicitly raise the guard limit."
        )


def _write_adapter_metadata(path: pathlib.Path, bundle: Any, args: argparse.Namespace, *, step: int, kind: str) -> None:
    """写 v4 off-policy adapter 自描述配置。

    这个 JSON 是后续 eval/probe/审计判断 adapter 来源的依据：能看出它来自 off-policy
    actor-learner、learner world_size、collector 数、snapshot 频率和 loss 权重。
    """

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
        "strict_vision_scope": bool(args.strict_vision_scope),
        "vision_lr_scale": float(args.vision_lr_scale),
        "max_vision_lr_scale": float(args.max_vision_lr_scale),
        "vision_clip_norm": float(args.vision_clip_norm),
        "language_clip_norm": float(args.language_clip_norm),
        "vision_guard_enabled": bool(args.vision_guard_enabled),
        "vision_guard_grad_norm_max": float(args.vision_guard_grad_norm_max),
        "vision_guard_param_norm_max": float(args.vision_guard_param_norm_max),
        "vision_guard_patience": int(args.vision_guard_patience),
        "loss_weights": {
            "a1": float(args.w_a1),
            "rs1": float(args.w_rs1),
            "a2": float(args.w_a2),
            "a3": float(args.w_a3),
            "s2": float(args.w_s2),
            "s3_status": float(args.w_s3_status),
            "s3_subgoal": float(args.w_s3_subgoal),
        },
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "sft_v4_adapter_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_snapshot_pointer(pointer: pathlib.Path) -> int | None:
    """读取当前已发布 snapshot 版本；不可读或非法时返回 None。"""

    try:
        text = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def publish_snapshot(bundle: Any, args: argparse.Namespace, *, step: int) -> pathlib.Path:
    """rank0 发布给 collectors 使用的 LoRA snapshot，并原子更新 current_version.txt。

    snapshot 只给 collector 做采集策略，不包含 optimizer/scheduler。发布顺序是：
    先写临时目录 -> rename 成 ``v_<step>`` -> 最后更新 pointer。collector 只看 pointer，
    因此不会加载半写入目录。若 pointer 已经指向同一个 ``v_<step>``，直接复用已发布目录，
    避免 final/resume 同 step 重发时短暂删除 collector 正在读取的版本。
    """

    step_int = int(step)
    latest = pathlib.Path(args.output_dir) / "latest_lora"
    latest.mkdir(parents=True, exist_ok=True)
    target = latest / f"v_{step_int}"
    pointer = latest / "current_version.txt"
    if target.exists() and _read_snapshot_pointer(pointer) == step_int:
        return target
    tmp = latest / f".tmp_v_{step_int}_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    bundle.unwrap().save_pretrained(str(tmp))
    _write_adapter_metadata(tmp, bundle, args, step=step, kind="snapshot")
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    # pointer 最后写，且同样用 tmp+replace，保证 collector 要么看到旧版本，要么看到新版本。
    pointer_tmp = latest / f"current_version.txt.tmp.{os.getpid()}"
    pointer_tmp.write_text(str(step_int), encoding="utf-8")
    os.replace(pointer_tmp, pointer)
    protected_versions = {step_int, max(0, step_int - 1)}
    versions = sorted(
        [p for p in latest.glob("v_*") if p.is_dir() and p.name[2:].isdigit()],
        key=lambda p: int(p.name[2:]),
    )
    keep = max(3, int(args.keep_snapshots))
    keep_tail = set(versions[-keep:])
    for old in versions:
        version = int(old.name[2:])
        if old in keep_tail or version in protected_versions or version >= step_int - 1:
            continue
        shutil.rmtree(old, ignore_errors=True)
    return target


def _load_adapter_state(bundle: Any, adapter_dir: pathlib.Path) -> None:
    """把 PEFT adapter 权重加载到当前 bundle，供 resume checkpoint 使用。"""

    adapter_dir = pathlib.Path(adapter_dir)
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_path), device=str(bundle.device))
    elif bin_path.exists():
        state = torch.load(str(bin_path), map_location=bundle.device)
    else:
        raise FileNotFoundError(f"adapter weights not found under {adapter_dir}")
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(bundle.unwrap(), state)


def load_checkpoint_if_requested(
    bundle: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    args: argparse.Namespace,
) -> int:
    """恢复 learner checkpoint，返回起始 global_step。

    checkpoint 目录由 ``save_checkpoint`` 生成，包含 PEFT adapter 权重和
    ``trainer_state.pt``。恢复后 rank0 会用同一个 step 重新发布 LoRA snapshot，
    collector 后续就能从恢复点继续采样。
    """

    ckpt_arg = str(getattr(args, "resume_from_checkpoint", "") or "").strip()
    if not ckpt_arg:
        return 0
    ckpt = pathlib.Path(ckpt_arg)
    if ckpt.name == "latest":
        candidates = sorted(
            [p for p in pathlib.Path(args.output_dir).glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        )
        if not candidates:
            raise FileNotFoundError(f"no checkpoint-* dirs under {args.output_dir}")
        ckpt = candidates[-1]
    if not ckpt.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {ckpt}")
    _load_adapter_state(bundle, ckpt)
    state_path = ckpt / "trainer_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"trainer_state.pt missing in checkpoint: {ckpt}")
    state = torch.load(str(state_path), map_location=bundle.device)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state.get("step", 0))


def save_checkpoint(bundle: Any, optimizer: torch.optim.Optimizer, scheduler: Any, args: argparse.Namespace, *, step: int) -> pathlib.Path:
    """rank0 保存可恢复训练 checkpoint。

    checkpoint 与 snapshot 分开：checkpoint 给恢复训练用，额外包含 optimizer/scheduler；
    snapshot 给 collector 用，只要求能加载 LoRA adapter。
    """

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
    """解析 learner 参数。

    通常不要手写这些参数，直接通过 ``launch_offpolicy.sh`` 的环境变量间接传入。手工调试
    时必须保证 ``--output-dir``、``--replay-dir`` 与 collectors 使用的是同一组目录。
    """

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
    p.add_argument("--max-vision-lr-scale", type=float, default=0.25)
    p.add_argument("--language-clip-norm", type=float, default=1.0)
    p.add_argument("--vision-clip-norm", type=float, default=0.3)
    p.add_argument("--vision-guard-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vision-guard-grad-norm-max", type=float, default=10.0)
    p.add_argument("--vision-guard-param-norm-max", type=float, default=200.0)
    p.add_argument("--vision-guard-patience", type=int, default=3)
    p.add_argument("--w-a1", type=float, default=0.2)
    p.add_argument("--w-rs1", type=float, default=1.0,
                   help="L_RS1 weight: step1 ROAD_STRUCTURE label CE (D25 = 1.0).")
    p.add_argument("--w-a2", type=float, default=0.2)
    p.add_argument("--w-a3", type=float, default=0.2)
    p.add_argument("--w-s2", type=float, default=1.0)
    p.add_argument("--w-s3-status", type=float, default=1.0)
    p.add_argument("--w-s3-subgoal", type=float, default=1.0)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument(
        "--startup-replay-timeout-sec",
        type=float,
        default=600.0,
        help="Maximum seconds to wait for the first replay trajectory before failing; <=0 disables.",
    )
    p.add_argument("--snapshot-every-steps", type=int, default=1000)
    p.add_argument("--save-steps", type=int, default=5000)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--keep-snapshots", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument("--check", action="store_true")
    p.add_argument("--learner-world-size", type=int, default=2)
    p.add_argument("--collector-processes", type=int, default=2)
    p.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="",
        help="Path to checkpoint-N dir, or 'latest' to use the newest checkpoint under output-dir.",
    )
    return p.parse_args()


def main() -> None:
    """learner DDP 主入口。

    主循环只做一件事：等所有 learner rank 都抽到 trajectory，然后各自算一条 trajectory
    的 loss 并 backward。rank0 周期保存 snapshot/checkpoint；训练结束写 STOP，让 collector
    在 episode 边界优雅退出。
    """

    args = parse_args()
    _validate_args(args)
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
    # 确保 rank0 已经创建目录，其余 rank 再开始加载/发布，避免远端文件系统 race。
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
    trainable_params = language_params + vision_params
    has_vision_lora = bool(vision_params)
    if has_vision_lora and is_rank0(rank):
        print(
            "[learn][warn] vision LoRA is enabled, but v4 off-policy learner keeps image prefill under no_grad; "
            "DDP will use find_unused_parameters=True and visual gradients may stay zero.",
            flush=True,
        )
    if dist.is_available() and dist.is_initialized():
        # DDP 包装前先同步 LoRA 初始权重，避免各 rank 随机初始化不同。
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
    start_step = load_checkpoint_if_requested(bundle, optimizer, scheduler, args)
    tb = SummaryWriter(log_dir=str(output_dir / "tb")) if (is_rank0(rank) and _TB_AVAILABLE) else None
    rng = random.Random(int(args.seed) + rank * 9973)
    if is_rank0(rank):
        # 初始 snapshot 是 collector 的启动门闩；resume 时直接发布恢复 step 的策略。
        publish_snapshot(bundle, args, step=start_step)
        print(
            f"[learn] output={output_dir} replay={replay_dir} world_size={world_size} "
            f"start_step={start_step}",
            flush=True,
        )

    global_step = int(start_step)
    fuse_stopped = False
    startup_timed_out = False
    startup_timeout_reason = ""
    guard_bad_steps = 0
    stop_file = output_dir / "STOP"
    start = time.time()
    first_replay_wait_start: Optional[float] = None
    while global_step < int(args.max_steps):
        stop_requested = _sync_bool(
            stop_file.exists(),
            device,
            op=dist.ReduceOp.MAX if dist.is_available() and dist.is_initialized() else None,
        )
        if stop_requested:
            if is_rank0(rank):
                print(f"[learn] observed external stop file: {stop_file}", flush=True)
            break
        path, records = sample_valid_trajectory(replay_dir, rng)
        local_ready = records is not None
        all_ready = _sync_bool(local_ready, device, op=dist.ReduceOp.MIN if dist.is_available() and dist.is_initialized() else None)
        if not all_ready:
            # replay 为空时只 sleep，不做 forward/backward，所以不会有 NCCL watchdog 空等。
            now = time.time()
            waited = 0.0
            local_startup_timeout = False
            if global_step == start_step:
                if first_replay_wait_start is None:
                    first_replay_wait_start = now
                waited = now - first_replay_wait_start
                local_startup_timeout = (
                    float(args.startup_replay_timeout_sec) > 0
                    and waited >= float(args.startup_replay_timeout_sec)
                )
            startup_timeout_now = _sync_bool(
                local_startup_timeout,
                device,
                op=dist.ReduceOp.MAX if dist.is_available() and dist.is_initialized() else None,
            )
            # timeout 判定本身也要先 allreduce：否则某个 rank 先进入 barrier，
            # 另一个 rank 还在下一轮 _sync_bool，会造成 collective 顺序错位。
            waited = _sync_max_float(waited, device)
            if startup_timeout_now:
                startup_timed_out = True
                startup_timeout_reason = (
                    f"no replay trajectory after {waited:.1f}s under {replay_dir}; "
                    "check collector logs and replay/failed"
                )
                if is_rank0(rank):
                    st = replay.replay_stats(replay_dir)
                    (output_dir / "STOP").write_text("startup_replay_timeout\n", encoding="utf-8")
                    print(
                        f"[learn][error] no replay trajectory after {waited:.1f}s "
                        f"(ready={st.ready_count} pending={st.pending_count} failed={st.failed_count}); "
                        "collectors likely failed or data paths are wrong.",
                        flush=True,
                    )
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                break
            if is_rank0(rank):
                st = replay.replay_stats(replay_dir)
                print(f"[learn] waiting replay ready={st.ready_count} pending={st.pending_count}", flush=True)
            time.sleep(float(args.wait_replay_sec))
            continue
        assert records is not None
        bundle.model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value, stats = trajectory_backward(
            bundle,
            records,
            args,
            trainable_params=trainable_params,
        )
        lang_norm = torch.nn.utils.clip_grad_norm_(language_params, float(args.language_clip_norm)) if language_params else torch.tensor(0.0, device=device)
        vis_norm = torch.nn.utils.clip_grad_norm_(vision_params, float(args.vision_clip_norm)) if vision_params else torch.tensor(0.0, device=device)
        lang_norm_value = _sync_max_float(float(lang_norm), device)
        vis_norm_value = _sync_max_float(float(vis_norm), device)
        lang_param_norm_value = _sync_max_float(_param_norm(language_params), device)
        vis_param_norm_value = _sync_max_float(_param_norm(vision_params), device)

        guard_bad = False
        if bool(args.vision_guard_enabled) and vision_params:
            bad_grad = (not math.isfinite(vis_norm_value)) or vis_norm_value > float(args.vision_guard_grad_norm_max)
            bad_param = (not math.isfinite(vis_param_norm_value)) or vis_param_norm_value > float(args.vision_guard_param_norm_max)
            guard_bad = bool(bad_grad or bad_param)
            guard_bad_steps = guard_bad_steps + 1 if guard_bad else 0
        fuse_now = _sync_bool(
            guard_bad_steps >= int(args.vision_guard_patience),
            device,
            op=dist.ReduceOp.MAX if dist.is_available() and dist.is_initialized() else None,
        )
        if fuse_now:
            # 这里保存的是 optimizer.step() 之前的权重，也就是上一已完成 step 的安全权重；
            # 目录名用 after_step_<global_step> 避免误解成当前坏 step 已经写入 adapter。
            optimizer.zero_grad(set_to_none=True)
            fuse_stopped = True
            if is_rank0(rank):
                reason = (
                    "vision fuse triggered: "
                    f"grad_norm={vis_norm_value:.4f} "
                    f"(max={float(args.vision_guard_grad_norm_max):.4f}), "
                    f"param_norm={vis_param_norm_value:.4f} "
                    f"(max={float(args.vision_guard_param_norm_max):.4f}), "
                    f"bad_steps={guard_bad_steps}"
                )
                emergency = output_dir / f"fuse_stop_after_step_{global_step}"
                if emergency.exists():
                    shutil.rmtree(emergency, ignore_errors=True)
                bundle.unwrap().save_pretrained(str(emergency))
                _write_adapter_metadata(emergency, bundle, args, step=global_step, kind="fuse_stop")
                (emergency / "fuse_reason.txt").write_text(reason + "\n", encoding="utf-8")
                (output_dir / "STOP").write_text("vision_fuse\n", encoding="utf-8")
                print(f"[fuse-stop] {reason}", flush=True)
                print(f"[fuse-stop] emergency adapter -> {emergency}", flush=True)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            break
        optimizer.step()
        scheduler.step()
        global_step += 1

        if is_rank0(rank) and (global_step == 1 or global_step % max(1, args.logging_steps) == 0):
            frames = max(stats["frames"], 1.0)
            st = replay.replay_stats(replay_dir)
            lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else 0.0
            print(
                f"[learn] step={global_step}/{args.max_steps} loss={loss_value:.4f} "
                f"frames={int(frames)} step2={_safe_ratio(stats['step2'], frames):.3f} "
                f"step3={_safe_ratio(stats['step3'], frames):.3f} "
                f"rs_flip={_safe_ratio(stats['rs_flip'], frames):.3f} "
                f"noise={_safe_ratio(stats['noise'], frames):.3f} replay={st.ready_count} "
                f"|g|_lang={lang_norm_value:.3f} |g|_vis={vis_norm_value:.3f} "
                f"|w|_vis={vis_param_norm_value:.3f} guard_bad_steps={guard_bad_steps} "
                f"age={st.avg_age_minutes:.1f}m lr={lr:.2e} elapsed={(time.time() - start) / 60.0:.1f}m",
                flush=True,
            )
            if tb is not None:
                tb.add_scalar("train/loss_total", loss_value, global_step)
                tb.add_scalar("train/lr", float(lr), global_step)
                tb.add_scalar("train/grad_norm/language", lang_norm_value, global_step)
                tb.add_scalar("train/grad_norm/vision", vis_norm_value, global_step)
                tb.add_scalar("train/param_norm/lora_language", lang_param_norm_value, global_step)
                tb.add_scalar("train/param_norm/lora_vision", vis_param_norm_value, global_step)
                tb.add_scalar("train/vision_guard_bad_steps", float(guard_bad_steps), global_step)
                tb.add_scalar("train/replay/size", float(st.ready_count), global_step)
                tb.add_scalar("train/replay/avg_age_minutes", float(st.avg_age_minutes), global_step)
                tb.add_scalar("train/step2_trigger_rate", _safe_ratio(stats["step2"], frames), global_step)
                tb.add_scalar("train/step3_trigger_rate", _safe_ratio(stats["step3"], frames), global_step)
                tb.add_scalar("train/fire_rate/step2", _safe_ratio(stats["step2"], frames), global_step)
                tb.add_scalar("train/fire_rate/step3", _safe_ratio(stats["step3"], frames), global_step)
                tb.add_scalar("train/accuracy/road_structure", _safe_ratio(stats["step2"], frames), global_step)
                tb.add_scalar("train/phase_b_noise_rate", _safe_ratio(stats["noise"], frames), global_step)
                tb.add_scalar("train/rs_flip_rate", _safe_ratio(stats["rs_flip"], frames), global_step)
                tb.add_scalar("train/scene_flip_rate", _safe_ratio(stats["scene_flip"], frames), global_step)
                tb.add_scalar("train/gt_leak_skip_rate/step1", _safe_ratio(stats["leak1"], frames), global_step)
                tb.add_scalar("train/gt_leak_skip_rate/step2", _safe_ratio(stats["leak2"], frames), global_step)
                tb.add_scalar("train/gt_leak_skip_rate/step3", _safe_ratio(stats["leak3"], frames), global_step)
                tb.add_scalar("train/phase_a_frame_frac", _safe_ratio(stats["phase_a"], frames), global_step)
                for key, value in {
                    "a1": args.w_a1,
                    "rs1": args.w_rs1,
                    "a2": args.w_a2,
                    "a3": args.w_a3,
                    "s2": args.w_s2,
                    "s3_status": args.w_s3_status,
                    "s3_subgoal": args.w_s3_subgoal,
                }.items():
                    tb.add_scalar(f"train/loss_weight/{key}", float(value), global_step)
                for key in ("a1", "rs1", "a2", "a3", "s2", "s3_status", "s3_subgoal"):
                    tb.add_scalar(f"train/loss/{key}", stats[f"loss_{key}"] / frames, global_step)
                tb.add_scalar("train/loss/L_A1", stats["loss_a1"] / frames, global_step)
                tb.add_scalar("train/loss/L_RS1", stats["loss_rs1"] / frames, global_step)
                tb.add_scalar("train/loss/L_A2", stats["loss_a2"] / frames, global_step)
                tb.add_scalar("train/loss/L_SC", stats["loss_s2"] / frames, global_step)
                tb.add_scalar("train/loss/L_A3", stats["loss_a3"] / frames, global_step)
                tb.add_scalar("train/loss/L_ST", stats["loss_s3_status"] / frames, global_step)
                tb.add_scalar("train/loss/L_SG", stats["loss_s3_subgoal"] / frames, global_step)

        if is_rank0(rank) and int(args.snapshot_every_steps) > 0 and global_step % int(args.snapshot_every_steps) == 0:
            snap = publish_snapshot(bundle, args, step=global_step)
            if tb is not None:
                tb.add_scalar("train/lora_snapshot/version", float(global_step), global_step)
            print(f"[snapshot] {snap}", flush=True)
        if is_rank0(rank) and int(args.save_steps) > 0 and global_step % int(args.save_steps) == 0:
            ckpt = save_checkpoint(bundle, optimizer, scheduler, args, step=global_step)
            print(f"[checkpoint] {ckpt}", flush=True)

    if is_rank0(rank) and not fuse_stopped and not startup_timed_out:
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
    elif is_rank0(rank):
        if tb is not None:
            tb.flush()
            tb.close()
        if fuse_stopped:
            print("[done] skipped final adapter because vision fuse guard stopped training early", flush=True)
        else:
            print("[done] skipped final adapter because startup replay timed out", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    cleanup_distributed()
    if startup_timed_out:
        raise TimeoutError(startup_timeout_reason)


if __name__ == "__main__":
    main()
