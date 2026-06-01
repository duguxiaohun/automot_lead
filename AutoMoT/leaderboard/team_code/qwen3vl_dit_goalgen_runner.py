"""子目标 latent 生成路线 CLI 入口。

整体路线见 PROJECT_CONTEXT.md §15。本 runner 串起：

  1. 读取历史 + 当前 RGB（复用 image_io.load_lead_rgb_clip）。
  2. teacher-forced 调用本地 Qwen3-VL-Instruct，只做 prefill，切成 DiT 层级 KV。
  3. 用冻结 VAE 把"历史帧 + 子目标关键帧"编到 latent。
  4. 流匹配采样 (z_t, t, v_target)，跑一次 DiT-MoT 前向。
  5. 输出 step.json：提示词、Qwen KV 摘要、分段 KV 摘要、DiT 输入/输出形状、损失。

这是前向 + 损失的单步入口，不是完整训练循环；真正训练请走
`qwen3vl_local/goalgen/train_v1.py` 或 `train_v1.sh`。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# 与 standalone 范式 A runner 一致的路径与 offline 设置。
_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
_CHECKPOINT_DIR = _AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B-Instruct"
_DEFAULT_OUTPUT_ROOT = _AUTOMOT_ROOT / "eval_json" / "qwen3vl_dit_goalgen"
_DEFAULT_KEYFRAMES_JSON = pathlib.Path("/datashare/IOL4SGH/data/data/keyframes_all_scenarios.json")

for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _cli_value(name: str) -> Optional[str]:
    prefix = name + "="
    for i, item in enumerate(sys.argv[1:]):
        if item == name and i + 2 <= len(sys.argv[1:]):
            return sys.argv[i + 2]
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


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
    """单步 runner 默认自动挑 1 张空闲 GPU；显式 device / CUDA mask 时保持外部配置。"""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    if os.environ.get("GOALGEN_RUNNER_DISABLE_AUTO_GPU", "0") == "1":
        return
    device_arg = _cli_value("--device")
    if device_arg and device_arg != "auto":
        return
    selected = _pick_idle_gpus(1)
    if selected:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        print(f"[gpu] auto selected idle CUDA_VISIBLE_DEVICES={selected}; process uses cuda:0/auto")


_maybe_set_idle_gpu_mask()

import torch  # noqa: E402

from qwen3vl_local.engine import LocalQwen3VLInstructEngine  # noqa: E402
from qwen3vl_local.image_io import (  # noqa: E402
    auto_detect_scenario_from_route,
    build_synthetic_raw_and_model_input,
    load_lead_rgb_clip,
)
from qwen3vl_local.prompt_pipeline import DrivingMemory, EVENT_DESCRIPTIONS  # noqa: E402
from qwen3vl_local.goalgen.dit import (  # noqa: E402
    DiTMoT,
    DiTMoTConfig,
)
from qwen3vl_local.goalgen.flow import flow_matching_loss, sample_flow_batch  # noqa: E402
from qwen3vl_local.goalgen.keyframes import (  # noqa: E402
    KeyframeIndex,
    infer_run_id_from_route,
    load_keyframe_rgb,
)
from qwen3vl_local.goalgen.qwen_kv import (  # noqa: E402
    summarize_pooled_kv,
    teacher_forced_prefill,
)
from qwen3vl_local.goalgen.vae import FrozenVAE, default_vae_paths  # noqa: E402


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


def _prepare_images(args: argparse.Namespace) -> Tuple[str, str, List[Any]]:
    """读取历史 + 当前 RGB。返回 (scenario, run_id, model_input_images)。

    没有 route_dir 时退回合成图，run_id 留空（后续找不到关键帧，就只跑前向）。
    """

    scenario = args.scenario
    run_id = ""
    if args.route_dir:
        _raw, model_input_images = load_lead_rgb_clip(
            route_dir=args.route_dir,
            anchor=args.anchor,
            rgb_frame_step=args.rgb_frame_step,
            rgb_frame_count=args.num_frames,
        )
        auto_scenario = auto_detect_scenario_from_route(args.route_dir)
        if auto_scenario and auto_scenario != scenario:
            print(f"[scenario] 自动识别到 '{auto_scenario}'，覆盖命令行里的 '{scenario}'")
            scenario = auto_scenario
        run_id = args.run_id or infer_run_id_from_route(args.route_dir)
    else:
        _raw, model_input_images = build_synthetic_raw_and_model_input(
            num_frames=args.num_frames,
            raw_size=(1920, 1080),
            model_input_size=(1152, 384),
        )
    return scenario, run_id, model_input_images


def _resolve_target_keyframe(
    args: argparse.Namespace,
    scenario: str,
    run_id: str,
    subgoal_event: str,
) -> Optional[int]:
    """从 keyframes_all_scenarios.json 查 subgoal 对应帧。

    显式 --target-frame 优先；没给且 route_dir 可用时再查 JSON。两个都拿不到返回 None。
    """

    if args.target_frame is not None and args.target_frame >= 0:
        return int(args.target_frame)
    if not args.route_dir or not run_id:
        return None
    try:
        index = KeyframeIndex.load(pathlib.Path(args.keyframes_json) if args.keyframes_json else None)
    except FileNotFoundError as e:
        print(f"[keyframes] {e}")
        return None
    frame = index.find_frame_for_event(scenario, run_id, subgoal_event)
    if frame is None:
        print(
            f"[keyframes] miss: scenario={scenario} run_id={run_id} event={subgoal_event}; "
            "将跳过损失计算，仅做前向"
        )
    return frame


def _apply_memory_overrides(memory: DrivingMemory, args: argparse.Namespace) -> None:
    """应用命令行里的 STATUS/SUBGOAL 覆盖，并严格保持状态机契约。"""

    if args.status:
        if args.status not in memory.event_sequence:
            raise ValueError(
                f"对场景 {memory.scenario} 来说，STATUS '{args.status}' 不合法；"
                f"合法事件={memory.event_sequence}"
            )
        memory.status = args.status

    expected_subgoal = memory._next_event_after(memory.status)
    if expected_subgoal is None:
        raise ValueError(
            f"STATUS '{memory.status}' 在事件链 {memory.event_sequence} 中没有后续子目标"
        )

    if args.subgoal:
        if args.subgoal not in memory.event_sequence:
            raise ValueError(
                f"对场景 {memory.scenario} 来说，SUBGOAL '{args.subgoal}' 不合法；"
                f"合法事件={memory.event_sequence}"
            )
        if args.subgoal != expected_subgoal:
            raise ValueError(
                f"SUBGOAL '{args.subgoal}' 与 STATUS '{memory.status}' 不匹配。"
                f"期望的下一个事件是：'{expected_subgoal}'。"
            )
        memory.subgoal = args.subgoal
    else:
        memory.subgoal = expected_subgoal


def _build_dit(
    pooled_kv,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> DiTMoT:
    """根据 segmented KV 维度构造 DiT 配置并实例化。

    优先级（高 → 低）：
    1. ckpt payload 里的 dit_config —— 保证"训练时用什么配置，推理就用什么"，
       即使命令行默认值在代码迭代中漂了，也不会撞到形状不匹配。
    2. CLI 显式覆盖（patch_size/hidden_dim/n_heads/mlp_ratio/cond_dim/max_history_frames）。
       不显式传时不覆盖。
    3. num_layers **永远**等于 len(pooled_kv) —— DiT 层数和 KV 段数必须强对齐。
    4. v2 起 DiT (n_heads, head_dim) 必须严格等于 Qwen K/V (n_kv_heads, head_dim)；
       本函数在 instantiate 之前做一次形状预检，不匹配直接抛 RuntimeError。
    """

    payload = None
    saved_cfg_dict = None
    saved_args_dict = None
    if args.dit_checkpoint:
        ckpt_path = pathlib.Path(args.dit_checkpoint).resolve()
        # 先读 payload 以便从中拿 dit_config 反建模型；map_location=device 防止
        # 拉到 cuda:0 撞别人占用，让检查点直接落到当前进程使用的卡上。
        payload = torch.load(ckpt_path, map_location=device)
        saved_cfg_dict = payload.get("dit_config") if isinstance(payload, dict) else None
        # train_v1.save_checkpoint 把 vars(args) 整个存进 payload["args"]，里面包含
        # qwen_adapter_dir / qwen_adapter_merge，用于 Qwen 条件分布一致性校验。
        saved_args_dict = payload.get("args") if isinstance(payload, dict) else None

    # ---- Qwen 适配器一致性校验 ----
    # 静默风险：DiT 是用 LoRA-Qwen KV 训练的，但推理忘传 --qwen-adapter-dir，形状完全
    # 一致 strict load 也不报错，但 KV 数值分布漂掉，DiT 在错的语言条件下生成。
    # 默认严格抛错；消融实验想故意跨适配器对比时传 --allow-qwen-adapter-mismatch。
    if saved_args_dict is not None and args.dit_checkpoint:
        def _resolve_adapter(s: str) -> str:
            # 空串保持空（基础 Qwen）；非空 resolve 到绝对路径让"相对 vs 绝对"也能匹配。
            return str(pathlib.Path(s).resolve()) if s else ""

        saved_adapter = _resolve_adapter(saved_args_dict.get("qwen_adapter_dir", "") or "")
        current_adapter = _resolve_adapter(args.qwen_adapter_dir or "")
        saved_merge = bool(saved_args_dict.get("qwen_adapter_merge", True))
        current_merge = bool(args.qwen_adapter_merge)

        if saved_adapter != current_adapter:
            msg = (
                f"DiT 训练时 qwen_adapter_dir='{saved_adapter or '<base>'}'，"
                f"当前推理 qwen_adapter_dir='{current_adapter or '<base>'}'，不一致会导致 KV "
                f"分布漂移，DiT 输出无意义。"
                f" 解决：把 --qwen-adapter-dir 改成训练时同款；"
                f"故意做消融时传 --allow-qwen-adapter-mismatch。"
            )
            if not args.allow_qwen_adapter_mismatch:
                raise RuntimeError(msg)
            print(f"[dit] WARN: {msg}")
        elif saved_adapter and saved_merge != current_merge:
            # merge=True/False 数学上等价（LoRA delta 加进基础权重），数值差异在浮点精度量级；
            # 不阻断训练，只输出提示。
            print(
                f"[dit] 提示：qwen_adapter_merge 训练={saved_merge} 推理={current_merge}（数学等价，仅浮点精度差异）"
            )
        else:
            print(
                f"[dit] qwen_adapter 一致性检查通过："
                f"adapter='{current_adapter or '<base>'}' merge={current_merge}"
            )

    # 起手用 CLI / saved cfg 的全集打底，下面会被 saved_cfg 覆盖（如果有）。
    cli_kwargs = dict(
        latent_channels=4,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        num_layers=len(pooled_kv),
        cond_dim=args.cond_dim,
        max_history_frames=args.max_history_frames,
    )

    if saved_cfg_dict is not None:
        # 形状预检：ckpt 训练时 num_layers 与现在运行时不一致 → strict=True load 必然炸
        # attention 投影。提前断言给出双数字 + 原因，比 load_state_dict 的原始报错可读得多。
        # v2 不再有 language_kv_input_dim 字段；K/V 形状一致性由 DiTMoT.forward 内部断言。
        saved_layers = saved_cfg_dict.get("num_layers")
        runtime_layers = cli_kwargs["num_layers"]
        if saved_layers is not None and saved_layers != runtime_layers:
            raise RuntimeError(
                f"DiT ckpt num_layers={saved_layers}（训练时）≠ 运行时 KV 段数={runtime_layers}。"
                f" 解决：训练与推理保持同样的 --num-layers，"
                f"或修改 qwen_kv 段数与之对齐。"
            )

        # ckpt 存的字段以它为准；CLI 没显式提到的就接受 saved 值。
        merged = dict(saved_cfg_dict)
        # 兼容旧 v1 ckpt：v1 cfg 里有 language_kv_input_dim 字段，v2 DiTMoTConfig 已不接受 -> 弹掉。
        merged.pop("language_kv_input_dim", None)
        merged["num_layers"] = runtime_layers
        # ckpt 没存 latent_channels 时（旧 ckpt）保留 CLI 默认值。
        merged.setdefault("latent_channels", cli_kwargs["latent_channels"])
        cfg = DiTMoTConfig(**merged)
        print(f"[dit] 已从检查点 dit_config 重建配置：{ckpt_path}")
    else:
        cfg = DiTMoTConfig(**cli_kwargs)
        if args.dit_checkpoint:
            # 旧检查点（只存 state_dict 没存 cfg）警告：strict load 可能撞形状，给个提示。
            print(f"[dit] 警告：检查点里没有 dit_config，将退回命令行参数。"
                  f"如果训练时用了非默认几何配置，加载时可能形状不匹配。")

    # v2 强校验：DiT (n_heads, head_dim) 必须严格 = Qwen pooled K/V 形状。
    if pooled_kv:
        k0, _ = pooled_kv[0]
        kv_n_heads = int(k0.shape[1])
        kv_head_dim = int(k0.shape[3])
        if cfg.n_heads != kv_n_heads or (cfg.hidden_dim // cfg.n_heads) != kv_head_dim:
            raise RuntimeError(
                f"DiT cfg (n_heads={cfg.n_heads}, head_dim={cfg.hidden_dim // cfg.n_heads}) "
                f"与运行时 Qwen K/V (n_kv_heads={kv_n_heads}, head_dim={kv_head_dim}) 不匹配；"
                "v2 要求严格相同。请确认 Qwen 模型 / DiT cfg 一致（默认 8×128）。"
            )

    model = DiTMoT(cfg).to(device=device, dtype=dtype)

    if payload is not None:
        # v2: 默认走 EMA 权重（推理质量更稳）；ckpt 没存就回退裸 dit_state_dict。
        # 早期手工保存的裸 state_dict 没有 dict wrapper，走 else 分支。
        if isinstance(payload, dict) and args.use_ema and payload.get("ema_state_dict") is not None:
            state_dict = payload["ema_state_dict"]
            print(f"[dit] 使用 EMA 权重 (decay={payload.get('ema_decay', 'unknown')})")
        else:
            state_dict = payload.get("dit_state_dict", payload) if isinstance(payload, dict) else payload
            if isinstance(payload, dict) and args.use_ema:
                print("[dit] WARN: 检查点无 ema_state_dict；回退裸 DiT 权重")
        model.load_state_dict(state_dict, strict=True)
        print(f"[dit] 已加载检查点：{pathlib.Path(args.dit_checkpoint).resolve()}")
    else:
        print("[dit] 未提供 --dit-checkpoint；使用随机初始化的 DiT")

    print(
        f"[dit] hidden={cfg.hidden_dim} heads={cfg.n_heads} head_dim={cfg.hidden_dim // cfg.n_heads} "
        f"layers={cfg.num_layers} patch={cfg.patch_size}"
    )
    return model


# --------------------------------------------------------------------------- #
# Main run
# --------------------------------------------------------------------------- #


def run_once(args: argparse.Namespace) -> None:
    """单次前向 + 损失入口。整体串五步：图 → Qwen → VAE → DiT → 落盘。"""

    # step_000000 命名留给未来的多 step / 训练 loop；当前 runner 永远只写一个目录。
    save_root = pathlib.Path(args.save_root).resolve() if args.save_root else _DEFAULT_OUTPUT_ROOT
    step_dir = save_root / "step_000000"
    step_dir.mkdir(parents=True, exist_ok=True)

    # 1) 视觉输入：复用 image_io 同一套 load_lead_rgb_clip / 合成图逻辑；
    #    顺便在路径有效时自动识别 scenario 和 run_id。
    scenario, run_id, images = _prepare_images(args)
    memory = DrivingMemory.from_scenario(scenario)

    # 1.5) 若用户没显式 --status，按 (scenario, run_id, anchor) 反查真值 STATUS。
    # 不查直接用 DrivingMemory.from_scenario(scenario) 的默认 "initial" 会让 prompt
    # 在中后段 anchor 上撒谎（告诉 Qwen "你现在是 initial"），下游 KV 全错。
    # 查不到（路径无效 / anchor 不在已知区间）走 warning 兜底，让用户明确看到。
    if not args.status and run_id and args.route_dir:
        try:
            kf_index = KeyframeIndex.load(
                pathlib.Path(args.keyframes_json) if args.keyframes_json else None
            )
            derived_status = kf_index.find_status_for_anchor(scenario, run_id, args.anchor)
            if derived_status:
                memory.status = derived_status
                # subgoal 同步推到下一个事件，与数据构建器行为对齐。
                next_ev = memory._next_event_after(memory.status)
                if next_ev:
                    memory.subgoal = next_ev
                print(
                    f"[runner] auto-derived STATUS={memory.status} SUBGOAL={memory.subgoal} "
                    f"from (scenario={scenario}, run_id={run_id}, anchor={args.anchor})"
                )
            else:
                print(
                    f"[runner] WARN: 反查 STATUS 失败（run/anchor 在 keyframes JSON 中无匹配），"
                    f"沿用默认 STATUS='{memory.status}'。若 anchor={args.anchor} 不在 initial 段，"
                    "prompt 真值会撒谎，请显式传 --status。"
                )
        except FileNotFoundError as e:
            print(f"[runner] WARN: 无法加载 keyframes JSON ({e})，跳过 STATUS 反查")

    _apply_memory_overrides(memory, args)
    print(
        f"[runner] scenario={scenario} run_id={run_id or '<none>'} "
        f"STATUS={memory.status} SUBGOAL={memory.subgoal}"
    )

    # 2) teacher-forced Qwen prefill + segmented KV。
    # 这里仍然复用 §14.10 范式 A 的 engine 类，但传 max_gen_tokens=0 / do_sample=False，
    # 因为我们根本不进入 decode 路径——只用它的 load / prepare_inputs / prefill。
    # cache_system_prompt=False：teacher-forced 的 system 文本与范式 A 不同，
    # 跨 runner 共享 system cache 反而会触发 fallback 多花时间。
    engine = LocalQwen3VLInstructEngine(
        checkpoint_dir=pathlib.Path(args.checkpoint_dir).resolve(),
        device=args.device,
        torch_dtype=args.qwen_dtype,
        max_gen_tokens=0,
        temperature=0.0,
        do_sample=False,
        save_cache=False,
        cache_system_prompt=False,
    )
    # 显式 load + 可选 attach LoRA：teacher_forced_prefill 内部也会 lazy load，
    # 但 attach_lora_adapter 必须在 load 之后调，所以这里先把顺序固定下来。
    engine.load()
    if args.qwen_adapter_dir:
        engine.attach_lora_adapter(args.qwen_adapter_dir, merge=args.qwen_adapter_merge)

    # num_segments 与 DiT layers 强绑定（PLAN §0 默认 12）；后面 _build_dit 会用
    # len(prefill.pooled_kv) 作为 DiT num_layers，所以这俩永远一致。
    prefill = teacher_forced_prefill(
        engine=engine,
        memory=memory,
        images=images,
        num_segments=args.num_layers,
        kv_segment_mode=args.qwen_kv_segment_mode,
    )
    print(
        f"[qwen] prefill seq_len={prefill.seq_len} n_kv_heads={prefill.n_kv_heads} "
        f"head_dim={prefill.head_dim} qwen_layers={prefill.num_qwen_layers} "
        f"kv_segments={len(prefill.pooled_kv)} mode={prefill.kv_segment_mode}"
    )

    # 3) 冻结 VAE：加载并编出历史帧与（可选）目标关键帧 latent。
    # 默认路径 default_vae_paths() 指向 AutoMoT/vae_standalone/{config,weights}/...，
    # 不通过 sys.path 之外的任何路径访问；远程同步过去就能直接用。
    cfg_path, weights_path = default_vae_paths()
    vae = FrozenVAE.load(
        config_path=cfg_path,
        weights_path=weights_path,
        # VAE 与 Qwen 必须在同一个 device，否则 z_history / pooled_kv 跨设备时还得 .to()。
        device=engine.device,
        dtype=args.vae_dtype,
    )
    # images 是 oldest->newest 排序；DiT 直接看整段历史 latent，最后一帧是当前 anchor。
    z_history = vae.encode(images).unsqueeze(0)
    print(f"[vae] z_history 形状={tuple(z_history.shape)} dtype={z_history.dtype}")

    # 目标真值：从 keyframes_all_scenarios.json 查 subgoal 对应帧；找不到 -> None。
    target_frame = _resolve_target_keyframe(args, scenario, run_id, memory.subgoal)
    if target_frame is not None and target_frame <= args.anchor:
        raise ValueError(
            f"target_frame 必须在未来：target_frame={target_frame} <= anchor={args.anchor}"
        )
    z_target: Optional[torch.Tensor] = None
    target_meta: Dict[str, Any] = {"subgoal_event": memory.subgoal, "frame_idx": target_frame}
    if target_frame is not None and args.route_dir:
        try:
            # 与 image_io.load_lead_rgb_clip 同一套 cv2 解码路径，避免不同 JPEG
            # 解码器在像素级有微小差异污染 z_target。
            keyframe_img = load_keyframe_rgb(args.route_dir, target_frame)
            z_target = vae.encode([keyframe_img])
            target_meta["latent_shape"] = list(z_target.shape)
            print(f"[vae] z_target 帧={target_frame} 形状={tuple(z_target.shape)}")
        except Exception as e:
            # 单条样本失败不应该崩溃整个 runner；记录到 target_meta 后继续走 fallback。
            print(f"[vae] 读取目标关键帧失败：{e}")
            target_meta["error"] = str(e)

    # 4) DiT 构造 + 一次前向。
    # DiT 的 num_layers 由 KV 段数决定（与 args.num_layers 强一致）。
    # v2 起 (n_heads, head_dim) 直接 = Qwen (8, 128)，_build_dit 内部会做形状强校验。
    dit_dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dit_dtype = dit_dtype_map.get(args.dit_dtype, torch.float32)
    dit = _build_dit(prefill.pooled_kv, args, device=engine.device, dtype=dit_dtype)

    # 把所有视觉张量与语言 KV 统一到 dit_dtype，避免 attention 内部混合 dtype
    # 报 RuntimeError。Qwen prefill 默认 bf16，DiT 默认 fp32，必须在这里桥接。
    z_history_in = z_history.to(dtype=dit_dtype)
    pooled_kv_in = [
        (k.to(dtype=dit_dtype), v.to(dtype=dit_dtype))
        for k, v in prefill.pooled_kv
    ]

    if z_target is not None:
        z1 = z_target.to(dtype=dit_dtype)
    else:
        # 兜底：没有真值时随机一个 z1 做形状/数值通路验证。
        # 损失数字在这条路径下**不可信**，只能用来判断前向是否能跑通。
        z1 = torch.randn_like(z_history_in[:, -1])
        target_meta.setdefault("note", "没有真实目标关键帧；损失只是冒烟测试数字")

    # flow matching 采 (z_t, z0, t, v_target)。每次调用 t 和 z0 都重新随机，所以
    # 跑两次 runner 看到的损失数值会不同，这是正常的。
    batch = sample_flow_batch(z1=z1)
    # DiT 一次前向：视觉流 = (z_t, z_history)，语言流 = pooled_kv_in。
    dit.eval()
    with torch.no_grad():
        v_pred = dit(batch.z_t, z_history_in, batch.t, pooled_kv_in)
    # 损失是单个标量；当前 runner 不反传，只看数值是否合理。
    loss = flow_matching_loss(v_pred, batch.v_target)
    print(f"[dit] v_pred 形状={tuple(v_pred.shape)} 损失={float(loss):.6f}")

    # 5) 落盘 step.json。
    record = {
        "step_idx": 0,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "scenario": scenario,
        "run_id": run_id,
        "memory": memory.to_dict(),
        "status_description": EVENT_DESCRIPTIONS.get(memory.status, memory.status),
        "subgoal_description": EVENT_DESCRIPTIONS.get(memory.subgoal, memory.subgoal),
        "checkpoint_dir": str(engine.checkpoint_dir),
        # 适配器信息：未传时 dir 为空字符串、attached=False。基础模型与 LoRA 实验对比时
        # 这两个字段是 step.json 里最关键的可追溯线索（KV / 损失差异都跟它强相关）。
        "qwen_adapter": {
            "dir": args.qwen_adapter_dir or "",
            "attached": bool(args.qwen_adapter_dir),
            "merge": bool(args.qwen_adapter_merge) if args.qwen_adapter_dir else None,
        },
        "num_images": len(images),
        "qwen_kv": {
            "seq_len": prefill.seq_len,
            "n_kv_heads": prefill.n_kv_heads,
            "head_dim": prefill.head_dim,
            "num_qwen_layers": prefill.num_qwen_layers,
            "segment_mode": prefill.kv_segment_mode,
            "segmented": summarize_pooled_kv(prefill.pooled_kv),
        },
        "dit_config": {
            "hidden_dim": args.hidden_dim,
            "n_heads": args.n_heads,
            "num_layers": len(prefill.pooled_kv),
            "patch_size": args.patch_size,
            "mlp_ratio": args.mlp_ratio,
            "cond_dim": args.cond_dim,
            "max_history_frames": args.max_history_frames,
            "checkpoint": args.dit_checkpoint,
        },
        "latent_shapes": {
            "z_history": list(z_history.shape),
            "z_t": list(batch.z_t.shape),
            "v_pred": list(v_pred.shape),
        },
        "target": target_meta,
        "loss": float(loss),
    }
    (step_dir / "step.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[output] {step_dir / 'step.json'}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="真值强制 Qwen3-VL → DiT-MoT → 流匹配单步 runner")
    # Qwen / engine
    p.add_argument("--checkpoint-dir", type=str, default=str(_CHECKPOINT_DIR))
    p.add_argument("--device", default="auto")
    p.add_argument("--qwen-dtype", choices=["bfloat16", "float16", "float32", "auto"], default="bfloat16")
    # LoRA / PEFT adapter（与 train_v1 / eval_v1 同口径）。
    p.add_argument("--qwen-adapter-dir", type=str, default="",
                   help="可选 LoRA / PEFT 适配器目录；为空则跑基础 Qwen。"
                        " 与训练 / 评测同款，确保 KV 分布一致。")
    p.add_argument("--qwen-adapter-merge", action="store_true", default=True)
    p.add_argument("--no-qwen-adapter-merge", dest="qwen_adapter_merge", action="store_false")
    p.add_argument("--allow-qwen-adapter-mismatch", action="store_true", default=False,
                   help="允许 DiT ckpt 训练时的 qwen_adapter_dir 与当前 CLI 不一致；"
                        " 仅消融实验使用；默认抛错，防止 KV 分布漂移导致静默错误生成。")
    # Frames
    p.add_argument("--scenario", type=str, default="MergerIntoSlowTraffic")
    p.add_argument("--status", type=str, default=None, help="覆盖 memory.status；不传则用 DrivingMemory 默认 'initial'")
    p.add_argument("--subgoal", type=str, default=None, help="覆盖 memory.subgoal；必须等于 STATUS 的下一个事件")
    p.add_argument("--num-frames", type=int, default=4)
    p.add_argument(
        "--route-dir",
        type=str,
        default="/datashare/IOL4SGH/data/data/Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46",
    )
    p.add_argument("--anchor", type=int, default=12)
    p.add_argument("--rgb-frame-step", type=int, default=1)
    p.add_argument("--run-id", type=str, default=None,
                   help="覆盖自动从 route_dir 推断的 run_id；keyframes 查询用得到")
    # Target keyframe
    p.add_argument("--keyframes-json", type=str, default=str(_DEFAULT_KEYFRAMES_JSON),
                   help="keyframes_all_scenarios.json 路径")
    p.add_argument("--target-frame", type=int, default=None,
                   help="显式指定目标关键帧索引；不传则从 keyframes JSON 查")
    # VAE
    p.add_argument("--vae-dtype", type=str, default="float32",
                   help="VAE 内部 dtype；vae_only.yaml 默认关闭自动混精，所以这里推荐 float32")
    # DiT —— v2 默认值对齐 Qwen3-VL-4B-Instruct K/V (8, 128)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--num-layers", type=int, default=12, help="Qwen KV 分段数 = DiT 层数")
    p.add_argument("--cond-dim", type=int, default=256)
    p.add_argument("--max-history-frames", type=int, default=8)
    p.add_argument("--qwen-kv-segment-mode",
                   choices=["concat_layers", "select_last", "mean"],
                   default="select_last")
    p.add_argument("--dit-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--dit-checkpoint", type=str, default=None,
                   help="可选：加载 train_v1.py 保存的 latest.pt 或 checkpoint-*/goalgen_v1.pt")
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True,
                   help="加载 ckpt 时优先使用 EMA 权重；--no-use-ema 走原始 DiT 权重做对照。")
    # Output
    p.add_argument("--save-root", type=str, default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_once(args)


if __name__ == "__main__":
    main()
