"""SFT base simple 断点续训源码合同测试。

本测试故意不 import train.py；Windows 本地轻量环境可能无法初始化 torch DLL。
用源码 needle 守住 resume 入口、step 语义和 trainer_state 保存。
"""

from __future__ import annotations

import pathlib


def main() -> None:
    """验证 train.py/train.sh 具备从 checkpoint 继续训练的关键合同。"""

    root = pathlib.Path(__file__).resolve().parents[3]
    train_py = pathlib.Path(__file__).with_name("train.py")
    train_sh = pathlib.Path(__file__).with_name("train.sh")
    train_src = train_py.read_text(encoding="utf-8")
    shell_src = train_sh.read_text(encoding="utf-8")
    required_train = [
        'p.add_argument("--resume-from-checkpoint", type=str, default=None)',
        "resume_dir = pathlib.Path(args.resume_from_checkpoint).expanduser().resolve() if args.resume_from_checkpoint else None",
        "_apply_resume_config(args, resume_config)",
        "_load_adapter_weights(bundle, resume_dir)",
        'return checkpoint_dir / "trainer_state.pt"',
        "def _trim_tensorboard_for_resume(tb_dir: pathlib.Path, *, resume_step: int) -> None:",
        'p.add_argument("--resume-tb-trim", action=argparse.BooleanOptionalAction, default=True)',
        '_trim_tensorboard_for_resume(output_dir / "tb", resume_step=int(global_step))',
        'archive_dir = tb_dir.parent / "tb_resume_archive"',
        "_save_trainer_state(",
        'trainer_state = _load_trainer_state(resume_dir, device=device) if resume_dir is not None else None',
        'optimizer.load_state_dict(trainer_state["optimizer"])',
        'scheduler.load_state_dict(trainer_state["scheduler"])',
        "global_step = resume_step",
        "planned_new_steps = int(args.max_steps) if int(args.max_steps) > 0 else steps_per_epoch * int(args.num_epochs)",
        "total_steps = resume_step + planned_new_steps",
        "if global_step >= total_steps:",
        'p.add_argument("--fourbin-routes-per-batch", type=int, default=16)',
        'p.add_argument("--joint-target-balance-count", type=int, default=8)',
        'p.add_argument("--ue-event-loss-weight", type=float, default=1.0)',
        'p.add_argument("--ue-frame-repeat", type=int, default=1)',
        'p.add_argument("--ue-repeat-mode", choices=["none", "fixed", "inverse_sqrt"], default="none")',
        'p.add_argument("--regular-repeat-mode", choices=["none", "fixed", "inverse_sqrt"], default="none")',
        'p.add_argument("--joint-balance-drop-majority", action=argparse.BooleanOptionalAction, default=False)',
        '"fourbin_routes_per_batch": int(args.fourbin_routes_per_batch)',
        "pending_routes: List[SequenceRow] = []",
        "stats.fourbin_highway_ue",
        "memory/early_ue_event_re_rate_last_batch",
        "[memory] early_ue_effective ",
    ]
    required_shell = [
        'RESUME_FROM_CHECKPOINT="${2:-${RESUME_FROM_CHECKPOINT:-}}"',
        'RESUME_FROM_CHECKPOINT="${MODE_ARG}"',
        'OUTPUT_DIR="$(dirname "${RESUME_FROM_CHECKPOINT}")"',
        'ln -sfn "$(basename "${OUTPUT_DIR}")" "${OUTPUT_DIR_BASE}/latest"',
        'EXTRA_ARGS+=("--resume-tb-trim")',
        'COMMON_ARGS+=("--resume-from-checkpoint" "${RESUME_FROM_CHECKPOINT}")',
        '--ue-event-loss-weight "${UE_EVENT_LOSS_WEIGHT:-1.0}"',
        '--ue-frame-repeat "${UE_FRAME_REPEAT:-1}"',
        '--ue-repeat-mode "${UE_REPEAT_MODE:-none}"',
        '--regular-repeat-mode "${REGULAR_REPEAT_MODE:-none}"',
        '--joint-target-balance-count "${JOINT_TARGET_BALANCE_COUNT:-8}"',
        'if [[ "${JOINT_BALANCE_DROP_MAJORITY:-0}" == "1" ]]; then',
    ]
    for needle in required_train:
        assert needle in train_src, needle
    for needle in required_shell:
        assert needle in shell_src, needle
    forbidden = [
        "transition_frame_repeat",
        "transition_frame_window",
        "transition_label_mode",
        "transition_repeat_mode",
        "--transition-",
        "--train-sampling-mode",
        "--segment-length",
        "--segments-per-route",
        "--negative-segment-ratio",
        "q2_memory_override",
        "resample_event_for_q2",
        "resample_event_memory_for_q2",
        "selected_pos",
    ]
    for needle in forbidden:
        assert needle not in train_src, needle
        assert needle not in shell_src, needle
    print(f"[test_train_resume] ok ({root.name})")


if __name__ == "__main__":
    main()


