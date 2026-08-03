"""SFT base 断点续训源码合同测试。

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
    ]
    required_shell = [
        'RESUME_FROM_CHECKPOINT="${2:-${RESUME_FROM_CHECKPOINT:-}}"',
        'RESUME_FROM_CHECKPOINT="${MODE_ARG}"',
        'OUTPUT_DIR="$(dirname "${RESUME_FROM_CHECKPOINT}")"',
        'ln -sfn "$(basename "${OUTPUT_DIR}")" "${OUTPUT_DIR_BASE}/latest"',
        'EXTRA_ARGS+=("--resume-tb-trim")',
        'COMMON_ARGS+=("--resume-from-checkpoint" "${RESUME_FROM_CHECKPOINT}")',
    ]
    for needle in required_train:
        assert needle in train_src, needle
    for needle in required_shell:
        assert needle in shell_src, needle
    print(f"[test_train_resume] ok ({root.name})")


if __name__ == "__main__":
    main()
