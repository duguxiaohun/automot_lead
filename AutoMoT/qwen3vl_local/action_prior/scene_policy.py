"""特殊 RE 场景先验的自动策略；shell/Python 共用，恢复时保留原条件。"""

import argparse
import json
import os
from pathlib import Path

SCENE_PRIOR_POLICY = "dataset_planning_special_re_v1"
LEGACY_SCENE_PRIOR_POLICY = "legacy_explicit_scene_priors"
RE_CONTEXTS = frozenset(("RE2_NAVIGATION_TRANSITION", "RE2_PRIOR_OBSTACLE",
                         "RE2_RECOVERY_PENDING", "RE3", "RE5"))


def automatic_scene_priors(args):
    """仅干净的离线 planning 实验默认提供特殊 RE；不补回被扰动的条件。"""
    return bool(getattr(args, "dataset_priors", False)
                and getattr(args, "high_level_planning", False)
                and getattr(args, "condition_mode", "prior") == "prior"
                and float(getattr(args, "prior_noise", 0.0)) == 0.0)


def resolve_scene_priors(args):
    """新训练推导一次；恢复读取保存值，不能用新默认值改写旧 decoder 条件。"""
    if not hasattr(args, "event_balanced_scene_priors"):
        # 新 parser 显式设置 None；完全缺字段的是旧 checkpoint，历史行为为关闭。
        args.event_balanced_scene_priors = False
    if args.event_balanced_scene_priors is None:
        if getattr(args, "resume", ""):
            saved = json.loads(Path(args.resume).resolve().with_name("config.json").read_text())
            args.event_balanced_scene_priors = bool(saved.get("event_balanced_scene_priors", False))
            args.scene_prior_policy = saved.get("scene_prior_policy", LEGACY_SCENE_PRIOR_POLICY)
        else:
            args.event_balanced_scene_priors = automatic_scene_priors(args)
            args.scene_prior_policy = SCENE_PRIOR_POLICY
    else:
        # eval/checkpoint 与旧配置没有自动策略字段时，保持其原来的输入范围。
        args.scene_prior_policy = getattr(args, "scene_prior_policy", LEGACY_SCENE_PRIOR_POLICY)
    if args.scene_prior_policy not in (SCENE_PRIOR_POLICY, LEGACY_SCENE_PRIOR_POLICY):
        raise ValueError("unknown saved scene prior policy")
    return args.event_balanced_scene_priors


def scene_contexts(args, contexts):
    """自动模式只补特殊 RE，UE 仍来自实际 Phase1/2；旧合同保持旧范围。"""
    if not getattr(args, "event_balanced_scene_priors", False):
        return ()
    return tuple(str(c) for c in contexts if c and (
        getattr(args, "scene_prior_policy", LEGACY_SCENE_PRIOR_POLICY) != SCENE_PRIOR_POLICY
        or c in RE_CONTEXTS))


def main():
    """shell 在构建索引前查询同一策略；CLI 按顺序覆盖显式环境值。"""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("dataset_priors", "high_level_planning"):
        value = os.environ.get(name.upper(), "0")
        if value not in ("0", "1"):
            parser.error(f"{name.upper()} must be 0 or 1")
        parser.add_argument("--" + name.replace("_", "-"),
                            action=argparse.BooleanOptionalAction, default=value == "1")
    parser.add_argument("--prior-noise", type=float, default=float(os.environ.get("PRIOR_NOISE", "0")))
    parser.add_argument("--condition-mode", default="prior")
    args, extra = parser.parse_known_args()
    if "EVENT_BALANCED_SCENE_PRIORS" in os.environ or any(
        item.split("=", 1)[0] in ("--event-balanced-scene-priors", "--no-event-balanced-scene-priors")
        for item in extra
    ):
        parser.error("event-balanced-scene-priors was removed; dataset-priors + high-level-planning "
                     "automatically uses confirmed special RE when prior-noise is zero")
    print(int(automatic_scene_priors(args)))


if __name__ == "__main__":
    main()
