"""v12 场景目的、提示词身份及默认入口的 CPU 回归，不读取 RGB 或模型。"""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
from pathlib import Path

import pytest

from qwen3vl_local.sft_new_loop_phase3 import history_rgb, prompts
from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID


@pytest.mark.parametrize("mode", ["binary", "choice"])
@pytest.mark.parametrize("context_id", list(CONTEXT_BY_ID))
def test_purpose_is_scoped_conditional_and_independent_of_target(mode, context_id):
    """每题只加对应目的；STOP/RESUME/invalid 答案均不能改变可见输入。"""
    spec = prompts.make_prompt_spec(
        variant="all_random_order", answers={key: key == "STOP" for key in prompts.ACTION_KEYS}, seed_key="purpose",
        context_id=context_id, road_structure=CONTEXT_BY_ID[context_id].allowed_rs[0],
        action_output_mode=mode, current_speed_mps=5,
    )
    prompt = prompts.build_action_prompt(spec=spec)
    changed = replace(spec, invalid_context=True,
                      questions=tuple(replace(q, answer=not q.answer) for q in spec.questions))
    assert prompt == prompts.build_action_prompt(spec=changed)
    scene = prompt.split("[SCENE_CONTEXT]", 1)[1].split("[/SCENE_CONTEXT]", 1)[0]
    for action in CONTEXT_BY_ID[context_id].action_keys:
        text = prompts.action_description(context_id, action)
        assert prompt.count(text) == 1
        assert text not in scene
    assert "High-level purpose" not in scene
    assert "a scene alone does not establish" in prompt
    assert "four-frame history" in prompt
    assert "future_speeds" not in prompt
    target = prompts.build_action_target(spec)
    assert prompts.parse_action_output(target, spec=spec) == prompts.spec_answers(spec)


@pytest.mark.parametrize("mode", ["binary", "choice"])
@pytest.mark.parametrize("rgb", ["4rgb", "2rgb_endpoints"])
def test_purpose_edit_invalidates_prompt_contract(monkeypatch, mode, rgb):
    """目的属于模型实际输入，修改后两种输出/图像模式都必须拒绝旧指纹。"""
    before = prompts.action_prompt_sha256(action_output_mode=mode, history_rgb_mode=rgb)
    monkeypatch.setitem(prompts.CONTEXT_ACTION_DESCRIPTIONS["STATIC_BLOCKAGE"], "STOP", "Changed purpose.")
    assert before != prompts.action_prompt_sha256(action_output_mode=mode, history_rgb_mode=rgb)


def _cpu_functions(filename, names):
    """提取实际纯函数，避开 train/eval 模块导入时的 torch/GPU 初始化。"""
    path = Path(__file__).with_name(filename)
    tree = ast.parse(path.read_text())
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(argparse=argparse, _AUTOMOT_ROOT=path.parents[2],
                     DEFAULT_HISTORY_RGB_MODE=history_rgb.DEFAULT_HISTORY_RGB_MODE,
                     HISTORY_RGB_MODES=history_rgb.HISTORY_RGB_MODES,
                     DEFAULT_ACTION_OUTPUT_MODE=prompts.DEFAULT_ACTION_OUTPUT_MODE,
                     validate_action_output_mode=prompts.validate_action_output_mode,
                     validate_history_rgb_mode=history_rgb.validate_history_rgb_mode)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


def test_train_defaults_and_explicit_binary_override(monkeypatch):
    """实际 Python CLI 默认四帧单选，显式两帧/binary 仍能覆盖。"""
    parse = _cpu_functions("train.py", {"parse_args"})["parse_args"]
    monkeypatch.setattr("sys.argv", ["train.py"])
    args = parse()
    assert (args.history_rgb_mode, args.action_output_mode) == ("4rgb", "choice")
    assert "data_v21" in args.index
    monkeypatch.setattr("sys.argv", ["train.py", "--history-rgb-mode", "2rgb_endpoints", "--action-output-mode", "binary"])
    args = parse()
    assert (args.history_rgb_mode, args.action_output_mode) == ("2rgb_endpoints", "binary")


def test_eval_defaults_never_override_persisted_or_legacy_modes():
    """新 base 默认值不能悄悄改变旧 adapter 的图像/多标签语义。"""
    functions = _cpu_functions("eval.py", {"_resolve_history_rgb_mode", "_resolve_action_output_mode"})
    rgb = functions["_resolve_history_rgb_mode"]
    action = functions["_resolve_action_output_mode"]
    assert rgb(None, None)[0] == "4rgb"
    assert action(None, None)[0] == "choice"
    assert rgb(None, {})[0] == "4rgb"
    assert action(None, {})[0] == "binary"
    cfg = {"history_rgb_mode": "4rgb", "action_output_mode": "binary"}
    assert rgb(None, cfg)[0] == "4rgb"
    assert action(None, cfg)[0] == "binary"
    with pytest.raises(ValueError, match="conflicts"):
        action("choice", cfg)
