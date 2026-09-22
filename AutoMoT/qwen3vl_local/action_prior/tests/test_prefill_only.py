"""无真实权重验证直接图文 KV、生成开关、缓存隔离和合同恢复。"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pytest
import torch
from PIL import Image
from qwen3vl_local.action_prior import prompts
from qwen3vl_local.action_prior.config import parser, training_plan, validate_args
from qwen3vl_local.action_prior.runtime import PriorEngine
from qwen3vl_local.action_prior.text_cache import TextCache


@pytest.mark.parametrize("review", [False, True])
@pytest.mark.parametrize("high_level", [False, True])
def test_default_encodes_only_images_and_prior_prompt_and_isolates_cache(tmp_path, review, high_level):
    """默认无 decode/复核/fallback，命中缓存仍编码四图；同缓存开启摘要必须重新生成。"""
    images = [Image.new("RGB", (3, 3), (i, 0, 0)) for i in range(4)]
    navigation = "Current velocity is 4 m/s. Predict the driving actions now."
    labels_read, transcripts, generation_calls = [], [], []

    def read_labels(identity):
        """只提供确认事实；审计私有信息不得进入提示词。"""
        labels_read.append(identity)
        return dict(conditions={"ROAD_STRUCTURE": "R1", "UE3": "YES", "UE5": "NO"},
                    invalid={}, calls=[], private="hidden audit", future_waypoints=[[999, 999]])

    def build_messages(system, user, selected):
        """保留图像对象和顺序，模拟正式 engine 的图文消息边界。"""
        assert selected == images
        return [{"role": "system", "content": system},
                {"role": "user", "content": [*selected, user]}]

    def template(messages, **kwargs):
        """直接模式和摘要模式均关闭额外 generation prompt。"""
        assert kwargs == dict(tokenize=False, add_generation_prompt=False)
        transcripts.append(messages)
        return str(messages)

    cache_object = object()
    prefills = []

    def prefill(inputs):
        """模拟完整图文 forward 的 KV 和 M-RoPE 位置偏移。"""
        prefills.append(inputs)
        return SimpleNamespace(past_key_values=cache_object, rope_deltas=torch.tensor([[-3]]))

    def prepare(text, selected):
        """四图仍完整传给 processor，不因没有文本生成而丢失。"""
        assert selected == images
        return {"input_ids": torch.arange(len(text)).reshape(1, -1)}

    engine = SimpleNamespace(build_messages=build_messages, prepare_inputs=prepare,
                             processor=SimpleNamespace(apply_chat_template=template), prefill=prefill)
    labels = SimpleNamespace(path="synthetic", rows=1, priors=read_labels)
    cache = TextCache(tmp_path / "cache")
    runtime = PriorEngine(engine, {"identity": "same"}, labels=labels,
                          text_cache=cache, analysis_review=review, high_level_action_prior=high_level)

    def forbidden(*args, **kwargs):
        """默认路径若发生任何摘要生成就立即失败。"""
        raise AssertionError("prefill-only must not generate, review or fall back")

    runtime.generate_messages = forbidden
    for hit in (False, True):
        kv, offset = runtime.condition(images, navigation, "case", ("S", "R", 0))
        assert kv is cache_object
        assert offset == prefills[-1]["input_ids"].shape[-1] - 3
        assert runtime.last_audit["text_cache_hit"] is hit
        assert runtime.last_audit["analysis"] == ""
        assert runtime.last_audit["analysis_review_enabled"] is False
        assert runtime.last_audit["final_cache_content"] == "inputs_only"
        assert not runtime.last_audit["analysis_fallback"]
    assert len(labels_read) == 1 and len(prefills) == 2
    assert transcripts[0] == transcripts[1]
    assert [m["role"] for m in transcripts[0]] == ["system", "user"]
    user = transcripts[0][1]["content"]
    assert user[:4] == images
    assert "4 m/s" in user[-1]
    assert runtime.last_audit["high_level_action_prior"] is high_level
    assert "sustain speed increases" not in user[-1]
    assert transcripts[0][0]["content"] == prompts.system_prompt()
    assert prompts.EVENT_DESCRIPTIONS["UE3"] in user[-1]
    for forbidden_text in ("YES", "NO", "UE3", "UE5", "hidden audit", "999", "Write the concise", "Predict the driving actions"):
        assert forbidden_text not in user[-1]

    talk = PriorEngine(engine, {"identity": "same"}, labels=labels,
                       text_cache=cache, analysis_review=review, generate_analysis=True,
                       high_level_action_prior=high_level)
    draft = "Another vehicle enters the immediate corridor; maintain clearance along the route."

    def generate(system, prompt, selected, **kwargs):
        """生成分支仍保留摘要及可选独立文本复核。"""
        generation_calls.append(system)
        text = json.dumps(dict.fromkeys(prompts.REVIEW_KEYS, True)) if system == prompts.REVIEW_SYSTEM else draft
        return text, SimpleNamespace(decode_steps=[SimpleNamespace(is_eos=True)])

    talk.generate_messages = generate
    for hit in (False, True):
        talk.condition(images, navigation, "case", ("S", "R", 0))
        assert talk.last_audit["text_cache_hit"] is hit
        assert talk.last_audit["analysis"] == draft
        assert talk.last_audit["final_cache_content"] == "inputs_and_analysis"
        assert transcripts[-1][-1] == {"role": "assistant", "content": draft}
        assert transcripts[-1][0]["content"] == prompts.system_prompt(generate_analysis=True)
    assert len(labels_read) == 2 and len(prefills) == 4
    assert len(generation_calls) == 1 + int(review)
    assert generation_calls[0] == prompts.system_prompt(generate_analysis=True)

    other = PriorEngine(engine, {"identity": "same"}, labels=labels, text_cache=cache,
                        high_level_action_prior=not high_level)
    other.generate_messages = forbidden
    other.condition(images, navigation, "case", ("S", "R", 0))
    assert not other.last_audit["text_cache_hit"]
    assert len(labels_read) == 3
    assert transcripts[-1] == transcripts[0]  # 无动作时开关不改自然事实/系统文案。


@pytest.mark.parametrize("option", ["generate_analysis"])
def test_mode_is_in_identity_and_cannot_be_changed_as_prior_source(tmp_path, monkeypatch, option):
    """切换 KV 模式必须训练新 decoder，允许标签来源变化不能绕过模式检查。"""
    from test_dataset_priors import dataset_args
    from qwen3vl_local.action_prior.config import build_contract
    from qwen3vl_local.action_prior.contracts import decoder_identity, require_contract
    from qwen3vl_local.action_prior import provenance

    # 本项只测模式合同，固定无关的源码身份以免依赖本机没有的只读 runner。
    monkeypatch.setattr(provenance, "execution_fingerprint", lambda: {"fixture": "fixed"})

    args = dataset_args(tmp_path)
    direct = build_contract(args)
    setattr(args, option, True)
    talk = build_contract(args)
    assert direct["identity"] != talk["identity"]
    assert decoder_identity(direct) != decoder_identity(talk)
    with pytest.raises(ValueError):
        require_contract(direct, talk, allow_prior_source_change=True)


@pytest.mark.parametrize("dataset,generate,review,mode,expected", [
    (True, False, True, "history", 0), (True, True, False, "history", 1),
    (True, True, True, "history", 2), (False, False, True, "history", 9),
    (False, True, True, "history", 11), (False, False, True, "compare", 15),
])
def test_training_plan_reports_effective_generation_budget(dataset, generate, review, mode, expected):
    """关掉摘要后，预算仅保留上游先验问答，dataset 模式为零生成。"""
    args = parser().parse_args([])
    assert args.generate_analysis is False
    args.dataset_priors, args.generate_analysis, args.analysis_review = dataset, generate, review
    args.recheck_mode = mode
    from qwen3vl_local.action_prior.tests.test_data_capacity import balanced_rows
    rows = {split: balanced_rows(split) for split in ("train", "val", "test")}
    plan = training_plan(args, rows, 1)
    assert plan["cold_generations_per_unique_frame"] == expected
    assert plan["independent_analysis_review"] == (generate and review)
    assert plan["final_base_prefills_per_presentation"] == 1


def test_saved_legacy_args_do_not_silently_take_new_default():
    """旧配置缺开关时仍按历史摘要语义检查合同，不宣称兼容旧执行指纹。"""
    args = parser().parse_args([])
    del args.generate_analysis
    args.event_balance_source_identity = {"fixture": True}
    validate_args(args)
    assert args.generate_analysis is True


def test_disabled_summary_is_not_counted_as_model_accepted():
    """无摘要是独立分组，不能因 fallback=False 被算成模型审核通过。"""
    from qwen3vl_local.action_prior.metrics import sample_groups

    audit = dict(conditions={}, invalid={}, analysis_acceptance="disabled", analysis_fallback=False)
    groups = sample_groups(audit, {})
    assert "summary_disabled" in groups
    assert "summary_model_accepted" not in groups and "summary_fallback" not in groups
    assert "confirmation/all_confirmed" in groups
