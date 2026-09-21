"""单图 prompt、图片身份、M-RoPE 起点与缓存隔离，不加载真实 Qwen。"""
from types import SimpleNamespace
import pytest
import torch
from PIL import Image

from qwen3vl_local.action_prior.image_condition import current_prior_prompt, current_base_prefill
from qwen3vl_local.action_prior.runtime import PriorEngine
from qwen3vl_local.action_prior.text_cache import TextCache
from qwen3vl_local.action_prior import prompts, phase2_v3_prompts
from qwen3vl_local.sft_new_loop_phase1 import prompts as p1
from qwen3vl_local.sft_new_loop_phase2 import prompts as p2
from qwen3vl_local.action_expert_ablation import common


@pytest.mark.parametrize("module,phase", [(p1, 1), (p2, 2), (phase2_v3_prompts, 2)])
def test_current_prior_prompts_preserve_questions_without_claiming_history(module, phase):
    system, prompt = current_prior_prompt(module, phase, None)
    assert "one current stitched" in system and "one current stitched" in prompt
    for unavailable in ("four-frame history", "older frames", "older-frame", "across frames", "across the history", "every history frame"):
        assert unavailable not in system + prompt
    assert "Output exactly these lines and nothing else:" in prompt
    if phase == 1:
        for fact in ("HIGHWAY", "STATIC_OBSTACLE", "VULNERABLE", "TRAFFIC_LIGHT_ABNORMAL", "RS1"):
            assert fact in prompt
    else:
        assert "INVALID_EVENT_CONTEXT" in prompt
        assert "single-frame proximity cue" in prompt


def fake_engine():
    calls = []
    def messages(system, user, images):
        calls.append((system, user, images[:]))
        return [dict(role="system", content=system), dict(role="user", content=user)]
    engine = SimpleNamespace(
        build_messages=messages, apply_chat_template=lambda messages: str(messages),
        processor=SimpleNamespace(apply_chat_template=lambda messages, **kwargs: str(messages)),
        prepare_inputs=lambda chat, images: dict(input_ids=torch.ones(1, 20, dtype=torch.long)),
        prefill=lambda inputs: SimpleNamespace(past_key_values="cache", rope_deltas=torch.tensor([[-4]])))
    return engine, calls


def test_single_image_prefill_identity_and_offset():
    engine, calls = fake_engine()
    current = Image.new("RGB", (1152, 384), "red")
    cache, offset = current_base_prefill(SimpleNamespace(leadmot_qwen_engine=engine), [current], "nav", "Plan.")
    assert cache == "cache" and offset == 16
    assert calls[0][2] == [current] and calls[0][2][0].size == (1152, 384)
    assert "one current stitched" in calls[0][0]
    with pytest.raises(ValueError, match="exactly"):
        current_base_prefill(SimpleNamespace(leadmot_qwen_engine=engine), [current] * 4, "nav", "Plan.")


def test_prior_current_image_cache_isolated_from_four_images(tmp_path):
    engine, calls = fake_engine()
    read = []
    def labels(identity):
        read.append(identity)
        return dict(conditions={"ROAD_STRUCTURE": "R1"}, invalid={}, calls=[])
    source = SimpleNamespace(path="synthetic", rows=1, priors=labels)
    cache = TextCache(tmp_path / "cache")
    image = Image.new("RGB", (4, 4))
    for count in (4, 1):
        prior = PriorEngine(engine, dict(identity="same_test"), labels=source, text_cache=cache, rgb_frame_count=count)
        for _ in range(2):
            prior.condition([image] * count, "nav", "sample", ("S", "R", 0))
            assert len(calls[-1][2]) == count
            assert prior.last_audit["rgb_frame_count"] == count
        assert prior.last_audit["text_cache_hit"]
    assert len(read) == 2
    assert "chronological images" in calls[0][0]
    assert "one current stitched" in calls[-1][0]
    assert "chronological images" not in calls[-1][0]


@pytest.mark.parametrize("variant", ["qwen_simple", "bev_only"])
@pytest.mark.parametrize("count", [1, 4])
def test_ablation_validates_supported_image_counts(variant, count):
    args = common.parser(variant).parse_args(["--rgb-frame-count", str(count)])
    common.validate_args(args, variant)
    args.rgb_frame_count = 2
    with pytest.raises(ValueError, match="RGB"):
        common.validate_args(args, variant)


def test_single_scene_context_has_no_visible_history_instruction():
    prior = dict(conditions={}, rgb_frame_count=1, event_balanced_scene_contexts=("RE2_PRIOR_OBSTACLE",))
    assert "use visible history" not in prompts.prefill_prompt(prior, "nav")


def test_lora_prior_calls_both_phases_with_only_current_image(monkeypatch):
    from contextlib import nullcontext
    from qwen3vl_local.action_prior import runtime
    engine, calls = fake_engine()
    current = Image.new("RGB", (6, 6))
    prior = PriorEngine(engine, dict(identity="single", phase1=dict(metadata={}), phase2=dict(metadata={})),
                        labels=SimpleNamespace(path="setup", rows=0), rgb_frame_count=1)
    # 跳过真实 PEFT 权重加载，保留 condition 中的两阶段选图/提示词和最终 prefill。
    prior.labels = None
    prior.mode = lambda _: nullcontext()
    monkeypatch.setattr(runtime, "prompt_module", lambda phase, _: p1 if phase == 1 else p2)
    prior.contract["phase1"]["metadata"]["history_rgb_mode"] = "4rgb"
    prior.contract["phase2"]["metadata"]["history_rgb_mode"] = "2rgb_endpoints"
    requests = []
    def generate(system, prompt, images, history):
        requests.append((system, prompt, images))
        assert images == [current]
        assert "one current stitched" in system
        return "answer", None
    prior.generate_messages = generate
    def collect(ask, *args, **kwargs):
        for phase in (1, 2):
            ask(phase, None, ())
        return dict(conditions={}, invalid={}, calls=[])
    monkeypatch.setattr(runtime, "collect_priors", collect)
    prior.condition([current], "nav", "S/R/0")
    assert len(requests) == 2 and calls[-1][2] == [current]
    assert prior.last_audit["prior_image_policy"] == "current_only_adapted"
