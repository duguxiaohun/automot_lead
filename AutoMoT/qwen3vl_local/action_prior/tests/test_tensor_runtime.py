"""CPU 小模型验证 adapter 隔离与轨迹 attention 的真实反向传播。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from qwen3vl_local.action_prior.runtime import PriorEngine
from qwen3vl_local.action_prior.prompts import (
    valid_analysis,
    fallback_analysis,
    analysis_prompt,
)


class TinyBase(nn.Module):
    """只验证 LoRA 代数，不模拟 Qwen 视觉能力。"""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.q_proj.weight.data.fill_(1.0)

    def forward(self, x):
        return self.q_proj(x)


def test_two_loras_disable_to_exact_base(tmp_path):
    from peft import LoraConfig, get_peft_model

    for i in (1, 2):
        model = get_peft_model(
            TinyBase(),
            LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"], bias="none"),
        )
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data.fill_(float(i))
        model.save_pretrained(tmp_path / f"p{i}")
    engine = SimpleNamespace(
        model=TinyBase(), _last_decode_state=None, _system_prompt_cache=None
    )
    contract = {f"phase{i}": {"path": str(tmp_path / f"p{i}")} for i in (1, 2)}
    runtime = PriorEngine(engine, contract)
    x = torch.ones(1, 4)
    with runtime.mode("phase1"):
        a = engine.model(x)
    with runtime.mode("phase2"):
        b = engine.model(x)
    with runtime.mode("base"):
        c = engine.model(x)
        assert torch.equal(c, TinyBase()(x))
    assert not torch.equal(a, b) and not torch.equal(a, c)
    assert not any(p.requires_grad for p in engine.model.parameters())


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
def test_frozen_inference_kv_decoder_backward(compute_dtype):
    from qwen3vl_local.leadmot import (
        LeadMoTPlanningDecoder,
        LeadMoTPlanningDecoderConfig,
    )

    cfg = LeadMoTPlanningDecoderConfig(
        hidden_size=16,
        num_kv_heads=2,
        head_dim=8,
        num_heads=2,
        num_layers=2,
        rope_type="none",
        bev_channels=4,
        bev_grid=(2, 2),
    )
    model = LeadMoTPlanningDecoder(cfg)
    with torch.inference_mode():
        cached = [(torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 8)) for _ in range(2)]
    kv = [(k.detach().clone(), v.detach().clone()) for k, v in cached]
    from qwen3vl_local.action_prior.precision import decoder_forward

    out = decoder_forward(
        model,
        dict(
            pooled_kv=kv,
            bev=torch.randn(1, 4, 2, 2),
            speed=torch.ones(1),
            target_point=torch.ones(1, 2),
            target_point_next=torch.ones(1, 2),
            final_goal=torch.ones(1, 2),
            rope_position_offset=9,
        ),
        compute_dtype,
        torch.device("cpu"),
    )
    loss = (
        out["pred_route"].square().mean() + out["pred_future_waypoints"].square().mean()
    )
    loss.backward()
    assert out["pred_route"].shape == (1, 10, 2)
    assert out["pred_future_waypoints"].shape == (1, 8, 2)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert all(not k.requires_grad for k, v in kv)


def test_conditional_flow_matching_uses_one_kv_conditioning_and_joint_point_interaction():
    """每步 vector field 复用一次 KV 条件，但任一点 x_t 可影响其余点的速度。"""
    from qwen3vl_local.action_prior.flow_matching import (
        ConditionalFlowMatchingDecoder,
        FlowMatchingConfig,
        flow_matching_loss,
        make_training_flow,
    )
    from qwen3vl_local.leadmot import LeadMoTPlanningDecoderConfig

    config = LeadMoTPlanningDecoderConfig(
        hidden_size=16, num_kv_heads=2, head_dim=8, num_heads=2, num_layers=2,
        rope_type="none", bev_channels=4, bev_grid=(2, 2),
    )
    model = ConditionalFlowMatchingDecoder(
        config, FlowMatchingConfig(time_embed_dim=8, sample_steps=3)
    )
    kv = [(torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 8)) for _ in range(2)]
    route, waypoint = torch.randn(1, 10, 2), torch.randn(1, 8, 2)
    flow = make_training_flow(route, waypoint, model.flow_config)
    calls, original = [0], model._conditioning

    def counted(**kwargs):
        calls[0] += 1
        return original(**kwargs)

    model._conditioning = counted
    kwargs = dict(
        pooled_kv=kv, bev=torch.randn(1, 4, 2, 2), speed=torch.ones(1),
        target_point=torch.ones(1, 2), target_point_next=torch.ones(1, 2),
        final_goal=torch.ones(1, 2), rope_position_offset=9,
        flow_state=flow["flow_state"], flow_time=flow["flow_time"],
        flow_sample_noise=torch.zeros(1, 18, 2),
    )
    outputs = model(**kwargs)
    loss, route_mse, waypoint_mse = flow_matching_loss(
        outputs, flow["flow_target_velocity"], 10, 0.5, 1.0
    )
    loss.backward()
    assert calls == [1]
    assert outputs["flow_velocity"].shape == (1, 18, 2)
    assert outputs["pred_route"].shape == (1, 10, 2)
    assert outputs["pred_future_waypoints"].shape == (1, 8, 2)
    assert route_mse >= 0 and waypoint_mse >= 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    # 固定 conditioning，扰动 point 0 的当前 x_t 必须改变至少一个其它点的向量场。
    model.eval()
    with torch.no_grad():
        conditioning = original(
            pooled_kv=kv, bev=kwargs["bev"], speed=kwargs["speed"],
            target_point=kwargs["target_point"], target_point_next=kwargs["target_point_next"],
            final_goal=kwargs["final_goal"], rope_position_offset=9,
        )
        before = model._velocity(conditioning, flow["flow_state"], flow["flow_time"])
        changed = flow["flow_state"].clone()
        changed[:, 0, 0] += 1.0
        after = model._velocity(conditioning, changed, flow["flow_time"])
    assert not torch.allclose(before[:, 1:], after[:, 1:])


def test_evaluation_flow_is_fixed_per_sample_and_isolated_from_global_rng():
    """同一 sample/seed 在不同进程重算同一 eps、t、ODE 噪声，不消耗训练 RNG。"""
    from qwen3vl_local.action_prior.flow_matching import FlowMatchingConfig, make_evaluation_flow

    route, waypoint = torch.ones(1, 10, 2), torch.ones(1, 8, 2)
    sample = dict(scenario="S", run_id="R", anchor=7)
    config = FlowMatchingConfig(time_embed_dim=8)
    torch.manual_seed(123)
    state = torch.get_rng_state().clone()
    first = make_evaluation_flow(route, waypoint, config, sample, 2026)
    assert torch.equal(torch.get_rng_state(), state)
    second = make_evaluation_flow(route, waypoint, config, sample, 2026)
    assert all(torch.equal(first[key], second[key]) for key in first)
    changed = make_evaluation_flow(route, waypoint, config, dict(sample, anchor=8), 2026)
    assert not torch.equal(first["flow_sample_noise"], changed["flow_sample_noise"])


def test_route_policy_generator_is_reproducible_and_isolated_from_global_rng():
    """闭环每 route 的 FM 噪声是独立序列，不依赖 Traffic Manager 或 torch 全局 RNG。"""
    from qwen3vl_local.action_prior.flow_matching import route_policy_generator

    torch.manual_seed(123)
    state = torch.get_rng_state().clone()
    first = torch.randn(2, 18, 2, generator=route_policy_generator("cpu", 7, "42"))
    second = torch.randn(2, 18, 2, generator=route_policy_generator("cpu", 7, "42"))
    other = torch.randn(2, 18, 2, generator=route_policy_generator("cpu", 7, "43"))
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_joint_fm_sampling_is_dropout_free_and_can_be_disabled_for_training():
    """默认训练只回归向量场；显式诊断采样固定噪声时不额外消耗 dropout RNG。"""
    from qwen3vl_local.action_prior.flow_matching import (
        ConditionalFlowMatchingDecoder,
        FlowMatchingConfig,
    )
    from qwen3vl_local.leadmot import LeadMoTPlanningDecoderConfig

    config = LeadMoTPlanningDecoderConfig(
        hidden_size=16, num_kv_heads=2, head_dim=8, num_heads=2, num_layers=1,
        rope_type="none", bev_channels=4, bev_grid=(2, 2), dropout=0.4,
    )
    model = ConditionalFlowMatchingDecoder(config, FlowMatchingConfig(time_embed_dim=8, sample_steps=2))
    model.train()
    conditioning = torch.randn(1, 18, 16)
    model._conditioning = lambda **kwargs: conditioning
    state = torch.get_rng_state().clone()
    first = model(flow_sample_noise=torch.zeros(1, 18, 2))
    assert torch.equal(torch.get_rng_state(), state)
    second = model(flow_sample_noise=torch.zeros(1, 18, 2))
    assert torch.equal(first["pred_route"], second["pred_route"])
    disabled = model(
        flow_state=torch.zeros(1, 18, 2), flow_time=torch.zeros(1),
        sample_trajectory=False,
    )
    assert "flow_velocity" in disabled and "pred_route" not in disabled


def test_joint_fm_cpu_bf16_train_and_eval_paths_are_supported():
    """CPU BF16 eval/no_grad 不走 PyTorch 2.3 的混合 dtype MHA fastpath 崩溃路径。"""
    from qwen3vl_local.action_prior.flow_matching import (
        ConditionalFlowMatchingDecoder,
        FlowMatchingConfig,
    )
    from qwen3vl_local.action_prior.precision import decoder_forward
    from qwen3vl_local.leadmot import LeadMoTPlanningDecoderConfig

    config = LeadMoTPlanningDecoderConfig(
        hidden_size=16, num_kv_heads=2, head_dim=8, num_heads=2, num_layers=1,
        rope_type="none", bev_channels=4, bev_grid=(2, 2), dropout=0.1,
    )
    model = ConditionalFlowMatchingDecoder(config, FlowMatchingConfig(time_embed_dim=8, sample_steps=2))
    kv = [(torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 8))]
    kwargs = dict(
        pooled_kv=kv, bev=torch.randn(1, 4, 2, 2), speed=torch.ones(1),
        target_point=torch.ones(1, 2), target_point_next=torch.ones(1, 2),
        final_goal=torch.ones(1, 2), rope_position_offset=9,
        flow_state=torch.zeros(1, 18, 2), flow_time=torch.zeros(1),
        flow_sample_noise=torch.zeros(1, 18, 2),
    )
    model.train()
    train_out = decoder_forward(model, kwargs, torch.bfloat16, torch.device("cpu"))
    train_out["flow_velocity"].square().mean().backward()
    assert any(p.grad is not None for p in model.parameters())
    model.eval()
    with torch.no_grad():
        eval_out = decoder_forward(model, kwargs, torch.bfloat16, torch.device("cpu"))
    assert eval_out["flow_velocity"].dtype == torch.float32
    assert eval_out["pred_route"].shape == (1, 10, 2)


def test_analysis_fallback_and_no_action_gt_leak():
    priors = {
        "conditions": {"ROAD_STRUCTURE": None, "UE3": None},
        "future_waypoints": [[99, 99]],
        "raw": "private",
    }
    text = fallback_analysis(priors)
    from qwen3vl_local.action_prior.prompts import analysis_format_valid

    assert analysis_format_valid(text)
    assert not valid_analysis(text, priors)
    assert not valid_analysis("", priors)
    prompt = analysis_prompt(priors, "velocity=2. Predict the driving actions now")
    assert (
        "future_waypoints" not in prompt
        and "private" not in prompt
        and "Predict the driving actions" not in prompt
    )


def test_text_cache_image_and_contract_invalidation(tmp_path):
    from PIL import Image
    from qwen3vl_local.action_prior.text_cache import TextCache

    cache = TextCache(tmp_path / "cache.sqlite")
    images = [Image.new("RGB", (3, 3), "red") for _ in range(4)]
    key = cache.key("contract1", images, "nav", "case")
    value = {
        "conditions": {"UE3": None},
        "calls": [
            dict(
                phase=2,
                variant="all_random_order",
                keys=["UE3"],
                response="UE3: NO",
                prompt="test",
                history=[],
            )
        ],
    }
    cache.put(key, value)
    assert cache.get(key)["conditions"]["UE3"] is None
    assert cache.get(cache.key("contract2", images, "nav", "case")) is None
    images[-1].putpixel((0, 0), (0, 0, 0))
    assert cache.get(cache.key("contract1", images, "nav", "case")) is None


@pytest.mark.parametrize("review_case", ["pass", "reject", "malformed"])
def test_generated_and_cached_final_kv_always_base(tmp_path, monkeypatch, review_case):
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from qwen3vl_local.action_prior.text_cache import TextCache
    import qwen3vl_local.action_prior.runtime as rt
    from qwen3vl_local.action_prior import prompts
    import json

    for i in (1, 2):
        m = get_peft_model(
            TinyBase(),
            LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"], bias="none"),
        )
        m.save_pretrained(tmp_path / f"p{i}")
    engine = SimpleNamespace(
        model=TinyBase(), _last_decode_state=None, _system_prompt_cache=None
    )
    engine.build_messages = lambda system, user, images: [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    engine.processor = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: str(messages)
    )
    engine.prepare_inputs = lambda text, images: {
        "input_ids": torch.arange(len(text)).reshape(1, -1)
    }
    calls = []

    def prefill(inputs):
        with torch.inference_mode():
            result = engine.model(torch.ones(1, 4))
        calls.append(result.clone())
        return SimpleNamespace(past_key_values=result, rope_deltas=torch.tensor([[-3]]))

    engine.prefill = prefill
    contract = {f"phase{i}": {"path": str(tmp_path / f"p{i}")} for i in (1, 2)}
    contract["identity"] = "identity"
    from qwen3vl_local.action_prior import phase2_v3_prompts as event_prompts
    contract["phase2"]["metadata"] = dict(
        prompt_name=event_prompts.PROMPT_NAME, history_rgb_mode="4rgb",
        production_prompt_sha256=event_prompts.event_prompt_sha256(history_rgb_mode="4rgb"))
    runtime = PriorEngine(
        engine, contract, text_cache=TextCache(tmp_path / "text.sqlite")
    )
    counter = []

    def collect(ask, key, **kwargs):
        assert kwargs["event_module"] is event_prompts
        counter.append(key)
        return {
            "conditions": {"ROAD_STRUCTURE": "R1", "UE3": "YES"},
            "invalid": {},
            "calls": [],
        }

    monkeypatch.setattr(rt, "collect_priors", collect)
    draft = (
        "The surface-road corridor continues ahead while another vehicle enters the immediate "
        "path; at 4 m/s the forward navigation target and reduced clearance guide near-term planning."
    )
    generated_calls = []

    def generate(system, prompt, images, **kwargs):
        assert torch.equal(engine.model(torch.ones(1, 4)), TinyBase()(torch.ones(1, 4)))
        generated_calls.append((system, prompt, len(images)))
        if system == prompts.REVIEW_SYSTEM:
            assert len(images) == 0 and draft in json.loads(
                prompt.split("[DRAFT_SUMMARY]\n")[1].split(
                    "\n[/DRAFT_SUMMARY]"
                )[0]
            )
            checks = {k: True for k in prompts.REVIEW_KEYS}
            if review_case == "reject":
                checks["consistent"] = False
            text = json.dumps(checks) if review_case != "malformed" else "PASS"
        else:
            assert "VERIFIED_SUMMARY" not in prompt and draft not in prompt
            text = draft
        return text, SimpleNamespace(decode_steps=[SimpleNamespace(is_eos=True)])

    runtime.generate_messages = generate
    images = [Image.new("RGB", (3, 3)) for _ in range(4)]
    k1, o1 = runtime.condition(images, "nav", "case")
    k2, o2 = runtime.condition(images, "nav", "case")
    assert len(counter) == 1 and runtime.last_audit["text_cache_hit"]
    assert len(generated_calls) == 2  # 命中不再次生成/复核。
    assert runtime.last_audit["raw_analysis"] == draft
    assert runtime.last_audit["analysis_semantic_guarantee"] is False
    if review_case == "pass":
        assert runtime.last_audit["analysis"] == draft
        assert not runtime.last_audit["analysis_fallback"]
    else:
        assert runtime.last_audit["analysis_fallback"]
        assert runtime.last_audit["analysis"] != draft
    assert len(calls) == 2 and torch.equal(k1, k2) and o1 == o2
    assert all(torch.equal(k, TinyBase()(torch.ones(1, 4))) for k in calls)
