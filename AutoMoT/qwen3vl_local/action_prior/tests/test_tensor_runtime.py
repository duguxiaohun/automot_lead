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
    assert not valid_analysis("Scene: truncated", priors)
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
        "Scene: The accepted road structure is a lane-following surface corridor.\n"
        "Interaction: A vehicle is entering the immediate ego corridor.\n"
        "Planning context: At 4 m/s, the forward navigation target and accepted intrusion are relevant to the available corridor."
    )
    generated_calls = []

    def generate(system, prompt, images, **kwargs):
        assert torch.equal(engine.model(torch.ones(1, 4)), TinyBase()(torch.ones(1, 4)))
        generated_calls.append((system, prompt, len(images)))
        if system == prompts.REVIEW_SYSTEM:
            assert len(images) == 0 and draft in json.loads(
                prompt.split("[DRAFT_JSON_STRING]\n")[1].split(
                    "\n[/DRAFT_JSON_STRING]"
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
