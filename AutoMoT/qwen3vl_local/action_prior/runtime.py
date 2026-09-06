"""复用 LeadMoT 预处理/BEV/decoder，只替换 frozen language condition。"""

from __future__ import annotations
import contextlib
from pathlib import Path
from qwen3vl_local.action_prior import prompts
from qwen3vl_local.action_prior.priors import collect_priors
from qwen3vl_local.sft_new_loop_phase1 import prompts as p1
from qwen3vl_local.sft_new_loop_phase2 import prompts as p2
from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices
from qwen3vl_local.action_prior.prompt_versions import prompt_module
from qwen3vl_local.action_prior.progress import report


class PriorEngine:
    """共享一个 frozen base，两个独立 LoRA；不 merge，不混用不同 adapter 的 cache。"""

    def __init__(
        self,
        engine,
        contract,
        analysis_tokens=384,
        text_cache=None,
        recheck_mode="history",
    ):
        from peft import PeftModel
        from qwen3vl_local.engine import _inspect_lora_adapter

        self.engine, self.contract = engine, contract
        self.analysis_tokens = analysis_tokens
        self.text_cache = text_cache
        self.recheck_mode = recheck_mode
        for key in ("phase1", "phase2"):
            _inspect_lora_adapter(Path(contract[key]["path"]))
        report("setup/load_phase1_lora", announce=True, phase1_path=contract["phase1"]["path"])
        self.adapters = PeftModel.from_pretrained(
            engine.model,
            contract["phase1"]["path"],
            adapter_name="phase1",
            is_trainable=False,
            local_files_only=True,
        )
        report("setup/load_phase2_lora", announce=True, phase2_path=contract["phase2"]["path"])
        self.adapters.load_adapter(
            contract["phase2"]["path"],
            adapter_name="phase2",
            is_trainable=False,
            local_files_only=True,
        )
        # PEFT 仅管理启停；forward/decode 始终调用底层 Qwen 及本地 M-RoPE helper。
        self.engine.model = self.adapters.get_base_model()
        self.adapters.eval().requires_grad_(False)
        self.last_audit = None
        report("setup/loras_ready", announce=True)

    @contextlib.contextmanager
    def mode(self, name):
        """每次切换清除旧 cache；base 时整段生成都禁用所有 LoRA。"""
        self.engine._last_decode_state = None
        self.engine._system_prompt_cache = None
        try:
            if name == "base":
                with self.adapters.disable_adapter():
                    yield
            else:
                self.adapters.set_adapter(name)
                self.adapters.requires_grad_(False)
                yield
        finally:
            # PEFT enable_adapter_layers 在退出 disable 上下文时可能重新启用梯度。
            self.adapters.requires_grad_(False)

    def generate_messages(self, system, prompt, images, history=(), max_tokens=160):
        """真实 assistant 后续问答，重做完整图文 prefill 保证多轮 M-RoPE 对齐。"""
        from qwen3vl_local.engine import GenerationTrace

        engine = self.engine
        engine._last_decode_state = None
        if history:
            messages = engine.build_messages(system, history[0][0], images)
            messages.append({"role": "assistant", "content": history[0][1]})
            for user, assistant in history[1:]:
                messages.extend(
                    [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ]
                )
            messages.append({"role": "user", "content": prompt})
        else:
            messages = engine.build_messages(system, prompt, images)
        text = engine.apply_chat_template(messages)
        inputs = engine.prepare_inputs(text, images)
        trace = GenerationTrace(
            chat_text=text,
            input_summary={},
            final_cache_summary={},
            prefill_cache_summary={},
        )
        output = engine.prefill(inputs)
        engine.max_gen_tokens = max_tokens
        ids = engine.decode(inputs, output, trace)
        return (
            engine.processor.batch_decode(ids, skip_special_tokens=True)[0].strip(),
            trace,
        )

    def condition(self, images, navigation, sample_key):
        """返回纯 base 吃四张图+先验+生成分析后的 cache，不包含 LoRA 计算的 KV。"""
        if len(images) != 4:
            raise ValueError(
                "action prior requires four chronological stitched RGB images"
            )

        question_count = 0

        def ask(phase, spec, history):
            nonlocal question_count
            question_count += 1
            report(f"condition/phase{phase}_question", question_call=question_count,
                   question_has_history=bool(history))
            meta = self.contract[f"phase{phase}"]["metadata"]
            module = prompt_module(phase, meta)
            mode = meta["history_rgb_mode"]
            selected = [images[i] for i in history_rgb_indices(mode)]
            prompt = (module.build_phase1_prompt if phase == 1 else module.build_event_prompt)(
                spec=spec, history_rgb_mode=mode
            )
            with self.mode(f"phase{phase}"):
                text, _ = self.generate_messages(
                    module.SYSTEM_PROMPT, prompt, selected, history
                )
            return text, prompt

        key = (
            self.text_cache.key(
                self.contract["identity"], images, navigation, sample_key
            )
            if self.text_cache
            else None
        )

        def compute():
            report("condition/cache_miss", text_cache_hit=False)
            priors = collect_priors(ask, sample_key, recheck_mode=self.recheck_mode,
                                    event_module=prompt_module(2, self.contract["phase2"]["metadata"]))
            report("condition/base_analysis")
            with self.mode("base"):
                text, trace = self.generate_messages(
                    prompts.SYSTEM_PROMPT,
                    prompts.analysis_prompt(priors, navigation),
                    images,
                    max_tokens=self.analysis_tokens,
                )
            truncated = not any(s.is_eos for s in trace.decode_steps)
            raw_analysis = text
            review, review_raw = None, ""
            review_truncated = False
            if not truncated and prompts.analysis_format_valid(text):
                report("condition/base_review")
                # 第二次独立文本调用只审查蕴含关系；不继承生成 cache，也不重新看图分类。
                with self.mode("base"):
                    review_raw, review_trace = self.generate_messages(
                        prompts.REVIEW_SYSTEM,
                        prompts.review_prompt(priors, navigation, text),
                        [],
                        max_tokens=192,
                    )
                review_truncated = not any(s.is_eos for s in review_trace.decode_steps)
                if not review_truncated:
                    review = prompts.parse_review(review_raw)
            fallback = not prompts.valid_analysis(text, priors, review)
            if truncated:
                rejection = "generation_truncated"
            elif not prompts.analysis_format_valid(text):
                rejection = "generation_format"
            elif review_truncated or review is None:
                rejection = "review_truncated" if review_truncated else "review_format"
            elif not all(review.values()):
                rejection = "review_rejected"
            else:
                rejection = "none"
            if fallback:
                text = prompts.fallback_analysis(priors, navigation)
            from qwen3vl_local.action_prior.contracts import digest

            return dict(
                priors,
                analysis=text,
                raw_analysis=raw_analysis,
                analysis_fallback=fallback,
                analysis_truncated=truncated,
                analysis_rejection=rejection,
                analysis_review=review,
                analysis_review_raw=review_raw,
                analysis_review_truncated=review_truncated,
                reviewed_analysis_sha256=digest(raw_analysis),
                analysis_acceptance="fallback" if fallback else "base_model_review",
                analysis_semantic_guarantee=False,
            )

        report("condition/cache_lookup_or_lock", sample_key=sample_key, text_cache_hit=None,
               question_call=0)
        if self.text_cache:
            priors, cache_hit = self.text_cache.get_or_compute(key, compute)
        else:
            priors, cache_hit = compute(), False
        from qwen3vl_local.action_prior.contracts import digest

        if priors["analysis_fallback"]:
            accepted = priors["analysis"] == prompts.fallback_analysis(
                priors, navigation
            )
        else:
            accepted = priors.get("reviewed_analysis_sha256") == digest(
                priors["analysis"]
            ) and prompts.valid_analysis(
                priors["analysis"], priors, priors.get("analysis_review")
            )
        if not accepted:
            raise ValueError(
                "cached analysis lacks its paired review or valid fallback"
            )
        report("condition/base_final_prefill", text_cache_hit=cache_hit)
        with self.mode("base"):
            text = priors["analysis"]
            # 首次和缓存命中都完整重建相同 assistant transcript，保证 KV 分布完全一致。
            self.engine._last_decode_state = None
            messages = self.engine.build_messages(
                prompts.SYSTEM_PROMPT,
                prompts.analysis_prompt(priors, navigation),
                images,
            )
            messages.append({"role": "assistant", "content": text})
            chat = self.engine.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = self.engine.prepare_inputs(chat, images)
            output = self.engine.prefill(inputs)
            cache = output.past_key_values
            length = int(inputs["input_ids"].shape[-1])
            delta = output.rope_deltas
            if delta is None:
                raise RuntimeError("missing base Qwen M-RoPE delta")
            offset = length + int(delta.reshape(-1)[0].item())
        self.last_audit = dict(
            priors,
            contract_identity=self.contract["identity"],
            text_cache_hit=cache_hit,
            base_cache_tokens=length,
            rope_position_offset=offset,
        )
        report("condition/base_kv_ready")
        return cache, offset


def make_runtime(args, device, contract):
    """实例级 prefill 注入，不修改旧 runner 文件或其全局实现。"""
    from qwen3vl_local.leadmot.train import LeadMoTTrainRuntime

    class Runtime(LeadMoTTrainRuntime):
        """沿用已有状态和 BEV 的 forward，显式复制 inference KV 供 autograd 使用。"""

        def __init__(self):
            import torch
            from types import SimpleNamespace
            from qwen3vl_local.leadmot import train as old

            # 旧 backbone 构造硬写 pretrained=True；本入口已有完整 BEV 权重，禁止先下载 ImageNet。
            original_timm = old.mot_runner.timm

            def offline_create(*a, **kw):
                kw["pretrained"] = False
                return original_timm.create_model(*a, **kw)

            old.mot_runner.timm = SimpleNamespace(create_model=offline_create)
            try:
                super().__init__(args, device)
            finally:
                old.mot_runner.timm = original_timm
            weights = torch.load(
                args.lead_bev_ckpt, map_location="cpu", weights_only=False
            )
            weights = weights.get("model", weights)
            backbone = {
                k[len("backbone.") :]: v
                for k, v in weights.items()
                if k.startswith("backbone.")
            }
            self.runner.bev_encoder.backbone.load_state_dict(
                backbone or weights, strict=True
            )
            del weights, backbone
            import os
            from qwen3vl_local.action_prior.text_cache import TextCache

            cache = (
                TextCache(Path(args.output_dir) / "text_cache" / "shared_v2")
                if args.cache_priors
                else None
            )
            self.prior = PriorEngine(
                self.runner.leadmot_qwen_engine,
                contract,
                args.analysis_tokens,
                cache,
                args.recheck_mode,
            )
            self.base_prefill = self.runner._run_leadmot_qwen_prefill
            self.runner._run_leadmot_qwen_prefill = self.prefill_prior
            self.sample_key = ""

        def prefill_prior(self, rgb_pil_list, user_prompt):
            """仅 navigation 作为公开输入，不把 sample 字典送入 Qwen。"""
            if args.condition_mode == "base":
                # 同初始化/优化器/划分的原 base 条件消融，不能用同一个 decoder 临时切条件。
                with self.prior.mode("base"):
                    result = self.base_prefill(rgb_pil_list, user_prompt)
                self.prior.last_audit = dict(
                    conditions={},
                    invalid={},
                    calls=[],
                    analysis="",
                    analysis_truncated=False,
                    analysis_fallback=False,
                    condition_mode="base",
                )
                return result
            return self.prior.condition(rgb_pil_list, user_prompt, self.sample_key)

        def forward_sample(
            self, sample, decoder, decoder_config, decoder_dtype, clip=None
        ):
            self.sample_key = f"{sample['scenario']}/{sample['run_id']}:{sample['anchor']}:{args.seed}"

            def decoder_with_trainable_cache(**kwargs):
                report("train_or_eval/decoder_forward")
                # inference tensor 不能被可训练 attention 保存给 backward；在 inference_mode 外 clone。
                kwargs["pooled_kv"] = [
                    (k.detach().clone(), v.detach().clone())
                    for k, v in kwargs["pooled_kv"]
                ]
                from qwen3vl_local.action_prior.precision import decoder_forward

                return decoder_forward(decoder, kwargs, decoder_dtype, self.device)

            return super().forward_sample(
                sample,
                decoder_with_trainable_cache,
                decoder_config,
                __import__("torch").float32,
                clip,
            )

    return Runtime()
