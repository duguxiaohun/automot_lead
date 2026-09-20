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
from qwen3vl_local.action_prior.action_input import (
    HighLevelActionIndex, normalize_action, gate_action, ACTION_CONDITIONING_VERSION,
)


class PriorEngine:
    """共享一个 frozen base，两个独立 LoRA；不 merge，不混用不同 adapter 的 cache。

    传入 ``labels`` 表示改用数据集标定真值：完全不加载 LoRA，也不做任何先验问答。
    """

    def __init__(
        self,
        engine,
        contract,
        analysis_tokens=384,
        text_cache=None,
        recheck_mode="history",
        labels=None,
        analysis_review=True,
        generate_analysis=False,
        high_level_action_prior=False,
    ):
        self.engine, self.contract = engine, contract
        self.analysis_tokens = analysis_tokens
        self.text_cache = text_cache
        self.recheck_mode = recheck_mode
        self.labels = labels
        self.generate_analysis = bool(generate_analysis)
        self.high_level_action_prior = bool(high_level_action_prior)
        self.analysis_review = bool(analysis_review) and self.generate_analysis
        self.adapters = None
        self.last_audit = None
        if labels is not None:
            report("setup/dataset_prior_labels", announce=True,
                   prior_labels=labels.path, labeled_frames=labels.rows,
                   analysis_review=self.analysis_review)
            return
        from peft import PeftModel
        from qwen3vl_local.engine import _inspect_lora_adapter

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
        report("setup/loras_ready", announce=True)

    @contextlib.contextmanager
    def mode(self, name):
        """每次切换清除旧 cache；base 时整段生成都禁用所有 LoRA。"""
        self.engine._last_decode_state = None
        self.engine._system_prompt_cache = None
        if self.adapters is None:
            # 数据集先验从未加载 adapter，模型本身就是 frozen base。
            if name != "base":
                raise ValueError("dataset priors never run a LoRA adapter")
            yield
            return
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

    def condition(
        self, images, navigation, sample_key, identity=None, event_balanced_scene_contexts=(),
        high_level_action=None, high_level_action_contexts=(),
    ):
        """默认直接编码四图与先验提示词；显式开启时追加生成分析，最终 KV 均来自 base。"""
        if len(images) != 4:
            raise ValueError(
                "action prior requires four chronological stitched RGB images"
            )
        if self.labels is not None and identity is None:
            raise ValueError("dataset priors need the (scenario, run_id, frame) identity")

        question_count = 0
        # 单帧动作接口：当前查自动 Phase3 标注/显式索引，后续可接 Phase3 predictor。
        # 关闭时完全忽略外部字段，不改变默认图文输入。
        action = normalize_action(high_level_action) if self.high_level_action_prior else None
        action_contexts = tuple(high_level_action_contexts) if self.high_level_action_prior else ()
        from qwen3vl_local.action_prior.contracts import digest

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

        # 这些是全帧映射固有的离线事实，不能随本轮抽到的配额桶改变。以 tuple
        # 进入 cache key，避免同一 RGB 在不同 prompt 条件下误复用分析缓存。
        event_balanced_scene_contexts = tuple(
            str(value) for value in event_balanced_scene_contexts if value
        )
        key = (
            self.text_cache.key(
                self.contract["identity"], images, navigation,
                f"{sample_key}:event_contexts={event_balanced_scene_contexts}"
                f":generate_analysis={self.generate_analysis}"
                f":high_level_action_prior={self.high_level_action_prior}:action={digest(action)}"
                f":action_contexts={action_contexts}:action_policy={ACTION_CONDITIONING_VERSION}"
            )
            if self.text_cache
            else None
        )

        def compute():
            report("condition/cache_miss", text_cache_hit=False)
            if self.labels is not None:
                report("condition/dataset_labels")
                priors = self.labels.priors(identity)
            else:
                priors = collect_priors(ask, sample_key, recheck_mode=self.recheck_mode,
                                        event_module=prompt_module(2, self.contract["phase2"]["metadata"]))
            # 这是显式 opt-in 的离线 transition/evidence 条件；默认采样课程不改变
            # Qwen 输入。文本只会在 prompts.py 变成自然、无类别名的短句。
            priors = dict(priors, high_level_action_prior=self.high_level_action_prior)
            if self.high_level_action_prior:
                # 先获得实际消费的 Phase1/2 条件（含噪声/复核），再门控动作。
                effective, action_gate = gate_action(action, priors["conditions"], action_contexts,
                                                     event_balanced_scene_contexts)
                priors.update(high_level_action=effective, high_level_action_input=action,
                              high_level_action_contexts=action_contexts, high_level_action_gate=action_gate)
            if event_balanced_scene_contexts:
                priors = dict(
                    priors,
                    event_balanced_scene_contexts=event_balanced_scene_contexts,
                )
            if not self.generate_analysis:
                # 只缓存先验，不运行 base decode、摘要复核或 fallback；空文本不进最终 KV。
                return dict(
                    priors, analysis="", raw_analysis="", analysis_fallback=False,
                    analysis_truncated=False, analysis_rejection="none",
                    analysis_review=None, analysis_review_raw="",
                    analysis_review_truncated=False, analysis_review_enabled=False,
                    analysis_acceptance="disabled", analysis_semantic_guarantee=False,
                    generate_analysis=False,
                )
            report("condition/base_analysis")
            with self.mode("base"):
                text, trace = self.generate_messages(
                    prompts.system_prompt(generate_analysis=True),
                    prompts.analysis_prompt(priors, navigation),
                    images,
                    max_tokens=self.analysis_tokens,
                )
            truncated = not any(s.is_eos for s in trace.decode_steps)
            raw_analysis = text
            review, review_raw = None, ""
            review_truncated = False
            if self.analysis_review and not truncated and prompts.analysis_format_valid(text):
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
            # 截断文本即使恰好凑成三段也不能当完整分析，与复核开关无关。
            fallback = truncated or not prompts.valid_analysis(
                text, priors, review, require_review=self.analysis_review
            )
            if truncated:
                rejection = "generation_truncated"
            elif not prompts.analysis_format_valid(text):
                rejection = "generation_format"
            elif not self.analysis_review:
                rejection = "none"
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
                generate_analysis=True,
                analysis=text,
                raw_analysis=raw_analysis,
                analysis_fallback=fallback,
                analysis_truncated=truncated,
                analysis_rejection=rejection,
                analysis_review=review,
                analysis_review_raw=review_raw,
                analysis_review_truncated=review_truncated,
                reviewed_analysis_sha256=digest(raw_analysis),
                analysis_review_enabled=self.analysis_review,
                analysis_acceptance=(
                    "fallback"
                    if fallback
                    else "base_model_review" if self.analysis_review else "format_only"
                ),
                analysis_semantic_guarantee=False,
            )

        report("condition/cache_lookup_or_lock", sample_key=sample_key, text_cache_hit=None,
               question_call=0)
        if self.text_cache:
            priors, cache_hit = self.text_cache.get_or_compute(key, compute)
        else:
            priors, cache_hit = compute(), False
        from qwen3vl_local.action_prior.contracts import digest

        if not self.generate_analysis:
            accepted = (
                priors.get("generate_analysis") is False
                and priors.get("analysis") == ""
                and priors.get("raw_analysis") == ""
                and priors.get("analysis_acceptance") == "disabled"
                and not priors.get("analysis_fallback")
            )
        elif priors["analysis_fallback"]:
            accepted = priors["analysis"] == prompts.fallback_analysis(
                priors, navigation
            )
        else:
            accepted = (
                not priors.get("analysis_truncated")
                and priors.get("reviewed_analysis_sha256") == digest(priors["analysis"])
                and priors.get("analysis_review_enabled", True) == self.analysis_review
                and prompts.valid_analysis(
                    priors["analysis"],
                    priors,
                    priors.get("analysis_review"),
                    require_review=self.analysis_review,
                )
            )
        action_matches = True
        if self.high_level_action_prior:
            expected, expected_gate = gate_action(action, priors["conditions"], action_contexts,
                                                  event_balanced_scene_contexts)
            action_matches = (priors.get("high_level_action_input") == action
                              and priors.get("high_level_action") == expected
                              and priors.get("high_level_action_gate") == expected_gate
                              and tuple(priors.get("high_level_action_contexts", ())) == action_contexts)
        if (not accepted or priors.get("high_level_planning", False)
                or priors.get("high_level_action_prior", False) != self.high_level_action_prior
                or not action_matches):
            raise ValueError(
                "cached prior/analysis does not match the configured KV mode or review"
            )
        report("condition/base_final_prefill", text_cache_hit=cache_hit)
        with self.mode("base"):
            # 首次和缓存命中都完整 prefill；默认只编码 system/user 图文，生成开关
            # 开启后才使用旧摘要 prompt 和 assistant transcript。
            self.engine._last_decode_state = None
            messages = self.engine.build_messages(
                prompts.system_prompt(generate_analysis=self.generate_analysis),
                (prompts.analysis_prompt if self.generate_analysis else prompts.prefill_prompt)(
                    priors, navigation
                ),
                images,
            )
            if self.generate_analysis:
                messages.append({"role": "assistant", "content": priors["analysis"]})
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
            final_cache_content="inputs_and_analysis" if self.generate_analysis else "inputs_only",
        )
        report("condition/base_kv_ready")
        return cache, offset


def sample_condition_inputs(args, sample, action_index=None):
    """分开查表动作作用域和显式场景事实，防止动作索引绕过 Phase1/2。"""
    from qwen3vl_local.action_prior.scene_policy import scene_contexts
    identity = (str(sample["scenario"]), str(sample["run_id"]), int(sample["anchor"]))
    enabled = getattr(args, "high_level_action_prior", False) and action_index is not None
    return dict(
        high_level_action=action_index.candidate_evidence(identity) if enabled else None,
        high_level_action_contexts=action_index.planning_contexts(identity) if enabled else (),
        event_balanced_scene_contexts=scene_contexts(args, sample.get("event_balance_scene_contexts", ())),
    )


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
            labels = None
            if getattr(args, "dataset_priors", False):
                from qwen3vl_local.action_prior.dataset_labels import (
                    PriorLabelIndex,
                    PriorNoise,
                )

                labels = PriorLabelIndex(
                    args.prior_labels,
                    PriorNoise(
                        args.prior_noise, args.prior_noise_invalid_share, args.seed
                    ),
                )
            self.prior = PriorEngine(
                self.runner.leadmot_qwen_engine,
                contract,
                args.analysis_tokens,
                cache,
                args.recheck_mode,
                labels,
                args.analysis_review,
                generate_analysis=args.generate_analysis,
                high_level_action_prior=getattr(args, "high_level_action_prior", False),
            )
            self.action_index = (
                HighLevelActionIndex(args.high_level_action_index)
                if getattr(args, "high_level_action_prior", False) else None
            )
            if self.action_index and self.action_index.identity != contract["identity_payload"]["high_level_action_input"]:
                raise ValueError("high-level action file changed after contract preflight")
            self.base_prefill = self.runner._run_leadmot_qwen_prefill
            self.runner._run_leadmot_qwen_prefill = self.prefill_prior
            self.sample_key = ""
            self.sample_identity = None
            self.event_balanced_scene_contexts = ()
            self.high_level_action = None
            self.high_level_action_contexts = ()

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
            return self.prior.condition(
                rgb_pil_list, user_prompt, self.sample_key, self.sample_identity,
                self.event_balanced_scene_contexts,
                high_level_action=self.high_level_action,
                high_level_action_contexts=self.high_level_action_contexts,
            )

        def forward_sample(
            self,
            sample,
            decoder,
            decoder_config,
            decoder_dtype,
            clip=None,
            flow_state=None,
            flow_time=None,
            flow_sample_noise=None,
            sample_trajectory=True,
        ):
            self.sample_key = f"{sample['scenario']}/{sample['run_id']}:{sample['anchor']}:{args.seed}"
            self.sample_identity = (
                str(sample["scenario"]),
                str(sample["run_id"]),
                int(sample["anchor"]),
            )
            condition_inputs = sample_condition_inputs(args, sample, self.action_index)
            self.high_level_action = condition_inputs["high_level_action"]
            self.high_level_action_contexts = condition_inputs["high_level_action_contexts"]
            self.event_balanced_scene_contexts = condition_inputs["event_balanced_scene_contexts"]

            def decoder_with_trainable_cache(**kwargs):
                report("train_or_eval/decoder_forward")
                # inference tensor 不能被可训练 attention 保存给 backward；在 inference_mode 外 clone。
                kwargs["pooled_kv"] = [
                    (k.detach().clone(), v.detach().clone())
                    for k, v in kwargs["pooled_kv"]
                ]
                from qwen3vl_local.action_prior.precision import decoder_forward

                # action_prior v5 的 FM 训练传入 x_t/t；推理只给噪声，由 decoder 内部 Euler
                # 采样。这里不改变冻结 Qwen/BEV 的一次性准备和完整 KV 条件边界。
                if flow_state is not None:
                    kwargs["flow_state"] = flow_state
                    kwargs["flow_time"] = flow_time
                if flow_sample_noise is not None:
                    kwargs["flow_sample_noise"] = flow_sample_noise
                kwargs["sample_trajectory"] = sample_trajectory
                return decoder_forward(decoder, kwargs, decoder_dtype, self.device)

            return super().forward_sample(
                sample,
                decoder_with_trainable_cache,
                decoder_config,
                __import__("torch").float32,
                clip,
            )

    return Runtime()
