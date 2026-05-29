"""Explicit local Qwen3-VL-Instruct inference engine.

The public methods intentionally expose the same phases that matter in
AutoMoT-style reasoning experiments:

1. Build chat messages.
2. Apply the local processor chat template.
3. Tensorize text/images.
4. Prefill once and obtain ``past_key_values``.
5. Decode one token at a time while updating the cache.

The model and processor are loaded from a local checkpoint directory only.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .cache_utils import save_kv_cache, summarize_kv_cache


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


@dataclass
class DecodeStep:
    step: int
    token_id: int
    token_text: str
    is_eos: bool


@dataclass
class GenerationTrace:
    chat_text: str
    input_summary: Dict[str, Any]
    prefill_cache_summary: Dict[str, Any]
    final_cache_summary: Dict[str, Any]
    decode_steps: List[DecodeStep] = field(default_factory=list)
    prefill_cache_file: Optional[str] = None
    final_cache_file: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "chat_text": self.chat_text,
            "input_summary": self.input_summary,
            "prefill_cache_summary": self.prefill_cache_summary,
            "final_cache_summary": self.final_cache_summary,
            "decode_steps": [x.__dict__ for x in self.decode_steps],
            "prefill_cache_file": self.prefill_cache_file,
            "final_cache_file": self.final_cache_file,
        }


def _inputs_summary(inputs: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in inputs.items():
        out[k] = {
            "shape": list(v.shape) if hasattr(v, "shape") else None,
            "dtype": str(getattr(v, "dtype", None)),
            "device": str(getattr(v, "device", None)),
        }
    return out


class LocalQwen3VLInstructEngine:
    def __init__(
        self,
        checkpoint_dir: pathlib.Path,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        max_gen_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
        save_cache: bool = False,
    ):
        self.checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()
        self.requested_device = device
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_gen_tokens = max_gen_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.save_cache = save_cache

        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load local model/processor. No network access is allowed."""
        if self.model is not None:
            return
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"missing local checkpoint: {self.checkpoint_dir}")

        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError:
            try:
                from transformers import Qwen3VLForConditionalGeneration as ModelClass
            except ImportError:
                from transformers import AutoModelForVision2Seq as ModelClass

        self.device = self.requested_device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        dtype = dtype_map.get(self.torch_dtype, torch.bfloat16)

        print(f"[qwen3vl-local] load checkpoint={self.checkpoint_dir}")
        print(f"[qwen3vl-local] offline env HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")
        # 这里的 from_pretrained 只接收本地目录，并配合 local_files_only/offline env。
        # 它不是联网下载入口，而是 transformers 对本地 config/权重分片的标准加载器。
        self.model = ModelClass.from_pretrained(
            str(self.checkpoint_dir),
            torch_dtype=dtype,
            local_files_only=True,
            trust_remote_code=True,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            str(self.checkpoint_dir),
            local_files_only=True,
            trust_remote_code=True,
        )

    def build_messages(self, system_prompt: str, user_prompt: str, images: List[Any]) -> List[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in images]
                    + [{"type": "text", "text": user_prompt}]
                ),
            },
        ]

    def apply_chat_template(self, messages: List[dict]) -> str:
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def prepare_inputs(self, chat_text: str, images: List[Any]) -> Any:
        return self.processor(
            text=[chat_text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

    def prefill(self, inputs: Any) -> Any:
        return self.model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )

    def _eos_ids(self) -> set:
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if eos is None:
            eos = getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None)
        if eos is None:
            return set()
        if isinstance(eos, (list, tuple, set)):
            return {int(x) for x in eos}
        return {int(eos)}

    def _select_next_token(self, next_logits: Any) -> Any:
        import torch

        if self.do_sample:
            logits = next_logits / max(self.temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(next_logits, dim=-1, keepdim=True)

    def decode(
        self,
        inputs: Any,
        prefill_outputs: Any,
        trace: GenerationTrace,
        cache_dir: Optional[pathlib.Path] = None,
    ) -> Any:
        import torch

        eos_ids = self._eos_ids()
        attention_mask = inputs.get("attention_mask", None)
        decoded_input_ids = inputs["input_ids"]
        past_key_values = prefill_outputs.past_key_values
        next_logits = prefill_outputs.logits[:, -1, :]
        generated_tokens: List[Any] = []

        if self.save_cache and cache_dir is not None:
            trace.prefill_cache_file = save_kv_cache(
                past_key_values, pathlib.Path(cache_dir) / "prefill_past_key_values.pt"
            )

        for step in range(self.max_gen_tokens):
            next_token = self._select_next_token(next_logits)
            generated_tokens.append(next_token)
            decoded_input_ids = torch.cat([decoded_input_ids, next_token], dim=1)

            token_id = int(next_token[0, 0].item())
            token_text = self.processor.batch_decode(next_token, skip_special_tokens=False)[0]
            is_eos = token_id in eos_ids
            trace.decode_steps.append(DecodeStep(step, token_id, token_text, is_eos))
            if is_eos:
                break

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token, device=attention_mask.device)],
                    dim=1,
                )

            cache_position = torch.arange(
                decoded_input_ids.shape[1] - 1,
                decoded_input_ids.shape[1],
                device=decoded_input_ids.device,
            )
            model_inputs = self.model.prepare_inputs_for_generation(
                decoded_input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=True,
            )
            outputs = self.model(**model_inputs, return_dict=True)
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

        trace.final_cache_summary = summarize_kv_cache(past_key_values)
        if self.save_cache and cache_dir is not None:
            trace.final_cache_file = save_kv_cache(
                past_key_values, pathlib.Path(cache_dir) / "final_past_key_values.pt"
            )

        if not generated_tokens:
            return inputs["input_ids"].new_empty((1, 0))
        return torch.cat(generated_tokens, dim=1)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: List[Any],
        cache_dir: Optional[pathlib.Path] = None,
    ) -> tuple[str, GenerationTrace]:
        self.load()
        messages = self.build_messages(system_prompt, user_prompt, images)
        chat_text = self.apply_chat_template(messages)
        inputs = self.prepare_inputs(chat_text, images)
        prefill_outputs = self.prefill(inputs)

        trace = GenerationTrace(
            chat_text=chat_text,
            input_summary=_inputs_summary(inputs),
            prefill_cache_summary=summarize_kv_cache(prefill_outputs.past_key_values),
            final_cache_summary={},
        )
        new_ids = self.decode(inputs, prefill_outputs, trace, cache_dir=cache_dir)
        raw_text = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        return raw_text.lstrip("\n "), trace


def dump_trace(trace: GenerationTrace, out_dir: pathlib.Path) -> None:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generation_trace.json").write_text(
        json.dumps(trace.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
