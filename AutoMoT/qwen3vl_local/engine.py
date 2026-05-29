"""本地 Qwen3-VL-Instruct 显式推理引擎。

这个文件负责把“范式 A”的一次 VLM 调用拆成可观察的几个阶段：

1. 构造 HuggingFace chat messages。
2. 调用 processor.apply_chat_template 生成最终聊天文本。
3. 用 processor 把文本和图片一起转成模型输入张量。
4. 做一次 prefill，得到首步 logits 和 past_key_values。
5. 手写 token-by-token decode，持续更新 KV cache。

这里刻意不复用 AutoMoT 的 InterleaveInferencer。原因是 standalone
Qwen3-VL-Instruct 要走 HuggingFace 标准接口，真实图像 token 也必须由
structured image message + processor 生成，而不是靠 prompt 里的字符串占位。
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .cache_utils import save_kv_cache, summarize_kv_cache


# 这组三个环境变量是“只读本地 checkpoint”的第一道保险。
# from_pretrained 下面仍然会显式传 local_files_only=True，双保险防止误联网下载。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


@dataclass
class DecodeStep:
    """记录自回归解码时的单步 token 信息。

    trace 里保存它不是为了训练，而是为了排查模型为什么提前 EOS、为什么输出
    某个奇怪 token，或者对比贪心/采样两种生成策略的差异。
    """

    step: int
    token_id: int
    token_text: str
    is_eos: bool


@dataclass
class GenerationTrace:
    """一次完整生成的可复现实验记录。

    chat_text 是 processor 模板展开后的最终文本；input_summary 和 cache summary
    只保存 shape/dtype/device，不复制大张量值，所以可以安全写进 JSON 日志。
    如果命令行打开 --save-cache，真正的 KV 张量会另外保存成 .pt 文件。
    """

    chat_text: str
    input_summary: Dict[str, Any]
    prefill_cache_summary: Dict[str, Any]
    final_cache_summary: Dict[str, Any]
    decode_steps: List[DecodeStep] = field(default_factory=list)
    prefill_cache_file: Optional[str] = None
    final_cache_file: Optional[str] = None

    def to_dict(self) -> dict:
        """转换成 JSON 可序列化结构，供 runner 落盘到 generation_trace.json。"""

        return {
            "chat_text": self.chat_text,
            "input_summary": self.input_summary,
            "prefill_cache_summary": self.prefill_cache_summary,
            "final_cache_summary": self.final_cache_summary,
            "decode_steps": [x.__dict__ for x in self.decode_steps],
            "prefill_cache_file": self.prefill_cache_file,
            "final_cache_file": self.final_cache_file,
        }


def _tensor_summary(x: Any) -> Dict[str, Any]:
    """只提取张量元信息，避免把真实 tensor 值塞进日志。"""

    return {
        "shape": list(x.shape) if hasattr(x, "shape") else None,
        "dtype": str(getattr(x, "dtype", None)),
        "device": str(getattr(x, "device", None)),
    }


def _inputs_summary(inputs: Any) -> Dict[str, Any]:
    """汇总 processor 输出的模型输入张量。

    Qwen3-VL 常见 key 包括 input_ids、attention_mask、pixel_values、
    image_grid_thw。这里记录这些字段的 shape，就能确认“几张图变成了多少
    vision token”以及张量是否已经移动到正确设备。
    """

    out: Dict[str, Any] = {}
    for k, v in inputs.items():
        out[k] = _tensor_summary(v)
    return out


def _outputs_summary(outputs: Any) -> Dict[str, Any]:
    """汇总模型输出的常见字段，主要用于临时调试。

    当前主流程没有把它写进 trace，因为 prefill_cache_summary 已经覆盖了最关心
    的 cache 结构；保留这个 helper 是为了以后需要打印 logits/hidden_states
    形状时不用再临时手写。
    """

    try:
        out_keys = list(outputs.keys())
    except Exception:
        out_keys = [k for k in dir(outputs) if not k.startswith("_")]

    out: Dict[str, Any] = {"keys": out_keys}
    if hasattr(outputs, "logits") and getattr(outputs, "logits") is not None:
        out["logits"] = _tensor_summary(outputs.logits)
    if hasattr(outputs, "past_key_values") and getattr(outputs, "past_key_values") is not None:
        out["past_key_values_summary"] = summarize_kv_cache(outputs.past_key_values)

    for name in ("hidden_states", "encoder_last_hidden_state"):
        value = getattr(outputs, name, None)
        if value is not None:
            out[name] = _tensor_summary(value)
    return out


class LocalQwen3VLInstructEngine:
    """本地 Qwen3-VL-Instruct 的最小推理封装。

    这个类只关心“给一组图和一段 prompt，生成一段自由文本”。它不接 AutoMoT
    的 BEV、route head、waypoint head，也不碰 MoT 自定义 cache 格式。
    """

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
        # checkpoint_dir 必须是本地已经下载好的 Qwen3-VL-4B-Instruct 目录。
        self.checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()

        # requested_device 保留用户原始选择；load() 时再把 "auto" 解析成 cuda/cpu。
        self.requested_device = device
        self.device = device
        self.torch_dtype = torch_dtype

        # 生成控制参数：默认温度 0 + 不采样，即贪心解码，便于复现实验。
        self.max_gen_tokens = max_gen_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.save_cache = save_cache

        # 延迟加载，避免创建 runner 时就占显存；第一次 generate/prefill 前才加载。
        self.model = None
        self.processor = None

    def load(self) -> None:
        """加载本地 model 和 processor，不允许联网补文件。"""

        if self.model is not None:
            return
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"missing local checkpoint: {self.checkpoint_dir}")

        import torch
        from transformers import AutoProcessor

        # transformers 版本不同，Qwen3-VL 的模型类名称可能不同。
        # 这里按“新通用类 -> Qwen 专用类 -> 旧 vision2seq 通用类”的顺序兜底。
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

        # trust_remote_code=True 是为了使用 checkpoint 目录里的 Qwen3-VL 自定义代码；
        # local_files_only=True 保证 transformers 只读本地文件。
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
        """构造 Qwen processor 认识的 structured chat messages。

        重点：图片在这里以 {"type": "image", "image": img} 的结构化字段传入。
        这才是真正触发 processor 生成 vision token 的地方；user_prompt 里的文字
        只负责描述任务和 memory，不负责占位图片。
        """

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
        """把 structured messages 展开成模型最终看到的聊天文本。"""

        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def prepare_inputs(self, chat_text: str, images: List[Any]) -> Any:
        """把聊天文本和图片一起转成模型输入张量。

        text=[chat_text] 负责文本 token；images=images 负责视觉输入。processor 会
        根据 chat_text 里的视觉占位和 images 列表建立对应关系，并返回
        pixel_values/image_grid_thw 等视觉张量。
        """

        return self.processor(
            text=[chat_text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

    def prefill(self, inputs: Any) -> Any:
        """执行一次完整上下文前向，得到 logits 和初始 past_key_values。

        prefill 阶段会一次性处理所有文本 token 和 vision token，成本最高；返回的
        past_key_values 会在 decode 阶段复用，避免每生成一个 token 都重算整段图文上下文。
        """

        return self.model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )

    def _eos_ids(self) -> set:
        """读取当前模型的结束 token id 集合。"""

        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if eos is None:
            eos = getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None)
        if eos is None:
            return set()
        if isinstance(eos, (list, tuple, set)):
            return {int(x) for x in eos}
        return {int(eos)}

    def _select_next_token(self, next_logits: Any) -> Any:
        """从最后一个位置的 logits 里选出下一 token。"""

        import torch

        if self.do_sample:
            # 采样模式用于探索多样输出；temperature 越高越随机。
            logits = next_logits / max(self.temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        # 默认贪心模式可复现，适合跑对照实验和调试状态机解析。
        return torch.argmax(next_logits, dim=-1, keepdim=True)

    def decode(
        self,
        inputs: Any,
        prefill_outputs: Any,
        trace: GenerationTrace,
        cache_dir: Optional[pathlib.Path] = None,
    ) -> Any:
        """基于 prefill cache 自回归生成新 token。

        每一轮只把“已经生成的最后一个 token + 旧 past_key_values”喂回模型。
        prepare_inputs_for_generation 会根据当前 decoded_input_ids、attention_mask
        和 cache_position 组装增量推理需要的输入。
        """

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

            # 记录原始 token 文本时不跳过 special token，方便看到是否立刻 EOS。
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

            # cache_position 指向本轮新增 token 在完整序列中的位置。
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
        """执行完整生成流程，返回模型文本和 trace。

        这个方法是 runner 唯一需要调用的高层入口；内部仍然故意保留
        build_messages/apply_chat_template/prepare_inputs/prefill/decode 的显式步骤，
        方便定位“prompt、图片 token、KV cache、decode”分别出了什么问题。
        """

        self.load()

        # 1) 结构化消息：图片仍是 PIL/list 对象，还没有转 token。
        messages = self.build_messages(system_prompt, user_prompt, images)

        # 2) 聊天模板：processor 在文本中放入模型需要的视觉占位 token。
        chat_text = self.apply_chat_template(messages)

        # 3) 张量化：文本 token 与图片 tensor 在这里真正绑定。
        inputs = self.prepare_inputs(chat_text, images)

        # 真实 LEAD 4 帧常见 input_summary 形态示例：
        # input_ids/attention_mask: [1, 文本+视觉总 token 数]
        # pixel_values: [视觉 patch 数, hidden 输入维度]
        # image_grid_thw: [图片张数, 3]，每行是 time/height/width 网格。
        prefill_outputs = self.prefill(inputs)

        trace = GenerationTrace(
            chat_text=chat_text,
            input_summary=_inputs_summary(inputs),
            prefill_cache_summary=summarize_kv_cache(prefill_outputs.past_key_values),
            final_cache_summary={},
        )

        # 4) 自回归 decode：new_ids 只包含新生成的 token，不包含 prompt token。
        new_ids = self.decode(inputs, prefill_outputs, trace, cache_dir=cache_dir)
        raw_text = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        return raw_text.lstrip("\n "), trace


def dump_trace(trace: GenerationTrace, out_dir: pathlib.Path) -> None:
    """把 generation trace 单独落盘，方便不打开 step.json 也能看推理细节。"""

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generation_trace.json").write_text(
        json.dumps(trace.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
