"""SFT 训练入口 — 直接把 LoRA 挂到 Qwen3-VL-4B-Instruct 上，手写 DDP loop。

设计目标见 qwen3vl_local/sft/SFT_PLAN.md；运行命令见 qwen3vl_local/sft/SFT_RUN.md。

与旧版（ms-swift + loss_scale plugin）的关键区别：
- **不再依赖 ms-swift**。直接 `transformers + peft.LoraConfig + get_peft_model`
  把 LoRA 适配器加到 base 模型上，自己写训练 loop 和 collator。
- **不再离线物化 teacher cache / manifest**。每个 train batch 在进 student 之前，
  先禁用 adapter，并调用底层 Qwen base model 做 greedy generate 出 ANALYSIS；
  再启用 adapter 进 student forward。同一份模型权重，只占 1 份显存。
- **loss 直接写在 train step 里**：collator 给每个样本算好 per-token weight，
  loss = sum(F.cross_entropy * weight) / sum(weight)。当前内置权重表：
    * user / system prompt 段 = 0
    * ANALYSIS 起手字面 "ANALYSIS: " = 1.0
    * ANALYSIS body（teacher 输出文本） = ANALYSIS_WEIGHT（默认 0.5）
    * 段切换字面 "\nSTATUS: " / "\nSUBGOAL: " = 1.0
    * STATUS event_name / SUBGOAL event_name = 1.0
    * tail / EOS = 1.0

典型用法（从 AutoMoT/ 目录运行）：

```bash
# 推荐入口：bash launcher 会自动创建 run_<RUN_TAG>/ 子目录并挑空闲 GPU
bash qwen3vl_local/sft/train.sh

# 显式 pin 单卡
GPU_IDS=0 bash qwen3vl_local/sft/train.sh

# 直接调 Python 做轻量 sanity（不保存 checkpoint）
GPU_IDS=0 python qwen3vl_local/sft/train.py \
  --train-jsonl checkpoints/sft_data_pending/train.jsonl \
  --val-jsonl checkpoints/sft_data_pending/val.jsonl \
  --output-dir checkpoints/sft_lora_debug \
  --check
```

环境变量入口（与 train.sh 解耦，方便 python 直接调）：
- `SFT_ANALYSIS_WEIGHT`：ANALYSIS body 权重，默认 0.5。
- `SFT_TEACHER_MAX_NEW_TOKENS`：teacher generate 上限，默认 256。
- `SFT_TEACHER_TEMPERATURE`：teacher 采样温度，默认 0.0（greedy）。

DDP / 性能相关行为（v2 → 当前路线的健壮性补丁）：
- 单条坏样本不再让多卡训练整体 raise；改成"同进同退" all-reduce(MIN)，所有
  rank 一起 skip 这条样本继续训。详见 `_ddp_all_ranks_valid`。
- 梯度累积前 (grad_accum-1) 个 micro-step 包在 ``DDP.no_sync()`` 里，只在
  最后一个 micro-step / 尾批 finish_optimizer_step 触发一次 all-reduce(AVG)，
  把每个 optimizer step 的 all-reduce 次数从 grad_accum 次降到 1 次。
- 尾批不再人为放大梯度幅度（rescale ≡ 1.0），避免 cosine 末段尾批 step 比
  正常 step 强 ``grad_accum / accum_count`` 倍。
- LoRA 注入时 `gradient_checkpointing_enable` 强制 `use_reentrant=False`，
  避免老 transformers + find_unused_parameters=True 的反传图错位坑。
- `--skip-teacher`：跳过 teacher.generate，ANALYSIS 全部走固定 fallback；
  用于 sanity 链路调试，不可用于产线训练。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except Exception:
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False

from PIL import Image


# ---------------------------------------------------------------------------
# 常量与正则
# ---------------------------------------------------------------------------

PENDING_PLACEHOLDER = "__TEACHER_PENDING__"
FALLBACK_ANALYSIS = "Observations recorded."

# tokenize 边界 sanity 只在每个进程第一次走 build_student_inputs 时跑一次：
# 验证 (chat_prompt) tokenize ⊕ (assistant) tokenize 的拼接结果，与一次性
# tokenize(chat_prompt + assistant) 完全一致；避免 BPE 在边界合并 token 导致
# 训练时看到的 token 序列与推理时不同。通过一次即可 silent，无需每个 batch 跑。
_TOKENIZE_BOUNDARY_CHECKED = False

# teacher 输出后处理：与 build_teacher.py 完全同口径。
_TEACHER_PREFIX_RE = re.compile(r"^\s*ANALYSIS\s*:\s*", re.IGNORECASE)
_TEACHER_STOP_MARKERS = ("\nSTATUS:", "\nSUBGOAL:", "\n\n", "<|im_end|>")
_TEACHER_MAX_CHARS = 420
_TEACHER_MIN_CHARS = 80

# 切 ANALYSIS 正文 / STATUS 事件名 / SUBGOAL 事件名的字符范围。
_FULL_ASSIST_RE = re.compile(
    r"ANALYSIS:[ \t]*"
    r"(?P<analysis>[^\n]*?)"
    r"\s*\nSTATUS:[ \t]*"
    r"(?P<status>\S[^\n]*?)"
    r"\s*\nSUBGOAL:[ \t]*"
    r"(?P<subgoal>\S[^\n]*)",
    flags=re.DOTALL,
)


# teacher prompt 由 PRIVILEGED 块注入；与 build_teacher.py 同口径。
_TEACHER_SYSTEM_PROMPT = """You are a vision-grounded annotation teacher for an autonomous driving status-tracking task.

Input:
- 4 RGB frames (oldest -> newest), stitched three-camera view.
- MEMORY: the previous anchor (anchor-K) STATUS and EVENT_SEQUENCE.
- PRIVILEGED: the ground-truth current STATUS at the newest frame, and whether this anchor is KEEP (state unchanged) or ADVANCE (state moved forward from MEMORY STATUS).

Task:
Produce a single line of ANALYSIS that a student model (which does NOT see PRIVILEGED) could plausibly infer from images alone. Sentence order MUST be:
1. First sentence: concretely describe what is visible in the LAST frame.
2. Second sentence: describe what CHANGED between the earliest and the latest frame.
3. Third sentence: state whether the observed evidence supports staying at MEMORY STATUS or advancing to the current STATUS, tying it to the visual evidence above.

Length target (strict):
- Aim for 40-70 words total across the 3 sentences.
- Going under 25 words tends to skip the visual evidence step and become a bare conclusion - do NOT do that.
- Going over 90 words tends to invent extra details, repeat clauses, or drift off-task - do NOT do that.
- Each sentence should carry one specific visual fact; do not pad with hedging phrases ("it seems that", "we can observe that", "as we can see").

Constraints:
- Do NOT mention or reference the PRIVILEGED block; write as if from images only.
- Do NOT invent visual content not actually present.
- Be concise, grounded, factual; 2-4 sentences total, all on a single line.
- Do NOT output STATUS or SUBGOAL; only the ANALYSIS body text (no "ANALYSIS:" prefix).

Output EXACTLY one line of text (the ANALYSIS body, no prefix, no trailing newline)."""

_STUDENT_TAIL_MARKER = "Given the observations above and the memory context"


# ---------------------------------------------------------------------------
# 辅助函数：teacher 输出后处理
# ---------------------------------------------------------------------------

def _truncate_at_sentence_boundary(t: str, hard_limit: int) -> str:
    """把 teacher 长输出优先截在句子边界，避免半句话进入 student 监督。"""

    if len(t) <= hard_limit:
        return t
    window = t[:hard_limit]
    best = -1
    for punct in (". ", "! ", "? "):
        idx = window.rfind(punct)
        if idx > best:
            best = idx + 1
    if best > hard_limit // 2:
        return t[:best].rstrip()
    cut_pos = window.rfind(" ")
    return t[: cut_pos if cut_pos > 0 else hard_limit].rstrip()


def postprocess_teacher(text: str) -> Tuple[str, bool]:
    """teacher 输出后处理，返回 (清理后文本, 是否使用兜底)。与 build_teacher 同口径。"""

    if not text:
        return FALLBACK_ANALYSIS, True
    t = _TEACHER_PREFIX_RE.sub("", text.strip())
    cut = len(t)
    for stop in _TEACHER_STOP_MARKERS:
        i = t.find(stop)
        if i >= 0 and i < cut:
            cut = i
    t = t[:cut]
    t = re.sub(r"\s+", " ", t).strip()
    t = _truncate_at_sentence_boundary(t, _TEACHER_MAX_CHARS)
    if len(t) < _TEACHER_MIN_CHARS:
        return FALLBACK_ANALYSIS, True
    return t, False


def build_teacher_user_prompt(student_user_no_image: str, meta: Dict) -> str:
    """在 student user prompt 末尾、`Given ...` 句之前插入 PRIVILEGED 块。"""

    privileged = (
        "\n[PRIVILEGED]\n"
        f"CURRENT_GT_STATUS: {meta['target_status']}\n"
        f"TRANSITION: {meta['transition']}\n"
        f"PREV_STATUS: {meta['memory_in_status']}\n"
        "[/PRIVILEGED]\n\n"
        "Given the observations, memory, and privileged ground truth, "
        "output the ANALYSIS body that the student should plausibly produce from images alone."
    )
    idx = student_user_no_image.find(_STUDENT_TAIL_MARKER)
    if idx >= 0:
        return student_user_no_image[:idx].rstrip() + privileged
    return student_user_no_image.rstrip() + privileged


def strip_image_placeholders(user_content: str) -> str:
    """去掉 jsonl user 里的 <image> 占位（由 processor 通过 structured image 提供）。"""

    s = user_content.lstrip()
    while s.startswith("<image>"):
        s = s[len("<image>"):]
    return s.lstrip("\n")


# ---------------------------------------------------------------------------
# 数据集：读取 pending jsonl
# ---------------------------------------------------------------------------

class SftJsonlDataset(Dataset):
    """每个样本返回原始 dict（含 prompts / image paths / GT status / GT subgoal）。

    图片在 collate 阶段才真正读入 PIL，避免 Dataset 之间 DataLoader worker fork
    后重复 keep。多卡 / 多 worker 都按照样本 idx 走 DistributedSampler 分片，
    每个 rank 跑自己的子集 batch。
    """

    def __init__(self, jsonl_path: pathlib.Path):
        """一次性读入 jsonl 元数据；真正读图延迟到 train step。"""

        self.jsonl_path = pathlib.Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"sft jsonl not found: {self.jsonl_path}")
        self.rows: List[Dict] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.rows.append(row)

    def __len__(self) -> int:
        """返回样本条数，供 DataLoader / DistributedSampler 切分。"""

        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """取一条样本，并从 assistant GT 中解析 STATUS / SUBGOAL。"""

        row = self.rows[idx]
        assistant = row["messages"][2]["content"]
        # 从 GT assistant 反取 STATUS / SUBGOAL，避免要求 jsonl 都带 teacher_meta_input。
        m_status = re.search(r"^STATUS:\s*(\S+)", assistant, flags=re.MULTILINE)
        m_subgoal = re.search(r"^SUBGOAL:\s*(\S+)", assistant, flags=re.MULTILINE)
        gt_status = m_status.group(1) if m_status else ""
        gt_subgoal = m_subgoal.group(1) if m_subgoal else ""
        return {
            "scenario": row.get("scenario", ""),
            "run_id": row.get("run_id", ""),
            "anchor": int(row.get("anchor", -1)),
            "system_prompt": row["messages"][0]["content"],
            "user_prompt": strip_image_placeholders(row["messages"][1]["content"]),
            "image_paths": list(row.get("images", [])),
            "gt_status": gt_status,
            "gt_subgoal": gt_subgoal,
            "teacher_meta_input": row.get("teacher_meta_input") or {
                "target_status": gt_status,
                "target_subgoal": gt_subgoal,
                "memory_in_status": "",
                "transition": "keep",
            },
            "is_transition_sample": bool(row.get("is_transition_sample", False)),
        }


def collate_passthrough(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """不做 padding/tokenize，把样本 dict 列表原样传给 train step；
    真正的 prompt 拼装 / image 读入 / tokenize 都放在 train step 里，
    那里能同时拿到 teacher 输出（per-sample 不同长度）。"""

    return batch


# ---------------------------------------------------------------------------
# 模型加载（base + LoRA）
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """训练期共享的模型上下文：PEFT model、processor/tokenizer 和当前 device。"""

    model: Any
    processor: Any
    tokenizer: Any
    device: torch.device

    def unwrap(self):
        """从 DDP wrapper 里拿出真 PeftModel；用于保存和控制 adapter。"""

        return getattr(self.model, "module", self.model)

    def base_model_for_generation(self):
        """返回底层 Qwen 模型，避免 teacher generate 走 PeftModel wrapper。"""

        peft_model = self.unwrap()
        if hasattr(peft_model, "get_base_model"):
            return peft_model.get_base_model()
        return peft_model


def load_model_with_lora(
    model_dir: pathlib.Path,
    *,
    device: torch.device,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    gradient_checkpointing: bool,
) -> ModelBundle:
    """加载本地 Qwen3-VL-Instruct，并把 LoRA adapter 直接注入 base model。

    base 权重全冻结，PEFT 注入后只有 LoRA 参数可训练；这就是本 SFT 路线
    “不用外部 LoRA 插件 / 不走 ms-swift”的核心入口。
    """

    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError:
        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass

    print(f"[load] base model from {model_dir}")
    model = ModelClass.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer
    # 训练 / loss-mask sanity 都依赖 return_offsets_mapping=True，
    # 这是 Fast tokenizer 独占功能。提前 assert，避免训练跑几分钟才崩在深层调用栈。
    assert getattr(tokenizer, "is_fast", False), (
        "SFT 需要 Fast tokenizer (PreTrainedTokenizerFast)；"
        "当前 tokenizer 不是 Fast 版本，return_offsets_mapping 无法工作。"
    )

    # 冻结所有参数；PEFT 之后会自己把 LoRA 子矩阵设成 trainable。
    for p in model.parameters():
        p.requires_grad = False

    # 让 LoRA 走 transformers 的 gradient_checkpointing 路径。
    # use_reentrant=False 是关键：老 transformers 默认 True，和 DDP
    # find_unused_parameters=True 搭配会触发反传图错位（部分 step loss=NaN /
    # 反传卡死）。这里强制走非 reentrant 路径，规避老坑。
    if gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            except TypeError:
                # 极老 transformers 不支持 kwargs 入参，退到默认调用。
                model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            # 必要：gradient_checkpointing 下 input embeddings 需要 require_grad。
            model.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return ModelBundle(model=model, processor=processor, tokenizer=tokenizer, device=device)


# ---------------------------------------------------------------------------
# Teacher 单条生成
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_teacher_one_sample(
    bundle: ModelBundle,
    sample: Dict[str, Any],
    images_pil: List[Image.Image],
    *,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[str, bool]:
    """对单个样本跑 frozen base teacher，返回 (analysis_body, fallback)。

    DDP-safe：在 `with disable_adapter()` + `eval()` 上下文里跑；每个 rank 独立处理
    自己的样本，不需要跨 rank 同步。
    """

    peft_model = bundle.unwrap()
    base_model = bundle.base_model_for_generation()
    base_model.eval()
    # gradient_checkpointing 与 generate(use_cache=True) 互斥；transformers 会强制
    # 把 use_cache 改为 False，每生成 1 个 token 都要重算整段前缀，teacher 单条耗时
    # 从秒级飙到分钟级。这里临时关掉 GC，generate 完再恢复。
    gc_was_enabled = bool(getattr(base_model, "is_gradient_checkpointing", False))
    if gc_was_enabled and hasattr(base_model, "gradient_checkpointing_disable"):
        base_model.gradient_checkpointing_disable()
    messages = [
        {"role": "system", "content": _TEACHER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": img} for img in images_pil]
                + [{"type": "text", "text": build_teacher_user_prompt(
                    sample["user_prompt"], sample["teacher_meta_input"])}]
            ),
        },
    ]
    chat_text = bundle.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = bundle.processor(
        text=[chat_text],
        images=images_pil if images_pil else None,
        return_tensors="pt",
        padding=True,
    ).to(bundle.device)

    do_sample = temperature > 0.0
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=True,
        pad_token_id=bundle.tokenizer.pad_token_id or bundle.tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    # 关键：adapter 的开关仍由 PeftModel 管，但 generate 调底层 Qwen。
    # 这样避开 Qwen3-VL + PeftModel.generate 的 M-RoPE/prepare_inputs 兼容坑。
    try:
        with peft_model.disable_adapter():
            out = base_model.generate(**inputs, **gen_kwargs)
    finally:
        if gc_was_enabled and hasattr(base_model, "gradient_checkpointing_enable"):
            try:
                base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            except TypeError:
                base_model.gradient_checkpointing_enable()

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, prompt_len:]
    text = bundle.tokenizer.decode(gen_ids, skip_special_tokens=True)
    cleaned, fb = postprocess_teacher(text)
    return cleaned, fb


# ---------------------------------------------------------------------------
# Student：把 prompt + 完整 assistant 拼接为 input_ids，并构造 labels + 逐 token 权重
# ---------------------------------------------------------------------------

def build_student_inputs(
    bundle: ModelBundle,
    sample: Dict[str, Any],
    images_pil: List[Image.Image],
    teacher_analysis: str,
    *,
    analysis_weight: float,
    max_length: int,
) -> Optional[Dict[str, Any]]:
    """构造一条 student 训练序列。返回 None 表示样本被截掉（assistant 装不下）。

    步骤：
    1. 用 processor 把 prompt（system+user+图）apply_chat_template + tokenize 得到
       `inputs_prompt`，长度记为 L_prompt。
    2. 把 GT assistant 文本拼出来：
         ANALYSIS: <teacher_analysis>\nSTATUS: <gt_status>\nSUBGOAL: <gt_subgoal>
       再附 tokenizer.eos_token（如 `<|im_end|>`）作为 tail。
    3. 单独 tokenize assistant 文本（add_special_tokens=False）拿到 token id 序列。
    4. 拼接 prompt + assistant tokens → input_ids。labels：prompt 段 = -100，
       assistant 段 = 自身 id。
    5. 按字符范围计算 assistant token 的逐 token 权重。
    """

    user_content = (
        [{"type": "image", "image": img} for img in images_pil]
        + [{"type": "text", "text": sample["user_prompt"]}]
    )
    messages = [
        {"role": "system", "content": sample["system_prompt"]},
        {"role": "user", "content": user_content},
    ]
    chat_text = bundle.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs_prompt = bundle.processor(
        text=[chat_text],
        images=images_pil if images_pil else None,
        return_tensors="pt",
        padding=True,
    )

    prompt_ids = inputs_prompt["input_ids"][0]
    L_prompt = int(prompt_ids.shape[0])

    eos_token = bundle.tokenizer.eos_token or "<|im_end|>"

    # ---- 一次性 tokenize 边界 sanity ----
    # 训练时把 prompt 与 assistant 分两次 tokenize 再拼接；如果 Qwen 的 BPE 在
    # 边界处会跨段 merge，拼出来的 token 序列就和一次性 tokenize 的不一致，
    # 模型推理时看到的边界 token 与训练时不同，会让生成质量下降。
    # 这里只在每个进程第一次走该函数时做一次完整对比，通过后默认 silent。
    global _TOKENIZE_BOUNDARY_CHECKED
    if not _TOKENIZE_BOUNDARY_CHECKED:
        _TOKENIZE_BOUNDARY_CHECKED = True
        # 注意：apply_chat_template 走 processor 路径，含视觉 token；这里只比
        # 文本侧的边界，所以用 tokenizer 走纯文本 tokenize 验证。
        text_prompt = bundle.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        # processor(text=...) 与 tokenizer(text) 对纯文本应当一致；但 processor
        # 在多模态路径上可能注入视觉占位符，所以单独用 tokenizer 跑一次纯文本。
        text_only_prompt_ids = bundle.tokenizer(
            text_prompt, add_special_tokens=False,
        )["input_ids"]
        sample_assist = "ANALYSIS: probe analysis body.\nSTATUS: probe_status\nSUBGOAL: probe_subgoal"
        sample_full_text = text_prompt + sample_assist + eos_token + "\n"
        merged = bundle.tokenizer(
            sample_full_text, add_special_tokens=False,
        )["input_ids"]
        split_assist = bundle.tokenizer(
            sample_assist + eos_token + "\n", add_special_tokens=False,
        )["input_ids"]
        concat = list(text_only_prompt_ids) + list(split_assist)
        if concat != list(merged):
            # 不直接 raise，避免训练崩；只在 rank0 打 warn 并附上 diff 长度。
            print(f"[tokenize-boundary][warn] split={len(concat)} merged={len(merged)} "
                  f"BPE 在 prompt↔assistant 边界 merge 了 token；考虑改成一次性 tokenize。")
    # 对齐 Qwen chat 模板 assistant 段 `<|im_start|>assistant\n{content}<|im_end|>\n`：
    # 末尾 EOS 后再补一个 `\n`，让训练分布与 in-context 推理一致。
    assistant_text = (
        f"ANALYSIS: {teacher_analysis}\n"
        f"STATUS: {sample['gt_status']}\n"
        f"SUBGOAL: {sample['gt_subgoal']}"
    )
    assistant_text_with_eos = assistant_text + eos_token + "\n"

    # 对 assistant 做 tokenize 时使用 add_special_tokens=False；这里 processor 不需要再加图像 token。
    # 直接用 tokenizer 编码并取 offset，方便把逐 token 权重映射到文本片段。
    enc = bundle.tokenizer(
        assistant_text_with_eos,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    asst_ids = list(enc["input_ids"])
    asst_offsets = [tuple(x) for x in enc["offset_mapping"]]
    asst_ids_tensor = torch.tensor(asst_ids, dtype=prompt_ids.dtype)

    # 长度上限截断：assistant 太长直接丢样本（不在 prompt 段截，因为图像 token 不能截）。
    total_len = L_prompt + len(asst_ids)
    if total_len > max_length:
        max_asst = max_length - L_prompt
        if max_asst < 16:
            return None
        # 截断 assistant 末尾时一定要保留 EOS，否则模型学不到何时停止，
        # 推理阶段会无限续写。这里强制把 EOS / 末尾换行换到截断窗口末尾。
        tail_ids = asst_ids[-2:]    # `<|im_end|>` + `\n`（也可能 tokenizer 合并成 1 个 id）
        tail_offsets = asst_offsets[-2:]
        keep = max_asst - len(tail_ids)
        if keep < 1:
            return None
        asst_ids = asst_ids[:keep] + tail_ids
        asst_offsets = asst_offsets[:keep] + tail_offsets
        asst_ids_tensor = torch.tensor(asst_ids, dtype=prompt_ids.dtype)
        # 字符范围不再连续（中间截了一段），权重切段仍按 offset 走，没问题。

    # 拼成完整 input_ids。
    input_ids = torch.cat([prompt_ids, asst_ids_tensor], dim=0)
    labels = input_ids.clone()
    labels[:L_prompt] = -100

    # 逐 token 权重：assistant 段按字符范围切段。
    weights = torch.zeros(len(asst_ids), dtype=torch.float32)
    m = _FULL_ASSIST_RE.search(assistant_text_with_eos)
    if m is not None:
        a_start, a_end = m.span("analysis")
        s_start, s_end = m.span("status")
        g_start, g_end = m.span("subgoal")

        def _overlap(off: Tuple[int, int], lo: int, hi: int) -> bool:
            """判断一个 token 的 char offset 是否与目标片段相交。"""

            return off[0] < hi and off[1] > lo

        for i, off in enumerate(asst_offsets):
            off_t = (int(off[0]), int(off[1]))
            # ANALYSIS 正文
            if _overlap(off_t, a_start, a_end):
                weights[i] = analysis_weight
            # STATUS 事件名 / SUBGOAL 事件名
            elif _overlap(off_t, s_start, s_end) or _overlap(off_t, g_start, g_end):
                weights[i] = 1.0
            else:
                # 包括 "ANALYSIS: " 起手字面、"\nSTATUS: " / "\nSUBGOAL: " 段切换字面、
                # 末尾换行/EOS tail。统一 weight=1.0，保证结构字面也被监督。
                weights[i] = 1.0
    else:
        # 极端情况（teacher 输出毁掉了三段结构）：全段保守按 1.0 给 loss，不漏监督。
        weights[:] = 1.0

    # 完整序列上的逐 token 权重：prompt 段 = 0，assistant 段 = weights。
    full_weights = torch.zeros_like(input_ids, dtype=torch.float32)
    full_weights[L_prompt:L_prompt + len(asst_ids)] = weights

    # 视觉张量（pixel_values / image_grid_thw）保持 processor 输出的原形状直接带走。
    # 注意 Qwen3-VL processor 的视觉张量是 patch-flat 形式（无 batch 维），不能切 v[0]，
    # 也不该再 unsqueeze 加假 batch 维；student forward 时直接以这种"全局"形式传入。
    extra: Dict[str, Any] = {
        k: v for k, v in inputs_prompt.items()
        if k not in ("input_ids", "attention_mask")
    }
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "loss_weights": full_weights,
        "vision": extra,
        "assistant_text": assistant_text_with_eos,
        "L_prompt": L_prompt,
        "L_assistant": len(asst_ids),
    }


# ---------------------------------------------------------------------------
# Student 前向 + 加权 loss
# ---------------------------------------------------------------------------

def student_loss_one_sample(bundle: ModelBundle, packed: Dict[str, Any]) -> torch.Tensor:
    """对单个样本跑 student forward，返回标量 weighted loss。"""

    model = bundle.model
    device = bundle.device
    input_ids = packed["input_ids"].unsqueeze(0).to(device)
    attention_mask = packed["attention_mask"].unsqueeze(0).to(device)
    labels = packed["labels"].unsqueeze(0).to(device)
    weights = packed["loss_weights"].unsqueeze(0).to(device)

    kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    for k, v in packed["vision"].items():
        # 视觉张量来自 processor 原输出，已经是 forward 期望的形状（patch-flat，无 batch 维）。
        # 不要再 unsqueeze / squeeze，保持透传。
        if isinstance(v, torch.Tensor):
            kwargs[k] = v.to(device)
        else:
            kwargs[k] = v

    out = model(**kwargs, use_cache=False, return_dict=True)
    logits = out.logits  # (1, L, V)

    # 下一个 token 预测：logits[:, :-1] 预测 labels[:, 1:]
    # labels 在 build_student_inputs 里已经把 prompt 段（含 image token）写成 -100，
    # 这样 cross_entropy 的 ignore_index 可以直接屏蔽这些位置，省下大量图像 token
    # 的无效计算；同时也避免未来某天 weight 逻辑被改坏后，prompt 段没被 mask
    # 导致 prompt 泄漏。
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].contiguous()

    # ANALYSIS 正文的字符到 token 权重已经在 build_student_inputs 里映射到 input
    # 序列下标 i；预测它的位置是 logits[:, i-1]。所以 shift_weights 用 weights[1:]
    # 是对的（权重跟着 label 走，预测它的 logits 是它前一个位置）。

    active = shift_labels.ne(-100) & shift_weights.gt(0)
    if not bool(active.any()):
        return shift_logits.sum() * 0.0

    active_logits = shift_logits[active]
    active_labels = shift_labels[active]
    active_weights = shift_weights[active]
    loss_per_token = F.cross_entropy(
        active_logits,
        active_labels,
        reduction="none",
    )

    # 只对 labels 明确开放的 assistant token 计算交叉熵；prompt / 图像 token 不进入
    # softmax，避免未来 weight 逻辑漂移时 prompt 泄漏，也少算数千个视觉 token。
    weighted = loss_per_token * active_weights
    denom = active_weights.sum().clamp_min(1e-6)
    return weighted.sum() / denom


# ---------------------------------------------------------------------------
# 单 batch 训练（DDP 安全，逐样本顺序处理）
# ---------------------------------------------------------------------------

@dataclass
class StepStats:
    """单个 micro-batch 的轻量统计，用于日志和尾批梯度尺度校正。"""

    loss_sum: float = 0.0
    n_samples: int = 0
    n_teacher_fallback: int = 0
    n_skipped: int = 0


def _ddp_all_ranks_valid(local_valid: bool) -> bool:
    """DDP 训练时让所有 rank 对"当前样本是否进入 backward"达成一致。

    本训练循环是逐样本反传。若某个 rank 因图片缺失 / assistant 过长跳过，
    其它 rank 却进入 DDP 反传，collective 顺序会不一致，表现为训练卡死。

    实现策略："同进同退"——all-reduce MIN，只要任一 rank 报 invalid，所有 rank
    都视为 invalid 一起跳过；这样单条坏样本只丢一个 micro-batch，不会让多卡
    长跑训练在一张坏 jpg 上整体崩。

    单卡训练直接透传 local_valid。
    """

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return local_valid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flag = torch.tensor([1.0 if local_valid else 0.0], device=device, dtype=torch.float32)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item() > 0.5)


def run_step_on_batch(
    bundle: ModelBundle,
    batch: List[Dict[str, Any]],
    *,
    analysis_weight: float,
    teacher_max_new_tokens: int,
    teacher_temperature: float,
    max_length: int,
    loss_scale: float = 1.0,
    sync_grads: bool = True,
    skip_teacher: bool = False,
) -> Tuple[torch.Tensor, StepStats]:
    """对 batch 内每条样本依次跑 teacher → student，每条样本立即反传。

    ``loss_scale`` 用于梯度累积：每个 micro-step 的 loss 会先除以
    ``loss_scale`` 再反传，让 grad_accum 次累积后的梯度与"等效 batch_size
    = micro_batch_size * grad_accum"且 reduction='mean' 的 loss 算法对齐，避免
    lr 实际被放大 grad_accum 倍。

    ``sync_grads`` 控制 DDP 是否在本 micro-step 同步梯度：
    - False：本次 backward 包在 ``DDP.no_sync()`` 里，只在本 rank 累加本地梯度，
      不触发 all-reduce；
    - True：正常 backward，触发一次 all-reduce 把跨 rank 累积的梯度求平均。
    grad_accum > 1 时，前 (grad_accum - 1) 个 micro-step 应传 False，最后一个传
    True，避免每个 micro-step 都 all-reduce 浪费带宽。

    ``skip_teacher`` 为 True 时跳过 teacher.generate，用固定 fallback 文本替代
    ANALYSIS。专门用来做 student/优化器/DDP 链路 sanity，不依赖 teacher 现场跑。
    """

    stats = StepStats()
    total_loss = torch.zeros((), device=bundle.device, dtype=torch.float32)

    # DDP no_sync ctx 在 DDP 包装下才存在；单卡时退化为 nullcontext。
    if (not sync_grads) and hasattr(bundle.model, "no_sync"):
        sync_ctx = bundle.model.no_sync()
    else:
        sync_ctx = nullcontext()

    with sync_ctx:
        for sample in batch:
            # 读 RGB clip。
            try:
                images_pil = [Image.open(p).convert("RGB") for p in sample["image_paths"]]
                local_load_ok = True
            except (FileNotFoundError, OSError) as e:
                print(f"[step][warn] image load fail scenario={sample['scenario']} "
                      f"run={sample['run_id']} anchor={sample['anchor']}: {e}")
                images_pil = []
                local_load_ok = False

            # "同进同退"：任一 rank 读图失败，所有 rank 跳过本样本，避免
            # collective 顺序错位（teacher.generate 内部也走 forward，必须各 rank 一致）。
            if not _ddp_all_ranks_valid(local_load_ok):
                stats.n_skipped += 1
                continue

            # 阶段 A：teacher 现场生成 ANALYSIS（或走 skip_teacher 兜底）
            if skip_teacher:
                analysis_text = FALLBACK_ANALYSIS
                fb = True
            else:
                analysis_text, fb = run_teacher_one_sample(
                    bundle, sample, images_pil,
                    max_new_tokens=teacher_max_new_tokens,
                    temperature=teacher_temperature,
                )
            if fb:
                stats.n_teacher_fallback += 1

            # 阶段 B：student 前向并反传
            packed = build_student_inputs(
                bundle, sample, images_pil, analysis_text,
                analysis_weight=analysis_weight,
                max_length=max_length,
            )
            packed_ok = packed is not None
            # 再做一次"同进同退"：assistant 超长在不同 rank 上是确定性的（同一份
            # tokenizer + 同一份样本），但保险起见仍跨 rank 对齐，避免极端边界 case。
            if not _ddp_all_ranks_valid(packed_ok):
                stats.n_skipped += 1
                continue

            # student 训练前确保模型处于 train 模式（teacher 阶段调过 eval()）。
            bundle.model.train()
            loss = student_loss_one_sample(bundle, packed)
            # 立即反传释放显存：单样本反传；外层负责 optimizer.step()。
            # 先除以 loss_scale 再反传，使 grad_accum 累积梯度等效于
            # reduction='mean' 的 batch loss。
            scaled_loss = loss / max(loss_scale, 1.0)
            scaled_loss.backward()
            total_loss = total_loss.detach() + loss.detach()

            stats.loss_sum += float(loss.detach().item())
            stats.n_samples += 1

    return total_loss, stats


# ---------------------------------------------------------------------------
# 评估：跑 val 集，按相同流程计算 loss 与 STATUS 匹配率
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    bundle: ModelBundle,
    val_loader: DataLoader,
    *,
    analysis_weight: float,
    teacher_max_new_tokens: int,
    teacher_temperature: float,
    max_length: int,
    max_eval_samples: int,
    log_prefix: str = "[eval]",
) -> Dict[str, float]:
    """在 val_loader 上复用 teacher→student 训练目标，计算加权 loss。

    这里不做自由生成指标；STATUS / SUBGOAL 的生成质量由 sft/eval.py 评估。
    """

    bundle.model.eval()
    losses: List[float] = []
    fallback = 0
    skipped = 0
    seen = 0
    for batch in val_loader:
        for sample in batch:
            if max_eval_samples > 0 and seen >= max_eval_samples:
                break
            try:
                images_pil = [Image.open(p).convert("RGB") for p in sample["image_paths"]]
            except (FileNotFoundError, OSError):
                skipped += 1
                continue
            analysis_text, fb = run_teacher_one_sample(
                bundle, sample, images_pil,
                max_new_tokens=teacher_max_new_tokens,
                temperature=teacher_temperature,
            )
            if fb:
                fallback += 1
            packed = build_student_inputs(
                bundle, sample, images_pil, analysis_text,
                analysis_weight=analysis_weight,
                max_length=max_length,
            )
            if packed is None:
                skipped += 1
                continue
            loss = student_loss_one_sample(bundle, packed)
            losses.append(float(loss.item()))
            seen += 1
        if max_eval_samples > 0 and seen >= max_eval_samples:
            break

    mean_loss = sum(losses) / max(len(losses), 1)
    print(f"{log_prefix} samples={len(losses)} mean_loss={mean_loss:.4f} "
          f"teacher_fallback={fallback} skipped={skipped}")
    return {"loss": mean_loss, "samples": float(len(losses)),
            "teacher_fallback": float(fallback), "skipped": float(skipped)}


# ---------------------------------------------------------------------------
# 分布式辅助函数（类似 accelerate 的职责，但手写以避免额外依赖）
# ---------------------------------------------------------------------------

def setup_distributed() -> Tuple[int, int, int]:
    """初始化 torchrun 环境；单卡跑返回 (0, 1, 0)。"""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """退出前销毁分布式进程组，避免 torchrun 子进程悬挂。"""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_rank0(rank: int) -> bool:
    """只有 rank0 写日志、checkpoint 和 TensorBoard。"""

    return rank == 0


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析训练命令行参数；路径默认按远端 AutoMoT/ 当前目录书写。"""

    p = argparse.ArgumentParser(description="SFT train (LoRA inject + on-the-fly teacher)")
    p.add_argument("--train-jsonl", type=str, required=True)
    p.add_argument("--val-jsonl", type=str, default=None)
    p.add_argument("--model-dir", type=str, default=str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B-Instruct"))
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--per-device-batch-size", type=int, default=1,
                   help="batch 内串行走 teacher→student；保持 1 是最稳口径。")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=3584)
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=10000)
    p.add_argument("--eval-steps", type=int, default=10000)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--max-eval-samples", type=int, default=0,
                   help="eval 截到前 N；0 表示全量。check 模式自动设 8。")
    p.add_argument("--analysis-weight", type=float,
                   default=float(os.environ.get("SFT_ANALYSIS_WEIGHT", "0.5")))
    p.add_argument("--teacher-max-new-tokens", type=int,
                   default=int(os.environ.get("SFT_TEACHER_MAX_NEW_TOKENS", "256")))
    p.add_argument("--teacher-temperature", type=float,
                   default=float(os.environ.get("SFT_TEACHER_TEMPERATURE", "0.0")))
    p.add_argument("--max-steps", type=int, default=0,
                   help="0 表示按 num_epochs 跑全集；>0 用作 check 模式 / 小规模 sanity。")
    p.add_argument("--check", action="store_true",
                   help="check 模式：max_steps=2、不保存、ckpt-only sanity。")
    p.add_argument("--skip-teacher", action="store_true",
                   help="跳过 teacher.generate，ANALYSIS 全部用固定 fallback。"
                        "用于做 student / 优化器 / DDP 链路 sanity，把 teacher 现场"
                        "生成的不确定性排除掉。注意：训练出来的 LoRA 不可用于生产。")
    p.add_argument("--seed", type=int, default=20260601)
    return p.parse_args()


def make_optimizer(model, lr: float, weight_decay: float):
    """创建 AdamW；只优化 trainable 参数，也就是 LoRA adapter。"""

    # 只优化 trainable（即 LoRA 参数）。
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """创建 warmup + cosine decay 调度器，避免额外依赖 transformers.get_scheduler。"""

    # 余弦退火 + warmup。手写避免引入 transformers.get_scheduler 的隐式依赖。
    def lr_lambda(step: int) -> float:
        """把 optimizer step 映射成学习率倍率。"""

        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    """训练主入口：数据、LoRA、现场 teacher、student 反传、评估和保存。"""

    args = parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    output_dir = pathlib.Path(args.output_dir)
    if is_rank0(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[init] output_dir={output_dir}")
        print(f"[init] rank={rank} world_size={world_size} local_rank={local_rank} device={device}")

    bundle = load_model_with_lora(
        pathlib.Path(args.model_dir),
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=not args.no_grad_checkpoint,
    )

    if world_size > 1:
        bundle.model = torch.nn.parallel.DistributedDataParallel(
            bundle.model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,  # LoRA 只训练部分模块，视觉塔或部分层可能没有梯度。
        )

    # ---- 数据 ----
    train_ds = SftJsonlDataset(pathlib.Path(args.train_jsonl))
    val_ds = SftJsonlDataset(pathlib.Path(args.val_jsonl)) if args.val_jsonl else None
    if is_rank0(rank):
        print(f"[data] train={len(train_ds)}  val={len(val_ds) if val_ds else 0}")

    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        val_sampler = (DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
                       if val_ds else None)
        train_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=0,  # 0 是因为 PIL.Image + processor 都在 train step 现做，多 worker 没收益
        collate_fn=collate_passthrough,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.per_device_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_passthrough,
    ) if val_ds else None

    if args.check:
        # check 模式只跑 2 step 做链路 sanity，不保存 ckpt 也不进 eval。
        # （之前这里还设 max_eval_samples=8，但下游 eval 触发条件本身就排除了
        # check 模式，那个设置永远不会生效；现在删掉避免误读。）
        args.max_steps = 2 if args.max_steps == 0 else min(args.max_steps, 2)
        if is_rank0(rank):
            print("[check] max_steps=2, no save, eval skipped")

    # ---- 优化器 / 学习率调度器 ----
    optimizer = make_optimizer(bundle.model, args.learning_rate, args.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum)))
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps)

    # ---- TensorBoard ----
    tb_writer = None
    if is_rank0(rank) and _TB_AVAILABLE:
        tb_writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    # ---- 训练 loop ----
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    saved_ckpts: List[pathlib.Path] = []
    stop_training = False

    def finish_optimizer_step(
        epoch_idx: int,
        reason: str,
        loss_sum: float,
        n_samples: int,
    ) -> None:
        """完成一次 optimizer step，并集中处理日志、eval、checkpoint。

        梯度累积的正常整批和 epoch 末尾不足 grad_accum 的尾批都会走这里。

        与历史版本不同：不再为尾批做 ``expected_total / n_total`` 放大。
        放大策略会让尾批 (实际样本数 < grad_accum) 的更新强度比正常 step 大
        ``grad_accum / accum_count`` 倍，cosine LR 末段叠加这一下噪声很大。
        改为按实际样本数取均值（rescale ≡ 1.0），尾批就是一次轻量 step。
        """

        nonlocal global_step, saved_ckpts

        # 仅用于日志：跨 rank 汇总有效样本数。不再参与梯度 rescale。
        if world_size > 1 and torch.distributed.is_initialized():
            n_buf = torch.tensor([float(n_samples)], device=bundle.device, dtype=torch.float32)
            torch.distributed.all_reduce(n_buf, op=torch.distributed.ReduceOp.SUM)
            n_total = float(n_buf.item())
        else:
            n_total = float(n_samples)

        if n_total <= 0:
            optimizer.zero_grad(set_to_none=True)
            return

        trainable_params = [p for p in bundle.unwrap().parameters() if p.requires_grad]

        # 尾批 fallback：epoch 末尾的不足 grad_accum 的 micro-step 是用 no_sync()
        # 反传的，DDP 没自动 all-reduce 梯度；这里手动 all-reduce(AVG) 一次让
        # 各 rank 梯度对齐，再做 optimizer.step()。
        # "grad_accum" reason 的 step 在最后一个 micro-step 已经 sync 过了，跳过。
        if reason == "tail" and world_size > 1 and torch.distributed.is_initialized():
            avg_op = getattr(torch.distributed.ReduceOp, "AVG",
                             torch.distributed.ReduceOp.SUM)
            do_divide = (avg_op == torch.distributed.ReduceOp.SUM)
            for p in trainable_params:
                if p.grad is None:
                    continue
                torch.distributed.all_reduce(p.grad, op=avg_op)
                if do_divide:
                    p.grad.div_(float(world_size))
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        avg_loss = loss_sum / max(n_samples, 1)
        if is_rank0(rank) and (global_step % args.logging_steps == 0 or global_step == 1 or reason == "tail"):
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"[train] epoch={epoch_idx} step={global_step}/{total_steps} "
                  f"loss={avg_loss:.4f} lr={lr_now:.2e} samples={n_samples} "
                  f"reason={reason} elapsed={elapsed/60.0:.1f}min")
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", avg_loss, global_step)
                tb_writer.add_scalar("train/lr", lr_now, global_step)

        if (not args.check) and val_loader is not None \
                and args.eval_steps > 0 and global_step % args.eval_steps == 0:
            metrics = evaluate(
                bundle, val_loader,
                analysis_weight=args.analysis_weight,
                teacher_max_new_tokens=args.teacher_max_new_tokens,
                teacher_temperature=args.teacher_temperature,
                max_length=args.max_length,
                max_eval_samples=args.max_eval_samples,
                log_prefix=f"[eval@step{global_step}]",
            )
            if is_rank0(rank) and tb_writer is not None:
                for k, v in metrics.items():
                    tb_writer.add_scalar(f"val/{k}", v, global_step)

        if (not args.check) and args.save_steps > 0 \
                and global_step % args.save_steps == 0 and is_rank0(rank):
            ckpt_dir = output_dir / f"checkpoint-{global_step}"
            bundle.unwrap().save_pretrained(str(ckpt_dir))
            saved_ckpts.append(ckpt_dir)
            # 保留最近 save_total_limit 个 checkpoint。
            if args.save_total_limit > 0 and len(saved_ckpts) > args.save_total_limit:
                old = saved_ckpts.pop(0)
                try:
                    import shutil
                    shutil.rmtree(old)
                    print(f"[save] purged old ckpt {old}")
                except Exception as e:
                    print(f"[save][warn] purge {old} fail: {e}")
            print(f"[save] wrote {ckpt_dir}")

    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        accum_count = 0
        accum_loss = 0.0
        accum_samples = 0
        for batch in train_loader:
            # batch_size=1 时 batch 就是 [sample]。
            # loss_scale = grad_accum * per_device_batch_size：跨 micro-step 累积梯度后
            # 等效 reduction='mean' 的 batch loss，避免 lr 实际被放大。
            #
            # sync_grads：只有"本 micro-step 之后立刻 optimizer.step()"时才让 DDP 同步。
            # 前 (grad_accum-1) 个 micro-step 用 no_sync()，减少 all-reduce 频次。
            # batch 是 DataLoader 输出 (per_device_bs 条样本)；外层 accum_count 计的
            # 是 micro-step 次数（不是样本数）。
            is_last_micro = (accum_count + 1 >= args.grad_accum)
            _, stats = run_step_on_batch(
                bundle, batch,
                analysis_weight=args.analysis_weight,
                teacher_max_new_tokens=args.teacher_max_new_tokens,
                teacher_temperature=args.teacher_temperature,
                max_length=args.max_length,
                loss_scale=float(max(args.grad_accum, 1) * max(args.per_device_batch_size, 1)),
                sync_grads=(world_size <= 1) or is_last_micro,
                skip_teacher=args.skip_teacher,
            )
            accum_loss += stats.loss_sum
            accum_samples += stats.n_samples
            if stats.n_samples <= 0:
                continue
            accum_count += 1

            if accum_count >= args.grad_accum:
                finish_optimizer_step(epoch, "grad_accum", accum_loss, accum_samples)
                accum_loss = 0.0
                accum_samples = 0
                accum_count = 0

                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop_training = True
                    break
        if accum_count > 0 and not stop_training:
            # 尾批：必同步（is_last_micro=True 那批可能没遇到，这里再做一次
            # 兜底 all-reduce 让梯度跨 rank 求平均）。注意 finish_optimizer_step 已
            # 改成不再放大尾批梯度幅度。
            finish_optimizer_step(epoch, "tail", accum_loss, accum_samples)
            accum_loss = 0.0
            accum_samples = 0
            accum_count = 0
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
        if stop_training:
            break

    # ---- 最终保存 ----
    if is_rank0(rank) and not args.check:
        final_dir = output_dir / "final"
        bundle.unwrap().save_pretrained(str(final_dir))
        # 同时存一份 processor 配置，避免下游 eval/probe 误用别处 tokenizer
        # 而 silent 漂移（保存量很小，几十 KB）。
        try:
            bundle.processor.save_pretrained(str(final_dir))
        except Exception as e:
            print(f"[done][warn] save processor fail (skip): {e}")
        print(f"[done] final adapter → {final_dir}")
    if is_rank0(rank) and tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()
