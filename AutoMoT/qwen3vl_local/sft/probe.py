"""SFT 单样本级探针：随机选 N 个场景的样本，把"输入 + 输出 + 损失"全保存。

eval.py 是聚合视角：跑完整 val 出 keep_acc / early_advance / per_scenario，
看不到"具体某条样本模型当时看到了什么、回了什么、错在哪个 token"。

本脚本补这个空白：
- 按 scenario 随机抽 K 条样本（同 seed 可复现，便于多次跑横向对比）；
- 每条样本一个独立目录，把人工 review 需要的全部材料一次写齐：
    * 历史 4 张 RGB（符号链接到原文件，windows 退化为复制）；
    * 完整 system / user prompt 文本（去 <image> 占位，还原训练时模型实际看到的文字）；
    * GT assistant 全文（ANALYSIS + STATUS + SUBGOAL 三行）；
    * 模型预测 raw 文本（与 eval.py 同一推理路径）；
    * expert_analysis.txt / language_compare.json：base teacher 专家语言 vs 模型自己的 ANALYSIS；
    * **token-level loss**：teacher-forced 给 assistant 一遍，逐 token 给出 NLL；
      额外用与 train.py 内置 mask 同款 regex 算"weighted loss"，直接看到
      ANALYSIS body 0.5 / 结构字面 1.0 / 事件名 1.0 真正用于训练的损失分布；
    * meta.json：lora_dir / model_dir / scenario / run_id / anchor / 推理耗时；
    * overview.md：把上面所有内容合并到一页，单文件即可人工复查。

输出布局（与 train.sh 同根 — OUTPUT_DIR 平铺，B 方案）：
  <save_root>/eval_cases/
    <scenario>__<run_id>__<anchor>/
      input_images/00.jpg ... 03.jpg
      system_prompt.txt
      user_prompt.txt
      gt.txt
      pred.txt
      expert_analysis.txt
      language_compare.json
      token_loss.json
      meta.json
      overview.md

典型用法（**从 AutoMoT/ 目录运行**，远程默认 cwd）：

```bash
# 默认跑 base，抽 16 个场景样本（每场景 4 条）做人工复查
GPU_IDS=0 python qwen3vl_local/sft/probe.py \
  --save-root checkpoints/sft_lora/latest \
  --num-per-scenario 4 --seed 0

# 只有明确要看 LoRA 时才传 adapter；同 seed 选中样本完全一致，方便并排比较
GPU_IDS=0 python qwen3vl_local/sft/probe.py \
  --lora-dir checkpoints/sft_lora/latest/checkpoint-900 \
  --save-root checkpoints/sft_lora/latest \
  --num-per-scenario 4 --seed 0 --case-suffix "_lora"

# 只看 Accident / Construction 两个场景
GPU_IDS=0 python qwen3vl_local/sft/probe.py \
  --save-root checkpoints/sft_lora/latest \
  --scenarios Accident,Construction --num-per-scenario 6 --seed 7
```

设计取舍：
- 不接 torchrun。probe 输出量小、需要 per-sample 写文件，单卡顺序跑可读性最高；
  多卡分片反而会让 stdout 交错难读。需要并行时直接起多个 python 进程，各自跑
  不同 --scenarios 切分；
- 默认自动挑 1 张空闲 GPU，并覆盖外层残留的 `CUDA_VISIBLE_DEVICES`；
- token-level loss 走"两次 encode 拼接"的近似定位（precise offset_mapping 在
  Qwen3-VL processor 上需要特殊处理，工程价值不大）；
- 图像默认 symlink，windows 失败退化 copy。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# HF 离线开关必须在 import transformers 之前。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _cli_value(name: str) -> Optional[str]:
    """在 argparse/import torch 前轻量读取命令行值，便于先决定 CUDA mask。"""

    prefix = name + "="
    for i, item in enumerate(sys.argv[1:]):
        if item == name and i + 2 <= len(sys.argv[1:]):
            return sys.argv[i + 2]
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def _pick_idle_gpus(n: int = 1) -> str:
    """用 nvidia-smi 挑 n 张最空闲 GPU；没有 nvidia-smi 时返回空串。"""

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[1]), int(parts[2]), parts[0]))
        except ValueError:
            continue
    rows.sort(key=lambda x: (x[0], x[1], int(x[2]) if x[2].isdigit() else 9999))
    return ",".join(row[2] for row in rows[:n])


def _normalize_gpu_ids(value: str) -> str:
    """规范化 GPU id 列表字符串，去掉空白和空项。"""

    ids = [part.strip() for part in str(value).split(",") if part.strip()]
    return ",".join(ids)


def _maybe_set_idle_gpu_mask() -> None:
    """probe 默认自动挑 1 张空闲 GPU；--device 显式传 cpu / cuda[:N] 时不覆盖 CVD。"""
    device_arg = _cli_value("--device")
    if device_arg and device_arg.strip().lower() not in ("", "auto"):
        print(f"[gpu] using explicit --device {device_arg}; CUDA_VISIBLE_DEVICES is not modified")
        return
    pinned = _normalize_gpu_ids(os.environ.get("GPU_IDS", ""))
    if pinned:
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = pinned
        print(
            f"[gpu] using explicit GPU_IDS={pinned}; "
            f"process uses cuda:0/auto; previous={previous or '<unset>'}"
        )
        return
    selected = _pick_idle_gpus(1)
    if selected:
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        print(
            f"[gpu] auto selected idle CUDA_VISIBLE_DEVICES={selected}; "
            f"process uses cuda:0/auto; previous={previous or '<unset>'}"
        )


_maybe_set_idle_gpu_mask()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from qwen3vl_local.engine import LocalQwen3VLInstructEngine  # noqa: E402
from qwen3vl_local.prompt_pipeline import parse_vlm_output  # noqa: E402
from qwen3vl_local.sft.train import (  # noqa: E402
    _TEACHER_SYSTEM_PROMPT,
    build_teacher_user_prompt,
    postprocess_teacher,
)

# 与 train.py 内置权重 / check_loss_mask.py 同款 regex：ANALYSIS body 可通过 env 调权，
# 其它 assistant token（结构字面、事件名、tail/EOS）权重为 1。
ANALYSIS_WEIGHT = float(os.environ.get("SFT_ANALYSIS_WEIGHT", "0.5"))
_FULL_PATTERN = re.compile(
    r"ANALYSIS:[ \t]*"
    r"(?P<analysis>[^\n]*?)"
    r"\s*\nSTATUS:[ \t]*"
    r"(?P<status>\S[^\n]*?)"
    r"\s*\nSUBGOAL:[ \t]*"
    r"(?P<subgoal>\S[^\n]*)",
    flags=re.DOTALL,
)


# --------------------------------------------------------------------------- #
# 数据 / 图像 / prompt 辅助函数（与 eval.py 同口径，复制以避免互相 import）
# --------------------------------------------------------------------------- #

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取 jsonl 为 dict 列表，自动跳过空行。"""

    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def reconstruct_prompts(sample: Dict[str, Any]) -> Dict[str, Any]:
    """从 jsonl 还原 system / user 文本 + image 路径，与 eval 同口径。"""
    system = sample["messages"][0]["content"]
    user_raw = sample["messages"][1]["content"]
    # 训练 jsonl 在 user 文本前面塞了多个 <image> 占位（历史上用于让模板对齐图片）。
    # 本地 engine 使用结构化图像内容，不需要文本占位，去掉以免被当成普通文本。
    user = user_raw.lstrip()
    while user.startswith("<image>"):
        user = user[len("<image>"):]
    user = user.lstrip("\n")
    return {"system": system, "user": user, "images": sample["images"]}


def extract_assistant_target(sample: Dict[str, Any]) -> str:
    """assistant 段完整 GT 文本（包含 ANALYSIS + STATUS + SUBGOAL）。"""
    return sample["messages"][-1]["content"]


def _teacher_meta_for_probe(sample: Dict[str, Any]) -> Dict[str, str]:
    """还原训练同款 teacher prompt 的 PRIVILEGED 元信息。"""

    gt_parsed = parse_vlm_output(extract_assistant_target(sample))
    meta = sample.get("teacher_meta_input")
    if isinstance(meta, dict):
        return {
            "target_status": str(meta.get("target_status") or gt_parsed.get("status") or "unknown"),
            "transition": str(meta.get("transition") or ("advance" if sample.get("is_transition_sample") else "keep")),
            "memory_in_status": str(meta.get("memory_in_status") or "unknown"),
        }
    return {
        "target_status": str(gt_parsed.get("status") or "unknown"),
        "transition": "advance" if sample.get("is_transition_sample") else "keep",
        "memory_in_status": "unknown",
    }


def generate_expert_analysis(
    engine: LocalQwen3VLInstructEngine,
    sample: Dict[str, Any],
    pieces: Dict[str, Any],
    pil_images: List[Any],
) -> Dict[str, Any]:
    """用 base teacher prompt 生成专家 ANALYSIS，和模型自己的 ANALYSIS 做人工对照。"""

    meta = _teacher_meta_for_probe(sample)
    try:
        raw_text, _ = engine.generate(
            system_prompt=_TEACHER_SYSTEM_PROMPT,
            user_prompt=build_teacher_user_prompt(pieces["user"], meta),
            images=pil_images,
            cache_dir=None,
        )
        analysis, fallback = postprocess_teacher(raw_text)
        return {
            "analysis": analysis,
            "raw_text": raw_text,
            "fallback": fallback,
            "error": None,
            "teacher_meta_input": meta,
        }
    except Exception as exc:
        return {
            "analysis": None,
            "raw_text": "",
            "fallback": True,
            "error": str(exc),
            "teacher_meta_input": meta,
        }


# --------------------------------------------------------------------------- #
# 样本挑选
# --------------------------------------------------------------------------- #

def select_samples(
    samples: List[Dict[str, Any]],
    scenarios: Optional[List[str]],
    num_per_scenario: int,
    seed: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    """按 scenario 分桶后各抽 N 条；返回 (原始 idx, sample) 列表。

    保留原始 idx 是为了让 overview / token_loss 里能写回 val.jsonl 行号，方便
    与 eval.py 的 predictions.jsonl[sample_idx] 对应起来。
    """
    by_scenario: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, s in enumerate(samples):
        sc = s.get("scenario", "<unknown>")
        if scenarios and sc not in scenarios:
            continue
        by_scenario[sc].append((idx, s))

    if not by_scenario:
        raise RuntimeError(
            f"未找到任何匹配的场景样本：scenarios={scenarios}，请检查 --val-jsonl"
        )

    rng = random.Random(seed)
    picked: List[Tuple[int, Dict[str, Any]]] = []
    for sc in sorted(by_scenario.keys()):
        bucket = by_scenario[sc]
        rng.shuffle(bucket)
        picked.extend(bucket[:num_per_scenario])
    return picked


# --------------------------------------------------------------------------- #
# 图像落盘（符号链接 → 复制回退）
# --------------------------------------------------------------------------- #

def link_or_copy(src: str, dst: pathlib.Path) -> None:
    """优先创建符号链接；失败（Windows 无权限 / 跨盘 / 文件系统不支持）时退化为复制。"""
    src_path = pathlib.Path(src)
    if not src_path.exists():
        print(f"[probe][warn] 源图不存在，跳过：{src_path}")
        return
    if dst.exists():
        dst.unlink()
    try:
        os.symlink(src_path.resolve(), dst)
    except (OSError, NotImplementedError):
        shutil.copyfile(src_path, dst)


# --------------------------------------------------------------------------- #
# Token 级 loss
# --------------------------------------------------------------------------- #

@torch.no_grad()
def compute_token_loss(
    engine: LocalQwen3VLInstructEngine,
    sample: Dict[str, Any],
    pieces: Dict[str, Any],
    pil_images: List[Any],
    gt_text: str,
) -> Dict[str, Any]:
    """用 teacher-forced 方式跑一遍，给 assistant 段逐 token 拆出 NLL 和训练 loss 权重。

    流程：
      1. apply_chat_template(messages, add_generation_prompt=True) 得到"system+user
         +assistant 起点"的前缀文本 prefix_text；
      2. apply_chat_template(messages + assistant, add_generation_prompt=False) 得到
         整段聊天文本 full_text；
      3. processor 各编码一次，full_ids 长度 - prefix_ids 长度 = assistant token 数；
      4. 用 full_ids 跑一次 model.forward(use_cache=False, return_dict=True) 拿 logits；
      5. shift logits/labels 后对 assistant 区段算 cross entropy（不平均，留 per-token）；
      6. 用 _FULL_PATTERN 在 gt_text 上找 ANALYSIS / STATUS / SUBGOAL 三段 char span，
         按 train.py 同口径给 token 权重：ANALYSIS body=SFT_ANALYSIS_WEIGHT，
         其它 assistant token=1.0；
      7. 输出逐 token 列表 + 原始均值 / 加权均值 / 分段均值。

    注意：
      - 这里 weight 是按 *char 长度* + tokenizer 重切换算的近似（assistant_only
        encode 与 full encode 在边界 token 可能不完全一致），所以输出里多带一个
        `assistant_offset` 字段，便于人工核对；
      - vision token 数量大，但只跑一次 forward，逐条样本顺序处理。
    """
    processor = engine.processor
    model = engine.model
    device = next(model.parameters()).device

    # ---- 1) 拼 messages（带 assistant）----
    messages = engine.build_messages(pieces["system"], pieces["user"], pil_images)
    messages_with_gt = messages + [{"role": "assistant", "content": gt_text}]

    # ---- 2) 文本展开 + tokenize ----
    prefix_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        messages_with_gt, tokenize=False, add_generation_prompt=False,
    )

    # processor 同时处理视觉 token；只 tokenize 文本拿不到正确的 input_ids 长度，
    # 所以必须走完整 processor(text=..., images=...)。
    prefix_inputs = processor(
        text=[prefix_text],
        images=pil_images if pil_images else None,
        return_tensors="pt",
        padding=True,
    ).to(device)
    full_inputs = processor(
        text=[full_text],
        images=pil_images if pil_images else None,
        return_tensors="pt",
        padding=True,
    ).to(device)

    prefix_len = int(prefix_inputs["input_ids"].shape[1])
    full_len = int(full_inputs["input_ids"].shape[1])
    assistant_token_count = full_len - prefix_len
    if assistant_token_count <= 0:
        return {
            "error": "assistant_token_count <= 0；prefix/full encode 结果异常",
            "prefix_len": prefix_len,
            "full_len": full_len,
        }

    # ---- 3) 前向计算 ----
    outputs = model(
        **full_inputs,
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits  # [1, T, V]

    # shift：位置 t 的 logits 预测位置 t+1 的 token
    shift_logits = logits[:, :-1, :].float()
    labels = full_inputs["input_ids"]
    shift_labels = labels[:, 1:]

    # assistant 区段在 shift 后的范围：[prefix_len - 1, full_len - 1)
    # （shift_logits[i] 预测 labels[i+1]，所以预测 labels[prefix_len] 的位置是 i=prefix_len-1）
    start = max(0, prefix_len - 1)
    end = full_len - 1
    asst_logits = shift_logits[0, start:end, :]
    asst_labels = shift_labels[0, start:end]

    # 逐 token 交叉熵，reduction='none' 保留每个 token 的 NLL
    nll = F.cross_entropy(asst_logits, asst_labels, reduction="none").detach().float().cpu().tolist()
    asst_ids = asst_labels.detach().cpu().tolist()
    tokenizer = processor.tokenizer
    token_texts = tokenizer.convert_ids_to_tokens(asst_ids)
    token_strs = [tokenizer.decode([tid]) for tid in asst_ids]

    # ---- 4) assistant char-span → token weight ----
    match = _FULL_PATTERN.search(gt_text)
    if match is None:
        # 与 train.py 一致：格式漂移时保守全段按 1.0 监督，不漏训练信号。
        weights = [1.0] * len(nll)
        token_buckets = ["fallback_full"] * len(nll)
    else:
        analysis_start, analysis_end = match.span("analysis")
        status_start, status_end = match.span("status")
        subgoal_start, subgoal_end = match.span("subgoal")
        gt_enc = tokenizer(gt_text, return_offsets_mapping=True, add_special_tokens=False)
        gt_offsets = [tuple(x) for x in gt_enc["offset_mapping"]]

        def _overlap(off: Tuple[int, int], lo: int, hi: int) -> bool:
            """判断 token offset 是否与 assistant 文本片段相交。"""

            return off[0] < hi and off[1] > lo

        weights = []
        token_buckets = []
        for off in gt_offsets[:len(nll)]:
            off_t = (int(off[0]), int(off[1]))
            if _overlap(off_t, analysis_start, analysis_end):
                weights.append(ANALYSIS_WEIGHT)
                token_buckets.append("analysis")
            elif _overlap(off_t, status_start, status_end):
                weights.append(1.0)
                token_buckets.append("status")
            elif _overlap(off_t, subgoal_start, subgoal_end):
                weights.append(1.0)
                token_buckets.append("subgoal")
            else:
                weights.append(1.0)
                token_buckets.append("literal")
        # 完整 chat 模板通常会在 assistant 内容后追加 <|im_end|>，训练尾部/EOS 权重也是 1.0。
        while len(weights) < len(nll):
            weights.append(1.0)
            token_buckets.append("tail")
        weights = weights[:len(nll)]
        token_buckets = token_buckets[:len(nll)]

    # ---- 5) 汇总平均 ----
    mean_raw = float(sum(nll) / max(1, len(nll)))
    weighted_sum = sum(n * w for n, w in zip(nll, weights))
    weight_count = sum(weights)
    mean_weighted = float(weighted_sum / weight_count) if weight_count > 0 else 0.0

    def _bucket_mean(name: str) -> float:
        """计算指定分桶的平均 NLL；分桶为空时返回 0。"""

        vals = [n for n, b in zip(nll, token_buckets) if b == name]
        return float(sum(vals) / len(vals)) if vals else 0.0

    mean_analysis = _bucket_mean("analysis")
    mean_status = _bucket_mean("status")
    mean_subgoal = _bucket_mean("subgoal")
    mean_literal_tail = float(
        sum(n for n, b in zip(nll, token_buckets) if b in ("literal", "tail"))
        / max(1, sum(1 for b in token_buckets if b in ("literal", "tail")))
    )

    per_token = [
        {
            "i": k,
            "token_id": int(tid),
            "token": token_texts[k],
            "text": token_strs[k],
            "nll": float(nll[k]),
            "weight": float(weights[k]),
            "mask": float(weights[k]),  # 兼容旧版消费方的别名。
            "bucket": token_buckets[k],
        }
        for k, tid in enumerate(asst_ids)
    ]

    return {
        "prefix_len": prefix_len,
        "full_len": full_len,
        "assistant_token_count": assistant_token_count,
        "mean_loss_raw": mean_raw,
        "mean_loss_weighted_train": mean_weighted,
        "mean_loss_analysis_body": mean_analysis,
        "mean_loss_status": mean_status,
        "mean_loss_subgoal": mean_subgoal,
        "mean_loss_literal_tail": mean_literal_tail,
        "analysis_weight": ANALYSIS_WEIGHT,
        # 兼容旧 overview / jsonl 消费方；语义已不再是“唯一训练 loss”。
        "mean_loss_status_subgoal_only": float((mean_status + mean_subgoal) / 2.0),
        "mean_loss_masked_literals": mean_literal_tail,
        "per_token": per_token,
    }


# --------------------------------------------------------------------------- #
# 总览 markdown 渲染
# --------------------------------------------------------------------------- #

def _md_cell(value: Any, max_chars: int = 600) -> str:
    """把任意文本压成 Markdown 表格单元，避免 ANALYSIS 里的管道/换行撑坏表格。"""

    text = "" if value is None else str(value)
    text = text.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def render_overview_md(
    case_dir: pathlib.Path,
    sample: Dict[str, Any],
    pieces: Dict[str, Any],
    gt_text: str,
    pred_raw: str,
    pred_parsed: Dict[str, Optional[str]],
    token_loss: Dict[str, Any],
    elapsed: float,
    meta: Dict[str, Any],
    expert_info: Optional[Dict[str, Any]] = None,
) -> str:
    """生成一页 markdown，让人单文件即可复查这条 case。

    刻意把 system/user/gt/pred 全文嵌入：不靠点击别的文件就能阅读完整上下文。
    """
    gt_parsed = parse_vlm_output(gt_text)
    lines: List[str] = []
    lines.append(f"# Case: {sample.get('scenario')}/{sample.get('run_id')} anchor={sample.get('anchor')}")
    lines.append("")
    lines.append(f"- val.jsonl sample_idx: **{meta.get('sample_idx')}**")
    lines.append(f"- is_transition_sample: {sample.get('is_transition_sample', False)}")
    lines.append(f"- inference elapsed: {elapsed:.3f}s")
    lines.append(f"- lora_dir: `{meta.get('lora_dir') or '<base>'}`")
    lines.append(f"- model_dir: `{meta.get('model_dir')}`")
    lines.append("")

    lines.append("## Input images")
    for k, p in enumerate(sample.get("images", [])):
        lines.append(f"- `input_images/{k:02d}.jpg` ← `{p}`")
    lines.append("")

    lines.append("## GT vs Pred (parsed)")
    lines.append("| field | GT | Pred |")
    lines.append("|---|---|---|")
    lines.append(f"| status | `{gt_parsed.get('status')}` | `{pred_parsed.get('status')}` |")
    lines.append(f"| subgoal | `{gt_parsed.get('subgoal')}` | `{pred_parsed.get('subgoal')}` |")
    lines.append(f"| analysis (片段) | {(gt_parsed.get('analysis') or '')[:80]} | {(pred_parsed.get('analysis') or '')[:80]} |")
    lines.append("")

    if expert_info is not None:
        lines.append("## Expert vs Model Analysis")
        if expert_info.get("error"):
            lines.append(f"- expert generation error: `{expert_info.get('error')}`")
        lines.append("| source | analysis |")
        lines.append("|---|---|")
        lines.append(f"| expert teacher | {_md_cell(expert_info.get('analysis'))} |")
        lines.append(f"| model output | {_md_cell(pred_parsed.get('analysis'))} |")
        gt_analysis = gt_parsed.get("analysis") or ""
        if gt_analysis and gt_analysis != "__TEACHER_PENDING__":
            lines.append(f"| materialized GT | {_md_cell(gt_analysis)} |")
        lines.append("")

    lines.append("## Token-level loss summary")
    if "error" in token_loss:
        lines.append(f"- error: {token_loss['error']}")
    else:
        lines.append(f"- assistant_token_count: **{token_loss['assistant_token_count']}**")
        lines.append(f"- mean_loss_raw (所有 assistant token): **{token_loss['mean_loss_raw']:.4f}**")
        lines.append(f"- mean_loss_weighted_train (按训练权重汇总): **{token_loss['mean_loss_weighted_train']:.4f}**")
        lines.append(f"- mean_loss_analysis_body (weight={token_loss.get('analysis_weight', ANALYSIS_WEIGHT):.2f}): {token_loss['mean_loss_analysis_body']:.4f}")
        lines.append(f"- mean_loss_status / subgoal: {token_loss['mean_loss_status']:.4f} / {token_loss['mean_loss_subgoal']:.4f}")
        lines.append(f"- mean_loss_literal_tail (结构字面 + tail/EOS, weight=1): {token_loss['mean_loss_literal_tail']:.4f}")
        lines.append("")
        # 头 32 个 token 的逐 token loss 直接列在 markdown 里，方便人眼快速找到异常 token。
        lines.append("Top tokens (first 32, w=train weight):")
        lines.append("```")
        for tok in token_loss.get("per_token", [])[:32]:
            lines.append(
                f"  i={tok['i']:3d} w={tok['weight']:.2f} "
                f"bucket={tok.get('bucket', '?'):8s} nll={tok['nll']:7.3f} text={tok['text']!r}"
            )
        lines.append("```")
    lines.append("")

    lines.append("## System prompt")
    lines.append("```")
    lines.append(pieces["system"])
    lines.append("```")
    lines.append("")

    lines.append("## User prompt")
    lines.append("```")
    lines.append(pieces["user"])
    lines.append("```")
    lines.append("")

    lines.append("## GT (assistant)")
    lines.append("```")
    lines.append(gt_text)
    lines.append("```")
    lines.append("")

    lines.append("## Pred (assistant, raw output)")
    lines.append("```")
    lines.append(pred_raw or "<inference error>")
    lines.append("```")
    lines.append("")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> None:
    """probe 主入口：抽样、推理、计算 token loss，并写每个 case 的复查目录。"""

    parser = argparse.ArgumentParser(description="SFT 单样本级探针（随机场景 dump）")
    parser.add_argument("--val-jsonl", type=str,
                        default=str(_AUTOMOT_ROOT / "checkpoints" / "sft_data_pending" / "val.jsonl"))
    parser.add_argument("--model-dir", type=str,
                        default=str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B-Instruct"))
    parser.add_argument("--lora-dir", type=str,
                        default="",
                        help="可选 LoRA adapter；默认空串跑 base 模型且不会导入 peft。")
    parser.add_argument("--save-root", type=str,
                        default=str(_AUTOMOT_ROOT / "checkpoints" / "sft_lora" / "latest"),
                        help="case dump 写到 <save-root>/eval_cases/<scenario>__<run>__<anchor>/")
    parser.add_argument("--case-suffix", type=str, default="",
                        help="给 case 目录名加后缀（如 '_base' / '_ckpt300'），方便同一 sample 多次 dump 不互相覆盖")
    parser.add_argument("--scenarios", type=str, default="",
                        help="逗号分隔过滤；空则全场景。例：--scenarios Accident,Construction")
    parser.add_argument("--num-per-scenario", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--cache-system-prompt", action=argparse.BooleanOptionalAction, default=True)
    # 与 eval.py 同口径：merge_and_unload 是默认，避免 PeftModel wrapper 在
    # Qwen3-VL 上的 forward 错位（详见 PROJECT_CONTEXT.md §18.1）。--no-merge-lora
    # 仅作为调试 PEFT 自身行为的逃生开关。
    parser.add_argument("--merge-lora",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="加载 LoRA 后是否 merge_and_unload。默认 True；"
                             "--no-merge-lora 会复现已知 Qwen3-VL forward bug，仅调试用。")
    # 当前 ANALYSIS 是 teacher 蒸馏真值（80-150 token），96 会截断到只剩
    # ANALYSIS 段。默认 256 与 eval.py 一致。
    parser.add_argument("--max-gen-tokens", type=int, default=256,
                        help="自回归生成 token 数上限。默认 256；建议 ≥ 200。")
    parser.add_argument("--skip-token-loss", action="store_true",
                        help="不算 token-level loss（只 dump prompt/GT/pred + 图），加速 probe")
    parser.add_argument("--expert-compare",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="是否生成专家 ANALYSIS 并与模型自己的 ANALYSIS 对比。默认开启。")
    args = parser.parse_args()

    case_root = pathlib.Path(args.save_root) / "eval_cases"
    case_root.mkdir(parents=True, exist_ok=True)
    from qwen3vl_local.run_log import install_output_log
    install_output_log(case_root)

    samples = read_jsonl(args.val_jsonl)
    ds_ver = samples[0].get("dataset_version", "unknown") if samples else "unknown"
    print(f"[probe] dataset_version={ds_ver}")
    if ds_ver == "pending":
        print("[probe][note] dataset_version=pending: gt.txt 里 ANALYSIS 段是 __TEACHER_PENDING__ 占位；"
              "STATUS/SUBGOAL 对照与 token-level loss 不受影响，但 ANALYSIS body 不能做内容对比。"
              "需要 teacher ANALYSIS 真值，跑 build_teacher.py 离线物化 val。")
    scenarios_filter = [s.strip() for s in args.scenarios.split(",") if s.strip()] or None
    picked = select_samples(samples, scenarios_filter, args.num_per_scenario, args.seed)
    print(f"[probe] selected {len(picked)} samples from {len(samples)} total "
          f"(scenarios={scenarios_filter or 'ALL'}, per={args.num_per_scenario}, seed={args.seed})")

    # 启动 engine 并可选挂 LoRA（不传则跑 base）。
    engine = LocalQwen3VLInstructEngine(
        checkpoint_dir=pathlib.Path(args.model_dir),
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_gen_tokens=args.max_gen_tokens,
        temperature=0.0,
        do_sample=False,
        save_cache=False,
        cache_system_prompt=args.cache_system_prompt,
    )
    engine.load()
    if not args.skip_token_loss:
        tokenizer = engine.processor.tokenizer
        assert getattr(tokenizer, "is_fast", False), (
            "probe token-level loss needs a Fast tokenizer (PreTrainedTokenizerFast); "
            "current tokenizer is not Fast, so return_offsets_mapping is unavailable."
        )
    from PIL import Image  # type: ignore

    # 专家语言对照必须在 attach/merge LoRA 之前跑，保证 expert 是 base teacher prompt。
    expert_compare_by_idx: Dict[int, Dict[str, Any]] = {}
    if args.expert_compare:
        print("[probe] expert_compare enabled: generating base-teacher ANALYSIS before attaching LoRA")
        for sample_idx, sample in picked:
            pieces_for_teacher = reconstruct_prompts(sample)
            pil_images_for_teacher = [Image.open(p).convert("RGB") for p in pieces_for_teacher["images"]]
            expert_compare_by_idx[sample_idx] = generate_expert_analysis(
                engine, sample, pieces_for_teacher, pil_images_for_teacher
            )

    if args.lora_dir:
        # 与 eval.py 统一走 engine 的 LoRA attach 入口。merge=True 是推荐路径；
        # merge=False 仅用于诊断 PEFT wrapper 行为。
        engine.attach_lora_adapter(args.lora_dir, merge=args.merge_lora)

    summary_records: List[Dict[str, Any]] = []
    for sample_idx, sample in picked:
        scenario = sample.get("scenario", "unknown")
        run_id = sample.get("run_id", "norun")
        anchor = sample.get("anchor", "noanchor")
        case_name = f"{scenario}__{run_id}__{anchor}{args.case_suffix}"
        case_dir = case_root / case_name
        (case_dir / "input_images").mkdir(parents=True, exist_ok=True)

        # 1) 为历史图创建符号链接或复制
        for k, p in enumerate(sample.get("images", [])):
            link_or_copy(p, case_dir / "input_images" / f"{k:02d}.jpg")

        # 2) 拼 prompt 并推理
        pieces = reconstruct_prompts(sample)
        pil_images = [Image.open(p).convert("RGB") for p in pieces["images"]]
        gt_text = extract_assistant_target(sample)
        expert_info = expert_compare_by_idx.get(sample_idx)

        t0 = time.time()
        try:
            raw_text, _ = engine.generate(
                system_prompt=pieces["system"],
                user_prompt=pieces["user"],
                images=pil_images,
                cache_dir=None,
            )
            pred_parsed = parse_vlm_output(raw_text)
            inference_err: Optional[str] = None
        except Exception as e:
            raw_text = ""
            pred_parsed = {}
            inference_err = str(e)
            print(f"[probe][err] sample_idx={sample_idx} inference failed: {e}")
        elapsed = time.time() - t0

        # 3) token-level loss
        if not args.skip_token_loss:
            try:
                token_loss = compute_token_loss(engine, sample, pieces, pil_images, gt_text)
            except Exception as e:
                token_loss = {"error": f"token-loss failed: {e}"}
                print(f"[probe][warn] sample_idx={sample_idx} token-loss failed: {e}")
        else:
            token_loss = {"skipped": True}

        # 4) 写文件
        (case_dir / "system_prompt.txt").write_text(pieces["system"], encoding="utf-8")
        (case_dir / "user_prompt.txt").write_text(pieces["user"], encoding="utf-8")
        (case_dir / "gt.txt").write_text(gt_text, encoding="utf-8")
        (case_dir / "pred.txt").write_text(raw_text or "<inference error>", encoding="utf-8")
        if expert_info is not None:
            (case_dir / "expert_analysis.txt").write_text(
                str(expert_info.get("analysis") or "<expert generation error>"),
                encoding="utf-8",
            )
            language_compare = {
                "expert_analysis": expert_info.get("analysis"),
                "model_analysis": pred_parsed.get("analysis"),
                "gt_analysis": parse_vlm_output(gt_text).get("analysis"),
                "gt_analysis_is_pending": parse_vlm_output(gt_text).get("analysis") == "__TEACHER_PENDING__",
                "expert_error": expert_info.get("error"),
                "expert_fallback": expert_info.get("fallback"),
                "teacher_meta_input": expert_info.get("teacher_meta_input"),
            }
            with (case_dir / "language_compare.json").open("w", encoding="utf-8") as f:
                json.dump(language_compare, f, ensure_ascii=False, indent=2)
        with (case_dir / "token_loss.json").open("w", encoding="utf-8") as f:
            json.dump(token_loss, f, ensure_ascii=False, indent=2)

        meta = {
            "sample_idx": sample_idx,
            "scenario": scenario,
            "run_id": run_id,
            "anchor": anchor,
            "is_transition_sample": sample.get("is_transition_sample", False),
            "lora_dir": args.lora_dir,
            "model_dir": args.model_dir,
            "model_dtype": args.torch_dtype,
            "seed": args.seed,
            "case_suffix": args.case_suffix,
            "inference_elapsed_sec": elapsed,
            "inference_error": inference_err,
        }
        with (case_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        overview = render_overview_md(
            case_dir, sample, pieces, gt_text, raw_text, pred_parsed,
            token_loss, elapsed, meta, expert_info=expert_info,
        )
        (case_dir / "overview.md").write_text(overview, encoding="utf-8")

        # 5) 汇总一行（dump 完一条立即落盘，避免长跑中途崩了丢全部进度）
        summary_records.append({
            "sample_idx": sample_idx,
            "case_dir": str(case_dir),
            "scenario": scenario,
            "run_id": run_id,
            "anchor": anchor,
            "is_transition_sample": sample.get("is_transition_sample", False),
            "gt_status": parse_vlm_output(gt_text).get("status"),
            "pred_status": pred_parsed.get("status"),
            "expert_analysis": expert_info.get("analysis") if expert_info else None,
            "pred_analysis": pred_parsed.get("analysis"),
            "mean_loss_raw": token_loss.get("mean_loss_raw"),
            "mean_loss_weighted_train": token_loss.get("mean_loss_weighted_train"),
            "mean_loss_status_subgoal_only": token_loss.get("mean_loss_status_subgoal_only"),
            "elapsed_sec": elapsed,
            "inference_error": inference_err,
        })
        with (case_root / f"_index{args.case_suffix}.jsonl").open("w", encoding="utf-8") as f:
            for r in summary_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[probe] done {scenario}/{run_id}/anchor={anchor} → {case_dir}")

    print(f"\n[probe] all {len(summary_records)} cases dumped under {case_root}")
    print(f"[probe] index: {case_root / f'_index{args.case_suffix}.jsonl'}")


if __name__ == "__main__":
    main()
