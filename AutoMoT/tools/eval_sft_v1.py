"""SFT v1 离线评估 — 跑 val.jsonl，输出 4 个核心指标 + anchor=12 sanity。

复用 AutoMoT/qwen3vl_local/engine.py 的 LocalQwen3VLInstructEngine 做推理；
LoRA adapter 用 peft 加载到 base model。

指标（与 tools/SFT_V1_PLAN.md §8 一致）：
  - keep_accuracy:      保持类样本 STATUS == GT 的比例
  - advance_accuracy:   推进类样本 STATUS == GT 的比例
  - early_advance_rate: 保持类样本 STATUS == next(GT) 的比例（核心痛点）
  - anchor12_sanity:    anchor=12 fail case 上 STATUS 是否回到 initial

cache_system_prompt 默认开启：所有样本 system prompt 相同，prefix KV cache
复用可省约 50% 推理时间。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[1]
_PROJECT_ROOT = _THIS_FILE.parents[2]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.engine import LocalQwen3VLInstructEngine  # noqa: E402
from qwen3vl_local.prompt_pipeline import (  # noqa: E402
    get_full_sequence,
    parse_vlm_output,
)

# HF 离线开关。
import os  # noqa: E402
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


# ---------------------------------------------------------------------------
# LoRA 加载
# ---------------------------------------------------------------------------

def attach_lora_adapter(engine: LocalQwen3VLInstructEngine, adapter_dir: str) -> None:
    """把训好的 LoRA adapter 挂到 engine.model 上。

    engine.load() 已经把 base model 放到设备上，这里只需要 peft 包一层。
    """
    from peft import PeftModel
    print(f"[eval] attaching LoRA adapter from {adapter_dir}")
    engine.model = PeftModel.from_pretrained(
        engine.model,
        adapter_dir,
        is_trainable=False,
    )
    engine.model.eval()


# ---------------------------------------------------------------------------
# 数据集读取
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> List[Dict]:
    """逐行读 jsonl。空行容错。"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def extract_assistant_target(sample: Dict) -> Dict[str, str]:
    """从 messages[-1] 取出 GT 字段。"""
    assistant_text = sample["messages"][-1]["content"]
    parsed = parse_vlm_output(assistant_text)
    return {
        "status": parsed.get("status"),
        "subgoal": parsed.get("subgoal"),
    }


def reconstruct_prompts(sample: Dict) -> Dict[str, str]:
    """从 jsonl 还原 system_prompt / user_prompt 字符串与 image 路径。

    engine.generate 接受单独的 system_prompt + user_prompt + images 三件。
    user_content 在 build_sft_dataset_v1 里前置了多个 <image>，这里去掉。
    """
    system = sample["messages"][0]["content"]
    user_raw = sample["messages"][1]["content"]
    # 去掉前置的 <image>...<image>\n。
    user = user_raw.lstrip()
    while user.startswith("<image>"):
        user = user[len("<image>"):]
    user = user.lstrip("\n")
    return {"system": system, "user": user, "images": sample["images"]}


# ---------------------------------------------------------------------------
# 单样本推理
# ---------------------------------------------------------------------------

def predict_status(
    engine: LocalQwen3VLInstructEngine,
    sample: Dict,
    images_loader,
) -> Optional[str]:
    """对一条样本跑推理，解析出 STATUS。失败返回 None。"""
    pieces = reconstruct_prompts(sample)
    pil_images = images_loader(pieces["images"])
    raw_text, _ = engine.generate(
        system_prompt=pieces["system"],
        user_prompt=pieces["user"],
        images=pil_images,
        cache_dir=None,
    )
    parsed = parse_vlm_output(raw_text)
    return parsed.get("status")


def next_event_in_seq(scenario: str, status: Optional[str]) -> Optional[str]:
    if status is None:
        return None
    seq = get_full_sequence(scenario)
    try:
        idx = seq.index(status)
    except ValueError:
        return None
    return seq[idx + 1] if idx + 1 < len(seq) else None


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-jsonl", type=str,
                        default=str(_AUTOMOT_ROOT / "checkpoints" / "sft_v1_data" / "val.jsonl"))
    parser.add_argument("--model-dir", type=str,
                        default=str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B-Instruct"))
    parser.add_argument("--lora-dir", type=str,
                        default=str(_AUTOMOT_ROOT / "checkpoints" / "sft_v1_lora"),
                        help="设为空字符串则只评估 base 模型（baseline）。")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 表示评估全部 val 样本，>0 时只评估前 N 条做快速验收。")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--cache-system-prompt", action="store_true", default=True,
                        help="复用 system prompt 的 KV prefix，节省推理时间。")
    parser.add_argument("--output-json", type=str,
                        default=str(_AUTOMOT_ROOT / "eval_json" / "sft_v1_metrics.json"))
    args = parser.parse_args()

    samples = read_jsonl(args.val_jsonl)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"[eval] loaded {len(samples)} samples from {args.val_jsonl}")

    # 启动 engine + 可选挂 LoRA。
    engine = LocalQwen3VLInstructEngine(
        checkpoint_dir=pathlib.Path(args.model_dir),
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_gen_tokens=96,        # ANALYSIS + STATUS + SUBGOAL 一般 < 80 token
        temperature=0.0,
        do_sample=False,
        save_cache=False,
        cache_system_prompt=args.cache_system_prompt,
    )
    if args.lora_dir:
        attach_lora_adapter(engine, args.lora_dir)

    # eval 时 jsonl 已经给了绝对路径，直接 PIL 打开就够。
    # 与 runner load_lead_rgb_clip 的 model_input_size 处理保持一致。
    from PIL import Image  # type: ignore

    def images_loader(paths: List[str]):
        return [Image.open(p).convert("RGB") for p in paths]

    # 计数器。
    n_keep = n_keep_correct = n_early_adv = 0
    n_adv = n_adv_correct = 0
    per_scenario: Dict[str, Counter] = defaultdict(Counter)

    for i, sample in enumerate(samples):
        scenario = sample["scenario"]
        gt = extract_assistant_target(sample)
        gt_status = gt["status"]
        is_trans = sample.get("is_transition_sample", False)

        try:
            pred = predict_status(engine, sample, images_loader)
        except Exception as e:
            print(f"[err {i}] {e}")
            pred = None

        next_gt = next_event_in_seq(scenario, gt_status)

        if not is_trans:
            n_keep += 1
            if pred == gt_status:
                n_keep_correct += 1
                per_scenario[scenario]["keep_correct"] += 1
            elif pred is not None and pred == next_gt:
                n_early_adv += 1
                per_scenario[scenario]["early_advance"] += 1
            per_scenario[scenario]["keep_total"] += 1
        else:
            n_adv += 1
            if pred == gt_status:
                n_adv_correct += 1
                per_scenario[scenario]["adv_correct"] += 1
            per_scenario[scenario]["adv_total"] += 1

        if (i + 1) % 50 == 0:
            print(f"[eval] {i+1}/{len(samples)}  "
                  f"keep_acc={n_keep_correct/max(1,n_keep):.3f}  "
                  f"adv_acc={n_adv_correct/max(1,n_adv):.3f}  "
                  f"early_adv={n_early_adv/max(1,n_keep):.3f}")

    metrics = {
        "n_total": len(samples),
        "n_keep": n_keep,
        "n_advance": n_adv,
        "keep_accuracy": n_keep_correct / max(1, n_keep),
        "advance_accuracy": n_adv_correct / max(1, n_adv),
        "early_advance_rate": n_early_adv / max(1, n_keep),
        "per_scenario": {k: dict(v) for k, v in per_scenario.items()},
        "config": vars(args),
    }
    out_path = pathlib.Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[done] metrics written to {out_path}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_scenario"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
