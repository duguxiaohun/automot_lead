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

典型用法（**从 AutoMoT/ 目录运行**，远程默认 cwd）：

```bash
# 评估 LoRA adapter（默认路径 checkpoints/sft_v1_lora）
python tools/eval_sft_v1.py

# 只评估 base 模型，作为微调前 baseline
python tools/eval_sft_v1.py --lora-dir ""

# 快速验收前 32 条样本，并跳过 anchor=12 单例
python tools/eval_sft_v1.py --max-samples 32 --skip-anchor12-sanity

# 指定某个 checkpoint step 目录
python tools/eval_sft_v1.py \
  --lora-dir checkpoints/sft_v1_lora/checkpoint-300
```

评估逻辑：
- val.jsonl 里的 assistant message 是 GT，只用于提取 STATUS/SUBGOAL，不会喂给模型。
- user message 中训练用的 `<image>` 占位符会被去掉；真实图片通过 engine 的
  structured image content 传入，和 qwen3vl_instruct_paradigm_a_runner.py 保持一致。
- anchor12_sanity 是额外的单例检查：跑最初触发“过早推进”的 route/anchor，
  看模型是否把 STATUS 保持为 initial。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[1]
_PROJECT_ROOT = _THIS_FILE.parents[2]
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qwen3vl_local.engine import LocalQwen3VLInstructEngine  # noqa: E402
from qwen3vl_local.prompt_pipeline import (  # noqa: E402
    DrivingMemory,
    build_system_prompt,
    build_user_prompt,
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

    调用前必须先 `engine.load()`：
    - LocalQwen3VLInstructEngine 默认懒加载，构造函数不会立刻加载权重；
    - PeftModel.from_pretrained 需要一个已经存在的 base model；
    - 如果忘记 load，这里会把 None 传给 PEFT，评估在开始前就崩。
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

    为什么训练和评估这里不同：
    - ms-swift 训练侧用 `<image>` 文本占位符匹配顶层 images 路径；
    - 本项目本地 engine 走 HuggingFace processor 的 structured message，
      图片以 {"type": "image", "image": PIL} 形式传入；
    - 因此 eval 需要还原出“纯 user prompt”，避免 `<image>` 文本被模型当普通文本读。
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

def predict_full(
    engine: LocalQwen3VLInstructEngine,
    sample: Dict,
    images_loader,
) -> Tuple[str, Dict[str, Optional[str]]]:
    """跑一次推理，同时返回 (raw_text, parsed_dict)。

    parsed_dict 至少含 status / subgoal / analysis 三个字段；缺失字段为 None。
    比原来的 predict_status 多返回 raw_text，是为了让 predictions jsonl 能保留
    模型完整输出（用户人工 review case 时定位错在哪个段，比只有 status 直观）。
    """
    pieces = reconstruct_prompts(sample)
    pil_images = images_loader(pieces["images"])
    raw_text, _ = engine.generate(
        system_prompt=pieces["system"],
        user_prompt=pieces["user"],
        images=pil_images,
        cache_dir=None,
    )
    parsed = parse_vlm_output(raw_text)
    return raw_text, parsed


def predict_status(
    engine: LocalQwen3VLInstructEngine,
    sample: Dict,
    images_loader,
) -> Optional[str]:
    """对一条样本跑推理，解析出 STATUS。失败返回 None。

    保留旧签名供 anchor12 sanity 等老调用方使用；新代码请用 predict_full
    拿到 raw_text + parsed_dict。
    """
    _, parsed = predict_full(engine, sample, images_loader)
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


def build_rgb_paths_from_route(
    route_dir: str,
    anchor: int,
    *,
    frame_step: int = 1,
    frame_count: int = 4,
) -> List[str]:
    """按 runner 规则构造 RGB clip 路径，并兼容 0000/0001 起始命名。

    anchor12 sanity 不来自 val.jsonl，所以需要现场构造图片路径。
    这里复制 build_sft_dataset_v1.py 的路径容错逻辑，保证 sanity 单例和验证集样本
    使用同一种 RGB 对齐规则。
    """

    route = pathlib.Path(route_dir)
    rgb_dir = route / "rgb"
    desc = [max(anchor - i * frame_step, 0) for i in range(frame_count)]
    ordered = list(reversed(desc))

    if not rgb_dir.exists():
        return [str(rgb_dir / f"{idx:04d}.jpg") for idx in ordered]

    rgb_files = sorted(rgb_dir.glob("*.jpg"))
    if not rgb_files:
        return [str(rgb_dir / f"{idx:04d}.jpg") for idx in ordered]

    paths: List[str] = []
    for idx in ordered:
        exact = rgb_dir / f"{idx:04d}.jpg"
        if exact.exists():
            paths.append(str(exact))
        elif 0 <= idx < len(rgb_files):
            paths.append(str(rgb_files[idx]))
        else:
            paths.append(str(exact))
    return paths


def build_anchor_sanity_sample(args: argparse.Namespace) -> Dict:
    """构造 anchor=12 fail case 的单样本，用同一套 predict_status 评估。

    这个样本不是训练/验证集的一部分，而是固定回归测试：
    原始 base Qwen 在这个 early anchor 上容易把 Accident 的 STATUS 从 initial
    提前推进到 hazard_detect。LoRA v1 的底线就是这里要回到 initial。
    """

    memory = DrivingMemory.from_scenario(args.anchor12_scenario)
    image_paths = build_rgb_paths_from_route(
        args.anchor12_route_dir,
        args.anchor12_anchor,
    )
    image_description = (
        f"The {len(image_paths)} images above are ordered oldest to newest; "
        "the last image is the current moment."
    )
    user_text = build_user_prompt(memory, image_description=image_description)
    # 构造成和 jsonl 样本一样的形态，后续统一走 reconstruct_prompts()。
    # 这样 sanity 单例和 val 样本不会因为 prompt 复原路径不同而产生额外变量。
    user_content = "".join("<image>" for _ in image_paths) + "\n" + user_text

    return {
        "scenario": args.anchor12_scenario,
        "run_id": pathlib.Path(args.anchor12_route_dir).name,
        "anchor": args.anchor12_anchor,
        "images": image_paths,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": (
                    f"ANALYSIS: Observations recorded.\n"
                    f"STATUS: {args.anchor12_expected_status}\n"
                    f"SUBGOAL: {next_event_in_seq(args.anchor12_scenario, args.anchor12_expected_status)}"
                ),
            },
        ],
        "is_transition_sample": False,
    }


def run_anchor12_sanity(
    engine: LocalQwen3VLInstructEngine,
    args: argparse.Namespace,
    images_loader,
) -> Dict:
    """跑原始 anchor=12 fail case，返回可写入 metrics 的结果。

    这里捕获异常而不是直接 raise：
    - 远程数据路径可能暂时没挂载；
    - 用户可能只想先跑 val 指标；
    - metrics 里记录 error 比整个评估中断更方便排查。
    """

    if args.skip_anchor12_sanity:
        return {"enabled": False, "passed": None}

    sample = build_anchor_sanity_sample(args)
    try:
        pred = predict_status(engine, sample, images_loader)
        expected = args.anchor12_expected_status
        return {
            "enabled": True,
            "passed": pred == expected,
            "pred_status": pred,
            "expected_status": expected,
            "scenario": sample["scenario"],
            "run_id": sample["run_id"],
            "anchor": sample["anchor"],
            "images": sample["images"],
            "error": None,
        }
    except Exception as e:
        return {
            "enabled": True,
            "passed": False,
            "pred_status": None,
            "expected_status": args.anchor12_expected_status,
            "scenario": sample["scenario"],
            "run_id": sample["run_id"],
            "anchor": sample["anchor"],
            "images": sample["images"],
            "error": str(e),
        }


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
    # ---- 关于 --cache-system-prompt ----
    # 这个开关控制 engine.generate 是否复用 system prompt 的 KV cache prefix。
    # 所有 val 样本共享同一段 system prompt（约 400 token），开启后每条样本
    # 跳过 system prompt 的 prefill 重算，整轮 eval 时间约能省 50%。
    #
    # 旧实现 action="store_true" + default=True 的死锁：
    #   * store_true 只在命令行写 --cache-system-prompt 时把值设 True，否则 default；
    #   * default 又写死 True；
    #   * 结果：不管命令行写不写这个 flag，最终值永远是 True，没有任何方式关闭。
    #
    # 修法用 argparse.BooleanOptionalAction（Python 3.9+）：
    #   * 自动配对生成 --cache-system-prompt / --no-cache-system-prompt 两个 flag；
    #   * default=True 表示默认开启，用户加 --no-cache-system-prompt 即可关闭；
    #   * 关闭后 engine 每条样本重新 prefill system prompt（debug KV cache 路径用）。
    parser.add_argument("--cache-system-prompt",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="复用 system prompt 的 KV prefix，节省推理时间。"
                             "--no-cache-system-prompt 可关闭。")
    parser.add_argument("--output-json", type=str,
                        default=str(_AUTOMOT_ROOT / "eval_json" / "sft_v1_metrics.json"))
    # ---- 逐条预测落盘（#5.5）----
    # predictions-jsonl 写全部样本（用于离线对比 / 人工 review）；
    # predictions-diff-jsonl 只写 pred ≠ gt 的样本，方便快速看错误 case。
    # 空字符串表示跳过；默认开启全量 predictions，diff 默认与之同目录。
    parser.add_argument("--predictions-jsonl", type=str,
                        default=str(_AUTOMOT_ROOT / "eval_json" / "sft_v1_predictions.jsonl"),
                        help="逐条 prediction 落盘路径；空字符串关闭。")
    parser.add_argument("--predictions-diff-jsonl", type=str,
                        default=str(_AUTOMOT_ROOT / "eval_json" / "sft_v1_predictions_diff.jsonl"),
                        help="只落 pred_status != gt_status 的样本；空字符串关闭。")
    parser.add_argument("--skip-anchor12-sanity", action="store_true",
                        help="跳过原始 anchor=12 fail case 单例检查。")
    parser.add_argument("--anchor12-route-dir", type=str,
                        default="/data/lead_data/data/Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46")
    parser.add_argument("--anchor12-scenario", type=str, default="Accident")
    parser.add_argument("--anchor12-anchor", type=int, default=12)
    parser.add_argument("--anchor12-expected-status", type=str, default="initial")
    args = parser.parse_args()

    samples = read_jsonl(args.val_jsonl)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"[eval] loaded {len(samples)} samples from {args.val_jsonl}")

    # 启动 engine + 可选挂 LoRA。
    #
    # 注意：engine 构造函数只保存配置，不加载权重。这里显式 engine.load()，
    # 一方面让 PEFT 有 base model 可挂，另一方面让后续 predict_status 不再重复触发加载。
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
    engine.load()
    if args.lora_dir:
        attach_lora_adapter(engine, args.lora_dir)

    # eval 时 jsonl 已经给了绝对路径，直接 PIL 打开就够。
    # 与 runner load_lead_rgb_clip 一样保留 RGB 原图，不做额外 resize/crop；
    # Qwen processor 会自己处理 dynamic resolution。
    from PIL import Image  # type: ignore

    def images_loader(paths: List[str]):
        # 每次打开后立刻 convert("RGB")，避免 PIL 延迟读取导致文件句柄在生成期间才报错。
        return [Image.open(p).convert("RGB") for p in paths]

    # 先跑固定 fail case，方便日志最前面就看见“这次 LoRA 是否解决了原问题”。
    # 如果该 route 不存在，可用 --skip-anchor12-sanity 跳过。
    anchor12_sanity = run_anchor12_sanity(engine, args, images_loader)
    if anchor12_sanity.get("enabled"):
        print(f"[anchor12] {anchor12_sanity}")

    # 计数器。
    # keep/advance 分开统计：只看总体 accuracy 会掩盖“模型永远保持”或“模型总是提前推进”
    # 这两种完全不同的失败模式。
    n_keep = n_keep_correct = n_early_adv = 0
    n_adv = n_adv_correct = 0
    per_scenario: Dict[str, Counter] = defaultdict(Counter)
    # 逐条 prediction 缓存：只在 --predictions-jsonl 或 --predictions-diff-jsonl 启用时累计，
    # 避免 max_samples=0 跑完整 val 时占额外内存（一行 dict 也就几百字节，800 条无压力）。
    predictions_records: List[Dict[str, Any]] = []
    capture_predictions = bool(args.predictions_jsonl) or bool(args.predictions_diff_jsonl)

    for i, sample in enumerate(samples):
        scenario = sample["scenario"]
        gt = extract_assistant_target(sample)
        gt_status = gt["status"]
        gt_subgoal = gt.get("subgoal") if isinstance(gt, dict) else None
        is_trans = sample.get("is_transition_sample", False)

        # 用 predict_full 拿 raw_text + parsed 一起返回；旧 predict_status 改为 wrapper。
        raw_text: Optional[str] = None
        pred: Optional[str] = None
        pred_subgoal: Optional[str] = None
        err: Optional[str] = None
        try:
            raw_text, parsed = predict_full(engine, sample, images_loader)
            pred = parsed.get("status")
            pred_subgoal = parsed.get("subgoal")
        except Exception as e:
            print(f"[err {i}] {e}")
            err = str(e)

        # 对 keep 样本，pred == next(GT) 就是最关心的 early advance。
        # 其它错误（输出 None、跳到更后状态、输出非法状态）不会计入 early_advance，
        # 但会让 keep_accuracy 下降。
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

        if capture_predictions:
            # error_kind 按"为什么 pred 错"分类，方便后续 diff 文件直接做 Counter 统计：
            #   ok                 — pred == gt
            #   early_advance      — pred == next(gt)（keep 样本最关心的错误）
            #   none               — 没有解析到 status（输出格式坏）
            #   inference_error    — generate 阶段抛异常
            #   other              — 其它（跳更后状态 / 非法 token / advance 样本未对齐 / ...）
            if pred is None and err is not None:
                error_kind = "inference_error"
            elif pred is None:
                error_kind = "none"
            elif pred == gt_status:
                error_kind = "ok"
            elif not is_trans and pred == next_gt:
                error_kind = "early_advance"
            else:
                error_kind = "other"
            predictions_records.append({
                "sample_idx": i,
                "scenario": scenario,
                "run_id": sample.get("run_id"),
                "anchor": sample.get("anchor"),
                "is_transition_sample": is_trans,
                "gt_status": gt_status,
                "gt_subgoal": gt_subgoal,
                "pred_status": pred,
                "pred_subgoal": pred_subgoal,
                "raw_text": raw_text,
                "error_kind": error_kind,
                "error": err,
            })

        if (i + 1) % 50 == 0:
            print(f"[eval] {i+1}/{len(samples)}  "
                  f"keep_acc={n_keep_correct/max(1,n_keep):.3f}  "
                  f"adv_acc={n_adv_correct/max(1,n_adv):.3f}  "
                  f"early_adv={n_early_adv/max(1,n_keep):.3f}")

    # metrics 保持 JSON 可序列化，方便后续画曲线或比较多个 checkpoint。
    metrics = {
        "n_total": len(samples),
        "n_keep": n_keep,
        "n_advance": n_adv,
        "keep_accuracy": n_keep_correct / max(1, n_keep),
        "advance_accuracy": n_adv_correct / max(1, n_adv),
        "early_advance_rate": n_early_adv / max(1, n_keep),
        "anchor12_sanity": anchor12_sanity,
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

    # ---- 逐条 prediction 落盘（#5.5）----
    # 一行一条 JSON：方便用 `jq .error_kind` / pandas 直接做透视；diff 只挑 error_kind != "ok"，
    # 让人工查错时不被正确样本淹没。
    if args.predictions_jsonl and predictions_records:
        pred_path = pathlib.Path(args.predictions_jsonl)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w", encoding="utf-8") as f:
            for row in predictions_records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[done] predictions written to {pred_path} (n={len(predictions_records)})")
    if args.predictions_diff_jsonl and predictions_records:
        diff_rows = [r for r in predictions_records if r["error_kind"] != "ok"]
        diff_path = pathlib.Path(args.predictions_diff_jsonl)
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diff_path, "w", encoding="utf-8") as f:
            for row in diff_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # 顺便打印 error_kind 分布，让 stdout 直接给出错误结构概览。
        kinds = Counter(r["error_kind"] for r in predictions_records)
        print(f"[done] diff written to {diff_path} (n={len(diff_rows)}); error_kind={dict(kinds)}")


if __name__ == "__main__":
    main()
