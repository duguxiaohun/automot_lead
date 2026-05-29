"""Standalone Qwen3-VL-4B-Instruct paradigm-A runner.

这个脚本是“范式 A：让 VLM 直接输出 ANALYSIS/STATUS/SUBGOAL 文本”的入口。
它只加载本地 ``AutoMoT/checkpoints/Qwen3-VL-4B-Instruct``，不 import
``vlm_paradigm_a_runner.py``，也不调用 AutoMoT 的 InterleaveInferencer。

推理主体拆到 ``AutoMoT/qwen3vl_local``，便于本地魔改和逐段观察：

- ``prompt_pipeline.py``：场景状态机、memory、prompt、输出解析。
- ``image_io.py``：合成图和 LEAD route RGB 读取。
- ``engine.py``：本地模型加载、chat template、prefill、decode、KV cache。
- ``cache_utils.py``：KV cache 摘要和可选落盘。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# 路径约定：
# 本文件在 AutoMoT/leaderboard/team_code/ 下。
# _AUTOMOT_ROOT 指 AutoMoT/，用于找 checkpoint、eval_json 和 qwen3vl_local。
# _PROJECT_ROOT 指 automot_lead/，用于从不同工作目录启动脚本时也能稳定 import。
_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_PROJECT_ROOT = _THIS_FILE.parents[3]
_CHECKPOINT_DIR = _AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B-Instruct"
_DEFAULT_OUTPUT_ROOT = _AUTOMOT_ROOT / "eval_json" / "qwen3vl_instruct_paradigm_a"

# 单文件入口常从不同目录运行；显式补 sys.path 可避免相对 import 因 cwd 改变失效。
for _p in (str(_AUTOMOT_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# HuggingFace 离线开关。engine.load() 还会传 local_files_only=True，双保险。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from qwen3vl_local.engine import LocalQwen3VLInstructEngine, dump_trace  # noqa: E402
from qwen3vl_local.image_io import (  # noqa: E402
    auto_detect_scenario_from_route,
    build_synthetic_raw_and_model_input,
    load_lead_rgb_clip,
)
from qwen3vl_local.prompt_pipeline import (  # noqa: E402
    DrivingMemory,
    build_system_prompt,
    build_user_prompt,
    parse_vlm_output,
    update_memory,
)


@dataclass
class ParadigmAStepRecord:
    """一次范式 A 推理的主记录，对应最终写出的 step.json。

    它把输入、输出、解析结果和 memory 更新前后状态放在同一个结构里，方便以后
    对照模型原文、状态机解析、图片输入和 KV trace。
    """

    step_idx: int
    timestamp: str
    scenario: str
    checkpoint_dir: str
    num_images: int
    num_raw_images: int
    memory_before: dict
    system_prompt: str
    user_prompt: str
    raw_vlm_text: str
    parsed: Dict[str, Optional[str]]
    memory_after: dict
    save_dir: Optional[str] = None
    image_files: List[str] = field(default_factory=list)
    raw_image_files: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转成 JSON 可写结构。"""

        return {
            "step_idx": self.step_idx,
            "timestamp": self.timestamp,
            "scenario": self.scenario,
            "checkpoint_dir": self.checkpoint_dir,
            "num_images": self.num_images,
            "num_raw_images": self.num_raw_images,
            "memory_before": self.memory_before,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "raw_vlm_text": self.raw_vlm_text,
            "parsed": self.parsed,
            "memory_after": self.memory_after,
            "save_dir": self.save_dir,
            "image_files": self.image_files,
            "raw_image_files": self.raw_image_files,
        }


def _save_pil_list(imgs: List[Any], out_dir: pathlib.Path, prefix: str) -> List[str]:
    """保存一组 PIL 图片，并返回相对 step 目录的路径。

    raw_images 和 model_input_images 可能尺寸不同，所以调用方会用不同 prefix
    分别保存。非 PIL 对象会被跳过，避免调试时传入占位对象导致直接崩溃。
    """

    saved: List[str] = []
    for i, img in enumerate(imgs):
        fname = f"{prefix}{i:03d}.png"
        fpath = out_dir / fname
        if hasattr(img, "save"):
            img.save(str(fpath))
            saved.append(str(fpath.relative_to(out_dir.parent)))
    return saved


def dump_record(
    record: ParadigmAStepRecord,
    trace_dict: dict,
    images: List[Any],
    target_dir: pathlib.Path,
    raw_images: Optional[List[Any]] = None,
) -> None:
    """落盘一次完整推理记录。

    输出目录分三块：
    - inputs：prompt、memory_before、送入模型的图片和原始图片；
    - outputs：模型原文、解析结果、memory_after；
    - trace：chat template 展开文本、输入张量摘要、KV cache 摘要、decode token。
    """

    target_dir = pathlib.Path(target_dir)
    inputs_dir = target_dir / "inputs"
    outputs_dir = target_dir / "outputs"
    trace_dir = target_dir / "trace"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    # 保存 prompt 和初始 memory，便于复现模型看到的文字上下文。
    (inputs_dir / "system_prompt.txt").write_text(record.system_prompt, encoding="utf-8")
    (inputs_dir / "user_prompt.txt").write_text(record.user_prompt, encoding="utf-8")
    (inputs_dir / "memory_before.json").write_text(
        json.dumps(record.memory_before, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 保存视觉输入。image_* 是真正送模型的版本；raw_image_* 是读取/合成的原图版本。
    record.image_files = _save_pil_list(images, inputs_dir, "image_")
    if raw_images:
        record.raw_image_files = _save_pil_list(raw_images, inputs_dir, "raw_image_")
    record.save_dir = str(target_dir)

    # 保存模型输出和状态机解析结果。
    (outputs_dir / "raw_vlm_text.txt").write_text(record.raw_vlm_text, encoding="utf-8")
    (outputs_dir / "parsed.json").write_text(
        json.dumps(record.parsed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outputs_dir / "memory_after.json").write_text(
        json.dumps(record.memory_after, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # trace_dict 来自 engine.GenerationTrace，记录底层推理细节。
    (trace_dir / "generation_trace.json").write_text(
        json.dumps(trace_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target_dir / "step.json").write_text(
        json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare_images(args: argparse.Namespace) -> Tuple[str, List[Any], List[Any]]:
    """准备一次推理使用的 RGB 图片。

    route_dir 非空时读取真实 LEAD route 下的 rgb/*.jpg；route_dir 为空字符串时
    生成合成三色图，只用于验证链路。返回的 raw_images 用于人工检查，
    model_input_images 才是真正传给 Qwen processor 的图片列表。
    """

    scenario = args.scenario
    if args.route_dir:
        raw_images, model_input_images = load_lead_rgb_clip(
            route_dir=args.route_dir,
            anchor=args.anchor,
            rgb_frame_step=args.rgb_frame_step,
            rgb_frame_count=args.num_frames,
        )
        # LEAD 数据目录通常形如 data/<Scenario>/<route_name>/，可以从路径里推场景。
        auto_scenario = auto_detect_scenario_from_route(args.route_dir)
        if auto_scenario and auto_scenario != scenario:
            print(f"[scenario] auto-detected '{auto_scenario}', overriding '{scenario}'")
            scenario = auto_scenario
    else:
        raw_images, model_input_images = build_synthetic_raw_and_model_input(
            num_frames=args.num_frames,
            raw_size=(1920, 1080),
            model_input_size=(1152, 384),
        )
    return scenario, raw_images, model_input_images


def describe_image_inputs(num_images: int) -> str:
    """生成 user prompt 里的图片说明。

    这句话只是自然语言说明，不是视觉 token。真实图片 token 由
    engine.build_messages() 的 structured image content 生成。
    """

    if num_images <= 0:
        return "No visual observations are provided for this step."
    if num_images == 1:
        return "The image above is the current visual observation."
    return (
        f"The {num_images} images above are ordered oldest to newest; "
        "the last image is the current moment."
    )


def run_once(args: argparse.Namespace) -> None:
    """执行一次完整的 standalone Qwen3-VL-Instruct 范式 A 推理。"""

    checkpoint_dir = pathlib.Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Local Qwen3-VL-4B-Instruct checkpoint not found: {checkpoint_dir}"
        )

    save_root = pathlib.Path(args.save_root).resolve() if args.save_root else _DEFAULT_OUTPUT_ROOT
    step_dir = save_root / "step_000000"
    cache_dir = step_dir / "kv_cache"

    # 1) 准备视觉输入和初始 memory。
    scenario, raw_images, model_input_images = prepare_images(args)
    memory = DrivingMemory.from_scenario(scenario)

    # 2) 构造 prompt。system 规定角色和输出格式；user 注入 memory 和图片顺序说明。
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        memory,
        image_description=describe_image_inputs(len(model_input_images)),
    )

    # 3) 调用本地 Qwen 引擎。engine 内部显式展开 chat_template -> prefill -> decode。
    engine = LocalQwen3VLInstructEngine(
        checkpoint_dir=checkpoint_dir,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        save_cache=args.save_cache,
    )
    raw_text, trace = engine.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=model_input_images,
        cache_dir=cache_dir,
    )

    # 4) 解析模型自由文本，并用状态机约束更新 memory。
    parsed = parse_vlm_output(raw_text)
    new_memory = update_memory(memory, parsed)

    record = ParadigmAStepRecord(
        step_idx=0,
        timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        scenario=scenario,
        checkpoint_dir=str(checkpoint_dir),
        num_images=len(model_input_images),
        num_raw_images=len(raw_images),
        memory_before=memory.to_dict(),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_vlm_text=raw_text,
        parsed=parsed,
        memory_after=new_memory.to_dict(),
    )

    # 5) 保存完整实验材料。即便 raw_text 为空，也可以通过 trace 回看 decode 过程。
    dump_record(
        record=record,
        trace_dict=trace.to_dict(),
        images=model_input_images,
        target_dir=step_dir,
        raw_images=raw_images,
    )
    dump_trace(trace, step_dir / "trace")

    print("-" * 72)
    print(f"[checkpoint] {checkpoint_dir}")
    print(f"[output] {step_dir}")
    print(f"[raw text len] {len(raw_text)}")
    print(raw_text if raw_text else "<EMPTY STRING>")
    print(f"[parsed] {parsed}")
    print(f"[memory] {new_memory.to_dict()}")
    print(f"[cache summary] {step_dir / 'trace' / 'generation_trace.json'}")
    if args.save_cache:
        print(f"[cache tensors] {cache_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    """定义命令行参数。

    默认值尽量对齐旧的范式 A 对照脚本，但本脚本固定只跑本地
    Qwen3-VL-4B-Instruct checkpoint。
    """

    parser = argparse.ArgumentParser(description="Local Qwen3-VL-4B-Instruct paradigm-A runner")
    parser.add_argument("--checkpoint-dir", type=str, default=str(_CHECKPOINT_DIR),
                        help="本地 checkpoint 目录。默认：AutoMoT/checkpoints/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default="auto",
                        help="auto、cuda、cuda:0 或 cpu")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32", "auto"],
                        default="bfloat16")
    parser.add_argument("--max-gen-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true",
                        help="开启采样生成；默认关闭以便复现实验。")
    parser.add_argument("--save-cache", action="store_true",
                        help="用 torch.save 保存 prefill/final past_key_values 张量，文件可能很大。")
    parser.add_argument("--save-root", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="MergerIntoSlowTraffic")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument(
        "--route-dir",
        type=str,
        default="/data/lead_data/data/Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46",
        help=(
            "真实 LEAD route 目录，目录下需要有 rgb/*.jpg。"
            "传空字符串则使用合成图，仅用于验证链路，语义不可依赖。"
        ),
    )
    parser.add_argument("--anchor", type=int, default=12,
                        help="以哪一帧作为当前帧，从它往前采样 RGB clip。")
    parser.add_argument("--rgb-frame-step", type=int, default=1,
                        help="RGB clip 的帧间隔；LEAD 落盘帧本身约为 0.25 秒。")
    return parser


def main() -> None:
    """CLI 入口。"""

    args = build_arg_parser().parse_args()
    run_once(args)


if __name__ == "__main__":
    main()
