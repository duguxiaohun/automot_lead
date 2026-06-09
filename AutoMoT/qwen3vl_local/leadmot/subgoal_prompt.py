"""LeadMoT subgoal 模式专用 prompt + keyframe 图像加载 helper。

只在 LeadMoT decoder_config.use_subgoal=True 时使用。

设计目标：在不破坏 v1/v2 prefix 的情况下，让 frozen Qwen3-VL 看到额外信号：

- 历史 + 当前的 stitched RGB（与原 runner 完全一致，最后一帧是 current）
- 一张 SUBGOAL 关键帧 stitched RGB，作为 "what the scene must look like when the
  next event is reached" 的视觉真值
- 显式的 STATUS / SUBGOAL 真值文本块（含语义解释）
- 仍然附带原导航文本 (current target / next target / final destination / speed) —
  decoder 还要按这套 status token 对齐 tp/ntp/final_goal 语义，不能拿掉

prompt 大体复用 goalgen.prompt 的"teacher-forced"措辞，但 system / user 都做了
LeadMoT 专属定制：

- 不要求模型输出 STATUS/SUBGOAL；模型不消费文字输出，下游 decoder 只读 prefill
  past_key_values。
- system 段强调 "downstream is a planning decoder, not a text generator"，与
  goalgen 的 "downstream is a latent image generator" 区分开。
- user 段同时塞入 [GROUND_TRUTH_STATE] 块（status/subgoal + meaning）与
  导航文本（current/next target + final destination + speed），让 KV cache
  同时携带语义状态与几何状态。

注意：所有进模型字符串全英文（项目硬约定，与 prompt_pipeline 同源）；
中文仅出现在 docstring / 注释里。
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Tuple

from PIL import Image

from ..prompt_pipeline import (
    EVENT_DESCRIPTIONS,
    SCENARIO_LABELS,
    DrivingMemory,
    get_full_sequence,
)


# ---------------------------------------------------------------------------
# system / user prompt
# ---------------------------------------------------------------------------

# 与 mot_lead_offline_runner._LEADMOT_QWEN_SYSTEM_PROMPT 完全同构，
# 仅以最小增量加入 SUBGOAL 相关说明：
#   (a) 多了一张 SUBGOAL 参考图（最后一张），它表示"下一子任务达成时的场景"，
#       不是当前观测；
#   (b) STATUS（当前事件）/ SUBGOAL（下一事件）会作为 ground-truth 文本一并给出。
# 其余措辞、句式、坐标系说明、下游 decoder 角色都和非 subgoal 版保持一致。
_LEADMOT_SUBGOAL_SYSTEM_PROMPT = (
    "You are an autonomous driving model controlling an ego vehicle in real-world "
    "urban traffic. Your job is to perceive the surrounding scene from multi-view "
    "RGB and help plan a safe short-horizon trajectory under traffic rules. "
    "You will be given a sequence of stitched multi-view RGB images plus one "
    "SUBGOAL reference image, and a navigation hint with target points, a final "
    "destination, and current speed. All coordinates are in the ego frame "
    "(x_forward, y_left) measured in meters. The last image is the SUBGOAL "
    "reference: it shows what the scene should look like once the next sub-task "
    "(the SUBGOAL event) is reached, not a current observation. The ground-truth "
    "current STATUS and next SUBGOAL events are also provided as text. "
    "A downstream planning decoder consumes your hidden state to predict the "
    "ego-vehicle trajectory."
)


def build_leadmot_subgoal_system_prompt() -> str:
    """返回 LeadMoT subgoal 模式固定的 system prompt（全英文）。"""

    return _LEADMOT_SUBGOAL_SYSTEM_PROMPT


def _format_ground_truth_block(memory: DrivingMemory) -> str:
    """构造 [GROUND_TRUTH_STATE] ... [/GROUND_TRUTH_STATE] 块。

    格式约定与 goalgen.prompt._format_memory_block 保持一致：
    - SCENARIO 行 # scenario_label
    - EVENT_SEQUENCE 全量展开成 "- event: description"
    - STATUS / SUBGOAL 单独点名 + meaning，强化局部 attention 锚点
    """

    seq_desc_lines = [
        f"- {event}: {EVENT_DESCRIPTIONS.get(event, event)}"
        for event in memory.event_sequence
    ]
    seq_desc_str = "\n".join(seq_desc_lines)
    status_desc = EVENT_DESCRIPTIONS.get(memory.status, memory.status)
    subgoal_desc = EVENT_DESCRIPTIONS.get(memory.subgoal, memory.subgoal)
    return (
        "[GROUND_TRUTH_STATE]\n"
        f"SCENARIO: {memory.scenario}  # {memory.scenario_label}\n"
        "EVENT_SEQUENCE (each step explained in order):\n"
        f"{seq_desc_str}\n"
        f"STATUS (ground truth, current event): {memory.status}\n"
        f"  meaning: {status_desc}\n"
        f"SUBGOAL (ground truth, the next event to reach): {memory.subgoal}\n"
        f"  meaning: {subgoal_desc}\n"
        "  visual_reference: the last image above shows the scene at SUBGOAL.\n"
        "[/GROUND_TRUTH_STATE]"
    )


def describe_image_inputs(num_history_images: int) -> str:
    """统一描述 prefix 图像顺序：N 张历史/当前 + 1 张 subgoal 参考。

    LEAD 离线/在线两条路径都保证 num_history_images >= 1（runner.run_step 至少
    构造 1 帧当前观测），因此不再为 num_history_images <= 0 写兜底文案；真出现
    时直接 raise，把"prefix 没有当前观测"这种不变式破坏暴露出来，避免 prompt
    文本与实际 images 列表对不上。
    """

    if num_history_images < 1:
        raise ValueError(
            f"describe_image_inputs requires num_history_images >= 1, "
            f"got {num_history_images}"
        )
    if num_history_images == 1:
        return (
            "Two images are provided above. The first image is the current "
            "observation; the second image is the SUBGOAL reference."
        )
    return (
        f"{num_history_images + 1} images are provided above. The first "
        f"{num_history_images} images are recent observations ordered oldest to "
        "newest; the last of these is the current moment. The very last image "
        "is the SUBGOAL reference (what the scene should look like once the "
        "SUBGOAL event is reached)."
    )


def build_leadmot_subgoal_user_prompt(
    navigation_prompt: str,
    memory: DrivingMemory,
    num_history_images: int,
) -> str:
    """构造 LeadMoT subgoal 模式 user prompt。

    参数:
    - navigation_prompt：runner.build_cleaned_prompt_and_modes() 返回的导航文本，
      含 current/next target、final destination、speed、planning horizon 说明。
      原样塞进来，保证 decoder 拿到的 KV 仍然能对齐 tp/ntp/final_goal 语义。
    - memory：DrivingMemory（含 scenario / event_sequence / status / subgoal）。
    - num_history_images：除 subgoal 参考之外的历史/当前 RGB 数量。
    """

    image_caption = describe_image_inputs(num_history_images)
    ground_truth = _format_ground_truth_block(memory)
    # 末尾不再附"compress hidden state / reply ok"指令：
    # - decoder 直接吃 prefill KV，不消费任何文字输出；
    # - 与 non-subgoal 路径 (build_cleaned_prompt_and_modes) 的收尾保持完全一致。
    return (
        f"{image_caption}\n\n"
        f"{ground_truth}\n\n"
        f"{navigation_prompt}"
    )


# ---------------------------------------------------------------------------
# DrivingMemory 构造
# ---------------------------------------------------------------------------

def build_memory_from_sample(sample: Dict[str, Any]) -> DrivingMemory:
    """从 build_dataset 写入的字段恢复 DrivingMemory。

    依赖以下字段（由 build_dataset.py --with-subgoal-fields 写入）：
    - scenario：场景 token，如 "Accident"
    - status / subgoal：事件 token，如 "initial" / "hazard_detect"

    缺字段时按既有 DrivingMemory.from_dict 行为抛 KeyError，由调用方决定
    skip-sample 还是直接报错。
    """

    scenario = str(sample["scenario"])
    seq = get_full_sequence(scenario)
    scenario_label = SCENARIO_LABELS.get(scenario, scenario)
    return DrivingMemory(
        scenario=scenario,
        scenario_label=scenario_label,
        event_sequence=seq,
        status=str(sample["status"]),
        subgoal=str(sample["subgoal"]),
        completed_events=[],
    )


# ---------------------------------------------------------------------------
# subgoal 关键帧 RGB 加载
# ---------------------------------------------------------------------------

def _try_load_with_cv2(path: pathlib.Path) -> Image.Image:
    """走 cv2 解码再转 RGB，复用 LEAD/runner 的解码链路保持像素一致。"""

    import cv2  # 延迟 import：subgoal 关闭时 train.py 不应该被强制依赖 cv2。

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2 read failed: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, mode="RGB")


def load_subgoal_rgb(sample: Dict[str, Any]) -> Image.Image:
    """读取 sample 指定的 subgoal stitched RGB（三视角 1152x384）。

    优先按 ``subgoal_rgb_path`` 直接读；该字段由 build_dataset.py 在反查时已经
    校验过文件存在，因此正常路径下不再额外 stat。

    fallback：仅给 ``route_dir`` + ``subgoal_frame`` 时按 LEAD ``rgb/{:04d}.jpg``
    命名拼路径再读；这是 probe / 手工构造 sample 时的便利路径。
    """

    path_str = sample.get("subgoal_rgb_path")
    if path_str:
        path = pathlib.Path(path_str)
        if path.exists():
            return _try_load_with_cv2(path)
        # build_dataset 在 NFS 上写入的 path 可能已失效；fallback 到 route_dir + frame。

    route_dir = sample.get("route_dir")
    frame = sample.get("subgoal_frame")
    if route_dir is None or frame is None:
        raise FileNotFoundError(
            "sample lacks subgoal_rgb_path / (route_dir + subgoal_frame); "
            "cannot load SUBGOAL RGB. Did build_dataset run with --with-subgoal-fields?"
        )
    fallback = pathlib.Path(route_dir) / "rgb" / f"{int(frame):04d}.jpg"
    if not fallback.exists():
        raise FileNotFoundError(
            f"subgoal RGB not found: subgoal_rgb_path={path_str}, "
            f"fallback={fallback}"
        )
    return _try_load_with_cv2(fallback)


__all__ = [
    "build_leadmot_subgoal_system_prompt",
    "build_leadmot_subgoal_user_prompt",
    "describe_image_inputs",
    "build_memory_from_sample",
    "load_subgoal_rgb",
]
