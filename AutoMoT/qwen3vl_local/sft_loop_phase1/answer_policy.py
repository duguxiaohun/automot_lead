"""经 RGB 复核的第一轮四问答案策略。

RS/EVENT 只是不直接喂给 Qwen 的审计分层键，不能机械等同于图像事实。尤其 ``R3``
既可能是受控快速路，也可能是短暂的普通乡间道路候选；本模块只编码已经用逐帧 RGB
和原始标签复核过的组合结论。个别 topology 混合组合以显式子组覆盖，绝不静默按 Town
或场景名猜测。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple


ANSWER_KEYS = ("HIGHWAY", "OBSTACLE", "VULNERABLE", "TRAFFIC_LIGHT_ABNORMAL")

# 这些 EVENT 都表示会占用/侵入/急剧压缩 ego 可行驶空间的动态或静态交通对象。
# U-E4 不并入这里：行人/自行车由独立 VULNERABLE 问题学习，避免两个问题退化成同义重复。
OBSTACLE_EVENTS = frozenset({"U-E1", "U-E2", "U-E3", "U-E5", "U-E6", "U-E8"})
VULNERABLE_EVENTS = frozenset({"U-E4"})


# 2026-08-09 的全帧 RGB 审计确认的高速组合。这里刻意不用 ``rs == R3``：
# InterurbanActorFlow/R3 是乡间地面道路，不能回答 HIGHWAY；而 EnterActorFlow*/R1
# 是仍由匝道/受控主线拓扑支配的画面，必须回答 HIGHWAY。
_HIGHWAY_RS_BY_SCENARIO: Mapping[str, frozenset[str]] = {
    "EnterActorFlow": frozenset({"R1", "R3"}),
    "EnterActorFlowV2": frozenset({"R1", "R3"}),
    "HardBreakRoute": frozenset({"R3"}),
    "HighwayCutIn": frozenset({"R3"}),
    "HighwayExit": frozenset({"R3"}),
    "MergerIntoSlowTraffic": frozenset({"R3"}),
    "MergerIntoSlowTrafficV2": frozenset({"R3"}),
    "StaticCutIn": frozenset({"R3"}),
}


# 视觉子组必须由 RGB/topology 审核步骤显式写入；这些 override 不可以由 town、RS 或
# scenario 名单独触发。ParkedObstacle/Town12 的审计中同时存在普通地面道路和受控快速路，
# 因此保留一个可追溯的子组，而不是把整个 Town12 或整个场景翻成 HIGHWAY=YES。
TOPOLOGY_SUBGROUP_OVERRIDES: Tuple[Mapping[str, Any], ...] = (
    {
        "id": "parked_obstacle_town12_limited_access_fast_road",
        "scenario": "ParkedObstacle",
        "towns": ("Town12",),
        "rs_values": ("R1",),
        "topology_subgroup": "limited_access_fast_road",
        "answers_patch": {"HIGHWAY": True},
        "evidence_contract": (
            "Only assign this subgroup after RGB shows the current ego path is still governed by "
            "limited-access fast-road structure: separated carriageway plus ramp/merge/exit or "
            "other access-control evidence. A wide straight road, barrier, or Town12 alone is insufficient."
        ),
        "audit_evidence": (
            "PHASE1_FOUR_QUESTION_RGB_AUDIT_20260809.md §4; "
            "full_route_rgb_label_review_20260809/ParkedObstacle/Town12/"
            "{Town12_Rep0_1006_0_route0_01_10_14_44_57,"
            "Town12_Rep0_2967_1_route0_01_10_20_56_17,"
            "Town12_Rep0_962_1_route0_01_09_14_36_56}"
        ),
    },
)


def topology_subgroup_overrides() -> List[Dict[str, Any]]:
    """返回可序列化的显式 topology 子组目录，供答案表和训练构建器共同使用。"""

    return [
        {
            **dict(item),
            "towns": list(item.get("towns", ())),
            "rs_values": list(item.get("rs_values", ())),
            "answers_patch": dict(item.get("answers_patch", {})),
        }
        for item in TOPOLOGY_SUBGROUP_OVERRIDES
    ]


def _highway_default(scenario: str, rs: str) -> bool:
    """只使用实图确认过的 scenario/RS 对，不以 RS 标签自身作推断。"""

    return str(rs) in _HIGHWAY_RS_BY_SCENARIO.get(str(scenario), frozenset())


def _apply_topology_override(
    answers: Dict[str, bool],
    *,
    scenario: str,
    rs: str,
    town: Optional[str],
    topology_subgroup: Optional[str],
) -> Dict[str, bool]:
    """应用由上游视觉审核明确赋值的子组覆盖。

    ``town`` 缺失时不匹配 Town 专属覆盖，避免训练构建器在没有 route 级证据时扩大标签。
    """

    if not topology_subgroup:
        return answers
    out = dict(answers)
    for item in TOPOLOGY_SUBGROUP_OVERRIDES:
        if str(item["scenario"]) != str(scenario):
            continue
        if str(item["topology_subgroup"]) != str(topology_subgroup):
            continue
        if town is None or str(town) not in set(item.get("towns", ())):
            continue
        rs_values = set(item.get("rs_values", ()))
        if rs_values and str(rs) not in rs_values:
            continue
        out.update({key: bool(value) for key, value in dict(item["answers_patch"]).items()})
    return out


def resolve_group_answers(
    scenario: str,
    rs: str,
    event: str,
    *,
    town: Optional[str] = None,
    topology_subgroup: Optional[str] = None,
) -> Dict[str, bool]:
    """返回一个已定义组合的四项统一 YES/NO 答案。

    - HIGHWAY 只由实图复核过的 scenario/RS 对给出；R3 不是充分条件。
    - U-E1/2/3/5/6/8 是可影响 ego 的动态或静态交通障碍；用户指定 U-E2 不因边界帧
      遮挡而降为 NO。
    - U-E4 独立回答弱势参与者。
    - U-E7 既可能是灯故障，也可能只是无灯路权不可靠；只有已经逐帧 RGB 审计为信号灯
      缺陷的 CrossJunctionDefectTrafficLight 才回答交通灯异常。
    """

    answers = {
        "HIGHWAY": _highway_default(str(scenario), str(rs)),
        "OBSTACLE": str(event) in OBSTACLE_EVENTS,
        "VULNERABLE": str(event) in VULNERABLE_EVENTS,
        "TRAFFIC_LIGHT_ABNORMAL": str(scenario) == "CrossJunctionDefectTrafficLight" and str(event) == "U-E7",
    }
    return _apply_topology_override(
        answers,
        scenario=str(scenario),
        rs=str(rs),
        town=town,
        topology_subgroup=topology_subgroup,
    )


def answer_rationale(scenario: str, rs: str, event: str) -> Dict[str, str]:
    """为人工表格给出每项 YES/NO 的可审计中文理由。"""

    answers = resolve_group_answers(scenario, rs, event)
    highway_reason = (
        "该 scenario/RS 组合已由跨 Town 全帧 RGB 确认为受控主线、匝道、合流或驶出拓扑"
        if answers["HIGHWAY"]
        else "RGB 审计未确认当前组合存在受控通行拓扑；直、宽、空、快、护栏或 R3 标签本身都不足以回答高速"
    )
    return {
        "HIGHWAY": highway_reason,
        "OBSTACLE": f"{event}：该组合定义为可影响 ego 的动态/静态交通障碍" if answers["OBSTACLE"] else f"{event}：没有该组合级的可交互障碍语义",
        "VULNERABLE": "U-E4：行人/骑行者等弱势参与者横穿或进入冲突区" if answers["VULNERABLE"] else f"{event}：没有弱势参与者冲突语义",
        "TRAFFIC_LIGHT_ABNORMAL": (
            "CrossJunctionDefectTrafficLight/U-E7：RGB 审计后的受控路口信号失效/矛盾语义"
            if answers["TRAFFIC_LIGHT_ABNORMAL"]
            else "正常灯态、他车违规或一般路权问题都不等于交通灯异常"
        ),
    }
