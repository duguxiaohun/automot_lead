"""Phase1/2 到 Phase3 的纯调度接口；不读取离线未来真值，不代替视觉 gate。

regular_gates 必须来自当前可见拓扑/导航或已发生的恢复状态。缺失与 NO 分开；
返回 recheck 表示尚有缺口，即使已有部分可提问 context，也不能宣布场景已完整恢复。
"""
from dataclasses import dataclass
from typing import Mapping

from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import (
    abnormal_events_from_answers, context_for_event, road_structure_from_answers)
from qwen3vl_local.sft_new_loop_phase3.transition_state import RecoveryState


@dataclass(frozen=True)
class DispatchPlan:
    road_structure: str
    context_ids: tuple[str, ...]
    recheck: tuple[str, ...]


@dataclass(frozen=True)
class CandidatePlan:
    """待模型核对的常规情境和已经由上游确定的异常分别保存。"""
    road_structure: str
    established_contexts: tuple[str, ...]
    probe_contexts: tuple[str, ...]
    recheck: tuple[str, ...]


def plan_candidate_requests(answers: Mapping[str, bool], *,
                            recovery: RecoveryState | None = None,
                            navigation_lane_transition: bool | None = None) -> CandidatePlan:
    """无需外部 RE 视觉分类器，直接用 Phase3 的 INVALID 行核对常规候选。

    R3 只决定可以补问 R-E3，不确认它存在。R5 同理。R-E2 可来自持续恢复状态
    或当前导航；未知导航也允许核对通用候选，绝不从最终目标 y 符号造变道侧。
    调用者必须保持候选身份，INVALID=NO 只表示未被驳回，不等于精确事件分类成绩。
    """
    if navigation_lane_transition is not None and type(navigation_lane_transition) is not bool:
        raise ValueError('navigation_lane_transition requires bool or None')
    base = plan_requests(answers, recovery=recovery)
    errors = tuple(key for key in base.recheck if not key.startswith('REGULAR_GATE:'))
    if errors:
        return CandidatePlan(base.road_structure, base.context_ids, (), errors)
    abnormal = tuple(ctx for ctx in base.context_ids if ctx != 'POST_BYPASS_RETURN')
    if abnormal:
        return CandidatePlan(base.road_structure, abnormal, (), ())
    probes = list(base.context_ids)
    if navigation_lane_transition is not False and 'POST_BYPASS_RETURN' not in probes:
        probes.append('POST_BYPASS_RETURN')
    if base.road_structure == 'R3':
        probes.append('RAMP_MERGE_EXIT')
    elif base.road_structure == 'R5':
        probes.append('UNSIGNALIZED_PRIORITY')
    return CandidatePlan(base.road_structure, (), tuple(probes), ())


def candidate_response(context_id: str, answers: Mapping[str, bool | None]) -> str:
    """核对完整输出；驳回普通候选可回常规流程，不能清除 RecoveryState。

    NOT_REJECTED_NO_ACTION 特意不叫 EVENT_PRESENT，避免把低可见度下的
    invalid=NO 或动作全 NO 冒充常规事件已成立/恢复已完成。
    """
    from qwen3vl_local.sft_new_loop_phase3.context_taxonomy import CONTEXT_BY_ID
    expected = set(CONTEXT_BY_ID[context_id].action_keys) | {'INVALID_ACTION_CONTEXT'}
    if set(answers) != expected or any(type(v) is not bool for v in answers.values()):
        return 'RECHECK_FORMAT'
    if answers['INVALID_ACTION_CONTEXT']:
        return ('RECHECK_FORMAT' if any(answers[k] for k in expected - {'INVALID_ACTION_CONTEXT'})
                else 'CANDIDATE_REJECTED')
    status = action_response_status(answers)
    return {'ACTION_PRESENT': 'NOT_REJECTED_ACTION',
            'NO_LISTED_ACTION': 'NOT_REJECTED_NO_ACTION'}.get(status, status)


def plan_requests(answers: Mapping[str, bool], *,
                  regular_gates: Mapping[str, bool] | None = None,
                  recovery: RecoveryState | None = None) -> DispatchPlan:
    """输入严格 parser 已转换的 bool；并发异常逐个提问，不按主事件优先级吞掉。

    不接受 raw taxonomy 的历史 U7 名字；TRAFFIC_LIGHT_ABNORMAL=YES 本身才是证据。
    这里消费 gate 而不预测 gate，单凭 R3 或 R5 不产生相应常规事件。
    """
    if any(type(value) is not bool for value in answers.values()):
        raise ValueError("answers must contain parsed bool values; missing questions must be omitted")
    gates = dict(regular_gates or {})
    if any(key not in ("R-E2", "R-E3", "R-E5") or type(value) is not bool
           for key, value in gates.items()):
        raise ValueError("regular_gates require explicit bool R-E2/R-E3/R-E5 observations")
    rs = road_structure_from_answers(answers)
    if rs == "UNKNOWN":
        return DispatchPlan(rs, (), ("ROAD_STRUCTURE",))
    events = abnormal_events_from_answers(answers)
    contexts, recheck = [], []
    if answers.get("INVALID_EVENT_CONTEXT") is True or (
            answers.get("INVALID_EVENT_CONTEXT") is not False
            and any(answers.get(key) is True for key in ("UE1", "UE3", "UE5", "UE6"))):
        recheck.append("PHASE2_EVENT_CONTEXT")
    for event in events:
        context = context_for_event(event)
        if rs in context.allowed_rs:
            contexts.append(context.context_id)
        else:
            recheck.append(f"RS_EVENT_CONFLICT:{event}")
    # 先处理仍在发生的异常；保留恢复状态，不由动作全 NO 清除。
    if contexts or recheck:
        return DispatchPlan(rs, tuple(contexts), tuple(recheck))
    if recovery is not None and recovery.candidate():
        contexts.append(recovery.candidate())
    for event in ("R-E2", "R-E3", "R-E5"):
        context = context_for_event(event)
        if rs not in context.allowed_rs:
            if gates.get(event) is True:
                recheck.append(f"RS_EVENT_CONFLICT:{event}")
            continue
        if gates.get(event) is True:
            if context.context_id not in contexts:
                contexts.append(context.context_id)
        elif event not in gates and context.context_id not in contexts:
            recheck.append(f"REGULAR_GATE:{event}")
    return DispatchPlan(rs, tuple(contexts), tuple(recheck))


def action_response_status(answers: Mapping[str, bool | None]) -> str:
    """INVALID/格式错应回查上游，动作全 NO 只表示本次无需所问动作。

    不修改 RecoveryState，也不把本函数当低层驾驶控制器。未问的横向行不补 NO。
    """
    if not answers or any(type(v) is not bool for v in answers.values()):
        return "RECHECK_FORMAT"
    if "INVALID_ACTION_CONTEXT" not in answers:
        return "RECHECK_FORMAT"
    if answers["INVALID_ACTION_CONTEXT"]:
        return "RECHECK_CONTEXT"
    speed = sum(answers.get(k, False) for k in ("STOP", "RESUME", "DECELERATE"))
    lateral = sum(answers.get(k, False) for k in ("LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT"))
    if speed > 1 or lateral > 1:
        return "RECHECK_ACTION_CONFLICT"
    return "ACTION_PRESENT" if any(v for k, v in answers.items() if k != "INVALID_ACTION_CONTEXT") else "NO_LISTED_ACTION"
