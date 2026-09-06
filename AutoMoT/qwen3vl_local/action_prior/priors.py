"""只用 RGB 问答生成先验；不接受未来轨迹、事件 GT 或 Phase3 输入。"""

from __future__ import annotations
from collections import defaultdict
from qwen3vl_local.sft_new_loop_phase1 import prompts as p1
from qwen3vl_local.sft_new_loop_phase2 import prompts as default_event_prompts

PROTOCOL_VERSION = "rs_event_recheck_modes_v2"


def strict_answers(text, keys):
    """逐行/顺序/数量严格解析，禁止把格式失败当 NO。"""
    lines = str(text).strip().splitlines()
    if len(lines) != len(keys):
        return {k: None for k in keys}
    result = {}
    for line, key in zip(lines, keys):
        if line.strip() not in (f"{key}: YES", f"{key}: NO"):
            return {k: None for k in keys}
        result[key] = line.strip().split(": ")[1]
    return result


def reconcile(first, second, keys):
    """一致仅表示 condition 接受，不作为 GT 正确率。"""
    values, invalid = {}, {}
    for key in keys:
        a, b = first.get(key), second.get(key)
        reason = (
            "format"
            if a not in ("YES", "NO") or b not in ("YES", "NO")
            else "disagreement" if a != b else None
        )
        values[key] = None if reason else a
        if reason:
            invalid[key] = reason
    return values, invalid


def collect_priors(ask, sample_key, recheck_mode="history", event_module=None):
    """ask(phase, spec, history) 接受实际多轮历史，返回原始回答与 prompt。

    先遍历两个 EVENT 域，避免 RS 门控漏掉 interrupted junction 上的 UE3。
    EVENT checkpoint 没有训练过 GROUP/DETAIL；复核使用它已训练的域内全问。
    """
    if recheck_mode not in ("history", "independent", "compare"):
        raise ValueError("recheck_mode must be history/independent/compare")
    p2 = event_module or default_event_prompts
    calls, comparisons, mode_disagreements = [], [], []

    def query(phase, spec, history=()):
        text, prompt = ask(phase, spec, list(history))
        row = dict(
            phase=phase,
            variant=spec.variant,
            keys=list(spec.output_keys),
            prompt=prompt,
            response=text,
            history=list(history),
        )
        calls.append(row)
        return strict_answers(text, spec.output_keys), (prompt, text)

    def recheck(phase, spec, initial, turn, scope):
        """compare 同时记录两种复核，condition 仍按 history 接受，不能当作正确率。"""
        chosen = None
        mode_answers = {}
        modes = (
            ("independent", "history") if recheck_mode == "compare" else (recheck_mode,)
        )
        for mode in modes:
            answer, next_turn = query(phase, spec, [turn] if mode == "history" else [])
            keys = [k for k in spec.output_keys if k in initial]
            _, errors = reconcile(initial, answer, keys)
            comparisons.append(
                dict(
                    scope=scope,
                    mode=mode,
                    keys=keys,
                    errors=errors,
                    compared_fields=len(keys),
                    same_prompt=next_turn[0] == turn[0],
                    accepted_fields=len(keys) - len(errors),
                )
            )
            mode_answers[mode] = answer
            chosen = answer
        if recheck_mode == "compare":
            _, cross_errors = reconcile(
                mode_answers["independent"], mode_answers["history"], spec.output_keys
            )
            mode_disagreements.append(dict(scope=scope, errors=cross_errors))
        return chosen

    spec = p1.make_prompt_spec(
        variant="all_random_order", answers={}, seed_key=sample_key
    )
    first, first_turn = query(1, spec)
    conditions = dict(first)
    invalid = {k: "format" for k, v in first.items() if v is None}
    hierarchy = []
    # 对四个 RS 全部复核，包括 NO；不能只确认第一次 YES 的类别。
    groups = {
        "RS1": "PLAIN_LANE_FOLLOWING_CORRIDOR",
        "RS2": "OPEN_SURFACE_PATH",
        "RS4": "JUNCTION_CONTROL_ZONE",
        "RS5": "LOCAL_RIGHT_OF_WAY_RULE",
    }
    for key, group in groups.items():
        hs = p1.make_prompt_spec(
            variant="hierarchical_probe",
            answers={},
            seed_key=f"{sample_key}:{key}",
            group_id=group,
            detail_key=key,
        )
        second = recheck(1, hs, first, first_turn, key)
        values, errors = reconcile(first, second, (*p1.PHASE1_ANSWER_KEYS, key))
        for name, value in values.items():
            if name not in invalid:
                conditions[name] = value
        invalid.update(errors)
        # GROUP 要与第一次完整 RS 向量的集合 membership 一致；R3 是完整全 NO。
        members = p1.GROUP_DEFINITIONS[group][3]
        if all(first.get(k) in ("YES", "NO") for k in p1.PHASE2_ANSWER_KEYS):
            expected = (
                "YES"
                if any(
                    first[k] == "YES" and k.replace("RS", "R") in members
                    for k in p1.PHASE2_ANSWER_KEYS
                )
                else "NO"
            )
            if second.get("GROUP") != expected:
                invalid[key] = (
                    "group_format"
                    if second.get("GROUP") is None
                    else "group_disagreement"
                )
        hierarchy.append(second.get("RS_HIGHWAY"))
    # HIGHWAY 与 RS_HIGHWAY 不是同一标签，不能强行要求相等。
    if None in hierarchy or len(set(hierarchy)) != 1:
        invalid["RS_HIGHWAY"] = "format" if None in hierarchy else "disagreement"
        conditions["RS_HIGHWAY"] = None
    else:
        conditions["RS_HIGHWAY"] = hierarchy[0]
    if all(first.get(k) in ("YES", "NO") for k in p1.PHASE2_ANSWER_KEYS):
        if sum(first[k] == "YES" for k in p1.PHASE2_ANSWER_KEYS) > 1:
            for k in p1.PHASE2_ANSWER_KEYS:
                invalid[k] = "multiple_rs_yes"
    for domain in p2.QUESTION_DOMAINS:
        answers = {p2.DOMAIN_ANSWER_KEYS[domain]: True}
        es = p2.make_prompt_spec(
            variant="all_random_order",
            answers=answers,
            seed_key=f"{sample_key}:{domain}:first",
        )
        a, turn = query(2, es)
        rs = p2.make_prompt_spec(
            variant="all_random_order",
            answers=answers,
            seed_key=f"{sample_key}:{domain}:recheck",
        )
        # 多题域尽量改变输出顺序；单题 UE6 域无合法的新排列，必须如实记录 same_prompt。
        for attempt in range(16):
            if rs.output_keys != es.output_keys:
                break
            rs = p2.make_prompt_spec(
                variant="all_random_order",
                answers=answers,
                seed_key=f"{sample_key}:{domain}:recheck:{attempt}",
            )
        b = recheck(2, rs, a, turn, domain)
        values, errors = reconcile(a, b, es.output_keys)
        domain_key = f"{domain}/{p2.INVALID_KEY}"
        conditions[domain_key] = values.pop(p2.INVALID_KEY)
        if p2.INVALID_KEY in errors:
            invalid[domain_key] = errors.pop(p2.INVALID_KEY)
        for key in p2.event_keys_for_domain(domain):
            conditions[key] = values[key]
            if conditions[domain_key] != "NO":
                invalid[key] = (
                    "domain_inapplicable"
                    if conditions[domain_key] == "YES"
                    else "domain_unconfirmed"
                )
        invalid.update(errors)
    for key in invalid:
        conditions[key] = None
    rs_values = [conditions.get(k) for k in p1.PHASE2_ANSWER_KEYS]
    conditions["ROAD_STRUCTURE"] = (
        next(
            (
                k.replace("RS", "R")
                for k in p1.PHASE2_ANSWER_KEYS
                if conditions[k] == "YES"
            ),
            "R3",
        )
        if all(v is not None for v in rs_values)
        else None
    )
    counts = defaultdict(int)
    for reason in invalid.values():
        counts[reason] += 1
    return dict(
        conditions=conditions,
        invalid=invalid,
        invalid_counts=dict(counts),
        calls=calls,
        recheck_mode=recheck_mode,
        recheck_comparisons=comparisons,
        recheck_mode_disagreements=mode_disagreements,
        condition_acceptance_policy=(
            "independent_consistency"
            if recheck_mode == "independent"
            else "history_consistency"
        ),
        compare_requires_consensus=False,
        consistency_is_accuracy=False,
    )
