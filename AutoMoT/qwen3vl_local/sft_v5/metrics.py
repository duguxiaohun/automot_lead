"""SFT v5 probe/eval 共用的自由生成指标。

本模块只处理已经解析好的逐帧记录，不加载 Qwen，也不依赖 CUDA。这样小样本 probe
和大样本 eval 可以共享完全相同的假阳性、假阴性和变化帧指标定义，避免两个入口对
同一个数字给出不同解释。

指标分三层：逐帧 RS/EVENT 正确性、只在 RS gate 通过时统计的 conditional Q2、以及
把 RS 错误导致 Q2 跳过也计错的 end-to-end EVENT。变化检测另外比较相邻“实际连续且
都执行过模型”的帧对；随机 probe 的不连续样本不会混入 transition F1。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


METRIC_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "rs_acc": {
        "meaning": "每帧实际使用的 RS（新 RS_SLOW 输出或复用 memory）与真值一致的比例。",
        "direction": "higher_is_better",
    },
    "rs_slow_trigger_rate": {
        "meaning": "实际运行低频 RS_SLOW 的帧比例；稳定区默认约 25%，recovery 会提高该值。",
        "direction": "diagnostic",
    },
    "rs_slow_acc": {
        "meaning": "只在实际运行 RS_SLOW 的帧上计算 RS 选择准确率。",
        "direction": "higher_is_better",
    },
    "rs_transition_acc": {
        "meaning": "仅在相邻真值 RS 发生变化的首帧上计算 RS 准确率。",
        "direction": "higher_is_better",
    },
    "rs_stable_acc": {
        "meaning": "仅在真值 RS 未发生变化的帧上计算 RS 准确率。",
        "direction": "higher_is_better",
    },
    "rs_change_detection_precision": {
        "meaning": "模型判为 RS 发生变化的帧中，真值也在该帧变化的比例。",
        "direction": "higher_is_better",
    },
    "rs_change_detection_recall": {
        "meaning": "所有真实 RS 变化首帧中，模型预测 RS 也在该帧发生变化的比例。",
        "direction": "higher_is_better",
    },
    "rs_change_detection_f1": {
        "meaning": "RS 变化帧检测 precision 与 recall 的调和平均。",
        "direction": "higher_is_better",
    },
    "rs_change_false_positive_rate": {
        "meaning": "真值 RS 稳定的相邻帧对中，模型错误切换 RS 的比例。",
        "direction": "lower_is_better",
    },
    "abnormal_acc": {
        "meaning": "EVENT_FAST 选择的 RE/UE family 与真值 normal/abnormal 是否一致。",
        "direction": "higher_is_better",
    },
    "abnormal_precision": {
        "meaning": "预测为异常的帧中真实异常的比例 TP/(TP+FP)，衡量误报纯度。",
        "direction": "higher_is_better",
    },
    "abnormal_recall": {
        "meaning": "所有真实异常帧中 EVENT_FAST 选择任意 UE 的比例 TP/GT_UE。",
        "direction": "higher_is_better",
    },
    "abnormal_f1": {
        "meaning": "EVENT_FAST 的 UE/RE precision 与 recall 调和平均。",
        "direction": "higher_is_better",
    },
    "abnormal_specificity": {
        "meaning": "所有真实 RE 帧中 EVENT_FAST 正确选择 RE 的比例 TN/GT_RE。",
        "direction": "higher_is_better",
    },
    "abnormal_false_positive_rate": {
        "meaning": "真实 RE 帧被 EVENT_FAST 选成任意 UE 的比例 FP/GT_RE。",
        "direction": "lower_is_better",
    },
    "abnormal_false_negative_rate": {
        "meaning": "真实 UE 帧未被 EVENT_FAST 选成 UE 的比例，包含 RE、跳过和非法输出。",
        "direction": "lower_is_better",
    },
    "abnormal_invalid_rate": {
        "meaning": "EVENT_FAST 未产生可解析 RE/UE 选择的帧比例。",
        "direction": "lower_is_better",
    },
    "abnormal_boundary_acc": {
        "meaning": "仅在 RE/UE 真值切换首帧上计算 EVENT_FAST family 准确率。",
        "direction": "higher_is_better",
    },
    "ue_entry_detection_precision": {
        "meaning": "模型判为从 RE 进入 UE 的帧中，真值也在该帧进入 UE 的比例。",
        "direction": "higher_is_better",
    },
    "ue_entry_detection_recall": {
        "meaning": "所有真实 RE->UE 进入帧中，模型也在该帧进入 UE 的比例。",
        "direction": "higher_is_better",
    },
    "ue_entry_detection_f1": {
        "meaning": "UE 进入帧检测 precision 与 recall 的调和平均。",
        "direction": "higher_is_better",
    },
    "ue_entry_false_positive_rate": {
        "meaning": "真值未进入 UE 的相邻帧对中，模型错报 RE->UE 的比例。",
        "direction": "lower_is_better",
    },
    "ue_exit_detection_precision": {
        "meaning": "模型判为从 UE 退出到 RE 的帧中，真值也在该帧退出 UE 的比例。",
        "direction": "higher_is_better",
    },
    "ue_exit_detection_recall": {
        "meaning": "所有真实 UE->RE 退出帧中，模型也在该帧退出 UE 的比例。",
        "direction": "higher_is_better",
    },
    "ue_exit_detection_f1": {
        "meaning": "UE 退出帧检测 precision 与 recall 的调和平均。",
        "direction": "higher_is_better",
    },
    "ue_exit_false_positive_rate": {
        "meaning": "真值未退出 UE 的相邻帧对中，模型错报 UE->RE 的比例。",
        "direction": "lower_is_better",
    },
    "q2_trigger_rate": {
        "meaning": "RS gate 正确并实际运行 EVENT_FAST 的帧比例。",
        "direction": "diagnostic",
    },
    "q2_skip_due_rs_rate": {
        "meaning": "因 RS_SLOW 输出或复用 RS memory 错误而跳过 EVENT_FAST 的帧比例。",
        "direction": "lower_is_better",
    },
    "event_acc_when_rs_correct": {
        "meaning": "实际进入 Q2 的帧中 EVENT 标签按动态多标签容错规则计正确的比例。",
        "direction": "higher_is_better",
    },
    "q2_ue_precision": {
        "meaning": "Q2 预测为某个 U-E* 的帧中真实为 UE 的比例。",
        "direction": "higher_is_better",
    },
    "q2_ue_recall": {
        "meaning": "进入 Q2 的真实 UE 帧中被预测为任意 U-E* 的比例。",
        "direction": "higher_is_better",
    },
    "q2_ue_f1": {
        "meaning": "Q2 UE/RE 二分类 precision 与 recall 的调和平均。",
        "direction": "higher_is_better",
    },
    "q2_false_positive_rate": {
        "meaning": "进入 Q2 的真实 RE 帧被错报为任意 U-E* 的比例。",
        "direction": "lower_is_better",
    },
    "q2_false_negative_rate": {
        "meaning": "进入 Q2 的真实 UE 帧未预测为 U-E* 的比例，包含 RE 和无法解析输出。",
        "direction": "lower_is_better",
    },
    "q2_invalid_rate": {
        "meaning": "进入 Q2 后 EVENT 选项无法解析的比例。",
        "direction": "lower_is_better",
    },
    "ue_acc": {
        "meaning": "进入 Q2 的真实 UE 帧中，具体 U-E* 标签按多标签容错规则正确的比例。",
        "direction": "higher_is_better",
    },
    "re_acc": {
        "meaning": "进入 Q2 的真实 RE 帧中正确输出 RE 的比例。",
        "direction": "higher_is_better",
    },
    "event_end_to_end_acc": {
        "meaning": "以所有帧为分母的 EVENT 严格准确率；Q1 RS 错导致 Q2 未触发也计错。",
        "direction": "higher_is_better",
    },
    "ue_end_to_end_recall": {
        "meaning": "所有真实 UE 帧中最终具体 U-E* 标签正确的比例，包含 Q1 门控失败。",
        "direction": "higher_is_better",
    },
    "event_end_to_end_false_positive_rate": {
        "meaning": "所有真实 RE 帧中最终输出任意 U-E* 的比例。",
        "direction": "lower_is_better",
    },
    "route_rs_all_correct_ratio": {
        "meaning": "整条 route 每一帧 RS 都正确的 route 比例。",
        "direction": "higher_is_better",
    },
    "route_abnormal_f1_macro": {
        "meaning": "先逐 route 计算 EVENT_FAST RE/UE F1，再对有定义的 route 等权平均。",
        "direction": "higher_is_better",
    },
    "route_ue_f1_macro": {
        "meaning": "先逐 route 计算 Q2 UE/RE F1，再对有定义的 route 等权平均。",
        "direction": "higher_is_better",
    },
    "mean_resets_per_100_frames": {
        "meaning": "每 100 帧实际应用的 GT memory 强制纠错次数；student closed-loop 测试中应为 0。",
        "direction": "lower_is_better",
    },
    "mean_training_reset_recommendations_per_100_frames": {
        "meaning": "兼容旧报告：每 100 帧出现 Q1 RS 错或 Q2 非法的即时失败信号次数；v5 延迟修复课程不会据此在下一帧直接 reset。",
        "direction": "diagnostic",
    },
    "rs_wrong_memory_copy_rate": {
        "meaning": "输入 RS memory 已知且错误时，Q1 仍原样输出该错误 RS 的比例；直接衡量 memory shortcut。",
        "direction": "lower_is_better",
    },
    "rs_wrong_or_unknown_memory_recovery_rate": {
        "meaning": "输入 RS memory 错误或 UNKNOWN 时，Q1 在当前帧自行恢复到 GT RS 的比例。",
        "direction": "higher_is_better",
    },
    "event_wrong_memory_copy_rate": {
        "meaning": "进入 Q2 且 EVENT memory 已知错误时，Q2 仍复制该错误 EVENT 的比例。",
        "direction": "lower_is_better",
    },
    "event_wrong_or_unknown_memory_recovery_rate": {
        "meaning": "进入 Q2 且 EVENT memory 错误或 UNKNOWN 时，Q2 自行恢复到可接受 EVENT 的比例。",
        "direction": "higher_is_better",
    },
    "mean_valid_frames_per_route": {
        "meaning": "每条参与评估 route 的平均有效帧数，用于审计评估规模。",
        "direction": "diagnostic",
    },
}


TRANSITION_METRIC_NAMES = (
    "rs_change_detection_precision",
    "rs_change_detection_recall",
    "rs_change_detection_f1",
    "rs_change_false_positive_rate",
    "ue_entry_detection_precision",
    "ue_entry_detection_recall",
    "ue_entry_detection_f1",
    "ue_entry_false_positive_rate",
    "ue_exit_detection_precision",
    "ue_exit_detection_recall",
    "ue_exit_detection_f1",
    "ue_exit_false_positive_rate",
)


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    """安全计算比例；无有效分母时返回 ``None``。

    ``None`` 表示当前评估集合没有这种样本，例如全程无 UE 时 recall 没有定义；它与
    “存在 UE 但一个都没检出”的 0.0 含义不同，写报告时必须保留这个区别。
    """

    if int(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def _f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    """由 precision/recall 计算 F1，并保留未定义的 ``None`` 语义。"""

    if precision is None or recall is None:
        return None
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _transition_outcome(gt: bool, pred: Optional[bool]) -> str:
    """把一个变化检测结果编码为 TP/FP/FN/TN/invalid。

    ``pred=None`` 通常表示前后任一离散输出无法解析；它单列为 invalid，不按“未变化”
    处理，否则会虚高 TN 或把格式错误错误地归为 FN。
    """

    if pred is None:
        return "invalid"
    if gt and pred:
        return "TP"
    if not gt and pred:
        return "FP"
    if gt and not pred:
        return "FN"
    return "TN"


def transition_case_from_row(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """将统一逐帧记录压缩为一条真值/预测变化对比。

    只有上一个原始时间帧也实际跑过模型时，``transition_pair_evaluated`` 才为
    true。随机 probe 中的跳帧不会被伪造成变化帧指标。返回值保留前后状态和三类
    outcome，供 transition_report.json 直接定位具体错帧。
    """

    if not bool(row.get("transition_pair_evaluated")):
        return None
    # 先把可能来自 JSON 的 0/1/None 统一成严格 bool/Optional[bool]，后续混淆矩阵
    # 不需要知道 eval/probe 的序列化细节。
    gt_rs_change = bool(row.get("gt_rs_change"))
    gt_ue_entry = bool(row.get("gt_ue_entry"))
    gt_ue_exit = bool(row.get("gt_ue_exit"))
    pred_rs_change_raw = row.get("pred_rs_change")
    pred_ue_entry_raw = row.get("pred_ue_entry")
    pred_ue_exit_raw = row.get("pred_ue_exit")
    pred_rs_change = None if pred_rs_change_raw is None else bool(pred_rs_change_raw)
    pred_ue_entry = None if pred_ue_entry_raw is None else bool(pred_ue_entry_raw)
    pred_ue_exit = None if pred_ue_exit_raw is None else bool(pred_ue_exit_raw)
    return {
        "scenario": row.get("scenario"),
        "route_id": row.get("route_id"),
        "frame_id": row.get("frame_id"),
        "previous_frame_id": row.get("previous_frame_id"),
        "previous_gt_rs_label": row.get("previous_gt_rs_label"),
        "gt_rs_label": row.get("gt_rs_label"),
        "previous_pred_rs_label": row.get("previous_pred_rs_label"),
        "pred_rs_label": row.get("pred_rs_label"),
        "previous_gt_abnormal": row.get("previous_gt_abnormal"),
        "gt_abnormal": row.get("gt_abnormal"),
        "previous_pred_abnormal": row.get("previous_pred_abnormal"),
        "pred_abnormal": row.get("pred_abnormal"),
        "gt_rs_change": gt_rs_change,
        "pred_rs_change": pred_rs_change,
        "rs_change_outcome": _transition_outcome(gt_rs_change, pred_rs_change),
        "gt_ue_entry": gt_ue_entry,
        "pred_ue_entry": pred_ue_entry,
        "ue_entry_outcome": _transition_outcome(gt_ue_entry, pred_ue_entry),
        "gt_ue_exit": gt_ue_exit,
        "pred_ue_exit": pred_ue_exit,
        "ue_exit_outcome": _transition_outcome(gt_ue_exit, pred_ue_exit),
    }


def transition_case_is_informative(case: Mapping[str, Any]) -> bool:
    """判断一个 frame pair 是否值得在人工变化报告中展开。

    全部为稳定 TN 的 pair 仍进入总体分母，但不必逐条写进精简人工列表；真实变化、
    模型报变化或相应 invalid 才是需要回看的信息样本。
    """

    return any(
        bool(case.get(key))
        for key in (
            "gt_rs_change",
            "pred_rs_change",
            "gt_ue_entry",
            "pred_ue_entry",
            "gt_ue_exit",
            "pred_ue_exit",
        )
    )


def build_transition_fields(
    *,
    pair_evaluated: bool,
    previous_frame_id: Optional[int],
    previous_gt_rs_label: Optional[str],
    gt_rs_label: str,
    previous_pred_rs_label: Optional[str],
    pred_rs_label: Optional[str],
    previous_gt_abnormal: Optional[bool],
    gt_abnormal: bool,
    previous_pred_abnormal: Optional[bool],
    pred_abnormal: Optional[bool],
) -> Dict[str, Any]:
    """用相邻帧真值/预测状态生成统一变化检测字段。

    ``pair_evaluated`` 表示上一帧也实际跑过当前模型。模型任一帧输出
    无法解析时，对应预测变化保留 ``None``，后续指标按 invalid 处理。真值变化仍可
    计算并写日志，但只有 ``transition_pair_evaluated=True`` 才进入变化指标分母。
    """

    # GT 前态和预测前态分开判定：有 GT 只能说明真实边界可定义，不代表模型边界可定义。
    has_gt_previous = previous_gt_rs_label is not None and previous_gt_abnormal is not None
    gt_rs_change = bool(has_gt_previous and previous_gt_rs_label != gt_rs_label)
    gt_ue_entry = bool(has_gt_previous and not bool(previous_gt_abnormal) and bool(gt_abnormal))
    gt_ue_exit = bool(has_gt_previous and bool(previous_gt_abnormal) and not bool(gt_abnormal))
    pred_rs_change = (
        None
        if not pair_evaluated or previous_pred_rs_label is None or pred_rs_label is None
        else previous_pred_rs_label != pred_rs_label
    )
    pred_ue_entry = (
        None
        if not pair_evaluated or previous_pred_abnormal is None or pred_abnormal is None
        else not bool(previous_pred_abnormal) and bool(pred_abnormal)
    )
    pred_ue_exit = (
        None
        if not pair_evaluated or previous_pred_abnormal is None or pred_abnormal is None
        else bool(previous_pred_abnormal) and not bool(pred_abnormal)
    )
    return {
        "transition_pair_evaluated": bool(pair_evaluated and has_gt_previous),
        "previous_frame_id": previous_frame_id,
        "previous_gt_rs_label": previous_gt_rs_label,
        "previous_pred_rs_label": previous_pred_rs_label,
        "previous_gt_abnormal": previous_gt_abnormal,
        "previous_pred_abnormal": previous_pred_abnormal,
        "gt_rs_change": gt_rs_change,
        "pred_rs_change": pred_rs_change,
        "gt_ue_entry": gt_ue_entry,
        "pred_ue_entry": pred_ue_entry,
        "gt_ue_exit": gt_ue_exit,
        "pred_ue_exit": pred_ue_exit,
    }


@dataclass
class _BinaryCounts:
    """可流式更新的严格二分类混淆矩阵。

    positive 的具体语义由持有者决定：abnormal/q2 中 positive=UE，RS change 中
    positive=发生变化，entry/exit 中 positive=对应边界发生。统一实现可确保所有 FPR、
    FNR、invalid 的分母口径完全一致。
    """

    total: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    invalid: int = 0
    gt_positive: int = 0
    gt_negative: int = 0
    predicted_positive: int = 0

    def update(self, gt: bool, pred: Optional[bool]) -> None:
        """加入一条二分类记录；预测为 ``None`` 时单列 invalid，绝不伪装成负类。"""

        self.total += 1
        self.gt_positive += int(gt)
        self.gt_negative += int(not gt)
        if pred is None:
            self.invalid += 1
            return
        self.predicted_positive += int(pred)
        if gt and pred:
            self.tp += 1
        elif not gt and pred:
            self.fp += 1
        elif not gt and not pred:
            self.tn += 1
        else:
            self.fn += 1

    def summary(self) -> Dict[str, Any]:
        """按累计混淆矩阵计算严格指标，无分母时返回 ``None``。

        invalid 被计入 total，因此会降低 accuracy/invalid_rate；在 FNR 中，真实正类里
        除 TP 外的所有样本（含 invalid）都算漏检，符合安全评估的严格口径。
        """

        precision = _ratio(self.tp, self.predicted_positive)
        recall = _ratio(self.tp, self.gt_positive)
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "invalid": self.invalid,
            "gt_positive": self.gt_positive,
            "gt_negative": self.gt_negative,
            "accuracy": _ratio(self.tp + self.tn, self.total),
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "specificity": _ratio(self.tn, self.gt_negative),
            "false_positive_rate": _ratio(self.fp, self.gt_negative),
            "false_negative_rate": _ratio(self.gt_positive - self.tp, self.gt_positive),
            "invalid_rate": _ratio(self.invalid, self.total),
        }


@dataclass
class _GroupCounts:
    """per-RS/per-EVENT 的轻量计数器，只保留汇总所需字段而不持有逐帧对象。"""

    frames: int = 0
    rs_correct: int = 0
    abnormal_correct: int = 0
    q2_triggered: int = 0
    q2_event_correct: int = 0

    def update(self, row: Mapping[str, Any]) -> None:
        """把统一逐帧记录加入当前 GT RS 或 GT EVENT 分组。

        ``event_acc_when_rs_correct`` 只以实际触发 Q2 的帧为分母；abnormal_acc 则覆盖
        分组内所有帧，RS gate 失败导致无 EVENT 输出时不会被当成正确。
        """

        self.frames += 1
        self.rs_correct += int(bool(row.get("rs_gate_correct", row.get("q1_rs_correct"))))
        pred_event_is_ue = row.get("pred_event_is_ue")
        self.abnormal_correct += int(
            pred_event_is_ue is not None
            and bool(pred_event_is_ue) == bool(row.get("gt_abnormal"))
        )
        if bool(row.get("q2_triggered")):
            self.q2_triggered += 1
            self.q2_event_correct += int(bool(row.get("q2_event_correct")))

    def summary(self) -> Dict[str, Any]:
        """返回该分组样本量、RS/family 准确率以及 RS gate 后的 conditional Q2 准确率。"""

        return {
            "frames": self.frames,
            "rs_acc": _ratio(self.rs_correct, self.frames),
            "abnormal_acc": _ratio(self.abnormal_correct, self.frames),
            "q2_triggered": self.q2_triggered,
            "event_acc_when_rs_correct": _ratio(self.q2_event_correct, self.q2_triggered),
        }


class StudentMetricsAccumulator:
    """probe/eval 共用的 O(类别数) 内存流式指标器。

    调用方对每个 frame 调一次 :meth:`update`，结束后调用 :meth:`summary`。对象只保存
    计数和少量 per-RS/per-EVENT 分组，不保存 prompt、图片、logits 或逐帧字典，因此可
    用于全量 validation。输入 row 的字段合同由 eval/probe 共同维护。
    """

    def __init__(self) -> None:
        """初始化总体分类、变化边界、conditional Q2、端到端与 memory 依赖计数器。"""

        # 总帧、Q1 RS 和 reset 计数使用所有评估帧作为分母。
        self.frames = 0
        self.q2_triggered = 0
        self.q2_skipped_rs_wrong = 0
        self.q1_triggered = 0
        self.q1_checked_correct = 0
        self.rs_correct = 0
        self.rs_transition_frames = 0
        self.rs_transition_correct = 0
        self.rs_stable_frames = 0
        self.rs_stable_correct = 0
        self.abnormal_boundary_frames = 0
        self.abnormal_boundary_correct = 0
        self.event_correct = 0
        self.q2_candidate_mismatch = 0
        self.q2_ue_total = 0
        self.q2_ue_exact_correct = 0
        self.q2_re_total = 0
        self.q2_re_exact_correct = 0
        self.all_ue_total = 0
        self.all_ue_exact_correct = 0
        self.all_re_total = 0
        self.end_to_end_fp = 0
        self.reset_count = 0
        self.training_reset_recommendation_count = 0
        self.rs_memory_known_wrong = 0
        self.rs_memory_unknown = 0
        self.rs_memory_copied_when_wrong = 0
        self.rs_memory_recovered = 0
        self.event_memory_known_wrong = 0
        self.event_memory_unknown = 0
        self.event_memory_copied_when_wrong = 0
        self.event_memory_recovered = 0
        # normal/abnormal 直接由 EVENT_FAST 的 RE/UE family 得到；保留 abnormal_binary
        # 键名兼容旧报告，但不再存在独立 Q1 ABNORMAL 输出。
        self.abnormal_binary = _BinaryCounts()
        self.q2_binary = _BinaryCounts()
        # 三套变化检测矩阵比较“相邻两帧是否变化”，与当前帧的
        # RS gate 与 EVENT_FAST 的 RE/UE family 准确率是两个不同问题。
        self.rs_change_binary = _BinaryCounts()
        self.ue_entry_binary = _BinaryCounts()
        self.ue_exit_binary = _BinaryCounts()
        self.per_rs: Dict[str, _GroupCounts] = defaultdict(_GroupCounts)
        self.per_event: Dict[str, _GroupCounts] = defaultdict(_GroupCounts)

    def update(self, row: Mapping[str, Any]) -> None:
        """消费一帧统一记录，并更新所有流式计数器。

        调用方每个实际评估 frame 必须恰好调用一次。更新顺序为：全帧 RS/memory 指标 →
        边界指标 → 全量 UE/RE 分母 → conditional Q2 → per-label 分组。这里不保存 row，
        所以全量 eval 的内存不会随帧数增长。
        """

        # normal/abnormal 不再来自单独问答，而由 EVENT canonical label 是否为 U-E* 推出。
        gt_abnormal = bool(row.get("gt_abnormal"))
        pred_abnormal_raw = row.get("pred_event_is_ue", row.get("pred_abnormal"))
        pred_abnormal = None if pred_abnormal_raw is None else bool(pred_abnormal_raw)
        q1_triggered = bool(row.get("q1_triggered", True))
        q1_rs_correct = bool(row.get("q1_rs_correct"))
        rs_gate_correct = bool(row.get("rs_gate_correct", q1_rs_correct))
        q2_triggered = bool(row.get("q2_triggered"))

        # Q1 统计对所有帧生效；RS/UE 边界标记由 eval/probe 按原始 route 相邻帧生成。
        self.frames += 1
        self.q2_skipped_rs_wrong += int(
            bool(row.get("q2_skipped_rs_wrong", not rs_gate_correct))
        )
        self.reset_count += int(bool(row.get("reset_next")))
        self.training_reset_recommendation_count += int(
            bool(row.get("would_reset_under_training"))
        )
        self.q1_triggered += int(q1_triggered)
        self.q1_checked_correct += int(q1_triggered and q1_rs_correct)
        self.rs_correct += int(rs_gate_correct)
        self.rs_memory_known_wrong += int(bool(row.get("memory_rs_input_known_wrong")))
        self.rs_memory_unknown += int(bool(row.get("memory_rs_input_unknown")))
        self.rs_memory_copied_when_wrong += int(bool(row.get("memory_rs_copied_when_wrong")))
        self.rs_memory_recovered += int(bool(row.get("memory_rs_recovered")))
        self.event_memory_known_wrong += int(bool(row.get("memory_event_input_known_wrong")))
        self.event_memory_unknown += int(bool(row.get("memory_event_input_unknown")))
        self.event_memory_copied_when_wrong += int(bool(row.get("memory_event_copied_when_wrong")))
        self.event_memory_recovered += int(bool(row.get("memory_event_recovered")))
        # abnormal_binary 是端到端 family 指标：RS gate 失败时 pred_abnormal=None，按
        # invalid/漏检处理；q2_binary 则只在下方 q2_triggered 分支更新。
        self.abnormal_binary.update(gt_abnormal, pred_abnormal)
        if bool(row.get("rs_transition")):
            self.rs_transition_frames += 1
            self.rs_transition_correct += int(rs_gate_correct)
        else:
            self.rs_stable_frames += 1
            self.rs_stable_correct += int(rs_gate_correct)
        if bool(row.get("abnormal_transition")):
            self.abnormal_boundary_frames += 1
            self.abnormal_boundary_correct += int(
                pred_abnormal is not None and pred_abnormal == gt_abnormal
            )
        if bool(row.get("transition_pair_evaluated")):
            # 三类变化分别建混淆矩阵，不能用“当前帧分类正确”替代边界检测正确。
            pred_rs_change_raw = row.get("pred_rs_change")
            pred_ue_entry_raw = row.get("pred_ue_entry")
            pred_ue_exit_raw = row.get("pred_ue_exit")
            self.rs_change_binary.update(
                bool(row.get("gt_rs_change")),
                None if pred_rs_change_raw is None else bool(pred_rs_change_raw),
            )
            self.ue_entry_binary.update(
                bool(row.get("gt_ue_entry")),
                None if pred_ue_entry_raw is None else bool(pred_ue_entry_raw),
            )
            self.ue_exit_binary.update(
                bool(row.get("gt_ue_exit")),
                None if pred_ue_exit_raw is None else bool(pred_ue_exit_raw),
            )

        # 先记录全量 UE/RE 分母，再单独进入 conditional Q2 分支。Q1 RS 错而未触发
        # Q2 的 UE 会降低端到端 recall，但不会污染 q2_ue_re_confusion 的条件分母。
        self.all_ue_total += int(gt_abnormal)
        self.all_re_total += int(not gt_abnormal)
        if q2_triggered:
            # conditional Q2 回答“上游 RS 已正确时 EVENT 本身学得怎样”；具体 UE 标签
            # event_ok 与 family pred_event_is_ue 同时保留，前者更严格。
            self.q2_triggered += 1
            pred_event_is_ue_raw = row.get("pred_event_is_ue")
            pred_event_is_ue = None if pred_event_is_ue_raw is None else bool(pred_event_is_ue_raw)
            self.q2_binary.update(gt_abnormal, pred_event_is_ue)
            event_ok = bool(row.get("q2_event_correct"))
            self.event_correct += int(event_ok)
            self.q2_candidate_mismatch += int(bool(row.get("q2_candidate_mismatch")))
            if gt_abnormal:
                self.q2_ue_total += 1
                self.q2_ue_exact_correct += int(event_ok)
            else:
                self.q2_re_total += 1
                self.q2_re_exact_correct += int(event_ok)
                self.end_to_end_fp += int(bool(pred_event_is_ue))
        if gt_abnormal:
            self.all_ue_exact_correct += int(bool(row.get("q2_event_correct")))

        # 分组只存小型计数器，便于定位某个 RS/EVENT 的系统性问题。
        rs_key = str(row.get("gt_rs_label") or "UNKNOWN")
        event_key = str(row.get("gt_event_label") or "UNKNOWN")
        self.per_rs[rs_key].update(row)
        self.per_event[event_key].update(row)

    def summary(self) -> Dict[str, Any]:
        """物化累计计数、扁平指标、分组指标和机器可读定义。

        返回值同时保留 ``metrics`` 子字典和顶层同名标量，兼容旧画图脚本；混淆矩阵与
        样本量也一并输出，避免只比较 F1 而忽略分母。调用本函数不会清空累计状态。
        """

        abnormal = self.abnormal_binary.summary()
        q2 = self.q2_binary.summary()
        rs_change = self.rs_change_binary.summary()
        ue_entry = self.ue_entry_binary.summary()
        ue_exit = self.ue_exit_binary.summary()
        # metrics 是面向画图/比较的扁平标量；外层同时保留原始计数和混淆矩阵，避免
        # 只看一个比率时无法判断样本量是否足够。
        metrics = {
            "rs_acc": _ratio(self.rs_correct, self.frames),
            "rs_slow_trigger_rate": _ratio(self.q1_triggered, self.frames),
            "rs_slow_acc": _ratio(self.q1_checked_correct, self.q1_triggered),
            "rs_transition_acc": _ratio(self.rs_transition_correct, self.rs_transition_frames),
            "rs_stable_acc": _ratio(self.rs_stable_correct, self.rs_stable_frames),
            "rs_change_detection_precision": rs_change["precision"],
            "rs_change_detection_recall": rs_change["recall"],
            "rs_change_detection_f1": rs_change["f1"],
            "rs_change_false_positive_rate": rs_change["false_positive_rate"],
            "abnormal_acc": abnormal["accuracy"],
            "abnormal_precision": abnormal["precision"],
            "abnormal_recall": abnormal["recall"],
            "abnormal_f1": abnormal["f1"],
            "abnormal_specificity": abnormal["specificity"],
            "abnormal_false_positive_rate": abnormal["false_positive_rate"],
            "abnormal_false_negative_rate": abnormal["false_negative_rate"],
            "abnormal_invalid_rate": abnormal["invalid_rate"],
            "abnormal_boundary_acc": _ratio(
                self.abnormal_boundary_correct,
                self.abnormal_boundary_frames,
            ),
            "ue_entry_detection_precision": ue_entry["precision"],
            "ue_entry_detection_recall": ue_entry["recall"],
            "ue_entry_detection_f1": ue_entry["f1"],
            "ue_entry_false_positive_rate": ue_entry["false_positive_rate"],
            "ue_exit_detection_precision": ue_exit["precision"],
            "ue_exit_detection_recall": ue_exit["recall"],
            "ue_exit_detection_f1": ue_exit["f1"],
            "ue_exit_false_positive_rate": ue_exit["false_positive_rate"],
            "q2_trigger_rate": _ratio(self.q2_triggered, self.frames),
            "q2_skip_due_rs_rate": _ratio(self.q2_skipped_rs_wrong, self.frames),
            "event_acc_when_rs_correct": _ratio(self.event_correct, self.q2_triggered),
            "q2_ue_precision": q2["precision"],
            "q2_ue_recall": q2["recall"],
            "q2_ue_f1": q2["f1"],
            "q2_false_positive_rate": q2["false_positive_rate"],
            "q2_false_negative_rate": q2["false_negative_rate"],
            "q2_invalid_rate": q2["invalid_rate"],
            "ue_acc": _ratio(self.q2_ue_exact_correct, self.q2_ue_total),
            "re_acc": _ratio(self.q2_re_exact_correct, self.q2_re_total),
            "event_end_to_end_acc": _ratio(self.event_correct, self.frames),
            "ue_end_to_end_recall": _ratio(self.all_ue_exact_correct, self.all_ue_total),
            "event_end_to_end_false_positive_rate": _ratio(self.end_to_end_fp, self.all_re_total),
            "rs_wrong_memory_copy_rate": _ratio(
                self.rs_memory_copied_when_wrong,
                self.rs_memory_known_wrong,
            ),
            "rs_wrong_or_unknown_memory_recovery_rate": _ratio(
                self.rs_memory_recovered,
                self.rs_memory_known_wrong + self.rs_memory_unknown,
            ),
            "event_wrong_memory_copy_rate": _ratio(
                self.event_memory_copied_when_wrong,
                self.event_memory_known_wrong,
            ),
            "event_wrong_or_unknown_memory_recovery_rate": _ratio(
                self.event_memory_recovered,
                self.event_memory_known_wrong + self.event_memory_unknown,
            ),
        }
        return {
            "frames": self.frames,
            "q1_rs_correct": self.rs_correct,
            "event_family_correct": abnormal["tp"] + abnormal["tn"],
            # 旧结果 schema 兼容别名；当前不存在 Q1 ABNORMAL 问题。
            "q1_abnormal_correct": abnormal["tp"] + abnormal["tn"],
            "q1_triggered": self.q1_triggered,
            "q1_checked_correct": self.q1_checked_correct,
            "q2_triggered": self.q2_triggered,
            "q2_skipped_rs_wrong": self.q2_skipped_rs_wrong,
            "q2_event_correct": self.event_correct,
            "q2_candidate_mismatch": self.q2_candidate_mismatch,
            "q2_invalid_output": q2["invalid"],
            "q2_ue_total": self.q2_ue_total,
            "q2_ue_correct": self.q2_ue_exact_correct,
            "q2_re_total": self.q2_re_total,
            "q2_re_correct": self.q2_re_exact_correct,
            # legacy key 只为旧 comparison.json 兼容保留；它历史上实际统计的是 RS
            # 错帧数而非真正 reset 次数。新消费方应读取 rs_wrong_frames/reset_count。
            "rs_wrong_resets": self.frames - self.rs_correct,
            "rs_wrong_frames": self.frames - self.rs_correct,
            "reset_count": self.reset_count,
            "training_reset_recommendation_count": self.training_reset_recommendation_count,
            "memory_dependency_counts": {
                "rs_known_wrong": self.rs_memory_known_wrong,
                "rs_unknown": self.rs_memory_unknown,
                "rs_copied_when_wrong": self.rs_memory_copied_when_wrong,
                "rs_recovered": self.rs_memory_recovered,
                "event_known_wrong": self.event_memory_known_wrong,
                "event_unknown": self.event_memory_unknown,
                "event_copied_when_wrong": self.event_memory_copied_when_wrong,
                "event_recovered": self.event_memory_recovered,
            },
            "rs_transition_frames": self.rs_transition_frames,
            "abnormal_boundary_frames": self.abnormal_boundary_frames,
            "abnormal_confusion": abnormal,
            "q2_ue_re_confusion": q2,
            "rs_change_confusion": rs_change,
            "ue_entry_confusion": ue_entry,
            "ue_exit_confusion": ue_exit,
            "metrics": metrics,
            **metrics,
            "per_rs": {key: value.summary() for key, value in sorted(self.per_rs.items())},
            "per_event": {key: value.summary() for key, value in sorted(self.per_event.items())},
            "metric_definitions": METRIC_DEFINITIONS,
        }


def summarize_student_predictions(frame_logs: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """便捷地从逐帧记录列表生成完整学生指标。

    大规模流式 eval 可直接持有 :class:`StudentMetricsAccumulator` 并逐条 ``update``；
    probe 已有小型 frame list 时使用本 helper 更方便，两条路径结果完全相同。
    """

    accumulator = StudentMetricsAccumulator()
    for row in frame_logs:
        accumulator.update(row)
    return accumulator.summary()


def build_transition_report(
    frame_logs: List[Mapping[str, Any]],
    *,
    summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """生成小样本变化帧对比报告。

    ``summary`` 已计算时可传入避免重复汇总；未传则现场计算。报告只保存状态、outcome、
    混淆矩阵和定义，不复制 prompt/logits/RGB，因此适合长期留作错帧审计产物。
    """

    resolved_summary = dict(summary or summarize_student_predictions(frame_logs))
    cases = [case for row in frame_logs if (case := transition_case_from_row(row)) is not None]
    return {
        "description": "逐相邻帧对比真值变化与模型预测变化；TP/FP/FN/TN 均按变化首帧计数。",
        "evaluated_pairs": len(cases),
        "informative_cases": sum(transition_case_is_informative(case) for case in cases),
        "metrics": {name: resolved_summary.get(name) for name in TRANSITION_METRIC_NAMES},
        "confusions": {
            "rs_change": resolved_summary.get("rs_change_confusion"),
            "ue_entry": resolved_summary.get("ue_entry_confusion"),
            "ue_exit": resolved_summary.get("ue_exit_confusion"),
        },
        "metric_definitions": {name: METRIC_DEFINITIONS[name] for name in TRANSITION_METRIC_NAMES},
        "cases": cases,
    }
