"""Bench2Drive 专用 action_prior agent：只替换模型入口，沿用 LEAD 传感器与 PID。"""

import os
import sys
import json
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[key] = "1"
from qwen3vl_local.eval_carla import agent as shared
from qwen3vl_local.action_prior.carla_runtime import ActionPriorRunner


def get_entry_point():
    """leaderboard 反射加载。"""
    return "ActionPriorAgent"


class ActionPriorAgent(shared.MOTLeadAgent):
    """保留全部在线预处理/控制，模型输入禁止接触指标专用 hero 真值。"""

    def _route_endpoint(self, route_id):
        """正式 benchmark 的最后一个导航 waypoint，不借用训练路线终点。"""
        path = Path(os.environ["ROUTES"])
        route = next(
            r
            for r in ET.parse(path).getroot().findall("route")
            if int(r.get("id")) == route_id
        )
        point = route.findall("waypoints/position")[-1]
        return dict(
            endpoint=[float(point.get(k, "0")) for k in ("x", "y", "z")],
            town=route.get("town"),
            scenario=route.find("scenarios/scenario").get("type"),
            source="benchmark_route_xml_endpoint",
            xml_path=str(path),
            route_id=route_id,
        )

    def _create_runner(self, device, rope_type):
        """恢复独立 action 合同，LoRA 永不 merge 到 base。"""
        self.runner = ActionPriorRunner(
            self.leadmot_ckpt_path,
            device,
            os.environ["SAVE_PATH"],
            shared._LEADMOT_USE_EMA,
        )
        self.qwen_backbone_contract = self.runner.contract
        args = self.runner.args
        shared._TP_LOOKAHEAD_S = args.target_point_lookahead_s
        shared._NTP_LOOKAHEAD_S = args.next_target_point_lookahead_s
        shared._MIN_LOOKAHEAD_M = args.tp_min_lookahead_m
        shared._USE_FINAL_GOAL = self.runner.leadmot_config.use_final_goal
        if self.step_stride != 5:
            raise ValueError("action_prior requires 4 Hz inference, STEP_STRIDE=5")
        self.metric_info = {}
        self.prior_counts = __import__("collections").Counter()
        self._last_audit_index = -1

    def run_step(self, input_data, timestamp):
        """运动学真值仅用于舒适性审计；每两 tick 记录一次，与指标 dt=0.1s 对齐。"""
        control = super().run_step(input_data, timestamp)
        if self.step % 2 == 0:
            if getattr(self, "hero_actor", None) is None:
                self.get_hero()
            self.metric_info[str(self.step)] = self.get_metric_info()
        if self.runner.index != self._last_audit_index:
            from qwen3vl_local.action_prior.train import audit_counts

            audit = self.runner.runtime.prior.last_audit
            if audit:
                self.prior_counts.update(audit_counts(audit))
                # 只保存有限原始案例；全程计数仍完整。
                if self.runner.index <= 12:
                    (self.save_path / f"prior_{self.runner.index:04d}.json").write_text(
                        json.dumps(audit, ensure_ascii=False)
                    )
            self._last_audit_index = self.runner.index
        return control

    def destroy(self):
        """在父类释放模型之前落盘指标；不依赖视频是否开启。"""
        if hasattr(self, "save_path"):
            (self.save_path / "metric_info.json").write_text(
                json.dumps(getattr(self, "metric_info", {}))
            )
            (self.save_path / "prior_counts.json").write_text(
                json.dumps(dict(getattr(self, "prior_counts", {})))
            )
            samples = sorted(self.runner.latencies) if hasattr(self, "runner") else []
            latency = dict(
                inferences=len(samples),
                mean_seconds=sum(samples) / len(samples) if samples else None,
                p95_seconds=(
                    samples[min(len(samples) - 1, int(len(samples) * 0.95))]
                    if samples
                    else None
                ),
                real_time_verified=False,
            )
            (self.save_path / "latency.json").write_text(json.dumps(latency))
        super().destroy()
