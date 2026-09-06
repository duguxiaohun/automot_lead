"""闭环恢复 action_prior，并将实时 clip 交给训练同款 forward（不构造 GT）。"""

import argparse
import os
from pathlib import Path


class ActionPriorRunner:
    """适配现有 CARLA agent 的 run_clip 与 LiDAR 对齐接口。"""

    def __init__(self, checkpoint, device, output_dir, use_ema=True):
        import torch
        from qwen3vl_local.action_prior.config import build_contract, validate_args
        from qwen3vl_local.action_prior.contracts import require_contract
        from qwen3vl_local.action_prior.runtime import make_runtime
        from qwen3vl_local.leadmot import (
            LeadMoTPlanningDecoder,
            LeadMoTPlanningDecoderConfig,
        )
        from qwen3vl_local.leadmot.train import _dtype

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("schema") != "action_prior_checkpoint_v2":
            raise ValueError("closed loop requires action_prior_checkpoint_v2")
        self.args = args = argparse.Namespace(**state["args"])
        args.phase1_adapter = state["qwen_backbone"]["phase1"]["path"]
        args.phase2_adapter = state["qwen_backbone"]["phase2"]["path"]
        for key in (
            "model_dir",
            "lead_bev_ckpt",
            "phase1_adapter",
            "phase2_adapter",
            "phase1_training_index",
            "phase2_training_index",
        ):
            override = os.environ.get("ACTION_" + key.upper())
            if override:
                setattr(args, key, override)
        validate_args(args)
        self.contract = build_contract(args)
        require_contract(state["qwen_backbone"], self.contract)
        self.leadmot_config = LeadMoTPlanningDecoderConfig(**state["decoder_config"])
        if self.leadmot_config.use_subgoal:
            raise ValueError("online subgoal RGB unavailable")
        args.output_dir = str(Path(output_dir).resolve())
        args.cache_priors = False  # 在线图像通常不重复；不把测试轨迹缓存混入训练目录。
        self.runtime = make_runtime(args, torch.device(device), self.contract)
        self.decoder = LeadMoTPlanningDecoder(self.leadmot_config).to(
            device=device, dtype=torch.float32
        )
        self.decoder.load_state_dict(
            state["ema_state_dict"]["shadow"] if use_ema else state["decoder"],
            strict=True,
        )
        self.decoder.eval().requires_grad_(False)
        self.dtype = _dtype(args.decoder_dtype)
        self.index = 0
        self.latencies = []
        # 多 GPU setup 同时写同一模型溯源，使用独立临时文件原子发布。
        import json
        import tempfile

        fd, temporary = tempfile.mkstemp(
            dir=Path(output_dir).parent, suffix=".contract.tmp"
        )
        with os.fdopen(fd, "w") as handle:
            json.dump(
                dict(
                    contract=self.contract,
                    args=vars(args),
                    decoder_config=state["decoder_config"],
                ),
                handle,
            )
        os.replace(temporary, Path(output_dir).parent / "model_contract.json")
        self.route_key = os.environ.get("ROUTES_SUBSET", "online")

    def _align_lidar_points_to_anchor(self, *args, **kwargs):
        """沿用训练 runner 的坐标对齐实现。"""
        return self.runtime.runner._align_lidar_points_to_anchor(*args, **kwargs)

    def run_clip(self, clip, **kwargs):
        """只消费 RGB/点云/当前状态/导航，不访问离线未来轨迹或 expert 标签。"""
        import torch

        sample = dict(scenario="online", run_id=self.route_key, anchor=self.index)
        self.index += 1
        import time

        started = time.perf_counter()
        with torch.no_grad():
            result = self.runtime.forward_sample(
                sample, self.decoder, self.leadmot_config, self.dtype, clip=clip
            )
        outputs = [
            dict(
                leadmot_route=result["pred_route"].detach().float().cpu(),
                leadmot_future_waypoints=result["pred_future_waypoints"]
                .detach()
                .float()
                .cpu(),
            )
        ]
        self.latencies.append(time.perf_counter() - started)  # CPU 拷贝已同步 CUDA。
        return outputs
