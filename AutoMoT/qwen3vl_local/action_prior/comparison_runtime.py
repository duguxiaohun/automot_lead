"""恢复三种已有模型的评测条件；所有损失与轨迹采样调用原共享评估器。"""
import argparse
from pathlib import Path

from qwen3vl_local.action_prior.comparison_cases import resolve_path, write_json, case_id
from qwen3vl_local.action_prior.progress import observed

PATH_FIELDS = ("data_root", "data_dir", "model_dir", "lead_bev_ckpt", "event_balance_index",
               "high_level_action_index", "prior_labels", "phase1_training_index", "phase2_training_index")


def restore_args(state, checkpoint, overrides):
    """恢复保存条件，路径可搬迁；不自动生成新标签或重选 LoRA。"""
    args = argparse.Namespace(**state["args"])
    variant = state.get("ablation_variant", "action_prior")
    expected_schema = "action_prior_checkpoint_v4" if variant == "action_prior" else "action_expert_ablation_checkpoint_v1"
    if state.get("schema") != expected_schema or variant not in ("action_prior", "bev_only", "qwen_simple"):
        raise ValueError("不支持的 checkpoint schema/variant")
    if state.get("trajectory_decoder") != "conditional_joint_trajectory_flow_matching_v2":
        raise ValueError("需要 joint trajectory FM checkpoint")
    args.selection_manifest = args.selection_output = args.lora_bundle = ""
    for name in PATH_FIELDS:
        value = overrides.get(name) or getattr(args, name, "")
        if name == "model_dir" and variant == "bev_only":
            continue
        if value:
            setattr(args, name, str(resolve_path(value, run=Path(checkpoint).parent)))
    if variant == "action_prior":
        if args.dataset_priors:
            args.phase1_adapter = args.phase2_adapter = ""
        else:
            from qwen3vl_local.action_prior.lora_bundle import restore_paths
            local = restore_paths(state["qwen_backbone"], checkpoint) if state["qwen_backbone"].get("phase1") else {"phase1": "", "phase2": ""}
            args.phase1_adapter, args.phase2_adapter = local["phase1"], local["phase2"]
    args.output_dir = str(Path(checkpoint).parent)
    return args, variant


def check_contract(state, args, variant):
    """使用正式入口的源码/权重/条件校验，新增对比工具不放宽合同。"""
    from qwen3vl_local.action_prior.contracts import file_hash
    if variant == "action_prior":
        from qwen3vl_local.action_prior.config import validate_args, build_contract
        from qwen3vl_local.action_prior.contracts import require_contract
        validate_args(args)
        contract = build_contract(args)
        require_contract(state["qwen_backbone"], contract)
    else:
        from qwen3vl_local.action_expert_ablation.common import validate_args, build_contract, require_contract
        validate_args(args, variant)
        contract = build_contract(args, variant)
        require_contract(state["condition_contract"], contract)
    for split in ("train", "val", "test"):
        if file_hash(Path(args.data_dir) / f"{split}.jsonl") != state["dataset_hashes"][split]:
            raise ValueError(f"{split} 索引不匹配 checkpoint")
    return contract


class CaptureRuntime:
    """只旁路保存实际预处理出的 PIL 输入与状态，不改变 decoder 参数。"""
    def __init__(self, runtime, output, variant):
        self.runtime, self.output, self.variant = runtime, Path(output), variant
        self.current_sample = None
        original = runtime.runner._prepare_inference_inputs
        self.original_prepare = original

        def capture(*args, **kwargs):
            """PIL 来自真实输入准备函数，避免重建历史帧或误判 BGR。"""
            result = original(*args, **kwargs)
            folder = self.output / case_id(self.current_sample)
            folder.mkdir(parents=True, exist_ok=True)
            # bev_only 真正使用当前 stitched RGB；其它变体也保留各自实际 Qwen 历史。
            images = result[0][-1:] if self.variant == "bev_only" else result[0]
            for index, im in enumerate(images):
                im.save(folder / f"input_rgb_{index:02d}.png")
            status = result[2].detach().float().cpu().tolist()[0]
            write_json(folder / "input.json", dict(
                speed_mps=status[0], target_point=status[1:3], next_target_point=status[3:5],
                final_goal=status[5:7], rgb_count=len(images),
                rgb_role="BEV current RGB" if self.variant == "bev_only" else "Qwen RGB history; BEV uses current RGB",
                action_token=self.current_sample.get("action_token"),
                action_token_id=self.current_sample.get("action_token_id")))
            return result
        runtime.runner._prepare_inference_inputs = capture

    def __getattr__(self, name):
        """评估器仍读取原 runtime 的 device/prior/audit。"""
        return getattr(self.runtime, name)

    def forward_sample(self, sample, *args, **kwargs):
        """标记当前帧供图像旁路保存。"""
        self.current_sample = sample
        return self.runtime.forward_sample(sample, *args, **kwargs)


@observed
def evaluate_worker(job):
    """单模型子进程运行，退出后释放 Qwen/BEV 显存。"""
    import os
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    from qwen3vl_local.action_prior.launch import ensure_gpu
    ensure_gpu()
    import torch
    from qwen3vl_local.action_prior.contracts import file_hash
    from qwen3vl_local.action_prior.comparison_cases import read_json
    from qwen3vl_local.action_prior.progress import current
    from qwen3vl_local.action_prior.flow_matching import ConditionalFlowMatchingDecoder, FlowMatchingConfig
    from qwen3vl_local.leadmot import LeadMoTPlanningDecoderConfig
    checkpoint = job["checkpoint"]
    if file_hash(checkpoint) != job["checkpoint_sha256"]:
        raise ValueError("计划后 checkpoint 被替换；请重新运行对比")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args, variant = restore_args(state, checkpoint, job["overrides"])
    contract = check_contract(state, args, variant)
    # 新生成的缓存/运行产物留在本次对比，不写回被测训练目录。
    args.output_dir = job["output"]
    # seed 单独作为配对评估协议；合同先按训练 seed 检查，保留先验噪声原定义。
    device = torch.device("cuda", 0)
    config = LeadMoTPlanningDecoderConfig(**state["decoder_config"])
    model = ConditionalFlowMatchingDecoder(config, FlowMatchingConfig(**state["flow_config"])).to(device=device, dtype=torch.float32)
    model.load_state_dict(state["ema_state_dict"]["shadow"], strict=True)
    del state
    if variant == "action_prior":
        from qwen3vl_local.action_prior.runtime import make_runtime
        from qwen3vl_local.action_prior.train import evaluate
        runtime = make_runtime(args, device, contract)
    else:
        from qwen3vl_local.action_expert_ablation.common import make_runtime, evaluate
        runtime = make_runtime(args, device, variant)
    # runtime 的训练条件 seed 保持原值；仅评估器的 eps/t/ODE seed 统一。
    import copy
    eval_args = copy.copy(args)
    eval_args.seed = job["seed"]
    eval_args.num_workers = job["workers"]
    eval_args.output_dir = job["output"]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.decoder_dtype]
    for split, path in job["rows"].items():
        out = Path(job["output"]) / split
        current().configure(out, "comparison")
        rows = read_json(path)
        if file_hash(path) != job["row_hashes"][split]:
            raise ValueError("计划后的 case 清单被修改")
        if not rows:
            write_json(out / "metrics.json", dict(samples=0, reason="no_cases_requested_or_available"))
            continue
        if variant == "action_prior":
            from qwen3vl_local.action_prior.provenance import annotate_upstream
            annotate_upstream(rows, contract["upstream_sources"])
        capture = CaptureRuntime(runtime, out / "inputs", variant)
        # 每个 split 结束恢复原函数，避免包装层叠加。
        try:
            metrics = evaluate(capture, model, config, rows, eval_args, dtype, 0, 1, 0, out / "cases")
            write_json(out / "metrics.json", metrics)
        finally:
            # capture closure 中的原始方法由 __init__ 保存。
            runtime.runner._prepare_inference_inputs = capture.original_prepare
