"""LeadMoT planning decoder 的配置。

本子包按从 ``AutoMoT`` 目录运行来写路径，例如：
``python qwen3vl_local/leadmot/train.py``。

默认值对齐 LEAD CARLA 设置：
- route head 预测 10 个 ego-frame route 点；
- waypoint head 预测 8 个未来 ego-frame waypoint；
- hidden_size=1024，对齐 Qwen3-VL-4B K/V 的 8 heads * 128 dim。

RoPE 模式：
- ``mrope``：给 LeadMoT 生成 token 使用 Qwen3-VL 风格 M-RoPE；
- ``mhrope``：head-wise multi-axis RoPE，用于消融；
- ``none``：生成 token 不加 RoPE。
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple


QWEN_BACKBONE_CONTRACT_SCHEMA = "leadmot_qwen_backbone_v1"
_ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")
_SFT_ADAPTER_CONFIG_NAMES = (
    "sft_new_loop_phase2_adapter_config.json",
    "sft_v5_adapter_config.json",
    "sft_v4_adapter_config.json",
    "sft_v3_adapter_config.json",
    "sft_v2_adapter_config.json",
)


def _sha256_file(path: Path) -> str:
    """流式计算文件 SHA256，避免把 adapter 权重整体读进内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _find_adapter_weight(adapter_dir: Path) -> Path:
    """定位 PEFT adapter 权重文件。"""

    for name in _ADAPTER_WEIGHT_NAMES:
        candidate = adapter_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"adapter weights not found under {adapter_dir}; expected one of {_ADAPTER_WEIGHT_NAMES}"
    )


def _find_sft_adapter_config(adapter_dir: Path) -> Path | None:
    """返回当前 adapter 的项目级自描述配置（如果存在）。"""

    for name in _SFT_ADAPTER_CONFIG_NAMES:
        candidate = adapter_dir / name
        if candidate.is_file():
            return candidate
    return None


def _adapter_fingerprint(adapter_dir: Path) -> tuple[str, dict[str, str], Path | None]:
    """对 PEFT 配置、权重和项目自描述配置生成稳定组合指纹。"""

    peft_config = adapter_dir / "adapter_config.json"
    if not peft_config.is_file():
        raise FileNotFoundError(f"adapter_config.json not found under {adapter_dir}")
    weight = _find_adapter_weight(adapter_dir)
    sft_config = _find_sft_adapter_config(adapter_dir)
    files = [peft_config, weight]
    if sft_config is not None:
        files.append(sft_config)
    per_file = {path.name: _sha256_file(path) for path in files}
    digest = hashlib.sha256()
    for name, value in sorted(per_file.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), per_file, sft_config


def build_qwen_backbone_contract(
    model_dir: str | Path,
    adapter_dir: str | Path | None = None,
) -> dict[str, Any]:
    """构建 LeadMoT checkpoint 绑定的 frozen Qwen/base+LoRA 合同。

    合同使用 base ``config.json`` 与 adapter 实际权重指纹，不用绝对路径充当身份，
    因而 checkpoint 搬到另一台机器后仍可通过显式新路径恢复同一组权重。
    """

    model_path = Path(model_dir).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Qwen model directory not found: {model_path}")
    model_config = model_path / "config.json"
    if not model_config.is_file():
        raise FileNotFoundError(f"Qwen config.json not found: {model_config}")
    contract: dict[str, Any] = {
        "schema": QWEN_BACKBONE_CONTRACT_SCHEMA,
        "base_model_dir": str(model_path),
        "base_config_sha256": _sha256_file(model_config),
        "adapter_enabled": False,
        "adapter_dir": "",
        "adapter_sha256": "",
        "adapter_file_sha256": {},
        "adapter_metadata": {},
    }
    raw_adapter = "" if adapter_dir is None else str(adapter_dir).strip()
    if not raw_adapter or raw_adapter.lower() in {"none", "base"}:
        return contract

    adapter_path = Path(raw_adapter).expanduser().resolve()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Qwen adapter directory not found: {adapter_path}")
    fingerprint, per_file, sft_config_path = _adapter_fingerprint(adapter_path)
    metadata: dict[str, Any] = {}
    if sft_config_path is not None:
        raw = json.loads(sft_config_path.read_text(encoding="utf-8"))
        keep = (
            "schema",
            "route",
            "dataset_name",
            "prompt_name",
            "production_prompt_sha256",
            "history_rgb_mode",
            "history_rgb_count",
            "history_rgb_selected_indices",
            "base_model_dir",
            "lora_vision_scope",
            "global_step",
            "seed",
        )
        metadata = {key: raw[key] for key in keep if key in raw}
        metadata["config_file"] = sft_config_path.name
    contract.update(
        {
            "adapter_enabled": True,
            "adapter_dir": str(adapter_path),
            "adapter_sha256": fingerprint,
            "adapter_file_sha256": per_file,
            "adapter_metadata": metadata,
        }
    )
    return contract


def resolve_qwen_adapter_dir(
    requested: str | Path | None,
    expected_contract: Mapping[str, Any] | None,
) -> str:
    """按 checkpoint 合同解析 eval/CARLA 的 adapter 路径。

    ``auto`` 优先恢复 checkpoint 记录的路径；模型搬家后由调用方显式传新路径，
    后续 SHA256 校验仍会保证它是同一份 adapter。
    """

    raw = "auto" if requested is None else str(requested).strip()
    auto = raw.lower() in {"", "auto"}
    expected_enabled = bool(expected_contract and expected_contract.get("adapter_enabled", False))
    if expected_enabled:
        candidate = str(expected_contract.get("adapter_dir", "")) if auto else raw
        if not candidate:
            raise ValueError(
                "checkpoint requires a Qwen adapter but records no usable path; "
                "pass --qwen-adapter-dir explicitly"
            )
        path = Path(candidate).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"checkpoint Qwen adapter path is unavailable: {path}; "
                "pass --qwen-adapter-dir with the relocated adapter"
            )
        return str(path)
    if auto:
        return ""
    return str(Path(raw).expanduser().resolve())


def require_qwen_backbone_match(
    expected_contract: Mapping[str, Any] | None,
    actual_contract: Mapping[str, Any],
    source: str | Path,
) -> None:
    """拒绝用错误的 base/LoRA prefix 分布加载 LeadMoT decoder checkpoint。"""

    actual_adapter = bool(actual_contract.get("adapter_enabled", False))
    if not expected_contract:
        if actual_adapter:
            raise ValueError(
                f"{source} is a legacy checkpoint without qwen_backbone metadata; "
                "refusing to attach a Qwen adapter because its prefix distribution is unknown"
            )
        return
    if expected_contract.get("schema") != QWEN_BACKBONE_CONTRACT_SCHEMA:
        raise ValueError(f"{source} has unsupported qwen_backbone schema: {expected_contract.get('schema')!r}")
    checks = (
        "base_config_sha256",
        "adapter_enabled",
        "adapter_sha256",
    )
    mismatches = {
        key: {"expected": expected_contract.get(key), "actual": actual_contract.get(key)}
        for key in checks
        if expected_contract.get(key) != actual_contract.get(key)
    }
    if mismatches:
        raise ValueError(
            f"Qwen backbone mismatch for {source}: "
            f"{json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}"
        )


@dataclass
class LeadMoTPlanningDecoderConfig:
    """LeadMoT decoder 用到的形状和结构开关。

    frozen Qwen prefix K/V 在 attention 前不经过 Linear 投影，因此
    ``hidden_size`` 必须等于 ``num_kv_heads * head_dim``。
    """

    # 生成 token 的 hidden 宽度，必须和 frozen Qwen K/V 宽度一致。
    hidden_size: int = 1024
    qwen_hidden_size: int = 2560
    point_dim: int = 2

    # Frozen Qwen prefix K/V 布局：(B, num_kv_heads, seq, head_dim)。
    num_kv_heads: int = 8
    head_dim: int = 128
    num_qwen_layers: int = 36
    kv_segment_mode: str = "select_last"

    # 只给生成 Q/K 用的 RoPE 配置。Qwen prefix K 已在 prefill 内带位置编码，
    # 这里绝不能重复旋转。
    rope_type: str = "mrope"
    rope_theta: float = 5000000.0
    mrope_section_dim: Tuple[int, int, int] = (16, 24, 24)
    mrope_section_head: Tuple[int, int, int] = (3, 3, 2)

    # LEAD BEV encoder 输出形状：(B, 512, 10, 12)。
    bev_channels: int = 512
    bev_grid: Tuple[int, int] = (10, 12)

    # 是否在 gen 序列里加入 BEV token（亦即"快推理是否融合 BEV 信息"）。
    # - True（默认）：gen 序列 = BEV(120) + speed + tp + ntp + final_goal + route + waypoint = 142 token；
    # - False：消融配置，gen 序列只含 22 个 status/query token
    #   （speed + tp + ntp + final_goal + route_q 10 + waypoint_q 8），
    #   decoder 完全靠 frozen Qwen prefix K/V + ego 状态做 planning，BEV encoder 仍可外部跑
    #   （只是 decoder 不接它的输出）。state_dict 在两档之间**不兼容**（bev_projector 一档存在
    #   一档不存在），切换时必须从头训或单独 warm start。
    use_bev: bool = True

    # final_goal token：第 4 个 status token，喂 LeadMoT decoder（默认启用）。
    # 与 tp/ntp 共享 WaypointInputAdaptor MLP，让坐标语义在同一空间。
    # 训练侧用 meta["next_target_points"][-1]（LEAD 采集保存的剩余 route 终点）转 ego；
    # 在线侧用 scenario_picker 对应 route XML 的最后一个 waypoint 转 ego。
    # **注意**：开启后 gen sequence 多 1 个 token，老 LeadMoT ckpt **不兼容**。
    use_final_goal: bool = True

    # 是否在 frozen Qwen3-VL prefix 里追加一张 SUBGOAL 关键帧 RGB + 显式 STATUS/SUBGOAL
    # 文本块。决定 prefix prompt 的语义，不改变 decoder 结构与 gen 序列长度，
    # 因此 state_dict 在 use_subgoal=True/False 之间**形状兼容**——但 prefix KV 分布
    # 差异巨大，cross-load 会让 attention 完全错配。
    # - True：build_dataset --with-subgoal-fields 必须按 scenario/run_id/anchor
    #   反查 keyframes，并写
    #   scenario/run_id/status/subgoal/subgoal_frame/subgoal_rgb_path 字段；
    #   train/eval/probe 走 LeadMoTTrainRuntime._run_subgoal_qwen_prefill 分支，
    #   offline runner 走 LeadOfflineMoTRunner._run_leadmot_qwen_prefill_subgoal，
    #   多喂 1 张 subgoal stitched RGB + 新的 system + STATUS/SUBGOAL 块。
    # - False：等价于现行 v1/v2 prefix，runner._run_leadmot_qwen_prefill 路径不变。
    # 与 use_bev **正交**：4 种组合都允许，但 prefix 不兼容需要分别训练。
    # 只支持离线训练/eval/probe/offline runner；eval_carla 在线 agent 会显式拒绝。
    use_subgoal: bool = False

    # Query 数量对齐 LEAD planning 标签。
    num_route_queries: int = 10
    num_waypoint_queries: int = 8
    waypoint_dt: float = 0.25

    # Decoder 深度：把 36 层 Qwen 压到 12 个 pooled-prefix block。
    num_layers: int = 12
    num_heads: int = 8
    mlp_ratio: float = 8.0 / 3.0
    dropout: float = 0.0

    speed_dim: int = 1
    target_point_dim: int = 2

    def total_gen_tokens(self) -> int:
        """返回 packed generated-token 序列长度。

        status_token 数：speed + tp + ntp (+ final_goal 若启用) = 3 或 4。
        use_bev=True + use_final_goal=True：BEV(120) + 4 status + 10 route + 8 wp = 142
        use_bev=True + use_final_goal=False：BEV(120) + 3 status + 18 query = 141（旧 ckpt 已舍弃）
        use_bev=False + use_final_goal=True：4 status + 18 query = 22
        use_bev=False + use_final_goal=False：3 status + 18 query = 21（旧 ckpt 已舍弃）
        """
        bev_tokens = self.bev_grid[0] * self.bev_grid[1] if self.use_bev else 0
        status_tokens = 4 if self.use_final_goal else 3
        return bev_tokens + status_tokens + self.num_route_queries + self.num_waypoint_queries

    def slice_layout(self):
        """返回 packed generated sequence 的 [start, end) 切片。

        这里必须和 ``LeadMoTPlanningDecoder._build_gen_sequence`` 的拼接顺序同步。
        两个 head 只读取 route 和 waypoint 对应切片。
        use_bev=False 时不放 "bev" 键，下游访问 layout["bev"] 应该先判断 use_bev。
        """
        idx = 0
        layout = {}
        if self.use_bev:
            bev_tokens = self.bev_grid[0] * self.bev_grid[1]
            layout["bev"] = (idx, idx + bev_tokens); idx += bev_tokens
        layout["speed"] = (idx, idx + 1); idx += 1
        layout["tp"] = (idx, idx + 1); idx += 1
        layout["ntp"] = (idx, idx + 1); idx += 1
        # final_goal 紧跟 ntp，与其它 status token 同段位置。
        # 不启用时该 key 不存在，下游访问 layout["final_goal"] 应先判断 use_final_goal。
        if self.use_final_goal:
            layout["final_goal"] = (idx, idx + 1); idx += 1
        layout["route"] = (idx, idx + self.num_route_queries); idx += self.num_route_queries
        layout["waypoint"] = (idx, idx + self.num_waypoint_queries)
        return layout

    @property
    def ffn_hidden_size(self) -> int:
        """SwiGLU feed-forward block 的内部宽度。"""
        return int(self.hidden_size * self.mlp_ratio)

    def active_mrope_section(self) -> Tuple[int, int, int]:
        """返回当前 RoPE 模式需要的 section 配置。"""
        if self.rope_type in {"mrope", "none"}:
            return self.mrope_section_dim
        if self.rope_type == "mhrope":
            return self.mrope_section_head
        raise ValueError(f"Unknown rope_type: {self.rope_type!r}")

    def validate_qwen_kv_shape(self) -> None:
        """检查直接 attention 到 Qwen prefix K/V 所需的不变量。"""
        if self.hidden_size != self.num_kv_heads * self.head_dim:
            raise ValueError(
                f"hidden_size must equal num_kv_heads * head_dim: "
                f"{self.hidden_size} != {self.num_kv_heads} * {self.head_dim}"
            )
        if self.num_heads != self.num_kv_heads:
            raise ValueError(
                f"num_heads must equal num_kv_heads: {self.num_heads} != {self.num_kv_heads}"
            )
        if self.rope_type not in {"mrope", "mhrope", "none"}:
            raise ValueError(f"rope_type must be 'mrope', 'mhrope', or 'none': {self.rope_type!r}")
        if self.rope_type == "none":
            return
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {self.head_dim}")
        if self.rope_type == "mrope":
            if sum(self.mrope_section_dim) != self.head_dim // 2:
                raise ValueError(
                    f"M-RoPE section sum {sum(self.mrope_section_dim)} must equal "
                    f"head_dim//2={self.head_dim // 2}"
                )
        else:
            if sum(self.mrope_section_head) > self.num_heads:
                raise ValueError(
                    f"MH-RoPE head section sum {sum(self.mrope_section_head)} "
                    f"cannot exceed num_heads={self.num_heads}"
                )
