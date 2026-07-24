import copy
from typing import List, Tuple, Optional, Dict, Any
import re
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_causal_mask
from data.reasoning.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    add_special_tokens,
    prepare_attention_mask_per_sample,
)
from .qwen3vl_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.cache_utils.taylorseer import cache_init

import sys
# sys.path.insert(0, ...)  # Removed: use pip-installed transformers
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer as Qwen3Tokenizer
import os

# Qwen3VL model paths - centralized configuration
# Tokenizer path: contains tokenizer files (tokenizer.json, vocab.json, etc.)
QWEN3VL_TOKENIZER_PATH = None  # Set via ModelArguments.model_path
# Processor path: contains proper Qwen3VL config with model_type
QWEN3VL_PROCESSOR_PATH = None  # Set via ModelArguments.qwen3vl_path
# For backward compatibility
QWEN3VL_MODEL_PATH = QWEN3VL_TOKENIZER_PATH

# Auto-detect paths if not explicitly set
_automot_dir = os.path.dirname(os.path.abspath(__file__))
# automot.py is at Automot/mot/modeling/automotive/automot.py
# Automot root is 3 levels up
_mot_dp_root = os.path.dirname(os.path.dirname(os.path.dirname(_automot_dir)))

if QWEN3VL_TOKENIZER_PATH is None:
    _default_tokenizer_path = os.path.join(_mot_dp_root, "checkpoints", "mot", "0025000")
    if os.path.isdir(_default_tokenizer_path):
        QWEN3VL_TOKENIZER_PATH = _default_tokenizer_path

if QWEN3VL_PROCESSOR_PATH is None:
    _default_processor_path = os.path.join(_mot_dp_root, "checkpoints")
    if os.path.isdir(_default_processor_path) and os.path.isfile(os.path.join(_default_processor_path, "preprocessor_config.json")):
        QWEN3VL_PROCESSOR_PATH = _default_processor_path

# Use local Qwen3VL tokenizer
# Workaround for HuggingFace validation error with local paths
# Temporarily disable repo_id validation by monkey-patching
import huggingface_hub.utils._validators as validators
original_validate_repo_id = validators.validate_repo_id

def patched_validate_repo_id(repo_id):
    # Skip validation if it looks like a local path
    if repo_id and (repo_id.startswith('/') or repo_id.startswith('./')):
        return
    return original_validate_repo_id(repo_id)

validators.validate_repo_id = patched_validate_repo_id

if QWEN3VL_TOKENIZER_PATH is not None:
    try:
        tokenizer = Qwen3Tokenizer.from_pretrained(QWEN3VL_TOKENIZER_PATH, local_files_only=True, trust_remote_code=True)
    finally:
        # Restore original validation
        validators.validate_repo_id = original_validate_repo_id
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
else:
    validators.validate_repo_id = original_validate_repo_id
    tokenizer = None
    new_token_ids = None
    num_new_tokens = 0

class AutoMoTConfig(PretrainedConfig):
    """
    AutoMoT Configuration for Qwen3VL integration.
    
    This configuration adapts the original AutoMoTive config to work with 
    Qwen3VL's vision model and qwen3vl_navit text processing.
    """
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vision_config=None,  # Changed from vit_config to vision_config
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        # Qwen3VL specific vision parameters
        vision_spatial_merge_size=2,
        vision_max_num_patches=4096,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        num_waypoints=8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vision_config = vision_config  # Qwen3VL vision config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        # self._attn_implementation = "sdpa"  # default attn implementation
        # Qwen3VL vision specific
        self.vision_spatial_merge_size = vision_spatial_merge_size
        self.vision_max_num_patches = vision_max_num_patches
        
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift
        self.num_waypoints = num_waypoints
        self.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

class WaypointInputAdaptor(nn.Module):
    """
    将形状为 [B, N, 2] 的目标点输入映射为 [B, N, token_size] 的特征表示。

    参数：
        token_size: 输出特征维度。
        hidden_size: 隐藏层维度。
        norm_layer: 可选归一化层。
    """
    
    def __init__(
        self, 
        token_size: int = 2560,
        hidden_size: int = 256,
        hidden_size2: int = 512,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_layer = norm_layer
        
        # MLP: 2 -> 256 -> 512 -> 2560
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size),            # 2 -> 256
            nn.ReLU(True), 
            nn.Linear(hidden_size, hidden_size2), # 256 -> 512
            nn.ReLU(True), 
            nn.Linear(hidden_size2, token_size)   # 512 -> 2560
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        返回：
            形状为 [B, N, token_size] 的编码结果。
        """
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = self.mlp(x)
        return x

class RouteHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int = 256,
        future_waypoints: int = 20,
    ):
        super().__init__()
        self.future_waypoints = future_waypoints

        # 可学习查询向量
        self.query = nn.Parameter(
            0.02 * torch.randn(1, future_waypoints, hidden_size)
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim * 2),
            nn.SiLU(True),
            nn.Linear(mlp_dim * 2, mlp_dim),
            nn.SiLU(True),
            nn.Linear(mlp_dim, 2, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        参数：
            features: 形状为 (B, W, hidden_size) 的特征。
        返回：
            route: 形状为 (B, W, 2) 的路径坐标。
        """
        # 增量坐标转绝对坐标
        route = self.mlp(features).cumsum(dim=1)
        return route

    def build_queries(self, batch_size: int) -> torch.Tensor:
        """
        返回：
            形状为 (B, W, hidden_size) 的查询向量。
        """
        return self.query.expand(batch_size, -1, -1)


class WaypointsHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int = 512,
        mlp_hidden: int = 256,
        num_waypoints: int = 6,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints

        self.query = nn.Parameter(
            0.02 * torch.randn(1, num_waypoints, hidden_size)
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.SiLU(True),
            nn.Linear(mlp_dim, mlp_hidden),
            nn.SiLU(True),
            nn.Linear(mlp_hidden, 2, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        参数：
            features: 形状为 (B, T, hidden_size) 的特征。
        返回：
            waypoints: 形状为 (B, T, 2) 的轨迹点。
        """
        waypoints = self.mlp(features)
        return waypoints

    def build_queries(self, batch_size: int) -> torch.Tensor:
        return self.query.expand(batch_size, -1, -1)


#class AutoMoT(PreTrainedModel):
class AutoMoT(Qwen3VLPreTrainedModel):
    """
    AutoMoT 主模型（Qwen3VL 对齐版）。

    这个类负责把多模态输入（文本 / 图像 / BEV / 规划相关 token）组织成统一的 packed 序列，
    再调用 language_model（Qwen3VLTextModel 的 MoT 版本）完成前向与缓存推理。

    主要职责：
    1. 模态对齐：将视觉特征、BEV 特征、速度/目标点等映射到统一 hidden_size。
    2. 序列组包：按索引把不同模态 token 写入 packed_sequence。
    3. 损失计算：语言 CE、route/traj/velocity 等任务损失。
    4. 增量推理：构建/更新 KV-cache，并提供多种 prepare_* 与 forward_cache_* 工具函数。

    约定：
    - `und` 路径主要服务理解分支；
    - `gen` 路径主要服务动作/生成分支（MoT 模式下会走不同投影与索引集合）。
    """
    config_class = AutoMoTConfig
    base_model_prefix = 'automot'

    def __init__(self, language_model, vision_model, config: AutoMoTConfig):
        """
        初始化 AutoMoT 的核心子模块。

        参数：
        - language_model: 文本主干（通常是 Qwen3VLForConditionalGenerationMoT 的 text 部分）。
        - vision_model: 视觉编码器（Qwen3VL 视觉分支或其兼容实现）。
        - config: AutoMoTConfig，包含 MoT、视觉、位置编码、任务头等配置。
        """
        super().__init__(config)    

        # ===== 基础主干与维度配置 =====
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_mot = "MoT" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        # ===== 任务头与输入适配器 =====
        self.route_head = RouteHead(hidden_size=self.hidden_size)
        self.target_point_encoder = WaypointInputAdaptor(token_size=self.hidden_size)
        
        # TransFuser projector: 1512 -> hidden_size (2560)
        self.bev_encoder_proj = nn.Linear(1512, self.hidden_size, bias=True)
        
        self.velocity_encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, self.hidden_size),
        )

        # 轨迹与速度预测头。
        self.waypoints_head = WaypointsHead(hidden_size=self.hidden_size)
        self.velocity_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_size, 3),
        )

        # 常用特殊 token id，后续在 prompt 组织和日志调试中会用到。
        self.start_id = int(tokenizer.encode('<|im_start|>', add_special_tokens=False)[-1])
        self.end_id = int(tokenizer.encode('<|im_end|>', add_special_tokens=False)[-1])
        self.comma_id = int(tokenizer.encode(',', add_special_tokens=False)[-1])
        print("start id, end id, comma id are:", self.start_id, self.end_id, self.comma_id)
        self.user_prompt = "<|im_start|>user\n"
        self.assistant_prompt = "<|im_end|>\n<|im_start|>assistant"
        self.generation_start_token = "\n"

        # ===== visual_gen 分支：推理/动作相关查询 token 与投影 =====
        if config.visual_gen:
            self.vision_model = vision_model
            self.reasoning_query_dim = config.reasoning_query_dim
            self.reasoning_query_tokens = config.reasoning_query_tokens
            self.reasoning_queries = nn.Embedding(
                num_embeddings=self.reasoning_query_tokens,
                embedding_dim=self.reasoning_query_dim,
            )
            self.reasoning_projector = MLPconnector(self.reasoning_query_dim, self.hidden_size, config.connector_act)
            self.action_query_dim = config.action_query_dim
            self.action_query_tokens = config.action_query_tokens
            self.route_queries = nn.Embedding(
                num_embeddings=20,
                embedding_dim=self.action_query_dim,
            )
            self.route_projector = MLPconnector(self.action_query_dim, self.hidden_size, config.connector_act)
            self.waypoint_queries = nn.Embedding(
                num_embeddings=6,
                embedding_dim=self.action_query_dim,
            )
            self.waypoint_projector = MLPconnector(self.action_query_dim, self.hidden_size, config.connector_act)

        # ===== visual_und 分支：视觉理解链路与官方 processor =====
        if config.visual_und:
            # Qwen3VL vision model with pretrained alignment
            # No connector or position embedding needed - already handled internally!
            self.vision_model = vision_model  # Qwen3VLVisionModel
            
            # 使用 AutoProcessor 创建官方 Qwen3VL 处理器
            from transformers import AutoProcessor
            self.vision_processor = AutoProcessor.from_pretrained(QWEN3VL_PROCESSOR_PATH, local_files_only=True, trust_remote_code=True) if QWEN3VL_PROCESSOR_PATH else None


        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config

    def _init_weights(self):
        """
        额外权重初始化入口。

        说明：当前仅在 visual_gen 条件下对 `llm2vae` 做零初始化。
        注意：若配置中未创建 `llm2vae`，这里会依赖外部调用路径避免触发。
        """
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def ce_from_dict(self, logits: torch.Tensor, prob_dict: dict, order=("accelerate", "constant", "slow")):
        """
        将离散概率字典监督转换为交叉熵损失。

        典型场景：`prob_dict` 不是 one-hot，而是例如
        `{"accelerate": 0.2, "constant": 0.7, "slow": 0.1}` 这类软标签。
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        device, dtype = logits.device, logits.dtype
        t = torch.tensor([float(prob_dict.get(k.strip().lower(), 0.0)) for k in order],
                        device=device, dtype=dtype)
        t = torch.clamp(t, min=1e-8)
        t = t / t.sum()
        log_q = F.log_softmax(logits, dim=-1)
        return -(t * log_q).sum(dim=-1).mean() 

    def scores_from_start(self, logp, t_start, BANK, alpha=1.0, ce_quota: int = 10):
        """
        从起始位置 `t_start` 计算候选 token 序列集合 `BANK` 的长度归一化得分。

        - `logp`: 每个时刻的对数概率张量，形状约 [T, V]（或等价索引方式）。
        - `BANK`: 候选序列列表，每个元素是 token id 列表。
        - `alpha`: 长度惩罚系数，按 $L^\alpha$ 归一化。
        - `ce_quota`: 只在窗口 `[t_start, t_start + ce_quota)` 内评估，防止越界。
        """
        if t_start is None:
            return None
        hard_cap = logp.size(0)
        if ce_quota is not None and ce_quota > 0:
            hard_cap = min(hard_cap, t_start + ce_quota)

        outs = []
        for ids in BANK:
            Lk = len(ids)
            if t_start + Lk > hard_cap:
                outs.append(logp.new_tensor(-1e9))
                continue
            lp = 0.0
            for i, tid in enumerate(ids):
                lp += logp[t_start + i, tid]
            outs.append(lp / (Lk ** alpha)) 
        return torch.stack(outs, dim=0)

    def vad_traj_loss(self, pred_offset, gt_offset, lam_pos_1s: float = 0.2):
        """
        轨迹偏移监督损失（带时间步权重）。

        参数：
            pred_offset, gt_offset: 形状均为 (B, 6, 2)。
            lam_pos_1s: 预留参数（当前实现未直接使用）。

        说明：
            6 个时间步的权重为 3,3,2,2,1,1。
        """
        assert pred_offset.shape == gt_offset.shape
        assert gt_offset.dim() == 3 and gt_offset.size(1) == 6 and gt_offset.size(2) == 2

        # (6,) -> (1,6,1) -> broadcast to (B,6,2)
        w = gt_offset.new_tensor([3, 3, 2, 2, 1, 1]).view(1, 6, 1)
        
        l1 = (pred_offset - gt_offset).abs() * w
        avg_factor = w.sum() * gt_offset.size(0) * gt_offset.size(2)  # B * sum(w) * 2
        loss = l1.sum() / avg_factor.clamp(min=1.0)
        return loss

    def ade_fde_loss(
        self,
        pred,              # (B, T, 2)
        gt,                # (B, T, 2)
        mask=None,
        n_1s=2,
        n_2s=4,
        n_3s=6,
    ):
        """
        轨迹误差评估（按 1s / 2s / 3s 分段计算 L2）。

        参数：
            pred: 预测轨迹，形状 (B, T, 2)。
            gt: 真实轨迹，形状 (B, T, 2)。
            mask: 可选有效位掩码，形状 (B, T)。
            n_1s, n_2s, n_3s: 对应 1 秒、2 秒、3 秒采用的点数。

        返回：
            l2_1s: 前 1 秒平均 L2 误差。
            l2_2s: 前 2 秒平均 L2 误差。
            l2_3s: 前 3 秒平均 L2 误差。
        """
        disp = torch.norm(pred - gt, dim=-1)  # (B, T)
        B, T = disp.shape
        
        n_1s = min(n_1s, T)
        n_2s = min(n_2s, T)
        n_3s = min(n_3s, T)
        
        disp_1s = disp[:, :n_1s]  # (B, 4)
        disp_2s = disp[:, :n_2s]  # (B, 8)
        disp_3s = disp[:, :n_3s]  # (B, 12)
        
        if mask is not None:
            mask = mask.float()
            denom = mask.sum().clamp(min=1.0)
            ade = (disp * mask).sum() / denom
            
            m1 = mask[:, :n_1s]
            m2 = mask[:, :n_2s]
            m3 = mask[:, :n_3s]
            l2_1s = (disp_1s * m1).sum() / m1.sum().clamp(min=1.0)
            l2_2s = (disp_2s * m2).sum() / m2.sum().clamp(min=1.0)
            l2_3s = (disp_3s * m3).sum() / m3.sum().clamp(min=1.0)
        else:
            ade = disp.mean()
            l2_1s = disp_1s.mean()
            l2_2s = disp_2s.mean()
            l2_3s = disp_3s.mean()
        return l2_1s, l2_2s, l2_3s

    def position_to_offset(self, traj_gt: torch.Tensor):
        """
        绝对轨迹 -> 增量轨迹。

        输入/输出形状均为 `[B, T, 2]`：
        - 第 0 帧 offset 取原始位置；
        - 第 t 帧 offset = pos[t] - pos[t-1]。
        """
        offset = traj_gt.clone()
        offset[..., 1:, :] = traj_gt[..., 1:, :] - traj_gt[..., :-1, :]
        offset[..., 0, :] = traj_gt[..., 0, :]
        return offset

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        ce_loss_weights: Optional[torch.BoolTensor] = None,
        traj_loss_indexes: Optional[torch.LongTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_und_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_und_text_indexes: Optional[torch.LongTensor] = None,
        packed_gen_text_indexes: Optional[torch.LongTensor] = None,
        packed_reasoning_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        traj_gt: Optional[torch.Tensor] = None,
        route_gt: Optional[torch.Tensor] = None,
        # for visual generation
        packed_action_token_indexes: Optional[torch.LongTensor] = None,
        route_loss_indexes: Optional[torch.LongTensor] = None,
        v_indexes: Optional[torch.LongTensor] = None,
        future_speeds_tensors: Optional[torch.Tensor] = None,
        target_point_indexes: Optional[torch.LongTensor] = None,
        action_query_token_seqlens: List[int] = None,
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
        image_tensor_list: Optional[torch.Tensor] = None,
        image_grid_thw_list: Optional[torch.Tensor] = None,
        v_target_point: Optional[torch.Tensor] = None,
        probs: Optional[List[Dict[str, Any]]] = None,
        ### for bev encoder tokens
        bev_feature: Optional[torch.Tensor] = None,
        packed_bev_indexes: Optional[torch.LongTensor] = None,

    ) -> torch.Tensor:
        """
        训练阶段主前向：将多模态 token 组装到统一 packed 序列，经过语言模型后计算多任务损失。

        参数：
            sequence_length: packed 序列总长度。
            packed_text_ids: 一维文本 token id。
            packed_text_indexes: 文本 token 在 packed 序列中的位置索引。
            sample_lens: 每个样本在 packed 序列中的长度。
            nested_attention_masks: 每个样本的二维注意力掩码列表，0 表示可见，-inf 表示屏蔽。
            packed_position_ids: packed 后的位置 id（支持 Qwen3VL 的位置编码格式）。

            packed_vit_tokens: 视觉输入（patch 化后或处理器输出的视觉张量）。
            packed_vit_position_ids: 视觉 token 对应位置或网格信息。
            packed_vit_token_indexes: 视觉 token 在 packed 序列中的索引。
            vit_token_seqlens: 每张图对应视觉 token 数量。
            packed_label_ids: 语言监督标签。
            ce_loss_indexes: 交叉熵损失计算位置。

            padded_latent / patchified_vae_latent_shapes / packed_latent_position_ids /
            packed_vae_token_indexes / packed_timesteps / mse_loss_indexes:
                视觉生成相关输入（当前版本中部分保留为兼容字段）。

            bev_feature: BEV 编码器输出，典型形状为 [B, 1512, 8, 8]。
            packed_bev_indexes: BEV token 在 packed 序列中的索引（通常每样本 64 个）。

        返回：
            包含可用损失项的字典（会自动过滤 None），例如 ce、traj_loss、route_loss、velocity_loss、l2 指标等。
        """
        # ========== 块1：初始化 Packed 序列结构 ==========
        # 步骤1：将文本 token ID 转换为嵌入向量
        # - packed_text_ids：一维文本 token ID 数组，包含所有样本的所有文本 token
        # - 输出：[N_text, hidden_size]，其中 N_text 是所有样本的文本 token 总数
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        
        # 步骤2：创建零张量作为 packed 序列的容器
        # - packed_sequence：形状 [sequence_length, hidden_size]，将包含所有类型的 token 嵌入
        # - 各类型 token（文本、视觉、BEV、路由、轨迹等）会被分别计算然后加入这个容器
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        
        # 步骤3：将文本嵌入放入 packed 序列的对应位置
        # - packed_text_indexes：指示文本 token 应该放在 packed_sequence 中的位置
        # - 这样是为了在多模态情况下，不同模态的 token 可以对齐到统一的位置
        packed_sequence[packed_text_indexes] = packed_text_embedding

        # ========== 块2：构建注意力掩码 ==========
        # 注意力掩码用于控制 Transformer 中哪些位置可以看到哪些位置
        
        if nested_attention_masks is None:
            # 如果没有预先提供注意力掩码，则根据样本长度和注意模式动态生成
            
            # 步骤1：创建稀疏掩码
            # - sample_lens：每个样本在 packed 序列中占用的长度
            # - split_lens/attn_modes：可能指定不同的注意力分割方式（如交叉注意）
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            
            # 步骤2：计算序列总长度（用于创建注意力矩阵）
            seqlen = sum(sample_lens)
            
            # 步骤3：将稀疏掩码转换为高效的块掩码格式
            # - B=1 表示批大小为 1（因为样本已经 packed 成一个序列）
            # - H=num_heads：多头注意力的头数
            # - BLOCK_SIZE=128：块大小，用于优化内存访问
            # - _compile=True：启用编译优化
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            # 如果已经提供了嵌套的注意力掩码，直接使用
            attention_mask = nested_attention_masks

        # ========== 块3：处理视觉特征（Vision Understanding 分支）==========
        # 当启用了视觉理解分支时，需要将图像特征编码并加入 packed 序列
        
        deepstack_visual_embeds = None  # 初始化视觉嵌入容器
        visual_pos_masks = None  # 初始化视觉位置掩码
        
        if self.config.visual_und:
            # 步骤1：计算每张图对应的视觉 token 的累积长度
            # - vit_token_seqlens：列表或张量，包含每张图的 token 数（通常为 256/400 等）
            # - cumsum：计算累积和，后面的 pad 在开头补 0，便于后续索引计算
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            
            # 步骤2：获取最大的单张图 token 数
            # - max_seqlen：用于填充和内存分配
            max_seqlen = torch.max(vit_token_seqlens).item()
            
            # 步骤3：从原始图像张量提取 Qwen3VL 视觉特征
            # - image_tensor_list：包含原始图像张量的列表
            # - image_grid_thw_list：图像的网格形状信息（用于 Qwen3VL 处理）
            # - get_image_features：调用 Qwen3VL processor，返回 patch 嵌入和多层视觉嵌入
            packed_vit_token_embed , deepstack_image_embeds = self.get_image_features(image_tensor_list, image_grid_thw_list)
            
            # 步骤4：连接来自多张图像的视觉 token
            # - packed_vit_token_embed：原本是列表 [img1_tokens, img2_tokens, ...]
            # - 连接后形状为 [总视觉token数, hidden_size]
            packed_vit_token_embed = torch.cat(packed_vit_token_embed, dim=0)
            
            # 步骤5：将视觉 token 嵌入放入 packed 序列的对应位置
            # - packed_vit_token_indexes：视觉 token 在 packed_sequence 中的目标位置
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed
            
            # 步骤6：保存多层视觉嵌入供后续使用
            # - deepstack_visual_embeds：包含多个层级的视觉特征（用于深层融合）
            deepstack_visual_embeds = deepstack_image_embeds
            
            # 步骤7：创建视觉位置掩码
            # - 掩码用于标记哪些位置是视觉 token，哪些是其他类型
            # - packed_sequence.shape[0] 是序列总长度
            visual_pos_masks = torch.zeros(
                packed_sequence.shape[0], 
                dtype=torch.bool, 
                device=packed_sequence.device
            )
            # 将视觉 token 对应的位置标记为 True
            visual_pos_masks[packed_vit_token_indexes] = True

        # ========== 块4：处理 BEV（Bird's Eye View）特征 ==========
        # BEV 是从鸟瞰角度获得的环境俯视图特征（来自 TransFuser 等编码器）
        # 这些特征用于空间推理和轨迹预测
        
        if self.config.visual_und:
            if bev_feature is not None and packed_bev_indexes is not None:
                # 步骤1：获取 BEV 特征并记录其形状（用于调试）
                x = bev_feature
                print("TransFuser 特征形状:", x.shape)
                
                # 步骤2：根据输入维度，灵活处理不同格式的 BEV 特征
                if x.dim() == 4:
                    # 情况1：标准 4D 张量 [B, C, H, W]（如 CNN 特征图）
                    B, C, H, W = x.shape
                    assert C == 1512, f"期望通道数为 1512，实际为 {C}"
                    assert H * W == 64, f"期望空间位置 8x8=64，实际为 {H}x{W}={H*W}"
                    
                    # 重塑过程：
                    # - flatten(2)：将 H,W 维度合并为单个空间维 -> [B, 1512, 64]
                    # - transpose(1,2)：交换通道和空间维 -> [B, 64, 1512]
                    x = x.flatten(2).transpose(1, 2)  # [B, 64, 1512]
                    
                    # 再塑形为 2D 张量便于线性投影
                    # [B, 64, 1512] -> [B*64, 1512]（共 B*64 个 BEV token）
                    bev_tok = x.reshape(-1, 1512)  # [B*64, 1512]
                    
                    # 步骤3a：通过投影层将 BEV 特征调整到模型隐藏维度
                    # bev_encoder_proj：线性层 1512 -> hidden_size(2560)
                    bev_tok = self.bev_encoder_proj(bev_tok)  # [B*64, 2560]

                elif x.dim() == 3:
                    # 情况2：已经是 3D 张量 [B, N, C]（N=64 个 token，C=1512 通道）
                    B, N, C = x.shape
                    assert N == 64 and C == 1512, f"期望形状 [B, 64, 1512]，实际为 [{B}, {N}, {C}]"
                    bev_tok = x.reshape(-1, 1512)  # [B*64, 1512]
                    bev_tok = self.bev_encoder_proj(bev_tok)  # [B*64, 2560]

                elif x.dim() == 2:
                    # 情况3：已经是 2D 张量 [N, C]（N 为总 token 数，不按批分离）
                    N, C = x.shape
                    assert C == 1512, f"期望最后一维为 1512，实际为 {C}"
                    bev_tok = self.bev_encoder_proj(x)  # [N, 2560]

                else:
                    raise ValueError(f"bev_feature 维度必须是 2/3/4，实际为 {x.dim()}")

                # 步骤4：将 BEV token 移到正确的设备和数据类型
                # - 确保与 packed_sequence 在同一个 GPU/CPU 上
                # - dtype 保持一致（通常是 float32 或 float16）
                bev_tok = bev_tok.to(device=packed_sequence.device, dtype=packed_sequence.dtype)
                packed_bev_indexes = packed_bev_indexes.to(device=packed_sequence.device)

                # 步骤5：验证 BEV token 和索引的合法性
                assert packed_bev_indexes.ndim == 1, "packed_bev_indexes 必须是 1D 张量"
                assert bev_tok.shape[0] == packed_bev_indexes.shape[0], \
                    f"长度不一致: bev_tok={bev_tok.shape[0]} vs packed_bev_indexes={packed_bev_indexes.shape[0]}"
                assert bev_tok.shape[1] == packed_sequence.shape[1], \
                    f"隐藏维不一致: bev_tok={bev_tok.shape[1]} vs hidden={packed_sequence.shape[1]}"

                # 步骤6：将 BEV token 嵌入放入 packed 序列的对应位置
                # - packed_bev_indexes 指示了各个 BEV token 应该位于 packed_sequence 中的位置
                packed_sequence[packed_bev_indexes] = bev_tok

        # ========== 块5：处理目标点（Target Points）特征 ==========
        # 目标点是规划阶段给定的中间目标点，引导模型生成合理的轨迹
        # 通常有 2 个目标点（近期和远期）
        
        if target_point_indexes is not None and v_target_point is not None:
            # v_target_point 的格式：(B, >=7) =
            # [速度, 目标点1_x, 目标点1_y, 目标点2_x, 目标点2_y, final_goal_x, final_goal_y]
            # legacy fast head 仍只消费前两个目标点；final_goal 通过 prompt 提供语义约束。
            assert v_target_point.dim() == 2 and v_target_point.size(1) >= 7, \
                f"v_target_point 形状非法: {tuple(v_target_point.shape)}"

            # target_point_indexes 应该包含偶数个元素（2 个点/样本）
            assert target_point_indexes.numel() % 2 == 0, \
                f"target_point_indexes 必须为偶数，当前为 {target_point_indexes.numel()}"

            # 验证批大小一致性
            B_v = v_target_point.size(0)  # 样本数
            B_tp = target_point_indexes.numel() // 2  # 根据索引计算的样本数

            assert B_v == B_tp, \
                f"批大小不一致: v_target_point B={B_v} vs target_point_indexes B={B_tp} (len={target_point_indexes.numel()})"

            # 步骤1：提取并重塑目标点坐标
            # - v_target_point[:, 1:5]：提取坐标部分（跳过速度），形状 (B, 4)
            # - reshape(B_v, 2, 2)：转换为 (B, 2点, 2坐标) 格式
            target_points = v_target_point[:, 1:5].reshape(B_v, 2, 2)  # (B, 2, 2)
            
            # 步骤2：通过 MLP 编码器将目标点坐标映射到嵌入空间
            # - self.target_point_encoder：通常是一个 MLP（2 -> 256 -> hidden_size）
            # - 输出形状：(B, 2, hidden_size)
            target_point_embed = self.target_point_encoder(target_points)  # (B, 2, hidden_size)
            
            # 步骤3：展平为 2D 张量便于插入 packed_sequence
            # - (B, 2, hidden_size) -> (B*2, hidden_size)
            # - B*2 个目标点 token，每个都是 hidden_size 维
            packed_target_point_embed = target_point_embed.reshape(-1, target_point_embed.size(-1))
            
            # 步骤4：将目标点 token 插入 packed 序列的指定位置
            # - target_point_indexes：指定了 B*2 个目标点 token 在 packed_sequence 中的位置
            packed_sequence[target_point_indexes] = packed_target_point_embed


        # ========== 块6：处理速度特征 ==========
        # 速度是自车当前移动的速度值，需要编码后作为速度预测任务的输入信息
        # 每个样本只有 1 个速度值
        
        if v_indexes is not None and v_target_point is not None:
            # 检查输入合法性
            assert v_target_point.dim() == 2 and v_target_point.size(1) >= 1, \
            f"v_target_point 形状非法: {tuple(v_target_point.shape)}"

            # 验证批大小一致性
            B_v = v_target_point.size(0)  # 样本数
            B_idx = v_indexes.numel()  # v_indexes 中的元素数

            assert B_v == B_idx, \
                f"批大小不一致: v_target_point B={B_v} vs v_indexes B={B_idx} (len={v_indexes.numel()})"

            # 步骤1：提取速度值
            # - v_target_point[:, 0:1]：提取第一列（速度），形状 (B, 1)
            # - 保持二维便于后续处理
            velocity = v_target_point[:, 0:1]  # (B, 1)
            
            # 步骤2：通过 MLP 编码器将速度值映射到嵌入空间
            # - self.velocity_encoder：MLP（1 -> 64 -> hidden_size）
            # - 输出形状：(B, hidden_size)
            velocity_embed = self.velocity_encoder(velocity)  # (B, hidden_size)

            # 步骤3：将速度嵌入放入 packed 序列的指定位置
            # - v_indexes：保存了 B 个速度 token 在 packed_sequence 中的位置
            packed_sequence[v_indexes] = velocity_embed


        # ========== 块7：处理推理（Reasoning）查询 token ==========
        # 推理 token 用于在生成分支中表示完整的推理状态
        # 这些可学习的 token 帮助模型进行多步推理
        
        if self.config.visual_gen:
            if packed_reasoning_token_indexes is not None:
                # 步骤1：计算推理 token 的总数和批大小
                # - packed_reasoning_token_indexes 包含了所有推理 token 的位置
                batch_query_count = packed_reasoning_token_indexes.shape[0]  # 总推理 token 数
                # 根据每批推理 token 数计算批大小
                batch_size = batch_query_count // self.reasoning_query_tokens
                
                # 步骤2：创建推理查询 token 的索引
                # - torch.arange(reasoning_query_tokens)：生成 [0, 1, ..., reasoning_query_tokens-1]
                # - 这些索引用于从预定义的可学习 embedding 中查询
                reasoning_token_indices = torch.arange(self.reasoning_query_tokens, device=packed_sequence.device)
                
                # 步骤3：从可学习的查询库中查询推理 token
                # - self.reasoning_queries：可学习的 embedding 查询表 [reasoning_query_tokens, hidden_size]
                # - 输出形状：[reasoning_query_tokens, hidden_size]
                reasoning_tokens = self.reasoning_queries(reasoning_token_indices)
                
                # 步骤4：为批处理复制推理 token
                # - unsqueeze(0)：[q_tokens, hidden] -> [1, q_tokens, hidden]
                # - repeat(batch_size, 1, 1)：复制到 [batch_size, q_tokens, hidden]
                packed_reasoning_tokens = reasoning_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
                
                # 步骤5：展平为 2D 张量
                # - [batch_size, q_tokens, hidden] -> [batch_size*q_tokens, hidden]
                packed_reasoning_tokens = packed_reasoning_tokens.view(-1, reasoning_tokens.shape[-1])
                
                # 步骤6：通过投影层调整推理 token 的表示
                # - self.reasoning_projector：线性投影或变换
                packed_reasoning_query_embed = self.reasoning_projector(packed_reasoning_tokens)
                
                # 步骤7：将推理 token 嵌入插入 packed 序列
                packed_sequence[packed_reasoning_token_indexes] = packed_reasoning_query_embed

        # ========== 块8：处理路由（Route）查询 token ==========
        # 路由 token 用于预测自车的移动路线（20 个时间步的轨迹）
        # 每个样本有 20 个路由 token（对应 10 秒的驾驶轨迹，每 0.5 秒一个）
        
        if route_loss_indexes is not None:
            # 步骤1：计算路由 token 的总数和批大小
            batch_query_count = route_loss_indexes.shape[0]  # 总路由 token 数
            batch_size = batch_query_count // 20  # 每样本 20 个 token
            
            # 步骤2：创建路由查询索引
            # - torch.arange(20)：生成 [0, 1, ..., 19]
            # - 用于从可学习的路由查询库中索取
            query_ids = torch.arange(20, device=packed_sequence.device)
            
            # 步骤3：从可学习的路由查询库中获取路由 token
            # - self.route_queries：可学习的 embedding 查询表 [20, hidden_size]
            # - 每个查询对应轨迹的一个时间步
            route_tokens = self.route_queries(query_ids)  # [20, hidden_size]
            
            # 步骤4：为批处理复制路由 token
            # - unsqueeze(0)：[20, hidden] -> [1, 20, hidden]
            # - repeat(batch_size, 1, 1)：复制到 [batch_size, 20, hidden]
            packed_route_tokens = route_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
            
            # 步骤5：展平便于处理
            # - [batch_size, 20, hidden] -> [batch_size*20, hidden]
            packed_route_tokens = packed_route_tokens.view(-1, route_tokens.shape[-1])
            
            # 步骤6：通过投影层处理路由 token
            # - self.route_projector：可能是线性层或包含激活函数的 MLP
            packed_route_query_embed = self.route_projector(packed_route_tokens)
            
            # 步骤7：将路由 token 嵌入放入 packed 序列
            # - route_loss_indexes：指定了 batch_size*20 个路由 token 的位置
            packed_sequence[route_loss_indexes] = packed_route_query_embed   
        # ========== 块9：处理轨迹（Trajectory/Waypoints）查询 token ==========
        # legacy AutoMoT 轨迹 token：内部 head 保留 6 个 waypoint、0.5s 间隔。
        # 这只描述原 fast head 的监督维度；当前 prompt/LeadMoT tp/ntp 语义仍对齐 2s 视野。
        # 每个样本有 6 个轨迹 token
        
        if traj_loss_indexes is not None:
            # 步骤1：计算轨迹 token 的总数和批大小
            batch_query_count = traj_loss_indexes.shape[0]  # 总轨迹 token 数
            batch_size = batch_query_count // 6  # 每样本 6 个 legacy 轨迹 token，间隔 0.5 秒
            
            # 步骤2：创建轨迹查询索引
            # - torch.arange(6)：生成 [0, 1, 2, 3, 4, 5]
            # - 对应 6 个时间步的预测点
            query_ids = torch.arange(6, device=packed_sequence.device)
            
            # 步骤3：从可学习的轨迹查询库中获取轨迹 token
            # - self.waypoint_queries：可学习的 embedding 查询表 [6, hidden_size]
            # - 每个查询对应一个未来时间点
            waypoint_tokens = self.waypoint_queries(query_ids)  # [6, hidden_size]
            
            # 步骤4：为批处理复制轨迹 token
            # - unsqueeze(0)：[6, hidden] -> [1, 6, hidden]
            # - repeat(batch_size, 1, 1)：复制到 [batch_size, 6, hidden]
            packed_waypoint_tokens = waypoint_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
            
            # 步骤5：展平便于处理
            # - [batch_size, 6, hidden] -> [batch_size*6, hidden]
            packed_waypoint_tokens = packed_waypoint_tokens.view(-1, waypoint_tokens.shape[-1])
            
            # 步骤6：通过投影层处理轨迹 token
            # - self.waypoint_projector：线性投影或 MLP
            packed_waypoint_query_embed = self.waypoint_projector(packed_waypoint_tokens)
            
            # 步骤7：将轨迹 token 嵌入放入 packed 序列
            # - traj_loss_indexes：指定了 batch_size*6 个轨迹 token 的位置
            packed_sequence[traj_loss_indexes] = packed_waypoint_query_embed         
        # ========== 块10：设置 MoT（Mixture-of-Thoughts）分支 ==========
        # MoT 模式将 token 分为两个分支：
        # 1. 理解分支（Understanding）：处理文本和视觉输入，进行环境感知
        # 2. 生成分支（Generation）：处理规划相关的 token（BEV、目标点、查询等），生成决策
        # 两个分支使用不同的 transformer 层或路由机制
        
        extra_inputs = {}  # 用于传递给语言模型的额外输入
        if self.use_mot:
            # 只有在视觉理解启用时才进行分支划分
            if packed_vit_token_indexes is not None:
                # 步骤1：构建理解分支的 token 索引
                # - 理解分支包含：文本 token 和视觉 token
                # - 这些 token 用于编码场景理解信息
                packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
                
                # 步骤2：构建生成分支的 token 索引
                # - 生成分支包含：BEV token、目标点、速度、推理 token、路由 token、轨迹 token
                # - 这些 token 用于生成驾驶决策
                packed_gen_token_indexes = torch.cat(
                    [packed_bev_indexes, target_point_indexes, v_indexes, 
                     packed_reasoning_token_indexes, route_loss_indexes, traj_loss_indexes], 
                    dim=0
                )
            
            # 步骤3：将两个分支的索引传递给语言模型
            # - 语言模型会根据这些索引选择激活不同的层或计算
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        # ========== 块11：通过语言模型进行多模态融合 ==========
        # 这是整个网络的核心计算，将所有 token 通过 Transformer 进行交互和特征融合
        
        # 步骤1：调用语言模型进行前向传播
        # - packed_sequence：包含所有类型 token 的 [sequence_length, hidden_size] 张量
        # - sample_lens：各样本的长度，用于重建原始批处理结构
        # - attention_mask：注意力掩码，控制哪些 token 可以相互交互
        # - deepstack_visual_embeds：多层视觉特征（如果有的话）
        # - visual_pos_masks：标记视觉 token 的位置
        # - packed_position_ids：RoPE 位置 id，支持 Qwen3VL 的 3D 位置编码
        # - extra_inputs：MoT 分支索引等额外参数
        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
            visual_pos_masks=visual_pos_masks,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )
        # 输出形状：[sequence_length, hidden_size]
        # 包含所有输入 token 经过 Transformer 编码后的最终隐藏状态

        # ========== 块12：计算文本语言建模损失（Cross-Entropy Loss）==========
        # 这是自编码（自回归文本预测）任务的损失
        # 模型需要根据上文预测下一个 token
        
        # mse = None
        # if self.config.visual_gen:
        #     packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
        #     target = noise - packed_latent_clean # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
        #     has_mse = packed_timesteps > 0
        #     mse = (packed_mse_preds - target[has_mse]) ** 2
        
        ce = None  # 初始化交叉熵损失为 None（可能不计算）
        
        if ce_loss_indexes is not None and len(ce_loss_indexes) > 0:
            # 步骤1：提取需要计算语言损失的 token 位置的隐藏状态
            # - ce_loss_indexes：标记了哪些 token 位置需要计算文本预测损失
            # - 这通常是文本 token 部分
            ce_hidden = last_hidden_state[ce_loss_indexes]  # [N_ce, hidden_size]
            
            # 步骤2：通过语言模型的输出头预测 token logits
            # - self.language_model.lm_head：线性层，hidden_size -> vocab_size
            # - 输出是 logits，表示每个 token 位置预测每个词汇的概率
            packed_ce_preds = self.language_model.lm_head(ce_hidden)  # [N_ce, vocab_size]
            
            # 步骤3：获取预测的 token ID（用于调试日志）
            predicted_token_ids = torch.argmax(packed_ce_preds, dim=-1)
            
            # 步骤4：尝试从 tokenizer 解码预测的文本，以便进行调试观察
            tokenizer = getattr(self, 'tokenizer', None)
            
            if tokenizer is None and hasattr(self.config, 'tokenizer'):
                tokenizer = self.config.tokenizer
            
            try:
                # 尝试解码预测的文本并保存日志
                lab = packed_label_ids
                # 过滤掉 padding token（id=-100）
                gt_ids = [t for t in lab if t != -100]
                N = len(gt_ids)
                # gt_text = tokenizer.decode(gt_ids, skip_special_tokens=False)
                pred = predicted_token_ids
                if torch.is_tensor(pred):
                    pred = pred.detach().cpu().tolist()
                pred = pred[:N] 
                # 将预测的 token ID 解码回文本
                predicted_text = tokenizer.decode(pred, skip_special_tokens=False)
                # 提取消息段（通常被 <|im_start|> 和 <|im_end|> 包围）
                segments = re.findall(r"<\|im_start\|>.*?<\|im_end\|>", predicted_text, flags=re.DOTALL)

                # 将解码结果写入调试日志
                with open("output_updatedv2_debug_speed.log", "a", encoding="utf-8") as f:
                    if segments:
                        for seg in segments:
                            f.write(seg.replace("\n", "\\n") + "\n")
                    else:
                        f.write("Predicted text: " + predicted_text.replace("\n", "\\n") + "\n")
            except Exception as e:
                print("Tokenizer 解码失败:", e)

            # 步骤5：计算交叉熵损失
            # - F.cross_entropy：逐位置比较预测 logits 与真实标签
            # - reduction="none"：保留每个位置的损失值，不求平均
            # - 这允许后续根据需要进行加权或其他处理
            ce = F.cross_entropy(
                packed_ce_preds, 
                packed_label_ids,
                reduction="none",
            )
        # ========== 块13：计算速度预测损失 ==========
        # 模型需要基于当前场景预测未来的车速（可能是 2 维：加速度和转向角速度）
        
        velocity_loss = None  # 初始化速度损失
        
        if traj_gt is not None and v_indexes is not None:
            # 步骤1：提取速度 token 对应的隐藏状态
            # - v_indexes：指定了各个样本的速度 token 位置
            # - 每个样本有 1 个速度 token
            velocity_feats = last_hidden_state[v_indexes]  # (B, hidden_size)
            
            # 步骤2：通过速度预测头解码速度值
            # - self.velocity_head：线性层或 MLP，hidden_size -> 2（如加速度和转向速度）
            # - 输出形状：(B, 2)
            velocity = self.velocity_head(velocity_feats)  # (B, 2)
            
            # 步骤3：计算平滑 L1 损失
            # - future_speeds_tensors：真实的目标速度
            # - Smooth L1 对异常值更鲁棒，是 L1 和 L2 的折中
            velocity_loss = F.smooth_l1_loss(velocity, future_speeds_tensors)  # 标量

        # ========== 块14：计算路由预测损失 ==========
        # 路由是一条长期轨迹（10 秒，20 个时间步），引导车辆朝向最终目标
        
        route_loss = None  # 初始化路由损失
        
        if route_gt is not None and route_loss_indexes is not None:
            # 步骤1：提取路由 token 对应的隐藏状态
            # - route_loss_indexes：所有样本的所有路由 token 的位置
            # - 形状：(B*20, hidden_size)，例如 (220, hidden_size)，其中 B=11 个样本
            route_feats = last_hidden_state[route_loss_indexes]
            
            # 步骤2：定义路由时间步数和计算批大小
            # - T_route = 20：每个样本有 20 个时间步的路由预测
            # - B：根据 token 总数反推样本数
            T_route = 20
            B = route_feats.shape[0] // T_route
            
            # 步骤3：通过路由头预测路由坐标
            # - self.route_head：MLP，hidden_size -> 2（x, y 坐标）
            # - 输出初始形状：(B*20, 2)，例如 (220, 2)
            pred_routes = self.route_head(route_feats)
            
            # 步骤4：重塑预测结果为批处理格式
            # - (B*20, 2) -> (B, 20, 2)：每个样本一个 20 步的路由
            pred_routes = pred_routes.view(B, T_route, 2)
            
            # 步骤5：验证并重塑真实标签
            # - route_gt 可能是 2D 或 3D 张量，需要统一为 (B, 20, 2) 的格式
            if route_gt.dim() == 2 and route_gt.size(-1) == 2:
                # 如果是 (N, 2) 的扁平格式，重塑为 (B, 20, 2)
                route_gt_valid = route_gt.view(B, T_route, 2)
            elif route_gt.dim() == 3:
                # 如果已是 (B_orig, T, 2) 的格式，取前 B 个样本
                route_gt_valid = route_gt[:B]
            else:
                raise ValueError(f"route_gt 形状不符合预期: {route_gt.shape}")
            
            # 步骤6：计算 L1 损失（逐坐标的平均绝对误差）
            # - L1 损失对异常值不敏感，适合轨迹预测
            route_loss = F.l1_loss(pred_routes, route_gt_valid)  # 标量

        # ========== 块15：计算轨迹预测损失及评价指标 ==========
        # legacy AutoMoT 短期精细轨迹 head：6 个 0.5s 时间步，用于细粒度运动控制。
        
        traj_loss = None  # 初始化轨迹损失
        l2_1s = l2_2s = l2_3s = l2_avg = None  # 初始化评价指标
        
        if traj_gt is not None and traj_loss_indexes is not None:
            # 步骤1：提取轨迹 token 对应的隐藏状态
            # - traj_loss_indexes：所有样本的所有轨迹 token 的位置处
            # - 形状：(B*6, hidden_size)，例如 (132, hidden_size)，其中 B=22 个样本
            traj_feats = last_hidden_state[traj_loss_indexes]
            
            # 步骤2：定义轨迹时间步数和计算批大小
            # - T_traj = 6：每个样本有 6 个 legacy 时间步（每 0.5 秒一个）
            # - B：根据 token 总数反推样本数
            T_traj = 6
            B = traj_feats.shape[0] // T_traj
            
            # 步骤3：通过轨迹头预测轨迹坐标
            # - self.waypoints_head：MLP，hidden_size -> 2（x, y 坐标）
            # - 输出初始形状：(B*6, 2)，例如 (132, 2)
            pred_trajs = self.waypoints_head(traj_feats)
            
            # 步骤4：重塑预测结果为批处理格式
            # - (B*6, 2) -> (B, 6, 2)：每个样本一条 6 步的轨迹
            pred_trajs = pred_trajs.view(B, T_traj, 2)
            
            # 步骤5：验证并重塑真实标签
            # - traj_gt 可能是 2D 或 3D 张量，需要统一为 (B, 6, 2) 的格式
            if traj_gt.dim() == 2 and traj_gt.size(-1) == 2:
                # 如果是 (N, 2) 的扁平格式，重塑为 (B, 6, 2)
                traj_gt_valid = traj_gt.view(B, T_traj, 2)
            elif traj_gt.dim() == 3:
                # 如果已是 (B_orig, T, 2) 的格式，取前 B 个样本
                traj_gt_valid = traj_gt[:B]
            else:
                raise ValueError(f"traj_gt 形状不符合预期: {traj_gt.shape}")
            
            # 步骤6：将真实轨迹转换为增量格式
            # - position_to_offset：将绝对坐标转为相对位移
            # - 第 0 帧保留原始位置，后续帧为差分
            traj_offset = self.position_to_offset(traj_gt_valid)  # (B, 6, 2)
            
            # 步骤7：计算轨迹预测的加权损失
            # - vad_traj_loss：考虑不同时间步重要性的加权 L1 损失
            # - 通常早期帧权重更高，因为控制越immediate 效应越大
            traj_loss = self.vad_traj_loss(pred_trajs, traj_offset)  # 标量
            
            # 步骤8：计算评价指标
            # 为了评估轨迹预测的长期精度，需要将增量转回绝对坐标
            # - cumsum(dim=1)：沿时间维求累积和，恢复为绝对轨迹
            pred_trajs_abs = pred_trajs.cumsum(dim=1)  # (B, 6, 2)：累积位移
            
            # 步骤9：计算不同时间段的误差指标
            # - ade_fde_loss：计算平均位移误差（Average Displacement Error）
            #   和最终位移误差（Final Displacement Error）
            # - l2_1s, l2_2s, l2_3s：分别是 1 秒、2 秒、3 秒时的 L2 误差
            l2_1s, l2_2s, l2_3s = self.ade_fde_loss(pred_trajs_abs, traj_gt_valid)
            
            # 步骤10：计算三个时段的平均误差
            # - 用于简化报告和早期停止判决
            l2_avg = (l2_1s + l2_2s + l2_3s) / 3
        # ========== 块16：汇总所有损失和指标 ==========
        # 将计算得到的所有损失项和评价指标整合为字典返回
        
        # 步骤1：创建包含所有可能的损失项的字典
        # - ce：交叉熵损失（文本预测）
        # - traj_loss：轨迹预测的加权损失
        # - route_loss：路由预测的 L1 损失
        # - velocity_loss：速度预测的平滑 L1 损失
        # - l2_1s, l2_2s, l2_3s：1、2、3 秒时的轨迹误差指标
        # - l2_avg：上述三个指标的平均值
        result = dict(
            ce=ce,  # 如果不计算，则为 None
            traj_loss=traj_loss,  # 如果不计算，则为 None
            route_loss=route_loss,  # 如果不计算，则为 None
            velocity_loss=velocity_loss,  # 如果不计算，则为 None
            l2_1s=l2_1s,  # 如果不计算，则为 None
            l2_2s=l2_2s,  # 如果不计算，则为 None
            l2_3s=l2_3s,  # 如果不计算，则为 None
            l2_avg=l2_avg,  # 如果不计算，则为 None
        )
        
        # 步骤2：过滤掉值为 None 的项
        # - 只返回实际计算的损失和指标，避免后续代码中的 None 错误
        return {k: v for k, v in result.items() if v is not None}
    
    def extract_all_eos_spans(self, valid_ids, eos_token_id):
        """按 eos_token_id 将 token 序列切分为若干段（每段均以 eos 结尾）。"""
        outputs = []
        temp = []
        for tid in valid_ids:
            temp.append(tid)
            if tid == eos_token_id:
                outputs.append(temp)
                temp = []
        return outputs

    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        """将多条文本提示词编码并打包为增量推理输入。"""
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            if '<|im_start|>' not in prompt:
                # prompt = "You are a smart autonomous agent and driving an self-driving car. Keep the necessary contents only in the answer. " + prompt + self.assistant_prompt
                prompt += self.assistant_prompt


            text_ids = tokenizer.encode(prompt)
            # text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        device = self.language_model.model.embed_tokens.weight.device
        
        packed_text_ids_tensor = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
        
        attention_mask = torch.ones(1, len(packed_text_ids_tensor), device=device, dtype=torch.long)
        
        text_position_ids_3d, rope_deltas = self.language_model.get_rope_index(
            input_ids=packed_text_ids_tensor.unsqueeze(0),
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=attention_mask
        )
        
        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int, device=device),
            "packed_text_ids": packed_text_ids_tensor,
            "packed_text_position_ids": text_position_ids_3d,
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
        }

        return generation_input, newlens, new_rope

    def _cached_prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        """缓存版本的提示词打包函数（含显式 BOS/EOS）。"""
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        device = self.language_model.model.embed_tokens.weight.device
        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int, device=device),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long, device=device),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long, device=device),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        """使用文本 token 执行一次增量前向，并回写 KV 缓存。"""
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_mot:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        定位序列中图像（及视频）占位符 token 的位置，生成布尔掩码，并校验占位符数量与视觉特征数量一致。

        背景说明：
            在多模态推理中，文本序列里的图像位置首先用特殊 token（<|image_pad|>，ID=151655）占位。
            后续需要精确找到这些占位符的位置，才能将视觉编码器输出的特征嵌入注入到正确位置。
            本函数就是生成这个"哪些位置是图像 token"的 True/False 布尔掩码。

        参数：
            input_ids:
                token ID 序列，形状 [seq_len] 或 [B, seq_len]。
                若提供，直接与 image_token_id 比较来找占位符。
                若为 None，则退而从 inputs_embeds 中反查（性能较低）。
            inputs_embeds:
                对应的嵌入向量，形状 [seq_len, hidden_size] 或 [B, seq_len, hidden_size]。
                掩码最终会被扩展到与此张量相同的形状，便于后续 masked_scatter 注入。
            image_features:
                视觉编码器输出的图像特征，形状 [N_tokens, hidden_size]。
                用于校验：其总元素数应等于 inputs_embeds 中被掩码选中部分的元素数。
            video_features:
                视频特征（当前已注释掉，预留接口，暂不使用）。

        返回：
            special_image_mask:
                布尔掩码，形状与 inputs_embeds 相同，True 的位置表示原本是图像占位符。
                上层会用 inputs_embeds[special_image_mask] = image_features.view(-1) 来注入视觉特征。
            None:
                视频掩码（当前版本已禁用），固定返回 None。
        """

        # ========== 步骤1：找到图像占位符的位置 ==========
        # 有两种路径，取决于调用方是否提供了 input_ids：
        #
        # 路径A（input_ids 为 None）：从嵌入向量反查
        #   - 将图像占位符的 token ID 转为嵌入向量，然后逐位置比较
        #   - all(-1)：要求 hidden_size 维度上每个元素都匹配，才算是该 token
        #   - 缺点：需要额外的 embedding 查找和全维度比较，开销较大
        #   - 使用场景：调用方没有保留原始 input_ids（例如只有 inputs_embeds）
        if input_ids is None:
            # 将 image_token_id（如 151655）转为嵌入向量，形状 [1, hidden_size]
            # 然后与 inputs_embeds 逐位置对比，找到嵌入值完全一致的位置
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            # inputs_embeds 某位置与图像嵌入的所有 hidden_size 维度都相等 -> 该位置是图像 token
            # all(-1)：对最后一维（hidden_size 维）做 AND，结果形状从 [..., hidden_size] 变为 [...]
            special_image_mask = special_image_mask.all(-1)

            # 同理处理视频占位符（当前代码保留但未实际使用，因视频分支已注释）
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            # 路径B（input_ids 存在）：直接用 token ID 比较（高效）
            # - self.config.image_token_id = 151655（<|image_pad|>）
            # - 结果形状与 input_ids 相同，True 表示该位置是图像占位符
            # - 例：input_ids = [1234, 151655, 151655, ..., 5678]
            #        mask     = [False,  True,   True, ..., False]
            special_image_mask = input_ids == self.config.image_token_id
            # 视频掩码当前已注释（视频分支未启用）
            # special_video_mask = input_ids == self.config.video_token_id

        # ========== 步骤2：统计图像占位符数量（用于后续校验） ==========
        # sum() 对布尔掩码求和 = True 的个数 = 序列中图像 token 的总数
        # 例：4 张图每张 128 个 token -> n_image_tokens = 512
        n_image_tokens = special_image_mask.sum()

        # ========== 步骤3：将掩码扩展到与 inputs_embeds 相同的形状 ==========
        # 目的：让掩码可以直接用于 inputs_embeds 的 masked_scatter 或索引操作
        #
        # 变换过程示例（以 2D 序列为例）：
        #   special_image_mask: [seq_len]            （每个位置一个 bool）
        #   unsqueeze(-1):      [seq_len, 1]          （在最后添加维度）
        #   expand_as(...):     [seq_len, hidden_size]（复制 hidden_size 列）
        #
        # 这样 inputs_embeds[special_image_mask] 就能选出所有图像位置的嵌入向量（形状 [N_tokens, hidden_size]）
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

        # ========== 步骤4：校验占位符数量与视觉特征数量一致 ==========
        # 只有在提供了 image_features 时才做校验
        # inputs_embeds[special_image_mask].numel()：被掩码选中的元素总数
        #   = n_image_tokens * hidden_size
        # image_features.numel()：视觉特征的总元素数
        #   = N_features * hidden_size
        # 若两者不等，说明占位符数量与特征数量不匹配，后续注入会出错，提前报错
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"图像特征与图像 token 数量不一致: tokens={n_image_tokens}, features={image_features.shape[0]}"
            )

        # ========== 视频掩码（已注释，预留接口）==========
        # 视频分支逻辑与图像相同，当前版本暂未启用
        # n_video_tokens = special_video_mask.sum()
        # special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        # if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
        #     raise ValueError(
        #         f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
        #     )

        # 返回图像掩码（已扩展到 hidden_size 维），视频掩码固定为 None
        return special_image_mask, None

    def get_input_embeddings(self):
        """返回底层语言模型的输入 embedding 层。"""
        return self.language_model.get_input_embeddings()

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        将视觉输入编码为可注入 LLM 的视觉 token embedding，并返回 deepstack 分支所需特征。

        参数：
            pixel_values:
                视觉处理器输出的图像张量（可能是多图打包后的形式）。
                在本项目常见场景下，它会被送入 Qwen3VL 视觉编码器得到 patch/token 级视觉特征。
            image_grid_thw:
                每张图对应的三维网格信息 [T, H, W]，形状通常为 [num_images, 3]。
                后续用于计算每张图应切分出的视觉 token 数量。

        返回：
            image_embeds:
                按“每张图”切分后的视觉 embedding 元组，供上层与文本占位符位置对齐替换。
            deepstack_image_embeds:
                deepstack 模块使用的视觉特征（保持视觉模型原始输出语义）。
        """

        # 第一步：将输入像素张量转换到视觉模型期望的 dtype（如 fp16/bf16）。
        # 目的：避免 dtype 不一致引发额外 cast 或算子报错，同时减少显存开销。
        # [4*512, 1536]  [4, 3]

        pixel_values = pixel_values.type(self.vision_model.dtype)

        # 第二步：送入视觉编码器。
        # image_embeds: 拼接后的视觉 token 表示（跨多张图拼在一起）。
        # deepstack_image_embeds: deepstack 路径使用的视觉特征。
        # 这里要注意两者的“组织方式”可能不一样：
        # 1. image_embeds 往往先表示为“所有图像 token 拼接后的总张量”；
        # 2. deepstack_image_embeds 往往已经按 deepstack 注入层组织好，因此可能是 list。
        #    例如 [(512, 2560), (512, 2560), (512, 2560)] 更像是
        #    “3 个 deepstack 层各自对应一份视觉特征”，而不是“3 张图”。
        # 其中 512 通常表示当前 batch 内所有视觉 token 总数，2560 是与 LLM 对齐后的 hidden size。

        image_embeds, deepstack_image_embeds = self.vision_model(pixel_values, grid_thw=image_grid_thw)
        # (512, 2560) [(512, 2560), (512, 2560), (512, 2560)]

        # 第三步：根据每张图的网格尺寸计算该图对应的 token 数。
        # image_grid_thw.prod(-1) = T*H*W。
        # 再除以 spatial_merge_size^2，是因为视觉模型在空间上做了 merge/downsample。
        # 得到 split_sizes 后，可用于把“拼接后的 image_embeds”按图像粒度切回去。
        # t=1, h=16, w=32, spatial_merge_size=2
        split_sizes = (image_grid_thw.prod(-1) // self.vision_model.spatial_merge_size**2).tolist()
        # Calculated split_sizes for images: [128, 128, 128, 128]

        # 第四步：按每张图的 token 数切分 embedding。
        # 这样上层可逐图处理，或与对应的 <|image_pad|> 占位符一一对齐。
        # 如果 image_embeds 此时还是一个总张量，就按 split_sizes 切成“每张图一份”；
        # 如果视觉模型已经直接返回 list/tuple，说明它已经完成了分组，这里就不重复切分。
        if not isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.split(image_embeds, split_sizes)
        # torch.Size([128, 2560], torch.Size([128, 2560]), torch.Size([128, 2560]), torch.Size([128, 2560]))

        # 最终返回时：
        # 1. image_embeds 是“按图像切分”的结果，供上层替换文本中的 image placeholder；
        # 2. deepstack_image_embeds 保持“按 deepstack 层组织”的结构，供语言模型内部逐层注入视觉信息。
        return image_embeds, deepstack_image_embeds

    def prepare_vit_images_qwen3vl(self, curr_kvlens, curr_rope, images, new_token_ids):
        """使用 Qwen3VL 官方处理器打包图像输入，生成可用于缓存更新的张量。

        参数：
            curr_kvlens: 当前历史 KV 缓存长度。
            curr_rope: 当前 RoPE 位置。
            images: PIL 图像列表。
            new_token_ids: 特殊 token id 字典。

        返回：
            generation_input: 前向所需张量字典。
            newlens: 更新后的 KV 长度。
            new_rope: 更新后的 RoPE 位置。
        """
            
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            im_start_ids = tokenizer.encode(self.user_prompt, add_special_tokens=False)
            packed_text_ids.extend(im_start_ids)
            packed_text_indexes.extend(range(_curr, _curr + len(im_start_ids)))
            packed_indexes.extend(range(curr, curr + len(im_start_ids)))
            curr += len(im_start_ids)
            _curr += len(im_start_ids)

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            # 使用官方 Qwen3VL 处理器处理图像
            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
            
            # 处理器输出对应的视觉 token（包含完整视觉序列）
            vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
            num_vision_tokens = len(vision_token_ids)
            
            # 处理器返回的 token 基本都是 <|image_pad|>，后续会替换为视觉 embedding
            packed_text_ids.extend(vision_token_ids.tolist())
            packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # 该段全部位置都需要视觉 embedding
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # 保存视觉输入
            packed_vit_tokens.append(pixel_values)
            packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
            vit_token_seqlens.append(num_vision_tokens)
            # 记录视觉 token 索引
            packed_vit_token_indexes.extend(vit_positions)

            # Update position counters
            curr += num_vision_tokens
            _curr += num_vision_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            total_seq_len = len(im_start_ids) + 1 + num_vision_tokens + 1  # +1 起始图像 token, +1 结束图像 token
            packed_seqlens.append(total_seq_len)
            newlens.append(curr_kvlen + total_seq_len)
            new_rope.append(curr_position_id + 1)

        device = self.language_model.model.embed_tokens.weight.device
        
        packed_text_ids_tensor = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
        packed_vit_token_indexes_tensor = torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device)
        packed_vit_position_ids_tensor = torch.stack(packed_vit_position_ids, dim=0).to(device)
        
        attention_mask = torch.ones(1, len(packed_text_ids_tensor), device=device, dtype=torch.long)
        
        position_ids_3d, rope_deltas = self.language_model.get_rope_index(
            input_ids=packed_text_ids_tensor.unsqueeze(0), 
            image_grid_thw=packed_vit_position_ids_tensor,
            video_grid_thw=None, 
            attention_mask=attention_mask
        )
        
        generation_input = {
            "packed_text_ids": packed_text_ids_tensor,
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0).to(device),  # 拼接所有图像像素张量
            "packed_vit_position_ids": packed_vit_position_ids_tensor,  # 堆叠 grid_thw
            "packed_vit_token_indexes": packed_vit_token_indexes_tensor,
            "packed_position_ids": position_ids_3d, 
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
        }

        return generation_input, newlens, new_rope

    def prepare_kv_cache(self, curr_kvlens, curr_rope, user_prompt, instruction_prompt, images, new_token_ids, tokenizer):
        """
        为多模态推理准备生成输入，包含3D位置ID和RoPE位置编码。
        该函数主要用途是将文本和视觉信息打包，计算正确的位置编码，用于模型的K-V缓存更新和推理。
        
        参数说明：
            curr_kvlens: 当前K-V缓存长度列表，记录历史输入的token数量
            curr_rope: 当前RoPE位置列表，记录位置编码的进度
            user_prompt: 用户输入提示词（通常是问题或指令）
            instruction_prompt: 指令提示词（给模型的额外指导）
            images: PIL图像列表，待处理的输入图像
            new_token_ids: 特殊token ID字典，包含图像开始/结束标记等
            tokenizer: 分词器，用于将文本转换为token ID
            
        返回值说明：
            generation_input: 模型前向传播所需的处理后张量字典
            newlens: 更新后的K-V缓存长度列表
            new_rope: 更新后的RoPE位置列表
        """
            
        # ==================== 初始化追踪列表 ====================
        # 这些列表用来记录各个模态token的位置和内容，后续会统一打包成张量
        packed_vit_token_indexes = list()  # 视觉token在最终序列中的索引位置
        vit_token_seqlens = list()  # 每张图像的视觉token序列长度
        packed_vit_tokens = list()  # 存储图像的像素值张量
        packed_vit_position_ids = list()  # 图像的网格位置信息(T,H,W)
        
        packed_text_ids = list()  # 所有文本token的ID序列,图像用151655代替
        packed_text_indexes = list()  # 文本token在最终序列的索引位置
        
        packed_seqlens = list()  # 每个样本的总序列长度
        packed_position_ids = list()  # 3D位置ID（将在后续计算）
        packed_indexes = list()  # K-V缓存中的索引位置
        packed_key_value_indexes = list()  # K-V缓存的访问索引
        
        # _curr: 在打包序列中的当前位置（用于索引映射）
        # curr: 在K-V缓存中的当前位置（用于缓存访问）
        _curr = curr = 0
        newlens = list()  # 用于存储更新后的缓存长度
        new_rope = list()  # 用于存储新的位置编码
        
        # 用于注意力掩码的参数
        split_lens = list()  # 每个部分（prompt、image、instruction）的长度
        attn_modes = list()  # 每个部分的注意力模式（causal或full）
        nested_attention_masks = list()  # 嵌套的注意力掩码张量列表

        # ==================== 处理旧的K-V缓存 ====================
        curr_position_id = 0
        # 如果有历史的K-V缓存，需要将其位置信息继承
        if curr_kvlens and curr_rope:
            # 获取当前RoPE位置（用于位置编码的连续性）
            curr_position_id = curr_rope[0]
            # 为每个历史缓存段建立索引映射
            for curr_kvlen in curr_kvlens:
                # 记录这段缓存的起止位置
                packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
                curr += curr_kvlen
        
        # ==================== 编码用户提示词 ====================
        # 将用户输入的文本转换为token ID序列
        user_prompt_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
        # len = 16

        # 将用户提示词token添加到打包序列
        packed_text_ids.extend(user_prompt_ids)
        
        # 记录这些token在打包序列中的位置（用于后续的embedding查找）
        packed_text_indexes.extend(range(_curr, _curr + len(user_prompt_ids)))
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        
        # 记录这些token在K-V缓存中的位置
        packed_indexes.extend(range(curr, curr + len(user_prompt_ids)))
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]        

        # 更新当前位置计数器
        curr += len(user_prompt_ids)
        _curr += len(user_prompt_ids)
        
        # 记录用户提示词段落的长度和注意力模式
        split_lens.append(len(user_prompt_ids))
        attn_modes.append('causal')  # 用户提示词使用因果注意力（自回归）

        # ==================== 处理图像序列 ====================
        # 逐张处理输入的图像，提取视觉特征并打包
        for image in images:
            # ----- 添加图像开始标记 -----
            # 在序列中插入特殊的"图像开始"token，标记图像内容的起始位置
            packed_text_ids.append(new_token_ids['start_of_image'])
            # size=512x256
            # Inserted start_of_image token with ID 151652 at position 16 in packed sequence and position 16 in K-V cache.
            # Inserted start_of_image token with ID 151652 at position 146 in packed sequence and position 146 in K-V cache.
            # Inserted start_of_image token with ID 151652 at position 276 in packed sequence and position 276 in K-V cache.
            # Inserted start_of_image token with ID 151652 at position 406 in packed sequence and position 406 in K-V cache.

            packed_text_indexes.append(_curr)  # 打包序列中的位置
            packed_indexes.append(curr)  # K-V缓存中的位置
            curr += 1
            _curr += 1

            # ----- 使用官方Qwen3VL处理器提取视觉特征 -----
            # 调用官方的视觉处理器（Qwen3VL processor），该处理器能够：
            # 1. 将图像调整到标准大小
            # 2. 提取图像特征
            # 3. 生成图像网格位置信息
            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            
            # 提取处理后的结果：
            pixel_values = processed["pixel_values"]  # 图像像素值张量，形状为([512, 1536])
            grid_thw = processed["image_grid_thw"]  # 图像网格的时间/高度/宽度信息，形状为 [1, 3]
            # 形状为 [1, 3], 里面是[ 1, 16, 32]
            # 获取处理器生成的vision token ID序列
            # 这些通常是 <|image_pad|> token，用来占位视觉特征
            vision_token_ids = processed["input_ids"][0]  # 移除batch维度，得到1D张量
            # Processed image into 128 vision tokens.
            # Vision token IDs: [151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655]
            
            num_vision_tokens = len(vision_token_ids)  # 该图像的token数量（通常与视觉特征的patch数有关）
            
            # ----- 将vision token添加到打包序列 -----
            # 所有返回的token都是 <|image_pad|> token，需要对应的视觉embedding
            packed_text_ids.extend(vision_token_ids.tolist())
            
            # 记录这些vision token在打包序列中的位置
            packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
            
            # 记录这些token在K-V缓存中的位置
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # 所有位置都需要视觉embedding（无需跳过特殊token）
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # ----- 存储视觉数据 -----
            # 保存像素值张量，用于后续的视觉编码器处理
            packed_vit_tokens.append(pixel_values)
            
            # 保存图像网格位置信息（移除batch维度），用于位置编码
            packed_vit_position_ids.append(grid_thw[0])  # 形状为 [3]，即 [T, H, W]
            
            # 记录该图像的vision token序列长度
            vit_token_seqlens.append(num_vision_tokens)
            
            # 所有position都需要vision embedding
            packed_vit_token_indexes.extend(vit_positions)
            
            # ----- 更新位置计数器 -----
            # 为下一个处理步骤准备新的位置偏移
            curr += num_vision_tokens
            _curr += num_vision_tokens

            # ----- 添加图像结束标记 -----
            # 在序列中插入特殊的"图像结束"token，标记图像内容的结束位置
            packed_text_ids.append(new_token_ids['end_of_image'])
            # Inserted end_of_image token with ID 151653 at position 145 in packed sequence and position 145 in K-V cache.
            # Inserted end_of_image token with ID 151653 at position 275 in packed sequence and position 275 in K-V cache.
            # Inserted end_of_image token with ID 151653 at position 405 in packed sequence and position 405 in K-V cache.
            # Inserted end_of_image token with ID 151653 at position 535 in packed sequence and position 535 in K-V cache.


            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
            # 16 + 4 * (128 + 2) = 536
            
            # 记录该图像段（包括开始/结束标记）的总长度和注意力模式
            # num_vision_tokens + 2: vision token数量 + start_of_image token + end_of_image token
            split_lens.append(num_vision_tokens + 2)
            attn_modes.append('full')  # 图像部分使用全注意力（可以看到所有token）

        


        # ==================== 编码指令提示词 ====================
        # 对指令进行清理和编码，遵循特定的格式化规则
        
        # 清理指令提示词：移除开头的特殊模态标记（如<image>、<lidar>等）

        # input term is: Your current and next target point is (8.560000, 0.054513), (17.120000, -0.015540),
        # your final destination is (50.000000, -0.120000), and your current velocity is 8.56 m/s.
        # Predict the driving actions ( now, +1s, +2s) and plan the trajectory for the next 2 seconds.
        instruction_prompt = self.clean_instruction_prompt(instruction_prompt)
        
        # 按照规定格式拼接指令和结束标记
        # <|im_end|> 是Qwen3VL的标准结束符，表示多模态输入的结束
        full_instruction_prompt = instruction_prompt + "<|im_end|>"
        
        # 使用分词器将完整指令编码为token ID序列
        instruction_prompt_ids = tokenizer.encode(full_instruction_prompt, add_special_tokens=False)
        # Instruction prompt length: 89

        # 将指令token添加到打包序列
        packed_text_ids.extend(instruction_prompt_ids)
        # Packed text IDs length: 625

        packed_text_indexes.extend(range(_curr, _curr + len(instruction_prompt_ids)))
        packed_indexes.extend(range(curr, curr + len(instruction_prompt_ids)))
        
        # 更新位置计数器
        curr += len(instruction_prompt_ids)
        _curr += len(instruction_prompt_ids)
        # 625
        
        # 记录指令段落的长度和注意力模式
        split_lens.append(len(instruction_prompt_ids))
        attn_modes.append('causal')  # 指令部分同样使用因果注意力

        # ==================== 计算总序列长度和缓存长度 ====================
        # 记录当前打包序列的总长度（将用于后续的张量构造）
        total_seq_len = _curr
        packed_seqlens.append(total_seq_len)
        
        # 计算新的K-V缓存长度：历史缓存长度 + 新增的token数量
        # 这个值用于后续的缓存管理和内存分配
        total_curr_kvlen = sum(curr_kvlens) if curr_kvlens else 0
        curr_position_start = curr_rope[0] if curr_rope else 0
        newlens.append(total_curr_kvlen + total_seq_len)
        # Total sequence length: 626, Total KV cache length: 0, Current position start: 0

        # ==================== 构造张量并计算位置编码 ====================
        # 获取模型所在的设备（CPU或GPU），以保证所有张量都在统一的设备上
        device = self.language_model.model.embed_tokens.weight.device
        
        # 将Python列表转换为PyTorch张量，便于模型处理
        packed_text_ids_tensor = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
        # torch.Size([626]) 
       
        packed_vit_token_indexes_tensor = torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device)
        # torch.Size([512])  128 * 4 = 512, tensor([17, 18, ..., 528, 529], device='cuda:0')

        # 将图像网格位置信息堆叠成一个张量
        # 输入形状：多个 [3] 张量
        # 输出形状：[num_images, 3]，其中3代表 [T, H, W]
        packed_vit_position_ids_tensor = torch.stack(packed_vit_position_ids, dim=0).to(device)
        # torch.Size([4, 3]) t=1, h=16, w=32 

        
        # 创建注意力掩码：所有位置都参与注意力计算
        # 形状为 [1, 总序列长度]，1表示batch size为1
        attention_mask = torch.ones(1, len(packed_text_ids_tensor), device=device, dtype=torch.long)

        # ----- 计算3D位置ID和RoPE -----
        # 调用语言模型的官方方法计算3D位置编码
        # 该方法遵循Qwen3VL的官方实现，支持多模态位置编码
        position_ids_3d, rope_deltas = self.language_model.get_rope_index(
            input_ids=packed_text_ids_tensor.unsqueeze(0),  # torch.Size([1,626]) token化的值
            image_grid_thw=packed_vit_position_ids_tensor,  # [4, 3]：图像网格信息
            video_grid_thw=None,  # 视频网格信息（当前不使用）
            attention_mask=attention_mask,  # [1, seq_len]：注意力掩码
            tokenizer=tokenizer  # 用于特殊token的识别
        )
        # torch.Size([3, 1, 626])
        
        
        # 获取新的RoPE位置：当前最大位置 + 1
        # 这用于下一帧的位置编码，确保位置编码的连续性
        new_rope.append(position_ids_3d[0].max().item() + 1)
        # [178]

        
        # ----- 构建嵌套注意力掩码 -----
        # 根据不同段落的长度和注意力模式构建分块的注意力掩码
        # 该掩码确保正确的注意力机制（因果或全）应用于各个段落
        nested_attention_masks.append(
            prepare_attention_mask_per_sample(split_lens, attn_modes).to(device)
        )
        
        # 将K-V缓存长度转换为张量
        key_values_lens_tensor = torch.tensor(curr_kvlens, dtype=torch.int, device=device)
        # Current KV lengths: [0], Key-Value lengths tensor: tensor([0], device='cuda:0', dtype=torch.int32)
        
        # ==================== 组织返回的字典 ====================
        # 将处理后的所有数据打包成一个字典，供模型的前向传播使用
        generation_input = {
            # ----- 文本相关 -----
            "packed_text_ids": packed_text_ids_tensor,  # torch.Size([626])， 所有文本token
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),  # 文本token的位置映射
            
            # ----- 注意力掩码 -----
            "nested_attention_masks": nested_attention_masks,  # 嵌套的注意力掩码列表
            
            # ----- 视觉特征相关 -----
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),  # 每张图的vision token数量
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0).to(device),  # 拼接所有图像的像素值
            "packed_vit_position_ids": packed_vit_position_ids_tensor,  # 图像网格位置信息
            "packed_vit_token_indexes": packed_vit_token_indexes_tensor,  # vision token的位置映射
            
            # ----- 位置编码相关 -----
            "packed_position_ids": position_ids_3d,  # 3D位置ID（用于RoPE）
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),  # 每个样本的序列长度
            
            # ----- K-V缓存相关 -----
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),  # 缓存中的索引映射
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),  # K-V索引
            "key_values_lens": torch.tensor(key_values_lens_tensor, dtype=torch.int, device=device),  # K-V长度
        }

        # generation_input
            # packed_text_ids torch.Size([626])， 所有文本token
            # packed_text_indexes torch.Size([626])，文本token的位置索引
            # nested_attention_masks list(len=1) 嵌套的注意力掩码列表
                # nested_attention_masks[0]: shape=(626, 626)
            # vit_token_seqlens [128, 128, 128, 128] 每张图的vision token数量
            # packed_vit_tokens torch.Size([4*512, 1536]) 拼接所有图像的像素值
            # packed_vit_position_ids torch.Size([4, 3]) THW 图像网格位置信息
            # packed_vit_token_indexes: torch.Size([512]) vision token的位置索引
            # packed_position_ids torch.Size([3, 1, 626]) 3D位置ID（用于RoPE）
                # [prepare_kv_cache] packed_position_ids 维度语义: dim0=RoPE轴(t/h/w), dim1=batch, dim2=packed序列token
                # [prepare_kv_cache] packed_position_ids.shape = (3, 1, 626)
                # [prepare_kv_cache] t-axis: shape=(1, 626), min=0, max=177
                # [prepare_kv_cache] t-axis sample0[:32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17]
                # [prepare_kv_cache] h-axis: shape=(1, 626), min=0, max=177
                # [prepare_kv_cache] h-axis sample0[:32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17]
                # [prepare_kv_cache] w-axis: shape=(1, 626), min=0, max=177
                # [prepare_kv_cache] w-axis sample0[:32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
            # packed_seqlens [626] # 每个样本的序列长度
            # packed_indexes torch.Size([626]) 缓存中的索引映射
            # packed_key_value_indexes torch.Size([0]) K-V索引（当前无历史
            # key_values_lens [0], K-V长度（当前为0）


        # 返回三个值：
        # 1. generation_input：供模型前向传播的张量字典
        # 2. newlens：更新后的K-V缓存长度（用于缓存管理）
        # 3. new_rope：更新后的RoPE位置（用于下一帧的位置编码）

        # [627] [179]
        return generation_input, newlens, new_rope


    def prepare_fast_kvcache(self, curr_kvlens, curr_rope, trans_feat, new_token_ids, tokenizer, reasoning_learnable_tokens, action_learnable_tokens, target_point_max_num_tokens, v_num_token, num_route_tokens, num_traj_tokens):
        """为快速 KV 缓存推理准备 BEV / 目标点 / 动作 token 的 packed 输入。

        这个方法的职责和 prepare_fast_generation 类似，但输入从“图像 token”换成了“BEV 特征 + 规划 token”。
        它主要完成以下几件事：
        1. 预留历史 KV cache 的索引位置；
        2. 把 BEV 特征、目标点、速度、推理 token、动作 token 按顺序放进 packed 序列；
        3. 构造分段注意力掩码，控制不同片段是 full 还是 causal；
        4. 生成与 Qwen3VL / 下游语言模型一致的 3D position_ids；
        5. 返回后续 forward_inference 需要的所有索引与长度信息。

        参数：
            curr_kvlens: 历史 KV cache 的长度列表。
            curr_rope: 历史 RoPE 位置列表。
            trans_feat: BEV / TransFuser 特征图，通常形状为 [B, C, H, W] 或等价形式。
            new_token_ids: 特殊 token id 字典。
            tokenizer: 分词器，用于生成必要的文本 token。
            reasoning_learnable_tokens: 推理阶段预留的可学习 token 数量。
            action_learnable_tokens: 动作阶段预留的可学习 token 数量。
            target_point_max_num_tokens: 目标点相关 token 上限。
            v_num_token: 速度相关 token 数量。
            num_route_tokens: 路径规划 token 数量。
            num_traj_tokens: 轨迹预测 token 数量。

        返回：
            generation_input: 供模型推理使用的张量字典。
            newlens: 更新后的 KV 长度。
            new_rope: 更新后的 RoPE 起点。
        """

        # 这些列表分别记录 BEV token、目标点 token、速度 token、推理 token 和动作 token 在 packed 序列中的位置。
        packed_bev_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes, packed_reasoning_token_indexes, packed_action_token_indexes = list(), list(), list()
        target_point_indexes, v_indexes = list(), list()
        _curr = curr = 0
        newlens, new_rope = list(), list()
        split_lens, attn_modes, nested_attention_masks = list(), list(), list()

        # 先把历史 KV cache 的索引段占出来，这样当前样本的新增 token 会顺着历史缓存继续往后排。
        if curr_kvlens and curr_rope:
            for curr_kvlen in curr_kvlens:
                packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
                curr += curr_kvlen

        # BEV 特征是这一段输入的核心视觉/空间表征，按 H*W 展平后会占据一整段 packed token 位置。
        # bev_feature (1, 1512, 8, 8)  
        bev_token_max_num_tokens = trans_feat.shape[-1] * trans_feat.shape[-2]
        # 64

        packed_bev_token_indexes.extend(range(_curr, _curr + bev_token_max_num_tokens))
        curr = curr + bev_token_max_num_tokens
        _curr = _curr + bev_token_max_num_tokens

        # BEV 段是完整空间上下文，因此使用 full attention。
        split_lens.append(bev_token_max_num_tokens)
        attn_modes.append('full')
        
        # 目标点 token：通常表示未来轨迹规划中的关键锚点或 waypoint。
        target_point_indexes.extend(range(_curr, _curr + 2))
        curr += target_point_max_num_tokens
        _curr += target_point_max_num_tokens  

        # 速度 token：一个样本一个速度槽位，用于把当前车速显式注入模型。
        v_indexes.append(_curr)
        curr += v_num_token
        _curr += v_num_token
        attn_modes.append("full")
        split_lens.append(target_point_max_num_tokens + v_num_token)        

        # 追加推理阶段的可学习 token，给模型留出额外的“思考槽位”。
        packed_reasoning_token_indexes.extend(range(_curr, _curr + reasoning_learnable_tokens))
        packed_indexes.extend(range(curr, curr + reasoning_learnable_tokens))
        curr += reasoning_learnable_tokens
        _curr += reasoning_learnable_tokens

        # 这一段的总长度更新，用于后续 KV 长度回写。
        total_seq_len = _curr 
        packed_seqlens.append(total_seq_len)
        split_lens.append(reasoning_learnable_tokens)
        attn_modes.append('full')

        # 再追加动作阶段的可学习 token，通常用于 route / traj 两类输出。
        packed_action_token_indexes.extend(range(_curr, _curr + num_route_tokens + num_traj_tokens))
        packed_indexes.extend(range(curr, curr + num_route_tokens + num_traj_tokens))

        # route token 段：用于中长程路径输出或规划分支。
        curr += num_route_tokens
        _curr += num_route_tokens
        split_lens.append(num_route_tokens)
        attn_modes.append('full')

        # traj token 段：用于未来轨迹/waypoint 预测分支。
        curr += num_traj_tokens
        _curr += num_traj_tokens
        split_lens.append(num_traj_tokens)
        attn_modes.append('full')

        # 后续所有张量统一放到语言模型 embedding 所在设备上，避免 device mismatch。
        device = self.language_model.model.embed_tokens.weight.device

        # 组合分段注意力掩码，使 BEV、目标点、推理 token 和动作 token 按预期模式工作。
        nested_attention_masks.append(
            prepare_attention_mask_per_sample(split_lens, attn_modes).to(device)
        )       

        # 这里没有显式文本 token，因此 packed_text_ids 为空张量，仅作为接口占位。
        packed_text_ids_tensor = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
        packed_bev_token_indexes_tensor = torch.tensor(packed_bev_token_indexes, dtype=torch.long, device=device)
        # packed_vit_position_ids_tensor = torch.stack(packed_vit_position_ids, dim=0).to(device)
        
        # attention_mask = torch.ones(1, len(packed_text_ids_tensor), device=device, dtype=torch.long)

        # 原本这里可以调用官方的 3D RoPE 计算函数；当前实现直接用连续位置替代，逻辑更轻量。
        # position_ids_3d, rope_deltas = self.language_model.get_rope_index_fast_thinking(
        #     input_ids=packed_text_ids_tensor.unsqueeze(0),
        #     image_grid_thw=packed_vit_position_ids_tensor,
        #     video_grid_thw=None, 
        #     attention_mask=attention_mask,
        #     num_learnable_tokens=reasoning_learnable_tokens + action_learnable_tokens + target_point_max_num_tokens + v_num_token,
        #     tokenizer=tokenizer
        # )
        # 这里把整段序列统一映射成 1D position_ids，再扩展成 Qwen3VL 期望的 3 轴格式。
        total_len = bev_token_max_num_tokens + reasoning_learnable_tokens + action_learnable_tokens + target_point_max_num_tokens + v_num_token
        # 101 = 64 + 8 + 26 + 2 + 1
        position_ids_1d = torch.arange(total_len, device=packed_bev_token_indexes_tensor.device).unsqueeze(0).expand(1, -1)
        position_ids_3d = position_ids_1d.unsqueeze(0).expand(3, -1, -1)
        rope_deltas = torch.zeros(1, 1, device=packed_bev_token_indexes_tensor.device)

        # 下一轮生成时的 RoPE 起点，等于当前序列最大位置再往后一个位置。
        new_rope.append(position_ids_3d[0].max().item() + 1)

        # 组织推理输入字典，把所有索引、位置和缓存长度统一返回给上层。
        generation_input = {
            "packed_text_ids": packed_text_ids_tensor,
            "nested_attention_masks": nested_attention_masks,
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "target_point_indexes": torch.tensor(target_point_indexes, dtype=torch.long, device=device),
            "v_indexes": torch.tensor(v_indexes, dtype=torch.long, device=device),
            "packed_reasoning_token_indexes": torch.tensor(packed_reasoning_token_indexes, dtype=torch.long, device=device),
            "packed_action_token_indexes": torch.tensor(packed_action_token_indexes, dtype=torch.long, device=device),
            "packed_bev_token_indexes": packed_bev_token_indexes_tensor,
            "packed_position_ids": position_ids_3d, 
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
        }


        # generation_input
            # packed_text_ids 为空张量，仅作为接口占位。
            # nested_attention_masks list(len=1) 嵌套的注意力掩码列表
                # nested_attention_masks[0]: shape=(bev_len+reasoning_learnable_tokens+action_learnable_tokens+target_point_max_num_tokens+v_num_token, bev_len+reasoning_learnable_tokens+action_learnable_tokens+target_point_max_num_tokens+v_num_token)
            # packed_text_indexes 为空张量，仅作为接口占位。
            # target_point_indexes shape=(2,) 目标点 token 在 packed 序列中的位置索引。  
            # v_indexes shape=(1,) 速度 token 在 packed 序列中的位置索引。      
            # packed_reasoning_token_indexes shape=(reasoning_learnable_tokens,) 推理阶段可学习 token 的位置索引。
            # packed_action_token_indexes shape=(num_route_tokens+num_traj_tokens,) 动作阶段可学习 token 的位置索引。
            # packed_bev_token_indexes shape=(bev_token_max_num_tokens,) BEV token 在 packed 序列中的位置索引。
            # packed_position_ids shape=(3, 1, total_len=101) 3D位置ID（用于RoPE）
                # sample=[[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]]
            # packed_seqlens shape=(1) values=[75] target_point_max_num_tokens + v_num_token + reasoning_learnable_tokens + action_learnable_tokens + bev_token_max_num_tokens
            # packed_indexes shape=(34,) values=[692, ..., 723] reasoning_learnable_tokens + action_learnable_tokens
            # packed_key_value_indexes shape=(625,) values=[0, 1, 2, ..., 623, 626] 历史 KV cache 的索引段



        # newlens len=0, shape=(0,), values=[]
        # new_rope len=1, shape=(1,), values=[101] 下一轮生成时的 RoPE 起点
        
        
        return generation_input, newlens, new_rope 

    def clean_instruction_prompt(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            return prompt
        return re.sub(
            r'^(?:\s*(?:<image>|<lidar>|<front>|<trans>))+',
            '',
            prompt
        ).lstrip()

    def prepare_generation(self, curr_kvlens, curr_rope, user_prompt, instruction_prompt, images, new_token_ids, tokenizer):
        """构建带 3D position_ids 的多模态生成输入（标准路径）。

        该函数会联合文本与视觉输入，按 Qwen3VL 规则计算位置编码。

        参数：
            curr_kvlens: 当前历史 KV 长度。
            curr_rope: 当前 RoPE 位置。
            images: PIL 图像列表。
            new_token_ids: 特殊 token id 字典。
            tokenizer: 分词器。

        返回：
            generation_input: 前向所需张量字典。
            newlens: 更新后的 KV 长度。
            new_rope: 更新后的 RoPE 位置。
        """
            
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()
        _curr = curr = 0
        newlens, new_rope = list(), list()
        
        if curr_kvlens and curr_rope:
            for curr_kvlen in curr_kvlens:
                packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
                curr += curr_kvlen
        
        user_prompt_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
        packed_text_ids.extend(user_prompt_ids)
        packed_text_indexes.extend(range(_curr, _curr + len(user_prompt_ids)))
        packed_indexes.extend(range(curr, curr + len(user_prompt_ids)))
        curr += len(user_prompt_ids)
        _curr += len(user_prompt_ids)
        
        for image in images:
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
            
            # Get the processed tokens from processor (includes complete vision token sequence)
            vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
            num_vision_tokens = len(vision_token_ids)
            
            # All tokens returned by processor are <|image_pad|> tokens needing vision embeddings
            packed_text_ids.extend(vision_token_ids.tolist())
            packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # All positions need vision embeddings (no special tokens to skip)
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # Store vision data
            packed_vit_tokens.append(pixel_values)
            packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
            vit_token_seqlens.append(num_vision_tokens)
            # All positions need vision embeddings
            packed_vit_token_indexes.extend(vit_positions)
            
            # Update position counters
            curr += num_vision_tokens
            _curr += num_vision_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

        full_instruction_prompt = instruction_prompt + self.assistant_prompt
        instruction_prompt_ids = tokenizer.encode(full_instruction_prompt, add_special_tokens=False)
        packed_text_ids.extend(instruction_prompt_ids)
        packed_text_indexes.extend(range(_curr, _curr + len(instruction_prompt_ids)))
        packed_indexes.extend(range(curr, curr + len(instruction_prompt_ids)))
        curr += len(instruction_prompt_ids)
        _curr += len(instruction_prompt_ids)

        total_seq_len = _curr
        packed_seqlens.append(total_seq_len)
        
        total_curr_kvlen = sum(curr_kvlens) if curr_kvlens else 0
        newlens.append(total_curr_kvlen + total_seq_len)

        device = self.language_model.model.embed_tokens.weight.device
        
        packed_text_ids_tensor = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
        packed_vit_token_indexes_tensor = torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device)
        packed_vit_position_ids_tensor = torch.stack(packed_vit_position_ids, dim=0).to(device)
        
        attention_mask = torch.ones(1, len(packed_text_ids_tensor), device=device, dtype=torch.long)
        
        position_ids_3d, rope_deltas = self.language_model.get_rope_index(
            input_ids=packed_text_ids_tensor.unsqueeze(0), 
            image_grid_thw=packed_vit_position_ids_tensor,  # [num_images, 3]
            video_grid_thw=None, 
            attention_mask=attention_mask,
            tokenizer=tokenizer 
        )
        new_rope.append(position_ids_3d[0].max().item() + 1)

        # Add indexes for the current sequence being processed
        current_sequence_length = len(packed_text_ids)
        # For the current sequence, use the starting KV cache position, not the accumulated curr
        kv_cache_start = sum(curr_kvlens) if curr_kvlens else 0
        packed_key_value_indexes.extend(range(kv_cache_start, kv_cache_start + current_sequence_length))
        
        generation_input = {
            "packed_text_ids": packed_text_ids_tensor,
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0).to(device),  # Concatenate pixel_values
            "packed_vit_position_ids": packed_vit_position_ids_tensor,  # Stack grid_thw
            "packed_vit_token_indexes": packed_vit_token_indexes_tensor,
            "packed_position_ids": position_ids_3d,
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(newlens, dtype=torch.int, device=device),
        }

        return generation_input, newlens, new_rope

    def prepare_fast_generation(self, curr_kvlens, curr_rope, user_prompt, instruction_prompt, images, new_token_ids, num_learnable_tokens, tokenizer):
        """为快速推理阶段准备多模态输入。

        这部分逻辑的核心目标是：
        1. 把历史 KV 缓存的索引先占位出来；
        2. 按“文本 + 视觉 + 可学习 token”的顺序组织当前序列；
        3. 为不同片段构造分段注意力掩码；
        4. 计算官方 Qwen3VL 需要的 3D position_ids 和 RoPE 位置；
        5. 返回后续 forward / cache update 所需的全部张量。

        参数：
            curr_kvlens: 当前历史 KV cache 的长度列表。
            curr_rope: 当前历史 RoPE 位置列表。
            user_prompt: 用户输入文本。
            instruction_prompt: 指令文本，会在图像后面拼接。
            images: PIL 图像列表，最后两张会被拆给生成分支。
            new_token_ids: 特殊 token 的 id 字典。
            tokenizer: 文本分词器。

        返回：
            generation_input: 供模型 forward 使用的张量字典。
            newlens: 更新后的 KV 长度。
            new_rope: 更新后的 RoPE 起点。

        约定：
            images 的结构通常是 [历史图像..., 当前视角图像, 当前视角副本, lidar 图像]，
            其中“当前视角副本”和“lidar 图像”会分配给第二个 transformer 分支。
        """

        # 这些列表分别记录：视觉 token 的位置、上下文中的文本位置、生成分支与理解分支的拆分位置等。
        packed_vit_token_indexes, packed_und_vit_token_indexes, packed_gen_vit_token_indexes = list(), list(), list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes, packed_und_text_indexes, packed_gen_text_indexes = list(), list(), list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()
        packed_learnable_token_indexes = list()
        split_lens, attn_modes, nested_attention_masks = list(), list(), list()
        _curr = curr = 0
        newlens, new_rope = list(), list()
        
        # 先把历史 KV cache 的位置索引占出来，保证当前样本追加时不会打乱缓存顺序。
        if curr_kvlens and curr_rope:
            curr_position_id = curr_rope[0]
            for curr_kvlen in curr_kvlens:
                packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
                curr += curr_kvlen
        
        # 先编码用户提示词，它通常是整段序列最前面的因果上下文。
        user_prompt_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
        packed_text_ids.extend(user_prompt_ids)
        packed_text_indexes.extend(range(_curr, _curr + len(user_prompt_ids)))
        packed_indexes.extend(range(curr, curr + len(user_prompt_ids)))
        curr += len(user_prompt_ids)
        _curr += len(user_prompt_ids)
        # 用户提示词使用因果注意力，表示后面的内容只能看见它，不能反向看未来位置。
        split_lens.append(len(user_prompt_ids))
        attn_modes.append('causal')

        # 这里把输入图像拆成两个部分：
        # und 分支负责前面的视觉理解，gen 分支负责最后两张图像对应的生成相关视觉信息。
        images_und = images[:-2]  # All images except the last two for und transformer
        images_gen = images[-2:]  # The last two images for gen transformer
        
        # 2) 处理理解分支（und）的图像。
        # 每张图都会插入 start_of_image -> vision tokens -> end_of_image 的完整结构。
        for image in images_und:
            # 图像开始标记：告诉语言模型后续要进入视觉片段。
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            # 使用官方 Qwen3VL processor 把 PIL 图像转换成像素张量和网格信息。
            # 这里返回的 input_ids 一般是一串 <|image_pad|> 占位 token，真正的视觉语义要由视觉编码器补齐。
            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
            
            # 提取这一张图对应的视觉 token 序列长度。
            vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
            num_vision_tokens = len(vision_token_ids)
            
            # 将这些占位 token 追加到文本序列里，后续会在 packed_sequence 中被视觉 embedding 替换。
            packed_text_ids.extend(vision_token_ids.tolist())
            # 这里不把视觉 token 放进 packed_text_indexes，是因为当前实现里视觉 token 由专门的索引集合管理。
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # 这一段里所有位置都需要视觉 embedding，因此直接把整段位置都记录下来。
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # 保存图像像素值和网格尺寸，供后续 get_image_features / RoPE 计算使用。
            packed_vit_tokens.append(pixel_values)
            packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
            # 记录视觉 token 序列长度，便于按图像切分和批处理。
            vit_token_seqlens.append(num_vision_tokens)
            # und 分支的视觉 token 索引，以及总视觉 token 索引都要记录下来。
            packed_und_vit_token_indexes.extend(vit_positions)
            packed_vit_token_indexes.extend(vit_positions)
            
            # 更新当前位置游标：视觉 token 也会占用 packed 序列和 KV cache 的位置。
            curr += num_vision_tokens
            _curr += num_vision_tokens

            # 图像结束标记：结束这段视觉上下文。
            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
            # 图像段使用全注意力，表示这一段内部 token 可以彼此自由交互。
            split_lens.append(num_vision_tokens+2)  # +2 for start and end tokens
            attn_modes.append('full')

        # 3) 在理解分支图像后面拼接指令提示词。
        # 这里先清理掉开头可能残留的模态占位符，再补上结束符。
        instruction_prompt = self.clean_instruction_prompt(instruction_prompt)
        full_instruction_prompt = instruction_prompt + "<|im_end|>"
        instruction_prompt_ids = tokenizer.encode(full_instruction_prompt, add_special_tokens=False)
        packed_text_ids.extend(instruction_prompt_ids)
        packed_text_indexes.extend(range(_curr, _curr + len(instruction_prompt_ids)))
        packed_indexes.extend(range(curr, curr + len(instruction_prompt_ids)))
        curr += len(instruction_prompt_ids)
        _curr += len(instruction_prompt_ids)
        # 指令段这里也使用因果注意力，符合标准自回归生成的约束。
        split_lens.append(len(instruction_prompt_ids))  # +2 for start and end tokens
        attn_modes.append('causal')
        
        # 4) 处理生成分支的图像。
        # 这一段的索引会单独进入 packed_gen_* 集合，用于区分生成相关视觉 token。
        for image in images_gen:
            # 生成分支图像的起始标记。
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_gen_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            # 同样通过官方 processor 预处理图像，并取出对应的视觉占位 token。
            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
            
            # 这一张生成分支图像对应的视觉 token 数量。
            vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
            num_vision_tokens = len(vision_token_ids)
            
            # 追加视觉占位 token 到文本序列，并只记录生成分支相关的视觉索引。
            packed_text_ids.extend(vision_token_ids.tolist())
            # packed_text_indexes 在这里仍然不直接扩展视觉占位段，实际由后面的索引集合来控制。
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # 这一段的所有位置都需要视觉 embedding，因此完整保留位置索引。
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # 保存该图像的视觉输入与网格信息，后面统一堆叠后送入位置编码和视觉编码器。
            packed_vit_tokens.append(pixel_values)
            packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
            vit_token_seqlens.append(num_vision_tokens)
            # 生成分支和总视觉 token 集合都记录该图像对应的位置。
            packed_gen_vit_token_indexes.extend(vit_positions)
            packed_vit_token_indexes.extend(vit_positions)
            
            # 更新位置游标，为 end_of_image 和后续 learnable tokens 留出空间。
            curr += num_vision_tokens
            _curr += num_vision_tokens

            # 图像结束标记：结束生成分支的图像片段。
            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_gen_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
            # 生成分支图像段使用全注意力，便于图像内部的全局交互。
            split_lens.append(num_vision_tokens+2)  # +2 for start and end tokens
            attn_modes.append('full')
        
        # 5) 在序列末尾追加可学习 token。
        # 这些 token 常用于额外的推理/规划/动作表征，是模型内部可训练的“槽位”。
        packed_learnable_token_indexes.extend(range(_curr, _curr + num_learnable_tokens))
        packed_indexes.extend(range(curr, curr + num_learnable_tokens))
        curr += num_learnable_tokens
        _curr += num_learnable_tokens
        split_lens.append(num_learnable_tokens)
        attn_modes.append('full')
        
        # 6) 汇总理解分支文本索引，排除生成分支图像相关 token。
        und_idxs = [
            idx for idx in packed_text_indexes
            if idx not in packed_gen_text_indexes
        ]
        packed_und_text_indexes = und_idxs

        # 当前样本的总序列长度，以及更新后的 KV cache 长度。
        total_seq_len = _curr
        packed_seqlens.append(total_seq_len)
        total_curr_kvlen = sum(curr_kvlens) if curr_kvlens else 0
        curr_position_start = curr_rope[0] if curr_rope else 0
        newlens.append(total_curr_kvlen + total_seq_len)

        # 所有后续张量都移动到语言模型 embedding 所在设备上，避免 device mismatch。
        device = self.language_model.model.embed_tokens.weight.device
        
        # 将 Python 列表转换成张量，供后续 embedding 替换和位置编码使用。
        packed_text_ids_tensor = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
        packed_vit_token_indexes_tensor = torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device)
        packed_und_vit_token_indexes_tensor = torch.tensor(packed_und_vit_token_indexes, dtype=torch.long, device=device)
        packed_gen_vit_token_indexes_tensor = torch.tensor(packed_gen_vit_token_indexes, dtype=torch.long, device=device)
        packed_vit_position_ids_tensor = torch.stack(packed_vit_position_ids, dim=0).to(device)
        
        # attention_mask 全 1 表示当前序列所有位置都参与位置编码与后续注意力计算。
        attention_mask = torch.ones_like(packed_text_ids_tensor).unsqueeze(0)

        # 计算官方 Qwen3VL 风格的 3D position_ids。
        # 这里会把文本 token、图像 token 和可学习 token 的位置关系统一编码出来。
        position_ids_3d, rope_deltas = self.language_model.get_rope_index_fast_thinking(
            input_ids=packed_text_ids_tensor.unsqueeze(0),  # [1, seq_len]
            image_grid_thw=packed_vit_position_ids_tensor,  # [num_images, 3]
            video_grid_thw=None,  
            attention_mask=attention_mask,
            num_learnable_tokens=num_learnable_tokens,
            tokenizer=tokenizer 
        )
        # 新一轮生成的 RoPE 起点，等于当前位置编码最大值加 1。
        new_rope.append(position_ids_3d[0].max().item() + 1)

        # 当前序列所占用的 KV cache 区间，按历史缓存结束位置向后连续分配。
        current_sequence_length = len(packed_text_ids)
        kv_cache_start = sum(curr_kvlens) if curr_kvlens else 0
        packed_key_value_indexes.extend(range(kv_cache_start, kv_cache_start + current_sequence_length))

        # 构造分段注意力掩码，使不同片段按 causal / full 的方式工作。
        nested_attention_masks.append(
            prepare_attention_mask_per_sample(split_lens, attn_modes).to(device)
        )

        # 快速生成模式下，把 <|image_pad|> token 从纯文本序列里剔除，避免它们进入文本侧的解码流程。
        packed_text_ids_tensor = packed_text_ids_tensor[packed_text_ids_tensor != 151655]  # remove all <|image_pad|> tokens for fast generation

        # 汇总生成输入字典，供后续 forward / inference 阶段直接使用。
        generation_input = {
            "packed_text_ids": packed_text_ids_tensor.to(device=device, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "packed_gen_text_indexes": torch.tensor(packed_gen_text_indexes, dtype=torch.long, device=device),
            "packed_und_text_indexes": torch.tensor(packed_und_text_indexes, dtype=torch.long, device=device),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),
            "nested_attention_masks": nested_attention_masks,
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0).to(device),  # Concatenate pixel_values
            "packed_vit_position_ids": packed_vit_position_ids_tensor,  # Stack grid_thw
            "packed_vit_token_indexes": packed_vit_token_indexes_tensor,
            "packed_und_vit_token_indexes": packed_und_vit_token_indexes_tensor,
            "packed_gen_vit_token_indexes": packed_gen_vit_token_indexes_tensor,
            "packed_learnable_token_indexes": torch.tensor(packed_learnable_token_indexes, dtype=torch.long, device=device),
            "packed_position_ids": position_ids_3d,  
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(newlens, dtype=torch.int, device=device),
            "curr": int(curr),
        }

        return generation_input, newlens, new_rope

    def _cached_prepare_vit_images_qwen3vl(self, curr_kvlens, curr_rope, images, new_token_ids):
        """缓存版本的视觉输入打包（使用 Qwen3VL 官方处理器）。

        参数：
            curr_kvlens: 当前历史 KV 长度。
            curr_rope: 当前 RoPE 位置。
            images: PIL 图像列表。
            new_token_ids: 特殊 token id 字典。

        返回：
            generation_input: 前向所需张量字典。
            newlens: 更新后的 KV 长度。
            new_rope: 更新后的 RoPE 位置。
        """
            
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            # 使用官方 Qwen3VL 处理器处理图像
            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
            
            # 处理器输出的视觉 token（含完整视觉序列）
            vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
            num_vision_tokens = len(vision_token_ids)
            
            # 处理器返回的 token 基本都是 <|image_pad|>，后续需要视觉 embedding
            packed_text_ids.extend(vision_token_ids.tolist())
            packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # 该段所有位置都需要视觉 embedding
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # 更新位置游标
            curr += num_vision_tokens
            _curr += num_vision_tokens
            
            # 保存视觉数据
            packed_vit_tokens.append(pixel_values)
            packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
            vit_token_seqlens.append(num_vision_tokens)
            # 记录视觉 token 位置
            packed_vit_token_indexes.extend(vit_positions)

            # 记录位置与长度
            packed_position_ids.extend([curr_position_id] * num_vision_tokens)
            packed_seqlens.append(num_vision_tokens)
            newlens.append(curr_kvlen + num_vision_tokens)
            new_rope.append(curr_position_id + 1)

        device = self.language_model.model.embed_tokens.weight.device
        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long, device=device),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0).to(device),  # 拼接像素张量
            "packed_vit_position_ids": torch.stack(packed_vit_position_ids, dim=0).to(device),  # 堆叠 grid_thw
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long, device=device),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        """使用视觉 token 执行一次增量前向并更新 KV 缓存。"""
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        # packed_sequence[packed_text_indexes] = packed_text_embedding

        # Qwen3VL 视觉处理：packed_vit_tokens 为像素值，packed_vit_position_ids 为 grid_thw
        image_embeds, deepstack_image_embeds = self.get_image_features(packed_vit_tokens, packed_vit_position_ids)
        image_embeds = torch.cat(image_embeds, dim=0).to(packed_text_embedding.device, packed_text_embedding.dtype)
        image_mask, _ = self.get_placeholder_mask(
            packed_text_ids, inputs_embeds=packed_text_embedding, image_features=image_embeds
        )
        packed_text_embedding = packed_text_embedding.masked_scatter(image_mask, image_embeds)
        packed_sequence[packed_text_indexes] = packed_text_embedding

        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds

        # packed_vit_token_embed, packed_deepstack_image_embed = self.vision_model(
        #     hidden_states=packed_vit_tokens,
        #     grid_thw=packed_vit_position_ids,
        # )
        # visual_pos_masks = None
        # deepstack_visual_embeds = None
        
        # Qwen3VL 路径不需要额外 connector 或手写位置编码
        # 这里保留 dtype 对齐相关说明
        # if packed_vit_token_embed.dtype != packed_sequence.dtype:
        #     packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        # packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_mot:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids, 
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            # args for deepstack
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    @torch.no_grad
    def forward_cache_update_generation(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        nested_attention_masks: list = None,
    ):
        # 该函数的职责：
        # 1) 将 packed 文本 token 先映射到 embedding；
        # 2) 计算视觉特征，并把 <|image_pad|> 对应位置替换为真实视觉 embedding；
        # 3) 按 packed 索引组装成模型 forward_inference 需要的 packed_query_sequence；
        # 4) 结合 3D RoPE 位置、注意力掩码和历史 KV，更新并返回新的 KV cache。
        #
        # 关键输入张量语义（典型形状示例）：
        # packed_text_ids: [626]，包含文本及视觉占位符 token id。
        # packed_text_indexes: [626]，每个 token 在 packed_sequence 中的目标下标。
        # packed_vit_tokens: [4*512, 1536]，视觉编码器输入像素特征（已打包）。
        # packed_vit_position_ids: [4, 3]，每张图的 THW 网格信息。
        # packed_position_ids: [3, 1, 626]，Qwen3VL 的 3D RoPE 位置（t/h/w 三轴）。
        # packed_seqlens: [1]，每个样本在 packed 后的总序列长度。
        # packed_indexes: [626]，当前 query token 在“KV 视角”下的连续索引。
        # packed_key_value_indexes/key_values_lens：历史 KV 读取与写回的索引和长度信息。
        # nested_attention_masks: 分块注意力掩码（控制不同段是 causal 还是 full）。
        """使用 3D 位置编码执行生成分支缓存更新（Qwen3VL 路径）。"""
        # 第一步：把 token id 映射到语言模型 embedding 空间。
        # 输出形状通常为 [N_text, hidden_size]。
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        # torch.Size([626, 2560])

        # 第二步：为本次 query 构造一个“完整 packed 序列容器”。
        # 容器长度 = 所有样本 packed 长度之和，初始全 0，后续按索引回填。
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        # torch.Size([626, 2560])
        
        # 第三步：通过视觉分支提取图像 token 对应的 embedding。
        # get_image_features 返回：
        # image_embeds: 用于替换文本中的视觉占位符；
        # deepstack_image_embeds: 给 deepstack 模块使用的视觉特征。

        # [4*512, 1536]  [4, 3]
        # t=1, h=16, w=32 
        image_embeds, deepstack_image_embeds = self.get_image_features(packed_vit_tokens, packed_vit_position_ids)
        # [(128, 2560), (128, 2560), (128, 2560), (128, 2560)]
        #  [(512, 2560), (512, 2560), (512, 2560)]
        # 1. image_embeds 往往先表示为“所有图像 token 拼接后的总张量”；
        # 2. deepstack_image_embeds 往往已经按 deepstack 注入层组织好，因此可能是 list。
        #    例如 [(512, 2560), (512, 2560), (512, 2560)] 更像是
        #    “3 个 deepstack 层各自对应一份视觉特征”，而不是“3 张图”。
        # 其中 512 通常表示当前 batch 内所有视觉 token 总数，2560 是与 LLM 对齐后的 hidden size。
        

        # 将多张图的 image_embeds 拼接为 [N_vit_token, hidden_size]，并与文本 embedding 对齐 device/dtype。
        image_embeds = torch.cat(image_embeds, dim=0).to(packed_text_embedding.device, packed_text_embedding.dtype)
        # (4*128, 2560)

        # 第四步：定位文本序列里需要被视觉特征替换的位置（即 image placeholder 位置）。
        # image_mask 是布尔掩码，True 的位置将被 image_embeds 按序填充。
        image_mask, _ = self.get_placeholder_mask(
            packed_text_ids, inputs_embeds=packed_text_embedding, image_features=image_embeds
        )

        # 第五步：把文本 embedding 中的视觉占位符替换成真实视觉 embedding。
        # masked_scatter 会按 mask 的 True 顺序从 image_embeds 取值写入。
        packed_text_embedding = packed_text_embedding.masked_scatter(image_mask, image_embeds)
        # torch.Size([626, 2560])

        # 第六步：将替换后的 embedding 回填到 packed_sequence 的指定位置。
        # packed_text_indexes 建立了“局部文本序列 -> 全局 packed 序列”的映射。
        packed_sequence[packed_text_indexes] = packed_text_embedding
        # torch.Size([626, 2560])
        
        # 第七步：整理 deepstack 所需的附加输入。
        # image_mask 原形状末维通常为 1，这里压缩后得到 [N_text] 布尔向量。
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
        
        # 第八步：根据模型开关决定是否给 forward_inference 传入 mode="und"。
        # 该参数用于启用特定分支逻辑（例如多任务/双分支推理）。
        extra_inputs = {}
        if self.use_mot:
            extra_inputs = {"mode": "und"}
            
        # 第九步：执行一次带缓存更新的推理。
        # 关键点：
        # 1) packed_query_sequence + packed_query_indexes/past_key_values 共同定义读写位置；
        # 2) packed_query_position_ids 提供 3D RoPE 位置；
        # 3) attention_mask 使用外部构造的 nested_attention_masks，支持分段注意力模式；
        # 4) update_past_key_values=True 表示本次 query 会写回 KV cache。
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids, 
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            attention_mask = nested_attention_masks,
            is_causal=False,
            # args for deepstack
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **extra_inputs,
        )

        # 第十步：取出更新后的 KV cache，供下一轮增量解码复用。
        past_key_values = output.past_key_values

        # 同时返回 packed_position_ids，便于上层在后续步骤继续使用/调试位置编码。
        return past_key_values, packed_position_ids

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        """准备文本增量生成的起始 token 与对应索引。"""
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            gen_start_id = tokenizer.encode('\n', add_special_tokens=False)
            packed_start_tokens.extend(gen_start_id)
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        device = self.language_model.model.embed_tokens.weight.device
        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long, device=device),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
        }

        return generation_input
    def prepare_fast_generation_senna(self, curr_kvlens, curr_rope, user_prompt, instruction_prompt, images, new_token_ids, num_learnable_tokens, action_learnable_tokens, tokenizer):
        """Senna 快速生成路径：构建带 3D 位置编码的文本+视觉打包输入。

        参数：
            curr_kvlens: 当前历史 KV 长度。
            curr_rope: 当前 RoPE 位置。
            images: 图像列表。
            new_token_ids: 特殊 token id 字典。
            tokenizer: 分词器。

        返回：
            generation_input: 前向所需张量字典。
            newlens: 更新后的 KV 长度。
            new_rope: 更新后的 RoPE 位置。

        说明：
            常见输入形式为 [image, image, image, image, image(当前视角), image(当前视角副本), image(lidar)]。
            其中当前视角副本与 lidar 图像会用于第二个 transformer 分支。
        """

        packed_vit_token_indexes, packed_und_vit_token_indexes, packed_gen_vit_token_indexes = list(), list(), list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes, packed_und_text_indexes, packed_gen_text_indexes = list(), list(), list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()
        packed_learnable_token_indexes, packed_action_token_indexes = list(), list()
        split_lens, attn_modes, nested_attention_masks = list(), list(), list()
        _curr = curr = 0
        newlens, new_rope = list(), list()
        
        curr_position_id = 0
        # print(f"DEBUG: curr_kvlens = {curr_kvlens}")
        # print(f"DEBUG: curr_rope = {curr_rope}")
        if curr_kvlens and curr_rope:
            curr_position_id = curr_rope[0]
            for curr_kvlen in curr_kvlens:
                packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
                curr += curr_kvlen
        # print(f"DEBUG: initial packed_key_value_indexes = {packed_key_value_indexes}")
        # print(f"DEBUG: curr after existing KV = {curr}")
        
        user_prompt_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
        packed_text_ids.extend(user_prompt_ids)
        packed_text_indexes.extend(range(_curr, _curr + len(user_prompt_ids)))
        packed_indexes.extend(range(curr, curr + len(user_prompt_ids)))
        curr += len(user_prompt_ids)
        _curr += len(user_prompt_ids)
        split_lens.append(len(user_prompt_ids))
        attn_modes.append('causal')
        images_und = images[:-2]  # All images except the last two for und transformer
        images_gen = images[-2:]  # The last two images for gen transformer
        # 2. process und images
        for image in images_und:
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            # Use official Qwen3VL processor to process image
            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
            
            # Get the processed tokens from processor (includes complete vision token sequence)
            vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
            num_vision_tokens = len(vision_token_ids)
            
            # All tokens returned by processor are <|image_pad|> tokens needing vision embeddings
            packed_text_ids.extend(vision_token_ids.tolist())
            #packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # All positions need vision embeddings (no special tokens to skip)
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # Store vision data
            packed_vit_tokens.append(pixel_values)
            packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
            vit_token_seqlens.append(num_vision_tokens)
            # All positions need vision embeddings
            packed_und_vit_token_indexes.extend(vit_positions)
            packed_vit_token_indexes.extend(vit_positions)
            
            # Update position counters
            curr += num_vision_tokens
            _curr += num_vision_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
            split_lens.append(num_vision_tokens+2)  # +2 for start and end tokens
            attn_modes.append('full')

        # 3. add instruction_prompt + assistant_prompt after images_und
        assistant_prompt = "<|im_end|>"
        instruction_prompt = self.clean_instruction_prompt(instruction_prompt)
        print("Cleaned instruction prompt:", instruction_prompt)
        full_instruction_prompt = instruction_prompt + assistant_prompt
        instruction_prompt_ids = tokenizer.encode(full_instruction_prompt, add_special_tokens=False)
        packed_text_ids.extend(instruction_prompt_ids)
        packed_text_indexes.extend(range(_curr, _curr + len(instruction_prompt_ids)))
        packed_indexes.extend(range(curr, curr + len(instruction_prompt_ids)))
        curr += len(instruction_prompt_ids)
        _curr += len(instruction_prompt_ids)
        split_lens.append(len(instruction_prompt_ids))  # +2 for start and end tokens
        attn_modes.append('causal')
        # 4. process gen images
        for image in images_gen:
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_gen_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            # Use official Qwen3VL processor to process image
            processed = self.vision_processor(images=[image], text=["<|image_pad|>"], return_tensors="pt")
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
            
            # Get the processed tokens from processor (includes complete vision token sequence)
            vision_token_ids = processed["input_ids"][0]  # Remove batch dimension
            num_vision_tokens = len(vision_token_ids)
            
            # All tokens returned by processor are <|image_pad|> tokens needing vision embeddings
            packed_text_ids.extend(vision_token_ids.tolist())
            #packed_text_indexes.extend(range(_curr, _curr + num_vision_tokens))
            packed_indexes.extend(range(curr, curr + num_vision_tokens))
            
            # All positions need vision embeddings (no special tokens to skip)
            vit_positions = list(range(_curr, _curr + num_vision_tokens))
            
            # Store vision data
            packed_vit_tokens.append(pixel_values)
            packed_vit_position_ids.append(grid_thw[0])  # Remove batch dimension
            
            vit_token_seqlens.append(num_vision_tokens)
            # All positions need vision embeddings
            packed_gen_vit_token_indexes.extend(vit_positions)
            packed_vit_token_indexes.extend(vit_positions)
            
            # Update position counters
            curr += num_vision_tokens
            _curr += num_vision_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_gen_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
            split_lens.append(num_vision_tokens+2)  # +2 for start and end tokens
            attn_modes.append('full')
        # 5. add 8 learnable tokens to the end
        packed_learnable_token_indexes.extend(range(_curr, _curr + num_learnable_tokens))
        packed_indexes.extend(range(curr, curr + num_learnable_tokens))
        curr += num_learnable_tokens
        _curr += num_learnable_tokens
        split_lens.append(num_learnable_tokens)
        attn_modes.append('full')
        # 5. add 1 action learnable tokroot/qihang_projects/AutoMoTive_qihang_action/evaluation/eval_automot_fast_thinking_senna.shens to the end
        packed_action_token_indexes.extend(range(_curr, _curr + action_learnable_tokens))
        packed_indexes.extend(range(curr, curr + action_learnable_tokens))
        curr += action_learnable_tokens
        _curr += action_learnable_tokens
        split_lens.append(action_learnable_tokens)
        attn_modes.append('full')

        und_idxs = [
            idx for idx in packed_text_indexes
            if idx not in packed_gen_text_indexes
        ]
        packed_und_text_indexes = und_idxs
        total_seq_len = _curr
        packed_seqlens.append(total_seq_len)
        total_curr_kvlen = sum(curr_kvlens) if curr_kvlens else 0
        curr_position_start = curr_rope[0] if curr_rope else 0
        newlens.append(total_curr_kvlen + total_seq_len)
        # new_rope.append(curr_position_start + total_seq_len)

        device = self.language_model.model.embed_tokens.weight.device
        
        packed_text_ids_tensor = torch.tensor(packed_text_ids, dtype=torch.long, device=device)
        packed_vit_token_indexes_tensor = torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device)
        packed_und_vit_token_indexes_tensor = torch.tensor(packed_und_vit_token_indexes, dtype=torch.long, device=device)
        packed_gen_vit_token_indexes_tensor = torch.tensor(packed_gen_vit_token_indexes, dtype=torch.long, device=device)
        packed_vit_position_ids_tensor = torch.stack(packed_vit_position_ids, dim=0).to(device)
        
        attention_mask = torch.ones_like(packed_text_ids_tensor).unsqueeze(0)
        position_ids_3d, rope_deltas = self.language_model.get_rope_index_fast_thinking(
            input_ids=packed_text_ids_tensor.unsqueeze(0),  # [1, seq_len]
            image_grid_thw=packed_vit_position_ids_tensor,  # [num_images, 3]
            video_grid_thw=None,  
            attention_mask=attention_mask,
            num_learnable_tokens = action_learnable_tokens + num_learnable_tokens,
            tokenizer=tokenizer 
        )
        new_rope.append(position_ids_3d[0].max().item() + 1)

        # Add indexes for the current sequence being processed
        current_sequence_length = len(packed_text_ids)
        # For the current sequence, use the starting KV cache position, not the accumulated curr
        kv_cache_start = sum(curr_kvlens) if curr_kvlens else 0
        packed_key_value_indexes.extend(range(kv_cache_start, kv_cache_start + current_sequence_length))
        nested_attention_masks.append(
            prepare_attention_mask_per_sample(split_lens, attn_modes).to(device)
        )
        packed_text_ids_tensor = packed_text_ids_tensor[packed_text_ids_tensor != 151655]  # remove all <|image_pad|> tokens for fast generation
        generation_input = {
            "packed_text_ids": packed_text_ids_tensor.to(device=device, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            "packed_gen_text_indexes": torch.tensor(packed_gen_text_indexes, dtype=torch.long, device=device),
            "packed_und_text_indexes": torch.tensor(packed_und_text_indexes, dtype=torch.long, device=device),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int, device=device),
            "nested_attention_masks": nested_attention_masks,
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0).to(device),  # Concatenate pixel_values
            "packed_vit_position_ids": packed_vit_position_ids_tensor,  # Stack grid_thw
            "packed_vit_token_indexes": packed_vit_token_indexes_tensor,
            "packed_action_token_indexes": torch.tensor(packed_action_token_indexes, dtype=torch.long, device=device),
            "packed_und_vit_token_indexes": packed_und_vit_token_indexes_tensor,
            "packed_gen_vit_token_indexes": packed_gen_vit_token_indexes_tensor,
            "packed_learnable_token_indexes": torch.tensor(packed_learnable_token_indexes, dtype=torch.long, device=device),
            "packed_position_ids": position_ids_3d,
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int, device=device),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long, device=device),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long, device=device),
            "key_values_lens": torch.tensor(newlens, dtype=torch.int, device=device),
            "curr": int(curr),
        }

        return generation_input, newlens, new_rope
    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        """基于 KV 缓存执行自回归文本生成。"""
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_mot:
                extra_inputs = {"mode": "und"}

            # 文本生成阶段：必要时将 1D 位置 id 转为 3D 格式
            if packed_query_position_ids.dim() == 1:
                position_ids_3d = packed_query_position_ids.unsqueeze(0).expand(3, -1)
            else:
                position_ids_3d = packed_query_position_ids

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=position_ids_3d,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # 当前仅支持 batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)
