"""
LEAD video -> AutoMoT offline runner.

中文说明：
- 该文件是离线桥接实现，不修改原有在线 Agent。
- 目标：把 LEAD route 中的某个 anchor 帧（含必要历史窗口）切成 AutoMoT 所需的
  时序输入，复用 AutoMoT 现有 Qwen3-VL 慢推理链路。
- 入口语义：显式指定 anchor（route 内绝对帧索引），由采样参数反推需要的历史长度。
  clip 内 anchor 永远是最后一帧；当 anchor 太靠前导致历史不足时，会重复 frame 0
  补齐并打印 warning（不报错，但需注意数据有重复）。
- 关键采样规则：RGB 默认使用 [t, t-1, t-2, t-3]（按时间顺序喂入）。
- BEV/LiDAR：clip 只保存原始点云 (`lidar_points`)，栅格化在 `_prepare_inference_inputs`
  里完成 —— 跨帧对齐到 anchor ego-local 后调本文件内的 `lead_rasterize_lidar`
  (LEAD 风格：±40m × [-32, 64]m / 4 px/m / z ∈ [-4, 10] 闭区间含地面) 直接出
  (320, 384) 单通道，与 LEAD TransfuserBackbone 训练分布一致。
- BEV encoder 已切换为本文件底部抄过来的 LEAD TransfuserBackbone（单帧 tfv6 框架），
  权重通过 `LEAD_BEV_CKPT_PATH` 常量加载 LEAD tfv6_resnet34 backbone-only ckpt
  （已实测 missing=0 / unexpected=0，state_dict 完全匹配 LEAD 训练分布）。
- 快推理（AutoMoT 自家 bev_encoder_proj + heads + queries）默认禁用，因为 LEAD
  trans_feat shape (1, 512, 10, 12) 与原 AutoMoT 期望 (1, 1512, 8, 8) 不兼容；
  要启用 LEAD 版快推理需要重设计整个 decoder 链路（见 PROJECT_CONTEXT.md §12）。
  python mot_lead_offline_runner.py --enable-leadmot-planning

"""

from __future__ import annotations

import argparse
import lzma
import os
import pathlib
import pickle
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import HfArgumentParser, AutoTokenizer

try:
    import laspy
except Exception:
    laspy = None


# 允许从 AutoMoT 工程根导入 mot 包与 team_code 工具。
_THIS_FILE = pathlib.Path(__file__).resolve()
# /home/cruser1/lda/AutoMoT/leaderboard/team_code/mot_lead_offline_runner.py

_AUTOMOT_ROOT = _THIS_FILE.parents[2]  # .../AutoMoT
# /home/cruser1/lda/AutoMoT

_AUTOMOT_PROJECT_ROOT = _THIS_FILE.parents[2] / "Automot"
# /home/cruser1/lda/AutoMoT/Automot

_MOT_ROOT = _AUTOMOT_PROJECT_ROOT / "mot"
# /home/cruser1/lda/AutoMoT/Automot/mot

# 按照 mot_b2d_agent.py 的方式配置路径
projects_root = str(_AUTOMOT_ROOT)
mot_dp_path = str(_AUTOMOT_PROJECT_ROOT)
mot_path = str(_MOT_ROOT)

for path in [projects_root, mot_dp_path, mot_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# ！重要：在导入automot模块前，必须设置tokenizer路径
# 因为automot.py模块初始化时会创建全局tokenizer对象
_qwen3vl_path = str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B")
# Setting Qwen3VL tokenizer/processor path to: /home/cruser1/lda/AutoMoT/checkpoints/Qwen3-VL-4B

import mot.modeling.automot.automot as _automot_module_preset
_automot_module_preset.QWEN3VL_TOKENIZER_PATH = _qwen3vl_path
_automot_module_preset.QWEN3VL_PROCESSOR_PATH = _qwen3vl_path
# 强制重新初始化tokenizer
if not hasattr(_automot_module_preset, '_tokenizer_reinitialized'):
    from transformers import AutoTokenizer as Qwen3Tokenizer
    from data.reasoning.data_utils import add_special_tokens as _add_special_tokens
    try:
        _tmp_tokenizer = Qwen3Tokenizer.from_pretrained(_qwen3vl_path, local_files_only=True, trust_remote_code=True)
        _tmp_tokenizer, _, _ = _add_special_tokens(_tmp_tokenizer)
        _automot_module_preset.tokenizer = _tmp_tokenizer
        _automot_module_preset._tokenizer_reinitialized = True
        print(f"✓ Pre-initialized tokenizer from {_qwen3vl_path}")
    except Exception as e:
        print(f"Warning: Could not pre-initialize tokenizer: {e}")
        # 尝试加载Qwen3VLForConditionalGenerationMoT的tokenizer作为备选
        try:
            _tmp_tokenizer = Qwen3Tokenizer.from_pretrained(
                str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B"),
                local_files_only=True,
                trust_remote_code=True,
            )
            _tmp_tokenizer, _, _ = _add_special_tokens(_tmp_tokenizer)
            _automot_module_preset.tokenizer = _tmp_tokenizer
            print(f"✓ Pre-initialized tokenizer (fallback) from Qwen3-VL-4B")
        except Exception as e2:
            print(f"Error initializing tokenizer: {e2}")

from data.reasoning.data_utils import add_special_tokens
from mot.evaluation.inference import InterleaveInferencer
from mot.modeling.automot import AutoMoT
# LeadMoT 子包：frozen Qwen prefix K/V + LEAD BEV 风格快推理 decoder
# - 输入：池化后的 Qwen K/V (12 段) + LEAD BEV (B,512,10,12) + ego status
# - 输出：(B,10,2) pred_route + (B,8,2) pred_future_waypoints，天然对齐 LEAD planning_decoder 契约
# - 详细架构见 AutoMoT/qwen3vl_local/leadmot/ARCHITECTURE.md
from qwen3vl_local.leadmot import LeadMoTPlanningDecoder, LeadMoTPlanningDecoderConfig

# standalone Qwen3-VL-4B-Instruct 引擎：leadmot 路径**必须**用这一份拿 cache，
# 而不是 AutoMoT InterleaveInferencer（那条路绑 AutoMoT checkpoint 里的 Qwen/MoT 路径，
# 权重与 standalone Qwen 不同，K/V 分布也不同 ——> 训练用 standalone、推理用 AutoMoT 会
# 显著掉点）。leadmot 训练侧若也走 `qwen3vl_local.engine.LocalQwen3VLInstructEngine`，
# 这里推理路径与训练 cache 来源天然同源。详见 ARCHITECTURE.md §6 和 PROJECT_CONTEXT.md §11.6。
from qwen3vl_local.engine import LocalQwen3VLInstructEngine

# Qwen3-VL-4B-Instruct 本地 checkpoint 路径（与 qwen3vl_instruct_paradigm_a_runner.py 一致）
# 该目录对应 HuggingFace repo_id=Qwen/Qwen3-VL-4B-Instruct，用户远程环境已下载。
# 必须 local_files_only=True（engine.load() 内部已设），禁止联网补文件。
_QWEN_INSTRUCT_CHECKPOINT_DIR = _AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B-Instruct"

# leadmot 推理时给 standalone Qwen 用的 system prompt。
# 设计目标：让 Qwen3-VL 接收 4 帧前视图 + driving prompt，产生丰富的 K/V cache 给
# LeadMoT decoder 当 prefix 用。文本输出本身不消费——leadmot 只读 prefill 阶段的
# past_key_values。训练时也必须用同一个 system + user prompt 文本，cache 才同源。
# 注：所有进 LLM 的 prompt 必须英文（项目硬约定）。
_LEADMOT_QWEN_SYSTEM_PROMPT = (
    "You are an autonomous driving model. You will be given a sequence of front-view "
    "RGB images and a navigation hint with target points and current speed. A downstream "
    "planning decoder consumes your hidden state to predict the ego-vehicle trajectory."
)
# 注：BEV encoder 已切换为 LEAD TransfuserBackbone(见本文件底部 LeadBEVEncoder 类),
# 不再依赖 AutoMoT 自带的 BEVEncoderBackboneExtractor。保留 bev_encoder_utils 因为
# normalize_angle / algin_lidar 仍被 LiDAR 跨帧对齐逻辑使用。
import mot.modeling.bev_encoder.bev_encoder_utils as bev_encoder_t_u

from team_code.automot_utils import (
    InferenceArguments,
    ModelArguments,
    build_cleaned_prompt_and_modes,
    inverse_conversion_2d,
    load_model_mot,
)


# ============================================================================
# LEAD TransfuserBackbone(本地复制版本)
#
# 由于离线 runner 工作时无法 import lead/ 包(两仓库互相看不见),这里把 LEAD 中
# TransfuserBackbone 所需的最小代码"抄"过来:
#   - lead/lead/tfv6/transfuser_backbone.py: TransfuserBackbone / GPT / Block / SelfAttention
#   - lead/lead/tfv6/transfuser_utils.py: normalize_imagenet
#   - lead/lead/data_loader/carla_dataset_utils.py: rasterize_lidar
# 同步在 LeadBevConfig 里固定了 carla_leaderboard_mode=True 下的全部 backbone 默认值
# (取自 lead/lead/training/config_training.py)。
#
# 改动相对 LEAD 原文件极小:
#   - 去掉 lead.* 的 import,GPT/Block/SelfAttention 重命名为 _Lead 前缀避免歧义
#   - 去掉 jaxtyping/beartype 装饰器,保留语义
#   - top_down(LEAD planning head 用)被保留为参数注册,前向不调用,
#     以便加载 LEAD ckpt 时 backbone.up_conv* / c5_conv / upsample* 不报 missing key
#
# 权重导入窗口:LEAD_BEV_CKPT_PATH 常量。当前指向 LEAD tfv6_resnet34 backbone-only ckpt
# (HuggingFace ln2697/tfv6/tfv6_resnet34/model_0030_0.pth 提取),实测 strict=False 加载
# missing=0 / unexpected=0,state_dict 100% 匹配。LeadBEVEncoder._load_lead_weights
# 兼容两种格式:(a) 完整 LEAD ckpt → 按 backbone.* 前缀过滤;(b) 预提取的 backbone-only
# (无前缀) → 整个 dict 直接加载。设为 None 则走随机初始化(仅用于跑通 forward shape)。
# ============================================================================


@dataclass
class LeadBevConfig:
    """LEAD TransfuserBackbone 所需的最小配置。

    默认值取自 `lead/lead/training/config_training.py`(carla_leaderboard_mode=True)。
    任何字段如与训练时实际值不符,加载 ckpt 会因 shape mismatch 报错。
    """
    # ---- backbone branches ----
    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    LTF: bool = False
    img_vert_anchors: int = 12        # final_image_height(384) // 32
    img_horz_anchors: int = 36        # num_used_cameras(3) * width(384) // 32
    lidar_vert_anchors: int = 10      # lidar_height_pixel(320) // 32
    lidar_horz_anchors: int = 12      # lidar_width_pixel(384) // 32
    bev_features_chanels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2
    perspective_downsample_factor: int = 1
    lidar_height_pixel: int = 320     # (max_y - min_y) * pixels_per_meter = 80 * 4
    lidar_width_pixel: int = 384      # (max_x - min_x) * pixels_per_meter = 96 * 4
    channel_last: bool = False        # 推理走默认连续内存(LEAD 训练 True,但推理时无收益)
    # ---- GPT fusion transformer ----
    block_exp: int = 4
    n_layer: int = 2
    n_head: int = 4
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    gpt_linear_layer_init_mean: float = 0.0
    gpt_linear_layer_init_std: float = 0.02
    gpt_layer_norm_init_weight: float = 1.0
    # ---- LEAD rasterize_lidar(carla_leaderboard_mode=True) ----
    pixels_per_meter: float = 4.0
    min_x_meter: int = -32
    max_x_meter: int = 64             # 前向 64m(不对称,前看远后看近)
    min_y_meter: int = -40
    max_y_meter: int = 40
    hist_max_per_pixel: int = 5
    min_height_lidar: float = -4.0
    max_height_lidar: float = 10.0    # LEAD 训练保留 z ∈ [-4, 10] 闭区间(含地面)
    # ---- runtime dtype ----
    # backbone 内部按 self.config.torch_float_type cast 输入;LEAD 训练时 a100/l40s 上
    # 自动切 bf16,但我们离线推理走 float32 保数值稳定(normalize_imagenet 数值范围大)。
    torch_float_type: torch.dtype = torch.float32


def lead_rasterize_lidar(
    points_xyz: np.ndarray,
    config: LeadBevConfig,
) -> np.ndarray:
    """LEAD 风格 LiDAR 栅格化(抄自 lead/lead/data_loader/carla_dataset_utils.py)。

    输入: (N, 3) ego-local 点云(米),CARLA 朝向(x 前 y 右 z 上)。
    输出: (H, W) float32 [0, 1],H = (max_y-min_y)*ppm = 320, W = (max_x-min_x)*ppm = 384。
          row = y(右为正),col = x(前为正)。

    z 过滤: [min_height_lidar, max_height_lidar] 闭区间(含地面层)。
    """
    H = int((config.max_y_meter - config.min_y_meter) * int(config.pixels_per_meter))
    W = int((config.max_x_meter - config.min_x_meter) * int(config.pixels_per_meter))
    if points_xyz.size == 0:
        return np.zeros((H, W), dtype=np.float32)

    z = points_xyz[..., 2]
    mask = (z >= config.min_height_lidar) & (z <= config.max_height_lidar)
    pts = points_xyz[mask]

    xbins = np.linspace(
        config.min_x_meter,
        config.max_x_meter,
        (config.max_x_meter - config.min_x_meter) * int(config.pixels_per_meter) + 1,
    )
    ybins = np.linspace(
        config.min_y_meter,
        config.max_y_meter,
        (config.max_y_meter - config.min_y_meter) * int(config.pixels_per_meter) + 1,
    )
    hist = np.histogramdd(pts[:, :2], bins=(xbins, ybins))[0]
    hist = np.clip(hist, 0, config.hist_max_per_pixel) / float(config.hist_max_per_pixel)
    # .T 后 row=y(右为正), col=x(前为正),与 LEAD/AutoMoT BEV encoder splat 同款轴序。
    return hist.T.astype(np.float32)


def _normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """ImageNet 均值方差归一化(等价 LEAD transfuser_utils.normalize_imagenet)。"""
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x


class _LeadSelfAttention(nn.Module):
    """多头自注意力(抄自 tfv6/transfuser_backbone.py: SelfAttention)。"""

    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float, resid_pdrop: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.dropout = attn_pdrop
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))
        return y


class _LeadBlock(nn.Module):
    """Transformer block(抄自 tfv6/transfuser_backbone.py: Block)。"""

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = _LeadSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _LeadGPT(nn.Module):
    """GPT-style cross-modal fusion(抄自 tfv6/transfuser_backbone.py: GPT)。"""

    def __init__(self, n_embd: int, config: LeadBevConfig):
        super().__init__()
        self.n_embd = n_embd
        self.config = config
        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                config.img_vert_anchors * config.img_horz_anchors
                + config.lidar_vert_anchors * config.lidar_horz_anchors,
                n_embd,
            )
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(
            *[
                _LeadBlock(n_embd, config.n_head, config.block_exp,
                          config.attn_pdrop, config.resid_pdrop)
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor: torch.Tensor, lidar_tensor: torch.Tensor):
        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        image_tensor = (
            image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        )
        lidar_tensor = (
            lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        )
        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)
        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        img_token_n = self.config.img_vert_anchors * self.config.img_horz_anchors
        image_tensor_out = (
            x[:, :img_token_n, :]
            .view(bz, img_h, img_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        lidar_tensor_out = (
            x[:, img_token_n:, :]
            .view(bz, lidar_h, lidar_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return image_tensor_out, lidar_tensor_out


class LeadTransfuserBackbone(nn.Module):
    """LEAD TransFuser backbone(抄自 tfv6/transfuser_backbone.py: TransfuserBackbone)。

    主要差异:
    - 去掉 lead.* import,改用本文件内 _normalize_imagenet / _LeadGPT 等
    - 去掉 jaxtyping / beartype 装饰器
    - top_down 方法删除(LEAD 单帧 backbone 推理路径不用),但 c5_conv/up_conv*/upsample*
      参数保留(LEAD ckpt 里仍存这些权重,strict=False 加载不影响)
    """

    def __init__(self, device: torch.device, config: LeadBevConfig) -> None:
        super().__init__()
        self.device = device
        self.config = config

        # Image branch
        self.image_encoder = timm.create_model(
            config.image_architecture, pretrained=True, features_only=True
        )
        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (config.img_vert_anchors, config.img_horz_anchors)
        )
        image_start_index = 0
        if len(self.image_encoder.return_layers) > 4:
            image_start_index += 1
        self.num_image_features = self.image_encoder.feature_info.info[
            image_start_index + 3
        ]["num_chs"]

        # LiDAR branch
        self.lidar_encoder = timm.create_model(
            config.lidar_architecture,
            pretrained=False,
            in_chans=2 if config.LTF else 1,
            features_only=True,
        )
        lidar_start_index = 0
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_start_index += 1
        self.num_lidar_features = self.lidar_encoder.feature_info.info[
            lidar_start_index + 3
        ]["num_chs"]
        self.lidar_channel_to_img = nn.ModuleList(
            [
                nn.Conv2d(
                    self.lidar_encoder.feature_info.info[lidar_start_index + i]["num_chs"],
                    self.image_encoder.feature_info.info[image_start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )
        self.img_channel_to_lidar = nn.ModuleList(
            [
                nn.Conv2d(
                    self.image_encoder.feature_info.info[image_start_index + i]["num_chs"],
                    self.lidar_encoder.feature_info.info[lidar_start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )
        self.avgpool_lidar = nn.AdaptiveAvgPool2d(
            (config.lidar_vert_anchors, config.lidar_horz_anchors)
        )

        # Fusion transformers
        self.transformers = nn.ModuleList(
            [
                _LeadGPT(
                    n_embd=self.image_encoder.feature_info.info[image_start_index + i]["num_chs"],
                    config=config,
                )
                for i in range(4)
            ]
        )

        # Post-fusion convs(top_down 路径用,推理 forward 不调用,但参数保留以
        # 兼容 LEAD ckpt 的 backbone.* state_dict)
        self.perspective_upsample_factor = (
            self.image_encoder.feature_info.info[image_start_index + 3]["reduction"]
            // config.perspective_downsample_factor
        )
        self.upsample = nn.Upsample(
            scale_factor=config.bev_upsample_factor,
            mode="bilinear",
            align_corners=False,
        )
        self.upsample2 = nn.Upsample(
            size=(
                config.lidar_height_pixel // config.bev_down_sample_factor,
                config.lidar_width_pixel // config.bev_down_sample_factor,
            ),
            mode="bilinear",
            align_corners=False,
        )
        self.up_conv5 = nn.Conv2d(
            config.bev_features_chanels,
            config.bev_features_chanels,
            (3, 3),
            padding=1,
        )
        self.up_conv4 = nn.Conv2d(
            config.bev_features_chanels,
            config.bev_features_chanels,
            (3, 3),
            padding=1,
        )
        self.c5_conv = nn.Conv2d(
            self.num_lidar_features, config.bev_features_chanels, (1, 1)
        )

    def forward(self, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """LEAD-style forward。

        Args:
            data: {'rgb': (B, 3, H, W) [0, 255],
                   'rasterized_lidar': (B, 1, H_bev, W_bev) [0, 1]}
        Returns:
            (lidar_features (B, 512, 10, 12), image_features (B, 512, 12, 36)) for resnet34。
        """
        rgb = data["rgb"].to(
            self.device, dtype=self.config.torch_float_type, non_blocking=True
        )
        if self.config.LTF:
            x = torch.linspace(0, 1, self.config.lidar_width_pixel)
            y = torch.linspace(0, 1, self.config.lidar_height_pixel)
            y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
            lidar = torch.zeros(
                (rgb.shape[0], 2, self.config.lidar_height_pixel, self.config.lidar_width_pixel),
                device=rgb.device,
            )
            lidar[:, 0] = y_grid.unsqueeze(0)
            lidar[:, 1] = x_grid.unsqueeze(0)
        else:
            lidar = data["rasterized_lidar"].to(
                self.device, dtype=self.config.torch_float_type, non_blocking=True
            )
        return self._forward(rgb, lidar)

    def _forward(self, image: torch.Tensor, lidar: torch.Tensor):
        image_features = _normalize_imagenet(image)
        lidar_features = lidar

        if self.config.channel_last:
            image_features = image_features.to(memory_format=torch.channels_last)
            if lidar_features is not None:
                lidar_features = lidar_features.to(memory_format=torch.channels_last)

        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())

        if len(self.image_encoder.return_layers) > 4:
            image_features = self._forward_layer_block(
                image_layers, self.image_encoder.return_layers, image_features
            )
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self._forward_layer_block(
                lidar_layers, self.lidar_encoder.return_layers, lidar_features
            )

        for i in range(4):
            image_features = self._forward_layer_block(
                image_layers, self.image_encoder.return_layers, image_features
            )
            lidar_features = self._forward_layer_block(
                lidar_layers, self.lidar_encoder.return_layers, lidar_features
            )
            image_features, lidar_features = self._fuse_features(
                image_features, lidar_features, i
            )
        return lidar_features, image_features

    @staticmethod
    def _forward_layer_block(layers, return_layers, features):
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def _fuse_features(self, image_features, lidar_features, layer_idx: int):
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)
        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

        image_features_layer, lidar_features_layer = self.transformers[layer_idx](
            image_embd_layer, lidar_embd_layer
        )
        lidar_features_layer = self.img_channel_to_lidar[layer_idx](lidar_features_layer)
        image_features_layer = F.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        lidar_features_layer = F.interpolate(
            lidar_features_layer,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        image_features = image_features + image_features_layer
        lidar_features = lidar_features + lidar_features_layer
        return image_features, lidar_features


# LEAD BEV encoder backbone-only ckpt 路径。
# 当前指向已从 HuggingFace ln2697/tfv6/tfv6_resnet34/model_0030_0.pth 提取的
# backbone 子集(前缀已剥),实测 strict=False 加载 missing=0 / unexpected=0,
# state_dict 与 LeadTransfuserBackbone 100% 匹配。
#
# 加载逻辑(见 LeadBEVEncoder._load_lead_weights):
#   1. torch.load → 若含 "model" 子 dict 则取出
#   2. 优先按 `backbone.*` 前缀过滤(用于直接喂完整 LEAD ckpt 时);若过滤后为空
#      则视为已预提取的 backbone-only ckpt(无前缀),整个 dict 直接加载
#   3. strict=False 加载到 self.backbone,容忍 ckpt 里有其它非 backbone 条目
# 设为 None 则走随机初始化(仅用于跑通 forward shape 验证)。
LEAD_BEV_CKPT_PATH: str | None = "/home/cruser1/lda/AutoMoT/checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth"


class LeadBEVEncoder(nn.Module):
    """对齐原 BEVEncoderBackboneExtractor 接口的 LEAD wrapper。

    - 实例化: LeadBEVEncoder(config, device, ckpt_path)
    - 调用:  out = wrapper(rgb=(B,3,384,1152) [0,255], lidar_bev=(B,1,320,384) [0,1])
    - 返回 dict 兼容原 AutoMoT key 风格(便于将来切回快推理时少改下游):
        - bev_feature:        (B, 512, 10, 12)  ← LEAD lidar branch 输出(作为 trans_feat)
        - image_feature_grid: (B, 512, 12, 36)  ← LEAD image branch 输出

    注意 bev_feature shape != AutoMoT 原 (1, 1512, 8, 8)。
    runner 默认禁用快推理(enable_fast_inference=False),此 trans_feat 不被消费。
    """

    def __init__(
        self,
        config: LeadBevConfig,
        device: torch.device,
        ckpt_path: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.backbone = LeadTransfuserBackbone(device=device, config=config)
        if ckpt_path is not None:
            self._load_lead_weights(ckpt_path)
        else:
            print("[LeadBEVEncoder] LEAD_BEV_CKPT_PATH 未设置,backbone 走随机初始化"
                  "(仅用于跑通 forward shape 验证;正式使用前请填权重路径)")
        self.backbone.to(device)
        self.backbone.eval()

    def _load_lead_weights(self, ckpt_path: str) -> None:
        """从 LEAD .pth 加载 backbone.* 子集(strict=False)。"""
        ckpt_p = pathlib.Path(ckpt_path)
        if not ckpt_p.exists():
            raise FileNotFoundError(f"LEAD BEV ckpt not found: {ckpt_path}")
        full_sd = torch.load(str(ckpt_p), map_location="cpu")
        # LEAD training_utils.py 保存格式:整个 model.state_dict() 或 {"model": state_dict}
        if isinstance(full_sd, dict) and "model" in full_sd and isinstance(full_sd["model"], dict):
            full_sd = full_sd["model"]
        backbone_sd = {
            k[len("backbone."):]: v
            for k, v in full_sd.items()
            if k.startswith("backbone.")
        }
        if not backbone_sd:
            # 若 ckpt 已只含 backbone 子模块(无前缀),直接用
            backbone_sd = full_sd
        missing, unexpected = self.backbone.load_state_dict(backbone_sd, strict=False)
        print(f"[LeadBEVEncoder] loaded {ckpt_path}: "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"  first 5 missing: {missing[:5]}")
        if unexpected:
            print(f"  first 5 unexpected: {unexpected[:5]}")

    def forward(
        self,
        rgb: torch.Tensor,
        lidar_bev: torch.Tensor,
    ) -> dict:
        """对齐 BEVEncoderBackboneExtractor.forward(rgb=, lidar_bev=) 调用风格。"""
        target_dtype = self.config.torch_float_type
        rgb_cast = rgb.to(self.device, dtype=target_dtype)
        lidar_cast = lidar_bev.to(self.device, dtype=target_dtype)
        data = {"rgb": rgb_cast, "rasterized_lidar": lidar_cast}
        with torch.no_grad():
            lidar_feat, image_feat = self.backbone(data)
        return {
            "bev_feature": lidar_feat,         # (B, 512, 10, 12)
            "image_feature_grid": image_feat,  # (B, 512, 12, 36)
        }


def _cache_tensor_to_bhsd(x: torch.Tensor) -> torch.Tensor:
    """把单个 Qwen cache 张量统一成 LeadMoT attention 需要的 4D 形状。

    目标形状
    ========
    LeadMoT 的 `PrefixKVAttention` 直接把 frozen Qwen 的 K/V 拼到 gen 自己
    算出的 K/V 后面，所以语言 cache 必须是：

        (B, num_kv_heads, seq_len, head_dim)

    对 Qwen3-VL-4B-Instruct 来说就是 `(B, 8, S, 128)`。

    为什么这里要兼容 3D / 4D
    ==========================
    - HF 标准 `past_key_values` 通常已经是 4D: `(B, H_kv, S, D)`。
    - AutoMoT 自定义 `NaiveCache` 为了在线 cache 复用，可能把 batch 维压掉，
      存成 3D: `(S, H_kv, D)`。

    本函数只做**形状规整 + detach**，不做 dtype/device 迁移、不做线性投影、
    不改 cache 数值；这保证 LeadMoT 读到的就是 frozen Qwen 原始 K/V。
    """
    # 类型防御：cache 里如果混进 list/None，后续 shape 访问会报很隐晦的错；
    # 这里提前给出具体类型，方便定位是哪一路 cache 解析失败。
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Qwen cache item must be a tensor, got {type(x).__name__}")

    if x.ndim == 4:
        # HF 标准形态，已经是 (B, H_kv, S, D)。detach 切断慢推理计算图，
        # 因为这里是 frozen Qwen cache，只作为 prefix memory 消费。
        return x.detach()
    if x.ndim == 3:
        # AutoMoT NaiveCache 存的是 (S, H_kv, D)，缺 batch 维且维度顺序不同。
        # permute(1,0,2) -> (H_kv, S, D)，unsqueeze(0) -> (1, H_kv, S, D)。
        # contiguous() 是为了后续 attention concat / view 时拿到连续内存布局。
        return x.detach().permute(1, 0, 2).unsqueeze(0).contiguous()

    # 只允许 3D/4D。2D 代表 head 维或 seq 维丢了，5D 通常代表外面又包了一层 batch，
    # 都不能被安全猜测，直接抛错比静默 reshape 更安全。
    raise ValueError(f"Unsupported Qwen cache tensor shape: {tuple(x.shape)}")


def _qwen_cache_to_layer_list(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """把任意形态的 Qwen past_key_values 统一成 list[(K, V)] 的逐层格式。

    背景
    ====
    AutoMoT 慢推理产出的 `gen_context["past_key_values"]` 在不同环境下类型可能不同：
    - HuggingFace 新版 `DynamicCache`：带 `to_legacy_cache()` 方法
    - AutoMoT 自定义 `NaiveCache`：带 `.key_cache` / `.value_cache` 属性（可能是 dict 也可能是 list）
    - HF legacy tuple：直接是 `list[(k, v), ...]`

    本函数统一三种形态，输出 `list[(k, v)]`，每个 K/V 都过 `_cache_tensor_to_bhsd`
    标准化成 (B, num_kv_heads, S, head_dim) 4D 张量，便于 LeadMoT decoder 直接消费。

    参数:
        past_key_values: 任意上述格式的 Qwen KV cache 对象
    返回:
        list[(K, V)]，长度 = Qwen 层数（Qwen3-VL-4B-Instruct 是 36）
        每个 K/V 形状 (B, num_kv_heads=8, S, head_dim=128)
    """
    # None 检查：调用方应保证慢推理已经跑完
    if past_key_values is None:
        raise ValueError("gen_context does not contain past_key_values")

    # HF 新版 Cache 对象提供 to_legacy_cache() 转回旧版 tuple 格式，统一处理
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()

    # 最终结果容器：每个元素是某一层的 (K, V) tuple
    layers: list[tuple[torch.Tensor, torch.Tensor]] = []

    # ---- 路径 1：AutoMoT NaiveCache (有 key_cache/value_cache 属性) ----
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        key_cache = past_key_values.key_cache
        value_cache = past_key_values.value_cache

        # NaiveCache 内部可能用 dict（key=layer_id）也可能用 list（index=layer_id）
        # 两种都要处理：取出 layer_ids 排序，然后用统一的 getter 取值
        if isinstance(key_cache, dict):
            layer_ids = sorted(key_cache.keys())
            get_key = key_cache.get
            get_value = value_cache.get
        else:
            layer_ids = range(len(key_cache))
            get_key = key_cache.__getitem__
            get_value = value_cache.__getitem__

        # 逐层取出 K/V，过 _cache_tensor_to_bhsd 标准化形状，跳过 None 层（防御性）
        for layer_id in layer_ids:
            k = get_key(layer_id)
            v = get_value(layer_id)
            if k is None or v is None:
                continue
            layers.append((_cache_tensor_to_bhsd(k), _cache_tensor_to_bhsd(v)))
        return layers

    # ---- 路径 2：HF legacy tuple 格式 list[(k, v), ...] ----
    if isinstance(past_key_values, (list, tuple)):
        for layer in past_key_values:
            # 空层跳过
            if layer is None:
                continue
            # 每层必须是 (K, V) 二元组（HF 标准）
            if not isinstance(layer, (list, tuple)) or len(layer) < 2:
                raise ValueError(f"Invalid legacy cache layer item: {type(layer).__name__}")
            layers.append((_cache_tensor_to_bhsd(layer[0]), _cache_tensor_to_bhsd(layer[1])))
        return layers

    # 既不是 NaiveCache 也不是 legacy tuple，类型不识别
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values).__name__}")


def _segment_qwen_cache_for_leadmot(
    past_key_values: Any,
    config: LeadMoTPlanningDecoderConfig,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """把 36 层 Qwen K/V 按 select_last 池化成 12 段 LeadMoT decoder 用的 prefix K/V。

    池化策略
    ========
    把 Qwen num_layers (36) 切成 config.num_layers (12) 段，每段取最后一层的 K/V。
    例如：
    - 段 0: 取 Qwen 第 2 层 K/V（段 [0,1,2]）
    - 段 1: 取 Qwen 第 5 层 K/V（段 [3,4,5]）
    - ...
    - 段 11: 取 Qwen 第 35 层 K/V（段 [33,34,35]）

    这与 goalgen `segment_kv_for_dit(mode='select_last')` 同义；
    runner 内联一份是为了减少跨子包依赖（goalgen 是另一条路线）。

    为什么 select_last
    ==================
    每段取最后一层而不是平均/拼接：
    - 最后一层语义最丰富，最接近 Qwen 最终理解
    - 显存最省（只取 1 层而不是 3 层平均/拼接）
    - 跟 goalgen v2 实测效果一致

    参数:
        past_key_values: frozen Qwen prefill 输出的 KV cache（任意形态）
        config:          LeadMoTPlanningDecoderConfig，提供 num_layers / num_kv_heads / head_dim
    返回:
        list[(K, V)] 长度 = config.num_layers (12)，每段一对 K/V
    """
    # 先把任意形态 cache 统一成 list[(K, V)] 标准形状
    layers = _qwen_cache_to_layer_list(past_key_values)

    # Qwen 层数必须 >= LeadMoT 层数，否则池化分段会不够分
    # 实际场景：Qwen3-VL-4B-Instruct 36 层 >> LeadMoT 默认 12 层，永远够
    if len(layers) < config.num_layers:
        raise ValueError(
            f"Qwen cache layers ({len(layers)}) fewer than LeadMoT layers ({config.num_layers})"
        )

    # 当前只实现了 select_last，其它模式（mean / concat_layers）暂未支持
    # 如果未来要支持，参考 goalgen.qwen_kv.segment_kv_for_dit
    if config.kv_segment_mode != "select_last":
        raise ValueError(f"Unsupported kv_segment_mode={config.kv_segment_mode!r}")

    # 池化结果容器
    selected: list[tuple[torch.Tensor, torch.Tensor]] = []

    # 逐段计算：第 seg_idx 段对应 Qwen 哪一层
    for seg_idx in range(config.num_layers):
        # 段右边界（half-open）：(seg_idx+1) * len(layers) / num_layers
        # 比如 12 段 / 36 层时：第 0 段右边界 = 3，最后一层是 index 2
        # round 是为了处理 num_qwen_layers 不能被 num_layers 整除的边界情况
        end = round((seg_idx + 1) * len(layers) / config.num_layers)
        # select_last：段内最后一层 index = end - 1
        # max(0, ...) 保险防止边界算成负数
        layer_idx = max(0, end - 1)
        k, v = layers[layer_idx]

        # 顺便再校验一遍 config 自洽（防御性，避免下游 attention 报错）
        config.validate_qwen_kv_shape()

        # 这一段 K/V 的形状必须严格匹配 (num_kv_heads, head_dim)
        # 否则 PrefixKVAttention 直接拼接时会 shape mismatch
        expected = (config.num_kv_heads, config.head_dim)
        actual_k = (int(k.shape[1]), int(k.shape[-1]))
        actual_v = (int(v.shape[1]), int(v.shape[-1]))
        if actual_k != expected or actual_v != expected:
            raise ValueError(
                f"Qwen cache shape mismatch at layer {layer_idx}: "
                f"k={tuple(k.shape)}, v={tuple(v.shape)}, expected heads/dim={expected}"
            )

        selected.append((k, v))
    return selected


@dataclass
class OfflineGroup:
    """单个离线样本组的描述信息。"""

    anchor_t: int
    rgb_indices_desc: list[int]
    rgb_indices_asc: list[int]


class LeadOfflineMoTRunner:
    """
    LEAD clip 的离线推理执行器。

    中文说明：
    - 初始化 AutoMoT + BEV encoder。
    - 接收 LEAD video clip（包含 T 帧）并按 AutoMoT 在线策略重组输入。
    - 支持“一个 clip 生成多组样本”，提高数据利用率。
    """

    def __init__(
        self,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._setup_model()

    def _setup_model(self) -> None:
        """模型初始化：复用 AutoMoT 在线推理配置。"""
        parser = HfArgumentParser((ModelArguments, InferenceArguments))
        model_args, inference_args = parser.parse_args_into_dataclasses(args=[])
        
        # 修正model_path：checkpoints应该指向AutoMoT根目录下的checkpoints
        import os
        actual_model_path = str(_AUTOMOT_ROOT / "checkpoints" / "AutoMoT")
        if os.path.isfile(os.path.join(actual_model_path, "model.safetensors")) or \
           os.path.isfile(os.path.join(actual_model_path, "model.safetensors.index.json")):
            model_args.model_path = actual_model_path
            print(f"✓ Using model path: {model_args.model_path}")
        
        # 修正qwen3vl_path
        model_args.qwen3vl_path = str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B")
        
        self.inference_args = inference_args

        # 设置 tokenizer 和 processor 路径（在模型加载前）
        import mot.modeling.automot.automot as _automot_module
        _automot_module.QWEN3VL_TOKENIZER_PATH = model_args.qwen3vl_path
        _automot_module.QWEN3VL_PROCESSOR_PATH = model_args.qwen3vl_path

        # 1) 主模型
        # 修正automot_utils中的路径变量以使用正确的model_path
        from team_code import automot_utils as _automot_utils_module
        _automot_utils_module._AUTOMOT_ROOT = str(_AUTOMOT_ROOT)
        # 直接覆盖 dataclass 默认值，确保 load_model_mot() 解析到正确 checkpoint。
        _automot_utils_module.ModelArguments.__dataclass_fields__["model_path"].default = model_args.model_path
        _automot_utils_module.ModelArguments.__dataclass_fields__["qwen3vl_path"].default = model_args.qwen3vl_path

        self.automot: AutoMoT = load_model_mot(self.device)

        # 2) 获取tokenizer（已在load_model_mot中全局初始化）
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.qwen3vl_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        self.automot.language_model.tokenizer = tokenizer

        # 3) inferencer
        self.inferencer = InterleaveInferencer(
            model=self.automot,
            vae_model=None,
            tokenizer=tokenizer,
            vae_transform=None,
            vit_transform=None,
            new_token_ids=new_token_ids,
            max_num_tokens=inference_args.max_num_tokens,
            visual_gen=True,
            visual_und=True,
        )

        # 4) BEV encoder：用 LEAD TransfuserBackbone(单帧 tfv6 框架,本文件底部抄过来)
        # 数据预处理(RGB shape / LiDAR 栅格 / 视野范围 / z 过滤)同步切到 LEAD 风格,
        # 详见 _prepare_inference_inputs。快推理路径默认禁用(trans_feat 不会被消费,
        # 即便 LEAD backbone 输出 shape (1, 512, 10, 12) 与 AutoMoT 原 (1, 1512, 8, 8)
        # 不兼容也无影响),见 run_step 的 enable_fast_inference 参数。
        self.bev_encoder_config = LeadBevConfig()
        self.bev_encoder = LeadBEVEncoder(
            config=self.bev_encoder_config,
            device=self.device,
            ckpt_path=LEAD_BEV_CKPT_PATH,
        )

        # ============================================================
        # LeadMoT decoder：lazy init 占位
        # ============================================================
        # 默认不实例化 decoder。原因：
        # - decoder 约 153M 参数，fp32 占 ~584MB，bf16 占 ~292MB
        # - 大部分慢推理工作流（仅取 Qwen gen_context 做下游分析）不需要这个 decoder
        # - 让用户显式 --enable-leadmot-planning 才付出显存代价
        #
        # 真正构建在 _ensure_leadmot_decoder() 里，run_step 内首次进入
        # enable_leadmot_planning 分支时触发。
        #
        # 四个占位字段：
        # - leadmot_config:  配置实例，lazy build 时填充
        # - leadmot_decoder: nn.Module 实例，lazy build 时填充
        # - leadmot_dtype:   默认 bf16，可在 lazy build 前修改成别的 dtype
        # - leadmot_qwen_engine: standalone Qwen3-VL-4B-Instruct 推理引擎（lazy）。
        #   leadmot 路径**专用**这一份，与 AutoMoT InterleaveInferencer 解耦，
        #   保证 cache 与 leadmot 训练时同源（详见模块顶部 import 注释）。
        self.leadmot_config: LeadMoTPlanningDecoderConfig | None = None
        self.leadmot_decoder: LeadMoTPlanningDecoder | None = None
        self.leadmot_dtype: torch.dtype = torch.bfloat16
        self.leadmot_qwen_engine: LocalQwen3VLInstructEngine | None = None

        print(f"✓ Model initialized on {self.device}")
        print("  - AutoMoT loaded")
        print("  - BEV Encoder (LEAD TransfuserBackbone) loaded")
        print("  - LeadMoT decoder lazy (will be built on first --enable-leadmot-planning)")
        print("  - Inferencer initialized")

    def _ensure_leadmot_decoder(self) -> None:
        """首次调用时按 self.leadmot_dtype 在 self.device 上构建 LeadMoT decoder。

        幂等：decoder 已经构建过会直接 return，不重复构建。

        构建步骤：
        1. 实例化默认配置 LeadMoTPlanningDecoderConfig（CARLA 模式）
        2. 立刻 validate_qwen_kv_shape() 抛出配置错误（hidden / num_kv_heads / head_dim 不自洽）
        3. 实例化 LeadMoTPlanningDecoder
        4. .to(device, dtype) 搬到推理设备并转 bf16
        5. .eval() 关闭 dropout（本子包默认 dropout=0，但保留 .eval() 调用作为约定）
        """
        # 幂等检查：已经构建过就直接返回，不重复初始化
        if self.leadmot_decoder is not None:
            return

        # 默认配置：LEAD CARLA 模式（route=10, waypoint=8, hidden=1024 等）
        # 如果未来要覆盖配置，在这里改成 LeadMoTPlanningDecoderConfig(num_layers=8, ...) 之类
        self.leadmot_config = LeadMoTPlanningDecoderConfig()

        # 早校验：配置错的话立即 raise，不要延迟到 forward 内部 attention 报错
        self.leadmot_config.validate_qwen_kv_shape()

        # 实例化 + 搬设备 + 转 dtype + 切 eval
        # 链式 .to(...).eval() 在 PyTorch 里都返回 self，方便链式赋值
        self.leadmot_decoder = (
            LeadMoTPlanningDecoder(self.leadmot_config)
            .to(device=self.device, dtype=self.leadmot_dtype)
            .eval()
        )
        print(f"✓ LeadMoT decoder lazy-built on {self.device} ({self.leadmot_dtype})")

    def _ensure_leadmot_qwen_engine(self) -> None:
        """首次进入 enable_leadmot_planning 分支时 lazy 构建 standalone Qwen 引擎。

        参考用法照搬 `goalgen/qwen_kv.py:teacher_forced_prefill` 与
        `qwen3vl_instruct_paradigm_a_runner.py` 的 engine 实例化套路：
            engine = LocalQwen3VLInstructEngine(checkpoint_dir, device, torch_dtype="bfloat16")
            engine.load()
        engine.load() 是幂等的（已加载就跳过），所以这里 _ensure 拦了第二次构建
        以避免每次 run_step 都重新建实例。

        显存代价
        ========
        Qwen3-VL-4B-Instruct bf16 约 8 GB 显存（参数 + 激活），首次 prefill 还会
        额外吃几 GB autograd-disabled 工作区。所以这一步只在用户显式开
        --enable-leadmot-planning 时才付出。
        """
        # 幂等：已经构建过就直接返回
        if self.leadmot_qwen_engine is not None:
            return

        # 路径校验：本地必须已有 Qwen3-VL-4B-Instruct checkpoint，禁止联网补文件
        if not _QWEN_INSTRUCT_CHECKPOINT_DIR.exists():
            raise FileNotFoundError(
                f"Missing local Qwen3-VL-4B-Instruct checkpoint: {_QWEN_INSTRUCT_CHECKPOINT_DIR}. "
                f"Place HuggingFace `Qwen/Qwen3-VL-4B-Instruct` files there before "
                f"running --enable-leadmot-planning."
            )

        # 构造 engine。torch_dtype 用字符串 "bfloat16"（engine 内部自己 map 到 torch.bfloat16）
        # max_gen_tokens=1 是因为我们只要 prefill 阶段的 past_key_values，不消费 decode 输出；
        # 但 engine.prefill 不强制走 generate 路径，所以这个参数实际上对 leadmot 路径无影响，
        # 留默认值即可。
        self.leadmot_qwen_engine = LocalQwen3VLInstructEngine(
            checkpoint_dir=_QWEN_INSTRUCT_CHECKPOINT_DIR,
            device=self.device,
            torch_dtype="bfloat16",
        )

        # engine.load() 内部做 from_pretrained + .to(device) + .eval()，offline 严格模式
        self.leadmot_qwen_engine.load()
        print(
            f"✓ LeadMoT Qwen engine (standalone Qwen3-VL-4B-Instruct) "
            f"lazy-loaded on {self.device}"
        )

    @torch.no_grad()
    def _run_leadmot_qwen_prefill(
        self,
        rgb_pil_list: list,
        user_prompt: str,
    ) -> Any:
        """用 standalone Qwen 跑一次 prefill，返回 past_key_values 给 leadmot 池化用。

        参数:
            rgb_pil_list: 4 帧 PIL Image，三视角拼接 (1152, 384) 风格，与 runner 慢路径同款
            user_prompt:  runner 已构造的 prompt_cleaned（含 target_point + 速度文本）

        返回:
            past_key_values：HuggingFace 标准 cache 对象（或 NaiveCache 兼容），36 层
            每层 (B, num_kv_heads=8, S, head_dim=128)。可直接喂
            `_segment_qwen_cache_for_leadmot` 池化。

        与 goalgen.qwen_kv.teacher_forced_prefill 的差异
        =================================================
        - 我们不用 build_teacher_system_prompt / build_teacher_user_prompt（那是
          DiT teacher-forced 训练专用 system prompt）
        - 我们用 _LEADMOT_QWEN_SYSTEM_PROMPT + runner 自己的 prompt_cleaned，保证
          runner 推理时和 leadmot 训练时用同一对 (system, user)
        - 其它步骤（build_messages -> apply_chat_template -> prepare_inputs -> prefill）
          完全照抄
        """
        # engine 必须已经 lazy 加载（调用方保证 _ensure_leadmot_qwen_engine 先跑）
        engine = self.leadmot_qwen_engine
        if engine is None:
            raise RuntimeError(
                "leadmot_qwen_engine is None — call _ensure_leadmot_qwen_engine() first."
            )

        # 标准三步：构造 messages -> 应用 chat template -> 处理多模态输入
        messages = engine.build_messages(
            system_prompt=_LEADMOT_QWEN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            images=rgb_pil_list,
        )
        chat_text = engine.apply_chat_template(messages)
        inputs = engine.prepare_inputs(chat_text, rgb_pil_list)

        # 跑 prefill。装饰器 @torch.no_grad 已经保证了不带 autograd state；
        # engine.prefill 内部传 use_cache=True，返回 outputs.past_key_values
        outputs = engine.prefill(inputs)
        return outputs.past_key_values

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        """把 tensor/list 统一转 numpy。"""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _ensure_hwc_uint8(img: np.ndarray) -> np.ndarray:
        """把输入统一为 HWC uint8 RGB。"""
        arr = img
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3:
            raise ValueError(f"RGB frame ndim invalid: {arr.ndim}")

        # 若是 float，按 0~1 归一化转 uint8。
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)

        # 只保留前三通道。
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

        return arr

    @staticmethod
    def _lead_lidar_to_rgb_bev(rasterized_lidar: np.ndarray) -> np.ndarray:
        """把 LEAD 风格 BEV 栅格 (H, W) 或 (1, H, W) 转成 3 通道可视化 RGB(仅日志)。

        - 输入: (320, 384) 或 (1, 320, 384) float32 [0, 1]
        - 输出: (320, 384, 3) uint8(不送入模型,仅供调试 PIL Image 显示)
        """
        arr = rasterized_lidar
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"rasterized_lidar shape invalid: {arr.shape}")
        arr_u8 = (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        bev = np.repeat(arr_u8[:, :, None], 3, axis=2)
        return bev

    @staticmethod
    def _lead_lidar_to_bev_encoder_channel(rasterized_lidar: np.ndarray) -> np.ndarray:
        """把栅格化结果整成 (1, H, W) float32,送入 LEAD BEV encoder。

        输入: (H, W) 或 (1, H, W) float32 [0, 1]。LEAD 训练分布是
        H=320(=80m * 4 px/m), W=384(=96m * 4 px/m)。
        输出: (1, H, W) float32 [0, 1],unsqueeze batch 后 (1, 1, 320, 384) 即可
        喂给 LeadBEVEncoder.forward(lidar_bev=...)。
        """
        arr = rasterized_lidar
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3:
            raise ValueError(f"rasterized_lidar shape invalid: {arr.shape}")
        return arr.astype(np.float32)

    @staticmethod
    def _align_lidar_points_to_anchor(
        points_xyz: np.ndarray,
        src_pos_xy: np.ndarray,
        src_theta: float,
        anchor_pos_xy: np.ndarray,
        anchor_theta: float,
    ) -> np.ndarray:
        """将历史帧 LiDAR 点云对齐到当前 anchor 帧自车局部坐标系（ego local）。

        坐标系说明：
        - 入参 points_xyz：已经是 src 帧的自车局部坐标点云（ego-local）。
          LEAD LAZ 文件在录制时已完成 sensor->ego 转换；读取时无需再次转换。
        - src_pos_xy/src_theta 与 anchor_pos_xy/anchor_theta：均为世界坐标系下的自车位姿。
        - 返回值：anchor 帧自车局部坐标系（anchor ego-local）下的点云。

        正确变换推导（供参考）：
            x_a = R(θ_a)^T @ (R(θ_s) @ x_s + p_s - p_a)
            平移项应为 R(θ_s)^T @ (p_a - p_s)，旋转项为 θ_a - θ_s。

        本函数使用严格正确的实现（平移项用 R(src_theta).T）。
        在线 agent _align_lidar_bev_encoder 用的是 R(anchor_theta).T，
        属已知近似误差；相邻帧旋转差小时误差可忽略，此处离线版已修正，与其不同。
        """
        if points_xyz.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        pos_diff = np.array(
            [float(anchor_pos_xy[0] - src_pos_xy[0]), float(anchor_pos_xy[1] - src_pos_xy[1]), 0.0],
            dtype=np.float32,
        )
        rot_diff = float(bev_encoder_t_u.normalize_angle(float(anchor_theta - src_theta)))

        # 正确平移项：用 R(src_theta).T 将世界位移旋转到 src 帧局部坐标，
        # 而非 R(anchor_theta).T（在线 agent 用的是后者，属已知近似误差）。
        rotation_matrix = np.array(
            [
                [np.cos(src_theta), -np.sin(src_theta), 0.0],
                [np.sin(src_theta), np.cos(src_theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        pos_diff_local = rotation_matrix.T @ pos_diff
        aligned = bev_encoder_t_u.algin_lidar(points_xyz, pos_diff_local, rot_diff)
        return np.asarray(aligned, dtype=np.float32)

    def _build_group_indices(self, clip_len: int, anchor_t: int, 
                            rgb_frame_step: int = 1, rgb_frame_count: int = 4) -> OfflineGroup:
        """
        构造一个 anchor 的 RGB 采样索引。

        参数说明：
        - rgb_frame_step: 帧间隔（默认 1，对应 LEAD 的 0.25s，使用 5 对应 AutoMoT 的 1.25s）
        - rgb_frame_count: 采样帧数（默认 4）
        
        规则：采样 [t, t-step, t-2*step, ...] 共 rgb_frame_count 帧。
        clip 长度不足时会被 clamp 到 0，确保索引有效。
        """
        desc = [max(anchor_t - i * rgb_frame_step, 0) for i in range(rgb_frame_count)]
        asc = list(reversed(desc))
        return OfflineGroup(anchor_t=anchor_t, rgb_indices_desc=desc, rgb_indices_asc=asc)

    def build_group_for_last_frame(self, clip_len: int,
                                   rgb_frame_step: int = 1, rgb_frame_count: int = 4) -> list[OfflineGroup]:
        """
        为 clip 的最后一帧构造单个 group。
        
        参数说明：
        - rgb_frame_step: RGB 采样间隔（默认 1）
        - rgb_frame_count: RGB 采样帧数（默认 4）
        
        返回包含单个 group 的列表，便于与原有 run_clip 接口兼容。
        """
        anchor_t = clip_len - 1  # 最后一帧作为 anchor
        group = self._build_group_indices(
            clip_len=clip_len,
            anchor_t=anchor_t,
            rgb_frame_step=rgb_frame_step,
            rgb_frame_count=rgb_frame_count
        )
        return [group]

    def _prepare_inference_inputs(self, lead_clip: dict[str, Any], group: OfflineGroup,
                                  bev_frame_step: int = 1, bev_frame_count: int = 1):
        """把 LEAD clip + group 转为 inferencer 输入。"""
        rgb_clip = self._to_numpy(lead_clip["rgb"])  # 期望 (T,C,H,W) 或 (T,H,W,C)
        # 默认 anchor=12, max_history=3 -> clip_len=4：(4, 384, 1152, 3)

        use_pose_aligned_lidar = all(k in lead_clip for k in ("lidar_points", "pos_global", "theta"))
        if not use_pose_aligned_lidar:
            raise ValueError(
                "缺少 lidar_points/pos_global/theta，无法执行与在线策略一致的 LiDAR 坐标对齐流程。"
            )

        lidar_points_clip = lead_clip.get("lidar_points", None)
        # 变长列表 list[4]，每帧 shape ≈ (N_i, 3)，N_i 因帧而异（示例: ~4676~34890）

        pos_global_clip = np.asarray(self._to_numpy(lead_clip.get("pos_global", [])), dtype=np.float32)
        # (4, 2)

        theta_clip = np.asarray(self._to_numpy(lead_clip.get("theta", [])), dtype=np.float32)
        # (4,)

        speed_clip = self._to_numpy(lead_clip["speed"])
        # (4,)  range: [7.5151e-06, 7.9911]

        tp_clip = self._to_numpy(lead_clip["target_point"])
        # (4, 2)  range: [-0.000894547, 10.5777]

        ntp_clip = self._to_numpy(lead_clip["target_point_next"])
        # (4, 2)  range: [-0.000629115, 25.0991]

        t = group.anchor_t

        # 1) RGB 多帧（4 帧）
        rgb_pil_list: list[Image.Image] = []
        for idx in group.rgb_indices_asc:
            rgb_i = rgb_clip[idx]
            # (384, 1152, 3)

            rgb_hwc = self._ensure_hwc_uint8(rgb_i)
            # (384, 1152, 3)  range: [0, 239]
            # 直接使用 LEAD 的三视角拼接图（前/左前/右前）。
            rgb_pil_list.append(Image.fromarray(rgb_hwc, mode="RGB"))
        
        # 2) LiDAR/BEV 多帧采样与融合（与 RGB 一样支持步长和帧数）
        bev_indices_desc = [max(t - i * bev_frame_step, 0) for i in range(bev_frame_count)]
        bev_indices_asc = list(reversed(bev_indices_desc))

        # 与在线策略同步：先把历史帧点云统一对齐到“当前 anchor 帧自车局部坐标系”，再栅格化。
        anchor_pos_xy = np.asarray(pos_global_clip[t], dtype=np.float32).reshape(-1)[:2]

        anchor_theta = float(np.asarray(theta_clip[t]).reshape(-1)[0])

        aligned_chunks: list[np.ndarray] = []

        for idx in bev_indices_asc:
            pts = np.asarray(lidar_points_clip[idx], dtype=np.float32)
            # 默认 anchor=12, bev_count=1, bev_step=1 -> bev_indices_asc=[3]（clip 内 idx，仅 anchor 单帧）
            # Frame 3: original lidar_points shape ≈ (N, 3), dtype: float32  # N 因帧而异
            
            if pts.ndim != 2 or pts.shape[1] < 3:
                raise ValueError(f"lidar_points[{idx}] shape invalid: {pts.shape}")
            pts = pts[:, :3]

            src_pos_xy = np.asarray(pos_global_clip[idx], dtype=np.float32).reshape(-1)[:2]
            src_theta = float(np.asarray(theta_clip[idx]).reshape(-1)[0])
            pts_aligned = self._align_lidar_points_to_anchor(
                points_xyz=pts,
                src_pos_xy=src_pos_xy,
                src_theta=src_theta,
                anchor_pos_xy=anchor_pos_xy,
                anchor_theta=anchor_theta,
            )
            aligned_chunks.append(pts_aligned)

        fused_points = (
            np.concatenate(aligned_chunks, axis=0) if aligned_chunks else np.zeros((0, 3), dtype=np.float32)
        )
        #  shape=(35926, 3), x_range=(-125.073,111.427), y_range=(-39.984,40.016), z_range=(-1.070,10.030)
        # (N, 3), N 因帧而异(单帧 anchor 默认 ~3.3e4-3.6e4)
        # LEAD 风格栅格化:±40m × [-32, 64]m / 4 px/m / z 过滤 [-4, 10] 闭区间(含地面),
        # 直接出 (320, 384) 单通道直方图,与 LEAD BEV encoder 训练分布完全一致。
        lidar_i = lead_rasterize_lidar(fused_points, self.bev_encoder_config)
        # shape: (320, 384), float32, range=[0, 1]

        lidar_bev_rgb = self._lead_lidar_to_rgb_bev(lidar_i)
        # shape: (320, 384, 3), dtype: uint8, range=[0, 255]

        lidar_pil_list = [Image.fromarray(lidar_bev_rgb, mode="RGB")]
        # PIL lidar 仅作调试日志:在线/离线的 InterleaveInferencer.__call__ 都把 lidar 参数注释掉了,
        # 不会进慢路径 prompt,所以这里的 PIL 内容不影响推理。

        # 3) target_point_speed
        speed = float(np.asarray(speed_clip[t]).reshape(-1)[0])
        tp = np.asarray(tp_clip[t]).reshape(-1)
        ntp = np.asarray(ntp_clip[t]).reshape(-1)
        target_point_speed = torch.tensor(
            [[speed, float(tp[0]), float(tp[1]), float(ntp[0]), float(ntp[1])]],
            dtype=torch.float32,
            device=self.device,
        )

        # 4) BEV encoder 输入(LEAD 风格,直接喂三视角拼接 RGB,不做 crop):
        # - RGB: LEAD 训练就是 (3, 384, 1152) 三视角横向拼接,backbone 内部 normalize_imagenet
        # - LiDAR: (1, 320, 384) 单通道 [0, 1],与 lead/data_loader/carla_dataset_utils.py
        #   rasterize_lidar 输出完全一致
        # LEAD .jpg 已是 1 次 JPEG 压缩,这里直接用 PIL 解码结果,不做二次 encode。
        bev_rgb = np.array(rgb_pil_list[-1], dtype=np.uint8)
        # (384, 1152, 3) uint8 RGB, range≈[0, 234]

        bev_rgb = np.transpose(bev_rgb, (2, 0, 1))
        bev_rgb_tensor = torch.from_numpy(bev_rgb).float().unsqueeze(0).to(self.device)
        # torch.Size([1, 3, 384, 1152]), float32,range=[0, ~235]
        # 不主动 .to(bf16):LEAD backbone 内部会按 config.torch_float_type(float32) cast,
        # 强行 bf16 会绕一圈无收益,且与 LEAD 训练时 normalize_imagenet 数值不一致。

        bev_lidar_1ch = self._lead_lidar_to_bev_encoder_channel(lidar_i)
        # shape: (1, 320, 384) float32 [0, 1]

        bev_lidar_tensor = torch.from_numpy(bev_lidar_1ch).float().unsqueeze(0).to(self.device)
        # torch.Size([1, 1, 320, 384]), float32

        # [_prepare_inference_inputs Return Values Stats]
        # - rgb_pil_list: list[4], first image size=(1152, 384), mode=RGB
        # - lidar_pil_list: list[1], first image size=(384, 320), mode=RGB  (PIL.size = W×H)
        # - target_point_speed: shape=(1, 5), dtype=float32, range≈[0, 25]
        # - bev_rgb_tensor: torch.Size([1, 3, 384, 1152]), float32, range≈[0, 235]
        # - bev_lidar_tensor: torch.Size([1, 1, 320, 384]), float32, range=[0, 1]
        # - bev_indices_desc: [3]   # anchor=12, bev_count=1, 仅 anchor 单帧(clip 内 idx)
        # - bev_indices_asc: [3]
        return rgb_pil_list, lidar_pil_list, target_point_speed, bev_rgb_tensor, bev_lidar_tensor, bev_indices_desc, bev_indices_asc

    @torch.no_grad()
    def run_step(self, lead_clip: dict[str, Any], anchor_t: int,
                 gen_context=None, timestamp: float = 0.0,
                 rgb_frame_step: int = 1, rgb_frame_count: int = 4,
                 bev_frame_step: int = 1, bev_frame_count: int = 1,
                 enable_fast_inference: bool = False,
                 enable_leadmot_planning: bool = False) -> dict[str, Any]:
        """离线版 run_step:输入 LEAD clip + anchor 帧,输出慢推理 gen_context(默认)
        或慢+快推理 text/traj/route。

        参数:
        - rgb_frame_step / rgb_frame_count: RGB 历史采样,默认 1 帧步长 × 4 帧
        - bev_frame_step / bev_frame_count: BEV/LiDAR 历史采样,默认 1 步 × 1 帧
          (LEAD 单帧 .laz 已含 5 累积 sweep,无需跨帧再拼)
        - enable_fast_inference: 是否走 AutoMoT 快推理路径。
          默认 False —— 当前 BEV encoder 已替换为 LEAD TransfuserBackbone,
          输出 trans_feat shape=(1, 512, 10, 12),与 AutoMoT 自家 bev_encoder_proj
          期望的 (1, 1512, 8, 8) 不兼容,启用快推理会 shape mismatch 崩溃。
          快推理 LEAD 版需要重新设计 projector/head/queries,见 PROJECT_CONTEXT.md §12。
          想跑快推理时(为了 debug/对照),手动传 True 让 ipath 走通(预期会报错)。

        关键流程:
        1. 准备数据 (LEAD 风格 BEV/LiDAR 输入,多帧 RGB)
        2. 调用 LEAD BEV encoder 得到 trans_feat(本身只是为了 keep 接口完整,
           当 enable_fast_inference=False 时实际不被消费,所以即便随机权重也无碍)
        3. 若 gen_context 为空 → 走慢推理一次,初始化 KV cache
        4. 可选:enable_fast_inference=True 时再走快推理路径
        """
        group = self._build_group_indices(
            clip_len=int(self._to_numpy(lead_clip["rgb"]).shape[0]),
            anchor_t=anchor_t,
            rgb_frame_step=rgb_frame_step,
            rgb_frame_count=rgb_frame_count
        )

        rgb_pil_list, lidar_pil_list, target_point_speed, bev_rgb_tensor, bev_lidar_tensor, bev_indices_desc, bev_indices_asc = self._prepare_inference_inputs(
            lead_clip, group, bev_frame_step=bev_frame_step, bev_frame_count=bev_frame_count
        )

        prompt_cleaned, understanding_output, reasoning_output = build_cleaned_prompt_and_modes(target_point_speed)

        # LEAD BEV encoder 前向(单帧 transfuser 框架)。
        # 输入: bev_rgb_tensor (1, 3, 384, 1152) [0, 235], bev_lidar_tensor (1, 1, 320, 384) [0, 1]
        # 输出: {bev_feature: (1, 512, 10, 12) [-2.79, 11.89], image_feature_grid: (1, 512, 12, 36) [-4.53, 50.80]}
        # 快推理禁用时这个输出不被消费,但保留 forward 调用以验证 backbone shape。
        with torch.no_grad():
            bev_encoder_output = self.bev_encoder(
                rgb=bev_rgb_tensor,
                lidar_bev=bev_lidar_tensor,
            )
        trans_feat = bev_encoder_output["bev_feature"]  # (1, 512, 10, 12) LEAD 风格

        # ========== 慢推理(Qwen3-VL frozen) ==========
        if gen_context is None:
            slow_input_lists = rgb_pil_list + [prompt_cleaned]
            # shapes: [(384, 1152, 3), (384, 1152, 3), (384, 1152, 3), (384, 1152, 3), 'str(len=214)']
            # input term is: <PIL.Image.Image image mode=RGB size=1152x384 at 0x7F0213313E80>
            # input term is: <PIL.Image.Image image mode=RGB size=1152x384 at 0x7F0213313D90>
            # input term is: <PIL.Image.Image image mode=RGB size=1152x384 at 0x7F0213313C40>
            # input term is: <PIL.Image.Image image mode=RGB size=1152x384 at 0x7F0213313D60>
            # input term is: Your current and next target point is (9.696953, 0.001055), (26.754944, 0.004383), 
            # and your current velocity is 6.39 m/s. Predict the driving actions ( now, +1s, +2s) and 
            # plan the trajectory for the next 3 seconds.
            
            gen_context = self.inferencer.kv_cache_fixed_inference(slow_input_lists)

        # ========== 快推理(AutoMoT 自家训练的下游 head) ==========
        # 默认禁用:LEAD trans_feat (1, 512, 10, 12) 与 AutoMoT bev_encoder_proj 期望
        # (1, 1512, 8, 8) shape 不兼容。要彻底启用 LEAD 版快推理,需重训整个 decoder 链路,
        # 见 PROJECT_CONTEXT.md §12 "未来工作"。
        # ============================================================
        # LEAD-MoT 快推理路径（frozen Qwen prefix K/V + LEAD BEV）
        # ============================================================
        # 这条路是 opt-in 的：必须用户传 --enable-leadmot-planning 才会跑。
        # 默认关闭原因：
        # 1. decoder 是架构层组装，权重未训练，预测值无意义（随机输出）
        # 2. 需要加载 checkpoint 才能产生真实预测（TODO：等训完接 --leadmot-ckpt）
        # 3. 即使不出预测，跑一遍 forward 也会占用 ~300MB bf16 显存 + 计算
        #
        # 调用方负责：
        # - 用 standalone LocalQwen3VLInstructEngine 单独 prefill 拿 past_key_values
        # - 提供 LEAD BEV (B,512,10,12) 和 ego status (B,5)
        # 本节负责：
        # - 池化 Qwen K/V 36 层 -> 12 段 prefix K/V
        # - 调 decoder forward 拿到 pred_route / pred_future_waypoints
        # - 打印 shape 方便调试

        # 三个输出字段先置 None，enable_leadmot_planning=False 时直接保持 None
        # 调用方拿到 None 就知道这条路没跑
        leadmot_route = None
        leadmot_future_waypoints = None
        leadmot_gen_hidden = None

        if enable_leadmot_planning:
            # ---- 首次进入时 lazy 构建 standalone Qwen engine 和 decoder ----
            # 两个 lazy build 都是幂等的（已加载就跳过），重复调用零成本
            # standalone Qwen 引擎专用于 leadmot 路径，与 AutoMoT InterleaveInferencer
            # 共用一个 device 但不共享权重（前者 = Qwen3-VL-4B-Instruct，
            # 后者 = AutoMoT checkpoint 里的 Qwen/MoT 路径）
            self._ensure_leadmot_qwen_engine()
            self._ensure_leadmot_decoder()

            # ---- 用 standalone Qwen 独立跑 prefill 拿 cache（cache 同源关键步骤）----
            # 不复用 gen_context["past_key_values"]——那是 AutoMoT InterleaveInferencer
            # 用 AutoMoT MoT checkpoint 算出来的，与 leadmot 训练时的 standalone Qwen 不同源
            # （主干同为 Qwen3-VL-4B 但权重已被 AutoMoT 训练改写，K/V 分布不同）。
            # 这里独立跑一次 standalone Qwen prefill，保证训练/推理 cache 来源一致。
            # 代价：每次 leadmot step 多跑一次 Qwen prefill（数百毫秒级），可接受。
            leadmot_past_key_values = self._run_leadmot_qwen_prefill(
                rgb_pil_list=rgb_pil_list,
                user_prompt=prompt_cleaned,
            )

            # ---- 池化 Qwen K/V：36 层 -> 12 段 prefix K/V ----
            # select_last 模式：每段取最后一层（语义最丰富、显存最省）
            # 返回 list[(K, V)] 长度 = 12，每个 K/V 形状 (B, 8, S, 128)
            pooled_kv = _segment_qwen_cache_for_leadmot(
                leadmot_past_key_values,
                self.leadmot_config,
            )

            # target_point_speed 是 _prepare_inference_inputs 里组好的 5 维状态：
            #   [speed, target_point_x, target_point_y, next_target_point_x, next_target_point_y]
            # 这里只做 device/dtype 对齐，不做数值归一化：
            # - decoder 默认是 bf16，裸 Linear 不会自动接收 fp32 输入；
            # - speed/tp/ntp 仍保持原始米制/秒制值，符合 AutoMoT status token 做法。
            status = target_point_speed.to(self.device, dtype=self.leadmot_dtype)

            # LeadMoTPlanningDecoder.forward 做四件事：
            # 1. BEV:    (B,512,10,12) -> 120 个 1024 维 token；
            # 2. Status: speed/tp/ntp -> 3 个 1024 维 token；
            # 3. Query:  route 10 个 + waypoint 8 个可学 query；
            # 4. Blocks: 141 个 gen token 逐层读 pooled_kv，最后 Linear+cumsum 出 LEAD 轨迹契约。
            leadmot_out = self.leadmot_decoder(
                pooled_kv=pooled_kv,
                bev=trans_feat,
                speed=status[:, 0],
                target_point=status[:, 1:3],
                target_point_next=status[:, 3:5],
            )

            # 三个输出都保持 GPU tensor 到 return 前再统一 detach：
            # - pred_route:            LEAD route 契约 (B,10,2)，空间路径点；
            # - pred_future_waypoints: LEAD future_waypoints 契约 (B,8,2)，4Hz 2s 轨迹；
            # - gen_hidden:            debug/可视化用的最终 141 token hidden，不是控制必需。
            leadmot_route = leadmot_out["pred_route"]
            leadmot_future_waypoints = leadmot_out["pred_future_waypoints"]
            leadmot_gen_hidden = leadmot_out["gen_hidden"]
            print("[LeadMoT] prefix-KV planning success (decoder is untrained unless checkpoint is loaded)")
            print(f"  leadmot_route_shape: {tuple(leadmot_route.shape)}")
            print(f"  leadmot_future_waypoints_shape: {tuple(leadmot_future_waypoints.shape)}")

        gen_text = None
        gen_traj = None
        route = None
        reasoning_hidden_states = None
        if enable_fast_inference:
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                gen_text, gen_traj, route, reasoning_hidden_states = self.inferencer.based_kv_cache_context_fast_qwen3vl_dp(
                    trans_feat=trans_feat,
                    gen_context=gen_context,
                    reasoning_tokens=getattr(self.automot.config, 'reasoning_query_tokens', 8),
                    action_tokens=getattr(self.automot.config, 'action_query_tokens', 26),
                    v_target_point=target_point_speed,
                )
            traj_shape = tuple(gen_traj.shape) if isinstance(gen_traj, torch.Tensor) else None
            route_shape = tuple(route.shape) if isinstance(route, torch.Tensor) else None
            rhs_shape = (
                tuple(reasoning_hidden_states.shape)
                if isinstance(reasoning_hidden_states, torch.Tensor)
                else None
            )
            print("[based_kv_cache_context_fast_qwen3vl_dp] success (LEAD trans_feat 路径,可能与原训练分布不一致)")
            print(f"  text: {str(gen_text)[:200]}")
            print(f"  traj_shape: {traj_shape}")
            print(f"  route_shape: {route_shape}")
            print(f"  reasoning_hidden_states_shape: {rhs_shape}")
        else:
            print("[run_step] enable_fast_inference=False:跳过快推理,仅返回慢推理 gen_context")

        # 返回 dict 里所有 tensor 字段都 detach 到 CPU，避免多 clip 收集 outputs 时
        # 持续挂 GPU 引用累积显存。gen_context 例外：它含 past_key_values，下一帧
        # 慢推理还要直接复用同设备的 cache。
        # _detach_cpu_float : 轨迹/路径这类小张量统一转 fp32，省得下游算 L1/ADE/numpy
        #                      可视化时到处补 .float()
        # _detach_cpu       : 大张量保留原 dtype（通常 bf16），下游需要 fp32 自己转
        def _detach_cpu_float(x: Any) -> Any:
            return x.detach().float().cpu() if isinstance(x, torch.Tensor) else x

        def _detach_cpu(x: Any) -> Any:
            return x.detach().cpu() if isinstance(x, torch.Tensor) else x

        return {
            "timestamp": timestamp,
            "anchor_t": anchor_t,
            "rgb_indices_desc": group.rgb_indices_desc,
            "rgb_indices_asc": group.rgb_indices_asc,
            "bev_indices_desc": bev_indices_desc,
            "bev_indices_asc": bev_indices_asc,
            "prompt": prompt_cleaned,
            "text": gen_text,
            "traj": _detach_cpu_float(gen_traj),
            "route": _detach_cpu_float(route),
            "leadmot_route": _detach_cpu_float(leadmot_route),
            "leadmot_future_waypoints": _detach_cpu_float(leadmot_future_waypoints),
            "leadmot_gen_hidden": _detach_cpu(leadmot_gen_hidden),     # bf16, debug/扩展用
            # trans_feat 当前转 CPU 仅供 shape/debug 用，没有下游 GPU 消费方。
            # 若未来 run_step 后还要接别的 GPU head（如新 BEV planning），需把这一行
            # 改成保留 GPU 引用（去掉 _detach_cpu 包装），并自行管理显存释放。
            "trans_feat": _detach_cpu(trans_feat),                     # LEAD BEV (1, 512, 10, 12)
            "gen_context": gen_context,                                # GPU 引用保留,下一帧 cache 复用
        }

    @torch.no_grad()
    def run_clip(self, lead_clip: dict[str, Any],
                 rgb_frame_step: int = 1, rgb_frame_count: int = 4,
                 bev_frame_step: int = 1, bev_frame_count: int = 1,
                 enable_leadmot_planning: bool = False) -> list[dict[str, Any]]:
        """
        处理整段 LEAD clip，生成单组推理结果（基于最后一帧）。

        参数说明：
        - rgb_frame_step: RGB 采样间隔（默认 1 对应 LEAD 的 0.25s，5 对应 AutoMoT 的 1.25s）
        - rgb_frame_count: RGB 采样帧数（默认 4）
        - bev_frame_step: BEV 采样间隔（默认 1）
        - bev_frame_count: BEV 采样帧数（默认 1，对齐 LEAD 单帧 .laz 含 5 sweep 的训练分布）
        
        返回包含单个推理结果的列表。
        """

        # [Clip Stats]
        #     - rgb: shape=(4, 384, 1152, 3), dtype=uint8, range=[0, 255]
        #     - lidar_points: list[4] (变长), dtype=float32, points/frame(min/max/total)=33763/35926/138854, range=[-125.073, 117.508]
        #     - pos_global: shape=(4, 2), dtype=float32, range=[88.7583, 229.458]
        #     - theta: shape=(4,), dtype=float32, range=[1.59468, 1.59501]
        #     - speed: shape=(4,), dtype=float32, range=[6.38579, 7.9911]
        #     - target_point: shape=(4, 2), dtype=float32, range=[0.000619971, 10.4444]
        #     - target_point_next: shape=(4, 2), dtype=float32, range=[0.00338461, 26.7549]
    
        clip_len = int(self._to_numpy(lead_clip["rgb"]).shape[0])
        
        # 以 clip 最后一帧作为 anchor（对应当前时刻）。
        # 注意:anchor_t 是 clip 内局部索引(0..clip_len-1),不是 route 内全局帧号;
        # 全局帧号见 build_clip_from_real_lead_route 打印的 "[load] anchor=... reading frames [a, b]"。
        anchor_t = clip_len - 1
        print(f"Processing clip-local anchor_t={anchor_t} (clip last frame, clip_len={clip_len})...")
        
        out = self.run_step(
            lead_clip=lead_clip, 
            anchor_t=anchor_t,
            gen_context=None,
            timestamp=0.25 * anchor_t,
            rgb_frame_step=rgb_frame_step,
            rgb_frame_count=rgb_frame_count,
            bev_frame_step=bev_frame_step,
            bev_frame_count=bev_frame_count,
            enable_leadmot_planning=enable_leadmot_planning
        )
        
        outputs: list[dict[str, Any]] = []
        gen_context = out.pop("gen_context")  # 提取缓存供下一次使用（预留接口）
        outputs.append(out)

        # 打印最终缓存上下文（符号名按用户要求）
        print("#sym:gen_context")
        if isinstance(gen_context, dict):
            print(f"  keys: {list(gen_context.keys())}")
            kv_lens = gen_context.get("kv_lens", None)
            ropes = gen_context.get("ropes", None)
            pkv = gen_context.get("past_key_values", None)
            print(f"  kv_lens: {kv_lens}")
            print(f"  ropes: {ropes}")
            print(f"  past_key_values_type: {type(pkv).__name__ if pkv is not None else None}")
            try:
                print(f"  past_key_values_layers: {len(pkv)}")
            except Exception:
                pass
        else:
            print(f"  value: {gen_context}")
        
        return outputs


def _print_clip_tensor_stats(clip: dict[str, Any]) -> None:
    """打印关键字段的 shape、dtype 与数值范围，便于快速核对输入数据。"""
    keys = ["rgb", "lidar_points", "pos_global", "theta",
            "speed", "target_point", "target_point_next"]
    print("\n[Clip Stats]")
    for k in keys:
        if k not in clip:
            print(f"  - {k}: <missing>")
            continue

        val = clip[k]

        # 特殊处理：变长列表（如逐帧点云 lidar_points），各元素 shape 不一致，
        # 直接 np.asarray 会报 inhomogeneous shape，需要逐帧统计。
        if isinstance(val, (list, tuple)):
            elems = [np.asarray(LeadOfflineMoTRunner._to_numpy(v)) for v in val]
            if len(elems) == 0:
                print(f"  - {k}: list[0] (empty)")
                continue
            shapes = [e.shape for e in elems]
            if len(set(shapes)) > 1:
                # 变长：逐帧报告点数 + 全局数值范围
                dtype_str = str(elems[0].dtype)
                counts = [s[0] if len(s) > 0 else 0 for s in shapes]
                nonempty = [e.reshape(-1) for e in elems if e.size > 0]
                all_vals = np.concatenate(nonempty) if nonempty else np.array([], dtype=elems[0].dtype)
                rng = (
                    f"[{float(all_vals.min()):.6g}, {float(all_vals.max()):.6g}]"
                    if all_vals.size > 0 else "[empty]"
                )
                print(
                    f"  - {k}: list[{len(elems)}] (变长), dtype={dtype_str}, "
                    f"points/frame(min/max/total)={min(counts)}/{max(counts)}/{sum(counts)}, "
                    f"range={rng}"
                )
                continue
            # 同构列表，可以正常堆叠
            arr = np.stack(elems, axis=0)
        else:
            arr = np.asarray(LeadOfflineMoTRunner._to_numpy(val))

        shape_str = str(tuple(arr.shape))
        dtype_str = str(arr.dtype)

        if arr.size == 0:
            print(f"  - {k}: shape={shape_str}, dtype={dtype_str}, range=[empty]")
            continue

        if np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_:
            arr_min = np.nanmin(arr)
            arr_max = np.nanmax(arr)
            print(
                f"  - {k}: shape={shape_str}, dtype={dtype_str}, "
                f"range=[{float(arr_min):.6g}, {float(arr_max):.6g}]"
            )
        else:
            print(f"  - {k}: shape={shape_str}, dtype={dtype_str}, range=[non-numeric]")

def _extract_tp_ntp_from_future_frames(
    *,
    current_meta: dict[str, Any],
    current_frame_idx: int,
    total_frames: int,
    meta_dir: pathlib.Path,
    meta_cache: dict[int, dict[str, Any]],
    tp_lookahead_s: float,
    ntp_lookahead_s: float,
    frame_interval_s: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """按未来帧真值位置生成 target_point 与 target_point_next（ego 坐标）。

    规则：
    - tp 取 t + round(tp_lookahead_s / dt) 帧的真值位置
    - ntp 取 t + round(ntp_lookahead_s / dt) 帧的真值位置
    - 超出末帧时 clamp 到最后一帧
    - 使用当前帧 (pos_global, theta) 做 inverse_conversion_2d
    """

    def _load_meta(frame_idx: int) -> dict[str, Any]:
        if frame_idx in meta_cache:
            return meta_cache[frame_idx]
        p = meta_dir / f"{frame_idx:04d}.pkl"
        if not p.exists():
            raise FileNotFoundError(f"missing future meta file: {p}")
        with lzma.open(p, "rb") as f:
            m = pickle.load(f)
        meta_cache[frame_idx] = m
        return m

    cur_pos = np.asarray(current_meta.get("pos_global", [0.0, 0.0]), dtype=np.float32).reshape(-1)
    if cur_pos.shape[0] < 2:
        cur_pos = np.array([0.0, 0.0], dtype=np.float32)
    cur_pos = cur_pos[:2].astype(np.float32)
    # Current position (global): [229.65828  80.57458]

    cur_theta = float(np.asarray(current_meta.get("theta", 0.0), dtype=np.float32).reshape(-1)[0])
    # Current theta: 1.595121 rad

    dt = max(1e-6, float(frame_interval_s))
    tp_offset = int(round(float(tp_lookahead_s) / dt))
    ntp_offset = int(round(float(ntp_lookahead_s) / dt))
    tp_idx = min(max(0, current_frame_idx + max(0, tp_offset)), total_frames - 1)
    ntp_idx = min(max(0, current_frame_idx + max(0, ntp_offset)), total_frames - 1)

    tp_meta = _load_meta(tp_idx)
    ntp_meta = _load_meta(ntp_idx)

    tp_world = np.asarray(tp_meta.get("pos_global", [0.0, 0.0]), dtype=np.float32).reshape(-1)
    # Loaded TP meta for frame 6: pos_global=[2.29582611e+02 8.36773529e+01 1.07219234e-01], theta=1.594943686327662    

    ntp_world = np.asarray(ntp_meta.get("pos_global", [0.0, 0.0]), dtype=np.float32).reshape(-1)
    # Loaded NTP meta for frame 12: pos_global=[229.32587     94.25194      0.23421314], theta=1.5950125892970224

    if tp_world.shape[0] < 2:
        tp_world = np.array([0.0, 0.0], dtype=np.float32)
    if ntp_world.shape[0] < 2:
        ntp_world = np.array([0.0, 0.0], dtype=np.float32)
    tp_world = tp_world[:2].astype(np.float32)
    ntp_world = ntp_world[:2].astype(np.float32)

    tp_ego = inverse_conversion_2d(tp_world, cur_pos, cur_theta)
    ntp_ego = inverse_conversion_2d(ntp_world, cur_pos, cur_theta)
    # Computed TP (ego): [3.10369810e+00 1.80789903e-04]
    # Computed NTP (ego): [ 1.36813994e+01 -3.43967825e-04]
    return np.asarray(tp_ego, dtype=np.float32), np.asarray(ntp_ego, dtype=np.float32)


def _extract_pose_from_meta(meta: dict[str, Any]) -> tuple[np.ndarray, float]:
    """从 LEAD meta 提取全局位置与航向，用于跨帧 LiDAR 对齐与 tp/ntp 转 ego。

    严格对齐 LEAD 训练默认配置（use_noisy_tp=False）：
    - 位置：固定用 `pos_global`（真值），不回退到 filtered / noisy
    - 朝向：固定用 `theta`（compass 经 normalize_angle + unwrap）
    缺失任一字段视为数据异常直接 raise。
    """
    if "pos_global" not in meta:
        raise KeyError("meta 缺少 pos_global 字段；LEAD 训练默认走真值位姿，请检查数据完整性")
    arr = np.asarray(meta["pos_global"], dtype=np.float32).reshape(-1)
    if arr.shape[0] < 2:
        raise ValueError(f"pos_global 维度不足 2: shape={arr.shape}")
    pos_xy = arr[:2].astype(np.float32)

    if "theta" not in meta:
        raise KeyError("meta 缺少 theta 字段")
    theta = float(np.asarray(meta["theta"], dtype=np.float32).reshape(-1)[0])

    return pos_xy, theta


def build_clip_from_real_lead_route(
    route_dir: str,
    anchor: int = 12,
    rgb_frame_step: int = 1,
    rgb_frame_count: int = 4,
    bev_frame_step: int = 1,
    bev_frame_count: int = 1,
    tp_lookahead_s: float = 1.5,
    ntp_lookahead_s: float = 3.0,
    frame_interval_s: float = 0.25,
) -> dict[str, Any]:
    """
    从真实 LEAD route 目录构造 runner 所需 clip。

    新语义：显式指定 anchor（route 内的绝对帧索引，0-based），由采样参数反推
    需要读取的最早历史帧。clip 内 anchor 永远是最后一帧。

    参数：
    - anchor: 待处理的 anchor 帧索引（route 内绝对索引）。
    - rgb_frame_step / rgb_frame_count: RGB 历史采样步长与帧数。
    - bev_frame_step / bev_frame_count: BEV/LiDAR 历史采样步长与帧数。
    - tp_lookahead_s / ntp_lookahead_s / frame_interval_s: target_point 未来帧设置。

    行为：
    - 计算 max_history = max((rgb_count-1)*rgb_step, (bev_count-1)*bev_step)
    - ideal_start = anchor - max_history；若 < 0 则 clamp 到 0 并 warning（补 0 行为
      由 _build_group_indices 内的 max(..., 0) 完成，会重复 frame 0 的数据）。
    - 校验 anchor 必须落在 [0, total_frames-1] 内。

    目录要求：
    - rgb/*.jpg
    - metas/*.pkl (xz 压缩 pickle)
    - lidar/*.laz
    """
    route = pathlib.Path(route_dir)
    if not route.exists():
        raise FileNotFoundError(f"route_dir not found: {route_dir}")
    if laspy is None:
        raise ImportError("laspy is required to read LEAD lidar .laz files")

    rgb_dir = route / "rgb"
    meta_dir = route / "metas"
    lidar_dir = route / "lidar"
    for p in (rgb_dir, meta_dir, lidar_dir):
        if not p.exists():
            raise FileNotFoundError(f"missing subdir in route: {p}")

    rgb_files = sorted(rgb_dir.glob("*.jpg"))
    if not rgb_files:
        raise FileNotFoundError(f"no rgb frames found in {rgb_dir}")

    total_frames = len(rgb_files)

    # 校验 anchor 合理性
    if anchor < 0:
        raise ValueError(f"anchor 必须 >= 0，当前 anchor={anchor}")
    if anchor >= total_frames:
        raise ValueError(
            f"anchor={anchor} 超出 route 范围（route 总帧数={total_frames}，"
            f"合法范围 [0, {total_frames - 1}]）"
        )

    # 根据采样参数反推需要读取的最早历史帧
    max_history = max(
        (max(1, rgb_frame_count) - 1) * max(1, rgb_frame_step),
        (max(1, bev_frame_count) - 1) * max(1, bev_frame_step),
    )
    ideal_start = anchor - max_history
    if ideal_start < 0:
        pad_count = -ideal_start
        print(
            f"[警告] anchor={anchor} 历史不足：需要 {max_history} 帧历史，"
            f"但 route 起点仅到 frame 0，将通过重复 frame 0 补 {pad_count} 次（"
            f"补 0 数据会有重复，可能略偏离训练分布）"
        )
        actual_start = 0
    else:
        actual_start = ideal_start

    print(
        f"[load] anchor={anchor}, max_history={max_history}, "
        f"reading frames [{actual_start}, {anchor}] ({anchor - actual_start + 1} frames), "
        f"total_route_frames={total_frames}"
    )

    rgb_list = []
    speed_list = []
    tp_list = []
    ntp_list = []
    pos_global_list = []
    theta_list = []
    lidar_points_list = []
    meta_cache: dict[int, dict[str, Any]] = {}

    for i in range(actual_start, anchor + 1):
        stem = f"{i:04d}"
        rgb_path = rgb_dir / f"{stem}.jpg"
        meta_path = meta_dir / f"{stem}.pkl"
        lidar_path = lidar_dir / f"{stem}.laz"

        if not rgb_path.exists() or not meta_path.exists() or not lidar_path.exists():
            raise FileNotFoundError(f"missing frame assets at index {i}")

        # RGB
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"failed to read image: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_list.append(rgb)

        # meta (xz compressed pickle)
        with lzma.open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_cache[i] = meta
        speed = float(meta.get("speed", 0.0))
        # 使用 route-time 模式：按未来帧真值位置生成目标点
        tp, ntp = _extract_tp_ntp_from_future_frames(
            current_meta=meta,
            current_frame_idx=i,
            total_frames=len(rgb_files),
            meta_dir=meta_dir,
            meta_cache=meta_cache,
            tp_lookahead_s=tp_lookahead_s,
            ntp_lookahead_s=ntp_lookahead_s,
            frame_interval_s=frame_interval_s,
        )
        pos_xy, theta = _extract_pose_from_meta(meta)
        # pos_global=[7.713916e-02 8.248221e+01], theta=1.595

        speed_list.append(speed)
        tp_list.append(tp)
        ntp_list.append(ntp)
        pos_global_list.append(pos_xy)
        theta_list.append(theta)

        # 读取 LAZ 点云。
        # 坐标系说明：LEAD 数据录制时已将传感器坐标转换为 ego-local（即自车局部坐标系），
        # 因此 LAZ 文件直接存储的就是 ego-local 坐标，此处无需任何 sensor->ego 转换。
        # 这与 LEAD 训练数据集加载器（carla_dataset_video.py）的行为一致——
        # 训练时也是直接读取 LAZ 后用于 rasterize_lidar，不做二次坐标转换。
        # 真正送入模型的多帧 LiDAR，会在 _prepare_inference_inputs 中统一对齐到
        # 当前 anchor 帧自车局部坐标系（anchor ego-local）后再栅格化。
        las = laspy.read(str(lidar_path))
        pts = np.stack([las.x, las.y, las.z], axis=1).astype(np.float32)
        # (34890, 3)
        
        lidar_points_list.append(pts)

    clip = {
        "rgb": np.stack(rgb_list, axis=0),
        "lidar_points": lidar_points_list,
        "pos_global": np.stack(pos_global_list, axis=0).astype(np.float32),
        "theta": np.asarray(theta_list, dtype=np.float32),
        "speed": np.asarray(speed_list, dtype=np.float32),
        "target_point": np.stack(tp_list, axis=0).astype(np.float32),
        "target_point_next": np.stack(ntp_list, axis=0).astype(np.float32),
    }
    return clip


def _auto_select_gpu() -> str:
    """自动选择使用率最低的 GPU 设备。
    
    如果 nvidia-smi 不可用或没有 GPU，默认返回 'cuda:0'。
    """
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if output.returncode != 0:
            print("[警告] nvidia-smi 查询失败，使用默认 cuda:0")
            return "cuda:0"
        
        lines = output.stdout.strip().split("\n")
        min_usage = float("inf")
        best_gpu = 0
        
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    gpu_id = int(parts[0])
                    used = float(parts[1])
                    if used < min_usage:
                        min_usage = used
                        best_gpu = gpu_id
                except (ValueError, IndexError):
                    continue
        
        device = f"cuda:{best_gpu}"
        print(f"[自动检测] 选择 GPU: {device} (显存使用: {min_usage:.0f}MB)")
        return device
    except Exception as e:
        print(f"[警告] 自动检测 GPU 失败: {e}，使用默认 cuda:0")
        return "cuda:0"


def main():
    parser = argparse.ArgumentParser(
        description="LEAD 离线数据桥接到 AutoMoT 推理脚本（仅支持真实 route-dir 输入）"
    )
    parser.add_argument(
        "--route-dir",
        type=str,
        default='/datashare/IOL4SGH/data/data/Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46',
        help="真实 LEAD 路由目录，目录下需包含 rgb、metas、lidar 三个子目录。",
    )
    parser.add_argument(
        "--anchor",
        type=int,
        default=12,
        help="待处理的 anchor 帧索引（route 内绝对索引，0-based）。"
             "实际读取的历史范围由 rgb/bev 的 step/count 反推；anchor 必须 < route 总帧数。"
             "若历史不足（anchor 太靠前）会触发补 0（重复 frame 0），打印 warning 但不报错。",
    )
    parser.add_argument(
        "--rgb-frame-step",
        type=int,
        default=1,
        help="RGB 历史采样步长（单位: 帧）。默认 1 表示相邻帧（LEAD 每帧约 0.25s）；设为 5 约等于 1.25s 间隔（更接近原 AutoMoT 习惯）。",
    )
    parser.add_argument(
        "--rgb-frame-count",
        type=int,
        default=4,
        help="每次推理使用多少张历史 RGB 图像。默认 4，对应采样序列 [t, t-step, t-2*step, t-3*step]（不足会自动夹到 0）。",
    )
    parser.add_argument(
        "--bev-frame-step",
        type=int,
        default=1,
        help="BEV/LiDAR 历史采样步长（单位: 帧）。默认 1，表示连续相邻帧。",
    )
    parser.add_argument(
        "--bev-frame-count",
        type=int,
        default=1,
        help="每次推理融合多少帧 BEV/LiDAR。默认 1（对齐 LEAD 训练：单帧 .laz 已含 5 累积 sweep）；"
             "设为 2 会拼 [t-step, t] 共 10 sweep，密度×2 时间窗×2，偏离训练分布。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="推理设备。默认 auto 自动选择显存占用最低 GPU；也可手动指定如 cuda:7 或 cpu。",
    )
    parser.add_argument(
        "--enable-leadmot-planning",
        action="store_true",
        help="Enable experimental LEAD-MoT route/waypoint head from frozen Qwen prefix K/V + LEAD BEV.",
    )
    parser.add_argument(
        "--tp-lookahead-s",
        type=float,
        default=1.5,
        help="target_point 预测时长（秒）。默认 1.5。将按未来真值帧位置计算。",
    )
    parser.add_argument(
        "--ntp-lookahead-s",
        type=float,
        default=3.0,
        help="target_point_next 预测时长（秒）。默认 3.0。将按未来真值帧位置计算。",
    )
    parser.add_argument(
        "--frame-interval-s",
        type=float,
        default=0.25,
        help="数据帧时间间隔（秒）。用于把秒数换算成未来帧索引。默认 0.25。",
    )
    args = parser.parse_args()
    
    # 自动检测 GPU（如果指定了 'auto'）
    if args.device == "auto":
        args.device = _auto_select_gpu()

    clip = build_clip_from_real_lead_route(
        route_dir=args.route_dir,
        anchor=max(0, args.anchor),
        rgb_frame_step=max(1, args.rgb_frame_step),
        rgb_frame_count=max(1, args.rgb_frame_count),
        bev_frame_step=max(1, args.bev_frame_step),
        bev_frame_count=max(1, args.bev_frame_count),
        tp_lookahead_s=float(args.tp_lookahead_s),
        ntp_lookahead_s=float(args.ntp_lookahead_s),
        frame_interval_s=float(args.frame_interval_s),
    )
    print(f"Using real route dir: {args.route_dir}")
    print(f"Anchor frame: {args.anchor}")
    print(f"Using route-time mode (tp_lookahead_s={args.tp_lookahead_s}s, ntp_lookahead_s={args.ntp_lookahead_s}s)")

    # _print_clip_tensor_stats(clip)
    
    # [Clip Stats]
    #     - rgb: shape=(4, 384, 1152, 3), dtype=uint8, range=[0, 255]
    #     - lidar_points: list[4] (变长), dtype=float32, points/frame(min/max/total)=33763/35926/138854, range=[-125.073, 117.508]
    #     - pos_global: shape=(4, 2), dtype=float32, range=[88.7583, 229.458]
    #     - theta: shape=(4,), dtype=float32, range=[1.59468, 1.59501]
    #     - speed: shape=(4,), dtype=float32, range=[6.38579, 7.9911]
    #     - target_point: shape=(4, 2), dtype=float32, range=[0.000619971, 10.4444]
    #     - target_point_next: shape=(4, 2), dtype=float32, range=[0.00338461, 26.7549]
    
    runner = LeadOfflineMoTRunner(device=args.device)
    outputs = runner.run_clip(
        clip, 
        rgb_frame_step=max(1, args.rgb_frame_step),
        rgb_frame_count=max(1, args.rgb_frame_count),
        bev_frame_step=max(1, args.bev_frame_step),
        bev_frame_count=max(1, args.bev_frame_count),
        enable_leadmot_planning=bool(args.enable_leadmot_planning)
    )

    print(f"\nGenerated {len(outputs)} inference result(s)")
    for i, out in enumerate(outputs):
        leadmot_route_shape = (
            tuple(out["leadmot_route"].shape)
            if isinstance(out.get("leadmot_route"), torch.Tensor)
            else None
        )
        leadmot_waypoints_shape = (
            tuple(out["leadmot_future_waypoints"].shape)
            if isinstance(out.get("leadmot_future_waypoints"), torch.Tensor)
            else None
        )
        print(
            f"\n[Result {i}]  anchor_t={out['anchor_t']}, "
            f"rgb_frames={out['rgb_indices_asc']}, "
            f"bev_frames={out['bev_indices_asc']}, "
            f"text={str(out['text'])[:80]}..., "
            f"leadmot_route_shape={leadmot_route_shape}, "
            f"leadmot_waypoints_shape={leadmot_waypoints_shape}"
        )


if __name__ == "__main__":
    main()
