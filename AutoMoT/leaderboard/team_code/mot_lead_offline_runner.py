"""
LEAD video -> AutoMoT offline runner.

中文说明：
- 该文件是离线桥接实现，不修改原有在线 Agent。
- 目标：把 LEAD route 中的某个 anchor 帧（含必要历史窗口）切成 AutoMoT 所需的
  时序输入，复用 AutoMoT 现有模型初始化与推理链路。
- 入口语义：显式指定 anchor（route 内绝对帧索引），由采样参数反推需要的历史长度。
  clip 内 anchor 永远是最后一帧；当 anchor 太靠前导致历史不足时，会重复 frame 0
  补齐并打印 warning（不报错，但需注意数据有重复）。
- 关键采样规则：RGB 默认使用 [t, t-1, t-2, t-3]（按时间顺序喂入）。
- BEV/LiDAR：clip 只保存原始点云 (`lidar_points`)，栅格化在 `_prepare_inference_inputs`
  里完成 —— 跨帧对齐到 anchor ego-local 后调 AutoMoT 的 `lidar_to_histogram_features`
  (±32m / 4 px/m / z>0.2 切地面) 直接出 (1, 256, 256)，与训练分布一致。
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
import torch
from PIL import Image
from safetensors.torch import load_file
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
            _tmp_tokenizer = Qwen3Tokenizer.from_pretrained(str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B"), trust_remote_code=True)
            _tmp_tokenizer, _, _ = _add_special_tokens(_tmp_tokenizer)
            _automot_module_preset.tokenizer = _tmp_tokenizer
            print(f"✓ Pre-initialized tokenizer (fallback) from Qwen3-VL-4B")
        except Exception as e2:
            print(f"Error initializing tokenizer: {e2}")

from data.reasoning.data_utils import add_special_tokens
from mot.evaluation.inference import InterleaveInferencer
from mot.modeling.automot import AutoMoT
from mot.modeling.bev_encoder.backbone_extractor import BEVEncoderBackboneExtractor
import mot.modeling.bev_encoder.bev_encoder_utils as bev_encoder_t_u

from team_code.automot_utils import (
    InferenceArguments,
    ModelArguments,
    build_cleaned_prompt_and_modes,
    inverse_conversion_2d,
    load_model_mot,
)
from team_code.bev_data_utils import lidar_to_histogram_features


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
        tokenizer = AutoTokenizer.from_pretrained(model_args.qwen3vl_path)
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

        # 4) BEV encoder（用于 trans_feat）
        ckpt_dir = _AUTOMOT_ROOT / "checkpoints" / "AutoMoT"
        combined_ckpt = ckpt_dir / "model.safetensors"
        combined_sd = load_file(str(combined_ckpt))
        bev_state_dict = {
            k[len("bev_encoder.") :]: v
            for k, v in combined_sd.items()
            if k.startswith("bev_encoder.")
        }
        del combined_sd

        self.bev_encoder = BEVEncoderBackboneExtractor(
            config_path=str(ckpt_dir),
            device=str(self.device),
            state_dict=bev_state_dict,
        )
        del bev_state_dict

        self.bev_encoder.eval()
        self.bev_encoder = self.bev_encoder.to(torch.bfloat16)
        self.bev_encoder_config = self.bev_encoder.config

        print(f"✓ Model initialized on {self.device}")
        print("  - AutoMoT loaded")
        print("  - BEV Encoder loaded")
        print("  - Inferencer initialized")

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
        """
        把 (1,H,W) BEV 栅格转成 3 通道可视化 RGB 图（仅日志/调试用）。

        说明：
        - 输入：(1,256,256) float32 [0,1]，来自 lidar_to_histogram_features(config) 输出
        - 输出：(256,256,3) uint8，用于 PIL Image 显示（不送入模型）
        """
        arr = rasterized_lidar
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"rasterized_lidar shape invalid: {arr.shape}")
        # 值域已是 [0,1]，直接转 uint8
        arr_u8 = (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        bev = np.repeat(arr_u8[:, :, None], 3, axis=2)
        return bev

    @staticmethod
    def _lead_lidar_to_bev_encoder_channel(rasterized_lidar: np.ndarray) -> np.ndarray:
        """
        把栅格化结果透传给 BEV encoder（只做 shape/dtype 兜底）。

        输入：(1,H,W) float32 [0,1]。当前主链路使用 AutoMoT 风格栅格化
        (lidar_to_histogram_features)，H=W=256；不再需要任何 resize。
        输出：(1,H,W) float32 [0,1]，直接送入 BEVEncoderBackboneExtractor.forward()。
        """
        arr = rasterized_lidar
        # 统一为 (1, H, W) float32
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
        # (35926, 3)
        # 改用 AutoMoT 风格栅格化（±32m / 4 px/m / z>0.2 切地面），直接出 (1, 256, 256)。
        # 这样 BEV encoder 输出 trans_feat 自然是 (1, 1512, 8, 8)，与训练分布一致；
        # 同时无需再做 cv2.resize 二次插值。bev_encoder_config 来自 AutoMoT 自己的 config。
        lidar_i = lidar_to_histogram_features(fused_points, self.bev_encoder_config)
        # shape: (1, 256, 256), float32, range=[0, 1]

        lidar_bev_rgb = self._lead_lidar_to_rgb_bev(lidar_i)
        # shape: (256, 256, 3), dtype: uint8, range=[0, 255]

        lidar_pil_list = [Image.fromarray(lidar_bev_rgb, mode="RGB")]
        # PIL lidar 仅作调试日志：在线/离线的 InterleaveInferencer.__call__ 都把 lidar 参数注释掉了，
        # 不会进慢路径 prompt，所以这里的 PIL 内容不影响推理。

        # 3) target_point_speed
        speed = float(np.asarray(speed_clip[t]).reshape(-1)[0])
        tp = np.asarray(tp_clip[t]).reshape(-1)
        ntp = np.asarray(ntp_clip[t]).reshape(-1)
        target_point_speed = torch.tensor(
            [[speed, float(tp[0]), float(tp[1]), float(ntp[0]), float(ntp[1])]],
            dtype=torch.float32,
            device=self.device,
        )

        # 4) BEV encoder 输入（来自最后一帧三视角拼接 RGB 与 AutoMoT 风格栅格化 LiDAR）
        # 注意：LEAD .jpg 已经经过 1 次 JPEG 压缩（与 AutoMoT 训练分布一致），
        # 这里直接复用 PIL 解码结果，避免再 encode/decode 引入二次压缩伪影。
        bev_rgb = np.array(rgb_pil_list[-1], dtype=np.uint8)
        # (384, 1152, 3) uint8 RGB, range≈[0, 234]

        # 裁剪 RGB 到训练时使用的视野范围
        bev_rgb = bev_encoder_t_u.crop_array(self.bev_encoder_config, bev_rgb)
        # (384, 1024, 3)

        bev_rgb = np.transpose(bev_rgb, (2, 0, 1))
        bev_rgb_tensor = torch.from_numpy(bev_rgb).float().unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        # torch.Size([1, 3, 384, 1024]),


        # shape: (1, 256, 256), float32, range=[0, 1]
        bev_lidar_1ch = self._lead_lidar_to_bev_encoder_channel(lidar_i)
        # shape: (1, 256, 256) —— 已是 AutoMoT BEV encoder 训练分布的 shape，无需再 resize

        bev_lidar_tensor = torch.from_numpy(bev_lidar_1ch).float().unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        # torch.Size([1, 1, 256, 256])



        # [_prepare_inference_inputs Return Values Stats]
        # - rgb_pil_list: list[4], first image size=(1152, 384), mode=RGB
        # - lidar_pil_list: list[1], first image size=(256, 256), mode=RGB
        # - target_point_speed: shape=(1, 5), dtype=float32, range=[0.00132107, 25.0991]
        # - bev_rgb_tensor torch.Size([1, 3, 384, 1024]),: range=[0, 235]
        # - bev_lidar_tensor torch.Size([1, 1, 256, 256]),: range=[0, 1]
        # - bev_indices_desc: [3]   # 默认 anchor=12, bev_count=1, 仅 anchor 单帧（clip 内 idx）
        # - bev_indices_asc: [3]
        return rgb_pil_list, lidar_pil_list, target_point_speed, bev_rgb_tensor, bev_lidar_tensor, bev_indices_desc, bev_indices_asc

    @torch.no_grad()
    def run_step(self, lead_clip: dict[str, Any], anchor_t: int,
                 gen_context=None, timestamp: float = 0.0,
                 rgb_frame_step: int = 1, rgb_frame_count: int = 4,
                 bev_frame_step: int = 1, bev_frame_count: int = 1) -> dict[str, Any]:
        """
        离线版 run_step：
        - 输入：一个 LEAD clip + 指定 anchor 帧 + KV缓存上下文
        - 输出：AutoMoT 的 text/traj/route + 组装元数据

        参数说明：
        - rgb_frame_step: RGB 采样间隔（默认 1 对应 LEAD 的 0.25s，5 对应 AutoMoT 的 1.25s）
        - rgb_frame_count: RGB 采样帧数（默认 4）
        - bev_frame_step: BEV 采样间隔（默认 1）
        - bev_frame_count: BEV 采样帧数（默认 1）—— LEAD 单帧 .laz 已内含 5 累积 sweep，
          直接用 anchor 单帧即对齐 LEAD 训练分布；改为 2 会拼到 10 sweep / 0.5s，密度×2 时间窗×2
        
        关键流程：
        1. 准备数据 (BEV encoder, 多帧RGB)
        2. 若gen_context为空，初始化KV缓存
        3. 用fast推理路径生成轨迹和文本
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

        with torch.no_grad():
            # bev_rgb_tensor -> shape=(1, 3, 384, 1024), dtype=torch.bfloat16, range=[0, 235]
            # bev_lidar_tensor -> shape=(1, 1, 256, 256), dtype=torch.bfloat16, range=[0, 1]
            bev_encoder_output = self.bev_encoder(
                rgb=bev_rgb_tensor,
                lidar_bev=bev_lidar_tensor,
            )
        trans_feat = bev_encoder_output["bev_feature"]  # (1, 1512, 8, 8) —— 与训练分布一致
        # bev_encoder_output keys: ['bev_feature', 'bev_feature_upscale', 'fused_features', 'image_feature_grid']
        # bev_feature: tensor, shape=(1, 1512, 8, 8), dtype=torch.bfloat16  —— 切到 AutoMoT 栅格后回到训练 shape
        # bev_feature_upscale: tensor, shape=(1, 64, 64, 64), dtype=torch.bfloat16
        # fused_features: tensor, shape=(1, 1512, 8, 8), dtype=torch.bfloat16
        # image_feature_grid: tensor, shape=(1, 1512, 12, 32), dtype=torch.bfloat16  —— 来自 RGB，仍是三视图视野
        
        
        # ========== KV缓存推理 ==========
        # 第一次调用时初始化KV缓存
        if gen_context is None:
            # 构建slow_input_lists：图像列表 + 文本提示
            slow_input_lists = rgb_pil_list + [prompt_cleaned]
            # 调用kv_cache_fixed_inference获取初始化的gen_context
            gen_context = self.inferencer.kv_cache_fixed_inference(slow_input_lists)
        
        # 使用fast推理路径，复用gen_context的KV缓存
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            gen_text, gen_traj, route, reasoning_hidden_states = self.inferencer.based_kv_cache_context_fast_qwen3vl_dp(
                trans_feat=trans_feat,
                gen_context=gen_context,
                reasoning_tokens=getattr(self.automot.config, 'reasoning_query_tokens', 8),
                action_tokens=getattr(self.automot.config, 'action_query_tokens', 26),
                v_target_point=target_point_speed,
            )

        # 打印 fast-path 真实输出，确认已跑通 based_kv_cache_context_fast_qwen3vl_dp。
        traj_shape = tuple(gen_traj.shape) if isinstance(gen_traj, torch.Tensor) else None
        route_shape = tuple(route.shape) if isinstance(route, torch.Tensor) else None
        rhs_shape = (
            tuple(reasoning_hidden_states.shape)
            if isinstance(reasoning_hidden_states, torch.Tensor)
            else None
        )
        print("[based_kv_cache_context_fast_qwen3vl_dp] success")
        print(f"  text: {str(gen_text)[:200]}")
        print(f"  traj_shape: {traj_shape}")
        print(f"  route_shape: {route_shape}")
        print(f"  reasoning_hidden_states_shape: {rhs_shape}")

        return {
            "timestamp": timestamp,
            "anchor_t": anchor_t,
            "rgb_indices_desc": group.rgb_indices_desc,
            "rgb_indices_asc": group.rgb_indices_asc,
            "bev_indices_desc": bev_indices_desc,
            "bev_indices_asc": bev_indices_asc,
            "prompt": prompt_cleaned,
            "text": gen_text,
            "traj": gen_traj,
            "route": route,
            "gen_context": gen_context,  # 返回缓存供下一次使用
        }

    @torch.no_grad()
    def run_clip(self, lead_clip: dict[str, Any],
                 rgb_frame_step: int = 1, rgb_frame_count: int = 4,
                 bev_frame_step: int = 1, bev_frame_count: int = 1) -> list[dict[str, Any]]:
        """
        处理整段 LEAD clip，生成单组推理结果（基于最后一帧）。

        参数说明：
        - rgb_frame_step: RGB 采样间隔（默认 1 对应 LEAD 的 0.25s，5 对应 AutoMoT 的 1.25s）
        - rgb_frame_count: RGB 采样帧数（默认 4）
        - bev_frame_step: BEV 采样间隔（默认 1）
        - bev_frame_count: BEV 采样帧数（默认 1，对齐 LEAD 单帧 .laz 含 5 sweep 的训练分布）
        
        返回包含单个推理结果的列表。
        """
        clip_len = int(self._to_numpy(lead_clip["rgb"]).shape[0])
        
        # 以 clip 最后一帧作为 anchor（对应当前时刻）
        anchor_t = clip_len - 1
        print(f"Processing anchor_t={anchor_t} (last frame)...")
        
        out = self.run_step(
            lead_clip=lead_clip, 
            anchor_t=anchor_t,
            gen_context=None,
            timestamp=0.25 * anchor_t,
            rgb_frame_step=rgb_frame_step,
            rgb_frame_count=rgb_frame_count,
            bev_frame_step=bev_frame_step,
            bev_frame_count=bev_frame_count
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
        default='/data/lead_data/data/Accident/Town03_Rep0_route_001783_route0_01_11_02_37_46',
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
        bev_frame_count=max(1, args.bev_frame_count)
    )

    print(f"\nGenerated {len(outputs)} inference result(s)")
    for i, out in enumerate(outputs):
        print(
            f"\n[Result {i}]  anchor_t={out['anchor_t']}, "
            f"rgb_frames={out['rgb_indices_asc']}, "
            f"bev_frames={out['bev_indices_asc']}, "
            f"text={str(out['text'])[:80]}..."
        )


if __name__ == "__main__":
    main()
