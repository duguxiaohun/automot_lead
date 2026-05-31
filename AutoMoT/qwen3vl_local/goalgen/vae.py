"""冻结 VAE 封装，桥接 AutoMoT/vae_standalone。

设计要点：
- 所有"碰 vae_standalone 源码"的操作都集中在这里，外部模块只看 FrozenVAE 这一个类。
- 不去改 vae_standalone/ 的任何文件；只把它加进 sys.path 后复用 instantiate_from_config。
- 加载后 eval() + requires_grad_(False)，全程冻结。
- 输入约定：[-1,1] 归一化的 RGB 张量 [B,3,H,W]，H/W 必须 64 倍数。
- 输出 latent 已乘 scale_factor。
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms


_THIS_FILE = pathlib.Path(__file__).resolve()
# qwen3vl_local/goalgen/vae.py -> AutoMoT/vae_standalone
_AUTOMOT_ROOT = _THIS_FILE.parents[2]
_VAE_STANDALONE_DIR = _AUTOMOT_ROOT / "vae_standalone"


def _ensure_vae_standalone_on_path() -> None:
    """把 vae_standalone 临时加进 sys.path。

    vae_standalone 的 import 用相对自身根目录的形式（from vwm.util import ...），
    所以这里只加它的根目录，不污染包名空间。
    """

    p = str(_VAE_STANDALONE_DIR)
    if not _VAE_STANDALONE_DIR.exists():
        raise FileNotFoundError(f"找不到 vae_standalone：{_VAE_STANDALONE_DIR}")
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass
class VAEConfig:
    """从 vae_only.yaml 抽出的必要字段。"""

    scale_factor: float
    disable_autocast: bool


class FrozenVAE:
    """加载好的 first_stage VAE 实例，提供 encode/decode 两个最小接口。

    encode 接收 PIL 列表或张量列表，统一规整成 [B,3,H,W]，归一化到 [-1,1]，
    返回乘过 scale_factor 的 latent；decode 走相反方向，用于可视化。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: VAEConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        # 训练里 VAE 全程冻结，所以这里直接 eval + 关掉 grad。
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        # 与 vae_reconstruct.preprocess_one_image 完全一致的归一化。
        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        self.latent_mean = torch.zeros(1, 4, 1, 1, device=self.device, dtype=self.dtype)
        self.latent_std = torch.ones(1, 4, 1, 1, device=self.device, dtype=self.dtype)
        self.latent_stats_enabled = False

    @classmethod
    def load(
        cls,
        config_path: pathlib.Path,
        weights_path: pathlib.Path,
        device: str = "cuda",
        dtype: str = "float32",
    ) -> "FrozenVAE":
        """按 vae_standalone 同样的方式加载模型 + safetensors 权重。

        三步：
        1. 把 vae_standalone 加进 sys.path；
        2. 用 OmegaConf 读 vae_only.yaml，instantiate_from_config 拿到模型骨架；
        3. 用 safetensors.load_file 读权重，strict=False 是因为 vae_only.safetensors
           只包含 first_stage 部分，原始权重里没有损失 / 正则器的可选键。
        """

        # 必须在 import vwm 之前注入 sys.path，否则下一行会 ModuleNotFoundError。
        _ensure_vae_standalone_on_path()
        from omegaconf import OmegaConf  # noqa: E402
        from safetensors.torch import load_file  # noqa: E402
        from vwm.util import instantiate_from_config  # noqa: E402

        if not config_path.exists():
            raise FileNotFoundError(f"找不到 VAE 配置：{config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"找不到 VAE 权重：{weights_path}")

        # OmegaConf 读 yaml；first_stage_config 是 vwm 风格的 target+params 配置。
        cfg_yaml = OmegaConf.load(str(config_path))
        first_stage_model = instantiate_from_config(cfg_yaml.first_stage_config)

        # 加权重。strict=False 容忍权重和模型可选模块之间的微小差异；
        # 实际场景下 missing/unexpected 应该都很少（多数 < 5），所以直接打印 len 即可。
        sd = load_file(str(weights_path))
        missing, unexpected = first_stage_model.load_state_dict(sd, strict=False)
        print(f"[goalgen.vae] 缺失键={len(missing)} 非预期键={len(unexpected)}")

        # 字符串 dtype 转 torch.dtype。默认 fp32：vae_only.yaml 关了自动混精，
        # 用 bf16 会有少量重构精度损失，对 z1 监督质量有影响。
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.float32)
        torch_device = torch.device(device)

        # 一次性把模型搬到目标 device 和 dtype，避免后续编码/解码时频繁迁移。
        first_stage_model = first_stage_model.to(device=torch_device, dtype=torch_dtype)

        # 把 scale_factor 和 autocast 开关从 yaml 抽出来，封装成 dataclass，
        # 避免 encode/decode 里反复访问 OmegaConf。
        vae_cfg = VAEConfig(
            scale_factor=float(cfg_yaml.scale_factor),
            disable_autocast=bool(cfg_yaml.disable_first_stage_autocast),
        )
        return cls(model=first_stage_model, cfg=vae_cfg, device=torch_device, dtype=torch_dtype)

    @staticmethod
    def _validate_shape(h: int, w: int) -> None:
        # VAE 下采 8 倍 + 内部还有 conv 对齐要求；统一限制 64 的倍数。
        if h % 64 != 0 or w % 64 != 0:
            raise ValueError(f"VAE input H/W 必须是 64 的倍数，得到 H={h} W={w}")

    def pil_to_tensor(self, images: List[Image.Image]) -> torch.Tensor:
        """PIL 列表 -> [B,3,H,W] 归一化张量，并校验形状。"""

        tensors: List[torch.Tensor] = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            self._validate_shape(h, w)
            tensors.append(self._to_tensor(img))
        return torch.stack(tensors, dim=0).to(device=self.device, dtype=self.dtype)

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Enable per-channel latent normalization on scaled VAE latents."""

        mean = mean.detach().to(device=self.device, dtype=self.dtype).view(1, 4, 1, 1)
        std = std.detach().to(device=self.device, dtype=self.dtype).view(1, 4, 1, 1)
        self.latent_mean = mean
        self.latent_std = std.clamp_min(1e-6)
        self.latent_stats_enabled = True

    def clear_latent_stats(self) -> None:
        self.latent_mean = torch.zeros(1, 4, 1, 1, device=self.device, dtype=self.dtype)
        self.latent_std = torch.ones(1, 4, 1, 1, device=self.device, dtype=self.dtype)
        self.latent_stats_enabled = False

    def latent_stats_dict(self) -> Optional[Dict[str, List[float]]]:
        if not self.latent_stats_enabled:
            return None
        return {
            "mean": [float(x) for x in self.latent_mean.view(-1).detach().float().cpu()],
            "std": [float(x) for x in self.latent_std.view(-1).detach().float().cpu()],
        }

    def load_latent_stats_dict(self, stats: Dict[str, List[float]]) -> None:
        self.set_latent_stats(
            torch.tensor(stats["mean"], dtype=torch.float32),
            torch.tensor(stats["std"], dtype=torch.float32),
        )

    @torch.no_grad()
    def encode_raw(self, images: List[Image.Image]) -> torch.Tensor:
        """PIL images -> scaled VAE latents, before per-channel normalization."""

        x = self.pil_to_tensor(images)
        with torch.autocast("cuda", enabled=(not self.cfg.disable_autocast and self.device.type == "cuda")):
            z = self.model.encode(x)
        return z * self.cfg.scale_factor

    @torch.no_grad()
    def encode(self, images: List[Image.Image]) -> torch.Tensor:
        """PIL 列表 -> latent，乘过 scale_factor。

        关键步骤：
        1. pil_to_tensor 内部完成 RGB 校验 / 64 倍数校验 / [-1,1] 归一化；
        2. autocast 仅在 cuda 且 yaml 没禁用时开启；默认 yaml 关了 autocast，所以
           走 fp32 路径；
        3. encode 后**必须**乘 scale_factor，否则 z 数值范围与 Vista 原训练分布不一致，
           DiT 学到的 v_target 就漂掉了。
        """

        z = self.encode_raw(images)
        if self.latent_stats_enabled:
            z = (z - self.latent_mean) / self.latent_std
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """latent -> [-1,1] 范围的 RGB 张量；仅推理 / 可视化用。"""

        # 强制对齐 (device, dtype)：DiT 训练 / 推理时 z 通常是 bf16，VAE 权重默认 fp32；
        # 不转换会在 model.decode 第一层 Conv2d 上抛 dtype mismatch（且错误堆栈在
        # C++ 端不好定位）。这里 .to 是 no-op 当 z 已经匹配，所以无副作用。
        z = z.to(device=self.device, dtype=self.dtype)
        if self.latent_stats_enabled:
            z = z * self.latent_std + self.latent_mean
        z_in = z / self.cfg.scale_factor
        # VideoDecoder 需要 timesteps；单帧时给当前 batch 大小即可。
        from vwm.modules.autoencoding.temporal_ae import VideoDecoder  # noqa: E402
        if isinstance(self.model.decoder, VideoDecoder):
            kwargs = {"timesteps": z_in.shape[0]}
        else:
            kwargs = {}
        with torch.autocast("cuda", enabled=(not self.cfg.disable_autocast and self.device.type == "cuda")):
            return self.model.decode(z_in, **kwargs)

    def latent_shape_for(self, height: int, width: int, batch: int = 1) -> Tuple[int, int, int, int]:
        """根据输入分辨率推潜变量形状；下采 8 倍 + z_channels=4。"""

        self._validate_shape(height, width)
        return (batch, 4, height // 8, width // 8)


def default_vae_paths() -> Tuple[pathlib.Path, pathlib.Path]:
    """返回项目里 vae_standalone 的默认配置与权重路径。"""

    cfg = _VAE_STANDALONE_DIR / "config" / "vae_only.yaml"
    weights = _VAE_STANDALONE_DIR / "weights" / "vae_only.safetensors"
    return cfg, weights
