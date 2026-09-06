"""轨迹 decoder 的 FP32 参数与 BF16 autocast；冻结模型有自己的精度配置。"""

import torch

PRECISION_POLICY = "fp32_parameters_adamw_ema_autocast_v2"


def decoder_forward(decoder, kwargs, compute_dtype, device):
    """只在可训练 decoder 前向启用 autocast，loss/backward 留在上下文外。"""
    with torch.autocast(
        device_type=device.type,
        dtype=compute_dtype,
        enabled=compute_dtype == torch.bfloat16,
    ):
        result = decoder(**kwargs)
    # L1/末点 loss 和轨迹指标使用 FP32；转换保持梯度连接。
    return {k: v.float() if k.startswith("pred_") else v for k, v in result.items()}
