"""Local Qwen3-VL-Instruct helpers for paradigm-A experiments.

这个包只服务 standalone Qwen3-VL-Instruct 范式 A：

- prompt_pipeline：构造驾驶状态机 prompt，并解析 VLM 文本输出。
- image_io：读取 LEAD RGB clip 或生成合成测试图。
- engine：通过 HuggingFace 标准接口显式 prefill/decode。
- cache_utils：记录或保存 KV cache 结构。
- mrope_utils：本地复算 Qwen3-VL 增量 decode 的 M-RoPE position_ids，避免
  PEFT/Transformers prepare_inputs_for_generation 丢 cache_position。

不要在这里接 AutoMoT 的 InterleaveInferencer；两条链路的模型结构和 checkpoint
语义不同。
"""
