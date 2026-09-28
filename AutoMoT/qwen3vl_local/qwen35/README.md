# Qwen3.5-4B 本地迁移（2026-09-28）

当前 SFT 各入口、Phase1/2/3、Action 主线与 qwen_simple、GoalGen、LeadMoT
以及道路事件探测入口，默认基座统一为 `AutoMoT/checkpoints/Qwen3.5-4B`。
`bev_only` 仍不加载 Qwen。包名 `qwen3vl_local` 和旧 engine 导入名保留兼容外部调用，
实际模型实现只从本目录 `vendor/` 导入。历史审计材料和冻结 Phase3 release 不改写。

## 本地源码与依赖

`vendor/` 包含官方 Transformers **5.3.0** 发布包中的完整 Qwen3.5 模型、配置、
分词器，以及官方复用的 Qwen3VL processor/video processor、Qwen2VL image processor。
这些类名称中的 Qwen2/Qwen3 是上游复用关系，不代表运行旧模型。
模型类和处理器不走 AutoModel 动态注册、远程仓库代码或在线 API。
Transformers 的共享框架、PyTorch、PEFT 等仍是本地环境依赖。
上游路径、wheel SHA256、原始/本地文件 SHA256、补丁说明和 Apache-2.0 许可证见
`vendor/UPSTREAM.json`、`vendor/LICENSE`。

本地改动集中在 `backend.py`、`processor.py`、`adapters.py`、`integration.py`
和 `vendor/modeling_qwen3_5.py`，不修改 site-packages。
上游源码默认版本不是“自动跟随最新”：运行时明确要求 `transformers==5.3.0`。
更换框架版本必须重新验证本地模型副本。

依赖清单见 `requirements.txt`。建议使用独立环境保留旧 run 的环境。
测试环境为 torch 2.5.1+cu124、torchvision 0.20.1、Transformers 5.3.0、
PEFT 0.18.1、Accelerate 1.12.0；新环境还需要匹配 CUDA 的 torch/torchvision、
Pillow、NumPy 及原项目依赖。训练机已有其它 torch/torchvision 配对时需重新验收。
离线安装使用预先准备的本地 wheel，例如在 `AutoMoT` 下：

```bash
python -m pip install --no-index --find-links /path/to/local/wheels -r qwen3vl_local/qwen35/requirements.txt
python -m qwen3vl_local.qwen35.preflight
python -m qwen3vl_local.qwen35.preflight --action
```

这三个命令不下载模型。完整模型文件必须提前放在默认目录，或传入本地 `--model-dir`。
目录应包含 `config.json`、完整 safetensors 权重及索引（若分片）、tokenizer 文件、
`chat_template.jinja`、`preprocessor_config.json`、`video_preprocessor_config.json`。`generation_config.json` 为可选文件；官方仓库未提供时，Transformers 从本地
`config.json` 推导默认生成配置，无需自行造文件。必需资产缺失时预检失败，不联网补文件。
运行时强制 HF/Transformers/Datasets offline、关闭 telemetry，拒绝远程图片/视频 URL。
不使用 GGUF、纯文本导出或旧 Qwen3-VL adapter 代替此多模态基座。

## 针对 Qwen3.5 的适配

- 图像使用模型目录内官方参数：patch16、merge2、mean/std=.5。
  1152×384 拼接图按完整图输入，每张产生 432 个视觉 token；1/2/4 图入口保留。
- 结构化问答和规划 prefill 默认 `enable_thinking=False`，训练和推理一致。
  空 `<think>...</think>` 是官方非思考格式的一部分，答案 loss mask 从其后开始，
  不把空思考段当答案监督。独立自由问答可显式通过 processor 请求 thinking，
  不能把这样的生成结果混进原结构化评估或既有 checkpoint 条件。
- 本地 processor 在非思考模式下保留历史 assistant 的空思考段，保证多轮
  全量模板与已缓存前缀字节一致；system-only prefix 也有单独兼容。
  tokenizer 默认左 padding；v5 手工右 padding 的缓存续写也显式保留 suffix mask。
  DeltaNet 只消费有效 token，padding 不推进卷积/循环状态；输出 scatter 回原打分位置。
  整行空 suffix 保留原状态和 next logits，普通注意力仍使用完整物理缓存 mask。
  ragged 分支按样本处理；无 padding 时保留原批量路径，CUDA 吞吐尚未验证。
- LoRA 除普通注意力/MLP 外，覆盖线性注意力的 `in_proj_qkv`、`in_proj_z`、
  `in_proj_a`、`in_proj_b`、`out_proj`。视觉 scope 原开关仍生效，不默认解冻视觉塔。
- Qwen3.5-4B 为 32 层，其中 3/7/11/15/19/23/27/31（从 0 编号）有 token K/V。
  线性层的 convolution/recurrent state 在续写、分支复制、batch 选择和缓存保存时保留；
  不伪装成逐 token K/V。修复固定上游版本的多 token cached suffix 状态重置，
  以及同一模型跨图片/纯文本 prefill 的旧 RoPE delta 污染。
  system-prefix 缓存续写及后续 decode 返回当前缓存自带的 delta，不能返回模型对象上其它分支的旧值。
- GoalGen/LeadMoT 新训默认 8 个 decoder block，每个对应一个普通注意力层，
  K/V 布局由 8×128 改为 **4×256**，总宽度仍为 1024。
  LeadMoT 的 generated Q/K 对齐 theta=1e7、partial factor=.25、
  interleaved M-RoPE=[11,11,10]；prefix K 不重复旋转。
  历史 `mhrope`/`none` 仍是显式消融，不是模型原生位置编码。
- Action 通过本地 bridge 接入上述分段；外部 runner 仍需实际存在。
  单靠本地 bridge 测试不能证明未知版本 runner 的完整兼容性。

## 新旧模型合同

Qwen3-VL 的 Phase1/2/3 LoRA、旧 Action/GoalGen checkpoint 不能直接挂到新基座续训。
各 SFT 保存路径（含 emergency）写入 `qwen35_backend.json` 与 `qwen35_base_assets.json`。
后者绑定实际 safetensors 权重（含索引指向的全部分片）、配置、模板、tokenizer 和处理参数。
模型加载前流式读取权重计算 SHA256，不以目录名、尺寸或时间戳代替内容身份；增加启动磁盘读取成本。
未通过 LocalModel 加载并绑定权重的模型不能保存为受支持的 adapter。
加载 adapter 校验本地源码/默认思考模式/基础资产，缺失或不一致拒绝。
Action 选择、复制、打包同时携带这些文件；执行指纹覆盖新增本地源码。
Action 自动选优在比较指标前按当前基座资产过滤，缺资产合同或缺权重哈希的候选拒绝；
同次扫描只计算一次基座哈希，显式指定候选同样校验，实际模型加载仍再次校验。
LeadMoT 与 GoalGen 共用 v2 backbone 合同，纯基座也必须校验权重与输入资产。
GoalGen 完整/轻量 checkpoint 都保存合同，评估和 `--init-from-ckpt` 初始化时强制核对；
同路径替换 adapter 会被拒绝，内容相同的目录搬迁允许。旧路径 mismatch 开关不能绕过内容校验。
GoalGen 另保存并严格核对 `qwen_kv_segment_mode`，评估/初始化均执行；缺失则拒绝。
已有 checkpoint 可读取其保存的 args 中的模式，不能按当前默认值猜测；新 checkpoint 显式字段
与 args 冲突也拒绝。即使默认 8 段时不同模式数值相同，也保持合同一致。
backend 已更新为 `qwen35_local_v2`；之前不含权重身份的 v1 合同不自动补写或放行。
旧 run 应使用原源码、原基座和原环境，不改其 manifest/hash 绕过检查。

本次不改变 Phase3 的动作真值、阈值、split、事件配额或稳定 `v23_io1` 数据构建快照。
模型升级是新模型实验，不能据 CPU 回归宣称效果提高。Phase1/2 如使用模型先验需重训，
dataset-priors 路径仍无需 Phase1/2 adapter。原训练脚本直接使用新默认目录；例如：

```bash
bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
# 显式选卡沿用项目约定：
GPU_IDS=0 bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```

先让本地预检通过，再在训练机执行；这些示例不表示本机已具备完整运行资源。

## 验证与边界

小型随机初始化的真实 Qwen3.5 网络覆盖多图、非零视觉 RoPE delta、单/多 token 缓存续写、
padding、分支选择、RoPE 数值对照、LoRA 注入/梯度、本地 save/load 和禁止联网。
真实 processor 与 Phase3 loss-mask 路径覆盖两图/四图、binary/choice 四种组合。
同时运行现有 Phase3、Action 训练/恢复及稳定 release 相关 CPU 回归。

本机没有 `checkpoints/Qwen3.5-4B` 完整权重，也缺少
`leaderboard/team_code/mot_lead_offline_runner.py`。因此本次不包含真实 4B/GPU 训练、
外部 runner 的端到端验收或模型效果对比。精确测试结果和剩余项见 PROJECT_CONTEXT 同日条目。
CUDA 的 FlashAttention/causal-conv1d/FLA 加速需在目标训练机验证；这些可选库缺失时，
本地模型保留上游 PyTorch 实现，但不能据此承诺相同吞吐。

官方依据：
- https://huggingface.co/Qwen/Qwen3.5-4B
- https://huggingface.co/docs/transformers/model_doc/qwen3_5
- https://github.com/huggingface/transformers/tree/v5.3.0/src/transformers/models/qwen3_5
