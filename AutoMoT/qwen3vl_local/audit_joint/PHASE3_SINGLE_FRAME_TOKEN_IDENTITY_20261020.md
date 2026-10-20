# 单帧方案一：历史/输出 token 一致与自然缺失

用户明确不添加人工 UNKNOWN。当前训练继续直接读取保存的历史标签及既定 nearest_short 填补结果，没有随机遮蔽、历史 dropout 或未来标签补填。UNKNOWN 表示信息缺失，不是 KEEP，也不是第七种输出动作；提示词明确要求结合图像、当前前提及其它已知历史判断。它保留正常 attention，不删除其位置。

## 完整提示词中的 token 校验

历史每格 A–F 改为独立行，使用普通文本 `[HISTORY_ACTION_i]` / `[/HISTORY_ACTION_i]` 标记定位。没有注册特殊 token、扩展词表或新增 embedding。这可避免通常的空格前缀 BPE 合并；最终仍以实际 tokenizer 的结果为准。

每次 `objective.pack` 都在聊天模板、图像占位展开后的完整输入上检查：解码/重编码 ID 不变；每个历史标记唯一且动作文字精确；已知动作只对应一个 token，且 ID 必须等于同动作的输出候选 ID，offset 不能混入前后字符。不满足即停止，不通过静默替换 tokenizer 或扩词表绕过。UNKNOWN 可以由多个现有 token 表示，不要求其与任何输出动作 ID 对齐。

正式 CPU `prepare` 用所选本机模型的 tokenizer，把六种动作及 UNKNOWN 分别放入全部四个历史位置测试，写入 `single_frame_preflight.json/action_token_identity`，记录实际候选 ID、词表大小及每个槽位的检查结果。这些合成探针仅用于校验，不加入训练数据。正式训练/评测仍逐输入检查。保存合同同时记录动作编码协议、提示词哈希和 UNKNOWN 策略。

相同 token ID 表示使用同一词表条目，不代表输入 embedding 和输出 head 权重一定绑定，更不代表新增了可训练动作 embedding；当前仍沿用已有词表和既定 LoRA 训练方案。

## 自然 UNKNOWN 的评测

现有四种输入评测（完整历史、无历史、原始未填补历史、空白图像）保留；无历史仅是评测输入对照。新增按 `native_unknown_slots_0..4` 与 `input_unknown_slots_0..4` 分组：前者统计填补前缺失，后者统计实际送入该评测变体的缺失。报告各组锚点数、有效窗口分母及对应命中/分布指标，不把不同分母的准确率混为一个对照。

自然缺失可以使训练包含信息不足的样本；能否提高鲁棒性仍需实际模型结果。缺失可能集中在某些事件/动作边界，分组成绩不能直接解释为随机噪声的因果效果。当前不主动增加 UNKNOWN。

## 数据与验证边界

本轮只改输入编码/运行时校验/报告，不改 Phase3 原规则、过去与未来标签、短缺口填补、event/action 配额或有效窗口 loss。已有数据通过显式 runtime-only refresh 核验原始来源及旧绑定，写入新的 `checkpoints/phase3_single_frame_v1/token_identity_train` 和 `token_identity_val`；旧目录保留。`records` 必须逐字段相同，无重新标定。新提示词/合同不能直接用于恢复旧提示词训练断点；接续仍要求源码与合同匹配，不能只改 SHA 绕过。

新增回归使用项目 Qwen tokenizer 实现、官方模板、真实图像处理器和带空格合并的合成词表：旧 `s: A` 与独立 `A` 的 ID 确实不同，新四格全部通过；另用会合并换行与字母的 tokenizer 证明独立候选检查通过时，完整输入校验仍能拒绝错误 ID。还覆盖重复/缺失标记、文本篡改、UNKNOWN attention、词表/数据不变及缺失分组分母。

本机未找到正式服务器模型的 tokenizer 资产，因此这些回归不是正式模型 tokenizer 已通过的证明。正式资产检查已接入 prepare，待服务器执行。CPU 测试也不证明 H20 资源或性能恢复；未做 GPU 训练、未 push。本轮测试及数据核验结果见 `phase3_single_frame_token_identity_verification_20261020.json`。
