"""SFT v1 loss_scale sanity check — 在 token 级别可视化 ANALYSIS 段 mask。

主体检查不依赖 GPU。脚本会先用 HuggingFace tokenizer + 你已经生成的
jsonl 样本，把 assistant content 按字符 → token 映射展开，标出三段：

    [MASK]  对应 ANALYSIS: Observations recorded.\n   （swift loss_scale 应权重 0）
    [LOSS]  对应 STATUS: <event_name>\n               （应算 loss）
    [LOSS]  对应 SUBGOAL: <event_name>                 （应算 loss）

随后会可选调用 `tools/sft_v1_loss_scale_plugin.py` 里的 ms-swift 插件本体，
确认 STATUS/SUBGOAL event_name 确实位于插件返回的 loss 段。若当前环境没装
ms-swift，这一步会打印 WARN；远程训练环境必须让这一步通过。

典型用法（**从 AutoMoT/ 目录运行**，远程默认 cwd）：

```bash
# 默认看 train.jsonl 第一条
python tools/check_loss_mask.py

# 看第 N 条
python tools/check_loss_mask.py --sample-idx 7

# 指定 tokenizer 目录
python tools/check_loss_mask.py \
    --tokenizer-dir checkpoints/Qwen3-VL-4B-Instruct
```

观察要点：
- ANALYSIS 段每个 token 都应被列为 [MASK]，token 数应在 5-10 之间
- STATUS / SUBGOAL 的字面前缀和 event_name token 应被列为 [LOSS]
- 如果 STATUS event_name 只有 1 个 token，说明 BPE 把它当成完整词；
  这是 v1 监督信号最稠密的位置，绝对不能被 mask
- 如果发现 ANALYSIS 段有 token 被标 [LOSS]（或反过来），说明
  PLACEHOLDER_ANALYSIS 跟 loss_scale 插件 regex 不匹配，必须修
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import os
import pathlib
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# 本文件位于 AutoMoT/tools/。parents[1]=AutoMoT/，parents[2]=automot_lead 仓库根。
# 这里推断路径不是为了 sys.path import 别的模块——本脚本不 import qwen3vl_local——
# 而是为了让默认的 --jsonl / --tokenizer-dir 参数能"开箱即用"地指向本项目约定位置。
_THIS_FILE = pathlib.Path(__file__).resolve()
_AUTOMOT_ROOT = _THIS_FILE.parents[1]
_PROJECT_ROOT = _THIS_FILE.parents[2]

# HF 离线开关。tokenizer 也走本地缓存，避免在远程跑时因联网失败而崩。
# setdefault 保证若用户事先 export 过同名变量，本脚本不会覆盖（典型场景：调试时
# 临时想联网下载 tokenizer，可以 unset 这三个再跑）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# 必须与 sft_v1_loss_scale_plugin.py 里的 regex 字面语义完全一致。
#
# regex 语义拆开：
#   ANALYSIS:          —— 锚定占位段开头，跟 build_sft_dataset_v1.py 的
#                          PLACEHOLDER_ANALYSIS = "Observations recorded." 前缀对齐
#   .*?                —— 非贪婪匹配，吃掉 "Observations recorded." 这段任意内容
#                          （包含可能的换行——配合 flags=re.DOTALL）
#   (?=\nSTATUS:)      —— 前瞻断言：吃到 \nSTATUS: 之前为止，但不消费它
#                          这样匹配 range 就是 "ANALYSIS: ... " 直到 \n 之前，
#                          \n 本身落到下一段（STATUS 行）保留给 loss 算
#
# 同步规则：本常量与 sft_v1_loss_scale_plugin.py 的 _ANALYSIS_REGEX 必须同时改、
# 同时验。如果哪边偷偷改了一边，可能出现"本脚本上对、但 swift 训练时不对"的假象。
LOSS_SCALE_REGEX = r"ANALYSIS:.*?(?=\nSTATUS:)"


def load_first_sample(jsonl_path: pathlib.Path, idx: int) -> Dict:
    """从 jsonl 取第 idx 条样本（0-based）。

    用途：sanity 脚本只需要 1 条样本就够。jsonl 可能很大（默认 8400 条），
    所以不一次性 json.load 整个文件——而是逐行扫，遇到目标 idx 就返回。

    边界：
    - jsonl 文件里可能夹杂空行（编辑器误存或合并产生），strip 后空行直接跳过；
    - idx 计数只算"非空"行（与 build_sft_dataset_v1.py 写入时的顺序一致）；
    - idx 超出文件长度时主动抛 IndexError，便于命令行立刻看到错误而不是返回 None。
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == idx:
                return json.loads(line)
    raise IndexError(f"jsonl {jsonl_path} has fewer than {idx+1} samples")


def find_mask_char_range(text: str) -> Optional[Tuple[int, int]]:
    """在 assistant text 上按 LOSS_SCALE_REGEX 找 ANALYSIS 段的"字符"区间。

    为什么是字符 range 而不是 token range：
    - swift 的 loss_scale 实现细节（按 char 还是按 token 匹配）我们无法 100%
      复刻，但 swift 拿到的输入文本和我们这里的 text 是同一份；
    - 只要我们能找出"哪些字符应该被 mask"，再通过 tokenizer 的 offsets_mapping
      把它映射回 token，就能判断"在 token 级 swift 应该把哪些 token mask 掉"；
    - 这样不依赖 swift 内部行为，只依赖"swift 用的也是这个 regex + 同一文本"
      这一个已知假设。

    返回值：
    - [start, end) 半开区间，end 落在 '\\nSTATUS:' 之前（regex 用了前瞻断言不消费）；
    - 匹配不到返回 None，调用方会把所有 token 都标 [LOSS] 并触发警告，
      因为这种情况下 swift 训练时整段 ANALYSIS 都会算 loss——是危险信号。
    """
    # re.DOTALL 让 . 也能匹配 \n。理论上 PLACEHOLDER_ANALYSIS 是单行不含 \n，
    # 但用户后续可能换成多行占位，加 DOTALL 让 regex 更鲁棒。
    m = re.search(LOSS_SCALE_REGEX, text, flags=re.DOTALL)
    if m is None:
        return None
    return m.start(), m.end()


def tokenize_with_offsets(tokenizer, text: str) -> Tuple[List[int], List[Tuple[int, int]], List[str]]:
    """对 assistant text 做 tokenize，同时拿到 char↔token 映射与单 token 解码字符串。

    HuggingFace fast tokenizer 的 ``return_offsets_mapping=True`` 会额外返回
    ``offset_mapping``：每个 token 在原始字符串中占用的字符 range（半开区间）。
    没有 offsets 的话，我们只能拿到 token id 序列，但无从知道"token #i 对应
    原文哪几个字符"，也就没法用上面 find_mask_char_range 的字符 range 来分段。

    参数：
    - tokenizer：HuggingFace ``PreTrainedTokenizerFast`` 实例（必须 fast，
      普通 Python 实现的 slow tokenizer 不支持 offsets）。
    - text：要 tokenize 的字符串。本脚本只 tokenize assistant 段，不包含
      system / user / image token，因为 loss_scale 只作用在 assistant response 上。

    为什么 ``add_special_tokens=False``：
    - 我们只想观察 ANALYSIS / STATUS / SUBGOAL 段本身的 token；
    - 如果加 special token，前面会多出 ``<|im_start|>assistant\\n`` 这种额外 id，
      它们的 offsets 会指向"虚空"或被映射成 (0,0)，干扰可视化与 mask 判定；
    - swift 训练时这些 special token 是模板渲染的一部分，loss_scale regex 不会
      命中它们，所以本脚本省略是安全的。

    返回三元组：
    - ids[i]：token id 列表，整数；
    - offsets[i]：(char_start, char_end) 半开区间，标识 token #i 在 text 里位置；
    - decoded[i]：把单个 token id 解码回字符串，方便打印时眼看"这个 token 长啥样"。
      ``clean_up_tokenization_spaces=False`` 保留原始空格 / 连接符，否则
      Qwen tokenizer 可能把前导空格清掉，让"是不是空白 token"难以辨认。
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    # tokenizer 返回的 input_ids / offset_mapping 是 list-like；显式转 list/tuple
    # 让后续 indexing 与 print 行为稳定，避免某些 tokenizer 后端返回 numpy 数组。
    ids = list(enc["input_ids"])
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    decoded = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in ids]
    return ids, offsets, decoded


def classify_tokens(
    offsets: List[Tuple[int, int]],
    mask_range: Optional[Tuple[int, int]],
) -> List[str]:
    """把每个 token 标成 ``[MASK]``（loss 权重 0）或 ``[LOSS]``（正常算 loss）。

    判定逻辑：对每个 token 的字符 range (s, e)，判断它是否与 mask_range
    (ms, me) 有非空交集。半开区间相交的标准式是 ``s < me and e > ms``：
    - ``s < me``：token 的起点在 mask 区间结束之前；
    - ``e > ms``：token 的终点在 mask 区间开始之后；
    - 两者同时成立 ⇒ token 至少与 mask_range 有 1 个字符重叠 ⇒ 视为 [MASK]。

    边界情况：
    - 跨边界 token：BPE 可能把 "Observations recorded.\\nSTATUS" 中
      ``recorded.`` 与 ``\\n`` 切成同一 token，这种 token 既覆盖 mask_range
      又覆盖 STATUS: 之前的 \\n。本判定下它会算 [MASK]，是保守选择
      （宁可少算一点 STATUS 段的 loss 也不要让 ANALYSIS 漏到 [LOSS]）。
      实测 Qwen3-VL tokenizer 不会出现这种跨边界 token，因为换行 \\n 单独成 token。
    - mask_range is None：regex 在 text 上完全匹配不到，意味着 PLACEHOLDER_ANALYSIS
      或 STATUS: 段哪个被改坏了。这种情况下不能贸然全标 [MASK]——那样训练会
      0 梯度 NaN——所以全标 [LOSS] 并由 main() 打印 [WARN] 提示。
    """
    tags: List[str] = []
    if mask_range is None:
        return ["[LOSS]"] * len(offsets)
    ms, me = mask_range
    for (s, e) in offsets:
        # 经典"半开区间相交"判定，等价于 `not (e <= ms or s >= me)`。
        if s < me and e > ms:
            tags.append("[MASK]")
        else:
            tags.append("[LOSS]")
    return tags


def print_token_table(
    ids: List[int],
    offsets: List[Tuple[int, int]],
    decoded: List[str],
    tags: List[str],
) -> None:
    """打印 token 对照表，让人眼看清楚每个 token 的 mask 状态。

    输出格式（每行一个 token）：
        idx  tag    id  char_range  decoded
        ----------------------------------------------------------
           0 [MASK]  19394   [0,9)    'ANALYSIS:'
           1 [MASK]    220   [9,10)   ' '
           2 [MASK]   4571  [10,13)   'Obs'
           ...
           7 [LOSS]    198  [32,33)   '\\n'
           8 [LOSS]  31650  [33,40)   'STATUS:'

    阅读要点：
    - idx：从 0 开始的 token 序号；
    - tag：[MASK]=训练时这个 token 不算 loss，[LOSS]=算 loss；
    - id：token 在词表里的整数 id，方便和 swift 训练日志的 input_ids 对照；
    - char_range：token 占用的原文字符区间 [start, end)；可看 token 边界；
    - decoded：单 token 解码字符串。``repr()`` 包裹让空格、换行可视。

    转义 \\n / \\r 是为了让换行 token 不破坏表格对齐；只有这两个常见控制符
    需要处理，其它（如 \\t）在 assistant content 里不出现。
    """
    print(f"{'idx':>4} {'tag':<6} {'id':>7} {'char_range':>14}  decoded")
    print("-" * 80)
    for i, (tid, off, tok, tag) in enumerate(zip(ids, offsets, decoded, tags)):
        # 让 \n / \r 在表格里以字面形式出现，避免实际换行打断行对齐。
        repr_tok = tok.replace("\n", "\\n").replace("\r", "\\r")
        print(f"{i:>4} {tag:<6} {tid:>7} {f'[{off[0]},{off[1]})':>14}  {repr_tok!r}")


def summarize(tags: List[str], decoded: List[str]) -> Dict[str, int]:
    """聚合统计：[MASK] / [LOSS] token 数 + 解码后总长度。

    用途：表格打印完后，给出一行总结让人快速判断是否健康。main() 还会基于
    这里的 n_loss / n_mask 触发警告：
    - n_loss < 5：STATUS / SUBGOAL 段 token 太少，说明监督信号稀薄
      （正常情况下 STATUS / SUBGOAL 行加起来应该有 8-15 个有效 token）；
    - n_mask == 0：regex 完全没命中或命中区间不含任何 token，训练时
      ANALYSIS 段会全部算 loss，模型会学成"先抄占位句再答 STATUS"。

    joined_length 主要给调试用：和原始 text 长度对比，能看出 tokenizer 是否
    丢了字符（理论上不会，但 fast tokenizer 偶发 normalization 问题时有用）。
    """
    n_mask = sum(1 for t in tags if t == "[MASK]")
    n_loss = sum(1 for t in tags if t == "[LOSS]")
    text_joined = "".join(decoded)
    return {
        "n_mask": n_mask,
        "n_loss": n_loss,
        "joined_length": len(text_joined),
    }


def parse_status_lines(text: str) -> Tuple[Optional[str], Optional[str]]:
    """抽取 assistant 输出里的 STATUS / SUBGOAL 事件名。"""
    status_match = re.search(r"^STATUS:\s*(.+)$", text, flags=re.MULTILINE)
    subgoal_match = re.search(r"^SUBGOAL:\s*(.+)$", text, flags=re.MULTILINE)
    status = status_match.group(1).strip() if status_match else None
    subgoal = subgoal_match.group(1).strip() if subgoal_match else None
    return status, subgoal


def print_plugin_loss_scale_check(text: str) -> None:
    """调用 ms-swift 插件本体，确认它把 STATUS/SUBGOAL 留在 loss 段。

    这一步比上面的纯 regex/token 表更接近训练侧：swift 训练时会通过
    `--external_plugins tools/sft_v1_loss_scale_plugin.py` 注册并调用同一个
    `SftV1AnalysisMaskLossScale.get_loss_scale()`。如果这里的 loss 段没有包含
    STATUS / SUBGOAL 的 event_name，`bash tools/sft_v1_train.sh check` 的低 loss
    就不能继续信任。
    """
    print()
    print("===== plugin loss_scale sanity =====")
    plugin_path = _THIS_FILE.with_name("sft_v1_loss_scale_plugin.py")
    try:
        spec = importlib.util.spec_from_file_location("sft_v1_loss_scale_plugin", plugin_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load plugin spec: {plugin_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loss_scale = module.SftV1AnalysisMaskLossScale()
        parts, scales = loss_scale.get_loss_scale(text)
    except Exception as exc:
        print(f"[WARN] 无法加载/调用 ms-swift loss_scale 插件：{exc!r}")
        print("       如果远程已安装 ms-swift，请优先修这个问题再跑训练。")
        print("====================================")
        return

    cursor = 0
    for idx, (part, scale) in enumerate(zip(parts, scales)):
        start = text.find(part, cursor) if isinstance(part, str) else -1
        end = start + len(part) if start >= 0 and isinstance(part, str) else -1
        cursor = end if end >= 0 else cursor
        preview = part.replace("\n", "\\n").replace("\r", "\\r") if isinstance(part, str) else repr(part)
        print(f"[plugin] seg={idx} scale={scale} chars=[{start},{end}) text={preview!r}")

    loss_text = "".join(part for part, scale in zip(parts, scales)
                        if scale > 0 and isinstance(part, str))
    mask_text = "".join(part for part, scale in zip(parts, scales)
                        if scale == 0 and isinstance(part, str))
    status, subgoal = parse_status_lines(text)
    checks: Sequence[Tuple[str, Optional[str]]] = (
        ("STATUS", status),
        ("SUBGOAL", subgoal),
    )
    for label, value in checks:
        if not value:
            print(f"[WARN] {label} event_name 解析失败。")
            continue
        in_loss = value in loss_text
        in_mask = value in mask_text
        print(f"[plugin] {label} event_name={value!r} in_loss={in_loss} in_mask={in_mask}")
        if not in_loss or in_mask:
            print(f"[WARN] {label} event_name 没有被正确保留为 loss token。")

    if not any(scale == 0 for scale in scales):
        print("[WARN] 插件没有产生任何 0 权重段，ANALYSIS 可能会参与 loss。")
    if not any(scale > 0 for scale in scales):
        print("[WARN] 插件没有产生任何 loss 段，STATUS/SUBGOAL 可能被全 mask。")
    print("====================================")


def main():
    """主流程：
    1. 解析命令行 → 拿到 jsonl 路径 + sample 编号 + tokenizer 目录
    2. 从 jsonl 取 1 条样本，抽出 assistant content
    3. 在 assistant content 上跑 LOSS_SCALE_REGEX，确认是否能找到 ANALYSIS 段
    4. 用 Qwen3-VL tokenizer 把 content tokenize 出 offsets
    5. 把字符 mask range 映射到 token tags
    6. 打印对照表 + summary + 必要的 [WARN]

    各阶段退出码（便于 CI / shell 脚本检测）：
        0  正常结束（即使带 [WARN] 也算 0，需要人工看输出判定）
        2  jsonl 文件不存在
        3  transformers 包未装
        4  tokenizer_dir 找不到 / 不是合法 HF 目录
    """
    parser = argparse.ArgumentParser()
    # 默认 jsonl 是 build_sft_dataset_v1.py 的默认输出位置，
    # 这样最常见命令 `python tools/check_loss_mask.py`（从 AutoMoT/ cwd）就能跑通。
    parser.add_argument("--jsonl", type=str,
                        default=str(_AUTOMOT_ROOT / "checkpoints" / "sft_v1_data" / "train.jsonl"))
    # 大多数时间看第 0 条就够；遇到特殊 scenario 想看时再调。
    parser.add_argument("--sample-idx", type=int, default=0)
    # tokenizer 与 LoRA base model 同一个目录（HF tokenizer 文件和 model
    # 权重通常存在一起）。若远程把 tokenizer 单独放，可显式 override。
    parser.add_argument("--tokenizer-dir", type=str,
                        default=str(_AUTOMOT_ROOT / "checkpoints" / "Qwen3-VL-4B-Instruct"))
    args = parser.parse_args()

    jsonl_path = pathlib.Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"[err] jsonl not found: {jsonl_path}", file=sys.stderr)
        sys.exit(2)

    # ---- 阶段 1：读样本 + 显示 assistant text 原文 ----
    # 把 assistant 段直接打印一次，让用户先肉眼确认占位句和 STATUS / SUBGOAL 是不是
    # 长成预期的样子。如果连原文都看着不对（比如 PLACEHOLDER_ANALYSIS 被改了），
    # 后面 regex 和 tokenize 都是浪费时间。
    sample = load_first_sample(jsonl_path, args.sample_idx)
    assistant_text = sample["messages"][-1]["content"]
    print(f"[load] jsonl={jsonl_path} sample_idx={args.sample_idx}")
    print(f"[load] scenario={sample.get('scenario')} run_id={sample.get('run_id')}"
          f" anchor={sample.get('anchor')}")
    print()
    print("===== assistant text =====")
    print(assistant_text)
    print("==========================")
    print()

    # ---- 阶段 2：在 char 级用 regex 找 mask range ----
    # 这一步与 tokenize 无关，是字符串纯文本匹配。如果这里就 match 不到，说明
    # PLACEHOLDER_ANALYSIS 与 LOSS_SCALE_REGEX 之间已经漂移了，必须先修配置再来跑。
    mask_range = find_mask_char_range(assistant_text)
    if mask_range is None:
        print("[WARN] LOSS_SCALE regex 在 assistant text 上匹配不到。"
              "ANALYSIS 段不会被 mask，训练时整段都会算 loss！")
    else:
        # 把匹配区间切片打印，让用户确认"是不是只圈住了 ANALYSIS 段、没溢出到 STATUS"。
        masked_str = assistant_text[mask_range[0]:mask_range[1]]
        print(f"[mask] regex matched chars [{mask_range[0]},{mask_range[1]})  "
              f"-> {masked_str!r}")

    print()
    # ---- 阶段 3：加载 tokenizer（远程必装；本地可能因依赖版本不兼容失败）----
    # 用 try/except 包住 import 是因为 transformers 在某些环境会因 torch 版本错
    # 在 import 时就抛 AttributeError；这里只接 ImportError，更严重的情况留给
    # python 默认 traceback 暴露细节。本地环境用户能看到完整堆栈方便排查。
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError:
        print("[err] transformers 未安装。请在远程或装好 transformers 的环境运行：\n"
              "      pip install transformers", file=sys.stderr)
        sys.exit(3)

    tokenizer_dir = pathlib.Path(args.tokenizer_dir)
    if not tokenizer_dir.exists():
        print(f"[err] tokenizer_dir 不存在: {tokenizer_dir}\n"
              f"      可用 --tokenizer-dir 指向有 Qwen3-VL tokenizer 文件的目录。",
              file=sys.stderr)
        sys.exit(4)

    # local_files_only=True：与 runner 同款离线策略，不允许 hub 联网下载；
    # trust_remote_code=True：Qwen3-VL tokenizer 类在 modeling 包里，必须信任。
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
        trust_remote_code=True,
    )

    # ---- 阶段 4：tokenize → 分类 → 打印对照表 ----
    # tokenize_with_offsets 用 fast tokenizer 拿到 offsets，前提是 from_pretrained
    # 返回的是 fast 版本。Qwen3-VL 默认配 fast tokenizer，所以这里不主动 use_fast 参数。
    ids, offsets, decoded = tokenize_with_offsets(tokenizer, assistant_text)
    tags = classify_tokens(offsets, mask_range)
    print_token_table(ids, offsets, decoded, tags)

    # ---- 阶段 5：聚合指标 + 阈值告警 ----
    # 这两个 WARN 阈值的意义见 summarize() docstring。CI / 自动化场景如果想把
    # WARN 升级成 fail，可以在外面 grep 输出里有没有 "[WARN]" 决定退出码。
    summary = summarize(tags, decoded)
    print()
    print(f"[summary] total tokens = {len(ids)}, "
          f"mask = {summary['n_mask']}, loss = {summary['n_loss']}")
    if summary["n_loss"] < 5:
        print("[WARN] 算 loss 的 token 太少（<5），STATUS/SUBGOAL 监督信号可能稀薄。"
              "确认 PLACEHOLDER_ANALYSIS 是不是把 STATUS 行也吞掉了。")
    if summary["n_mask"] == 0:
        print("[WARN] 没有任何 token 被 mask。检查 LOSS_SCALE_REGEX 与 PLACEHOLDER_ANALYSIS。")
    print_plugin_loss_scale_check(assistant_text)


if __name__ == "__main__":
    main()
