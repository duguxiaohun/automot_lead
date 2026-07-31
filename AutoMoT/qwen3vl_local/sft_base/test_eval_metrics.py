"""SFT base eval metrics 源码合同测试。

本测试故意不 import eval.py；eval.py 会 import torch 和模型训练入口，在本地轻量
环境里可能因为 CUDA/torch DLL 初始化失败而无法加载。这里用源码合同守住
RS×候选数分层的关键落点。
"""

from __future__ import annotations

import pathlib


def main() -> None:
    """验证 EVENT 指标同时按候选数和 RS×候选数分层输出。"""

    eval_py = pathlib.Path(__file__).with_name("eval.py")
    src = eval_py.read_text(encoding="utf-8")
    required = [
        'rs_bucket_prefix = f"q2_rs_{frame.rs_label}_candidate_count_{candidate_count_bucket}"',
        'rs_prefix = f"q2_rs_{rs}_candidate_count_{n}"',
        'rs_candidate_count_report[rs] = rs_report_by_count',
        '"q2_rs_candidate_count_report": rs_candidate_count_report',
        "`q2_candidate_count_report` 和 `q2_rs_candidate_count_report`",
    ]
    for needle in required:
        assert needle in src, needle
    print("[test_eval_metrics] ok")


if __name__ == "__main__":
    main()
