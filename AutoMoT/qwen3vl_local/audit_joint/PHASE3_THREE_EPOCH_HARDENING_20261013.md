# Phase3 三轮入口：失败诊断与常驻回归补强

2026-10-13。接续 `PHASE3_THREE_EPOCH_REVIEW_20261013.md`。三轮正常训练路径的准入结论不变；没有新依据修改标定、阈值、提示词或训练超参。

## 发现与修复

原 `package()` 在保护逻辑外解析 `epoch_result.json`。临时目录写入截断 JSON 后，原生函数抛出 `JSONDecodeError`，没有生成 ZIP。这是失败交接缺口，不是正常三轮训练已经失败的证据。

现在损坏、非对象及非法轮数进入 `handoff.errors`，报告不可读/缺失成员进入 `missing_files`，继续生成 `incomplete` 包并返回 2。原文件不覆盖。shell 将同一 `EPOCHS` 显式传入打包；即使守护在启动前拦下、尚无状态文件，仍记录请求的三轮及两模式缺失的逐轮证据。请求与结果轮数矛盾不能通过。手工打包未传轮数时仍兼容读取有效状态，旧单轮结果默认一轮。

把此前临时验证固化在 `tests/test_server_phase3_epoch.py`：真实 CPU 优化器/原生三轮循环，每轮十事件各 4；第 41 / 81 个输入分别拒绝，立即停止，仅保留之前完整轮次的消费回执。图像打包、损失和线性模型为合成替代，不能当作 Qwen 或四卡验证。保留正式 Phase3 训练源码，避免改变已验收的生产合同。

## 验证

- 51 项相关测试通过，2.07 秒，其中本轮新增 12 项。覆盖成功/失败编排、原生循环、损坏状态、错误轮数与 CLI。
- 原生 CLI：无状态/截断 JSON 两种失败场景均退出 2，生成 544 / 661 字节包，外层 SHA、成员 SHA、CRC 全部通过；均保留三轮范围。
- 新 475 源清单全部匹配；实际已收首轮 ZIP 的原生三轮准入仍 `verified`。shell 语法检查通过。
- 首次误用了缺 torch 的默认 Python；改用本机已有 pvi 环境。新测试初版漏计独立 `INVALID=0` 项，出现 3 个断言失败，修正断言后通过；失败日志保留。未修改生产计数行为。
- 本轮不重复既有完整 2419 项回归、服务器 readiness、建库或首轮。

新清单为 `release_sources_phase3_three_epoch_hardening_20261013.json`；旧 multiepoch 清单及独立复审报告不改。轻量回执 `phase3_three_epoch_hardening_verification_20261013.json`，原始证据 `checkpoints/phase3_three_epoch_hardening_20261013/`。

## 下一步

发布并沿既有同步路径更新服务器后，继续按 `PHASE3_ADMISSION_FIRST_EPOCH_20261013.md` 顶部命令执行固定三轮。无需重建数据或重复首轮。两模式每轮十事件各 1024；val/test 各事件 32/38。最终交 `phase3_epoch_report.zip` 和 `phase3_epoch_package.json`，不上传权重。

训练成功仍只表示开发实验链路与等量消费通过；质量看三轮结果，已经查看过的 test 不能称未曝光盲测。Phase4 的 U-E5 真实正例、新 v53 正式池及跨任务隔离仍未关闭。本轮没有远程/GPU 训练、新 RGB 目视或标签回放，未 push。
