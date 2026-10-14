# v57收尾：失败交接修复，继续正式池验收

本次发布与服务器最新步骤见[SERVER_V57_RELEASE_RGB_20261014.md](SERVER_V57_RELEASE_RGB_20261014.md)，下文“未push”描述保留为此前审计时点。

2026-10-14。STOP停车位置与横穿规则已原生接入，上一轮准入方向不变。本轮不扩大本地标定审计，也没有发现新的修改阈值或提示词依据。

## 已复现并修复

`server_finish.package()`直接解析各JSON。临时目录只写入截断的`build_result.json`，旧入口即抛`JSONDecodeError`，没有生成ZIP。长时间建库被中断后，这会阻止交回已有诊断；不是STOP标签缺陷，也不要求重新建库来修复。

现在逐文件收集：损坏JSON、非对象JSON、无效预算、超出诊断预算的文件保留路径、可读取的SHA/长度、限长预览及原因，继续生成诊断包。单个JSON输入限10MB、正常成员序列化总预算20MB，给错误、完成回执和日志留余量，最终仍由原打包器验证30MB限制、CRC与成员SHA。保留原始文件和正常报告，不用截断内容冒充完整报告。

出现收集错误时`finish_receipt.status=blocked_or_interrupted`并写明`diagnostics_complete=false`。外层包`verified`只表示包完整性通过，不能据此开训。已经生成的handoff仍不覆盖。磁盘完全无法写入或输出目录不可写不在这项容错保证内。

两份Phase4入口改用新的`release_sources_v57_handoff_20261014.json`，501个文件匹配；旧发布清单原样保留。此次仅交接器、测试及入口发布指针变化，生产任务仍v57，老师/数据/短视图合同不变，已有效的v57回放无需因此重建；旧v54–56不能混入新版正式池。

## 实际验证与RGB复核

- 121项相关回归通过；1项需要torch/peft的模型参数计数测试明确未选入，未宣称完整模型测试通过。范围包括交接失败、关键支持、开发模式豁免、首轮消费回执、STOP目录及绑定失效。
- 合成截断报告经真实package入口生成967字节诊断ZIP，CRC/成员SHA通过，内部状态仍阻塞。
- 两模式分别原生重载既有20条R-E5/proceed YES、4物理组；属于同一批既有样本，不相加为40个独立样本。
- 实际复看AccidentTwoWays Town12 `26_0` 第32–38帧、Town13 `1153_0` 第35–41帧，14个不同RGB，42份源SHA核验通过。都是已暴露开发窗口，非盲审、非新增数据生成。Town12两侧车辆仍可见，Town13右侧车辆及前方车辆仍在画面中；画面不能单独证明完整停车历史和米级净空。本轮没有足以改变现有规则或提示词的新证据。
- 未运行远程、GPU、正式池重建或训练，未push。没有修改用户`.vscode`。

完整本机证据：`checkpoints/v57_finish_handoff_20261014/`；轻量回执见`v57_finish_handoff_verification_20261014.json`。

## 接下来只做这些

发布同步后，沿[现有v57服务器命令](STOP_POSITION_V57_20261014.md)使用新输出目录运行`server_phase4_existing.sh`，只使用现有数据。不要重复Phase3、readiness或STOP盘点。本轮未发布，不能假定服务器已取得这些文件。

1. 一次v57正式CPU池建库；检查实际过滤后的关键监督、物理路线隔离、两模式配对，以及train/val/test十事件等量。局部CPU计划每事件1024次不能替代全池及GPU消费证据。
2. 完成或失败均交回`handoff/fullflow_report.zip`和`handoff/fullflow_package.json`，均不包含RGB或权重。先确认包内准入状态，不仅看ZIP verified。
3. 通过后两模式各完整一轮、保存重载和固定test，交回首轮回执。无需逐题人工标注；U-E5放行YES缺格仍只在existing_data_development下记录限制，不补采、不用其它边补格。YES:NO不要求等量，十事件之间仍要求1:1。

联合G0与实际模型能力仍不能由本轮诊断或CPU计划代替验收。
