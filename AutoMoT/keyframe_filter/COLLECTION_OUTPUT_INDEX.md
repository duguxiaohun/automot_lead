# keyframe_filter collection_output 索引与保留规则

`AutoMoT/keyframe_filter/collection_output/` 是本机自动调研、逐帧标注和 RGB 审计证据目录。它包含很多大 JSON、contact sheet 和临时 summary，默认不入库、不 push；需要复查时优先读本文和相关小型 MD，再决定是否打开本地证据或重新小范围生成。

## 1. 总原则

- `collection_output/*_result.json` 是逐 scenario 帧级 RS/EVENT 标签大文件，很多单文件数百 MB 到数 GB；它们是 SFT / probe 的本地数据源，不通过 git 携带。
- `collection_output/multi_scenario_collection.json` 是聚合大文件，当前约 17 GB，不入库、不 push。
- `collection_output/frame_rs_annotation_summary.json` 是小 summary，但可由 result 重新生成；除非未来明确改成稳定配置，否则不加入白名单。
- RGB sheet、montage、candidate anomalies、route/town/scenario/global summary、logs 都是本地证据缓存，不入库、不 push。
- 需要共享的新结论，应沉淀为 `README.md`、规则文档、审计归档 MD 或小型规则配置，而不是提交大 JSON/RGB。

## 2. 当前主要内容

| 内容 | 用途 | Git 规则 |
|---|---|---|
| `*_result.json` | 每个 scenario 的逐帧 RS/EVENT 标注结果，供 SFT/probe/build_dataset 读取 | 不入库；远端需要时同步数据或重跑生成 |
| `multi_scenario_collection.json` | 多 scenario 聚合输出 | 不入库；体量过大 |
| `frame_rs_annotation_summary.json` | 当前输出分布摘要 | 默认不入库，可再生 |
| `phase1_four_question_audit/` | Phase1 四问审计主目录 | 只放行轻量 JSON/JSONL/matrix，RGB/summary 不入库 |
| `phase1_fullframe_rgb_real_visual_v1/` | Phase1 real visual v1 早期证据 | 已被 dated final review supersede，并已物理删除 |
| `phase1_fullframe_rgb_original_labels/` | Phase1 original-label 旧证据 | 已被最终审计口径取代，并已物理删除 |
| `phase1_fullframe_rgb/` | Phase1 早期局部调试证据 | 已物理删除 |

Phase1 目录的详细主入口、legacy/superseded 关系和白名单例外见
[`PHASE1_COLLECTION_OUTPUT_INDEX.md`](PHASE1_COLLECTION_OUTPUT_INDEX.md)。

## 3. 代码读取关系

当前代码/训练常见读取方式：

- `AutoMoT/qwen3vl_local/sft_v5/`：数据来自 `AutoMoT/keyframe_filter/collection_output/*_result.json`，但训练前跳过 `noScenarios_result.json`、异常时长 route、数据缺失 skip、缺 XML/RGB/meta/annotation 的 route。
- `AutoMoT/qwen3vl_local/sft_base/` / `sft_base_simple/`：复用同一批 `collection_output/*_result.json` 构造 RS/EVENT 或二分类基线。
- `AutoMoT/keyframe_filter/quick_start.py annotate-rs`：生成或重建 `*_result.json` / summary。
- `AutoMoT/keyframe_filter/rs_full_frame_review.py`：读取 LEAD RGB + 当前标注并输出本地 contact sheet / review summary。

因此不要随意改名或移动 `collection_output/*_result.json`。如果确实要迁移路径，必须同步修改对应 build_dataset/eval/probe 文档和代码里的默认路径或 CLI 参数说明。

## 4. 推荐复用流程

### 查 ROAD_STRUCTURE / EVENT 当前规则

1. 先读 `README.md` 的快速说明。
2. 读 `ROAD_EVENT_CLASSIFICATION_PLAN.md` 看总语义、runtime 门控和错帧回查流程。
3. 读 `ROAD_EVENT_CANDIDATE_MAPPING.md` 看 Qwen/probe 可解析候选池。
4. 读 `ROAD_STRUCTURE_PER_SCENARIO_RULES.md` 看逐场景规则、阈值、证据口径。
5. 若问题涉及 2026-07 老 RGB 审计，读 `ROAD_EVENT_RGB_AUDIT_ARCHIVE_202607.md`，不要恢复已删除的散落旧 MD。

### 查 Phase1 四问标签

1. 先读 `PHASE1_COLLECTION_OUTPUT_INDEX.md`。
2. 再读 `PHASE1_FOUR_QUESTION_RGB_AUDIT_20260809.md`。
3. 标签主表用 `phase1_four_question_audit/phase1_four_question_answer_table.json`。
4. 需要证据时复用 `phase1_four_question_audit/full_route_rgb_label_review_20260809/` 的本地 sheet。

### 重跑或新增证据

- 小范围规则调试：优先输出到 `/tmp/...`。
- 需要保留到 workspace 供多轮复查：输出到 `collection_output/<task_name>_YYYYMMDD/` 或明确命名的 scenario result。
- 新增高成本人工复核：同步新增/更新小型 MD，说明证据路径、覆盖范围、判定口径和复用规则。
- 不要把新 RGB 目录、large JSON、per-route summary 直接加入 git。

## 5. 白名单例外

目前 `collection_output/` 下唯一允许精确 add / push 的是 Phase1 四问轻量标签结果：

- `phase1_four_question_audit/phase1_four_question_answer_table.json`
- `phase1_four_question_audit/answer_table_partial.json`
- `phase1_four_question_audit/manual_visual_audit_notes.jsonl`
- `phase1_four_question_audit/critical_batch/phase1_four_question_matrix.json`
- `phase1_four_question_audit/highway_flow_batch/phase1_four_question_matrix.json`
- `phase1_four_question_audit/motion_parking_batch/phase1_four_question_matrix.json`
- `phase1_four_question_audit/obstacle_single_batch/phase1_four_question_matrix.json`
- `phase1_four_question_audit/obstacle_twoways_batch/phase1_four_question_matrix.json`
- `phase1_four_question_audit/remaining_batch/phase1_four_question_matrix.json`
- `phase1_four_question_audit/signal_control_batch/phase1_four_question_matrix.json`
- `phase1_four_question_audit/vehicle_turning_batch/phase1_four_question_matrix.json`
- `phase1_four_question_audit/full_route_rgb_label_review_20260809/manual_full_sheet_notes_20260809.jsonl`
- `phase1_four_question_audit/full_route_rgb_label_review_20260809/manual_table_gap_combo_notes_20260810.jsonl`

`no_scenarios_batch`、RGB、summary、logs 和 candidate anomalies 不在例外内。

## 6. 清理边界

- 可以删除或重建可再生的本地 RGB/contact sheet/cache，但删除前必须确认没有当前任务 notes/table 依赖该路径。
- 2026-08-10 已清理 Phase1 顶层旧 RGB 目录：`phase1_fullframe_rgb_real_visual_v1/`、
  `phase1_fullframe_rgb_original_labels/`、`phase1_fullframe_rgb/`；不要恢复它们。
- 2026-08-10 已清理 Phase1 主目录内部未被最终表/notes 引用的旧视觉目录：
  `manual_visual_contact_chunks/`、`manual_visual_run_overviews/`、`manual_visual_town_triplets/`，
  以及已排除的 `no_scenarios_batch/`；不要恢复它们。
- Phase1 的 `full_route_rgb_label_review_20260809/` 已被 notes 引用，默认不要移动；若要释放磁盘，先生成迁移 manifest 并更新 notes/table 路径。
- Phase1 各 `*_batch/sheets/*.jpg` 虽不入库，但目前仍被答案表和 gap notes 的
  `evidence_sheet` 字段引用，暂时保留本地。
- `*_result.json` 如果被 SFT/probe 任务使用，不能在未同步训练路径的情况下改名或删除。
- 如果只是想减少 git 噪音，优先改 `.gitignore` 或文档索引，不要把数据文件硬塞进版本库。
