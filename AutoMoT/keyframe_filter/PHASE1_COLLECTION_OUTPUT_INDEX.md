# Phase1 collection_output 索引与复用规则

这个文档专门解释 `AutoMoT/keyframe_filter/collection_output/` 里 Phase1 四问相关目录为什么很多、哪些是最终结果、哪些只是 RGB 证据缓存。后续如果继续复核 `scenario × RS × EVENT` 四问答案，先读这里，再读审计摘要，不要一上来重新批量生成 RGB sheet。

## 结论先行

- 最终四问答案表只看：
  `collection_output/phase1_four_question_audit/phase1_four_question_answer_table.json`
- 最终人工 notes 只看：
  `collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809/manual_full_sheet_notes_20260809.jsonl`
  和
  `collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809/manual_table_gap_combo_notes_20260810.jsonl`
- 最终 RGB 证据缓存只看：
  `collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809/`
- `phase1_fullframe_rgb_real_visual_v1/` 与最终目录基本是同一批视觉证据的早期版本；最终目录多了人工 notes 和 gap notes，后续默认不要再优先使用 v1。
- `phase1_fullframe_rgb_original_labels/` 是旧 original-label 全帧扫图证据，体量更大，已被本轮 real visual + manual notes 审计口径取代；只在追溯旧生成过程时查看。
- `phase1_fullframe_rgb/` 是早期局部调试目录，只覆盖少量 `CrossJunctionDefectTrafficLight`，不是全量入口。

## 目录对应关系

| 路径 | 性质 | 后续怎么用 |
|---|---|---|
| `phase1_four_question_audit/phase1_four_question_answer_table.json` | 最终 `scenario × RS × EVENT` 四问监督标签 | 训练/提示词/复核都以它为主表 |
| `phase1_four_question_audit/answer_table_partial.json` | 早期 partial 表 | 只作历史 diff，不覆盖最终表 |
| `phase1_four_question_audit/manual_visual_audit_notes.jsonl` | 早期 route 级人工 notes | 只作补充证据 |
| `phase1_four_question_audit/*_batch/phase1_four_question_matrix.json` | 按场景族拆分的轻量 matrix | 用来追踪组合级来源；`no_scenarios_batch` 不参与本轮四问 |
| `phase1_four_question_audit/*_batch/sheets/*.jpg` | 早期 batch RGB contact sheet | 本地证据缓存，不 push；只有 gap 回查时再看 |
| `phase1_four_question_audit/manual_visual_contact_chunks/` | 早期人工 contact chunk | 本地证据缓存，不作为最终入口 |
| `phase1_four_question_audit/manual_visual_run_overviews/` | 早期 run overview | 本地证据缓存，不作为最终入口 |
| `phase1_four_question_audit/manual_visual_town_triplets/` | 早期 town triplet 抽样图 | 本地证据缓存，不作为最终入口 |
| `phase1_four_question_audit/full_route_rgb_label_review_20260809/` | 最终全量 RGB + RS/EVENT 标签证据缓存 | 需要看 RGB 时优先使用这里 |
| `phase1_fullframe_rgb_real_visual_v1/` | real visual v1 早期输出 | 已被 `full_route_rgb_label_review_20260809/` supersede |
| `phase1_fullframe_rgb_original_labels/` | original-label 全帧旧输出 | 只追溯旧标签时查看 |
| `phase1_fullframe_rgb/` | 早期局部调试输出 | 不作为全量审计入口 |

## 当前证据体量

这几个目录看起来“重复”，主要因为审计从小批量矩阵、早期 contact sheet、全帧 original-label、real-visual v1，最后收束到带人工 notes 的 dated final review。保留它们是为了证据链可回查，不代表每个目录都是一个新的最终版本。

| 目录 | 文件数 | 约大小 | 说明 |
|---|---:|---:|---|
| `phase1_four_question_audit/full_route_rgb_label_review_20260809/` | 5,093 | 2.0 GB | 最终 dated review，含 3,642 张 JPG、1,447 个 JSON、人工 notes |
| `phase1_fullframe_rgb_real_visual_v1/` | 5,091 | 1.9 GB | 与最终 dated review 的相对文件几乎一致，少人工 notes/gap notes |
| `phase1_fullframe_rgb_original_labels/` | 14,209 | 4.4 GB | 旧 original-label 全帧证据，已不作为主入口 |
| `phase1_fullframe_rgb/` | 139 | 37 MB | 早期局部调试 |

## 复用流程

以后做类似“逐帧 RGB + 标签复核”任务时，按这个顺序走：

1. 先读最终答案表：
   `collection_output/phase1_four_question_audit/phase1_four_question_answer_table.json`
2. 再读最终人工 notes：
   `full_route_rgb_label_review_20260809/manual_full_sheet_notes_20260809.jsonl`
3. 若某个组合只在表里、全量 notes 没覆盖，再读：
   `full_route_rgb_label_review_20260809/manual_table_gap_combo_notes_20260810.jsonl`
4. 需要打开 RGB 时，优先使用 notes 里记录的 `sheet` 路径，或到：
   `full_route_rgb_label_review_20260809/<Scenario>/<Town>/<run_id>/sheets/all_frames_*.jpg`
5. 只有当上述证据缺失、损坏或需要新增采样维度时，才生成新的 RGB sheet；新输出必须放到带日期的新目录，例如：
   `phase1_four_question_audit/full_route_rgb_label_review_YYYYMMDD/`
   并同步更新本索引和审计摘要。

## 不直接物理合并目录的原因

最终表和 JSONL notes 中已经记录了证据 sheet 路径。直接移动或删除旧 RGB 目录会造成两类问题：

- 旧路径失效，之后很难复查“当时到底看的是哪张图”。
- 早期 matrix / gap notes 指向的辅助 sheet 可能被破坏，影响追责和回归。

因此当前采用“逻辑合并”：用本索引指定唯一主入口，把早期目录标成 legacy/superseded。若确实需要释放磁盘空间，应单独执行一次 archive/cleanup 任务：先重写 notes/table 中涉及的路径或生成迁移 manifest，再移动到统一 `_legacy_phase1_rgb/` 或删除；不要在普通复核任务里顺手清理。

## Git / push 边界

允许精确 add / push 的只有轻量标签结果：

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

RGB contact sheet、per-route/per-town/per-scenario/global summary、candidate anomalies、logs、montage 等仍是本地证据缓存，不入库、不 push。
