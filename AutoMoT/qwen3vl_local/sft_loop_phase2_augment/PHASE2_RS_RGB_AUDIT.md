# Phase2 RS RGB Audit

## Evidence Scope

Phase2 does not create a second copy of LEAD RGB. It reuses the completed Phase1 full-route visual audit at:

`keyframe_filter/collection_output/phase1_four_question_audit/full_route_rgb_label_review_20260809/`

远程训练环境不需要携带这个不入库的大目录。`phase2_rgb_audit_coverage.json` 是随
`sft_loop_phase2` 提交的紧凑覆盖证明，只保存场景/Town 的已审计 route 数；构建索引时远程会自动读取它，本地有完整审计目录时才额外读取并核对原始审计摘要。

The audit summaries report completed full-frame review for 42 scenarios, 197 scenario-Town pairs, and 582 routes. Every scenario-Town pair has at least one completed route (in practice the existing protocol sampled three routes per Town). `visual_audit.py` verifies this contract before writing its lightweight manifest.

The local checkout contains both the completed contact sheets and the raw LEAD RGB through `lead_data/<Scenario>` symlinks. The contact sheets are the chronological human-review entry point; the dataset builder reads the original RGB paths. The conclusions below therefore come from actual frame sequences with RS overlays, not from scenario names or synthetic descriptions.

## What The RGB Shows

### R1: Ordinary Same-Direction Road

- Representative foggy local-road sheets show an ego lane continuing beside sidewalks, fences, buildings, shoulders, or ordinary traffic without an immediate controlled junction. A broad/straight/empty road remains R1 when no other rule structure is visible.
- R1 is also the correct exit state around signalized or priority scenarios when the current frame is before/after their local conflict area. It must not inherit R4/R5 from the scenario title.
- Roundabout and parking-side road segments remain R1 unless a separate visible signal/priority junction controls ego.

### R2: Opposing-Lane Sharing

- Reviewed TwoWays sheets show the useful distinction is not the obstacle itself but the usable corridor: centreline/opposite lane is close, oncoming traffic participates, or parked/blocked lateral space leaves a passage that needs an opposite-lane decision.
- Rain/night contact sheets can make lane lines and oncoming geometry unreadable. The current labels explicitly carry `r2_lacks_xodr_opposite_lane_confirmation` on such frames. These are retained as `visual_label_risk` rather than treated as clean visual positives.
- A median-separated carriageway, normal multi-lane same-direction traffic, or a curbside vehicle that leaves ego's lane open is not R2.

### R4: Signal-Controlled Intersection

- Signalized left-turn and T-junction sheets show a local signal mast/overhead head, cross street or turning geometry, stop-line/crosswalk/median opening, and normally a local signal-controlled conflict area. The colour itself is not the class: working hardware at this junction is.
- Defective-light frames remain R4 when the signal hardware is visible; Phase1's abnormal-light question handles its failure state.
- Several reviewed rows have only weak/far signal evidence, especially in fog. Reasons such as `r4_meta_tl_without_strong_context_requires_rgb_confirmation` are not silently converted into clean R4 samples.

### R5: Priority / Unsignalized Intersection

- Non-signalized left-turn sheets show local T/cross geometry, STOP marking/stop-yield behaviour, side-road conflict, or cross traffic with no working traffic-signal rule.
- T-junction geometry is included: a T is R5 when it is unlit or priority/stop/yield controlled, and R4 when signal hardware governs it.
- Some annotations explicitly flag `nonsignalized_with_signal_topology_conflict`. These must be reviewed as visual-label-risk rows, never used to teach Qwen that a visible signalized junction is R5.

### R3 Robust Negative

- HighwayExit sheets visibly contain a controlled multi-lane corridor with barriers and separated traffic. There is no reason to force it into R1 merely because lane markings continue.
- In Phase2, R3 is intentionally encoded as `RS1=NO, RS2=NO, RS4=NO, RS5=NO`. It is reported separately as `is_highway_negative`, so it cannot disappear inside ordinary NO examples.

## Dataset Label Contract

| Question | YES exactly when | NO includes |
|---|---|---|
| `RS1` | `primary_road_structure == R1` | R2/R3/R4/R5 |
| `RS2` | `primary_road_structure == R2` | R1/R3/R4/R5 |
| `RS4` | `primary_road_structure == R4` | R1/R2/R3/R5 |
| `RS5` | `primary_road_structure == R5` | R1/R2/R3/R4 |

The source label is never rewritten. Each row records `visual_label_risk` and `visual_label_risk_reasons`. Clean dataset construction excludes only the listed visual-risk rows by default; `--include-visual-risk` creates an explicit robustness/noisy-label variant. Both variants preserve route-disjoint train/val/test splits and the final eight-bin exact-balance protocol.

## Prompt Changes Must Follow Evidence

Before changing `prompts.py`, first run the fixed base or prompt-aligned LoRA audit with `--audit-prompt`, inspect `task_cases/<RS>/cases*.jsonl` plus the matching `error_cases/<RS>/rgb/`, and decide whether the failure is:

1. visible evidence missed by the model;
2. a rule boundary that needs a clearer prompt; or
3. a `visual_label_risk`/unobservable label that must not be taught as a clean RGB fact.

Do not change the prompt merely because a scenario is named `TwoWays`, `Signalized`, `T_Junction`, or `Highway`.
