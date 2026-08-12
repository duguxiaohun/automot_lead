# Phase1 Prompt-Aligned LoRA RGB Error Audit

## Scope

This review uses the prompt-aligned production LoRA evaluation at
`lora_static_obstacle_refined_4gpu/20260812_091804`. It reads the saved four-frame
RGB error cases before changing a prompt. Scenario and event names are used only to
locate samples; every conclusion below is based on the visible sequence.

Focused 1:1 production results were `HIGHWAY` F1 `0.9508`, `STATIC_OBSTACLE` F1
`0.7500`, `VULNERABLE` F1 `0.8966`, and `TRAFFIC_LIGHT_ABNORMAL` F1 `0.8468`.
The remaining errors are asymmetric: all four focused tasks have zero false
positives except `STATIC_OBSTACLE`, which has one. Therefore this iteration must
improve recall without broadly relaxing the YES boundary.

## Static Obstacle Review

The 25 focused static-obstacle false negatives do not form one visual class.
Many histories are nighttime, heavy fog, normal moving traffic, an event that is
no longer ahead of ego, or a scene where path overlap cannot be recovered from
RGB. Examples include parked/accident cases `00041`, `00059`, `00062`, `00067`,
`00091`, `00196`, `00356`, `00362`, `00412`, `00432`, `00452`, `00478`, and
`00497`. They are expected limits of the inexpensive `U-E2` route-state proxy;
they are not evidence for asking the model to guess a static obstacle from a
scenario prior.

One repeated, readable pattern is different:

- `case_00201_ConstructionObstacleTwoWays_f58`: all four frames show an
  orange/yellow work-zone closure object fixed in the forward lane. It remains
  small because it is far away, but its road position and lane diversion are
  visible.
- `case_00246_ConstructionObstacle_f36`: all four frames show the same
  orange/yellow lane-closure facility centered in the ego corridor. It remains
  road-fixed while the nearby blue vehicle changes its relative position.

These are genuine visual false negatives, not label-only positives. The old
negative rule grouped all distant objects together. The production prompt now
draws the needed boundary: a distant object stays NO only when its path overlap
cannot be seen; a readable, road-fixed orange/yellow lane-closure facility that
occupies or diverts the traced ego lane is YES.

The sole focused false positive, `case_00163_DynamicObjectCrossing_f107`, shows
ordinary vehicles in a night scene rather than a visible fixed lane closure. It
does not repeat as a separate appearance pattern, so no additional NO wording is
added in this pass.

## Traffic-Light Review

All 17 focused traffic-light false negatives are U-E7 rows. Their RGB histories
include dense fog, darkness, tiny signal heads, signal heads outside the camera
views, and normal-looking red-versus-green phasing. Some broad daytime junctions
show lights, but do not expose two readable conflicting approach arms in the
same conflict area. There is no single visible witness pattern that can be
expanded safely without turning normal phasing into false positives.

The traffic-light rule is therefore unchanged. The current model must continue
to require a readable contradictory or broken-signal witness instead of using a
U-E7/scenario prior. These rows document observability noise in the current
route-state proxy, not a prompt rule to loosen.

## Decision

Only the narrowly evidenced distant-but-readable construction-closure boundary
changes. `answer_policy.py`, the `U-E2` supervision contract, frame index,
route-disjoint split, and eight balanced 1:1 bins are unchanged. Because this is
only a production-prompt change, do not rebuild the dataset. Train a new LoRA
with the new prompt SHA, then compare it with the current adapter using the same
fixed index and a production (non-audit) evaluation. Run a separate audit prompt
only for RGB evidence review.

## Follow-Up Audit: New Prompt-Aligned LoRA

The later audit run at
`lora_static_obstacle_distant_closure_v2_audit_4gpu/20260812_161058` uses an
adapter aligned with the prior distant-closure prompt revision. Its
`STATIC_OBSTACLE` audit recall is `46/64`, but the production model still misses
the two original readable construction cases. The four RGB frames were reread:

- `case_00201_ConstructionObstacleTwoWays_f58` contains a small orange/yellow
  roadwork trailer/board in the centre of the ego lane in every frame. The audit
  text sees the orange object but calls it a slowly approaching vehicle. Its
  claimed motion is not supported by the road-fixed image sequence.
- `case_00246_ConstructionObstacle_f36` contains a small orange/yellow closure
  facility aligned with the ego lane in every frame. The audit text instead says
  that there is no cone, barrier, or parked vehicle ahead, so this is a
  near-horizon object miss.

The same audit can correctly identify a similar yellow trailer plus cones as a
stationary lane closure (`ConstructionObstacleTwoWays`, case `013`, frame `38`),
so the model has the concept. The failure boundary is specifically distant
search and trailer-versus-vehicle identification, not lack of a construction
category.

All ten audit-only static false positives were then reread across all four RGB
frames: `053`, `055`, `117`, `163`, `174`, `270`, `316`, `368`, `460`, and `499`.
They are ordinary lead/parked vehicles, a yellow taxi, a blue van, an emergency
vehicle visibly crossing a junction, or dark/foggy traffic whose static state is
not readable. None is a small orange/yellow roadwork trailer or lane-closure
board fixed in the ego lane. The next production prompt revision is consequently
limited to: scan the ego lane from near pavement through the vanishing point,
recognize the raised board/base/cone/trailer structure, and require road-relative
self-motion before calling that object dynamic. It does not relax ordinary-
vehicle or unreadable-scene negatives.

## Final Prompt Selection

The v3 follow-up wording was evaluated with a newly trained, prompt-aligned LoRA
on the same fixed 512-case test index. It did not repair either production target:
`case_00201` and `case_00246` remained `STATIC_OBSTACLE=NO`. The focused static
result changed from v2 `TP/FP/FN/TN = 39/1/25/63`, F1 `0.7500`, to v3
`38/0/26/64`, F1 `0.7451`. Exact four-answer accuracy also changed from `0.8496`
to `0.8457`; traffic-light F1 fell from `0.8571` to `0.8364`.

Therefore the final Phase1 production prompt is the v2 content, with production
SHA-256 `827b59181b391657c7bd3c97640241902d8905f4cada90861ebb37d931bb633a`.
The v3 near-to-horizon and trailer-structure wording remains a documented
experiment, not a deployed rule. Freeze the current v2 prompt, retain only a
prompt-aligned final adapter, and move future static-obstacle gains to a separate
visual-label, crop/attention, or evidence-supervised experiment rather than
lengthening this answer-only prompt.
