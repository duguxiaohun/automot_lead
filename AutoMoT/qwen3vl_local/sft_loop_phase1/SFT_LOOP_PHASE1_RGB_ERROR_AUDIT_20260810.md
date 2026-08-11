# Phase1 RGB Error Audit

## Scope

This audit reviews every main-question error exported by the LoRA evaluation at
`sft_loop_phase1_eval/lora_zero_shot_prompt_4gpu/20260810_211451`:

- 104 main-question errors, each with four stitched RGB history frames;
- HIGHWAY 8, OBSTACLE 32, VULNERABLE 17, and TRAFFIC_LIGHT_ABNORMAL 47;
- RGB was reviewed frame by frame, not inferred from scenario or event names.

`YES` means the fact must be visible in the current four-frame history. A
scenario-level answer can still be true later in the route, but is unsuitable as
supervision for a frame where the needed visual evidence is absent.

## Findings

| Task | RGB finding | Cases | Consequence |
| --- | --- | --- | --- |
| HIGHWAY | Clear model false positives: an ordinary T-junction road or a built-up/local road was promoted from guardrails, lanes, darkness, or fog. | 33, 345, 465 | Keep the limited-access topology requirement. Darkness/fog must not turn a weak cue into highway evidence. |
| HIGHWAY | The controlled corridor is visible, but the model rejected it because it searched too literally for a ramp or merge. | 61 | Retain mainline-without-ramp positive rule. |
| HIGHWAY | A traffic-light-controlled rural arterial was labeled highway although the four frames do not show a controlled-access chain. | 103, 507; also label-boundary concern for 251, 426 | Review labels/definition before pushing highway recall further; prompt changes alone cannot resolve contradictory visual criteria. |
| OBSTACLE | Clear missed path conflicts: an ambulance crosses the junction, vehicles/queues occupy the junction or only usable gap, a parked/crossing car intrudes, or a construction board closes the lane. | 30, 54, 91, 225, 243, 316, 368, 402, 414, 497 | Add an explicit path-overlap test and inspect all history frames. |
| OBSTACLE | The route-level positive event is not visible in the four frames: clear road, ordinary lead vehicle, or nearly black/fog-obscured view. | 4, 8, 12, 46, 74, 79, 89, 121, 213, 254, 259, 275, 394, 404, 440, 492 | Mark as unobservable for a current-RGB label; do not teach the model to guess from the scenario. |
| OBSTACLE | The model predicted `YES` when the annotation is `NO`, but a construction board visibly blocks the ego lane. | 348 | High-confidence label error for this frame. Cases 70 and 215 need a tighter lane/path check, not broad nearby-vehicle positives. |
| VULNERABLE | Clear small people occur at a corner/crosswalk or close sidewalk edge, often visible in only one frame. | 132, 325, 372 | Add a deliberate second pass over crosswalks, curb corners, sidewalks, and image edges; one clear frame is enough. |
| VULNERABLE | The positive scenario is not visually supported in this short history, especially at night; several frames show only vehicles or an empty road. | 3, 18, 67, 127, 178, 195, 228, 274, 326, 413, 490, 503 | Current frame labels need an observable/unknown state rather than scenario-wide `YES`. |
| VULNERABLE | A person-shaped roadside target is visible near the right-side sidewalk, but it is outside the immediate forward/turning conflict area in the reviewed frame. | 458 | Do not count mere visibility as a label error; this is a boundary case for the decision-relevance rule. |
| TRAFFIC_LIGHT_ABNORMAL | Clear true model false negatives: two or more heads governing crossing approaches are green at the same junction. The model often described these images as a “consistent normal green phase”. | 6, 24, 72, 118, 124, 142, 257, 303, 317, 320, 352, 388, 392, 410, 436, 472, 484 | Force a same-junction witness-pair check. A visible pair of crossing green approaches is `YES`, even when only one history frame is clear. |
| TRAFFIC_LIGHT_ABNORMAL | Signal-fault `YES` is not visually provable in the current history: lamps are unreadable from fog/glare/night, only one head is seen, or no relevant head is visible. | 13, 55, 81, 90, 104, 116, 176, 204, 242, 246, 248, 269, 318, 319, 327, 342, 347, 449, 451, 453, 491 | This is the largest source of irreducible light-task error. It is not valid RGB-only supervision unless the task is redefined as “scenario fault active”, which the prompt intentionally must not infer. |
| TRAFFIC_LIGHT_ABNORMAL | The model called a normal red/green phase abnormal because it did not map the heads to crossing movements. | 205, 281, 289 | `YES` now requires a visible conflicting GREEN witness pair or a clearly broken head; red plus green alone remains normal by default. |

## Signal-Light Detail

The clearest defect pattern is not merely “many green pixels.” In the reviewed
RGB, a valid witness consists of readable illuminated green heads on two arms
whose vehicles would enter the same cross, T, angled, or multi-arm junction
conflict box. Cases 6, 24, 72, 118, 124, 142, 257, 303, 388, 392, 410, and 472
show this directly. Cases 317, 320, 352, 436, and 484 show incompatible signal
states through the same junction geometry. Several valid examples are obvious
only in one of the four images, so requiring the newest image alone would throw
away the evidence.

Conversely, cases 205, 281, and 289 show why “red plus green” is not a defect:
the colors govern different non-conflicting approaches. A model must first map
signal heads to the same conflict box, then compare permissions.

## Prompt Change Applied

`prompts.py` now adds three RGB-grounded checks:

1. A dedicated pass over all four frames and all three views for small road
   users, vehicle noses, door/cone intrusions, crosswalks, curb corners, and
   signal heads.
2. An obstacle path-overlap rule: partial intrusion into ego's lane, junction
   box, turning arc, or only usable gap is enough; a dramatic crash shape is
   unnecessary.
3. A signal visual-witness rule: identify one junction, assign readable heads
   to approaches, and require two green permissions for paths through the same
   conflict box (or an obviously broken head). The audit output must name the
   witness frame/heads instead of asserting that a phase is “normal”.

## Required Dataset Follow-up

The answer table is indexed by `scenario + RS + EVENT`; it cannot express that
the event exists in a route while the four current RGB frames are before it,
after it, occluded, or too dark to observe. Do not silently flip every listed
case: some labels may be correct as a route state. Add a frame-level audit field
for each question, for example `visible_yes`, `visible_no`, or
`not_observable`, then:

- train the RGB-only four-question model only on `visible_yes` / `visible_no`;
- exclude `not_observable` rows from the corresponding focused train/eval bin;
- retain the original scenario/RS/EVENT answer as route-state metadata for
  later memory or non-visual tasks;
- manually correct high-confidence reversed rows, beginning with obstacle case
  348; keep vulnerable case 458 as a decision-distance boundary example unless
  a fuller route review proves it enters ego's conflict area.

The next evaluation should compare the current prompt with this change on the
same fixed test index and separately report metrics after excluding
`not_observable` labels. Otherwise the light, obstacle, and vulnerable F1s
continue to mix perception errors with unanswerable frames.

## Prompt-Delta Evaluation (2026-08-11)

The follow-up comparison uses the exact same 512 fixed evaluation cases and
the same LoRA adapter. The prior run is
`sft_loop_phase1_eval_old/lora_zero_shot_prompt_4gpu/20260810_211451`; the
RGB-feedback run is
`sft_loop_phase1_eval/lora_zero_shot_prompt_4gpu/20260811_092904`.

| Primary task | Prior F1 | RGB-feedback F1 | TP / FP / FN / TN change | RGB conclusion |
| --- | ---: | ---: | --- | --- |
| HIGHWAY | 0.9385 | 0.9385 | unchanged: 61 / 5 / 3 / 59 | The highway topology rules are stable. Do not perturb them. |
| OBSTACLE | 0.6863 | 0.7037 | 35 / 3 / 29 / 61 -> 38 / 6 / 26 / 58 | The path-overlap pass recovered three real positives, but also turned normal following/roadside parking into three false positives. |
| VULNERABLE | 0.8496 | 0.8673 | 48 / 1 / 16 / 63 -> 49 / 0 / 15 / 64 | The close-target pass helped. The remaining night example 408 is visually too dark to verify from RGB. |
| TRAFFIC_LIGHT_ABNORMAL | 0.4598 | 0.5055 | 20 / 3 / 44 / 61 -> 23 / 4 / 41 / 60 | The witness wording recovered three positives, but the adapter still does not reliably connect its written evidence to the final answer. |

Overall four-answer exact match increased from `0.6992` to `0.7148`.

### New Obstacle Boundary

The regression cases are visually specific rather than abstract:

- 350: a yellow taxi is parked at the curb/parking side, outside the ego lane;
- 446: a white car with brake lights is a normal same-lane lead vehicle;
- 463: parked/oncoming cars compress the visual street view but do not occupy
  ego's usable lane;
- 480: a black SUV ahead is ordinary following traffic with a usable gap.

The new positive 221 shows the complementary rule: a white sedan occupies the
marked turn/ego corridor at the junction, so it remains an obstacle even though
it resembles a parked car. The prompt therefore now requires *actual crossing
or occupation of the usable corridor*, rather than any nearby body, brake light,
or apparent short gap.

### New Signal Boundary

Some signal improvements are real: cases 72, 142, 392, 410, and 484 became
positive after the multi-view check. But the model also outputs statements such
as "no conflicting green" followed by a final `YES`, or describes several
green heads without proving their approaches conflict. Conversely, it misses
examples where an older history frame is the only clear contradictory frame.

The prompt is therefore tightened to a single operational test: `YES` requires
a readable pair of green heads that authorize crossing paths in one identifiable
junction, or an obviously broken governing head. Multiple green lights, an
all-red/all-green appearance, or a scenario name alone is not evidence until
their approach geometry is readable. This is deliberately more faithful to the
RGB task, but it exposes the remaining label problem: many `YES` rows describe
an active scenario state that is invisible in their current four frames (for
example vulnerable case 408 is almost black; several traffic-light rows show
only a normal phase or no readable signal head).

Do not enlarge the data/epoch budget or treat a result as final on the present
labels yet. A prompt-aligned LoRA run on the existing fixed index is still
needed to measure the answer-only loop, but it must be followed by
per-question `visible_yes` / `visible_no` / `not_observable` status for its
current-frame samples, starting with the 128 primary signal-light cases and
then the obstacle/vulnerable error cases. The next larger training set should
only balance observable YES/NO samples for its focused task. This is a
label-schema refinement, not a wholesale replacement of the scenario + RS +
EVENT answer table.

## Over-Tightening Check (2026-08-11)

The existing LoRA was trained with an earlier prompt, then evaluated with the
strict witness/ordinary-lead wording above at
`sft_loop_phase1_eval/lora_rgb_boundary_refined_4gpu/20260811_122138`. It is
therefore a prompt-compatibility regression, not a valid measurement of how a
LoRA trained on that wording would perform. The fixed 512-case comparison is
still useful because it exposes where the instruction became too hard:

| Primary task | RGB-feedback F1 | Over-tight F1 | TP / FP / FN / TN change |
| --- | ---: | ---: | --- |
| HIGHWAY | 0.9385 | 0.9385 | unchanged: 61 / 5 / 3 / 59 |
| OBSTACLE | 0.7037 | 0.6408 | 38 / 6 / 26 / 58 -> 33 / 6 / 31 / 58 |
| VULNERABLE | 0.8673 | 0.8393 | 49 / 0 / 15 / 64 -> 47 / 1 / 17 / 63 |
| TRAFFIC_LIGHT_ABNORMAL | 0.5055 | 0.1972 | 23 / 4 / 41 / 60 -> 7 / 0 / 57 / 64 |

The signal rule collapsed to almost-always `NO`: it removed four false
positives, but discarded sixteen true positives. In case 142, the four RGB
frames visibly show simultaneous green heads on distinct arms of a broad
intersection, yet the model writes "no conflicting green" and answers `NO`.
It cannot reliably perform an exact lane-by-lane topology proof from this
stitched view. The corrected middle rule therefore accepts a clear pair of
green heads facing distinct approach arms in one broad conflict area, while
still rejecting red-versus-green normal phasing, same-arm/same-gantry heads,
and separate junctions.

The obstacle rule had the same failure mode. Case 97 shows a braking lead
growing closer through the fog history, and case 338 shows a parked orange
vehicle occupying an unmarked travel lane. Both affect ego immediately and
must remain `YES`. The corrected rule rejects a single brake-light frame or a
clearly marked parking bay, but accepts visible range closing, a stopped queue,
or a vehicle occupying the travel lane.

Next protocol: run base Qwen with the corrected middle prompt first; retain the
base audit run as the visual prompt-diagnosis tool, and treat the base
production run only as an answer-only lower bound. Retain the old adapter run
only as a compatibility note; then train a new adapter on this same production
prompt and unchanged index before treating LoRA F1 as a prompt-quality result.
No dataset rebuild is required for that prompt-aligned training. The separate
`visible_yes` / `visible_no` / `not_observable` audit remains required before
the next larger dataset/label iteration.

## Production Base Lower Bound (2026-08-11)

`sft_loop_phase1_eval/base_rgb_middle_boundary_4gpu/20260811_125440` runs the
corrected middle prompt in `prompt_mode=production`: four YES/NO lines only,
with no `EVIDENCE_*` request. All 512 responses parse correctly, so this is not
a formatting failure, but the untrained base is extremely conservative:

| Primary task | TP / FP / FN / TN | F1 |
| --- | --- | ---: |
| HIGHWAY | 6 / 0 / 58 / 64 | 0.1714 |
| OBSTACLE | 0 / 0 / 64 / 64 | 0.0000 |
| VULNERABLE | 41 / 0 / 23 / 64 | 0.7810 |
| TRAFFIC_LIGHT_ABNORMAL | 0 / 0 / 64 / 64 | 0.0000 |

The earlier base runs used the longer audit/evidence prompt and therefore must
not be compared numerically with this production run. The evidence scaffold
elicits visual inspection from the frozen base; the four-line production prompt
is the answer-only task that the LoRA must learn. Treat this result as the
untrained answer-only floor and parser contract. The decisive next result is a
new LoRA trained with this exact production prompt, followed by a separate
audit run of that same adapter for RGB error analysis.
