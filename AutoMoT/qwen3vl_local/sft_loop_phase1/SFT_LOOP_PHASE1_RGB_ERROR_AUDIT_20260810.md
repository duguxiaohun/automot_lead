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
| VULNERABLE | A person is visibly present near the right-side sidewalk, while the frame annotation says `NO`. | 458 | High-confidence label error for this frame. |
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
  348 and vulnerable case 458.

The next evaluation should compare the current prompt with this change on the
same fixed test index and separately report metrics after excluding
`not_observable` labels. Otherwise the light, obstacle, and vulnerable F1s
continue to mix perception errors with unanswerable frames.
