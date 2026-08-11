# Static Obstacle Prompt Audit

## Scope

This audit defines the replacement for the retired mixed `OBSTACLE` question in
the first no-memory loop. The new output is:

```text
STATIC_OBSTACLE: YES|NO
```

It is supervised only by `U-E2`. Dynamic `U-E1/U-E3/U-E5/U-E6/U-E8` events are
intentionally excluded and will be queried in a later dynamic-obstacle loop.

The rule below was derived from the full-route RGB sheets and the corresponding
manual notes, not from scenario names. Reviewed U-E2 examples cover accident,
accident-two-ways, construction, construction-two-ways, parked-obstacle,
parked-obstacle-two-ways, and vehicle-opens-door-two-ways routes across towns.

## What Is Visible In RGB

The positive visual pattern is a road-fixed object which occupies the usable
ego corridor. In the reviewed sheets, that appears as one of these forms:

- A crashed, disabled, or parked car is diagonal to the lane or straddles its
  usable width. The lane line, curb, and car body show that ego must move around
  it rather than simply follow it.
- A construction board, cones, barrier line, or work vehicle closes the lane.
  The decisive cue is the closed/narrowed passage, not the orange color alone.
- A road-side car has an open door protruding into the travel lane. The door is
  a fixed geometric intrusion even if the vehicle otherwise resembles parking.
- On a narrow two-way road, a stationary car remains road-fixed while ego gets
  closer; the open space is not ego-width unless ego borrows around it.

The reviewed negative boundary is equally important. A normal lead vehicle,
queue at a signal, ambulance crossing a junction, cut-in car, oncoming
intruder, or red-light-running car can demand braking, but it is moving and is
therefore not a static obstacle in this loop. A car wholly in a marked bay or
shoulder, a background parked car with a clear lane, and a residual crash car
after ego has passed are also `NO`.

## Short-History Rule

The prompt asks the model to compare all four RGB frames against lane markings,
curb, road edge, and background. A static object stays fixed in road coordinates
while ego closes distance or drives around it; a moving vehicle changes its
road-relative position. It explicitly tells the model to answer `NO` when this
distinction is not visible, rather than guessing static from brake lights,
darkness, fog, a short gap, or a scenario prior.

This is necessary because some U-E2 route-state labels are only weakly visible
in a particular four-frame history. Examples include fog/rain/night sheets and
brief U-E2-to-recovery overlaps. Those samples remain part of the current U-E2
contract, but they must be tagged `not_observable` before any final RGB-only
benchmark is interpreted. Do not manufacture a positive visual cue from the
scenario name to make such rows easier.

## Implementation Contract

- `prompts.py` emits `STATIC_OBSTACLE`, never the legacy `OBSTACLE` key.
- `build_dataset.py` accepts the frozen old answer table as an audit source, but
  derives the new label solely as `primary_event == U-E2`.
- The dataset name, adapter route, prompt name, target order, and eval parser
  are changed together. Existing mixed-obstacle frame indices and adapters are
  rejected instead of silently evaluated against new answers.
- Rebuild `checkpoints/sft_loop_phase1_data/` before base or LoRA evaluation.
  The evaluation's `STATIC_OBSTACLE` focus module remains YES/NO 1:1.

## Evidence Consulted

- `ROAD_EVENT_CLASSIFICATION_PLAN.md` defines U-E2 as static ego-path blocking
  and separates it from U-E3 dynamic cut-in and U-E5/U-E6 dynamic conflicts.
- Full-route RGB notes confirm nighttime construction cones/barriers, parked
  obstruction cores, narrow-road static cars, and open-door intrusion. They
  also document normal background parking and post-obstacle recovery as false.
- The prior mixed-obstacle error audit identified that broad path-conflict
  wording created false positives for ordinary following and curbside parking.
  That wording is retired rather than merely tightened.
