# noScenarios RS Research

- rule_kind: `noscenario`
- road_candidates: `R1, R4`
- sampled towns: `Town03, Town04, Town05, Town06, Town07, Town10HD, Town15`
- auto_input_complete: `True`
- map_rgb_alignment_status: `not_checked`
- manual_final_complete: `False`

See `rules/scenario_rule_design.md` for the current scenario-specific logic.
See `maps/`, `rgb/`, `meta/`, `xml/`, and `xodr/` for the evidence chain.
Before changing runtime thresholds, check `maps/*route_trigger_ego_trace.png` and `rgb/*sample_contact_sheet.jpg`, then fill threshold provenance in `rules/thresholds.json`.
