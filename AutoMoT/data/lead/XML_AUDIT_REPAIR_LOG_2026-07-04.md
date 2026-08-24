# LEAD XML Audit / Repair Log - 2026-07-04

Target root: `AutoMoT/data/lead`  
Source root: `AutoMoT/data/data_routes`

## Summary

- Target XML checked: 9294
- Source routes indexed: 9893
- Normalized source matches: 9293
- Repaired XML: 0
- Deleted XML: 0
- Kept with source-traceability note: 1

## Method

- Parsed every target XML under `AutoMoT/data/lead`.
- Indexed source routes by scenario, town, and source file key; used global `(town, route_key)` fallback for known cross-scenario source layout.
- Compared normalized route content. The known source typo `weathis_juncer` is normalized to `weather` before comparison.
- Deleted nothing because no target XML was both abnormal and unrecoverable.

## Kept With Note

`ParkedObstacle/Town12_route_Town12_route15.xml` is kept. The exact source key `Town12_route15` is absent from `data_routes`, but the target XML parses cleanly and matches multiple equivalent Town12/ParkedObstacle route-id-18 source candidates:

- `AutoMoT/data/data_routes/50x38_Town12/ParkedObstacle/1006_0.xml#route[0]`
- `AutoMoT/data/data_routes/50x38_Town12/ParkedObstacle/1006_1.xml#route[0]`
- `AutoMoT/data/data_routes/50x38_Town12/ParkedObstacle/Town12.xml#route[0]`
- `AutoMoT/data/data_routes/50x38_Town12/ParkedObstacle/Town12.xml#route[15]`

This is a source-traceability ambiguity, not an abnormal target XML.
