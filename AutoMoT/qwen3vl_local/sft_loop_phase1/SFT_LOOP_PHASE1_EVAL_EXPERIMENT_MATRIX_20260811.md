# Phase1 Evaluation Experiment Matrix

## Scope And Rules

This record summarizes every completed Phase1 four-question evaluation found on
2026-08-11. All six runs use the same fixed 512 `case_index` values. Each
primary-task module contains 128 samples with that primary question balanced at
64 `YES` and 64 `NO`; every sample still answers all four questions.

`audit` means the user prompt requests four visible `EVIDENCE_*` observations
before the four answers. It does **not** include scenario, RS, EVENT, focus
task, answer-table values, or GT labels. It gives the autoregressive model an
external observation scratchpad, so its F1 must not be compared directly with
the answer-only `production` prompt. The legacy metrics predate explicit
`prompt_mode` metadata, but their saved prompt and `audit_prompt=true` show that
they are audit runs.

The `F1` columns below are the focused 1:1 main-question F1 values, not the
unbalanced side-question metrics.

## Completed Runs

| ID | Model | Prompt behavior | Mode | Exact | Highway F1 | Obstacle F1 | Vulnerable F1 | Light F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | Base Qwen | Original rules | audit | 0.5371 | 0.9008 | 0.4286 | 0.7379 | 0.1127 |
| B | Existing LoRA | Original rules | audit | 0.6992 | 0.9385 | 0.6863 | 0.8496 | 0.4598 |
| C | Base Qwen | RGB-feedback: close-target pass, path-overlap, signal witness | audit | 0.5469 | 0.9032 | 0.4421 | 0.7500 | 0.0308 |
| D | Existing LoRA | Same RGB-feedback rules | audit | 0.7148 | 0.9385 | 0.7037 | 0.8673 | 0.5055 |
| E | Existing LoRA | Over-tight boundary: exact signal topology proof and stricter ordinary-lead exclusion | audit | 0.6797 | 0.9385 | 0.6408 | 0.8393 | 0.1972 |
| F | Base Qwen | Corrected middle boundary | production | 0.5000 | 0.1714 | 0.0000 | 0.7810 | 0.0000 |
| G | Base Qwen | Corrected middle boundary | audit | 0.5527 | 0.8960 | 0.2892 | 0.7379 | 0.0000 |
| H | Existing LoRA | Corrected middle boundary | production | 0.7598 | 0.9333 | 0.5556 | 0.8496 | 0.6800 |

Run directories:

- A: `sft_loop_phase1_eval_old/base_zero_shot_prompt_4gpu/`
- B: `sft_loop_phase1_eval_old/lora_zero_shot_prompt_4gpu/20260810_211451/`
- C: `sft_loop_phase1_eval/base_rgb_feedback_4gpu/20260811_000022/`
- D: `sft_loop_phase1_eval/lora_zero_shot_prompt_4gpu/20260811_092904/`
- E: `sft_loop_phase1_eval/lora_rgb_boundary_refined_4gpu/20260811_122138/`
- F: `sft_loop_phase1_eval/base_rgb_middle_boundary_4gpu/20260811_125440/`
- G: `sft_loop_phase1_eval_new/base_rgb_middle_boundary_audit_4gpu/20260811_160335/`
- H: `sft_loop_phase1_eval_new/lora_legacy_middle_production_4gpu/20260811_211331/`

The old LoRA is the adapter under
`checkpoints/sft_loop_phase1_runs/latest/final`, trained before the current
middle-boundary prompt and before prompt-content SHA-256 was stored. It is
therefore a legacy compatibility model, not a prompt-aligned model for D or E.

## What The Matrix Shows

### 1. LoRA Helps Strongly Under The Original Audit Scaffold

B minus A is a fair same-prompt audit comparison. Exact match improves by
`+0.1621`; focused F1 changes are Highway `+0.0377`, Obstacle `+0.2577`,
Vulnerable `+0.1117`, and Light `+0.3471`. The adapter is not merely fixing
formatting: the gains are concentrated in positive recall for obstacle and
traffic-light anomaly.

### 2. RGB-Feedback Rules Help The Existing LoRA, But This Is Compatibility Evidence

D minus B changes exact match by `+0.0156`. Highway remains `0.9385`, while
Obstacle rises `0.6863 -> 0.7037`, Vulnerable rises `0.8496 -> 0.8673`, and
Light rises `0.4598 -> 0.5055`. This confirms that the close-target pass and
simple multi-arm green-light cue are useful to the old adapter. It does not
prove the final ceiling of those rules because the adapter did not train on
their exact wording.

For the frozen base, C minus A is only a small exact gain (`+0.0098`). It helps
highway, obstacle, and vulnerable slightly, but Light drops `0.1127 -> 0.0308`.
The frozen base needs the explicit evidence scaffold and still cannot reliably
apply the light rule.

### 3. The Strict-Boundary Version Must Be Rejected

E versus D keeps Highway unchanged but loses exact match `0.7148 -> 0.6797`.
The large failure is Light: `23 / 4 / 41 / 60` becomes `7 / 0 / 57 / 64`
(TP / FP / FN / TN). It makes the model nearly always answer `NO`, including
clear broad-junction frames with green heads on distinct approach arms. Obstacle
also loses five true positives. The corrected middle prompt retains simple
distinct-arm visual evidence while excluding red-versus-green normal phasing,
same-arm heads, and marked parking bays.

### 4. Production Base Is An Answer-Only Floor, Not A Prompt Winner

F uses the current production prompt: four YES/NO lines only. It has no parse
failures, but the frozen base answers no obstacle and no traffic-light anomaly
positive samples. Its focused confusion counts are:

| Question | TP / FP / FN / TN |
| --- | --- |
| HIGHWAY | 6 / 0 / 58 / 64 |
| OBSTACLE | 0 / 0 / 64 / 64 |
| VULNERABLE | 41 / 0 / 23 / 64 |
| TRAFFIC_LIGHT_ABNORMAL | 0 / 0 / 64 / 64 |

This is not answer leakage in audit mode and not a parser issue. The audit
prompt makes the model write its own RGB observations before the answer, which
gives it more autoregressive test-time computation. Production intentionally
does not request that scratchpad. Therefore F cannot be compared numerically
with A/C, and it must not decide whether the visual rules are good.

### 5. The Legacy LoRA Proves That Answer-Only SFT Works

H fills the first missing production cell: the legacy adapter receives the same
current middle-boundary production prompt as F and is evaluated on the same 512
fixed cases. Its adapter lacks a prompt-content SHA-256 because it predates that
metadata, so this remains a compatibility result rather than a prompt-aligned
final score. It is nevertheless decisive evidence about the answer-only route:

| Primary task | Base production F1 (F) | Legacy LoRA production F1 (H) | Main TP / FP / FN / TN in H |
| --- | ---: | ---: | --- |
| HIGHWAY | 0.1714 | 0.9333 | 56 / 0 / 8 / 64 |
| OBSTACLE | 0.0000 | 0.5556 | 25 / 1 / 39 / 63 |
| VULNERABLE | 0.7810 | 0.8496 | 48 / 1 / 16 / 63 |
| TRAFFIC_LIGHT_ABNORMAL | 0.0000 | 0.6800 | 34 / 2 / 30 / 62 |

Exact match rises `0.5000 -> 0.7598`. Relative to the base under the same
production prompt, the adapter restores 65 highway positives, 29 obstacle
positives, and 34 signal-light positives that the frozen model answered `NO`.
Thus four-line answer-only SFT can learn the required positive decisions; an
audit scratchpad is useful for diagnosis, but is not required in deployment.

G is intentionally not used to select the final production prompt. Frozen base
audit F1 is respectable for highway and vulnerable road users, but its obstacle
and light recall are lower than the earlier RGB-feedback audit runs. H shows
that the LoRA, not the frozen base, is the relevant decision model for the
answer-only loop. Keep the corrected middle boundary now; do not make another
prompt edit before measuring a prompt-aligned adapter.

## Evidence Supervision Status

Current training calls `build_phase1_prompt(audit=False)` and supervises only
the four YES/NO value tokens. `EVIDENCE_*` is not present in the target and has
no loss. Consequently:

- audit runs are for prompt/RGB diagnosis only;
- production runs are the answer-only deployment contract;
- do not train with `--audit-prompt` until there is an explicit, reviewed
  evidence target for each sample.

## Missing Experiments And Decision Rule

Cell 1, existing legacy LoRA + current middle-boundary `production` prompt, is
now complete as H. The only missing critical experiment is a newly trained,
prompt-aligned LoRA + that same `production` prompt. Freeze the current middle
boundary and train a new adapter with the unchanged `frame_index.jsonl`; no
dataset rebuild is needed for that text-only change. This new production run is
the first valid formal result for the answer-only loop. Run the same new adapter
again with `--audit-prompt` only to inspect RGB failures; do not compare that
audit F1 to its production F1.

Before a larger data iteration, add per-question frame-level
`visible_yes` / `visible_no` / `not_observable` status. The answer table is
scenario + RS + EVENT scoped and still cannot distinguish a currently visible
event from an event that is dark, occluded, or absent in these four RGB frames.
