# SFT v2 Plan

SFT v2 is an independent LoRA route for serial choice supervision. It does not
replace `qwen3vl_local/sft/`.

The task is now explicitly two-stage:

1. Feed the RGB clip and scene-choice prompt, then ask the model to output only
   `SCENE`.
2. If the predicted scene is valid, append a new user prompt for that predicted
   scene's event sequence to the same conversation, reuse the scene-step KV, and
   ask the model to output only `STATUS` and `SUBGOAL`.

The second stage should reuse the first stage's visual/prompt KV cache instead
of re-encoding the RGB clip. If stage-1 predicts a scene outside
`SCENE_CHOICES`, eval stops that sample immediately. If stage-1 predicts a valid
but wrong scene, stage-2 still uses the predicted scene's event sequence, and
serial metrics count the downstream status/subgoal as wrong unless the scene is
also correct.

## Data

`build_dataset.py` reuses the old SFT keyframe timeline and keep/advance
sampling logic:

- Input is still 4 LEAD stitched RGB frames, ordered oldest to newest.
- `SCENE` supervision comes from the run scenario.
- `STATUS` supervision comes from the GT interval containing the anchor frame.
- `SUBGOAL` is the next event in `prompt_pipeline.get_full_sequence(scene)`.
- `PREVIOUS_STATUS_HINT` comes from `anchor - K` to preserve memory semantics.
- By default, `--samples-per-scenario 0` keeps all valid candidates. Positive
  values enable the old per-scenario downsampling path.
- Train rows use `--wrong-scene-ratio` augmentation by default. A subset of
  stage-2 prompts intentionally lists a wrong selected scene. The previous hint
  and supervised `STATUS/SUBGOAL` are phase-mapped into that selected scene's
  own `EVENT_SEQUENCE`, so the model sees the eval-time mismatch case without
  violating the second-stage choice constraints.
  Validation rows are not augmented.

Each jsonl row stores `stage_messages.scene` and `stage_messages.status`.
Only the scene stage carries image placeholders; status stage is appended as a
text-only follow-up turn.

## Prompt

`prompts.py` is the only prompt source:

- Stage 1 uses `SCENE_SYSTEM_PROMPT` and `build_scene_user_prompt(...)`.
- Stage 2 appends `build_status_user_prompt(selected_scene=...)` as a follow-up
  user turn. The status/subgoal output constraints are embedded in that user
  prompt because the active system turn remains the original scene/status
  classifier system prompt.
- Stage 1 lists all `SCENE_CHOICES`.
- Stage 2 lists only the selected scene's `EVENT_SEQUENCE` and event
  descriptions.

## Loss

`train.py` loads local `Qwen3-VL-4B-Instruct`, freezes the base model, injects
PEFT LoRA, and runs one multi-turn teacher-forced forward per sample:

- first assistant turn: supervise only the `SCENE` value token span.
- second assistant turn: supervise only the `STATUS` and `SUBGOAL` value token spans.
- prompt/image/system tokens have zero loss.
- format tokens such as `SCENE:` / `STATUS:` / newlines have zero loss.
- wrong-scene augmented rows still supervise the second assistant turn, but the
  supervised values are legal events from the selected scene.
- there is no teacher, no ANALYSIS, no pending placeholder, and no offline
  teacher cache.

This makes status/subgoal training condition on the earlier scene turn in the
same context while still masking loss to value tokens only.

## Eval

`eval.py` free-generates with the same two-stage protocol:

- Generate stage-1 `SCENE`.
- If scene is invalid, stop the sample and count `invalid_scene`.
- If scene is valid, append stage-2 prompt from the predicted scene, not GT, and
  continue decoding from the scene-step KV cache.
- Generate `STATUS/SUBGOAL`.
- The previous-status hint is phase-mapped into the predicted scene before
  stage 2, so the follow-up prompt always remains internally consistent.

Main metrics:

- `scene_accuracy`
- `status_accuracy`
- `subgoal_accuracy`
- `all_accuracy`
- `status_raw_accuracy` / `subgoal_raw_accuracy` for diagnostics only
- `invalid_scene_rate`
- `invalid_status_for_pred_scene_rate`
- `subgoal_not_next_rate`
- `status_kv_reuse_rate` / `status_kv_fallback_rate`
- `valid_total` plus `*_valid_scene` metrics, where invalid-scene rows are
  excluded from the denominator.

`status_accuracy` and `subgoal_accuracy` are serial metrics: scene must also be
correct.
