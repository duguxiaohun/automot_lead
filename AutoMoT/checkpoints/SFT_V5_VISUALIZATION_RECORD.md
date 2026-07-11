# SFT v5 Visualization Record

This file is a lightweight record under `AutoMoT/checkpoints/` for the SFT v5
teacher/student visualization workflow. It does not contain model weights or
generated probe cases; those should stay under run-specific output directories.

## Purpose

SFT v5 uses two questions per frame:

- Q1: road structure option and abnormal-event existence.
- Q2: event option under the Q1 road structure.

The visualization workflow dumps exactly what the student sees, what the
privileged teacher prompt contains, what the cleaned teacher target looks like,
the copied RGB history, labels, memory transitions, and route-level timelines.

## Static Prompt/Teacher Dump

Run from `AutoMoT/`:

```bash
python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --output-dir checkpoints/sft_v5_runs/latest/probe \
  --num-cases 24 \
  --with-teacher
```

This does not load Qwen. It writes teacher/student prompts, cleaned teacher
targets, labels, memory JSON, RGB copies, timeline JSON/PNG, and manifest JSON.

## Student Output Dump

Run from `AutoMoT/`:

```bash
GPU_IDS=0 python qwen3vl_local/sft_v5/probe.py \
  --index checkpoints/sft_v5_data/val_sequence_index.jsonl \
  --model-dir checkpoints/Qwen3-VL-4B-Instruct \
  --adapter-dir checkpoints/sft_v5_runs/latest/final \
  --output-dir checkpoints/sft_v5_runs/latest/probe_with_model \
  --num-cases 8 \
  --with-model \
  --with-teacher
```

`--with-model` loads the student adapter and fills `q1_student_output.txt` /
`q2_student_output.txt`. `--with-teacher` is a compatibility flag: v5 always
writes teacher privileged prompt and cleaned teacher target, but does not load a
second teacher model in probe.

## Output Layout

```text
probe*/
  manifest.json
  route_<idx>__<scenario>__<route_id>/
    timeline.json
    timeline.png
    frame_<frame_id>/
      rgb_00.jpg
      rgb_01.jpg
      rgb_02.jpg
      rgb_03.jpg
      rgb_paths.json
      q1_student_prompt.txt
      q1_student_output.txt
      q1_teacher_prompt.txt
      q1_teacher_target.txt
      q2_student_prompt.txt
      q2_student_output.txt
      q2_teacher_prompt.txt
      q2_teacher_target.txt
      step1_user.txt
      step1_student.txt
      step1_teacher_user.txt
      step1_teacher.txt
      step2_user.txt
      step2_student.txt
      step2_teacher_user.txt
      step2_teacher.txt
      labels.json
      memory_before.json
      memory_after.json
      flags.json
```

Timeline colors:

- Red: Q1 RS is wrong and the next frame resets to `GT RS + RE`.
- Blue: Q1 RS is correct and Q2 is entered.
- Green: static teacher-forced dump without student generation.
- Gray: ordinary frame without a highlighted transition.

## Review Checklist

- `q1_student_prompt.txt` and `q2_student_prompt.txt` must not contain
  `XML_WEATHER`, `ANSWER_`, `REFERENCE`, ground-truth labels, or scenario names.
- `q1_teacher_prompt.txt` may contain XML weather and GT fields; teacher target
  files must be cleaned back to the student perspective.
- `q2_student_prompt.txt` should show `RE` plus the allowed `U-E*` candidates;
  `RE` should include the current frame's `regular_event_codes` description.
- `memory_before.json` and `memory_after.json` should follow the v5 state
  machine: Q1 RS wrong stops Q2 and resets next frame; invalid Q2 does not
  pollute memory.
