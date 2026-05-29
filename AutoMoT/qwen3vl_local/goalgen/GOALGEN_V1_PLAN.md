# GoalGen v1 Plan

## Goal

GoalGen v1 trains a latent-space DiT-MoT generator conditioned on frozen
Qwen3-VL-Instruct KV cache.

The route is:

```text
history RGB -> frozen Qwen prefill -> segmented token-level KV
history RGB -> frozen VAE -> z_history
subgoal keyframe RGB -> frozen VAE -> z1
z0 ~ N(0, I), z_t = (1 - t) z0 + t z1
DiT-MoT(z_t, z_history, t, segmented KV) -> v_pred
loss = MSE(v_pred, z1 - z0)
```

Qwen and VAE are frozen. Only DiT-MoT trains.

## Data

Dataset builder:

```text
qwen3vl_local/goalgen/build_dataset_v1.py
```

It mirrors `tools/build_sft_dataset_v1.py`:

1. Read `/datashare/IOL4SGH/data/data/keyframes_all_scenarios.json`.
2. Keep only `Completed` / `Perfect` runs.
3. Expand keyframes into per-run status timelines.
4. For each valid `anchor`, set:
   - `status` = GT status at `anchor`
   - `subgoal` = next event after `status`
   - `target_frame` = keyframe where `subgoal` starts
5. Keep only samples with `target_frame > anchor`.
6. Stratify per scenario and `status->subgoal`.

Default size is up to `1000` samples per scenario. Use
`--samples-per-scenario 0` for all valid anchors.

Schema:

```json
{
  "scenario": "Accident",
  "run_id": "...",
  "anchor": 12,
  "status": "initial",
  "subgoal": "hazard_detect",
  "target_event": "hazard_detect",
  "target_frame": 37,
  "history_frames": [9, 10, 11, 12],
  "history_rgb_paths": ["..."],
  "current_rgb_path": ".../rgb/0012.jpg",
  "target_rgb_path": ".../rgb/0037.jpg",
  "memory": {
    "scenario": "Accident",
    "event_sequence": ["initial", "...", "final"],
    "status": "initial",
    "subgoal": "hazard_detect",
    "completed_events": ["initial"]
  }
}
```

## Model Defaults

- LEAD RGB: `1152x384`
- VAE latent: `[B, 4, 48, 144]`
- DiT patch size: `2`
- DiT latent token grid: `(24, 72) = 1728 tokens`
- DiT vision tokens with builder default 4 history frames:
  `z_t` tokens + `4 * z_history` tokens = `8640`
- DiT hidden dim: `768`
- DiT heads: `12`
- DiT layers: `12`
- Qwen segmented KV: default `select_last`, Qwen 36 layers -> 12 segments,
  each segment is the last layer of its 3-layer group (token-level K/V,
  shape `[B, 8, S, 128]`). `concat_layers` is kept as an ablation option.
- Qwen KV input dim: `8 * 128 = 1024` (same under `select_last` and `concat_layers`)

If any default shape changes, update this file and the runbook together.

## Training

Training entry:

```text
qwen3vl_local/goalgen/train_v1.py
qwen3vl_local/goalgen/train_v1.sh
```

Launcher modes:

- `check`: one GPU, two optimizer steps.
- `single`: one GPU training.
- `ddp`: multi-GPU DDP, auto GPU selection, auto free port.

Optimizer:

- AdamW on DiT parameters only.
- LR `1e-4`.
- Weight decay `0.01`.
- Cosine schedule.
- Warmup ratio `0.02`.

Checkpoints store only DiT state and optimizer/scheduler state for training
resume diagnostics.

## Parameter Budget (default config)

DiT-MoT only (Qwen and VAE are frozen and not counted):

| Module | Approx params |
|---|---|
| Patchify x2 (Conv2d 4 -> 768, kernel=2) | ~12K x 2 |
| Type embedding (2, 768) | 1.5K |
| Timestep MLP (cond_dim=256, 4x) | ~0.5M |
| Per JointAttention (q/k/v/o + lang_k/v) | ~3.9M |
| Per MLP (768 -> 3072 -> 768) | ~4.7M |
| Per AdaLN modulation (256 -> 4608) | ~1.2M |
| Per block total | ~9.8M |
| 12 blocks | ~118M |
| Unpatchify Linear(768 -> 16) | ~12K |
| **DiT total (rough)** | **~120M** |

bf16 weights are about 240MB. Qwen 4B and VAE add their own memory but no
gradients.

## Memory Budget (H20 96GB, batch=1, bf16 DiT)

| Stage | Estimated GPU mem |
|---|---|
| Qwen3-VL-4B-Instruct (bf16) | ~8GB |
| VAE (fp32) | ~0.4GB |
| Qwen prefill (~2300 tokens, 36 layers KV) | ~3-4GB |
| Segmented 12 KV memories, select_last (bf16) | ~1GB |
| DiT-MoT weights (bf16) | ~0.25GB |
| DiT forward activations (vision N=8640, language S~2300 per block under default `select_last`) | moderate; if you switch to `concat_layers` language tokens triple to ~6900 and memory grows accordingly |
| DiT backward + AdamW state | ~2GB |
| **Total per rank (training, batch=1)** | **~17-20GB** |

In DDP each rank loads its own Qwen + VAE, so 8 ranks cost about 8x the Qwen
memory across the cluster. This is wasteful but acceptable for v1; v2 should
offline-cache segmented KV and history/target latents instead.

## Risks and fallbacks

| Risk | Trigger | Fallback |
|---|---|---|
| KV segment count != DiT layers | `pooled_kv segments ... != DiT layers ...` raised inside DiT.forward | Keep `--num-layers` and the segmenter `num_segments` equal. Default 12 on both sides. |
| `language_kv_input_dim` hardcoded but base model changed | DiT first language Linear shape mismatch -> RuntimeError | `train_v1.py` now infers it from the first sample's segmented KV. Pass `--language-kv-input-dim auto` (default). |
| Full KV mode OOM | `concat_layers` triples language memory per DiT block | Stay on default `QWEN_KV_SEGMENT_MODE=select_last`; only switch to `concat_layers` for ablation; use `mean` only for old ablation. |
| dtype mix between bf16 Qwen KV and fp32 DiT | RuntimeError inside SDPA | Trainer casts segmented KV / z_history / z1 to `dit_dtype` before forward. Keep `--qwen-dtype` and `--dit-dtype` compatible or rely on the explicit `.to(dtype=dit_dtype)`. |
| Qwen prefill too long (LEAD num_frames > 4) | Qwen prefill OOM on H20 96GB | Drop `--num-frames`, or quantize Qwen with `--qwen-dtype float16`. |
| Target keyframe missing on disk | `RGB image not found: .../rgb/NNNN.jpg` | Re-run `build_dataset_v1.py` from the host that mounts LEAD data. |
| Wrong scenario string passed to `--status` / `--subgoal` | Runner raises because STATUS/SUBGOAL violates the event chain | Prefer using STATUS only; runner derives SUBGOAL. For training, source memory from `build_dataset_v1.py` jsonl. |
| Target frame is not future | Runner raises `target_frame must be in the future` | Use dataset-built samples or choose `--target-frame > --anchor`. |
| DDP rank hang | One rank does extra `loss.backward()` due to shard length mismatch | `usable_per_epoch = (N // world_size) * world_size` already keeps all shards equal. Do not change shard slicing logic. |
| Slow training dominated by Qwen prefill | Each sample triggers a full Qwen forward | v1 limit; v2 must precompute and cache segmented KV + history/target latents to disk. |

## Default shapes sync (read this before changing any geometry)

The default shapes table above is the contract used by both the dataset builder
and the trainer. If any of these values is changed:

1. Update this file's defaults section.
2. Update the matching numbers in `GOALGEN_V1_RUN.md`.
3. Re-run `build_dataset_v1.py` if `RGB_FRAME_COUNT` or `RGB_FRAME_STEP`
   changes (the jsonl encodes specific frame indices).
4. Re-train DiT from scratch if `patch_size`, `hidden_dim`, `n_heads`, or
   `num_layers` changes (old checkpoints become incompatible).

Do not let code drift away from these documents.

## Known Limits

- v1 computes Qwen KV and VAE latents on the fly. It is simple but slow.
- DiT now consumes all history latents from `history_rgb_paths` directly.
- No EMA / CFG / cached latent dataset yet.
- No decoded-image eval yet; use loss and velocity cosine as early sanity only.

## v1 / v2 Boundary

What v1 explicitly does NOT do (kept here so future agents do not expand scope):

- No precomputed segmented KV / latent cache.
- No EMA on DiT weights.
- No classifier-free guidance dropout on the language stream.
- No multi-target supervision (one subgoal keyframe per sample).
- No decoded-image evaluation; loss + velocity cosine only.

v2 will start once v1 forward + train smoke passes on the remote H20 cluster.
