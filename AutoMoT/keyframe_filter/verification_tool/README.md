# Keyframe Verification Viewer

Local HTML web app to verify key frames in `keyframes_4scenarios.json`.

## Setup (uv)

```bash
cd /home/cruser1/lda/lead/cache_ln/data/verification_tool
uv sync
```

## Run

```bash
cd /home/cruser1/lda/lead/cache_ln/data/verification_tool
uv run -- python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Optional env vars:

- `KEYFRAME_JSON_PATH` (default: `/home/cruser1/lda/lead/cache_ln/data/keyframes_4scenarios.json`)
- `DATASET_ROOT` (default: `/home/cruser1/lda/lead/cache_ln/data`)

Open:

- `http://127.0.0.1:8000`
