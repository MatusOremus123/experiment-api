# Experiment API — Parkinson's Gait (SRH Berlin IoT/Data Science)

A FastAPI server that ingests IoT walking-trial data from a Raspberry Pi, runs
the feature-extraction pipeline **once on upload**, stores the results in SQLite,
and serves them to the frontend team.


```
experiment (a patient)  ->  exercise (a walking trial)  ->  exercise_data (processed features)
```

Each ~14 m walk produces three synchronized streams, uploaded together when the
recording stops:

| stream | file                     | format                                  |
|--------|--------------------------|-----------------------------------------|
| motion | `motion_[ts].csv`        | accel + gyro, 50 Hz (`time,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z`) |
| audio  | `audio_[ts].wav`         | 48 kHz mono, 32-bit                     |
| video  | `video_[ts].h264`        | 1280×720 raw H.264                      |

---

## ⚠️ Before you run: drop in the real feature-extraction functions

The brief said `extract_step_features` / `extract_audio_features` /
`extract_video_features` already live in the repo root — but this repo was empty
when the server was scaffolded. So the three modules below currently contain
**lightweight PLACEHOLDER stand-ins** that let the whole API + DB + pipeline flow
run and be tested end to end without heavy dependencies:

- `step_features.py`  → `extract_step_features(motion_csv_path) -> dict`
- `audio_features.py` → `extract_audio_features(audio_wav_path) -> dict`
- `video_features.py` → `extract_video_features(video_h264_path) -> dict`

**To integrate the real pipeline:** replace the body of each of those three files
with your real implementation, keeping the **function name** and the
**`(path) -> dict` signature** identical. `server.py` imports them by those names,
so nothing else changes.

> **Schema-mapping note (pending team decision):** our pipeline produces *gait*
> features (`cadence`, `step_regularity`, `mean_rotation`, mouth-opening, sound
> pressure, …). The professor's `ExerciseData` schema names
> `mouthOpening` / `soundPressure` / `footSpeed` / `stepLengths`. These do **not**
> map 1:1. As instructed, we store and serve our **actual** feature dict as-is
> under `features` and do **not** invent fields we don't produce. The exact
> field-name mapping to the reference schema is left for the team to decide (see
> the comment in `database.py::exercise_data_to_dict`).

---

## Run it

```bash
# 1. install deps (a venv is recommended)
pip install -r requirements.txt

# 2. start the server
uvicorn server:app --reload
```

- API base:            <http://localhost:8000>
- **Interactive docs:  <http://localhost:8000/docs>**  ← for the frontend team
- OpenAPI JSON:        <http://localhost:8000/openapi.json>

SQLite lives in `experiment.db` (override with `EXPERIMENT_DB`); uploaded files go
to `data/<exerciseId>/` (override with `EXPERIMENT_DATA_DIR`). CORS is open to all
origins so a browser frontend can call the API directly.

### Smoke test (no server needed)

`test_flow.py` generates tiny valid sample files and drives the whole
create → start → stop(+upload) → data → export flow in-process:

```bash
py test_flow.py     # or: python test_flow.py
```

---

## Endpoints

| Method & path | Purpose |
|---|---|
| `POST /experiments` | Create experiment (patient metadata) |
| `GET /experiments` | List experiments (paginated) |
| `GET /experiments/{id}` | Get one experiment |
| `DELETE /experiments/{id}` | Delete experiment (cascades to exercises + data) |
| `POST /experiments/{id}/exercises` | Create exercise (a trial) |
| `GET /experiments/{id}/exercises` | List exercises for an experiment |
| `GET /exercises/{id}` | Get one exercise |
| `POST /exercises/{id}/recording/start` | Mark recording started |
| `POST /exercises/{id}/recording/stop` | **Upload the 3 files → run pipeline once → store features** |
| `GET /exercises/{id}/data` | Return the stored processed features (never reprocesses) |
| `DELETE /exercises/{id}/data` | Clear stored data (keeps the exercise) |
| `GET /exercises/{id}/export` | CSV export of the stored features |

### Example curl for each

```bash
BASE=http://localhost:8000

# --- Experiments ---
# Create
curl -s -X POST $BASE/experiments \
  -H 'Content-Type: application/json' \
  -d '{"patientNumber":"P-001","age":67,"height":172,"weight":80,"properties":{"notes":"baseline"}}'
# -> {"id":"<EXP_ID>", ...}

# List
curl -s "$BASE/experiments?page=1&pageSize=20"

# Get one
curl -s $BASE/experiments/<EXP_ID>

# Delete
curl -s -X DELETE $BASE/experiments/<EXP_ID> -w '%{http_code}\n'

# --- Exercises ---
# Create (under an experiment)
curl -s -X POST $BASE/experiments/<EXP_ID>/exercises \
  -H 'Content-Type: application/json' \
  -d '{"properties":{"trial":"1"}}'
# -> {"id":"<EX_ID>", ...}

# List for an experiment
curl -s $BASE/experiments/<EXP_ID>/exercises

# Get one exercise
curl -s $BASE/exercises/<EX_ID>

# --- Recording ---
# Start
curl -s -X POST $BASE/exercises/<EX_ID>/recording/start

# Stop  ← THE KEY ONE: multipart upload of the three streams
curl -s -X POST $BASE/exercises/<EX_ID>/recording/stop \
  -F "motion=@motion_1720000000.csv;type=text/csv" \
  -F "audio=@audio_1720000000.wav;type=audio/wav" \
  -F "video=@video_1720000000.h264;type=application/octet-stream"

# --- Data ---
# Stored processed features (JSON)
curl -s $BASE/exercises/<EX_ID>/data

# Clear stored data
curl -s -X DELETE $BASE/exercises/<EX_ID>/data -w '%{http_code}\n'

# CSV export
curl -s $BASE/exercises/<EX_ID>/export -o exercise.csv
```

---

## How processing works (the important requirement)

Processing happens **exactly once**, on `POST /recording/stop`:

1. The three uploaded files are saved to `data/<exerciseId>/`.
2. `extract_step_features`, `extract_audio_features`, and `extract_video_features`
   are each called on their file. Each result is stored under its own namespace
   (`features.motion`, `features.audio`, `features.video`) so keys never collide.
3. The merged dict is written to the `exercise_data.features` column as JSON.

Extraction is **fault-tolerant**: if one stream fails, whatever succeeded is still
stored, and a per-stream note is recorded under `errors` (e.g.
`{"errors": {"video": "RuntimeError: ..."}}`).

`GET /data` and `GET /export` only ever **read** the stored JSON — they never
re-run the pipeline. This satisfies the professor's requirement that
"processed data is stored in a database so it isn't always processed on demand."

## Project layout

```
server.py          FastAPI app: endpoints, CORS, /docs, upload+process flow
database.py        SQLite schema + row<->dict helpers (camelCase for the frontend)
step_features.py   extract_step_features   (PLACEHOLDER — replace with real)
audio_features.py  extract_audio_features  (PLACEHOLDER — replace with real)
video_features.py  extract_video_features  (PLACEHOLDER — replace with real)
test_flow.py       in-process end-to-end smoke test
requirements.txt   deps (web stack + pipeline deps)
```
