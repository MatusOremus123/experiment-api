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

## ⚠️ Setup: this API imports the pipeline from the srh project repo

The feature-extraction code lives in the separate
[`srh-ss26-iot-project`](https://github.com/MonsterDeveloper/srh-ss26-iot-project)
repo — **this API does not copy it**. On startup, `pipeline.py` adds that repo to
`sys.path` and imports the three functions from it:

- `extract_step_features(motion_csv_path) -> dict`
- `extract_audio_features(audio_wav_path) -> dict`
- `extract_video_features(video_h264_path) -> dict`

Importing in place (rather than copying) also lets `extract_video_features` find
its `models/face_landmarker.task`, whose path is relative to that file.

**Point the API at your clone** with the `SRH_PROJECT_PATH` env var. It defaults
to a sibling folder named `srh-ss26-iot-project1`:

```
GitHub/experiment-api          <- this API
GitHub/srh-ss26-iot-project1   <- the pipeline + model + sample data
```

The srh repo's dependencies (librosa, mediapipe, opencv-python, soundfile,
numpy, pandas, scipy) must be installed in the environment running this API —
they're included in `requirements.txt`.

> **Schema-mapping note (pending team decision):** our pipeline produces *gait /
> voice / mouth* features (`step_count`, `cadence_time_domain`, `step_regularity`,
> `mean_rotation`, `mean_loudness`, `mean_mouth_opening`, …). The professor's
> `ExerciseData` schema names `mouthOpening` / `soundPressure` / `footSpeed` /
> `stepLengths`. These do **not** map 1:1. As instructed, we store and serve our
> **actual** feature dict as-is under `features` (namespaced per stream) and do
> **not** invent fields we don't produce. The exact field-name mapping to the
> reference schema is left for the team to decide (see the comment in
> `database.py::exercise_data_to_dict`).

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

`test_flow.py` picks a **real** sample walk from the srh repo's
`collected_sample_data/` and drives the whole create → start → stop(+upload) →
data → export flow in-process:

```bash
py test_flow.py     # or: python test_flow.py
```

Any stream whose dependency isn't installed on the machine is reported under
`errors` instead of failing the run; the test asserts on the streams that
produced features (motion always must).

---

## Endpoints

Paths and names follow the professor's `openapi.yaml` (the source of truth).

| Method & path | Purpose | In yaml |
|---|---|:--:|
| `POST /experiments` | Create experiment (patient metadata) | ✅ |
| `GET /experiments` | List experiments (paginated) | ✅ |
| `GET /experiments/{id}` | Get one experiment | ✅ |
| `PATCH /experiments/{id}` | Partial update of an experiment | ✅ |
| `DELETE /experiments/{id}` | Delete experiment (cascades to exercises + data) | ✅ |
| `POST /experiments/{id}/exercises` | Create exercise (a trial) | ✅ |
| `GET /experiments/{id}/exercises` | List exercises of an experiment | ✅ |
| `GET /exercises` | List all exercises (paginated) | ✅ |
| `GET /exercises/{id}` | Get one exercise | ✅ |
| `DELETE /exercises/{id}` | Delete an exercise completely | ✅ |
| `POST /exercises/{id}/recording/start` | Mark recording started | ✅ |
| `POST /exercises/{id}/recording/stop` | **Upload the 3 files → run pipeline once → store `ExerciseData`** | ✅ * |
| `GET /exercises/{id}/data` | Return stored `ExerciseData` (never reprocesses) | ✅ |
| `DELETE /exercises/{id}/data` | Clear stored data (keeps the exercise) | ✅ |
| `GET /experiments/{id}/export` | CSV: one row per exercise, all features | ➕ |

`*` the yaml's `stop` is a bare action; we extend it with the 3-file multipart
upload the Pi sends (documented in our auto-generated `/docs`). `➕` export isn't
in the yaml — it comes from the Session 5 architecture (dashboard "Download all").

**`condition` + `repetition`** (3 conditions × 3 reps = 9 exercises per experiment)
are not first-class in the yaml, so they live in the exercise's `properties`
(`{"condition": "normal", "repetition": "1"}`). The export reads them from there.

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

# Update (partial)
curl -s -X PATCH $BASE/experiments/<EXP_ID> \
  -H 'Content-Type: application/json' \
  -d '{"age":68,"properties":{"visit":"2"}}'

# Delete
curl -s -X DELETE $BASE/experiments/<EXP_ID> -w '%{http_code}\n'

# --- Exercises ---
# Create (under an experiment) — condition + repetition go in properties
curl -s -X POST $BASE/experiments/<EXP_ID>/exercises \
  -H 'Content-Type: application/json' \
  -d '{"properties":{"condition":"normal","repetition":"1"}}'
# -> {"id":"<EX_ID>", ...}

# List for an experiment / list all (paginated) / get one
curl -s $BASE/experiments/<EXP_ID>/exercises
curl -s "$BASE/exercises?page=1&pageSize=20"
curl -s $BASE/exercises/<EX_ID>

# Delete an exercise completely
curl -s -X DELETE $BASE/exercises/<EX_ID> -w '%{http_code}\n'

# --- Recording ---
# Start
curl -s -X POST $BASE/exercises/<EX_ID>/recording/start

# Stop  ← THE KEY ONE: multipart upload of the three streams
curl -s -X POST $BASE/exercises/<EX_ID>/recording/stop \
  -F "motion=@motion_1720000000.csv;type=text/csv" \
  -F "audio=@audio_1720000000.wav;type=audio/wav" \
  -F "video=@video_1720000000.h264;type=application/octet-stream"

# --- Data ---
# Stored processed ExerciseData (JSON)
curl -s $BASE/exercises/<EX_ID>/data

# Clear stored data (keeps the exercise)
curl -s -X DELETE $BASE/exercises/<EX_ID>/data -w '%{http_code}\n'

# CSV export — whole experiment, one row per exercise
curl -s $BASE/experiments/<EXP_ID>/export -o experiment.csv
```

---

## How processing works (the important requirement)

Processing happens **exactly once**, on `POST /recording/stop`:

1. The three uploaded files are saved to `data/<exerciseId>/`.
2. `pipeline.process_recording` runs the extractors (video decoded once for both
   scalars and the mouth-opening series) and **assembles the `ExerciseData`
   payload**.
3. The assembled payload is written to `exercise_data.payload` as JSON.

`GET /data` and `GET /export` only ever **read** the stored JSON — they never
re-run the pipeline. This satisfies the professor's requirement that
"processed data is stored in a database so it isn't always processed on demand."

Extraction is **fault-tolerant**: if one stream fails (bad file, or a missing dep
like mediapipe on this machine), whatever succeeded is still stored and a
per-stream note is recorded under `errors`.

### `GET /data` payload (hybrid `ExerciseData`)

We return the professor's `openapi` `ExerciseData` shape, filling what our
pipeline can produce today and flagging the rest under `_notes`:

| Field | Status |
|---|---|
| `mouthOpening.values` | ✅ real per-frame `[vertical, null]` series (horizontal not produced yet) |
| `aggregates` | ✅ `stepLength` & `footSpeed`(walking speed) from the fixed **14 m** route; scalar averages/medians |
| `soundPressure.values` | ⏳ empty — our loudness is dBFS (relative), not calibrated SPL (needs the Pi's `spl.csv`) |
| `footSpeed.values` | ⏳ empty — per-sample foot speed not derived by the current motion pipeline |
| `features` | ✅ our raw 14 scalar features, verbatim (namespaced `motion`/`audio`/`video`) |

The exact final mapping to the reference schema is a **pending team decision**;
`_notes` in every payload states what's provisional and why.

## Project layout

```
server.py          FastAPI app: endpoints, CORS, /docs, upload+process flow
database.py        SQLite schema + row<->dict helpers (camelCase for the frontend)
pipeline.py        bridge to the srh extractors + assembles the ExerciseData payload
test_flow.py       in-process end-to-end smoke test (uses real srh sample data)
requirements.txt   deps (web stack + srh pipeline deps)
```

The `extract_*_features` functions and the MediaPipe model are **not** in this
repo — they live in `srh-ss26-iot-project` (see setup section above).

---

## Known gaps / open items

Tracked here so the team can pick them up. None block the current create → record →
process → store → serve flow, which works end to end.

### 1. Raw signals the spec wants but the pipeline doesn't produce yet
The `openapi` `ExerciseData` asks for full per-sample signals; our Session-04
extractors currently output ~14 **scalar** features, so these are served empty/partial
with a note (see the `GET /data` payload table above):

- **`soundPressure` (calibrated SPL)** — our audio loudness is **dBFS (relative)**, not
  Pascal/dB. Calibrated SPL is meant to come from the Pi's pre-computed `spl.csv`.
  ⚠️ The Session 5 diagram has the Pi writing **5 files** (`accel.csv · gyro.csv ·
  spl.csv · audio.wav · mouth.csv`), but the actual sample data / current upload is
  **3 files** (`motion.csv · audio.wav · video.h264`). Reconcile the Pi's file contract.
- **`footSpeed` (per-sample, cm/s)** — not derived by the motion pipeline. Only the
  *aggregate* foot speed (= walking speed, 14 m ÷ duration) is available today.
- **Horizontal mouth opening** — the video extractor produces vertical opening only, so
  `mouthOpening.values` tuples are `[vertical, null]`.

### 2. Integration seams not wired
- **Pi → API upload**: the Raspberry Pi's Flask recorder does not yet POST to
  `/recording/stop`; the flow has only been driven with recorded sample files.
- **API → dashboard**: the Streamlit dashboard still reads local files, not
  `GET /exercises/{id}/data` / `GET /experiments/{id}/export`.

### 3. Pending team decisions
- **Final `ExerciseData` field mapping** — the hybrid is provisional (see payload table).
- **Units** (open in the architecture diagram): sound pressure in Pa vs dB vs RMS;
  mouth opening as fraction-of-frame vs mm.
- **`condition` / `repetition`** currently live in exercise `properties`; decide whether
  they should become first-class fields (the yaml keeps exercises generic).

### 4. Environment note
The pipeline deps (librosa, mediapipe, opencv, …) must be installed in whatever runs
`uvicorn`. Since this is a separate repo from `srh-ss26-iot-project` (which uses `uv`),
they're duplicated in `requirements.txt`. If a machine lacks one, that stream degrades
to a per-stream `error` rather than crashing the server.
