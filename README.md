# ContextMatch

Qwen3-14B-AWQ pipeline for ranking the supplied `top_700.jsonl` candidates
against the Redrob Senior AI Engineer job description.

The model scores fixed dimensions; Python calculates totals and applies
disqualifier caps. Forty diverse candidates are manually reviewed before the
final run. Eight become prompt anchors and 32 remain a calibration holdout.

## 1. Server setup

On the Linux GPU server:

```bash
nvidia-smi
python3 --version
free -h
df -h .

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -r requirements-server.txt
```

Install the CUDA-compatible vLLM wheel recommended by the current vLLM
installation documentation if the generic installation does not match the
server driver.

Start the local server:

```bash
bash scripts/start_vllm.sh
```

The default `GPU_MEMORY_UTILIZATION=0.40` reserves about 16 GB on a 40 GB GPU.
If model loading fails, inspect actual free VRAM and raise it gradually, for
example:

```bash
GPU_MEMORY_UTILIZATION=0.45 bash scripts/start_vllm.sh
```

The server uses an 8K context, prefix caching, structured JSON, and a reasoning
parser. Bulk requests explicitly disable thinking; only disqualifier
adjudication enables it.

## 2. Validate and create the calibration set

Place the full shortlisted records at `top_700.jsonl`. The current file
contains 641 valid, unique records. Commands accept any valid count by default;
pass `--expected-count 641` when an exact count check is desired.

```bash
contextmatch validate-input --input top_700.jsonl --expected-count 641

contextmatch score \
  --input top_700.jsonl \
  --output runs/calibration/initial_scores.jsonl \
  --report runs/calibration/initial_timing.json

contextmatch select-calibration \
  --input top_700.jsonl \
  --scores runs/calibration/initial_scores.jsonl \
  --size 40 \
  --output calibration/reviews.json
```

Open `calibration/reviews.json`. For every record:

- Set `review_status` to `approved` if the draft is correct.
- Otherwise set it to `corrected`, copy `draft_assessment` into
  `corrected_assessment`, and edit the incorrect dimensions, evidence,
  concerns, flags, or confidence.
- Add useful `reviewer_notes`.
- Do not leave any record as `pending`.

Build the eight prompt anchors and 32-record holdout:

```bash
contextmatch build-calibration \
  --reviews calibration/reviews.json \
  --anchors-output calibration/anchors.json \
  --holdout-output calibration/holdout.json
```

Evaluate prompt calibration:

```bash
contextmatch evaluate-calibration \
  --anchors calibration/anchors.json \
  --holdout calibration/holdout.json \
  --output calibration/report.json
```

Calibration passes when at least 80% of holdout totals are within 10 points,
mean absolute error is at most 8, and no reviewed disqualifier is missed.
Revise rubric wording or reviewed anchors before a final run if it fails.

## 3. Rank and export

```bash
contextmatch run \
  --input top_700.jsonl \
  --anchors calibration/anchors.json \
  --output team_xxx.csv \
  --artifacts-dir runs/final
```

The run:

1. Scores all candidates once.
2. Repeats ranks 70–180, confidence below 0.75, and conflicting cases.
3. Uses thinking-mode adjudication when disqualifier flags disagree.
4. Comparatively reranks the leading 150 in three randomized rounds of groups
   of ten.
5. Combines 80% rubric score with 20% comparative percentile.
6. Generates factual, unique, 8–49 word reasoning for the final 100.
7. Writes intermediate JSONL artifacts and the final CSV.

Validate independently:

```bash
contextmatch validate-output \
  --input top_700.jsonl \
  --submission team_xxx.csv

python India_runs_data_and_ai_challenge/validate_submission.py team_xxx.csv
```

## Performance tuning

The bulk target is below one second average per candidate, excluding model
loading, calibration, repeats, comparative reranking, and final reasoning.
Start with concurrency 16, then benchmark 8, 16, and 32:

```bash
contextmatch score \
  --input top_700.jsonl \
  --anchors calibration/anchors.json \
  --concurrency 16 \
  --output runs/benchmark.jsonl \
  --report runs/benchmark.json
```

Do not increase concurrency if it causes out-of-memory errors, long queueing,
or lower throughput.
