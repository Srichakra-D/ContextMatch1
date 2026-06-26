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

## 2. Validate and scan integrity

Place the full shortlisted records at `top_700.jsonl`. The current file
contains 641 valid, unique records. Commands accept any valid count by default;
pass `--expected-count 641` when an exact count check is desired.

```bash
contextmatch validate-input --input top_700.jsonl --expected-count 641
```

Run the deterministic knowledge-base integrity scan before starting Qwen:

```bash
contextmatch scan-integrity \
  --input top_700.jsonl \
  --expected-count 641 \
  --knowledge-base knowledge_base.json \
  --output-dir runs/integrity
```

This creates:

```text
runs/integrity/integrity_report.jsonl
runs/integrity/verified_failures.jsonl
runs/integrity/suspicious_candidates.jsonl
runs/integrity/summary.json
```

Only `verified_failure` receives score zero. Expert skills with zero usage are
treated as suspicious metadata and receive a small score penalty instead of
hard exclusion. Other suspicious findings, if any, are retained for audit but
hidden from Qwen. Technology release-date mismatches are ignored entirely,
including both skill-duration metadata and dated career-text mentions, because
those fields are too noisy for honeypot detection in this dataset. With the
compact actionable knowledge base, the current scan reports 83 verified
failures, 21 suspicious candidates, and 537 clean candidates.

## 3. Create a fresh calibration set

The old calibration must not be reused because integrity filtering changes the
eligible population. Run initial scoring with the integrity report:

```bash
contextmatch score \
  --input top_700.jsonl \
  --expected-count 641 \
  --knowledge-base knowledge_base.json \
  --integrity-report runs/integrity/integrity_report.jsonl \
  --output runs/calibration/initial_scores.jsonl \
  --report runs/calibration/initial_timing.json

contextmatch select-calibration \
  --input top_700.jsonl \
  --expected-count 641 \
  --scores runs/calibration/initial_scores.jsonl \
  --size 40 \
  --output calibration/reviews.json
```

Verified failures skip Qwen and receive deterministic score zero.
`select-calibration` excludes them while keeping suspicious candidates eligible.

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
  --knowledge-base knowledge_base.json \
  --integrity-report runs/integrity/integrity_report.jsonl \
  --output calibration/report.json
```

This produces the **Stage 1 comparison outputs**:

- `calibration/stage_1_calibration_predictions.jsonl`: complete Qwen
  assessments for the 32 holdout candidates.
- `calibration/stage_1_calibration_comparison.csv`: reviewed score versus
  predicted score, absolute error, confidence, and disqualifier agreement.
- `calibration/report.json`: aggregate calibration pass/fail metrics.

Calibration passes when at least 80% of holdout totals are within 10 points,
mean absolute error is at most 8, and no reviewed disqualifier is missed.
Revise rubric wording or reviewed anchors before a final run if it fails.

## 4. Rank and export

```bash
contextmatch run \
  --input top_700.jsonl \
  --expected-count 641 \
  --anchors calibration/anchors.json \
  --knowledge-base knowledge_base.json \
  --integrity-report runs/integrity/integrity_report.jsonl \
  --output team_xxx.csv \
  --artifacts-dir runs/final
```

The run:

1. Assigns verified failures score zero without calling Qwen.
2. Scores clean and suspicious candidates with Qwen; suspicious warnings are
   not shown to Qwen. `expert_skill_zero_usage` receives a 5-point deterministic
   penalty, capped at 10 points.
3. Repeats ranks 70–180, confidence below 0.75, and conflicting cases.
4. Uses thinking-mode adjudication when disqualifier flags disagree.
5. Comparatively reranks the leading 150 eligible candidates in three
   randomized rounds of groups of ten.
6. Combines 80% rubric score with 20% comparative percentile.
7. Generates factual, unique, 8–49 word reasoning for the final 100.
8. Rejects the output if a verified failure enters the top 100.

### Three-stage comparison outputs

The pipeline preserves explicit outputs for each evaluation stage:

**Stage 1 — Calibration evaluation**

Located in `calibration/` after `evaluate-calibration`:

- `stage_1_calibration_comparison.csv`
- `stage_1_calibration_predictions.jsonl`
- `report.json`

This stage contains only the 32 reviewed holdout candidates. It measures
whether Qwen follows the manually corrected scoring standard; it is not a
ranking of all 641 candidates.

**Stage 2 — Individual scoring and re-evaluation**

Located in `runs/final/` after `contextmatch run`:

- `stage_2_individual_ranking.csv`
- `stage_2_individual_assessments.jsonl`

The CSV contains all 641 candidates and shows initial-pass rank/score,
post-repeat rank/score, rank movement, confidence, score caps, disqualifiers,
integrity status/reasons, knowledge facts, and whether the candidate was
repeated or adjudicated.

**Stage 3 — Comparative reranking**

Located in `runs/final/`:

- `stage_3_comparative_ranking.csv`
- `stage_3_comparative_assessments.jsonl`

The CSV compares Stage 2 rank and score with final rank, comparative
percentile, final score, rank movement, top-150 comparison participation, and
top-100 selection.

Validate independently:

```bash
contextmatch validate-output \
  --input top_700.jsonl \
  --expected-count 641 \
  --knowledge-base knowledge_base.json \
  --integrity-report runs/integrity/integrity_report.jsonl \
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
  --expected-count 641 \
  --anchors calibration/anchors.json \
  --knowledge-base knowledge_base.json \
  --integrity-report runs/integrity/integrity_report.jsonl \
  --concurrency 16 \
  --output runs/benchmark.jsonl \
  --report runs/benchmark.json
```

Do not increase concurrency if it causes out-of-memory errors, long queueing,
or lower throughput.
