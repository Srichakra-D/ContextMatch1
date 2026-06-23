# ContextMatch Project Plan

## 1. Objective

Rank the strongest 100 candidates for Redrob's Senior AI Engineer — Founding
Team role.

The final output is a CSV with:

```text
candidate_id,rank,score,reasoning
```

Each selected candidate must have:

- A unique rank from 1 through 100.
- A numeric score that decreases or remains equal as rank increases.
- Candidate-specific reasoning of fewer than 50 words.
- Reasoning supported entirely by facts in the candidate profile.

The current candidate pool is `top_700.jsonl`, which contains 641 shortlisted
candidates. These records were selected from the original 100,000-candidate
dataset.

## 2. Model and Infrastructure

The ranking model is:

```text
Qwen/Qwen3-14B-AWQ
```

It runs through vLLM on a shared Linux server with an NVIDIA 40 GB GPU.

Configuration goals:

- Use approximately 16 GB VRAM as a safe target.
- Use 4-bit AWQ quantization.
- Disable extended thinking during bulk scoring.
- Use batching to keep average bulk-scoring time below one second per
  candidate.
- Allow slower inference for uncertain-case adjudication and final reasoning.

The first model launch downloads the model from Hugging Face. Later launches
reuse the cached files.

## 3. Candidate Data Verification

Before model scoring:

1. Validate every candidate against the provided candidate schema.
2. Check candidate ID format and uniqueness.
3. Confirm required fields and date formats.
4. Detect verified profile contradictions, including:
   - Current-role mismatches.
   - Materially incorrect employment durations.
   - Expert skills with zero usage duration.
5. Verify selected candidate records against the original 100,000-record
   dataset.

The `verification/` directory contains:

- `top40.json`: the current 40-candidate calibration set.
- `verify_candidate_subset.py`: exact record-comparison utility.

## 4. JD-Based Scoring Rubric

Qwen receives a condensed representation of the job description, explicit
scoring rules, disqualifier definitions, calibration examples, and one
candidate profile.

Career history is the primary evidence source. Skills, headlines, and summaries
only corroborate career evidence.

The model assigns integer scores across seven dimensions:

| Dimension | Maximum |
|---|---:|
| Production retrieval, search, ranking, and recommendation | 25 |
| Evaluation, relevance metrics, and A/B testing | 20 |
| Production ML and Python engineering | 15 |
| Product shipping and measurable outcomes | 15 |
| Ownership, seniority, mentoring, and founding-team fit | 10 |
| NLP, LLM, and useful secondary skills | 5 |
| Location, availability, notice period, and engagement | 10 |
| **Total** | **100** |

Python calculates the total instead of asking the model to perform the final
arithmetic.

## 5. Disqualifiers and Score Caps

Verified impossible profiles receive a score of zero.

Other explicit JD disqualifiers apply maximum score caps:

| Condition | Maximum score |
|---|---:|
| Pure research without production deployment | 30 |
| Recent LLM/API-only work without earlier production ML | 30 |
| No recent hands-on production coding | 40 |
| Entire career in services without product experience | 40 |
| Primarily CV, speech, or robotics without meaningful NLP/IR | 45 |

Uncertain concerns do not trigger hard caps. They remain normal scoring
penalties or are sent for additional review.

## 6. Prompt Calibration

The model is calibrated before the final ranking. This is prompt calibration,
not model-weight training.

### Initial assessment

Qwen first scores all 641 candidates using only the rubric.

### Calibration set

The system selects 40 diverse candidates covering:

- Excellent profiles.
- Strong profiles.
- Borderline profiles.
- Weak profiles.
- Disqualified profiles.
- Integrity-risk profiles.

Qwen drafts an assessment for each candidate. The user and model review every
draft together:

- Correct assessments are approved.
- Incorrect assessments are corrected.
- No record remains pending.

### Anchors and holdout

The 40 reviewed candidates are divided into:

- Eight calibration anchors included in future prompts.
- Thirty-two holdout candidates used to test calibration quality.

Calibration passes when:

- At least 80% of holdout totals are within 10 points of reviewed totals.
- Mean absolute error is no more than eight points.
- No reviewed disqualifier is missed.

If calibration fails, adjust the rubric wording or anchor examples before the
final run.

QLoRA fine-tuning is deferred. It should only be considered if prompt
calibration remains inconsistent after correction.

## 7. Final Ranking Pipeline

The system preserves separate outputs for three stages so their scores and rank
changes can be compared.

### Integrity stage

Before calibration or Qwen scoring, scan all 641 candidates using
`knowledge_base.json`.

This implementation phase is intentionally limited to `top_700.jsonl`.
Scanning or ranking the original 100,000-candidate file is out of scope.

- `verified_failure`: assign score zero and skip Qwen.
- `suspicious`: retain for audit, apply no penalty, and hide the warning from
  Qwen.
- `clean`: continue normally.

The current compact-knowledge-base scan identifies 104 verified failures, 66
suspicious candidates, and 471 clean candidates. Generate a new 40-candidate calibration
set after this scan; do not reuse calibration created before integrity
integration.

### Stage 1: calibration evaluation

After the 40 manual reviews are split into eight anchors and 32 holdout
candidates, Qwen re-scores the holdout set. The output records reviewed score,
predicted score, absolute error, confidence, and disqualifier agreement.

Stage 1 measures whether the model understands the rubric. It contains only the
32 holdout candidates and is not the final 641-candidate ranking.

### Stage 2: individual scoring and re-evaluation

Score all 641 candidates independently using:

- The condensed JD.
- The full scoring rubric.
- The eight reviewed anchors.
- The candidate's relevant profile, career history, skills, and behavioral
  signals.

The model returns:

- Seven dimension scores.
- Specific evidence.
- Concerns.
- Disqualifier flags.
- Confidence.

Verified failures do not invoke Qwen and cannot be selected as calibration
anchors. Suspicious candidates are evaluated exactly like clean candidates.

### Repeat uncertain candidates

Run a second assessment for:

- Candidates initially ranked between 70 and 180.
- Candidates with confidence below 0.75.
- Candidates whose evidence is internally conflicting.

Average repeated dimension scores.

If the two runs disagree about a disqualifier, use a slower thinking-mode
assessment to adjudicate the disagreement.

The Stage 2 output records every candidate's initial-pass rank and score,
post-repeat rank and score, movement, confidence, caps, disqualifiers, and
whether repeat scoring or adjudication occurred.

### Stage 3: comparative reranking

Take the leading 150 candidates and compare them in overlapping randomized
groups of ten.

Verified failures are excluded from comparison groups and cannot enter the
final top 100.

Run three comparison rounds and aggregate group preferences into a comparative
percentile.

Calculate the final score:

```text
final score = 80% rubric score + 20% comparative percentile
```

Sort by final score. Break exact score ties using candidate ID ascending.

The Stage 3 output records Stage 2 rank, comparative percentile, final score,
final rank, rank movement, and whether the candidate entered the final top 100.

## 8. Final Reasoning

Generate reasoning only for the final top 100.

Each reasoning entry must:

- Contain one or two sentences.
- Use fewer than 50 words.
- Refer to specific candidate evidence.
- Connect that evidence to the JD.
- Mention a material concern when one exists.
- Avoid unsupported claims and generic praise.
- Be substantively different from other candidates' reasoning.

Invalid or duplicated reasoning is regenerated or replaced with a factual
fallback assembled from verified evidence.

## 9. Validation and Deliverables

The completed run produces:

- Initial model assessments.
- Repeated and adjudicated assessments.
- Comparative-reranking results.
- Final scores and reasonings.
- Runtime and throughput reports.
- The final top-100 CSV.

Before submission, validate:

- Exactly 100 rows.
- Ranks 1 through 100 appear once each.
- Candidate IDs are unique and exist in the supplied candidate pool.
- Scores are finite and non-increasing.
- Equal-score ties follow candidate ID order.
- Reasoning is factual, unique, and under 50 words.

Run both the project validator and the organizer-provided validator.

## 10. Current Status

Completed:

- Ranking architecture.
- JD scoring rubric.
- Qwen/vLLM integration.
- Calibration workflow.
- Repeat scoring and adjudication logic.
- Comparative reranking.
- Reasoning generation.
- CSV validation.
- Candidate subset verification.
- Automated tests.

Remaining:

1. Set up the project on the Linux GPU server.
2. Start Qwen3-14B-AWQ through vLLM.
3. Run initial scoring.
4. Review and correct the 40 calibration candidates.
5. Evaluate calibration.
6. Run the final ranking.
7. Inspect and validate the final top-100 CSV.

Detailed installation and command instructions are in `README.md`.
