# ContextMatch — Project Handoff

> This file records the earlier design discussion. The implemented and current
> workflow is documented in `README.md`: it consumes `top_700.jsonl` directly,
> uses Qwen3-14B-AWQ through vLLM, calibrates with 40 reviewed candidates, and
> does not build a shortlist from the 100,000-record pool.

## Current Goal

Generate a high-quality provisional ranking of the top 100 candidates for the
Redrob Senior AI Engineer job description. This result will be manually and
experimentally verified later.

The immediate objective is answer quality. The official five-minute CPU
submission constraint is not part of this first experimental phase. A fast,
reproducible submission system can be developed later using the teacher
results.

## Available Data

The challenge bundle is in `India_runs_data_and_ai_challenge/`.

Important files:

- `candidates.jsonl`: 100,000 candidate records, approximately 480 MB.
- `job_description.docx`: Senior AI Engineer — Founding Team role.
- `candidate_schema.json`: candidate record schema.
- `submission_spec.docx`: output and evaluation requirements.
- `redrob_signals_doc.docx`: behavioral-signal definitions.
- `validate_submission.py`: CSV format validator.

The required final CSV columns are:

```text
candidate_id,rank,score,reasoning
```

Requirements for the provisional result:

- Exactly 100 candidates.
- Unique ranks from 1 through 100.
- Scores must be non-increasing by rank.
- Reasoning must be candidate-specific, factual, and below 50 words.
- Detailed reasoning is generated only for the final top 100.

## Agreed Model Strategy

Use an open-weight LLM as an unrestricted teacher:

- Model: `Qwen/Qwen3-14B`.
- Quantization: 4-bit.
- Target hardware: NVIDIA RTX 5080 Laptop GPU with 16 GB VRAM.
- Bulk scoring mode: non-thinking.
- Final reasoning mode: may be slower and more detailed.
- Training or QLoRA is not currently required; how to teach or fine-tune the
  model will be discussed later.

Do not send all 100 candidates or the entire dataset in one prompt. Ranking
requires multiple controlled calls and global sorting.

## Performance Requirement

Bulk LLM scoring should average less than one second per shortlisted candidate.
This is an amortized throughput target:

- A batch of 32 candidates taking 16 seconds equals 0.5 seconds per candidate.
- Individual request latency does not need to be below one second.
- Bulk output should contain only candidate IDs and compact numeric scores.
- Explanations during bulk scoring are intentionally disabled.

Scoring all 100,000 candidates with an LLM would still be too slow even at this
rate. A fast, high-recall programmatic shortlist is required first.

## Planned Pipeline

```text
100,000 candidate JSONL records
        |
        v
Streaming feature extraction and integrity checks
        |
        v
High-recall shortlist of approximately 2,000 candidates
        |
        v
Compact candidate representations
        |
        v
Batched Qwen3-14B score-only inference
        |
        v
Global sorting and stronger review near the cutoff
        |
        v
Final top 100
        |
        v
Qwen-generated factual reasoning under 50 words
        |
        v
Validated CSV
```

## Ranking Priorities

Career evidence is the primary source of truth. Skills are noisy and should
only corroborate career evidence.

Strong positive evidence:

- Production embeddings-based retrieval.
- Search, ranking, recommendation, or matching systems.
- Hybrid retrieval, vector databases, BM25, FAISS, or similar infrastructure.
- Learning-to-rank and behavioral reranking.
- NDCG, MRR, MAP, offline evaluation, online A/B testing, and feedback loops.
- Production deployment, monitoring, index refresh, drift handling, latency,
  and real-user outcomes.
- Strong Python and hands-on ML engineering.
- Product-company shipping experience.
- Ownership, technical judgment, and mentoring.

Negative evidence:

- Pure research without production deployment.
- AI experience limited to recent LangChain or API-based demos.
- No recent hands-on coding.
- Entire career in consulting/services without product experience.
- Primarily computer vision, speech, or robotics without meaningful NLP/IR.
- Keyword-heavy skills unsupported by career history.
- Impossible timelines or expert skills with zero usage duration.

Behavioral signals such as recent activity, response rate, notice period, and
relocation should modify the technical-fit score, not replace it.

## Dataset Findings

- The data contains 100,000 candidates.
- Career descriptions are highly synthetic: only 44 unique description
  templates were found.
- Skills are deliberately noisy and frequently contradict titles or career
  descriptions.
- Strong search/ranking candidates are rare and concentrated in a small number
  of high-quality career templates.
- Approximately 55 obvious integrity anomalies were identified, including
  inconsistent role durations and expert skills with zero usage.
- Around 99 candidates contain career evidence matching retrieval, evaluation,
  and production concepts under a simple keyword analysis. This is only an
  exploratory count, not the final ranking.
- The provided sample submission is structurally valid but intentionally
  low-quality.

## Next Implementation Step

Start by implementing a dependency-free streaming shortlist tool that:

1. Reads `candidates.jsonl` one record at a time.
2. Extracts career-evidence, title, experience, logistics, behavioral, and
   integrity features.
3. Uses broad recall-oriented rules so potentially strong candidates are not
   removed early.
4. Keeps approximately 2,000 candidates.
5. Writes compact JSONL records for batched LLM scoring.
6. Includes tests and reports shortlist runtime and feature distributions.

After the shortlist is inspected, integrate the Qwen3-14B runtime appropriate
for the RTX machine's operating system. The operating system of that machine
has not yet been confirmed.

## Local Environment

The current workspace is:

```text
/Users/srichakra/My_Projects/ContextMatch1
```

The currently inspected local machine is macOS ARM, not the RTX 5080 machine.
Python 3.12 is installed. The project directory is not currently initialized as
a Git repository.
