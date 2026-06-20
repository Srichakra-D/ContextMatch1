# Idea 1: Unconstrained Teacher and Constrained Student

## Objective

Build an expensive, unconstrained ranking system that acts as a high-quality
teacher. Use its judgments to train a compact student ranker that can reproduce
the ranking under the hackathon constraints:

- Maximum 5 minutes wall-clock time
- Maximum 16 GB RAM
- CPU only
- No network access during ranking

The teacher is not official ground truth. It is a development oracle used to
compare approaches and generate training supervision.

## Why Use a Teacher Ensemble

A single large model can misunderstand the job description, reward polished
profiles, miss synthetic traps, or apply inconsistent standards. The teacher
should therefore combine:

1. Deterministic integrity checks
2. High-recall embedding retrieval
3. A dedicated cross-encoder reranker
4. One or more strong reasoning models
5. Confidence estimates based on model agreement

## Proposed Pipeline

```text
100,000 candidates
        |
        v
Deterministic integrity and honeypot checks
        |
        v
Embedding retrieval: top 3,000-5,000
        |
        v
Cross-encoder reranking: top 500-1,000
        |
        v
Large reasoning-model evaluation
        |
        v
Consensus labels, pairwise preferences, evidence, and confidence
        |
        v
Train a compact student ranker
```

## 1. Integrity and Honeypot Detection

Programmatically identify factual contradictions before using semantic models:

- Employment before a company's founding date
- Role duration inconsistent with start and end dates
- Total career duration inconsistent with stated experience
- Expert skills with zero months of usage
- Current role inconsistent with the profile
- Overlapping or otherwise impossible career timelines
- Job-description hard disqualifiers

These checks should produce explicit features and penalties. A language model
should not override a verified factual contradiction.

## 2. High-Recall Retrieval

Use a strong open embedding model such as `Qwen3-Embedding-8B`.

Retrieve candidates using multiple representations of the job requirements:

- Production retrieval and hybrid search
- Ranking, recommendation, and learning-to-rank
- Evaluation frameworks and online experimentation
- Strong Python and production ML engineering
- Product-company and shipping experience
- Founding-team ownership and technical judgment

Career descriptions should receive more weight than self-declared skills.
Retrieve a union of the results from all queries to avoid losing strong
plain-language candidates.

## 3. Cross-Encoder Reranking

Use a dedicated reranker such as `Qwen3-Reranker-8B` on the retrieval shortlist.

The reranker should receive:

- The complete job-description rubric
- Candidate profile and career history
- Instructions to prioritize demonstrated career evidence
- Instructions to penalize skills-only keyword matching
- Integrity flags from the deterministic layer

Keep approximately the strongest 500-1,000 candidates for expensive reasoning.

## 4. Reasoning-Model Teacher

Possible free, locally runnable teacher models include:

- A quantized Qwen reasoning model that fits the available hardware
- `DeepSeek-R1-Distill-Qwen-32B` or a smaller quantized variant

The available machine has an NVIDIA RTX 5080 Laptop GPU with approximately
16 GB VRAM and 32 GB system RAM. Large quantized models may require partial CPU
offloading and can run slowly, which is acceptable during teacher generation.

Evaluate each shortlisted candidate with structured output:

```json
{
  "production_retrieval": 4,
  "ranking_evaluation": 3,
  "product_shipping": 3,
  "python_engineering": 3,
  "seniority_fit": 2,
  "location_logistics": 1,
  "availability": 1,
  "disqualifier": false,
  "evidence": [
    "Candidate-specific supporting fact"
  ],
  "concerns": [
    "Candidate-specific weakness"
  ],
  "overall_relevance": 4
}
```

Run multiple evaluations with:

- Randomized candidate order
- Slightly different rubric wording
- At least two independent model families where practical

Model disagreement should reduce label confidence rather than being hidden.

## 5. Teacher Supervision

The teacher should generate several forms of supervision:

- Graded relevance labels
- Pairwise candidate preferences
- Per-dimension scores
- Hard disqualification labels
- Evidence spans from the candidate profile
- Confidence based on agreement and stability

Pairwise preferences are particularly useful because the final problem is
ranking rather than absolute score prediction.

## 6. Student Ranker

Train a compact learning-to-rank model, with LightGBM LambdaMART as the initial
choice.

Potential student features:

- Current and historical title categories
- Years and recency of relevant experience
- Career-supported retrieval and ranking evidence
- Evaluation and A/B-testing evidence
- Production deployment evidence
- Python and ML engineering evidence
- Product-company versus services-only history
- Job tenure and switching patterns
- Location, relocation, work mode, and notice period
- Activity and recruiter-response signals
- GitHub activity and external validation
- Integrity and honeypot flags
- Compact embedding or reranker scores

Do not use candidate IDs or exact synthetic-template identifiers as features.
The student should learn general matching logic, not memorize this dataset.

## 7. Local Evaluation

The teacher score is not an estimate of the hidden leaderboard score. Use it to
compare student versions and test specific engineering choices.

Measure:

- Student NDCG against teacher relevance labels
- Pairwise agreement with the teacher ensemble
- Agreement between independent teacher models
- Retrieval recall of teacher-approved candidates
- Honeypot rejection rate
- Ranking stability under prompt and scoring changes
- Performance on high-confidence versus uncertain examples
- Runtime and peak memory under the official constraints

Use group-based holdouts:

- Hold out complete career-description templates
- Hold out companies
- Hold out title groups

This is more reliable than a random candidate split because random splits can
leak synthetic generation patterns into both training and evaluation.

## 8. Final Production Path

The final submission pipeline should:

1. Read candidates in a streaming pass.
2. Compute deterministic and semantic features.
3. Apply the trained student ranker.
4. Remove hard integrity failures.
5. Select and order the top 100.
6. Generate factual reasoning from extracted evidence.
7. Validate the CSV.

The submission should not depend on network calls, large teacher models, or
manual lookup tables.

## Main Risk

The student may faithfully reproduce the teacher's mistakes. Mitigate this with
an ensemble, confidence weighting, deterministic checks, counterfactual tests,
and holdouts that expose template memorization.

The goal is not to claim that the teacher represents true relevance. The goal
is to create a substantially stronger and more consistent development signal
than manual labels from one person or simple keyword heuristics.
