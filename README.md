# Redrob Ranker — Intelligent Candidate Discovery & Ranking

Top-100 candidate ranker for the **India.runs / Redrob** Senior AI Engineer JD.
CPU-only, no network at ranking time, ~75 s on 100k candidates, ~1 GB RAM.

## Reproduce the submission (single command)

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv candidates.jsonl   # spec check
```

`candidates.jsonl` is the released pool (the gzipped `.jsonl.gz` also works after `gunzip`).
The ranking step makes **no external API calls** and runs on CPU within the 5-minute budget.

## Approach (short)

The fixed JD is parsed into a structured requirement model. Each candidate is scored from:

1. **Lexical relevance** — a local hashing TF vector (1–2 grams, L2-normed) cosine’d to a JD query. No model download, no network.
1. **Production IR/ML evidence** — retrieval / ranking / recsys / vector-search / NLP / eval terms mined from **career-history text**, *not* the `skills[]` list. This is what lets a plain-language (“Tier-5”) engineer who *describes* building a recommendation system outrank someone who merely lists AI skills.
1. **Experience-band fit** — Gaussian peak at ~7 yrs (JD’s 5–9 band).
1. **Product-company exposure** and Python/SWE signal.

Then **rule gates** handle the documented traps, **multipliers** finish the score, and the top 100 get deterministic ranks and **fact-grounded reasoning**.

## Repo map

```
rank.py                   # the ranker: extract() -> compute_scores() -> reasoning() -> CSV
validate_submission.py    # spec validator (Section 3): 100 rows, unique ranks, monotone score, ids exist
requirements.txt          # numpy, scikit-learn
submission_metadata.yaml  # portal metadata mirror (fill before submitting)
ARCHITECTURE.md           # full design + the trap-handling rationale (defend-your-work reference)
sandbox_app.py            # Streamlit sandbox: upload <=100 candidates -> ranked CSV
```

## Trap handling (why the ranking is trustworthy)

- **Honeypots** → floored via internal-consistency checks (advanced/expert skill with 0 months used; summed tenure > years of experience; a single role longer than the whole career; end-date before start-date; impossible education years). **0 honeypots in the top 100.**
- **Keyword-stuffers** (AI skills listed, wrong role / no career evidence) → penalised. **0 in top 100.**
- **Non-engineering roles**, **services-only** careers, **research-only** (no production), **CV/speech/robotics-only**, **LangChain-wrapper-only**, **title-chasers** → penalised per the JD’s explicit “do NOT want” list.
- **Availability** → recency, recruiter-response-rate, open-to-work and interview-completion form a multiplier, so perfect-on-paper-but-unreachable candidates are down-weighted.

## Compute & reproducibility

|Constraint|This system                       |
|----------|----------------------------------|
|Runtime   |~75 s wall-clock on 100k (≤ 5 min)|
|Memory    |~1 GB peak (≤ 16 GB)              |
|Compute   |CPU only                          |
|Network   |none at ranking time              |

## AI tools declaration

Built with assistance from Claude and Cursor; all engineering decisions (feature design, trap rules, weighting, validation) are documented in `ARCHITECTURE.md`.