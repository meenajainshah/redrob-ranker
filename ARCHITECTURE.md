# ARCHITECTURE — Redrob Ranker

## Context & Why

- **Type:** Technical document — design + defence reference for the ranking system.
- **Reason:** The challenge scores top-100 quality (NDCG@10/50, MAP, P@10) and runs a Stage-3 reproduction + Stage-5 “defend your work” interview. The design has to be reproducible under hard compute limits and explainable line by line.
- **How it helps:** One map from JD → ranked CSV, with the rationale for every scoring choice and trap rule.
- **Status:** Implemented in `rank.py`; produces a validated 100-row submission in ~75 s / ~1 GB, CPU-only, no network.

## 1. Design constraints (drove every decision)

CPU-only, no network, ≤5 min, ≤16 GB at ranking time. **Consequence:** no LLM call per candidate, no hosted embeddings. We use a *local* hashing TF vectoriser for lexical relevance and deterministic feature/rule scoring — fast, reproducible, and explainable. (Production note: a sentence-transformer / BGE-E5 embedding stage is a drop-in upgrade for the lexical component; the interface is isolated so it can be swapped without touching the scorer.)

## 2. Pipeline

```
JD (fixed)                         candidates.jsonl (100k)
   │  parsed into requirement model        │  one streaming pass
   ▼                                        ▼
JD query string ──► HashingVectorizer ──► cosine            extract():
                                            │               role identity · IR/ML evidence (career text)
                                            ▼               · experience band · product exposure
                              compute_scores(): lexical + structured fit
                                            │   − trap penalties  × availability × location
                                            │   honeypot ⇒ floor
                                            ▼
                              sort (score, evidence, cos, id) ─► top 100
                                            ▼
                              monotone score [0.40–0.99] + deterministic reasoning ─► submission.csv
```

## 3. The requirement model (from the JD)

- **Must:** AI/ML/IR engineering identity; production embeddings/retrieval/vector-search/ranking/recsys; strong Python; ranking-eval familiarity (NDCG/MRR/MAP).
- **Positive:** 5–9 yrs (peak ~7); product-company (not services-only); NLP/IR focus; LTR, fine-tuning, HR-tech, distributed systems, OSS; Noida/Pune/Hyderabad/Mumbai/Delhi-NCR or willing to relocate; active & responsive on-platform.
- **Down-weight / disqualify (the JD’s explicit list):** research-only with no production; recent LangChain-wrapper “AI” without depth; services-firm-only careers; title-chasers; CV/speech/robotics-only without NLP/IR.

## 4. Why evidence comes from career text, not `skills[]`

The dataset’s keyword-stuffers fill `skills[]` with AI terms. So skill-list matching *is* the trap. We mine retrieval/ranking/vector/NLP/eval/production language from **career-history descriptions and the summary**, where genuine builders describe what they shipped. This is the mechanism that (a) surfaces plain-language strong candidates and (b) ignores stuffed skill lists.

## 5. Scoring (`compute_scores`)

`score = role_prior + 2.3·evidence + 7·lexical_cos + 2.2·exp_fit + product + python + nice_to_haves`
then `− Σ penalties`, then `× availability × location`; honeypots → floor.
Title is a **weak prior** (2.0 ML / 1.2 software); evidence is the main driver, so a strongly-evidenced engineer outranks a weakly-evidenced ML-titled one.

## 6. Trap rules

- **Honeypot floor:** advanced/expert skill with 0 months used · summed tenure > YoE·12+12 · single role > whole career · end<start dates · impossible education years.
- **Penalties:** keyword-stuffer (−5), non-engineering role (−6), services-only (−2.5), research-only (−3), CV/speech-only (−2.5), LangChain-only (−2.5), title-chaser (−1.5), long notice (−0.6).
- **Availability multiplier** (≈0.45–1.0): recency, recruiter-response-rate, open-to-work, interview-completion.
- **Location multiplier:** India-hub 1.05 · India 1.0 · willing-to-relocate 0.92 · outside+no-relocate 0.80 (no visa sponsorship).

## 7. Explainability & anti-hallucination

Reasoning is built **only** from the candidate’s extracted fields (title, YoE, which evidence categories were found, product flag, response rate, recency, open-to-work, concrete concerns). Because it is generated from real values — not free text — it cannot reference skills the candidate doesn’t have. No LLM at ranking time. Reasonings are distinct and rank-consistent (concerns lower the rank and appear in the text together).

## 8. Results (local, on the released pool)

- Format: 100 rows, ranks 1–100 unique, scores non-increasing — validator passes.
- Top-100 composition: 0 honeypots, 0 keyword-stuffers, 0 non-engineering roles; experience median ~6.7 yrs (JD ideal 6–8); 98/100 India-based or relocating.
- Runtime ~75 s, peak ~1 GB, CPU-only, no network.

## 9. Known limitations / next

- Lexical relevance is TF-based; a local embedding reranker (BGE/E5, CPU, precomputed) would improve semantic recall — interface is isolated for this.
- Honeypot detection is rule-based; with the hidden set it can be tightened.
- Weights are set from the JD’s stated priorities, not tuned to a leaderboard (there is none); they are deliberately interpretable for the Stage-5 defence.