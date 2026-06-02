#!/usr/bin/env python3
"""
rank.py — Intelligent Candidate Discovery & Ranking (Redrob / India.runs)

Ranks the top-100 candidates from candidates.jsonl for the Senior AI Engineer JD.
Constraint-compliant: CPU-only, no network, single streaming pass, <5 min, <16 GB.

Pipeline:
  1. Parse JD into a structured requirement model (the JD is fixed; encoded below).
  2. Stream candidates -> extract structured features + a text blob (one pass).
  3. Lexical relevance: hashing TF vectors, cosine to a JD query (local, no model download).
  4. Structured fit score: role identity + production IR/ML evidence + experience band
     + product-company + nice-to-haves.
  5. Rule gates (the traps): honeypot -> floor; keyword-stuffing / non-engineering /
     consulting-only / research-only / wrong-domain / title-chaser -> penalties.
  6. Behavioral availability multiplier (recency, response, open-to-work, interview).
  7. Rank, take top-100, deterministic tie-break, monotone score, specific reasoning.

Usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""
import argparse, json, csv, math, re, sys
from datetime import date
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# ----------------------------- JD requirement model -----------------------------
# Parsed from job_description.md (Senior AI Engineer, Redrob AI). Fixed, so encoded.
ROLE_ML = re.compile(r"\b(machine learning|ml engineer|ai engineer|applied scientist|"
    r"data scientist|research engineer|mlops|ml platform|search engineer|"
    r"relevance engineer|nlp engineer)\b")
SOFTWARE_TITLE = re.compile(r"\b(software|backend|full[\s-]?stack|frontend|web|java|\.net|"
    r"python|golang|cloud|devops|sre|platform|data engineer|analytics engineer|"
    r"developer|programmer|sde|machine learning|ml|ai|data scientist|nlp)\b")
NON_TECH_TITLE = re.compile(r"\b(hr|human resource|recruit|marketing|sales|account|"
    r"content writer|copywriter|graphic|ux writer|customer support|operations manager|"
    r"business analyst|project manager|program manager|civil engineer|mechanical engineer|"
    r"electrical engineer|designer|administrator|finance|legal|teacher)\b")

# Core IR/ML/retrieval/ranking evidence (rewarded most when it appears in CAREER text,
# not just the skills[] list — this is how plain-language strong candidates surface and
# keyword-stuffers do not).
EVIDENCE = {
    "retrieval/ranking": re.compile(r"\b(retrieval|ranking|rank(ed|ing)?|learning[\s-]?to[\s-]?rank|"
        r"ltr|recommend(er|ation)?|recsys|relevance|search engine|semantic search|matching system)\b"),
    "embeddings/vector": re.compile(r"\b(embedding|vector (search|database|index|store)|"
        r"faiss|pinecone|weaviate|qdrant|milvus|opensearch|elasticsearch|bm25|hybrid search|ann)\b"),
    "nlp/llm": re.compile(r"\b(nlp|natural language|transformer|bert|sentence[\s-]?transformer|"
        r"bge|e5|llm|language model|fine[\s-]?tun|lora|qlora|peft|rag)\b"),
    "eval": re.compile(r"\b(ndcg|mrr|map@|mean average precision|a/?b test|offline (eval|metric)|"
        r"ranking metric|recall@|precision@)\b"),
    "production/scale": re.compile(r"\b(production|deployed|in production|at scale|real[\s-]?time|"
        r"latency|throughput|millions of|serving|inference|pipeline|to real users)\b"),
}
PYTHON = re.compile(r"\bpython\b")
CONSULTING = re.compile(r"\b(tcs|tata consultancy|infosys|wipro|accenture|cognizant|capgemini|"
    r"hcl tech|hcltech|tech mahindra|ltimindtree|larsen)\b")
CV_SPEECH_ROBO = re.compile(r"\b(computer vision|image classification|object detection|opencv|"
    r"speech recognition|asr|tts|robotics|slam|lidar|autonomous)\b")
RESEARCHY = re.compile(r"\b(phd|ph\.d|postdoc|research scholar|university|institute of technology research|"
    r"academic|publication|paper|thesis)\b")
LANGCHAIN_WRAP = re.compile(r"\b(langchain|llama[\s-]?index|prompt engineering|chatgpt wrapper|openai api)\b")
INDIA_HUB = re.compile(r"\b(noida|pune|hyderabad|bengaluru|bangalore|mumbai|delhi|gurgaon|gurugram|ncr)\b")
AI_SKILL = re.compile(r"\b(machine learning|deep learning|nlp|llm|transformer|pytorch|tensorflow|"
    r"embedding|rag|fine[\s-]?tun|retrieval|ranking|recommendation|vector|bert|hugging ?face|"
    r"scikit|xgboost|mlops|computer vision|reinforcement)\b")

REFERENCE_DAY = date(2026, 5, 27)  # max last_active_date in dataset (computed in recon)

JD_QUERY = ("senior ai engineer machine learning embeddings retrieval ranking "
            "recommendation relevance vector search hybrid search semantic search "
            "nlp information retrieval learning to rank ndcg mrr evaluation a/b test "
            "production deployed at scale python sentence transformers bge e5 faiss "
            "pinecone qdrant elasticsearch fine-tuning llm reranking recsys")

def make_vectorizer():
    # stateless hashing vectoriser -> identical behaviour on full pool or a small sample
    return HashingVectorizer(n_features=2**18, alternate_sign=False, norm="l2",
                             ngram_range=(1, 2), stop_words="english")

def days_since(d):
    try:
        y, m, dd = map(int, d.split("-")); return (REFERENCE_DAY - date(y, m, dd)).days
    except Exception:
        return 365

def parse_iso(d):
    try:
        y, m, dd = map(int, d.split("-")); return date(y, m, dd)
    except Exception:
        return None

# ----------------------------- feature extraction -----------------------------
def extract(c):
    """Return (compact feature dict). One candidate -> features. No raw dict retained."""
    p = c["profile"]; sig = c["redrob_signals"]; ch = c.get("career_history", [])
    edu = c.get("education", []); skills = c.get("skills", [])
    yoe = float(p.get("years_of_experience", 0) or 0)
    title = (p.get("current_title", "") or "").lower()

    # text blobs ------------------------------------------------------------
    career_txt = " ".join(((j.get("title", "") or "") + " " + (j.get("description", "") or "")[:320])
                          for j in ch)
    blob = " ".join([p.get("headline", "") or "", (p.get("summary", "") or "")[:500],
                     career_txt, " ".join(s.get("name", "") for s in skills)]).lower()

    # role identity ---------------------------------------------------------
    is_ml_role = bool(ROLE_ML.search(title)) or bool(ROLE_ML.search(blob[:400]))
    is_software = bool(SOFTWARE_TITLE.search(title))
    is_nontech = bool(NON_TECH_TITLE.search(title)) and not is_software and not is_ml_role

    # production IR/ML evidence in CAREER text (weighted) --------------------
    ev_hits = {k: len(rx.findall(career_txt.lower())) for k, rx in EVIDENCE.items()}
    ev_retrieval = ev_hits["retrieval/ranking"]
    ev_vector = ev_hits["embeddings/vector"]
    ev_nlp = ev_hits["nlp/llm"]
    ev_eval = ev_hits["eval"]
    ev_prod = ev_hits["production/scale"]
    evidence_score = (min(ev_retrieval, 4) * 2.2 + min(ev_vector, 4) * 2.0 +
                      min(ev_nlp, 4) * 1.3 + min(ev_eval, 3) * 1.6 + min(ev_prod, 5) * 0.6)
    has_python = bool(PYTHON.search(blob))

    # experience band fit (peak ~7, band 5-9 soft) --------------------------
    exp_fit = math.exp(-((yoe - 7.0) ** 2) / (2 * 3.0 ** 2))  # gaussian, 0..1

    # product vs services ---------------------------------------------------
    companies = " ".join((j.get("company", "") or "") for j in ch).lower()
    consulting_hits = len(CONSULTING.findall(companies + " " + (p.get("current_company","") or "").lower()))
    big_product = any(j.get("company_size") in ("1-10","11-50","51-200","201-500","501-1000")
                      for j in ch)  # startup/product-scale exposure
    services_only = consulting_hits >= max(1, len(ch))  # every listed co is consulting

    # nice-to-haves ---------------------------------------------------------
    nice = 0.0
    if re.search(r"\b(xgboost|learning to rank|lambdamart)\b", blob): nice += 0.6
    if re.search(r"\b(lora|qlora|peft|fine[\s-]?tun)\b", blob): nice += 0.5
    if re.search(r"\b(hr[\s-]?tech|recruit|talent|marketplace)\b", blob): nice += 0.5
    if re.search(r"\b(distributed|kubernetes|spark|kafka|scale)\b", blob): nice += 0.3
    if sig.get("github_activity_score", -1) is not None and sig.get("github_activity_score", -1) > 40: nice += 0.4

    # ----- trap signals -----
    # keyword stuffing: many AI skills declared, but role non-tech AND ~no career evidence
    ai_skill_count = sum(1 for s in skills if AI_SKILL.search((s.get("name","") or "").lower()))
    advanced_ai = sum(1 for s in skills if AI_SKILL.search((s.get("name","") or "").lower())
                      and s.get("proficiency") in ("advanced","expert"))
    keyword_stuffer = (ai_skill_count >= 6 and (is_nontech or (evidence_score < 1.5 and not is_ml_role)))

    # research-only (no production evidence)
    research_only = bool(RESEARCHY.search(blob)) and ev_prod == 0 and not big_product
    # CV/speech/robotics primary without NLP/IR
    cv_primary = bool(CV_SPEECH_ROBO.search(blob)) and (ev_retrieval + ev_vector + ev_nlp) == 0
    # langchain-wrapper-only recent AI w/o depth
    langchain_only = bool(LANGCHAIN_WRAP.search(blob)) and (ev_retrieval + ev_vector + ev_eval) == 0 and yoe < 4
    # title chaser: many jobs, short average tenure
    durations = [j.get("duration_months", 0) or 0 for j in ch]
    avg_tenure = (sum(durations) / len(durations)) if durations else 24
    title_chaser = len(ch) >= 4 and avg_tenure < 16

    # ----- honeypot detection (force to floor) -----
    hp = False
    for s in skills:
        if s.get("proficiency") in ("advanced", "expert") and (s.get("duration_months", 1) == 0):
            hp = True; break
    tot_dur = sum(durations)
    if tot_dur > yoe * 12 + 12: hp = True
    if any(d > yoe * 12 + 6 for d in durations): hp = True   # one role longer than whole career
    for j in ch:
        sd, ed = parse_iso(j.get("start_date", "")), parse_iso(j.get("end_date") or "")
        if sd and ed and ed < sd: hp = True
        if sd and sd > REFERENCE_DAY: hp = True
    for e in edu:
        sy, ey = e.get("start_year"), e.get("end_year")
        if isinstance(sy, int) and isinstance(ey, int) and ey < sy: hp = True

    # ----- behavioral availability multiplier -----
    rec = days_since(sig.get("last_active_date", ""))
    recency = 1.0 if rec <= 30 else max(0.45, 1.0 - (rec - 30) / 300.0)
    resp = float(sig.get("recruiter_response_rate", 0) or 0)
    icr = float(sig.get("interview_completion_rate", 0) or 0)
    otw = 1.0 if sig.get("open_to_work_flag") else 0.55
    avail = 0.45 + 0.55 * (0.40 * recency + 0.30 * resp + 0.20 * otw + 0.10 * icr)

    # ----- location multiplier -----
    loc = ((p.get("location","") or "") + " " + (p.get("country","") or "")).lower()
    in_india = "india" in loc or bool(INDIA_HUB.search(loc))
    relocate = bool(sig.get("willing_to_relocate"))
    if in_india: location_mult = 1.05 if INDIA_HUB.search(loc) else 1.0
    elif relocate: location_mult = 0.92
    else: location_mult = 0.80  # outside India, won't relocate, no visa sponsorship

    return dict(cid=c["candidate_id"], blob=blob, yoe=yoe,
                is_ml_role=is_ml_role, is_software=is_software, is_nontech=is_nontech,
                evidence_score=evidence_score, has_python=has_python, exp_fit=exp_fit,
                services_only=services_only, big_product=big_product, nice=nice,
                keyword_stuffer=keyword_stuffer, research_only=research_only,
                cv_primary=cv_primary, langchain_only=langchain_only, title_chaser=title_chaser,
                honeypot=hp, avail=avail, location_mult=location_mult,
                ai_skill_count=ai_skill_count, advanced_ai=advanced_ai,
                ev_retrieval=ev_retrieval, ev_vector=ev_vector, ev_nlp=ev_nlp, ev_eval=ev_eval,
                resp=resp, recency_days=rec, otw=bool(sig.get("open_to_work_flag")),
                notice=int(sig.get("notice_period_days", 0) or 0),
                gh=float(sig.get("github_activity_score", -1) or -1))

# ----------------------------- main ranking -----------------------------
def rank_feats(feats, topk=100):
    """feats: list from extract(). Returns list of (cid, rank, score_str, reasoning)."""
    hv = make_vectorizer()
    X = hv.transform(f["blob"] for f in feats)
    qv = hv.transform([JD_QUERY])
    cos = np.asarray(X.dot(qv.T).todense()).ravel()
    scores = compute_scores(feats, cos)
    n = len(feats)
    order = sorted(range(n), key=lambda i: (-scores[i], -feats[i]["evidence_score"],
                                            -cos[i], feats[i]["cid"]))
    top = order[:min(topk, n)]
    raw = np.array([scores[i] for i in top], dtype=np.float64)
    lo, hi = raw.min(), raw.max()
    norm = (raw - lo) / (hi - lo) if hi > lo else np.linspace(1, 0, len(raw))
    out_scores = np.round(0.40 + 0.59 * norm, 4)
    return [(feats[i]["cid"], pos, f"{out_scores[pos-1]:.4f}", reasoning(feats[i]))
            for pos, i in enumerate(top, start=1)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.jsonl")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

    feats = []
    with open(args.candidates, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                feats.append(extract(json.loads(line)))
    print(f"[1/2] parsed {len(feats)} candidates", file=sys.stderr)
    rows = rank_feats(feats, args.topk)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        w.writerows(rows)
    print(f"[2/2] wrote {args.out} ({len(rows)} rows)", file=sys.stderr)

# ----------------------------- scoring -----------------------------
def compute_scores(feats, cos):
    """Combine lexical relevance + structured fit + trap penalties + multipliers.
    Title gives a modest prior; production IR/ML *evidence in career text* is the
    main driver, so a strong plain-language ('Tier-5') engineer can outrank a
    weakly-evidenced ML-titled one — the behaviour the JD explicitly asks for."""
    n = len(feats)
    scores = np.zeros(n, dtype=np.float64)
    for i, f in enumerate(feats):
        role = 2.0 if f["is_ml_role"] else (1.2 if f["is_software"] else 0.0)  # modest prior
        s = (role
             + 2.3 * f["evidence_score"]            # production IR/ML evidence (career text) — main driver
             + 7.0 * cos[i]                          # lexical JD relevance
             + 2.2 * f["exp_fit"]                    # experience band
             + (1.0 if f["big_product"] else 0.0)    # product/startup exposure
             + (0.6 if f["has_python"] else 0.0)
             + f["nice"])
        # penalties (the traps)
        if f["is_nontech"]:      s -= 6.0
        if f["keyword_stuffer"]: s -= 5.0
        if f["services_only"]:   s -= 2.5
        if f["research_only"]:   s -= 3.0
        if f["cv_primary"]:      s -= 2.5
        if f["langchain_only"]:  s -= 2.5
        if f["title_chaser"]:    s -= 1.5
        if f["notice"] > 60:     s -= 0.6
        # multipliers
        s = s * f["avail"] * f["location_mult"]
        if f["honeypot"]:        s = -1e6            # floor honeypots
        scores[i] = s
    return scores

# ----------------------------- reasoning (deterministic, specific, no hallucination) -----------------------------
def reasoning(f):
    """1-2 sentences built only from this candidate's real, extracted fields."""
    # role + experience
    parts = []
    base = f"{f['yoe']:.1f} yrs"
    if f["is_ml_role"]:
        parts.append(f"AI/ML engineer ({base})")
    elif f["is_software"]:
        parts.append(f"software engineer ({base})")
    else:
        parts.append(f"non-engineering profile ({base})")
    # evidence
    ev = []
    if f["ev_retrieval"]: ev.append("retrieval/ranking work")
    if f["ev_vector"]: ev.append("vector/search infra")
    if f["ev_nlp"]: ev.append("NLP/LLM exposure")
    if f["ev_eval"]: ev.append("ranking-eval experience")
    if ev:
        parts.append("evidence of " + ", ".join(ev[:3]))
    elif f["evidence_score"] < 1:
        parts.append("limited direct retrieval/ranking evidence")
    if f["big_product"]:
        parts.append("product-company background")
    # availability signal
    sig = f"recruiter response {f['resp']:.2f}"
    if f["recency_days"] <= 30: sig += ", active recently"
    elif f["recency_days"] > 150: sig += f", inactive {f['recency_days']}d"
    if f["otw"]: sig += ", open to work"
    parts.append(sig)
    # honest concerns
    concerns = []
    if f["keyword_stuffer"]: concerns.append("AI skills listed but unsupported by role/history")
    if f["services_only"]: concerns.append("services-firm-only career")
    if f["title_chaser"]: concerns.append("short average tenure")
    if f["notice"] > 60: concerns.append(f"{f['notice']}-day notice")
    if f["location_mult"] < 0.85: concerns.append("outside India, no relocation")
    txt = "; ".join(parts)
    if concerns:
        txt += ". Concern: " + ", ".join(concerns[:2]) + "."
    else:
        txt += "."
    return txt[:300]

if __name__ == "__main__":
    main()
