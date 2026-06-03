#!/usr/bin/env python3
"""Sandbox demo for the Redrob ranker (submission_spec Section 10.5).

Upload a CSV or JSONL of candidates -> ranks them against the Senior AI Engineer JD
with the exact rank.py pipeline (CPU-only, no network) -> shows the table and a
downloadable CSV. A built-in sample is included as a one-tap fallback.
Run: streamlit run sandbox_app.py
"""
import io, json, csv
import numpy as np
import streamlit as st
import rank as R  # extract(), compute_scores(), make_vectorizer(), JD_QUERY, reasoning()

CAP = 5000  # hard processing cap so the demo always runs inside Streamlit's limits

# ---- built-in synthetic sample (no challenge data; shows the trap handling) ----
def _c(cid,title,yoe,summ,loc,country,skills,sig):
    return {"candidate_id":cid,"profile":{"current_title":title,"years_of_experience":yoe,
        "headline":title,"summary":summ,"current_company":"","location":loc,"country":country},
        "career_history":[{"title":title,"company":"","company_size":"51-200","description":summ,
        "duration_months":int(yoe*12),"start_date":"2020-01-01","end_date":None,"is_current":True}],
        "education":[{"start_year":2012,"end_year":2016}],"skills":skills,"redrob_signals":sig}
def _sk(n,p,m): return {"name":n,"proficiency":p,"duration_months":m,"endorsements":10}
_S=lambda **k:{"last_active_date":"2026-05-20","recruiter_response_rate":0.8,
    "interview_completion_rate":0.7,"open_to_work_flag":True,"willing_to_relocate":True,
    "notice_period_days":30,"github_activity_score":55,**k}
SAMPLE=[
 _c("CAND_S01","Senior Machine Learning Engineer",7,
   "Built the recommendation and ranking stack at a product company: embedding-based retrieval with a vector index, hybrid search, learning-to-rank reranker, A/B tests and NDCG/MRR offline evaluation, deployed to production at scale for millions of users.",
   "Pune","India",[_sk("Python","expert",84),_sk("Information Retrieval","advanced",40)],_S()),
 _c("CAND_S02","Machine Learning Engineer",6,
   "Shipped a recommendation system: embedding retrieval plus a Qdrant vector database, a ranking model with offline NDCG evaluation, and low-latency real-time serving.",
   "Bengaluru","India",[_sk("Python","expert",72),_sk("Recommender Systems","advanced",60)],_S(recruiter_response_rate=0.88)),
 _c("CAND_S04","Backend Engineer",6,
   "Backend engineer who recently built a product recommendation pipeline end to end: candidate retrieval and a ranking service deployed to production for real users. Strong Python, moving deeper into ranking and retrieval.",
   "Noida","India",[_sk("Python","advanced",72)],_S(github_activity_score=70)),
 _c("CAND_S05","Data Scientist",5,
   "Built forecasting models and analytics dashboards; some classification models in production.",
   "Mumbai","India",[_sk("Python","advanced",60),_sk("Machine Learning","intermediate",30)],_S(recruiter_response_rate=0.6,github_activity_score=20)),
 _c("CAND_S06","HR Manager",4,
   "HR manager handling end-to-end recruitment, onboarding, payroll and employee engagement programs.",
   "Delhi","India",[_sk("Machine Learning","advanced",12),_sk("Deep Learning","advanced",12),_sk("NLP","advanced",12),_sk("LLM","expert",6),_sk("Transformers","expert",6)],_S(github_activity_score=0)),
 _c("CAND_S07","AI Engineer",9,"Led all AI initiatives.","Remote","India",
   [_sk("Machine Learning","expert",0),_sk("Deep Learning","expert",0),_sk("LLM","expert",0)],_S()),
]
# the honeypot (CAND_S07): "expert" skills with 0 months used -> floored by the ranker


# ---- CSV schema (flat, human-readable) -> nested record the ranker expects ----
CSV_COLUMNS = ["candidate_id","current_title","years_of_experience","summary","skills",
               "location","country","company","company_size","last_active_date",
               "recruiter_response_rate","interview_completion_rate","open_to_work",
               "willing_to_relocate","notice_period_days","github_activity_score"]

def _num(v, d):
    try: return float(v)
    except Exception: return d
def _bool(v, d=False):
    s = str(v).strip().lower()
    return s in ("1","true","yes","y","t") if s else d

def csv_row_to_record(row, i):
    g = lambda k, d="": (row.get(k) or d)
    yoe = _num(g("years_of_experience"), 0.0)
    summary = g("summary")
    skills = [{"name": s.strip(), "proficiency": "advanced", "duration_months": int(yoe*12)}
              for s in g("skills").split(",") if s.strip()]
    return {
        "candidate_id": g("candidate_id", f"ROW_{i+1}"),
        "profile": {"current_title": g("current_title"), "years_of_experience": yoe,
                    "headline": g("current_title"), "summary": summary,
                    "current_company": g("company"), "location": g("location"),
                    "country": g("country", "India")},
        "career_history": [{"title": g("current_title"), "company": g("company"),
                    "company_size": g("company_size", "51-200"), "description": summary,
                    "duration_months": int(yoe*12), "start_date": "2020-01-01",
                    "end_date": None, "is_current": True}],
        "education": [{"start_year": 2012, "end_year": 2016}],
        "skills": skills,
        "redrob_signals": {"last_active_date": g("last_active_date", "2026-05-01"),
                    "recruiter_response_rate": _num(g("recruiter_response_rate"), 0.7),
                    "interview_completion_rate": _num(g("interview_completion_rate"), 0.6),
                    "open_to_work_flag": _bool(g("open_to_work"), True),
                    "willing_to_relocate": _bool(g("willing_to_relocate"), True),
                    "notice_period_days": int(_num(g("notice_period_days"), 30)),
                    "github_activity_score": _num(g("github_activity_score"), 30)},
    }

def parse_upload(name, data_bytes):
    """Return (records, total_seen). Accepts CSV or JSONL/JSON; caps to CAP rows."""
    text = data_bytes.decode("utf-8", errors="ignore")
    recs = []
    if name.lower().endswith(".csv"):
        rd = list(csv.DictReader(io.StringIO(text)))
        total = len(rd)
        for i, row in enumerate(rd[:CAP]):
            recs.append(csv_row_to_record(row, i))
    else:
        lines = [l for l in text.splitlines() if l.strip()]
        total = len(lines)
        if total <= 1 and text.strip().startswith("["):       # JSON array
            arr = json.loads(text); total = len(arr); recs = arr[:CAP]
        else:
            for l in lines[:CAP]:
                try: recs.append(json.loads(l))
                except Exception: pass
    return recs, total

# ---- ranking for display (same pipeline as rank.py; robust score scaling) ----
def rank_for_display(records, topk):
    feats = [R.extract(c) for c in records]
    hv = R.make_vectorizer(); X = hv.transform(f["blob"] for f in feats); qv = hv.transform([R.JD_QUERY])
    cos = np.asarray(X.dot(qv.T).todense()).ravel()
    scores = R.compute_scores(feats, cos)
    order = sorted(range(len(feats)), key=lambda i: (-scores[i], -feats[i]["evidence_score"],
                                                      -cos[i], feats[i]["cid"]))
    top = order[:min(topk, len(feats))]
    raw = np.array([scores[i] for i in top], float)
    finite = raw[raw > -1e5]
    lo = finite.min() if len(finite) else raw.min(); hi = raw.max()
    norm = np.clip((raw - lo)/(hi - lo) if hi > lo else np.ones_like(raw), 0, 1)
    out = np.round(0.40 + 0.59*norm, 4)
    return [{"rank": pos, "candidate_id": feats[i]["cid"], "score": f"{out[pos-1]:.4f}",
             "reasoning": R.reasoning(feats[i])} for pos, i in enumerate(top, 1)]

def to_csv(rows):
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in rows: w.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])
    return buf.getvalue()

TEMPLATE = ("candidate_id,current_title,years_of_experience,summary,skills,location,country,"
    "recruiter_response_rate,open_to_work,willing_to_relocate\n"
    "C1,Senior ML Engineer,7,Built production retrieval and ranking systems; shipped a "
    "recommendation engine with embeddings and learning-to-rank at a product company,"
    "\"Python,Information Retrieval,Recommender Systems\",Pune,India,0.85,true,true\n"
    "C2,HR Manager,4,Handled recruitment payroll and engagement,"
    "\"Machine Learning,Deep Learning,NLP,LLM\",Delhi,India,0.6,true,true\n")

# ----------------------------- UI -----------------------------
st.set_page_config(page_title="Redrob Ranker — Sandbox", layout="wide")
st.title("Redrob Ranker — Sandbox")
st.caption("Ranks candidates against the Senior AI Engineer JD using the exact rank.py pipeline "
           "— CPU-only, no network. Upload your own candidates (CSV or JSONL) and download the ranking, "
           "or try the built-in sample.")

mode = st.radio("Input", ["Upload CSV / JSONL", "Built-in sample"], horizontal=True)
records, total = None, 0

if mode.startswith("Upload"):
    with st.expander("CSV format (click to see columns + download a template)"):
        st.write("One candidate per row. `summary` carries the career text the model reads. "
                 "`skills` is comma-separated. Missing columns fall back to sensible defaults.")
        st.code(", ".join(CSV_COLUMNS), language="text")
        st.download_button("Download CSV template", TEMPLATE, "candidates_template.csv", "text/csv")
    up = st.file_uploader("Upload candidates (.csv or .jsonl)", type=["csv", "jsonl", "json", "txt"])
    if up is not None:
        records, total = parse_upload(up.name, up.getvalue())
        if records:
            msg = f"Loaded **{len(records)}** candidates"
            if total > CAP: msg += f" (capped at {CAP} of {total})"
            st.write(msg + ".")
        else:
            st.error("Couldn't read any candidates. Use the CSV template above, or JSONL with one JSON object per line.")
else:
    records = SAMPLE; total = len(SAMPLE)
    st.write(f"Built-in sample: **{len(SAMPLE)}** candidates (includes a keyword-stuffer and a honeypot to show the trap handling).")

if records:
    n = len(records)
    maxk = min(100, n)
    topk = st.slider("How many to rank", 1, maxk, min(10, maxk)) if maxk > 1 else maxk
    if st.button("Rank candidates", type="primary"):
        rows = rank_for_display(records, topk)
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.download_button("Download ranked CSV", to_csv(rows), "ranked.csv", "text/csv")
