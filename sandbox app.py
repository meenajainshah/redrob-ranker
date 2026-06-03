#!/usr/bin/env python3
"""Sandbox demo for the Redrob ranker (submission_spec Section 10.5).

Self-contained: ranks a built-in synthetic sample on click (no upload needed), and
optionally ranks an uploaded JSONL. Uses the exact extract()/compute_scores() from
rank.py — CPU-only, no network. Run: streamlit run sandbox_app.py
"""
import io, json, csv
import numpy as np
import streamlit as st
import rank as R  # extract(), compute_scores(), make_vectorizer(), JD_QUERY, reasoning()

# --- built-in synthetic sample (no challenge data; shows the trap handling) ---
def _c(cid,title,yoe,head,summ,comp,sz,loc,country,career,skills,sig):
    return {"candidate_id":cid,"profile":{"current_title":title,"years_of_experience":yoe,
        "headline":head,"summary":summ,"current_company":comp,"location":loc,"country":country},
        "career_history":career,"education":[{"start_year":2012,"end_year":2016}],
        "skills":skills,"redrob_signals":sig}
def _j(t,c,sz,d,m,cur=True,sd="2021-01-01",ed=None):
    return {"title":t,"company":c,"company_size":sz,"description":d,"duration_months":m,
            "start_date":sd,"end_date":ed,"is_current":cur}
def _s(n,p,m): return {"name":n,"proficiency":p,"duration_months":m,"endorsements":10}
_S=lambda **k:{"last_active_date":"2026-05-20","recruiter_response_rate":0.8,
    "interview_completion_rate":0.7,"open_to_work_flag":True,"willing_to_relocate":True,
    "notice_period_days":30,"github_activity_score":55,**k}

SAMPLE=[
 _c("CAND_S01","Senior Machine Learning Engineer",7,"ML engineer — search & recommendations",
   "Build production retrieval and ranking systems; shipped semantic search and a recommendation engine to millions of users with learning-to-rank and NDCG/MRR evaluation.",
   "ShopWave","51-200","Pune","India",
   [_j("Senior ML Engineer","ShopWave","51-200","Built the recommendation and ranking stack: embedding-based retrieval with a vector index, hybrid search, learning-to-rank reranker, A/B tests and NDCG/MRR offline evaluation, deployed to production at scale.",40),
    _j("ML Engineer","DataNimbus","201-500","NLP and information retrieval; semantic search with sentence transformers and FAISS in production.",44,False,"2017-05-01","2020-12-01")],
   [_s("Python","expert",84),_s("Information Retrieval","advanced",40),_s("PyTorch","advanced",50)],_S()),
 _c("CAND_S02","Machine Learning Engineer",6,"Recommender systems & vector search",
   "Recommendation systems engineer; recsys and personalization with embeddings and nearest-neighbour retrieval at a product company.",
   "Streamly","11-50","Bengaluru","India",
   [_j("ML Engineer","Streamly","11-50","Shipped the recommendation system: embedding retrieval + Qdrant vector database, ranking model with offline NDCG evaluation, low-latency real-time serving.",60)],
   [_s("Python","expert",72),_s("Recommender Systems","advanced",60)],_S(recruiter_response_rate=0.88)),
 _c("CAND_S04","Backend Engineer",6,"Backend engineer, building toward ML",
   "Backend engineer; recently built a recommendation pipeline for our marketplace; strong Python, moving deeper into ranking and retrieval.",
   "KartHub","51-200","Noida","India",
   [_j("Backend Engineer","KartHub","51-200","Built the product recommendation pipeline end to end: candidate retrieval and a ranking service deployed to production for real users. Heavy Python.",50)],
   [_s("Python","advanced",72)],_S(github_activity_score=70)),
 _c("CAND_S05","Data Scientist",5,"Data scientist — analytics & ML",
   "Dashboards, forecasting and some ML models.","InsightCo","201-500","Mumbai","India",
   [_j("Data Scientist","InsightCo","201-500","Forecasting models and analytics dashboards; some classification models in production.",60)],
   [_s("Python","advanced",60),_s("Machine Learning","intermediate",30)],_S(recruiter_response_rate=0.6,github_activity_score=20)),
 _c("CAND_S06","HR Manager",4,"HR Manager | People Ops",
   "HR manager handling recruitment, payroll and engagement.","PeopleFirst","501-1000","Delhi","India",
   [_j("HR Manager","PeopleFirst","501-1000","End-to-end recruitment, onboarding, payroll and engagement programs.",48)],
   [_s("Machine Learning","advanced",12),_s("Deep Learning","advanced",12),_s("NLP","advanced",12),_s("LLM","expert",6),_s("Transformers","expert",6),_s("RAG","advanced",6),_s("Computer Vision","advanced",6)],_S(github_activity_score=0)),
 _c("CAND_S07","AI Engineer",9,"AI expert","Expert across the entire AI stack.","GhostCorp","51-200","Remote","India",
   [_j("AI Engineer","GhostCorp","51-200","Led all AI initiatives.",130,True,"2015-01-01")],
   [_s("Machine Learning","expert",0),_s("Deep Learning","expert",0),_s("LLM","expert",0)],_S()),
]

def rank_for_display(records, topk):
    feats=[R.extract(c) for c in records]
    hv=R.make_vectorizer(); X=hv.transform(f["blob"] for f in feats); qv=hv.transform([R.JD_QUERY])
    cos=np.asarray(X.dot(qv.T).todense()).ravel()
    scores=R.compute_scores(feats,cos)
    order=sorted(range(len(feats)),key=lambda i:(-scores[i],-feats[i]["evidence_score"],-cos[i],feats[i]["cid"]))
    top=order[:min(topk,len(feats))]
    raw=np.array([scores[i] for i in top],float)
    finite=raw[raw>-1e5]                       # ignore honeypot floor when scaling
    lo=finite.min() if len(finite) else raw.min(); hi=raw.max()
    norm=np.clip((raw-lo)/(hi-lo) if hi>lo else np.ones_like(raw),0,1)
    out=np.round(0.40+0.59*norm,4)
    return [{"rank":pos,"candidate_id":feats[i]["cid"],"score":f"{out[pos-1]:.4f}",
             "reasoning":R.reasoning(feats[i])} for pos,i in enumerate(top,1)]

st.set_page_config(page_title="Redrob Ranker — Sandbox", layout="wide")
st.title("Redrob Ranker — Sandbox")
st.caption("Ranks candidates against the Senior AI Engineer JD using the exact rank.py pipeline — "
           "CPU-only, no network. Try the built-in sample (it includes a keyword-stuffer and a honeypot "
           "to show the trap handling), or upload your own JSONL.")

mode = st.radio("Input", ["Built-in sample", "Upload JSONL (<=100)"], horizontal=True)
records = None
if mode.startswith("Built-in"):
    records = SAMPLE
    st.write(f"Built-in sample: **{len(SAMPLE)}** candidates.")
else:
    up = st.file_uploader("candidates.jsonl", type=["jsonl","json","txt"])
    if up is not None:
        txt = up.read().decode("utf-8")
        records = [json.loads(l) for l in txt.splitlines() if l.strip()]
        if not records:
            try: records = json.loads(txt)
            except Exception:
                st.error("Expected JSONL or a JSON array."); records = None
        if records:
            records = records[:100]; st.write(f"Loaded **{len(records)}** candidates.")

topk = st.slider("How many to rank", 3, 100, min(10, len(records) if records else 10))

if records and st.button("Rank candidates", type="primary"):
    rows = rank_for_display(records, topk)
    st.dataframe(rows, use_container_width=True, hide_index=True)
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in rows:
        w.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])
    st.download_button("Download ranked CSV", buf.getvalue(), "ranked.csv", "text/csv")
