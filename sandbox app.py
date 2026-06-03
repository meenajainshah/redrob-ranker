#!/usr/bin/env python3
"""Sandbox demo for the Redrob ranker (submission_spec Section 10.5).

Two ways to see it work:
  • Demo pool — a built-in synthetic pool (strong ML engineers, generic software,
    non-engineering roles, keyword-stuffers, services-only careers, honeypots). Ranks
    on click so you can watch real matches surface and traps get buried. No upload.
  • Upload — your own candidates as CSV or JSONL (a small chunk; the full 465 MB
    candidates.jsonl is over Streamlit's limit and is meant for local rank.py, not the web app).

Uses the exact extract()/compute_scores() from rank.py. CPU-only, no network.
Run: streamlit run sandbox_app.py
"""
import io, json, csv, random
import numpy as np
import streamlit as st
import rank as R

CAP = 5000  # processing cap so the demo always runs inside Streamlit's limits

# ============================ synthetic discovery pool ============================
_FIRST = ["Aarav","Diya","Kabir","Ira","Vivaan","Anaya","Reyansh","Myra","Arjun","Sara",
          "Dev","Tara","Rohan","Nisha","Aditya","Meera","Karan","Zoya","Veer","Ishan"]
_PROD = ["ShopWave","Streamly","KartHub","FinML Labs","Nexa","Bolt AI","Loop","Vyom","Terra","Glimpse"]
_SVC  = ["Infosys","TCS","Wipro","Cognizant","Capgemini","Accenture"]
_CITY = ["Pune","Bengaluru","Hyderabad","Noida","Gurugram","Mumbai","Delhi NCR","Chennai"]

def _rec(cid,title,yoe,summary,career,skills,loc,country,sig):
    return {"candidate_id":cid,"profile":{"current_title":title,"years_of_experience":yoe,
        "headline":title,"summary":summary,"current_company":career[0]["company"] if career else "",
        "location":loc,"country":country},"career_history":career,
        "education":[{"start_year":2010,"end_year":2014}],"skills":skills,"redrob_signals":sig}
def _job(t,c,sz,d,m,cur=True,sd="2021-01-01",ed=None):
    return {"title":t,"company":c,"company_size":sz,"description":d,"duration_months":m,
            "start_date":sd,"end_date":ed,"is_current":cur}
def _sk(n,p,m): return {"name":n,"proficiency":p,"duration_months":m,"endorsements":10}
def _sig(rng,active=True,resp=None,open_w=True,reloc=True,gh=50):
    return {"last_active_date":"2026-05-%02d"%rng.randint(1,27) if active else "2025-%02d-15"%rng.randint(9,12),
            "recruiter_response_rate":resp if resp is not None else round(rng.uniform(0.6,0.95),2),
            "interview_completion_rate":round(rng.uniform(0.5,0.9),2),"open_to_work_flag":open_w,
            "willing_to_relocate":reloc,"notice_period_days":rng.choice([15,30,30,60,90]),
            "github_activity_score":gh}

def _strong_ml(i,rng):
    t=rng.choice(["Senior Machine Learning Engineer","Machine Learning Engineer","Applied Scientist","Search Relevance Engineer"])
    yoe=rng.randint(5,9); c=rng.choice(_PROD)
    d=rng.choice([
      "Built the recommendation and ranking stack: embedding-based retrieval with a vector index, hybrid search and a learning-to-rank reranker, with NDCG/MRR offline evaluation, deployed to production at scale.",
      "Owned search relevance: semantic retrieval with sentence embeddings + FAISS, ranking model tuned on NDCG, A/B tested and shipped to millions of users.",
      "Built recsys and personalization: candidate retrieval over a Qdrant vector database, ranking with offline MAP/MRR evaluation, low-latency real-time serving."])
    return _rec(f"CAND_{i:05d}",t,yoe,d,[_job(t,c,rng.choice(["11-50","51-200","201-500"]),d,yoe*12)],
        [_sk("Python","expert",yoe*12),_sk("Information Retrieval","advanced",40),_sk("PyTorch","advanced",36)],
        rng.choice(_CITY),"India",_sig(rng,gh=rng.randint(40,90)))
def _sw_recsys(i,rng):
    yoe=rng.randint(5,8); c=rng.choice(_PROD)
    d="Built the product recommendation pipeline end to end — candidate retrieval and a ranking service deployed to production for real users. Strong Python; moving deeper into ranking and retrieval."
    return _rec(f"CAND_{i:05d}","Backend Engineer",yoe,d,[_job("Backend Engineer",c,"51-200",d,yoe*12)],
        [_sk("Python","advanced",yoe*12),_sk("Backend","advanced",yoe*12)],rng.choice(_CITY),"India",_sig(rng,gh=rng.randint(40,80)))
def _ml_mid(i,rng):
    yoe=rng.randint(3,6); c=rng.choice(_PROD)
    d="Worked on ML models: classification and some NLP; built data pipelines and trained models, a couple shipped to production."
    return _rec(f"CAND_{i:05d}",rng.choice(["Machine Learning Engineer","Data Scientist"]),yoe,d,
        [_job("ML Engineer",c,"201-500",d,yoe*12)],[_sk("Python","advanced",yoe*12),_sk("Machine Learning","intermediate",30)],
        rng.choice(_CITY),"India",_sig(rng,resp=round(rng.uniform(0.4,0.8),2),gh=rng.randint(10,50)))
def _data_sci(i,rng):
    yoe=rng.randint(3,7)
    d="Built dashboards, forecasting and analytics; SQL-heavy reporting with some classification models."
    return _rec(f"CAND_{i:05d}","Data Scientist",yoe,d,[_job("Data Scientist",rng.choice(_PROD),"201-500",d,yoe*12)],
        [_sk("SQL","expert",yoe*12),_sk("Python","advanced",40),_sk("Machine Learning","intermediate",20)],
        rng.choice(_CITY),"India",_sig(rng,resp=round(rng.uniform(0.4,0.7),2),gh=rng.randint(0,30)))
def _sw_generic(i,rng):
    t=rng.choice(["Software Engineer","Full Stack Developer","Backend Engineer","Frontend Engineer","DevOps Engineer"])
    yoe=rng.randint(2,10)
    d="Built and maintained web services and APIs; microservices, CI/CD and cloud deployments. No ML work."
    return _rec(f"CAND_{i:05d}",t,yoe,d,[_job(t,rng.choice(_PROD),"201-500",d,yoe*12)],
        [_sk("Java","advanced",yoe*12),_sk("Python","intermediate",24)],rng.choice(_CITY),"India",_sig(rng,gh=rng.randint(0,40)))
def _non_eng(i,rng):
    t=rng.choice(["HR Manager","Marketing Manager","Sales Executive","Mechanical Engineer","Accountant",
                  "Operations Manager","Business Analyst","Civil Engineer","Content Writer","Project Manager"])
    yoe=rng.randint(2,12)
    d="Managed teams, processes and stakeholders in a non-technical function; reporting and operations."
    return _rec(f"CAND_{i:05d}",t,yoe,d,[_job(t,"GenCorp","501-1000",d,yoe*12)],
        [_sk("Communication","advanced",yoe*12),_sk("Excel","advanced",yoe*12)],rng.choice(_CITY),"India",_sig(rng,gh=0))
def _stuffer(i,rng):
    t=rng.choice(["HR Manager","Marketing Manager","Business Analyst","Operations Manager","Sales Executive"])
    yoe=rng.randint(2,8)
    d="Generalist role across operations and people functions."
    sks=[_sk(s,"advanced",rng.randint(6,18)) for s in ["Machine Learning","Deep Learning","NLP","LLM","Transformers","RAG","Computer Vision","PyTorch","Recommendation"]]
    return _rec(f"CAND_{i:05d}",t,yoe,d,[_job(t,"GenCorp","501-1000",d,yoe*12)],sks,rng.choice(_CITY),"India",_sig(rng,gh=0))
def _services(i,rng):
    yoe=rng.randint(4,12); c=rng.choice(_SVC)
    d="Delivered and maintained enterprise applications for large clients; support and managed services."
    return _rec(f"CAND_{i:05d}","Senior Consultant",yoe,d,
        [_job("Senior Consultant",c,"1000+",d,min(yoe*12,60)),_job("Consultant",rng.choice(_SVC),"1000+","Client support and maintenance.",40,False,"2016-01-01","2019-05-01")],
        [_sk("Java","advanced",yoe*12),_sk("Python","intermediate",18)],rng.choice(_CITY),"India",_sig(rng,resp=round(rng.uniform(0.3,0.6),2)))
def _honeypot(i,rng):
    yoe=rng.randint(3,5); c=rng.choice(_PROD)
    # impossible: a single role far longer than the whole career + 'expert' skills with 0 months
    return _rec(f"CAND_{i:05d}","AI Engineer",yoe,"Expert across the entire AI stack.",
        [_job("AI Engineer",c,"51-200","Led all AI initiatives.",yoe*12+90,True,"2014-01-01")],
        [_sk("Machine Learning","expert",0),_sk("Deep Learning","expert",0),_sk("LLM","expert",0)],
        rng.choice(_CITY),"India",_sig(rng))
def _inactive_strong(i,rng):
    yoe=rng.randint(6,9)
    d="Built ranking and retrieval models with embeddings and vector search, deployed to production."
    return _rec(f"CAND_{i:05d}","Machine Learning Engineer",yoe,d,[_job("ML Engineer","GlobalAI","501-1000",d,yoe*12)],
        [_sk("Python","expert",yoe*12),_sk("Information Retrieval","advanced",40)],"San Francisco","United States",
        _sig(rng,active=False,resp=0.2,open_w=False,reloc=False,gh=30))

_ARCH = [(_strong_ml,5),(_sw_recsys,4),(_ml_mid,7),(_data_sci,9),(_sw_generic,22),
         (_non_eng,30),(_stuffer,9),(_services,9),(_honeypot,3),(_inactive_strong,2)]
def generate_pool(n=600, seed=42):
    rng = random.Random(seed)
    fns = [f for f,w in _ARCH for _ in range(w)]   # weighted bag
    return [rng.choice(fns)(i, rng) for i in range(n)]

# ============================ CSV upload support ============================
CSV_COLUMNS = ["candidate_id","current_title","years_of_experience","summary","skills","location",
               "country","company","company_size","last_active_date","recruiter_response_rate",
               "interview_completion_rate","open_to_work","willing_to_relocate","notice_period_days","github_activity_score"]
def _num(v,d):
    try: return float(v)
    except Exception: return d
def _bool(v,d=False):
    s=str(v).strip().lower(); return s in ("1","true","yes","y","t") if s else d
def csv_row_to_record(row,i):
    g=lambda k,d="":(row.get(k) or d); yoe=_num(g("years_of_experience"),0.0); summary=g("summary")
    skills=[{"name":s.strip(),"proficiency":"advanced","duration_months":int(yoe*12)} for s in g("skills").split(",") if s.strip()]
    return {"candidate_id":g("candidate_id",f"ROW_{i+1}"),
        "profile":{"current_title":g("current_title"),"years_of_experience":yoe,"headline":g("current_title"),
                   "summary":summary,"current_company":g("company"),"location":g("location"),"country":g("country","India")},
        "career_history":[{"title":g("current_title"),"company":g("company"),"company_size":g("company_size","51-200"),
                   "description":summary,"duration_months":int(yoe*12),"start_date":"2020-01-01","end_date":None,"is_current":True}],
        "education":[{"start_year":2012,"end_year":2016}],"skills":skills,
        "redrob_signals":{"last_active_date":g("last_active_date","2026-05-01"),
                   "recruiter_response_rate":_num(g("recruiter_response_rate"),0.7),
                   "interview_completion_rate":_num(g("interview_completion_rate"),0.6),
                   "open_to_work_flag":_bool(g("open_to_work"),True),"willing_to_relocate":_bool(g("willing_to_relocate"),True),
                   "notice_period_days":int(_num(g("notice_period_days"),30)),"github_activity_score":_num(g("github_activity_score"),30)}}
def parse_upload(name,data):
    text=data.decode("utf-8",errors="ignore"); recs=[]
    if name.lower().endswith(".csv"):
        rd=list(csv.DictReader(io.StringIO(text))); total=len(rd)
        for i,row in enumerate(rd[:CAP]): recs.append(csv_row_to_record(row,i))
    else:
        lines=[l for l in text.splitlines() if l.strip()]; total=len(lines)
        if total<=1 and text.strip().startswith("["):
            arr=json.loads(text); total=len(arr); recs=arr[:CAP]
        else:
            for l in lines[:CAP]:
                try: recs.append(json.loads(l))
                except Exception: pass
    return recs,total
TEMPLATE=("candidate_id,current_title,years_of_experience,summary,skills,location,country,recruiter_response_rate,open_to_work,willing_to_relocate\n"
 "C1,Senior ML Engineer,7,Built production retrieval and ranking and a recommendation engine with embeddings and learning-to-rank at a product company,\"Python,Information Retrieval,Recommender Systems\",Pune,India,0.85,true,true\n"
 "C2,HR Manager,4,Handled recruitment payroll and engagement,\"Machine Learning,Deep Learning,NLP,LLM\",Delhi,India,0.6,true,true\n")

# ============================ ranking for display ============================
def rank_for_display(records, topk):
    feats=[R.extract(c) for c in records]
    hv=R.make_vectorizer(); X=hv.transform(f["blob"] for f in feats); qv=hv.transform([R.JD_QUERY])
    cos=np.asarray(X.dot(qv.T).todense()).ravel(); scores=R.compute_scores(feats,cos)
    order=sorted(range(len(feats)),key=lambda i:(-scores[i],-feats[i]["evidence_score"],-cos[i],feats[i]["cid"]))
    top=order[:min(topk,len(feats))]; raw=np.array([scores[i] for i in top],float)
    finite=raw[raw>-1e5]; lo=finite.min() if len(finite) else raw.min(); hi=raw.max()
    norm=np.clip((raw-lo)/(hi-lo) if hi>lo else np.ones_like(raw),0,1); out=np.round(0.40+0.59*norm,4)
    return [{"rank":pos,"candidate_id":feats[i]["cid"],"title":records[i]["profile"]["current_title"],
             "score":f"{out[pos-1]:.4f}","reasoning":R.reasoning(feats[i])} for pos,i in enumerate(top,1)]
def to_csv(rows):
    buf=io.StringIO(); w=csv.writer(buf); w.writerow(["candidate_id","rank","score","reasoning"])
    for r in rows: w.writerow([r["candidate_id"],r["rank"],r["score"],r["reasoning"]])
    return buf.getvalue()

# ============================ UI ============================
st.set_page_config(page_title="Redrob Ranker — Sandbox", layout="wide")
st.title("Redrob Ranker — Sandbox")
st.caption("Ranks candidates against the Senior AI Engineer JD using the exact rank.py pipeline — CPU-only, no network.")

mode = st.radio("Input", ["Demo pool (synthetic)", "Upload CSV / JSONL"], horizontal=True)
records, total = None, 0

if mode.startswith("Demo"):
    pool_n = st.select_slider("Pool size", options=[100,300,600,1000], value=600)
    records = generate_pool(pool_n); total = len(records)
    st.write(f"Synthetic pool of **{pool_n}** candidates — a realistic mix: strong ML engineers, generic software, "
             "non-engineering roles, keyword-stuffers (AI skills, wrong job), services-only careers, and honeypots. "
             "Rank it and watch the real matches surface while the traps stay out of the top.")
else:
    with st.expander("CSV format (columns + template)"):
        st.write("One candidate per row. `summary` carries the career text the model reads; `skills` is comma-separated. "
                 "Missing columns use sensible defaults. The full 465 MB candidates.jsonl won't upload (Streamlit's 200 MB cap) — "
                 "upload a small chunk if you want to try the real data.")
        st.code(", ".join(CSV_COLUMNS), language="text")
        st.download_button("Download CSV template", TEMPLATE, "candidates_template.csv", "text/csv")
    up = st.file_uploader("Upload candidates (.csv or .jsonl)", type=["csv","jsonl","json","txt"])
    if up is not None:
        records, total = parse_upload(up.name, up.getvalue())
        if records:
            msg=f"Loaded **{len(records)}** candidates"
            if total>CAP: msg+=f" (capped at {CAP} of {total})"
            st.write(msg+".")
        else:
            st.error("Couldn't read any candidates. Use the CSV template, or JSONL with one JSON object per line.")

if records:
    n=len(records); maxk=min(100,n)
    topk = st.slider("How many to rank", 1, maxk, min(10,maxk)) if maxk>1 else maxk
    if st.button("Rank candidates", type="primary"):
        rows = rank_for_display(records, topk)
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.download_button("Download ranked CSV", to_csv(rows), "ranked.csv", "text/csv")
