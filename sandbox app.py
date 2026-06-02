#!/usr/bin/env python3
"""Sandbox demo for the Redrob ranker (satisfies submission_spec Section 10.5).

Upload a small candidates JSONL (<=100 records, same schema) and the app runs the
exact ranking pipeline from rank.py and returns a ranked CSV. CPU-only, no network.

Run locally:   streamlit run sandbox_app.py
Deploy free:   Streamlit Community Cloud / HuggingFace Spaces (point at this repo).
"""
import io, json, csv
import streamlit as st
import rank as R  # same extract() / rank_feats() that produce the real submission

st.set_page_config(page_title="Redrob Ranker — Sandbox", layout="wide")
st.title("Redrob Ranker — Sandbox")
st.caption("Upload up to 100 candidate records (JSONL, the released schema). "
           "Runs the exact rank.py pipeline — CPU-only, no network — and returns a ranked CSV.")

up = st.file_uploader("candidates.jsonl (<=100 records)", type=["jsonl", "json", "txt"])
topk = st.slider("How many to rank", 5, 100, 20)

if up is not None:
    text = up.read().decode("utf-8")
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not records:  # maybe it's a JSON array
        try:
            records = json.loads(text)
        except Exception:
            st.error("Could not parse — expected JSONL or a JSON array of candidate objects.")
            st.stop()
    records = records[:100]
    st.write(f"Loaded **{len(records)}** candidates.")

    feats = [R.extract(c) for c in records]
    rows = R.rank_feats(feats, topk=min(topk, len(feats)))

    st.subheader("Ranked candidates")
    st.dataframe(
        [{"rank": r, "candidate_id": cid, "score": sc, "reasoning": rsn}
         for (cid, r, sc, rsn) in rows],
        use_container_width=True, hide_index=True,
    )

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["candidate_id", "rank", "score", "reasoning"])
    w.writerows(rows)
    st.download_button("Download ranked CSV", buf.getvalue(),
                       file_name="ranked_sample.csv", mime="text/csv")
else:
    st.info("Tip: take the first 100 lines of candidates.jsonl to try it — "
            "`head -100 candidates.jsonl > sample.jsonl`")
