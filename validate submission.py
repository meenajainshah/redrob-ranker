#!/usr/bin/env python3
"""Validate a submission CSV against the Redrob spec (Section 3). Run before uploading.
Usage: python validate_submission.py submission.csv candidates.jsonl"""
import csv, json, sys

def main(sub, pool):
    rows = list(csv.DictReader(open(sub, encoding="utf-8")))
    errs = []
    if list(rows[0].keys()) != ["candidate_id", "rank", "score", "reasoning"]:
        errs.append(f"header must be candidate_id,rank,score,reasoning (got {list(rows[0].keys())})")
    if len(rows) != 100:
        errs.append(f"must be exactly 100 data rows (got {len(rows)})")
    ranks = [int(r["rank"]) for r in rows]
    if sorted(ranks) != list(range(1, 101)):
        errs.append("ranks must be each integer 1..100 exactly once")
    ids = [r["candidate_id"] for r in rows]
    if len(set(ids)) != len(ids):
        errs.append("duplicate candidate_ids")
    sc = [float(r["score"]) for r in rows]
    if any(sc[i] < sc[i + 1] for i in range(len(sc) - 1)):
        errs.append("score must be non-increasing with rank")
    if len(set(sc)) == 1:
        errs.append("all scores identical — model isn't differentiating")
    # candidate_ids must exist in pool
    valid = set()
    with open(pool, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                valid.add(json.loads(line)["candidate_id"])
    missing = [c for c in ids if c not in valid]
    if missing:
        errs.append(f"{len(missing)} candidate_id(s) not in pool, e.g. {missing[:3]}")
    if errs:
        print("INVALID:"); [print("  -", e) for e in errs]; sys.exit(1)
    print(f"VALID: {len(rows)} rows, ranks 1-100 unique, scores non-increasing, all ids in pool.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "submission.csv",
         sys.argv[2] if len(sys.argv) > 2 else "candidates.jsonl")
