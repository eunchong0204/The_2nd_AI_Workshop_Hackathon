"""
Step 2 -- Classify every MedDRA PT in unique_pts.csv into the fixed taxonomy in taxonomy.py
-- one LLM call per batch, keep = (category in taxonomy.KEEP) -- and write the
drug-agnostic PT blocklist that Step 3 consumes:
  pt_types.csv  -> step3 (pt, pt_type, keep)

Resumable: PTs already in pt_types.csv are skipped on re-run.

Run:
    uv run python step2_classify_pts.py                  # full run (25 PTs/call)
    uv run python step2_classify_pts.py --limit 500      # quick test
    uv run python step2_classify_pts.py --batch-size 10  # smaller batches for a small model
"""



###########
# 1. Set-up
###########
from __future__ import annotations

import argparse
import csv

import pandas as pd
from tqdm import tqdm

from utils import config, taxonomy
from utils.llm import chat_json


############
# 2. Helpers
############
def load_pending(limit: int | None) -> list[str]:
    if not config.UNIQUE_PTS_CSV.exists():
        raise SystemExit(f"{config.UNIQUE_PTS_CSV} not found -- run step1_run_ebgm.R first.")
    pts = pd.read_csv(config.UNIQUE_PTS_CSV)["pt"].dropna().astype(str).tolist()
    if limit:
        pts = pts[:limit]
    done: set[str] = set()
    if config.PT_TYPES_CSV.exists():
        done = set(pd.read_csv(config.PT_TYPES_CSV)["pt"].astype(str))
    return [p for p in pts if p not in done]


def classify_batch(pts: list[str], system: str) -> list[dict]:
    data = chat_json(system, taxonomy.classification_user_prompt(pts))
    rows = data.get("results", data) if isinstance(data, dict) else data

    exact, lower = {}, {}
    for r in rows:
        pt = str(r.get("pt", "")).strip()
        cat = str(r.get("pt_type", "")).strip().upper().replace(" ", "_")
        exact[pt] = cat
        lower[pt.lower()] = cat

    out = []
    for pt in pts:
        cat = exact.get(pt) or lower.get(pt.lower()) or ""
        if cat not in taxonomy.CATEGORIES:
            cat = taxonomy.CATCH_ALL
        out.append({"pt": pt, "pt_type": cat, "keep": cat in taxonomy.KEEP})
    return out


def summarize(types: pd.DataFrame) -> None:
    """Report how many PTs survive the blocklist."""
    kept = int(types["keep"].sum())
    print(f"PTs: keep {kept}/{len(types)} (blocklist -> {config.PT_TYPES_CSV})")


#########
# 3. Main
#########
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 2: classify all PTs")
    ap.add_argument("--batch-size", type=int, default=25, help="PTs per model call")
    ap.add_argument("--limit", type=int, default=None, help="only the top-N PTs (quick test)")
    args = ap.parse_args()

    system = taxonomy.classification_system_prompt()
    print(config.describe(),
          f"| {len(taxonomy.CATEGORIES)} categories | keep={sorted(taxonomy.KEEP)}")

    pending = load_pending(args.limit)
    if pending:
        print(f"{len(pending)} PTs to classify in batches of {args.batch_size}")
        config.PT_TYPES_CSV.parent.mkdir(parents=True, exist_ok=True)
        write_header = not config.PT_TYPES_CSV.exists()
        with open(config.PT_TYPES_CSV, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["pt", "pt_type", "keep"])
            if write_header:
                writer.writeheader()
            for i in tqdm(range(0, len(pending), args.batch_size), unit="batch"):
                for row in classify_batch(pending[i : i + args.batch_size], system):
                    writer.writerow(row)
                fh.flush()
    else:
        print("All PTs already classified.")

    df = pd.read_csv(config.PT_TYPES_CSV)
    print("\n" + df["pt_type"].value_counts().to_string())
    summarize(df)


if __name__ == "__main__":
    main()
