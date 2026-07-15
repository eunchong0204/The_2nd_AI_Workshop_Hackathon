"""
Step 4 -- Score and rank each drug's Step 3 candidates so you know which NOVEL ADR to look at 
FIRST, reading adr_results.csv and writing the table you review:
  scored_adrs.csv  -> one row per (drug, PT), ranked WITHIN each drug

Each PT is scored on five components, every one normalised to [0, 1] where higher is
better, then combined by a weighted sum (equal weights by default -- DEFAULT_WEIGHTS):

  dispro          log(EB05)                 -- disproportionality signal strength
  literature      max cos(drug->pt, papers) -- semantic support from cited papers
  severity        cos(pt, death anchors)    -- semantic closeness to a fatal event
  plausibility    cos(pt, labeled PTs +     -- biological relatedness to what this
                       indication)             drug is already known to do/treat
  low_confounding 1 - log1p(median_suspects) -- FEWER suspect drugs on the report = cleaner

dispro (EB05) and confounding come from signals.csv (R stage); severity, plausibility
and literature use OpenAI embeddings (utils/embed.py) -- literature over the per-paper
citations in pair_litref.csv. Confounding is the only "more is worse" signal, so it is
inverted into a "low-confounding" score before the sum.

IMPORTANT: ranking is PER DRUG. final_score is min-max normalised within each drug and
`rank` restarts at 1 for every drug, so scores are NOT comparable across drugs -- do
not sort the whole file by final_score. Filter to one drug, then read its ranking.

Each drug is scored against the SAME label Step 3 used: the brand is taken from Step
3's `requested_brand` column (--brand overrides it), and fetch_label's cache is keyed
by (drug, brand), so this costs no extra API call.

Run:
    uv run python step4_score_pts.py                                    # every drug in adr_results
    uv run python step4_score_pts.py --drug "MEDROXYPROGESTERONE ACETATE"  --brand Provera
    uv run python step4_score_pts.py --drugs-csv drug_brands.csv        # a batch of drugs
"""



###########
# 1. Set-up
###########
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import config, openfda
from utils.drugs import add_drug_args, resolve
from utils.embed import embed, unit

# Anchors for the "how close to death" severity axis. A PT's severity is its cosine
# similarity to the NEAREST of these.
SEVERITY_ANCHORS = [
    "death",
    "died",
    "fatal outcome",
    "sudden death",
    "life-threatening event",
]

# Equal weights across all components.
DEFAULT_WEIGHTS = {
    "dispro": 0.20,
    "literature": 0.20,
    "severity": 0.20,
    "plausibility": 0.20,
    "low_confounding": 0.20,
}


##############
# 2. Features
##############
def _minmax(s: pd.Series) -> pd.Series:
    """Scale to [0, 1]. A constant column maps to a neutral 0.5 (no info)."""
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def enrich_from_signals(df: pd.DataFrame, drug: str) -> pd.DataFrame:
    """Left-join the R-stage columns (EB05, suspect count) onto `df` by PT.

    `df` needs at least a `pt` column. Anything already present is overwritten by
    the authoritative signals.csv values.
    """
    sig = pd.read_csv(config.SIGNALS_CSV)
    sig = sig[sig["prod_ai"].astype(str).str.upper() == drug.upper()]
    cols = ["pt", "EBGM", "QUANT_05", "median_suspects"]
    sig = sig[cols].drop_duplicates("pt").set_index("pt")

    out = df.copy()
    joined = sig.reindex(out["pt"].astype(str).values).reset_index(drop=True)
    for c in ("EBGM", "QUANT_05", "median_suspects"):
        out[c] = joined[c].values
    return out


def indication_text(drug: str) -> str:
    """Fetch the drug's INDICATIONS AND USAGE section (for the plausibility anchor)."""
    label = openfda.fetch_label(drug)
    return label.get("sections", {}).get("indications_and_usage", "") or drug


_LITREF_DF: pd.DataFrame | None = None


def _litref_table() -> pd.DataFrame:
    global _LITREF_DF
    if _LITREF_DF is None:
        _LITREF_DF = pd.read_csv(
            config.PROCESSED_DIR / "pair_litref.csv",
            dtype=str, keep_default_na=False,
        )
    return _LITREF_DF


def literature_relevance(drug: str, pts: list[str]) -> tuple[np.ndarray, list[str]]:
    """For each PT, the MAX cosine between the claim "The adverse reaction of
    {drug} is {pt}" and any paper cited for that (drug, pt) pair.

    Returns (sims aligned to `pts`, winning-citation text per PT; "" if none).
    """
    lit = _litref_table()
    lit = lit[lit["prod_ai"].str.upper() == drug.upper()]
    by_pt: dict[str, list[str]] = {}
    for pt, ref in zip(lit["pt"], lit["lit_ref"]):
        ref = ref.strip()
        if ref:
            by_pt.setdefault(pt, []).append(ref)

    q_vecs = unit(embed([f"The adverse reaction of {drug} is {pt}" for pt in pts]))
    all_cites = sorted({c for cites in by_pt.values() for c in cites})
    cite_idx = {c: i for i, c in enumerate(all_cites)}
    cite_vecs = unit(embed(all_cites)) if all_cites else np.zeros((0, q_vecs.shape[1]))

    sims = np.zeros(len(pts))
    winner = [""] * len(pts)
    for i, pt in enumerate(pts):
        cites = by_pt.get(str(pt), [])
        if not cites:
            continue
        cos = cite_vecs[[cite_idx[c] for c in cites]] @ q_vecs[i]
        j = int(cos.argmax())
        sims[i], winner[i] = float(cos[j]), cites[j]
    return sims, winner


###########
# 3. Score
###########
def score_drug(
    df: pd.DataFrame,
    drug: str,
    *,
    brand_name: str | None = None,
    weights: dict[str, float] | None = None,
    use_indication: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Score and rank ONE drug's candidate PTs.

    `df` must have `pt` and `status` columns (status in {NOVEL, ALREADY_LABELED,
    DISEASE_RELATED, ...}); EBGM etc. are pulled from signals.csv regardless.

    Returns (scored_df sorted best-first, summary dict). The scored frame carries
    the raw feature values, the five normalised [0,1] components, `final_score`,
    `rank`, and `percentile` -- all relative to THIS drug only.
    """
    weights = weights or DEFAULT_WEIGHTS
    d = enrich_from_signals(df, drug).reset_index(drop=True)
    pts = d["pt"].astype(str).tolist()

    label = openfda.fetch_label(drug, brand_name)
    meta = openfda.label_meta(label)

    # embed PTs, severity anchors, and the indication once.
    pt_vecs = unit(embed(pts))

    # severity: cosine to the NEAREST death anchor (max over anchors)
    anchor_vecs = unit(embed(SEVERITY_ANCHORS))           # (k, d) unit vectors
    sev = pt_vecs @ anchor_vecs.T                         # (n, k) PT <-> each anchor
    d["sev_sim"] = sev.max(axis=1)                        # nearest-anchor cosine
    d["severity_by"] = [SEVERITY_ANCHORS[i] for i in sev.argmax(axis=1)]  # which anchor won

    # biological plausibility: cosine to the NEAREST anchor (max) in the pool of {all labeled PTs} + {the indication}. 
    labeled = (d["status"].astype(str).str.upper() == "ALREADY_LABELED").to_numpy()
    n_lab = int(labeled.sum())
    if n_lab >= 2:
        sims = pt_vecs @ pt_vecs[labeled].T          # (n, n_lab) cosine PT<->each labeled PT
        lab_idx = np.where(labeled)[0]
        sims[lab_idx, np.arange(n_lab)] = -np.inf    # leave-one-out: drop each self-match
        pool_names = list(d["pt"].astype(str).to_numpy()[labeled])  # column -> labeled PT name
        if use_indication:
            ind = label.get("sections", {}).get("indications_and_usage", "") or drug
            ind_vec = unit(embed([ind]))[0]
            sims = np.hstack([sims, (pt_vecs @ ind_vec)[:, None]])  # indication as one anchor
            pool_names.append("(indication)")
        d["plaus_sim"] = sims.max(axis=1)            # nearest anchor in the pool
        d["plaus_by"] = [pool_names[j] for j in sims.argmax(axis=1)]  # which anchor won
    else:
        d["plaus_sim"] = np.nan  # not enough labeled PTs -> neutral after _minmax
        d["plaus_by"] = ""

    # literature: MAX semantic similarity of the "drug -> pt" claim to any cited paper.
    lit_sim, lit_by = literature_relevance(drug, pts)
    d["lit_sim"] = lit_sim
    d["literature_by"] = lit_by

    # normalise every component to [0, 1], higher = better (WITHIN this drug)
    eb05 = pd.to_numeric(d["QUANT_05"], errors="coerce").fillna(
        pd.to_numeric(d["EBGM"], errors="coerce")
    )
    n_susp = pd.to_numeric(d["median_suspects"], errors="coerce")
    n_susp = n_susp.fillna(n_susp.median())

    d["dispro"] = _minmax(np.log(eb05.clip(lower=1e-6)))
    d["literature"] = _minmax(pd.Series(lit_sim, index=d.index))
    d["severity"] = _minmax(d["sev_sim"])
    d["plausibility"] = _minmax(d["plaus_sim"])
    d["low_confounding"] = 1.0 - _minmax(np.log1p(n_susp))

    comp = ["dispro", "literature", "severity", "plausibility", "low_confounding"]
    d["final_score"] = sum(weights[c] * d[c] for c in comp)

    d = d.sort_values("final_score", ascending=False).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)
    d["percentile"] = 100 * (1 - (d["rank"] - 1) / max(len(d) - 1, 1))
    for k, v in meta.items():           # stamp which label this ranking used
        d[k] = v

    summary = {
        "drug": drug,
        "weights": weights,
        "n_pts": len(d),
        "n_labeled": n_lab,
        "label": meta,
        "mean_score_by_status": d.groupby("status")["final_score"].mean().to_dict(),
    }
    return d, summary


########
# 4. Main
########
def step3_brand(rows: pd.DataFrame) -> str | None:
    """The brand Step 3 pinned this drug to (its `requested_brand`), or None if it
    compared against the ingredient-level label. Reusing it guarantees Step 4 scores
    against the very same SPL Step 3 judged novel-vs-labeled on."""
    if "requested_brand" not in rows.columns:
        return None
    seen = [str(b).strip() for b in rows["requested_brand"].dropna().unique()
            if str(b).strip() and str(b).strip().lower() != "nan"]
    return seen[0] if seen else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 4: score & rank the Step 3 shortlist")
    add_drug_args(ap)                  # --drug / --brand / --drugs-csv (see utils/drugs.py)
    ap.add_argument("--out", default=None,
                    help=f"output CSV (default: {config.SCORED_ADRS_CSV})")
    args = ap.parse_args()

    selected, brands = resolve(ap, args)   # selected=None -> every drug in adr_results

    if not config.ADR_RESULTS_CSV.exists():
        raise SystemExit(f"{config.ADR_RESULTS_CSV} not found -- run Step 3 first.")
    res = pd.read_csv(config.ADR_RESULTS_CSV)

    if selected:
        wanted = {d.upper() for d in selected}
        res = res[res["prod_ai"].astype(str).str.upper().isin(wanted)]
        if res.empty:
            raise SystemExit("none of the selected drugs appear in adr_results.csv")

    out_path = Path(args.out) if args.out else config.SCORED_ADRS_CSV
    print(f"embed={config.EMBED_MODEL} | drugs: {res['prod_ai'].nunique()} "
          f"| rows: {len(res)}")

    frames = []
    for drug, rows in tqdm(res.groupby("prod_ai", sort=False), unit="drug"):
        # An explicit --brand / --drugs-csv brand wins; otherwise reuse Step 3's.
        brand = brands.get(str(drug).upper()) or step3_brand(rows)
        scored, _ = score_drug(rows, str(drug), brand_name=brand)
        frames.append(scored)

    out = pd.concat(frames, ignore_index=True).sort_values(["prod_ai", "rank"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\nWrote {out_path} ({len(out)} rows, {out['prod_ai'].nunique()} drugs)")
    print(out["status"].value_counts().to_string())
    print("\nNOTE: rank/final_score are PER DRUG -- filter to one drug before reading "
          "the ranking.")


if __name__ == "__main__":
    main()
