"""
Step 3 -- Compare each drug's candidate AEs (its signals.csv rows, kept to Step 2's PTs)
against the drug's OpenFDA label and assign every AE one status with the SAME
model used in Step 2, then write the single table Step 4 scores:
  adr_results.csv  -> every AE + status; the validation set

Status is a hierarchical per-AE call: (a) the drug's indication or the disease
itself -- incl. progression / non-response -> DISEASE_RELATED; else (b) described
anywhere on the label -> ALREADY_LABELED; else (c) NOVEL (a potential unlabeled ADR).

Resumable: (prod_ai, pt) pairs already in adr_results.csv are skipped.

Run:
    uv run python step3_compare_to_label.py                 # all drugs, ingredient-level
    uv run python step3_compare_to_label.py --no-blocklist  # configs (1)/(2): no Step 2 filter
    uv run python step3_compare_to_label.py --drug "MEDROXYPROGESTERONE ACETATE" --brand Provera
    uv run python step3_compare_to_label.py --drugs-csv drug_brands.csv  # batch: run the drugs in the file

By default each drug is compared against its ingredient-level OpenFDA label (the most
recent SPL). Give --brand (with --drug), or a 'brand' column in --drugs-csv, to pin a
drug to ONE product's label; the product used is recorded in the requested_brand column.
"""



###########
# 1. Set-up
###########
from __future__ import annotations

import argparse
import csv

import pandas as pd
from tqdm import tqdm

from utils import config, openfda
from utils.drugs import add_drug_args, resolve
from utils.llm import chat_json

VALID_STATUS = {"DISEASE_RELATED", "ALREADY_LABELED", "NOVEL"}
_FIELDS = ["prod_ai", "pt", "n_reports", "EBGM", "QUANT_05", "status", "reasoning",
           "requested_brand", "label_brand", "label_manufacturer", "label_effective_time"]

_SYSTEM = (
    "You are a senior pharmacovigilance expert comparing candidate PTs for one drug "
    "against that drug's official FDA label.\n\n"
    "Definitions:\n"
    "- Adverse Drug Reaction (ADR): a noxious and unintended response for which a "
    "causal relationship with the medicinal product is at least a reasonable "
    "possibility.\n"
    "- MedDRA Preferred Term (PT): one coded medical concept recorded in a FAERS "
    "report.\n\n"
    "Background:\n"
    "The label text is provided as reference material and may include any of the "
    "drug's sections -- indications, boxed warning, adverse reactions (narrative text "
    "and incidence tables), warnings, precautions, and contraindications. Treat the "
    "ENTIRE label as the source of truth; do not limit your search to any single "
    "section.\n\n"
    "Classification rules:\n"
    "For each candidate PT, make a hierarchical decision and assign exactly ONE "
    "status:\n"
    "1. If the PT is really the drug's INDICATION, the underlying disease being "
    "treated, or that disease worsening / the drug failing to work (so it's "
    "confounding or treatment failure, not a reaction) -> DISEASE_RELATED.\n"
    "2. Else if the label describes the PT as something the drug can cause -- a risk, "
    "warning, or adverse experience of taking it -- in ANY section -> ALREADY_LABELED. "
    "A bare mention is NOT enough: a condition named only as a pre-existing risk "
    "factor or a contraindicated patient population (e.g. \"contraindicated in "
    "patients with hepatic impairment\") is not ALREADY_LABELED.\n"
    "3. Else -> NOVEL (a potential unlabeled ADR).\n\n"
    "Respond only with valid JSON in this exact structure:\n"
    '{"results":[{"pt":"<verbatim>","status":"<DISEASE_RELATED|ALREADY_LABELED|NOVEL>",'
    '"reasoning":"<one short sentence>"}]}\n'
    "Do not include Markdown, comments, additional keys, the label text, or any text "
    "outside the JSON."
)


############
# 2. Helpers
############
def load_signals(use_blocklist: bool, drugs: list[str] | None) -> pd.DataFrame:
    if not config.SIGNALS_CSV.exists():
        raise SystemExit(
            f"{config.SIGNALS_CSV} not found -- run step1_run_ebgm.R (it now writes signals.csv)."
        )
    sig = pd.read_csv(config.SIGNALS_CSV)

    if use_blocklist:
        if not config.PT_TYPES_CSV.exists():
            raise SystemExit(
                f"{config.PT_TYPES_CSV} not found -- run Step 2 first, or pass --no-blocklist."
            )
        keep_pts = set(pd.read_csv(config.PT_TYPES_CSV).query("keep")["pt"].astype(str))
        before = len(sig)
        sig = sig[sig["pt"].astype(str).isin(keep_pts)]
        print(f"blocklist: kept {len(sig)}/{before} signals ({len(keep_pts)} clinical PTs)")

    if drugs:
        wanted = {d.upper() for d in drugs}
        sig = sig[sig["prod_ai"].str.upper().isin(wanted)]

    return sig.sort_values("EBGM", ascending=False)


def build_user_prompt(drug: str, label: dict, pts: list[dict]) -> str:
    label_text = openfda.label_to_text(label) or "(no label sections available)"
    pt_lines = "\n".join(f"{i}. {r['pt']}" for i, r in enumerate(pts, 1))
    return (
        f"DRUG (active ingredient): {drug}\n\n"
        f"FDA LABEL:\n{label_text}\n\n"
        f"Classify the following {len(pts)} PTs:\n{pt_lines}"
    )


def _label_meta(label: dict) -> dict:
    """Reference-label provenance stamped on every row -- see openfda.label_meta."""
    return openfda.label_meta(label)


def classify_drug(drug: str, rows: list[dict], batch_size: int,
                  brand: str | None = None) -> list[dict]:
    label = openfda.fetch_label(drug, brand)
    meta = _label_meta(label)   # stamped on every row so the scope is on the record
    meta["requested_brand"] = brand or ""   # product we pinned to ("" = ingredient-level)
    if not label.get("found"):
        # No SPL to compare against -- record so it's accounted for, but these
        # are not eligible for the novel shortlist (can't confirm "unlabeled").
        return [dict(r, status="NO_LABEL", reasoning="no OpenFDA label found", **meta)
                for r in rows]

    results: list[dict] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        data = chat_json(_SYSTEM, build_user_prompt(drug, label, batch), max_tokens=8192)
        # Small models sometimes ignore the schema and return a dict with no
        # "results" key (or a list of bare strings). Default to an empty list and
        # keep only dict entries so a malformed reply degrades to "NOVEL" for the
        # whole batch instead of raising -- Step 3 stays resumable/robust.
        parsed = data.get("results", []) if isinstance(data, dict) else data
        if not isinstance(parsed, list):
            parsed = []
        by_pt = {str(p.get("pt", "")).strip(): p for p in parsed if isinstance(p, dict)}
        for r in batch:
            hit = by_pt.get(r["pt"], {})
            # normalise case/spacing so a lowercase or spaced reply still maps to a
            # canonical status instead of silently collapsing to the fallback.
            status = str(hit.get("status", "NOVEL")).strip().upper().replace(" ", "_")
            if status not in VALID_STATUS:
                status = "NOVEL"
            results.append(dict(r, status=status, reasoning=hit.get("reasoning", ""), **meta))
    return results


########
# 3. Main
########
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 3: per-drug label comparison")
    ap.add_argument("--no-blocklist", action="store_true",
                    help="skip Step 2 filtering (comparison configs 1 and 2)")
    add_drug_args(ap)                  # --drug / --brand / --drugs-csv (see utils/drugs.py)
    ap.add_argument("--batch-size", type=int, default=50, help="AEs per model call")
    ap.add_argument("--max-drugs", type=int, default=None, help="cap number of drugs (smoke test)")
    args = ap.parse_args()

    drugs, brands = resolve(ap, args)   # drugs=None -> every drug, ingredient-level

    print(config.describe(),
          f"| blocklist={not args.no_blocklist}"
          f" | drugs: {len(drugs) if drugs else 'all'} | brand-pinned: {len(brands)}")
    sig = load_signals(use_blocklist=not args.no_blocklist, drugs=drugs)

    # Resume: which (prod_ai, pt) pairs are already written?
    done: set[tuple[str, str]] = set()
    if config.ADR_RESULTS_CSV.exists():
        prev = pd.read_csv(config.ADR_RESULTS_CSV)
        done = set(zip(prev["prod_ai"].astype(str), prev["pt"].astype(str)))

    grouped = list(sig.groupby("prod_ai", sort=False))
    if args.max_drugs:
        grouped = grouped[: args.max_drugs]

    config.ADR_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not config.ADR_RESULTS_CSV.exists()
    with open(config.ADR_RESULTS_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for drug, drug_sig in tqdm(grouped, unit="drug"):
            rows = [
                {k: r[k] for k in ("prod_ai", "pt", "n_reports", "EBGM", "QUANT_05")}
                for _, r in drug_sig.iterrows()
                if (str(r["prod_ai"]), str(r["pt"])) not in done
            ]
            if not rows:
                continue
            brand = brands.get(str(drug).upper())   # None -> ingredient-level label
            for out in classify_drug(str(drug), rows, args.batch_size, brand):
                writer.writerow(out)
            fh.flush()

    res = pd.read_csv(config.ADR_RESULTS_CSV)
    counts = res["status"].value_counts()

    print(f"\nWrote {config.ADR_RESULTS_CSV} ({len(res)} rows)")
    print(counts.to_string())
    print(f"\nNovel candidate ADRs: {int(counts.get('NOVEL', 0))} "
          '(status == "NOVEL", rank by EBGM)')


if __name__ == "__main__":
    main()
