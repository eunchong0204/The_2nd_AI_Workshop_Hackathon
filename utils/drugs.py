"""
The shared --drug / --brand / --drugs-csv CLI contract for the step 3-4 scripts.

Step 3 and Step 4 select drugs the same way, so the flags, the guards and the
prod_ai,brand CSV loader are defined ONCE here rather.

  --drug "MEDROXYPROGESTERONE ACETATE" [--brand Depo-Provera]  -> one drug
  --drugs-csv drug_brands.csv                                  -> many (prod_ai + optional brand)
  (neither)                                                    -> every drug, ingredient-level

A drug with no brand falls back to its ingredient-level label (the most recent SPL).
Brand strings must match OpenFDA EXACTLY -- use openfda.list_brands() to find them
and openfda.check_brand() to verify them before a real run.
"""



###########
# 1. Set-up
###########
from __future__ import annotations

import argparse

import pandas as pd


###############
# 2. Public API
###############
def add_drug_args(ap: argparse.ArgumentParser) -> None:
    """Add the shared drug-selection flags to a parser."""
    ap.add_argument("--drug", default=None,
                    help='ONE active ingredient; quote if it has spaces, e.g. '
                         '--drug "MEDROXYPROGESTERONE ACETATE"')
    ap.add_argument("--brand", default=None,
                    help="pin --drug to ONE product's label (requires --drug; for many "
                         "drug-brand pairs use --drugs-csv)")
    ap.add_argument("--drugs-csv", dest="drugs_csv", default=None,
                    help="CSV of drugs to run: a 'prod_ai' column (required) + optional "
                         "'brand' column to pin a drug to one product's label")


def load_drug_brands(path: str) -> tuple[list[str], dict[str, str]]:
    """Read a drug CSV: a 'prod_ai' column (required) + optional 'brand' column.

    Returns (drugs, brands) -- the drug list to run, and {UPPER(prod_ai): brand} for
    the rows that give a brand (others fall back to the ingredient-level label).
    """
    df = pd.read_csv(path)
    if "prod_ai" not in df.columns:
        raise SystemExit(f"{path}: needs a 'prod_ai' column (optional 'brand').")
    drugs = [str(d).strip() for d in df["prod_ai"] if str(d).strip()]
    brands: dict[str, str] = {}
    if "brand" in df.columns:
        brands = {str(d).strip().upper(): str(b).strip()
                  for d, b in zip(df["prod_ai"], df["brand"])
                  if str(d).strip() and str(b).strip() and str(b).strip().lower() != "nan"}
    return drugs, brands


def resolve(ap: argparse.ArgumentParser,
            args: argparse.Namespace) -> tuple[list[str] | None, dict[str, str]]:
    """Turn the parsed flags into (drugs, brands).

    `drugs` is None when no selection was given -> "every drug". `brands` maps
    UPPER(prod_ai) -> brand for the drugs pinned to a specific product.
    """
    if args.brand and not args.drug:
        ap.error("--brand requires --drug; for many drug-brand pairs use --drugs-csv")
    if args.drug and args.drugs_csv:
        ap.error("use --drug (one) OR --drugs-csv (many), not both")

    if args.drugs_csv:
        drugs, brands = load_drug_brands(args.drugs_csv)
        if not drugs:
            ap.error(f"{args.drugs_csv}: no drugs found in the 'prod_ai' column")
        return drugs, brands
    if args.drug:
        return [args.drug], ({args.drug.upper(): args.brand} if args.brand else {})
    return None, {}          # no selection -> every drug, ingredient-level
