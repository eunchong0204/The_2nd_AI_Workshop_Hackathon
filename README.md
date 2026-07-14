# Novel ADR detection from FAERS

Surfaces **novel, unlabeled adverse drug reactions** by combining classical
pharmacovigilance statistics with an LLM stage:

1. **R / MGPS** — mine FAERS for drug–AE pairs that are reported far more often than
   chance (EBGM disproportionality).
2. **Python / LLM** — throw away the pairs that aren't real clinical events, check each
   survivor against the drug's official FDA label, and rank whatever is left over.

The output is a per-drug shortlist: adverse events that FAERS reports
disproportionately and that the drug's label **does not** mention.

📊 **[Analysis report → `analysis_report.ipynb`](analysis_report.ipynb)** — methods, results, and validation.

---

## 1. Setup

### Requirements

| | |
|---|---|
| R | ≥ 4.4 (developed on 4.5.3) |
| Python | 3.11 (pinned in `.python-version`) |
| Disk | ~5 GB — FAERS zips ~600 MB, derived tables ~1.5 GB, embedding cache ~750 MB |
| RAM | ~8 GB — Step 4 loads `pair_litref.csv` (≈490 MB) fully into memory |
| Keys | an **OpenAI key** (always — see below) and an OpenFDA key (free) |

### 1a. R environment (`renv`)

`renv` pins exact R package versions in `renv.lock`.

```r
renv::restore()         # run inside R, from the project root
```

### 1b. Python environment (`uv`)

```bash
uv sync                 # creates .venv from pyproject.toml + .python-version
```

### 1c. Configuration (`.env`)

```bash
cp .env.example .env     # then edit .env and fill in your keys
```

Edit these:

| variable | what it does |
|---|---|
| `LLM_PROVIDER` | `ollama` (local, free) or `openai`. Used by **both** Step 2 and Step 3. |
| `LLM_MODEL` | override the per-provider default (`llama3.1:8b` / `gpt-4o-mini`) |
| `OPENAI_API_KEY` | **required regardless of `LLM_PROVIDER`** — see the note below |
| `OPENFDA_API_KEY` | free key from [open.fda.gov](https://open.fda.gov/apis/authentication/). Without one you're capped. |

> **The OpenAI key is not optional.** Step 4's embeddings always go to OpenAI, so you need `OPENAI_API_KEY` even when `LLM_PROVIDER=ollama`.

For local runs, install [Ollama](https://ollama.com) and pull the model first:

```bash
ollama pull llama3.1:8b
ollama serve
```

### 1d. Get the data

Download the FAERS quarterly ASCII zips from the
[FDA FAERS page](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html)
and drop them in `data/raw/`, named `faers_ascii_<YYYY>q<Q>.zip` —
for example `data/raw/faers_ascii_2024q1.zip`. Leave them zipped: Step 0 reads
each archive directly.
---

## 2. Usage

Run the steps in order. Each one reads the previous step's output from `data/processed/`.

### Step 0 — build the drug–AE tables (R)

```bash
Rscript step0_process_data.R
```

Stacks every quarter in `data/raw/`, keeps the latest version of each case, filters to
primary/secondary suspect drugs, drops bulk reports (> 10 suspect drugs), and pairs
every suspect drug with every reaction on the report.

→ `drug_reac_pairs.csv` · `pair_features.csv` · `pair_litref.csv`

### Step 1 — EBGM disproportionality (R)

```bash
Rscript step1_run_ebgm.R
```

Fits the MGPS model with `openEBGM` and keeps the pairs that survive **N ≥ 5 and EB05 > 2**.

→ `signals.csv` · `unique_pts.csv`

```bash
Rscript step1_run_sensitivity.R    # optional: fit diagnostic, writes nothing
```

### Step 2 — drop the PTs that aren't clinical events (LLM)

```bash
uv run python step2_classify_pts.py                  # full run
uv run python step2_classify_pts.py --limit 500      # quick test
uv run python step2_classify_pts.py --batch-size 10  # smaller batches for a small model
```

Classifies every PT into one of seven drug-agnostic categories
(`utils/taxonomy.py`). Only `CLINICAL_EVENT` and `LAB_OR_INVESTIGATION` are kept;
lab-context, product complaints, medication errors, and lack-of-efficacy terms are
blocked.

→ `pt_types.csv`

### Step 3 — compare against the FDA label (LLM)

```bash
uv run python step3_compare_to_label.py                       # every drug
uv run python step3_compare_to_label.py --drugs-csv drug_brands.csv
uv run python step3_compare_to_label.py --drug "FEBUXOSTAT" --brand ULORIC
```

Fetches each drug's OpenFDA label and assigns every candidate AE exactly one status:

| status | meaning |
|---|---|
| `disease_related` | the indication, the underlying disease, or it worsening — a confound, not a reaction |
| `already_labeled` | the label describes the drug as causing it |
| `novel` | **neither → a potential unlabeled ADR** |
| `no_label` | no OpenFDA label found; not eligible for the shortlist |

→ `adr_results.csv` (every AE + status)

### Step 4 — score and rank (embeddings)

```bash
uv run python step4_score_pts.py                              # every drug
uv run python step4_score_pts.py --drugs-csv drug_brands.csv
uv run python step4_score_pts.py --drug "FEBUXOSTAT" --brand ULORIC --out my_ranking.csv
```

Scores each candidate AE on five components, each normalised to [0, 1] and combined by a weighted
sum:

| component | what it measures |
|---|---|
| `dispro` | log(EB05) — disproportionality strength |
| `literature` | similarity to papers cited on the reports |
| `severity` | closeness to a fatal outcome |
| `plausibility` | biological relatedness to what the drug already does |
| `low_confounding` | fewer suspect drugs on the report = cleaner attribution |

> **Ranking is per drug.** `final_score` is min-max normalised *within* each drug
> Filter to one drug, then read its ranking.

→ `scored_adrs.csv`

### Reading the results

```python
import pandas as pd

scored = pd.read_csv("data/processed/scored_adrs.csv")

# ranking is per drug -- always filter to ONE drug first
one = scored[scored.prod_ai == "FEBUXOSTAT"].sort_values("rank")

# the shortlist: unlabeled AEs for this drug, best first
one[one.status == "novel"][
    ["rank", "pt", "final_score", "dispro", "literature",
     "severity", "plausibility", "low_confounding"]
].head(10)
```

---

## 3. Folder structure

```
.
├── data/                       # (local) nothing here is in the repo
│   ├── raw/                    # FAERS quarterly zips -- you download these
│   ├── processed/              # every derived table (steps 0-4 write here)
│   ├── labels/                 # OpenFDA label cache, one JSON per drug
│   └── embeddings/             # embedding cache, keyed by model
│
├── step0_process_data.R
├── step1_run_ebgm.R
├── step1_run_sensitivity.R
├── step2_classify_pts.py
├── step3_compare_to_label.py
├── step4_score_pts.py
│
├── utils/
│   ├── config.py
│   ├── llm.py
│   ├── embed.py
│   ├── openfda.py
│   ├── taxonomy.py
│   └── drugs.py
│
│  # ---- R environment (renv) ----
├── renv.lock
├── .Rprofile
├── renv/
│   ├── activate.R
│   ├── settings.json
│   └── .gitignore
│
│  # ---- Python environment (uv) ----
├── pyproject.toml
├── uv.lock
├── .python-version
│
├── drug_brands.csv             # curated drug -> brand map for 100 drug-brand analysis
├── analysis_report.ipynb
└── .env.example
```

---
