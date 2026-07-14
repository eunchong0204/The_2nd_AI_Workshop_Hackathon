# Step 0 -- Combine FAERS (FDA Adverse Event Reporting System) quarterly raw data 
# in data/raw/faers_ascii_*.zip, filter it, and derive the processed tables the
# downstream steps consume:
#   drug_reac_pairs.csv  -> step 1 (one row per case-drug-AE occurrence)
#   pair_features.csv    -> step 4 (per-pair confoundedness: median suspect-drug count)
#   pair_litref.csv      -> step 4 (per-pair literature citations, for scoring)



###########
# 1. Set-up
###########
library(readr)
library(dplyr)
library(purrr)
library(stringr)
library(tidyr)

data_dir <- "data/raw"
processed_dir <- "data/processed"
dir.create(processed_dir, showWarnings = FALSE, recursive = TRUE)
zip_files <- list.files(data_dir, pattern = "^faers_ascii_.*\\.zip$", full.names = TRUE)


######################
# 2. Import FAERS data
######################
if (length(zip_files) == 0) {
  stop("No faers_ascii_*.zip files found in '", data_dir, "'")
}

table_types <- c("DEMO", "DRUG", "REAC", "OUTC", "THER", "INDI", "RPSR")

# e.g. faers_ascii_2025q3.zip -> 2025Q3
quarter_label <- function(zip_path) {
  m <- str_match(basename(zip_path), "faers_ascii_(\\d{4})[qQ](\\d)")
  paste0(m[, 2], "Q", m[, 3])
}

# FAERS is $-delimited; all columns are read as character.
read_faers_table <- function(zip_path, table_type) {
  entries <- unzip(zip_path, list = TRUE)$Name
  target <- entries[str_detect(entries, paste0("^ASCII/", table_type, "\\d{2}Q\\d\\.txt$"))]

  if (length(target) == 0) {
    message("  [skip] ", table_type, " not found in ", basename(zip_path))
    return(NULL)
  }
  if (length(target) > 1) {
    warning("Multiple ", table_type, " files matched in ", basename(zip_path), "; using the first")
    target <- target[1]
  }

  con <- unz(zip_path, target)
  df <- read_delim(
    con,
    delim = "$",
    col_types = cols(.default = col_character()),
    na = c("", "NA"),
    progress = FALSE
  )
  df$quarter <- quarter_label(zip_path)
  message("  [ok] ", table_type, " ", quarter_label(zip_path), ": ", nrow(df), " rows")
  df
}

# One combined dataframe per table type, stacking all quarters.
faers <- map(table_types, function(tt) {
  message("Combining table: ", tt)
  parts <- map(zip_files, read_faers_table, table_type = tt)
  parts <- compact(parts)
  bind_rows(parts)
}) %>% set_names(str_to_lower(table_types))

# demo, drug, reac, outc, ther, indi, rpsr
invisible(list2env(faers, envir = .GlobalEnv))


###################
# 3. Data filtering
###################
# 3-1. Deduplicate case versions: Keep only the latest caseversion per caseid.
keep_ids <- demo %>%
  mutate(caseversion = as.integer(caseversion)) %>%
  group_by(caseid) %>%
  slice_max(caseversion, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  pull(primaryid)

demo <- demo %>% filter(primaryid %in% keep_ids)
drug <- drug %>% filter(primaryid %in% keep_ids)
reac <- reac %>% filter(primaryid %in% keep_ids)
message(sprintf("caseversion dedup: kept %d latest-version reports", length(keep_ids)))

# 3-2. Keep only PS/SS = primary/secondary suspect and drop rows with a missing prod_ai.
drug <- drug %>% filter(role_cod %in% c("PS", "SS"), !is.na(prod_ai))
message(sprintf("PS/SS + prod_ai filter: kept %d suspect-drug rows", nrow(drug)))

# 3-3. Drop bulk multi-drug reports
# A few reports list dozens of suspect drugs (up to 168 vs a 99th-percentile of
# 9). The case-level join below pairs every suspect drug with every reaction, so
# one such report fabricates many spurious drug-AE co-occurrences. Drop reports
# with > 10 distinct suspect ingredients.
MAX_SUSPECT_DRUGS <- 10
bulk_ids <- drug %>%
  distinct(primaryid, prod_ai) %>%
  count(primaryid, name = "n_suspect") %>%
  filter(n_suspect > MAX_SUSPECT_DRUGS) %>%
  pull(primaryid)

drug <- drug %>% filter(!primaryid %in% bulk_ids)
message(sprintf("bulk-report cap: dropped %d reports with > %d suspect drugs",
                length(bulk_ids), MAX_SUSPECT_DRUGS))


#######################################
# 4. Create tables for downstream steps
#######################################
# 4-1. Create drug_reac_pairs.csv
join_keys <- c("primaryid", "caseid", "quarter")

drug_reac_pairs <- drug %>%
  distinct(primaryid, caseid, quarter, prod_ai) %>%
  inner_join(
    reac %>% distinct(primaryid, caseid, quarter, pt),
    by = join_keys,
    relationship = "many-to-many"
  )

write_csv(drug_reac_pairs, file.path(processed_dir, "drug_reac_pairs.csv"))

# 4-2. Create pair_features.csv
# For the scoring step: median_suspects (confoundedness) is the median number of
# suspect drugs listed on the reports carrying this pair. It counts every distinct
# suspect ingredient on the report INCLUDING this pair's own drug, so the minimum is
# 1 (a report where this drug is the sole suspect), not 0.
report_feats <- drug %>%
  distinct(primaryid, prod_ai) %>%
  count(primaryid, name = "n_suspect")

pair_features <- drug_reac_pairs %>%
  left_join(report_feats, by = "primaryid") %>%
  group_by(prod_ai, pt) %>%
  summarise(
    n_reports       = n(),
    median_suspects = median(n_suspect),
    .groups = "drop"
  )
write_csv(pair_features, file.path(processed_dir, "pair_features.csv"))

# 4-3. create pair_litref.csv
# Individual literature citations per pair, for the embedding/scoring stage.
# A lit_ref field can pack several citations joined by "; ". That separator is
# imperfect -- "; " also appears between authors and around year;volume within one
# citation -- but the downstream score takes a MAX cosine per pair, so mis-split
# fragments (an author name, "110 (6): 311-316.") are harmless: they are
# semantically far from the query and never win. It then drops text-less fragments (no 4+ letter
# word) and dedup per (prod_ai, pt). Only 8.6% of reports carry a lit_ref.
pair_litref <- drug_reac_pairs %>%
  left_join(demo %>% distinct(primaryid, lit_ref), by = "primaryid") %>%
  filter(!is.na(lit_ref)) %>%
  mutate(lit_ref = str_squish(lit_ref)) %>%
  separate_rows(lit_ref, sep = "; ") %>%             # split packed citations
  mutate(lit_ref = str_squish(lit_ref)) %>%          # tidy each fragment
  filter(str_detect(lit_ref, "[A-Za-z]{4,}")) %>%    # drop text-less junk fragments
  distinct(prod_ai, pt, lit_ref)                     # dedup per (drug, pt)
write_csv(pair_litref, file.path(processed_dir, "pair_litref.csv"))
