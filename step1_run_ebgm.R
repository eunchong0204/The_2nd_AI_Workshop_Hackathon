# Step 1 -- Score every drug-AE pair in drug_reac_pairs.csv with the FDA's MGPS 
# (Multi-item Gamma Poisson Shrinker) model via openEBGM, and derive the EBGM 
# (Empirical Bayes Geometric Mean) signal tables the downstream steps consume:
#   signals.csv     -> steps 2-4  (per-pair EBGM/EB05 survivors: N>=5 & EB05>2)
#   unique_pts.csv  -> step 2     (drug-agnostic PTs appearing in any signal)



###########
# 1. Set-up
###########
library(readr)
library(dplyr)
library(openEBGM)

processed_dir <- "data/processed"

# Read drug_pt pairs data
pairs <- read_csv(file.path(processed_dir, "drug_reac_pairs.csv"), show_col_types = FALSE)

# Data input for openEBGM: one row per (report, drug, event).
data_in <- pairs %>%
  distinct(primaryid, prod_ai, pt) %>%
  transmute(id = primaryid, var1 = prod_ai, var2 = pt)


#############################
# 2. Fit MGPS hyperparameters
#############################
# 2-1. Process the raw counts, then squash to get representative points.
processed <- processRaw(data_in)
squashed <- autoSquash(processed)

# 2-2. Fit the 5 MGPS hyperparameters with hyperEM().
# hyperEM() is more numerically stable than autoHyper() with near-degenerate situation.
theta_init_vec <- c(alpha1 = 0.2, beta1 = 0.1, alpha2 = 2, beta2 = 4, p = 1 / 3)
hyper <- hyperEM(squashed, theta_init_vec = theta_init_vec, squashed = TRUE,
                 method = "score", print_level = 1, track = TRUE)

# Print score_norm and tracking to check the fit.
cat("Print score_norm in hyperEM()\n")
print(hyper$score_norm)
cat("Print tail of tracking in hyperEM()\n")
print(tail(hyper$tracking))


###############################
# 3. Compute EBGM signal scores
###############################
scores <- ebScores(processed, hyper, quantiles = c(5, 95), digits = 2)

ebgm_scores <- scores$data %>%
  rename(prod_ai = var1, pt = var2, n_reports = N) %>%
  arrange(desc(EBGM))
#write_csv(ebgm_scores, file.path(processed_dir, "ebgm_scores.csv"))


###################################
# 4. Filter signals & write outputs
###################################
# 4-1. Filter to the surviving signals, then create signals.csv.
pair_features <- read_csv(file.path(processed_dir, "pair_features.csv"),
                          show_col_types = FALSE) %>%
  select(prod_ai, pt, median_suspects)

# The filtering criteria are:
#   - N >= 5: at least 5 reports
#   - QUANT_05 > 2: 5th percentile of the EBGM score is greater than 2
# This is a heuristic to filter out low-quality signals, and deliberately lenient to evaluate a scoring system.
# For the real situation, the threshold should be adjusted based on the data distribution and the desired sensitivity.
signals <- ebgm_scores %>%
  filter(n_reports >= 5, QUANT_05 > 2) %>%
  left_join(pair_features, by = c("prod_ai", "pt")) %>%
  arrange(desc(EBGM))

n_scored <- nrow(ebgm_scores)
n_eligible <- sum(ebgm_scores$n_reports >= 5)
cat(sprintf("%d signals pass (N >= 5 & EB05 > 2) -- %.2f%% of %d scored pairs, %.2f%% of %d with N >= 5\n",
            nrow(signals), 100 * nrow(signals) / n_scored, n_scored,
            100 * nrow(signals) / n_eligible, n_eligible))
write_csv(signals, file.path(processed_dir, "signals.csv"))

# 4-2. Drug-agnostic list of PTs that appear in any signal -- input to step2.
unique_pts <- signals %>%
  count(pt, name = "n_pairs") %>%
  arrange(desc(n_pairs))
write_csv(unique_pts, file.path(processed_dir, "unique_pts.csv"))
