# Step 1 -- Sensitivity check for step1_run_ebgm.R's MGPS fit: refit the SAME model 
# on only high-volume drugs (>= 10,000 reports) and common PTs (>= 100 reports) 
# to show it converges cleanly (alpha1 off its bound, score_norm -> 0) once the near-null
# low-count noise is removed. Full-FAERS E is unchanged -- diagnostic only, writes
# nothing (prints score_norm + tracking).



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
# 2-1. Process the raw counts. E stays from the FULL table (comparator = all of FAERS).
processed <- processRaw(data_in)

# 2-2. Restrict the FIT to high-volume drugs & common PTs, then squash. Unlike the
# >= 5 filter this is NOT lossless -- it drops pairs that could be signals -- so this
# is a fit diagnostic, not a signal run.
MIN_DRUG_REPORTS <- 10000
MIN_PT_REPORTS   <- 100

eligible_drugs <- data_in %>%
  distinct(id, var1) %>%
  count(var1, name = "n_drug_reports") %>%
  filter(n_drug_reports >= MIN_DRUG_REPORTS) %>%
  pull(var1)

eligible_pts <- data_in %>%
  distinct(id, var2) %>%
  count(var2, name = "n_pt_reports") %>%
  filter(n_pt_reports >= MIN_PT_REPORTS) %>%
  pull(var2)

fit_cells <- processed[
  processed$var1 %in% eligible_drugs & processed$var2 %in% eligible_pts,
]
squashed_fit <- autoSquash(fit_cells)
message("fit cells: ", nrow(fit_cells),
        " (of ", nrow(processed), "); squashed for fit: ", nrow(squashed_fit))

# 2-3. Fit the 5 MGPS hyperparameters with hyperEM().
# On this well-populated subset alpha1 leaves its bound (~0.044) and score_norm -> 0.
theta_init_vec <- c(alpha1 = 0.2, beta1 = 0.1, alpha2 = 2, beta2 = 4, p = 1 / 3)
hyper <- hyperEM(squashed_fit, theta_init_vec = theta_init_vec, squashed = TRUE,
                 method = "score", print_level = 1, track = TRUE)

# Print score_norm and tracking to check the fit.
cat("Print score_norm in hyperEM()\n")
print(hyper$score_norm)
cat("Print tail of tracking in hyperEM()\n")
print(tail(hyper$tracking))
