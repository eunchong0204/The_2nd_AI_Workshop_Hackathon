# One-time R environment setup with renv (the R analog of Python's uv).
#
# YOU run this once to capture your exact package versions into renv.lock.
# OTHERS then reproduce your library with a single call: renv::restore()
# (see the "For others" note at the bottom, and SETUP.md).
#
#   Rscript setup_R.R
#
# renv creates a project-local library (renv/library), an renv.lock manifest,
# and an .Rprofile that auto-activates the project every time R starts here --
# so step0_process_data.R / step1_run_ebgm.R transparently use the pinned versions.

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}

# Every package the R steps actually load -- this is the complete set of
# library() calls across step0_process_data.R, step1_run_ebgm.R and
# step1_run_sensitivity.R. Nothing in R plots, so there is no ggplot2 here.
pkgs <- c(
  "readr", "dplyr", "purrr", "stringr", "tidyr",  # data wrangling (tidyr: separate_rows)
  "openEBGM"                                      # MGPS/EBGM signal detection
)

# bare = TRUE: set up the project scaffolding without renv's auto-discovery
# install, so we control exactly what goes in the lockfile below.
renv::init(bare = TRUE)

# Pull the versions already on your machine into the project library, then pin.
renv::install(pkgs)
renv::snapshot(packages = pkgs, prompt = FALSE)

message("\nrenv.lock written. Commit renv.lock + .Rprofile + renv/activate.R so ",
        "others can reproduce with renv::restore().")

# For others (after cloning): open R in this folder and run
#   renv::restore()
# renv reads renv.lock and installs the exact same package versions.
