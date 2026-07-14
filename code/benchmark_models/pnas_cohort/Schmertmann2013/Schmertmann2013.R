###############################################################################
## Schmertmann (2014) Bayesian cohort-fertility forecast
## "Bayesian Forecasting of Cohort Fertility", JASA 109(506): 500-513
##
## Runnable driver adapted for the deep-fert benchmark suite. It fuses the three
## reference scripts shipped with the paper --
##     Schmertmann_2013_preprocessing.R  (data + shape-penalty calibration)
##     Schmertmann2013_penalties.R       (penalty matrices + weight calibration)
##     Schmertmann2013_forecasts.R       (per-country posterior mean surface)
## -- into one pipeline that produces cohort forecasts in the SAME long format
## and over the SAME jump-off years as the other cohort benchmarks in this repo
## (FreezeRates, Lee1993, Myrskyla2013, deBeer1985and1989).
##
## Input data (follows Schmertmann's description):
##   The model is fed native HFD cohort data in the asfrVV format (asfr/asfrVV.txt)
##   with matching exposures (exposure/exposVV.txt) -- fertility by single age,
##   birth cohort, and country. Age = ARDY, cohort taken straight from the Cohort
##   column (= Year - ARDY). No period (RR) -> cohort approximation is used.
##   Output is keyed to the same 0-based numeric country index the other
##   benchmarks emit (order of first appearance in asfr/asfrRR.txt) so scoring
##   keys line up.
##
## Genuinely-complete-cohort requirement:
##   Schmertmann's prior deliberately leaves the oldest 10 cohorts of each
##   40-cohort surface unconstrained, so the model assumes they are fully
##   observed. We enforce that: a (country, jump-off year) is forecast only if
##   those 10 cohorts are complete in the data (after the paper's narrow
##   ages-15-19 backcast). Combinations without the required history are skipped,
##   NOT backfilled. The model mathematics (penalties, posterior) are unchanged.
##
## Prior calibration:
##   Shape and time-series priors are calibrated on genuinely complete cohorts
##   born in CALIB_COHORTS (1900-1949, as in the original), which the VV data's
##   pre-1950 depth supports for the long-series countries.
###############################################################################

suppressWarnings(suppressMessages({
  required_packages <- c("tidyverse", "here", "MASS", "Matrix")
  install_if_missing <- function(pkg) {
    if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  }
  invisible(lapply(required_packages, install_if_missing))
  invisible(lapply(required_packages, library, character.only = TRUE))
}))
# dplyr::select / filter must win over MASS::select etc.
select <- dplyr::select
filter <- dplyr::filter

path <- here("data")

## ---- model configuration (mirrors the paper) --------------------------------
age1 <- 15
age2 <- 44
age  <- age1:age2
A    <- length(age)          # 30 ages
NC   <- 40                   # cohorts per forecast surface (JOY-54 .. JOY-15)
joys <- c(1985, 1990, 1995, 2000, 2005, 2010)
CALIB_COHORTS <- 1900:1949   # completed historical cohorts used to calibrate priors (as in the original)
NITER <- 30                  # penalty-weight calibration iterations

## =============================================================================
## 1. DATA (native VV) -- rates & exposure as age x cohort, keyed by the numeric
##    country index used across the benchmark suite (0-based, = geos_key order)
## =============================================================================

## --- Code -> numeric country index (order of first appearance in asfrRR.txt).
##     split_period_data.py assigns this index and the other benchmarks emit it,
##     so mapping VV's HFD Codes through it keeps Schmertmann's output keys aligned.
rr_codes <- read.table(file.path(path, "asfr", "asfrRR.txt"),
                       skip = 3, header = FALSE,
                       colClasses = c("character", "NULL", "NULL", "NULL"))[[1]]
code_levels <- unique(rr_codes)                       # order of first appearance
code2idx    <- setNames(seq_along(code_levels) - 1L, code_levels)

## --- rates: native HFD cohort format (asfrVV). Age = ARDY; cohort straight from
##     the Cohort column (= Year - ARDY). Open age/cohort groups ("12-", "1939+")
##     become NA under coercion and are dropped, leaving clean 15-44 cells.
asfr <- read.table(file.path(path, "asfr", "asfrVV.txt"),
                   skip = 2, header = TRUE, as.is = TRUE)
asfr$Age    <- suppressWarnings(as.integer(asfr$ARDY))
asfr$Cohort <- suppressWarnings(as.integer(asfr$Cohort))
asfr$Rate   <- suppressWarnings(as.numeric(asfr$ASFR))
asfr <- asfr[!is.na(asfr$Age) & asfr$Age %in% age &
             !is.na(asfr$Cohort) & !is.na(asfr$Rate), ]
asfr$Country <- code2idx[asfr$Code]
asfr <- asfr[!is.na(asfr$Country), ]

## --- exposure: native HFD cohort format (exposVV), matched the same way
expo <- read.table(file.path(path, "exposure", "exposVV.txt"),
                   skip = 2, header = TRUE, as.is = TRUE)
expo$Age      <- suppressWarnings(as.integer(expo$ARDY))
expo$Cohort   <- suppressWarnings(as.integer(expo$Cohort))
expo$Exposure <- suppressWarnings(as.numeric(expo$Exposure))
expo <- expo[!is.na(expo$Age) & expo$Age %in% age &
             !is.na(expo$Cohort) & !is.na(expo$Exposure), ]
expo$Country <- code2idx[expo$Code]
expo <- expo[!is.na(expo$Country), ]

## --- helper: build an age x cohort matrix from long (Age, Cohort, value) rows
build_mat <- function(df, valcol) {
  cohs <- sort(unique(df$Cohort))
  m <- matrix(NA_real_, nrow = A, ncol = length(cohs),
              dimnames = list(as.character(age), as.character(cohs)))
  m[cbind(match(df$Age, age), match(df$Cohort, cohs))] <- df[[valcol]]
  m
}

countries <- sort(unique(asfr$Country))
rates <- list(); expos <- list()
for (k in countries) {
  kk <- as.character(k)
  rates[[kk]] <- build_mat(asfr[asfr$Country == k, ], "Rate")
  ek <- expo[expo$Country == k, ]
  expos[[kk]] <- if (nrow(ek) > 0) build_mat(ek, "Exposure") else NULL
}

## =============================================================================
## 2. SHAPE-PENALTY CALIBRATION  (X, M, omega, omega.plus)
##    -- straight port of Schmertmann_2013_preprocessing.R lines 172-210,
##       restricted to complete cohorts in CALIB_COHORTS.
## =============================================================================
PHI <- NULL
for (kk in names(rates)) {
  R  <- rates[[kk]]
  ch <- as.numeric(colnames(R))
  keep <- (ch %in% CALIB_COHORTS) & apply(R, 2, function(x) !any(is.na(x)))
  if (any(keep)) PHI <- cbind(PHI, R[, keep, drop = FALSE])
}
stopifnot(!is.null(PHI), ncol(PHI) > A)
message(sprintf("Shape calibration: %d complete cohort schedules", ncol(PHI)))

ss <- svd(PHI)
X  <- ss$u[, 1:3] %*% diag(c(-1, 1, -1))              # top-3 principal components
M  <- diag(A) - X %*% solve(t(X) %*% X) %*% t(X)      # residual projection (30x30)
eps   <- M %*% PHI
omega <- eps %*% t(eps) / ncol(eps)                   # shape-residual covariance

nullity <- qr(X)$rank                                 # = 3
r_om    <- ncol(omega) - nullity                      # = 27 nonzero eigenvalues
ei      <- eigen(omega, symmetric = TRUE)
Ur      <- ei$vec[, 1:r_om]
omega.plus <- Ur %*% diag(1 / ei$val[1:r_om]) %*% t(Ur)   # Moore-Penrose inverse

## =============================================================================
## 3. PENALTY MATRICES (Kj) + WEIGHT CALIBRATION -> K
##    -- port of Schmertmann2013_penalties.R
## =============================================================================
time.weights <- list(
  freeze.slope = c(-4/30, -3/30, -2/30, -1/30, 10/30 + 1, -1),
  freeze.rate  = c(1, -1)
)

## age-specific SDs of freeze-rate / freeze-slope residuals over historical data
result <- list()
for (typ in names(time.weights)) {
  wt <- time.weights[[typ]]; n <- length(wt)
  sd.p <- setNames(rep(NA_real_, A), as.character(age))
  for (a in age) {
    x <- NULL
    for (kk in names(rates)) {
      R  <- rates[[kk]]; ch <- as.numeric(colnames(R))
      y  <- R[as.character(a), ch %in% CALIB_COHORTS]
      y  <- y[is.finite(y)]
      if (length(y) >= n) {
        AA <- matrix(0, length(y) - (n - 1), length(y))
        for (j in n:length(y)) AA[j - n + 1, j - ((n - 1):0)] <- wt
        x <- c(x, AA %*% y)
      }
    }
    sd.p[as.character(a)] <- sd(x, na.rm = TRUE)
  }
  result[[typ]] <- list(sd = sd.p)
}

evec <- function(i, n) matrix(diag(n)[, i], ncol = 1)

Kj <- list(); target <- c(); group <- c()
Mrank <- qr(M)$rank                                   # 27 -> shape penalty target

## shape penalty: one per cohort 11..40 (first 10 cohorts are always complete)
for (j in 11:NC) {
  nm <- paste0("shape", j)
  group[nm] <- "shape"; target[nm] <- Mrank
  MG <- t(evec(j, NC)) %x% M
  Kj[[nm]] <- Matrix(t(MG) %*% omega.plus %*% MG, sparse = TRUE)      # text Eq 10
}

## time-series penalties: one per age for each residual type
for (typ in names(time.weights)) {
  wt <- time.weights[[typ]]; n <- length(wt)
  W   <- matrix(0, NC - (n - 1), NC)
  cmr <- col(W) - row(W)
  for (j in 1:n) W[cmr == (j - 1)] <- wt[j]
  W <- W[nrow(W) - 29:0, ]                            # keep rows for cohorts 11..40
  for (a in age) {
    nm <- paste0(typ, a)
    group[nm] <- typ; target[nm] <- nrow(W)
    ix <- which(age == a)
    s2 <- (result[[typ]]$sd[as.character(a)])^2
    WH <- W %*% (diag(NC) %x% t(evec(ix, A)))
    Kj[[nm]] <- (1 / s2) * Matrix(t(WH) %*% WH, sparse = TRUE)        # text Eq 13/14
  }
}

## iteratively calibrate weights so E[penalty_j] matches its target
tr_prod <- function(Ksp, Kpl) sum(Ksp * Kpl)          # tr(AB)=sum(A*B) for symmetric B
Ktarget <- target[names(Kj)]
w <- setNames(rep(1, length(Kj)), names(Kj))
rankK <- NA_integer_
for (i in 1:NITER) {
  K <- Matrix(0, A * NC, A * NC, sparse = TRUE)
  for (j in names(Kj)) K <- K + w[j] * Kj[[j]]
  Kd <- as.matrix((K + t(K)) / 2)
  ei <- eigen(Kd, symmetric = TRUE)
  if (is.na(rankK)) { rankK <- sum(ei$val > 1e-6); message(sprintf("rank of K = %d", rankK)) }
  Ur    <- ei$vec[, 1:rankK]
  Kplus <- Ur %*% diag(1 / ei$val[1:rankK]) %*% t(Ur)
  Kplus <- (Kplus + t(Kplus)) / 2
  e <- sapply(names(Kj), function(j) tr_prod(Kj[[j]], Kplus))
  w <- w * e / Ktarget
}

## final weighted penalty matrix
K <- Matrix(0, A * NC, A * NC, sparse = TRUE)
for (j in names(Kj)) K <- K + w[j] * Kj[[j]]
K_dense <- as.matrix((K + t(K)) / 2)
message("Penalty matrix K calibrated.")

## =============================================================================
## 4. FORECASTS  -- posterior mean surface per (country, jump-off year)
##    -- port of Schmertmann2013_forecasts.R core (mu only)
## =============================================================================

## long-format helper, identical to the other benchmark drivers
cohort_matrix_to_long <- function(mat, country, joy, method, key) {
  if (is.null(mat) || all(is.na(mat))) return(NULL)
  ages    <- as.numeric(rownames(mat))
  cohorts <- as.numeric(colnames(mat))
  ex <- expand.grid(Age = ages, Year = cohorts)
  ex$Rate        <- as.vector(as.matrix(mat))
  ex$Country     <- country
  ex$JumpOffYear <- joy
  ex$Method      <- method
  ex$Key         <- key
  ex[!is.na(ex$Rate), c("Country", "Year", "Age", "Rate", "JumpOffYear", "Method", "Key")]
}

all_forecasts <- list(); all_obs <- list()
method_name <- "Schmertmann2013"

for (joy in joys) {
  last.coh <- joy - age1
  coh      <- last.coh - (NC - 1):0                   # ascending JOY-54 .. JOY-15

  for (kk in names(rates)) {
    Data <- rates[[kk]]; Expo <- expos[[kk]]
    if (is.null(Expo)) next

    surf  <- matrix(NA_real_, A, NC, dimnames = list(as.character(age), as.character(coh)))
    women <- matrix(NA_real_, A, NC, dimnames = list(as.character(age), as.character(coh)))

    ix <- match(colnames(surf), colnames(Data))
    ok <- which(!is.na(ix)); surf[, ok] <- Data[, ix[ok]]
    ixe <- match(colnames(women), colnames(Expo))
    oke <- which(!is.na(ixe)); women[, oke] <- Expo[, ixe[oke]]

    ## Narrow freeze-rate backcast, exactly as in Schmertmann's forecasts.R:
    ## only ages 15-19 in the first 5 (oldest) cohorts may be gap-filled -- enough
    ## to rescue a near-complete series without materially changing the forecast.
    ## Only this FITTING copy is touched; the emitted truth (surf) stays clean.
    surf_fit  <- surf
    women_fit <- women
    for (r in which(age %in% 15:19)) {
      for (cc in 1:5) {
        if (is.na(surf_fit[r, cc])) {
          nxt <- which(!is.na(surf_fit[r, ])); nxt <- nxt[nxt > cc]
          if (length(nxt)) {
            surf_fit[r, cc]  <- surf_fit[r, min(nxt)]
            women_fit[r, cc] <- women_fit[r, min(nxt)]
          }
        }
      }
    }

    ## Genuinely-complete-cohort requirement: the oldest 10 cohorts (which the
    ## prior does NOT constrain, and which are entirely in the past at the jump-off)
    ## must be fully observed. If any cell is still missing we lack the data the
    ## model assumes -- skip this (country, joy) rather than fabricate it.
    if (any(is.na(surf_fit[, 1:10]))) next

    masked <- outer(age, coh, "+") > joy               # cells in the future at JOY
    vis    <- !is.na(surf_fit) & !is.na(women_fit) & !masked
    if (sum(vis) <= A) next                            # too little data to fit

    ok_fit <- tryCatch({
      viz <- which(as.vector(vis))
      d  <- numeric(A * NC); s0 <- numeric(A * NC)
      d[viz]  <- women_fit[vis] / pmax(surf_fit[vis], 1e-6)   # inverse variances
      s0[viz] <- surf_fit[vis]

      Amat <- K_dense
      diag(Amat) <- diag(Amat) + d                      # t(V) PSIINV V + K
      VPOST <- solve(Amat)
      MPOST <- VPOST %*% (d * s0)                        # posterior mean vector
      mu <- matrix(MPOST, A, NC, dimnames = list(as.character(age), as.character(coh)))
      mu[mu < 0] <- 0                                   # clamp tiny negative posterior rates, as in other benchmarks
      TRUE -> ok; mu
    }, error = function(e) {
      message(sprintf("  skip country %s JOY %d: %s", kk, joy, e$message)); NULL
    })
    if (is.null(ok_fit)) next
    mu <- ok_fit

    ctry <- as.integer(kk)
    key  <- paste0(method_name, "_", ctry)
    fc <- cohort_matrix_to_long(mu,   ctry, joy, method_name, key)
    ob <- cohort_matrix_to_long(surf, ctry, joy, method_name, key)   # RR observed truth
    if (!is.null(fc)) all_forecasts[[length(all_forecasts) + 1]] <- fc
    if (!is.null(ob)) all_obs[[length(all_obs) + 1]]             <- ob
  }
  message(sprintf("Finished JOY %d", joy))
}

## =============================================================================
## 5. WRITE OUTPUT  (distinct filenames -- the existing schmertmann_*_cohort.csv
##    belong to the separate Schmertmann-2003 model)
## =============================================================================
forecasts_df <- bind_rows(all_forecasts)
obs_df       <- bind_rows(all_obs)

write.csv(forecasts_df, file = file.path(path, "schmertmann2013_forecasts_cohort.csv"), row.names = FALSE)
write.csv(obs_df,       file = file.path(path, "schmertmann2013_obs_cohort.csv"),       row.names = FALSE)

message(sprintf("Saved %d forecast rows to schmertmann2013_forecasts_cohort.csv", nrow(forecasts_df)))
message(sprintf("Saved %d observed rows to schmertmann2013_obs_cohort.csv",       nrow(obs_df)))
