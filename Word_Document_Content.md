# Delta-Hedging Simulation and PnL Attribution: Where Does the Money Come From in a Hedged Option Book?

## Abstract

This project builds a complete simulation and attribution framework for the most
important question in practical options trading: when a trader sells an option and
delta-hedges it, what determines whether the book makes or loses money? We implement
Black-Scholes pricing and Greeks, a self-financing discrete delta-hedging engine
with explicit cash accounting, and a step-by-step PnL attribution that decomposes
every hedging period into theta, gamma, vega, and delta-mismatch components plus a
small discretization residual. Controlled Monte Carlo experiments on geometric
Brownian motion (GBM) paths confirm the classical theory: hedged PnL is a bet on
realized versus implied variance, hedging frequency controls dispersion but not the
mean, and proportional transaction costs create a breakeven rebalancing frequency
beyond which hedging destroys the volatility edge. A real-data workflow applies the
same machinery to rolling one-month at-the-money straddles on five years of SPY
prices (2020-2025, downloaded via yfinance), where a regression of episode hedged
PnL on the variance risk premium is strongly positive and highly significant (slope
36.4, R-squared 0.42, p below 1e-140 across 1,215 episodes), with 89% of episodes
carrying the theoretically correct PnL sign. All synthetic results are clearly
labeled as simulation output and are never presented as market evidence.

## 1. Introduction and Motivation

The Black-Scholes framework promises that an option can be replicated by a
continuously rebalanced portfolio of the underlying and cash. In practice,
rebalancing is discrete, volatility is unknown at trade time, and every trade costs
money. The gap between the frictionless theory and the implementable strategy is
exactly where option trading PnL lives. Understanding that gap requires three
pieces of machinery that this project develops from scratch: a pricing and Greeks
engine, a hedging simulator with honest cash and stock accounting, and an
attribution scheme that explains, period by period, why the hedged book made or
lost money.

The study centers on a short at-the-money (ATM) one-month straddle, the canonical
volatility-selling trade. Selling a straddle collects theta and pays gamma; the
delta hedge strips out the first-order directional exposure, leaving a position
whose PnL is driven by the difference between realized and implied variance.

## 2. Methodology

### 2.1 Black-Scholes pricing and Greeks

We implement closed-form Black-Scholes values for European calls, puts, and
straddles, together with delta, gamma, theta (per year), and vega (per unit of
absolute volatility). At expiry, values fall back to the intrinsic payoff and all
Greeks are set to zero, since no time value remains to hedge. The implementation is
validated against put-call parity, delta bounds, and finite-difference
approximations of gamma before any experiment runs.

### 2.2 The discrete delta-hedging engine

The hedged book at time t is

    Pi_t = -V(S_t, t) + h_t S_t + B_t

where V is the long straddle value, h_t is the number of shares held, and B_t is a
cash account earning the continuously compounded risk-free rate r. At each
rebalancing date, h is reset to the straddle delta (the position is short the
straddle, so the fair hedge is plus delta shares). Every trade is self-financing:
shares are bought or sold out of the cash account, and proportional transaction
costs of k basis points of trade value are deducted. At expiry the straddle settles
at its payoff and the share position is closed. Terminal PnL is the liquidation
value of the whole portfolio, and it is independently reconstructed as the sum of
all step PnLs minus costs; the two must agree to floating-point precision, which is
our internal accounting consistency check.

### 2.3 PnL attribution

Applying a second-order Taylor expansion of the option value to the step PnL

    dPi = -dV + h dS + r B dt

gives the attribution identity used throughout the project:

    dPi = theta + gamma + vega + delta_mismatch + residual - costs

with theta = -Theta dt (the short book collects time decay), gamma = -0.5 Gamma
dS^2 (the short book pays for convexity), vega = -Vega d_sigma (zero while the
marking vol is held constant), and delta_mismatch = (h - Delta) dS capturing the
stale hedge between rebalancing dates. The residual collects third-order terms and
the interest on the option premium, and it must remain a small share of the gross
components for the attribution to be meaningful.

### 2.4 Monte Carlo experiments on GBM paths

Four controlled experiments run on simulated GBM paths of one-month maturity:

1. Baseline: realized vol equals the hedge vol, hedging every simulated step. The
   mean terminal PnL must be statistically indistinguishable from zero.
2. Volatility mismatch: realized vol varies from 0.10 to 0.30 while the hedge vol
   stays at 0.20. Mean PnL must respond monotonically to the realized minus implied
   variance gap, with the short straddle profiting when realized vol is below
   implied.
3. Hedging frequency: rebalancing intervals range from every step to never. The
   mean PnL stays near zero, but the standard deviation must shrink as frequency
   rises.
4. Transaction costs: with a small volatility edge (realized 0.18 versus implied
   0.20), proportional costs are increased from 0 to 10 basis points and the
   breakeven rebalancing frequency, where mean net PnL crosses zero, is located.

### 2.5 Real-data workflow

In real mode the script loads daily closes from a user CSV (schema: date, close) or
downloads them with yfinance. For each trading day it sells a one-month ATM
straddle struck at that day's close, uses the lagged 21-day realized volatility as
the hedge-volatility proxy (a stand-in for the implied vol quote we would observe
in the market), delta-hedges daily for 21 trading days, and records the terminal
PnL, its attribution, and the forward realized volatility. The result is an
episode-level panel linking hedged PnL to the ex-post variance gap. We ran this
workflow on five years of SPY closes (2020-01-02 to 2024-12-31, 1,258 daily
observations, 1,215 rolling straddle episodes) downloaded live via yfinance, and
regressed the episode hedged PnL on the variance risk premium (implied minus
realized variance) using ordinary least squares.

## 3. Step-by-Step Application

1. Run the sanity suite. The script first checks put-call parity, delta bounds,
   finite-difference gamma, and agreement between the scalar and vectorized hedging
   engines on a common path.
2. Run the synthetic demo (`--mode demo`). This produces four summary CSVs, a
   per-path baseline PnL file, a step-detail file for one illustrative path, four
   figures, and a correctness-check table.
3. Inspect the baseline. Confirm that the mean hedged PnL is within two standard
   errors of zero and that the PnL histogram is centered at zero.
4. Inspect the attribution. Confirm that mean theta and mean gamma nearly cancel at
   matched vol and that the median residual is a small percentage of gross
   component PnL.
5. Inspect the vol-mismatch grid. Confirm the monotone relationship between mean
   PnL and the variance gap, and locate the zero crossing at matched vol.
6. Inspect the frequency study. Confirm that PnL dispersion shrinks with hedging
   frequency while the mean is stable.
7. Inspect the cost study. Read off the breakeven rebalancing frequency at each
   cost level.
8. Run the real-data study (`--mode real --csv your_data.csv` or `--ticker SPY`)
   and interpret the episode panel, in particular the OLS regression of hedged
   PnL on the variance risk premium and the fraction of episodes with the
   theoretically correct PnL sign.

## 4. Output Interpretation

Representative synthetic demo results (1,500 GBM paths per configuration; these are
simulation outputs, not market evidence):

- Baseline at matched vol: straddle premium 4.60, mean hedged PnL -0.007 (about
  -0.15% of premium, within two standard errors of zero), standard deviation 0.31.
- Attribution at matched vol: mean theta +2.30, mean gamma -2.30, median residual
  share 1.7% of gross component PnL, maximum reconstruction error about 1e-13.
- Vol mismatch: mean PnL falls monotonically from +2.29 at realized vol 0.10 to
  -2.30 at realized vol 0.30, crossing zero at the matched point.
- Frequency: PnL standard deviation falls from 3.52 when the hedge is never
  rebalanced to 0.31 at the densest grid, while the mean stays near zero
  throughout.
- Costs: with a 2-vol-point edge, the mean net PnL stays positive at 10 basis
  points only up to roughly 2 rebalances per day and turns negative at 4
  rebalances per day, giving a concrete breakeven frequency.

Real-data results (SPY, 2020-2025, 1,215 rolling one-month straddle episodes, not
simulation output):

- Mean hedged PnL per episode is -0.42 (std 6.80), reflecting that implied
  volatility was on average slightly below subsequently realized volatility over
  this particular five-year window, which included the March 2020 volatility
  spike visible in `fig5_real_rolling_straddle.png`.
- Regressing episode hedged PnL on the variance risk premium (implied squared
  minus realized squared) gives slope 36.4, intercept -0.44, R-squared 0.42, and
  p-value below 1e-140: a strong, highly significant positive relationship.
- 89.1% of episodes have hedged PnL whose sign agrees with the sign of the
  variance risk premium, satisfying the "large majority of episodes" correctness
  requirement; the remaining episodes cluster near a variance risk premium close
  to zero, where sampling noise in the 21-day forward realized vol window
  dominates the small expected PnL.

Interpretation guidance: a loss on a hedged short straddle is not a math error; it
is the expected outcome when realized variance exceeds the implied variance that
was sold. The correct benchmark for a hedged book is zero PnL at matched vol, not
the full premium.

## 5. Validation

The script enforces the following correctness checks automatically and writes them
to `correctness_checks.csv`:

- Continuous or frequent hedging at matched realized and implied vol produces
  near-zero average terminal PnL (two-standard-error test). Passed.
- Attribution components reconcile to total PnL within a small discretization
  residual (median residual share below 5%, reconstruction error at machine
  precision). Passed.
- Mean PnL responds monotonically to realized minus implied variance. Passed.
- PnL standard deviation shrinks with hedging frequency. Passed.
- Costs are nonnegative and create a breakeven hedging frequency. Passed.
- Option values are nonnegative, put-call parity holds, deltas are bounded, and
  the scalar and vectorized engines agree on identical paths. Passed.
- Real-data check (SPY, 2020-2025): the regression of hedged PnL on the variance
  risk premium is positive with p below 0.05, and the PnL sign agrees with the
  variance risk premium sign in more than half of episodes (89.1% observed).
  Written to `outputs_real/real_rolling_straddle_summary.csv` as
  `variance_risk_premium_check_passed`. Passed.

## 6. Underfitting and Overfitting, in Numerical and Financial Terms

Underfitting in this project means using a model or a hedging rule that is too
coarse to capture the risk it claims to control. Hedging too infrequently is the
clearest case: the stale delta between rebalances leaves first-order directional
exposure that the Taylor attribution assigns to the delta-mismatch term, and the
PnL dispersion is then dominated by path noise rather than by the variance view the
trader intended to express. Simulating too few paths is the statistical analogue:
standard errors on mean PnL stay large, and spurious nonzero means survive the
zero-mean test. Ignoring transaction costs is a structural underfit: the model
concludes that more frequent hedging is always better, a conclusion that is false
in any market with frictions.

Overfitting means tuning to the noise of a particular realization or regime.
Choosing a hedge vol or a rebalancing rule because it maximized PnL on one set of
simulated paths, or on one historical window, fits the idiosyncratic path noise
rather than the variance relationship and will not generalize out of sample. The
defenses built into this project are the use of fresh random seeds per experiment,
the two-standard-error test rather than point estimates, and the separation between
the synthetic validation world (where the truth is known by construction) and the
real-data study (where the truth is not).

Model misspecification sits between the two. GBM with constant vol is deliberately
simple: it is the correct data-generating process for validating the engine, but
real underlyings exhibit jumps, stochastic vol, and vol-of-vol, all of which
inflate the attribution residual and the delta-mismatch term relative to what the
demo shows. The real-data workflow treats the observed residual share as a
diagnostic of exactly this misspecification. The regime question is closely
related: a hedging rule calibrated in a low-vol regime under-hedges in a crisis
regime, which is an out-of-sample failure that no amount of in-sample tuning can
fix.

## 7. Limitations

- GBM is an idealized data-generating process; it validates the engine but does not
  represent real return distributions.
- The real-data study uses lagged realized vol as a proxy for implied vol because
  option quote data is not part of the deliverable; absolute PnL levels in real
  mode should therefore be read as illustrative.
- The hedge is delta-only; gamma and vega are unhedged, which is the point of the
  study but is not a complete risk-management policy.
- Discrete hedging of a jumpy underlying can produce losses far larger than the
  GBM experiments suggest.
- Interest accrual uses a constant continuously compounded rate.

## 8. Conclusion

The project delivers a validated, self-financing delta-hedging simulator with
exact PnL attribution. The synthetic experiments reproduce every classical
prediction of delta-hedging theory: replication at matched vol, monotone response
to the variance gap, dispersion control through frequency, and a cost-driven
breakeven rebalancing frequency. The real-data workflow carries the same machinery
to five years of SPY prices, where an OLS regression of episode hedged PnL on the
variance risk premium (slope 36.4, R-squared 0.42, p below 1e-140, 89.1% correct
sign match) provides a clean empirical confirmation of the mechanism outside the
controlled simulation world. The framework is a foundation for extensions to
stochastic volatility, gamma and vega hedging, and band-based rebalancing rules.
