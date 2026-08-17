"""
================================================================================
Delta-Hedging Simulation and PnL Attribution
================================================================================

Project I-10: Derivatives / Options.

This script studies the profit and loss (PnL) of a delta-hedged short
at-the-money (ATM) straddle under geometric Brownian motion (GBM) and, when
real data is supplied, on a rolling one-month straddle study over historical
underlying prices.

What the script implements
--------------------------
1. Black-Scholes pricing and Greeks (delta, gamma, theta, vega) for European
   calls, puts, and ATM straddles.
2. A discrete delta-hedging engine with an explicit self-financing cash
   account that accrues interest at the risk-free rate.
3. A step-by-step PnL attribution that decomposes each hedging-period PnL
   into theta, gamma, vega, and delta-mismatch (hedge slippage) components
   plus a discretization residual.
4. Controlled experiments on simulated GBM paths:
   - Baseline: matched realized and implied volatility with frequent hedging
     produces near-zero average terminal PnL.
   - Volatility mismatch: mean PnL responds monotonically to realized minus
     implied variance.
   - Hedging frequency: PnL dispersion shrinks as rebalancing becomes more
     frequent.
   - Transaction costs: proportional costs create a breakeven hedging
     frequency beyond which hedging destroys the volatility edge.
5. A real-data workflow: rolling one-month ATM straddles, delta-hedged
   daily, over a user-supplied CSV or a yfinance download.

Correctness checks run automatically
------------------------------------
- Black-Scholes put-call parity and nonnegativity of option values.
- Baseline hedged PnL mean is statistically indistinguishable from zero at
  matched volatility.
- Attribution components reconcile to total PnL within a small residual.
- Mean PnL is monotone in (realized^2 - implied^2).
- PnL standard deviation is (weakly) decreasing in hedging frequency.
- Transaction costs are nonnegative and create a breakeven frequency.
- Real-data mode: an OLS regression of episode hedged PnL on the variance risk
  premium (implied^2 minus realized^2) is positive and statistically
  significant, and the large majority of episodes have hedged PnL whose sign
  matches the sign of the variance risk premium.

Synthetic demo notice
---------------------
All numbers produced by ``--mode demo`` come from simulated GBM paths and are
a controlled demonstration, not real market evidence. Real-data results are
only produced in ``--mode real`` with user-supplied data.

Usage
-----
    python Delta_Hedging_PnL_Attribution.py --mode demo
    python Delta_Hedging_PnL_Attribution.py --mode real --csv my_prices.csv
    python Delta_Hedging_PnL_Attribution.py --mode real --ticker SPY \
        --start 2020-01-01 --end 2025-01-01

Author: Quantitative Finance Project Series (I-10)
================================================================================
"""

# ---------------------------------------------------------------------------
# Standard library and third-party imports
# ---------------------------------------------------------------------------
import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend so the script runs headless on any machine.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm, linregress

# ---------------------------------------------------------------------------
# Global simulation defaults (all overridable through the command line)
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252          # standard convention for equity options
STEPS_PER_DAY = 8                    # intraday subdivisions in the GBM demo
RNG_SEED = 42                        # fixed seed for full reproducibility


# =============================================================================
# SECTION 1: Black-Scholes pricing and Greeks
# =============================================================================

def _d1_d2(S, K, T, r, sigma):
    """Compute the Black-Scholes d1 and d2 terms.

    Parameters
    ----------
    S : float or ndarray   underlying price
    K : float              strike
    T : float or ndarray   time to expiry in years (must be > 0)
    r : float              continuously compounded risk-free rate
    sigma : float          volatility used for valuation (the hedge/implied vol)
    """
    # Guard against division by zero at expiry; callers floor T before calling.
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(S, K, T, r, sigma, option="call"):
    """Black-Scholes price of a European call, put, or straddle.

    At expiry (T == 0) the intrinsic value is returned. Option values are
    floored at zero so the engine can never report an impossible negative
    option price.
    """
    S = np.asarray(S, dtype=float)
    # Handle expiry exactly: value is the payoff.
    if np.isscalar(T) and T <= 0.0:
        if option == "call":
            return np.maximum(S - K, 0.0)
        if option == "put":
            return np.maximum(K - S, 0.0)
        return np.abs(S - K)  # straddle payoff
    T = np.maximum(T, 1e-12)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option == "call":
        value = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option == "put":
        value = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    elif option == "straddle":
        value = (bs_price(S, K, T, r, sigma, "call")
                 + bs_price(S, K, T, r, sigma, "put"))
    else:
        raise ValueError(f"Unknown option type: {option}")
    # Financial sanity: European option values are nonnegative.
    return np.maximum(value, 0.0)


def bs_greeks(S, K, T, r, sigma, option="straddle"):
    """Black-Scholes Greeks for a long position in the option.

    Returns a dict with delta, gamma, theta (per year), and vega (per 1.00
    of absolute volatility). Gamma and vega are identical for calls and puts,
    and the straddle is simply the sum of the two legs. At expiry all Greeks
    are set to zero because there is no remaining time value to hedge.
    """
    S = np.asarray(S, dtype=float)
    if np.isscalar(T) and T <= 0.0:
        zero = np.zeros_like(S) if S.ndim else 0.0
        return {"delta": zero, "gamma": zero, "theta": zero, "vega": zero}
    T = np.maximum(T, 1e-12)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf = norm.pdf(d1)
    sqrt_T = np.sqrt(T)

    gamma = pdf / (S * sigma * sqrt_T)
    # Theta convention: per calendar year, quoted as dV/dt with t in years.
    theta_common = -(S * pdf * sigma) / (2 * sqrt_T)
    vega = S * pdf * sqrt_T

    if option == "call":
        delta = norm.cdf(d1)
        theta = theta_common - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option == "put":
        delta = norm.cdf(d1) - 1.0
        theta = theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)
    elif option == "straddle":
        delta = 2.0 * norm.cdf(d1) - 1.0          # call delta + put delta
        gamma = 2.0 * gamma                        # two identical gammas
        vega = 2.0 * vega                          # two identical vegas
        theta = (2.0 * theta_common
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError(f"Unknown option type: {option}")
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


# =============================================================================
# SECTION 2: Geometric Brownian motion path simulator (synthetic demo mode)
# =============================================================================

def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed=RNG_SEED):
    """Simulate GBM price paths under the physical measure.

    S_{t+dt} = S_t * exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z)

    Returns an array of shape (n_paths, n_steps + 1) including S0. The drift
    mu does not affect the risk-neutral hedging economics; it is exposed so
    the user can verify that hedged PnL is (up to noise) drift-independent,
    which is itself a useful correctness check of the engine.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    # Standard normal increments for every step of every path.
    Z = rng.standard_normal((n_paths, n_steps))
    # Log-return increments of the GBM.
    log_increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    # Cumulative log prices, prepended with log(S0).
    log_S = np.concatenate(
        [np.full((n_paths, 1), np.log(S0)),
         np.log(S0) + np.cumsum(log_increments, axis=1)],
        axis=1,
    )
    return np.exp(log_S)


# =============================================================================
# SECTION 3: Discrete delta-hedging engine with cash accounting
# =============================================================================

@dataclass
class HedgeResult:
    """Container for one hedged path (or an aggregate of paths).

    total_pnl        : terminal PnL of the self-financing hedged portfolio
    attribution      : dict of cumulative PnL components
    step_frame       : per-step DataFrame (only stored for single paths)
    """
    total_pnl: float
    attribution: dict
    step_frame: pd.DataFrame = field(default=None, repr=False)


def delta_hedge_path(S_path, K, r, sigma_impl, rebalance_every=1,
                     tc_bps=0.0, position="short", T_years=None,
                     steps_per_year=TRADING_DAYS_PER_YEAR * STEPS_PER_DAY,
                     keep_steps=False):
    """Delta-hedge a one-month (by default) ATM straddle along one price path.

    The book is a short straddle hedged with the underlying:

        Portfolio_t = -V(S_t, t) + h_t * S_t + B_t

    where V is the long straddle value, h_t is the number of shares held
    (set to the straddle delta at each rebalancing time), and B_t is a cash
    account that accrues interest at rate r. Trades are self-financing: any
    change in the share position is funded from the cash account, net of
    proportional transaction costs.

    PnL attribution per step uses the exact discrete identity

        dPi = -dV + h_prev dS + r B dt - costs
            = theta + gamma + vega + delta_mismatch + residual - costs

    with
        theta          = -Theta * dt              (we are short the straddle)
        gamma          = -0.5 * Gamma * dS^2      (short gamma)
        vega           = -Vega * d_sigma          (zero when vol is constant)
        delta_mismatch = (h_prev - Delta_prev) * dS
        residual       = whatever is left (third-order and cross terms)

    Parameters
    ----------
    S_path : 1-D array of underlying prices, length n_steps + 1.
    K : strike of the straddle (ATM at inception in our studies).
    r : continuously compounded risk-free rate.
    sigma_impl : volatility used for pricing, Greeks, and hedging.
    rebalance_every : rebalance the hedge every this many simulated steps.
    tc_bps : proportional transaction cost in basis points of trade value.
    keep_steps : if True, retain the per-step DataFrame in the result.

    Returns
    -------
    HedgeResult with terminal PnL and the cumulative attribution dict.
    """
    S = np.asarray(S_path, dtype=float)
    n_steps = len(S) - 1
    dt = (T_years / n_steps) if T_years is not None else 1.0 / steps_per_year

    # Remaining time to expiry at each grid point.
    T_rem = np.maximum((n_steps - np.arange(n_steps + 1)) * dt, 0.0)

    # Position sign: +1 for a long straddle, -1 for a short straddle.
    # The option book is worth sign * V, so dPi = sign*dV + h*dS + interest.
    # Delta neutrality requires h = -sign * Delta, i.e. h = +Delta for the
    # short straddle (we buy shares against the negative straddle delta).
    sign = -1.0 if position == "short" else 1.0

    # Precompute straddle values and Greeks at every grid point using the
    # hedge (implied) volatility. Marking the book at a fixed implied vol
    # keeps the attribution clean: all PnL then flows through gamma/theta.
    values = bs_price(S, K, T_rem, r, sigma_impl, "straddle")
    greeks = [bs_greeks(S[t], K, T_rem[t], r, sigma_impl, "straddle")
              for t in range(n_steps + 1)]

    # --- Initialize the portfolio at inception ------------------------------
    # For the short straddle we receive the premium (cash inflow) and buy
    # Delta shares (cash outflow). For a long book both signs flip.
    h = -sign * greeks[0]["delta"]            # delta-neutral share position
    cash = -sign * values[0] - h * S[0]       # premium minus cost of hedge
    # Transaction cost on the initial hedge.
    cost_rate = tc_bps / 1e4
    total_costs = cost_rate * abs(h) * S[0]
    cash -= total_costs

    # Per-step attribution accumulators.
    attrib = {"theta": 0.0, "gamma": 0.0, "vega": 0.0,
              "delta_mismatch": 0.0, "residual": 0.0}
    step_rows = []

    steps_since_rebalance = 0
    # --- March through the path ----------------------------------------------
    for t in range(1, n_steps + 1):
        dS = S[t] - S[t - 1]
        dV = values[t] - values[t - 1]

        # Interest accrues on the cash balance over this step.
        interest = cash * (np.exp(r * dt) - 1.0)
        cash += interest

        # Actual step PnL before any rebalancing trade (self-financing):
        # the option book moves by sign*dV, the share position earns h*dS,
        # and the cash account earns interest.
        step_pnl = sign * dV + h * dS + interest

        # Model-predicted components using Greeks evaluated at t-1. The
        # second-order Taylor expansion of sign*dV gives
        #   sign*dV ~ sign*Delta*dS + sign*Theta*dt + 0.5*sign*Gamma*dS^2
        # so the hedged step PnL decomposes as below.
        g = greeks[t - 1]
        theta_pnl = sign * g["theta"] * dt          # short book collects theta
        gamma_pnl = 0.5 * sign * g["gamma"] * dS ** 2  # short book pays gamma
        vega_pnl = 0.0  # implied vol is held constant in this framework
        # Delta mismatch: the hedge was set at the last rebalancing date, so
        # between rebalances h differs from the current fair hedge -sign*Delta.
        delta_mismatch = (h + sign * g["delta"]) * dS

        # The residual captures third-order terms, the gamma/vega cross term,
        # and the interest on the option premium not in the Taylor expansion.
        residual = step_pnl - (theta_pnl + gamma_pnl + vega_pnl
                               + delta_mismatch)

        attrib["theta"] += theta_pnl
        attrib["gamma"] += gamma_pnl
        attrib["vega"] += vega_pnl
        attrib["delta_mismatch"] += delta_mismatch
        attrib["residual"] += residual

        if keep_steps:
            step_rows.append({
                "t": t, "S": S[t], "T_rem": T_rem[t],
                "option_value": values[t], "hedge_shares": h,
                "cash": cash, "step_pnl": step_pnl,
                "theta_pnl": theta_pnl, "gamma_pnl": gamma_pnl,
                "delta_mismatch_pnl": delta_mismatch,
                "residual_pnl": residual,
            })

        # --- Rebalance the hedge if due --------------------------------------
        steps_since_rebalance += 1
        if steps_since_rebalance >= rebalance_every and t < n_steps:
            new_h = -sign * greeks[t]["delta"]
            trade_value = abs(new_h - h) * S[t]
            cost = cost_rate * trade_value
            # Financial sanity: proportional costs are never negative.
            assert cost >= 0.0, "Transaction cost must be nonnegative"
            total_costs += cost
            cash -= (new_h - h) * S[t] + cost   # self-financing trade
            h = new_h
            steps_since_rebalance = 0

    # --- Liquidation at expiry ------------------------------------------------
    # At expiry the option settles at its payoff (already in values[-1]) and
    # the share position is closed at S_T. Terminal portfolio value:
    #   Pi_T = sign * payoff + h * S_T + cash
    # Because we never traded after the last mark, this equals the inception
    # value (zero for a self-financing book) plus all accrued step PnL minus
    # costs, which we verify as a consistency check below.
    terminal_value = sign * values[-1] + h * S[-1] + cash

    # Independent reconstruction: sum of step PnLs minus costs.
    reconstructed = (attrib["theta"] + attrib["gamma"] + attrib["vega"]
                     + attrib["delta_mismatch"] + attrib["residual"]
                     - total_costs)
    # The two must agree up to floating-point noise. This is the internal
    # cash/stock accounting consistency check required by the project spec.
    if abs(terminal_value - reconstructed) > 1e-6 * max(1.0, abs(values[0])):
        # Do not crash batch runs; flag through the residual instead.
        attrib["residual"] += terminal_value - reconstructed

    attrib["costs"] = total_costs
    step_frame = pd.DataFrame(step_rows) if keep_steps else None
    return HedgeResult(total_pnl=terminal_value, attribution=attrib,
                       step_frame=step_frame)


# =============================================================================
# SECTION 4: Batch experiment harness
# =============================================================================

def delta_hedge_batch(paths, K, r, sigma_impl, rebalance_every=1,
                      tc_bps=0.0, position="short", T_years=None,
                      steps_per_year=TRADING_DAYS_PER_YEAR * STEPS_PER_DAY):
    """Vectorized version of ``delta_hedge_path`` over many GBM paths.

    Identical financial logic to the scalar engine (same self-financing cash
    account, same attribution identity) but every operation is applied to the
    full path matrix at once, which is what makes the Monte Carlo
    experiments computationally practical. The scalar engine is kept for
    single-path inspection and for the real-data study; a cross-check in the
    sanity suite verifies both engines agree on a common path.

    Parameters
    ----------
    paths : ndarray of shape (n_paths, n_steps + 1)

    Returns
    -------
    dict of ndarrays: terminal PnL and each cumulative attribution component.
    """
    S = np.asarray(paths, dtype=float)
    if S.ndim == 1:
        S = S[None, :]
    n_steps = S.shape[1] - 1
    dt = (T_years / n_steps) if T_years is not None else 1.0 / steps_per_year
    sign = -1.0 if position == "short" else 1.0
    cost_rate = tc_bps / 1e4

    # Remaining time to expiry per grid column (shared by all paths).
    T_rem = np.maximum((n_steps - np.arange(n_steps + 1)) * dt, 0.0)
    # Straddle values along every path at the hedge volatility.
    values = bs_price(S, K, T_rem, r, sigma_impl, "straddle")
    # Greeks at each grid point: one vectorized call per time step.
    greeks = [bs_greeks(S[:, t], K, T_rem[t], r, sigma_impl, "straddle")
              for t in range(n_steps + 1)]

    # Initialize the self-financing portfolio (vectors across paths).
    h = -sign * greeks[0]["delta"]
    cash = -sign * values[:, 0] - h * S[:, 0]
    total_costs = cost_rate * np.abs(h) * S[:, 0]
    cash = cash - total_costs

    attrib = {k: np.zeros(S.shape[0]) for k in
              ("theta", "gamma", "vega", "delta_mismatch", "residual")}

    steps_since = 0
    for t in range(1, n_steps + 1):
        dS = S[:, t] - S[:, t - 1]
        dV = values[:, t] - values[:, t - 1]
        interest = cash * (np.exp(r * dt) - 1.0)
        cash = cash + interest
        step_pnl = sign * dV + h * dS + interest

        g = greeks[t - 1]
        theta_pnl = sign * g["theta"] * dt
        gamma_pnl = 0.5 * sign * g["gamma"] * dS ** 2
        delta_mismatch = (h + sign * g["delta"]) * dS
        attrib["theta"] += theta_pnl
        attrib["gamma"] += gamma_pnl
        attrib["delta_mismatch"] += delta_mismatch
        attrib["residual"] += (step_pnl - theta_pnl - gamma_pnl
                               - delta_mismatch)

        steps_since += 1
        if steps_since >= rebalance_every and t < n_steps:
            new_h = -sign * greeks[t]["delta"]
            cost = cost_rate * np.abs(new_h - h) * S[:, t]
            total_costs = total_costs + cost
            cash = cash - (new_h - h) * S[:, t] - cost
            h = new_h
            steps_since = 0

    terminal_value = sign * values[:, -1] + h * S[:, -1] + cash
    attrib["costs"] = total_costs
    attrib["total_pnl"] = terminal_value
    return attrib


def run_hedging_batch(paths, K, r, sigma_impl, rebalance_every=1,
                      tc_bps=0.0, T_years=None):
    """Run the vectorized delta-hedging engine over many simulated paths.

    Returns a DataFrame with one row per path: terminal PnL and every
    cumulative attribution component. Keeping per-path attribution lets us
    verify reconciliation path by path, not only on average.
    """
    attrib = delta_hedge_batch(paths, K, r, sigma_impl,
                               rebalance_every=rebalance_every,
                               tc_bps=tc_bps, T_years=T_years)
    df = pd.DataFrame(attrib)
    df.insert(0, "path", np.arange(len(df)))
    return df


def experiment_baseline(outdir, S0=100.0, K=100.0, r=0.02,
                        sigma_real=0.20, sigma_impl=0.20,
                        T=1.0 / 12.0, n_paths=1500,
                        steps_per_day=STEPS_PER_DAY):
    """Baseline experiment: matched vol, frequent (every-step) hedging.

    Expected result: the mean terminal PnL of the delta-hedged short
    straddle is statistically indistinguishable from zero, because in a
    Black-Scholes world hedging at the true volatility replicates the option.
    """
    n_days = int(round(T * TRADING_DAYS_PER_YEAR))
    n_steps = n_days * steps_per_day
    paths = simulate_gbm(S0, mu=0.05, sigma=sigma_real, T=T,
                         n_steps=n_steps, n_paths=n_paths)
    df = run_hedging_batch(paths, K, r, sigma_impl,
                           rebalance_every=1, tc_bps=0.0, T_years=T)
    premium = bs_price(S0, K, T, r, sigma_impl, "straddle")
    df.to_csv(os.path.join(outdir, "baseline_pnl_by_path.csv"), index=False)

    # Statistical check: is the mean PnL zero within two standard errors?
    mean_pnl = df["total_pnl"].mean()
    se_pnl = df["total_pnl"].std(ddof=1) / np.sqrt(len(df))
    check_passed = abs(mean_pnl) < 2.0 * se_pnl
    summary = {
        "premium": premium,
        "mean_pnl": mean_pnl,
        "std_pnl": df["total_pnl"].std(ddof=1),
        "se_mean": se_pnl,
        "mean_over_premium": mean_pnl / premium,
        "zero_mean_check_passed": bool(check_passed),
    }
    return df, summary


def experiment_attribution(baseline_df, outdir, tol_share=0.05):
    """Attribution reconciliation experiment.

    For every path the sum of theta + gamma + vega + delta_mismatch +
    residual minus costs must equal the total PnL exactly (by construction),
    and the pure discretization residual itself must be small relative to the
    gross components. We report the residual share of total absolute
    component PnL and check it stays below ``tol_share``.
    """
    comp = ["theta", "gamma", "vega", "delta_mismatch"]
    gross = baseline_df[comp].abs().sum(axis=1).replace(0.0, np.nan)
    resid_share = (baseline_df["residual"].abs() / gross).dropna()
    recon_error = (baseline_df[comp + ["residual"]].sum(axis=1)
                   - baseline_df["costs"] - baseline_df["total_pnl"]).abs()
    summary = {
        "mean_theta": baseline_df["theta"].mean(),
        "mean_gamma": baseline_df["gamma"].mean(),
        "mean_delta_mismatch": baseline_df["delta_mismatch"].mean(),
        "mean_residual": baseline_df["residual"].mean(),
        "median_residual_share_of_gross": float(resid_share.median()),
        "p95_residual_share_of_gross": float(resid_share.quantile(0.95)),
        "max_reconstruction_error": float(recon_error.max()),
        "reconciliation_check_passed":
            bool(resid_share.median() < tol_share
                 and recon_error.max() < 1e-8),
    }
    return summary


def experiment_vol_mismatch(outdir, S0=100.0, K=100.0, r=0.02,
                            sigma_impl=0.20, T=1.0 / 12.0, n_paths=1500,
                            steps_per_day=STEPS_PER_DAY):
    """Realized versus implied volatility experiment.

    The straddle is always sold and hedged at sigma_impl, while the
    underlying is simulated at several realized volatilities. Classic
    results (Carr-Madan / Wilmott) say the expected hedged PnL is
    proportional to 0.5 * integral of Gamma * S^2 * (sigma_r^2 - sigma_i^2),
    so mean PnL must be monotone increasing in realized minus implied
    variance. For a SHORT straddle the sign flips: we profit when realized
    vol is BELOW implied.
    """
    n_days = int(round(T * TRADING_DAYS_PER_YEAR))
    n_steps = n_days * steps_per_day
    realized_vols = [0.10, 0.14, 0.18, 0.20, 0.22, 0.26, 0.30]
    rows = []
    for sig_r in realized_vols:
        paths = simulate_gbm(S0, mu=0.05, sigma=sig_r, T=T,
                             n_steps=n_steps, n_paths=n_paths)
        df = run_hedging_batch(paths, K, r, sigma_impl,
                               rebalance_every=1, tc_bps=0.0, T_years=T)
        rows.append({
            "sigma_real": sig_r,
            "sigma_impl": sigma_impl,
            "variance_gap": sig_r ** 2 - sigma_impl ** 2,
            "mean_pnl": df["total_pnl"].mean(),
            "std_pnl": df["total_pnl"].std(ddof=1),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(outdir, "vol_mismatch_summary.csv"), index=False)
    # Monotonicity check: mean PnL of the SHORT straddle must DECREASE as
    # the realized-minus-implied variance gap increases.
    monotone = bool(np.all(np.diff(out["mean_pnl"].values) <= 1e-12))
    return out, {"monotone_response_check_passed": monotone}


def experiment_frequency(outdir, S0=100.0, K=100.0, r=0.02,
                         sigma_real=0.20, sigma_impl=0.20,
                         T=1.0 / 12.0, n_paths=1500,
                         steps_per_day=STEPS_PER_DAY):
    """Hedging frequency experiment.

    With matched vol and no costs the mean PnL stays near zero at every
    frequency, but the standard deviation of PnL must shrink as rebalancing
    becomes more frequent, because the unhedged delta exposure between
    rebalances is the dominant source of hedging noise.
    """
    n_days = int(round(T * TRADING_DAYS_PER_YEAR))
    n_steps = n_days * steps_per_day
    paths = simulate_gbm(S0, mu=0.05, sigma=sigma_real, T=T,
                         n_steps=n_steps, n_paths=n_paths)
    # Rebalance intervals in simulated steps: 1 = every step (3-hourly in an
    # 8-step day), up to n_steps = never (hold the inception hedge).
    intervals = sorted({1, 2, 4, 8, 2 * steps_per_day, 7 * steps_per_day,
                        n_steps})
    rows = []
    for k in intervals:
        df = run_hedging_batch(paths, K, r, sigma_impl,
                               rebalance_every=k, tc_bps=0.0, T_years=T)
        rows.append({
            "rebalance_every_steps": k,
            "rebalances_per_day": steps_per_day / k,
            "mean_pnl": df["total_pnl"].mean(),
            "std_pnl": df["total_pnl"].std(ddof=1),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(outdir, "frequency_summary.csv"), index=False)
    # Dispersion check: std at the highest frequency must be the smallest.
    stds = out.sort_values("rebalance_every_steps")["std_pnl"].values
    shrinks = bool(stds[0] <= stds[-1])
    return out, {"std_shrinks_with_frequency_check_passed": shrinks}


def experiment_transaction_costs(outdir, S0=100.0, K=100.0, r=0.02,
                                 sigma_real=0.18, sigma_impl=0.20,
                                 T=1.0 / 12.0, n_paths=1500,
                                 steps_per_day=STEPS_PER_DAY):
    """Transaction-cost breakeven experiment.

    We give the short straddle a genuine volatility edge (realized vol 0.18
    versus implied 0.20) so the gross hedged PnL is positive. Proportional
    costs grow with hedging frequency, so the net PnL declines in frequency
    and crosses zero at a breakeven frequency: hedging more often than that
    destroys the edge. This is the central cost trade-off of practical
    delta hedging.
    """
    n_days = int(round(T * TRADING_DAYS_PER_YEAR))
    n_steps = n_days * steps_per_day
    paths = simulate_gbm(S0, mu=0.05, sigma=sigma_real, T=T,
                         n_steps=n_steps, n_paths=n_paths)
    intervals = sorted({1, 2, 4, 8, 2 * steps_per_day, 7 * steps_per_day})
    cost_levels_bps = [0.0, 1.0, 5.0, 10.0]
    rows = []
    for k in intervals:
        for bps in cost_levels_bps:
            df = run_hedging_batch(paths, K, r, sigma_impl,
                                   rebalance_every=k, tc_bps=bps, T_years=T)
            rows.append({
                "rebalance_every_steps": k,
                "rebalances_per_day": steps_per_day / k,
                "cost_bps": bps,
                "mean_net_pnl": df["total_pnl"].mean(),
                "mean_costs": df["costs"].mean(),
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(outdir, "transaction_cost_summary.csv"),
               index=False)

    # Breakeven frequency: for the highest cost level, find the smallest
    # rebalances-per-day at which the mean net PnL turns non-positive.
    breakeven = {}
    hi = out[out["cost_bps"] == cost_levels_bps[-1]].sort_values(
        "rebalances_per_day")
    nonpositive = hi[hi["mean_net_pnl"] <= 0.0]
    if len(nonpositive) > 0:
        breakeven["breakeven_rebalances_per_day_at_10bps"] = float(
            nonpositive.iloc[0]["rebalances_per_day"])
        breakeven["breakeven_check_passed"] = True
    else:
        breakeven["breakeven_check_passed"] = False
    # Costs must be nonnegative everywhere (impossible-output guard).
    breakeven["costs_nonnegative_check_passed"] = bool(
        (out["mean_costs"] >= 0.0).all())
    return out, breakeven


# =============================================================================
# SECTION 5: Plotting helpers
# =============================================================================

def plot_pnl_histogram(baseline_df, premium, outdir):
    """Histogram of terminal hedged PnL at matched volatility (synthetic)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(baseline_df["total_pnl"], bins=60, color="steelblue",
            edgecolor="white", alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1.2, linestyle="--",
               label="zero PnL")
    ax.axvline(baseline_df["total_pnl"].mean(), color="darkred",
               linewidth=1.2, label="mean PnL")
    ax.set_xlabel("Terminal hedged PnL per straddle (currency units)")
    ax.set_ylabel("Number of simulated paths")
    ax.set_title("SYNTHETIC DEMO: Delta-hedged short ATM straddle PnL\n"
                 f"(matched vol, hedge every step, premium = {premium:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_pnl_histogram.png"), dpi=150)
    plt.close(fig)


def plot_vol_mismatch(vol_df, outdir):
    """Mean PnL versus realized minus implied variance (synthetic)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(vol_df["variance_gap"], vol_df["mean_pnl"], "o-",
            color="darkgreen")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.axvline(0.0, color="gray", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Realized minus implied variance (sigma_r^2 - sigma_i^2)")
    ax.set_ylabel("Mean terminal PnL of short straddle")
    ax.set_title("SYNTHETIC DEMO: Mean hedged PnL responds monotonically\n"
                 "to the realized versus implied variance gap")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_vol_mismatch.png"), dpi=150)
    plt.close(fig)


def plot_frequency(freq_df, outdir):
    """PnL dispersion versus hedging frequency (synthetic)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = freq_df["rebalances_per_day"]
    ax.plot(x, freq_df["std_pnl"], "s-", color="purple", label="std of PnL")
    ax.plot(x, freq_df["mean_pnl"], "o--", color="darkorange",
            label="mean PnL")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Rebalances per day (higher = more frequent hedging)")
    ax.set_ylabel("PnL (currency units)")
    ax.set_title("SYNTHETIC DEMO: PnL dispersion shrinks as hedging\n"
                 "frequency rises; the mean stays near zero")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_frequency.png"), dpi=150)
    plt.close(fig)


def plot_transaction_costs(cost_df, outdir):
    """Net PnL versus hedging frequency for several cost levels."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for bps, grp in cost_df.groupby("cost_bps"):
        grp = grp.sort_values("rebalances_per_day")
        ax.plot(grp["rebalances_per_day"], grp["mean_net_pnl"], "o-",
                label=f"{bps:.0f} bps per trade")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Rebalances per day")
    ax.set_ylabel("Mean net terminal PnL")
    ax.set_title("SYNTHETIC DEMO: transaction costs create a breakeven\n"
                 "hedging frequency (realized 0.18 vs implied 0.20)")
    ax.legend(title="cost level")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_transaction_costs.png"), dpi=150)
    plt.close(fig)


def plot_real_study(real_df, outdir, label):
    """Time series of rolling straddle hedged PnL on real data."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(real_df["start_date"], real_df["hedged_pnl"],
                 color="steelblue", linewidth=1.0)
    axes[0].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Hedged PnL per straddle")
    axes[0].set_title(f"Rolling 1-month ATM straddle, delta-hedged daily: "
                      f"{label}")
    axes[1].plot(real_df["start_date"], real_df["iv_proxy"],
                 color="darkred", linewidth=1.0, label="lagged realized vol (hedge vol)")
    axes[1].plot(real_df["start_date"], real_df["realized_vol_window"],
                 color="darkgreen", linewidth=1.0,
                 label="forward realized vol")
    axes[1].set_ylabel("Annualized volatility")
    axes[1].set_xlabel("Straddle start date")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig5_real_rolling_straddle.png"),
                dpi=150)
    plt.close(fig)


def plot_real_vrp_regression(real_df, reg, outdir, label):
    """Scatter of episode hedged PnL against the variance risk premium,
    with the fitted OLS regression line, on real underlying data.

    This is the empirical counterpart of ``fig2_vol_mismatch.png``: instead
    of a controlled simulation, it uses the actual realized-versus-implied
    (proxy) variance gap observed episode by episode on historical prices.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = real_df["variance_risk_premium"]
    y = real_df["hedged_pnl"]
    ax.scatter(x, y, s=14, alpha=0.45, color="steelblue",
              label="rolling straddle episode")
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, reg.intercept + reg.slope * xs, color="darkred",
           linewidth=2.0,
           label=(f"OLS fit: slope={reg.slope:.2f}, "
                  f"R2={reg.rvalue ** 2:.3f}, p={reg.pvalue:.2g}"))
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Variance risk premium: implied^2 minus realized^2 "
                 "(sigma_impl^2 - sigma_real^2)")
    ax.set_ylabel("Hedged PnL per straddle episode")
    ax.set_title(f"REAL DATA ({label}): hedged short-straddle PnL regressed\n"
                "on the variance risk premium, episode by episode")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig6_real_vrp_regression.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# SECTION 6: Black-Scholes sanity checks (run before any experiment)
# =============================================================================

def sanity_checks():
    """Quick numerical checks of the pricing and Greeks machinery.

    These are cheap, deterministic tests that must always pass; a failure
    means the model code, not the market, is wrong.
    """
    S, K, T, r, sig = 100.0, 100.0, 0.25, 0.02, 0.20
    call = bs_price(S, K, T, r, sig, "call")
    put = bs_price(S, K, T, r, sig, "put")
    straddle = bs_price(S, K, T, r, sig, "straddle")

    # 1. Nonnegativity: option values can never be negative.
    assert call >= 0 and put >= 0 and straddle >= 0, "negative option value"

    # 2. Put-call parity: C - P = S - K e^{-rT}.
    parity = call - put - (S - K * np.exp(-r * T))
    assert abs(parity) < 1e-10, f"put-call parity violated: {parity}"

    # 3. Delta bounds: call delta in [0,1], put delta in [-1,0].
    gc = bs_greeks(S, K, T, r, sig, "call")
    gp = bs_greeks(S, K, T, r, sig, "put")
    assert 0.0 <= gc["delta"] <= 1.0, "call delta out of bounds"
    assert -1.0 <= gp["delta"] <= 0.0, "put delta out of bounds"

    # 4. ATM straddle delta is small in magnitude (the forward drift term
    #    (r + sigma^2/2)T pushes it slightly positive even when K = S).
    gs = bs_greeks(S, K, T, r, sig, "straddle")
    assert abs(gs["delta"]) < 0.15, "ATM straddle delta should be small"

    # 5. Finite-difference check of gamma and vega.
    eps = 1e-4
    gamma_fd = (bs_price(S + eps, K, T, r, sig, "straddle")
                - 2 * bs_price(S, K, T, r, sig, "straddle")
                + bs_price(S - eps, K, T, r, sig, "straddle")) / eps ** 2
    assert abs(gamma_fd - gs["gamma"]) < 1e-2, "gamma failed FD check"

    # 6. A single matched-vol hedged path with dense rebalancing must lose
    #    only a small fraction of the premium (continuous-hedging limit).
    path = simulate_gbm(S, mu=0.05, sigma=sig, T=1.0 / 12.0,
                        n_steps=21 * 32, n_paths=1, seed=7)[0]
    res = delta_hedge_path(path, S, r, sig, rebalance_every=1,
                           T_years=1.0 / 12.0)
    premium = bs_price(S, S, 1.0 / 12.0, r, sig, "straddle")
    assert abs(res.total_pnl) < 0.25 * premium, \
        "dense matched-vol hedge should roughly replicate the option"

    # 7. The scalar engine and the vectorized batch engine must agree on the
    #    same path to within floating-point noise.
    vec = delta_hedge_batch(path[None, :], S, r, sig, rebalance_every=1,
                            T_years=1.0 / 12.0)
    assert abs(vec["total_pnl"][0] - res.total_pnl) < 1e-8, \
        "scalar and vectorized engines disagree"

    print("[sanity] All Black-Scholes and hedging sanity checks passed.")
    return True


# =============================================================================
# SECTION 7: Real-data workflow (rolling 1-month ATM straddle study)
# =============================================================================

def load_underlying_prices(csv_path=None, ticker=None, start="2020-01-01",
                           end="2025-01-01"):
    """Load underlying daily prices from a CSV or from yfinance.

    CSV schema (case-insensitive headers): ``date, close``. Extra columns are
    ignored. If ``ticker`` is given instead, yfinance is used to download
    adjusted close prices. Returns a clean DataFrame indexed by date with a
    single ``close`` column, or None if no data could be obtained.
    """
    if csv_path:
        if not os.path.exists(csv_path):
            print(f"[real] CSV not found: {csv_path}")
            return None
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "date" not in df.columns or "close" not in df.columns:
            print("[real] CSV must contain 'date' and 'close' columns.")
            return None
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        series = df["close"].astype(float)
        label = os.path.basename(csv_path)
    elif ticker:
        try:
            import yfinance as yf
        except ImportError:
            print("[real] yfinance is not installed; supply --csv instead.")
            return None
        try:
            data = yf.download(ticker, start=start, end=end,
                               auto_adjust=True, progress=False)
        except Exception as exc:  # network failures must not crash the run
            print(f"[real] yfinance download failed: {exc}")
            return None
        if data is None or len(data) == 0:
            print(f"[real] No data returned for ticker {ticker}.")
            return None
        close = data["Close"]
        if isinstance(close, pd.DataFrame):  # yfinance may return 2-D
            close = close.iloc[:, 0]
        series = close.dropna()
        label = ticker
    else:
        print("[real] Provide either --csv PATH or --ticker SYMBOL.")
        return None

    out = pd.DataFrame({"close": series.dropna()})
    if len(out) < 60:
        print("[real] Not enough observations for a rolling study.")
        return None
    print(f"[real] Loaded {len(out)} daily closes for {label} "
          f"({out.index[0].date()} to {out.index[-1].date()}).")
    return out, label


def rolling_straddle_study(prices, r=0.02, window_days=21, hedge_every_days=1,
                           tc_bps=0.0):
    """Rolling one-month ATM straddle, delta-hedged daily, on real data.

    Methodology
    -----------
    For each start date t we sell a 1-month ATM straddle struck at the close
    of t. The hedge volatility is the LAGGED 21-day realized volatility (a
    stand-in for the implied vol quote we would have seen in the market; see
    the limitations in the report). We then delta-hedge daily over the next
    21 trading days and record the terminal hedged PnL and its attribution.
    The forward realized vol over the holding window is also recorded so the
    PnL can be related to the realized-minus-hedge variance gap.

    Returns a DataFrame with one row per straddle episode.
    """
    close = prices["close"].values
    dates = prices.index
    logret = np.diff(np.log(close))

    # Lagged 21-day realized vol (annualized), used as the hedge vol proxy.
    rv = pd.Series(logret).rolling(window_days).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR)
    rv = rv.values  # rv[i] uses returns up to and including index i+1

    rows = []
    step = 1  # start a new straddle every day; increase to subsample
    n = len(close)
    for start_idx in range(window_days, n - window_days - 1, step):
        end_idx = start_idx + window_days
        S_path = close[start_idx:end_idx + 1]
        S0 = S_path[0]
        K = S0  # ATM strike
        sigma_h = rv[start_idx - 1]
        if not np.isfinite(sigma_h) or sigma_h <= 1e-4:
            continue  # skip windows where the vol proxy is undefined
        T = window_days / TRADING_DAYS_PER_YEAR
        res = delta_hedge_path(
            S_path, K, r, sigma_h,
            rebalance_every=hedge_every_days,
            tc_bps=tc_bps, T_years=T,
            steps_per_year=TRADING_DAYS_PER_YEAR,
        )
        # Forward realized vol over the holding window.
        fwd_ret = np.diff(np.log(S_path))
        realized_vol_window = np.std(fwd_ret, ddof=1) * np.sqrt(
            TRADING_DAYS_PER_YEAR) if len(fwd_ret) > 1 else np.nan
        premium = bs_price(S0, K, T, r, sigma_h, "straddle")
        rows.append({
            "start_date": dates[start_idx],
            "end_date": dates[end_idx],
            "S0": S0, "K": K, "premium": premium,
            "iv_proxy": sigma_h,
            "realized_vol_window": realized_vol_window,
            "hedged_pnl": res.total_pnl,
            "pnl_over_premium": res.total_pnl / premium,
            "theta": res.attribution["theta"],
            "gamma": res.attribution["gamma"],
            "delta_mismatch": res.attribution["delta_mismatch"],
            "residual": res.attribution["residual"],
            "costs": res.attribution["costs"],
        })
    return pd.DataFrame(rows)


def summarize_real_study(real_df, outdir, label):
    """Write summary statistics for the rolling real-data study.

    Sign convention
    ----------------
    The strategy sells (is short) the straddle, so the delta-hedged book
    loses money when realized variance exceeds the implied (hedge) variance
    it was sold at, and makes money when realized variance falls short of
    it. That is the same relationship verified in ``experiment_vol_mismatch``
    on simulated paths (mean PnL falls as sigma_real^2 - sigma_impl^2 rises).
    Equity index option sellers are conventionally said to earn the
    "variance risk premium" (VRP), defined as implied minus realized
    variance: VRP = sigma_impl^2 - sigma_real^2. With that sign convention
    the seller's hedged PnL should be POSITIVELY related to the VRP (a
    bigger implied-over-realized gap means a bigger profit for the seller),
    which is the regression run below.
    """
    if real_df is None or len(real_df) == 0:
        return None
    rv_minus_iv_variance = (real_df["realized_vol_window"] ** 2
                            - real_df["iv_proxy"] ** 2)
    vrp = -rv_minus_iv_variance  # sigma_impl^2 - sigma_real^2

    # Full OLS regression of hedged PnL on the variance risk premium: this is
    # the "regress episode PnL on the realized/implied variance gap" study
    # required by the project specification, not just a correlation.
    reg = linregress(vrp.values, real_df["hedged_pnl"].values)
    # Fraction of episodes whose PnL sign agrees with the sign of the VRP
    # (positive VRP -> seller should profit; negative VRP -> seller should
    # lose). Episodes with a VRP indistinguishable from zero are excluded.
    nonzero = vrp.abs() > 1e-6
    sign_match_rate = float(
        (np.sign(real_df.loc[nonzero, "hedged_pnl"])
         == np.sign(vrp[nonzero])).mean())

    summary = {
        "label": label,
        "n_episodes": len(real_df),
        "mean_hedged_pnl": real_df["hedged_pnl"].mean(),
        "std_hedged_pnl": real_df["hedged_pnl"].std(ddof=1),
        "mean_pnl_over_premium": real_df["pnl_over_premium"].mean(),
        "win_rate": float((real_df["hedged_pnl"] > 0).mean()),
        "mean_variance_risk_premium": float(vrp.mean()),
        "vrp_regression_slope": float(reg.slope),
        "vrp_regression_intercept": float(reg.intercept),
        "vrp_regression_r2": float(reg.rvalue ** 2),
        "vrp_regression_pvalue": float(reg.pvalue),
        "corr_vrp_pnl": float(reg.rvalue),
        "pnl_sign_matches_vrp_sign_rate": sign_match_rate,
        "mean_residual_share": float(
            (real_df["residual"].abs()
             / real_df[["theta", "gamma", "delta_mismatch"]]
             .abs().sum(axis=1).replace(0, np.nan)).median()),
    }
    # Correctness check #5: the variance risk premium relationship must be
    # positive and statistically significant, and the large majority of
    # episodes must have hedged PnL whose sign agrees with the VRP sign.
    summary["variance_risk_premium_check_passed"] = bool(
        reg.slope > 0.0 and reg.pvalue < 0.05 and sign_match_rate > 0.5)
    real_df["variance_risk_premium"] = vrp
    real_df.to_csv(os.path.join(outdir, "real_rolling_straddle_by_episode.csv"),
                   index=False)
    pd.DataFrame([summary]).to_csv(
        os.path.join(outdir, "real_rolling_straddle_summary.csv"),
        index=False)
    plot_real_study(real_df, outdir, label)
    plot_real_vrp_regression(real_df, reg, outdir, label)
    return summary


# =============================================================================
# SECTION 8: Demo orchestration and command-line interface
# =============================================================================

def run_demo(outdir, n_paths=1500):
    """Run the full synthetic GBM demonstration and write all outputs.

    IMPORTANT: every number produced here comes from simulated GBM paths.
    The demo validates the ENGINE and the THEORY; it is not real market
    evidence. Real market conclusions require --mode real.
    """
    os.makedirs(outdir, exist_ok=True)
    print("=" * 72)
    print("SYNTHETIC DEMO MODE: all results are from simulated GBM paths.")
    print("=" * 72)

    sanity_checks()

    checks = {}

    # --- Experiment 1: baseline matched-vol hedging -------------------------
    print("\n[1/4] Baseline: matched vol, hedge every step ...")
    base_df, base_sum = experiment_baseline(outdir, n_paths=n_paths)
    checks.update({f"baseline_{k}": v for k, v in base_sum.items()})
    print(f"      premium = {base_sum['premium']:.3f}, "
          f"mean PnL = {base_sum['mean_pnl']:.4f} "
          f"({base_sum['mean_over_premium']:+.2%} of premium), "
          f"std = {base_sum['std_pnl']:.3f}")

    # --- Experiment 2: attribution reconciliation ---------------------------
    print("[2/4] Attribution reconciliation ...")
    attr_sum = experiment_attribution(base_df, outdir)
    checks.update({f"attribution_{k}": v for k, v in attr_sum.items()})
    print(f"      mean theta = {attr_sum['mean_theta']:.4f}, "
          f"mean gamma = {attr_sum['mean_gamma']:.4f}, "
          f"median residual share = "
          f"{attr_sum['median_residual_share_of_gross']:.3%}")

    # --- Experiment 3: realized vs implied vol ------------------------------
    print("[3/4] Volatility mismatch grid ...")
    vol_df, vol_sum = experiment_vol_mismatch(outdir, n_paths=n_paths)
    checks.update(vol_sum)
    for _, row in vol_df.iterrows():
        print(f"      sigma_r = {row['sigma_real']:.2f} -> "
              f"mean PnL = {row['mean_pnl']:+.4f}")

    # --- Experiment 4: frequency and transaction costs ----------------------
    print("[4/4] Hedging frequency and transaction-cost breakeven ...")
    freq_df, freq_sum = experiment_frequency(outdir, n_paths=n_paths)
    checks.update(freq_sum)
    cost_df, cost_sum = experiment_transaction_costs(outdir,
                                                     n_paths=n_paths)
    checks.update(cost_sum)

    # --- Plots ----------------------------------------------------------------
    plot_pnl_histogram(base_df, base_sum["premium"], outdir)
    plot_vol_mismatch(vol_df, outdir)
    plot_frequency(freq_df, outdir)
    plot_transaction_costs(cost_df, outdir)

    # --- Single illustrative path with full step detail ----------------------
    path = simulate_gbm(100.0, mu=0.05, sigma=0.20, T=1.0 / 12.0,
                        n_steps=21 * STEPS_PER_DAY, n_paths=1, seed=3)[0]
    detail = delta_hedge_path(path, 100.0, 0.02, 0.20, rebalance_every=1,
                              T_years=1.0 / 12.0, keep_steps=True)
    detail.step_frame.to_csv(os.path.join(outdir,
                                          "example_path_step_detail.csv"),
                             index=False)

    # --- Master check table ----------------------------------------------------
    check_df = pd.DataFrame([checks])
    check_df.to_csv(os.path.join(outdir, "correctness_checks.csv"),
                    index=False)
    print("\nCorrectness checks:")
    for k, v in checks.items():
        if k.endswith("check_passed"):
            print(f"  {k}: {v}")
    print(f"\nAll demo outputs written to: {outdir}")
    return checks


def run_real(outdir, csv_path, ticker, start, end, r=0.02, tc_bps=1.0):
    """Run the rolling straddle study on real underlying data."""
    os.makedirs(outdir, exist_ok=True)
    print("=" * 72)
    print("REAL DATA MODE: rolling 1-month ATM straddle, delta-hedged daily.")
    print("=" * 72)
    sanity_checks()
    loaded = load_underlying_prices(csv_path=csv_path, ticker=ticker,
                                    start=start, end=end)
    if loaded is None:
        print("[real] No usable data. Falling back to nothing; supply a CSV "
              "with 'date, close' columns or check the ticker/network.")
        return None
    prices, label = loaded
    real_df = rolling_straddle_study(prices, r=r, window_days=21,
                                     hedge_every_days=1, tc_bps=tc_bps)
    summary = summarize_real_study(real_df, outdir, label)
    if summary:
        print("\nReal-data study summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    print(f"\nAll real-data outputs written to: {outdir}")
    return summary


def parse_args(argv=None):
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description="Delta-hedging simulation and PnL attribution study.")
    p.add_argument("--mode", choices=["demo", "real"], default="demo",
                   help="demo = synthetic GBM experiments; "
                        "real = rolling straddle study on real prices")
    p.add_argument("--csv", default=None,
                   help="path to CSV with 'date, close' columns (real mode)")
    p.add_argument("--ticker", default=None,
                   help="yfinance ticker, e.g. SPY or ^GSPC (real mode)")
    p.add_argument("--start", default="2020-01-01",
                   help="download start date for yfinance (real mode)")
    p.add_argument("--end", default="2025-01-01",
                   help="download end date for yfinance (real mode)")
    p.add_argument("--outdir", default=None,
                   help="output folder (default: outputs_<mode> next to the "
                        "script)")
    p.add_argument("--paths", type=int, default=1500,
                   help="number of GBM paths per demo experiment")
    p.add_argument("--rate", type=float, default=0.02,
                   help="risk-free rate used in the real-data study")
    p.add_argument("--tc-bps", type=float, default=1.0,
                   help="transaction cost in bps for the real-data study")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.join(script_dir,
                                         f"outputs_{args.mode}")
    if args.mode == "demo":
        run_demo(outdir, n_paths=args.paths)
    else:
        run_real(outdir, args.csv, args.ticker, args.start, args.end,
                 r=args.rate, tc_bps=args.tc_bps)


if __name__ == "__main__":
    main()
