# LinkedIn Project Portfolio Entry

## Title

Delta-Hedging Simulation and PnL Attribution: Rebuilding the Economics of an Options Trading Desk in Python

## Description

Where does the PnL of a hedged option book actually come from? I built a complete
simulation and attribution framework to answer that question the way a derivatives
desk would.

The engine prices European straddles with Black-Scholes, runs a self-financing
discrete delta hedge with explicit cash accounting and transaction costs, and then
decomposes every hedging period into theta, gamma, vega, and delta-mismatch PnL,
reconciling to the total at machine precision.

Highlights:

- Monte Carlo study on thousands of geometric Brownian motion paths confirming the
  core theorem of options trading: a delta-hedged book is a pure bet on realized
  versus implied variance. Mean PnL responded monotonically to the variance gap,
  from +2.3 to -2.3 currency units as realized vol moved from 10% to 30% against a
  20% implied.
- Hedging frequency study showing PnL dispersion falling by more than 10x from a
  never-rebalanced hedge to a dense grid, while the mean stays pinned at zero.
- Transaction-cost analysis identifying the breakeven rebalancing frequency at
  which hedging more often destroys the volatility edge.
- A real-data workflow that replays rolling one-month ATM straddles, delta-hedged
  daily, over five years of underlying prices from CSV or yfinance, linking hedged
  PnL to the ex-post variance gap episode by episode. Run on five years of SPY
  prices (2020-2025, 1,215 episodes), it produces a regression of episode PnL on
  the variance risk premium with slope 36.4, R-squared 0.42, and p below 1e-140.
- A built-in validation suite: put-call parity, finite-difference Greeks,
  zero-mean PnL at matched vol, attribution reconciliation, and accounting
  consistency between the cash account and the liquidation value.

All synthetic results are clearly labeled as simulation output; the framework is
designed so the same code runs on real market data with one command-line flag.

## Skills (comma-separated)

Options Pricing, Black-Scholes Model, Greeks, Delta Hedging, PnL Attribution,
Monte Carlo Simulation, Geometric Brownian Motion, Volatility Trading, Transaction
Cost Analysis, Risk Management, Quantitative Finance, Python, NumPy, pandas,
Matplotlib, SciPy, yfinance, Financial Modeling, Backtesting, Model Validation,
Derivatives, Statistical Testing

## Media Recommendations

1. Figure 1 (fig1_pnl_histogram.png): the terminal PnL histogram of the hedged
   short straddle at matched vol, centered on zero. A strong visual headline.
2. Figure 2 (fig2_vol_mismatch.png): mean PnL versus the realized-minus-implied
   variance gap. The single most important result; shows the monotone relationship
   that defines volatility trading.
3. Figure 4 (fig4_transaction_costs.png): net PnL versus hedging frequency at
   several cost levels, with the zero line crossing. Great for discussing the
   practical cost trade-off.
4. Figure 5 (fig5_real_rolling_straddle.png, real mode): rolling straddle PnL on
   actual prices next to lagged and forward realized vol. Use this when discussing
   the real-data extension.
5. Figure 6 (fig6_real_vrp_regression.png, real mode): the real-data regression of
   episode PnL on the variance risk premium, with the fitted line. Pairs well with
   Figure 2 to show the same law confirmed in simulation and in real SPY data.
6. A short screen recording of running `--mode demo` end to end, showing the
   correctness checks printing True one by one.
