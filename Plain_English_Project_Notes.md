# Plain English Project Notes: Delta-Hedging Simulation and PnL Attribution

## What we made

A Python program that acts like a small options trading desk. It sells a one-month
at-the-money straddle (a bet that the stock will stay quiet), hedges the
directional risk by trading the stock, and then explains, step by step, exactly
where every dollar of profit or loss came from. It can run on fake stock paths we
generate on purpose (so we know the true answer and can check the math), or on
real price history from a CSV file or yfinance.

## The concept

When you sell an option you collect a premium up front. If you do nothing, your
profit depends mostly on which way the stock moves, which is just a bet on
direction. Professional volatility traders do not want that bet. They trade the
stock against the option position, a technique called delta hedging, to cancel the
directional exposure. What remains is a much cleaner bet: you make money if the
stock turns out to be LESS jumpy than the option price implied, and you lose money
if it turns out MORE jumpy. In other words, a hedged option book is a bet on
realized versus implied volatility, not on direction.

There are three practical frictions that decide whether this works:

1. You cannot hedge continuously. Every gap between rebalances leaves a stale
   hedge and adds noise to your PnL.
2. Every trade costs money. Hedge more often and the noise shrinks but the costs
   pile up. There is a breakeven frequency beyond which hedging more destroys
   your edge.
3. You never know future volatility when you trade. Your hedge settings are based
   on a forecast, and being wrong has a price.

The program measures all three.

## How we made it

1. Built Black-Scholes pricing plus the Greeks (delta, gamma, theta, vega) from
   scratch, and verified them with put-call parity and finite differences.
2. Built a hedging engine that keeps honest books: a share position and a cash
   account that earns interest, with every trade paid out of the cash account.
   At expiry everything is liquidated and the final value is checked against the
   sum of all daily PnLs.
3. Built an attribution layer that splits each period's PnL into time decay
   (theta), convexity cost (gamma), stale-hedge slippage (delta mismatch), and a
   small leftover (residual). The pieces must add back up to the total.
4. Ran controlled experiments on simulated paths: matched volatility (should
   break even on average), volatility mismatch (PnL should move monotonically
   with the realized minus implied variance gap), hedging frequency (noise should
   shrink as we hedge more), and transaction costs (find the breakeven
   frequency).
5. Added a real-data mode that replays rolling one-month straddles over five
   years of prices and relates each episode's PnL to how jumpy the market
   actually was. Running it on five years of SPY prices (2020-2025, 1,215
   episodes) gives a regression of episode PnL on the implied-minus-realized
   variance gap with slope 36.4, R-squared 0.42, and a p-value far below any
   conventional significance threshold, so the same mechanism shows up outside
   the simulated world too.

## The steps to run it

- `python Delta_Hedging_PnL_Attribution.py --mode demo` runs the full synthetic
  study and writes figures and CSV summaries into `outputs_demo/`.
- `python Delta_Hedging_PnL_Attribution.py --mode real --csv prices.csv` (or
  `--ticker SPY`) runs the rolling straddle study on real data.

## Interview-ready explanation

"I built a delta-hedging simulator with full PnL attribution. The headline result
is that a delta-hedged short straddle is a clean bet on realized versus implied
variance: when I simulated the stock at the same vol I hedged with, the mean PnL
was statistically zero; when I moved realized vol from 10% to 30% against a 20%
implied, mean PnL moved monotonically from about +2.3 to -2.3 units. The
attribution reconciled to machine precision, with theta and gamma nearly
canceling at matched vol. I also showed the practical trade-off: hedging more
often cut PnL dispersion by over 10x, but with 10 basis points of transaction
costs the edge disappeared beyond roughly 2 rebalances a day, so there is a
breakeven hedging frequency. I validated everything with sanity checks: put-call
parity, finite-difference Greeks, zero-mean tests, and accounting consistency
between the cash account and liquidation value. I then ran the same engine on
five years of real SPY prices and regressed 1,215 rolling straddle episodes'
hedged PnL on the variance risk premium: slope 36.4, R-squared 0.42, p below
1e-140, with 89% of episodes carrying the theoretically correct PnL sign."

If asked about weaknesses: "The synthetic world uses geometric Brownian motion,
which has no jumps and constant vol, so it validates the engine rather than the
market. In real mode I use lagged realized vol as a stand-in for implied vol,
which is fine for studying relationships but not for quoting absolute PnL. And
the hedge is delta-only, so gamma and vega risk are deliberately left open."

## One-line summary

Sell the straddle, hedge the direction, keep score honestly: what is left is a
pure volatility bet, and this project proves it in simulation and measures it in
data.
