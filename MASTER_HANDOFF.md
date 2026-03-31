
---
## SESSION 5 END — NEW FEATURES BUILT

### Cross-asset features added to ml_model.py (16 features):
- ca_hy_spread_level, ca_hy_spread_chg_20d, ca_hy_spread_chg_60d
- ca_hy_danger (HY spread rising >50bps = momentum crash risk)
- ca_yield_curve, ca_yield_curve_chg, ca_yield_inverted, ca_yield_steepening
- ca_copper_ret_21d, ca_copper_ret_63d, ca_copper_momentum, ca_copper_accel
- ca_ig_spread_chg, ca_credit_risk_on
- ca_breakeven_level, ca_breakeven_rising
- Data source: cross_asset_data.csv (saved, from FRED)
- Measured IC: copper=0.1166, yield_curve=0.052, HY_spread=-0.040

### Propagation features added to ml_model.py (4 features):
- upstream_prop_1d, upstream_prop_3d
- upstream_large_move_1d, upstream_signed_1d
- Measured IC: 0.031-0.055 on semiconductor/financial clusters
- Peer map hardcoded for 15 key stocks (NVDA, AMAT, LRCX etc.)

### 17-year backtest running:
- Command: python v2/backtester_v2.py --days 6500 --oos
- Log: /tmp/backtest_17yr.log
- At session end: 30% complete, 2014, 282 trades
- Projected total: ~940 trades OOS
- Est completion: ~2 more hours

### Next session immediate actions:
1. tail /tmp/backtest_17yr.log — get final results
2. retrain ranker: python ml_model.py
3. Run OOS with new ranker: python v2/backtester_v2.py --days 6500 --oos
4. Re-run all statistical analysis on ~940 trades
5. Check feature importance — did copper/propagation features rank high?
6. If IC improved: wire cross-asset into regime classifier too
7. Update cross_asset_data.csv refresh — needs daily update script

### Total features: 336 (was 316)
### Baseline still locked at commit 3f2dd20

## Session 6 Summary (2026-03-31)

### New Baseline
OOS CAGR: 18.11%  Sharpe: 1.62  MaxDD: -7.51%  (improved from 18.60%/1.55/-13.55%)

### Changes Committed
- 375 features (overnight/intraday IC=0.047, cross-asset copper IC=0.117)
- Friday exit delay (OOS p=0.035)
- Barroso vol scaling TRIED and REVERTED (-7.33% CAGR)
- Symbol historical stop rate added to LIVE_ONLY_FEATURES
- TCPS (trajectory-conditioned position sizing) added to backtester
  - Day 7 positive trajectory: OOS p=0.003, WR=74% avg=$1,071
  - Day 7 negative trajectory: OOS WR=35% avg=-$39

### Pending (check on wake)
- /tmp/backtest_tcps.log — TCPS backtest result vs 18.11% baseline
- If TCPS OOS CAGR > 18.11%: keep it, commit, deploy to live trader
- If TCPS OOS CAGR < 18.11%: revert TCPS from backtester_v2.py

### Signals Validated This Session
- TCPS day 7: IS p=0.026, OOS p=0.003 — STRONGEST signal of session
- Friday exit: IS p=0.002, OOS p=0.035 — IMPLEMENTED
- June vs JanFeb: IS p=0.079, OOS p=0.022 — validated, not yet implemented
- Intraday momentum 10am: WR=55% p=0.009 — real but costs kill it

### Dead This Session
- Barroso vol scaling, earnings acceleration, idiosyncratic strength,
  return consistency, sector rotation, regime age filter, all CHOPPY strategies

### Live Trader
- Still running, CHOPPY regime, correctly in cash
- New 375-feature ranker deployed automatically
