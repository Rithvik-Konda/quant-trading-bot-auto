
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

## Session 7 Summary (2026-03-31 / 2026-04-01)

### Key Findings
- 22.56% baseline (commit 21bf4d0) does NOT reproduce — was artifact of older ranker
- Real baseline is 18.11% OOS CAGR (HEAD commit 9d3989f)
- Ranker only had 204 features despite 375 being computed — overnight/intraday/earnings missing
- LIVE_ONLY_FEATURES bug: eps_beat_rate, eps_avg_surprise, eps_beat_streak wrongly excluded
  → These have real time-series data in cache_earnings/ going back to 2006
  → Fixed: removed from LIVE_ONLY_FEATURES in both build_panel and train_ranker

### Changes Made
- ml_model.py: removed eps_beat_rate/eps_avg_surprise/eps_beat_streak from LIVE_ONLY_FEATURES
- strategy_choppy.py: added mean reversion entry path (RSI2 < 15 + 200d SMA + quality > 0.5)
- backtester_v2.py: added MR entry block + MR exit logic (mr_max_hold, mr_sma_exit)
- Retrain running: /tmp/retrain.log (adds 16 new features including overnight/intraday)
- MR CHOPPY backtest running: /tmp/backtest_mr_choppy.log

### Pending (check on next session)
- /tmp/retrain.log → did retrain complete? New rankers saved?
- /tmp/backtest_mr_choppy.log → does MR improve 2019/2025/2026 CHOPPY years?
  Compare: 2019 was -6.1%, 2025 was -2.1% — both CHOPPY-dominant years
- If MR backtest improves OOS CAGR: commit + run new retrain with regime-aware MR features
- If retrain improves IC: backtest again with new rankers

### Research Completed This Session
- Overnight/intraday decomposition: IC=0.047 on YOUR data (intraday positive, overnight NEGATIVE)
- Supply chain momentum: IC 0.01-0.02 incremental, needs EDGAR parsing
- Quality-filtered mean reversion: 3.6x improvement over vanilla (Zhu et al 2019)
- VRP engine: realistic 8-12% on allocated capital, near-zero correlation to momentum
- Dynamic capital allocation: risk parity + 30% Sharpe tilt, monthly rebalancing
- PEAD dead for large caps post-2006, text-based PEAD.txt still works (3.9 bps/day)
- Same-weekday momentum (Da & Zhang 2024): IC 0.02-0.03, worth adding as feature

### Next Priority Order
1. Validate MR CHOPPY backtest result
2. If improved: commit, run retrain with regime-aware features
3. Add same-weekday momentum as LightGBM feature
4. Build PEAD text engine (FinBERT on earnings calls)
5. After 60 days live: add cash-secured put selling on CHOPPY idle cash
