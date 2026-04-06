"""
backtest_choppy_ml.py — Standalone CHOPPY ML Strategy Backtester
=================================================================
ISOLATED: touches ZERO baseline code. Uses its own capital allocation.
Strategy: when regime=CHOPPY, rank stocks using choppy_ranker.joblib
          and hold top-ranked names until regime exits CHOPPY or
          rank decays significantly.

Key facts about the model:
  IC=0.1241 OOS (trained 2021 H1, tested 2019)
  18 features: vol_60d, atr_pct, mom_60d dominate
  Beta/correlation features learned zero importance — model found
  pure vol+momentum signal, not defensive rotation

Capital allocation: runs on SEPARATE capital sleeve (25% of portfolio)
This means results are additive to momentum baseline, not a replacement.
"""
import os, sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

sys.path.insert(0, os.path.expanduser('~/ai_trading_bot_v2'))
sys.path.insert(0, os.path.expanduser('~/ai_trading_bot_v2/v2'))

from backtester_clean import fetch_history, apply_fill_cost, calc_stats
from regime_classifier import RegimeClassifier, build_regime_series, compute_signals, load_macro_data, CHOPPY, TRENDING_BULL, BEAR
from strategy_choppy_ml import compute_choppy_features

# ── Configuration ────────────────────────────────────────────────────────────
CHOPPY_MODEL   = os.path.expanduser('~/ai_trading_bot_v2/choppy_ranker.joblib')
INITIAL_CAP    = 25_000        # 25% sleeve of $100k portfolio
MAX_POSITIONS  = 3             # max simultaneous CHOPPY positions
RISK_PER_TRADE = 0.04          # 4% of sleeve per trade (Kelly-style)
STOP_PCT       = 0.08          # 8% stop loss (wider — CHOPPY is noisy)
TAKE_PROFIT    = 0.20          # 20% take profit
MAX_HOLD_DAYS  = 20            # 20d max hold (shorter — CHOPPY mean reverts)
ML_RANK_MIN    = 0.75          # top 25% of CHOPPY scores
SLIPPAGE_BPS   = 20            # same as baseline
UNIVERSE_FILE  = os.path.expanduser('~/ai_trading_bot_v2/config.py')

START_DATE     = '2017-01-01'
END_DATE       = '2025-12-31'
CACHE_DIR      = os.path.expanduser('~/ai_trading_bot_v2/cache_choppy_bt')
os.makedirs(CACHE_DIR, exist_ok=True)

SEP = "=" * 60


def load_universe() -> list:
    """Load watchlist from config.py."""
    sys.path.insert(0, os.path.expanduser('~/ai_trading_bot_v2'))
    import config
    syms = getattr(config, 'WATCHLIST', getattr(config, 'SYMBOLS', []))
    if not syms:
        import re
        txt = open(UNIVERSE_FILE).read()
        matches = re.findall(r"'([A-Z]{1,5})'", txt)
        syms = list(dict.fromkeys(matches))
    print(f"Universe: {len(syms)} symbols")
    return syms


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone and time-of-day from index, lowercase columns."""
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = pd.DatetimeIndex(idx.date)
    df.columns = [c.lower() for c in df.columns]
    return df


def load_prices(symbols: list) -> dict:
    """Load price history for all symbols."""
    prices = {}
    errors = []
    for i, sym in enumerate(symbols):
        if i % 50 == 0:
            print(f"  Loading {i}/{len(symbols)}  loaded={len(prices)}...")
        try:
            cache = os.path.join(CACHE_DIR, f"{sym}.parquet")
            if os.path.exists(cache):
                df = pd.read_parquet(cache)
                if len(df) < 100:
                    os.remove(cache)
                    raise ValueError(f"stale cache {len(df)} rows")
            else:
                df = fetch_history(sym, days=4000)
                if df is None or len(df) < 100:
                    continue
                df = _clean_index(df)
                df.to_parquet(cache)
                prices[sym] = df
                continue
            df = _clean_index(df)
            prices[sym] = df
        except Exception as e:
            errors.append(f"{sym}: {e}")
    if errors[:3]:
        print(f"  Sample errors: {errors[:3]}")
    print(f"  Loaded {len(prices)} symbols")
    return prices


def precompute_choppy_scores(prices: dict, ranker: dict) -> dict:
    """
    Pre-compute CHOPPY ML scores for all symbols at all dates.
    Returns: dict[symbol] -> pd.Series(date -> score)
    """
    model     = ranker['model']
    feat_cols = ranker['features']
    scores    = {}

    print(f"Computing CHOPPY features for {len(prices)} symbols...")
    for i, (sym, df) in enumerate(prices.items()):
        if i % 50 == 0:
            print(f"  {i}/{len(prices)}...")
        try:
            feats = compute_choppy_features(df, symbol=sym)
            X = feats.reindex(columns=feat_cols, fill_value=0).fillna(0)
            preds = model.predict(X.values)
            # Normalize index to date-only for consistent lookup
            idx = pd.to_datetime(feats.index).normalize()
            scores[sym] = pd.Series(preds, index=idx)
        except Exception as e:
            pass

    print(f"  Scores computed for {len(scores)} symbols")
    return scores


def cross_sectional_rank(scores: dict, date) -> dict:
    """
    Rank all stocks cross-sectionally on a given date.
    Returns dict[sym] -> percentile rank (0-1).
    """
    day_scores = {}
    for sym, s in scores.items():
        if date in s.index:
            val = s[date]
            if pd.notna(val):
                day_scores[sym] = val

    if not day_scores:
        return {}

    vals  = np.array(list(day_scores.values()))
    ranks = vals.argsort().argsort() / max(len(vals) - 1, 1)
    return {sym: float(r) for sym, r in zip(day_scores.keys(), ranks)}


def run_backtest():
    print(f"\n{SEP}")
    print("  CHOPPY ML STANDALONE BACKTEST")
    print(f"  Capital sleeve: ${INITIAL_CAP:,}")
    print(f"  Period: {START_DATE} → {END_DATE}")
    print(f"  Max positions: {MAX_POSITIONS}")
    print(f"  ML rank min: {ML_RANK_MIN:.0%}")
    print(SEP)

    # ── Load model ───────────────────────────────────────────
    if not os.path.exists(CHOPPY_MODEL):
        print("ERROR: choppy_ranker.joblib not found. Run: python strategy_choppy_ml.py --train")
        return
    ranker = joblib.load(CHOPPY_MODEL)
    print(f"Model IC={ranker['ic']:.4f}  features={len(ranker['features'])}  trained={ranker['trained'][:10]}")

    # ── Load data ────────────────────────────────────────────
    symbols = load_universe()
    prices  = load_prices(symbols)

    # ── Regime data ──────────────────────────────────────────
    print("Loading macro data for regime classification...")
    spy_df, hyg_df, vix_df = load_macro_data()

    # ── Pre-compute scores ───────────────────────────────────
    choppy_scores = precompute_choppy_scores(prices, ranker)

    # ── Build trading dates ──────────────────────────────────
    spy_prices = prices.get('SPY', next(iter(prices.values())))
    all_dates  = spy_prices.index[
        (spy_prices.index >= pd.Timestamp(START_DATE)) &
        (spy_prices.index <= pd.Timestamp(END_DATE))
    ]

    # ── State ────────────────────────────────────────────────
    cash      = float(INITIAL_CAP)
    positions = {}    # sym -> {entry_price, qty, entry_date, stop, highest}
    trades    = []
    equity    = []



    print("Building regime series (with hysteresis)...")
    regime_series = build_regime_series(spy_df, hyg_df, vix_df, all_dates)
    print(f"Regime dist: {regime_series.value_counts().to_dict()}")

    print(f"\nRunning main loop ({len(all_dates)} days)...")

    for step, date in enumerate(all_dates):
        date_str = str(date.date())

        # ── Regime ───────────────────────────────────────────
        regime = regime_series.get(date, CHOPPY)

        # ── Close prices for open positions ──────────────────
        close_prices = {}
        for sym in list(positions.keys()):
            df = prices.get(sym)
            if df is not None and date in df.index:
                close_prices[sym] = float(df.loc[date, 'close'])

        # ── Mark to market equity ─────────────────────────────
        pos_value = sum(
            close_prices.get(sym, pos['entry_price']) * pos['qty']
            for sym, pos in positions.items()
        )
        equity.append({'date': date, 'equity': cash + pos_value})

        # ── Exit logic ───────────────────────────────────────
        for sym in list(positions.keys()):
            pos   = positions[sym]
            close = close_prices.get(sym, pos['entry_price'])
            hold_d = (date - pd.Timestamp(pos['entry_date'])).days

            # Track highest price
            pos['highest'] = max(pos['highest'], close)

            exit_reason = None

            # Stop loss
            if close <= pos['stop']:
                exit_reason = 'stop'

            # Take profit
            elif close >= pos['entry_price'] * (1 + TAKE_PROFIT):
                exit_reason = 'take_profit'

            # Max hold
            elif hold_d >= MAX_HOLD_DAYS:
                exit_reason = 'max_hold'

            # Regime exit — CHOPPY strategy only runs in CHOPPY
            elif regime != CHOPPY:
                exit_reason = 'regime_exit'

            # Rank decay exit — if rank drops below 25th percentile
            # AND position is underwater, cut it
            elif hold_d >= 5:
                ranks = cross_sectional_rank(choppy_scores, date)
                current_rank = ranks.get(sym, 0.5)
                unrealized_pct = (close - pos['entry_price']) / pos['entry_price']
                if current_rank < 0.25 and unrealized_pct < 0.0:
                    exit_reason = 'rank_decay'

            if exit_reason:
                fill, comm = apply_fill_cost(close, pos['qty'], 'sell')
                pnl = (fill - pos['entry_price']) * pos['qty'] - comm
                cash += fill * pos['qty'] - comm

                trades.append({
                    'symbol':      sym,
                    'entry_date':  pos['entry_date'],
                    'exit_date':   date_str,
                    'entry_price': pos['entry_price'],
                    'exit_price':  fill,
                    'qty':         pos['qty'],
                    'pnl':         pnl,
                    'reason':      exit_reason,
                    'hold_days':   hold_d,
                    'regime':      CHOPPY,
                })
                del positions[sym]

        # ── Entry logic — only in CHOPPY regime ──────────────
        if regime == CHOPPY and len(positions) < MAX_POSITIONS:
            # Get cross-sectional ranks for today
            ranks = cross_sectional_rank(choppy_scores, date)

            # Filter to high-rank stocks not already held
            candidates = [
                (sym, rank) for sym, rank in ranks.items()
                if rank >= ML_RANK_MIN
                and sym not in positions
                and sym in prices
                and date in prices[sym].index
            ]

            # Sort by rank descending
            candidates.sort(key=lambda x: -x[1])

            for sym, rank in candidates:
                if len(positions) >= MAX_POSITIONS:
                    break

                px = float(prices[sym].loc[date, 'close'])
                if px <= 0:
                    continue

                # Kelly-style sizing
                stop_px      = px * (1 - STOP_PCT)
                risk_per_shr = px - stop_px
                risk_budget  = INITIAL_CAP * RISK_PER_TRADE
                qty          = int(risk_budget / risk_per_shr) if risk_per_shr > 0 else 0

                # Cap at 30% of sleeve
                max_dollars = INITIAL_CAP * 0.30
                qty = min(qty, int(max_dollars / px))

                if qty <= 0:
                    continue

                fill, comm = apply_fill_cost(px, qty, 'buy')
                cost = fill * qty + comm

                if cost > cash:
                    continue

                cash -= cost
                positions[sym] = {
                    'entry_price': fill,
                    'qty':         qty,
                    'entry_date':  date_str,
                    'stop':        stop_px,
                    'highest':     fill,
                }

        if step % 250 == 0:
            pct = step / len(all_dates) * 100
            print(f"  {pct:.0f}%  {date_str}  regime={regime[:4]}  "
                  f"positions={len(positions)}  trades={len(trades)}  "
                  f"cash=${cash:,.0f}")

    # ── Force-close remaining positions ──────────────────────
    final_date = all_dates[-1]
    for sym, pos in list(positions.items()):
        df = prices.get(sym)
        close = float(df.loc[final_date, 'close']) if (df is not None and final_date in df.index) else pos['entry_price']
        fill, comm = apply_fill_cost(close, pos['qty'], 'sell')
        pnl = (fill - pos['entry_price']) * pos['qty'] - comm
        cash += fill * pos['qty'] - comm
        trades.append({
            'symbol': sym, 'entry_date': pos['entry_date'],
            'exit_date': str(final_date.date()), 'pnl': pnl,
            'reason': 'forced_close', 'hold_days': (final_date - pd.Timestamp(pos['entry_date'])).days,
        })

    # ── Results ───────────────────────────────────────────────
    eq = pd.DataFrame(equity).set_index('date')['equity']
    df_trades = pd.DataFrame(trades)

    print(f"\n{SEP}")
    print("  CHOPPY ML BACKTEST RESULTS")
    print(SEP)

    if len(df_trades) == 0:
        print("NO TRADES EXECUTED — regime may not have been CHOPPY during test period")
        return

    total_ret  = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    years      = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr       = ((eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1) * 100
    daily_ret  = eq.pct_change().dropna()
    sharpe     = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    roll_max   = eq.cummax()
    max_dd     = ((eq - roll_max) / roll_max).min() * 100
    win_rate   = (df_trades['pnl'] > 0).mean() * 100

    print(f"  Total Return : {total_ret:>8.2f}%")
    print(f"  CAGR         : {cagr:>8.2f}%")
    print(f"  Sharpe       : {sharpe:>8.2f}")
    print(f"  Max Drawdown : {max_dd:>8.2f}%")
    print(f"  Trades       : {len(df_trades):>8}")
    print(f"  Win Rate     : {win_rate:>8.1f}%")

    print(f"\n  Exit breakdown:")
    for reason, g in df_trades.groupby('reason'):
        wr = (g['pnl'] > 0).mean() * 100
        print(f"    {reason:<20} {len(g):>4}  WR={wr:.0f}%  avg=${g['pnl'].mean():>8,.0f}")

    print(f"\n  Year-by-year:")
    df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
    df_trades['year'] = df_trades['exit_date'].dt.year

    yr_eq = eq.copy()
    yr_eq.index = pd.to_datetime(yr_eq.index)
    yr_starts = yr_eq.groupby(yr_eq.index.year).first()
    yr_ends   = yr_eq.groupby(yr_eq.index.year).last()
    yr_rets   = (yr_ends / yr_starts - 1) * 100

    for y in sorted(yr_rets.index):
        if 2017 <= y <= 2025:
            tyr = df_trades[df_trades['year'] == y]
            wr  = (tyr['pnl'] > 0).mean() * 100 if len(tyr) > 0 else 0
            mark = '✓' if yr_rets[y] > 0 else '✗'
            print(f"    {y}  {yr_rets[y]:>7.1f}%  trades={len(tyr):>3}  WR={wr:.0f}%  {mark}")

    # Save trades
    out = os.path.expanduser('~/ai_trading_bot_v2/choppy_ml_trades.csv')
    df_trades.to_csv(out, index=False)
    print(f"\n  Trades saved to: {out}")

    print(f"\n{SEP}")
    print("  CORRELATION WITH MOMENTUM BASELINE")
    print(SEP)
    print("  (Lower = more uncorrelated = more portfolio value)")

    # Load baseline equity for correlation
    baseline_eq_path = os.path.expanduser('~/ai_trading_bot_v2/equity_v2.csv')
    if os.path.exists(baseline_eq_path):
        base = pd.read_csv(baseline_eq_path, index_col=0, parse_dates=True).squeeze()
        base.index = base.index.tz_localize(None)
        base_ret  = base.pct_change().dropna()
        choppy_ret = eq.pct_change().dropna()
        common = base_ret.index.intersection(choppy_ret.index)
        if len(common) > 100:
            corr = base_ret[common].corr(choppy_ret[common])
            print(f"  Correlation vs baseline: {corr:.3f}")
            print(f"  (Target: < 0.30 for genuine diversification)")

    print(SEP)


if __name__ == '__main__':
    run_backtest()
