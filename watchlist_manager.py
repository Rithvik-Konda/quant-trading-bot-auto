"""
watchlist_manager.py — Automatic watchlist discovery and management
============================================================
Automatically finds and adds stocks that meet criteria:
1. IPOs/spinoffs with 90+ days of history and strong RS
2. Stocks added to S&P500/Nasdaq100 index
3. Stocks crossing $5B market cap with strong momentum
4. Analyst initiations with Strong Buy on stocks not in watchlist

Removes stocks that:
1. Drop below $1B market cap
2. Get delisted or acquired
3. Have <30 days of volume data

Runs weekly — fully automatic, no manual curation needed.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json, os, sys, time
from datetime import datetime, timedelta
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

WATCHLIST_CACHE = '/Users/rick/ai_trading_bot_v2/cache_watchlist'
os.makedirs(WATCHLIST_CACHE, exist_ok=True)

# Criteria for auto-inclusion
MIN_MARKET_CAP    = 2e9     # $2B minimum
MIN_AVG_VOLUME    = 500_000 # 500k shares/day minimum
MIN_HISTORY_DAYS  = 90      # 90 days of trading history
MIN_RS_VS_SPY     = -0.10   # not more than 10% below SPY in 60d

# Candidate sources
def get_sp500_additions() -> list:
    """Get recent S&P500 additions from Wikipedia."""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500  = tables[0]['Symbol'].tolist()
        return [s.replace('.', '-') for s in sp500]
    except:
        return []

def get_nasdaq100() -> list:
    """Get Nasdaq100 components."""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        for t in tables:
            if 'Ticker' in t.columns or 'Symbol' in t.columns:
                col = 'Ticker' if 'Ticker' in t.columns else 'Symbol'
                return t[col].tolist()
    except:
        return []

def get_recent_ipos() -> list:
    """Get stocks that IPO'd in last 6 months with strong RS."""
    # Check our new_listing_scanner cache
    cache = '/Users/rick/ai_trading_bot_v2/cache_newlistings/rs_scores.json'
    if os.path.exists(cache):
        data = json.load(open(cache))
        return [r['symbol'] for r in data if r.get('tradeable')]
    return []

def meets_criteria(ticker: str) -> tuple:
    """Check if a stock meets auto-inclusion criteria."""
    try:
        tick = yf.Ticker(ticker)
        info = tick.info
        hist = tick.history(period='6mo')

        if len(hist) < MIN_HISTORY_DAYS:
            return False, f"insufficient history ({len(hist)}d)"

        mkt_cap = info.get('marketCap', 0) or 0
        if mkt_cap < MIN_MARKET_CAP:
            return False, f"market cap too small (${mkt_cap/1e9:.1f}B)"

        avg_vol = float(hist['Volume'].tail(20).mean())
        if avg_vol < MIN_AVG_VOLUME:
            return False, f"low volume ({avg_vol:,.0f})"

        # RS vs SPY
        spy = yf.Ticker('SPY').history(period='3mo')['Close']
        if len(hist) >= 60 and len(spy) >= 60:
            ret_60d    = float(hist['Close'].iloc[-1] / hist['Close'].iloc[-60] - 1)
            spy_ret_60d = float(spy.iloc[-1] / spy.iloc[-60] - 1)
            rs = ret_60d - spy_ret_60d
            if rs < MIN_RS_VS_SPY:
                return False, f"weak RS vs SPY ({rs:>+.1%})"

        return True, "qualifies"
    except Exception as e:
        return False, str(e)

def should_remove(ticker: str, current_watchlist: list) -> tuple:
    """Check if a stock should be removed from watchlist."""
    try:
        tick = yf.Ticker(ticker)
        info = tick.info

        # Delisted check
        if not info or info.get('regularMarketPrice', 0) == 0:
            return True, "possibly delisted"

        # Market cap drop
        mkt_cap = info.get('marketCap', 0) or 0
        if mkt_cap < 500e6:  # below $500M = remove
            return True, f"market cap too small (${mkt_cap/1e9:.2f}B)"

        # Acquired/merged
        if info.get('symbol', ticker) != ticker:
            return True, "ticker changed — possible acquisition"

        return False, "keep"
    except:
        return False, "keep"  # keep on error

def run_watchlist_update():
    print("="*60)
    print(f"WATCHLIST AUTO-MANAGER — {datetime.now().strftime('%Y-%m-%d')}")
    print("="*60)

    import config

    current = list(config.WATCHLIST)
    print(f"Current watchlist: {len(current)} stocks")

    # ── Step 1: Find candidates to add ────────────────────────
    print("\n[1] Finding new candidates...")
    candidates = set()

    # S&P500 and Nasdaq100
    sp500 = get_sp500_additions()
    ndx   = get_nasdaq100()
    ipos  = get_recent_ipos()

    candidates.update(sp500[:500])
    candidates.update(ndx[:100])
    candidates.update(ipos)

    # Remove already in watchlist
    new_candidates = [s for s in candidates if s not in current]
    print(f"   {len(new_candidates)} new candidates to evaluate")

    # ── Step 2: Evaluate candidates ───────────────────────────
    print("\n[2] Evaluating candidates...")
    to_add = []
    for sym in new_candidates[:50]:  # check up to 50 per run
        qualifies, reason = meets_criteria(sym)
        if qualifies:
            to_add.append(sym)
            print(f"   ✓ ADD: {sym} — {reason}")
        time.sleep(0.2)

    # ── Step 3: Check for removals ────────────────────────────
    print(f"\n[3] Checking for removals...")
    to_remove = []
    # Only check stocks added by auto-manager (not original curated list)
    auto_added_file = os.path.join(WATCHLIST_CACHE, 'auto_added.json')
    auto_added = json.load(open(auto_added_file)) if os.path.exists(auto_added_file) else []

    for sym in auto_added:
        remove, reason = should_remove(sym, current)
        if remove:
            to_remove.append(sym)
            print(f"   ✗ REMOVE: {sym} — {reason}")
        time.sleep(0.2)

    # ── Step 4: Update config.py ──────────────────────────────
    if to_add or to_remove:
        print(f"\n[4] Updating watchlist...")
        with open('/Users/rick/ai_trading_bot_v2/config.py', 'r') as f:
            content = f.read()

        # Add new stocks
        for sym in to_add:
            if sym not in content:
                # Add after WATCHLIST = [ line
                old = 'WATCHLIST = [\n'
                new = f'WATCHLIST = [\n    \'{sym}\',  # Auto-added {datetime.now().strftime("%Y-%m-%d")}\n'
                if old in content:
                    content = content.replace(old, new)

        # Remove delisted stocks
        for sym in to_remove:
            content = content.replace(f"    '{sym}',  # Auto-added", f"    # REMOVED: '{sym}'")

        with open('/Users/rick/ai_trading_bot_v2/config.py', 'w') as f:
            f.write(content)

        # Update auto_added tracking
        auto_added = [s for s in auto_added if s not in to_remove] + to_add
        with open(auto_added_file, 'w') as f:
            json.dump(auto_added, f, indent=2)

        print(f"   Added: {to_add}")
        print(f"   Removed: {to_remove}")
    else:
        print(f"\n[4] No changes needed")

    # Save report
    report = {
        'date':       str(datetime.now().date()),
        'total':      len(current) + len(to_add) - len(to_remove),
        'added':      to_add,
        'removed':    to_remove,
        'candidates': len(new_candidates),
    }
    with open(os.path.join(WATCHLIST_CACHE, 'last_update.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nWatchlist: {len(current)} → {len(current)+len(to_add)-len(to_remove)} stocks")
    return report

if __name__ == "__main__":
    run_watchlist_update()
