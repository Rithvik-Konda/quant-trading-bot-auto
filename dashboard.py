"""
dashboard.py — Daily performance summary
Run anytime to see current state of paper trading
"""
import os, sys, json
import pandas as pd
from datetime import datetime
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

ALPACA_KEY    = os.environ.get('ALPACA_KEY', '')
ALPACA_SECRET = os.environ.get('ALPACA_SECRET', '')

def run_dashboard():
    print("="*65)
    print(f"TRADING DASHBOARD — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*65)

    # Account
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET,
                           base_url='https://paper-api.alpaca.markets')
        acct = api.get_account()
        equity = float(acct.equity)
        cash   = float(acct.cash)
        start  = 100_000
        total_return = (equity - start) / start
        print(f"\n💰 ACCOUNT")
        print(f"   Portfolio:    ${equity:>10,.2f}")
        print(f"   Cash:         ${cash:>10,.2f}")
        print(f"   Total return: {total_return:>+10.2%}")

        # Positions
        positions = api.list_positions()
        if positions:
            print(f"\n📊 OPEN POSITIONS ({len(positions)})")
            total_unreal = 0
            for p in positions:
                unreal = float(p.unrealized_pl)
                pct    = float(p.unrealized_plpc)
                total_unreal += unreal
                bar = "▓" * int(abs(pct)*20) if abs(pct) < 0.5 else "▓"*10
                print(f"   {p.symbol:<6}: {p.qty:>5} @ ${float(p.avg_entry_price):>8.2f}  "
                      f"PnL={unreal:>+8.2f} ({pct:>+.1%}) {bar}")
            print(f"   {'─'*45}")
            print(f"   Unrealized total: ${total_unreal:>+,.2f}")
        else:
            print(f"\n📊 No open positions")
    except Exception as e:
        print(f"\n⚠️  Alpaca not available: {e}")

    # Today's scanner
    scan_file = '/Users/rick/ai_trading_bot_v2/cache_scanner/daily_scan.json'
    if os.path.exists(scan_file):
        scan = json.load(open(scan_file))
        print(f"\n🌡️  MARKET CONDITIONS")
        print(f"   Regime:  {scan.get('regime')}")
        print(f"   Breadth: {scan.get('breadth', 0):.0%}")
        print(f"   Trading: {'✓ YES' if scan.get('can_trade') else '✗ NO'}")
        bearish = scan.get('bearish_news', [])
        if bearish:
            print(f"   ⚠️  Bearish news: {bearish}")

    # VIX signal
    vix_file = '/Users/rick/ai_trading_bot_v2/cache_vix/vix_signal.json'
    if os.path.exists(vix_file):
        vix = json.load(open(vix_file))
        print(f"\n📉 VIX: {vix.get('vix_current')} ({vix.get('signal')})  "
              f"Size scalar: {vix.get('size_scalar', 1):.0%}")

    # Hot themes
    theme_file = '/Users/rick/ai_trading_bot_v2/cache_themes/hot_themes.json'
    if os.path.exists(theme_file):
        themes = json.load(open(theme_file))
        if themes:
            print(f"\n🔥 HOT THEMES")
            for name, data in list(themes.items())[:3]:
                print(f"   {name}: RS={data.get('rs_60d',0):>+.1%}")

    # Cyclical signals
    cyc_file = '/Users/rick/ai_trading_bot_v2/cache_cyclical/inflection_scores.json'
    if os.path.exists(cyc_file):
        cyc = json.load(open(cyc_file))
        top_cyc = [r for r in cyc if r.get('inflection_score', 0) >= 4
                   and r.get('analyst_upside', 0) > 0.10][:3]
        if top_cyc:
            print(f"\n⚡ CYCLICAL INFLECTIONS")
            for r in top_cyc:
                print(f"   {r['symbol']}: score={r['inflection_score']}  "
                      f"upside={r.get('analyst_upside',0):>+.0%}")

    # Daily P&L log
    log_file = '/Users/rick/ai_trading_bot_v2/cache_alpaca/daily_log.json'
    if os.path.exists(log_file):
        log = json.load(open(log_file))
        if log:
            print(f"\n📈 PERFORMANCE HISTORY")
            for entry in log[-5:]:
                pnl = entry.get('unrealized_pnl', 0)
                port = entry.get('portfolio', 100000)
                ret  = (port - 100000) / 100000
                print(f"   {entry['date']}: ${port:>10,.0f}  {ret:>+.2%}  "
                      f"open={entry.get('n_positions',0)}")

    # Pairs signals
    pairs_file = '/Users/rick/ai_trading_bot_v2/cache_pairs/signals.json'
    if os.path.exists(pairs_file):
        pairs = json.load(open(pairs_file))
        tradeable = [p for p in pairs.get('pairs', [])
                    if p.get('action') == 'trade']
        if tradeable:
            print(f"\n⚖️  PAIRS SIGNALS")
            for p in tradeable:
                print(f"   {p['sym1']}/{p['sym2']}: z={p['z_score']:>+.2f}  {p['signal']}")

    print(f"\n{'='*65}")

if __name__ == "__main__":
    run_dashboard()
