"""
daily_report.py — End of day P&L summary
Run at 4:00pm ET via cron
"""
import os, sys, json
import pandas as pd
from datetime import datetime
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')

ALPACA_KEY    = os.environ.get('ALPACA_KEY', '')
ALPACA_SECRET = os.environ.get('ALPACA_SECRET', '')
CACHE_DIR     = '/Users/rick/ai_trading_bot_v2/cache_alpaca'

def run_report():
    import alpaca_trade_api as tradeapi
    api = tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, base_url='https://paper-api.alpaca.markets')

    account   = api.get_account()
    portfolio = float(account.portfolio_value)
    cash      = float(account.cash)
    equity    = float(account.equity)

    positions = api.list_positions()

    print("="*60)
    print(f"DAILY REPORT — {datetime.now().strftime('%Y-%m-%d')}")
    print("="*60)
    print(f"Portfolio value: ${portfolio:>10,.2f}")
    print(f"Cash:            ${cash:>10,.2f}")
    print(f"Open positions:  {len(positions)}")

    total_unrealized = 0
    print(f"\nPositions:")
    for p in positions:
        unreal = float(p.unrealized_pl)
        unreal_pct = float(p.unrealized_plpc)
        total_unrealized += unreal
        print(f"  {p.symbol:<6}: {p.qty:>4} shares  "
              f"entry=${float(p.avg_entry_price):>8.2f}  "
              f"current=${float(p.current_price):>8.2f}  "
              f"PnL=${unreal:>+8.2f} ({unreal_pct:>+.1%})")

    print(f"\nTotal unrealized: ${total_unrealized:>+,.2f}")

    # Load scan for context
    scan_file = f'{CACHE_DIR}/../cache_scanner/daily_scan.json'
    if os.path.exists(scan_file):
        scan = json.load(open(scan_file))
        print(f"\nToday's regime: {scan.get('regime')}")
        print(f"Breadth: {scan.get('breadth', 0):.0%}")

    # Save daily snapshot
    os.makedirs(CACHE_DIR, exist_ok=True)
    snapshot = {
        'date': str(datetime.now().date()),
        'portfolio': portfolio,
        'cash': cash,
        'unrealized_pnl': total_unrealized,
        'n_positions': len(positions),
    }
    log_file = os.path.join(CACHE_DIR, 'daily_log.json')
    history = json.load(open(log_file)) if os.path.exists(log_file) else []
    history.append(snapshot)
    with open(log_file, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved to cache_alpaca/daily_log.json")
    print("="*60)

if __name__ == "__main__":
    run_report()
