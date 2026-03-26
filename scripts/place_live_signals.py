"""
place_live_signals.py — Execute today's top ML candidates on Alpaca paper
Run after generate_signals.py. Places market orders for top candidates
that aren't already in portfolio. Respects regime and vol filters.
"""
import sys, os, pandas as pd
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2')
sys.path.insert(0, '/Users/rick/ai_trading_bot_v2/v2')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import config

def place_signals(dry_run=True):
    # Load today's signals
    sig_path = '/Users/rick/ai_trading_bot_v2/signals_today.csv'
    if not os.path.exists(sig_path):
        print("ERROR: run generate_signals.py first")
        return
    
    signals = pd.read_csv(sig_path)
    candidates = signals[signals['candidate'] == True].head(6)
    regime = signals['regime'].iloc[0] if len(signals) else 'CHOPPY'
    
    print(f"Regime: {regime}")
    print(f"Candidates: {len(candidates)}")
    
    # Connect to Alpaca
    client = TradingClient(
        os.environ.get("ALPACA_API_KEY"),
        os.environ.get("ALPACA_SECRET_KEY"),
        paper=True
    )
    account = client.get_account()
    equity = float(account.equity)
    buying_power = float(account.buying_power)
    print(f"Account equity: ${equity:,.2f}")
    print(f"Buying power: ${buying_power:,.2f}")
    
    # Get current positions
    positions = {p.symbol: p for p in client.get_all_positions()}
    print(f"Current positions: {list(positions.keys())}")
    
    # Max positions by regime
    max_pos = {"TRENDING_BULL": 6, "CHOPPY": 4, "BEAR": 2}.get(regime, 4)
    slots_available = max(0, max_pos - len(positions))
    print(f"Slots available: {slots_available}")
    
    if slots_available == 0:
        print("Portfolio full for this regime")
        return
    
    # Position sizing — equal weight across max positions
    position_size = min(equity * 0.20, buying_power / max(slots_available, 1))
    
    orders_placed = 0
    for _, row in candidates.iterrows():
        sym = row['symbol']
        if sym in positions:
            print(f"  {sym}: already in portfolio, skipping")
            continue
        if slots_available <= 0:
            break
            
        px = float(row['px'])
        vol = float(row['vol60'])
        
        # Vol-adjusted size
        if vol > 0.50:
            size = position_size * 0.50
        elif vol > 0.30:
            size = position_size * 0.75
        else:
            size = position_size
            
        qty = max(1, int(size / px))
        cost = qty * px
        
        print(f"  {sym}: ML={row['ml_score']:.3f} px=${px:.2f} vol={vol:.2f} qty={qty} cost=${cost:,.0f}")
        
        if dry_run:
            print(f"    [DRY RUN] would place BUY {qty} {sym}")
        else:
            try:
                order = client.submit_order(MarketOrderRequest(
                    symbol=sym,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                ))
                print(f"    ORDER PLACED: {order.id}")
                orders_placed += 1
                slots_available -= 1
            except Exception as e:
                print(f"    ERROR: {e}")
    
    print(f"\n{'DRY RUN — ' if dry_run else ''}Orders: {orders_placed}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Actually place orders (default: dry run)")
    args = parser.parse_args()
    place_signals(dry_run=not args.live)
