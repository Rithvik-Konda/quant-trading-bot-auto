"""
live_trader.py — Alpaca paper trading integration
Setup: export ALPACA_API_KEY="..." and ALPACA_SECRET_KEY="..."
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

API_KEY    = os.environ.get("ALPACA_API_KEY", "")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

def get_account():
    from alpaca.trading.client import TradingClient
    client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    account = client.get_account()
    print(f"Equity:        ${float(account.equity):,.2f}")
    print(f"Buying power:  ${float(account.buying_power):,.2f}")
    print(f"Cash:          ${float(account.cash):,.2f}")
    return client, account

def get_positions(client):
    positions = client.get_all_positions()
    if not positions:
        print("No open positions")
    for p in positions:
        print(f"  {p.symbol}: {p.qty} shares @ ${float(p.avg_entry_price):.2f} "
              f"| unrealized P&L: ${float(p.unrealized_pl):.2f} "
              f"({float(p.unrealized_plpc)*100:.1f}%)")
    return positions

def place_market_order(client, symbol, qty, side="buy"):
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side=="buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    result = client.submit_order(order)
    print(f"Order placed: {side.upper()} {qty} {symbol} — ID: {result.id}")
    return result

if __name__ == "__main__":
    if not API_KEY:
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("Get free paper trading keys at: https://alpaca.markets")
        sys.exit(1)
    print("=== Alpaca Paper Trading ===")
    client, account = get_account()
    print("\nOpen positions:")
    get_positions(client)
