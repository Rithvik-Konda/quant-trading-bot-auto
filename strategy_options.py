"""
strategy_options.py — Options Overlay for High-Conviction Signals
=================================================================
Two strategies:

1. REGIME FLIP CALLS — SPY calls when regime transitions BEAR→TRENDING_BULL
   - 21-day ATM calls, 3% of capital
   - Exit: up 50% OR 5 days before expiry OR regime flips back to BEAR

2. VOL-CAP COMPLEMENT — Stock calls when vol sizing reduces position
   - When NVDA/AVGO/VRT etc get 75% or 50% sized due to high vol,
     use the "cut" capital (1-2% of total) to buy ATM calls instead
   - Captures upside the vol cap removed, keeps defined downside
   - Only for ML rank > 0.90 entries

Academic basis:
- Levered upside with defined downside is the correct structure for
  momentum signals — you want asymmetric payoff, not symmetric risk
- Options pricing on high-vol momentum stocks is often cheap relative
  to realized vol during strong trend phases (vol crush post-entry)
"""
import os
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd

def get_options_client():
    from alpaca.trading.client import TradingClient
    return TradingClient(
        os.environ.get("ALPACA_API_KEY"),
        os.environ.get("ALPACA_SECRET_KEY"),
        paper=True
    )

def get_atm_call(client, symbol: str, days_out: int = 21) -> Optional[dict]:
    """
    Find the closest ATM call option expiring ~days_out from today.
    Returns dict with contract details or None.
    """
    try:
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import ContractType, ExerciseStyle

        # Get current price
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        data_client = StockHistoricalDataClient(
            os.environ.get("ALPACA_API_KEY"),
            os.environ.get("ALPACA_SECRET_KEY"),
        )
        quote = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
        current_price = float(quote[symbol].ask_price)

        # Target expiry
        target_date = datetime.now() + timedelta(days=days_out)
        expiry_from = target_date - timedelta(days=7)
        expiry_to   = target_date + timedelta(days=7)

        # Find ATM call
        req = GetOptionContractsRequest(
            underlying_symbols=[symbol],
            contract_type=ContractType.CALL,
            expiration_date_gte=expiry_from.date(),
            expiration_date_lte=expiry_to.date(),
            strike_price_gte=str(round(current_price * 0.97, 0)),
            strike_price_lte=str(round(current_price * 1.03, 0)),
        )
        contracts = client.get_option_contracts(req)
        if not contracts or len(contracts.option_contracts) == 0:
            return None

        # Pick closest to ATM and closest expiry
        best = min(
            contracts.option_contracts,
            key=lambda c: (abs(float(c.strike_price) - current_price),
                          abs((pd.Timestamp(c.expiration_date) - pd.Timestamp.now()).days - days_out))
        )
        return {
            "symbol":      best.symbol,
            "underlying":  symbol,
            "strike":      float(best.strike_price),
            "expiry":      str(best.expiration_date),
            "current_px":  current_price,
        }
    except Exception as e:
        print(f"[options] get_atm_call error for {symbol}: {e}")
        return None


def place_call_order(client, contract_symbol: str, qty: int = 1) -> Optional[object]:
    """Place a market order for a call option contract."""
    try:
        from alpaca.trading.requests import MarketOrderRequest, OptionLegRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

        order = MarketOrderRequest(
            symbol=contract_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.SIMPLE,
        )
        result = client.submit_order(order)
        print(f"[options] Order placed: BUY {qty} {contract_symbol} — ID: {result.id}")
        return result
    except Exception as e:
        print(f"[options] Order error: {e}")
        return None


def regime_flip_call(
    portfolio_value: float = 100_000,
    capital_pct: float = 0.03,
    days_out: int = 21,
) -> None:
    """
    Buy SPY calls when regime just flipped BEAR→TRENDING_BULL.
    Call this from the live trader on regime change detection.
    """
    client = get_options_client()
    capital = portfolio_value * capital_pct
    print(f"[options] Regime flip detected — allocating ${capital:,.0f} to SPY calls")

    contract = get_atm_call(client, "SPY", days_out)
    if contract is None:
        print("[options] No suitable SPY contract found")
        return

    # Estimate contracts — each SPY contract = 100 shares
    # Rough premium estimate: ATM call ≈ 2-3% of underlying
    est_premium_per_contract = contract["current_px"] * 0.025 * 100
    qty = max(1, int(capital / est_premium_per_contract))

    print(f"[options] SPY ATM call: strike={contract['strike']} expiry={contract['expiry']}")
    print(f"[options] Est premium/contract: ${est_premium_per_contract:,.0f} → buying {qty} contracts")

    place_call_order(client, contract["symbol"], qty)


def vol_cap_complement_call(
    symbol: str,
    cut_capital: float,
    days_out: int = 21,
) -> None:
    """
    When vol sizing reduces a stock position, buy calls with the cut capital.
    E.g. NVDA normally gets $25k position → vol cap reduces to $18,750 (75%)
    → call vol_cap_complement_call("NVDA", 6250) to buy calls with the $6,250 cut
    Only for high-conviction entries (ML rank > 0.90).
    """
    if cut_capital < 500:
        print(f"[options] {symbol}: cut capital ${cut_capital:.0f} too small, skipping")
        return

    client = get_options_client()
    contract = get_atm_call(client, symbol, days_out)
    if contract is None:
        print(f"[options] {symbol}: no suitable contract found")
        return

    est_premium_per_contract = contract["current_px"] * 0.025 * 100
    qty = max(1, int(cut_capital / est_premium_per_contract))

    print(f"[options] {symbol} vol-cap complement: strike={contract['strike']} expiry={contract['expiry']}")
    print(f"[options] Cut capital: ${cut_capital:,.0f} → buying {qty} contracts")

    place_call_order(client, contract["symbol"], qty)


if __name__ == "__main__":
    print("Testing options connectivity...")
    client = get_options_client()
    account = client.get_account()
    print(f"Account equity: ${float(account.equity):,.2f}")
    print(f"Options level: {account.options_approved_level}")

    print("\nTesting SPY ATM call lookup...")
    contract = get_atm_call(client, "SPY", days_out=21)
    if contract:
        print(f"Found: {contract}")
    else:
        print("No contract found — market may be closed")

    print("\nTesting NVDA ATM call lookup...")
    contract = get_atm_call(client, "NVDA", days_out=21)
    if contract:
        print(f"Found: {contract}")
