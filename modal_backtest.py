import modal
import os

app = modal.App("quant-backtest")

volume = modal.Volume.from_name("backtest-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "pandas", "numpy", "scikit-learn", "lightgbm",
        "yfinance", "joblib", "scipy"
    ])
    .add_local_dir(
        os.path.expanduser("~/ai_trading_bot_v2"),
        remote_path="/root/ai_trading_bot_v2"
    )
)

@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=3600,
    volumes={"/results": volume},
)
def run_backtest(config: dict) -> dict:
    import sys, os, json
    sys.path.insert(0, "/root/ai_trading_bot_v2/v2")
    sys.path.insert(0, "/root/ai_trading_bot_v2")
    os.chdir("/root/ai_trading_bot_v2/v2")

    import strategy_trending as strat_bull
    import strategy_choppy as strat_chop

    if "max_hold_days" in config:
        strat_bull.MAX_HOLD_DAYS = config["max_hold_days"]
        strat_bull.StrategyParams.__dataclass_fields__["max_hold_days"].default = config["max_hold_days"]
    if "choppy_max_positions" in config:
        strat_chop.MAX_POSITIONS = config["choppy_max_positions"]
        strat_chop.StrategyParams.__dataclass_fields__["max_positions"].default = config["choppy_max_positions"]
    if "ml_rank_min" in config:
        strat_bull.ML_RANK_MIN = config["ml_rank_min"]
        strat_bull.StrategyParams.__dataclass_fields__["ml_rank_min"].default = config["ml_rank_min"]

    from backtester_v2 import run_backtest_v2
    from backtester_clean import calc_stats
    import pandas as pd

    equity, trades, stats = run_backtest_v2(days=3650, verbose=False)

    oos_cutoff = pd.Timestamp("2022-01-01")
    eq_oos = equity[equity.index >= oos_cutoff]
    eq_oos_norm = eq_oos / eq_oos.iloc[0] * 100000
    trades_oos = [t for t in trades if str(t.exit_date) >= "2022-01-01"]
    stats_oos = calc_stats(eq_oos_norm, trades_oos)

    annual = {}
    for yr, a in stats_oos.get("annual", {}).items():
        annual[str(yr)] = {"cagr": a["cagr"], "sharpe": a["sharpe"], "maxdd": a["max_drawdown"]}

    result = {
        "config": config,
        "oos_cagr":   stats_oos["cagr"],
        "oos_sharpe": stats_oos["sharpe"],
        "oos_maxdd":  stats_oos["max_drawdown"],
        "full_cagr":  stats["cagr"],
        "annual":     annual,
    }

    name = config.get("name", "unknown")
    with open(f"/results/{name}.json", "w") as f:
        json.dump(result, f)
    volume.commit()
    print(f"[done] {name}: OOS CAGR={result['oos_cagr']:.1%}, Sharpe={result['oos_sharpe']:.2f}")
    return result


@app.function(volumes={"/results": volume}, timeout=60)
def read_results():
    import os, json
    results = []
    for fname in sorted(os.listdir("/results")):
        if fname.endswith(".json"):
            with open(f"/results/{fname}") as f:
                results.append(json.load(f))
    return results


@app.local_entrypoint()
def main():
    import sys

    if "--read" in sys.argv:
        results = read_results.remote()
        if not results:
            print("No results yet — jobs may still be running.")
            return
        print(f"\n{'Config':<20} {'OOS CAGR':>10} {'Sharpe':>8} {'Max DD':>10} {'Full CAGR':>10}")
        print("-" * 62)
        results.sort(key=lambda x: x["oos_cagr"], reverse=True)
        for r in results:
            name = r["config"].get("name", "?")
            print(f"{name:<20} {r['oos_cagr']:>9.1%}  {r['oos_sharpe']:>7.2f}  {r['oos_maxdd']:>9.1%}  {r['full_cagr']:>9.1%}")
        print(f"\nBest: {results[0]['config']['name']} — OOS CAGR {results[0]['oos_cagr']:.1%}")
        return

    configs = [
        {"name": "baseline",     "max_hold_days": 26},
        {"name": "hold_30",      "max_hold_days": 30},
        {"name": "hold_22",      "max_hold_days": 22},
        {"name": "ml_085",       "max_hold_days": 26, "ml_rank_min": 0.85},
        {"name": "ml_090",       "max_hold_days": 26, "ml_rank_min": 0.90},
        {"name": "choppy_1",     "max_hold_days": 26, "choppy_max_positions": 1},
        {"name": "hold30_ml090", "max_hold_days": 30, "ml_rank_min": 0.90},
    ]

    print(f"Submitting {len(configs)} jobs...")
    # spawn fires jobs without waiting for results — true detach
    for config in configs:
        run_backtest.spawn(config)
    print("All 7 jobs submitted. Check results in ~30 min:")
    print("  /opt/homebrew/bin/python3.11 -m modal run modal_backtest.py -- --read")
