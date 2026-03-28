#!/bin/bash
# Run first Sunday of each month
# Crontab: 0 2 * * 0 /bin/bash ~/ai_trading_bot_v2/scripts/monthly_retrain.sh

# Only run on first Sunday of month
DAY=$(date +%d)
if [ "$DAY" -gt 7 ]; then
    echo "Not first Sunday — skipping retrain"
    exit 0
fi

cd ~/ai_trading_bot_v2
echo "Starting monthly retrain $(date)"

# Clear feature cache so fresh features are computed
rm -rf ~/ai_trading_bot_v2/cache_prices/feat_cache/
echo "Feature cache cleared"

# Retrain with latest data
/opt/homebrew/bin/python3.11 ml_model.py --rolling
echo "Retrain complete $(date)"

# Clear today ML ranks so scanner rescores tomorrow
rm -f ~/ai_trading_bot_v2/cache_alpaca/today_ml_ranks.json
rm -f ~/ai_trading_bot_v2/cache_alpaca/prev_ml_ranks.json
echo "ML rank cache cleared — will rescore tomorrow"
