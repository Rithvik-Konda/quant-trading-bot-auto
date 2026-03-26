#!/bin/bash
# Run first Sunday of each month
# Add to crontab: 0 2 * * 0 /bin/bash ~/ai_trading_bot_v2/scripts/monthly_retrain.sh
cd ~/ai_trading_bot_v2
echo "Starting monthly retrain $(date)"
/opt/homebrew/bin/python3.11 ml_model.py --rolling
echo "Retrain complete $(date)"
