WATCHLIST = [

# ===== SEMICONDUCTORS / AI =====
"NVDA","AVGO","AMD","MU","MRVL","QCOM","TXN","INTC","ADI",
"AMAT","LRCX","KLAC","SMCI","ANET","ARM",

# ===== MEGA CAP TECH =====
"MSFT","META","AMZN","GOOGL","AAPL","TSLA",

# ===== CLOUD / SOFTWARE =====
"SNOW","MDB","DDOG","NET","TEAM","ZS","SHOP","NOW",
"CRWD","PANW","PLTR","ADBE","ORCL","CRM","INTU",
"WDAY","OKTA","DOCU","HUBS","SPLK","FTNT",

# ===== FINANCIALS =====
"JPM","BAC","C","GS","MS","BLK","SCHW","AXP",
"SPGI","ICE","CME","AIG","CB","PGR","TRV",
"USB","PNC","TFC","COF","BK",

# ===== HEALTHCARE =====
"UNH","LLY","JNJ","PFE","ABBV","MRK","TMO","ISRG",
"DHR","ABT","BMY","AMGN","GILD","VRTX","REGN",
"ZTS","SYK","BSX","MDT","CI",

# ===== CONSUMER STAPLES =====
"PG","KO","PEP","COST","WMT","PM","MO","CL",
"KMB","GIS","HSY","EL",

# ===== CONSUMER DISCRETIONARY =====
"HD","LOW","NKE","MCD","SBUX","CMG","MAR",
"BKNG","TJX","ROST","EBAY","ETSY","ULTA",

# ===== INDUSTRIALS =====
"CAT","DE","HON","LMT","BA","RTX","UPS","FDX",
"WM","GE","PH","ETN","ITW","EMR","ROK",
"GD","NOC","CSX","UNP","NSC",

# ===== ENERGY =====
"XOM","CVX","COP","EOG","SLB","PSX","VLO","MPC",
"OXY","HAL",

# ===== UTILITIES =====
"NEE","DUK","SO","D","EXC","AEP","SRE","PEG",

# ===== MATERIALS =====
"LIN","APD","NEM","FCX","ECL","SHW","DD","DOW",

# ===== INFRASTRUCTURE / DATA =====
"VRT","EQIX","DLR","AMT","CCI","SBAC","PLD","PSA"

]

SECTOR_ROTATION_ENABLED = False   # handled by adaptive leadership
SECTOR_LOOKBACK_DAYS    = 60
TOP_SECTORS_TO_TRADE    = 3

SECTOR_ETFS = {
    "XLK": ["NVDA","AVGO","AMD","MU","MRVL","QCOM","TXN","INTC","ADI","AMAT","LRCX","KLAC",
            "SMCI","ANET","ARM","MSFT","AAPL","ORCL","CRM","INTU","ADBE","SNOW","MDB","DDOG",
            "NET","TEAM","ZS","NOW","CRWD","PANW","PLTR","FTNT","VRT","DELL"],
    "XLY": ["AMZN","TSLA","HD","LOW","NKE","MCD","SBUX","CMG","MAR","BKNG","TJX","ROST","EBAY","ETSY","ULTA","SHOP"],
    "XLC": ["META","GOOGL"],
    "XLF": ["JPM","BAC","C","GS","MS","BLK","SCHW","AXP","SPGI","ICE","CME","AIG","CB","PGR","TRV","USB","PNC","TFC","COF","BK"],
    "XLV": ["UNH","LLY","JNJ","PFE","ABBV","MRK","TMO","ISRG","DHR","ABT","BMY","AMGN","GILD","VRTX","REGN","ZTS","SYK","BSX","MDT","CI"],
    "XLP": ["PG","KO","PEP","COST","WMT","PM","MO","CL","KMB","GIS","HSY","EL"],
    "XLI": ["CAT","DE","HON","LMT","BA","RTX","UPS","FDX","WM","GE","PH","ETN","ITW","EMR","ROK","GD","NOC","CSX","UNP","NSC"],
    "XLE": ["XOM","CVX","COP","EOG","SLB","PSX","VLO","MPC","OXY","HAL"],
    "XLU": ["NEE","DUK","SO","D","EXC","AEP","SRE","PEG","VST","CEG","NRG"],
    "XLB": ["LIN","APD","NEM","FCX","ECL","SHW","DD","DOW"],
    "XLRE": ["EQIX","DLR","AMT","CCI","SBAC","PLD","PSA"],
}

BENCHMARK_SYMBOL = "SPY"
INITIAL_CAPITAL  = 100_000

# ─── Stock classification ─────────────────────────────────────────────────────
# Tags every stock with market behavior. Used to gate short candidates.
# Defensives hold value or rise in bear markets — never short them.
# Utilities and REITs are rate-sensitive — never short in rising-rate env.
# Without this table the filter in strategy_core defaults every stock to
# "cyclical" and does nothing.
STOCK_CLASSIFICATION = {
    # Defensives — never short
    "PG":   "defensive", "KO":   "defensive", "PEP":  "defensive",
    "WMT":  "defensive", "PM":   "defensive", "MO":   "defensive",
    "CL":   "defensive", "KMB":  "defensive", "GIS":  "defensive",
    "HSY":  "defensive", "COST": "defensive", "MCD":  "defensive",
    "SBUX": "defensive", "JNJ":  "defensive", "ABBV": "defensive",
    "MRK":  "defensive", "PFE":  "defensive", "BMY":  "defensive",
    "AMGN": "defensive", "GILD": "defensive", "ABT":  "defensive",
    "MDT":  "defensive", "UNH":  "defensive", "EL":   "defensive",
    # Utilities — never short
    "NEE":  "utility",   "DUK":  "utility",   "SO":   "utility",
    "D":    "utility",   "EXC":  "utility",   "AEP":  "utility",
    "SRE":  "utility",   "PEG":  "utility",
    # REITs — never short
    "EQIX": "reit",      "DLR":  "reit",      "AMT":  "reit",
    "CCI":  "reit",      "SBAC": "reit",      "PLD":  "reit",
    "PSA":  "reit",
}
SHORT_INELIGIBLE_TYPES = {"defensive", "utility", "reit"}

# ─── Position sizing ──────────────────────────────────────────────────────────
MAX_POSITIONS        = 4
MAX_TOTAL_EXPOSURE   = 1.45   # gross long exposure cap (× capital)
MAX_POSITION_WEIGHT  = 0.35
MAX_POSITION_DOLLARS = 40_000
RISK_PER_TRADE       = 0.026

# ─── Exit parameters ──────────────────────────────────────────────────────────
FIXED_STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT     = 0.40
MAX_HOLD_DAYS       = 12
MAX_TRADES_PER_DAY  = 10

# ─── Signal thresholds ────────────────────────────────────────────────────────
RULE_THRESHOLD      = 0.05
ML_THRESHOLD        = 0.00
ML_RANK_MIN_PCT     = 0.80
COMBINED_SCORE_MIN  = 0.18

WEIGHTS = {
    "technical": 0.50,
    "volume":    0.30,
    "sentiment": 0.20,
}

# ─── Trend filter ─────────────────────────────────────────────────────────────
TREND_FILTER_ENABLED = True
TREND_SMA_FAST       = 50
TREND_SMA_SLOW       = 200

# ─── ATR stops ────────────────────────────────────────────────────────────────
USE_ATR_STOPS       = True
ATR_PERIOD          = 14
ATR_STOP_MULTIPLIER = 1.8

# ─── Adaptive stop parameters (no hardcoding — all tunable) ──────────────────
# Stop width scales with realized vol of the individual stock.
# As a position moves into profit the stop tightens to lock in gains.
STOP_VOL_MULTIPLIER       = 2.0   # base ATR × this in normal vol
STOP_PROFIT_TIGHTEN_START = 0.08  # start tightening when profit exceeds this
STOP_PROFIT_TIGHTEN_RATE  = 0.50  # fraction of profit gain to give back
STOP_MIN_PCT              = 0.015 # never tighter than this
STOP_MAX_PCT              = 0.12  # never wider than this

# ─── Regime detection ────────────────────────────────────────────────────────
# Continuous 0-1 regime score replaces binary bear/bull switch.
# Score feeds smoothly into position sizing and long/short ratio.
ENABLE_REGIME_FILTER   = True
REGIME_SENSITIVITY     = 0.5    # how quickly the score responds to new data
REGIME_BREADTH_LOOKBACK= 50     # days to compute market breadth
REGIME_VOL_LOOKBACK    = 20     # days for realized vol estimate
SPY_CRASH_HALT_PCT     = -0.03  # hard intraday halt still kept as circuit breaker
REALIZED_VOL_HALT      = 0.40   # raised — previous 0.32 was too trigger-happy
BEAR_POSITION_SCALAR   = 0.75   # used only if regime score unavailable

# Thresholds on 0-1 regime score
REGIME_BULL_THRESHOLD    = 0.60  # above this → full long book
REGIME_NEUTRAL_THRESHOLD = 0.45  # between neutral and bull → long + short
REGIME_BEAR_THRESHOLD    = 0.45  # below this → short-heavy or short-only

# Regime component weights (must sum to 1.0)
# Breadth raised: 80% of stocks below 200MA is an unambiguous bear signal
# MA structure lowered: SPY MAs lag — breadth and vol lead
REGIME_WEIGHT_MA_STRUCTURE = 0.20  # was 0.30 — lags, can mask bear early
REGIME_WEIGHT_MOMENTUM     = 0.25  # unchanged
REGIME_WEIGHT_VOL          = 0.20  # unchanged
REGIME_WEIGHT_BREADTH      = 0.20  # was 0.10 — leading indicator, raised
REGIME_WEIGHT_VOL_TREND    = 0.08  # unchanged
REGIME_WEIGHT_DRAWDOWN     = 0.07  # unchanged

# After a regime_exit, don't re-enter longs for this many trading days.
REGIME_EXIT_COOLDOWN_DAYS  = 10

# Short reentry cooldown: if a short stop fires on a symbol, block re-entry
# for this many days. Prevents repeatedly shorting the same stock that keeps
# bouncing (e.g. KMB shorted 5 times in 2022, stopped twice).
SHORT_REENTRY_COOLDOWN_DAYS = 60

# Staggered entries: max new long positions opened per day.
# Prevents entering 4 correlated positions on the same day and all stopping
# out together when market gaps down (e.g. January 4, 2022).
MAX_NEW_ENTRIES_PER_DAY = 2

# Regime must hold this many consecutive days before officially flipping.
# Stops same-day entry/exit churn during volatile oscillations.
REGIME_MIN_DAYS_BEFORE_FLIP = 5

# VIX spike early warning: halt new long entries when vol spikes sharply
# even before the regime score has moved. Catches early bear signals.
VIX_SPIKE_WINDOW_DAYS  = 5     # lookback window for vol comparison
VIX_SPIKE_THRESHOLD    = 0.30  # 30% jump in realized vol triggers halt

# Short sector exclusion: don't short stocks in outperforming sectors.
# Prevents shorting energy in 2022 when energy was +60%.
SHORT_SECTOR_EXCLUDE_LOOKBACK   = 60    # days to measure sector performance
SHORT_SECTOR_EXCLUDE_THRESHOLD  = 0.05  # sectors up >5% are excluded from shorts

# ─── Long/short parameters ────────────────────────────────────────────────────
LONG_ENTRY_THRESHOLD  = 0.80   # top X% of ML ranking → long candidates
SHORT_ENTRY_THRESHOLD = 0.20   # bottom X% of ML ranking → short candidates
SHORT_MAX_POSITIONS   = 4      # max simultaneous short positions
SHORT_RISK_PER_TRADE  = 0.020  # slightly smaller risk per short
SHORT_MAX_DOLLARS     = 35_000 # max dollars per short position

# Net exposure by regime (long_weight - short_weight as fraction of capital)
NET_EXPOSURE_BULL    =  1.0   # 100% net long
NET_EXPOSURE_NEUTRAL =  0.0   # market neutral
NET_EXPOSURE_BEAR    = -0.5   # net short

# Short exit parameters
SHORT_TAKE_PROFIT_PCT = 0.20   # cover if down 20%
SHORT_STOP_PCT        = 0.08   # stop out if up 8% against us
SHORT_MAX_HOLD_DAYS   = 15     # cover after this many days

# ─── Correlation / diversification ───────────────────────────────────────────
CORRELATION_LOOKBACK_DAYS      = 60
CORRELATION_PENALTY_START      = 0.55
CORRELATION_PENALTY_MULT       = 0.50
MAX_CORRELATED_BUCKET_WEIGHT   = 0.35

CORRELATION_BUCKETS = {
    "NVDA": "semis", "AVGO": "semis", "AMD": "semis", "MU": "semis",
    "SMCI": "semis", "MRVL": "semis", "ANET": "semis", "KLAC": "semis",
    "LRCX": "semis", "AMAT": "semis", "QCOM": "semis", "TXN": "semis",
    "INTC": "semis", "ADI": "semis",  "ARM": "semis",
    "MSFT": "software", "PLTR": "software", "CRWD": "software",
    "PANW": "software", "SNOW": "software", "MDB":  "software",
    "DDOG": "software", "NET":  "software", "TEAM": "software",
    "ZS":   "software", "SHOP": "software", "NOW":  "software",
    "VST":  "power",    "CEG":  "power",    "NRG":  "power",
    "VRT":  "power",    "ETN":  "industrial","PH":  "industrial",
    "GE":   "industrial","DELL":"hardware",
    "META": "mega_cap", "AMZN": "mega_cap", "GOOGL":"mega_cap",
    "AAPL": "mega_cap", "TSLA": "mega_cap",
}

ROTATION_SCORE_GAP = 0.08

# ─── Execution costs ─────────────────────────────────────────────────────────
SLIPPAGE_BPS           = 5
COMMISSION_PER_SHARE   = 0.005
COMMISSION_MAX_PCT     = 0.005

COOLDOWN_AFTER_WIN_MINUTES  = 10
COOLDOWN_AFTER_LOSS_MINUTES = 30

ALLOW_PYRAMIDING        = False
MAX_PYRAMID_ADDS        = 0
PYRAMID_MIN_R_MULTIPLE  = 1.5
PYRAMID_SIZE_FRACTION   = 0.50

# ─── Adaptive leadership ─────────────────────────────────────────────────────
ADAPTIVE_LEADERSHIP_ENABLED = True
LEADERSHIP_UPDATE_FREQ_DAYS = 5
LEADERSHIP_THRESHOLD        = 0.62
LEADERSHIP_TOP_N            = 4
LEADERSHIP_MIN_MULTIPLIER   = 0.60
LEADERSHIP_BOOST_CAP        = 1.50