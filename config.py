WATCHLIST = [

# ===== SEMICONDUCTORS / AI HARDWARE =====
"NVDA","AVGO","AMD","MU","MRVL","QCOM","TXN","INTC","ADI",
"AMAT","LRCX","KLAC","SMCI","ANET","ARM","MPWR","ON",
"SWKS","NXPI","SNPS","CDNS","ANSS","STX","WDC",

# ===== MEGA CAP TECH =====
"MSFT","META","AMZN","GOOGL","AAPL","TSLA",

# ===== CLOUD / SOFTWARE =====
"SNOW","MDB","DDOG","NET","TEAM","ZS","SHOP","NOW",
"CRWD","PANW","PLTR","ADBE","ORCL","CRM","INTU",
"WDAY","OKTA","DOCU","HUBS","SPLK","FTNT",
"GTLB","BILL","CFLT","S","ESTC","MNDY","TOST","SMAR",

# ===== HARDWARE / ENTERPRISE TECH =====
"DELL","HPQ","HPE","PSTG","NTAP","CDW","JNPR","FFIV",
"ZBRA","CGNX","TDY","TRMB",

# ===== INTERNET / DIGITAL MEDIA =====
"NFLX","UBER","ABNB","DASH","PINS","SNAP","RDDT",
"TTD","ZG","IAC",

# ===== FINTECH =====
"V","MA","PYPL","SQ","AFRM","UPST","SOFI","NU",
"ALLY","HOOD","COIN",

# ===== FINANCIALS — BANKS =====
"JPM","BAC","C","GS","MS","BLK","SCHW","AXP",
"SPGI","ICE","CME","AIG","CB","PGR","TRV",
"USB","PNC","TFC","COF","BK","MTB","CFG",
"HBAN","RF","KEY","FITB","ZION","CMA","WAL",

# ===== FINANCIALS — INSURANCE / ASSET MGMT =====
"MET","PRU","AFL","ALL","HIG","LNC",
"RJF","LPLA","WTW","AON","MMC","AJG","BRO","ERIE",
"BX","KKR","APO","CG","ARES",

# ===== HEALTHCARE — PHARMA / BIOTECH =====
"UNH","LLY","JNJ","PFE","ABBV","MRK","TMO","ISRG",
"DHR","ABT","BMY","AMGN","GILD","VRTX","REGN",
"ZTS","SYK","BSX","MDT","CI","CVS","HUM","CNC",
"MRNA","BNTX","BIIB","ALNY","INCY","NBIX","JAZZ",

# ===== HEALTHCARE — DEVICES / SERVICES =====
"EW","HOLX","DXCM","PODD","ALGN","IART",
"HCA","THC","UHS","LH","DGX","EXAS","NTRA",
"VEEV","DOCS",

# ===== CONSUMER STAPLES =====
"PG","KO","PEP","COST","WMT","PM","MO","CL",
"KMB","GIS","HSY","EL","MDLZ","CAG","CPB",
"SJM","HRL","TSN","KHC","MKC","CHD","CLX",
"KR","BJ","SFM",

# ===== CONSUMER DISCRETIONARY — RETAIL =====
"HD","LOW","NKE","MCD","SBUX","CMG","MAR",
"BKNG","TJX","ROST","EBAY","ETSY","ULTA",
"TGT","BBY","ANF","URBN","WSM","RH",
"BOOT","DRVN",

# ===== CONSUMER DISCRETIONARY — AUTOS / LEISURE =====
"F","GM","RIVN","HOG","POOL",
"LVS","MGM","WYNN","CZR","PENN","DKNG",
"DIS","PARA","WBD","CNK","IMAX",

# ===== INDUSTRIALS — AEROSPACE / DEFENSE =====
"LMT","BA","RTX","GD","NOC","HII","TXT","KTOS",
"RKLB",

# ===== INDUSTRIALS — MACHINERY / EQUIPMENT =====
"CAT","DE","HON","UPS","FDX","WM","GE","PH",
"ETN","ITW","EMR","ROK","CSX","UNP","NSC",
"MMM","DOV","IR","XYL","GNRC",
"FAST","GWW","MSC","WSO",

# ===== INDUSTRIALS — CONSTRUCTION =====
"PWR","PRIM","MTZ","STRL",
"VMC","MLM","EXP","SUM",
"MAS","TREX","AZEK",
"NVR","PHM","DHI","LEN","TOL","KBH",
"TMHC","SKY",

# ===== INDUSTRIALS — TRANSPORTATION / LOGISTICS =====
"CHRW","EXPD","XPO","ODFL","SAIA",
"JBHT","KNX","WERN","ARCB",
"MATX","ZIM",

# ===== ENERGY — OIL & GAS =====
"XOM","CVX","COP","EOG","SLB","PSX","VLO","MPC",
"OXY","HAL","DVN","FANG","APA","MRO","HES",
"KMI","WMB","OKE","EPD","ET","MPLX",

# ===== ENERGY — CLEAN / POWER =====
"NEE","VST","CEG","NRG","AES","CWEN",
"ENPH","SEDG","FSLR","ARRY",
"BE","PLUG",

# ===== UTILITIES =====
"DUK","SO","D","EXC","AEP","SRE","PEG",
"EIX","PCG","PPL","WEC","ES","CNP","CMS",
"NI","PNW",

# ===== MATERIALS =====
"LIN","APD","NEM","FCX","ECL","SHW","DD","DOW",
"PPG","RPM","EMN","CE","OLN",
"AA","NUE","STLD","CMC","ATI","RS",
"ALB","MP","AVY","PKG","IP","WRK","AMCR",

# ===== REAL ESTATE =====
"EQIX","DLR","AMT","CCI","SBAC","PLD","PSA",
"SPG","O","VICI","WELL","VTR",
"EQR","AVB","ESS","MAA",
"KIM","REG","STAG","EGP","REXR",

# ===== INFRASTRUCTURE / DATA CENTERS =====
"VRT","NDAQ","MSCI","CBOE","FDS",

# ===== COMMUNICATIONS =====
"T","VZ","TMUS","CHTR","CMCSA","LBRDA",

]

SECTOR_ROTATION_ENABLED = False   # handled by adaptive leadership
SECTOR_LOOKBACK_DAYS    = 60
TOP_SECTORS_TO_TRADE    = 3

SECTOR_ETFS = {
    "XLK": [
        "NVDA","AVGO","AMD","MU","MRVL","QCOM","TXN","INTC","ADI","AMAT","LRCX","KLAC",
        "SMCI","ANET","ARM","MSFT","AAPL","ORCL","CRM","INTU","ADBE","SNOW","MDB","DDOG",
        "NET","TEAM","ZS","NOW","CRWD","PANW","PLTR","FTNT","VRT","DELL","HPQ","HPE",
        "PSTG","NTAP","CDW","JNPR","FFIV","MPWR","ON","NXPI","SNPS","CDNS","ANSS",
        "ZBRA","CGNX","TDY","TRMB","GTLB","BILL","CFLT","S","ESTC","MNDY","TOST","SMAR",
    ],
    "XLC": [
        "META","GOOGL","NFLX","T","VZ","TMUS","CMCSA","CHTR","DIS","PARA","WBD",
        "SNAP","PINS","RDDT","TTD","IAC","LBRDA",
    ],
    "XLY": [
        "AMZN","TSLA","HD","LOW","NKE","MCD","SBUX","CMG","MAR","BKNG","TJX","ROST",
        "EBAY","ETSY","ULTA","SHOP","TGT","BBY","F","GM","RIVN","LVS","MGM","WYNN",
        "CZR","PENN","DKNG","ABNB","DASH","UBER","ANF","URBN","WSM","RH","POOL",
        "HOG","BOOT","DIS","PARA","WBD","CNK","IMAX",
    ],
    "XLP": [
        "PG","KO","PEP","COST","WMT","PM","MO","CL","KMB","GIS","HSY","EL",
        "MDLZ","CAG","CPB","SJM","HRL","TSN","KHC","MKC","CHD","CLX","KR","BJ","SFM",
    ],
    "XLF": [
        "JPM","BAC","C","GS","MS","BLK","SCHW","AXP","SPGI","ICE","CME","AIG","CB",
        "PGR","TRV","USB","PNC","TFC","COF","BK","MTB","CFG","HBAN","RF","KEY","FITB",
        "ZION","CMA","WAL","V","MA","PYPL","SQ","AFRM","UPST","SOFI","NU","ALLY",
        "HOOD","COIN","MET","PRU","AFL","ALL","HIG","LNC","AON","MMC","AJG","BRO",
        "ERIE","RJF","LPLA","WTW","BX","KKR","APO","CG","ARES",
    ],
    "XLV": [
        "UNH","LLY","JNJ","PFE","ABBV","MRK","TMO","ISRG","DHR","ABT","BMY","AMGN",
        "GILD","VRTX","REGN","ZTS","SYK","BSX","MDT","CI","CVS","HUM","CNC","MRNA",
        "BNTX","BIIB","ALNY","INCY","NBIX","JAZZ","EW","HOLX","DXCM","PODD","ALGN",
        "HCA","THC","UHS","LH","DGX","EXAS","NTRA","VEEV","DOCS",
    ],
    "XLI": [
        "CAT","DE","HON","LMT","BA","RTX","UPS","FDX","WM","GE","PH","ETN","ITW",
        "EMR","ROK","GD","NOC","CSX","UNP","NSC","MMM","DOV","IR","XYL","GNRC",
        "FAST","GWW","MSC","WSO","PWR","PRIM","MTZ","STRL","ODFL","SAIA","JBHT",
        "KNX","XPO","CHRW","EXPD","HII","TXT","KTOS","RKLB",
    ],
    "XLE": [
        "XOM","CVX","COP","EOG","SLB","PSX","VLO","MPC","OXY","HAL","DVN","FANG",
        "APA","MRO","HES","KMI","WMB","OKE","EPD","ET","MPLX",
        "NEE","VST","CEG","NRG","AES","CWEN","ENPH","SEDG","FSLR","ARRY","BE","PLUG",
    ],
    "XLU": [
        "DUK","SO","D","EXC","AEP","SRE","PEG","EIX","PCG","PPL","WEC","ES","CNP",
        "CMS","NI","PNW",
    ],
    "XLB": [
        "LIN","APD","NEM","FCX","ECL","SHW","DD","DOW","PPG","RPM","EMN","CE","OLN",
        "AA","NUE","STLD","CMC","ATI","RS","ALB","MP","AVY","PKG","IP","WRK","AMCR",
    ],
    "XLRE": [
        "EQIX","DLR","AMT","CCI","SBAC","PLD","PSA","SPG","O","VICI","WELL","VTR",
        "EQR","AVB","ESS","MAA","KIM","REG","STAG","EGP","REXR",
    ],
}

BENCHMARK_SYMBOL = "SPY"
INITIAL_CAPITAL  = 100_000




# ─── Position sizing ──────────────────────────────────────────────────────────
MAX_POSITIONS        = 6
MAX_TOTAL_EXPOSURE   = 1.60   # gross long exposure cap (× capital)
MAX_POSITION_WEIGHT  = 0.35
MAX_POSITION_DOLLARS = 40_000
RISK_PER_TRADE       = 0.035

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
STOP_VOL_MULTIPLIER       = 2.5   # base ATR × this in normal vol
STOP_PROFIT_TIGHTEN_START = 0.12  # start tightening when profit exceeds this
STOP_PROFIT_TIGHTEN_RATE  = 0.50  # fraction of profit gain to give back
STOP_MIN_PCT              = 0.035 # never tighter than this
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
    "INTC": "semis", "ADI": "semis",  "ARM": "semis",  "MPWR": "semis",
    "ON": "semis",   "NXPI": "semis", "SNPS": "semis", "CDNS": "semis",
    "MSFT": "software", "PLTR": "software", "CRWD": "software",
    "PANW": "software", "SNOW": "software", "MDB":  "software",
    "DDOG": "software", "NET":  "software", "TEAM": "software",
    "ZS":   "software", "SHOP": "software", "NOW":  "software",
    "GTLB": "software", "BILL": "software", "CFLT": "software",
    "S":    "software", "MNDY": "software", "TOST": "software",
    "VST":  "power",    "CEG":  "power",    "NRG":  "power",
    "VRT":  "power",    "ETN":  "industrial","PH":  "industrial",
    "GE":   "industrial","DELL":"hardware",  "HPQ": "hardware",
    "HPE":  "hardware", "PSTG":"hardware",
    "META": "mega_cap", "AMZN": "mega_cap", "GOOGL":"mega_cap",
    "AAPL": "mega_cap", "TSLA": "mega_cap",
    "JPM":  "banks",    "BAC":  "banks",    "C":    "banks",
    "GS":   "banks",    "MS":   "banks",    "USB":  "banks",
    "PNC":  "banks",    "TFC":  "banks",    "COF":  "banks",
    "MTB":  "banks",    "CFG":  "banks",    "HBAN": "banks",
    "RF":   "banks",    "KEY":  "banks",    "FITB": "banks",
    "BX":   "alt_asset","KKR":  "alt_asset","APO":  "alt_asset",
    "CG":   "alt_asset","ARES": "alt_asset",
    "XOM":  "energy",   "CVX":  "energy",   "COP":  "energy",
    "EOG":  "energy",   "OXY":  "energy",   "DVN":  "energy",
    "FANG": "energy",   "APA":  "energy",   "MRO":  "energy",
    "NVR":  "homebuilder","PHM":"homebuilder","DHI": "homebuilder",
    "LEN":  "homebuilder","TOL":"homebuilder","KBH": "homebuilder",
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