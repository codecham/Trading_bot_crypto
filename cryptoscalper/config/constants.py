# cryptoscalper/config/constants.py
"""
Constantes globales du projet CryptoScalper AI.

Toutes les valeurs fixes doivent être définies ici.
Jamais de "magic numbers" dans le code !
"""

# ============================================
# BINANCE API
# ============================================

# URLs
BINANCE_API_URL = "https://api.binance.com"
BINANCE_TESTNET_API_URL = "https://testnet.binance.vision"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"
BINANCE_TESTNET_WS_URL = "wss://testnet.binance.vision/ws"

# Timeframes (intervalles de bougies)
KLINE_INTERVAL_1M = "1m"
KLINE_INTERVAL_3M = "3m"
KLINE_INTERVAL_5M = "5m"
KLINE_INTERVAL_15M = "15m"
KLINE_INTERVAL_1H = "1h"

# Limites API
MAX_KLINES_PER_REQUEST = 1000
MAX_STREAMS_PER_CONNECTION = 1024
RATE_LIMIT_REQUESTS_PER_MINUTE = 1200
RATE_LIMIT_ORDERS_PER_SECOND = 10

# ============================================
# TRADING
# ============================================

# Ordre minimum sur Binance (en USDT)
MIN_ORDER_VALUE_USDT = 10.0

# Frais de trading Binance (0.1% = 0.001)
BINANCE_FEE_PERCENT = 0.001

# Frais aller-retour (achat + vente)
ROUND_TRIP_FEE_PERCENT = BINANCE_FEE_PERCENT * 2

# Slippage estimé pour les ordres market
ESTIMATED_SLIPPAGE_PERCENT = 0.0005  # 0.05%

# Durée maximale d'une position (en secondes)
MAX_POSITION_DURATION_SECONDS = 300  # 5 minutes

# Timeout pour les ordres (en secondes)
ORDER_TIMEOUT_SECONDS = 30

# ============================================
# SCANNER
# ============================================

# Paires exclues (stablecoins, etc.)
EXCLUDED_PAIRS = frozenset([
    "USDCUSDT",
    "BUSDUSDT", 
    "TUSDUSDT",
    "EURUSDT",
    "DAIUSDT",
    "FDUSDUSDT",
    "USDPUSDT",
])

# Quote asset pour le trading
QUOTE_ASSET = "USDT"

# Nombre de paires à surveiller au maximum
MAX_PAIRS_TO_SCAN = 150

# Intervalle de rafraîchissement de la liste des paires (en secondes)
PAIRS_REFRESH_INTERVAL_SECONDS = 3600  # 1 heure

# ============================================
# INDICATEURS TECHNIQUES
# ============================================

# RSI
RSI_PERIOD_DEFAULT = 14
RSI_PERIOD_FAST = 7
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# MACD
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# Bollinger Bands
BOLLINGER_PERIOD = 20
BOLLINGER_STD_DEV = 2

# ATR (Average True Range)
ATR_PERIOD = 14

# EMA (Exponential Moving Average)
EMA_FAST_PERIOD = 5
EMA_MEDIUM_PERIOD = 10
EMA_SLOW_PERIOD = 20

# Stochastic
STOCHASTIC_K_PERIOD = 14
STOCHASTIC_D_PERIOD = 3

# ============================================
# MACHINE LEARNING
# ============================================

# Features
TOTAL_FEATURES_COUNT = 42

# Label (target)
PREDICTION_HORIZON_MINUTES = 3
MIN_PRICE_CHANGE_FOR_LABEL = 0.002  # 0.2%

# Train/Val/Test split
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# XGBoost defaults
XGBOOST_N_ESTIMATORS = 200
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.05

# ============================================
# RISK MANAGEMENT
# ============================================

# Limites de perte
MAX_DAILY_LOSS_PERCENT = 0.10  # 10% du capital
MAX_DRAWDOWN_PERCENT = 0.25  # 25% du capital initial

# Limites de trading
MAX_TRADES_PER_HOUR = 20
MAX_TRADES_PER_DAY = 100
MAX_CONSECUTIVE_LOSSES = 5

# Position sizing
DEFAULT_POSITION_SIZE_PERCENT = 0.20  # 20% du capital
MIN_POSITION_SIZE_USDT = 10.0  # Minimum Binance

# ============================================
# LOGGING
# ============================================

# Formats de date
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATETIME_FORMAT_MS = "%Y-%m-%d %H:%M:%S.%f"

# Fichiers de log
LOG_DIR = "logs"
MAIN_LOG_FILE = "cryptoscalper.log"
TRADES_LOG_FILE = "trades.log"
ERRORS_LOG_FILE = "errors.log"

# ============================================
# PATHS
# ============================================

# Dossiers
DATA_DIR = "data"
MODELS_DIR = "models/saved"
CACHE_DIR = "cache"

# Fichiers
MODEL_FILENAME = "xgb_model_latest.joblib"
SCALER_FILENAME = "scaler_latest.joblib"
TRADES_HISTORY_FILENAME = "trades_history.csv"