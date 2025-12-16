from dataclasses import dataclass
from typing import Optional, List, Dict
import pandas as pd
import numpy as np

import ta
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator, AroonIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator

from cryptoscalper.config.constants import (
    RSI_PERIOD_DEFAULT,
    RSI_PERIOD_FAST,
    EMA_FAST_PERIOD,
    EMA_MEDIUM_PERIOD,
    EMA_SLOW_PERIOD,
    MACD_FAST_PERIOD,
    MACD_SLOW_PERIOD,
    MACD_SIGNAL_PERIOD,
    BOLLINGER_PERIOD,
    BOLLINGER_STD_DEV,
    ATR_PERIOD,
    STOCHASTIC_K_PERIOD,
    STOCHASTIC_D_PERIOD,
)
from cryptoscalper.utils.logger import logger


@dataclass
class OrderbookData:
    """Données de l'orderbook pour le calcul des features."""
    bids: List[tuple]
    asks: List[tuple]
    
    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0
    
    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0
    
    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return 0.0


@dataclass
class FeatureConfig:
    """Configuration pour le calcul des features."""
    rsi_period: int = RSI_PERIOD_DEFAULT
    rsi_period_fast: int = RSI_PERIOD_FAST
    ema_fast: int = EMA_FAST_PERIOD
    ema_medium: int = EMA_MEDIUM_PERIOD
    ema_slow: int = EMA_SLOW_PERIOD
    macd_fast: int = MACD_FAST_PERIOD
    macd_slow: int = MACD_SLOW_PERIOD
    macd_signal: int = MACD_SIGNAL_PERIOD
    bollinger_period: int = BOLLINGER_PERIOD
    bollinger_std: float = BOLLINGER_STD_DEV
    atr_period: int = ATR_PERIOD
    stoch_k: int = STOCHASTIC_K_PERIOD
    stoch_d: int = STOCHASTIC_D_PERIOD
    roc_short: int = 5
    roc_long: int = 10
    returns_periods: List[int] = None
    
    def __post_init__(self):
        if self.returns_periods is None:
            self.returns_periods = [1, 5, 15]


@dataclass
class FeatureSet:
    """Ensemble de features calculées pour une paire."""
    symbol: str
    features: Dict[str, float]
    timestamp: pd.Timestamp
    
    def to_series(self) -> pd.Series:
        return pd.Series(self.features, name=self.symbol)
    
    def to_dict(self) -> Dict[str, float]:
        return self.features.copy()
    
    @property
    def count(self) -> int:
        return len(self.features)


class FeatureEngine:
    """
    Moteur de calcul des features techniques.
    VERSION CORRIGÉE - Alignée avec dataset.py
    """
    
    FEATURE_NAMES = [
        # Momentum (10)
        "rsi_14", "rsi_7", "stoch_k", "stoch_d", "williams_r",
        "roc_5", "roc_10", "momentum_5", "cci", "cmo",
        # Tendance (8)
        "ema_5_ratio", "ema_10_ratio", "ema_20_ratio",
        "macd_line", "macd_signal", "macd_histogram",
        "adx", "aroon_oscillator",
        # Volatilité (6)
        "bb_width", "bb_position", "atr", "atr_percent",
        "returns_std", "hl_range_avg",
        # Orderbook (8)
        "spread_percent", "orderbook_imbalance", "bid_depth", "ask_depth",
        "depth_ratio", "bid_pressure", "ask_pressure", "midprice_distance",
        # Volume (5)
        "volume_relative", "obv_slope", "volume_delta", "vwap_distance", "ad_line",
        # Price Action (5)
        "returns_1m", "returns_5m", "returns_15m",
        "consecutive_green", "candle_body_ratio"
    ]
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self._min_periods = 30
    
    @property
    def feature_count(self) -> int:
        return len(self.FEATURE_NAMES)
    
    def compute_features(
        self,
        df: pd.DataFrame,
        orderbook: Optional[OrderbookData] = None,
        symbol: str = "UNKNOWN"
    ) -> FeatureSet:
        """Calcule toutes les features pour une paire."""
        features = {}
        
        if len(df) < self._min_periods:
            logger.warning(f"Pas assez de données pour {symbol}: {len(df)} < {self._min_periods}")
            return self._empty_feature_set(symbol)
        
        features.update(self._compute_momentum_features(df))
        features.update(self._compute_trend_features(df))
        features.update(self._compute_volatility_features(df))
        features.update(self._compute_orderbook_features(df, orderbook))
        features.update(self._compute_volume_features(df))
        features.update(self._compute_price_action_features(df))
        
        return FeatureSet(symbol=symbol, features=features, timestamp=pd.Timestamp.now())
    
    def _empty_feature_set(self, symbol: str) -> FeatureSet:
        features = {name: np.nan for name in self.FEATURE_NAMES}
        return FeatureSet(symbol=symbol, features=features, timestamp=pd.Timestamp.now())
    
    def _compute_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule les 10 features de momentum (alignées avec dataset.py)."""
        features = {}
        close = df["close"]
        high = df["high"]
        low = df["low"]
        current_price = close.iloc[-1]
        
        # RSI
        features["rsi_14"] = self._safe_last(RSIIndicator(close, window=14).rsi())
        features["rsi_7"] = self._safe_last(RSIIndicator(close, window=7).rsi())
        
        # Stochastic
        stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
        features["stoch_k"] = self._safe_last(stoch.stoch())
        features["stoch_d"] = self._safe_last(stoch.stoch_signal())
        
        # Williams %R
        features["williams_r"] = self._safe_last(WilliamsRIndicator(high, low, close, lbp=14).williams_r())
        
        # ROC
        features["roc_5"] = self._safe_last(ROCIndicator(close, window=5).roc())
        features["roc_10"] = self._safe_last(ROCIndicator(close, window=10).roc())
        
        # Momentum NORMALISÉ (aligné avec dataset.py)
        momentum_series = close.diff(5) / close * 100
        features["momentum_5"] = self._safe_last(momentum_series)
        
        # CCI
        features["cci"] = self._safe_last(CCIIndicator(high, low, close, window=20).cci())
        
        # CMO
        features["cmo"] = self._calculate_cmo(close, 14)
        
        return features
    
    def _compute_trend_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule les 8 features de tendance (alignées avec dataset.py)."""
        features = {}
        close = df["close"]
        high = df["high"]
        low = df["low"]
        current_price = close.iloc[-1]
        
        # EMA ratios
        ema_5 = EMAIndicator(close, window=5).ema_indicator()
        ema_10 = EMAIndicator(close, window=10).ema_indicator()
        ema_20 = EMAIndicator(close, window=20).ema_indicator()
        
        features["ema_5_ratio"] = current_price / self._safe_last(ema_5)
        features["ema_10_ratio"] = current_price / self._safe_last(ema_10)
        features["ema_20_ratio"] = current_price / self._safe_last(ema_20)
        
        # MACD normalisé
        macd = MACD(close, window_fast=12, window_slow=26, window_sign=9)
        macd_line = macd.macd() / close * 100
        macd_signal = macd.macd_signal() / close * 100
        macd_histogram = macd.macd_diff() / close * 100
        
        features["macd_line"] = self._safe_last(macd_line)
        features["macd_signal"] = self._safe_last(macd_signal)
        features["macd_histogram"] = self._safe_last(macd_histogram)
        
        # ADX
        features["adx"] = self._safe_last(ADXIndicator(high, low, close, window=14).adx())
        
        # Aroon
        aroon = AroonIndicator(high, low, window=25)
        features["aroon_oscillator"] = self._safe_last(aroon.aroon_indicator())
        
        return features
    
    def _compute_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule les 6 features de volatilité (alignées avec dataset.py)."""
        features = {}
        close = df["close"]
        high = df["high"]
        low = df["low"]
        current_price = close.iloc[-1]
        
        # Bollinger Bands
        bb = BollingerBands(close, window=20, window_dev=2)
        bb_high = bb.bollinger_hband()
        bb_low = bb.bollinger_lband()
        bb_mid = bb.bollinger_mavg()
        
        # BB Width
        bb_width = (bb_high - bb_low) / bb_mid
        features["bb_width"] = self._safe_last(bb_width)
        
        # BB Position AVEC -0.5 (aligné avec dataset.py)
        bb_pos = (close - bb_low) / (bb_high - bb_low) - 0.5
        features["bb_position"] = self._safe_last(bb_pos)
        
        # ATR normalisé
        atr = AverageTrueRange(high, low, close, window=14)
        atr_series = atr.average_true_range() / close * 100
        features["atr"] = self._safe_last(atr_series)
        features["atr_percent"] = features["atr"]
        
        # Returns std
        returns_std = close.pct_change().rolling(20).std()
        features["returns_std"] = self._safe_last(returns_std)
        
        # HL range avg
        hl_range = (high - low) / close
        hl_range_avg = hl_range.rolling(20).mean()
        features["hl_range_avg"] = self._safe_last(hl_range_avg)
        
        return features
    
    def _compute_orderbook_features(self, df: pd.DataFrame, orderbook: Optional[OrderbookData]) -> Dict[str, float]:
        """Calcule les 8 features de l'orderbook."""
        if orderbook is None or not orderbook.bids or not orderbook.asks:
            return {
                "spread_percent": np.nan,
                "orderbook_imbalance": np.nan,
                "bid_depth": np.nan,
                "ask_depth": np.nan,
                "depth_ratio": np.nan,
                "bid_pressure": np.nan,
                "ask_pressure": np.nan,
                "midprice_distance": np.nan,
            }
        
        current_price = df["close"].iloc[-1]
        
        # Spread
        spread = orderbook.best_ask - orderbook.best_bid
        spread_percent = (spread / orderbook.best_bid) * 100 if orderbook.best_bid else np.nan
        
        # Imbalance
        bid_volume = sum(qty for _, qty in orderbook.bids)
        ask_volume = sum(qty for _, qty in orderbook.asks)
        total = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total if total > 0 else 0
        
        # Depths
        bid_depth = sum(price * qty for price, qty in orderbook.bids)
        ask_depth = sum(price * qty for price, qty in orderbook.asks)
        
        return {
            "spread_percent": spread_percent,
            "orderbook_imbalance": imbalance,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "depth_ratio": bid_depth / ask_depth if ask_depth > 0 else np.nan,
            "bid_pressure": bid_volume,
            "ask_pressure": ask_volume,
            "midprice_distance": (current_price - orderbook.mid_price) / orderbook.mid_price * 100 if orderbook.mid_price else np.nan,
        }
    
    def _compute_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule les 5 features de volume (alignées avec dataset.py)."""
        features = {}
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        
        # Volume relatif
        volume_sma = volume.rolling(20).mean()
        features["volume_relative"] = self._safe_last(volume / volume_sma)
        
        # OBV slope normalisé (ALIGNÉ avec dataset.py)
        obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        obv_mean = obv.abs().rolling(20).mean()
        obv_slope = obv.diff(5) / obv_mean.replace(0, np.nan)
        features["obv_slope"] = self._safe_last(obv_slope)
        
        # Volume delta normalisé (ALIGNÉ avec dataset.py)
        volume_delta_raw = volume * np.sign(close.diff())
        volume_delta_norm = volume_delta_raw / volume_sma.replace(0, np.nan)
        features["volume_delta"] = self._safe_last(volume_delta_norm)
        
        # VWAP distance (ALIGNÉ avec dataset.py - rolling 20)
        vwap = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        vwap_distance = (close - vwap) / vwap * 100
        features["vwap_distance"] = self._safe_last(vwap_distance)
        
        # A/D Line normalisé (ALIGNÉ avec dataset.py)
        ad = AccDistIndexIndicator(high, low, close, volume).acc_dist_index()
        ad_mean = ad.abs().rolling(20).mean()
        ad_norm = ad.diff(5) / ad_mean.replace(0, np.nan)
        features["ad_line"] = self._safe_last(ad_norm)
        
        return features
    
    def _compute_price_action_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule les 5 features de price action (alignées avec dataset.py)."""
        features = {}
        close = df["close"]
        open_price = df["open"]
        high = df["high"]
        low = df["low"]
        
        # Returns
        features["returns_1m"] = self._safe_last(close.pct_change(1) * 100)
        features["returns_5m"] = self._safe_last(close.pct_change(5) * 100)
        features["returns_15m"] = self._safe_last(close.pct_change(15) * 100)
        
        # Consecutive green (aligné avec dataset.py)
        is_green = (close > open_price).astype(int)
        consec = is_green.groupby((is_green != is_green.shift()).cumsum()).cumsum() * is_green
        features["consecutive_green"] = self._safe_last(consec)
        
        # Candle body ratio
        body = abs(close - open_price)
        total_range = high - low
        ratio = body / total_range.replace(0, np.nan)
        features["candle_body_ratio"] = self._safe_last(ratio)
        
        return features
    
    def _calculate_cmo(self, close: pd.Series, period: int) -> float:
        """Calcule le CMO (aligné avec dataset.py)."""
        delta = close.diff()
        gains = delta.where(delta > 0, 0).rolling(period).sum()
        losses = (-delta.where(delta < 0, 0)).rolling(period).sum()
        cmo = 100 * (gains - losses) / (gains + losses)
        return self._safe_last(cmo)
    
    def _safe_last(self, series: pd.Series) -> float:
        """Retourne la dernière valeur ou NaN."""
        if series is None or len(series) == 0:
            return np.nan
        val = series.iloc[-1]
        return val if not pd.isna(val) else np.nan
    
    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        """Calcule un ratio de manière sécurisée."""
        if denominator is None or denominator == 0 or np.isnan(denominator):
            return np.nan
        if numerator is None or np.isnan(numerator):
            return np.nan
        return numerator / denominator


def get_feature_names() -> List[str]:
    """Retourne la liste des noms de features."""
    return FeatureEngine.FEATURE_NAMES.copy()


def compute_features_for_symbol(
    df: pd.DataFrame,
    orderbook: Optional[OrderbookData] = None,
    symbol: str = "UNKNOWN",
    config: Optional[FeatureConfig] = None
) -> Dict[str, float]:
    """Fonction utilitaire pour calculer les features d'une paire."""
    engine = FeatureEngine(config)
    feature_set = engine.compute_features(df, orderbook, symbol)
    return feature_set.to_dict()