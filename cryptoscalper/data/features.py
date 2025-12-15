# cryptoscalper/data/features.py
"""
Feature Engine - Calcul des indicateurs techniques pour le modèle ML.

Calcule 42 features réparties en 6 catégories:
- Momentum (10): RSI, Stochastic, Williams %R, ROC, CCI, CMO
- Tendance (8): EMA ratios, MACD, ADX, Aroon
- Volatilité (6): Bollinger Bands, ATR, écart-type
- Orderbook (8): Spread, imbalance, depth, pression
- Volume (5): Volume relatif, OBV, VWAP, A/D
- Price Action (5): Returns, chandeliers consécutifs

Usage:
    engine = FeatureEngine()
    features = engine.compute_features(df, orderbook)
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import pandas as pd
import numpy as np

# Librairie ta pour les indicateurs techniques
import ta
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator, AroonIndicator
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


# ============================================
# DATACLASSES
# ============================================

@dataclass
class OrderbookData:
    """Données de l'orderbook pour le calcul des features."""
    
    bids: List[tuple]  # [(price, quantity), ...]
    asks: List[tuple]  # [(price, quantity), ...]
    
    @property
    def best_bid(self) -> float:
        """Meilleur prix d'achat."""
        return self.bids[0][0] if self.bids else 0.0
    
    @property
    def best_ask(self) -> float:
        """Meilleur prix de vente."""
        return self.asks[0][0] if self.asks else 0.0
    
    @property
    def mid_price(self) -> float:
        """Prix médian."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return 0.0


@dataclass
class FeatureConfig:
    """Configuration pour le calcul des features."""
    
    # Périodes RSI
    rsi_period: int = RSI_PERIOD_DEFAULT
    rsi_period_fast: int = RSI_PERIOD_FAST
    
    # Périodes EMA
    ema_fast: int = EMA_FAST_PERIOD
    ema_medium: int = EMA_MEDIUM_PERIOD
    ema_slow: int = EMA_SLOW_PERIOD
    
    # MACD
    macd_fast: int = MACD_FAST_PERIOD
    macd_slow: int = MACD_SLOW_PERIOD
    macd_signal: int = MACD_SIGNAL_PERIOD
    
    # Bollinger
    bollinger_period: int = BOLLINGER_PERIOD
    bollinger_std: float = BOLLINGER_STD_DEV
    
    # ATR
    atr_period: int = ATR_PERIOD
    
    # Stochastic
    stoch_k: int = STOCHASTIC_K_PERIOD
    stoch_d: int = STOCHASTIC_D_PERIOD
    
    # ROC
    roc_short: int = 5
    roc_long: int = 10
    
    # Returns
    returns_periods: List[int] = None
    
    def __post_init__(self):
        if self.returns_periods is None:
            self.returns_periods = [1, 5, 15]  # 1m, 5m, 15m


@dataclass
class FeatureSet:
    """Ensemble de features calculées pour une paire."""
    
    symbol: str
    features: Dict[str, float]
    timestamp: pd.Timestamp
    
    def to_series(self) -> pd.Series:
        """Convertit en pandas Series."""
        return pd.Series(self.features, name=self.symbol)
    
    def to_dict(self) -> Dict[str, float]:
        """Retourne le dictionnaire des features."""
        return self.features.copy()
    
    @property
    def count(self) -> int:
        """Nombre de features."""
        return len(self.features)


# ============================================
# FEATURE ENGINE
# ============================================

class FeatureEngine:
    """
    Moteur de calcul des features techniques.
    
    Calcule 42 indicateurs techniques à partir des données OHLCV
    et de l'orderbook.
    
    Usage:
        engine = FeatureEngine()
        
        # Pour une paire
        features = engine.compute_features(df, orderbook)
        
        # Pour plusieurs paires
        batch = engine.compute_features_batch(data_dict)
    """
    
    # Liste des noms de features (pour référence)
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
        """
        Initialise le Feature Engine.
        
        Args:
            config: Configuration des indicateurs (optionnel)
        """
        self.config = config or FeatureConfig()
        self._min_periods = 30  # Minimum de bougies requis
    
    @property
    def feature_count(self) -> int:
        """Nombre total de features."""
        return len(self.FEATURE_NAMES)
    
    def compute_features(
        self,
        df: pd.DataFrame,
        orderbook: Optional[OrderbookData] = None,
        symbol: str = "UNKNOWN"
    ) -> FeatureSet:
        """
        Calcule toutes les features pour une paire.
        
        Args:
            df: DataFrame avec colonnes OHLCV (open, high, low, close, volume)
            orderbook: Données de l'orderbook (optionnel)
            symbol: Symbole de la paire
            
        Returns:
            FeatureSet avec les 42 features
        """
        features = {}
        
        # Vérifier les données
        if len(df) < self._min_periods:
            logger.warning(f"Pas assez de données pour {symbol}: {len(df)} < {self._min_periods}")
            return self._empty_feature_set(symbol)
        
        # 1. Features Momentum
        momentum_features = self._compute_momentum_features(df)
        features.update(momentum_features)
        
        # 2. Features Tendance
        trend_features = self._compute_trend_features(df)
        features.update(trend_features)
        
        # 3. Features Volatilité
        volatility_features = self._compute_volatility_features(df)
        features.update(volatility_features)
        
        # 4. Features Orderbook
        orderbook_features = self._compute_orderbook_features(df, orderbook)
        features.update(orderbook_features)
        
        # 5. Features Volume
        volume_features = self._compute_volume_features(df)
        features.update(volume_features)
        
        # 6. Features Price Action
        price_action_features = self._compute_price_action_features(df)
        features.update(price_action_features)
        
        return FeatureSet(
            symbol=symbol,
            features=features,
            timestamp=pd.Timestamp.now()
        )
    
    def compute_features_batch(
        self,
        data_dict: Dict[str, tuple]
    ) -> pd.DataFrame:
        """
        Calcule les features pour plusieurs paires.
        
        Args:
            data_dict: {symbol: (df, orderbook)} pour chaque paire
            
        Returns:
            DataFrame avec une ligne par paire
        """
        results = []
        
        for symbol, (df, orderbook) in data_dict.items():
            try:
                feature_set = self.compute_features(df, orderbook, symbol)
                results.append(feature_set.to_series())
            except Exception as e:
                logger.error(f"Erreur calcul features {symbol}: {e}")
                results.append(self._empty_feature_set(symbol).to_series())
        
        if not results:
            return pd.DataFrame(columns=self.FEATURE_NAMES)
        
        return pd.DataFrame(results)
    
    def _empty_feature_set(self, symbol: str) -> FeatureSet:
        """Retourne un FeatureSet avec des NaN."""
        features = {name: np.nan for name in self.FEATURE_NAMES}
        return FeatureSet(symbol=symbol, features=features, timestamp=pd.Timestamp.now())
    
    # =========================================
    # MOMENTUM FEATURES (10)
    # =========================================
    
    def _compute_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule les 10 features de momentum."""
        features = {}
        close = df["close"]
        high = df["high"]
        low = df["low"]
        
        # RSI 14 périodes
        features["rsi_14"] = self._safe_last(
            RSIIndicator(close, window=self.config.rsi_period).rsi()
        )
        
        # RSI 7 périodes (plus réactif)
        features["rsi_7"] = self._safe_last(
            RSIIndicator(close, window=self.config.rsi_period_fast).rsi()
        )
        
        # Stochastic
        stoch = StochasticOscillator(
            high, low, close,
            window=self.config.stoch_k,
            smooth_window=self.config.stoch_d
        )
        features["stoch_k"] = self._safe_last(stoch.stoch())
        features["stoch_d"] = self._safe_last(stoch.stoch_signal())
        
        # Williams %R
        features["williams_r"] = self._safe_last(
            WilliamsRIndicator(high, low, close, lbp=14).williams_r()
        )
        
        # ROC (Rate of Change)
        features["roc_5"] = self._safe_last(
            ROCIndicator(close, window=self.config.roc_short).roc()
        )
        features["roc_10"] = self._safe_last(
            ROCIndicator(close, window=self.config.roc_long).roc()
        )
        
        # Momentum simple (variation sur 5 périodes)
        features["momentum_5"] = self._calculate_momentum(close, 5)
        
        # CCI (Commodity Channel Index)
        features["cci"] = self._calculate_cci(high, low, close, 20)
        
        # CMO (Chande Momentum Oscillator)
        features["cmo"] = self._calculate_cmo(close, 14)
        
        return features
    
    def _calculate_momentum(self, close: pd.Series, period: int) -> float:
        """Calcule le momentum simple."""
        if len(close) < period:
            return np.nan
        return close.iloc[-1] - close.iloc[-period]
    
    def _calculate_cci(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> float:
        """Calcule le CCI."""
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad)
        return self._safe_last(cci)
    
    def _calculate_cmo(self, close: pd.Series, period: int) -> float:
        """Calcule le CMO (Chande Momentum Oscillator)."""
        delta = close.diff()
        gains = delta.where(delta > 0, 0).rolling(window=period).sum()
        losses = (-delta.where(delta < 0, 0)).rolling(window=period).sum()
        cmo = 100 * (gains - losses) / (gains + losses)
        return self._safe_last(cmo)
    
    # =========================================
    # TREND FEATURES (8)
    # =========================================
    
    def _compute_trend_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule les 8 features de tendance."""
        features = {}
        close = df["close"]
        high = df["high"]
        low = df["low"]
        current_price = close.iloc[-1]
        
        # EMA ratios (prix actuel / EMA)
        ema_5 = EMAIndicator(close, window=self.config.ema_fast).ema_indicator()
        ema_10 = EMAIndicator(close, window=self.config.ema_medium).ema_indicator()
        ema_20 = EMAIndicator(close, window=self.config.ema_slow).ema_indicator()
        
        features["ema_5_ratio"] = self._safe_ratio(current_price, self._safe_last(ema_5))
        features["ema_10_ratio"] = self._safe_ratio(current_price, self._safe_last(ema_10))
        features["ema_20_ratio"] = self._safe_ratio(current_price, self._safe_last(ema_20))
        
        # MACD
        macd = MACD(
            close,
            window_fast=self.config.macd_fast,
            window_slow=self.config.macd_slow,
            window_sign=self.config.macd_signal
        )
        features["macd_line"] = self._safe_last(macd.macd())
        features["macd_signal"] = self._safe_last(macd.macd_signal())
        features["macd_histogram"] = self._safe_last(macd.macd_diff())
        
        # ADX (Average Directional Index)
        adx = ADXIndicator(high, low, close, window=14)
        features["adx"] = self._safe_last(adx.adx())
        
        # Aroon Oscillator
        aroon = AroonIndicator(high, low, window=25)
        aroon_up = self._safe_last(aroon.aroon_up())
        aroon_down = self._safe_last(aroon.aroon_down())
        features["aroon_oscillator"] = aroon_up - aroon_down if not np.isnan(aroon_up) and not np.isnan(aroon_down) else np.nan
        
        return features
    
    # =========================================
    # VOLATILITY FEATURES (6)
    # =========================================
    
    def _compute_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule les 6 features de volatilité."""
        features = {}
        close = df["close"]
        high = df["high"]
        low = df["low"]
        current_price = close.iloc[-1]
        
        # Bollinger Bands
        bb = BollingerBands(
            close,
            window=self.config.bollinger_period,
            window_dev=self.config.bollinger_std
        )
        bb_high = self._safe_last(bb.bollinger_hband())
        bb_low = self._safe_last(bb.bollinger_lband())
        bb_mid = self._safe_last(bb.bollinger_mavg())
        
        # BB Width (largeur relative)
        features["bb_width"] = self._safe_ratio(bb_high - bb_low, bb_mid) if bb_mid else np.nan
        
        # BB Position (0 = bande basse, 1 = bande haute)
        features["bb_position"] = self._calculate_bb_position(current_price, bb_low, bb_high)
        
        # ATR
        atr = AverageTrueRange(high, low, close, window=self.config.atr_period)
        atr_value = self._safe_last(atr.average_true_range())
        features["atr"] = atr_value
        features["atr_percent"] = self._safe_ratio(atr_value, current_price) * 100
        
        # Écart-type des returns
        returns = close.pct_change()
        features["returns_std"] = returns.rolling(window=20).std().iloc[-1] if len(returns) >= 20 else np.nan
        
        # Range High-Low moyen
        hl_range = (high - low) / close
        features["hl_range_avg"] = hl_range.rolling(window=20).mean().iloc[-1] if len(hl_range) >= 20 else np.nan
        
        return features
    
    def _calculate_bb_position(
        self,
        price: float,
        bb_low: float,
        bb_high: float
    ) -> float:
        """Calcule la position dans les Bollinger Bands (0-1)."""
        if np.isnan(bb_low) or np.isnan(bb_high) or bb_high == bb_low:
            return np.nan
        return (price - bb_low) / (bb_high - bb_low)
    
    # =========================================
    # ORDERBOOK FEATURES (8)
    # =========================================
    
    def _compute_orderbook_features(
        self,
        df: pd.DataFrame,
        orderbook: Optional[OrderbookData]
    ) -> Dict[str, float]:
        """Calcule les 8 features de l'orderbook."""
        features = {}
        current_price = df["close"].iloc[-1]
        
        # Si pas d'orderbook, retourner des NaN
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
        
        # Spread bid-ask
        features["spread_percent"] = self._calculate_spread(orderbook)
        
        # Imbalance
        features["orderbook_imbalance"] = self._calculate_imbalance(orderbook)
        
        # Depths
        bid_depth, ask_depth = self._calculate_depths(orderbook)
        features["bid_depth"] = bid_depth
        features["ask_depth"] = ask_depth
        features["depth_ratio"] = self._safe_ratio(bid_depth, ask_depth)
        
        # Pressions
        features["bid_pressure"] = self._calculate_bid_pressure(orderbook, current_price)
        features["ask_pressure"] = self._calculate_ask_pressure(orderbook, current_price)
        
        # Distance au midprice
        features["midprice_distance"] = self._calculate_midprice_distance(orderbook, current_price)
        
        return features
    
    def _calculate_spread(self, orderbook: OrderbookData) -> float:
        """Calcule le spread en pourcentage."""
        if orderbook.best_bid == 0:
            return np.nan
        spread = orderbook.best_ask - orderbook.best_bid
        return (spread / orderbook.best_bid) * 100
    
    def _calculate_imbalance(self, orderbook: OrderbookData) -> float:
        """Calcule l'imbalance bid/ask (-1 à +1)."""
        bid_volume = sum(qty for _, qty in orderbook.bids)
        ask_volume = sum(qty for _, qty in orderbook.asks)
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        return (bid_volume - ask_volume) / total
    
    def _calculate_depths(self, orderbook: OrderbookData) -> tuple:
        """Calcule les profondeurs bid et ask."""
        bid_depth = sum(price * qty for price, qty in orderbook.bids)
        ask_depth = sum(price * qty for price, qty in orderbook.asks)
        return bid_depth, ask_depth
    
    def _calculate_bid_pressure(
        self,
        orderbook: OrderbookData,
        current_price: float
    ) -> float:
        """Calcule la pression acheteuse (volume proche du prix)."""
        threshold = current_price * 0.001  # 0.1% du prix
        pressure = sum(
            qty for price, qty in orderbook.bids
            if current_price - price <= threshold
        )
        return pressure
    
    def _calculate_ask_pressure(
        self,
        orderbook: OrderbookData,
        current_price: float
    ) -> float:
        """Calcule la pression vendeuse (volume proche du prix)."""
        threshold = current_price * 0.001  # 0.1% du prix
        pressure = sum(
            qty for price, qty in orderbook.asks
            if price - current_price <= threshold
        )
        return pressure
    
    def _calculate_midprice_distance(
        self,
        orderbook: OrderbookData,
        current_price: float
    ) -> float:
        """Calcule la distance entre le prix actuel et le midprice."""
        mid = orderbook.mid_price
        if mid == 0:
            return np.nan
        return (current_price - mid) / mid * 100
    
    # =========================================
    # VOLUME FEATURES (5)
    # =========================================
    
    def _compute_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule les 5 features de volume."""
        features = {}
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        
        # Volume relatif (vs moyenne 20 périodes)
        volume_sma = volume.rolling(window=20).mean()
        features["volume_relative"] = self._safe_ratio(volume.iloc[-1], volume_sma.iloc[-1])
        
        # OBV slope (pente de l'OBV sur 5 périodes)
        obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        features["obv_slope"] = self._calculate_slope(obv, 5)
        
        # Volume delta (approximation buy vs sell)
        features["volume_delta"] = self._calculate_volume_delta(df)
        
        # VWAP distance
        features["vwap_distance"] = self._calculate_vwap_distance(df)
        
        # A/D Line (Accumulation/Distribution)
        ad = AccDistIndexIndicator(high, low, close, volume).acc_dist_index()
        features["ad_line"] = self._calculate_slope(ad, 5)
        
        return features
    
    def _calculate_slope(self, series: pd.Series, period: int) -> float:
        """Calcule la pente normalisée d'une série."""
        if len(series) < period:
            return np.nan
        
        y = series.iloc[-period:].values
        x = np.arange(period)
        
        # Régression linéaire simple
        if np.all(np.isnan(y)):
            return np.nan
        
        # Retirer les NaN
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        
        slope = np.polyfit(x[mask], y[mask], 1)[0]
        
        # Normaliser par la moyenne
        mean_val = np.nanmean(y)
        if mean_val != 0:
            return slope / abs(mean_val)
        return slope
    
    def _calculate_volume_delta(self, df: pd.DataFrame) -> float:
        """
        Estime le delta de volume (buy vs sell).
        
        Utilise la position du close dans le range high-low.
        """
        close = df["close"].iloc[-1]
        high = df["high"].iloc[-1]
        low = df["low"].iloc[-1]
        volume = df["volume"].iloc[-1]
        
        if high == low:
            return 0.0
        
        # Position du close dans le range (0 = bas, 1 = haut)
        position = (close - low) / (high - low)
        
        # Delta = volume * (2 * position - 1)
        # Si close en haut → delta positif (plus d'achats)
        # Si close en bas → delta négatif (plus de ventes)
        return volume * (2 * position - 1)
    
    def _calculate_vwap_distance(self, df: pd.DataFrame) -> float:
        """Calcule la distance au VWAP en pourcentage."""
        # VWAP simplifié sur les données disponibles
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        
        current_price = df["close"].iloc[-1]
        current_vwap = vwap.iloc[-1]
        
        if current_vwap == 0 or np.isnan(current_vwap):
            return np.nan
        
        return (current_price - current_vwap) / current_vwap * 100
    
    # =========================================
    # PRICE ACTION FEATURES (5)
    # =========================================
    
    def _compute_price_action_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule les 5 features de price action."""
        features = {}
        close = df["close"]
        open_price = df["open"]
        high = df["high"]
        low = df["low"]
        
        # Returns sur différentes périodes
        for period in self.config.returns_periods:
            key = f"returns_{period}m"
            features[key] = self._calculate_returns(close, period)
        
        # Nombre de chandeliers verts consécutifs
        features["consecutive_green"] = self._count_consecutive_green(df)
        
        # Ratio corps/mèches de la dernière bougie
        features["candle_body_ratio"] = self._calculate_candle_body_ratio(
            open_price.iloc[-1],
            close.iloc[-1],
            high.iloc[-1],
            low.iloc[-1]
        )
        
        return features
    
    def _calculate_returns(self, close: pd.Series, period: int) -> float:
        """Calcule le rendement sur N périodes."""
        if len(close) <= period:
            return np.nan
        return (close.iloc[-1] - close.iloc[-period - 1]) / close.iloc[-period - 1] * 100
    
    def _count_consecutive_green(self, df: pd.DataFrame) -> int:
        """Compte le nombre de chandeliers verts consécutifs."""
        is_green = df["close"] > df["open"]
        
        count = 0
        for val in reversed(is_green.values):
            if val:
                count += 1
            else:
                break
        
        return count
    
    def _calculate_candle_body_ratio(
        self,
        open_price: float,
        close: float,
        high: float,
        low: float
    ) -> float:
        """
        Calcule le ratio corps/range total de la bougie.
        
        1.0 = corps complet (pas de mèches)
        0.0 = doji (tout en mèches)
        """
        total_range = high - low
        if total_range == 0:
            return 0.0
        
        body = abs(close - open_price)
        return body / total_range
    
    # =========================================
    # UTILS
    # =========================================
    
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


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def get_feature_names() -> List[str]:
    """Retourne la liste des noms de features."""
    return FeatureEngine.FEATURE_NAMES.copy()


def compute_features_for_symbol(
    df: pd.DataFrame,
    orderbook: Optional[OrderbookData] = None,
    symbol: str = "UNKNOWN",
    config: Optional[FeatureConfig] = None
) -> Dict[str, float]:
    """
    Fonction utilitaire pour calculer les features d'une paire.
    
    Args:
        df: DataFrame OHLCV
        orderbook: Données orderbook (optionnel)
        symbol: Symbole de la paire
        config: Configuration (optionnel)
        
    Returns:
        Dictionnaire des features
    """
    engine = FeatureEngine(config)
    feature_set = engine.compute_features(df, orderbook, symbol)
    return feature_set.to_dict()