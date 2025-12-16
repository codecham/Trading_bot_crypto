# cryptoscalper/data/dataset.py
"""
Module de préparation du dataset pour l'entraînement ML.

Responsabilités :
- Calcul des features sur données historiques
- Création des labels (hausse ≥X% en Y minutes)
- Split temporel train/validation/test
- Vérification et équilibrage des classes

Usage:
    builder = DatasetBuilder()
    dataset = builder.build_from_file("data_cache/BTCUSDT_1m.parquet")
    
    # Ou depuis plusieurs fichiers
    dataset = builder.build_from_multiple(["BTCUSDT", "ETHUSDT"])
    
    # Split
    train, val, test = dataset.split_temporal()
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np

from cryptoscalper.data.features import (
    FeatureEngine,
    FeatureConfig,
    OrderbookData,
    get_feature_names,
)
from cryptoscalper.data.historical import (
    HistoricalDataDownloader,
    get_cached_data_path,
    load_cached_data,
)
from cryptoscalper.config.constants import (
    PREDICTION_HORIZON_MINUTES,
    MIN_PRICE_CHANGE_FOR_LABEL,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    DATA_DIR,
)
from cryptoscalper.utils.logger import logger


# ============================================
# CONSTANTES
# ============================================

# Minimum de lignes requises pour calculer les features
MIN_ROWS_FOR_FEATURES = 50

# Colonnes requises dans les données brutes
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


# ============================================
# DATACLASSES
# ============================================

@dataclass
class LabelConfig:
    """Configuration pour la création des labels basés sur SL/TP."""
    
    # Horizon maximum de prédiction (en minutes) - timeout
    horizon_minutes: int = PREDICTION_HORIZON_MINUTES  # 15 min recommandé
    
    # Seuil de Take Profit (ex: 0.005 = +0.5%)
    threshold_percent: float = MIN_PRICE_CHANGE_FOR_LABEL  # TP
    
    # Seuil de Stop Loss (ex: 0.003 = -0.3%)
    stop_loss_percent: float = 0.003  # SL
    
    # Mode de labeling: 'sltp' (nouveau) ou 'legacy' (ancien)
    mode: str = 'sltp'


@dataclass
class SplitConfig:
    """Configuration pour le split train/val/test."""
    
    train_ratio: float = TRAIN_RATIO      # 0.70
    val_ratio: float = VALIDATION_RATIO   # 0.15
    test_ratio: float = TEST_RATIO        # 0.15
    
    def __post_init__(self):
        """Vérifie que les ratios somment à 1."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Les ratios doivent sommer à 1.0, got {total}")


@dataclass
class DatasetStats:
    """Statistiques du dataset."""
    
    total_rows: int = 0
    valid_rows: int = 0  # Après suppression des NaN
    label_1_count: int = 0
    label_0_count: int = 0
    feature_count: int = 0
    symbols_count: int = 0
    date_start: Optional[datetime] = None
    date_end: Optional[datetime] = None
    nan_features_dropped: int = 0
    
    @property
    def label_ratio(self) -> float:
        """Ratio label_1 / total."""
        if self.valid_rows == 0:
            return 0.0
        return self.label_1_count / self.valid_rows
    
    @property
    def is_balanced(self) -> bool:
        """Vérifie si les classes sont équilibrées (ratio entre 0.3 et 0.7)."""
        return 0.3 <= self.label_ratio <= 0.7
    
    def summary(self) -> str:
        """Retourne un résumé des stats."""
        return (
            f"Dataset: {self.valid_rows:,} lignes, {self.feature_count} features\n"
            f"Labels: {self.label_1_count:,} positifs ({self.label_ratio:.1%}), "
            f"{self.label_0_count:,} négatifs ({1-self.label_ratio:.1%})\n"
            f"Période: {self.date_start} → {self.date_end}\n"
            f"Équilibré: {'✅ Oui' if self.is_balanced else '⚠️ Non'}"
        )


@dataclass
class PreparedDataset:
    """Dataset préparé pour l'entraînement."""
    
    # Données
    features: pd.DataFrame  # X
    labels: pd.Series       # y
    timestamps: pd.Series   # Pour le split temporel
    
    # Métadonnées
    symbols: List[str]
    feature_names: List[str]
    stats: DatasetStats
    label_config: LabelConfig
    
    def __len__(self) -> int:
        return len(self.features)
    
    def split_temporal(
        self,
        config: Optional[SplitConfig] = None
    ) -> Tuple["PreparedDataset", "PreparedDataset", "PreparedDataset"]:
        """
        Split temporel du dataset.
        
        IMPORTANT: Ne pas utiliser de split aléatoire pour les séries temporelles !
        
        Args:
            config: Configuration du split
            
        Returns:
            (train, validation, test) datasets
        """
        config = config or SplitConfig()
        
        n = len(self)
        train_end = int(n * config.train_ratio)
        val_end = int(n * (config.train_ratio + config.val_ratio))
        
        # Split
        train = self._slice(0, train_end)
        val = self._slice(train_end, val_end)
        test = self._slice(val_end, n)
        
        logger.info(
            f"📊 Split temporel: "
            f"Train={len(train):,} ({config.train_ratio:.0%}), "
            f"Val={len(val):,} ({config.val_ratio:.0%}), "
            f"Test={len(test):,} ({config.test_ratio:.0%})"
        )
        
        return train, val, test
    
    def _slice(self, start: int, end: int) -> "PreparedDataset":
        """Crée un sous-dataset."""
        features = self.features.iloc[start:end].reset_index(drop=True)
        labels = self.labels.iloc[start:end].reset_index(drop=True)
        timestamps = self.timestamps.iloc[start:end].reset_index(drop=True)
        
        # Recalculer les stats
        stats = DatasetStats(
            total_rows=len(features),
            valid_rows=len(features),
            label_1_count=int(labels.sum()),
            label_0_count=int(len(labels) - labels.sum()),
            feature_count=len(self.feature_names),
            symbols_count=len(self.symbols),
            date_start=timestamps.iloc[0] if len(timestamps) > 0 else None,
            date_end=timestamps.iloc[-1] if len(timestamps) > 0 else None,
        )
        
        return PreparedDataset(
            features=features,
            labels=labels,
            timestamps=timestamps,
            symbols=self.symbols,
            feature_names=self.feature_names,
            stats=stats,
            label_config=self.label_config
        )
    
    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convertit en arrays numpy pour sklearn/xgboost."""
        X = self.features.values
        y = self.labels.values
        return X, y
    
    def save(self, path: Path) -> None:
        """Sauvegarde le dataset en parquet."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combiner features et labels
        df = self.features.copy()
        df["label"] = self.labels
        df["timestamp"] = self.timestamps
        
        df.to_parquet(path, index=False)
        logger.info(f"💾 Dataset sauvegardé: {path}")
    
    @classmethod
    def load(cls, path: Path, label_config: Optional[LabelConfig] = None) -> "PreparedDataset":
        """Charge un dataset depuis un fichier parquet."""
        df = pd.read_parquet(path)
        
        labels = df["label"]
        timestamps = df["timestamp"]
        features = df.drop(columns=["label", "timestamp"])
        
        feature_names = list(features.columns)
        
        stats = DatasetStats(
            total_rows=len(df),
            valid_rows=len(df),
            label_1_count=int(labels.sum()),
            label_0_count=int(len(labels) - labels.sum()),
            feature_count=len(feature_names),
            date_start=timestamps.iloc[0] if len(timestamps) > 0 else None,
            date_end=timestamps.iloc[-1] if len(timestamps) > 0 else None,
        )
        
        logger.info(f"📂 Dataset chargé: {path} ({len(df):,} lignes)")
        
        return cls(
            features=features,
            labels=labels,
            timestamps=timestamps,
            symbols=["unknown"],
            feature_names=feature_names,
            stats=stats,
            label_config=label_config or LabelConfig()
        )


# ============================================
# DATASET BUILDER
# ============================================

class DatasetBuilder:
    """
    Constructeur de dataset pour l'entraînement ML.
    
    Prend des données OHLCV brutes et produit un dataset
    avec features calculées et labels.
    
    Usage:
        builder = DatasetBuilder()
        
        # Depuis un fichier
        dataset = builder.build_from_file("data/BTCUSDT_1m.parquet")
        
        # Depuis plusieurs symboles
        dataset = builder.build_from_symbols(["BTCUSDT", "ETHUSDT"])
        
        # Split
        train, val, test = dataset.split_temporal()
    """
    
    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        label_config: Optional[LabelConfig] = None
    ):
        """
        Initialise le builder.
        
        Args:
            feature_config: Configuration des features
            label_config: Configuration des labels
        """
        self._feature_engine = FeatureEngine(feature_config)
        self._label_config = label_config or LabelConfig()
    
    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> PreparedDataset:
        """
        Construit un dataset depuis un DataFrame OHLCV.
        
        Args:
            df: DataFrame avec colonnes OHLCV
            symbol: Nom du symbole
            
        Returns:
            PreparedDataset prêt pour l'entraînement
        """
        logger.info(f"🔧 Construction dataset {symbol} ({len(df):,} lignes)...")
        
        # Vérifier les colonnes requises
        self._validate_dataframe(df)
        
        # Créer les labels
        df_with_labels = self._create_labels(df)
        
        # Calculer les features
        df_with_features = self._compute_all_features(df_with_labels)
        
        # Nettoyer (supprimer NaN)
        df_clean, nan_dropped = self._clean_dataset(df_with_features)
        
        # Extraire features, labels, timestamps
        feature_names = get_feature_names()
        features = df_clean[feature_names]
        labels = df_clean["label"].astype(int)
        timestamps = df_clean["open_time"] if "open_time" in df_clean.columns else pd.Series(range(len(df_clean)))
        
        # Calculer les stats
        stats = self._compute_stats(df, df_clean, features, labels, nan_dropped)
        
        logger.info(f"✅ Dataset {symbol}: {stats.summary()}")
        
        return PreparedDataset(
            features=features.reset_index(drop=True),
            labels=labels.reset_index(drop=True),
            timestamps=timestamps.reset_index(drop=True),
            symbols=[symbol],
            feature_names=feature_names,
            stats=stats,
            label_config=self._label_config
        )
    
    def build_from_file(
        self,
        path: Path,
        symbol: Optional[str] = None
    ) -> PreparedDataset:
        """
        Construit un dataset depuis un fichier parquet/csv.
        
        Args:
            path: Chemin du fichier
            symbol: Nom du symbole (déduit du nom de fichier si non fourni)
            
        Returns:
            PreparedDataset
        """
        path = Path(path)
        
        # Déduire le symbole du nom de fichier
        if symbol is None:
            symbol = path.stem.split("_")[0].upper()
        
        # Charger les données
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, parse_dates=["open_time", "close_time"])
        
        logger.info(f"📂 Chargé {path.name}: {len(df):,} lignes")
        
        return self.build_from_dataframe(df, symbol)
    
    def build_from_symbols(
        self,
        symbols: List[str],
        data_dir: Path = Path(DATA_DIR)
    ) -> PreparedDataset:
        """
        Construit un dataset combiné depuis plusieurs symboles.
        
        Args:
            symbols: Liste des symboles
            data_dir: Dossier contenant les fichiers de données
            
        Returns:
            PreparedDataset combiné
        """
        logger.info(f"📦 Construction dataset multi-symboles ({len(symbols)} paires)...")
        
        all_features = []
        all_labels = []
        all_timestamps = []
        total_nan_dropped = 0
        
        for symbol in symbols:
            path = get_cached_data_path(symbol, data_dir)
            
            if not path.exists():
                logger.warning(f"⚠️ Fichier non trouvé pour {symbol}: {path}")
                continue
            
            try:
                dataset = self.build_from_file(path, symbol)
                
                all_features.append(dataset.features)
                all_labels.append(dataset.labels)
                all_timestamps.append(dataset.timestamps)
                total_nan_dropped += dataset.stats.nan_features_dropped
                
            except Exception as e:
                logger.error(f"❌ Erreur {symbol}: {e}")
        
        if not all_features:
            raise ValueError("Aucun dataset valide construit")
        
        # Combiner
        features = pd.concat(all_features, ignore_index=True)
        labels = pd.concat(all_labels, ignore_index=True)
        timestamps = pd.concat(all_timestamps, ignore_index=True)
        
        # Trier par timestamp
        sort_idx = timestamps.argsort()
        features = features.iloc[sort_idx].reset_index(drop=True)
        labels = labels.iloc[sort_idx].reset_index(drop=True)
        timestamps = timestamps.iloc[sort_idx].reset_index(drop=True)
        
        # Stats
        stats = DatasetStats(
            total_rows=len(features),
            valid_rows=len(features),
            label_1_count=int(labels.sum()),
            label_0_count=int(len(labels) - labels.sum()),
            feature_count=len(get_feature_names()),
            symbols_count=len(symbols),
            date_start=timestamps.iloc[0] if len(timestamps) > 0 else None,
            date_end=timestamps.iloc[-1] if len(timestamps) > 0 else None,
            nan_features_dropped=total_nan_dropped
        )
        
        logger.info(f"✅ Dataset combiné: {stats.summary()}")
        
        return PreparedDataset(
            features=features,
            labels=labels,
            timestamps=timestamps,
            symbols=symbols,
            feature_names=get_feature_names(),
            stats=stats,
            label_config=self._label_config
        )
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Vérifie que le DataFrame a les colonnes requises."""
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")
        
        if len(df) < MIN_ROWS_FOR_FEATURES:
            raise ValueError(
                f"Pas assez de lignes: {len(df)} < {MIN_ROWS_FOR_FEATURES}"
            )
    
    def _create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les labels SL/TP de manière VECTORISÉE (rapide).
        
        Label = 1 si TP atteint avant SL dans les N prochaines minutes.
        """
        import numpy as np
        
        df = df.copy()
        
        horizon = self._label_config.horizon_minutes
        tp_threshold = self._label_config.threshold_percent
        sl_threshold = self._label_config.stop_loss_percent
        
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)
        
        # Pré-allouer les arrays
        labels = np.zeros(n, dtype=np.int32)
        future_returns = np.zeros(n, dtype=np.float64)
        
        # Pour chaque bougie future (1 à horizon), calculer si TP/SL atteint
        tp_reached_at = np.full(n, np.inf)  # Quand le TP est atteint
        sl_reached_at = np.full(n, np.inf)  # Quand le SL est atteint
        
        for j in range(1, horizon + 1):
            # Indices valides (pas de débordement)
            valid = np.arange(n - j)
            future_idx = valid + j
            
            # Prix d'entrée et seuils
            entry_prices = closes[valid]
            tp_prices = entry_prices * (1 + tp_threshold)
            sl_prices = entry_prices * (1 - sl_threshold)
            
            # High et Low futurs
            future_highs = highs[future_idx]
            future_lows = lows[future_idx]
            
            # TP atteint à cette bougie ?
            tp_hit = (future_highs >= tp_prices) & (tp_reached_at[valid] == np.inf)
            tp_reached_at[valid[tp_hit]] = j
            
            # SL atteint à cette bougie ?
            sl_hit = (future_lows <= sl_prices) & (sl_reached_at[valid] == np.inf)
            sl_reached_at[valid[sl_hit]] = j
            
            # Meilleur return
            current_returns = (future_highs - entry_prices) / entry_prices
            future_returns[valid] = np.maximum(future_returns[valid], current_returns)
        
        # Label = 1 si TP atteint AVANT SL (ou SL jamais atteint)
        labels = (tp_reached_at < sl_reached_at).astype(np.int32)
        
        df["label"] = labels
        df["future_return"] = future_returns
        
        return df

    
    def _compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule toutes les features de manière VECTORISÉE (rapide)."""
        import ta
        import numpy as np
        
        logger.debug("Calcul des features (vectorisé)...")
        
        df = df.copy()
        
        # === MOMENTUM ===
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_7'] = ta.momentum.rsi(df['close'], window=7)
        
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
        df['roc_5'] = ta.momentum.roc(df['close'], window=5)
        df['roc_10'] = ta.momentum.roc(df['close'], window=10)
        
        # Momentum simple
        df['momentum_5'] = df['close'].diff(5)
        
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        
        # CMO (Chande Momentum Oscillator) - calcul manuel
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).sum()
        loss = (-delta.where(delta < 0, 0)).rolling(14).sum()
        df['cmo'] = 100 * (gain - loss) / (gain + loss)
        
        # === TREND ===
        df['ema_5'] = ta.trend.ema_indicator(df['close'], window=5)
        df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        
        df['ema_5_ratio'] = df['close'] / df['ema_5']
        df['ema_10_ratio'] = df['close'] / df['ema_10']
        df['ema_20_ratio'] = df['close'] / df['ema_20']
        
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd_line'] = macd.macd() / df['close'] * 100
        df['macd_signal'] = macd.macd_signal() / df['close'] * 100
        df['macd_histogram'] = macd.macd_diff() / df['close'] * 100
        
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        
        aroon = ta.trend.AroonIndicator(df['high'], df['low'], window=25)
        df['aroon_oscillator'] = aroon.aroon_indicator()
        
        # === VOLATILITY ===
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        df['bb_position'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband()) - 0.5
        
        atr_raw = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr_raw / df['close'] * 100  # Normalisé
        df['atr_percent'] = df['atr']  # Identique
        
        df['returns_std'] = df['close'].pct_change().rolling(20).std()
        df['hl_range_avg'] = ((df['high'] - df['low']) / df['close']).rolling(20).mean()
        
        # === VOLUME ===
        df['volume_relative'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # OBV slope
        obv = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_slope'] = obv.diff(5) / 5
        
        # Volume delta (approximation)
        df['volume_delta'] = df['volume'] * np.sign(df['close'].diff())
        
        # VWAP distance
        vwap = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['vwap_distance'] = (df['close'] - vwap) / vwap * 100
        
        # A/D Line
        df['ad_line'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
        df['ad_line'] = df['ad_line'].diff(5)  # Variation sur 5 périodes
        
        # === PRICE ACTION ===
        df['returns_1m'] = df['close'].pct_change(1) * 100
        df['returns_5m'] = df['close'].pct_change(5) * 100
        df['returns_15m'] = df['close'].pct_change(15) * 100
        
        # Bougies consécutives vertes
        is_green = (df['close'] > df['open']).astype(int)
        df['consecutive_green'] = is_green.groupby((is_green != is_green.shift()).cumsum()).cumsum() * is_green
        
        # Ratio corps/mèche
        body = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        df['candle_body_ratio'] = body / total_range.replace(0, np.nan)
        
        # === ORDERBOOK (placeholders pour données historiques) ===
        df['spread_percent'] = 0.0
        df['orderbook_imbalance'] = 0.0
        df['bid_depth'] = 0.0
        df['ask_depth'] = 0.0
        df['depth_ratio'] = 0.0
        df['bid_pressure'] = 0.0
        df['ask_pressure'] = 0.0
        df['midprice_distance'] = 0.0
        
        # Supprimer les colonnes temporaires
        for col in ['ema_5', 'ema_10', 'ema_20']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        
        logger.debug(f"✅ Features calculées: {len(df):,} lignes")
        
        return df
    
    def _clean_dataset(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, int]:
        """Supprime les lignes avec NaN et les dernières lignes sans label."""
        import warnings
        
        original_len = len(df)
        
        # Supprimer les lignes où le label est NaN (fin du dataset)
        df = df.dropna(subset=["label"])
        
        # Supprimer les lignes avec trop de features NaN
        feature_names = get_feature_names()
        features_present = df[feature_names].notna().sum(axis=1)
        min_features = len(feature_names) * 0.8  # Au moins 80% des features
        df = df[features_present >= min_features]
        
        # Remplir les NaN restants par la médiane (avec suppression des warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for col in feature_names:
                if df[col].isna().any():
                    median_val = df[col].median()
                    # Si la médiane est NaN (colonne vide), utiliser 0
                    if pd.isna(median_val):
                        median_val = 0.0
                    df[col] = df[col].fillna(median_val)
        
        nan_dropped = original_len - len(df)
        
        return df, nan_dropped
    
    def _compute_stats(
        self,
        df_original: pd.DataFrame,
        df_clean: pd.DataFrame,
        features: pd.DataFrame,
        labels: pd.Series,
        nan_dropped: int
    ) -> DatasetStats:
        """Calcule les statistiques du dataset."""
        return DatasetStats(
            total_rows=len(df_original),
            valid_rows=len(df_clean),
            label_1_count=int(labels.sum()),
            label_0_count=int(len(labels) - labels.sum()),
            feature_count=len(features.columns),
            symbols_count=1,
            date_start=df_clean["open_time"].iloc[0] if "open_time" in df_clean.columns else None,
            date_end=df_clean["open_time"].iloc[-1] if "open_time" in df_clean.columns else None,
            nan_features_dropped=nan_dropped
        )


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def prepare_dataset(
    symbol: str,
    data_dir: Path = Path(DATA_DIR),
    label_config: Optional[LabelConfig] = None
) -> PreparedDataset:
    """
    Fonction utilitaire pour préparer rapidement un dataset.
    
    Args:
        symbol: Symbole (ex: "BTCUSDT")
        data_dir: Dossier des données
        label_config: Configuration des labels
        
    Returns:
        PreparedDataset
    """
    builder = DatasetBuilder(label_config=label_config)
    path = get_cached_data_path(symbol, data_dir)
    return builder.build_from_file(path, symbol)


def analyze_class_balance(labels: pd.Series) -> Dict[str, float]:
    """
    Analyse l'équilibre des classes.
    
    Args:
        labels: Série de labels (0/1)
        
    Returns:
        Dict avec les statistiques d'équilibre
    """
    total = len(labels)
    positive = int(labels.sum())  # Convertir en int Python natif
    negative = total - positive
    
    positive_ratio = positive / total if total > 0 else 0
    
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_ratio": positive_ratio,
        "negative_ratio": negative / total if total > 0 else 0,
        "imbalance_ratio": max(positive, negative) / min(positive, negative) if min(positive, negative) > 0 else float('inf'),
        "is_balanced": bool(0.3 <= positive_ratio <= 0.7) if total > 0 else False  # Convertir en bool Python natif
    }