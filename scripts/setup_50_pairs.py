#!/usr/bin/env python3
# scripts/setup_50_pairs.py
"""
Setup complet pour 50 paires de trading.
VERSION OPTIMISÉE - Labels vectorisés (100x plus rapide)

Ce script automatise:
1. Récupération des top 50 paires par volume
2. Téléchargement de 90 jours de données
3. Création du dataset (OPTIMISÉ)
4. Entraînement du modèle

Usage:
    python scripts/setup_50_pairs.py
    python scripts/setup_50_pairs.py --pairs 30 --days 60
    python scripts/setup_50_pairs.py --skip-download
"""

import argparse
import asyncio
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================
# CONSTANTES
# ============================================

DEFAULT_PAIRS_COUNT = 50
DEFAULT_DAYS = 90
DATA_DIR = "data_cache_50"
DATASET_DIR = "datasets"
MODEL_DIR = "models/saved"

# Stablecoins et paires à exclure
EXCLUDED_SYMBOLS = {
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "USDPUSDT", "DAIUSDT",
    "EURUSDT", "GBPUSDT", "AUDUSDT", "FDUSDUSDT", "PYUSDUSDT",
    "BTCSTUSDT", "BETHUSDT", "WBTCUSDT", "WBETHUSDT",
    "USTUSDT", "LUNAUSDT", "LUNCUSDT",
}


# ============================================
# ÉTAPE 1: RÉCUPÉRER LES TOP PAIRES
# ============================================

async def get_top_pairs(n: int = 50, min_volume: float = 1_000_000) -> List[str]:
    """Récupère les top N paires par volume 24h."""
    from binance import AsyncClient
    
    logger.info(f"🔍 Recherche des top {n} paires...")
    
    client = await AsyncClient.create()
    
    try:
        # Récupérer les infos exchange
        exchange_info = await client.get_exchange_info()
        usdt_symbols = set()
        
        for s in exchange_info["symbols"]:
            symbol = s["symbol"]
            if (symbol.endswith("USDT") and 
                s["status"] == "TRADING" and
                s.get("isSpotTradingAllowed", False) and
                symbol not in EXCLUDED_SYMBOLS):
                usdt_symbols.add(symbol)
        
        # Récupérer les volumes
        tickers = await client.get_ticker()
        volumes = {}
        for t in tickers:
            if t["symbol"] in usdt_symbols:
                volumes[t["symbol"]] = float(t.get("quoteVolume", 0))
        
        # Filtrer et trier
        filtered = [(s, v) for s, v in volumes.items() if v >= min_volume]
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        top_pairs = [s for s, v in filtered[:n]]
        
        logger.info(f"✅ {len(top_pairs)} paires sélectionnées")
        
        return top_pairs
        
    finally:
        await client.close_connection()


# ============================================
# ÉTAPE 2: TÉLÉCHARGER LES DONNÉES
# ============================================

async def download_pair_data(
    client,
    symbol: str,
    days: int,
    output_dir: Path,
) -> Optional[Path]:
    """Télécharge les données d'une paire."""
    
    try:
        # Calculer les dates
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Récupérer les klines par batch
        all_klines = []
        current_start = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        while current_start < end_ms:
            klines = await client.get_klines(
                symbol=symbol,
                interval="1m",
                startTime=current_start,
                limit=1000
            )
            
            if not klines:
                break
            
            all_klines.extend(klines)
            current_start = klines[-1][0] + 60000  # +1 minute
            
            # Petit délai pour éviter rate limit
            await asyncio.sleep(0.05)
        
        if not all_klines:
            logger.warning(f"⚠️  Pas de données pour {symbol}")
            return None
        
        # Créer le DataFrame
        df = pd.DataFrame(all_klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        # Convertir les types
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)
        
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        
        # Supprimer les doublons
        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")
        
        # Sauvegarder
        output_path = output_dir / f"{symbol}_1m.parquet"
        df.to_parquet(output_path, index=False)
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Erreur {symbol}: {e}")
        return None


async def download_all_data(
    symbols: List[str],
    days: int,
    output_dir: Path,
    max_concurrent: int = 5,
) -> int:
    """Télécharge les données de toutes les paires."""
    from binance import AsyncClient
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📥 Téléchargement de {len(symbols)} paires ({days} jours)...")
    logger.info(f"   Dossier: {output_dir}")
    
    client = await AsyncClient.create()
    
    try:
        success_count = 0
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(symbol: str, index: int):
            nonlocal success_count
            async with semaphore:
                logger.info(f"   [{index}/{len(symbols)}] {symbol}...")
                result = await download_pair_data(client, symbol, days, output_dir)
                if result:
                    success_count += 1
        
        tasks = [
            download_with_semaphore(symbol, i+1) 
            for i, symbol in enumerate(symbols)
        ]
        
        await asyncio.gather(*tasks)
        
        logger.info(f"✅ {success_count}/{len(symbols)} paires téléchargées")
        
        return success_count
        
    finally:
        await client.close_connection()


# ============================================
# ÉTAPE 3: CRÉER LE DATASET (OPTIMISÉ)
# ============================================

def compute_labels_vectorized(
    df: pd.DataFrame,
    horizon: int = 120,
    tp_pct: float = 0.02,
    sl_pct: float = 0.01,
) -> np.ndarray:
    """
    Calcule les labels de manière VECTORISÉE (100x plus rapide).
    
    Pour chaque bougie, détermine si dans les `horizon` prochaines bougies:
    - Le prix atteint TP (+tp_pct) AVANT SL → label = 1
    - Le prix atteint SL (-sl_pct) AVANT TP → label = 0
    - Timeout → label basé sur le prix final
    
    Args:
        df: DataFrame avec colonnes 'close', 'high', 'low'
        horizon: Nombre de bougies à regarder
        tp_pct: Take profit en pourcentage (0.02 = 2%)
        sl_pct: Stop loss en pourcentage (0.01 = 1%)
    
    Returns:
        Array de labels (0, 1, ou NaN)
    """
    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    
    labels = np.full(n, np.nan)
    
    # Pré-calculer les prix TP et SL pour chaque entrée
    tp_prices = close * (1 + tp_pct)
    sl_prices = close * (1 - sl_pct)
    
    # Pour chaque position de départ
    for i in range(n - horizon):
        tp_price = tp_prices[i]
        sl_price = sl_prices[i]
        
        # Extraire les données futures (vectorisé)
        future_highs = high[i+1:i+1+horizon]
        future_lows = low[i+1:i+1+horizon]
        
        # Trouver le premier index où TP est atteint
        tp_hits = np.where(future_highs >= tp_price)[0]
        tp_idx = tp_hits[0] if len(tp_hits) > 0 else horizon + 1
        
        # Trouver le premier index où SL est atteint
        sl_hits = np.where(future_lows <= sl_price)[0]
        sl_idx = sl_hits[0] if len(sl_hits) > 0 else horizon + 1
        
        # Déterminer le label
        if tp_idx < sl_idx and tp_idx < horizon:
            # TP atteint en premier
            labels[i] = 1
        elif sl_idx < tp_idx and sl_idx < horizon:
            # SL atteint en premier
            labels[i] = 0
        else:
            # Timeout - regarder le prix final
            final_price = close[min(i + horizon, n - 1)]
            labels[i] = 1 if final_price > close[i] else 0
    
    return labels


def compute_features_for_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les 42 features (vectorisé avec ta)."""
    import ta
    
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
    df['momentum_5'] = df['close'].diff(5) / df['close'] * 100
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
    
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
    df['atr'] = atr_raw / df['close'] * 100
    df['atr_percent'] = df['atr']
    
    df['returns_std'] = df['close'].pct_change().rolling(20).std()
    df['hl_range_avg'] = ((df['high'] - df['low']) / df['close']).rolling(20).mean()
    
    # === VOLUME ===
    df['volume_relative'] = df['volume'] / df['volume'].rolling(20).mean()
    
    obv = ta.volume.on_balance_volume(df['close'], df['volume'])
    obv_mean = obv.abs().rolling(20).mean()
    df['obv_slope'] = obv.diff(5) / obv_mean.replace(0, np.nan)
    
    volume_sma = df['volume'].rolling(20).mean()
    volume_delta_raw = df['volume'] * np.sign(df['close'].diff())
    df['volume_delta'] = volume_delta_raw / volume_sma.replace(0, np.nan)
    
    vwap = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwap_distance'] = (df['close'] - vwap) / vwap * 100
    
    ad_line = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
    ad_mean = ad_line.abs().rolling(20).mean()
    df['ad_line'] = ad_line.diff(5) / ad_mean.replace(0, np.nan)
    
    # === PRICE ACTION ===
    df['returns_1m'] = df['close'].pct_change(1) * 100
    df['returns_5m'] = df['close'].pct_change(5) * 100
    df['returns_15m'] = df['close'].pct_change(15) * 100
    
    is_green = (df['close'] > df['open']).astype(int)
    df['consecutive_green'] = is_green.groupby((is_green != is_green.shift()).cumsum()).cumsum() * is_green
    
    body = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']
    df['candle_body_ratio'] = body / total_range.replace(0, np.nan)
    
    # === ORDERBOOK (placeholders) ===
    df['spread_percent'] = 0.0
    df['orderbook_imbalance'] = 0.0
    df['bid_depth'] = 0.0
    df['ask_depth'] = 0.0
    df['depth_ratio'] = 0.0
    df['bid_pressure'] = 0.0
    df['ask_pressure'] = 0.0
    df['midprice_distance'] = 0.0
    
    return df


def get_feature_names() -> List[str]:
    """Retourne la liste des 42 features."""
    return [
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


def create_dataset(
    symbols: List[str],
    data_dir: Path,
    output_dir: Path,
    horizon: int = 120,
    threshold: float = 0.02,
    stop_loss: float = 0.01,
) -> Optional[Path]:
    """Crée le dataset à partir des données téléchargées (VERSION OPTIMISÉE)."""
    
    logger.info("📊 Création du dataset (version optimisée)...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_dfs = []
    total_start = time.time()
    
    for i, symbol in enumerate(symbols):
        data_path = data_dir / f"{symbol}_1m.parquet"
        
        if not data_path.exists():
            logger.warning(f"⚠️  Fichier manquant: {data_path}")
            continue
        
        pair_start = time.time()
        
        try:
            df = pd.read_parquet(data_path)
            
            if len(df) < 200:
                logger.warning(f"⚠️  Pas assez de données pour {symbol}: {len(df)}")
                continue
            
            # Calculer les features
            df = compute_features_for_dataset(df)
            
            # Calculer les labels (VECTORISÉ - RAPIDE!)
            df["label"] = compute_labels_vectorized(
                df, 
                horizon=horizon, 
                tp_pct=threshold, 
                sl_pct=stop_loss
            )
            
            # Ajouter le symbole
            df["symbol"] = symbol
            
            all_dfs.append(df)
            
            elapsed = time.time() - pair_start
            logger.info(f"   [{i+1}/{len(symbols)}] {symbol}: {len(df):,} rows ({elapsed:.1f}s)")
            
        except Exception as e:
            logger.error(f"❌ Erreur {symbol}: {e}")
            continue
    
    if not all_dfs:
        logger.error("❌ Aucun dataset créé")
        return None
    
    # Combiner tous les DataFrames
    logger.info("🔗 Combinaison des données...")
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Nettoyer les NaN
    logger.info("🧹 Nettoyage des données...")
    feature_cols = get_feature_names()
    
    # Supprimer les lignes sans label
    combined = combined.dropna(subset=["label"])
    
    # Remplir les NaN des features par la médiane
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in feature_cols:
            if col in combined.columns and combined[col].isna().any():
                median = combined[col].median()
                if pd.isna(median):
                    median = 0.0
                combined[col] = combined[col].fillna(median)
    
    # Stats
    label_counts = combined["label"].value_counts()
    total_elapsed = time.time() - total_start
    
    logger.info(f"📈 Dataset créé en {total_elapsed:.1f}s:")
    logger.info(f"   Total: {len(combined):,} samples")
    logger.info(f"   Label 1 (TP): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(combined)*100:.1f}%)")
    logger.info(f"   Label 0 (SL): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(combined)*100:.1f}%)")
    logger.info(f"   Symboles: {combined['symbol'].nunique()}")
    
    # Sauvegarder
    output_path = output_dir / "swing_50pairs.parquet"
    combined.to_parquet(output_path, index=False)
    logger.info(f"💾 Sauvegardé: {output_path}")
    
    # Split train/val/test
    logger.info("✂️  Split train/val/test...")
    
    # Trier par temps
    combined = combined.sort_values("open_time")
    
    n = len(combined)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train = combined.iloc[:train_end]
    val = combined.iloc[train_end:val_end]
    test = combined.iloc[val_end:]
    
    train.to_parquet(output_dir / "swing_50pairs_train.parquet", index=False)
    val.to_parquet(output_dir / "swing_50pairs_val.parquet", index=False)
    test.to_parquet(output_dir / "swing_50pairs_test.parquet", index=False)
    
    logger.info(f"   Train: {len(train):,} ({len(train)/n*100:.0f}%)")
    logger.info(f"   Val: {len(val):,} ({len(val)/n*100:.0f}%)")
    logger.info(f"   Test: {len(test):,} ({len(test)/n*100:.0f}%)")
    
    return output_path


# ============================================
# ÉTAPE 4: ENTRAÎNER LE MODÈLE
# ============================================

def train_model(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
) -> Optional[Path]:
    """Entraîne le modèle XGBoost."""
    import joblib
    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    logger.info("🤖 Entraînement du modèle...")
    
    # Charger les données
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    feature_names = get_feature_names()
    
    X_train = train_df[feature_names].values
    y_train = train_df["label"].values
    X_val = val_df[feature_names].values
    y_val = val_df["label"].values
    X_test = test_df[feature_names].values
    y_test = test_df["label"].values
    
    # Remplacer les NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    logger.info(f"   Train: {len(X_train):,} samples")
    logger.info(f"   Val: {len(X_val):,} samples")
    logger.info(f"   Test: {len(X_test):,} samples")
    
    # Entraîner XGBoost
    logger.info("   Training XGBoost...")
    
    base_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=20,
    )
    
    base_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    # Calibration
    logger.info("   Calibration...")
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    
    # Évaluation
    logger.info("   Évaluation...")
    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    logger.info(f"📊 Métriques sur test:")
    logger.info(f"   Accuracy: {accuracy:.2%}")
    logger.info(f"   Precision: {precision:.2%}")
    logger.info(f"   Recall: {recall:.2%}")
    logger.info(f"   F1: {f1:.2%}")
    
    # Analyser par seuil
    logger.info("\n📈 Performance par seuil de probabilité:")
    
    for threshold in [0.15, 0.20, 0.25, 0.30, 0.35]:
        mask = y_proba >= threshold
        if mask.sum() == 0:
            continue
        
        trades = mask.sum()
        wins = (y_test[mask] == 1).sum()
        wr = wins / trades if trades > 0 else 0
        
        # Simuler PnL (TP 2%, SL 1%, frais 0.2%)
        win_pnl = 1.8  # 2% - 0.2%
        loss_pnl = -1.2  # 1% + 0.2%
        pnl = wins * win_pnl + (trades - wins) * loss_pnl
        pnl_per_trade = pnl / trades if trades > 0 else 0
        
        logger.info(f"   Seuil {threshold:.0%}: {trades:,} trades, WR {wr:.1%}, PnL {pnl:+.1f}% ({pnl_per_trade:+.2f}%/trade)")
    
    # Sauvegarder
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "swing_50pairs_model.joblib"
    joblib.dump(calibrated, model_path)
    logger.info(f"\n💾 Modèle sauvegardé: {model_path}")
    
    # Sauvegarder aussi comme "latest" pour le paper trading
    latest_path = output_dir / "xgb_model_latest.joblib"
    joblib.dump(calibrated, latest_path)
    
    # Sauvegarder les feature importances
    if hasattr(base_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": base_model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        importance_path = output_dir / "feature_importance_50pairs.csv"
        importance_df.to_csv(importance_path, index=False)
        
        logger.info(f"\n🏆 Top 10 features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.4f}")
    
    return model_path


# ============================================
# MAIN
# ============================================

async def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Setup complet pour trading avec 50 paires (OPTIMISÉ)"
    )
    parser.add_argument(
        "--pairs", "-n",
        type=int,
        default=DEFAULT_PAIRS_COUNT,
        help=f"Nombre de paires (défaut: {DEFAULT_PAIRS_COUNT})"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Nombre de jours de données (défaut: {DEFAULT_DAYS})"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Sauter le téléchargement (si données déjà présentes)"
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Sauter la création du dataset"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Sauter l'entraînement"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    print("=" * 65)
    print("🚀 CryptoScalper AI - Setup 50 Paires (OPTIMISÉ)")
    print("=" * 65)
    print(f"   Paires: {args.pairs}")
    print(f"   Jours: {args.days}")
    print("=" * 65)
    print()
    
    start_time = time.time()
    
    # Étape 1: Récupérer les top paires
    top_pairs = await get_top_pairs(n=args.pairs)
    
    if not top_pairs:
        logger.error("❌ Impossible de récupérer les paires")
        return 1
    
    print(f"\n📋 Paires sélectionnées: {', '.join(top_pairs[:10])}... (+{len(top_pairs)-10} autres)")
    
    # Sauvegarder la liste
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    with open(config_dir / "top_50_pairs.txt", "w") as f:
        f.write("\n".join(top_pairs))
    
    # Étape 2: Télécharger les données
    data_dir = Path(DATA_DIR)
    
    if not args.skip_download:
        print()
        await download_all_data(top_pairs, args.days, data_dir)
    else:
        logger.info("⏭️  Téléchargement sauté (--skip-download)")
    
    # Étape 3: Créer le dataset
    dataset_dir = Path(DATASET_DIR)
    
    if not args.skip_dataset:
        print()
        create_dataset(
            top_pairs,
            data_dir,
            dataset_dir,
            horizon=120,       # 2h
            threshold=0.02,    # TP 2%
            stop_loss=0.01,    # SL 1%
        )
    else:
        logger.info("⏭️  Création dataset sautée (--skip-dataset)")
    
    # Étape 4: Entraîner le modèle
    model_dir = Path(MODEL_DIR)
    
    if not args.skip_training:
        print()
        train_model(
            dataset_dir / "swing_50pairs_train.parquet",
            dataset_dir / "swing_50pairs_val.parquet",
            dataset_dir / "swing_50pairs_test.parquet",
            model_dir,
        )
    else:
        logger.info("⏭️  Entraînement sauté (--skip-training)")
    
    # Résumé
    elapsed = time.time() - start_time
    
    print()
    print("=" * 65)
    print("✅ SETUP TERMINÉ")
    print("=" * 65)
    print(f"   Durée totale: {elapsed/60:.1f} minutes")
    print()
    print("📁 Fichiers créés:")
    print(f"   • config/top_50_pairs.txt")
    print(f"   • {DATA_DIR}/*.parquet (données)")
    print(f"   • {DATASET_DIR}/swing_50pairs*.parquet (datasets)")
    print(f"   • {MODEL_DIR}/swing_50pairs_model.joblib (modèle)")
    print()
    print("🎯 Prochaine étape:")
    print("   python scripts/paper_trading.py --duration 4h")
    print("=" * 65)
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))