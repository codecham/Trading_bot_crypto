#!/usr/bin/env python3
# scripts/train_model_50pairs.py
"""
Entraîne le modèle XGBoost sur le dataset 50 paires.
Version simplifiée sans calibration (évite les problèmes de compatibilité).

Usage:
    python scripts/train_model_50pairs.py
"""

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

# Setup logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level="INFO",
    colorize=True
)


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


def train_model():
    """Entraîne le modèle XGBoost."""
    import joblib
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Chemins
    dataset_dir = Path("datasets")
    output_dir = Path("models/saved")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = dataset_dir / "swing_50pairs_train.parquet"
    val_path = dataset_dir / "swing_50pairs_val.parquet"
    test_path = dataset_dir / "swing_50pairs_test.parquet"
    
    # Vérifier que les fichiers existent
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            logger.error(f"❌ Fichier manquant: {path}")
            return None
    
    logger.info("🤖 Entraînement du modèle...")
    start_time = time.time()
    
    # Charger les données
    logger.info("   Chargement des données...")
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
    
    # Remplacer les NaN et Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.info(f"   Train: {len(X_train):,} samples")
    logger.info(f"   Val: {len(X_val):,} samples")
    logger.info(f"   Test: {len(X_test):,} samples")
    
    # Entraîner XGBoost
    logger.info("   Training XGBoost...")
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=20,
        n_jobs=-1,  # Utiliser tous les CPU
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    train_time = time.time() - start_time
    logger.info(f"   Training terminé en {train_time:.1f}s")
    
    # Évaluation
    logger.info("   Évaluation...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    logger.info(f"\n📊 Métriques sur test set:")
    logger.info(f"   Accuracy:  {accuracy:.2%}")
    logger.info(f"   Precision: {precision:.2%}")
    logger.info(f"   Recall:    {recall:.2%}")
    logger.info(f"   F1 Score:  {f1:.2%}")
    
    # Analyser par seuil de probabilité
    logger.info(f"\n📈 Performance par seuil de probabilité:")
    logger.info(f"   {'Seuil':<8} {'Trades':>10} {'WR':>8} {'PnL':>10} {'PnL/trade':>12}")
    logger.info(f"   {'-'*50}")
    
    best_threshold = 0.20
    best_pnl_per_trade = -999
    
    for threshold in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40]:
        mask = y_proba >= threshold
        trades = mask.sum()
        
        if trades == 0:
            continue
        
        wins = (y_test[mask] == 1).sum()
        wr = wins / trades if trades > 0 else 0
        
        # Simuler PnL (TP 2%, SL 1%, frais 0.2%)
        win_pnl = 1.8   # 2% - 0.2%
        loss_pnl = -1.2  # 1% + 0.2%
        total_pnl = wins * win_pnl + (trades - wins) * loss_pnl
        pnl_per_trade = total_pnl / trades if trades > 0 else 0
        
        logger.info(f"   {threshold:<8.0%} {trades:>10,} {wr:>7.1%} {total_pnl:>+10.1f}% {pnl_per_trade:>+11.3f}%")
        
        # Trouver le meilleur seuil
        if pnl_per_trade > best_pnl_per_trade and trades >= 1000:
            best_pnl_per_trade = pnl_per_trade
            best_threshold = threshold
    
    logger.info(f"\n   💡 Meilleur seuil recommandé: {best_threshold:.0%}")
    
    # Sauvegarder le modèle
    model_path = output_dir / "swing_50pairs_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"\n💾 Modèle sauvegardé: {model_path}")
    
    # Copier aussi comme "latest"
    latest_path = output_dir / "xgb_model_latest.joblib"
    joblib.dump(model, latest_path)
    logger.info(f"💾 Copié vers: {latest_path}")
    
    # Sauvegarder les feature importances
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        importance_path = output_dir / "feature_importance_50pairs.csv"
        importance_df.to_csv(importance_path, index=False)
        
        logger.info(f"\n🏆 Top 10 features les plus importantes:")
        for _, row in importance_df.head(10).iterrows():
            bar = "█" * int(row['importance'] * 50)
            logger.info(f"   {row['feature']:<20} {row['importance']:.4f} {bar}")
    
    # Résumé final
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ ENTRAÎNEMENT TERMINÉ")
    logger.info(f"{'='*60}")
    logger.info(f"   Durée totale: {total_time:.1f}s")
    logger.info(f"   Modèle: {model_path}")
    logger.info(f"   Seuil recommandé: {best_threshold:.0%}")
    logger.info(f"\n🎯 Prochaine étape:")
    logger.info(f"   python scripts/paper_trading.py --duration 4h --threshold {best_threshold}")
    logger.info(f"{'='*60}")
    
    return model_path


if __name__ == "__main__":
    train_model()