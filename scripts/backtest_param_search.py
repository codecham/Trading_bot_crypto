#!/usr/bin/env python3
# scripts/backtest_param_search.py
"""
Recherche des meilleurs paramètres TP/SL par backtest.

Teste plusieurs combinaisons de:
- Take Profit (0.5% à 2%)
- Stop Loss (0.3% à 1%)
- Seuils de probabilité

Usage:
    python scripts/backtest_param_search.py
"""

import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
from enum import Enum

import numpy as np
import pandas as pd
import joblib
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
        "rsi_14", "rsi_7", "stoch_k", "stoch_d", "williams_r",
        "roc_5", "roc_10", "momentum_5", "cci", "cmo",
        "ema_5_ratio", "ema_10_ratio", "ema_20_ratio",
        "macd_line", "macd_signal", "macd_histogram",
        "adx", "aroon_oscillator",
        "bb_width", "bb_position", "atr", "atr_percent",
        "returns_std", "hl_range_avg",
        "spread_percent", "orderbook_imbalance", "bid_depth", "ask_depth",
        "depth_ratio", "bid_pressure", "ask_pressure", "midprice_distance",
        "volume_relative", "obv_slope", "volume_delta", "vwap_distance", "ad_line",
        "returns_1m", "returns_5m", "returns_15m",
        "consecutive_green", "candle_body_ratio"
    ]


@dataclass
class BacktestParams:
    """Paramètres de backtest."""
    tp_pct: float
    sl_pct: float
    timeout_min: int
    threshold: float
    fee_pct: float = 0.001  # 0.1% par trade


@dataclass 
class BacktestResult:
    """Résultat d'un backtest."""
    params: BacktestParams
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    pnl_pct: float
    max_dd: float
    tp_exits: int
    sl_exits: int
    timeout_exits: int
    avg_duration: float


def run_fast_backtest(
    data: pd.DataFrame,
    probas: np.ndarray,
    params: BacktestParams,
) -> BacktestResult:
    """
    Backtest rapide et simplifié.
    
    Simule un seul trade à la fois, sans gestion de capital complexe.
    """
    n = len(data)
    
    # Stats
    trades = 0
    wins = 0
    losses = 0
    tp_exits = 0
    sl_exits = 0
    timeout_exits = 0
    total_pnl = 0.0
    durations = []
    
    # Equity tracking
    equity = 100.0  # Base 100
    peak = 100.0
    max_dd = 0.0
    
    # État
    in_trade = False
    entry_idx = 0
    entry_price = 0.0
    
    # Colonnes
    close = data['close'].values
    high = data['high'].values
    low = data['low'].values
    times = data['open_time'].values
    
    i = 0
    while i < n:
        if not in_trade:
            # Chercher une entrée
            if probas[i] >= params.threshold:
                in_trade = True
                entry_idx = i
                entry_price = close[i]
                tp_price = entry_price * (1 + params.tp_pct)
                sl_price = entry_price * (1 - params.sl_pct)
                timeout_idx = min(i + params.timeout_min, n - 1)
            i += 1
        else:
            # Vérifier la sortie
            # Check SL
            if low[i] <= sl_price:
                pnl = -params.sl_pct - params.fee_pct * 2
                losses += 1
                sl_exits += 1
                exit_type = "SL"
            # Check TP
            elif high[i] >= tp_price:
                pnl = params.tp_pct - params.fee_pct * 2
                wins += 1
                tp_exits += 1
                exit_type = "TP"
            # Check Timeout
            elif i >= timeout_idx:
                final_pnl = (close[i] - entry_price) / entry_price
                pnl = final_pnl - params.fee_pct * 2
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                timeout_exits += 1
                exit_type = "TO"
            else:
                # Toujours en trade
                i += 1
                continue
            
            # Trade fermé
            trades += 1
            total_pnl += pnl
            durations.append(i - entry_idx)
            
            # Update equity
            equity *= (1 + pnl)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
            
            in_trade = False
            i += 1
    
    # Calculer les résultats
    win_rate = wins / trades * 100 if trades > 0 else 0
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    return BacktestResult(
        params=params,
        total_trades=trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        pnl_pct=total_pnl * 100,
        max_dd=max_dd,
        tp_exits=tp_exits,
        sl_exits=sl_exits,
        timeout_exits=timeout_exits,
        avg_duration=avg_duration,
    )


def main():
    """Point d'entrée principal."""
    print("=" * 70)
    print("🔍 CryptoScalper AI - Recherche de Paramètres Optimaux")
    print("=" * 70)
    
    # Charger le modèle
    model_path = "models/saved/swing_50pairs_model.joblib"
    logger.info(f"Chargement du modèle: {model_path}")
    model = joblib.load(model_path)
    
    # Charger les données de test
    data_path = "datasets/swing_50pairs_test.parquet"
    logger.info(f"Chargement des données: {data_path}")
    data = pd.read_parquet(data_path)
    
    # Trier par temps
    data = data.sort_values('open_time').reset_index(drop=True)
    
    logger.info(f"   {len(data):,} samples")
    logger.info(f"   {data['symbol'].nunique()} symboles")
    
    # Calculer les probas une seule fois
    logger.info("Calcul des probabilités...")
    feature_names = get_feature_names()
    X = data[feature_names].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    probas = model.predict_proba(X)[:, 1]
    
    # Paramètres à tester
    tp_values = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]  # 0.5% à 2%
    sl_values = [0.003, 0.005, 0.0075, 0.01]  # 0.3% à 1%
    timeout_values = [60, 120, 240]  # 1h, 2h, 4h
    threshold_values = [0.20, 0.25, 0.30, 0.35, 0.40]
    
    results = []
    total_tests = len(tp_values) * len(sl_values) * len(timeout_values) * len(threshold_values)
    
    logger.info(f"\n🧪 Lancement de {total_tests} backtests...\n")
    
    test_num = 0
    start_time = time.time()
    
    for tp in tp_values:
        for sl in sl_values:
            # Skip si ratio R:R < 1
            if tp / sl < 1.0:
                continue
                
            for timeout in timeout_values:
                for threshold in threshold_values:
                    test_num += 1
                    
                    params = BacktestParams(
                        tp_pct=tp,
                        sl_pct=sl,
                        timeout_min=timeout,
                        threshold=threshold,
                    )
                    
                    result = run_fast_backtest(data, probas, params)
                    results.append(result)
                    
                    # Progress
                    if test_num % 20 == 0:
                        elapsed = time.time() - start_time
                        eta = elapsed / test_num * (total_tests - test_num)
                        logger.info(f"   Progression: {test_num}/{total_tests} ({elapsed:.0f}s, ETA: {eta:.0f}s)")
    
    # Trier par PnL
    results.sort(key=lambda x: x.pnl_pct, reverse=True)
    
    # Afficher les meilleurs résultats
    print("\n" + "=" * 70)
    print("🏆 TOP 20 MEILLEURES CONFIGURATIONS")
    print("=" * 70)
    
    print(f"\n{'#':<3} {'TP':>6} {'SL':>6} {'T/O':>5} {'Seuil':>6} {'Trades':>7} {'WR':>6} {'PnL':>8} {'MaxDD':>6} {'TP%':>5} {'SL%':>5} {'TO%':>5}")
    print("-" * 85)
    
    for i, r in enumerate(results[:20], 1):
        p = r.params
        tp_pct = r.tp_exits / r.total_trades * 100 if r.total_trades > 0 else 0
        sl_pct = r.sl_exits / r.total_trades * 100 if r.total_trades > 0 else 0
        to_pct = r.timeout_exits / r.total_trades * 100 if r.total_trades > 0 else 0
        
        print(f"{i:<3} {p.tp_pct*100:>5.1f}% {p.sl_pct*100:>5.1f}% {p.timeout_min:>4}m {p.threshold*100:>5.0f}% "
              f"{r.total_trades:>7} {r.win_rate:>5.1f}% {r.pnl_pct:>+7.1f}% {r.max_dd:>5.1f}% "
              f"{tp_pct:>4.0f}% {sl_pct:>4.0f}% {to_pct:>4.0f}%")
    
    # Afficher les pires
    print("\n" + "-" * 70)
    print("📉 5 PIRES CONFIGURATIONS")
    print("-" * 70)
    
    for i, r in enumerate(results[-5:], 1):
        p = r.params
        print(f"{i:<3} TP:{p.tp_pct*100:.1f}% SL:{p.sl_pct*100:.1f}% T/O:{p.timeout_min}m Seuil:{p.threshold*100:.0f}% "
              f"→ {r.total_trades} trades, WR:{r.win_rate:.1f}%, PnL:{r.pnl_pct:+.1f}%")
    
    # Analyser les patterns
    print("\n" + "=" * 70)
    print("📊 ANALYSE DES PATTERNS")
    print("=" * 70)
    
    # Meilleur TP
    best_by_tp = {}
    for r in results:
        tp = r.params.tp_pct
        if tp not in best_by_tp or r.pnl_pct > best_by_tp[tp].pnl_pct:
            best_by_tp[tp] = r
    
    print("\n📈 Meilleur résultat par TP:")
    for tp in sorted(best_by_tp.keys()):
        r = best_by_tp[tp]
        print(f"   TP {tp*100:.1f}%: PnL {r.pnl_pct:+.1f}%, WR {r.win_rate:.1f}%, {r.total_trades} trades")
    
    # Meilleur seuil
    best_by_threshold = {}
    for r in results:
        t = r.params.threshold
        if t not in best_by_threshold or r.pnl_pct > best_by_threshold[t].pnl_pct:
            best_by_threshold[t] = r
    
    print("\n📈 Meilleur résultat par seuil:")
    for t in sorted(best_by_threshold.keys()):
        r = best_by_threshold[t]
        print(f"   Seuil {t*100:.0f}%: PnL {r.pnl_pct:+.1f}%, WR {r.win_rate:.1f}%, {r.total_trades} trades")
    
    # Profitable combinations
    profitable = [r for r in results if r.pnl_pct > 0]
    print(f"\n✅ Combinaisons rentables: {len(profitable)}/{len(results)}")
    
    if profitable:
        print("\n🎯 CONFIGURATION RECOMMANDÉE:")
        best = profitable[0]
        p = best.params
        print(f"   TP: {p.tp_pct*100:.2f}%")
        print(f"   SL: {p.sl_pct*100:.2f}%")
        print(f"   Timeout: {p.timeout_min} min")
        print(f"   Seuil: {p.threshold*100:.0f}%")
        print(f"   → PnL attendu: {best.pnl_pct:+.1f}%")
        print(f"   → Win Rate: {best.win_rate:.1f}%")
        print(f"   → Max Drawdown: {best.max_dd:.1f}%")
    else:
        print("\n❌ AUCUNE CONFIGURATION RENTABLE TROUVÉE")
        print("   Le modèle ne semble pas capable de prédire les mouvements de prix.")
        print("   Recommandations:")
        print("   1. Revoir le labeling des données")
        print("   2. Essayer une approche différente (prédire direction, pas TP)")
        print("   3. Ajouter des features (order book réel, sentiment, etc.)")
    
    # Sauvegarder tous les résultats
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    
    results_data = []
    for r in results:
        results_data.append({
            'tp_pct': r.params.tp_pct * 100,
            'sl_pct': r.params.sl_pct * 100,
            'timeout_min': r.params.timeout_min,
            'threshold': r.params.threshold * 100,
            'trades': r.total_trades,
            'wins': r.wins,
            'losses': r.losses,
            'win_rate': r.win_rate,
            'pnl_pct': r.pnl_pct,
            'max_dd': r.max_dd,
            'tp_exits': r.tp_exits,
            'sl_exits': r.sl_exits,
            'timeout_exits': r.timeout_exits,
        })
    
    df = pd.DataFrame(results_data)
    output_path = output_dir / f"param_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\n💾 Résultats sauvegardés: {output_path}")
    
    print("\n" + "=" * 70)
    elapsed = time.time() - start_time
    print(f"✅ Recherche terminée en {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()