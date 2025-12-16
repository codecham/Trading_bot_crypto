# scripts/validate_features.py
"""
Script de validation de l'alignement des features.
Compare les features calculées par dataset.py vs features.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

from cryptoscalper.data.features import FeatureEngine, get_feature_names

console = Console()


def compute_features_dataset_method(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les features avec la méthode du dataset.py (vectorisée).
    COPIE EXACTE de _compute_all_features pour comparaison.
    """
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
    
    # Momentum NORMALISÉ
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


def compute_features_engine_method(df: pd.DataFrame) -> dict:
    """Calcule les features avec FeatureEngine (temps réel)."""
    engine = FeatureEngine()
    fs = engine.compute_features(df, symbol='TEST')
    return fs.features


def compare_features(df: pd.DataFrame):
    """Compare les deux méthodes de calcul."""
    feature_names = get_feature_names()
    
    # Méthode dataset (vectorisée)
    df_dataset = compute_features_dataset_method(df.copy())
    
    # Méthode engine (temps réel) - sur les 60 dernières lignes
    features_engine = compute_features_engine_method(df.tail(60))
    
    # Comparer la dernière ligne du dataset avec engine
    last_idx = len(df_dataset) - 1
    
    table = Table(title="Comparaison Features: Dataset vs Engine")
    table.add_column("Feature", style="cyan")
    table.add_column("Dataset", justify="right")
    table.add_column("Engine", justify="right")
    table.add_column("Diff %", justify="right")
    table.add_column("Status")
    
    mismatches = []
    
    for name in feature_names:
        if name in df_dataset.columns and name in features_engine:
            val_dataset = df_dataset[name].iloc[last_idx]
            val_engine = features_engine[name]
            
            # Calculer la différence
            if pd.isna(val_dataset) or pd.isna(val_engine):
                diff_pct = "NaN"
                status = "⚠️"
            elif val_dataset == 0 and val_engine == 0:
                diff_pct = "0.0%"
                status = "✅"
            elif val_dataset == 0:
                diff_pct = "∞"
                status = "❌"
            else:
                diff = abs(val_engine - val_dataset) / abs(val_dataset) * 100
                diff_pct = f"{diff:.1f}%"
                if diff < 1:
                    status = "✅"
                elif diff < 10:
                    status = "⚠️"
                else:
                    status = "❌"
                    mismatches.append((name, val_dataset, val_engine, diff))
            
            table.add_row(
                name,
                f"{val_dataset:.6f}" if not pd.isna(val_dataset) else "NaN",
                f"{val_engine:.6f}" if not pd.isna(val_engine) else "NaN",
                diff_pct,
                status
            )
    
    console.print(table)
    
    if mismatches:
        console.print(f"\n[red]❌ {len(mismatches)} features avec >10% de différence:[/red]")
        for name, ds, eng, diff in mismatches:
            console.print(f"   {name}: dataset={ds:.6f}, engine={eng:.6f}, diff={diff:.1f}%")
    else:
        console.print("\n[green]✅ Toutes les features sont alignées (<10% de différence)[/green]")
    
    return mismatches


def main():
    console.print("[bold blue]🔍 Validation de l'alignement des features[/bold blue]\n")
    
    # Charger des données de test
    df = pd.read_parquet('data_cache/BTCUSDT_1m.parquet').tail(100)
    console.print(f"Données: {len(df)} lignes de BTCUSDT\n")
    
    mismatches = compare_features(df)
    
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
