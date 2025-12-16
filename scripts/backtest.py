#!/usr/bin/env python3
# scripts/backtest.py
"""
Script CLI pour exécuter un backtest de la stratégie.

Ce script permet de:
- Charger des données historiques
- Charger un modèle ML entraîné
- Exécuter le backtest avec différentes configurations
- Générer des rapports de performance

Usage:
    # Backtest avec modèle ML
    python scripts/backtest.py --data data/BTCUSDT_1m.parquet --model models/saved/xgb_model_latest.joblib
    
    # Backtest simple avec config personnalisée
    python scripts/backtest.py --data data/BTCUSDT_1m.csv --capital 50 --sl 0.005 --tp 0.004
    
    # Backtest avec génération de rapport
    python scripts/backtest.py --data data/BTCUSDT_1m.parquet --report reports/

Options:
    --data          Chemin vers les données (CSV ou Parquet)
    --model         Chemin vers le modèle ML (optionnel)
    --symbol        Symbole de la paire (défaut: déduit du fichier)
    --capital       Capital initial (défaut: 30)
    --position      Taille position en % (défaut: 0.20)
    --sl            Stop loss en % (défaut: 0.004)
    --tp            Take profit en % (défaut: 0.003)
    --fee           Frais par transaction (défaut: 0.001)
    --prob          Probabilité minimum (défaut: 0.65)
    --report        Dossier pour les rapports (optionnel)
    --verbose       Mode verbeux
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from cryptoscalper.utils.logger import setup_logger, logger
from cryptoscalper.backtest import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    load_historical_data,
)
from cryptoscalper.backtest.reports import BacktestReport, generate_report


def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Exécute un backtest de la stratégie CryptoScalper AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Données
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Chemin vers les données historiques (CSV ou Parquet)",
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Chemin vers le modèle ML (optionnel, sinon backtest simple)",
    )
    
    parser.add_argument(
        "--symbol", "-s",
        type=str,
        default=None,
        help="Symbole de la paire (déduit du fichier si non spécifié)",
    )
    
    # Configuration
    parser.add_argument(
        "--capital", "-c",
        type=float,
        default=30.0,
        help="Capital initial en USDT (défaut: 30)",
    )
    
    parser.add_argument(
        "--position",
        type=float,
        default=0.20,
        help="Taille de position en %% du capital (défaut: 0.20)",
    )
    
    parser.add_argument(
        "--sl",
        type=float,
        default=0.004,
        help="Stop loss en %% (défaut: 0.004 = 0.4%%)",
    )
    
    parser.add_argument(
        "--tp",
        type=float,
        default=0.003,
        help="Take profit en %% (défaut: 0.003 = 0.3%%)",
    )
    
    parser.add_argument(
        "--fee",
        type=float,
        default=0.001,
        help="Frais par transaction (défaut: 0.001 = 0.1%%)",
    )
    
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0005,
        help="Slippage estimé (défaut: 0.0005 = 0.05%%)",
    )
    
    parser.add_argument(
        "--prob",
        type=float,
        default=0.65,
        help="Probabilité minimum pour trader (défaut: 0.65)",
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.55,
        help="Confiance minimum (défaut: 0.55)",
    )
    
    parser.add_argument(
        "--max-duration",
        type=int,
        default=5,
        help="Durée max d'une position en minutes (défaut: 5)",
    )
    
    # Output
    parser.add_argument(
        "--report", "-r",
        type=str,
        default=None,
        help="Dossier pour sauvegarder les rapports",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbeux (affiche plus de détails)",
    )
    
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Mode simple: génère des signaux aléatoires (pour tests)",
    )
    
    parser.add_argument(
        "--signal-freq",
        type=int,
        default=50,
        help="Fréquence des signaux en mode simple (1 signal toutes les N bougies)",
    )
    
    return parser.parse_args()


def infer_symbol_from_path(filepath: str) -> str:
    """
    Déduit le symbole depuis le nom du fichier.
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        Symbole (ex: BTCUSDT)
    """
    filename = Path(filepath).stem.upper()
    
    # Patterns courants
    for quote in ["USDT", "BTC", "ETH", "BUSD"]:
        if quote in filename:
            # Extraire le base asset
            parts = filename.split("_")
            for part in parts:
                if quote in part:
                    return part
            # Sinon prendre le premier qui contient le quote
            return filename.split("_")[0] if "_" in filename else filename
    
    return "BTCUSDT"  # Default


def load_data(filepath: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier.
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        DataFrame avec les données
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
    
    logger.info(f"📂 Chargement des données: {filepath}")
    
    # Utiliser la fonction du module backtest
    df = load_historical_data(filepath)
    
    logger.info(f"   {len(df)} bougies chargées")
    logger.info(f"   Période: {df.index[0]} → {df.index[-1]}")
    
    return df


def create_simple_signals(df: pd.DataFrame, freq: int = 50) -> pd.Series:
    """
    Crée des signaux simples pour test.
    
    Args:
        df: DataFrame des données
        freq: Fréquence des signaux
        
    Returns:
        Series de signaux (0 ou 1)
    """
    signals = pd.Series(0, index=df.index)
    
    # Warmup de 60 bougies
    warmup = 60
    
    # Générer des signaux périodiques
    for i in range(warmup, len(df) - 10, freq):
        signals.iloc[i] = 1
    
    n_signals = signals.sum()
    logger.info(f"📊 {n_signals} signaux générés (mode simple)")
    
    return signals


def run_backtest_with_model(
    df: pd.DataFrame,
    model_path: str,
    config: BacktestConfig,
    symbol: str,
) -> BacktestResult:
    """
    Exécute le backtest avec un modèle ML.
    
    Args:
        df: Données historiques
        model_path: Chemin vers le modèle
        config: Configuration du backtest
        symbol: Symbole
        
    Returns:
        BacktestResult
    """
    from cryptoscalper.models import MLPredictor
    from cryptoscalper.data.features import FeatureEngine
    
    logger.info(f"🤖 Chargement du modèle: {model_path}")
    
    predictor = MLPredictor.from_file(model_path)
    feature_engine = FeatureEngine()
    
    logger.info("🚀 Démarrage du backtest avec modèle ML...")
    
    engine = BacktestEngine(config)
    
    # Callback de progression
    def progress(current: int, total: int):
        if current % 1000 == 0:
            pct = current / total * 100
            print(f"\r   Progression: {pct:.1f}% ({current}/{total})", end="", flush=True)
    
    result = engine.run(
        data=df,
        predictor=predictor,
        feature_engine=feature_engine,
        symbol=symbol,
        progress_callback=progress,
    )
    
    print()  # Nouvelle ligne après la progression
    
    return result


def run_backtest_simple(
    df: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig,
    symbol: str,
) -> BacktestResult:
    """
    Exécute un backtest simple avec des signaux pré-calculés.
    
    Args:
        df: Données historiques
        signals: Signaux de trading
        config: Configuration du backtest
        symbol: Symbole
        
    Returns:
        BacktestResult
    """
    logger.info("🚀 Démarrage du backtest simple...")
    
    engine = BacktestEngine(config)
    result = engine.run_simple(df, signals, symbol=symbol)
    
    return result


def print_results(result: BacktestResult) -> None:
    """Affiche les résultats du backtest."""
    print(result.summary())


def main() -> int:
    """Point d'entrée du script."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(level=log_level)
    
    print("=" * 65)
    print("🤖 CryptoScalper AI - Backtest")
    print("=" * 65)
    
    try:
        # 1. Charger les données
        df = load_data(args.data)
        
        # 2. Déterminer le symbole
        symbol = args.symbol or infer_symbol_from_path(args.data)
        logger.info(f"📈 Symbole: {symbol}")
        
        # 3. Créer la configuration
        config = BacktestConfig(
            initial_capital=args.capital,
            position_size_pct=args.position,
            fee_percent=args.fee,
            slippage_percent=args.slippage,
            stop_loss_pct=args.sl,
            take_profit_pct=args.tp,
            min_probability=args.prob,
            min_confidence=args.confidence,
            max_position_duration_minutes=args.max_duration,
        )
        
        logger.info(f"⚙️ Configuration:")
        logger.info(f"   Capital: {config.initial_capital:.2f} USDT")
        logger.info(f"   Position: {config.position_size_pct:.0%}")
        logger.info(f"   SL: {config.stop_loss_pct:.2%} | TP: {config.take_profit_pct:.2%}")
        logger.info(f"   Frais: {config.fee_percent:.2%} | Slippage: {config.slippage_percent:.2%}")
        
        # 4. Exécuter le backtest
        if args.model and not args.simple:
            # Backtest avec modèle ML
            result = run_backtest_with_model(df, args.model, config, symbol)
        else:
            # Backtest simple
            signals = create_simple_signals(df, freq=args.signal_freq)
            result = run_backtest_simple(df, signals, config, symbol)
        
        # 5. Afficher les résultats
        print("\n")
        print_results(result)
        
        # 6. Générer les rapports si demandé
        if args.report:
            print(f"\n📄 Génération des rapports...")
            report = generate_report(
                result=result,
                output_dir=args.report,
                prefix=f"backtest_{symbol}",
            )
            logger.info(f"✅ Rapports sauvegardés dans {args.report}")
        
        # 7. Résumé final
        print("\n" + "=" * 65)
        print("📊 RÉSUMÉ")
        print("=" * 65)
        
        emoji_pnl = "✅" if result.total_return >= 0 else "❌"
        emoji_wr = "✅" if result.win_rate >= 0.5 else "⚠️"
        
        print(f"  {emoji_pnl} Rendement: {result.total_return:+.2f} USDT ({result.total_return_pct:+.2%})")
        print(f"  {emoji_wr} Win Rate: {result.win_rate:.1%}")
        print(f"  📊 Trades: {result.total_trades}")
        print(f"  📉 Max Drawdown: {result.max_drawdown:.2%}")
        print(f"  📈 Profit Factor: {result.profit_factor:.2f}")
        print("=" * 65)
        
        # Verdict
        if result.total_return > 0 and result.win_rate > 0.5 and result.profit_factor > 1.0:
            print("🎉 Stratégie potentiellement viable !")
        elif result.total_return > 0:
            print("🤔 Résultats mitigés, ajustements recommandés")
        else:
            print("⚠️  Stratégie non profitable, révision nécessaire")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return 1
    except ImportError as e:
        logger.error(f"❌ Module manquant: {e}")
        logger.info("💡 Vérifiez que tous les modules sont installés")
        return 1
    except Exception as e:
        logger.exception(f"❌ Erreur: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())