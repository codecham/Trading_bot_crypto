# scripts/prepare_dataset.py
"""
Script pour préparer le dataset d'entraînement.

Usage:
    # Préparer un dataset depuis les données téléchargées
    python scripts/prepare_dataset.py --symbols BTCUSDT,ETHUSDT --output datasets/train_dataset.parquet
    
    # Avec configuration personnalisée
    python scripts/prepare_dataset.py --symbols BTCUSDT --horizon 5 --threshold 0.003
"""

import argparse
import sys
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import setup_logger, logger
from cryptoscalper.data.dataset import (
    DatasetBuilder,
    LabelConfig,
    SplitConfig,
    prepare_dataset,
    analyze_class_balance,
)
from cryptoscalper.data.historical import is_data_cached


# Symboles par défaut
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Prépare le dataset pour l'entraînement ML"
    )
    
    # Symboles
    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Liste de symboles séparés par des virgules"
    )
    
    # Configuration des labels
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Horizon de prédiction en minutes (défaut: 3)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.002,
        help="Seuil de hausse pour label=1 (défaut: 0.002 = 0.2%%)"
    )
    
    # Sortie
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/prepared_dataset.parquet",
        help="Fichier de sortie"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data_cache",
        help="Dossier des données sources"
    )
    
    # Options
    parser.add_argument(
        "--split",
        action="store_true",
        help="Sauvegarder aussi les splits train/val/test"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbose"
    )
    
    return parser.parse_args()


def main():
    """Point d'entrée principal."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(level=log_level)
    
    print("=" * 60)
    print("🔧 CryptoScalper AI - Préparation Dataset")
    print("=" * 60)
    
    # Parser les symboles
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # Vérifier que les données existent
    data_dir = Path(args.data_dir)
    missing = [s for s in symbols if not is_data_cached(s, data_dir)]
    if missing:
        logger.error(f"❌ Données manquantes pour: {missing}")
        logger.info("Lancez d'abord: python scripts/download_data.py --symbols ...")
        return 1
    
    # Configuration
    label_config = LabelConfig(
        horizon_minutes=args.horizon,
        threshold_percent=args.threshold
    )
    
    print(f"\n📋 Configuration:")
    print(f"   Symboles: {', '.join(symbols)}")
    print(f"   Horizon: {args.horizon} minutes")
    print(f"   Seuil: {args.threshold:.2%}")
    print(f"   Sortie: {args.output}")
    print()
    
    # Construire le dataset
    builder = DatasetBuilder(label_config=label_config)
    
    try:
        dataset = builder.build_from_symbols(symbols, data_dir)
    except Exception as e:
        logger.error(f"❌ Erreur construction dataset: {e}")
        return 1
    
    # Afficher les stats
    print("\n" + "=" * 60)
    print("📊 STATISTIQUES DU DATASET")
    print("=" * 60)
    print(dataset.stats.summary())
    
    # Analyse de l'équilibre
    balance = analyze_class_balance(dataset.labels)
    print(f"\n📈 Équilibre des classes:")
    print(f"   Ratio positifs: {balance['positive_ratio']:.1%}")
    print(f"   Ratio négatifs: {balance['negative_ratio']:.1%}")
    print(f"   Déséquilibre: {balance['imbalance_ratio']:.2f}x")
    
    # Sauvegarder
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output_path)
    
    # Split si demandé
    if args.split:
        print("\n📂 Sauvegarde des splits...")
        train, val, test = dataset.split_temporal()
        
        base_name = output_path.stem
        train.save(output_path.parent / f"{base_name}_train.parquet")
        val.save(output_path.parent / f"{base_name}_val.parquet")
        test.save(output_path.parent / f"{base_name}_test.parquet")
    
    print("\n" + "=" * 60)
    print("✅ DATASET PRÉPARÉ AVEC SUCCÈS")
    print("=" * 60)
    print(f"   Fichier: {args.output}")
    print(f"   Lignes: {len(dataset):,}")
    print(f"   Features: {dataset.stats.feature_count}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())