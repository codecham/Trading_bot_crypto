# scripts/download_data.py
"""
Script pour télécharger les données historiques depuis Binance.

Usage:
    # Télécharger BTC et ETH sur 180 jours
    python scripts/download_data.py --symbols BTCUSDT,ETHUSDT --days 180
    
    # Télécharger le top 20 des paires par volume
    python scripts/download_data.py --top 20 --days 180
    
    # Télécharger une liste prédéfinie
    python scripts/download_data.py --preset default --days 180
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import setup_logger, logger
from cryptoscalper.data.binance_client import BinanceClient
from cryptoscalper.data.historical import (
    HistoricalDataDownloader,
    MultiSymbolDownloader,
    DownloadConfig,
)
from cryptoscalper.data.symbols import get_tradeable_symbols


# Symboles par défaut pour l'entraînement
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT",
    "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "ETCUSDT",
]

# Symboles minimaux pour tests rapides
MINIMAL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Télécharge les données historiques depuis Binance"
    )
    
    # Source des symboles
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--symbols",
        type=str,
        help="Liste de symboles séparés par des virgules (ex: BTCUSDT,ETHUSDT)"
    )
    group.add_argument(
        "--top",
        type=int,
        help="Télécharger les N paires avec le plus de volume"
    )
    group.add_argument(
        "--preset",
        choices=["default", "minimal", "all"],
        help="Utiliser une liste prédéfinie"
    )
    
    # Paramètres de téléchargement
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Nombre de jours à télécharger (défaut: 180)"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1m",
        help="Intervalle des bougies (défaut: 1m)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_cache",
        help="Dossier de sortie (défaut: data_cache)"
    )
    
    # Options
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Format de sauvegarde (défaut: parquet)"
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=10_000_000,
        help="Volume minimum 24h pour --top (défaut: 10M)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbose"
    )
    
    return parser.parse_args()


async def get_symbols(args) -> list:
    """Récupère la liste des symboles selon les arguments."""
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",")]
    
    if args.preset == "minimal":
        return MINIMAL_SYMBOLS
    
    if args.preset == "default":
        return DEFAULT_SYMBOLS
    
    # Pour --top ou --preset all, on doit scanner les paires
    async with BinanceClient() as client:
        if args.top:
            symbols = await get_tradeable_symbols(
                client._client,
                min_volume=args.min_volume,
                max_pairs=args.top
            )
            return symbols
        
        if args.preset == "all":
            symbols = await get_tradeable_symbols(
                client._client,
                min_volume=args.min_volume,
                max_pairs=100
            )
            return symbols
    
    # Par défaut
    return DEFAULT_SYMBOLS


async def main():
    """Point d'entrée principal."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(level=log_level)
    
    print("=" * 60)
    print("📥 CryptoScalper AI - Téléchargement Données Historiques")
    print("=" * 60)
    
    # Récupérer les symboles
    symbols = await get_symbols(args)
    
    print(f"\n📋 Symboles à télécharger: {len(symbols)}")
    print(f"   {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
    print(f"📆 Période: {args.days} jours")
    print(f"📊 Intervalle: {args.interval}")
    print(f"📁 Dossier: {args.output_dir}")
    print(f"💾 Format: {args.format}")
    print()
    
    # Confirmer
    if len(symbols) > 5:
        confirm = input(f"Télécharger {len(symbols)} symboles ? (y/N) ")
        if confirm.lower() != "y":
            print("Annulé.")
            return 0
    
    # Configurer
    config = DownloadConfig(
        interval=args.interval,
        days=args.days,
        output_dir=Path(args.output_dir),
        save_format=args.format
    )
    
    # Télécharger
    start_time = datetime.now()
    
    async with BinanceClient() as client:
        downloader = MultiSymbolDownloader(client._client, config)
        results = await downloader.download_all(
            symbols=symbols,
            days=args.days,
            save=True
        )
    
    # Résumé final
    duration = (datetime.now() - start_time).total_seconds()
    success_count = len([r for r in results.values() if r.rows_count > 0])
    total_rows = sum(r.rows_count for r in results.values())
    
    print("\n" + "=" * 60)
    print("📊 TÉLÉCHARGEMENT TERMINÉ")
    print("=" * 60)
    print(f"   Symboles réussis: {success_count}/{len(symbols)}")
    print(f"   Total lignes: {total_rows:,}")
    print(f"   Durée totale: {duration:.1f}s")
    print(f"   Fichiers dans: {args.output_dir}/")
    
    # Lister les fichiers créés
    print("\n📁 Fichiers créés:")
    for symbol, result in results.items():
        if result.file_path:
            size_mb = result.file_path.stat().st_size / (1024 * 1024)
            print(f"   {result.file_path.name}: {result.rows_count:,} lignes ({size_mb:.1f} MB)")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))