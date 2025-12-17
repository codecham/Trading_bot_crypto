#!/usr/bin/env python3
# scripts/get_top_pairs.py
"""
Récupère les meilleures paires de trading sur Binance.

Critères de sélection:
- Paires USDT uniquement
- Volume 24h > 1M USDT
- Exclut les stablecoins
- Trié par volume décroissant

Usage:
    python scripts/get_top_pairs.py
    python scripts/get_top_pairs.py --top 50 --min-volume 1000000
    python scripts/get_top_pairs.py --save  # Sauvegarde dans un fichier
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Dict

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from binance import AsyncClient


# ============================================
# CONSTANTES
# ============================================

# Stablecoins et paires à exclure
EXCLUDED_SYMBOLS = {
    # Stablecoins
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "USDPUSDT", "DAIUSDT",
    "EURUSDT", "GBPUSDT", "AUDUSDT", "FDUSDUSDT", "PYUSDUSDT",
    # Paires leveraged/spéciales
    "BTCSTUSDT", "BETHUSDT", "WBTCUSDT", "WBETHUSDT",
    # Autres à éviter
    "USTUSDT", "LUNAUSDT", "LUNCUSDT",  # Terra
}

# Paramètres par défaut
DEFAULT_TOP_N = 50
DEFAULT_MIN_VOLUME = 1_000_000  # 1M USDT


# ============================================
# FONCTIONS
# ============================================

async def get_all_usdt_pairs(client: AsyncClient) -> List[Dict]:
    """
    Récupère toutes les paires USDT tradables.
    
    Returns:
        Liste de dict avec symbol et infos
    """
    exchange_info = await client.get_exchange_info()
    
    usdt_pairs = []
    for symbol_info in exchange_info["symbols"]:
        symbol = symbol_info["symbol"]
        
        # Filtrer
        if not symbol.endswith("USDT"):
            continue
        if symbol_info["status"] != "TRADING":
            continue
        if not symbol_info.get("isSpotTradingAllowed", False):
            continue
        if symbol in EXCLUDED_SYMBOLS:
            continue
        
        usdt_pairs.append({
            "symbol": symbol,
            "baseAsset": symbol_info["baseAsset"],
            "quoteAsset": symbol_info["quoteAsset"],
        })
    
    return usdt_pairs


async def get_volumes_24h(client: AsyncClient) -> Dict[str, float]:
    """
    Récupère le volume 24h de toutes les paires.
    
    Returns:
        Dict symbol -> volume en USDT
    """
    tickers = await client.get_ticker()
    
    volumes = {}
    for ticker in tickers:
        symbol = ticker["symbol"]
        # quoteVolume = volume en USDT pour les paires USDT
        volumes[symbol] = float(ticker.get("quoteVolume", 0))
    
    return volumes


async def get_top_pairs(
    top_n: int = DEFAULT_TOP_N,
    min_volume: float = DEFAULT_MIN_VOLUME,
) -> List[str]:
    """
    Récupère les top N paires par volume.
    
    Args:
        top_n: Nombre de paires à retourner
        min_volume: Volume minimum 24h en USDT
        
    Returns:
        Liste des symboles triés par volume
    """
    client = await AsyncClient.create()
    
    try:
        # Récupérer les paires USDT
        print("📊 Récupération des paires USDT...")
        usdt_pairs = await get_all_usdt_pairs(client)
        print(f"   {len(usdt_pairs)} paires USDT trouvées")
        
        # Récupérer les volumes
        print("📈 Récupération des volumes 24h...")
        volumes = await get_volumes_24h(client)
        
        # Filtrer par volume minimum et trier
        pairs_with_volume = []
        for pair in usdt_pairs:
            symbol = pair["symbol"]
            volume = volumes.get(symbol, 0)
            
            if volume >= min_volume:
                pairs_with_volume.append({
                    "symbol": symbol,
                    "volume_24h": volume,
                    "base": pair["baseAsset"],
                })
        
        # Trier par volume décroissant
        pairs_with_volume.sort(key=lambda x: x["volume_24h"], reverse=True)
        
        # Prendre le top N
        top_pairs = pairs_with_volume[:top_n]
        
        return top_pairs
        
    finally:
        await client.close_connection()


def format_volume(volume: float) -> str:
    """Formate le volume pour affichage."""
    if volume >= 1_000_000_000:
        return f"{volume / 1_000_000_000:.1f}B"
    elif volume >= 1_000_000:
        return f"{volume / 1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"{volume / 1_000:.1f}K"
    else:
        return f"{volume:.0f}"


def print_pairs_table(pairs: List[Dict]) -> None:
    """Affiche les paires dans un tableau."""
    print("\n" + "=" * 50)
    print(f"{'#':<4} {'Symbol':<12} {'Base':<8} {'Volume 24h':>15}")
    print("=" * 50)
    
    for i, pair in enumerate(pairs, 1):
        vol_str = format_volume(pair["volume_24h"])
        print(f"{i:<4} {pair['symbol']:<12} {pair['base']:<8} {vol_str:>15} USDT")
    
    print("=" * 50)


def get_symbols_list(pairs: List[Dict]) -> List[str]:
    """Extrait la liste des symboles."""
    return [p["symbol"] for p in pairs]


def get_symbols_string(pairs: List[Dict]) -> str:
    """Retourne les symboles en string séparés par virgule."""
    return ",".join(get_symbols_list(pairs))


async def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Récupère les meilleures paires Binance par volume"
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Nombre de paires (défaut: {DEFAULT_TOP_N})"
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=DEFAULT_MIN_VOLUME,
        help=f"Volume minimum 24h en USDT (défaut: {DEFAULT_MIN_VOLUME:,.0f})"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Sauvegarder dans config/top_pairs.txt"
    )
    parser.add_argument(
        "--format",
        choices=["table", "list", "comma", "python"],
        default="table",
        help="Format de sortie"
    )
    
    args = parser.parse_args()
    
    print(f"🔍 Recherche des top {args.top} paires (volume min: {args.min_volume:,.0f} USDT)")
    print()
    
    # Récupérer les paires
    pairs = await get_top_pairs(top_n=args.top, min_volume=args.min_volume)
    
    if not pairs:
        print("❌ Aucune paire trouvée avec ces critères")
        return 1
    
    # Afficher selon le format
    if args.format == "table":
        print_pairs_table(pairs)
        print(f"\n📋 {len(pairs)} paires sélectionnées")
        print(f"\n💡 Pour copier la liste:")
        print(f"   {get_symbols_string(pairs)}")
        
    elif args.format == "list":
        for p in pairs:
            print(p["symbol"])
            
    elif args.format == "comma":
        print(get_symbols_string(pairs))
        
    elif args.format == "python":
        symbols = get_symbols_list(pairs)
        print("TOP_PAIRS = [")
        for i, s in enumerate(symbols):
            comma = "," if i < len(symbols) - 1 else ""
            print(f'    "{s}"{comma}')
        print("]")
    
    # Sauvegarder si demandé
    if args.save:
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        output_path = config_dir / "top_pairs.txt"
        with open(output_path, "w") as f:
            f.write("\n".join(get_symbols_list(pairs)))
        
        print(f"\n💾 Sauvegardé dans: {output_path}")
        
        # Aussi sauvegarder en format comma pour le download
        comma_path = config_dir / "top_pairs_comma.txt"
        with open(comma_path, "w") as f:
            f.write(get_symbols_string(pairs))
        
        print(f"💾 Sauvegardé dans: {comma_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))