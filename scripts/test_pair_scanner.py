# scripts/test_pair_scanner.py
"""
Script de test pour le scanner de paires.

Usage:
    python scripts/test_pair_scanner.py
"""

import asyncio
import sys
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import setup_logger, logger
from cryptoscalper.data.binance_client import BinanceClient
from cryptoscalper.data.pair_scanner import PairScanner


async def test_scan_pairs():
    """Test du scan des paires."""
    print("\n🔍 Test scan des paires USDT...")
    
    async with BinanceClient() as client:
        scanner = PairScanner(client._client)
        result = await scanner.scan(min_volume_24h=1_000_000, max_pairs=50)
        
        print(f"  ✅ Scan terminé en {result.scan_duration_ms:.0f}ms")
        print(f"     Total paires trouvées: {result.total_pairs_found}")
        print(f"     Après filtres: {result.pairs_after_filters}")
        
        # Top 10 par volume
        print("\n  📊 Top 10 par volume 24h:")
        for i, pair in enumerate(result.get_top_by_volume(10), 1):
            print(
                f"     {i:2}. {pair.symbol:<12} "
                f"Vol: ${pair.volume_24h/1e6:>8.1f}M  "
                f"Prix: ${pair.price:>10,.2f}  "
                f"Chg: {pair.price_change_percent_24h:>+6.2f}%"
            )
    
    return True


async def test_top_volatility():
    """Test récupération paires les plus volatiles."""
    print("\n📈 Test paires les plus volatiles...")
    
    async with BinanceClient() as client:
        scanner = PairScanner(client._client)
        result = await scanner.scan(min_volume_24h=5_000_000, max_pairs=100)
        
        print("  ✅ Top 10 par volatilité 24h:")
        for i, pair in enumerate(result.get_top_by_volatility(10), 1):
            print(
                f"     {i:2}. {pair.symbol:<12} "
                f"Volatilité: {pair.volatility_24h*100:>5.2f}%  "
                f"Range: ${pair.low_24h:,.2f} - ${pair.high_24h:,.2f}"
            )
    
    return True


async def test_bullish_pairs():
    """Test récupération paires en hausse."""
    print("\n🟢 Test paires en hausse (bullish)...")
    
    async with BinanceClient() as client:
        scanner = PairScanner(client._client)
        result = await scanner.scan(min_volume_24h=10_000_000, max_pairs=50)
        
        bullish = result.get_bullish_pairs()
        bearish_count = len(result.pairs) - len(bullish)
        
        print(f"  ✅ Paires en hausse: {len(bullish)}/{len(result.pairs)}")
        print(f"     Paires en baisse: {bearish_count}")
        
        if bullish:
            # Top 5 hausses
            top_gainers = sorted(bullish, key=lambda p: p.price_change_percent_24h, reverse=True)[:5]
            print("\n  🚀 Top 5 hausses:")
            for pair in top_gainers:
                print(f"     {pair.symbol:<12} +{pair.price_change_percent_24h:.2f}%")
    
    return True


async def test_get_symbols_list():
    """Test récupération liste de symboles."""
    print("\n📋 Test liste de symboles pour WebSocket...")
    
    async with BinanceClient() as client:
        scanner = PairScanner(client._client)
        symbols = await scanner.get_symbols_list(min_volume_24h=5_000_000, max_pairs=20)
        
        print(f"  ✅ {len(symbols)} symboles récupérés:")
        print(f"     {', '.join(symbols[:10])}...")
    
    return True


async def test_single_pair_info():
    """Test récupération info d'une paire."""
    print("\n🔎 Test info paire unique (BTCUSDT)...")
    
    async with BinanceClient() as client:
        scanner = PairScanner(client._client)
        pair = await scanner.get_pair_info("BTCUSDT")
        
        if pair:
            print(f"  ✅ {pair.symbol}")
            print(f"     Prix: ${pair.price:,.2f}")
            print(f"     Volume 24h: ${pair.volume_24h/1e6:.1f}M")
            print(f"     Variation 24h: {pair.price_change_percent_24h:+.2f}%")
            print(f"     Volatilité 24h: {pair.volatility_24h*100:.2f}%")
            print(f"     Trades 24h: {pair.trades_count_24h:,}")
            return True
        else:
            print("  ❌ Paire non trouvée")
            return False


async def main():
    """Exécute tous les tests."""
    print("=" * 60)
    print("🧪 CryptoScalper AI - Test Scanner de Paires")
    print("=" * 60)
    
    setup_logger(level="WARNING")  # Moins de logs pour lisibilité
    
    results = []
    
    try:
        results.append(("Scan Paires", await test_scan_pairs()))
        results.append(("Top Volatilité", await test_top_volatility()))
        results.append(("Paires Bullish", await test_bullish_pairs()))
        results.append(("Liste Symboles", await test_get_symbols_list()))
        results.append(("Info Paire", await test_single_pair_info()))
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        logger.exception("Erreur test scanner")
        return 1
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 Scanner de paires opérationnel !")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))