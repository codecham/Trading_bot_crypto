# scripts/test_binance_connection.py
"""
Script de test pour vérifier la connexion à Binance.

Usage:
    python scripts/test_binance_connection.py
"""

import asyncio
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import setup_logger, logger
from cryptoscalper.data.binance_client import BinanceClient


async def test_connection():
    """Teste la connexion de base."""
    print("\n🔌 Test de connexion à Binance...")
    
    async with BinanceClient() as client:
        # Test ping
        is_alive = await client.ping()
        if is_alive:
            print("  ✅ Ping OK - Connexion établie")
        else:
            print("  ❌ Ping échoué")
            return False
        
        # Afficher le mode
        mode = "🧪 TESTNET" if client.is_testnet else "🔴 PRODUCTION"
        print(f"  📍 Mode: {mode}")
    
    return True


async def test_get_price():
    """Teste la récupération du prix."""
    print("\n💰 Test récupération du prix BTC...")
    
    async with BinanceClient() as client:
        ticker = await client.get_price("BTCUSDT")
        print(f"  ✅ BTCUSDT = {ticker.price:,.2f} USDT")
        print(f"     Timestamp: {ticker.timestamp}")
    
    return True


async def test_get_klines():
    """Teste la récupération des bougies."""
    print("\n📊 Test récupération des klines...")
    
    async with BinanceClient() as client:
        klines = await client.get_klines("BTCUSDT", interval="1m", limit=5)
        print(f"  ✅ {len(klines)} bougies récupérées")
        
        # Afficher la dernière bougie
        last = klines[-1]
        print(f"     Dernière bougie:")
        print(f"     - Open:  {last.open:,.2f}")
        print(f"     - High:  {last.high:,.2f}")
        print(f"     - Low:   {last.low:,.2f}")
        print(f"     - Close: {last.close:,.2f}")
        print(f"     - Volume: {last.volume:,.4f}")
    
    return True


async def test_get_orderbook():
    """Teste la récupération de l'orderbook."""
    print("\n📖 Test récupération de l'orderbook...")
    
    async with BinanceClient() as client:
        orderbook = await client.get_orderbook("BTCUSDT", limit=5)
        print(f"  ✅ Orderbook récupéré")
        print(f"     Best Bid: {orderbook.best_bid:,.2f}")
        print(f"     Best Ask: {orderbook.best_ask:,.2f}")
        print(f"     Spread: {orderbook.spread:,.2f} ({orderbook.spread_percent:.4%})")
    
    return True


async def test_get_balance():
    """Teste la récupération du solde."""
    print("\n💵 Test récupération du solde USDT...")
    
    async with BinanceClient() as client:
        balance = await client.get_account_balance("USDT")
        print(f"  ✅ Solde USDT: {balance:,.2f}")
        
        # Sur testnet, on devrait avoir du solde fictif
        if client.is_testnet and balance > 0:
            print("     (Solde fictif du testnet)")
    
    return True


async def test_invalid_symbol():
    """Teste la gestion d'erreur pour symbole invalide."""
    print("\n⚠️  Test gestion d'erreur (symbole invalide)...")
    
    from cryptoscalper.utils.exceptions import InvalidSymbolError
    
    async with BinanceClient() as client:
        try:
            await client.get_price("INVALIDPAIR")
            print("  ❌ Aurait dû lever une exception")
            return False
        except InvalidSymbolError as e:
            print(f"  ✅ Exception correctement levée: {e}")
            return True


async def main():
    """Exécute tous les tests."""
    print("=" * 55)
    print("🧪 CryptoScalper AI - Test Connexion Binance")
    print("=" * 55)
    
    # Setup logger en mode DEBUG pour voir les détails
    setup_logger(level="INFO")
    
    results = []
    
    try:
        results.append(("Connexion", await test_connection()))
        results.append(("Get Price", await test_get_price()))
        results.append(("Get Klines", await test_get_klines()))
        results.append(("Get Orderbook", await test_get_orderbook()))
        results.append(("Get Balance", await test_get_balance()))
        results.append(("Invalid Symbol", await test_invalid_symbol()))
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        logger.exception("Erreur lors des tests")
        return 1
    
    # Résumé
    print("\n" + "=" * 55)
    print("📊 RÉSULTATS")
    print("=" * 55)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 55)
    
    if all_passed:
        print("🎉 Tous les tests sont passés ! Connexion Binance OK.")
        return 0
    else:
        print("⚠️  Certains tests ont échoué.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)