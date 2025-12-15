# scripts/test_websocket.py
"""
Script de test pour le WebSocket temps réel.

Usage:
    python scripts/test_websocket.py
"""

import asyncio
import sys
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import setup_logger, logger
from cryptoscalper.data.binance_client import BinanceClient
from cryptoscalper.data.pair_scanner import PairScanner
from cryptoscalper.data.websocket_manager import WebSocketManager, TickerData


# Compteur global pour le test
ticker_count = 0


def on_ticker_received(ticker: TickerData) -> None:
    """Callback appelé à chaque ticker reçu."""
    global ticker_count
    ticker_count += 1
    
    # Afficher seulement tous les 100 messages pour ne pas surcharger
    if ticker_count % 100 == 0:
        print(f"  📨 {ticker_count} tickers reçus | Dernier: {ticker.symbol} @ ${ticker.price:,.2f}")


async def test_websocket_basic():
    """Test basique du WebSocket avec quelques paires."""
    print("\n🔌 Test WebSocket basique (5 paires, 10 secondes)...")
    
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    
    async with BinanceClient() as client:
        manager = WebSocketManager(client._client)
        manager.on_ticker(on_ticker_received)
        
        await manager.start(symbols)
        
        print(f"  ✅ WebSocket démarré pour {len(symbols)} paires")
        print("  ⏳ Collecte pendant 10 secondes...")
        
        # Attendre 10 secondes
        await asyncio.sleep(10)
        
        # Afficher les états
        print("\n  📊 État des paires:")
        for symbol in symbols:
            state = manager.get_pair_state(symbol)
            if state and state.current_price > 0:
                change_1m = state.change_1m_percent
                change_str = f"{change_1m:+.2f}%" if change_1m else "N/A"
                print(f"     {symbol:<10} ${state.current_price:>10,.2f}  (1m: {change_str})")
        
        # Stats
        print(f"\n  📈 Stats WebSocket:")
        print(f"     Messages reçus: {manager.stats.messages_received}")
        print(f"     Messages/sec: {manager.stats.messages_per_second:.1f}")
        print(f"     Erreurs: {manager.stats.errors_count}")
        
        await manager.stop()
    
    return manager.stats.messages_received > 0


async def test_websocket_with_scanner():
    """Test WebSocket avec paires du scanner."""
    print("\n🔍 Test WebSocket avec Scanner (top 20 paires, 15 secondes)...")
    
    async with BinanceClient() as client:
        # Scanner les meilleures paires
        scanner = PairScanner(client._client)
        symbols = await scanner.get_symbols_list(min_volume_24h=10_000_000, max_pairs=20)
        
        print(f"  ✅ {len(symbols)} paires sélectionnées par le scanner")
        
        # Démarrer le WebSocket
        manager = WebSocketManager(client._client)
        await manager.start(symbols)
        
        print("  ⏳ Collecte pendant 15 secondes...")
        await asyncio.sleep(15)
        
        # Top movers
        top_movers = manager.get_top_movers(n=5, timeframe="1m")
        
        if top_movers:
            print("\n  🚀 Top 5 movers:")
            for i, state in enumerate(top_movers, 1):
                # Afficher la meilleure variation disponible
                change = state.change_1m_percent
                period = "1m"
                if change is None:
                    change = state.change_since_start_percent
                    period = "start"
                
                if change is not None:
                    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    print(f"     {i}. {direction} {state.symbol:<10} {change:+.4f}% ({period})")
        else:
            print("\n  ⚠️  Pas assez de données pour les top movers")
        
        await manager.stop()
    
    return len(top_movers) > 0


async def test_price_history():
    """Test de l'historique des prix."""
    print("\n📜 Test historique des prix (BTCUSDT, 30 secondes)...")
    
    async with BinanceClient() as client:
        manager = WebSocketManager(client._client)
        await manager.start(["BTCUSDT"])
        
        print("  ⏳ Collecte de l'historique pendant 30 secondes...")
        
        # Afficher le prix toutes les 10 secondes
        for i in range(3):
            await asyncio.sleep(10)
            state = manager.get_pair_state("BTCUSDT")
            
            if state:
                history_len = len(state.price_history)
                print(f"  [{(i+1)*10}s] Prix: ${state.current_price:,.2f} | Historique: {history_len} points")
        
        # Vérifier l'historique
        state = manager.get_pair_state("BTCUSDT")
        if state:
            print(f"\n  ✅ Historique collecté: {len(state.price_history)} points")
            
            if state.price_history:
                first_price = state.price_history[0][1]
                last_price = state.price_history[-1][1]
                variation = (last_price - first_price) / first_price * 100
                print(f"     Premier prix: ${first_price:,.2f}")
                print(f"     Dernier prix: ${last_price:,.2f}")
                print(f"     Variation: {variation:+.4f}%")
        
        await manager.stop()
    
    return len(state.price_history) > 0 if state else False


async def main():
    """Exécute tous les tests."""
    global ticker_count
    
    print("=" * 60)
    print("🧪 CryptoScalper AI - Test WebSocket Temps Réel")
    print("=" * 60)
    
    setup_logger(level="WARNING")
    
    results = []
    
    try:
        ticker_count = 0
        results.append(("WebSocket Basic", await test_websocket_basic()))
        
        ticker_count = 0
        results.append(("WebSocket + Scanner", await test_websocket_with_scanner()))
        
        results.append(("Price History", await test_price_history()))
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        logger.exception("Erreur test WebSocket")
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
        print("🎉 WebSocket temps réel opérationnel !")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))