# scripts/test_phase2.py
"""
Tests d'intégration pour la Phase 2 - Collecte Temps Réel.

Teste :
- WebSocket Manager (connexion, reconnexion, streams)
- Data Collector (interface unifiée)
- Tous les types de données (ticker, klines, orderbook)

Usage:
    python scripts/test_phase2.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import setup_logger, logger
from cryptoscalper.data.binance_client import BinanceClient
from cryptoscalper.data.websocket_manager import WebSocketManager, TickerData, KlineData, DepthData
from cryptoscalper.data.collector import DataCollector, CollectorConfig


# Compteurs pour les callbacks
callback_counts = {
    "ticker": 0,
    "kline": 0,
    "depth": 0
}


def on_ticker(data: TickerData) -> None:
    """Callback ticker."""
    callback_counts["ticker"] += 1


def on_kline(data: KlineData) -> None:
    """Callback kline."""
    callback_counts["kline"] += 1


def on_depth(data: DepthData) -> None:
    """Callback depth."""
    callback_counts["depth"] += 1


# =========================================
# TESTS WEBSOCKET MANAGER
# =========================================

async def test_ws_connection():
    """Test 2.1.1 - Connexion WebSocket unique."""
    print("\n📡 Test 2.1.1 - Connexion WebSocket...")
    
    async with BinanceClient() as client:
        manager = WebSocketManager(client._client)
        await manager.start(["BTCUSDT", "ETHUSDT"])
        
        assert manager.is_running, "WebSocket devrait être running"
        assert len(manager.symbols) == 2, "Devrait avoir 2 symboles"
        
        await asyncio.sleep(3)
        
        assert manager.stats.messages_received > 0, "Devrait avoir reçu des messages"
        
        await manager.stop()
        assert not manager.is_running, "WebSocket devrait être arrêté"
    
    print("  ✅ Connexion WebSocket OK")
    return True


async def test_ws_reconnection():
    """Test 2.1.3 - Gestion reconnexion (test des mécanismes)."""
    print("\n🔄 Test 2.1.3 - Mécanismes de reconnexion...")
    
    async with BinanceClient() as client:
        manager = WebSocketManager(client._client)
        
        # Vérifier les constantes de reconnexion
        assert manager.RECONNECT_DELAY_INITIAL == 1.0
        assert manager.RECONNECT_DELAY_MAX == 60.0
        assert manager.RECONNECT_DELAY_MULTIPLIER == 2.0
        
        await manager.start(["BTCUSDT"])
        await asyncio.sleep(2)
        
        # Vérifier que le compteur de reconnexion existe
        assert hasattr(manager.stats, "reconnections")
        
        await manager.stop()
    
    print("  ✅ Mécanismes de reconnexion présents")
    return True


async def test_ws_error_handling():
    """Test 2.1.4 - Gestion des erreurs."""
    print("\n⚠️  Test 2.1.4 - Gestion des erreurs...")
    
    async with BinanceClient() as client:
        manager = WebSocketManager(client._client)
        await manager.start(["BTCUSDT"])
        
        await asyncio.sleep(2)
        
        # Vérifier que le compteur d'erreurs existe et est initialisé
        assert hasattr(manager.stats, "errors_count")
        initial_errors = manager.stats.errors_count
        
        # Le manager devrait gérer les erreurs sans crash
        # (on ne peut pas facilement simuler une erreur ici)
        
        await manager.stop()
    
    print("  ✅ Gestion des erreurs OK")
    return True


async def test_ws_ticker_stream():
    """Test 2.2.1 - Stream ticker (prix)."""
    print("\n💰 Test 2.2.1 - Stream ticker...")
    
    global callback_counts
    callback_counts["ticker"] = 0
    
    async with BinanceClient() as client:
        manager = WebSocketManager(client._client)
        manager.on_ticker(on_ticker)
        
        await manager.start(["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        await asyncio.sleep(5)
        
        # Vérifier qu'on a reçu des tickers
        assert callback_counts["ticker"] > 0, "Devrait avoir reçu des tickers"
        assert manager.stats.ticker_messages > 0, "Stats ticker > 0"
        
        # Vérifier les prix
        for symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
            state = manager.get_pair_state(symbol)
            assert state is not None, f"{symbol} devrait avoir un state"
            assert state.current_price > 0, f"{symbol} devrait avoir un prix > 0"
        
        print(f"     Tickers reçus: {callback_counts['ticker']}")
        
        await manager.stop()
    
    print("  ✅ Stream ticker OK")
    return True


async def test_ws_kline_stream():
    """Test 2.2.2 - Stream klines 1m."""
    print("\n📊 Test 2.2.2 - Stream klines...")
    
    global callback_counts
    callback_counts["kline"] = 0
    
    async with BinanceClient() as client:
        manager = WebSocketManager(client._client)
        manager.on_kline(on_kline)
        
        await manager.start(
            ["BTCUSDT", "ETHUSDT"],
            subscribe_klines=True
        )
        
        # Attendre assez pour recevoir des klines
        await asyncio.sleep(10)
        
        # Vérifier qu'on a reçu des klines
        assert manager.stats.kline_messages > 0, "Devrait avoir reçu des klines"
        
        # Vérifier la kline courante
        btc_state = manager.get_pair_state("BTCUSDT")
        if btc_state and btc_state.current_kline:
            kline = btc_state.current_kline
            print(f"     Kline BTC: O={kline.open:.2f} H={kline.high:.2f} L={kline.low:.2f} C={kline.close:.2f}")
        
        print(f"     Klines reçues: {manager.stats.kline_messages}")
        
        await manager.stop()
    
    print("  ✅ Stream klines OK")
    return True


async def test_ws_depth_stream():
    """Test 2.2.3 - Stream orderbook."""
    print("\n📖 Test 2.2.3 - Stream orderbook...")
    
    global callback_counts
    callback_counts["depth"] = 0
    
    async with BinanceClient() as client:
        manager = WebSocketManager(client._client)
        manager.on_depth(on_depth)
        
        await manager.start(
            ["BTCUSDT"],
            subscribe_depth=True,
            depth_level=10
        )
        
        await asyncio.sleep(5)
        
        # Vérifier qu'on a reçu des depths
        assert manager.stats.depth_messages > 0, "Devrait avoir reçu des orderbooks"
        
        # Vérifier l'orderbook
        btc_state = manager.get_pair_state("BTCUSDT")
        if btc_state and btc_state.current_depth:
            depth = btc_state.current_depth
            print(f"     Best Bid: ${depth.best_bid:,.2f}")
            print(f"     Best Ask: ${depth.best_ask:,.2f}")
            print(f"     Spread: {depth.spread_percent:.4f}%")
            print(f"     Imbalance: {depth.imbalance:+.2f}")
        
        print(f"     Depths reçus: {manager.stats.depth_messages}")
        
        await manager.stop()
    
    print("  ✅ Stream orderbook OK")
    return True


async def test_ws_price_history():
    """Test 2.2.4 - Buffer circulaire historique."""
    print("\n📜 Test 2.2.4 - Buffer circulaire historique...")
    
    async with BinanceClient() as client:
        manager = WebSocketManager(client._client)
        await manager.start(["BTCUSDT"])
        
        # Collecter pendant 15 secondes
        await asyncio.sleep(15)
        
        state = manager.get_pair_state("BTCUSDT")
        assert state is not None
        
        # Vérifier l'historique
        history_len = len(state.price_history)
        assert history_len > 0, "Devrait avoir un historique"
        
        print(f"     Points dans l'historique: {history_len}")
        
        # Tester la récupération par timestamp
        if history_len >= 5:
            first_ts, first_price = state.price_history[0]
            last_ts, last_price = state.price_history[-1]
            
            print(f"     Premier: ${first_price:,.2f} @ {first_ts.strftime('%H:%M:%S')}")
            print(f"     Dernier: ${last_price:,.2f} @ {last_ts.strftime('%H:%M:%S')}")
            
            change = state.change_since_start_percent
            if change:
                print(f"     Variation: {change:+.4f}%")
        
        await manager.stop()
    
    print("  ✅ Buffer circulaire OK")
    return True


# =========================================
# TESTS DATA COLLECTOR
# =========================================

async def test_collector_interface():
    """Test 2.3.1 - Interface unifiée Data Collector."""
    print("\n🎯 Test 2.3.1 - Interface unifiée DataCollector...")
    
    config = CollectorConfig(
        symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        enable_websocket=True,
        subscribe_klines=True,
        subscribe_depth=True
    )
    
    async with DataCollector(config) as collector:
        assert collector.is_running, "Collector devrait être running"
        
        await asyncio.sleep(5)
        
        # Test get_price
        btc_price = collector.get_price("BTCUSDT")
        assert btc_price is not None and btc_price > 0, "BTC price devrait être > 0"
        print(f"     get_price('BTCUSDT'): ${btc_price:,.2f}")
        
        # Test get_all_prices
        prices = collector.get_all_prices()
        assert len(prices) >= 2, "Devrait avoir plusieurs prix"
        print(f"     get_all_prices(): {len(prices)} paires")
        
        # Test get_pair_state
        state = collector.get_pair_state("BTCUSDT")
        assert state is not None
        print(f"     get_pair_state(): OK")
        
        # Test get_depth
        depth = collector.get_depth("BTCUSDT")
        if depth:
            print(f"     get_depth(): spread={depth.spread_percent:.4f}%")
        
        # Test get_current_kline
        kline = collector.get_current_kline("BTCUSDT")
        if kline:
            print(f"     get_current_kline(): close=${kline.close:,.2f}")
        
        # Test get_top_movers
        movers = collector.get_top_movers(n=3)
        print(f"     get_top_movers(): {len(movers)} paires")
    
    print("  ✅ Interface unifiée OK")
    return True


async def test_collector_rest_api():
    """Test 2.3.2 - Données REST via Collector."""
    print("\n🌐 Test 2.3.2 - Données REST via Collector...")
    
    config = CollectorConfig(
        symbols=["BTCUSDT"],
        enable_websocket=False  # Désactiver WebSocket pour ce test
    )
    
    async with DataCollector(config) as collector:
        # Test fetch_price (REST)
        price = await collector.fetch_price("BTCUSDT")
        assert price is not None and price > 0
        print(f"     fetch_price(): ${price:,.2f}")
        
        # Test fetch_klines (REST)
        klines = await collector.fetch_klines("BTCUSDT", limit=10)
        assert len(klines) == 10
        print(f"     fetch_klines(): {len(klines)} bougies")
        
        # Test fetch_orderbook (REST)
        orderbook = await collector.fetch_orderbook("BTCUSDT", limit=5)
        assert orderbook is not None
        assert len(orderbook.bids) == 5
        print(f"     fetch_orderbook(): {len(orderbook.bids)} bids, {len(orderbook.asks)} asks")
    
    print("  ✅ Données REST OK")
    return True


async def test_collector_summary():
    """Test 2.3.3 - Statistiques et résumé."""
    print("\n📊 Test 2.3.3 - Statistiques Collector...")
    
    config = CollectorConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        subscribe_klines=True,
        subscribe_depth=True
    )
    
    async with DataCollector(config) as collector:
        await asyncio.sleep(5)
        
        summary = collector.get_summary()
        
        print(f"     Running: {summary['running']}")
        print(f"     Uptime: {summary['uptime_seconds']:.1f}s")
        print(f"     Symbols: {summary['symbols_count']}")
        print(f"     Messages reçus: {summary['websocket']['messages_received']}")
        print(f"     Messages/sec: {summary['websocket']['messages_per_second']:.1f}")
        print(f"     Streams actifs: ticker={summary['streams']['ticker']}, "
              f"klines={summary['streams']['klines']}, "
              f"depth={summary['streams']['depth']}")
        
        assert summary["running"] is True
        assert summary["websocket"]["messages_received"] > 0
    
    print("  ✅ Statistiques OK")
    return True


# =========================================
# MAIN
# =========================================

async def main():
    """Exécute tous les tests de la Phase 2."""
    print("=" * 65)
    print("🧪 CryptoScalper AI - Tests Phase 2: Collecte Temps Réel")
    print("=" * 65)
    
    setup_logger(level="WARNING")
    
    results = []
    
    try:
        # 2.1 WebSocket Manager
        print("\n" + "─" * 40)
        print("📦 2.1 WebSocket Manager")
        print("─" * 40)
        results.append(("2.1.1 Connexion WS", await test_ws_connection()))
        results.append(("2.1.3 Reconnexion", await test_ws_reconnection()))
        results.append(("2.1.4 Erreurs", await test_ws_error_handling()))
        
        # 2.2 Streams de données
        print("\n" + "─" * 40)
        print("📦 2.2 Streams de données")
        print("─" * 40)
        results.append(("2.2.1 Stream ticker", await test_ws_ticker_stream()))
        results.append(("2.2.2 Stream klines", await test_ws_kline_stream()))
        results.append(("2.2.3 Stream depth", await test_ws_depth_stream()))
        results.append(("2.2.4 Buffer historique", await test_ws_price_history()))
        
        # 2.3 Data Collector
        print("\n" + "─" * 40)
        print("📦 2.3 Data Collector")
        print("─" * 40)
        results.append(("2.3.1 Interface unifiée", await test_collector_interface()))
        results.append(("2.3.2 Données REST", await test_collector_rest_api()))
        results.append(("2.3.3 Statistiques", await test_collector_summary()))
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        logger.exception("Erreur test Phase 2")
        return 1
    
    # Résumé
    print("\n" + "=" * 65)
    print("📊 RÉSULTATS PHASE 2")
    print("=" * 65)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("─" * 65)
    print(f"  Total: {passed}/{len(results)} tests passés")
    print("=" * 65)
    
    if failed == 0:
        print("🎉 Phase 2 - Collecte Temps Réel : VALIDÉE !")
        return 0
    else:
        print(f"⚠️  {failed} test(s) échoué(s)")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))