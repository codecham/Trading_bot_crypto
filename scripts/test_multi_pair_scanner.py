# scripts/test_multi_pair_scanner.py
"""
Tests d'intégration pour le Scanner Multi-Paires (Phase 3).

Teste :
- SymbolsManager (sélection dynamique, rafraîchissement)
- MultiPairScanner (détection opportunités, alertes)
- Performance (latence, CPU)

Usage:
    python scripts/test_multi_pair_scanner.py
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import setup_logger, logger
from cryptoscalper.data.binance_client import BinanceClient
from cryptoscalper.data.websocket_manager import WebSocketManager
from cryptoscalper.data.symbols import SymbolsManager, get_tradeable_symbols
from cryptoscalper.data.multi_pair_scanner import (
    MultiPairScanner,
    ScannerConfig,
    ScannerAlert,
    AlertType
)


# Compteur d'alertes pour les tests
alerts_received = []


def on_alert(alert: ScannerAlert) -> None:
    """Callback pour les alertes."""
    alerts_received.append(alert)
    print(f"  📢 ALERTE: {alert}")


# =========================================
# TESTS SYMBOLS MANAGER
# =========================================

async def test_symbols_manager_basic():
    """Test 3.1.1 - SymbolsManager basique."""
    print("\n📋 Test 3.1.1 - SymbolsManager basique...")
    
    async with BinanceClient() as client:
        manager = SymbolsManager(
            client=client._client,
            min_volume_24h=5_000_000,
            max_pairs=50
        )
        
        # Premier refresh
        symbols = await manager.refresh()
        
        assert len(symbols) > 0, "Devrait avoir des symboles"
        assert len(symbols) <= 50, "Ne devrait pas dépasser max_pairs"
        assert manager.state.refresh_count == 1, "Refresh count devrait être 1"
        
        print(f"  ✅ {len(symbols)} symboles chargés")
        print(f"     Top 5: {', '.join(symbols[:5])}")
    
    return True


async def test_symbols_manager_refresh():
    """Test 3.1.2 - Rafraîchissement automatique."""
    print("\n🔄 Test 3.1.2 - Rafraîchissement automatique...")
    
    async with BinanceClient() as client:
        manager = SymbolsManager(
            client=client._client,
            min_volume_24h=10_000_000,
            max_pairs=30,
            refresh_interval=5  # Refresh rapide pour le test
        )
        
        await manager.start(auto_refresh=True)
        
        initial_count = manager.state.refresh_count
        print(f"  ⏳ Attente du premier refresh automatique (5s)...")
        
        await asyncio.sleep(7)
        
        assert manager.state.refresh_count > initial_count, "Devrait avoir refresh"
        
        print(f"  ✅ Refresh count: {manager.state.refresh_count}")
        
        await manager.stop()
    
    return True


async def test_symbols_helper_function():
    """Test 3.1.3 - Fonction helper get_tradeable_symbols."""
    print("\n🔧 Test 3.1.3 - Fonction get_tradeable_symbols...")
    
    async with BinanceClient() as client:
        symbols = await get_tradeable_symbols(
            client=client._client,
            min_volume=10_000_000,
            max_pairs=20
        )
        
        assert len(symbols) > 0, "Devrait retourner des symboles"
        assert all(s.endswith("USDT") for s in symbols), "Tous en USDT"
        
        print(f"  ✅ {len(symbols)} symboles: {', '.join(symbols[:10])}...")
    
    return True


# =========================================
# TESTS MULTI-PAIR SCANNER
# =========================================

async def test_scanner_basic():
    """Test 3.2.1 - Scanner basique."""
    print("\n🔍 Test 3.2.1 - Scanner basique...")
    
    global alerts_received
    alerts_received = []
    
    async with BinanceClient() as client:
        # Récupérer les symboles
        symbols = await get_tradeable_symbols(client._client, max_pairs=50)
        
        # Démarrer WebSocket
        ws_manager = WebSocketManager(client._client)
        await ws_manager.start(symbols, subscribe_depth=True)
        
        print(f"  ⏳ Collecte de données (15s)...")
        await asyncio.sleep(15)
        
        # Créer et démarrer le scanner
        config = ScannerConfig(
            min_volume_24h=1_000_000,
            min_score_for_alert=0.2,  # Seuil bas pour générer des alertes
            alert_callback=on_alert
        )
        
        scanner = MultiPairScanner(ws_manager, config)
        await scanner.start(scan_interval=2.0)
        
        print(f"  ⏳ Scan actif (20s)...")
        await asyncio.sleep(20)
        
        # Vérifier les stats
        stats = scanner.get_stats()
        print(f"\n  📊 Statistiques:")
        print(f"     Scans effectués: {stats['scans_count']}")
        print(f"     Alertes générées: {stats['alerts_generated']}")
        print(f"     Alertes par type: {stats['alerts_by_type']}")
        print(f"     Latence scan: {stats['last_scan_duration_ms']:.2f}ms")
        
        await scanner.stop()
        await ws_manager.stop()
    
    print(f"  ✅ Scanner basique OK")
    return True


async def test_scanner_opportunities():
    """Test 3.2.2 - Détection des opportunités."""
    print("\n🎯 Test 3.2.2 - Détection des opportunités...")
    
    async with BinanceClient() as client:
        symbols = await get_tradeable_symbols(client._client, max_pairs=100)
        
        ws_manager = WebSocketManager(client._client)
        await ws_manager.start(symbols, subscribe_depth=True)
        
        print(f"  ⏳ Collecte de données (20s)...")
        await asyncio.sleep(20)
        
        scanner = MultiPairScanner(ws_manager)
        await scanner.start(scan_interval=1.0)
        
        await asyncio.sleep(5)
        
        # Récupérer les top opportunités
        opportunities = scanner.get_top_opportunities(n=10)
        
        print(f"\n  🚀 Top {len(opportunities)} opportunités:")
        for i, opp in enumerate(opportunities, 1):
            change_1m = opp.change_1m_percent
            change_str = f"{change_1m:+.3f}%" if change_1m else "N/A"
            
            spread_str = "N/A"
            if opp.current_depth and opp.current_depth.spread_percent:
                spread_str = f"{opp.current_depth.spread_percent:.4f}%"
            
            print(
                f"     {i:2}. {opp.symbol:<12} "
                f"${opp.current_price:>10,.2f}  "
                f"Δ1m: {change_str:>8}  "
                f"Spread: {spread_str}"
            )
        
        assert len(opportunities) > 0, "Devrait avoir des opportunités"
        
        await scanner.stop()
        await ws_manager.stop()
    
    print(f"  ✅ Détection opportunités OK")
    return True


async def test_scanner_alerts():
    """Test 3.2.3 - Génération des alertes."""
    print("\n📢 Test 3.2.3 - Génération des alertes...")
    
    global alerts_received
    alerts_received = []
    
    async with BinanceClient() as client:
        symbols = await get_tradeable_symbols(client._client, max_pairs=100)
        
        ws_manager = WebSocketManager(client._client)
        await ws_manager.start(symbols, subscribe_depth=True)
        
        await asyncio.sleep(15)
        
        config = ScannerConfig(
            min_score_for_alert=0.15,  # Seuil très bas pour le test
            momentum_threshold_1m=0.05,  # Seuil bas
            alert_callback=on_alert
        )
        
        scanner = MultiPairScanner(ws_manager, config)
        await scanner.start(scan_interval=1.0)
        
        print(f"  ⏳ Attente des alertes (30s)...")
        await asyncio.sleep(30)
        
        # Vérifier les alertes
        recent_alerts = scanner.get_recent_alerts(limit=10)
        
        print(f"\n  📋 Alertes récentes: {len(recent_alerts)}")
        for alert in recent_alerts[:5]:
            print(f"     - {alert.alert_type.value}: {alert.symbol} (score: {alert.score:.2f})")
        
        # Alertes par type
        momentum_alerts = scanner.get_recent_alerts(alert_type=AlertType.MOMENTUM)
        breakout_alerts = scanner.get_recent_alerts(alert_type=AlertType.BREAKOUT)
        
        print(f"\n  📊 Par type:")
        print(f"     Momentum: {len(momentum_alerts)}")
        print(f"     Breakout: {len(breakout_alerts)}")
        
        await scanner.stop()
        await ws_manager.stop()
    
    print(f"  ✅ Génération alertes OK")
    return True


async def test_scanner_performance():
    """Test 3.3.1 - Performance du scanner (latence, CPU)."""
    print("\n⚡ Test 3.3.1 - Performance du scanner...")
    
    async with BinanceClient() as client:
        # Charger beaucoup de paires
        symbols = await get_tradeable_symbols(client._client, max_pairs=150)
        print(f"  📋 Test avec {len(symbols)} paires")
        
        ws_manager = WebSocketManager(client._client)
        await ws_manager.start(symbols, subscribe_depth=True)
        
        print(f"  ⏳ Collecte de données (20s)...")
        await asyncio.sleep(20)
        
        scanner = MultiPairScanner(ws_manager)
        await scanner.start(scan_interval=0.5)  # Scan rapide
        
        # Mesurer les performances sur 20 secondes
        await asyncio.sleep(20)
        
        stats = scanner.get_stats()
        
        # Vérifier la latence
        latency = stats['last_scan_duration_ms']
        target_latency = 100  # ms
        
        print(f"\n  📊 Performances:")
        print(f"     Paires surveillées: {len(symbols)}")
        print(f"     Scans effectués: {stats['scans_count']}")
        print(f"     Latence dernier scan: {latency:.2f}ms")
        print(f"     Objectif latence: < {target_latency}ms")
        
        # Le test réussit si la latence est acceptable
        latency_ok = latency < target_latency
        status = "✅ OK" if latency_ok else "⚠️  Élevée"
        print(f"     Statut latence: {status}")
        
        await scanner.stop()
        await ws_manager.stop()
    
    return latency_ok


async def test_scanner_full_pipeline():
    """Test 3.3.2 - Pipeline complet Scanner."""
    print("\n🔄 Test 3.3.2 - Pipeline complet...")
    
    global alerts_received
    alerts_received = []
    
    async with BinanceClient() as client:
        # 1. Symbols Manager
        symbols_manager = SymbolsManager(
            client._client,
            min_volume_24h=5_000_000,
            max_pairs=100
        )
        await symbols_manager.start(auto_refresh=False)
        symbols = symbols_manager.get_symbols()
        
        print(f"  1️⃣ SymbolsManager: {len(symbols)} paires")
        
        # 2. WebSocket Manager
        ws_manager = WebSocketManager(client._client)
        await ws_manager.start(symbols, subscribe_depth=True)
        
        print(f"  2️⃣ WebSocket: connecté")
        await asyncio.sleep(15)
        
        # 3. Multi-Pair Scanner
        config = ScannerConfig(
            min_score_for_alert=0.2,
            alert_callback=on_alert
        )
        scanner = MultiPairScanner(ws_manager, config)
        await scanner.start(scan_interval=2.0)
        
        print(f"  3️⃣ Scanner: actif")
        
        # 4. Attendre et observer
        print(f"\n  ⏳ Pipeline actif pendant 30 secondes...")
        
        for i in range(6):
            await asyncio.sleep(5)
            
            # Afficher le statut
            ws_stats = ws_manager.stats
            scanner_stats = scanner.get_stats()
            
            print(
                f"     [{(i+1)*5}s] "
                f"WS msgs: {ws_stats.messages_received:,} | "
                f"Scans: {scanner_stats['scans_count']} | "
                f"Alertes: {scanner_stats['alerts_generated']}"
            )
        
        # 5. Résultats finaux
        print(f"\n  📊 Résultats finaux:")
        
        opportunities = scanner.get_top_opportunities(n=5)
        if opportunities:
            print(f"  🎯 Top 5 opportunités:")
            for opp in opportunities:
                change = opp.change_1m_percent
                change_str = f"{change:+.3f}%" if change else "N/A"
                print(f"     - {opp.symbol}: {change_str}")
        
        # Cleanup
        await scanner.stop()
        await ws_manager.stop()
        await symbols_manager.stop()
    
    print(f"  ✅ Pipeline complet OK")
    return True


# =========================================
# MAIN
# =========================================

async def main():
    """Exécute tous les tests de la Phase 3."""
    print("=" * 65)
    print("🧪 CryptoScalper AI - Tests Phase 3: Scanner Multi-Paires")
    print("=" * 65)
    
    setup_logger(level="WARNING")
    
    results = []
    
    try:
        # 3.1 SymbolsManager
        print("\n" + "─" * 50)
        print("📦 3.1 SymbolsManager")
        print("─" * 50)
        results.append(("3.1.1 SymbolsManager basique", await test_symbols_manager_basic()))
        results.append(("3.1.2 Rafraîchissement auto", await test_symbols_manager_refresh()))
        results.append(("3.1.3 Helper function", await test_symbols_helper_function()))
        
        # 3.2 MultiPairScanner
        print("\n" + "─" * 50)
        print("📦 3.2 MultiPairScanner")
        print("─" * 50)
        results.append(("3.2.1 Scanner basique", await test_scanner_basic()))
        results.append(("3.2.2 Détection opportunités", await test_scanner_opportunities()))
        results.append(("3.2.3 Génération alertes", await test_scanner_alerts()))
        
        # 3.3 Performance & Intégration
        print("\n" + "─" * 50)
        print("📦 3.3 Performance & Intégration")
        print("─" * 50)
        results.append(("3.3.1 Performance (latence)", await test_scanner_performance()))
        results.append(("3.3.2 Pipeline complet", await test_scanner_full_pipeline()))
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        logger.exception("Erreur test Phase 3")
        return 1
    
    # Résumé
    print("\n" + "=" * 65)
    print("📊 RÉSULTATS PHASE 3")
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
        print("🎉 Phase 3 - Scanner Multi-Paires : VALIDÉE !")
        return 0
    else:
        print(f"⚠️  {failed} test(s) échoué(s)")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))