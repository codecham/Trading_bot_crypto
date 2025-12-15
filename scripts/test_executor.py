#!/usr/bin/env python3
# scripts/test_executor.py
"""
Tests d'intégration pour la Phase 8 - Executor & Trade Logger.

Tests couverts:
- 8.1 Order Manager (executor.py)
  - Création et configuration
  - Simulation paper trading
  - Ordre market BUY/SELL
  - Ordre OCO
  - Gestion positions
  
- 8.2 Position Tracker
  - Dataclass Position
  - Dataclass CompletedTrade
  - Suivi positions ouvertes
  
- 8.3 Trade Logger
  - Sauvegarde CSV
  - Calcul statistiques
  - Export et résumés

Usage:
    python scripts/test_executor.py
"""

import sys
import os
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import logger, setup_logger


# ============================================
# TESTS 8.1 - ORDER MANAGER
# ============================================

def test_executor_imports():
    """Test 8.1.1 - Imports du module executor."""
    print("\n📦 Test 8.1.1 - Imports executor...")
    
    try:
        from cryptoscalper.trading.executor import (
            TradeExecutor,
            ExecutorConfig,
            SymbolInfo,
            OrderResult,
            Position,
            CompletedTrade,
            OrderType,
            OrderSide,
            OrderStatus,
            PositionStatus,
            CloseReason,
            create_executor,
        )
        print("  ✅ Tous les imports réussis")
        return True
    except ImportError as e:
        print(f"  ❌ Erreur import: {e}")
        return False


def test_executor_config():
    """Test 8.1.2 - Configuration de l'executor."""
    print("\n⚙️ Test 8.1.2 - ExecutorConfig...")
    
    from cryptoscalper.trading.executor import ExecutorConfig
    
    # Config par défaut
    config = ExecutorConfig()
    assert config.testnet is True, "Testnet devrait être True par défaut"
    assert config.paper_trading is False, "Paper trading devrait être False par défaut"
    assert config.max_retries == 3, "Max retries devrait être 3"
    
    # Config personnalisée
    config = ExecutorConfig(
        testnet=False,
        paper_trading=True,
        max_retries=5,
        order_timeout_seconds=60,
    )
    assert config.testnet is False
    assert config.paper_trading is True
    assert config.max_retries == 5
    
    print("  ✅ ExecutorConfig OK")
    return True


def test_symbol_info():
    """Test 8.1.3 - SymbolInfo et validation."""
    print("\n📊 Test 8.1.3 - SymbolInfo...")
    
    from cryptoscalper.trading.executor import SymbolInfo
    
    info = SymbolInfo(
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        min_notional=10.0,
        min_qty=0.00001,
        step_size=0.00001,
        tick_size=0.01,
    )
    
    # Test round_quantity
    qty = info.round_quantity(0.123456789)
    assert qty == 0.12345, f"Quantité arrondie incorrecte: {qty}"
    
    # Test round_price
    price = info.round_price(42123.456)
    assert price == 42123.45, f"Prix arrondi incorrect: {price}"
    
    # Test validate_order - valide
    is_valid, reason = info.validate_order(qty=0.001, price=42000)
    assert is_valid is True, f"Ordre devrait être valide: {reason}"
    
    # Test validate_order - notional trop bas
    is_valid, reason = info.validate_order(qty=0.0001, price=42000)
    # 0.0001 * 42000 = 4.2 USDT < 10 USDT min
    assert is_valid is False, "Ordre devrait être invalide (notional)"
    assert "minimum" in reason.lower()
    
    print(f"     Symbol: {info.symbol}")
    print(f"     Step size: {info.step_size}")
    print(f"     Min notional: {info.min_notional}")
    print("  ✅ SymbolInfo OK")
    return True


def test_order_result():
    """Test 8.1.4 - OrderResult dataclass."""
    print("\n📋 Test 8.1.4 - OrderResult...")
    
    from cryptoscalper.trading.executor import (
        OrderResult, OrderSide, OrderType, OrderStatus
    )
    
    result = OrderResult(
        order_id=12345,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        status=OrderStatus.FILLED,
        executed_qty=0.001,
        executed_price=42000.0,
        commission=0.042,
        commission_asset="USDT",
    )
    
    assert result.is_filled is True
    assert result.notional_value == 42.0
    
    print(f"     Order ID: {result.order_id}")
    print(f"     Notional: {result.notional_value:.2f} USDT")
    print(f"     Status: {result.status.value}")
    print("  ✅ OrderResult OK")
    return True


def test_position_dataclass():
    """Test 8.2.1 - Position dataclass."""
    print("\n📊 Test 8.2.1 - Position dataclass...")
    
    from cryptoscalper.trading.executor import Position, OrderSide, PositionStatus
    
    position = Position(
        position_id="POS_BTCUSDT_1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=0.001,
        entry_price=42000.0,
        entry_time=datetime.utcnow() - timedelta(minutes=2),
        stop_loss_price=41800.0,
        take_profit_price=42200.0,
        status=PositionStatus.OPEN,
    )
    
    assert position.is_open is True
    assert position.notional_value == 42.0
    assert position.age_seconds >= 120
    
    # Test PnL
    pnl = position.current_pnl(42100.0)  # Prix actuel +100
    assert pnl == 0.1, f"PnL incorrect: {pnl}"  # 0.001 * 100 = 0.1 USDT
    
    pnl_pct = position.current_pnl_percent(42100.0)
    assert abs(pnl_pct - 0.00238) < 0.001  # ~0.238%
    
    print(f"     Position ID: {position.position_id}")
    print(f"     Entry: {position.entry_price}")
    print(f"     Age: {position.age_seconds}s")
    print(f"     PnL @ 42100: {pnl:+.4f} USDT ({pnl_pct:+.2%})")
    print("  ✅ Position dataclass OK")
    return True


def test_completed_trade():
    """Test 8.2.2 - CompletedTrade dataclass."""
    print("\n📊 Test 8.2.2 - CompletedTrade dataclass...")
    
    from cryptoscalper.trading.executor import (
        Position, CompletedTrade, OrderSide, PositionStatus, CloseReason
    )
    
    # Créer une position
    position = Position(
        position_id="POS_BTCUSDT_2",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=0.001,
        entry_price=42000.0,
        entry_time=datetime.utcnow() - timedelta(minutes=3),
        stop_loss_price=41800.0,
        take_profit_price=42200.0,
        status=PositionStatus.OPEN,
    )
    
    # Convertir en CompletedTrade
    completed = CompletedTrade.from_position(
        position=position,
        exit_price=42150.0,  # +150 USDT
        exit_time=datetime.utcnow(),
        close_reason=CloseReason.TAKE_PROFIT,
        commission=0.05,
    )
    
    assert completed.is_winner is True
    assert completed.pnl_usdt == 0.15  # 0.001 * 150
    assert completed.duration_seconds >= 180
    assert completed.close_reason == CloseReason.TAKE_PROFIT
    
    print(f"     Trade ID: {completed.trade_id}")
    print(f"     PnL: {completed.pnl_usdt:+.4f} USDT ({completed.pnl_percent:+.2%})")
    print(f"     Duration: {completed.duration_seconds}s")
    print(f"     Close reason: {completed.close_reason.value}")
    print("  ✅ CompletedTrade dataclass OK")
    return True


def test_executor_creation():
    """Test 8.1.5 - Création de l'executor."""
    print("\n🔧 Test 8.1.5 - Création TradeExecutor...")
    
    from cryptoscalper.trading.executor import TradeExecutor, ExecutorConfig
    
    # Création avec config paper trading
    config = ExecutorConfig(testnet=True, paper_trading=True)
    executor = TradeExecutor(config)
    
    assert executor.is_connected is False
    assert executor.is_paper_trading is True
    assert executor.open_positions_count == 0
    
    print(f"     Paper trading: {executor.is_paper_trading}")
    print(f"     Connected: {executor.is_connected}")
    print("  ✅ Création TradeExecutor OK")
    return True


async def test_executor_connect():
    """Test 8.1.6 - Connexion de l'executor (paper mode)."""
    print("\n🔌 Test 8.1.6 - Connexion executor...")
    
    from cryptoscalper.trading.executor import TradeExecutor, ExecutorConfig
    
    config = ExecutorConfig(testnet=True, paper_trading=True)
    
    async with TradeExecutor(config) as executor:
        assert executor.is_connected is True
        
        # Test ping/symbol info
        try:
            info = await executor.get_symbol_info("BTCUSDT")
            print(f"     Symbol info loaded: {info.symbol}")
            print(f"     Min notional: {info.min_notional}")
        except Exception as e:
            print(f"     ⚠️ Impossible de charger symbol info: {e}")
    
    print("  ✅ Connexion executor OK")
    return True


async def test_paper_trading_buy():
    """Test 8.1.7 - Paper trading BUY."""
    print("\n📝 Test 8.1.7 - Paper trading BUY...")
    
    from cryptoscalper.trading.executor import TradeExecutor, ExecutorConfig
    
    config = ExecutorConfig(testnet=True, paper_trading=True)
    
    async with TradeExecutor(config) as executor:
        # Simuler un achat
        result = await executor.execute_market_buy("BTCUSDT", 0.001)
        
        assert result.is_filled is True
        assert result.executed_qty == 0.001
        assert result.executed_price > 0
        
        print(f"     Qty: {result.executed_qty}")
        print(f"     Price: {result.executed_price:.2f}")
        print(f"     Commission: {result.commission:.4f}")
    
    print("  ✅ Paper trading BUY OK")
    return True


async def test_paper_trading_sell():
    """Test 8.1.8 - Paper trading SELL."""
    print("\n📝 Test 8.1.8 - Paper trading SELL...")
    
    from cryptoscalper.trading.executor import TradeExecutor, ExecutorConfig
    
    config = ExecutorConfig(testnet=True, paper_trading=True)
    
    async with TradeExecutor(config) as executor:
        # Simuler une vente
        result = await executor.execute_market_sell("BTCUSDT", 0.001)
        
        assert result.is_filled is True
        assert result.executed_qty == 0.001
        
        print(f"     Qty: {result.executed_qty}")
        print(f"     Price: {result.executed_price:.2f}")
    
    print("  ✅ Paper trading SELL OK")
    return True


async def test_paper_trading_oco():
    """Test 8.1.9 - Paper trading OCO."""
    print("\n📝 Test 8.1.9 - Paper trading OCO...")
    
    from cryptoscalper.trading.executor import TradeExecutor, ExecutorConfig
    
    config = ExecutorConfig(testnet=True, paper_trading=True)
    
    async with TradeExecutor(config) as executor:
        # Prix actuel
        price = await executor._get_current_price("BTCUSDT")
        
        # Placer un OCO
        oco_id, sl_id, tp_id = await executor.place_oco_order(
            symbol="BTCUSDT",
            quantity=0.001,
            stop_loss_price=price * 0.99,  # -1%
            take_profit_price=price * 1.01,  # +1%
        )
        
        assert oco_id is not None
        assert sl_id is not None
        assert tp_id is not None
        
        print(f"     OCO ID: {oco_id}")
        print(f"     SL ID: {sl_id}")
        print(f"     TP ID: {tp_id}")
    
    print("  ✅ Paper trading OCO OK")
    return True


# ============================================
# TESTS 8.3 - TRADE LOGGER
# ============================================

def test_trade_logger_imports():
    """Test 8.3.1 - Imports du module trade_logger."""
    print("\n📦 Test 8.3.1 - Imports trade_logger...")
    
    try:
        from cryptoscalper.utils.trade_logger import (
            TradeLogger,
            TradeRecord,
            TradingStatistics,
            DailyStatistics,
            create_trade_logger,
        )
        print("  ✅ Tous les imports réussis")
        return True
    except ImportError as e:
        print(f"  ❌ Erreur import: {e}")
        return False


def test_trade_record():
    """Test 8.3.2 - TradeRecord dataclass."""
    print("\n📋 Test 8.3.2 - TradeRecord...")
    
    from cryptoscalper.utils.trade_logger import TradeRecord
    
    record = TradeRecord(
        trade_id="TRADE_001",
        symbol="BTCUSDT",
        side="BUY",
        entry_time="2025-01-01T10:00:00",
        exit_time="2025-01-01T10:05:00",
        entry_price=42000.0,
        exit_price=42100.0,
        quantity=0.001,
        pnl_usdt=0.1,
        pnl_percent=0.00238,
        close_reason="take_profit",
        duration_seconds=300,
        commission=0.05,
    )
    
    # Test to_dict
    data = record.to_dict()
    assert data["trade_id"] == "TRADE_001"
    assert data["pnl_usdt"] == 0.1
    
    print(f"     Trade ID: {record.trade_id}")
    print(f"     PnL: {record.pnl_usdt:+.4f} USDT")
    print("  ✅ TradeRecord OK")
    return True


def test_trade_logger_creation():
    """Test 8.3.3 - Création TradeLogger."""
    print("\n📊 Test 8.3.3 - Création TradeLogger...")
    
    from cryptoscalper.utils.trade_logger import TradeLogger
    
    # Créer dans un dossier temporaire
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "trades.csv")
        
        trade_logger = TradeLogger(csv_path)
        
        assert trade_logger.trades_count == 0
        assert trade_logger.csv_path.exists() is False  # Pas encore de trades
        
        print(f"     CSV path: {trade_logger.csv_path}")
        print(f"     Trades count: {trade_logger.trades_count}")
    
    print("  ✅ Création TradeLogger OK")
    return True


def test_trade_logger_log():
    """Test 8.3.4 - Logging des trades."""
    print("\n📝 Test 8.3.4 - Logging des trades...")
    
    from cryptoscalper.utils.trade_logger import TradeLogger, TradeRecord
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "trades.csv")
        trade_logger = TradeLogger(csv_path)
        
        # Logger plusieurs trades
        trades = [
            TradeRecord(
                trade_id="T1", symbol="BTCUSDT", side="BUY",
                entry_time="2025-01-01T10:00:00", exit_time="2025-01-01T10:05:00",
                entry_price=42000, exit_price=42100, quantity=0.001,
                pnl_usdt=0.1, pnl_percent=0.00238, close_reason="take_profit",
                duration_seconds=300, commission=0.05
            ),
            TradeRecord(
                trade_id="T2", symbol="ETHUSDT", side="BUY",
                entry_time="2025-01-01T11:00:00", exit_time="2025-01-01T11:03:00",
                entry_price=2500, exit_price=2480, quantity=0.01,
                pnl_usdt=-0.2, pnl_percent=-0.008, close_reason="stop_loss",
                duration_seconds=180, commission=0.03
            ),
            TradeRecord(
                trade_id="T3", symbol="BTCUSDT", side="BUY",
                entry_time="2025-01-01T12:00:00", exit_time="2025-01-01T12:04:00",
                entry_price=42050, exit_price=42150, quantity=0.001,
                pnl_usdt=0.1, pnl_percent=0.00238, close_reason="take_profit",
                duration_seconds=240, commission=0.05
            ),
        ]
        
        for trade in trades:
            trade_logger.log_trade(trade)
        
        assert trade_logger.trades_count == 3
        assert trade_logger.csv_path.exists() is True
        
        print(f"     Trades loggés: {trade_logger.trades_count}")
    
    print("  ✅ Logging des trades OK")
    return True


def test_trade_logger_statistics():
    """Test 8.3.5 - Calcul des statistiques."""
    print("\n📈 Test 8.3.5 - Statistiques de trading...")
    
    from cryptoscalper.utils.trade_logger import TradeLogger, TradeRecord
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "trades.csv")
        trade_logger = TradeLogger(csv_path)
        
        # Logger des trades variés
        trades = [
            # 3 gagnants
            TradeRecord("T1", "BTCUSDT", "BUY", "2025-01-01T10:00:00", "2025-01-01T10:05:00",
                       42000, 42100, 0.001, 0.1, 0.00238, "take_profit", 300, 0.05),
            TradeRecord("T2", "BTCUSDT", "BUY", "2025-01-01T11:00:00", "2025-01-01T11:03:00",
                       42050, 42200, 0.001, 0.15, 0.00357, "take_profit", 180, 0.05),
            TradeRecord("T3", "ETHUSDT", "BUY", "2025-01-01T12:00:00", "2025-01-01T12:02:00",
                       2500, 2530, 0.01, 0.3, 0.012, "take_profit", 120, 0.03),
            # 2 perdants
            TradeRecord("T4", "BTCUSDT", "BUY", "2025-01-01T13:00:00", "2025-01-01T13:01:00",
                       42200, 42100, 0.001, -0.1, -0.00237, "stop_loss", 60, 0.05),
            TradeRecord("T5", "ETHUSDT", "BUY", "2025-01-01T14:00:00", "2025-01-01T14:02:00",
                       2520, 2480, 0.01, -0.4, -0.0159, "stop_loss", 120, 0.03),
        ]
        
        for trade in trades:
            trade_logger.log_trade(trade)
        
        # Calculer les stats
        stats = trade_logger.get_statistics()
        
        assert stats.total_trades == 5
        assert stats.winning_trades == 3
        assert stats.losing_trades == 2
        assert stats.win_rate == 0.6  # 3/5
        
        # PnL total = 0.1 + 0.15 + 0.3 - 0.1 - 0.4 = 0.05
        assert abs(stats.total_pnl_usdt - 0.05) < 0.001
        
        # Profit factor = (0.1 + 0.15 + 0.3) / (0.1 + 0.4) = 0.55 / 0.5 = 1.1
        assert abs(stats.profit_factor - 1.1) < 0.01
        
        print(f"     Total trades: {stats.total_trades}")
        print(f"     Win rate: {stats.win_rate:.1%}")
        print(f"     PnL total: {stats.total_pnl_usdt:+.4f} USDT")
        print(f"     Profit factor: {stats.profit_factor:.2f}")
        print(f"     Take profits: {stats.take_profit_count}")
        print(f"     Stop losses: {stats.stop_loss_count}")
    
    print("  ✅ Statistiques de trading OK")
    return True


def test_trade_logger_export():
    """Test 8.3.6 - Export JSON et résumé."""
    print("\n📤 Test 8.3.6 - Export et résumé...")
    
    from cryptoscalper.utils.trade_logger import TradeLogger, TradeRecord
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "trades.csv")
        json_path = os.path.join(tmpdir, "trades.json")
        summary_path = os.path.join(tmpdir, "summary.txt")
        
        trade_logger = TradeLogger(csv_path)
        
        # Logger des trades
        trade_logger.log_trade(TradeRecord(
            "T1", "BTCUSDT", "BUY", "2025-01-01T10:00:00", "2025-01-01T10:05:00",
            42000, 42100, 0.001, 0.1, 0.00238, "take_profit", 300, 0.05
        ))
        
        # Export JSON
        trade_logger.export_to_json(json_path)
        assert os.path.exists(json_path)
        
        with open(json_path) as f:
            data = json.load(f)
            assert "trades" in data
            assert "statistics" in data
        
        # Export résumé
        trade_logger.export_summary(summary_path)
        assert os.path.exists(summary_path)
        
        with open(summary_path) as f:
            content = f.read()
            assert "TRADING" in content
            assert "Win rate" in content
        
        print(f"     JSON exporté: {json_path}")
        print(f"     Résumé exporté: {summary_path}")
    
    print("  ✅ Export et résumé OK")
    return True


def test_trade_logger_reload():
    """Test 8.3.7 - Rechargement des trades depuis CSV."""
    print("\n🔄 Test 8.3.7 - Rechargement depuis CSV...")
    
    from cryptoscalper.utils.trade_logger import TradeLogger, TradeRecord
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "trades.csv")
        
        # Premier logger - écrire des trades
        logger1 = TradeLogger(csv_path)
        logger1.log_trade(TradeRecord(
            "T1", "BTCUSDT", "BUY", "2025-01-01T10:00:00", "2025-01-01T10:05:00",
            42000, 42100, 0.001, 0.1, 0.00238, "take_profit", 300, 0.05
        ))
        logger1.log_trade(TradeRecord(
            "T2", "ETHUSDT", "BUY", "2025-01-01T11:00:00", "2025-01-01T11:05:00",
            2500, 2520, 0.01, 0.2, 0.008, "take_profit", 300, 0.03
        ))
        
        assert logger1.trades_count == 2
        
        # Deuxième logger - recharger
        logger2 = TradeLogger(csv_path)
        
        assert logger2.trades_count == 2
        assert logger2.trades[0].trade_id == "T1"
        assert logger2.trades[1].trade_id == "T2"
        
        print(f"     Trades rechargés: {logger2.trades_count}")
    
    print("  ✅ Rechargement depuis CSV OK")
    return True


# ============================================
# TESTS D'INTÉGRATION
# ============================================

async def test_full_trade_cycle():
    """Test intégration - Cycle complet de trading."""
    print("\n🔄 Test intégration - Cycle complet...")
    
    from cryptoscalper.trading.executor import (
        TradeExecutor, ExecutorConfig, CloseReason
    )
    from cryptoscalper.utils.trade_logger import TradeLogger
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "trades.csv")
        trade_logger = TradeLogger(csv_path)
        
        config = ExecutorConfig(testnet=True, paper_trading=True)
        
        async with TradeExecutor(config) as executor:
            # 1. Obtenir le prix actuel
            price = await executor._get_current_price("BTCUSDT")
            print(f"     Prix BTC: {price:.2f}")
            
            # 2. Acheter
            buy_result = await executor.execute_market_buy("BTCUSDT", 0.001)
            entry_price = buy_result.executed_price
            print(f"     Achat @ {entry_price:.2f}")
            
            # 3. Créer une position manuellement (simulé)
            from cryptoscalper.trading.executor import Position, OrderSide, PositionStatus
            
            position = Position(
                position_id="POS_TEST_1",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                quantity=buy_result.executed_qty,
                entry_price=entry_price,
                entry_time=datetime.utcnow(),
                stop_loss_price=entry_price * 0.995,
                take_profit_price=entry_price * 1.005,
                status=PositionStatus.OPEN,
            )
            
            executor._open_positions[position.position_id] = position
            assert executor.open_positions_count == 1
            
            # 4. Fermer la position
            completed = await executor.close_position(
                position, 
                reason=CloseReason.TAKE_PROFIT
            )
            
            assert completed is not None
            print(f"     PnL: {completed.pnl_usdt:+.4f} USDT")
            
            # 5. Logger le trade
            trade_logger.log_trade(completed)
            
            # 6. Vérifier les stats
            stats = trade_logger.get_statistics()
            assert stats.total_trades == 1
            print(f"     Stats: {stats.total_trades} trade(s)")
    
    print("  ✅ Cycle complet OK")
    return True


# ============================================
# MAIN
# ============================================

async def run_test_safe(name: str, test_coro_or_func, is_network_test: bool = False):
    """
    Exécute un test de manière sûre avec gestion des erreurs.
    
    Args:
        name: Nom du test
        test_coro_or_func: Coroutine ou fonction de test
        is_network_test: Si True, les erreurs réseau sont marquées comme SKIP
        
    Returns:
        Tuple (name, result, skipped)
    """
    try:
        if asyncio.iscoroutinefunction(test_coro_or_func):
            result = await test_coro_or_func()
        elif asyncio.iscoroutine(test_coro_or_func):
            result = await test_coro_or_func
        else:
            result = test_coro_or_func() if callable(test_coro_or_func) else test_coro_or_func
        return (name, result, False)
    except Exception as e:
        error_msg = str(e).lower()
        if is_network_test and ("name resolution" in error_msg or "connect" in error_msg or "network" in error_msg):
            print(f"  ⏭️  SKIP (pas d'accès réseau)")
            return (name, None, True)  # Skipped
        print(f"  ❌ ERREUR: {e}")
        return (name, False, False)


async def main():
    """Exécute tous les tests de la Phase 8."""
    print("=" * 65)
    print("🧪 CryptoScalper AI - Tests Phase 8: Executor & Trade Logger")
    print("=" * 65)
    
    setup_logger(level="WARNING")
    
    results = []
    skipped_count = 0
    
    # Tests 8.1 - Order Manager
    print("\n" + "─" * 50)
    print("📦 8.1 Order Manager - Executor")
    print("─" * 50)
    
    # Tests offline
    results.append(await run_test_safe("8.1.1 Imports executor", test_executor_imports))
    results.append(await run_test_safe("8.1.2 ExecutorConfig", test_executor_config))
    results.append(await run_test_safe("8.1.3 SymbolInfo", test_symbol_info))
    results.append(await run_test_safe("8.1.4 OrderResult", test_order_result))
    results.append(await run_test_safe("8.1.5 Création executor", test_executor_creation))
    
    # Tests réseau (peuvent être skippés)
    results.append(await run_test_safe("8.1.6 Connexion", test_executor_connect, is_network_test=True))
    results.append(await run_test_safe("8.1.7 Paper BUY", test_paper_trading_buy, is_network_test=True))
    results.append(await run_test_safe("8.1.8 Paper SELL", test_paper_trading_sell, is_network_test=True))
    results.append(await run_test_safe("8.1.9 Paper OCO", test_paper_trading_oco, is_network_test=True))
    
    # Tests 8.2 - Position Tracker
    print("\n" + "─" * 50)
    print("📦 8.2 Position Tracker")
    print("─" * 50)
    results.append(await run_test_safe("8.2.1 Position dataclass", test_position_dataclass))
    results.append(await run_test_safe("8.2.2 CompletedTrade", test_completed_trade))
    
    # Tests 8.3 - Trade Logger
    print("\n" + "─" * 50)
    print("📦 8.3 Trade Logger")
    print("─" * 50)
    results.append(await run_test_safe("8.3.1 Imports trade_logger", test_trade_logger_imports))
    results.append(await run_test_safe("8.3.2 TradeRecord", test_trade_record))
    results.append(await run_test_safe("8.3.3 Création TradeLogger", test_trade_logger_creation))
    results.append(await run_test_safe("8.3.4 Logging trades", test_trade_logger_log))
    results.append(await run_test_safe("8.3.5 Statistiques", test_trade_logger_statistics))
    results.append(await run_test_safe("8.3.6 Export", test_trade_logger_export))
    results.append(await run_test_safe("8.3.7 Rechargement CSV", test_trade_logger_reload))
    
    # Test d'intégration
    print("\n" + "─" * 50)
    print("📦 8.4 Tests d'Intégration")
    print("─" * 50)
    results.append(await run_test_safe("8.4.1 Cycle complet", test_full_trade_cycle, is_network_test=True))
    
    # Résumé
    print("\n" + "=" * 65)
    print("📊 RÉSULTATS PHASE 8")
    print("=" * 65)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result, was_skipped in results:
        if was_skipped:
            status = "⏭️  SKIP"
            skipped += 1
        elif result:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        print(f"  {name}: {status}")
    
    print("─" * 65)
    print(f"  Total: {passed} passés, {skipped} skippés, {failed} échoués / {len(results)} tests")
    print("=" * 65)
    
    if failed == 0:
        if skipped > 0:
            print(f"🎉 Phase 8 - Executor & Trade Logger : VALIDÉE ! ({skipped} tests réseau skippés)")
        else:
            print("🎉 Phase 8 - Executor & Trade Logger : VALIDÉE !")
        return 0
    else:
        print(f"⚠️  {failed} test(s) échoué(s)")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))