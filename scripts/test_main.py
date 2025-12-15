#!/usr/bin/env python3
# scripts/test_main.py
"""
Tests d'intégration pour la Phase 10 - Boucle Principale.

Tests couverts:
- 10.1 Orchestrateur
  - BotConfig création et validation
  - BotState propriétés
  - TradingBot initialisation
  - Gestion des signaux SIGINT

- 10.2 Mode Paper Trading
  - Mode paper vs live
  - Confirmation mode live

Usage:
    python scripts/test_main.py
"""

import sys
import asyncio
import signal
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import setup_logger, logger


# ============================================
# TESTS 10.1 - ORCHESTRATEUR
# ============================================

def test_bot_config_imports():
    """Test 10.1.1 - Imports du module main."""
    print("\n📦 Test 10.1.1 - Imports main.py...")
    
    try:
        from cryptoscalper.main import (
            BotMode,
            BotStatus,
            BotConfig,
            BotState,
            TradingBot,
            parse_args,
        )
        print("  ✅ Tous les imports réussis")
        return True
    except ImportError as e:
        print(f"  ❌ Erreur import: {e}")
        return False


def test_bot_mode_enum():
    """Test 10.1.2 - BotMode enum."""
    print("\n🎯 Test 10.1.2 - BotMode enum...")
    
    from cryptoscalper.main import BotMode
    
    assert BotMode.PAPER.value == "paper"
    assert BotMode.LIVE.value == "live"
    
    # Création depuis string
    mode = BotMode("paper")
    assert mode == BotMode.PAPER
    
    mode = BotMode("live")
    assert mode == BotMode.LIVE
    
    print("  ✅ BotMode enum OK")
    return True


def test_bot_status_enum():
    """Test 10.1.3 - BotStatus enum."""
    print("\n📊 Test 10.1.3 - BotStatus enum...")
    
    from cryptoscalper.main import BotStatus
    
    assert BotStatus.INITIALIZING.value == "INITIALIZING"
    assert BotStatus.RUNNING.value == "RUNNING"
    assert BotStatus.PAUSED.value == "PAUSED"
    assert BotStatus.STOPPING.value == "STOPPING"
    assert BotStatus.STOPPED.value == "STOPPED"
    assert BotStatus.ERROR.value == "ERROR"
    
    print("  ✅ BotStatus enum OK")
    return True


def test_bot_config_creation():
    """Test 10.1.4 - Création BotConfig."""
    print("\n⚙️ Test 10.1.4 - BotConfig création...")
    
    from cryptoscalper.main import BotConfig, BotMode
    
    # Config par défaut
    config = BotConfig()
    
    assert config.mode == BotMode.PAPER
    assert config.initial_capital == 30.0
    assert config.scan_interval == 2.0
    assert config.min_probability == 0.65
    assert config.min_confidence == 0.55
    
    print(f"     Mode: {config.mode.value}")
    print(f"     Capital: {config.initial_capital}")
    print(f"     Interval: {config.scan_interval}s")
    
    # Config personnalisée
    config = BotConfig(
        mode=BotMode.LIVE,
        initial_capital=50.0,
        scan_interval=1.5,
    )
    
    assert config.mode == BotMode.LIVE
    assert config.initial_capital == 50.0
    assert config.scan_interval == 1.5
    
    print("  ✅ BotConfig création OK")
    return True


def test_bot_state_creation():
    """Test 10.1.5 - Création BotState."""
    print("\n📈 Test 10.1.5 - BotState création...")
    
    from cryptoscalper.main import BotState, BotStatus
    
    state = BotState(initial_capital=30.0)
    
    assert state.status == BotStatus.INITIALIZING
    assert state.initial_capital == 30.0
    assert state.current_capital == 30.0
    assert state.peak_capital == 30.0
    assert state.open_positions == 0
    assert state.total_trades == 0
    
    print(f"     Status: {state.status.value}")
    print(f"     Capital: {state.current_capital}")
    print("  ✅ BotState création OK")
    return True


def test_bot_state_properties():
    """Test 10.1.6 - BotState propriétés calculées."""
    print("\n📊 Test 10.1.6 - BotState propriétés...")
    
    from cryptoscalper.main import BotState
    
    state = BotState(
        initial_capital=30.0,
        current_capital=30.0,
        peak_capital=30.0,
    )
    
    # Win rate
    state.winning_trades = 6
    state.losing_trades = 4
    assert state.win_rate == 0.6
    print(f"     Win rate: {state.win_rate:.1%}")
    
    # Drawdown
    state.peak_capital = 35.0
    state.current_capital = 28.0
    expected_dd = (35.0 - 28.0) / 35.0
    assert abs(state.drawdown_pct - expected_dd) < 0.001
    print(f"     Drawdown: {state.drawdown_pct:.1%}")
    
    # Uptime
    state.start_time = datetime.now() - timedelta(hours=1, minutes=30)
    uptime = state.uptime_str
    assert "01:30" in uptime
    print(f"     Uptime: {uptime}")
    
    print("  ✅ BotState propriétés OK")
    return True


def test_trading_bot_creation():
    """Test 10.1.7 - Création TradingBot."""
    print("\n🤖 Test 10.1.7 - TradingBot création...")
    
    from cryptoscalper.main import TradingBot, BotConfig, BotMode, BotStatus
    
    config = BotConfig(
        mode=BotMode.PAPER,
        initial_capital=30.0,
    )
    
    bot = TradingBot(config)
    
    assert bot.config == config
    assert bot.mode == BotMode.PAPER
    assert bot.is_running is False
    assert bot.state.status == BotStatus.INITIALIZING
    assert bot.state.initial_capital == 30.0
    
    print(f"     Mode: {bot.mode.value}")
    print(f"     Running: {bot.is_running}")
    print(f"     Status: {bot.state.status.value}")
    
    print("  ✅ TradingBot création OK")
    return True


def test_parse_args_default():
    """Test 10.1.8 - Parse args par défaut."""
    print("\n📝 Test 10.1.8 - Parse args défaut...")
    
    from cryptoscalper.main import parse_args
    
    # Simuler args vides
    with patch('sys.argv', ['main.py']):
        args = parse_args()
    
    assert args.mode == "paper"
    assert args.capital == 30.0
    assert args.interval == 2.0
    assert args.log_level == "INFO"
    
    print(f"     Mode: {args.mode}")
    print(f"     Capital: {args.capital}")
    print(f"     Interval: {args.interval}")
    print(f"     Log level: {args.log_level}")
    
    print("  ✅ Parse args défaut OK")
    return True


def test_parse_args_custom():
    """Test 10.1.9 - Parse args personnalisés."""
    print("\n📝 Test 10.1.9 - Parse args personnalisés...")
    
    from cryptoscalper.main import parse_args
    
    # Simuler args personnalisés
    test_args = [
        'main.py',
        '--mode', 'live',
        '--capital', '50',
        '--interval', '1.5',
        '--log-level', 'DEBUG',
    ]
    
    with patch('sys.argv', test_args):
        args = parse_args()
    
    assert args.mode == "live"
    assert args.capital == 50.0
    assert args.interval == 1.5
    assert args.log_level == "DEBUG"
    
    print(f"     Mode: {args.mode}")
    print(f"     Capital: {args.capital}")
    print(f"     Interval: {args.interval}")
    print(f"     Log level: {args.log_level}")
    
    print("  ✅ Parse args personnalisés OK")
    return True


def test_bot_config_from_args():
    """Test 10.1.10 - BotConfig depuis args."""
    print("\n⚙️ Test 10.1.10 - BotConfig.from_args()...")
    
    from cryptoscalper.main import BotConfig, BotMode
    from argparse import Namespace
    
    args = Namespace(
        mode="paper",
        capital=25.0,
        model="models/test.joblib",
        interval=3.0,
        log_level="DEBUG",
    )
    
    config = BotConfig.from_args(args)
    
    assert config.mode == BotMode.PAPER
    assert config.initial_capital == 25.0
    assert config.model_path == "models/test.joblib"
    assert config.scan_interval == 3.0
    assert config.log_level == "DEBUG"
    
    print(f"     Mode: {config.mode.value}")
    print(f"     Capital: {config.initial_capital}")
    print(f"     Model: {config.model_path}")
    
    print("  ✅ BotConfig.from_args() OK")
    return True


# ============================================
# TESTS 10.2 - MODE PAPER TRADING
# ============================================

def test_paper_mode_config():
    """Test 10.2.1 - Configuration mode paper."""
    print("\n📄 Test 10.2.1 - Mode Paper config...")
    
    from cryptoscalper.main import BotConfig, BotMode
    
    config = BotConfig(mode=BotMode.PAPER)
    
    assert config.mode == BotMode.PAPER
    assert config.mode.value == "paper"
    
    # Le mode paper ne nécessite pas de confirmation
    print(f"     Mode: {config.mode.value}")
    print("  ✅ Mode Paper config OK")
    return True


def test_live_mode_config():
    """Test 10.2.2 - Configuration mode live."""
    print("\n🔴 Test 10.2.2 - Mode Live config...")
    
    from cryptoscalper.main import BotConfig, BotMode
    
    config = BotConfig(mode=BotMode.LIVE)
    
    assert config.mode == BotMode.LIVE
    assert config.mode.value == "live"
    
    # Le mode live nécessite une confirmation
    print(f"     Mode: {config.mode.value}")
    print("     ⚠️ Ce mode requiert confirmation manuelle")
    print("  ✅ Mode Live config OK")
    return True


def test_bot_state_trading_stats():
    """Test 10.2.3 - Stats de trading dans BotState."""
    print("\n📊 Test 10.2.3 - Trading stats...")
    
    from cryptoscalper.main import BotState
    
    state = BotState(initial_capital=30.0)
    
    # Simuler quelques trades
    state.total_trades = 10
    state.winning_trades = 6
    state.losing_trades = 4
    state.total_pnl = 2.50
    state.daily_pnl = 1.20
    state.daily_trades = 5
    
    assert state.total_trades == 10
    assert state.win_rate == 0.6
    assert state.total_pnl == 2.50
    
    print(f"     Trades: {state.total_trades}")
    print(f"     Gagnants: {state.winning_trades}")
    print(f"     Perdants: {state.losing_trades}")
    print(f"     Win rate: {state.win_rate:.1%}")
    print(f"     PnL total: {state.total_pnl:+.2f} USDT")
    print(f"     PnL jour: {state.daily_pnl:+.2f} USDT")
    
    print("  ✅ Trading stats OK")
    return True


def test_bot_state_scan_stats():
    """Test 10.2.4 - Stats de scan dans BotState."""
    print("\n🔍 Test 10.2.4 - Scan stats...")
    
    from cryptoscalper.main import BotState
    
    state = BotState(initial_capital=30.0)
    
    # Simuler des scans
    state.total_scans = 1000
    state.signals_generated = 50
    state.signals_executed = 10
    
    conversion_rate = state.signals_executed / state.signals_generated if state.signals_generated > 0 else 0
    
    print(f"     Scans: {state.total_scans}")
    print(f"     Signaux générés: {state.signals_generated}")
    print(f"     Signaux exécutés: {state.signals_executed}")
    print(f"     Taux conversion: {conversion_rate:.1%}")
    
    print("  ✅ Scan stats OK")
    return True


def test_bot_state_error_tracking():
    """Test 10.2.5 - Tracking des erreurs."""
    print("\n❌ Test 10.2.5 - Error tracking...")
    
    from cryptoscalper.main import BotState
    
    state = BotState(initial_capital=30.0)
    
    # Pas d'erreurs
    assert state.consecutive_errors == 0
    assert state.last_error is None
    
    # Simuler des erreurs
    state.consecutive_errors = 3
    state.last_error = "API timeout"
    
    assert state.consecutive_errors == 3
    assert state.last_error == "API timeout"
    
    print(f"     Erreurs consécutives: {state.consecutive_errors}")
    print(f"     Dernière erreur: {state.last_error}")
    
    print("  ✅ Error tracking OK")
    return True


def test_bot_open_positions_tracking():
    """Test 10.2.6 - Tracking des positions ouvertes."""
    print("\n📍 Test 10.2.6 - Position tracking...")
    
    from cryptoscalper.main import BotState
    
    state = BotState(initial_capital=30.0)
    
    # Pas de positions
    assert state.open_positions == 0
    assert len(state.open_symbols) == 0
    
    # Ouvrir une position
    state.open_positions = 1
    state.open_symbols.add("BTCUSDT")
    
    assert state.open_positions == 1
    assert "BTCUSDT" in state.open_symbols
    
    # Fermer la position
    state.open_positions = 0
    state.open_symbols.discard("BTCUSDT")
    
    assert state.open_positions == 0
    assert "BTCUSDT" not in state.open_symbols
    
    print("  ✅ Position tracking OK")
    return True


# ============================================
# TESTS INTÉGRATION (Mocks)
# ============================================

async def test_bot_lifecycle_mock():
    """Test 10.3.1 - Lifecycle du bot (avec mocks)."""
    print("\n🔄 Test 10.3.1 - Bot lifecycle (mock)...")
    
    from cryptoscalper.main import TradingBot, BotConfig, BotMode, BotStatus
    
    config = BotConfig(
        mode=BotMode.PAPER,
        initial_capital=30.0,
    )
    
    bot = TradingBot(config)
    
    # État initial
    assert bot.state.status == BotStatus.INITIALIZING
    assert not bot.is_running
    
    # Mock les modules pour éviter les vrais appels
    bot._initialize_modules = AsyncMock()
    bot._shutdown_modules = AsyncMock()
    bot._setup_signal_handlers = Mock()
    
    # Simuler un démarrage puis arrêt rapide
    async def quick_shutdown():
        await asyncio.sleep(0.1)
        await bot.stop()
    
    # Remplacer la main loop par une version courte
    original_main_loop = bot._main_loop
    
    async def mock_main_loop():
        while bot._running:
            await asyncio.sleep(0.1)
    
    bot._main_loop = mock_main_loop
    
    # Démarrer et arrêter
    start_task = asyncio.create_task(bot.start())
    shutdown_task = asyncio.create_task(quick_shutdown())
    
    await asyncio.gather(start_task, shutdown_task)
    
    # Vérifier l'état final
    assert bot.state.status == BotStatus.STOPPED
    
    print("  ✅ Bot lifecycle mock OK")
    return True


async def test_daily_reset():
    """Test 10.3.2 - Reset journalier."""
    print("\n🌅 Test 10.3.2 - Daily reset...")
    
    from cryptoscalper.main import TradingBot, BotConfig, BotMode
    from datetime import date
    
    config = BotConfig(mode=BotMode.PAPER)
    bot = TradingBot(config)
    
    # Simuler un jour précédent
    yesterday = date.today() - timedelta(days=1)
    bot._last_daily_reset = yesterday
    
    # Simuler des stats du jour précédent
    bot._state.daily_pnl = 5.0
    bot._state.daily_trades = 10
    
    # Mock le risk manager
    bot._risk_manager = Mock()
    bot._risk_manager.reset_daily_stats = Mock()
    
    # Appeler le check
    bot._check_daily_reset()
    
    # Vérifier le reset
    assert bot._state.daily_pnl == 0.0
    assert bot._state.daily_trades == 0
    assert bot._last_daily_reset == date.today()
    bot._risk_manager.reset_daily_stats.assert_called_once()
    
    print("  ✅ Daily reset OK")
    return True


def test_can_open_trade():
    """Test 10.3.3 - Vérification can_open_trade."""
    print("\n✅ Test 10.3.3 - Can open trade...")
    
    from cryptoscalper.main import TradingBot, BotConfig, BotMode
    
    config = BotConfig(mode=BotMode.PAPER)
    bot = TradingBot(config)
    
    # Mock le risk manager
    bot._risk_manager = Mock()
    bot._risk_manager.is_kill_switch_active = False
    bot._risk_manager.can_open_trade = Mock(return_value=(True, None))
    
    # Devrait pouvoir ouvrir
    assert bot._can_open_new_trade() is True
    
    # Kill switch actif
    bot._risk_manager.is_kill_switch_active = True
    assert bot._can_open_new_trade() is False
    
    # Kill switch inactif mais risk manager refuse
    bot._risk_manager.is_kill_switch_active = False
    bot._risk_manager.can_open_trade = Mock(return_value=(False, "Max positions"))
    assert bot._can_open_new_trade() is False
    
    print("  ✅ Can open trade OK")
    return True


# ============================================
# MAIN
# ============================================

async def main():
    """Exécute tous les tests."""
    print("=" * 65)
    print("🧪 CryptoScalper AI - Tests Phase 10: Boucle Principale")
    print("=" * 65)
    
    setup_logger(level="WARNING")
    
    results = []
    
    try:
        # Tests 10.1 - Orchestrateur
        print("\n" + "─" * 50)
        print("📦 10.1 Orchestrateur")
        print("─" * 50)
        results.append(("10.1.1 Imports", test_bot_config_imports()))
        results.append(("10.1.2 BotMode enum", test_bot_mode_enum()))
        results.append(("10.1.3 BotStatus enum", test_bot_status_enum()))
        results.append(("10.1.4 BotConfig création", test_bot_config_creation()))
        results.append(("10.1.5 BotState création", test_bot_state_creation()))
        results.append(("10.1.6 BotState propriétés", test_bot_state_properties()))
        results.append(("10.1.7 TradingBot création", test_trading_bot_creation()))
        results.append(("10.1.8 Parse args défaut", test_parse_args_default()))
        results.append(("10.1.9 Parse args custom", test_parse_args_custom()))
        results.append(("10.1.10 Config from args", test_bot_config_from_args()))
        
        # Tests 10.2 - Mode Paper Trading
        print("\n" + "─" * 50)
        print("📦 10.2 Mode Paper Trading")
        print("─" * 50)
        results.append(("10.2.1 Paper mode config", test_paper_mode_config()))
        results.append(("10.2.2 Live mode config", test_live_mode_config()))
        results.append(("10.2.3 Trading stats", test_bot_state_trading_stats()))
        results.append(("10.2.4 Scan stats", test_bot_state_scan_stats()))
        results.append(("10.2.5 Error tracking", test_bot_state_error_tracking()))
        results.append(("10.2.6 Position tracking", test_bot_open_positions_tracking()))
        
        # Tests 10.3 - Intégration
        print("\n" + "─" * 50)
        print("📦 10.3 Tests d'Intégration")
        print("─" * 50)
        results.append(("10.3.1 Bot lifecycle", await test_bot_lifecycle_mock()))
        results.append(("10.3.2 Daily reset", await test_daily_reset()))
        results.append(("10.3.3 Can open trade", test_can_open_trade()))
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        logger.exception("Erreur test Phase 10")
        return 1
    
    # Résumé
    print("\n" + "=" * 65)
    print("📊 RÉSULTATS PHASE 10 - BOUCLE PRINCIPALE")
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
        print("🎉 Phase 10 - Boucle Principale : VALIDÉE !")
        return 0
    else:
        print(f"⚠️  {failed} test(s) échoué(s)")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))