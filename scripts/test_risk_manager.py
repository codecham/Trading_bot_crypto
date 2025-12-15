#!/usr/bin/env python3
# scripts/test_risk_manager.py
"""
Tests d'intégration pour la Phase 7 - Risk Management.

Ce script teste:
- RiskConfig (validation, création)
- CompletedTrade (dataclass, properties)
- DailyLimits (limites journalières, reset)
- KillSwitch (drawdown, activation)
- RiskManager (autorisation trades, position sizing, tracking)

Usage:
    python scripts/test_risk_manager.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ajouter le projet au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cryptoscalper.utils.logger import setup_logger, logger


# ============================================
# TESTS RISK CONFIG
# ============================================

def test_risk_config_defaults():
    """Test 7.1.1 - RiskConfig avec valeurs par défaut."""
    print("\n⚙️ Test 7.1.1 - RiskConfig défaut...")
    
    from cryptoscalper.trading.risk_manager import RiskConfig
    
    config = RiskConfig()
    
    assert config.initial_capital == 30.0
    assert config.max_position_pct == 0.20
    assert config.min_position_usdt == 10.0
    assert config.max_daily_loss_pct == 0.10
    assert config.max_drawdown_pct == 0.25
    assert config.max_open_positions == 1
    assert config.max_trades_per_day == 100
    
    print(f"  ✅ Capital initial: {config.initial_capital} USDT")
    print(f"  ✅ Max position: {config.max_position_pct:.0%}")
    print(f"  ✅ Max drawdown: {config.max_drawdown_pct:.0%}")
    
    return True


def test_risk_config_custom():
    """Test 7.1.2 - RiskConfig avec valeurs personnalisées."""
    print("\n⚙️ Test 7.1.2 - RiskConfig personnalisé...")
    
    from cryptoscalper.trading.risk_manager import RiskConfig
    
    config = RiskConfig(
        initial_capital=50.0,
        max_position_pct=0.30,
        max_daily_loss_pct=0.15,
        max_drawdown_pct=0.30,
        max_trades_per_day=50
    )
    
    assert config.initial_capital == 50.0
    assert config.max_position_pct == 0.30
    assert config.max_daily_loss_pct == 0.15
    
    print(f"  ✅ Capital: {config.initial_capital} USDT")
    print(f"  ✅ Max position: {config.max_position_pct:.0%}")
    
    return True


def test_risk_config_validation():
    """Test 7.1.3 - Validation de RiskConfig."""
    print("\n⚙️ Test 7.1.3 - Validation RiskConfig...")
    
    from cryptoscalper.trading.risk_manager import RiskConfig
    
    # Config valide
    valid_config = RiskConfig(initial_capital=20.0)
    valid_config.validate()  # Ne doit pas lever d'exception
    print("  ✅ Config valide acceptée")
    
    # Capital négatif
    try:
        bad_config = RiskConfig(initial_capital=-10.0)
        bad_config.validate()
        print("  ❌ Devrait rejeter capital négatif")
        return False
    except ValueError:
        print("  ✅ Capital négatif rejeté")
    
    # Pourcentage invalide
    try:
        bad_config = RiskConfig(max_position_pct=1.5)
        bad_config.validate()
        print("  ❌ Devrait rejeter pourcentage > 1")
        return False
    except ValueError:
        print("  ✅ Pourcentage > 1 rejeté")
    
    return True


def test_risk_config_to_dict():
    """Test 7.1.4 - Conversion RiskConfig en dict."""
    print("\n⚙️ Test 7.1.4 - RiskConfig to_dict()...")
    
    from cryptoscalper.trading.risk_manager import RiskConfig
    
    config = RiskConfig(initial_capital=25.0)
    d = config.to_dict()
    
    assert "initial_capital" in d
    assert "max_position_pct" in d
    assert "max_drawdown_pct" in d
    assert d["initial_capital"] == 25.0
    
    print(f"  ✅ Dict contient {len(d)} clés")
    print(f"  ✅ initial_capital: {d['initial_capital']}")
    
    return True


# ============================================
# TESTS COMPLETED TRADE
# ============================================

def test_completed_trade_creation():
    """Test 7.1.5 - Création CompletedTrade."""
    print("\n📝 Test 7.1.5 - CompletedTrade création...")
    
    from cryptoscalper.trading.risk_manager import CompletedTrade, TradeOutcome
    
    now = datetime.utcnow()
    trade = CompletedTrade(
        symbol="BTCUSDT",
        side="BUY",
        entry_price=50000.0,
        exit_price=50150.0,
        quantity=0.001,
        entry_time=now - timedelta(minutes=2),
        exit_time=now,
        pnl_usdt=0.15,
        pnl_percent=0.003,
        fees_usdt=0.02,
        exit_reason="TP"
    )
    
    assert trade.symbol == "BTCUSDT"
    assert trade.outcome == TradeOutcome.WIN
    assert trade.duration_seconds == 120  # 2 minutes
    assert trade.net_pnl_usdt == 0.13  # 0.15 - 0.02
    
    print(f"  ✅ Trade: {trade.symbol} {trade.side}")
    print(f"  ✅ Outcome: {trade.outcome.value}")
    print(f"  ✅ Durée: {trade.duration_seconds}s")
    print(f"  ✅ PnL net: {trade.net_pnl_usdt:.4f} USDT")
    
    return True


def test_completed_trade_outcomes():
    """Test 7.1.6 - Outcomes de CompletedTrade."""
    print("\n📝 Test 7.1.6 - CompletedTrade outcomes...")
    
    from cryptoscalper.trading.risk_manager import CompletedTrade, TradeOutcome
    
    now = datetime.utcnow()
    base_args = {
        "symbol": "ETHUSDT",
        "side": "BUY",
        "entry_price": 3000.0,
        "exit_price": 3000.0,
        "quantity": 0.01,
        "entry_time": now - timedelta(minutes=1),
        "exit_time": now,
        "pnl_percent": 0.0,
    }
    
    # Trade gagnant
    win_trade = CompletedTrade(**{**base_args, "pnl_usdt": 0.50})
    assert win_trade.outcome == TradeOutcome.WIN
    print("  ✅ Win: pnl > 0")
    
    # Trade perdant
    loss_trade = CompletedTrade(**{**base_args, "pnl_usdt": -0.30})
    assert loss_trade.outcome == TradeOutcome.LOSS
    print("  ✅ Loss: pnl < 0")
    
    # Breakeven
    even_trade = CompletedTrade(**{**base_args, "pnl_usdt": 0.0})
    assert even_trade.outcome == TradeOutcome.BREAKEVEN
    print("  ✅ Breakeven: pnl = 0")
    
    return True


# ============================================
# TESTS DAILY LIMITS
# ============================================

def test_daily_limits_creation():
    """Test 7.2.1 - Création DailyLimits."""
    print("\n📊 Test 7.2.1 - DailyLimits création...")
    
    from cryptoscalper.trading.risk_manager import DailyLimits, RiskConfig
    
    config = RiskConfig(initial_capital=30.0, max_daily_loss_pct=0.10)
    limits = DailyLimits(config)
    
    assert limits.daily_pnl == 0.0
    assert limits.daily_trades_count == 0
    assert limits.remaining_daily_loss == 3.0  # 30 * 0.10
    
    print(f"  ✅ PnL initial: {limits.daily_pnl}")
    print(f"  ✅ Trades: {limits.daily_trades_count}")
    print(f"  ✅ Perte restante: {limits.remaining_daily_loss} USDT")
    
    return True


def test_daily_limits_can_trade():
    """Test 7.2.2 - Vérification DailyLimits.can_trade()."""
    print("\n📊 Test 7.2.2 - DailyLimits can_trade()...")
    
    from cryptoscalper.trading.risk_manager import (
        DailyLimits, RiskConfig, RejectionReason
    )
    
    config = RiskConfig(
        initial_capital=30.0,
        max_daily_loss_pct=0.10,
        max_trades_per_day=5
    )
    limits = DailyLimits(config)
    
    # Au début, on peut trader
    can_trade, reason = limits.can_trade()
    assert can_trade is True
    assert reason is None
    print("  ✅ Début: trading autorisé")
    
    return True


def test_daily_limits_loss_exceeded():
    """Test 7.2.3 - Perte journalière maximum."""
    print("\n📊 Test 7.2.3 - Perte journalière max...")
    
    from cryptoscalper.trading.risk_manager import (
        DailyLimits, RiskConfig, RejectionReason
    )
    
    config = RiskConfig(initial_capital=30.0, max_daily_loss_pct=0.10)
    limits = DailyLimits(config)
    
    # Simuler une grosse perte
    limits.register_loss(3.5)  # Plus que 10% de 30 = 3 USDT
    
    can_trade, reason = limits.can_trade()
    assert can_trade is False
    assert reason == RejectionReason.DAILY_LOSS_EXCEEDED
    
    print(f"  ✅ Perte journalière: {limits.daily_pnl:.2f} USDT")
    print(f"  ✅ Trading bloqué: {reason.value}")
    
    return True


def test_daily_limits_max_trades():
    """Test 7.2.4 - Nombre maximum de trades."""
    print("\n📊 Test 7.2.4 - Max trades par jour...")
    
    from cryptoscalper.trading.risk_manager import (
        DailyLimits, RiskConfig, CompletedTrade, RejectionReason
    )
    
    config = RiskConfig(
        initial_capital=30.0,
        max_trades_per_day=3  # Seulement 3 trades
    )
    limits = DailyLimits(config)
    
    # Créer un trade mock
    now = datetime.utcnow()
    mock_trade = CompletedTrade(
        symbol="BTCUSDT",
        side="BUY",
        entry_price=50000.0,
        exit_price=50050.0,
        quantity=0.001,
        entry_time=now - timedelta(minutes=1),
        exit_time=now,
        pnl_usdt=0.05,
        pnl_percent=0.001,
    )
    
    # Enregistrer 3 trades
    for i in range(3):
        limits.register_trade(mock_trade)
    
    assert limits.daily_trades_count == 3
    
    # Le 4ème devrait être bloqué
    can_trade, reason = limits.can_trade()
    assert can_trade is False
    assert reason == RejectionReason.MAX_TRADES_PER_DAY
    
    print(f"  ✅ Trades enregistrés: {limits.daily_trades_count}")
    print(f"  ✅ Trading bloqué: {reason.value}")
    
    return True


def test_daily_limits_reset():
    """Test 7.2.5 - Reset des limites journalières."""
    print("\n📊 Test 7.2.5 - Reset DailyLimits...")
    
    from cryptoscalper.trading.risk_manager import DailyLimits, RiskConfig
    
    config = RiskConfig(initial_capital=30.0)
    limits = DailyLimits(config)
    
    # Accumuler des pertes
    limits.register_loss(2.0)
    assert limits.daily_pnl == -2.0
    
    # Reset
    limits.reset()
    
    assert limits.daily_pnl == 0.0
    assert limits.daily_trades_count == 0
    
    print("  ✅ Reset effectué")
    print(f"  ✅ PnL après reset: {limits.daily_pnl}")
    
    return True


# ============================================
# TESTS KILL SWITCH
# ============================================

def test_kill_switch_creation():
    """Test 7.3.1 - Création KillSwitch."""
    print("\n🚨 Test 7.3.1 - KillSwitch création...")
    
    from cryptoscalper.trading.risk_manager import KillSwitch
    
    ks = KillSwitch(initial_capital=30.0, max_drawdown_pct=0.25)
    
    assert ks.is_active is False
    assert ks.peak_capital == 30.0
    assert ks.current_capital == 30.0
    assert ks.current_drawdown == 0.0
    
    print(f"  ✅ Kill switch créé (inactif)")
    print(f"  ✅ Peak capital: {ks.peak_capital}")
    print(f"  ✅ Drawdown: {ks.current_drawdown:.1%}")
    
    return True


def test_kill_switch_update_capital():
    """Test 7.3.2 - Update du capital."""
    print("\n🚨 Test 7.3.2 - KillSwitch update capital...")
    
    from cryptoscalper.trading.risk_manager import KillSwitch
    
    ks = KillSwitch(initial_capital=30.0, max_drawdown_pct=0.25)
    
    # Capital augmente
    ks.update(32.0)
    assert ks.peak_capital == 32.0
    assert ks.is_active is False
    print(f"  ✅ Peak mis à jour: {ks.peak_capital}")
    
    # Capital diminue mais pas de kill switch
    ks.update(28.0)
    assert ks.peak_capital == 32.0  # Peak ne change pas
    assert ks.current_drawdown == (32.0 - 28.0) / 32.0
    assert ks.is_active is False
    print(f"  ✅ Drawdown: {ks.current_drawdown:.1%} (pas d'activation)")
    
    return True


def test_kill_switch_activation():
    """Test 7.3.3 - Activation du kill switch."""
    print("\n🚨 Test 7.3.3 - Activation KillSwitch...")
    
    from cryptoscalper.trading.risk_manager import KillSwitch
    
    ks = KillSwitch(initial_capital=30.0, max_drawdown_pct=0.25)
    
    # Drawdown progressif
    ks.update(25.0)  # -16.7%
    assert ks.is_active is False
    print(f"  ✅ -16.7%: pas d'activation")
    
    # Franchir le seuil (25% de 30 = 7.5 USDT de perte = 22.5 USDT)
    activated = ks.update(22.0)  # -26.7% > 25%
    assert activated is True
    assert ks.is_active is True
    print(f"  ✅ -26.7%: ACTIVATION")
    
    # Vérifier le check
    can_trade, msg = ks.check()
    assert can_trade is False
    print(f"  ✅ Trading bloqué: {msg}")
    
    return True


def test_kill_switch_reset():
    """Test 7.3.4 - Reset du kill switch."""
    print("\n🚨 Test 7.3.4 - Reset KillSwitch...")
    
    from cryptoscalper.trading.risk_manager import KillSwitch
    
    ks = KillSwitch(initial_capital=30.0, max_drawdown_pct=0.25)
    
    # Activer
    ks.update(20.0)
    assert ks.is_active is True
    
    # Reset avec nouveau capital
    ks.reset(new_initial_capital=25.0)
    assert ks.is_active is False
    assert ks.current_capital == 25.0
    assert ks.peak_capital == 25.0
    
    print("  ✅ Kill switch reset")
    print(f"  ✅ Nouveau capital: {ks.current_capital}")
    
    return True


# ============================================
# TESTS RISK MANAGER
# ============================================

def test_risk_manager_creation():
    """Test 7.1.7 - Création RiskManager."""
    print("\n🛡️ Test 7.1.7 - RiskManager création...")
    
    from cryptoscalper.trading.risk_manager import RiskManager, RiskConfig
    
    config = RiskConfig(initial_capital=30.0)
    rm = RiskManager(config)
    
    assert rm.current_capital == 30.0
    assert rm.open_positions_count == 0
    assert rm.consecutive_losses == 0
    assert rm.is_kill_switch_active is False
    
    print(f"  ✅ RiskManager créé")
    print(f"  ✅ Capital: {rm.current_capital}")
    print(f"  ✅ Kill switch: {'actif' if rm.is_kill_switch_active else 'inactif'}")
    
    return True


def test_risk_manager_can_open_trade():
    """Test 7.1.8 - Autorisation d'ouvrir un trade."""
    print("\n🛡️ Test 7.1.8 - can_open_trade()...")
    
    from cryptoscalper.trading.risk_manager import RiskManager, RiskConfig
    
    config = RiskConfig(initial_capital=30.0)
    rm = RiskManager(config)
    
    # Au début, on peut trader
    can_trade, reason = rm.can_open_trade()
    assert can_trade is True
    assert reason is None
    
    print("  ✅ Trading autorisé au début")
    
    return True


def test_risk_manager_position_sizing_pct():
    """Test 7.1.9 - Position sizing par pourcentage."""
    print("\n🛡️ Test 7.1.9 - Position sizing (%)...")
    
    from cryptoscalper.trading.risk_manager import RiskManager, RiskConfig
    
    config = RiskConfig(
        initial_capital=30.0,
        max_position_pct=0.20  # 20%
    )
    rm = RiskManager(config)
    
    size = rm.calculate_position_size(current_capital=30.0)
    
    # Maximum: 30 * 0.20 = 6 USDT
    assert size <= 6.0
    assert size >= config.min_position_usdt or size == 0
    
    print(f"  ✅ Position size: {size:.2f} USDT (max: 6.0)")
    
    return True


def test_risk_manager_position_sizing_risk():
    """Test 7.1.10 - Position sizing par risque."""
    print("\n🛡️ Test 7.1.10 - Position sizing (risque)...")
    
    from cryptoscalper.trading.risk_manager import RiskManager, RiskConfig
    
    config = RiskConfig(
        initial_capital=100.0,
        max_position_pct=0.50,  # 50%
        max_loss_per_trade_pct=0.02,  # 2%
        min_position_usdt=5.0
    )
    rm = RiskManager(config)
    
    # Avec SL de 1%
    # Risk = 100 * 0.02 = 2 USDT
    # Size by risk = 2 / 0.01 = 200 USDT (mais limité par max_position_pct)
    # Size by pct = 100 * 0.50 = 50 USDT
    # Final = min(50, 200) = 50 USDT
    size = rm.calculate_position_size(stop_loss_pct=0.01)
    
    assert size <= 50.0
    print(f"  ✅ Position size: {size:.2f} USDT")
    
    # Avec SL de 5%
    # Size by risk = 2 / 0.05 = 40 USDT
    # Final = min(50, 40) = 40 USDT
    size2 = rm.calculate_position_size(stop_loss_pct=0.05)
    
    assert size2 <= 40.0
    print(f"  ✅ Position avec SL 5%: {size2:.2f} USDT")
    
    return True


def test_risk_manager_position_too_small():
    """Test 7.1.11 - Position trop petite."""
    print("\n🛡️ Test 7.1.11 - Position trop petite...")
    
    from cryptoscalper.trading.risk_manager import RiskManager, RiskConfig
    
    config = RiskConfig(
        initial_capital=5.0,  # Petit capital
        max_position_pct=0.20,  # 20% = 1 USDT
        min_position_usdt=10.0  # Minimum 10 USDT
    )
    rm = RiskManager(config)
    
    size = rm.calculate_position_size()
    
    # 5 * 0.20 = 1 USDT < 10 USDT minimum → 0
    assert size == 0.0
    
    print(f"  ✅ Position rejetée (trop petite): {size}")
    
    return True


def test_risk_manager_register_trade():
    """Test 7.1.12 - Enregistrement d'un trade."""
    print("\n🛡️ Test 7.1.12 - register_trade_result()...")
    
    from cryptoscalper.trading.risk_manager import (
        RiskManager, RiskConfig, CompletedTrade
    )
    
    config = RiskConfig(initial_capital=30.0)
    rm = RiskManager(config)
    
    # Trade gagnant
    now = datetime.utcnow()
    win_trade = CompletedTrade(
        symbol="BTCUSDT",
        side="BUY",
        entry_price=50000.0,
        exit_price=50150.0,
        quantity=0.001,
        entry_time=now - timedelta(minutes=2),
        exit_time=now,
        pnl_usdt=0.15,
        pnl_percent=0.003,
        fees_usdt=0.02,
        exit_reason="TP"
    )
    
    initial_capital = rm.current_capital
    rm.register_trade_result(win_trade)
    
    # Capital devrait augmenter de (0.15 - 0.02) = 0.13
    assert rm.current_capital == initial_capital + 0.13
    assert rm.daily_pnl == 0.13
    assert rm.consecutive_losses == 0
    
    print(f"  ✅ Trade WIN enregistré")
    print(f"  ✅ Capital: {rm.current_capital:.2f} USDT")
    print(f"  ✅ PnL jour: {rm.daily_pnl:.4f} USDT")
    
    return True


def test_risk_manager_consecutive_losses():
    """Test 7.1.13 - Tracking des pertes consécutives."""
    print("\n🛡️ Test 7.1.13 - Pertes consécutives...")
    
    from cryptoscalper.trading.risk_manager import (
        RiskManager, RiskConfig, CompletedTrade
    )
    
    config = RiskConfig(
        initial_capital=100.0,
        max_consecutive_losses=3
    )
    rm = RiskManager(config)
    
    now = datetime.utcnow()
    
    def make_loss_trade():
        return CompletedTrade(
            symbol="BTCUSDT",
            side="BUY",
            entry_price=50000.0,
            exit_price=49900.0,
            quantity=0.001,
            entry_time=now - timedelta(minutes=1),
            exit_time=now,
            pnl_usdt=-0.10,
            pnl_percent=-0.002,
            exit_reason="SL"
        )
    
    # 3 pertes consécutives
    for i in range(3):
        rm.register_trade_result(make_loss_trade())
    
    assert rm.consecutive_losses == 3
    
    # Vérifier que le 4ème trade est bloqué
    can_trade, reason = rm.can_open_trade()
    assert can_trade is False
    assert "consécutives" in reason.lower()
    
    print(f"  ✅ Pertes consécutives: {rm.consecutive_losses}")
    print(f"  ✅ Trading bloqué: {reason}")
    
    return True


def test_risk_manager_kill_switch():
    """Test 7.3.5 - Kill switch via RiskManager."""
    print("\n🛡️ Test 7.3.5 - Kill switch integration...")
    
    from cryptoscalper.trading.risk_manager import (
        RiskManager, RiskConfig, CompletedTrade
    )
    from cryptoscalper.utils.exceptions import KillSwitchActivatedError
    
    config = RiskConfig(
        initial_capital=30.0,
        max_drawdown_pct=0.25  # 25%
    )
    rm = RiskManager(config)
    
    now = datetime.utcnow()
    
    # Grosse perte qui déclenche le kill switch
    big_loss = CompletedTrade(
        symbol="BTCUSDT",
        side="BUY",
        entry_price=50000.0,
        exit_price=46000.0,
        quantity=0.001,
        entry_time=now - timedelta(minutes=5),
        exit_time=now,
        pnl_usdt=-8.0,  # -8 USDT sur 30 = -26.7%
        pnl_percent=-0.08,
        exit_reason="SL"
    )
    
    try:
        rm.register_trade_result(big_loss)
        print("  ❌ Devrait lever KillSwitchActivatedError")
        return False
    except KillSwitchActivatedError as e:
        print(f"  ✅ Kill switch activé: {e}")
    
    # Vérifier que le trading est bloqué
    assert rm.is_kill_switch_active is True
    can_trade, reason = rm.can_open_trade()
    assert can_trade is False
    
    print(f"  ✅ Trading bloqué: {reason}")
    
    return True


def test_risk_manager_open_positions():
    """Test 7.1.14 - Gestion des positions ouvertes."""
    print("\n🛡️ Test 7.1.14 - Positions ouvertes...")
    
    from cryptoscalper.trading.risk_manager import RiskManager, RiskConfig
    
    config = RiskConfig(
        initial_capital=30.0,
        max_open_positions=1
    )
    rm = RiskManager(config)
    
    # Ouvrir une position
    rm.register_position_opened("BTCUSDT")
    assert rm.open_positions_count == 1
    print(f"  ✅ Position ouverte: BTCUSDT")
    
    # Essayer d'ouvrir une 2ème
    can_trade, reason = rm.can_open_trade()
    assert can_trade is False
    assert "positions" in reason.lower()
    print(f"  ✅ 2ème position bloquée: {reason}")
    
    # Fermer la position
    rm.register_position_closed("BTCUSDT")
    assert rm.open_positions_count == 0
    
    # Maintenant on peut trader
    can_trade, _ = rm.can_open_trade()
    assert can_trade is True
    print("  ✅ Trading autorisé après fermeture")
    
    return True


def test_risk_manager_statistics():
    """Test 7.1.15 - Statistiques du RiskManager."""
    print("\n🛡️ Test 7.1.15 - Statistiques...")
    
    from cryptoscalper.trading.risk_manager import RiskManager, RiskConfig
    
    config = RiskConfig(initial_capital=30.0)
    rm = RiskManager(config)
    
    stats = rm.get_statistics()
    
    assert "config" in stats
    assert "current_capital" in stats
    assert "daily_limits" in stats
    assert "kill_switch" in stats
    
    print(f"  ✅ Statistiques disponibles: {list(stats.keys())}")
    
    # Affichage
    summary = str(rm)
    assert "RiskManager" in summary
    assert "Capital" in summary
    
    print(f"  ✅ Affichage:\n{summary}")
    
    return True


def test_risk_manager_full_reset():
    """Test 7.1.16 - Reset complet."""
    print("\n🛡️ Test 7.1.16 - Full reset...")
    
    from cryptoscalper.trading.risk_manager import (
        RiskManager, RiskConfig, CompletedTrade
    )
    
    config = RiskConfig(initial_capital=30.0)
    rm = RiskManager(config)
    
    # Simuler de l'activité
    rm.update_capital(25.0)
    rm.register_position_opened("BTCUSDT")
    
    now = datetime.utcnow()
    trade = CompletedTrade(
        symbol="ETHUSDT",
        side="BUY",
        entry_price=3000.0,
        exit_price=2990.0,
        quantity=0.01,
        entry_time=now - timedelta(minutes=1),
        exit_time=now,
        pnl_usdt=-0.10,
        pnl_percent=-0.003,
    )
    rm.register_trade_result(trade)
    
    # Reset complet
    rm.full_reset(new_capital=50.0)
    
    assert rm.current_capital == 50.0
    assert rm.open_positions_count == 0
    assert rm.consecutive_losses == 0
    assert rm.daily_pnl == 0.0
    
    print(f"  ✅ Reset complet effectué")
    print(f"  ✅ Nouveau capital: {rm.current_capital} USDT")
    
    return True


# ============================================
# TESTS FONCTIONS UTILITAIRES
# ============================================

def test_create_risk_manager_helper():
    """Test 7.1.17 - Fonction create_risk_manager()."""
    print("\n🔧 Test 7.1.17 - create_risk_manager()...")
    
    from cryptoscalper.trading.risk_manager import create_risk_manager
    
    rm = create_risk_manager(
        initial_capital=50.0,
        max_position_pct=0.30,
        max_drawdown_pct=0.20
    )
    
    assert rm.current_capital == 50.0
    assert rm.config.max_position_pct == 0.30
    assert rm.config.max_drawdown_pct == 0.20
    
    print(f"  ✅ RiskManager créé via helper")
    
    return True


def test_calculate_risk_reward_ratio():
    """Test 7.1.18 - Calcul ratio risque/récompense."""
    print("\n🔧 Test 7.1.18 - calculate_risk_reward_ratio()...")
    
    from cryptoscalper.trading.risk_manager import calculate_risk_reward_ratio
    
    # Entrée: 100, SL: 99, TP: 102
    # Risk: 1, Reward: 2 → R:R = 2.0
    rr = calculate_risk_reward_ratio(
        entry_price=100.0,
        stop_loss_price=99.0,
        take_profit_price=102.0
    )
    
    assert rr == 2.0
    print(f"  ✅ R:R ratio: {rr}")
    
    # Cas où risque > récompense
    rr2 = calculate_risk_reward_ratio(
        entry_price=100.0,
        stop_loss_price=98.0,
        take_profit_price=101.0
    )
    
    assert rr2 == 0.5  # Reward(1) / Risk(2)
    print(f"  ✅ R:R défavorable: {rr2}")
    
    return True


# ============================================
# MAIN
# ============================================

async def main():
    """Exécute tous les tests de la Phase 7."""
    print("=" * 65)
    print("🧪 CryptoScalper AI - Tests Phase 7: Risk Management")
    print("=" * 65)
    
    setup_logger(level="WARNING")
    
    results = []
    
    try:
        # Tests RiskConfig
        print("\n" + "─" * 50)
        print("📦 7.1.1-4 RiskConfig")
        print("─" * 50)
        results.append(("7.1.1 Config défaut", test_risk_config_defaults()))
        results.append(("7.1.2 Config custom", test_risk_config_custom()))
        results.append(("7.1.3 Validation", test_risk_config_validation()))
        results.append(("7.1.4 to_dict()", test_risk_config_to_dict()))
        
        # Tests CompletedTrade
        print("\n" + "─" * 50)
        print("📦 7.1.5-6 CompletedTrade")
        print("─" * 50)
        results.append(("7.1.5 Création", test_completed_trade_creation()))
        results.append(("7.1.6 Outcomes", test_completed_trade_outcomes()))
        
        # Tests DailyLimits
        print("\n" + "─" * 50)
        print("📦 7.2.1-5 DailyLimits")
        print("─" * 50)
        results.append(("7.2.1 Création", test_daily_limits_creation()))
        results.append(("7.2.2 can_trade()", test_daily_limits_can_trade()))
        results.append(("7.2.3 Loss exceeded", test_daily_limits_loss_exceeded()))
        results.append(("7.2.4 Max trades", test_daily_limits_max_trades()))
        results.append(("7.2.5 Reset", test_daily_limits_reset()))
        
        # Tests KillSwitch
        print("\n" + "─" * 50)
        print("📦 7.3.1-4 KillSwitch")
        print("─" * 50)
        results.append(("7.3.1 Création", test_kill_switch_creation()))
        results.append(("7.3.2 Update capital", test_kill_switch_update_capital()))
        results.append(("7.3.3 Activation", test_kill_switch_activation()))
        results.append(("7.3.4 Reset", test_kill_switch_reset()))
        
        # Tests RiskManager
        print("\n" + "─" * 50)
        print("📦 7.1.7-16 RiskManager")
        print("─" * 50)
        results.append(("7.1.7 Création", test_risk_manager_creation()))
        results.append(("7.1.8 can_open_trade()", test_risk_manager_can_open_trade()))
        results.append(("7.1.9 Position sizing %", test_risk_manager_position_sizing_pct()))
        results.append(("7.1.10 Position sizing risque", test_risk_manager_position_sizing_risk()))
        results.append(("7.1.11 Position trop petite", test_risk_manager_position_too_small()))
        results.append(("7.1.12 Register trade", test_risk_manager_register_trade()))
        results.append(("7.1.13 Pertes consécutives", test_risk_manager_consecutive_losses()))
        results.append(("7.3.5 Kill switch", test_risk_manager_kill_switch()))
        results.append(("7.1.14 Positions ouvertes", test_risk_manager_open_positions()))
        results.append(("7.1.15 Statistiques", test_risk_manager_statistics()))
        results.append(("7.1.16 Full reset", test_risk_manager_full_reset()))
        
        # Tests fonctions utilitaires
        print("\n" + "─" * 50)
        print("📦 7.1.17-18 Fonctions utilitaires")
        print("─" * 50)
        results.append(("7.1.17 create_risk_manager()", test_create_risk_manager_helper()))
        results.append(("7.1.18 R:R ratio", test_calculate_risk_reward_ratio()))
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        logger.exception("Erreur test Phase 7")
        return 1
    
    # Résumé
    print("\n" + "=" * 65)
    print("📊 RÉSULTATS PHASE 7 - RISK MANAGEMENT")
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
        print("🎉 Phase 7 - Risk Management : VALIDÉE !")
        return 0
    else:
        print(f"⚠️  {failed} test(s) échoué(s)")
        return 1


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))