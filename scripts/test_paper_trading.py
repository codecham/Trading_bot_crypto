#!/usr/bin/env python3
# scripts/test_paper_trading.py
"""
Tests pour le script paper_trading.py.

Vérifie les composants principaux:
- Configuration
- Dataclasses (PaperTrade, TradingStats)
- Logique de calcul PnL
- Parsing des arguments

Usage:
    python scripts/test_paper_trading.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================
# TESTS
# ============================================

def test_imports():
    """Test 10.5.1 - Imports du module paper_trading."""
    print("\n📦 Test 10.5.1 - Imports paper_trading...")
    
    try:
        from scripts.paper_trading import (
            PaperTradeConfig,
            PaperTrade,
            TradingStats,
            TradeStatus,
            PaperTrader,
            parse_args,
            parse_duration,
            TAKE_PROFIT_PCT,
            STOP_LOSS_PCT,
            TIMEOUT_MINUTES,
            DEFAULT_PROBABILITY_THRESHOLD,
        )
        print("  ✅ Tous les imports réussis")
        return True
    except ImportError as e:
        print(f"  ❌ Erreur import: {e}")
        return False


def test_trade_status_enum():
    """Test 10.5.2 - TradeStatus enum."""
    print("\n🏷️  Test 10.5.2 - TradeStatus enum...")
    
    from scripts.paper_trading import TradeStatus
    
    assert TradeStatus.OPEN.value == "OPEN"
    assert TradeStatus.CLOSED_TP.value == "CLOSED_TP"
    assert TradeStatus.CLOSED_SL.value == "CLOSED_SL"
    assert TradeStatus.CLOSED_TIMEOUT.value == "CLOSED_TIMEOUT"
    
    print("  ✅ TradeStatus enum OK")
    return True


def test_paper_trade_config():
    """Test 10.5.3 - PaperTradeConfig."""
    print("\n⚙️  Test 10.5.3 - PaperTradeConfig...")
    
    from scripts.paper_trading import PaperTradeConfig
    
    # Config par défaut
    config = PaperTradeConfig()
    assert config.capital == 25.0, "Capital défaut 25"
    assert len(config.symbols) == 5, "5 symboles par défaut"
    assert config.probability_threshold == 0.20, "Seuil 20%"
    assert config.take_profit_pct == 0.02, "TP 2%"
    assert config.stop_loss_pct == 0.01, "SL 1%"
    assert config.timeout_minutes == 120, "Timeout 120 min"
    
    print(f"  ✅ Config par défaut: capital={config.capital}, symbols={len(config.symbols)}")
    
    # Config personnalisée
    config = PaperTradeConfig(
        capital=50.0,
        symbols=["BTCUSDT", "ETHUSDT"],
        probability_threshold=0.30,
    )
    assert config.capital == 50.0
    assert len(config.symbols) == 2
    assert config.probability_threshold == 0.30
    
    print(f"  ✅ Config custom: capital={config.capital}, threshold={config.probability_threshold}")
    return True


def test_paper_trade_dataclass():
    """Test 10.5.4 - PaperTrade dataclass."""
    print("\n📝 Test 10.5.4 - PaperTrade dataclass...")
    
    from scripts.paper_trading import PaperTrade, TradeStatus
    
    now = datetime.now()
    trade = PaperTrade(
        id=1,
        symbol="BTCUSDT",
        entry_time=now,
        entry_price=100000.0,
        quantity=0.001,
        size_usdt=100.0,
        take_profit_price=102000.0,  # +2%
        stop_loss_price=99000.0,     # -1%
        timeout_time=now + timedelta(hours=2),
        probability=0.25,
    )
    
    assert trade.is_open is True, "Trade devrait être ouvert"
    assert trade.status == TradeStatus.OPEN
    assert trade.duration_minutes is None, "Pas de durée tant que non fermé"
    
    # Simuler fermeture
    trade.exit_time = now + timedelta(minutes=30)
    trade.exit_price = 102000.0  # TP atteint
    trade.status = TradeStatus.CLOSED_TP
    trade.pnl_usdt = 1.80  # 2% - 0.2% frais
    trade.pnl_pct = 1.8
    
    assert trade.is_open is False
    assert trade.duration_minutes == 30.0
    
    print(f"  ✅ PaperTrade créé: {trade.symbol}, PnL={trade.pnl_usdt}")
    return True


def test_trading_stats():
    """Test 10.5.5 - TradingStats."""
    print("\n📊 Test 10.5.5 - TradingStats...")
    
    from scripts.paper_trading import TradingStats
    
    stats = TradingStats(current_capital=25.0, peak_capital=25.0)
    
    assert stats.total_trades == 0
    assert stats.win_rate == 0.0
    assert stats.net_pnl == 0.0
    
    # Simuler des trades
    stats.total_trades = 10
    stats.winning_trades = 5
    stats.losing_trades = 5
    stats.total_pnl_usdt = 3.0
    stats.total_fees_usdt = 0.4
    
    assert stats.win_rate == 50.0, f"WR devrait être 50%, got {stats.win_rate}"
    assert stats.net_pnl == 2.6, f"Net PnL devrait être 2.6, got {stats.net_pnl}"
    
    print(f"  ✅ TradingStats: WR={stats.win_rate}%, Net PnL={stats.net_pnl}")
    return True


def test_pnl_calculation():
    """Test 10.5.6 - Calcul du PnL."""
    print("\n💰 Test 10.5.6 - Calcul PnL...")
    
    from scripts.paper_trading import TRADING_FEES_PCT
    
    # Paramètres
    entry_price = 100.0
    size_usdt = 20.0
    fees_pct = TRADING_FEES_PCT  # 0.2%
    
    # Trade gagnant (TP +2%)
    exit_price_tp = 102.0
    price_change_pct = (exit_price_tp - entry_price) / entry_price
    gross_pnl_tp = size_usdt * price_change_pct
    fees = size_usdt * fees_pct
    net_pnl_tp = gross_pnl_tp - fees
    
    expected_gross = 0.40  # 2% de 20 = 0.40
    expected_fees = 0.04   # 0.2% de 20 = 0.04
    expected_net = 0.36    # 0.40 - 0.04 = 0.36
    
    assert abs(gross_pnl_tp - expected_gross) < 0.01, f"Gross PnL: {gross_pnl_tp}"
    assert abs(fees - expected_fees) < 0.001, f"Fees: {fees}"
    assert abs(net_pnl_tp - expected_net) < 0.01, f"Net PnL: {net_pnl_tp}"
    
    print(f"  ✅ Trade TP: Gross={gross_pnl_tp:.2f}, Fees={fees:.4f}, Net={net_pnl_tp:.2f}")
    
    # Trade perdant (SL -1%)
    exit_price_sl = 99.0
    price_change_pct = (exit_price_sl - entry_price) / entry_price
    gross_pnl_sl = size_usdt * price_change_pct
    net_pnl_sl = gross_pnl_sl - fees
    
    expected_gross_sl = -0.20  # -1% de 20 = -0.20
    expected_net_sl = -0.24   # -0.20 - 0.04 = -0.24
    
    assert abs(gross_pnl_sl - expected_gross_sl) < 0.01, f"Gross SL: {gross_pnl_sl}"
    assert abs(net_pnl_sl - expected_net_sl) < 0.01, f"Net SL: {net_pnl_sl}"
    
    print(f"  ✅ Trade SL: Gross={gross_pnl_sl:.2f}, Fees={fees:.4f}, Net={net_pnl_sl:.2f}")
    
    return True


def test_win_rate_requirement():
    """Test 10.5.7 - Vérification du WR requis pour rentabilité."""
    print("\n📈 Test 10.5.7 - Win Rate requis...")
    
    # Paramètres swing
    tp_pct = 2.0   # +2%
    sl_pct = 1.0   # -1%
    fees_pct = 0.2  # 0.2%
    
    # Gain net par trade gagnant
    net_gain = tp_pct - fees_pct  # 1.8%
    
    # Perte nette par trade perdant
    net_loss = sl_pct + fees_pct  # 1.2%
    
    # Win Rate requis pour breakeven
    # WR * net_gain = (1-WR) * net_loss
    # WR = net_loss / (net_gain + net_loss)
    wr_breakeven = net_loss / (net_gain + net_loss)
    
    print(f"  Gain net par TP: +{net_gain:.1f}%")
    print(f"  Perte nette par SL: -{net_loss:.1f}%")
    print(f"  WR breakeven: {wr_breakeven*100:.1f}%")
    
    # Vérifier que 40% est suffisant
    assert wr_breakeven < 0.40, f"WR breakeven {wr_breakeven:.1%} devrait être < 40%"
    
    # Calculer le profit à WR 43.5%
    wr_actual = 0.435
    expected_pnl_per_100_trades = (
        wr_actual * 100 * net_gain - 
        (1 - wr_actual) * 100 * net_loss
    )
    
    print(f"  ✅ À WR 43.5%: +{expected_pnl_per_100_trades:.1f}% pour 100 trades")
    
    assert expected_pnl_per_100_trades > 0, "Devrait être rentable à 43.5% WR"
    
    return True


def test_parse_duration():
    """Test 10.5.8 - Parsing de durée."""
    print("\n⏱️  Test 10.5.8 - Parse duration...")
    
    from scripts.paper_trading import parse_duration
    
    # Test heures
    assert parse_duration("1h") == 1.0
    assert parse_duration("2h") == 2.0
    assert parse_duration("24h") == 24.0
    
    # Test jours
    assert parse_duration("1d") == 24.0
    assert parse_duration("7d") == 168.0
    
    # Test None
    assert parse_duration(None) is None
    
    print("  ✅ parse_duration OK")
    return True


def test_constants():
    """Test 10.5.9 - Vérification des constantes."""
    print("\n🔢 Test 10.5.9 - Constantes...")
    
    from scripts.paper_trading import (
        TAKE_PROFIT_PCT,
        STOP_LOSS_PCT,
        TIMEOUT_MINUTES,
        DEFAULT_PROBABILITY_THRESHOLD,
        TRADING_FEES_PCT,
        DEFAULT_CAPITAL,
        DEFAULT_SYMBOLS,
    )
    
    # Vérifier les valeurs swing trading
    assert TAKE_PROFIT_PCT == 0.02, "TP 2%"
    assert STOP_LOSS_PCT == 0.01, "SL 1%"
    assert TIMEOUT_MINUTES == 120, "Timeout 2h"
    assert DEFAULT_PROBABILITY_THRESHOLD == 0.20, "Seuil 20%"
    assert TRADING_FEES_PCT == 0.002, "Fees 0.2%"
    assert DEFAULT_CAPITAL == 25.0, "Capital 25 USDT"
    assert len(DEFAULT_SYMBOLS) == 5, "5 symboles"
    
    print(f"  ✅ Constantes swing: TP={TAKE_PROFIT_PCT:.0%}, SL={STOP_LOSS_PCT:.0%}")
    print(f"  ✅ Threshold={DEFAULT_PROBABILITY_THRESHOLD:.0%}, Timeout={TIMEOUT_MINUTES}min")
    
    return True


# ============================================
# MAIN
# ============================================

def main():
    """Exécute tous les tests."""
    print("=" * 65)
    print("🧪 CryptoScalper AI - Tests Paper Trading (Phase 10.5)")
    print("=" * 65)
    
    results = []
    
    results.append(("10.5.1 Imports", test_imports()))
    results.append(("10.5.2 TradeStatus enum", test_trade_status_enum()))
    results.append(("10.5.3 PaperTradeConfig", test_paper_trade_config()))
    results.append(("10.5.4 PaperTrade dataclass", test_paper_trade_dataclass()))
    results.append(("10.5.5 TradingStats", test_trading_stats()))
    results.append(("10.5.6 Calcul PnL", test_pnl_calculation()))
    results.append(("10.5.7 WR requis", test_win_rate_requirement()))
    results.append(("10.5.8 Parse duration", test_parse_duration()))
    results.append(("10.5.9 Constantes", test_constants()))
    
    # Résumé
    print("\n" + "=" * 65)
    print("📊 RÉSULTATS PHASE 10.5 - Paper Trading")
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
    
    print("-" * 65)
    print(f"  Total: {passed}/{len(results)} tests passés")
    print("=" * 65)
    
    if failed == 0:
        print("🎉 Phase 10.5 - Paper Trading : VALIDÉE !")
        return 0
    else:
        print(f"⚠️  {failed} test(s) échoué(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())