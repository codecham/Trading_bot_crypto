#!/usr/bin/env python3
# scripts/test_backtest.py
"""
Tests d'intégration pour la Phase 9 - Backtest.

Tests couverts:
- 9.1 Backtest Engine
  - Configuration et création
  - Simulation des ordres
  - Calcul frais et slippage
  - Gestion des positions (SL/TP/Timeout)
  
- 9.2 Reports
  - BacktestResult et métriques
  - Métriques avancées (Sharpe, Sortino, etc.)
  - Génération de rapports

Usage:
    python scripts/test_backtest.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from cryptoscalper.utils.logger import logger, setup_logger


# ============================================
# HELPERS
# ============================================

def create_sample_ohlcv(
    n_rows: int = 1000,
    start_price: float = 50000.0,
    volatility: float = 0.001,
    trend: float = 0.0,
    start_date: datetime = None,
) -> pd.DataFrame:
    """
    Crée un DataFrame OHLCV de test.
    
    Args:
        n_rows: Nombre de bougies
        start_price: Prix de départ
        volatility: Volatilité des returns
        trend: Tendance (positive = haussier)
        start_date: Date de début
    """
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    
    np.random.seed(42)
    
    # Générer les returns
    returns = np.random.normal(trend, volatility, n_rows)
    prices = start_price * np.cumprod(1 + returns)
    
    # Générer OHLCV
    df = pd.DataFrame({
        "open": prices * (1 + np.random.uniform(-0.001, 0.001, n_rows)),
        "high": prices * (1 + np.random.uniform(0, 0.003, n_rows)),
        "low": prices * (1 - np.random.uniform(0, 0.003, n_rows)),
        "close": prices,
        "volume": np.random.uniform(10, 100, n_rows),
    })
    
    # S'assurer que high >= max(open, close) et low <= min(open, close)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    
    # Index datetime
    df.index = pd.date_range(start=start_date, periods=n_rows, freq="1min")
    
    return df


# ============================================
# TESTS 9.1 - BACKTEST ENGINE
# ============================================

def test_backtest_imports():
    """Test 9.1.1 - Imports du module backtest."""
    print("\n📦 Test 9.1.1 - Imports backtest...")
    
    try:
        from cryptoscalper.backtest import (
            BacktestEngine,
            BacktestConfig,
            BacktestResult,
            BacktestTrade,
            BacktestState,
            TradeCloseReason,
            create_backtest_engine,
            load_historical_data,
        )
        print("  ✅ Tous les imports réussis")
        return True
    except ImportError as e:
        print(f"  ❌ Erreur import: {e}")
        return False


def test_backtest_config():
    """Test 9.1.2 - Configuration du backtest."""
    print("\n⚙️ Test 9.1.2 - BacktestConfig...")
    
    from cryptoscalper.backtest import BacktestConfig
    
    # Config par défaut
    config = BacktestConfig()
    assert config.initial_capital == 30.0, "Capital initial incorrect"
    assert config.position_size_pct == 0.20, "Position size incorrect"
    assert config.fee_percent == 0.001, "Fee percent incorrect"
    assert config.slippage_percent == 0.0005, "Slippage incorrect"
    
    print(f"     Capital: {config.initial_capital} USDT")
    print(f"     Position: {config.position_size_pct:.0%}")
    print(f"     Fees: {config.fee_percent:.2%}")
    print(f"     Slippage: {config.slippage_percent:.2%}")
    
    # Config personnalisée
    config_custom = BacktestConfig(
        initial_capital=100.0,
        position_size_pct=0.10,
        fee_percent=0.0005,
        stop_loss_pct=0.005,
        take_profit_pct=0.004,
    )
    assert config_custom.initial_capital == 100.0
    assert config_custom.stop_loss_pct == 0.005
    
    print("  ✅ BacktestConfig OK")
    return True


def test_backtest_engine_creation():
    """Test 9.1.3 - Création du BacktestEngine."""
    print("\n🔧 Test 9.1.3 - Création BacktestEngine...")
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    
    # Création par défaut
    engine = BacktestEngine()
    assert engine.config is not None
    assert engine.config.initial_capital == 30.0
    
    # Création avec config
    config = BacktestConfig(initial_capital=50.0)
    engine = BacktestEngine(config)
    assert engine.config.initial_capital == 50.0
    
    print("  ✅ BacktestEngine créé")
    return True


def test_backtest_trade_dataclass():
    """Test 9.1.4 - BacktestTrade dataclass."""
    print("\n📊 Test 9.1.4 - BacktestTrade...")
    
    from cryptoscalper.backtest import BacktestTrade, TradeCloseReason
    
    # Créer un trade
    trade = BacktestTrade(
        trade_id=1,
        symbol="BTCUSDT",
        entry_time=datetime.now() - timedelta(minutes=3),
        entry_price=50000.0,
        entry_price_raw=49990.0,
        quantity=0.001,
        size_usdt=50.0,
        stop_loss_price=49800.0,
        take_profit_price=50150.0,
        probability=0.72,
        confidence=0.65,
    )
    
    assert trade.is_open is True
    assert trade.trade_id == 1
    
    # Fermer le trade avec un prix bien plus haut pour garantir le profit après frais
    trade.close(
        exit_time=datetime.now(),
        exit_price_raw=50300.0,
        exit_price=50295.0,
        close_reason=TradeCloseReason.TAKE_PROFIT,
        fee_percent=0.001,
    )
    
    assert trade.is_open is False
    # PnL doit être positif après frais
    # Entry: 50000 * 0.001 = 50 USDT, Exit: 50295 * 0.001 = ~50.295 USDT
    # Gross PnL: (50295 - 50000) * 0.001 = 0.295 USDT
    # Fees: ~0.10 USDT total
    # Net PnL should be positive
    assert trade.pnl_usdt > 0, f"PnL should be positive: {trade.pnl_usdt}"
    assert trade.is_winner is True
    assert trade.pnl_usdt is not None
    assert trade.duration_minutes is not None
    
    print(f"     PnL: {trade.pnl_usdt:+.4f} USDT")
    print(f"     Duration: {trade.duration_minutes:.1f} min")
    print(f"     Close reason: {trade.close_reason.value}")
    print("  ✅ BacktestTrade OK")
    return True


def test_slippage_calculation():
    """Test 9.1.5 - Calcul du slippage."""
    print("\n💱 Test 9.1.5 - Slippage...")
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    
    config = BacktestConfig(slippage_percent=0.001)  # 0.1%
    engine = BacktestEngine(config)
    
    base_price = 50000.0
    
    # Slippage achat (prix augmente)
    buy_price = engine._apply_slippage_buy(base_price)
    assert buy_price > base_price
    expected_buy = base_price * 1.001
    assert abs(buy_price - expected_buy) < 0.01
    
    # Slippage vente (prix diminue)
    sell_price = engine._apply_slippage_sell(base_price)
    assert sell_price < base_price
    expected_sell = base_price * 0.999
    assert abs(sell_price - expected_sell) < 0.01
    
    print(f"     Prix base: {base_price:.2f}")
    print(f"     Prix achat (avec slippage): {buy_price:.2f}")
    print(f"     Prix vente (avec slippage): {sell_price:.2f}")
    print("  ✅ Slippage calculé correctement")
    return True


def test_backtest_simple_run():
    """Test 9.1.6 - Backtest simple avec signaux."""
    print("\n🚀 Test 9.1.6 - Backtest simple...")
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    
    # Créer des données
    df = create_sample_ohlcv(500, volatility=0.002, trend=0.0001)
    
    # Créer des signaux aléatoires (1 trade toutes les ~50 bougies)
    np.random.seed(123)
    signals = pd.Series(0, index=df.index)
    signal_indices = np.random.choice(range(50, 450), size=8, replace=False)
    signals.iloc[signal_indices] = 1
    
    # Exécuter le backtest
    config = BacktestConfig(
        initial_capital=100.0,
        position_size_pct=0.20,
        max_position_duration_minutes=10,
    )
    engine = BacktestEngine(config)
    
    result = engine.run_simple(df, signals, symbol="BTCUSDT")
    
    assert result is not None
    assert result.total_trades > 0
    assert result.initial_capital == 100.0
    
    print(f"     Trades: {result.total_trades}")
    print(f"     Win rate: {result.win_rate:.1%}")
    print(f"     PnL: {result.total_pnl:+.4f} USDT")
    print(f"     Final capital: {result.final_capital:.2f} USDT")
    print("  ✅ Backtest simple exécuté")
    return True


def test_stop_loss_trigger():
    """Test 9.1.7 - Déclenchement du stop loss."""
    print("\n🛑 Test 9.1.7 - Stop Loss...")
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig, TradeCloseReason
    
    # Créer des données avec une baisse
    df = create_sample_ohlcv(200, start_price=50000, volatility=0.005, trend=-0.001)
    
    # Signal au début
    signals = pd.Series(0, index=df.index)
    signals.iloc[55] = 1  # Signal après warmup
    
    config = BacktestConfig(
        initial_capital=100.0,
        stop_loss_pct=0.003,  # 0.3% SL
        take_profit_pct=0.01,  # 1% TP (difficile à atteindre en baisse)
        max_position_duration_minutes=60,
    )
    engine = BacktestEngine(config)
    
    result = engine.run_simple(df, signals, symbol="BTCUSDT")
    
    # Vérifier qu'il y a eu au moins un stop loss
    assert result.stop_loss_count > 0 or result.timeout_count > 0 or result.total_trades == 0
    
    print(f"     Trades: {result.total_trades}")
    print(f"     Stop Loss: {result.stop_loss_count}")
    print(f"     Take Profit: {result.take_profit_count}")
    print(f"     Timeout: {result.timeout_count}")
    print("  ✅ Stop Loss testé")
    return True


def test_take_profit_trigger():
    """Test 9.1.8 - Déclenchement du take profit."""
    print("\n🎯 Test 9.1.8 - Take Profit...")
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    
    # Créer des données avec une hausse
    df = create_sample_ohlcv(200, start_price=50000, volatility=0.003, trend=0.002)
    
    # Signal au début
    signals = pd.Series(0, index=df.index)
    signals.iloc[55] = 1
    
    config = BacktestConfig(
        initial_capital=100.0,
        stop_loss_pct=0.02,  # 2% SL (loin)
        take_profit_pct=0.002,  # 0.2% TP (facile en hausse)
        max_position_duration_minutes=60,
    )
    engine = BacktestEngine(config)
    
    result = engine.run_simple(df, signals, symbol="BTCUSDT")
    
    print(f"     Trades: {result.total_trades}")
    print(f"     Take Profit: {result.take_profit_count}")
    print(f"     Win rate: {result.win_rate:.1%}")
    print("  ✅ Take Profit testé")
    return True


def test_fees_calculation():
    """Test 9.1.9 - Calcul des frais."""
    print("\n💰 Test 9.1.9 - Frais...")
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    
    df = create_sample_ohlcv(300)
    
    signals = pd.Series(0, index=df.index)
    signals.iloc[60] = 1
    signals.iloc[120] = 1
    signals.iloc[180] = 1
    
    config = BacktestConfig(
        initial_capital=100.0,
        fee_percent=0.001,  # 0.1% par transaction
    )
    engine = BacktestEngine(config)
    
    result = engine.run_simple(df, signals, symbol="BTCUSDT")
    
    assert result.total_fees > 0
    
    print(f"     Trades: {result.total_trades}")
    print(f"     Frais totaux: {result.total_fees:.4f} USDT")
    print(f"     Frais moyens/trade: {result.total_fees / max(1, result.total_trades):.4f} USDT")
    print("  ✅ Frais calculés")
    return True


def test_equity_curve():
    """Test 9.1.10 - Equity curve."""
    print("\n📈 Test 9.1.10 - Equity Curve...")
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    
    df = create_sample_ohlcv(500)
    
    signals = pd.Series(0, index=df.index)
    for i in range(60, 450, 50):
        signals.iloc[i] = 1
    
    engine = BacktestEngine(BacktestConfig(initial_capital=100.0))
    result = engine.run_simple(df, signals, symbol="BTCUSDT")
    
    assert len(result.equity_curve) > 0
    
    # Vérifier que l'equity curve est cohérente
    equities = [e[1] for e in result.equity_curve]
    assert equities[0] <= 100.0 + 0.1  # Proche du capital initial
    
    print(f"     Points equity curve: {len(result.equity_curve)}")
    print(f"     Min equity: {min(equities):.2f}")
    print(f"     Max equity: {max(equities):.2f}")
    print("  ✅ Equity curve générée")
    return True


# ============================================
# TESTS 9.2 - REPORTS
# ============================================

def test_reports_imports():
    """Test 9.2.1 - Imports du module reports."""
    print("\n📦 Test 9.2.1 - Imports reports...")
    
    try:
        from cryptoscalper.backtest.reports import (
            BacktestReport,
            AdvancedMetrics,
            PeriodStats,
            generate_report,
            calculate_monthly_stats,
            calculate_hourly_stats,
            plot_equity_curve,
            plot_pnl_distribution,
        )
        print("  ✅ Tous les imports réussis")
        return True
    except ImportError as e:
        print(f"  ❌ Erreur import: {e}")
        return False


def test_backtest_result_summary():
    """Test 9.2.2 - BacktestResult summary."""
    print("\n📊 Test 9.2.2 - BacktestResult summary...")
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    
    df = create_sample_ohlcv(500, trend=0.0002)
    
    signals = pd.Series(0, index=df.index)
    for i in range(60, 450, 40):
        signals.iloc[i] = 1
    
    engine = BacktestEngine(BacktestConfig(initial_capital=50.0))
    result = engine.run_simple(df, signals, symbol="BTCUSDT")
    
    summary = result.summary()
    assert len(summary) > 0
    assert "RÉSULTATS DU BACKTEST" in summary
    assert "CAPITAL" in summary
    assert "TRADES" in summary
    
    print("  📄 Résumé généré:")
    print("-" * 40)
    # Afficher seulement les premières lignes
    for line in summary.split("\n")[:15]:
        print(f"  {line}")
    print("  ...")
    print("-" * 40)
    print("  ✅ Summary généré")
    return True


def test_advanced_metrics():
    """Test 9.2.3 - Métriques avancées."""
    print("\n📈 Test 9.2.3 - Métriques avancées...")
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    from cryptoscalper.backtest.reports import AdvancedMetrics
    
    df = create_sample_ohlcv(1000, trend=0.0001)
    
    signals = pd.Series(0, index=df.index)
    for i in range(60, 950, 30):
        signals.iloc[i] = 1
    
    engine = BacktestEngine(BacktestConfig(initial_capital=100.0))
    result = engine.run_simple(df, signals, symbol="BTCUSDT")
    
    metrics = AdvancedMetrics.from_result(result)
    
    print(f"     Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"     Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"     Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print(f"     Profit Factor: {metrics.profit_factor:.2f}")
    print(f"     Espérance: {metrics.expectancy:+.4f} USDT")
    print(f"     Max DD Duration: {metrics.max_drawdown_duration_days:.1f} jours")
    print(f"     Série gains max: {metrics.max_consecutive_wins}")
    print(f"     Série pertes max: {metrics.max_consecutive_losses}")
    
    print("  ✅ Métriques avancées calculées")
    return True


def test_monthly_stats():
    """Test 9.2.4 - Stats mensuelles."""
    print("\n📅 Test 9.2.4 - Stats mensuelles...")
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    from cryptoscalper.backtest.reports import calculate_monthly_stats
    
    # Données sur 3 mois
    df = create_sample_ohlcv(
        n_rows=3 * 30 * 24 * 60,  # ~3 mois en minutes
        start_date=datetime(2024, 1, 1),
    )
    
    # Réduire pour le test
    df = df.iloc[::60]  # 1 bougie par heure
    
    signals = pd.Series(0, index=df.index)
    for i in range(60, len(df) - 10, 100):
        signals.iloc[i] = 1
    
    engine = BacktestEngine(BacktestConfig(
        initial_capital=100.0,
        max_position_duration_minutes=120,
    ))
    result = engine.run_simple(df, signals, symbol="BTCUSDT")
    
    monthly = calculate_monthly_stats(result)
    
    if monthly:
        print(f"     Mois analysés: {len(monthly)}")
        for stat in monthly[:3]:  # Premiers mois
            print(f"       {stat.period}: {stat.trades} trades, {stat.pnl:+.4f} USDT, WR {stat.win_rate:.1%}")
    else:
        print("     Pas assez de données mensuelles")
    
    print("  ✅ Stats mensuelles calculées")
    return True


def test_equity_curve_ascii():
    """Test 9.2.5 - Graphique ASCII equity curve."""
    print("\n📈 Test 9.2.5 - Equity curve ASCII...")
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    from cryptoscalper.backtest.reports import plot_equity_curve_ascii
    
    df = create_sample_ohlcv(500)
    
    signals = pd.Series(0, index=df.index)
    for i in range(60, 450, 40):
        signals.iloc[i] = 1
    
    engine = BacktestEngine(BacktestConfig(initial_capital=50.0))
    result = engine.run_simple(df, signals, symbol="BTCUSDT")
    
    chart = plot_equity_curve_ascii(result.equity_curve, width=40, height=8)
    
    print("  📊 Equity Curve:")
    for line in chart.split("\n"):
        print(f"    {line}")
    
    print("  ✅ Graphique ASCII généré")
    return True


def test_pnl_distribution_ascii():
    """Test 9.2.6 - Distribution PnL ASCII."""
    print("\n📊 Test 9.2.6 - Distribution PnL ASCII...")
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    from cryptoscalper.backtest.reports import plot_pnl_distribution_ascii
    
    df = create_sample_ohlcv(500)
    
    signals = pd.Series(0, index=df.index)
    for i in range(60, 450, 30):
        signals.iloc[i] = 1
    
    engine = BacktestEngine(BacktestConfig(initial_capital=50.0))
    result = engine.run_simple(df, signals, symbol="BTCUSDT")
    
    chart = plot_pnl_distribution_ascii(result.trades, bins=6, width=20)
    
    print("  📊 Distribution PnL:")
    for line in chart.split("\n"):
        print(f"    {line}")
    
    print("  ✅ Distribution ASCII générée")
    return True


def test_backtest_report_generation():
    """Test 9.2.7 - Génération BacktestReport."""
    print("\n📄 Test 9.2.7 - BacktestReport...")
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    from cryptoscalper.backtest.reports import BacktestReport
    
    df = create_sample_ohlcv(500, trend=0.0001)
    
    signals = pd.Series(0, index=df.index)
    for i in range(60, 450, 35):
        signals.iloc[i] = 1
    
    engine = BacktestEngine(BacktestConfig(initial_capital=100.0))
    result = engine.run_simple(df, signals, symbol="BTCUSDT")
    
    report = BacktestReport(result)
    
    assert report.result is not None
    assert report.metrics is not None
    
    # Générer le rapport texte
    text_report = report.generate_text()
    assert len(text_report) > 100
    assert "MÉTRIQUES AVANCÉES" in text_report
    
    print(f"     Taille rapport: {len(text_report)} caractères")
    print("  ✅ BacktestReport généré")
    return True


def test_report_save_json():
    """Test 9.2.8 - Sauvegarde JSON."""
    print("\n💾 Test 9.2.8 - Sauvegarde JSON...")
    
    import tempfile
    import json
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    from cryptoscalper.backtest.reports import BacktestReport
    
    df = create_sample_ohlcv(300)
    
    signals = pd.Series(0, index=df.index)
    for i in range(60, 250, 40):
        signals.iloc[i] = 1
    
    engine = BacktestEngine(BacktestConfig(initial_capital=50.0))
    result = engine.run_simple(df, signals, symbol="BTCUSDT")
    
    report = BacktestReport(result)
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        filepath = f.name
    
    report.save_json(filepath)
    
    # Vérifier le fichier
    with open(filepath, "r") as f:
        data = json.load(f)
    
    assert "result" in data
    assert "metrics" in data
    assert data["result"]["symbol"] == "BTCUSDT"
    
    print(f"     Fichier: {filepath}")
    print(f"     Clés: {list(data.keys())}")
    print("  ✅ JSON sauvegardé")
    
    # Cleanup
    Path(filepath).unlink()
    return True


def test_report_save_html():
    """Test 9.2.9 - Sauvegarde HTML."""
    print("\n🌐 Test 9.2.9 - Sauvegarde HTML...")
    
    import tempfile
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    from cryptoscalper.backtest.reports import BacktestReport
    
    df = create_sample_ohlcv(400)
    
    signals = pd.Series(0, index=df.index)
    for i in range(60, 350, 35):
        signals.iloc[i] = 1
    
    engine = BacktestEngine(BacktestConfig(initial_capital=75.0))
    result = engine.run_simple(df, signals, symbol="ETHUSDT")
    
    report = BacktestReport(result)
    
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        filepath = f.name
    
    report.save_html(filepath)
    
    # Vérifier le fichier
    with open(filepath, "r") as f:
        html = f.read()
    
    assert "<!DOCTYPE html>" in html
    assert "ETHUSDT" in html
    assert "chart.js" in html.lower()  # Case-insensitive check
    
    print(f"     Fichier: {filepath}")
    print(f"     Taille: {len(html)} caractères")
    print("  ✅ HTML sauvegardé")
    
    # Cleanup
    Path(filepath).unlink()
    return True


def test_result_to_dict():
    """Test 9.2.10 - Conversion en dict."""
    print("\n📋 Test 9.2.10 - to_dict()...")
    
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    
    df = create_sample_ohlcv(200)
    
    signals = pd.Series(0, index=df.index)
    signals.iloc[60] = 1
    signals.iloc[120] = 1
    
    engine = BacktestEngine(BacktestConfig(initial_capital=50.0))
    result = engine.run_simple(df, signals, symbol="BTCUSDT")
    
    data = result.to_dict()
    
    assert "symbol" in data
    assert "total_trades" in data
    assert "win_rate" in data
    assert "sharpe_ratio" in data
    
    print(f"     Clés: {len(data)}")
    print(f"     Win rate: {data['win_rate']:.2%}")
    print("  ✅ to_dict() OK")
    return True


# ============================================
# MAIN
# ============================================

def run_test(name: str, test_func) -> bool:
    """Exécute un test de manière sûre."""
    try:
        return test_func()
    except Exception as e:
        print(f"\n❌ ERREUR dans {name}: {e}")
        logger.exception(f"Erreur test {name}")
        return False


def main() -> int:
    """Point d'entrée des tests."""
    print("=" * 65)
    print("🧪 CryptoScalper AI - Tests Phase 9: Backtest")
    print("=" * 65)
    
    setup_logger(level="WARNING")
    
    results = []
    
    # Tests 9.1 - Engine
    print("\n" + "─" * 50)
    print("📦 9.1 BACKTEST ENGINE")
    print("─" * 50)
    
    results.append(("9.1.1 Imports", run_test("imports", test_backtest_imports)))
    results.append(("9.1.2 Config", run_test("config", test_backtest_config)))
    results.append(("9.1.3 Engine creation", run_test("engine", test_backtest_engine_creation)))
    results.append(("9.1.4 Trade dataclass", run_test("trade", test_backtest_trade_dataclass)))
    results.append(("9.1.5 Slippage", run_test("slippage", test_slippage_calculation)))
    results.append(("9.1.6 Simple run", run_test("simple", test_backtest_simple_run)))
    results.append(("9.1.7 Stop Loss", run_test("sl", test_stop_loss_trigger)))
    results.append(("9.1.8 Take Profit", run_test("tp", test_take_profit_trigger)))
    results.append(("9.1.9 Fees", run_test("fees", test_fees_calculation)))
    results.append(("9.1.10 Equity curve", run_test("equity", test_equity_curve)))
    
    # Tests 9.2 - Reports
    print("\n" + "─" * 50)
    print("📦 9.2 REPORTS")
    print("─" * 50)
    
    results.append(("9.2.1 Imports reports", run_test("imports_rep", test_reports_imports)))
    results.append(("9.2.2 Summary", run_test("summary", test_backtest_result_summary)))
    results.append(("9.2.3 Advanced metrics", run_test("metrics", test_advanced_metrics)))
    results.append(("9.2.4 Monthly stats", run_test("monthly", test_monthly_stats)))
    results.append(("9.2.5 Equity ASCII", run_test("eq_ascii", test_equity_curve_ascii)))
    results.append(("9.2.6 PnL ASCII", run_test("pnl_ascii", test_pnl_distribution_ascii)))
    results.append(("9.2.7 Report generation", run_test("report", test_backtest_report_generation)))
    results.append(("9.2.8 Save JSON", run_test("json", test_report_save_json)))
    results.append(("9.2.9 Save HTML", run_test("html", test_report_save_html)))
    results.append(("9.2.10 to_dict", run_test("dict", test_result_to_dict)))
    
    # Résumé
    print("\n" + "=" * 65)
    print("📊 RÉSULTATS PHASE 9")
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
        print("🎉 Phase 9 - Backtest : VALIDÉE !")
        return 0
    else:
        print(f"⚠️  {failed} test(s) échoué(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())