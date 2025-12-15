# cryptoscalper/backtest/__init__.py
"""
Module Backtest - Validation de la stratégie sur données historiques.

Ce module contient:
- engine: Moteur de backtest avec simulation réaliste
- reports: Génération de rapports et graphiques

Usage:
    from cryptoscalper.backtest import BacktestEngine, BacktestConfig
    
    engine = BacktestEngine(BacktestConfig(initial_capital=30.0))
    result = engine.run(data, predictor, feature_engine)
    print(result.summary())
"""

from cryptoscalper.backtest.engine import (
    # Enums
    BacktestMode,
    TradeCloseReason,
    # Dataclasses
    BacktestConfig,
    BacktestTrade,
    BacktestState,
    BacktestResult,
    # Classes
    BacktestEngine,
    # Functions
    create_backtest_engine,
    load_historical_data,
)

__all__ = [
    # Enums
    "BacktestMode",
    "TradeCloseReason",
    # Dataclasses
    "BacktestConfig",
    "BacktestTrade",
    "BacktestState",
    "BacktestResult",
    # Classes
    "BacktestEngine",
    # Functions
    "create_backtest_engine",
    "load_historical_data",
]


# Import conditionnel des reports (si disponible)
try:
    from cryptoscalper.backtest.reports import (
        BacktestReport,
        generate_report,
        plot_equity_curve,
        plot_pnl_distribution,
        plot_monthly_returns,
    )
    __all__.extend([
        "BacktestReport",
        "generate_report",
        "plot_equity_curve",
        "plot_pnl_distribution",
        "plot_monthly_returns",
    ])
except ImportError:
    pass  # reports.py pas encore disponible