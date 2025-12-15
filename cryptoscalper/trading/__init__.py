# cryptoscalper/trading/__init__.py
"""
Module Trading - Gestion des signaux, risques et exécution.

Ce module contient:
- signals: Génération et gestion des signaux de trading
- risk_manager: Gestion du risque (position sizing, kill switch, limites)
- executor: Exécution des ordres (market, OCO, positions)
"""

# Executor (Phase 8)
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

__all__ = [
    # Executor
    "TradeExecutor",
    "ExecutorConfig",
    "SymbolInfo",
    "OrderResult",
    "Position",
    "CompletedTrade",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "PositionStatus",
    "CloseReason",
    "create_executor",
]


# Import conditionnel des signals (si disponible)
try:
    from cryptoscalper.trading.signals import (
        SignalGenerator,
        SignalConfig,
        TradeSignal,
        SignalType,
        SignalStrength,
        SignalStatus,
        create_signal_generator,
        filter_signals_by_strength,
    )
    __all__.extend([
        "SignalGenerator",
        "SignalConfig",
        "TradeSignal",
        "SignalType",
        "SignalStrength",
        "SignalStatus",
        "create_signal_generator",
        "filter_signals_by_strength",
    ])
except ImportError:
    pass  # signals.py pas encore disponible localement


# Import conditionnel du risk_manager (si disponible)
try:
    from cryptoscalper.trading.risk_manager import (
        RiskManager,
        RiskConfig,
        TradeRecord,
        DailyLimits,
        KillSwitch,
        RejectionReason,
        TradeOutcome,
        create_risk_manager,
        calculate_risk_reward_ratio,
    )
    __all__.extend([
        "RiskManager",
        "RiskConfig",
        "TradeRecord",
        "DailyLimits",
        "KillSwitch",
        "RejectionReason",
        "TradeOutcome",
        "create_risk_manager",
        "calculate_risk_reward_ratio",
    ])
except ImportError:
    pass  # risk_manager.py pas encore disponible localement