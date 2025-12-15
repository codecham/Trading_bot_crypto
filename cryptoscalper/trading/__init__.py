# cryptoscalper/trading/__init__.py
"""
Module Trading - Gestion des signaux, risques et exécution.

Ce module contient:
- risk_manager: Gestion du risque (position sizing, kill switch, limites)
- signals: Génération et gestion des signaux de trading (à synchroniser)
- executor: Exécution des ordres (à venir)
"""

# Risk Management (toujours disponible)
from cryptoscalper.trading.risk_manager import (
    RiskManager,
    RiskConfig,
    CompletedTrade,
    TradeRecord,
    DailyLimits,
    KillSwitch,
    RejectionReason,
    TradeOutcome,
    create_risk_manager,
    calculate_risk_reward_ratio,
)


__all__ = [
    # Risk Management
    "RiskManager",
    "RiskConfig",
    "CompletedTrade",
    "TradeRecord",
    "DailyLimits",
    "KillSwitch",
    "RejectionReason",
    "TradeOutcome",
    "create_risk_manager",
    "calculate_risk_reward_ratio",
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