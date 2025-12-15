# cryptoscalper/models/__init__.py
"""
Module ML pour CryptoScalper.

Contient:
- trainer.py : Entraînement du modèle XGBoost
- predictor.py : Prédiction en temps réel (Phase 6)
"""

from cryptoscalper.models.trainer import (
    ModelTrainer,
    XGBoostConfig,
    TrainingResult,
    EvaluationMetrics,
    FeatureImportance,
    print_threshold_analysis,
    find_optimal_threshold,
)

__all__ = [
    "ModelTrainer",
    "XGBoostConfig",
    "TrainingResult",
    "EvaluationMetrics",
    "FeatureImportance",
    "print_threshold_analysis",
    "find_optimal_threshold",
]