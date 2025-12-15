# cryptoscalper/models/__init__.py
"""
Module ML pour CryptoScalper.

Contient:
- trainer.py : Entraînement du modèle XGBoost
- evaluator.py : Évaluation et visualisations
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

from cryptoscalper.models.evaluator import (
    ModelEvaluator,
    EvaluationReport,
    evaluate_model_from_file,
)

__all__ = [
    # Trainer
    "ModelTrainer",
    "XGBoostConfig",
    "TrainingResult",
    "EvaluationMetrics",
    "FeatureImportance",
    "print_threshold_analysis",
    "find_optimal_threshold",
    # Evaluator
    "ModelEvaluator",
    "EvaluationReport",
    "evaluate_model_from_file",
]