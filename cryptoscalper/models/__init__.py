# cryptoscalper/models/__init__.py
"""
Module ML pour CryptoScalper.

Contient:
- trainer.py : Entraînement du modèle XGBoost
- evaluator.py : Évaluation et visualisations
- predictor.py : Prédiction en temps réel

Note: Les imports sont lazy pour éviter les dépendances circulaires.
Utilisez directement:
    from cryptoscalper.models.predictor import MLPredictor
"""

# Imports lazy - on n'importe que ce qui est demandé
__all__ = [
    # Predictor (Phase 6.1)
    "MLPredictor",
    "PredictionResult",
    "ModelMetadata",
    "load_predictor",
    "predict_single",
]


def __getattr__(name):
    """Import lazy des modules."""
    if name in ["MLPredictor", "PredictionResult", "ModelMetadata", 
                "load_predictor", "predict_single"]:
        from cryptoscalper.models.predictor import (
            MLPredictor,
            PredictionResult,
            ModelMetadata,
            load_predictor,
            predict_single,
        )
        return locals()[name]
    
    raise AttributeError(f"module 'cryptoscalper.models' has no attribute '{name}'")