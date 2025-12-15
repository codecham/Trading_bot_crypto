# cryptoscalper/models/trainer.py
"""
Module d'entraînement du modèle XGBoost.

Responsabilités :
- Entraînement XGBoost avec early stopping
- Calibration des probabilités (CalibratedClassifierCV)
- Évaluation (AUC, precision, recall, F1)
- Feature importance
- Sauvegarde du modèle et des métriques

Usage:
    trainer = ModelTrainer(config)
    result = trainer.train(train_dataset, val_dataset)
    trainer.save_model(result.model, "models/saved/model.joblib")
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from cryptoscalper.config.constants import (
    XGBOOST_N_ESTIMATORS,
    XGBOOST_MAX_DEPTH,
    XGBOOST_LEARNING_RATE,
    MODELS_DIR,
    MODEL_FILENAME,
)
from cryptoscalper.data.dataset import PreparedDataset
from cryptoscalper.utils.logger import logger


# ============================================
# CONSTANTES
# ============================================

# Paramètres XGBoost par défaut
DEFAULT_SUBSAMPLE = 0.8
DEFAULT_COLSAMPLE_BYTREE = 0.8
DEFAULT_MIN_CHILD_WEIGHT = 1
DEFAULT_EARLY_STOPPING_ROUNDS = 20
DEFAULT_RANDOM_STATE = 42

# Calibration
DEFAULT_CALIBRATION_METHOD = "isotonic"  # ou "sigmoid"
DEFAULT_CALIBRATION_CV = 5


# ============================================
# DATACLASSES
# ============================================

@dataclass
class XGBoostConfig:
    """Configuration pour l'entraînement XGBoost."""
    
    # Nombre d'arbres
    n_estimators: int = XGBOOST_N_ESTIMATORS  # 200
    
    # Profondeur max des arbres
    max_depth: int = XGBOOST_MAX_DEPTH  # 6
    
    # Taux d'apprentissage
    learning_rate: float = XGBOOST_LEARNING_RATE  # 0.05
    
    # Sous-échantillonnage
    subsample: float = DEFAULT_SUBSAMPLE  # 0.8
    colsample_bytree: float = DEFAULT_COLSAMPLE_BYTREE  # 0.8
    
    # Régularisation
    min_child_weight: int = DEFAULT_MIN_CHILD_WEIGHT
    reg_alpha: float = 0.0  # L1
    reg_lambda: float = 1.0  # L2
    
    # Gestion des classes déséquilibrées
    # Si None, calculé automatiquement
    scale_pos_weight: Optional[float] = None
    
    # Early stopping
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS
    
    # Reproductibilité
    random_state: int = DEFAULT_RANDOM_STATE
    
    # Métrique d'évaluation
    eval_metric: str = "auc"
    
    # Calibration
    calibrate: bool = True
    calibration_method: str = DEFAULT_CALIBRATION_METHOD
    calibration_cv: int = DEFAULT_CALIBRATION_CV
    
    def to_xgb_params(self) -> Dict[str, Any]:
        """Convertit en paramètres XGBoost."""
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "eval_metric": self.eval_metric,
            "use_label_encoder": False,
            "verbosity": 1,
        }
        
        if self.scale_pos_weight is not None:
            params["scale_pos_weight"] = self.scale_pos_weight
        
        return params


@dataclass
class EvaluationMetrics:
    """Métriques d'évaluation du modèle."""
    
    # Métriques de classification
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    
    # AUC
    roc_auc: float = 0.0
    
    # Matrice de confusion
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    
    # Détails par classe
    classification_report: str = ""
    
    # Courbes (pour visualisation)
    roc_curve: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    pr_curve: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    
    # Métriques à différents seuils
    metrics_by_threshold: Optional[Dict[float, Dict[str, float]]] = None
    
    def summary(self) -> str:
        """Retourne un résumé des métriques."""
        return (
            f"📊 Métriques:\n"
            f"   Accuracy:  {self.accuracy:.4f}\n"
            f"   Precision: {self.precision:.4f}\n"
            f"   Recall:    {self.recall:.4f}\n"
            f"   F1-Score:  {self.f1:.4f}\n"
            f"   ROC-AUC:   {self.roc_auc:.4f}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire (pour JSON)."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }


@dataclass
class FeatureImportance:
    """Importance des features."""
    
    # Feature name -> importance score
    importance_dict: Dict[str, float] = field(default_factory=dict)
    
    # Top N features
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    
    @classmethod
    def from_model(
        cls,
        model: xgb.XGBClassifier,
        feature_names: List[str],
        top_n: int = 20
    ) -> "FeatureImportance":
        """Crée depuis un modèle XGBoost."""
        importance = model.feature_importances_
        importance_dict = dict(zip(feature_names, importance))
        
        # Trier par importance décroissante
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return cls(
            importance_dict=importance_dict,
            top_features=sorted_features[:top_n]
        )
    
    def summary(self, top_n: int = 10) -> str:
        """Retourne un résumé des features importantes."""
        lines = ["🏆 Top Features:"]
        for i, (name, score) in enumerate(self.top_features[:top_n], 1):
            bar = "█" * int(score * 50)
            lines.append(f"   {i:2d}. {name:30s} {score:.4f} {bar}")
        return "\n".join(lines)


@dataclass
class TrainingResult:
    """Résultat complet de l'entraînement."""
    
    # Modèle entraîné (XGBoost ou calibré)
    model: Any
    
    # Modèle XGBoost brut (avant calibration)
    raw_model: xgb.XGBClassifier
    
    # Métriques sur validation
    val_metrics: EvaluationMetrics
    
    # Métriques sur test (si fourni)
    test_metrics: Optional[EvaluationMetrics] = None
    
    # Feature importance
    feature_importance: Optional[FeatureImportance] = None
    
    # Configuration utilisée
    config: Optional[XGBoostConfig] = None
    
    # Métadonnées
    trained_at: datetime = field(default_factory=datetime.now)
    training_time_seconds: float = 0.0
    n_train_samples: int = 0
    n_val_samples: int = 0
    n_features: int = 0
    feature_names: List[str] = field(default_factory=list)
    best_iteration: int = 0
    
    # Est calibré ?
    is_calibrated: bool = False
    
    def summary(self) -> str:
        """Retourne un résumé du training."""
        return (
            f"🎯 Résultat Training\n"
            f"{'=' * 50}\n"
            f"📅 Date: {self.trained_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"⏱️  Durée: {self.training_time_seconds:.1f}s\n"
            f"📊 Échantillons: {self.n_train_samples:,} train, {self.n_val_samples:,} val\n"
            f"🔢 Features: {self.n_features}\n"
            f"🌳 Best iteration: {self.best_iteration}\n"
            f"📐 Calibré: {'Oui' if self.is_calibrated else 'Non'}\n"
            f"\n{self.val_metrics.summary()}"
        )
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convertit en métadonnées pour sauvegarde."""
        return {
            "trained_at": self.trained_at.isoformat(),
            "training_time_seconds": self.training_time_seconds,
            "n_train_samples": self.n_train_samples,
            "n_val_samples": self.n_val_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "best_iteration": self.best_iteration,
            "is_calibrated": self.is_calibrated,
            "val_metrics": self.val_metrics.to_dict(),
            "test_metrics": self.test_metrics.to_dict() if self.test_metrics else None,
            "config": {
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
                "learning_rate": self.config.learning_rate,
            } if self.config else None,
        }


# ============================================
# MODEL TRAINER
# ============================================

class ModelTrainer:
    """
    Entraîneur de modèle XGBoost pour la prédiction de hausse de prix.
    
    Workflow:
    1. Prépare les données (X, y)
    2. Entraîne XGBoost avec early stopping
    3. Calibre les probabilités (optionnel)
    4. Évalue sur validation et test
    5. Calcule feature importance
    
    Usage:
        trainer = ModelTrainer()
        result = trainer.train(train_data, val_data)
        
        # Évaluer sur test
        result = trainer.evaluate(result.model, test_data)
        
        # Sauvegarder
        trainer.save_model(result.model, "model.joblib")
    """
    
    def __init__(self, config: Optional[XGBoostConfig] = None):
        """
        Initialise le trainer.
        
        Args:
            config: Configuration XGBoost (défaut si None)
        """
        self._config = config or XGBoostConfig()
        logger.info(f"🔧 ModelTrainer initialisé (n_estimators={self._config.n_estimators})")
    
    @property
    def config(self) -> XGBoostConfig:
        """Retourne la configuration."""
        return self._config
    
    # =========================================
    # ENTRAÎNEMENT
    # =========================================
    
    def train(
        self,
        train_data: PreparedDataset,
        val_data: PreparedDataset,
        test_data: Optional[PreparedDataset] = None,
        verbose: bool = True
    ) -> TrainingResult:
        """
        Entraîne le modèle XGBoost.
        
        Args:
            train_data: Dataset d'entraînement
            val_data: Dataset de validation
            test_data: Dataset de test (optionnel)
            verbose: Afficher les logs
            
        Returns:
            TrainingResult avec modèle et métriques
        """
        import time
        start_time = time.time()
        
        logger.info("🚀 Début de l'entraînement XGBoost...")
        
        # Préparer les données
        X_train, y_train = train_data.to_numpy()
        X_val, y_val = val_data.to_numpy()
        
        logger.info(f"   Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        logger.info(f"   Val: {X_val.shape[0]:,} samples")
        
        # Calculer scale_pos_weight si non défini
        config = self._config
        if config.scale_pos_weight is None:
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            config = XGBoostConfig(**{
                **vars(self._config),
                "scale_pos_weight": n_neg / n_pos if n_pos > 0 else 1.0
            })
            logger.info(f"   Scale pos weight: {config.scale_pos_weight:.2f}")
        
        # Créer le modèle
        model = xgb.XGBClassifier(**config.to_xgb_params())
        
        # Entraîner avec early stopping
        logger.info("🌳 Entraînement en cours...")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose
        )
        
        best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else config.n_estimators
        logger.info(f"   Best iteration: {best_iteration}")
        
        # Calibration (optionnel)
        calibrated_model = model
        is_calibrated = False
        
        if config.calibrate:
            logger.info("📐 Calibration des probabilités...")
            calibrated_model = self._calibrate_model(model, X_train, y_train)
            is_calibrated = True
            logger.info("   ✅ Calibration terminée")
        
        # Évaluer sur validation
        logger.info("📊 Évaluation sur validation...")
        val_metrics = self._evaluate(calibrated_model, X_val, y_val)
        
        # Évaluer sur test si fourni
        test_metrics = None
        if test_data is not None:
            logger.info("📊 Évaluation sur test...")
            X_test, y_test = test_data.to_numpy()
            test_metrics = self._evaluate(calibrated_model, X_test, y_test)
        
        # Feature importance (du modèle brut)
        feature_importance = FeatureImportance.from_model(
            model,
            train_data.feature_names
        )
        
        training_time = time.time() - start_time
        
        # Construire le résultat
        result = TrainingResult(
            model=calibrated_model,
            raw_model=model,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            feature_importance=feature_importance,
            config=config,
            training_time_seconds=training_time,
            n_train_samples=len(train_data),
            n_val_samples=len(val_data),
            n_features=len(train_data.feature_names),
            feature_names=train_data.feature_names,
            best_iteration=best_iteration,
            is_calibrated=is_calibrated,
        )
        
        logger.info(f"✅ Entraînement terminé en {training_time:.1f}s")
        logger.info(f"\n{result.summary()}")
        
        return result
    
    def _calibrate_model(
        self,
        model: xgb.XGBClassifier,
        X: np.ndarray,
        y: np.ndarray
    ) -> CalibratedClassifierCV:
        """
        Calibre les probabilités du modèle.
        
        La calibration rend les probabilités plus fiables :
        - Si le modèle prédit 70%, il devrait avoir raison ~70% du temps
        
        Args:
            model: Modèle XGBoost entraîné
            X: Features d'entraînement
            y: Labels d'entraînement
            
        Returns:
            Modèle calibré
        """
        calibrated = CalibratedClassifierCV(
            model,
            method=self._config.calibration_method,
            cv=self._config.calibration_cv
        )
        calibrated.fit(X, y)
        return calibrated
    
    # =========================================
    # ÉVALUATION
    # =========================================
    
    def _evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> EvaluationMetrics:
        """
        Évalue le modèle sur un dataset.
        
        Args:
            model: Modèle (XGBoost ou calibré)
            X: Features
            y: Labels vrais
            
        Returns:
            EvaluationMetrics
        """
        # Prédictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        # Métriques de base
        metrics = EvaluationMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, zero_division=0),
            recall=recall_score(y, y_pred, zero_division=0),
            f1=f1_score(y, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y, y_proba),
            confusion_matrix=confusion_matrix(y, y_pred),
            classification_report=classification_report(y, y_pred),
        )
        
        # Courbes
        metrics.roc_curve = roc_curve(y, y_proba)
        metrics.pr_curve = precision_recall_curve(y, y_proba)
        
        # Métriques à différents seuils
        metrics.metrics_by_threshold = self._compute_metrics_by_threshold(y, y_proba)
        
        return metrics
    
    def _compute_metrics_by_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> Dict[float, Dict[str, float]]:
        """
        Calcule les métriques à différents seuils de probabilité.
        
        Utile pour choisir le seuil optimal.
        """
        if thresholds is None:
            thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        
        results = {}
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Nombre de prédictions positives
            n_positive_pred = y_pred.sum()
            
            results[threshold] = {
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "n_predictions": n_positive_pred,
                "pct_predictions": n_positive_pred / len(y_pred) * 100,
            }
        
        return results
    
    def evaluate(
        self,
        model: Any,
        dataset: PreparedDataset
    ) -> EvaluationMetrics:
        """
        Évalue le modèle sur un PreparedDataset.
        
        Args:
            model: Modèle entraîné
            dataset: Dataset à évaluer
            
        Returns:
            Métriques d'évaluation
        """
        X, y = dataset.to_numpy()
        return self._evaluate(model, X, y)
    
    # =========================================
    # SAUVEGARDE / CHARGEMENT
    # =========================================
    
    def save_model(
        self,
        model: Any,
        path: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Sauvegarde le modèle avec joblib.
        
        Args:
            model: Modèle à sauvegarder
            path: Chemin de sauvegarde (défaut: models/saved/xgb_model_latest.joblib)
            metadata: Métadonnées à sauvegarder
            
        Returns:
            Chemin du fichier sauvegardé
        """
        if path is None:
            path = Path(MODELS_DIR) / MODEL_FILENAME
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le modèle
        joblib.dump(model, path)
        logger.info(f"💾 Modèle sauvegardé: {path}")
        
        # Sauvegarder les métadonnées
        if metadata:
            meta_path = path.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"💾 Métadonnées sauvegardées: {meta_path}")
        
        return path
    
    def save_training_result(
        self,
        result: TrainingResult,
        base_path: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Sauvegarde complète d'un résultat d'entraînement.
        
        Sauvegarde:
        - Modèle (.joblib)
        - Métadonnées (.json)
        - Feature importance (.csv)
        
        Args:
            result: Résultat d'entraînement
            base_path: Chemin de base (défaut: models/saved/)
            
        Returns:
            Dictionnaire des chemins sauvegardés
        """
        if base_path is None:
            base_path = Path(MODELS_DIR)
        
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Modèle
        model_path = base_path / MODEL_FILENAME
        paths["model"] = self.save_model(result.model, model_path, result.to_metadata())
        
        # Feature importance
        if result.feature_importance:
            fi_path = base_path / "feature_importance.csv"
            fi_df = pd.DataFrame(
                result.feature_importance.top_features,
                columns=["feature", "importance"]
            )
            fi_df.to_csv(fi_path, index=False)
            paths["feature_importance"] = fi_path
            logger.info(f"💾 Feature importance: {fi_path}")
        
        # Métriques par seuil
        if result.val_metrics.metrics_by_threshold:
            thresh_path = base_path / "metrics_by_threshold.csv"
            rows = []
            for thresh, metrics in result.val_metrics.metrics_by_threshold.items():
                rows.append({"threshold": thresh, **metrics})
            pd.DataFrame(rows).to_csv(thresh_path, index=False)
            paths["metrics_by_threshold"] = thresh_path
            logger.info(f"💾 Métriques par seuil: {thresh_path}")
        
        return paths
    
    @staticmethod
    def load_model(path: Path) -> Any:
        """
        Charge un modèle depuis un fichier.
        
        Args:
            path: Chemin du fichier .joblib
            
        Returns:
            Modèle chargé
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {path}")
        
        model = joblib.load(path)
        logger.info(f"📂 Modèle chargé: {path}")
        return model


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def print_threshold_analysis(metrics: EvaluationMetrics) -> None:
    """
    Affiche l'analyse des métriques par seuil.
    
    Aide à choisir le seuil optimal pour le trading.
    """
    if not metrics.metrics_by_threshold:
        print("Pas de métriques par seuil disponibles.")
        return
    
    print("\n📈 Analyse par seuil de probabilité:")
    print("=" * 70)
    print(f"{'Seuil':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'N Pred':<12} {'%':<8}")
    print("-" * 70)
    
    for threshold, m in sorted(metrics.metrics_by_threshold.items()):
        print(
            f"{threshold:<10.2f} "
            f"{m['precision']:<12.4f} "
            f"{m['recall']:<12.4f} "
            f"{m['f1']:<12.4f} "
            f"{m['n_predictions']:<12.0f} "
            f"{m['pct_predictions']:<8.1f}"
        )
    
    print("=" * 70)
    print("\n💡 Conseil: Pour le scalping, privilégiez un seuil élevé (0.65-0.75)")
    print("   - Moins de trades mais meilleure précision")


def find_optimal_threshold(
    metrics: EvaluationMetrics,
    min_precision: float = 0.55,
    min_predictions_pct: float = 5.0
) -> Optional[float]:
    """
    Trouve le seuil optimal selon les critères.
    
    Args:
        metrics: Métriques d'évaluation
        min_precision: Précision minimum requise
        min_predictions_pct: % minimum de prédictions positives
        
    Returns:
        Seuil optimal ou None si aucun ne convient
    """
    if not metrics.metrics_by_threshold:
        return None
    
    best_threshold = None
    best_f1 = 0.0
    
    for threshold, m in metrics.metrics_by_threshold.items():
        if m["precision"] >= min_precision and m["pct_predictions"] >= min_predictions_pct:
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_threshold = threshold
    
    return best_threshold