# cryptoscalper/models/predictor.py
"""
Module de prédiction ML pour le trading en temps réel.

Responsabilités :
- Charger le modèle XGBoost entraîné
- Faire des prédictions single et batch
- Calculer la confiance des prédictions
- Gérer le cache des features pour performance

Usage:
    predictor = MLPredictor.from_file("models/saved/xgb_model_latest.joblib")
    
    # Prédiction unique
    result = predictor.predict(feature_set)
    print(f"Proba: {result.probability:.2%}, Confiance: {result.confidence:.2%}")
    
    # Prédiction batch
    results = predictor.predict_batch([features1, features2, features3])
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

import numpy as np
import pandas as pd
import joblib

from cryptoscalper.config.constants import (
    MODELS_DIR,
    MODEL_FILENAME,
    SIGNAL_MIN_PROBABILITY,
    SIGNAL_MIN_CONFIDENCE,
)
from cryptoscalper.data.features import FeatureSet, FeatureEngine, get_feature_names
from cryptoscalper.utils.logger import logger


# ============================================
# CONSTANTES
# ============================================

# Seuils par défaut
DEFAULT_MIN_PROBABILITY = SIGNAL_MIN_PROBABILITY  # 0.65
DEFAULT_MIN_CONFIDENCE = SIGNAL_MIN_CONFIDENCE    # 0.55

# Seuil pour considérer une prédiction comme "confiante"
# Plus la proba est proche de 0.5, moins on est confiant
CONFIDENCE_NEUTRAL_POINT = 0.5


# ============================================
# DATACLASSES
# ============================================

@dataclass
class PredictionResult:
    """
    Résultat d'une prédiction ML.
    
    Attributes:
        symbol: Symbole de la paire tradée
        probability_up: Probabilité de hausse (0-1)
        probability_down: Probabilité de baisse (0-1)
        predicted_class: Classe prédite (0 ou 1)
        confidence: Score de confiance (0-1)
        timestamp: Moment de la prédiction
        features_used: Nombre de features utilisées
        model_version: Version du modèle utilisé
    """
    
    symbol: str
    probability_up: float
    probability_down: float
    predicted_class: int
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    features_used: int = 42
    model_version: str = "unknown"
    
    @property
    def is_bullish(self) -> bool:
        """Indique si la prédiction est haussière."""
        return self.predicted_class == 1
    
    @property
    def is_confident(self) -> bool:
        """Indique si la prédiction est confiante (> 55%)."""
        return self.confidence >= DEFAULT_MIN_CONFIDENCE
    
    @property
    def is_strong_signal(self) -> bool:
        """Indique si c'est un signal fort (proba >= 65% et confiant)."""
        return self.probability_up >= DEFAULT_MIN_PROBABILITY and self.is_confident
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "symbol": self.symbol,
            "probability_up": self.probability_up,
            "probability_down": self.probability_down,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "features_used": self.features_used,
            "model_version": self.model_version,
            "is_bullish": self.is_bullish,
            "is_strong_signal": self.is_strong_signal,
        }
    
    def __str__(self) -> str:
        """Représentation string lisible."""
        direction = "📈 HAUSSE" if self.is_bullish else "📉 BAISSE"
        strength = "💪 FORT" if self.is_strong_signal else "⚡ FAIBLE"
        return (
            f"{self.symbol}: {direction} | "
            f"Proba: {self.probability_up:.1%} | "
            f"Confiance: {self.confidence:.1%} | "
            f"{strength}"
        )


@dataclass
class ModelMetadata:
    """
    Métadonnées du modèle chargé.
    
    Attributes:
        trained_at: Date d'entraînement
        n_features: Nombre de features attendues
        feature_names: Liste des noms de features
        is_calibrated: Si le modèle est calibré
        val_auc: AUC sur validation
        model_path: Chemin du fichier modèle
    """
    
    trained_at: Optional[datetime] = None
    n_features: int = 42
    feature_names: List[str] = field(default_factory=list)
    is_calibrated: bool = False
    val_auc: float = 0.0
    model_path: str = ""
    
    @classmethod
    def from_json(cls, json_path: Path) -> "ModelMetadata":
        """Charge les métadonnées depuis un fichier JSON."""
        if not json_path.exists():
            logger.warning(f"Fichier métadonnées introuvable: {json_path}")
            return cls()
        
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            trained_at = None
            if "trained_at" in data:
                trained_at = datetime.fromisoformat(data["trained_at"])
            
            val_auc = 0.0
            if "val_metrics" in data and "roc_auc" in data["val_metrics"]:
                val_auc = data["val_metrics"]["roc_auc"]
            
            return cls(
                trained_at=trained_at,
                n_features=data.get("n_features", 42),
                feature_names=data.get("feature_names", []),
                is_calibrated=data.get("is_calibrated", False),
                val_auc=val_auc,
                model_path=str(json_path.with_suffix(".joblib")),
            )
        except Exception as e:
            logger.error(f"Erreur lecture métadonnées: {e}")
            return cls()


# ============================================
# ML PREDICTOR
# ============================================

class MLPredictor:
    """
    Prédicteur ML pour le scalping crypto.
    
    Charge un modèle XGBoost entraîné et fournit des prédictions
    en temps réel avec calcul de confiance.
    
    Workflow:
    1. Charger le modèle avec from_file()
    2. Passer des features via predict() ou predict_batch()
    3. Récupérer les résultats avec probabilités et confiance
    
    Exemple:
        predictor = MLPredictor.from_file("models/saved/model.joblib")
        
        # Avec un FeatureSet
        result = predictor.predict(feature_set)
        
        # Avec un dictionnaire de features
        result = predictor.predict_from_dict({"rsi_14": 65.2, ...}, "BTCUSDT")
        
        # Batch
        results = predictor.predict_batch([fs1, fs2, fs3])
    """
    
    def __init__(
        self,
        model: Any,
        metadata: Optional[ModelMetadata] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialise le predictor avec un modèle déjà chargé.
        
        Args:
            model: Modèle sklearn (XGBoost ou calibré)
            metadata: Métadonnées du modèle
            feature_names: Liste ordonnée des noms de features
        """
        self._model = model
        self._metadata = metadata or ModelMetadata()
        
        # Utiliser les feature names du metadata ou ceux par défaut
        if feature_names:
            self._feature_names = feature_names
        elif self._metadata.feature_names:
            self._feature_names = self._metadata.feature_names
        else:
            self._feature_names = get_feature_names()
        
        self._n_features = len(self._feature_names)
        
        logger.info(
            f"🤖 MLPredictor initialisé | "
            f"Features: {self._n_features} | "
            f"Calibré: {self._metadata.is_calibrated}"
        )
    
    # =========================================
    # FACTORY METHODS
    # =========================================
    
    @classmethod
    def from_file(
        cls,
        model_path: Optional[Union[str, Path]] = None
    ) -> "MLPredictor":
        """
        Charge le predictor depuis un fichier.
        
        Args:
            model_path: Chemin du fichier .joblib
                       Si None, utilise le chemin par défaut
        
        Returns:
            Instance de MLPredictor
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le modèle est invalide
        """
        if model_path is None:
            model_path = Path(MODELS_DIR) / MODEL_FILENAME
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable: {model_path}")
        
        logger.info(f"📂 Chargement modèle: {model_path}")
        
        # Charger le modèle
        try:
            model = joblib.load(model_path)
        except Exception as e:
            raise ValueError(f"Erreur chargement modèle: {e}") from e
        
        # Vérifier que le modèle a les méthodes requises
        if not hasattr(model, "predict_proba"):
            raise ValueError("Le modèle doit avoir une méthode predict_proba()")
        
        # Charger les métadonnées
        meta_path = model_path.with_suffix(".json")
        metadata = ModelMetadata.from_json(meta_path)
        metadata.model_path = str(model_path)
        
        return cls(model, metadata)
    
    # =========================================
    # PROPERTIES
    # =========================================
    
    @property
    def model(self) -> Any:
        """Retourne le modèle sous-jacent."""
        return self._model
    
    @property
    def metadata(self) -> ModelMetadata:
        """Retourne les métadonnées du modèle."""
        return self._metadata
    
    @property
    def feature_names(self) -> List[str]:
        """Retourne la liste des noms de features."""
        return self._feature_names.copy()
    
    @property
    def n_features(self) -> int:
        """Retourne le nombre de features attendues."""
        return self._n_features
    
    @property
    def is_calibrated(self) -> bool:
        """Indique si le modèle est calibré."""
        return self._metadata.is_calibrated
    
    # =========================================
    # PRÉDICTION SINGLE
    # =========================================
    
    def predict(self, feature_set: FeatureSet) -> PredictionResult:
        """
        Fait une prédiction pour une paire.
        
        Args:
            feature_set: Features calculées pour la paire
            
        Returns:
            PredictionResult avec probabilité et confiance
        """
        # Préparer les features
        X = self._prepare_features(feature_set.features)
        
        # Prédiction
        proba = self._model.predict_proba(X)[0]
        
        # Le modèle retourne [proba_classe_0, proba_classe_1]
        prob_down = proba[0]  # Classe 0 = pas de hausse
        prob_up = proba[1]    # Classe 1 = hausse
        
        # Classe prédite (argmax)
        predicted_class = int(np.argmax(proba))
        
        # Calcul de la confiance
        confidence = self._calculate_confidence(prob_up)
        
        return PredictionResult(
            symbol=feature_set.symbol,
            probability_up=float(prob_up),
            probability_down=float(prob_down),
            predicted_class=predicted_class,
            confidence=confidence,
            timestamp=datetime.now(),
            features_used=self._n_features,
            model_version=self._get_model_version(),
        )
    
    def predict_from_dict(
        self,
        features: Dict[str, float],
        symbol: str = "UNKNOWN"
    ) -> PredictionResult:
        """
        Fait une prédiction depuis un dictionnaire de features.
        
        Args:
            features: Dictionnaire {nom_feature: valeur}
            symbol: Symbole de la paire
            
        Returns:
            PredictionResult
        """
        # Convertir en FeatureSet
        feature_set = FeatureSet(
            symbol=symbol,
            features=features,
            timestamp=pd.Timestamp.now()
        )
        return self.predict(feature_set)
    
    def predict_from_array(
        self,
        X: np.ndarray,
        symbol: str = "UNKNOWN"
    ) -> PredictionResult:
        """
        Fait une prédiction depuis un array numpy.
        
        Args:
            X: Array de shape (n_features,) ou (1, n_features)
            symbol: Symbole de la paire
            
        Returns:
            PredictionResult
        """
        # Reshape si nécessaire
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Vérifier la dimension
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Attendu {self._n_features} features, reçu {X.shape[1]}"
            )
        
        # Prédiction
        proba = self._model.predict_proba(X)[0]
        prob_down, prob_up = proba[0], proba[1]
        predicted_class = int(np.argmax(proba))
        confidence = self._calculate_confidence(prob_up)
        
        return PredictionResult(
            symbol=symbol,
            probability_up=float(prob_up),
            probability_down=float(prob_down),
            predicted_class=predicted_class,
            confidence=confidence,
            timestamp=datetime.now(),
            features_used=self._n_features,
            model_version=self._get_model_version(),
        )
    
    # =========================================
    # PRÉDICTION BATCH
    # =========================================
    
    def predict_batch(
        self,
        feature_sets: List[FeatureSet]
    ) -> List[PredictionResult]:
        """
        Fait des prédictions pour plusieurs paires.
        
        Plus efficace que d'appeler predict() en boucle car
        le modèle peut traiter un batch en une seule fois.
        
        Args:
            feature_sets: Liste de FeatureSet
            
        Returns:
            Liste de PredictionResult dans le même ordre
        """
        if not feature_sets:
            return []
        
        # Préparer la matrice de features
        symbols = []
        X_list = []
        
        for fs in feature_sets:
            symbols.append(fs.symbol)
            X_list.append(self._prepare_features(fs.features).flatten())
        
        X = np.array(X_list)
        
        # Prédiction batch
        probas = self._model.predict_proba(X)
        
        # Construire les résultats
        results = []
        timestamp = datetime.now()
        model_version = self._get_model_version()
        
        for i, (symbol, proba) in enumerate(zip(symbols, probas)):
            prob_down, prob_up = proba[0], proba[1]
            predicted_class = int(np.argmax(proba))
            confidence = self._calculate_confidence(prob_up)
            
            results.append(PredictionResult(
                symbol=symbol,
                probability_up=float(prob_up),
                probability_down=float(prob_down),
                predicted_class=predicted_class,
                confidence=confidence,
                timestamp=timestamp,
                features_used=self._n_features,
                model_version=model_version,
            ))
        
        return results
    
    def predict_batch_dataframe(
        self,
        df: pd.DataFrame,
        symbol_column: Optional[str] = None
    ) -> List[PredictionResult]:
        """
        Fait des prédictions depuis un DataFrame.
        
        Args:
            df: DataFrame avec les features en colonnes
            symbol_column: Nom de la colonne symbole (optionnel)
            
        Returns:
            Liste de PredictionResult
        """
        # Extraire les features
        feature_cols = [c for c in self._feature_names if c in df.columns]
        
        if len(feature_cols) != self._n_features:
            missing = set(self._feature_names) - set(feature_cols)
            raise ValueError(f"Features manquantes: {missing}")
        
        X = df[feature_cols].values
        
        # Symboles
        if symbol_column and symbol_column in df.columns:
            symbols = df[symbol_column].tolist()
        else:
            symbols = [f"ROW_{i}" for i in range(len(df))]
        
        # Prédictions
        probas = self._model.predict_proba(X)
        
        results = []
        timestamp = datetime.now()
        model_version = self._get_model_version()
        
        for symbol, proba in zip(symbols, probas):
            prob_down, prob_up = proba[0], proba[1]
            predicted_class = int(np.argmax(proba))
            confidence = self._calculate_confidence(prob_up)
            
            results.append(PredictionResult(
                symbol=symbol,
                probability_up=float(prob_up),
                probability_down=float(prob_down),
                predicted_class=predicted_class,
                confidence=confidence,
                timestamp=timestamp,
                features_used=self._n_features,
                model_version=model_version,
            ))
        
        return results
    
    # =========================================
    # MÉTHODES PRIVÉES
    # =========================================
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Prépare les features pour le modèle.
        
        Assure que les features sont dans le bon ordre et remplace
        les NaN par 0 (ou une valeur par défaut).
        
        Args:
            features: Dictionnaire des features
            
        Returns:
            Array numpy de shape (1, n_features)
        """
        # Extraire les valeurs dans l'ordre correct
        values = []
        for name in self._feature_names:
            value = features.get(name, np.nan)
            # Remplacer NaN par 0 (le modèle gère mal les NaN)
            if pd.isna(value):
                value = 0.0
            values.append(value)
        
        return np.array(values).reshape(1, -1)
    
    def _calculate_confidence(self, probability_up: float) -> float:
        """
        Calcule un score de confiance basé sur la probabilité.
        
        La confiance est maximale quand la proba est proche de 0 ou 1,
        et minimale quand elle est proche de 0.5 (incertitude).
        
        Formule: confidence = 2 * |probability - 0.5|
        
        Exemples:
        - probability = 0.5 → confidence = 0 (incertain)
        - probability = 0.7 → confidence = 0.4
        - probability = 0.8 → confidence = 0.6
        - probability = 0.9 → confidence = 0.8
        - probability = 1.0 → confidence = 1.0 (très confiant)
        
        Args:
            probability_up: Probabilité de hausse (0-1)
            
        Returns:
            Score de confiance (0-1)
        """
        return 2 * abs(probability_up - CONFIDENCE_NEUTRAL_POINT)
    
    def _get_model_version(self) -> str:
        """Retourne une version du modèle pour tracking."""
        if self._metadata.trained_at:
            return self._metadata.trained_at.strftime("%Y%m%d_%H%M%S")
        return "unknown"
    
    # =========================================
    # UTILITAIRES
    # =========================================
    
    def get_top_predictions(
        self,
        predictions: List[PredictionResult],
        n: int = 5,
        min_probability: float = DEFAULT_MIN_PROBABILITY,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE
    ) -> List[PredictionResult]:
        """
        Filtre et trie les meilleures prédictions.
        
        Args:
            predictions: Liste de prédictions
            n: Nombre max de résultats
            min_probability: Seuil de probabilité minimum
            min_confidence: Seuil de confiance minimum
            
        Returns:
            Top N prédictions triées par probabilité décroissante
        """
        # Filtrer
        filtered = [
            p for p in predictions
            if p.probability_up >= min_probability
            and p.confidence >= min_confidence
            and p.is_bullish
        ]
        
        # Trier par probabilité décroissante
        sorted_preds = sorted(
            filtered,
            key=lambda p: p.probability_up,
            reverse=True
        )
        
        return sorted_preds[:n]
    
    def summary(self) -> str:
        """Retourne un résumé du predictor."""
        return (
            f"🤖 MLPredictor\n"
            f"{'=' * 40}\n"
            f"📂 Modèle: {self._metadata.model_path}\n"
            f"📅 Entraîné: {self._metadata.trained_at}\n"
            f"🔢 Features: {self._n_features}\n"
            f"📐 Calibré: {'Oui' if self._metadata.is_calibrated else 'Non'}\n"
            f"📊 AUC validation: {self._metadata.val_auc:.4f}"
        )


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def load_predictor(model_path: Optional[Union[str, Path]] = None) -> MLPredictor:
    """
    Fonction utilitaire pour charger rapidement un predictor.
    
    Args:
        model_path: Chemin du modèle (défaut si None)
        
    Returns:
        MLPredictor chargé
    """
    return MLPredictor.from_file(model_path)


def predict_single(
    features: Dict[str, float],
    symbol: str,
    model_path: Optional[Union[str, Path]] = None
) -> PredictionResult:
    """
    Fonction utilitaire pour une prédiction rapide.
    
    Note: Charge le modèle à chaque appel, donc inefficace pour
    des prédictions répétées. Utiliser MLPredictor directement
    pour de meilleures performances.
    
    Args:
        features: Dictionnaire des features
        symbol: Symbole de la paire
        model_path: Chemin du modèle
        
    Returns:
        PredictionResult
    """
    predictor = load_predictor(model_path)
    return predictor.predict_from_dict(features, symbol)