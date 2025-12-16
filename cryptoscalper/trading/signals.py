# cryptoscalper/trading/signals.py
"""
Module de génération de signaux de trading.

Responsabilités :
- Filtrer les prédictions ML selon les seuils
- Générer des signaux de trading (TradeSignal)
- Classer et prioriser les opportunités
- Gérer l'historique des signaux

Usage:
    generator = SignalGenerator(config)
    
    # Générer des signaux depuis des prédictions
    predictions = predictor.predict_batch(feature_sets)
    signals = generator.generate_signals(predictions, current_positions=["BTCUSDT"])
    
    # Obtenir les meilleurs signaux
    top_signals = generator.get_top_signals(n=5)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
from collections import deque

import numpy as np

from cryptoscalper.config.constants import (
    SIGNAL_MIN_PROBABILITY,
    SIGNAL_MIN_CONFIDENCE,
    SIGNAL_MIN_PREDICTED_MOVE,
    SIGNAL_VALIDITY_SECONDS,
    MAX_SIGNALS_HISTORY,
    DEFAULT_STOP_LOSS_PERCENT,
    DEFAULT_TAKE_PROFIT_PERCENT,
)
from cryptoscalper.models.predictor import PredictionResult
from cryptoscalper.utils.logger import logger


# ============================================
# ENUMS
# ============================================

class SignalType(Enum):
    """Type de signal de trading."""
    
    BUY = "BUY"      # Signal d'achat
    SELL = "SELL"    # Signal de vente (pour fermer une position)
    HOLD = "HOLD"    # Garder la position


class SignalStrength(Enum):
    """Force du signal."""
    
    WEAK = "WEAK"        # Faible (proba 55-65%)
    MODERATE = "MODERATE"  # Modéré (proba 65-75%)
    STRONG = "STRONG"    # Fort (proba 75-85%)
    VERY_STRONG = "VERY_STRONG"  # Très fort (proba > 85%)
    
    @classmethod
    def from_probability(cls, probability: float) -> "SignalStrength":
        """Détermine la force du signal depuis la probabilité."""
        if probability >= 0.85:
            return cls.VERY_STRONG
        elif probability >= 0.75:
            return cls.STRONG
        elif probability >= 0.65:
            return cls.MODERATE
        else:
            return cls.WEAK


class SignalStatus(Enum):
    """Statut d'un signal."""
    
    PENDING = "PENDING"      # En attente d'exécution
    EXECUTED = "EXECUTED"    # Exécuté
    EXPIRED = "EXPIRED"      # Expiré
    CANCELLED = "CANCELLED"  # Annulé


# ============================================
# DATACLASSES
# ============================================

@dataclass
class SignalConfig:
    """
    Configuration pour la génération de signaux.
    
    Attributes:
        min_probability: Seuil minimum de probabilité (0.65 par défaut)
        min_confidence: Seuil minimum de confiance (0.55 par défaut)
        min_predicted_move: Mouvement minimum prédit (0.002 = 0.2%)
        validity_seconds: Durée de validité d'un signal en secondes
        default_stop_loss_pct: Stop-loss par défaut en pourcentage
        default_take_profit_pct: Take-profit par défaut en pourcentage
        max_signals_per_symbol: Nombre max de signaux actifs par symbole
    """
    
    min_probability: float = SIGNAL_MIN_PROBABILITY
    min_confidence: float = SIGNAL_MIN_CONFIDENCE
    min_predicted_move: float = SIGNAL_MIN_PREDICTED_MOVE
    validity_seconds: int = SIGNAL_VALIDITY_SECONDS
    default_stop_loss_pct: float = DEFAULT_STOP_LOSS_PERCENT
    default_take_profit_pct: float = DEFAULT_TAKE_PROFIT_PERCENT
    max_signals_per_symbol: int = 1
    
    def validate(self) -> None:
        """Valide la configuration."""
        if not 0.1 <= self.min_probability <= 1.0:
            raise ValueError("min_probability doit être entre 0.1 et 1.0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence doit être entre 0.0 et 1.0")
        if self.validity_seconds <= 0:
            raise ValueError("validity_seconds doit être positif")


@dataclass
class TradeSignal:
    """
    Signal de trading généré par le système.
    
    Un TradeSignal représente une opportunité de trading détectée
    par le modèle ML avec toutes les informations nécessaires
    pour prendre une décision.
    
    Attributes:
        symbol: Paire de trading (ex: "BTCUSDT")
        signal_type: Type de signal (BUY, SELL, HOLD)
        probability: Probabilité du modèle (0-1)
        confidence: Score de confiance (0-1)
        strength: Force du signal
        entry_price: Prix d'entrée suggéré (optionnel)
        stop_loss_price: Prix de stop-loss suggéré
        take_profit_price: Prix de take-profit suggéré
        stop_loss_pct: Stop-loss en pourcentage
        take_profit_pct: Take-profit en pourcentage
        score: Score global du signal (pour ranking)
        timestamp: Moment de génération du signal
        valid_until: Date d'expiration du signal
        status: Statut du signal
        metadata: Informations additionnelles
    """
    
    symbol: str
    signal_type: SignalType
    probability: float
    confidence: float
    strength: SignalStrength
    
    # Prix
    entry_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PERCENT
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PERCENT
    
    # Scoring
    score: float = 0.0
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    valid_until: datetime = field(default_factory=lambda: datetime.now() + timedelta(seconds=SIGNAL_VALIDITY_SECONDS))
    
    # Statut
    status: SignalStatus = SignalStatus.PENDING
    
    # Métadonnées
    metadata: Dict = field(default_factory=dict)
    
    # Identifiant unique
    signal_id: str = field(default_factory=lambda: f"SIG_{datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    
    @property
    def is_valid(self) -> bool:
        """Vérifie si le signal est encore valide."""
        return (
            self.status == SignalStatus.PENDING
            and datetime.now() < self.valid_until
        )
    
    @property
    def is_buy(self) -> bool:
        """Indique si c'est un signal d'achat."""
        return self.signal_type == SignalType.BUY
    
    @property
    def is_sell(self) -> bool:
        """Indique si c'est un signal de vente."""
        return self.signal_type == SignalType.SELL
    
    @property
    def time_remaining(self) -> timedelta:
        """Temps restant avant expiration."""
        return max(timedelta(0), self.valid_until - datetime.now())
    
    @property
    def risk_reward_ratio(self) -> float:
        """Ratio risque/récompense."""
        if self.stop_loss_pct == 0:
            return 0.0
        return self.take_profit_pct / self.stop_loss_pct
    
    def calculate_sl_tp_prices(self, current_price: float) -> None:
        """
        Calcule les prix de SL/TP depuis les pourcentages.
        
        Args:
            current_price: Prix actuel du marché
        """
        self.entry_price = current_price
        
        if self.signal_type == SignalType.BUY:
            # Long: SL en dessous, TP au dessus
            self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
            self.take_profit_price = current_price * (1 + self.take_profit_pct)
        else:
            # Short: SL au dessus, TP en dessous
            self.stop_loss_price = current_price * (1 + self.stop_loss_pct)
            self.take_profit_price = current_price * (1 - self.take_profit_pct)
    
    def expire(self) -> None:
        """Marque le signal comme expiré."""
        self.status = SignalStatus.EXPIRED
    
    def execute(self) -> None:
        """Marque le signal comme exécuté."""
        self.status = SignalStatus.EXECUTED
    
    def cancel(self) -> None:
        """Annule le signal."""
        self.status = SignalStatus.CANCELLED
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire."""
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "probability": self.probability,
            "confidence": self.confidence,
            "strength": self.strength.value,
            "score": self.score,
            "entry_price": self.entry_price,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "risk_reward_ratio": self.risk_reward_ratio,
            "timestamp": self.timestamp.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "status": self.status.value,
            "is_valid": self.is_valid,
        }
    
    def __str__(self) -> str:
        """Représentation string lisible."""
        status_emoji = {
            SignalStatus.PENDING: "⏳",
            SignalStatus.EXECUTED: "✅",
            SignalStatus.EXPIRED: "⏰",
            SignalStatus.CANCELLED: "❌",
        }
        strength_emoji = {
            SignalStrength.WEAK: "🟡",
            SignalStrength.MODERATE: "🟠",
            SignalStrength.STRONG: "🔴",
            SignalStrength.VERY_STRONG: "🔥",
        }
        
        return (
            f"{status_emoji[self.status]} {self.symbol} | "
            f"{self.signal_type.value} | "
            f"Proba: {self.probability:.1%} | "
            f"Score: {self.score:.2f} | "
            f"{strength_emoji[self.strength]} {self.strength.value}"
        )


# ============================================
# SIGNAL GENERATOR
# ============================================

class SignalGenerator:
    """
    Générateur de signaux de trading.
    
    Analyse les prédictions ML et génère des signaux de trading
    filtrés et classés selon les critères configurés.
    
    Workflow:
    1. Recevoir des prédictions ML (PredictionResult)
    2. Filtrer selon les seuils (proba, confiance)
    3. Exclure les symboles déjà en position
    4. Calculer un score pour chaque signal
    5. Retourner les meilleurs signaux triés
    
    Exemple:
        generator = SignalGenerator(config)
        
        # Depuis des prédictions
        signals = generator.generate_signals(
            predictions,
            current_positions=["BTCUSDT", "ETHUSDT"]
        )
        
        # Top 5 signaux
        top = generator.get_top_signals(n=5)
    """
    
    def __init__(self, config: Optional[SignalConfig] = None):
        """
        Initialise le générateur de signaux.
        
        Args:
            config: Configuration des seuils (défaut si None)
        """
        self._config = config or SignalConfig()
        self._config.validate()
        
        # Historique des signaux (pour éviter les doublons)
        self._signals_history: deque = deque(maxlen=MAX_SIGNALS_HISTORY)
        
        # Signaux actifs (en attente)
        self._active_signals: Dict[str, TradeSignal] = {}
        
        # Compteur de signaux générés
        self._signals_generated = 0
        
        logger.info(
            f"📡 SignalGenerator initialisé | "
            f"min_proba={self._config.min_probability:.0%} | "
            f"min_conf={self._config.min_confidence:.0%}"
        )
    
    # =========================================
    # PROPERTIES
    # =========================================
    
    @property
    def config(self) -> SignalConfig:
        """Retourne la configuration."""
        return self._config
    
    @property
    def active_signals_count(self) -> int:
        """Nombre de signaux actifs."""
        return len(self._active_signals)
    
    @property
    def signals_generated(self) -> int:
        """Nombre total de signaux générés."""
        return self._signals_generated
    
    # =========================================
    # GÉNÉRATION DE SIGNAUX
    # =========================================
    
    def generate_signals(
        self,
        predictions: List[PredictionResult],
        current_positions: Optional[List[str]] = None,
        current_prices: Optional[Dict[str, float]] = None
    ) -> List[TradeSignal]:
        """
        Génère des signaux de trading depuis des prédictions ML.
        
        Args:
            predictions: Liste de prédictions du modèle
            current_positions: Symboles déjà en position (à exclure)
            current_prices: Prix actuels pour calculer SL/TP
            
        Returns:
            Liste de TradeSignal filtrés et triés par score
        """
        if not predictions:
            return []
        
        current_positions = set(current_positions or [])
        signals = []
        
        for prediction in predictions:
            # Vérifier si on peut générer un signal
            if not self._should_generate_signal(prediction, current_positions):
                continue
            
            # Créer le signal
            signal = self._create_signal_from_prediction(prediction)
            
            # Calculer les prix SL/TP si disponibles
            if current_prices and prediction.symbol in current_prices:
                signal.calculate_sl_tp_prices(current_prices[prediction.symbol])
            
            # Ajouter à la liste
            signals.append(signal)
            
            # Enregistrer
            self._register_signal(signal)
        
        # Trier par score décroissant
        signals.sort(key=lambda s: s.score, reverse=True)
        
        if signals:
            logger.info(
                f"📡 {len(signals)} signaux générés | "
                f"Top: {signals[0].symbol} (score={signals[0].score:.2f})"
            )
        
        return signals
    
    def generate_signal(
        self,
        prediction: PredictionResult,
        current_price: Optional[float] = None
    ) -> Optional[TradeSignal]:
        """
        Génère un signal pour une seule prédiction.
        
        Args:
            prediction: Prédiction du modèle
            current_price: Prix actuel pour SL/TP
            
        Returns:
            TradeSignal ou None si les seuils ne sont pas atteints
        """
        if not self._passes_thresholds(prediction):
            return None
        
        signal = self._create_signal_from_prediction(prediction)
        
        if current_price:
            signal.calculate_sl_tp_prices(current_price)
        
        self._register_signal(signal)
        return signal
    
    # =========================================
    # FILTRAGE ET VALIDATION
    # =========================================
    
    def _should_generate_signal(
        self,
        prediction: PredictionResult,
        excluded_symbols: Set[str]
    ) -> bool:
        """
        Vérifie si on doit générer un signal pour cette prédiction.
        
        Args:
            prediction: Prédiction à évaluer
            excluded_symbols: Symboles à exclure
            
        Returns:
            True si on peut générer un signal
        """
        # Exclure si déjà en position
        if prediction.symbol in excluded_symbols:
            return False
        
        # Exclure si un signal actif existe déjà pour ce symbole
        if prediction.symbol in self._active_signals:
            existing = self._active_signals[prediction.symbol]
            if existing.is_valid:
                return False
        
        # Vérifier les seuils
        return self._passes_thresholds(prediction)
    
    def _passes_thresholds(self, prediction: PredictionResult) -> bool:
        """
        Vérifie si une prédiction passe les seuils.
        
        Args:
            prediction: Prédiction à évaluer
            
        Returns:
            True si les seuils sont respectés
        """
        # Doit être un signal haussier
        if not prediction.is_bullish:
            return False
        
        # Probabilité minimum
        if prediction.probability_up < self._config.min_probability:
            return False
        
        # Confiance minimum
        if prediction.confidence < self._config.min_confidence:
            return False
        
        return True
    
    # =========================================
    # CRÉATION DE SIGNAL
    # =========================================
    
    def _create_signal_from_prediction(
        self,
        prediction: PredictionResult
    ) -> TradeSignal:
        """
        Crée un TradeSignal depuis une prédiction.
        
        Args:
            prediction: Prédiction du modèle
            
        Returns:
            TradeSignal configuré
        """
        # Déterminer la force du signal
        strength = SignalStrength.from_probability(prediction.probability_up)
        
        # Calculer le score global
        score = self._calculate_score(prediction)
        
        # Créer le signal
        signal = TradeSignal(
            symbol=prediction.symbol,
            signal_type=SignalType.BUY,  # Pour l'instant, que des BUY
            probability=prediction.probability_up,
            confidence=prediction.confidence,
            strength=strength,
            score=score,
            stop_loss_pct=self._config.default_stop_loss_pct,
            take_profit_pct=self._config.default_take_profit_pct,
            valid_until=datetime.now() + timedelta(seconds=self._config.validity_seconds),
            metadata={
                "model_version": prediction.model_version,
                "features_used": prediction.features_used,
            }
        )
        
        return signal
    
    def _calculate_score(self, prediction: PredictionResult) -> float:
        """
        Calcule un score global pour le ranking.
        
        Le score combine:
        - Probabilité (poids: 60%)
        - Confiance (poids: 40%)
        
        Args:
            prediction: Prédiction à scorer
            
        Returns:
            Score entre 0 et 1
        """
        # Pondération
        prob_weight = 0.6
        conf_weight = 0.4
        
        # Score pondéré
        score = (
            prob_weight * prediction.probability_up +
            conf_weight * prediction.confidence
        )
        
        return score
    
    # =========================================
    # GESTION DES SIGNAUX
    # =========================================
    
    def _register_signal(self, signal: TradeSignal) -> None:
        """Enregistre un nouveau signal."""
        self._signals_history.append(signal)
        self._active_signals[signal.symbol] = signal
        self._signals_generated += 1
    
    def get_active_signals(self) -> List[TradeSignal]:
        """
        Retourne tous les signaux actifs (valides et en attente).
        
        Returns:
            Liste de signaux actifs
        """
        # Nettoyer les signaux expirés
        self._cleanup_expired_signals()
        
        # Retourner les signaux actifs triés
        signals = [
            s for s in self._active_signals.values()
            if s.is_valid
        ]
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals
    
    def get_top_signals(
        self,
        n: int = 5,
        min_score: float = 0.0
    ) -> List[TradeSignal]:
        """
        Retourne les N meilleurs signaux actifs.
        
        Args:
            n: Nombre max de signaux
            min_score: Score minimum
            
        Returns:
            Top N signaux triés par score
        """
        signals = self.get_active_signals()
        
        # Filtrer par score minimum
        if min_score > 0:
            signals = [s for s in signals if s.score >= min_score]
        
        return signals[:n]
    
    def get_signal_by_symbol(self, symbol: str) -> Optional[TradeSignal]:
        """
        Retourne le signal actif pour un symbole.
        
        Args:
            symbol: Symbole à chercher
            
        Returns:
            TradeSignal ou None
        """
        signal = self._active_signals.get(symbol)
        if signal and signal.is_valid:
            return signal
        return None
    
    def mark_signal_executed(self, symbol: str) -> Optional[TradeSignal]:
        """
        Marque un signal comme exécuté.
        
        Args:
            symbol: Symbole du signal
            
        Returns:
            Signal mis à jour ou None
        """
        signal = self._active_signals.get(symbol)
        if signal:
            signal.execute()
            logger.info(f"✅ Signal exécuté: {symbol}")
            return signal
        return None
    
    def cancel_signal(self, symbol: str) -> Optional[TradeSignal]:
        """
        Annule un signal.
        
        Args:
            symbol: Symbole du signal
            
        Returns:
            Signal annulé ou None
        """
        signal = self._active_signals.get(symbol)
        if signal and signal.is_valid:
            signal.cancel()
            logger.info(f"❌ Signal annulé: {symbol}")
            return signal
        return None
    
    def _cleanup_expired_signals(self) -> int:
        """
        Nettoie les signaux expirés.
        
        Returns:
            Nombre de signaux nettoyés
        """
        expired = []
        
        for symbol, signal in self._active_signals.items():
            if not signal.is_valid and signal.status == SignalStatus.PENDING:
                signal.expire()
                expired.append(symbol)
        
        for symbol in expired:
            del self._active_signals[symbol]
        
        if expired:
            logger.debug(f"🧹 {len(expired)} signaux expirés nettoyés")
        
        return len(expired)
    
    def clear_all_signals(self) -> int:
        """
        Supprime tous les signaux actifs.
        
        Returns:
            Nombre de signaux supprimés
        """
        count = len(self._active_signals)
        self._active_signals.clear()
        logger.info(f"🗑️ {count} signaux supprimés")
        return count
    
    # =========================================
    # STATISTIQUES
    # =========================================
    
    def get_statistics(self) -> Dict:
        """
        Retourne des statistiques sur les signaux.
        
        Returns:
            Dictionnaire de statistiques
        """
        history_list = list(self._signals_history)
        
        if not history_list:
            return {
                "total_generated": 0,
                "active_count": 0,
                "executed_count": 0,
                "expired_count": 0,
                "avg_probability": 0,
                "avg_score": 0,
            }
        
        executed = [s for s in history_list if s.status == SignalStatus.EXECUTED]
        expired = [s for s in history_list if s.status == SignalStatus.EXPIRED]
        
        return {
            "total_generated": self._signals_generated,
            "active_count": len(self.get_active_signals()),
            "executed_count": len(executed),
            "expired_count": len(expired),
            "cancelled_count": len([s for s in history_list if s.status == SignalStatus.CANCELLED]),
            "avg_probability": np.mean([s.probability for s in history_list]),
            "avg_confidence": np.mean([s.confidence for s in history_list]),
            "avg_score": np.mean([s.score for s in history_list]),
        }
    
    def summary(self) -> str:
        """Retourne un résumé du générateur."""
        stats = self.get_statistics()
        return (
            f"📡 SignalGenerator\n"
            f"{'=' * 40}\n"
            f"⚙️  Config:\n"
            f"   - Min proba: {self._config.min_probability:.0%}\n"
            f"   - Min conf: {self._config.min_confidence:.0%}\n"
            f"   - Validité: {self._config.validity_seconds}s\n"
            f"   - SL: {self._config.default_stop_loss_pct:.2%}\n"
            f"   - TP: {self._config.default_take_profit_pct:.2%}\n"
            f"📊 Stats:\n"
            f"   - Générés: {stats['total_generated']}\n"
            f"   - Actifs: {stats['active_count']}\n"
            f"   - Exécutés: {stats['executed_count']}\n"
            f"   - Expirés: {stats['expired_count']}\n"
            f"   - Score moyen: {stats['avg_score']:.2f}"
        )


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def create_signal_generator(
    min_probability: float = SIGNAL_MIN_PROBABILITY,
    min_confidence: float = SIGNAL_MIN_CONFIDENCE,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PERCENT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PERCENT
) -> SignalGenerator:
    """
    Fonction utilitaire pour créer un générateur de signaux.
    
    Args:
        min_probability: Seuil de probabilité
        min_confidence: Seuil de confiance
        stop_loss_pct: Stop-loss en pourcentage
        take_profit_pct: Take-profit en pourcentage
        
    Returns:
        SignalGenerator configuré
    """
    config = SignalConfig(
        min_probability=min_probability,
        min_confidence=min_confidence,
        default_stop_loss_pct=stop_loss_pct,
        default_take_profit_pct=take_profit_pct
    )
    return SignalGenerator(config)


def filter_signals_by_strength(
    signals: List[TradeSignal],
    min_strength: SignalStrength = SignalStrength.MODERATE
) -> List[TradeSignal]:
    """
    Filtre les signaux par force minimale.
    
    Args:
        signals: Liste de signaux
        min_strength: Force minimale requise
        
    Returns:
        Signaux filtrés
    """
    strength_order = [
        SignalStrength.WEAK,
        SignalStrength.MODERATE,
        SignalStrength.STRONG,
        SignalStrength.VERY_STRONG
    ]
    
    min_index = strength_order.index(min_strength)
    
    return [
        s for s in signals
        if strength_order.index(s.strength) >= min_index
    ]