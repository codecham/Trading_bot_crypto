# cryptoscalper/utils/exceptions.py
"""
Exceptions personnalisées pour CryptoScalper AI.

Hiérarchie des exceptions :
- CryptoScalperError (base)
  ├── ConfigurationError
  ├── ConnectionError
  │   ├── APIConnectionError
  │   └── WebSocketConnectionError
  ├── TradingError
  │   ├── InsufficientBalanceError
  │   ├── OrderExecutionError
  │   ├── InvalidOrderError
  │   └── PositionError
  ├── DataError
  │   ├── InvalidSymbolError
  │   ├── DataFetchError
  │   └── FeatureCalculationError
  ├── MLError
  │   ├── ModelNotFoundError
  │   ├── PredictionError
  │   └── TrainingError
  └── RiskError
      ├── RiskLimitExceededError
      └── KillSwitchActivatedError
"""


class CryptoScalperError(Exception):
    """
    Exception de base pour toutes les erreurs du projet.
    
    Toutes les exceptions personnalisées héritent de celle-ci.
    """
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# ============================================
# CONFIGURATION ERRORS
# ============================================

class ConfigurationError(CryptoScalperError):
    """Erreur de configuration (fichier .env, settings, etc.)."""
    pass


# ============================================
# CONNECTION ERRORS
# ============================================

class ConnectionError(CryptoScalperError):
    """Erreur de connexion générique."""
    pass


class APIConnectionError(ConnectionError):
    """Erreur de connexion à l'API REST."""
    
    def __init__(self, message: str, status_code: int = None, **kwargs):
        details = kwargs
        if status_code:
            details["status_code"] = status_code
        super().__init__(message, details)


class WebSocketConnectionError(ConnectionError):
    """Erreur de connexion WebSocket."""
    pass


# ============================================
# TRADING ERRORS
# ============================================

class TradingError(CryptoScalperError):
    """Erreur liée au trading."""
    pass


class InsufficientBalanceError(TradingError):
    """Balance insuffisante pour exécuter le trade."""
    
    def __init__(self, required: float, available: float, asset: str = "USDT"):
        message = f"Balance insuffisante: {available:.4f} {asset} disponible, {required:.4f} requis"
        super().__init__(message, {"required": required, "available": available, "asset": asset})


class OrderExecutionError(TradingError):
    """Erreur lors de l'exécution d'un ordre."""
    
    def __init__(self, message: str, order_id: str = None, symbol: str = None):
        details = {}
        if order_id:
            details["order_id"] = order_id
        if symbol:
            details["symbol"] = symbol
        super().__init__(message, details)


class InvalidOrderError(TradingError):
    """Ordre invalide (paramètres incorrects)."""
    pass


class PositionError(TradingError):
    """Erreur liée à une position (ouverture, fermeture, suivi)."""
    pass


# ============================================
# DATA ERRORS
# ============================================

class DataError(CryptoScalperError):
    """Erreur liée aux données."""
    pass


class InvalidSymbolError(DataError):
    """Symbole de paire invalide ou non supporté."""
    
    def __init__(self, symbol: str):
        super().__init__(f"Symbole invalide ou non trouvé: {symbol}", {"symbol": symbol})


class DataFetchError(DataError):
    """Erreur lors de la récupération des données."""
    
    def __init__(self, message: str, symbol: str = None, endpoint: str = None):
        details = {}
        if symbol:
            details["symbol"] = symbol
        if endpoint:
            details["endpoint"] = endpoint
        super().__init__(message, details)


class FeatureCalculationError(DataError):
    """Erreur lors du calcul des features."""
    
    def __init__(self, message: str, feature_name: str = None):
        details = {}
        if feature_name:
            details["feature"] = feature_name
        super().__init__(message, details)


# ============================================
# ML ERRORS
# ============================================

class MLError(CryptoScalperError):
    """Erreur liée au Machine Learning."""
    pass


class ModelNotFoundError(MLError):
    """Modèle ML non trouvé sur le disque."""
    
    def __init__(self, model_path: str):
        super().__init__(f"Modèle non trouvé: {model_path}", {"path": model_path})


class PredictionError(MLError):
    """Erreur lors de la prédiction."""
    pass


class TrainingError(MLError):
    """Erreur lors de l'entraînement du modèle."""
    pass


# ============================================
# RISK ERRORS
# ============================================

class RiskError(CryptoScalperError):
    """Erreur liée à la gestion du risque."""
    pass


class RiskLimitExceededError(RiskError):
    """Limite de risque dépassée."""
    
    def __init__(self, limit_type: str, current_value: float, max_value: float):
        message = f"Limite de risque dépassée: {limit_type}"
        super().__init__(message, {
            "limit_type": limit_type,
            "current": current_value,
            "max": max_value
        })


class KillSwitchActivatedError(RiskError):
    """Kill switch activé (drawdown max atteint)."""
    
    def __init__(self, drawdown_percent: float, max_drawdown: float):
        message = "KILL SWITCH ACTIVÉ - Drawdown maximum atteint"
        super().__init__(message, {
            "drawdown": f"{drawdown_percent:.2%}",
            "max_drawdown": f"{max_drawdown:.2%}"
        })