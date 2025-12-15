# cryptoscalper/config/settings.py
"""
Module de configuration du bot CryptoScalper AI.

Utilise Pydantic pour la validation automatique des paramètres.
Les valeurs sont chargées depuis les variables d'environnement (.env).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional
from pathlib import Path


class BinanceSettings(BaseSettings):
    """Configuration de connexion à Binance."""
    
    model_config = SettingsConfigDict(
        env_prefix="BINANCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    api_key: str = Field(
        default="",
        description="Clé API Binance"
    )
    api_secret: str = Field(
        default="",
        description="Secret API Binance"
    )
    testnet: bool = Field(
        default=True,
        description="Utiliser le testnet (True) ou la production (False)"
    )
    
    @property
    def base_url(self) -> str:
        """Retourne l'URL de base selon le mode testnet/production."""
        if self.testnet:
            return "https://testnet.binance.vision"
        return "https://api.binance.com"
    
    @property
    def websocket_url(self) -> str:
        """Retourne l'URL WebSocket selon le mode testnet/production."""
        if self.testnet:
            return "wss://testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"


class TradingSettings(BaseSettings):
    """Configuration des paramètres de trading."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    initial_capital: float = Field(
        default=30.0,
        ge=10.0,
        description="Capital initial en USDT"
    )
    max_risk_per_trade: float = Field(
        default=0.02,
        ge=0.001,
        le=0.10,
        description="Risque max par trade (0.02 = 2%)"
    )
    max_total_exposure: float = Field(
        default=0.10,
        ge=0.01,
        le=0.50,
        description="Exposition totale max (0.10 = 10%)"
    )
    max_concurrent_trades: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Nombre max de trades simultanés"
    )
    
    @field_validator("max_risk_per_trade")
    @classmethod
    def validate_risk(cls, v: float) -> float:
        """Valide que le risque n'est pas trop élevé."""
        if v > 0.05:
            raise ValueError("Risque par trade trop élevé (max 5%)")
        return v


class ScannerSettings(BaseSettings):
    """Configuration du scanner de paires."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    min_volume_24h_usdt: float = Field(
        default=1_000_000,
        ge=100_000,
        alias="MIN_VOLUME_24H_USDT",
        description="Volume minimum 24h en USDT"
    )
    max_spread_percent: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.01,
        alias="MAX_SPREAD_PERCENT",
        description="Spread maximum acceptable"
    )
    scan_interval_seconds: int = Field(
        default=5,
        ge=1,
        le=60,
        alias="SCAN_INTERVAL_SECONDS",
        description="Intervalle entre les scans"
    )
    max_pairs: int = Field(
        default=150,
        ge=10,
        le=500,
        description="Nombre maximum de paires à surveiller"
    )


class MLSettings(BaseSettings):
    """Configuration du modèle de Machine Learning."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    prediction_threshold: float = Field(
        default=0.65,
        ge=0.5,
        le=0.95,
        alias="PREDICTION_THRESHOLD",
        description="Seuil de probabilité pour trader"
    )
    target_profit_percent: float = Field(
        default=0.003,
        ge=0.001,
        le=0.02,
        alias="TARGET_PROFIT_PERCENT",
        description="Objectif de profit (0.003 = 0.3%)"
    )
    stop_loss_percent: float = Field(
        default=0.005,
        ge=0.001,
        le=0.02,
        alias="STOP_LOSS_PERCENT",
        description="Stop-loss (0.005 = 0.5%)"
    )
    model_path: Path = Field(
        default=Path("models/saved/xgb_model_latest.joblib"),
        description="Chemin vers le modèle sauvegardé"
    )


class LoggingSettings(BaseSettings):
    """Configuration du logging."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Niveau de log (DEBUG, INFO, WARNING, ERROR)"
    )
    file_path: Path = Field(
        default=Path("logs/cryptoscalper.log"),
        alias="LOG_FILE_PATH",
        description="Chemin du fichier de log"
    )
    rotation: str = Field(
        default="10 MB",
        description="Taille max avant rotation"
    )
    retention: str = Field(
        default="7 days",
        description="Durée de conservation des logs"
    )
    
    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Valide que le niveau de log est valide."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Niveau invalide. Choix: {valid_levels}")
        return v_upper


class Settings(BaseSettings):
    """
    Configuration principale du bot CryptoScalper AI.
    
    Regroupe toutes les sous-configurations.
    
    Usage:
        settings = Settings()
        print(settings.binance.api_key)
        print(settings.trading.initial_capital)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Sous-configurations
    binance: BinanceSettings = Field(default_factory=BinanceSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    scanner: ScannerSettings = Field(default_factory=ScannerSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    def is_production(self) -> bool:
        """Vérifie si on est en mode production (argent réel)."""
        return not self.binance.testnet
    
    def get_risk_amount(self) -> float:
        """Calcule le montant risqué par trade en USDT."""
        return self.trading.initial_capital * self.trading.max_risk_per_trade
    
    def display_config(self) -> str:
        """Affiche un résumé de la configuration (sans secrets)."""
        mode = "🧪 TESTNET" if self.binance.testnet else "🔴 PRODUCTION"
        return f"""
╔══════════════════════════════════════════╗
║       CryptoScalper AI - Config          ║
╠══════════════════════════════════════════╣
║ Mode: {mode:<33}║
║ Capital: {self.trading.initial_capital:.2f} USDT{' ' * 24}║
║ Risque/trade: {self.trading.max_risk_per_trade:.1%}{' ' * 23}║
║ Stop-loss: {self.ml.stop_loss_percent:.2%}{' ' * 24}║
║ Take-profit: {self.ml.target_profit_percent:.2%}{' ' * 22}║
║ Log level: {self.logging.level:<28}║
╚══════════════════════════════════════════╝
"""


# Instance globale (singleton pattern)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Récupère l'instance unique des settings.
    
    Utilise le pattern singleton pour éviter de recharger
    le fichier .env à chaque appel.
    
    Returns:
        Instance de Settings
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Force le rechargement des settings depuis le fichier .env.
    
    Utile après modification du fichier .env en runtime.
    
    Returns:
        Nouvelle instance de Settings
    """
    global _settings
    _settings = Settings()
    return _settings