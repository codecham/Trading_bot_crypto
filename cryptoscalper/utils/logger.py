# cryptoscalper/utils/logger.py
"""
Module de logging pour CryptoScalper AI.

Utilise loguru pour un logging simple et puissant.
Configure automatiquement la sortie console et fichier.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# Supprimer le handler par défaut de loguru
logger.remove()


def setup_logger(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "7 days"
) -> None:
    """
    Configure le logger pour l'application.
    
    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Chemin du fichier de log (None = pas de fichier)
        rotation: Taille max avant rotation du fichier
        retention: Durée de conservation des fichiers de log
        
    Example:
        >>> setup_logger(level="DEBUG", log_file=Path("logs/bot.log"))
        >>> logger.info("Bot démarré!")
    """
    # Format pour la console (coloré)
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Format pour le fichier (plus détaillé)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    # Handler console
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True
    )
    
    # Handler fichier (si spécifié)
    if log_file is not None:
        # Créer le dossier si nécessaire
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_file),
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            encoding="utf-8"
        )


def setup_logger_from_settings() -> None:
    """
    Configure le logger depuis les settings de l'application.
    
    Charge automatiquement la configuration depuis le fichier .env.
    
    Example:
        >>> setup_logger_from_settings()
        >>> logger.info("Configuration chargée!")
    """
    # Import local pour éviter les imports circulaires
    from cryptoscalper.config.settings import get_settings
    
    settings = get_settings()
    
    setup_logger(
        level=settings.logging.level,
        log_file=settings.logging.file_path,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention
    )


def get_trade_logger() -> "logger":
    """
    Retourne un logger spécialisé pour les trades.
    
    Crée un fichier séparé pour l'historique des trades.
    
    Returns:
        Logger configuré pour les trades
    """
    trade_log_path = Path("logs/trades.log")
    trade_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    trade_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{message}"
    )
    
    # Créer un logger dédié aux trades
    trade_logger = logger.bind(trade=True)
    
    # Ajouter le handler fichier pour les trades
    logger.add(
        str(trade_log_path),
        format=trade_format,
        filter=lambda record: record["extra"].get("trade", False),
        rotation="1 day",
        retention="30 days",
        encoding="utf-8"
    )
    
    return trade_logger


# === Fonctions utilitaires de logging ===

def log_trade_signal(
    symbol: str,
    action: str,
    price: float,
    confidence: float
) -> None:
    """
    Log un signal de trading.
    
    Args:
        symbol: Paire tradée (ex: BTCUSDT)
        action: Action (BUY, SELL, HOLD)
        price: Prix au moment du signal
        confidence: Score de confiance (0-1)
    """
    emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
    logger.info(
        f"{emoji} SIGNAL | {symbol} | {action} @ {price:.2f} | "
        f"Confiance: {confidence:.1%}"
    )


def log_trade_executed(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    order_id: str
) -> None:
    """
    Log l'exécution d'un trade.
    
    Args:
        symbol: Paire tradée
        side: Côté (BUY, SELL)
        quantity: Quantité exécutée
        price: Prix d'exécution
        order_id: ID de l'ordre
    """
    emoji = "✅" if side == "BUY" else "💰"
    logger.info(
        f"{emoji} TRADE | {symbol} | {side} | "
        f"Qty: {quantity:.6f} @ {price:.2f} | "
        f"Order: {order_id}"
    )


def log_trade_result(
    symbol: str,
    pnl_usdt: float,
    pnl_percent: float,
    duration_seconds: int
) -> None:
    """
    Log le résultat d'un trade fermé.
    
    Args:
        symbol: Paire tradée
        pnl_usdt: Profit/perte en USDT
        pnl_percent: Profit/perte en pourcentage
        duration_seconds: Durée du trade en secondes
    """
    emoji = "🎉" if pnl_usdt > 0 else "😤"
    sign = "+" if pnl_usdt > 0 else ""
    
    logger.info(
        f"{emoji} RESULT | {symbol} | "
        f"PnL: {sign}{pnl_usdt:.4f} USDT ({sign}{pnl_percent:.2%}) | "
        f"Durée: {duration_seconds}s"
    )


def log_error_with_context(
    error: Exception,
    context: dict
) -> None:
    """
    Log une erreur avec son contexte.
    
    Args:
        error: L'exception levée
        context: Dictionnaire de contexte (symbol, action, etc.)
    """
    context_str = " | ".join(f"{k}={v}" for k, v in context.items())
    logger.error(f"❌ ERROR | {type(error).__name__}: {error} | {context_str}")


def log_bot_status(
    status: str,
    capital: float,
    open_positions: int,
    daily_pnl: float
) -> None:
    """
    Log le statut périodique du bot.
    
    Args:
        status: État du bot (RUNNING, PAUSED, STOPPED)
        capital: Capital actuel en USDT
        open_positions: Nombre de positions ouvertes
        daily_pnl: PnL du jour en USDT
    """
    emoji = "🤖" if status == "RUNNING" else "⏸️" if status == "PAUSED" else "🛑"
    sign = "+" if daily_pnl > 0 else ""
    
    logger.info(
        f"{emoji} STATUS | {status} | "
        f"Capital: {capital:.2f} USDT | "
        f"Positions: {open_positions} | "
        f"PnL jour: {sign}{daily_pnl:.2f} USDT"
    )


# Exporter le logger principal
__all__ = [
    "logger",
    "setup_logger",
    "setup_logger_from_settings",
    "get_trade_logger",
    "log_trade_signal",
    "log_trade_executed",
    "log_trade_result",
    "log_error_with_context",
    "log_bot_status"
]