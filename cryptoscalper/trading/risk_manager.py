# cryptoscalper/trading/risk_manager.py
"""
Module de gestion du risque.

Responsabilités :
- Contrôler la taille des positions (position sizing)
- Suivre les pertes journalières
- Activer le kill switch si nécessaire
- Approuver ou rejeter les trades

Usage:
    from cryptoscalper.trading.risk_manager import RiskManager, RiskConfig
    
    # Configuration
    config = RiskConfig(initial_capital=30.0, max_position_pct=0.20)
    risk_manager = RiskManager(config)
    
    # Vérifier si on peut trader
    can_trade, reason = risk_manager.can_open_trade(signal)
    
    # Calculer la taille de position
    size = risk_manager.calculate_position_size(signal, current_capital=25.0)
    
    # Enregistrer le résultat d'un trade
    risk_manager.register_trade_result(completed_trade)
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque

from cryptoscalper.config.constants import (
    MAX_DAILY_LOSS_PERCENT,
    MAX_DRAWDOWN_PERCENT,
    MAX_TRADES_PER_HOUR,
    MAX_TRADES_PER_DAY,
    MAX_CONSECUTIVE_LOSSES,
    DEFAULT_POSITION_SIZE_PERCENT,
    MIN_POSITION_SIZE_USDT,
    DEFAULT_STOP_LOSS_PERCENT,
    DEFAULT_TAKE_PROFIT_PERCENT,
)
from cryptoscalper.utils.exceptions import (
    RiskError,
    RiskLimitExceededError,
    KillSwitchActivatedError,
)
from cryptoscalper.utils.logger import logger


# ============================================
# ENUMS
# ============================================

class RejectionReason(Enum):
    """Raisons de rejet d'un trade."""
    
    KILL_SWITCH_ACTIVE = "Kill switch activé (drawdown max atteint)"
    DAILY_LOSS_EXCEEDED = "Perte journalière maximum atteinte"
    MAX_TRADES_PER_HOUR = "Nombre maximum de trades par heure atteint"
    MAX_TRADES_PER_DAY = "Nombre maximum de trades par jour atteint"
    MAX_CONSECUTIVE_LOSSES = "Nombre maximum de pertes consécutives atteint"
    MAX_POSITIONS_REACHED = "Nombre maximum de positions ouvertes atteint"
    POSITION_TOO_SMALL = "Taille de position inférieure au minimum"
    INSUFFICIENT_CAPITAL = "Capital insuffisant"
    COOLDOWN_ACTIVE = "Période de cooldown active"


class TradeOutcome(Enum):
    """Résultat d'un trade."""
    
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"


# ============================================
# DATACLASSES
# ============================================

@dataclass
class RiskConfig:
    """
    Configuration des paramètres de risque.
    
    Cette dataclass contient tous les paramètres qui contrôlent
    le comportement du Risk Manager.
    
    Attributes:
        initial_capital: Capital initial en USDT
        max_position_pct: Pourcentage max du capital par trade (0.20 = 20%)
        min_position_usdt: Taille minimum de position en USDT
        max_loss_per_trade_pct: Perte max par trade en % du capital
        max_daily_loss_pct: Perte journalière max en % du capital
        max_drawdown_pct: Drawdown max avant kill switch (% du capital initial)
        max_open_positions: Nombre max de positions simultanées
        max_trades_per_hour: Nombre max de trades par heure
        max_trades_per_day: Nombre max de trades par jour
        max_consecutive_losses: Nombre max de pertes consécutives
        max_position_duration_sec: Durée max d'une position en secondes
        default_stop_loss_pct: Stop-loss par défaut
        default_take_profit_pct: Take-profit par défaut
        cooldown_after_loss_sec: Cooldown après une perte (secondes)
        cooldown_after_consecutive_losses_sec: Cooldown après pertes consécutives
    """
    
    # Capital
    initial_capital: float = 30.0
    
    # Position sizing
    max_position_pct: float = DEFAULT_POSITION_SIZE_PERCENT  # 20%
    min_position_usdt: float = MIN_POSITION_SIZE_USDT  # 10 USDT
    
    # Limites de perte
    max_loss_per_trade_pct: float = 0.02  # 2% du capital
    max_daily_loss_pct: float = MAX_DAILY_LOSS_PERCENT  # 10%
    max_drawdown_pct: float = MAX_DRAWDOWN_PERCENT  # 25%
    
    # Limites de trades
    max_open_positions: int = 1  # Une seule position (capital limité)
    max_trades_per_hour: int = MAX_TRADES_PER_HOUR  # 20
    max_trades_per_day: int = MAX_TRADES_PER_DAY  # 100
    max_consecutive_losses: int = MAX_CONSECUTIVE_LOSSES  # 5
    
    # Timeout
    max_position_duration_sec: int = 300  # 5 minutes
    
    # SL/TP par défaut
    default_stop_loss_pct: float = DEFAULT_STOP_LOSS_PERCENT  # 0.4%
    default_take_profit_pct: float = DEFAULT_TAKE_PROFIT_PERCENT  # 0.3%
    
    # Cooldown
    cooldown_after_loss_sec: int = 0  # Pas de cooldown par défaut
    cooldown_after_consecutive_losses_sec: int = 60  # 1 min après pertes consécutives
    
    def validate(self) -> None:
        """
        Valide la configuration.
        
        Raises:
            ValueError: Si un paramètre est invalide
        """
        if self.initial_capital <= 0:
            raise ValueError("initial_capital doit être positif")
        if not 0 < self.max_position_pct <= 1.0:
            raise ValueError("max_position_pct doit être entre 0 et 1")
        if self.min_position_usdt <= 0:
            raise ValueError("min_position_usdt doit être positif")
        if not 0 < self.max_daily_loss_pct <= 1.0:
            raise ValueError("max_daily_loss_pct doit être entre 0 et 1")
        if not 0 < self.max_drawdown_pct <= 1.0:
            raise ValueError("max_drawdown_pct doit être entre 0 et 1")
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions doit être >= 1")
        if self.max_trades_per_day < 1:
            raise ValueError("max_trades_per_day doit être >= 1")
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire."""
        return {
            "initial_capital": self.initial_capital,
            "max_position_pct": self.max_position_pct,
            "min_position_usdt": self.min_position_usdt,
            "max_loss_per_trade_pct": self.max_loss_per_trade_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_open_positions": self.max_open_positions,
            "max_trades_per_hour": self.max_trades_per_hour,
            "max_trades_per_day": self.max_trades_per_day,
            "max_consecutive_losses": self.max_consecutive_losses,
            "max_position_duration_sec": self.max_position_duration_sec,
            "default_stop_loss_pct": self.default_stop_loss_pct,
            "default_take_profit_pct": self.default_take_profit_pct,
        }


@dataclass
class CompletedTrade:
    """
    Représente un trade terminé.
    
    Cette dataclass est utilisée pour enregistrer les résultats
    des trades et mettre à jour les statistiques de risque.
    
    Attributes:
        symbol: Symbole tradé (ex: "BTCUSDT")
        side: Côté du trade ("BUY" ou "SELL")
        entry_price: Prix d'entrée
        exit_price: Prix de sortie
        quantity: Quantité tradée
        entry_time: Timestamp d'entrée
        exit_time: Timestamp de sortie
        pnl_usdt: Profit/perte en USDT
        pnl_percent: Profit/perte en pourcentage
        fees_usdt: Frais payés en USDT
        exit_reason: Raison de sortie ("TP", "SL", "TIMEOUT", "MANUAL")
    """
    
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl_usdt: float
    pnl_percent: float
    fees_usdt: float = 0.0
    exit_reason: str = "UNKNOWN"
    
    @property
    def outcome(self) -> TradeOutcome:
        """Détermine le résultat du trade."""
        if self.pnl_usdt > 0.0001:  # Petit seuil pour éviter les erreurs de float
            return TradeOutcome.WIN
        elif self.pnl_usdt < -0.0001:
            return TradeOutcome.LOSS
        return TradeOutcome.BREAKEVEN
    
    @property
    def duration_seconds(self) -> int:
        """Durée du trade en secondes."""
        return int((self.exit_time - self.entry_time).total_seconds())
    
    @property
    def net_pnl_usdt(self) -> float:
        """PnL net (après frais)."""
        return self.pnl_usdt - self.fees_usdt
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "pnl_usdt": self.pnl_usdt,
            "pnl_percent": self.pnl_percent,
            "fees_usdt": self.fees_usdt,
            "net_pnl_usdt": self.net_pnl_usdt,
            "exit_reason": self.exit_reason,
            "outcome": self.outcome.value,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class TradeRecord:
    """
    Enregistrement simplifié d'un trade pour le tracking.
    
    Utilisé pour l'historique des trades (timestamps et résultats).
    """
    
    timestamp: datetime
    outcome: TradeOutcome
    pnl_usdt: float


# ============================================
# DAILY LIMITS
# ============================================

class DailyLimits:
    """
    Gère les limites journalières de trading.
    
    Suit les pertes, le nombre de trades, et reset automatiquement
    à minuit UTC.
    
    Attributes:
        config: Configuration de risque
        daily_pnl: PnL cumulé du jour
        daily_trades: Nombre de trades du jour
        last_reset: Date du dernier reset
    """
    
    def __init__(self, config: RiskConfig):
        """
        Initialise le tracker de limites journalières.
        
        Args:
            config: Configuration de risque
        """
        self._config = config
        self._max_daily_loss = config.initial_capital * config.max_daily_loss_pct
        
        # Stats journalières
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._daily_wins: int = 0
        self._daily_losses: int = 0
        self._last_reset: date = datetime.utcnow().date()
        
        # Historique horaire (pour limite par heure)
        self._hourly_trades: deque = deque(maxlen=200)
    
    # =========================================
    # PROPERTIES
    # =========================================
    
    @property
    def daily_pnl(self) -> float:
        """PnL du jour en USDT."""
        self._check_and_reset()
        return self._daily_pnl
    
    @property
    def daily_trades_count(self) -> int:
        """Nombre de trades du jour."""
        self._check_and_reset()
        return self._daily_trades
    
    @property
    def daily_win_rate(self) -> float:
        """Win rate du jour (0-1)."""
        self._check_and_reset()
        total = self._daily_wins + self._daily_losses
        if total == 0:
            return 0.0
        return self._daily_wins / total
    
    @property
    def hourly_trades_count(self) -> int:
        """Nombre de trades dans la dernière heure."""
        self._clean_hourly_trades()
        return len(self._hourly_trades)
    
    @property
    def remaining_daily_trades(self) -> int:
        """Trades restants pour le jour."""
        self._check_and_reset()
        return max(0, self._config.max_trades_per_day - self._daily_trades)
    
    @property
    def remaining_daily_loss(self) -> float:
        """Perte restante autorisée pour le jour."""
        self._check_and_reset()
        return max(0.0, self._max_daily_loss + self._daily_pnl)
    
    # =========================================
    # VÉRIFICATIONS
    # =========================================
    
    def can_trade(self) -> Tuple[bool, Optional[RejectionReason]]:
        """
        Vérifie si un trade est autorisé selon les limites journalières.
        
        Returns:
            Tuple (autorisé, raison_si_refusé)
        """
        self._check_and_reset()
        
        # Vérifier perte journalière
        if self._daily_pnl <= -self._max_daily_loss:
            return False, RejectionReason.DAILY_LOSS_EXCEEDED
        
        # Vérifier nombre de trades par jour
        if self._daily_trades >= self._config.max_trades_per_day:
            return False, RejectionReason.MAX_TRADES_PER_DAY
        
        # Vérifier nombre de trades par heure
        self._clean_hourly_trades()
        if len(self._hourly_trades) >= self._config.max_trades_per_hour:
            return False, RejectionReason.MAX_TRADES_PER_HOUR
        
        return True, None
    
    # =========================================
    # ENREGISTREMENT
    # =========================================
    
    def register_trade(self, trade: CompletedTrade) -> None:
        """
        Enregistre un trade terminé.
        
        Args:
            trade: Le trade complété
        """
        self._check_and_reset()
        
        # Mettre à jour les stats
        self._daily_pnl += trade.net_pnl_usdt
        self._daily_trades += 1
        
        if trade.outcome == TradeOutcome.WIN:
            self._daily_wins += 1
        elif trade.outcome == TradeOutcome.LOSS:
            self._daily_losses += 1
        
        # Enregistrer pour la limite horaire
        self._hourly_trades.append(datetime.utcnow())
        
        logger.debug(
            f"📊 DailyLimits | Trade enregistré | "
            f"PnL jour: {self._daily_pnl:+.2f} USDT | "
            f"Trades: {self._daily_trades}/{self._config.max_trades_per_day}"
        )
    
    def register_loss(self, amount: float) -> None:
        """
        Enregistre une perte (sans trade complet).
        
        Args:
            amount: Montant de la perte (positif)
        """
        self._check_and_reset()
        self._daily_pnl -= abs(amount)
    
    # =========================================
    # RESET
    # =========================================
    
    def _check_and_reset(self) -> None:
        """Reset automatique à minuit UTC."""
        today = datetime.utcnow().date()
        if today != self._last_reset:
            self.reset()
    
    def reset(self) -> None:
        """Reset manuel des stats journalières."""
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._daily_wins = 0
        self._daily_losses = 0
        self._last_reset = datetime.utcnow().date()
        self._hourly_trades.clear()
        
        logger.info("🔄 DailyLimits | Reset des stats journalières")
    
    def _clean_hourly_trades(self) -> None:
        """Supprime les trades de plus d'une heure."""
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        while self._hourly_trades and self._hourly_trades[0] < one_hour_ago:
            self._hourly_trades.popleft()
    
    # =========================================
    # STATISTIQUES
    # =========================================
    
    def get_statistics(self) -> Dict:
        """Retourne les statistiques journalières."""
        self._check_and_reset()
        return {
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "daily_wins": self._daily_wins,
            "daily_losses": self._daily_losses,
            "daily_win_rate": self.daily_win_rate,
            "hourly_trades": self.hourly_trades_count,
            "remaining_trades": self.remaining_daily_trades,
            "remaining_loss": self.remaining_daily_loss,
            "max_daily_loss": self._max_daily_loss,
        }


# ============================================
# KILL SWITCH
# ============================================

class KillSwitch:
    """
    Protection contre les pertes excessives.
    
    Le kill switch s'active automatiquement quand le drawdown
    atteint le seuil configuré. Une fois activé, il bloque
    tous les nouveaux trades.
    
    Attributes:
        initial_capital: Capital initial de référence
        max_drawdown_pct: Seuil de drawdown pour activation
        peak_capital: Plus haut capital atteint
        is_active: État du kill switch
    """
    
    def __init__(self, initial_capital: float, max_drawdown_pct: float):
        """
        Initialise le kill switch.
        
        Args:
            initial_capital: Capital initial
            max_drawdown_pct: Seuil de drawdown (0.25 = 25%)
        """
        self._initial_capital = initial_capital
        self._max_drawdown_pct = max_drawdown_pct
        self._peak_capital = initial_capital
        self._current_capital = initial_capital
        self._is_active = False
        self._activation_time: Optional[datetime] = None
        self._activation_drawdown: Optional[float] = None
    
    # =========================================
    # PROPERTIES
    # =========================================
    
    @property
    def is_active(self) -> bool:
        """Le kill switch est-il activé ?"""
        return self._is_active
    
    @property
    def peak_capital(self) -> float:
        """Plus haut capital atteint."""
        return self._peak_capital
    
    @property
    def current_capital(self) -> float:
        """Capital actuel."""
        return self._current_capital
    
    @property
    def current_drawdown(self) -> float:
        """Drawdown actuel en pourcentage (0-1)."""
        if self._peak_capital <= 0:
            return 0.0
        return (self._peak_capital - self._current_capital) / self._peak_capital
    
    @property
    def current_drawdown_from_initial(self) -> float:
        """Drawdown depuis le capital initial en pourcentage."""
        if self._initial_capital <= 0:
            return 0.0
        return (self._initial_capital - self._current_capital) / self._initial_capital
    
    @property
    def remaining_until_activation(self) -> float:
        """Capital restant avant activation (en USDT)."""
        threshold = self._initial_capital * (1 - self._max_drawdown_pct)
        return max(0.0, self._current_capital - threshold)
    
    # =========================================
    # UPDATE
    # =========================================
    
    def update(self, current_capital: float) -> bool:
        """
        Met à jour le capital et vérifie le drawdown.
        
        Args:
            current_capital: Capital actuel
            
        Returns:
            True si le kill switch vient de s'activer
        """
        self._current_capital = current_capital
        
        # Mettre à jour le peak
        if current_capital > self._peak_capital:
            self._peak_capital = current_capital
        
        # Calculer le drawdown depuis le capital initial
        drawdown = self.current_drawdown_from_initial
        
        # Vérifier si on doit activer
        if not self._is_active and drawdown >= self._max_drawdown_pct:
            self._activate(drawdown)
            return True
        
        return False
    
    def _activate(self, drawdown: float) -> None:
        """Active le kill switch."""
        self._is_active = True
        self._activation_time = datetime.utcnow()
        self._activation_drawdown = drawdown
        
        logger.critical(
            f"🚨 KILL SWITCH ACTIVÉ | "
            f"Drawdown: {drawdown:.1%} >= {self._max_drawdown_pct:.1%} | "
            f"Capital: {self._current_capital:.2f} / {self._initial_capital:.2f} USDT"
        )
    
    # =========================================
    # RESET
    # =========================================
    
    def reset(self, new_initial_capital: Optional[float] = None) -> None:
        """
        Reset le kill switch (après intervention manuelle).
        
        Args:
            new_initial_capital: Nouveau capital initial (optionnel)
        """
        if new_initial_capital is not None:
            self._initial_capital = new_initial_capital
            self._peak_capital = new_initial_capital
            self._current_capital = new_initial_capital
        
        self._is_active = False
        self._activation_time = None
        self._activation_drawdown = None
        
        logger.warning("⚠️ Kill switch RESET (intervention manuelle)")
    
    # =========================================
    # VÉRIFICATION
    # =========================================
    
    def check(self) -> Tuple[bool, Optional[str]]:
        """
        Vérifie l'état du kill switch.
        
        Returns:
            Tuple (peut_trader, message_si_non)
        """
        if self._is_active:
            message = (
                f"Kill switch actif depuis {self._activation_time} | "
                f"Drawdown: {self._activation_drawdown:.1%}"
            )
            return False, message
        return True, None
    
    # =========================================
    # STATISTIQUES
    # =========================================
    
    def get_statistics(self) -> Dict:
        """Retourne les statistiques du kill switch."""
        return {
            "is_active": self._is_active,
            "initial_capital": self._initial_capital,
            "peak_capital": self._peak_capital,
            "current_capital": self._current_capital,
            "current_drawdown": self.current_drawdown,
            "current_drawdown_from_initial": self.current_drawdown_from_initial,
            "max_drawdown_pct": self._max_drawdown_pct,
            "remaining_until_activation": self.remaining_until_activation,
            "activation_time": self._activation_time.isoformat() if self._activation_time else None,
            "activation_drawdown": self._activation_drawdown,
        }


# ============================================
# RISK MANAGER
# ============================================

class RiskManager:
    """
    Gestionnaire central du risque.
    
    Orchestre tous les aspects de la gestion du risque :
    - Position sizing
    - Limites journalières
    - Kill switch
    - Pertes consécutives
    - Cooldown après pertes
    
    Usage:
        config = RiskConfig(initial_capital=30.0)
        rm = RiskManager(config)
        
        # Vérifier si on peut trader
        can_trade, reason = rm.can_open_trade(signal)
        
        # Calculer la taille
        size = rm.calculate_position_size(signal, capital=25.0)
        
        # Après un trade
        rm.register_trade_result(completed_trade)
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        """
        Initialise le Risk Manager.
        
        Args:
            config: Configuration (utilise les défauts si None)
        """
        self._config = config or RiskConfig()
        self._config.validate()
        
        # Sous-composants
        self._daily_limits = DailyLimits(self._config)
        self._kill_switch = KillSwitch(
            initial_capital=self._config.initial_capital,
            max_drawdown_pct=self._config.max_drawdown_pct
        )
        
        # État
        self._current_capital = self._config.initial_capital
        self._open_positions: List[str] = []  # Symboles avec position ouverte
        
        # Tracking pertes consécutives
        self._consecutive_losses = 0
        self._last_trade_outcome: Optional[TradeOutcome] = None
        
        # Cooldown
        self._cooldown_until: Optional[datetime] = None
        
        # Historique des trades
        self._trade_history: deque = deque(maxlen=1000)
        
        logger.info(
            f"🛡️ RiskManager initialisé | "
            f"Capital: {self._config.initial_capital:.2f} USDT | "
            f"Max position: {self._config.max_position_pct:.0%} | "
            f"Max drawdown: {self._config.max_drawdown_pct:.0%}"
        )
    
    # =========================================
    # PROPERTIES
    # =========================================
    
    @property
    def config(self) -> RiskConfig:
        """Retourne la configuration."""
        return self._config
    
    @property
    def current_capital(self) -> float:
        """Capital actuel."""
        return self._current_capital
    
    @property
    def open_positions_count(self) -> int:
        """Nombre de positions ouvertes."""
        return len(self._open_positions)
    
    @property
    def consecutive_losses(self) -> int:
        """Nombre de pertes consécutives."""
        return self._consecutive_losses
    
    @property
    def daily_pnl(self) -> float:
        """PnL du jour."""
        return self._daily_limits.daily_pnl
    
    @property
    def is_kill_switch_active(self) -> bool:
        """Le kill switch est-il activé ?"""
        return self._kill_switch.is_active
    
    @property
    def is_in_cooldown(self) -> bool:
        """Est-on en période de cooldown ?"""
        if self._cooldown_until is None:
            return False
        return datetime.utcnow() < self._cooldown_until
    
    @property
    def cooldown_remaining_seconds(self) -> int:
        """Temps restant de cooldown en secondes."""
        if not self.is_in_cooldown:
            return 0
        return int((self._cooldown_until - datetime.utcnow()).total_seconds())
    
    # =========================================
    # VÉRIFICATION AUTORISATION TRADE
    # =========================================
    
    def can_open_trade(
        self,
        signal: Optional[object] = None,
        symbol: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Vérifie si un nouveau trade peut être ouvert.
        
        Args:
            signal: Signal de trading (optionnel)
            symbol: Symbole à trader (si pas de signal)
            
        Returns:
            Tuple (autorisé, raison_si_refusé)
        """
        # Extraire le symbole du signal si fourni
        trade_symbol = symbol
        if signal is not None and hasattr(signal, 'symbol'):
            trade_symbol = signal.symbol
        
        # 1. Vérifier le kill switch
        ks_ok, ks_msg = self._kill_switch.check()
        if not ks_ok:
            return False, RejectionReason.KILL_SWITCH_ACTIVE.value
        
        # 2. Vérifier les limites journalières
        daily_ok, daily_reason = self._daily_limits.can_trade()
        if not daily_ok:
            return False, daily_reason.value
        
        # 3. Vérifier les pertes consécutives
        if self._consecutive_losses >= self._config.max_consecutive_losses:
            return False, RejectionReason.MAX_CONSECUTIVE_LOSSES.value
        
        # 4. Vérifier le nombre de positions ouvertes
        if len(self._open_positions) >= self._config.max_open_positions:
            return False, RejectionReason.MAX_POSITIONS_REACHED.value
        
        # 5. Vérifier le cooldown
        if self.is_in_cooldown:
            remaining = self.cooldown_remaining_seconds
            return False, f"{RejectionReason.COOLDOWN_ACTIVE.value} ({remaining}s restantes)"
        
        # 6. Vérifier le capital suffisant
        if self._current_capital < self._config.min_position_usdt:
            return False, RejectionReason.INSUFFICIENT_CAPITAL.value
        
        return True, None
    
    # =========================================
    # POSITION SIZING
    # =========================================
    
    def calculate_position_size(
        self,
        signal: Optional[object] = None,
        current_capital: Optional[float] = None,
        stop_loss_pct: Optional[float] = None
    ) -> float:
        """
        Calcule la taille de position optimale.
        
        Utilise deux méthodes et prend le minimum :
        1. Pourcentage fixe du capital
        2. Basé sur le risque (stop-loss)
        
        Args:
            signal: Signal de trading (pour extraire SL)
            current_capital: Capital actuel (utilise self si None)
            stop_loss_pct: Stop-loss en % (utilise défaut si None)
            
        Returns:
            Taille de position en USDT (0 si non viable)
        """
        capital = current_capital or self._current_capital
        
        # Extraire le stop-loss du signal si disponible
        sl_pct = stop_loss_pct or self._config.default_stop_loss_pct
        if signal is not None:
            if hasattr(signal, 'stop_loss_pct') and signal.stop_loss_pct:
                sl_pct = signal.stop_loss_pct
            elif hasattr(signal, 'stop_loss_price') and hasattr(signal, 'entry_price'):
                if signal.entry_price and signal.stop_loss_price:
                    sl_pct = abs(signal.entry_price - signal.stop_loss_price) / signal.entry_price
        
        # Méthode 1: Pourcentage fixe du capital
        size_by_pct = capital * self._config.max_position_pct
        
        # Méthode 2: Basé sur le risque (stop-loss)
        risk_per_trade = capital * self._config.max_loss_per_trade_pct
        if sl_pct > 0:
            size_by_risk = risk_per_trade / sl_pct
        else:
            size_by_risk = size_by_pct
        
        # Prendre le minimum
        size = min(size_by_pct, size_by_risk)
        
        # Vérifier le minimum Binance
        if size < self._config.min_position_usdt:
            logger.warning(
                f"⚠️ Position trop petite: {size:.2f} USDT < "
                f"{self._config.min_position_usdt:.2f} USDT minimum"
            )
            return 0.0
        
        # Ne pas dépasser le capital disponible
        size = min(size, capital)
        
        logger.debug(
            f"📏 Position sizing | "
            f"Capital: {capital:.2f} | "
            f"Par %: {size_by_pct:.2f} | "
            f"Par risque: {size_by_risk:.2f} | "
            f"Final: {size:.2f} USDT"
        )
        
        return round(size, 2)
    
    # =========================================
    # GESTION DES POSITIONS
    # =========================================
    
    def register_position_opened(self, symbol: str) -> None:
        """
        Enregistre l'ouverture d'une position.
        
        Args:
            symbol: Symbole de la position
        """
        if symbol not in self._open_positions:
            self._open_positions.append(symbol)
            logger.info(f"📈 Position ouverte: {symbol} | Total: {len(self._open_positions)}")
    
    def register_position_closed(self, symbol: str) -> None:
        """
        Enregistre la fermeture d'une position.
        
        Args:
            symbol: Symbole de la position
        """
        if symbol in self._open_positions:
            self._open_positions.remove(symbol)
            logger.info(f"📉 Position fermée: {symbol} | Total: {len(self._open_positions)}")
    
    def get_open_positions(self) -> List[str]:
        """Retourne la liste des positions ouvertes."""
        return self._open_positions.copy()
    
    # =========================================
    # ENREGISTREMENT DES RÉSULTATS
    # =========================================
    
    def register_trade_result(self, trade: CompletedTrade) -> None:
        """
        Enregistre le résultat d'un trade terminé.
        
        Met à jour :
        - Le capital
        - Les limites journalières
        - Les pertes consécutives
        - Le kill switch
        
        Args:
            trade: Le trade complété
        """
        # 1. Mettre à jour le capital
        self._current_capital += trade.net_pnl_usdt
        
        # 2. Enregistrer dans les limites journalières
        self._daily_limits.register_trade(trade)
        
        # 3. Mettre à jour le kill switch
        was_activated = self._kill_switch.update(self._current_capital)
        
        # 4. Gérer les pertes consécutives
        self._update_consecutive_losses(trade.outcome)
        
        # 5. Retirer la position si elle était ouverte
        self.register_position_closed(trade.symbol)
        
        # 6. Ajouter à l'historique
        self._trade_history.append(TradeRecord(
            timestamp=trade.exit_time,
            outcome=trade.outcome,
            pnl_usdt=trade.net_pnl_usdt
        ))
        
        # 7. Gérer le cooldown si perte
        if trade.outcome == TradeOutcome.LOSS:
            self._apply_cooldown()
        
        # Log
        outcome_emoji = "🟢" if trade.outcome == TradeOutcome.WIN else "🔴" if trade.outcome == TradeOutcome.LOSS else "⚪"
        logger.info(
            f"{outcome_emoji} Trade enregistré | {trade.symbol} | "
            f"PnL: {trade.net_pnl_usdt:+.4f} USDT | "
            f"Capital: {self._current_capital:.2f} USDT | "
            f"Pertes consécutives: {self._consecutive_losses}"
        )
        
        # Alerte si kill switch activé
        if was_activated:
            raise KillSwitchActivatedError(
                self._kill_switch.current_drawdown_from_initial,
                self._config.max_drawdown_pct
            )
    
    def _update_consecutive_losses(self, outcome: TradeOutcome) -> None:
        """Met à jour le compteur de pertes consécutives."""
        if outcome == TradeOutcome.LOSS:
            self._consecutive_losses += 1
            logger.warning(f"⚠️ Perte consécutive #{self._consecutive_losses}")
        elif outcome == TradeOutcome.WIN:
            if self._consecutive_losses > 0:
                logger.info(f"✅ Série de pertes stoppée après {self._consecutive_losses} pertes")
            self._consecutive_losses = 0
        
        self._last_trade_outcome = outcome
    
    def _apply_cooldown(self) -> None:
        """Applique un cooldown après une perte."""
        # Cooldown simple après perte
        cooldown_seconds = self._config.cooldown_after_loss_sec
        
        # Cooldown plus long après pertes consécutives
        if self._consecutive_losses >= 3:
            cooldown_seconds = max(
                cooldown_seconds,
                self._config.cooldown_after_consecutive_losses_sec
            )
        
        if cooldown_seconds > 0:
            self._cooldown_until = datetime.utcnow() + timedelta(seconds=cooldown_seconds)
            logger.info(f"⏳ Cooldown activé: {cooldown_seconds}s")
    
    # =========================================
    # UPDATE CAPITAL
    # =========================================
    
    def update_capital(self, new_capital: float) -> None:
        """
        Met à jour le capital actuel.
        
        Args:
            new_capital: Nouveau capital
        """
        old_capital = self._current_capital
        self._current_capital = new_capital
        self._kill_switch.update(new_capital)
        
        if new_capital != old_capital:
            logger.debug(f"💰 Capital mis à jour: {old_capital:.2f} → {new_capital:.2f} USDT")
    
    # =========================================
    # RESET
    # =========================================
    
    def reset_daily_stats(self) -> None:
        """Reset les statistiques journalières."""
        self._daily_limits.reset()
        logger.info("🔄 Stats journalières reset")
    
    def reset_consecutive_losses(self) -> None:
        """Reset le compteur de pertes consécutives."""
        self._consecutive_losses = 0
        logger.info("🔄 Compteur de pertes consécutives reset")
    
    def reset_cooldown(self) -> None:
        """Annule le cooldown actif."""
        self._cooldown_until = None
        logger.info("🔄 Cooldown annulé")
    
    def reset_kill_switch(self, new_capital: Optional[float] = None) -> None:
        """
        Reset le kill switch (intervention manuelle).
        
        Args:
            new_capital: Nouveau capital initial (optionnel)
        """
        self._kill_switch.reset(new_capital)
        if new_capital:
            self._current_capital = new_capital
    
    def full_reset(self, new_capital: Optional[float] = None) -> None:
        """
        Reset complet du Risk Manager.
        
        Args:
            new_capital: Nouveau capital initial
        """
        capital = new_capital or self._config.initial_capital
        
        self._current_capital = capital
        self._open_positions.clear()
        self._consecutive_losses = 0
        self._cooldown_until = None
        self._trade_history.clear()
        
        self._daily_limits.reset()
        self._kill_switch.reset(capital)
        
        logger.warning(f"🔄 RESET COMPLET | Capital: {capital:.2f} USDT")
    
    # =========================================
    # STATISTIQUES
    # =========================================
    
    def get_statistics(self) -> Dict:
        """Retourne toutes les statistiques du Risk Manager."""
        return {
            "config": self._config.to_dict(),
            "current_capital": self._current_capital,
            "open_positions": self._open_positions.copy(),
            "open_positions_count": len(self._open_positions),
            "consecutive_losses": self._consecutive_losses,
            "is_in_cooldown": self.is_in_cooldown,
            "cooldown_remaining_seconds": self.cooldown_remaining_seconds,
            "daily_limits": self._daily_limits.get_statistics(),
            "kill_switch": self._kill_switch.get_statistics(),
            "total_trades": len(self._trade_history),
        }
    
    def get_trade_history(self, last_n: int = 50) -> List[Dict]:
        """
        Retourne l'historique des trades récents.
        
        Args:
            last_n: Nombre de trades à retourner
            
        Returns:
            Liste de dictionnaires
        """
        trades = list(self._trade_history)[-last_n:]
        return [
            {
                "timestamp": t.timestamp.isoformat(),
                "outcome": t.outcome.value,
                "pnl_usdt": t.pnl_usdt,
            }
            for t in trades
        ]
    
    # =========================================
    # REPRÉSENTATION
    # =========================================
    
    def __str__(self) -> str:
        """Représentation lisible."""
        stats = self.get_statistics()
        daily = stats["daily_limits"]
        ks = stats["kill_switch"]
        
        status = "🟢 OK"
        if self._kill_switch.is_active:
            status = "🔴 KILL SWITCH"
        elif daily["daily_pnl"] <= -daily["max_daily_loss"] * 0.8:
            status = "🟡 ATTENTION"
        
        return (
            f"🛡️ RiskManager | {status}\n"
            f"{'=' * 40}\n"
            f"💰 Capital: {self._current_capital:.2f} / {self._config.initial_capital:.2f} USDT\n"
            f"📊 Drawdown: {ks['current_drawdown_from_initial']:.1%} / {ks['max_drawdown_pct']:.1%}\n"
            f"📈 PnL jour: {daily['daily_pnl']:+.2f} USDT\n"
            f"📝 Trades jour: {daily['daily_trades']} / {self._config.max_trades_per_day}\n"
            f"🔄 Positions: {len(self._open_positions)} / {self._config.max_open_positions}\n"
            f"❌ Pertes consécutives: {self._consecutive_losses} / {self._config.max_consecutive_losses}"
        )


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def create_risk_manager(
    initial_capital: float = 30.0,
    max_position_pct: float = 0.20,
    max_daily_loss_pct: float = 0.10,
    max_drawdown_pct: float = 0.25
) -> RiskManager:
    """
    Fonction utilitaire pour créer un Risk Manager.
    
    Args:
        initial_capital: Capital initial en USDT
        max_position_pct: Taille max de position (0-1)
        max_daily_loss_pct: Perte journalière max (0-1)
        max_drawdown_pct: Drawdown max avant kill switch (0-1)
        
    Returns:
        RiskManager configuré
    """
    config = RiskConfig(
        initial_capital=initial_capital,
        max_position_pct=max_position_pct,
        max_daily_loss_pct=max_daily_loss_pct,
        max_drawdown_pct=max_drawdown_pct
    )
    return RiskManager(config)


def calculate_risk_reward_ratio(
    entry_price: float,
    stop_loss_price: float,
    take_profit_price: float
) -> float:
    """
    Calcule le ratio risque/récompense.
    
    Args:
        entry_price: Prix d'entrée
        stop_loss_price: Prix de stop-loss
        take_profit_price: Prix de take-profit
        
    Returns:
        Ratio R:R (ex: 2.0 = récompense double du risque)
    """
    risk = abs(entry_price - stop_loss_price)
    reward = abs(take_profit_price - entry_price)
    
    if risk == 0:
        return 0.0
    
    return reward / risk


# ============================================
# EXPORTS
# ============================================

__all__ = [
    # Enums
    "RejectionReason",
    "TradeOutcome",
    # Dataclasses
    "RiskConfig",
    "CompletedTrade",
    "TradeRecord",
    # Classes
    "DailyLimits",
    "KillSwitch",
    "RiskManager",
    # Fonctions
    "create_risk_manager",
    "calculate_risk_reward_ratio",
]