# cryptoscalper/backtest/engine.py
"""
Backtest Engine - Simulation de la stratégie sur données historiques.

Responsabilités :
- Charger et parcourir les données historiques
- Calculer les features pour chaque point dans le temps
- Générer des prédictions avec le modèle ML
- Simuler l'exécution des trades (frais + slippage)
- Tracker les performances

Usage:
    engine = BacktestEngine(config)
    result = await engine.run(data, predictor)
    print(result.summary())
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import pandas as pd
import numpy as np

from cryptoscalper.config.constants import (
    BINANCE_FEE_PERCENT,
    ROUND_TRIP_FEE_PERCENT,
    ESTIMATED_SLIPPAGE_PERCENT,
    DEFAULT_STOP_LOSS_PERCENT,
    DEFAULT_TAKE_PROFIT_PERCENT,
    SIGNAL_MIN_PROBABILITY,
    SIGNAL_MIN_CONFIDENCE,
)
from cryptoscalper.utils.logger import logger


# ============================================
# ENUMS
# ============================================

class BacktestMode(Enum):
    """Mode de backtest."""
    
    VECTORIZED = "vectorized"      # Rapide mais moins réaliste
    EVENT_DRIVEN = "event_driven"  # Plus lent mais plus réaliste


class TradeCloseReason(Enum):
    """Raison de fermeture d'un trade dans le backtest."""
    
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIMEOUT = "timeout"
    END_OF_DATA = "end_of_data"


# ============================================
# DATACLASSES
# ============================================

@dataclass
class BacktestConfig:
    """
    Configuration du backtest.
    
    Attributes:
        initial_capital: Capital initial en USDT
        position_size_pct: Taille de position en % du capital (0.0-1.0)
        fee_percent: Frais par transaction (0.001 = 0.1%)
        slippage_percent: Slippage estimé (0.0005 = 0.05%)
        stop_loss_pct: Stop loss par défaut (0.004 = 0.4%)
        take_profit_pct: Take profit par défaut (0.003 = 0.3%)
        min_probability: Probabilité minimum pour trade (0.65)
        min_confidence: Confiance minimum pour trade (0.55)
        max_position_duration_minutes: Durée max d'une position
        max_open_positions: Nombre max de positions ouvertes
        require_features: Vérifier que les features sont valides
    """
    
    initial_capital: float = 30.0
    position_size_pct: float = 0.20  # 20% du capital
    
    # Frais et slippage
    fee_percent: float = BINANCE_FEE_PERCENT  # 0.1%
    slippage_percent: float = ESTIMATED_SLIPPAGE_PERCENT  # 0.05%
    
    # SL/TP par défaut
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PERCENT  # 0.4%
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PERCENT  # 0.3%
    
    # Seuils de signal
    min_probability: float = SIGNAL_MIN_PROBABILITY  # 0.65
    min_confidence: float = SIGNAL_MIN_CONFIDENCE  # 0.55
    
    # Contraintes
    max_position_duration_minutes: int = 5
    max_open_positions: int = 1
    
    # Options
    require_features: bool = True
    use_trailing_stop: bool = False
    
    def __post_init__(self):
        """Validation des paramètres."""
        assert 0 < self.initial_capital, "Capital doit être positif"
        assert 0 < self.position_size_pct <= 1.0, "Position size doit être entre 0 et 1"
        assert 0 <= self.fee_percent < 0.01, "Frais suspects (> 1%)"
        assert 0 <= self.slippage_percent < 0.01, "Slippage suspect (> 1%)"
        assert 0 < self.stop_loss_pct < 0.1, "SL doit être entre 0 et 10%"
        assert 0 < self.take_profit_pct < 0.1, "TP doit être entre 0 et 10%"
    
    @property
    def total_fees_pct(self) -> float:
        """Frais totaux aller-retour."""
        return self.fee_percent * 2
    
    @property
    def total_slippage_pct(self) -> float:
        """Slippage total aller-retour."""
        return self.slippage_percent * 2


@dataclass
class BacktestTrade:
    """
    Représente un trade simulé pendant le backtest.
    
    Attributes:
        trade_id: Identifiant unique
        symbol: Paire tradée
        entry_time: Timestamp d'entrée
        entry_price: Prix d'entrée (après slippage)
        entry_price_raw: Prix brut (avant slippage)
        quantity: Quantité tradée
        size_usdt: Taille en USDT
        stop_loss_price: Prix de stop loss
        take_profit_price: Prix de take profit
        exit_time: Timestamp de sortie
        exit_price: Prix de sortie (après slippage)
        exit_price_raw: Prix brut (avant slippage)
        pnl_usdt: Profit/perte en USDT (après frais)
        pnl_percent: Profit/perte en %
        pnl_gross: PnL brut (avant frais)
        fees_paid: Frais payés
        close_reason: Raison de fermeture
        probability: Probabilité du modèle
        confidence: Confiance du modèle
    """
    
    trade_id: int
    symbol: str
    entry_time: datetime
    entry_price: float
    entry_price_raw: float
    quantity: float
    size_usdt: float
    stop_loss_price: float
    take_profit_price: float
    
    # Sortie (remplis à la fermeture)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_price_raw: Optional[float] = None
    pnl_usdt: Optional[float] = None
    pnl_percent: Optional[float] = None
    pnl_gross: Optional[float] = None
    fees_paid: Optional[float] = None
    close_reason: Optional[TradeCloseReason] = None
    
    # Métadonnées du signal
    probability: float = 0.0
    confidence: float = 0.0
    
    @property
    def is_open(self) -> bool:
        """Le trade est-il toujours ouvert ?"""
        return self.exit_time is None
    
    @property
    def is_winner(self) -> bool:
        """Le trade est-il gagnant ?"""
        return self.pnl_usdt is not None and self.pnl_usdt > 0
    
    @property
    def duration_minutes(self) -> Optional[float]:
        """Durée du trade en minutes."""
        if self.exit_time is None:
            return None
        return (self.exit_time - self.entry_time).total_seconds() / 60
    
    def close(
        self,
        exit_time: datetime,
        exit_price_raw: float,
        exit_price: float,
        close_reason: TradeCloseReason,
        fee_percent: float,
    ) -> None:
        """
        Ferme le trade et calcule le PnL.
        
        Args:
            exit_time: Timestamp de sortie
            exit_price_raw: Prix de sortie brut
            exit_price: Prix de sortie (après slippage)
            close_reason: Raison de fermeture
            fee_percent: Frais (0.001 = 0.1%)
        """
        self.exit_time = exit_time
        self.exit_price_raw = exit_price_raw
        self.exit_price = exit_price
        self.close_reason = close_reason
        
        # Calcul PnL brut
        price_change = self.exit_price - self.entry_price
        self.pnl_gross = price_change * self.quantity
        
        # Frais (entrée + sortie)
        entry_fees = self.size_usdt * fee_percent
        exit_value = self.quantity * self.exit_price
        exit_fees = exit_value * fee_percent
        self.fees_paid = entry_fees + exit_fees
        
        # PnL net
        self.pnl_usdt = self.pnl_gross - self.fees_paid
        self.pnl_percent = self.pnl_usdt / self.size_usdt if self.size_usdt > 0 else 0


@dataclass
class BacktestState:
    """
    État interne du backtest à un instant T.
    
    Attributes:
        current_capital: Capital actuel
        equity: Equity (capital + valeur positions ouvertes)
        open_trades: Liste des trades ouverts
        closed_trades: Liste des trades fermés
        total_trades: Nombre total de trades
        winning_trades: Nombre de trades gagnants
        total_fees: Total des frais payés
        peak_equity: Plus haut equity atteint
        max_drawdown: Drawdown maximum
    """
    
    current_capital: float
    equity: float
    open_trades: List[BacktestTrade] = field(default_factory=list)
    closed_trades: List[BacktestTrade] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    total_fees: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    
    # Historique pour les graphiques
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialisation."""
        self.peak_equity = self.current_capital
        self.equity = self.current_capital


# ============================================
# BACKTEST ENGINE
# ============================================

class BacktestEngine:
    """
    Moteur de backtest pour simuler la stratégie.
    
    Le backtest parcourt les données historiques et simule
    l'exécution des trades comme s'ils étaient réels, avec
    prise en compte des frais et du slippage.
    
    Usage:
        config = BacktestConfig(initial_capital=30.0)
        engine = BacktestEngine(config)
        
        # Avec predictor
        result = engine.run(
            data=historical_df,
            predictor=ml_predictor,
            feature_engine=feature_engine,
        )
        
        # Afficher le résumé
        print(result.summary())
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialise le moteur de backtest.
        
        Args:
            config: Configuration (défaut si None)
        """
        self._config = config or BacktestConfig()
        self._state: Optional[BacktestState] = None
        self._trade_counter = 0
        
        logger.info(
            f"📊 BacktestEngine initialisé | "
            f"Capital: {self._config.initial_capital:.2f} USDT | "
            f"Position: {self._config.position_size_pct:.0%} | "
            f"Fees: {self._config.fee_percent:.2%}"
        )
    
    # =========================================
    # PROPRIÉTÉS
    # =========================================
    
    @property
    def config(self) -> BacktestConfig:
        """Configuration du backtest."""
        return self._config
    
    @property
    def state(self) -> Optional[BacktestState]:
        """État actuel du backtest."""
        return self._state
    
    # =========================================
    # SIMULATION DES PRIX
    # =========================================
    
    def _apply_slippage_buy(self, price: float) -> float:
        """
        Applique le slippage pour un achat (prix augmente).
        
        Args:
            price: Prix brut
            
        Returns:
            Prix avec slippage
        """
        return price * (1 + self._config.slippage_percent)
    
    def _apply_slippage_sell(self, price: float) -> float:
        """
        Applique le slippage pour une vente (prix diminue).
        
        Args:
            price: Prix brut
            
        Returns:
            Prix avec slippage
        """
        return price * (1 - self._config.slippage_percent)
    
    # =========================================
    # GESTION DES TRADES
    # =========================================
    
    def _can_open_trade(self) -> bool:
        """Vérifie si on peut ouvrir un nouveau trade."""
        if self._state is None:
            return False
        
        # Vérifier le nombre de positions ouvertes
        if len(self._state.open_trades) >= self._config.max_open_positions:
            return False
        
        # Vérifier le capital disponible
        position_size = self._state.current_capital * self._config.position_size_pct
        if position_size < 5.0:  # Minimum 5 USDT
            return False
        
        return True
    
    def _open_trade(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        probability: float,
        confidence: float,
    ) -> BacktestTrade:
        """
        Ouvre un nouveau trade.
        
        Args:
            symbol: Paire tradée
            timestamp: Timestamp d'entrée
            price: Prix d'entrée brut
            probability: Probabilité du modèle
            confidence: Confiance du modèle
            
        Returns:
            BacktestTrade créé
        """
        self._trade_counter += 1
        
        # Calculer la taille de position
        size_usdt = self._state.current_capital * self._config.position_size_pct
        
        # Appliquer le slippage
        entry_price = self._apply_slippage_buy(price)
        
        # Calculer la quantité
        quantity = size_usdt / entry_price
        
        # Calculer SL/TP
        stop_loss_price = entry_price * (1 - self._config.stop_loss_pct)
        take_profit_price = entry_price * (1 + self._config.take_profit_pct)
        
        trade = BacktestTrade(
            trade_id=self._trade_counter,
            symbol=symbol,
            entry_time=timestamp,
            entry_price=entry_price,
            entry_price_raw=price,
            quantity=quantity,
            size_usdt=size_usdt,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            probability=probability,
            confidence=confidence,
        )
        
        # Mettre à jour l'état
        self._state.open_trades.append(trade)
        self._state.current_capital -= size_usdt
        self._state.total_trades += 1
        
        logger.debug(
            f"📈 OPEN #{trade.trade_id} | {symbol} | "
            f"Entry: {entry_price:.2f} | Size: {size_usdt:.2f} USDT | "
            f"SL: {stop_loss_price:.2f} | TP: {take_profit_price:.2f}"
        )
        
        return trade
    
    def _close_trade(
        self,
        trade: BacktestTrade,
        timestamp: datetime,
        price: float,
        reason: TradeCloseReason,
    ) -> None:
        """
        Ferme un trade.
        
        Args:
            trade: Trade à fermer
            timestamp: Timestamp de sortie
            price: Prix de sortie brut
            reason: Raison de fermeture
        """
        # Appliquer le slippage
        exit_price = self._apply_slippage_sell(price)
        
        # Fermer le trade
        trade.close(
            exit_time=timestamp,
            exit_price_raw=price,
            exit_price=exit_price,
            close_reason=reason,
            fee_percent=self._config.fee_percent,
        )
        
        # Mettre à jour l'état
        self._state.open_trades.remove(trade)
        self._state.closed_trades.append(trade)
        
        # Mettre à jour le capital
        exit_value = trade.quantity * exit_price
        self._state.current_capital += exit_value - (exit_value * self._config.fee_percent)
        
        # Compteurs
        self._state.total_fees += trade.fees_paid
        if trade.is_winner:
            self._state.winning_trades += 1
        
        emoji = "🎉" if trade.is_winner else "😤"
        logger.debug(
            f"{emoji} CLOSE #{trade.trade_id} | {trade.symbol} | "
            f"Exit: {exit_price:.2f} | PnL: {trade.pnl_usdt:+.4f} USDT ({trade.pnl_percent:+.2%}) | "
            f"Reason: {reason.value}"
        )
    
    def _check_stop_loss(
        self,
        trade: BacktestTrade,
        low_price: float,
    ) -> bool:
        """
        Vérifie si le stop loss est touché.
        
        Args:
            trade: Trade à vérifier
            low_price: Prix le plus bas de la bougie
            
        Returns:
            True si SL touché
        """
        return low_price <= trade.stop_loss_price
    
    def _check_take_profit(
        self,
        trade: BacktestTrade,
        high_price: float,
    ) -> bool:
        """
        Vérifie si le take profit est touché.
        
        Args:
            trade: Trade à vérifier
            high_price: Prix le plus haut de la bougie
            
        Returns:
            True si TP touché
        """
        return high_price >= trade.take_profit_price
    
    def _check_timeout(
        self,
        trade: BacktestTrade,
        current_time: datetime,
    ) -> bool:
        """
        Vérifie si le trade a dépassé la durée maximale.
        
        Args:
            trade: Trade à vérifier
            current_time: Timestamp actuel
            
        Returns:
            True si timeout
        """
        duration = (current_time - trade.entry_time).total_seconds() / 60
        return duration >= self._config.max_position_duration_minutes
    
    def _update_equity(self, current_prices: Dict[str, float]) -> None:
        """
        Met à jour l'equity en comptant les positions ouvertes.
        
        Args:
            current_prices: Prix actuels par symbole
        """
        open_value = 0.0
        for trade in self._state.open_trades:
            price = current_prices.get(trade.symbol, trade.entry_price)
            open_value += trade.quantity * price
        
        self._state.equity = self._state.current_capital + open_value
        
        # Mettre à jour le peak et drawdown
        if self._state.equity > self._state.peak_equity:
            self._state.peak_equity = self._state.equity
        
        current_drawdown = (self._state.peak_equity - self._state.equity) / self._state.peak_equity
        if current_drawdown > self._state.max_drawdown:
            self._state.max_drawdown = current_drawdown
    
    # =========================================
    # EXÉCUTION DU BACKTEST
    # =========================================
    
    def run(
        self,
        data: pd.DataFrame,
        predictor,  # MLPredictor
        feature_engine,  # FeatureEngine
        symbol: str = "BTCUSDT",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> "BacktestResult":
        """
        Exécute le backtest sur les données historiques.
        
        Le backtest parcourt les données bougie par bougie et simule
        la stratégie de trading avec le modèle ML.
        
        Args:
            data: DataFrame avec colonnes OHLCV (open, high, low, close, volume)
            predictor: MLPredictor pour les prédictions
            feature_engine: FeatureEngine pour calculer les features
            symbol: Symbole de la paire
            progress_callback: Callback(current, total) pour progression
            
        Returns:
            BacktestResult avec les statistiques
        """
        logger.info(f"🚀 Démarrage backtest | {symbol} | {len(data)} bougies")
        
        # Validation des données
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")
        
        # Initialiser l'état
        self._state = BacktestState(
            current_capital=self._config.initial_capital,
            equity=self._config.initial_capital,
        )
        self._trade_counter = 0
        
        # S'assurer que l'index est un DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            if "timestamp" in data.columns:
                data = data.set_index("timestamp")
            else:
                data.index = pd.to_datetime(data.index)
        
        # Période de warmup pour les indicateurs (50 bougies minimum)
        warmup_period = 50
        
        # Parcourir les données
        n_rows = len(data)
        for i in range(warmup_period, n_rows):
            # Données jusqu'à maintenant (pour éviter le look-ahead bias)
            current_data = data.iloc[:i+1]
            current_row = data.iloc[i]
            
            timestamp = data.index[i]
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
            
            current_price = float(current_row["close"])
            high_price = float(current_row["high"])
            low_price = float(current_row["low"])
            
            # 1. Vérifier les trades ouverts (SL/TP/Timeout)
            trades_to_close = []
            for trade in self._state.open_trades:
                # Vérifier SL en premier (exécuté au prix SL)
                if self._check_stop_loss(trade, low_price):
                    trades_to_close.append((trade, trade.stop_loss_price, TradeCloseReason.STOP_LOSS))
                # Puis TP (exécuté au prix TP)
                elif self._check_take_profit(trade, high_price):
                    trades_to_close.append((trade, trade.take_profit_price, TradeCloseReason.TAKE_PROFIT))
                # Puis timeout (exécuté au prix actuel)
                elif self._check_timeout(trade, timestamp):
                    trades_to_close.append((trade, current_price, TradeCloseReason.TIMEOUT))
            
            # Fermer les trades
            for trade, exit_price, reason in trades_to_close:
                self._close_trade(trade, timestamp, exit_price, reason)
            
            # 2. Chercher de nouvelles opportunités si pas de position ouverte
            if self._can_open_trade():
                try:
                    # Calculer les features
                    feature_set = feature_engine.compute_features(
                        df=current_data.tail(100),  # Dernières 100 bougies
                        orderbook=None,
                        symbol=symbol,
                    )
                    
                    # Obtenir la prédiction
                    prediction = predictor.predict_single(feature_set.features)
                    
                    # Vérifier les seuils
                    if (prediction.probability_up >= self._config.min_probability and
                        prediction.confidence >= self._config.min_confidence):
                        
                        # Ouvrir un trade
                        self._open_trade(
                            symbol=symbol,
                            timestamp=timestamp,
                            price=current_price,
                            probability=prediction.probability_up,
                            confidence=prediction.confidence,
                        )
                        
                except Exception as e:
                    logger.debug(f"Erreur prédiction à {timestamp}: {e}")
            
            # 3. Mettre à jour l'equity
            self._update_equity({symbol: current_price})
            
            # Enregistrer l'equity curve
            self._state.equity_curve.append((timestamp, self._state.equity))
            
            # Callback de progression
            if progress_callback and i % 100 == 0:
                progress_callback(i - warmup_period, n_rows - warmup_period)
        
        # Fermer les positions restantes
        for trade in list(self._state.open_trades):
            final_price = float(data.iloc[-1]["close"])
            final_time = data.index[-1]
            if isinstance(final_time, pd.Timestamp):
                final_time = final_time.to_pydatetime()
            self._close_trade(trade, final_time, final_price, TradeCloseReason.END_OF_DATA)
        
        logger.info(
            f"✅ Backtest terminé | "
            f"Trades: {self._state.total_trades} | "
            f"PnL: {self._state.equity - self._config.initial_capital:+.2f} USDT"
        )
        
        return BacktestResult.from_state(
            state=self._state,
            config=self._config,
            symbol=symbol,
            start_date=data.index[warmup_period].to_pydatetime() if isinstance(data.index[warmup_period], pd.Timestamp) else data.index[warmup_period],
            end_date=data.index[-1].to_pydatetime() if isinstance(data.index[-1], pd.Timestamp) else data.index[-1],
        )
    
    def run_simple(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        symbol: str = "BTCUSDT",
    ) -> "BacktestResult":
        """
        Exécute un backtest simple avec des signaux pré-calculés.
        
        Utile pour tester différentes stratégies rapidement sans
        recalculer les features à chaque fois.
        
        Args:
            data: DataFrame OHLCV
            signals: Series de signaux (1 = achat, 0 = rien)
            symbol: Symbole de la paire
            
        Returns:
            BacktestResult
        """
        logger.info(f"🚀 Backtest simple | {symbol} | {len(data)} bougies")
        
        # Initialiser l'état
        self._state = BacktestState(
            current_capital=self._config.initial_capital,
            equity=self._config.initial_capital,
        )
        self._trade_counter = 0
        
        # Parcourir les données
        n_rows = len(data)
        for i in range(n_rows):
            current_row = data.iloc[i]
            timestamp = data.index[i]
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
            
            current_price = float(current_row["close"])
            high_price = float(current_row["high"])
            low_price = float(current_row["low"])
            
            # Vérifier les trades ouverts
            trades_to_close = []
            for trade in self._state.open_trades:
                if self._check_stop_loss(trade, low_price):
                    trades_to_close.append((trade, trade.stop_loss_price, TradeCloseReason.STOP_LOSS))
                elif self._check_take_profit(trade, high_price):
                    trades_to_close.append((trade, trade.take_profit_price, TradeCloseReason.TAKE_PROFIT))
                elif self._check_timeout(trade, timestamp):
                    trades_to_close.append((trade, current_price, TradeCloseReason.TIMEOUT))
            
            for trade, exit_price, reason in trades_to_close:
                self._close_trade(trade, timestamp, exit_price, reason)
            
            # Ouvrir un trade si signal
            if self._can_open_trade() and i < len(signals) and signals.iloc[i] == 1:
                self._open_trade(
                    symbol=symbol,
                    timestamp=timestamp,
                    price=current_price,
                    probability=0.7,
                    confidence=0.6,
                )
            
            # Mettre à jour l'equity
            self._update_equity({symbol: current_price})
            self._state.equity_curve.append((timestamp, self._state.equity))
        
        # Fermer les positions restantes
        for trade in list(self._state.open_trades):
            final_price = float(data.iloc[-1]["close"])
            final_time = data.index[-1]
            if isinstance(final_time, pd.Timestamp):
                final_time = final_time.to_pydatetime()
            self._close_trade(trade, final_time, final_price, TradeCloseReason.END_OF_DATA)
        
        return BacktestResult.from_state(
            state=self._state,
            config=self._config,
            symbol=symbol,
            start_date=data.index[0].to_pydatetime() if isinstance(data.index[0], pd.Timestamp) else data.index[0],
            end_date=data.index[-1].to_pydatetime() if isinstance(data.index[-1], pd.Timestamp) else data.index[-1],
        )


# ============================================
# RÉSULTATS DU BACKTEST
# ============================================

@dataclass
class BacktestResult:
    """
    Résultats du backtest.
    
    Contient toutes les statistiques et métriques de performance
    calculées après l'exécution du backtest.
    """
    
    # Métadonnées
    symbol: str
    start_date: datetime
    end_date: datetime
    duration_days: float
    
    # Capital
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    
    # Trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # PnL
    total_pnl: float
    average_pnl: float
    best_trade: float
    worst_trade: float
    
    # Frais
    total_fees: float
    fees_percent_of_pnl: float
    
    # Risque
    max_drawdown: float
    profit_factor: float
    sharpe_ratio: float
    
    # Temps
    avg_trade_duration_minutes: float
    
    # Détails
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    
    # Raisons de fermeture
    take_profit_count: int = 0
    stop_loss_count: int = 0
    timeout_count: int = 0
    
    @classmethod
    def from_state(
        cls,
        state: BacktestState,
        config: BacktestConfig,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> "BacktestResult":
        """
        Crée un BacktestResult depuis l'état final.
        
        Args:
            state: État final du backtest
            config: Configuration utilisée
            symbol: Symbole tradé
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            BacktestResult calculé
        """
        trades = state.closed_trades
        n_trades = len(trades)
        
        # PnL des trades
        pnls = [t.pnl_usdt for t in trades if t.pnl_usdt is not None]
        
        if n_trades == 0:
            return cls(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                duration_days=(end_date - start_date).days,
                initial_capital=config.initial_capital,
                final_capital=state.current_capital,
                total_return=0.0,
                total_return_pct=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                average_pnl=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                total_fees=0.0,
                fees_percent_of_pnl=0.0,
                max_drawdown=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                avg_trade_duration_minutes=0.0,
                trades=trades,
                equity_curve=state.equity_curve,
            )
        
        # Calculs
        total_pnl = sum(pnls)
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]
        
        # Profit factor
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio (simplifié, annualisé)
        if len(pnls) > 1:
            returns = np.array(pnls) / config.initial_capital
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 24 * 12)  # Annualisé pour 5min
        else:
            sharpe = 0.0
        
        # Durée moyenne
        durations = [t.duration_minutes for t in trades if t.duration_minutes is not None]
        avg_duration = np.mean(durations) if durations else 0.0
        
        # Raisons de fermeture
        reasons = [t.close_reason for t in trades if t.close_reason]
        tp_count = sum(1 for r in reasons if r == TradeCloseReason.TAKE_PROFIT)
        sl_count = sum(1 for r in reasons if r == TradeCloseReason.STOP_LOSS)
        timeout_count = sum(1 for r in reasons if r == TradeCloseReason.TIMEOUT)
        
        return cls(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            duration_days=(end_date - start_date).days,
            initial_capital=config.initial_capital,
            final_capital=state.current_capital,
            total_return=state.current_capital - config.initial_capital,
            total_return_pct=(state.current_capital - config.initial_capital) / config.initial_capital,
            total_trades=n_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning) / n_trades if n_trades > 0 else 0.0,
            total_pnl=total_pnl,
            average_pnl=np.mean(pnls) if pnls else 0.0,
            best_trade=max(pnls) if pnls else 0.0,
            worst_trade=min(pnls) if pnls else 0.0,
            total_fees=state.total_fees,
            fees_percent_of_pnl=state.total_fees / abs(total_pnl) if total_pnl != 0 else 0.0,
            max_drawdown=state.max_drawdown,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            avg_trade_duration_minutes=avg_duration,
            trades=trades,
            equity_curve=state.equity_curve,
            take_profit_count=tp_count,
            stop_loss_count=sl_count,
            timeout_count=timeout_count,
        )
    
    def summary(self) -> str:
        """
        Génère un résumé textuel des résultats.
        
        Returns:
            Résumé formaté
        """
        lines = [
            "=" * 60,
            "📊 RÉSULTATS DU BACKTEST",
            "=" * 60,
            f"",
            f"📈 Symbole: {self.symbol}",
            f"📅 Période: {self.start_date.strftime('%Y-%m-%d')} → {self.end_date.strftime('%Y-%m-%d')} ({self.duration_days} jours)",
            f"",
            "─" * 60,
            "💰 CAPITAL",
            "─" * 60,
            f"  Initial:     {self.initial_capital:>10.2f} USDT",
            f"  Final:       {self.final_capital:>10.2f} USDT",
            f"  Rendement:   {self.total_return:>+10.2f} USDT ({self.total_return_pct:+.2%})",
            f"",
            "─" * 60,
            "📊 TRADES",
            "─" * 60,
            f"  Total:       {self.total_trades:>10}",
            f"  Gagnants:    {self.winning_trades:>10} ({self.win_rate:.1%})",
            f"  Perdants:    {self.losing_trades:>10}",
            f"",
            f"  Take Profit: {self.take_profit_count:>10}",
            f"  Stop Loss:   {self.stop_loss_count:>10}",
            f"  Timeout:     {self.timeout_count:>10}",
            f"",
            "─" * 60,
            "💵 PnL",
            "─" * 60,
            f"  Total PnL:   {self.total_pnl:>+10.4f} USDT",
            f"  Moyenne:     {self.average_pnl:>+10.4f} USDT",
            f"  Meilleur:    {self.best_trade:>+10.4f} USDT",
            f"  Pire:        {self.worst_trade:>+10.4f} USDT",
            f"",
            f"  Frais payés: {self.total_fees:>10.4f} USDT",
            f"",
            "─" * 60,
            "📉 RISQUE",
            "─" * 60,
            f"  Max Drawdown:   {self.max_drawdown:>8.2%}",
            f"  Profit Factor:  {self.profit_factor:>8.2f}",
            f"  Sharpe Ratio:   {self.sharpe_ratio:>8.2f}",
            f"",
            "─" * 60,
            "⏱️ TEMPS",
            "─" * 60,
            f"  Durée moyenne:  {self.avg_trade_duration_minutes:>8.1f} min",
            f"",
            "=" * 60,
        ]
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire."""
        return {
            "symbol": self.symbol,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "duration_days": self.duration_days,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "average_pnl": self.average_pnl,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "total_fees": self.total_fees,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "avg_trade_duration_minutes": self.avg_trade_duration_minutes,
            "take_profit_count": self.take_profit_count,
            "stop_loss_count": self.stop_loss_count,
            "timeout_count": self.timeout_count,
        }


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def create_backtest_engine(
    initial_capital: float = 30.0,
    position_size_pct: float = 0.20,
    fee_percent: float = 0.001,
    stop_loss_pct: float = 0.004,
    take_profit_pct: float = 0.003,
) -> BacktestEngine:
    """
    Crée un BacktestEngine avec des paramètres personnalisés.
    
    Args:
        initial_capital: Capital initial
        position_size_pct: Taille de position en %
        fee_percent: Frais par transaction
        stop_loss_pct: Stop loss en %
        take_profit_pct: Take profit en %
        
    Returns:
        BacktestEngine configuré
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        fee_percent=fee_percent,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )
    return BacktestEngine(config)


def load_historical_data(filepath: str) -> pd.DataFrame:
    """
    Charge les données historiques depuis un fichier.
    
    Supporte CSV et Parquet.
    
    Args:
        filepath: Chemin vers le fichier
        
    Returns:
        DataFrame avec index datetime
    """
    path = Path(filepath)
    
    if path.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif path.suffix == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Format non supporté: {path.suffix}")
    
    # Convertir en datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("open_time")
    
    # Renommer les colonnes si nécessaire
    col_mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=col_mapping)
    
    return df


__all__ = [
    # Enums
    "BacktestMode",
    "TradeCloseReason",
    # Dataclasses
    "BacktestConfig",
    "BacktestTrade",
    "BacktestState",
    "BacktestResult",
    # Classes
    "BacktestEngine",
    # Functions
    "create_backtest_engine",
    "load_historical_data",
]