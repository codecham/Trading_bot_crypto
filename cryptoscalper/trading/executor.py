# cryptoscalper/trading/executor.py
"""
Module d'exécution des ordres (Trade Executor).

Responsabilités :
- Exécuter les ordres market BUY
- Placer les ordres OCO (SL + TP)
- Gérer les positions ouvertes
- Suivre les stop-loss et take-profit
- Synchroniser avec l'exchange

Usage:
    executor = TradeExecutor(config)
    await executor.connect()
    
    # Exécuter un signal
    position = await executor.execute_signal(signal, size_usdt=10.0)
    
    # Fermer une position
    completed = await executor.close_position(position, reason="take_profit")
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_DOWN
import asyncio

from binance import AsyncClient
from binance.exceptions import BinanceAPIException, BinanceRequestException

from cryptoscalper.config.settings import get_settings
from cryptoscalper.utils.logger import logger
from cryptoscalper.utils.exceptions import (
    APIConnectionError,
    OrderExecutionError,
    InsufficientBalanceError,
)


# ============================================
# ENUMS
# ============================================

class OrderType(Enum):
    """Type d'ordre."""
    
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class OrderSide(Enum):
    """Côté de l'ordre."""
    
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Statut d'un ordre."""
    
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionStatus(Enum):
    """Statut d'une position."""
    
    PENDING = "PENDING"        # En attente d'ouverture
    OPEN = "OPEN"              # Position ouverte
    CLOSING = "CLOSING"        # En cours de fermeture
    CLOSED = "CLOSED"          # Position fermée


class CloseReason(Enum):
    """Raison de fermeture d'une position."""
    
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    MANUAL = "manual"
    TIMEOUT = "timeout"
    KILL_SWITCH = "kill_switch"
    ERROR = "error"


# ============================================
# DATACLASSES
# ============================================

@dataclass
class ExecutorConfig:
    """
    Configuration de l'executor.
    
    Attributes:
        testnet: Mode testnet ou production
        default_recv_window: Fenêtre de réception des ordres (ms)
        order_timeout_seconds: Timeout pour les ordres
        max_retries: Nombre max de tentatives en cas d'erreur
        retry_delay_seconds: Délai entre les tentatives
        sync_interval_seconds: Intervalle de synchronisation avec l'exchange
    """
    
    testnet: bool = True
    default_recv_window: int = 5000
    order_timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    sync_interval_seconds: int = 5
    
    # Paper trading (simulation sans ordres réels)
    paper_trading: bool = False


@dataclass
class SymbolInfo:
    """
    Informations sur une paire de trading.
    
    Attributes:
        symbol: Symbole de la paire
        base_asset: Asset de base (ex: BTC)
        quote_asset: Asset de cotation (ex: USDT)
        min_notional: Valeur minimum d'ordre
        min_qty: Quantité minimum
        step_size: Incrément de quantité
        tick_size: Incrément de prix
    """
    
    symbol: str
    base_asset: str
    quote_asset: str
    min_notional: float
    min_qty: float
    step_size: float
    tick_size: float
    
    def round_quantity(self, qty: float) -> float:
        """Arrondit la quantité selon step_size."""
        if self.step_size <= 0:
            return qty
        
        step = Decimal(str(self.step_size))
        quantity = Decimal(str(qty))
        rounded = (quantity // step) * step
        return float(rounded)
    
    def round_price(self, price: float) -> float:
        """Arrondit le prix selon tick_size."""
        if self.tick_size <= 0:
            return price
        
        tick = Decimal(str(self.tick_size))
        price_dec = Decimal(str(price))
        rounded = (price_dec // tick) * tick
        return float(rounded)
    
    def validate_order(self, qty: float, price: float) -> Tuple[bool, str]:
        """
        Valide un ordre selon les contraintes du symbole.
        
        Returns:
            (True, "") si valide, (False, "raison") sinon
        """
        notional = qty * price
        
        if qty < self.min_qty:
            return False, f"Quantité {qty} < minimum {self.min_qty}"
        
        if notional < self.min_notional:
            return False, f"Notional {notional:.2f} < minimum {self.min_notional}"
        
        return True, ""


@dataclass
class OrderResult:
    """
    Résultat d'un ordre exécuté.
    
    Attributes:
        order_id: ID de l'ordre sur Binance
        symbol: Paire tradée
        side: BUY ou SELL
        order_type: Type d'ordre
        status: Statut de l'ordre
        executed_qty: Quantité exécutée
        executed_price: Prix moyen d'exécution
        commission: Commission payée
        commission_asset: Asset de la commission
        timestamp: Timestamp de l'ordre
    """
    
    order_id: int
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    executed_qty: float
    executed_price: float
    commission: float = 0.0
    commission_asset: str = "USDT"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_filled(self) -> bool:
        """L'ordre est-il complètement rempli ?"""
        return self.status == OrderStatus.FILLED
    
    @property
    def notional_value(self) -> float:
        """Valeur totale de l'ordre."""
        return self.executed_qty * self.executed_price


@dataclass
class Position:
    """
    Représente une position ouverte.
    
    Attributes:
        position_id: ID unique de la position
        symbol: Paire tradée
        side: Côté (BUY = long)
        quantity: Quantité détenue
        entry_price: Prix d'entrée moyen
        entry_time: Timestamp d'entrée
        stop_loss_price: Prix du stop-loss
        take_profit_price: Prix du take-profit
        stop_loss_order_id: ID de l'ordre SL
        take_profit_order_id: ID de l'ordre TP
        status: Statut de la position
    """
    
    position_id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss_price: float
    take_profit_price: float
    stop_loss_order_id: Optional[int] = None
    take_profit_order_id: Optional[int] = None
    oco_order_id: Optional[int] = None  # ID de l'ordre OCO
    status: PositionStatus = PositionStatus.OPEN
    
    # Métadonnées
    signal_id: Optional[str] = None
    signal_score: Optional[float] = None
    
    @property
    def notional_value(self) -> float:
        """Valeur de la position au prix d'entrée."""
        return self.quantity * self.entry_price
    
    @property
    def age_seconds(self) -> int:
        """Âge de la position en secondes."""
        return int((datetime.utcnow() - self.entry_time).total_seconds())
    
    @property
    def is_open(self) -> bool:
        """La position est-elle ouverte ?"""
        return self.status == PositionStatus.OPEN
    
    def current_pnl(self, current_price: float) -> float:
        """Calcule le PnL non réalisé."""
        if self.side == OrderSide.BUY:
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity
    
    def current_pnl_percent(self, current_price: float) -> float:
        """Calcule le PnL non réalisé en pourcentage."""
        if self.entry_price == 0:
            return 0.0
        if self.side == OrderSide.BUY:
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price


@dataclass
class CompletedTrade:
    """
    Représente un trade terminé (position fermée).
    
    Attributes:
        trade_id: ID unique du trade
        symbol: Paire tradée
        side: Côté (BUY = long)
        entry_price: Prix d'entrée
        exit_price: Prix de sortie
        quantity: Quantité tradée
        entry_time: Timestamp d'entrée
        exit_time: Timestamp de sortie
        pnl_usdt: Profit/perte en USDT
        pnl_percent: Profit/perte en pourcentage
        close_reason: Raison de fermeture
        commission_total: Commission totale payée
    """
    
    trade_id: str
    symbol: str
    side: OrderSide
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl_usdt: float
    pnl_percent: float
    close_reason: CloseReason
    commission_total: float = 0.0
    
    # Métadonnées
    signal_id: Optional[str] = None
    signal_score: Optional[float] = None
    
    @property
    def duration_seconds(self) -> int:
        """Durée du trade en secondes."""
        return int((self.exit_time - self.entry_time).total_seconds())
    
    @property
    def is_winner(self) -> bool:
        """Le trade est-il gagnant ?"""
        return self.pnl_usdt > 0
    
    @classmethod
    def from_position(
        cls,
        position: Position,
        exit_price: float,
        exit_time: datetime,
        close_reason: CloseReason,
        commission: float = 0.0
    ) -> "CompletedTrade":
        """Crée un CompletedTrade depuis une Position."""
        pnl = position.current_pnl(exit_price)
        pnl_pct = position.current_pnl_percent(exit_price)
        
        return cls(
            trade_id=position.position_id,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=exit_time,
            pnl_usdt=pnl,
            pnl_percent=pnl_pct,
            close_reason=close_reason,
            commission_total=commission,
            signal_id=position.signal_id,
            signal_score=position.signal_score,
        )


# ============================================
# TRADE EXECUTOR
# ============================================

class TradeExecutor:
    """
    Exécute les trades sur Binance.
    
    Responsabilités:
    - Passer des ordres market BUY
    - Placer des ordres OCO (SL + TP)
    - Gérer les positions ouvertes
    - Fermer les positions
    - Synchroniser avec l'exchange
    
    Usage:
        executor = TradeExecutor(config)
        await executor.connect()
        
        # Exécuter un signal
        position = await executor.execute_signal(signal, size_usdt=10.0)
        
        # Fermer une position
        completed = await executor.close_position(position, reason="take_profit")
    """
    
    def __init__(self, config: Optional[ExecutorConfig] = None):
        """
        Initialise l'executor.
        
        Args:
            config: Configuration (utilise les défauts si None)
        """
        self._config = config or ExecutorConfig()
        self._settings = get_settings()
        
        # Client Binance
        self._client: Optional[AsyncClient] = None
        self._connected = False
        
        # Cache des informations symboles
        self._symbol_info_cache: Dict[str, SymbolInfo] = {}
        
        # Positions ouvertes
        self._open_positions: Dict[str, Position] = {}
        
        # Compteur pour IDs
        self._position_counter = 0
        
        # Stats
        self._orders_executed = 0
        self._orders_failed = 0
        
        logger.info(
            f"🔧 TradeExecutor initialisé | "
            f"Mode: {'PAPER' if self._config.paper_trading else 'LIVE'} | "
            f"Testnet: {self._config.testnet}"
        )
    
    # =========================================
    # PROPRIÉTÉS
    # =========================================
    
    @property
    def is_connected(self) -> bool:
        """L'executor est-il connecté ?"""
        return self._connected and self._client is not None
    
    @property
    def is_paper_trading(self) -> bool:
        """Mode paper trading actif ?"""
        return self._config.paper_trading
    
    @property
    def open_positions(self) -> List[Position]:
        """Liste des positions ouvertes."""
        return [p for p in self._open_positions.values() if p.is_open]
    
    @property
    def open_positions_count(self) -> int:
        """Nombre de positions ouvertes."""
        return len(self.open_positions)
    
    @property
    def orders_executed(self) -> int:
        """Nombre d'ordres exécutés."""
        return self._orders_executed
    
    # =========================================
    # CONNEXION
    # =========================================
    
    async def connect(self) -> None:
        """
        Établit la connexion avec Binance.
        
        Raises:
            APIConnectionError: Si la connexion échoue
        """
        if self._connected:
            logger.debug("Déjà connecté à Binance")
            return
        
        try:
            # Mode paper trading : pas de connexion réelle nécessaire
            # mais on se connecte quand même pour les infos de symboles
            if self._config.testnet:
                self._client = await AsyncClient.create(
                    api_key=self._settings.binance.api_key,
                    api_secret=self._settings.binance.api_secret,
                    testnet=True
                )
            else:
                self._client = await AsyncClient.create(
                    api_key=self._settings.binance.api_key,
                    api_secret=self._settings.binance.api_secret
                )
            
            self._connected = True
            
            mode = "TESTNET" if self._config.testnet else "PRODUCTION"
            paper = " (PAPER)" if self._config.paper_trading else ""
            logger.info(f"✅ TradeExecutor connecté à Binance ({mode}{paper})")
            
        except Exception as e:
            raise APIConnectionError(f"Impossible de se connecter: {e}")
    
    async def disconnect(self) -> None:
        """Ferme la connexion proprement."""
        if self._client:
            await self._client.close_connection()
            self._client = None
            self._connected = False
            logger.info("TradeExecutor déconnecté")
    
    async def __aenter__(self) -> "TradeExecutor":
        """Support du context manager async."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ferme la connexion à la sortie."""
        await self.disconnect()
    
    def _ensure_connected(self) -> None:
        """Vérifie que le client est connecté."""
        if not self.is_connected:
            raise APIConnectionError("Executor non connecté. Appelez connect() d'abord.")
    
    # =========================================
    # INFORMATIONS SYMBOLES
    # =========================================
    
    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """
        Récupère les informations d'un symbole.
        
        Args:
            symbol: Symbole de la paire (ex: "BTCUSDT")
            
        Returns:
            SymbolInfo avec les contraintes du symbole
        """
        self._ensure_connected()
        
        # Vérifier le cache
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]
        
        try:
            info = await self._client.get_symbol_info(symbol)
            
            if info is None:
                raise ValueError(f"Symbole {symbol} non trouvé")
            
            # Parser les filtres
            min_notional = 10.0  # Défaut
            min_qty = 0.0
            step_size = 0.0
            tick_size = 0.0
            
            for f in info.get("filters", []):
                filter_type = f.get("filterType")
                
                if filter_type == "NOTIONAL":
                    min_notional = float(f.get("minNotional", 10.0))
                elif filter_type == "MIN_NOTIONAL":
                    min_notional = float(f.get("minNotional", 10.0))
                elif filter_type == "LOT_SIZE":
                    min_qty = float(f.get("minQty", 0.0))
                    step_size = float(f.get("stepSize", 0.0))
                elif filter_type == "PRICE_FILTER":
                    tick_size = float(f.get("tickSize", 0.0))
            
            symbol_info = SymbolInfo(
                symbol=symbol,
                base_asset=info.get("baseAsset", ""),
                quote_asset=info.get("quoteAsset", ""),
                min_notional=min_notional,
                min_qty=min_qty,
                step_size=step_size,
                tick_size=tick_size,
            )
            
            # Mettre en cache
            self._symbol_info_cache[symbol] = symbol_info
            
            return symbol_info
            
        except BinanceAPIException as e:
            raise ValueError(f"Erreur récupération info symbole: {e.message}")
    
    # =========================================
    # EXÉCUTION DES ORDRES
    # =========================================
    
    async def execute_market_buy(
        self,
        symbol: str,
        quantity: float,
    ) -> OrderResult:
        """
        Exécute un ordre market BUY.
        
        Args:
            symbol: Paire de trading
            quantity: Quantité à acheter
            
        Returns:
            OrderResult avec les détails de l'exécution
            
        Raises:
            OrderExecutionError: Si l'ordre échoue
        """
        self._ensure_connected()
        
        # Récupérer les infos du symbole
        symbol_info = await self.get_symbol_info(symbol)
        
        # Arrondir la quantité
        quantity = symbol_info.round_quantity(quantity)
        
        logger.info(f"📤 Market BUY | {symbol} | Qty: {quantity}")
        
        # Mode paper trading
        if self._config.paper_trading:
            return await self._simulate_market_buy(symbol, quantity)
        
        # Exécution réelle
        try:
            order = await self._client.create_order(
                symbol=symbol,
                side="BUY",
                type="MARKET",
                quantity=quantity,
                recvWindow=self._config.default_recv_window,
            )
            
            result = self._parse_order_result(order, OrderSide.BUY, OrderType.MARKET)
            
            self._orders_executed += 1
            logger.info(
                f"✅ Market BUY exécuté | {symbol} | "
                f"Qty: {result.executed_qty} @ {result.executed_price:.2f}"
            )
            
            return result
            
        except BinanceAPIException as e:
            self._orders_failed += 1
            raise OrderExecutionError(
                f"Erreur ordre market BUY: {e.message}",
                symbol=symbol,
                error_code=e.code
            )
    
    async def execute_market_sell(
        self,
        symbol: str,
        quantity: float,
    ) -> OrderResult:
        """
        Exécute un ordre market SELL.
        
        Args:
            symbol: Paire de trading
            quantity: Quantité à vendre
            
        Returns:
            OrderResult avec les détails de l'exécution
        """
        self._ensure_connected()
        
        # Récupérer les infos du symbole
        symbol_info = await self.get_symbol_info(symbol)
        
        # Arrondir la quantité
        quantity = symbol_info.round_quantity(quantity)
        
        logger.info(f"📤 Market SELL | {symbol} | Qty: {quantity}")
        
        # Mode paper trading
        if self._config.paper_trading:
            return await self._simulate_market_sell(symbol, quantity)
        
        # Exécution réelle
        try:
            order = await self._client.create_order(
                symbol=symbol,
                side="SELL",
                type="MARKET",
                quantity=quantity,
                recvWindow=self._config.default_recv_window,
            )
            
            result = self._parse_order_result(order, OrderSide.SELL, OrderType.MARKET)
            
            self._orders_executed += 1
            logger.info(
                f"✅ Market SELL exécuté | {symbol} | "
                f"Qty: {result.executed_qty} @ {result.executed_price:.2f}"
            )
            
            return result
            
        except BinanceAPIException as e:
            self._orders_failed += 1
            raise OrderExecutionError(
                f"Erreur ordre market SELL: {e.message}",
                symbol=symbol,
                error_code=e.code
            )
    
    async def place_oco_order(
        self,
        symbol: str,
        quantity: float,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Place un ordre OCO (One-Cancels-Other) pour SL et TP.
        
        Un ordre OCO combine:
        - Un ordre LIMIT SELL au prix take_profit
        - Un ordre STOP_LOSS_LIMIT au prix stop_loss
        
        Args:
            symbol: Paire de trading
            quantity: Quantité à vendre
            stop_loss_price: Prix du stop-loss
            take_profit_price: Prix du take-profit
            
        Returns:
            Tuple (oco_order_id, stop_loss_order_id, take_profit_order_id)
        """
        self._ensure_connected()
        
        # Récupérer les infos du symbole
        symbol_info = await self.get_symbol_info(symbol)
        
        # Arrondir
        quantity = symbol_info.round_quantity(quantity)
        stop_loss_price = symbol_info.round_price(stop_loss_price)
        take_profit_price = symbol_info.round_price(take_profit_price)
        
        # Le stop limit trigger est légèrement au-dessus du stop loss
        stop_limit_price = symbol_info.round_price(stop_loss_price * 0.999)
        
        logger.info(
            f"📤 OCO Order | {symbol} | "
            f"SL: {stop_loss_price:.2f} | TP: {take_profit_price:.2f}"
        )
        
        # Mode paper trading
        if self._config.paper_trading:
            # Simuler les IDs
            fake_oco_id = 10000 + self._position_counter
            fake_sl_id = 10001 + self._position_counter
            fake_tp_id = 10002 + self._position_counter
            logger.info(f"📝 [PAPER] OCO simulé - IDs: {fake_oco_id}, {fake_sl_id}, {fake_tp_id}")
            return fake_oco_id, fake_sl_id, fake_tp_id
        
        # Exécution réelle
        try:
            oco_order = await self._client.create_oco_order(
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                price=str(take_profit_price),       # Prix TP (limit)
                stopPrice=str(stop_loss_price),     # Prix trigger SL
                stopLimitPrice=str(stop_limit_price),  # Prix limit SL
                stopLimitTimeInForce="GTC",
                recvWindow=self._config.default_recv_window,
            )
            
            # Parser la réponse
            oco_id = oco_order.get("orderListId")
            orders = oco_order.get("orderReports", [])
            
            sl_order_id = None
            tp_order_id = None
            
            for order in orders:
                if order.get("type") == "STOP_LOSS_LIMIT":
                    sl_order_id = order.get("orderId")
                elif order.get("type") == "LIMIT_MAKER":
                    tp_order_id = order.get("orderId")
            
            self._orders_executed += 1
            logger.info(
                f"✅ OCO placé | {symbol} | "
                f"OCO: {oco_id} | SL: {sl_order_id} | TP: {tp_order_id}"
            )
            
            return oco_id, sl_order_id, tp_order_id
            
        except BinanceAPIException as e:
            self._orders_failed += 1
            # L'OCO peut échouer si le prix est trop proche
            logger.error(f"❌ Erreur OCO: {e.message}")
            raise OrderExecutionError(
                f"Erreur placement OCO: {e.message}",
                symbol=symbol,
                error_code=e.code
            )
    
    async def cancel_order(
        self,
        symbol: str,
        order_id: int
    ) -> bool:
        """
        Annule un ordre.
        
        Args:
            symbol: Paire de trading
            order_id: ID de l'ordre à annuler
            
        Returns:
            True si annulé avec succès
        """
        self._ensure_connected()
        
        logger.info(f"📤 Cancel Order | {symbol} | ID: {order_id}")
        
        # Mode paper trading
        if self._config.paper_trading:
            logger.info(f"📝 [PAPER] Ordre {order_id} annulé")
            return True
        
        try:
            await self._client.cancel_order(
                symbol=symbol,
                orderId=order_id,
                recvWindow=self._config.default_recv_window,
            )
            logger.info(f"✅ Ordre {order_id} annulé")
            return True
            
        except BinanceAPIException as e:
            logger.error(f"❌ Erreur annulation ordre: {e.message}")
            return False
    
    async def cancel_oco_order(
        self,
        symbol: str,
        oco_order_id: int
    ) -> bool:
        """
        Annule un ordre OCO.
        
        Args:
            symbol: Paire de trading
            oco_order_id: ID de l'ordre OCO
            
        Returns:
            True si annulé avec succès
        """
        self._ensure_connected()
        
        logger.info(f"📤 Cancel OCO | {symbol} | ID: {oco_order_id}")
        
        # Mode paper trading
        if self._config.paper_trading:
            logger.info(f"📝 [PAPER] OCO {oco_order_id} annulé")
            return True
        
        try:
            await self._client.cancel_order(
                symbol=symbol,
                orderListId=oco_order_id,
                recvWindow=self._config.default_recv_window,
            )
            logger.info(f"✅ OCO {oco_order_id} annulé")
            return True
            
        except BinanceAPIException as e:
            logger.error(f"❌ Erreur annulation OCO: {e.message}")
            return False
    
    # =========================================
    # GESTION DES POSITIONS
    # =========================================
    
    async def execute_signal(
        self,
        signal,  # TradeSignal
        size_usdt: float,
        current_price: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Exécute un signal de trading complet.
        
        1. Place un ordre market BUY
        2. Place un ordre OCO (SL + TP)
        3. Crée et enregistre la position
        
        Args:
            signal: Signal de trading (TradeSignal)
            size_usdt: Taille de la position en USDT
            current_price: Prix actuel (optionnel, récupéré si None)
            
        Returns:
            Position si succès, None si échec
        """
        self._ensure_connected()
        
        symbol = signal.symbol
        
        # Récupérer le prix actuel si non fourni
        if current_price is None:
            current_price = await self._get_current_price(symbol)
        
        # Calculer la quantité
        symbol_info = await self.get_symbol_info(symbol)
        quantity = size_usdt / current_price
        quantity = symbol_info.round_quantity(quantity)
        
        # Valider l'ordre
        is_valid, reason = symbol_info.validate_order(quantity, current_price)
        if not is_valid:
            logger.error(f"❌ Ordre invalide: {reason}")
            return None
        
        logger.info(
            f"🚀 Exécution signal | {symbol} | "
            f"Size: {size_usdt:.2f} USDT | Score: {signal.score:.2f}"
        )
        
        try:
            # 1. Exécuter l'ordre market BUY
            buy_result = await self.execute_market_buy(symbol, quantity)
            
            if not buy_result.is_filled:
                logger.error(f"❌ Ordre BUY non rempli: {buy_result.status}")
                return None
            
            # 2. Calculer les prix SL/TP
            entry_price = buy_result.executed_price
            
            # Utiliser les prix du signal ou calculer depuis les pourcentages
            if signal.stop_loss_price and signal.stop_loss_price > 0:
                sl_price = signal.stop_loss_price
            else:
                sl_price = entry_price * (1 - signal.stop_loss_pct)
            
            if signal.take_profit_price and signal.take_profit_price > 0:
                tp_price = signal.take_profit_price
            else:
                tp_price = entry_price * (1 + signal.take_profit_pct)
            
            # Arrondir les prix
            sl_price = symbol_info.round_price(sl_price)
            tp_price = symbol_info.round_price(tp_price)
            
            # 3. Placer l'ordre OCO
            oco_id, sl_id, tp_id = await self.place_oco_order(
                symbol=symbol,
                quantity=buy_result.executed_qty,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
            )
            
            # 4. Créer la position
            self._position_counter += 1
            position_id = f"POS_{symbol}_{self._position_counter}"
            
            position = Position(
                position_id=position_id,
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=buy_result.executed_qty,
                entry_price=entry_price,
                entry_time=datetime.utcnow(),
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                stop_loss_order_id=sl_id,
                take_profit_order_id=tp_id,
                oco_order_id=oco_id,
                status=PositionStatus.OPEN,
                signal_id=getattr(signal, 'signal_id', None),
                signal_score=signal.score,
            )
            
            # Enregistrer la position
            self._open_positions[position_id] = position
            
            logger.info(
                f"✅ Position ouverte | {position_id} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}"
            )
            
            return position
            
        except OrderExecutionError as e:
            logger.error(f"❌ Erreur exécution signal: {e}")
            return None
        except Exception as e:
            logger.exception(f"❌ Erreur inattendue: {e}")
            return None
    
    async def close_position(
        self,
        position: Position,
        reason: CloseReason,
        current_price: Optional[float] = None,
    ) -> Optional[CompletedTrade]:
        """
        Ferme une position.
        
        1. Annule les ordres OCO en cours
        2. Exécute un ordre market SELL
        3. Crée le CompletedTrade
        
        Args:
            position: Position à fermer
            reason: Raison de fermeture
            current_price: Prix de sortie (optionnel)
            
        Returns:
            CompletedTrade si succès, None si échec
        """
        self._ensure_connected()
        
        if not position.is_open:
            logger.warning(f"Position {position.position_id} déjà fermée")
            return None
        
        position.status = PositionStatus.CLOSING
        
        logger.info(
            f"📤 Fermeture position | {position.position_id} | "
            f"Raison: {reason.value}"
        )
        
        try:
            # 1. Annuler l'OCO si présent
            if position.oco_order_id:
                await self.cancel_oco_order(position.symbol, position.oco_order_id)
            
            # 2. Vendre au marché
            sell_result = await self.execute_market_sell(
                symbol=position.symbol,
                quantity=position.quantity,
            )
            
            exit_price = sell_result.executed_price if sell_result.is_filled else current_price
            
            if exit_price is None:
                exit_price = await self._get_current_price(position.symbol)
            
            # 3. Créer le CompletedTrade
            completed = CompletedTrade.from_position(
                position=position,
                exit_price=exit_price,
                exit_time=datetime.utcnow(),
                close_reason=reason,
                commission=sell_result.commission if sell_result else 0.0,
            )
            
            # 4. Mettre à jour le statut
            position.status = PositionStatus.CLOSED
            
            # Retirer de la liste des positions ouvertes
            if position.position_id in self._open_positions:
                del self._open_positions[position.position_id]
            
            emoji = "🎉" if completed.is_winner else "😤"
            logger.info(
                f"{emoji} Position fermée | {position.position_id} | "
                f"PnL: {completed.pnl_usdt:+.4f} USDT ({completed.pnl_percent:+.2%}) | "
                f"Durée: {completed.duration_seconds}s"
            )
            
            return completed
            
        except Exception as e:
            logger.exception(f"❌ Erreur fermeture position: {e}")
            position.status = PositionStatus.OPEN
            return None
    
    async def close_all_positions(
        self,
        reason: CloseReason = CloseReason.KILL_SWITCH
    ) -> List[CompletedTrade]:
        """
        Ferme toutes les positions ouvertes.
        
        Args:
            reason: Raison de fermeture
            
        Returns:
            Liste des CompletedTrade
        """
        completed_trades = []
        
        positions_to_close = list(self._open_positions.values())
        
        if not positions_to_close:
            logger.info("Aucune position ouverte à fermer")
            return completed_trades
        
        logger.warning(
            f"⚠️ Fermeture de {len(positions_to_close)} positions | "
            f"Raison: {reason.value}"
        )
        
        for position in positions_to_close:
            completed = await self.close_position(position, reason)
            if completed:
                completed_trades.append(completed)
        
        return completed_trades
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """
        Récupère une position par son ID.
        
        Args:
            position_id: ID de la position
            
        Returns:
            Position ou None
        """
        return self._open_positions.get(position_id)
    
    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """
        Récupère la position ouverte pour un symbole.
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            Position ou None
        """
        for position in self._open_positions.values():
            if position.symbol == symbol and position.is_open:
                return position
        return None
    
    # =========================================
    # SYNCHRONISATION AVEC L'EXCHANGE
    # =========================================
    
    async def sync_with_exchange(self) -> None:
        """
        Synchronise l'état local avec Binance.
        
        Vérifie si des ordres OCO ont été exécutés
        et met à jour les positions en conséquence.
        """
        self._ensure_connected()
        
        if self._config.paper_trading:
            return  # Pas de sync en paper trading
        
        for position in list(self._open_positions.values()):
            if not position.is_open:
                continue
            
            try:
                # Vérifier le statut de l'ordre OCO
                if position.oco_order_id:
                    oco_status = await self._check_oco_status(
                        position.symbol,
                        position.oco_order_id
                    )
                    
                    if oco_status == "ALL_DONE":
                        # Déterminer quel ordre a été exécuté
                        close_reason = await self._determine_close_reason(position)
                        await self.close_position(position, close_reason)
                        
            except Exception as e:
                logger.error(f"Erreur sync position {position.position_id}: {e}")
    
    async def _check_oco_status(self, symbol: str, oco_id: int) -> str:
        """Vérifie le statut d'un ordre OCO."""
        try:
            # Note: Cette méthode peut nécessiter une adaptation selon l'API
            order_list = await self._client.get_order_list(orderListId=oco_id)
            return order_list.get("listOrderStatus", "UNKNOWN")
        except Exception:
            return "UNKNOWN"
    
    async def _determine_close_reason(self, position: Position) -> CloseReason:
        """Détermine la raison de fermeture d'une position."""
        try:
            current_price = await self._get_current_price(position.symbol)
            
            # Si le prix est proche du TP
            if current_price >= position.take_profit_price * 0.999:
                return CloseReason.TAKE_PROFIT
            
            # Si le prix est proche du SL
            if current_price <= position.stop_loss_price * 1.001:
                return CloseReason.STOP_LOSS
            
            return CloseReason.MANUAL
            
        except Exception:
            return CloseReason.MANUAL
    
    # =========================================
    # MÉTHODES PRIVÉES
    # =========================================
    
    def _parse_order_result(
        self,
        order: dict,
        side: OrderSide,
        order_type: OrderType
    ) -> OrderResult:
        """Parse la réponse d'un ordre Binance."""
        # Calculer le prix moyen d'exécution
        fills = order.get("fills", [])
        
        if fills:
            total_qty = sum(float(f["qty"]) for f in fills)
            total_value = sum(float(f["qty"]) * float(f["price"]) for f in fills)
            avg_price = total_value / total_qty if total_qty > 0 else 0.0
            
            # Commission totale
            commission = sum(float(f.get("commission", 0)) for f in fills)
            commission_asset = fills[0].get("commissionAsset", "USDT") if fills else "USDT"
        else:
            avg_price = float(order.get("price", 0))
            commission = 0.0
            commission_asset = "USDT"
        
        return OrderResult(
            order_id=order.get("orderId"),
            symbol=order.get("symbol"),
            side=side,
            order_type=order_type,
            status=OrderStatus(order.get("status", "NEW")),
            executed_qty=float(order.get("executedQty", 0)),
            executed_price=avg_price,
            commission=commission,
            commission_asset=commission_asset,
            timestamp=datetime.utcnow(),
        )
    
    async def _get_current_price(self, symbol: str) -> float:
        """Récupère le prix actuel d'un symbole."""
        try:
            ticker = await self._client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception:
            raise ValueError(f"Impossible de récupérer le prix de {symbol}")
    
    async def _simulate_market_buy(self, symbol: str, quantity: float) -> OrderResult:
        """Simule un ordre market BUY en paper trading."""
        # Récupérer le prix actuel
        current_price = await self._get_current_price(symbol)
        
        # Simuler un léger slippage (0.05%)
        executed_price = current_price * 1.0005
        
        # Simuler les frais (0.1%)
        commission = quantity * executed_price * 0.001
        
        self._position_counter += 1
        
        result = OrderResult(
            order_id=self._position_counter * 1000,
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            executed_qty=quantity,
            executed_price=executed_price,
            commission=commission,
            commission_asset="USDT",
        )
        
        logger.info(
            f"📝 [PAPER] Market BUY | {symbol} | "
            f"Qty: {quantity} @ {executed_price:.2f}"
        )
        
        return result
    
    async def _simulate_market_sell(self, symbol: str, quantity: float) -> OrderResult:
        """Simule un ordre market SELL en paper trading."""
        # Récupérer le prix actuel
        current_price = await self._get_current_price(symbol)
        
        # Simuler un léger slippage négatif (-0.05%)
        executed_price = current_price * 0.9995
        
        # Simuler les frais (0.1%)
        commission = quantity * executed_price * 0.001
        
        self._position_counter += 1
        
        result = OrderResult(
            order_id=self._position_counter * 1000,
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            executed_qty=quantity,
            executed_price=executed_price,
            commission=commission,
            commission_asset="USDT",
        )
        
        logger.info(
            f"📝 [PAPER] Market SELL | {symbol} | "
            f"Qty: {quantity} @ {executed_price:.2f}"
        )
        
        return result


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def create_executor(
    testnet: bool = True,
    paper_trading: bool = False
) -> TradeExecutor:
    """
    Crée un TradeExecutor avec les paramètres par défaut.
    
    Args:
        testnet: Utiliser le testnet
        paper_trading: Mode paper trading
        
    Returns:
        TradeExecutor configuré
    """
    config = ExecutorConfig(
        testnet=testnet,
        paper_trading=paper_trading,
    )
    return TradeExecutor(config)


__all__ = [
    # Enums
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "PositionStatus",
    "CloseReason",
    # Dataclasses
    "ExecutorConfig",
    "SymbolInfo",
    "OrderResult",
    "Position",
    "CompletedTrade",
    # Classes
    "TradeExecutor",
    # Functions
    "create_executor",
]