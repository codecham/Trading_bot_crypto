# cryptoscalper/data/websocket_manager.py
"""
Gestionnaire WebSocket pour la collecte de données temps réel.

Gère les connexions WebSocket à Binance pour recevoir :
- Prix en temps réel (mini ticker)
- Klines/bougies en temps réel
- Orderbook en temps réel

Fonctionnalités :
- Reconnexion automatique en cas de déconnexion
- Buffer circulaire pour historique des prix
- Callbacks pour traitement personnalisé
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import deque
from enum import Enum

from binance import AsyncClient, BinanceSocketManager

from cryptoscalper.config.settings import get_settings
from cryptoscalper.config.constants import KLINE_INTERVAL_1M
from cryptoscalper.utils.logger import logger
from cryptoscalper.utils.exceptions import WebSocketConnectionError


class StreamType(Enum):
    """Types de streams WebSocket disponibles."""
    TICKER = "ticker"
    KLINE = "kline"
    DEPTH = "depth"


@dataclass
class TickerData:
    """Données ticker temps réel."""
    
    symbol: str
    price: float
    price_change_percent: float
    high_24h: float
    low_24h: float
    volume_24h: float
    quote_volume_24h: float
    timestamp: datetime


@dataclass
class KlineData:
    """Données kline temps réel."""
    
    symbol: str
    interval: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool
    timestamp: datetime


@dataclass
class DepthData:
    """Données orderbook temps réel."""
    
    symbol: str
    bids: List[tuple]  # [(price, quantity), ...]
    asks: List[tuple]  # [(price, quantity), ...]
    timestamp: datetime
    
    @property
    def best_bid(self) -> Optional[float]:
        """Meilleur prix d'achat."""
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Meilleur prix de vente."""
        return self.asks[0][0] if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        """Spread en valeur absolue."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_percent(self) -> Optional[float]:
        """Spread en pourcentage."""
        if self.best_bid and self.spread:
            return (self.spread / self.best_bid) * 100
        return None
    
    @property
    def bid_volume(self) -> float:
        """Volume total des bids."""
        return sum(qty for _, qty in self.bids)
    
    @property
    def ask_volume(self) -> float:
        """Volume total des asks."""
        return sum(qty for _, qty in self.asks)
    
    @property
    def imbalance(self) -> float:
        """
        Déséquilibre bid/ask.
        > 0 = plus de pression acheteuse
        < 0 = plus de pression vendeuse
        """
        total = self.bid_volume + self.ask_volume
        if total == 0:
            return 0.0
        return (self.bid_volume - self.ask_volume) / total


@dataclass
class PairState:
    """
    État temps réel d'une paire.
    
    Maintient un historique glissant des prix et l'orderbook.
    """
    
    symbol: str
    current_price: float = 0.0
    price_change_percent: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    volume_24h: float = 0.0
    last_update: Optional[datetime] = None
    
    # Historique des prix (5 dernières minutes max)
    price_history: deque = field(default_factory=lambda: deque(maxlen=300))
    
    # Dernière kline
    current_kline: Optional[KlineData] = None
    
    # Dernier orderbook
    current_depth: Optional[DepthData] = None
    
    def _get_price_at_time(self, seconds_ago: int) -> Optional[float]:
        """
        Récupère le prix il y a N secondes.
        
        Args:
            seconds_ago: Nombre de secondes dans le passé
            
        Returns:
            Prix le plus proche de ce moment, ou None si pas de données
        """
        if not self.price_history:
            return None
        
        target_time = datetime.now() - timedelta(seconds=seconds_ago)
        
        # Chercher le prix le plus proche du timestamp cible
        closest_price = None
        min_diff = float('inf')
        
        for timestamp, price in self.price_history:
            diff = abs((timestamp - target_time).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_price = price
        
        # Retourner seulement si on a trouvé un prix dans une fenêtre raisonnable
        if min_diff < 30:
            return closest_price
        return None
    
    @property
    def price_1m_ago(self) -> Optional[float]:
        """Prix il y a ~1 minute."""
        return self._get_price_at_time(60)
    
    @property
    def price_5m_ago(self) -> Optional[float]:
        """Prix il y a ~5 minutes."""
        return self._get_price_at_time(300)
    
    @property
    def change_1m_percent(self) -> Optional[float]:
        """Variation sur 1 minute."""
        price_ago = self.price_1m_ago
        if price_ago and self.current_price and price_ago > 0:
            return (self.current_price - price_ago) / price_ago * 100
        return None
    
    @property
    def change_5m_percent(self) -> Optional[float]:
        """Variation sur 5 minutes."""
        price_ago = self.price_5m_ago
        if price_ago and self.current_price and price_ago > 0:
            return (self.current_price - price_ago) / price_ago * 100
        return None
    
    @property
    def change_since_start_percent(self) -> Optional[float]:
        """Variation depuis le début de la collecte."""
        if not self.price_history or not self.current_price:
            return None
        first_price = self.price_history[0][1]
        if first_price > 0:
            return (self.current_price - first_price) / first_price * 100
        return None
    
    def update_price(self, price: float) -> None:
        """Met à jour le prix et l'historique."""
        self.current_price = price
        self.last_update = datetime.now()
        self.price_history.append((self.last_update, price))
    
    def update_depth(self, depth: DepthData) -> None:
        """Met à jour l'orderbook."""
        self.current_depth = depth
    
    def update_kline(self, kline: KlineData) -> None:
        """Met à jour la kline courante."""
        self.current_kline = kline


@dataclass
class WebSocketStats:
    """Statistiques du WebSocket."""
    
    messages_received: int = 0
    ticker_messages: int = 0
    kline_messages: int = 0
    depth_messages: int = 0
    errors_count: int = 0
    reconnections: int = 0
    start_time: Optional[datetime] = None
    last_message_time: Optional[datetime] = None
    
    @property
    def uptime_seconds(self) -> float:
        """Durée depuis le démarrage."""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0
    
    @property
    def messages_per_second(self) -> float:
        """Taux de messages par seconde."""
        if self.uptime_seconds > 0:
            return self.messages_received / self.uptime_seconds
        return 0.0


class WebSocketManager:
    """
    Gestionnaire de connexions WebSocket Binance.
    
    Gère la collecte de données temps réel pour plusieurs paires
    avec reconnexion automatique.
    
    Usage:
        manager = WebSocketManager(client)
        await manager.start(["BTCUSDT", "ETHUSDT"])
        
        # Récupérer l'état d'une paire
        state = manager.get_pair_state("BTCUSDT")
        print(f"BTC: {state.current_price}")
        
        await manager.stop()
    """
    
    # Constantes pour la reconnexion
    RECONNECT_DELAY_INITIAL = 1.0  # Délai initial en secondes
    RECONNECT_DELAY_MAX = 60.0  # Délai maximum
    RECONNECT_DELAY_MULTIPLIER = 2.0  # Multiplicateur pour backoff exponentiel
    
    def __init__(self, client: AsyncClient):
        """
        Initialise le gestionnaire.
        
        Args:
            client: Client Binance async connecté
        """
        self._client = client
        self._bsm: Optional[BinanceSocketManager] = None
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._reconnect_delay = self.RECONNECT_DELAY_INITIAL
        
        # Configuration des streams
        self._symbols: List[str] = []
        self._subscribe_klines = False
        self._subscribe_depth = False
        self._kline_interval = KLINE_INTERVAL_1M
        self._depth_level = 10
        
        # État des paires
        self._pairs_state: Dict[str, PairState] = {}
        
        # Callbacks
        self._on_ticker_callback: Optional[Callable[[TickerData], None]] = None
        self._on_kline_callback: Optional[Callable[[KlineData], None]] = None
        self._on_depth_callback: Optional[Callable[[DepthData], None]] = None
        
        # Stats
        self.stats = WebSocketStats()
    
    @property
    def is_running(self) -> bool:
        """Vérifie si le WebSocket est actif."""
        return self._running
    
    @property
    def symbols(self) -> List[str]:
        """Liste des symboles surveillés."""
        return list(self._pairs_state.keys())
    
    def get_pair_state(self, symbol: str) -> Optional[PairState]:
        """Récupère l'état d'une paire."""
        return self._pairs_state.get(symbol)
    
    def get_all_states(self) -> Dict[str, PairState]:
        """Retourne l'état de toutes les paires."""
        return self._pairs_state.copy()
    
    def on_ticker(self, callback: Callable[[TickerData], None]) -> None:
        """Enregistre un callback pour les tickers."""
        self._on_ticker_callback = callback
    
    def on_kline(self, callback: Callable[[KlineData], None]) -> None:
        """Enregistre un callback pour les klines."""
        self._on_kline_callback = callback
    
    def on_depth(self, callback: Callable[[DepthData], None]) -> None:
        """Enregistre un callback pour l'orderbook."""
        self._on_depth_callback = callback
    
    async def start(
        self,
        symbols: List[str],
        subscribe_klines: bool = False,
        subscribe_depth: bool = False,
        kline_interval: str = KLINE_INTERVAL_1M,
        depth_level: int = 10
    ) -> None:
        """
        Démarre la collecte temps réel.
        
        Args:
            symbols: Liste des symboles à surveiller
            subscribe_klines: Souscrire aux klines
            subscribe_depth: Souscrire à l'orderbook
            kline_interval: Intervalle des klines
            depth_level: Profondeur de l'orderbook (5, 10, 20)
        """
        if self._running:
            logger.warning("WebSocket déjà en cours d'exécution")
            return
        
        # Sauvegarder la configuration pour reconnexion
        self._symbols = symbols
        self._subscribe_klines = subscribe_klines
        self._subscribe_depth = subscribe_depth
        self._kline_interval = kline_interval
        self._depth_level = depth_level
        
        logger.info(f"🔌 Démarrage WebSocket pour {len(symbols)} paires...")
        
        # Initialiser les états
        for symbol in symbols:
            self._pairs_state[symbol] = PairState(symbol=symbol)
        
        # Stats
        self._running = True
        self.stats = WebSocketStats(start_time=datetime.now())
        self._reconnect_delay = self.RECONNECT_DELAY_INITIAL
        
        # Démarrer les streams
        await self._start_streams()
        
        logger.info(f"✅ WebSocket démarré ({len(symbols)} paires)")
    
    async def _start_streams(self) -> None:
        """Démarre tous les streams configurés."""
        # Créer le socket manager
        self._bsm = BinanceSocketManager(self._client)
        
        # Stream des mini tickers
        streams = [f"{s.lower()}@miniTicker" for s in self._symbols]
        
        # Ajouter klines si demandé
        if self._subscribe_klines:
            streams.extend([
                f"{s.lower()}@kline_{self._kline_interval}" 
                for s in self._symbols
            ])
        
        # Ajouter depth si demandé
        if self._subscribe_depth:
            streams.extend([
                f"{s.lower()}@depth{self._depth_level}@1000ms" 
                for s in self._symbols
            ])
        
        # Créer le multiplex socket
        socket = self._bsm.multiplex_socket(streams)
        self._tasks.append(asyncio.create_task(
            self._handle_stream(socket)
        ))
    
    async def stop(self) -> None:
        """Arrête proprement le WebSocket."""
        if not self._running:
            return
        
        logger.info("🔌 Arrêt du WebSocket...")
        self._running = False
        
        # Annuler les tâches
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        logger.info("✅ WebSocket arrêté")
    
    async def _handle_stream(self, socket) -> None:
        """
        Gère le stream multiplexé avec reconnexion automatique.
        """
        while self._running:
            try:
                async with socket as stream:
                    logger.debug("Stream connecté")
                    self._reconnect_delay = self.RECONNECT_DELAY_INITIAL
                    
                    while self._running:
                        try:
                            msg = await asyncio.wait_for(
                                stream.recv(), 
                                timeout=30.0
                            )
                            self._process_message(msg)
                        except asyncio.TimeoutError:
                            # Pas de message depuis 30s, vérifier la connexion
                            logger.debug("Timeout WebSocket, vérification...")
                            continue
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break
                    
                self.stats.errors_count += 1
                logger.error(f"Erreur WebSocket: {e}")
                
                # Reconnexion avec backoff exponentiel
                await self._reconnect()
    
    async def _reconnect(self) -> None:
        """Gère la reconnexion avec backoff exponentiel."""
        if not self._running:
            return
        
        self.stats.reconnections += 1
        logger.warning(
            f"Reconnexion dans {self._reconnect_delay:.1f}s "
            f"(tentative #{self.stats.reconnections})"
        )
        
        await asyncio.sleep(self._reconnect_delay)
        
        # Augmenter le délai pour la prochaine fois
        self._reconnect_delay = min(
            self._reconnect_delay * self.RECONNECT_DELAY_MULTIPLIER,
            self.RECONNECT_DELAY_MAX
        )
        
        # Recréer le socket manager
        try:
            self._bsm = BinanceSocketManager(self._client)
            logger.info("Reconnexion réussie")
        except Exception as e:
            logger.error(f"Échec reconnexion: {e}")
    
    def _process_message(self, msg: dict) -> None:
        """Distribue le message au bon handler."""
        if "data" not in msg:
            return
        
        self.stats.messages_received += 1
        self.stats.last_message_time = datetime.now()
        
        # Identifier le type de stream
        stream = msg.get("stream", "")
        data = msg["data"]
        
        if "@miniTicker" in stream:
            self._process_ticker_message(data)
        elif "@kline" in stream:
            self._process_kline_message(data)
        elif "@depth" in stream:
            self._process_depth_message(data, stream)
    
    def _process_ticker_message(self, data: dict) -> None:
        """Traite un message ticker."""
        symbol = data.get("s", "")
        
        if symbol not in self._pairs_state:
            return
        
        self.stats.ticker_messages += 1
        
        # Mettre à jour l'état
        state = self._pairs_state[symbol]
        price = float(data.get("c", 0))
        state.update_price(price)
        state.high_24h = float(data.get("h", 0))
        state.low_24h = float(data.get("l", 0))
        state.volume_24h = float(data.get("v", 0))
        
        # Callback
        if self._on_ticker_callback:
            ticker_data = TickerData(
                symbol=symbol,
                price=price,
                price_change_percent=float(data.get("p", 0)),
                high_24h=state.high_24h,
                low_24h=state.low_24h,
                volume_24h=state.volume_24h,
                quote_volume_24h=float(data.get("q", 0)),
                timestamp=datetime.now()
            )
            self._on_ticker_callback(ticker_data)
    
    def _process_kline_message(self, data: dict) -> None:
        """Traite un message kline."""
        kline = data.get("k", {})
        symbol = kline.get("s", "")
        
        if symbol not in self._pairs_state:
            return
        
        self.stats.kline_messages += 1
        
        kline_data = KlineData(
            symbol=symbol,
            interval=kline.get("i", ""),
            open_time=datetime.fromtimestamp(kline.get("t", 0) / 1000),
            open=float(kline.get("o", 0)),
            high=float(kline.get("h", 0)),
            low=float(kline.get("l", 0)),
            close=float(kline.get("c", 0)),
            volume=float(kline.get("v", 0)),
            is_closed=kline.get("x", False),
            timestamp=datetime.now()
        )
        
        # Mettre à jour l'état
        self._pairs_state[symbol].update_kline(kline_data)
        
        # Callback
        if self._on_kline_callback:
            self._on_kline_callback(kline_data)
    
    def _process_depth_message(self, data: dict, stream: str) -> None:
        """Traite un message orderbook."""
        # Extraire le symbole du stream name (ex: "btcusdt@depth10@1000ms")
        symbol = stream.split("@")[0].upper()
        
        if symbol not in self._pairs_state:
            return
        
        self.stats.depth_messages += 1
        
        depth_data = DepthData(
            symbol=symbol,
            bids=[(float(b[0]), float(b[1])) for b in data.get("bids", [])],
            asks=[(float(a[0]), float(a[1])) for a in data.get("asks", [])],
            timestamp=datetime.now()
        )
        
        # Mettre à jour l'état
        self._pairs_state[symbol].update_depth(depth_data)
        
        # Callback
        if self._on_depth_callback:
            self._on_depth_callback(depth_data)
    
    def get_top_movers(self, n: int = 10, timeframe: str = "1m") -> List[PairState]:
        """
        Retourne les paires avec le plus de mouvement.
        
        Args:
            n: Nombre de paires à retourner
            timeframe: "1m", "5m" ou "start" (depuis début collecte)
            
        Returns:
            Liste des paires triées par mouvement décroissant
        """
        pairs_with_change = []
        
        for state in self._pairs_state.values():
            change = None
            
            if timeframe == "1m":
                change = state.change_1m_percent
            elif timeframe == "5m":
                change = state.change_5m_percent
            
            # Fallback sur la variation depuis le début
            if change is None:
                change = state.change_since_start_percent
            
            if change is not None:
                pairs_with_change.append((state, abs(change)))
        
        pairs_with_change.sort(key=lambda x: x[1], reverse=True)
        return [pair for pair, _ in pairs_with_change[:n]]