# cryptoscalper/data/websocket_manager.py
"""
Gestionnaire WebSocket pour la collecte de données temps réel.

Gère les connexions WebSocket à Binance pour recevoir :
- Prix en temps réel (mini ticker)
- Klines/bougies en temps réel
- Orderbook en temps réel
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import deque

from binance import AsyncClient, BinanceSocketManager

from cryptoscalper.config.settings import get_settings
from cryptoscalper.config.constants import KLINE_INTERVAL_1M
from cryptoscalper.utils.logger import logger
from cryptoscalper.utils.exceptions import WebSocketConnectionError


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
class PairState:
    """
    État temps réel d'une paire.
    
    Maintient un historique glissant des prix.
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
        # (moins de 30 secondes d'écart)
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


@dataclass
class WebSocketStats:
    """Statistiques du WebSocket."""
    
    messages_received: int = 0
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
    
    Gère la collecte de données temps réel pour plusieurs paires.
    
    Usage:
        manager = WebSocketManager(client)
        await manager.start(["BTCUSDT", "ETHUSDT"])
        
        # Récupérer l'état d'une paire
        state = manager.get_pair_state("BTCUSDT")
        print(f"BTC: {state.current_price}")
        
        await manager.stop()
    """
    
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
        
        # État des paires
        self._pairs_state: Dict[str, PairState] = {}
        
        # Callbacks
        self._on_ticker_callback: Optional[Callable[[TickerData], None]] = None
        self._on_kline_callback: Optional[Callable[[KlineData], None]] = None
        
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
        """
        Récupère l'état d'une paire.
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            PairState ou None si non surveillée
        """
        return self._pairs_state.get(symbol)
    
    def get_all_states(self) -> Dict[str, PairState]:
        """Retourne l'état de toutes les paires."""
        return self._pairs_state.copy()
    
    def on_ticker(self, callback: Callable[[TickerData], None]) -> None:
        """
        Enregistre un callback pour les tickers.
        
        Args:
            callback: Fonction appelée à chaque ticker reçu
        """
        self._on_ticker_callback = callback
    
    def on_kline(self, callback: Callable[[KlineData], None]) -> None:
        """
        Enregistre un callback pour les klines.
        
        Args:
            callback: Fonction appelée à chaque kline reçue
        """
        self._on_kline_callback = callback
    
    async def start(
        self,
        symbols: List[str],
        subscribe_klines: bool = False,
        kline_interval: str = KLINE_INTERVAL_1M
    ) -> None:
        """
        Démarre la collecte temps réel.
        
        Args:
            symbols: Liste des symboles à surveiller
            subscribe_klines: Aussi souscrire aux klines
            kline_interval: Intervalle des klines
        """
        if self._running:
            logger.warning("WebSocket déjà en cours d'exécution")
            return
        
        logger.info(f"🔌 Démarrage WebSocket pour {len(symbols)} paires...")
        
        # Initialiser les états
        for symbol in symbols:
            self._pairs_state[symbol] = PairState(symbol=symbol)
        
        # Créer le socket manager
        self._bsm = BinanceSocketManager(self._client)
        
        # Démarrer les streams
        self._running = True
        self.stats.start_time = datetime.now()
        
        # Stream des mini tickers (tous les symboles)
        ticker_socket = self._bsm.multiplex_socket([
            f"{s.lower()}@miniTicker" for s in symbols
        ])
        self._tasks.append(asyncio.create_task(
            self._handle_ticker_stream(ticker_socket)
        ))
        
        # Stream des klines (optionnel)
        if subscribe_klines:
            kline_socket = self._bsm.multiplex_socket([
                f"{s.lower()}@kline_{kline_interval}" for s in symbols
            ])
            self._tasks.append(asyncio.create_task(
                self._handle_kline_stream(kline_socket)
            ))
        
        logger.info(f"✅ WebSocket démarré ({len(symbols)} paires)")
    
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
    
    async def _handle_ticker_stream(self, socket) -> None:
        """Gère le stream des tickers."""
        try:
            async with socket as stream:
                while self._running:
                    try:
                        msg = await asyncio.wait_for(stream.recv(), timeout=30.0)
                        self._process_ticker_message(msg)
                    except asyncio.TimeoutError:
                        logger.warning("Timeout WebSocket ticker, reconnexion...")
                        self.stats.reconnections += 1
                    except Exception as e:
                        logger.error(f"Erreur ticker stream: {e}")
                        self.stats.errors_count += 1
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Erreur fatale ticker stream: {e}")
    
    async def _handle_kline_stream(self, socket) -> None:
        """Gère le stream des klines."""
        try:
            async with socket as stream:
                while self._running:
                    try:
                        msg = await asyncio.wait_for(stream.recv(), timeout=30.0)
                        self._process_kline_message(msg)
                    except asyncio.TimeoutError:
                        logger.warning("Timeout WebSocket kline, reconnexion...")
                        self.stats.reconnections += 1
                    except Exception as e:
                        logger.error(f"Erreur kline stream: {e}")
                        self.stats.errors_count += 1
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Erreur fatale kline stream: {e}")
    
    def _process_ticker_message(self, msg: dict) -> None:
        """Traite un message ticker."""
        if "data" not in msg:
            return
        
        data = msg["data"]
        symbol = data.get("s", "")
        
        if symbol not in self._pairs_state:
            return
        
        # Mettre à jour l'état
        state = self._pairs_state[symbol]
        price = float(data.get("c", 0))
        state.update_price(price)
        state.high_24h = float(data.get("h", 0))
        state.low_24h = float(data.get("l", 0))
        state.volume_24h = float(data.get("v", 0))
        
        # Stats
        self.stats.messages_received += 1
        self.stats.last_message_time = datetime.now()
        
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
    
    def _process_kline_message(self, msg: dict) -> None:
        """Traite un message kline."""
        if "data" not in msg:
            return
        
        data = msg["data"]
        kline = data.get("k", {})
        symbol = kline.get("s", "")
        
        if symbol not in self._pairs_state:
            return
        
        # Stats
        self.stats.messages_received += 1
        self.stats.last_message_time = datetime.now()
        
        # Callback
        if self._on_kline_callback:
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
            self._on_kline_callback(kline_data)
    
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
            
            # Essayer d'abord le timeframe demandé
            if timeframe == "1m":
                change = state.change_1m_percent
            elif timeframe == "5m":
                change = state.change_5m_percent
            
            # Fallback sur la variation depuis le début si pas assez de données
            if change is None:
                change = state.change_since_start_percent
            
            if change is not None:
                pairs_with_change.append((state, abs(change)))
        
        # Trier par mouvement absolu décroissant
        pairs_with_change.sort(key=lambda x: x[1], reverse=True)
        
        return [pair for pair, _ in pairs_with_change[:n]]