# cryptoscalper/data/collector.py
"""
Data Collector - Interface unifiée pour la collecte de données.

Combine les données REST et WebSocket en une seule interface simple.
Fournit un accès unifié à :
- Prix temps réel
- Klines (bougies)
- Orderbook
- Historique des prix
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from binance import AsyncClient

from cryptoscalper.data.binance_client import BinanceClient, Kline, OrderBook
from cryptoscalper.data.websocket_manager import (
    WebSocketManager,
    PairState,
    TickerData,
    KlineData,
    DepthData,
)
from cryptoscalper.config.constants import KLINE_INTERVAL_1M
from cryptoscalper.utils.logger import logger


@dataclass
class CollectorConfig:
    """Configuration du Data Collector."""
    
    # Symboles à surveiller
    symbols: List[str]
    
    # Options WebSocket
    enable_websocket: bool = True
    subscribe_klines: bool = True
    subscribe_depth: bool = True
    kline_interval: str = KLINE_INTERVAL_1M
    depth_level: int = 10
    
    # Options de données
    use_production_data: bool = True  # Données live de production


@dataclass
class CollectorStats:
    """Statistiques du collecteur."""
    
    start_time: Optional[datetime] = None
    symbols_count: int = 0
    websocket_running: bool = False
    total_updates: int = 0
    
    @property
    def uptime_seconds(self) -> float:
        """Temps depuis le démarrage."""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0


class DataCollector:
    """
    Interface unifiée pour la collecte de données.
    
    Combine :
    - REST API pour données historiques et ponctuelles
    - WebSocket pour données temps réel
    
    Usage:
        config = CollectorConfig(symbols=["BTCUSDT", "ETHUSDT"])
        collector = DataCollector(config)
        
        await collector.start()
        
        # Accéder aux données
        price = collector.get_price("BTCUSDT")
        state = collector.get_pair_state("BTCUSDT")
        klines = await collector.fetch_klines("BTCUSDT", limit=100)
        
        await collector.stop()
    """
    
    def __init__(self, config: CollectorConfig):
        """
        Initialise le collecteur.
        
        Args:
            config: Configuration du collecteur
        """
        self._config = config
        self._client: Optional[BinanceClient] = None
        self._ws_manager: Optional[WebSocketManager] = None
        self._running = False
        
        self.stats = CollectorStats(symbols_count=len(config.symbols))
    
    @property
    def is_running(self) -> bool:
        """Vérifie si le collecteur est actif."""
        return self._running
    
    @property
    def symbols(self) -> List[str]:
        """Liste des symboles surveillés."""
        return self._config.symbols
    
    async def start(self) -> None:
        """
        Démarre le collecteur.
        
        Initialise la connexion REST et WebSocket.
        """
        if self._running:
            logger.warning("Collecteur déjà en cours d'exécution")
            return
        
        logger.info(f"🚀 Démarrage du collecteur ({len(self.symbols)} paires)...")
        
        # Connexion REST
        self._client = BinanceClient(
            use_production_data=self._config.use_production_data
        )
        await self._client.connect()
        
        # WebSocket si activé
        if self._config.enable_websocket:
            self._ws_manager = WebSocketManager(self._client._client)
            await self._ws_manager.start(
                symbols=self._config.symbols,
                subscribe_klines=self._config.subscribe_klines,
                subscribe_depth=self._config.subscribe_depth,
                kline_interval=self._config.kline_interval,
                depth_level=self._config.depth_level
            )
            self.stats.websocket_running = True
        
        self._running = True
        self.stats.start_time = datetime.now()
        
        logger.info("✅ Collecteur démarré")
    
    async def stop(self) -> None:
        """Arrête proprement le collecteur."""
        if not self._running:
            return
        
        logger.info("🛑 Arrêt du collecteur...")
        
        # Arrêter WebSocket
        if self._ws_manager:
            await self._ws_manager.stop()
            self.stats.websocket_running = False
        
        # Déconnecter REST
        if self._client:
            await self._client.disconnect()
        
        self._running = False
        logger.info("✅ Collecteur arrêté")
    
    async def __aenter__(self) -> "DataCollector":
        """Support du context manager."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ferme à la sortie du context manager."""
        await self.stop()
    
    # =========================================
    # DONNÉES TEMPS RÉEL (WebSocket)
    # =========================================
    
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Récupère le prix actuel d'une paire (temps réel).
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            Prix actuel ou None si non disponible
        """
        state = self.get_pair_state(symbol)
        return state.current_price if state else None
    
    def get_pair_state(self, symbol: str) -> Optional[PairState]:
        """
        Récupère l'état complet d'une paire.
        
        Inclut: prix, historique, kline, orderbook.
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            PairState ou None si non surveillée
        """
        if self._ws_manager:
            return self._ws_manager.get_pair_state(symbol)
        return None
    
    def get_all_states(self) -> Dict[str, PairState]:
        """Récupère l'état de toutes les paires."""
        if self._ws_manager:
            return self._ws_manager.get_all_states()
        return {}
    
    def get_all_prices(self) -> Dict[str, float]:
        """
        Récupère les prix de toutes les paires surveillées.
        
        Returns:
            Dict {symbol: price}
        """
        prices = {}
        for symbol, state in self.get_all_states().items():
            if state.current_price > 0:
                prices[symbol] = state.current_price
        return prices
    
    def get_depth(self, symbol: str) -> Optional[DepthData]:
        """
        Récupère l'orderbook temps réel d'une paire.
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            DepthData ou None
        """
        state = self.get_pair_state(symbol)
        return state.current_depth if state else None
    
    def get_current_kline(self, symbol: str) -> Optional[KlineData]:
        """
        Récupère la kline en cours d'une paire.
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            KlineData ou None
        """
        state = self.get_pair_state(symbol)
        return state.current_kline if state else None
    
    def get_top_movers(self, n: int = 10, timeframe: str = "1m") -> List[PairState]:
        """
        Récupère les paires avec le plus de mouvement.
        
        Args:
            n: Nombre de paires à retourner
            timeframe: "1m", "5m" ou "start"
            
        Returns:
            Liste des paires triées par mouvement
        """
        if self._ws_manager:
            return self._ws_manager.get_top_movers(n, timeframe)
        return []
    
    # =========================================
    # DONNÉES HISTORIQUES (REST API)
    # =========================================
    
    async def fetch_price(self, symbol: str) -> Optional[float]:
        """
        Récupère le prix via REST API (ponctuel).
        
        Utiliser get_price() pour le temps réel.
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            Prix actuel
        """
        if self._client:
            ticker = await self._client.get_price(symbol)
            return ticker.price
        return None
    
    async def fetch_klines(
        self,
        symbol: str,
        interval: str = KLINE_INTERVAL_1M,
        limit: int = 100
    ) -> List[Kline]:
        """
        Récupère les klines historiques via REST API.
        
        Args:
            symbol: Symbole de la paire
            interval: Intervalle (1m, 5m, 15m, 1h, etc.)
            limit: Nombre de klines (max 1000)
            
        Returns:
            Liste de Kline
        """
        if self._client:
            return await self._client.get_klines(symbol, interval, limit)
        return []
    
    async def fetch_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBook]:
        """
        Récupère l'orderbook via REST API (snapshot).
        
        Utiliser get_depth() pour le temps réel.
        
        Args:
            symbol: Symbole de la paire
            limit: Profondeur
            
        Returns:
            OrderBook
        """
        if self._client:
            return await self._client.get_orderbook(symbol, limit)
        return None
    
    # =========================================
    # CALLBACKS
    # =========================================
    
    def on_ticker(self, callback) -> None:
        """Enregistre un callback pour les tickers."""
        if self._ws_manager:
            self._ws_manager.on_ticker(callback)
    
    def on_kline(self, callback) -> None:
        """Enregistre un callback pour les klines."""
        if self._ws_manager:
            self._ws_manager.on_kline(callback)
    
    def on_depth(self, callback) -> None:
        """Enregistre un callback pour l'orderbook."""
        if self._ws_manager:
            self._ws_manager.on_depth(callback)
    
    # =========================================
    # STATISTIQUES
    # =========================================
    
    def get_websocket_stats(self):
        """Récupère les stats du WebSocket."""
        if self._ws_manager:
            return self._ws_manager.stats
        return None
    
    def get_summary(self) -> dict:
        """
        Retourne un résumé de l'état du collecteur.
        
        Returns:
            Dict avec les infos principales
        """
        ws_stats = self.get_websocket_stats()
        
        return {
            "running": self._running,
            "uptime_seconds": self.stats.uptime_seconds,
            "symbols_count": len(self.symbols),
            "websocket": {
                "enabled": self._config.enable_websocket,
                "running": self.stats.websocket_running,
                "messages_received": ws_stats.messages_received if ws_stats else 0,
                "messages_per_second": ws_stats.messages_per_second if ws_stats else 0,
                "reconnections": ws_stats.reconnections if ws_stats else 0,
                "errors": ws_stats.errors_count if ws_stats else 0,
            },
            "streams": {
                "ticker": True,
                "klines": self._config.subscribe_klines,
                "depth": self._config.subscribe_depth,
            }
        }