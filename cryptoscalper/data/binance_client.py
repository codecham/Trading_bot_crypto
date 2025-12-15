# cryptoscalper/data/binance_client.py
"""
Client wrapper pour l'API Binance.

Fournit une interface simplifiée pour interagir avec Binance.
Supporte le mode testnet et production.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException, BinanceRequestException

from cryptoscalper.config.settings import get_settings
from cryptoscalper.config.constants import (
    KLINE_INTERVAL_1M,
    MAX_KLINES_PER_REQUEST,
)
from cryptoscalper.utils.logger import logger
from cryptoscalper.utils.exceptions import (
    APIConnectionError,
    InvalidSymbolError,
    DataFetchError,
)


@dataclass
class TickerPrice:
    """Prix actuel d'une paire."""
    
    symbol: str
    price: float
    timestamp: datetime


@dataclass
class Kline:
    """Une bougie (candlestick) OHLCV."""
    
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime
    quote_volume: float
    trades_count: int


@dataclass
class OrderBookEntry:
    """Une entrée du carnet d'ordres."""
    
    price: float
    quantity: float


@dataclass
class OrderBook:
    """Carnet d'ordres (orderbook) d'une paire."""
    
    symbol: str
    bids: List[OrderBookEntry]  # Ordres d'achat (prix décroissant)
    asks: List[OrderBookEntry]  # Ordres de vente (prix croissant)
    timestamp: datetime
    
    @property
    def best_bid(self) -> Optional[float]:
        """Meilleur prix d'achat."""
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Meilleur prix de vente."""
        return self.asks[0].price if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        """Spread bid-ask en valeur absolue."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_percent(self) -> Optional[float]:
        """Spread en pourcentage."""
        if self.best_bid and self.spread:
            return self.spread / self.best_bid
        return None


class BinanceClient:
    """
    Client pour interagir avec l'API Binance.
    
    Supporte le mode testnet et production.
    Utilise l'API async pour de meilleures performances.
    
    Usage:
        client = BinanceClient()
        await client.connect()
        price = await client.get_price("BTCUSDT")
        await client.disconnect()
        
    Ou avec context manager:
        async with BinanceClient() as client:
            price = await client.get_price("BTCUSDT")
    """
    
    def __init__(self):
        """Initialise le client (ne se connecte pas encore)."""
        self._settings = get_settings()
        self._client: Optional[AsyncClient] = None
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Vérifie si le client est connecté."""
        return self._connected and self._client is not None
    
    @property
    def is_testnet(self) -> bool:
        """Vérifie si on est en mode testnet."""
        return self._settings.binance.testnet
    
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
            self._client = await self._create_client()
            self._connected = True
            mode = "TESTNET" if self.is_testnet else "PRODUCTION"
            logger.info(f"✅ Connecté à Binance ({mode})")
        except Exception as e:
            raise APIConnectionError(f"Impossible de se connecter à Binance: {e}")
    
    async def _create_client(self) -> AsyncClient:
        """Crée le client async Binance."""
        if self.is_testnet:
            return await AsyncClient.create(
                api_key=self._settings.binance.api_key,
                api_secret=self._settings.binance.api_secret,
                testnet=True
            )
        return await AsyncClient.create(
            api_key=self._settings.binance.api_key,
            api_secret=self._settings.binance.api_secret
        )
    
    async def disconnect(self) -> None:
        """Ferme la connexion proprement."""
        if self._client:
            await self._client.close_connection()
            self._client = None
            self._connected = False
            logger.info("Déconnecté de Binance")
    
    async def __aenter__(self) -> "BinanceClient":
        """Support du context manager async."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ferme la connexion à la sortie du context manager."""
        await self.disconnect()
    
    def _ensure_connected(self) -> None:
        """Vérifie que le client est connecté."""
        if not self.is_connected:
            raise APIConnectionError("Client non connecté. Appelez connect() d'abord.")
    
    async def get_price(self, symbol: str) -> TickerPrice:
        """
        Récupère le prix actuel d'une paire.
        
        Args:
            symbol: Paire de trading (ex: "BTCUSDT")
            
        Returns:
            TickerPrice avec le prix actuel
            
        Raises:
            InvalidSymbolError: Si le symbole est invalide
            DataFetchError: Si la requête échoue
        """
        self._ensure_connected()
        
        try:
            ticker = await self._client.get_symbol_ticker(symbol=symbol)
            return TickerPrice(
                symbol=ticker["symbol"],
                price=float(ticker["price"]),
                timestamp=datetime.now()
            )
        except BinanceAPIException as e:
            if e.code == -1121:  # Invalid symbol
                raise InvalidSymbolError(symbol)
            raise DataFetchError(f"Erreur API Binance: {e.message}", symbol=symbol)
        except BinanceRequestException as e:
            raise DataFetchError(f"Erreur requête: {e}", symbol=symbol)
    
    async def get_all_prices(self) -> List[TickerPrice]:
        """
        Récupère les prix de toutes les paires.
        
        Returns:
            Liste de TickerPrice pour toutes les paires
        """
        self._ensure_connected()
        
        try:
            tickers = await self._client.get_all_tickers()
            now = datetime.now()
            return [
                TickerPrice(
                    symbol=t["symbol"],
                    price=float(t["price"]),
                    timestamp=now
                )
                for t in tickers
            ]
        except (BinanceAPIException, BinanceRequestException) as e:
            raise DataFetchError(f"Erreur récupération prix: {e}")
    
    async def get_klines(
        self,
        symbol: str,
        interval: str = KLINE_INTERVAL_1M,
        limit: int = 100
    ) -> List[Kline]:
        """
        Récupère les bougies (candlesticks) d'une paire.
        
        Args:
            symbol: Paire de trading (ex: "BTCUSDT")
            interval: Intervalle des bougies (ex: "1m", "5m", "1h")
            limit: Nombre de bougies à récupérer (max 1000)
            
        Returns:
            Liste de Kline (plus ancienne en premier)
        """
        self._ensure_connected()
        limit = min(limit, MAX_KLINES_PER_REQUEST)
        
        try:
            klines = await self._client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            return [self._parse_kline(k) for k in klines]
        except BinanceAPIException as e:
            if e.code == -1121:
                raise InvalidSymbolError(symbol)
            raise DataFetchError(f"Erreur récupération klines: {e.message}", symbol=symbol)
    
    def _parse_kline(self, kline: list) -> Kline:
        """Parse une kline brute en objet Kline."""
        return Kline(
            open_time=datetime.fromtimestamp(kline[0] / 1000),
            open=float(kline[1]),
            high=float(kline[2]),
            low=float(kline[3]),
            close=float(kline[4]),
            volume=float(kline[5]),
            close_time=datetime.fromtimestamp(kline[6] / 1000),
            quote_volume=float(kline[7]),
            trades_count=int(kline[8])
        )
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """
        Récupère le carnet d'ordres d'une paire.
        
        Args:
            symbol: Paire de trading (ex: "BTCUSDT")
            limit: Profondeur du carnet (5, 10, 20, 50, 100, 500, 1000)
            
        Returns:
            OrderBook avec les bids et asks
        """
        self._ensure_connected()
        
        try:
            depth = await self._client.get_order_book(symbol=symbol, limit=limit)
            return OrderBook(
                symbol=symbol,
                bids=[OrderBookEntry(float(b[0]), float(b[1])) for b in depth["bids"]],
                asks=[OrderBookEntry(float(a[0]), float(a[1])) for a in depth["asks"]],
                timestamp=datetime.now()
            )
        except BinanceAPIException as e:
            if e.code == -1121:
                raise InvalidSymbolError(symbol)
            raise DataFetchError(f"Erreur récupération orderbook: {e.message}", symbol=symbol)
    
    async def get_account_balance(self, asset: str = "USDT") -> float:
        """
        Récupère le solde d'un asset.
        
        Args:
            asset: L'asset dont on veut le solde (ex: "USDT", "BTC")
            
        Returns:
            Solde disponible (free balance)
        """
        self._ensure_connected()
        
        try:
            account = await self._client.get_account()
            for balance in account["balances"]:
                if balance["asset"] == asset:
                    return float(balance["free"])
            return 0.0
        except (BinanceAPIException, BinanceRequestException) as e:
            raise DataFetchError(f"Erreur récupération balance: {e}")
    
    async def ping(self) -> bool:
        """
        Teste la connexion avec Binance.
        
        Returns:
            True si la connexion fonctionne
        """
        self._ensure_connected()
        
        try:
            await self._client.ping()
            return True
        except Exception:
            return False