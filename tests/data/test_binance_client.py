# tests/data/test_binance_client.py
"""
Tests unitaires pour le client Binance.

Usage:
    pytest tests/data/test_binance_client.py -v
"""

import pytest
from datetime import datetime

from cryptoscalper.data.binance_client import (
    BinanceClient,
    TickerPrice,
    Kline,
    OrderBook,
    OrderBookEntry,
)
from cryptoscalper.utils.exceptions import (
    APIConnectionError,
    InvalidSymbolError,
)


class TestTickerPrice:
    """Tests pour la dataclass TickerPrice."""
    
    def test_create_ticker_price(self):
        """Création d'un TickerPrice valide."""
        ticker = TickerPrice(
            symbol="BTCUSDT",
            price=45000.0,
            timestamp=datetime.now()
        )
        
        assert ticker.symbol == "BTCUSDT"
        assert ticker.price == 45000.0
        assert isinstance(ticker.timestamp, datetime)


class TestKline:
    """Tests pour la dataclass Kline."""
    
    def test_create_kline(self):
        """Création d'une Kline valide."""
        now = datetime.now()
        kline = Kline(
            open_time=now,
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=1000.0,
            close_time=now,
            quote_volume=105000.0,
            trades_count=500
        )
        
        assert kline.open == 100.0
        assert kline.high == 110.0
        assert kline.low == 95.0
        assert kline.close == 105.0


class TestOrderBook:
    """Tests pour la dataclass OrderBook."""
    
    def test_orderbook_spread(self):
        """Calcul du spread bid-ask."""
        orderbook = OrderBook(
            symbol="BTCUSDT",
            bids=[OrderBookEntry(100.0, 1.0), OrderBookEntry(99.0, 2.0)],
            asks=[OrderBookEntry(101.0, 1.0), OrderBookEntry(102.0, 2.0)],
            timestamp=datetime.now()
        )
        
        assert orderbook.best_bid == 100.0
        assert orderbook.best_ask == 101.0
        assert orderbook.spread == 1.0
        assert orderbook.spread_percent == 0.01  # 1%
    
    def test_orderbook_empty(self):
        """Orderbook vide retourne None pour les propriétés."""
        orderbook = OrderBook(
            symbol="BTCUSDT",
            bids=[],
            asks=[],
            timestamp=datetime.now()
        )
        
        assert orderbook.best_bid is None
        assert orderbook.best_ask is None
        assert orderbook.spread is None


class TestBinanceClient:
    """Tests pour BinanceClient."""
    
    def test_client_not_connected_by_default(self):
        """Le client n'est pas connecté à l'initialisation."""
        client = BinanceClient()
        
        assert client.is_connected is False
    
    def test_ensure_connected_raises_error(self):
        """Appeler une méthode sans connexion lève une erreur."""
        client = BinanceClient()
        
        with pytest.raises(APIConnectionError):
            client._ensure_connected()


@pytest.mark.asyncio
class TestBinanceClientAsync:
    """Tests async pour BinanceClient (nécessite connexion)."""
    
    async def test_connect_and_disconnect(self):
        """Test connexion et déconnexion."""
        client = BinanceClient()
        
        await client.connect()
        assert client.is_connected is True
        
        await client.disconnect()
        assert client.is_connected is False
    
    async def test_context_manager(self):
        """Test du context manager."""
        async with BinanceClient() as client:
            assert client.is_connected is True
        
        # Après le context, devrait être déconnecté
        assert client.is_connected is False
    
    async def test_get_price_btcusdt(self):
        """Test récupération prix BTC."""
        async with BinanceClient() as client:
            ticker = await client.get_price("BTCUSDT")
            
            assert ticker.symbol == "BTCUSDT"
            assert ticker.price > 0
            assert isinstance(ticker.timestamp, datetime)
    
    async def test_get_price_invalid_symbol(self):
        """Test erreur symbole invalide."""
        async with BinanceClient() as client:
            with pytest.raises(InvalidSymbolError):
                await client.get_price("INVALIDPAIR123")
    
    async def test_get_klines(self):
        """Test récupération klines."""
        async with BinanceClient() as client:
            klines = await client.get_klines("BTCUSDT", limit=10)
            
            assert len(klines) == 10
            assert all(isinstance(k, Kline) for k in klines)
            assert all(k.close > 0 for k in klines)
    
    async def test_get_orderbook(self):
        """Test récupération orderbook."""
        async with BinanceClient() as client:
            orderbook = await client.get_orderbook("BTCUSDT", limit=10)
            
            assert orderbook.symbol == "BTCUSDT"
            assert len(orderbook.bids) == 10
            assert len(orderbook.asks) == 10
            assert orderbook.spread >= 0
    
    async def test_ping(self):
        """Test ping."""
        async with BinanceClient() as client:
            is_alive = await client.ping()
            
            assert is_alive is True