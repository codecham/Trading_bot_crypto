# cryptoscalper/data/pair_scanner.py
"""
Scanner de paires pour identifier les meilleures opportunités de trading.

Filtre les paires selon :
- Volume 24h minimum
- Spread acceptable
- Exclusion des stablecoins
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from binance import AsyncClient

from cryptoscalper.config.settings import get_settings
from cryptoscalper.config.constants import (
    QUOTE_ASSET,
    EXCLUDED_PAIRS,
    MAX_PAIRS_TO_SCAN,
)
from cryptoscalper.utils.logger import logger
from cryptoscalper.utils.exceptions import DataFetchError


@dataclass
class PairInfo:
    """Informations sur une paire de trading."""
    
    symbol: str
    base_asset: str
    quote_asset: str
    price: float
    volume_24h: float
    price_change_percent_24h: float
    high_24h: float
    low_24h: float
    trades_count_24h: int
    
    @property
    def volatility_24h(self) -> float:
        """Volatilité approximative sur 24h (high-low / price)."""
        if self.price == 0:
            return 0.0
        return (self.high_24h - self.low_24h) / self.price
    
    @property
    def is_bullish_24h(self) -> bool:
        """True si la paire est en hausse sur 24h."""
        return self.price_change_percent_24h > 0


@dataclass
class ScanResult:
    """Résultat d'un scan de paires."""
    
    pairs: List[PairInfo]
    total_pairs_found: int
    pairs_after_filters: int
    scan_timestamp: datetime
    scan_duration_ms: float
    
    def get_top_by_volume(self, n: int = 10) -> List[PairInfo]:
        """Retourne les N paires avec le plus de volume."""
        return sorted(self.pairs, key=lambda p: p.volume_24h, reverse=True)[:n]
    
    def get_top_by_volatility(self, n: int = 10) -> List[PairInfo]:
        """Retourne les N paires les plus volatiles."""
        return sorted(self.pairs, key=lambda p: p.volatility_24h, reverse=True)[:n]
    
    def get_bullish_pairs(self) -> List[PairInfo]:
        """Retourne uniquement les paires en hausse."""
        return [p for p in self.pairs if p.is_bullish_24h]


class PairScanner:
    """
    Scanner pour identifier les meilleures paires à trader.
    
    Filtre les paires USDT selon :
    - Volume 24h minimum (liquidité)
    - Spread acceptable
    - Exclusion des stablecoins
    
    Usage:
        scanner = PairScanner(client)
        result = await scanner.scan()
        top_pairs = result.get_top_by_volume(20)
    """
    
    def __init__(self, client: AsyncClient):
        """
        Initialise le scanner.
        
        Args:
            client: Client Binance async connecté
        """
        self._client = client
        self._settings = get_settings()
    
    async def scan(
        self,
        min_volume_24h: Optional[float] = None,
        max_pairs: Optional[int] = None,
        quote_asset: str = QUOTE_ASSET
    ) -> ScanResult:
        """
        Scanne et filtre les paires disponibles.
        
        Args:
            min_volume_24h: Volume minimum en USDT (défaut: depuis settings)
            max_pairs: Nombre max de paires à retourner (défaut: depuis constants)
            quote_asset: Asset de cotation (défaut: USDT)
            
        Returns:
            ScanResult avec les paires filtrées
        """
        start_time = datetime.now()
        
        # Valeurs par défaut
        min_volume = min_volume_24h or self._settings.scanner.min_volume_24h_usdt
        max_count = max_pairs or MAX_PAIRS_TO_SCAN
        
        logger.info(f"🔍 Scan des paires {quote_asset} (volume min: {min_volume:,.0f} USDT)")
        
        # Récupérer toutes les paires
        all_pairs = await self._fetch_all_pairs()
        total_found = len(all_pairs)
        
        # Filtrer
        filtered = self._filter_pairs(
            pairs=all_pairs,
            quote_asset=quote_asset,
            min_volume=min_volume
        )
        
        # Trier par volume et limiter
        filtered.sort(key=lambda p: p.volume_24h, reverse=True)
        filtered = filtered[:max_count]
        
        # Calculer la durée
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(
            f"✅ Scan terminé: {len(filtered)}/{total_found} paires "
            f"({duration_ms:.0f}ms)"
        )
        
        return ScanResult(
            pairs=filtered,
            total_pairs_found=total_found,
            pairs_after_filters=len(filtered),
            scan_timestamp=datetime.now(),
            scan_duration_ms=duration_ms
        )
    
    async def _fetch_all_pairs(self) -> List[PairInfo]:
        """Récupère les infos de toutes les paires."""
        try:
            tickers = await self._client.get_ticker()
            return [self._parse_ticker(t) for t in tickers]
        except Exception as e:
            raise DataFetchError(f"Erreur récupération tickers: {e}")
    
    def _parse_ticker(self, ticker: dict) -> PairInfo:
        """Parse un ticker en PairInfo."""
        symbol = ticker["symbol"]
        
        # Extraire base et quote (simplifié - suppose USDT en fin)
        quote = "USDT" if symbol.endswith("USDT") else ""
        base = symbol[:-len(quote)] if quote else symbol
        
        return PairInfo(
            symbol=symbol,
            base_asset=base,
            quote_asset=quote,
            price=float(ticker.get("lastPrice", 0)),
            volume_24h=float(ticker.get("quoteVolume", 0)),
            price_change_percent_24h=float(ticker.get("priceChangePercent", 0)),
            high_24h=float(ticker.get("highPrice", 0)),
            low_24h=float(ticker.get("lowPrice", 0)),
            trades_count_24h=int(ticker.get("count", 0))
        )
    
    def _filter_pairs(
        self,
        pairs: List[PairInfo],
        quote_asset: str,
        min_volume: float
    ) -> List[PairInfo]:
        """Applique tous les filtres sur les paires."""
        filtered = []
        
        for pair in pairs:
            # Filtre: quote asset (USDT)
            if pair.quote_asset != quote_asset:
                continue
            
            # Filtre: exclure stablecoins et paires blacklistées
            if pair.symbol in EXCLUDED_PAIRS:
                continue
            
            # Filtre: volume minimum
            if pair.volume_24h < min_volume:
                continue
            
            # Filtre: prix > 0 (paires actives)
            if pair.price <= 0:
                continue
            
            filtered.append(pair)
        
        return filtered
    
    async def get_pair_info(self, symbol: str) -> Optional[PairInfo]:
        """
        Récupère les infos d'une paire spécifique.
        
        Args:
            symbol: Symbole de la paire (ex: "BTCUSDT")
            
        Returns:
            PairInfo ou None si non trouvée
        """
        try:
            ticker = await self._client.get_ticker(symbol=symbol)
            return self._parse_ticker(ticker)
        except Exception:
            return None
    
    async def get_symbols_list(
        self,
        min_volume_24h: Optional[float] = None,
        max_pairs: Optional[int] = None
    ) -> List[str]:
        """
        Retourne uniquement la liste des symboles filtrés.
        
        Pratique pour initialiser les WebSockets.
        
        Args:
            min_volume_24h: Volume minimum
            max_pairs: Nombre max de paires
            
        Returns:
            Liste de symboles (ex: ["BTCUSDT", "ETHUSDT", ...])
        """
        result = await self.scan(min_volume_24h, max_pairs)
        return [p.symbol for p in result.pairs]