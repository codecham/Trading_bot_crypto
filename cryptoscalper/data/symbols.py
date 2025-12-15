# cryptoscalper/data/symbols.py
"""
Gestionnaire dynamique des symboles de trading.

Responsabilités :
- Sélection des meilleures paires à surveiller
- Rafraîchissement périodique de la liste
- Filtrage par volume, spread, et critères de liquidité
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Set

from binance import AsyncClient

from cryptoscalper.config.settings import get_settings
from cryptoscalper.config.constants import (
    QUOTE_ASSET,
    EXCLUDED_PAIRS,
    MAX_PAIRS_TO_SCAN,
    PAIRS_REFRESH_INTERVAL_SECONDS,
)
from cryptoscalper.utils.logger import logger


@dataclass
class SymbolInfo:
    """Informations sur un symbole."""
    
    symbol: str
    base_asset: str
    quote_asset: str
    volume_24h: float
    price: float
    price_change_percent: float
    
    @property
    def is_bullish(self) -> bool:
        """True si en hausse sur 24h."""
        return self.price_change_percent > 0


@dataclass
class SymbolsState:
    """État du gestionnaire de symboles."""
    
    symbols: List[str] = field(default_factory=list)
    last_refresh: Optional[datetime] = None
    refresh_count: int = 0
    last_error: Optional[str] = None


class SymbolsManager:
    """
    Gestionnaire dynamique des symboles à surveiller.
    
    Sélectionne automatiquement les meilleures paires selon :
    - Volume 24h minimum
    - Exclusion des stablecoins
    - Quote asset (USDT par défaut)
    
    Rafraîchit périodiquement la liste pour s'adapter au marché.
    
    Usage:
        manager = SymbolsManager(client)
        await manager.start()
        
        symbols = manager.get_symbols()  # Liste actuelle
        
        await manager.stop()
    """
    
    def __init__(
        self,
        client: AsyncClient,
        min_volume_24h: Optional[float] = None,
        max_pairs: Optional[int] = None,
        refresh_interval: Optional[int] = None,
        quote_asset: str = QUOTE_ASSET
    ):
        """
        Initialise le gestionnaire.
        
        Args:
            client: Client Binance async connecté
            min_volume_24h: Volume minimum en USDT (défaut: depuis settings)
            max_pairs: Nombre max de paires (défaut: depuis constants)
            refresh_interval: Intervalle de rafraîchissement en secondes
            quote_asset: Asset de cotation (défaut: USDT)
        """
        self._client = client
        self._settings = get_settings()
        
        # Configuration
        self._min_volume = min_volume_24h or self._settings.scanner.min_volume_24h_usdt
        self._max_pairs = max_pairs or MAX_PAIRS_TO_SCAN
        self._refresh_interval = refresh_interval or PAIRS_REFRESH_INTERVAL_SECONDS
        self._quote_asset = quote_asset
        
        # État
        self._symbols: List[str] = []
        self._symbols_info: dict[str, SymbolInfo] = {}
        self._running = False
        self._refresh_task: Optional[asyncio.Task] = None
        
        self.state = SymbolsState()
    
    @property
    def symbols(self) -> List[str]:
        """Liste des symboles actuels."""
        return list(self._symbols)
    
    @property
    def count(self) -> int:
        """Nombre de symboles surveillés."""
        return len(self._symbols)
    
    def get_symbols(self) -> List[str]:
        """Retourne la liste des symboles."""
        return self.symbols
    
    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Retourne les infos d'un symbole."""
        return self._symbols_info.get(symbol)
    
    def contains(self, symbol: str) -> bool:
        """Vérifie si un symbole est dans la liste."""
        return symbol in self._symbols
    
    async def start(self, auto_refresh: bool = True) -> None:
        """
        Démarre le gestionnaire.
        
        Args:
            auto_refresh: Si True, lance le rafraîchissement périodique
        """
        if self._running:
            logger.warning("SymbolsManager déjà en cours d'exécution")
            return
        
        logger.info("🔄 Démarrage du gestionnaire de symboles...")
        
        # Premier chargement
        await self.refresh()
        
        self._running = True
        
        # Lancer le rafraîchissement périodique
        if auto_refresh:
            self._refresh_task = asyncio.create_task(self._refresh_loop())
        
        logger.info(f"✅ SymbolsManager démarré ({self.count} symboles)")
    
    async def stop(self) -> None:
        """Arrête le gestionnaire."""
        if not self._running:
            return
        
        logger.info("🛑 Arrêt du gestionnaire de symboles...")
        
        self._running = False
        
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ SymbolsManager arrêté")
    
    async def refresh(self) -> List[str]:
        """
        Rafraîchit la liste des symboles.
        
        Returns:
            Nouvelle liste de symboles
        """
        logger.debug("Rafraîchissement de la liste des symboles...")
        
        try:
            # Récupérer tous les tickers
            tickers = await self._client.get_ticker()
            
            # Parser et filtrer
            symbols_info = self._parse_and_filter_tickers(tickers)
            
            # Trier par volume et limiter
            symbols_info.sort(key=lambda x: x.volume_24h, reverse=True)
            symbols_info = symbols_info[:self._max_pairs]
            
            # Mettre à jour l'état
            self._symbols = [s.symbol for s in symbols_info]
            self._symbols_info = {s.symbol: s for s in symbols_info}
            
            # Mettre à jour les stats
            self.state.symbols = self._symbols
            self.state.last_refresh = datetime.now()
            self.state.refresh_count += 1
            self.state.last_error = None
            
            logger.info(
                f"📋 Symboles rafraîchis: {len(self._symbols)} paires "
                f"(refresh #{self.state.refresh_count})"
            )
            
            return self._symbols
            
        except Exception as e:
            self.state.last_error = str(e)
            logger.error(f"Erreur rafraîchissement symboles: {e}")
            return self._symbols  # Garder l'ancienne liste
    
    def _parse_and_filter_tickers(self, tickers: list) -> List[SymbolInfo]:
        """Parse et filtre les tickers."""
        result = []
        
        for ticker in tickers:
            symbol = ticker.get("symbol", "")
            
            # Filtre: quote asset
            if not symbol.endswith(self._quote_asset):
                continue
            
            # Filtre: exclusions
            if symbol in EXCLUDED_PAIRS:
                continue
            
            # Filtre: volume minimum
            volume = float(ticker.get("quoteVolume", 0))
            if volume < self._min_volume:
                continue
            
            # Filtre: prix > 0
            price = float(ticker.get("lastPrice", 0))
            if price <= 0:
                continue
            
            # Extraire base asset
            base = symbol[:-len(self._quote_asset)]
            
            result.append(SymbolInfo(
                symbol=symbol,
                base_asset=base,
                quote_asset=self._quote_asset,
                volume_24h=volume,
                price=price,
                price_change_percent=float(ticker.get("priceChangePercent", 0))
            ))
        
        return result
    
    async def _refresh_loop(self) -> None:
        """Boucle de rafraîchissement périodique."""
        while self._running:
            try:
                await asyncio.sleep(self._refresh_interval)
                
                if not self._running:
                    break
                
                await self.refresh()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur dans la boucle de rafraîchissement: {e}")
                await asyncio.sleep(60)  # Attendre avant de réessayer
    
    def get_top_by_volume(self, n: int = 10) -> List[SymbolInfo]:
        """Retourne les N symboles avec le plus de volume."""
        sorted_symbols = sorted(
            self._symbols_info.values(),
            key=lambda x: x.volume_24h,
            reverse=True
        )
        return sorted_symbols[:n]
    
    def get_bullish_symbols(self) -> List[str]:
        """Retourne les symboles en hausse."""
        return [
            symbol for symbol, info in self._symbols_info.items()
            if info.is_bullish
        ]
    
    def get_summary(self) -> dict:
        """Retourne un résumé de l'état."""
        return {
            "count": self.count,
            "last_refresh": self.state.last_refresh.isoformat() if self.state.last_refresh else None,
            "refresh_count": self.state.refresh_count,
            "refresh_interval_seconds": self._refresh_interval,
            "min_volume_24h": self._min_volume,
            "max_pairs": self._max_pairs,
            "last_error": self.state.last_error
        }


async def get_tradeable_symbols(
    client: AsyncClient,
    min_volume: float = 1_000_000,
    max_pairs: int = 150
) -> List[str]:
    """
    Fonction utilitaire pour récupérer rapidement les symboles.
    
    Pour un usage ponctuel. Pour un usage continu, utiliser SymbolsManager.
    
    Args:
        client: Client Binance
        min_volume: Volume minimum 24h en USDT
        max_pairs: Nombre max de paires
        
    Returns:
        Liste de symboles triés par volume
    """
    manager = SymbolsManager(
        client=client,
        min_volume_24h=min_volume,
        max_pairs=max_pairs
    )
    
    return await manager.refresh()