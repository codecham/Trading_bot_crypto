# cryptoscalper/data/historical.py
"""
Module de téléchargement des données historiques depuis Binance.
VERSION OPTIMISÉE - Téléchargement parallèle.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Callable
import pandas as pd
import numpy as np

from binance import AsyncClient

from cryptoscalper.config.constants import (
    KLINE_INTERVAL_1M,
    MAX_KLINES_PER_REQUEST,
    DATA_DIR,
)
from cryptoscalper.utils.logger import logger


# ============================================
# CONSTANTES
# ============================================

DEFAULT_INTERVAL = KLINE_INTERVAL_1M
DEFAULT_DAYS = 180

# OPTIMISÉ: Binance permet 1200 req/min = 20/sec
# On utilise 0.05s = 20 req/sec (safe margin)
REQUEST_DELAY_SECONDS = 0.05

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
]


# ============================================
# DATACLASSES
# ============================================

@dataclass
class DownloadConfig:
    """Configuration pour le téléchargement."""
    
    interval: str = DEFAULT_INTERVAL
    days: int = DEFAULT_DAYS
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    output_dir: Path = field(default_factory=lambda: Path(DATA_DIR))
    save_format: str = "parquet"
    
    def __post_init__(self):
        if self.end_date is None:
            self.end_date = datetime.now()
        if self.start_date is None:
            self.start_date = self.end_date - timedelta(days=self.days)


@dataclass
class DownloadProgress:
    """Progression du téléchargement."""
    
    symbol: str
    total_expected: int = 0
    downloaded: int = 0
    start_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    
    @property
    def percent(self) -> float:
        if self.total_expected == 0:
            return 0.0
        return (self.downloaded / self.total_expected) * 100
    
    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def rate_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return self.downloaded / self.elapsed_seconds


@dataclass
class DownloadResult:
    """Résultat d'un téléchargement."""
    
    symbol: str
    df: pd.DataFrame
    start_date: datetime
    end_date: datetime
    rows_count: int
    duration_seconds: float
    file_path: Optional[Path] = None
    
    def summary(self) -> str:
        return (
            f"{self.symbol}: {self.rows_count:,} lignes "
            f"({self.start_date.date()} → {self.end_date.date()}) "
            f"en {self.duration_seconds:.1f}s"
        )


# ============================================
# HISTORICAL DATA DOWNLOADER
# ============================================

class HistoricalDataDownloader:
    """
    Télécharge les données historiques depuis Binance.
    VERSION OPTIMISÉE avec délai réduit.
    """
    
    def __init__(
        self,
        client: AsyncClient,
        config: Optional[DownloadConfig] = None
    ):
        self._client = client
        self._config = config or DownloadConfig()
        self._progress_callback: Optional[Callable[[DownloadProgress], None]] = None
    
    def on_progress(self, callback: Callable[[DownloadProgress], None]) -> None:
        self._progress_callback = callback
    
    async def download(
        self,
        symbol: str,
        interval: Optional[str] = None,
        days: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Télécharge les données historiques pour un symbole."""
        interval = interval or self._config.interval
        end_date = end_date or datetime.now()
        
        if start_date is not None:
            pass
        elif days is not None:
            start_date = end_date - timedelta(days=days)
        else:
            start_date = end_date - timedelta(days=self._config.days)
        
        total_minutes = int((end_date - start_date).total_seconds() / 60)
        
        logger.info(
            f"📥 {symbol} ({interval}): "
            f"{start_date.date()} → {end_date.date()} (~{total_minutes:,} klines)"
        )
        
        progress = DownloadProgress(
            symbol=symbol,
            total_expected=total_minutes,
            start_time=datetime.now()
        )
        
        all_klines = await self._download_in_batches(
            symbol=symbol,
            interval=interval,
            start_time=start_date,
            end_time=end_date,
            progress=progress
        )
        
        df = self._klines_to_dataframe(all_klines)
        
        logger.info(
            f"✅ {symbol}: {len(df):,} klines "
            f"en {progress.elapsed_seconds:.1f}s "
            f"({progress.rate_per_second:.0f}/s)"
        )
        
        return df
    
    async def _download_in_batches(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        progress: DownloadProgress
    ) -> List[list]:
        """Télécharge les klines par lots de 1000."""
        all_klines = []
        current_start = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        while current_start < end_ms:
            try:
                klines = await self._fetch_klines_batch(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start,
                    end_time=end_ms
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                progress.downloaded = len(all_klines)
                
                self._notify_progress(progress)
                
                current_start = klines[-1][6] + 1
                
                # Rate limiting optimisé
                await asyncio.sleep(REQUEST_DELAY_SECONDS)
                
            except Exception as e:
                error_msg = f"Erreur batch {symbol}: {e}"
                progress.errors.append(error_msg)
                logger.error(error_msg)
                # Retry avec délai plus long
                await asyncio.sleep(1.0)
                continue
        
        return all_klines
    
    async def _fetch_klines_batch(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int
    ) -> List[list]:
        """Récupère un lot de klines (max 1000)."""
        return await self._client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_time,
            endTime=end_time,
            limit=MAX_KLINES_PER_REQUEST
        )
    
    def _klines_to_dataframe(self, klines: List[list]) -> pd.DataFrame:
        """Convertit les klines brutes en DataFrame."""
        if not klines:
            return pd.DataFrame(columns=KLINE_COLUMNS)
        
        df = pd.DataFrame(klines, columns=KLINE_COLUMNS + ["ignore"])
        df = df.drop(columns=["ignore"])
        df = self._convert_dtypes(df)
        df = df.drop_duplicates(subset=["open_time"])
        df = df.sort_values("open_time").reset_index(drop=True)
        
        return df
    
    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convertit les types de colonnes."""
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        
        float_columns = [
            "open", "high", "low", "close", "volume",
            "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"
        ]
        for col in float_columns:
            df[col] = df[col].astype(float)
        
        df["trades_count"] = df["trades_count"].astype(int)
        
        return df
    
    def _notify_progress(self, progress: DownloadProgress) -> None:
        """Notifie la progression."""
        if progress.downloaded % 10000 == 0 or progress.downloaded == progress.total_expected:
            logger.debug(
                f"  {progress.symbol}: {progress.downloaded:,}/{progress.total_expected:,} "
                f"({progress.percent:.1f}%)"
            )
        
        if self._progress_callback:
            self._progress_callback(progress)
    
    def save_to_parquet(
        self,
        df: pd.DataFrame,
        path: Optional[Path] = None,
        symbol: str = "data"
    ) -> Path:
        """Sauvegarde le DataFrame en Parquet."""
        if path is None:
            path = self._config.output_dir / f"{symbol}_1m.parquet"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(path, index=False)
        logger.info(f"💾 Sauvegardé: {path} ({len(df):,} lignes)")
        
        return path
    
    def save_to_csv(
        self,
        df: pd.DataFrame,
        path: Optional[Path] = None,
        symbol: str = "data"
    ) -> Path:
        """Sauvegarde le DataFrame en CSV."""
        if path is None:
            path = self._config.output_dir / f"{symbol}_1m.csv"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(path, index=False)
        logger.info(f"💾 Sauvegardé: {path} ({len(df):,} lignes)")
        
        return path
    
    @staticmethod
    def load_from_parquet(path: Path) -> pd.DataFrame:
        """Charge un DataFrame depuis un fichier Parquet."""
        df = pd.read_parquet(path)
        logger.info(f"📂 Chargé: {path} ({len(df):,} lignes)")
        return df
    
    @staticmethod
    def load_from_csv(path: Path) -> pd.DataFrame:
        """Charge un DataFrame depuis un fichier CSV."""
        df = pd.read_csv(path, parse_dates=["open_time", "close_time"])
        logger.info(f"📂 Chargé: {path} ({len(df):,} lignes)")
        return df


# ============================================
# MULTI-SYMBOL DOWNLOADER (OPTIMISÉ)
# ============================================

class MultiSymbolDownloader:
    """
    Télécharge les données pour plusieurs symboles.
    VERSION OPTIMISÉE avec téléchargement parallèle.
    """
    
    def __init__(
        self,
        client: AsyncClient,
        config: Optional[DownloadConfig] = None
    ):
        self._client = client
        self._config = config or DownloadConfig()
    
    async def download_all(
        self,
        symbols: List[str],
        days: int = DEFAULT_DAYS,
        save: bool = True,
        parallel: int = 3  # OPTIMISÉ: 3 téléchargements en parallèle
    ) -> Dict[str, DownloadResult]:
        """
        Télécharge les données pour tous les symboles EN PARALLÈLE.
        
        Args:
            symbols: Liste des symboles
            days: Nombre de jours
            save: Sauvegarder automatiquement
            parallel: Nombre de téléchargements parallèles (défaut: 3)
            
        Returns:
            Dict {symbol: DownloadResult}
        """
        logger.info(f"📦 Téléchargement de {len(symbols)} symboles ({days} jours)...")
        logger.info(f"   Mode: {parallel} téléchargements parallèles")
        
        start_total = datetime.now()
        
        # Sémaphore pour limiter les téléchargements parallèles
        semaphore = asyncio.Semaphore(parallel)
        
        async def download_with_semaphore(symbol: str) -> tuple:
            async with semaphore:
                try:
                    result = await self._download_single(symbol, days, save)
                    return symbol, result
                except Exception as e:
                    logger.error(f"❌ Erreur {symbol}: {e}")
                    return symbol, None
        
        # Lancer tous les téléchargements en parallèle
        tasks = [download_with_semaphore(symbol) for symbol in symbols]
        completed = await asyncio.gather(*tasks)
        
        # Collecter les résultats
        results = {symbol: result for symbol, result in completed if result is not None}
        
        # Résumé
        total_duration = (datetime.now() - start_total).total_seconds()
        total_rows = sum(r.rows_count for r in results.values())
        
        logger.info(f"\n{'='*50}")
        logger.info(f"📊 RÉSUMÉ: {len(results)}/{len(symbols)} symboles téléchargés")
        logger.info(f"   Total: {total_rows:,} lignes en {total_duration:.1f}s")
        logger.info(f"   Vitesse: {total_rows/total_duration:,.0f} lignes/sec")
        
        return results
    
    async def _download_single(
        self,
        symbol: str,
        days: int,
        save: bool
    ) -> DownloadResult:
        """Télécharge un seul symbole."""
        start = datetime.now()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        downloader = HistoricalDataDownloader(self._client, self._config)
        df = await downloader.download(symbol, days=days)
        
        file_path = None
        if save and len(df) > 0:
            file_path = downloader.save_to_parquet(df, symbol=symbol)
        
        return DownloadResult(
            symbol=symbol,
            df=df,
            start_date=start_date,
            end_date=end_date,
            rows_count=len(df),
            duration_seconds=(datetime.now() - start).total_seconds(),
            file_path=file_path
        )


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

async def download_historical_data(
    client: AsyncClient,
    symbol: str,
    days: int = 180,
    save: bool = True
) -> pd.DataFrame:
    """Fonction utilitaire pour télécharger rapidement des données."""
    downloader = HistoricalDataDownloader(client)
    df = await downloader.download(symbol, days=days)
    
    if save and len(df) > 0:
        downloader.save_to_parquet(df, symbol=symbol)
    
    return df


def get_cached_data_path(symbol: str, data_dir: Path = Path(DATA_DIR)) -> Path:
    """Retourne le chemin du fichier cache pour un symbole."""
    return data_dir / f"{symbol}_1m.parquet"


def is_data_cached(symbol: str, data_dir: Path = Path(DATA_DIR)) -> bool:
    """Vérifie si les données sont en cache."""
    return get_cached_data_path(symbol, data_dir).exists()


def load_cached_data(symbol: str, data_dir: Path = Path(DATA_DIR)) -> Optional[pd.DataFrame]:
    """Charge les données depuis le cache si disponibles."""
    path = get_cached_data_path(symbol, data_dir)
    if path.exists():
        return HistoricalDataDownloader.load_from_parquet(path)
    return None