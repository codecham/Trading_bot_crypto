#!/usr/bin/env python3
# scripts/paper_trading.py
"""
CryptoScalper AI - Paper Trading en Temps Réel.

Script de simulation de trading sans argent réel.
Utilise le modèle swing trading pour scanner les cryptos et simuler des trades.

Usage:
    python scripts/paper_trading.py
    python scripts/paper_trading.py --capital 25 --threshold 0.20 --duration 24h
    python scripts/paper_trading.py --verbose --symbols BTCUSDT,ETHUSDT,SOLUSDT

Paramètres Swing Trading:
    - Take Profit: 2%
    - Stop Loss: 1%
    - Timeout: 120 min (2h)
    - Seuil probabilité: 20%
    - Position size: 20% du capital
"""

import argparse
import asyncio
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
import csv

import numpy as np
import pandas as pd
from loguru import logger

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.data.features import FeatureEngine, get_feature_names
from cryptoscalper.data.binance_client import BinanceClient


# ============================================
# CONSTANTES
# ============================================

# Paramètres Swing Trading
TAKE_PROFIT_PCT = 0.02          # 2%
STOP_LOSS_PCT = 0.01            # 1%
TIMEOUT_MINUTES = 120           # 2 heures
DEFAULT_PROBABILITY_THRESHOLD = 0.20  # 20%
DEFAULT_POSITION_SIZE_PCT = 0.20      # 20% du capital

# Configuration
DEFAULT_CAPITAL = 25.0
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
DEFAULT_MODEL_PATH = "models/saved/swing_final_model.joblib"
DEFAULT_SCAN_INTERVAL = 60      # 1 minute
DEFAULT_DATA_DIR = "data_cache"

# Frais Binance (aller-retour)
TRADING_FEES_PCT = 0.002        # 0.2%


# ============================================
# ENUMS
# ============================================

class TradeStatus(Enum):
    """Statut d'un trade simulé."""
    OPEN = "OPEN"
    CLOSED_TP = "CLOSED_TP"      # Take Profit
    CLOSED_SL = "CLOSED_SL"      # Stop Loss
    CLOSED_TIMEOUT = "CLOSED_TIMEOUT"


# ============================================
# DATACLASSES
# ============================================

@dataclass
class PaperTradeConfig:
    """Configuration du paper trading."""
    
    capital: float = DEFAULT_CAPITAL
    symbols: List[str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    model_path: str = DEFAULT_MODEL_PATH
    data_dir: str = DEFAULT_DATA_DIR
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD
    position_size_pct: float = DEFAULT_POSITION_SIZE_PCT
    scan_interval: int = DEFAULT_SCAN_INTERVAL
    take_profit_pct: float = TAKE_PROFIT_PCT
    stop_loss_pct: float = STOP_LOSS_PCT
    timeout_minutes: int = TIMEOUT_MINUTES
    verbose: bool = False
    log_file: str = "logs/paper_trades.csv"
    duration_hours: Optional[float] = None


@dataclass
class PaperTrade:
    """Représente un trade simulé."""
    
    id: int
    symbol: str
    entry_time: datetime
    entry_price: float
    quantity: float
    size_usdt: float
    take_profit_price: float
    stop_loss_price: float
    timeout_time: datetime
    probability: float
    status: TradeStatus = TradeStatus.OPEN
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_usdt: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    @property
    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN
    
    @property
    def duration_minutes(self) -> Optional[float]:
        if self.exit_time is None:
            return None
        return (self.exit_time - self.entry_time).total_seconds() / 60


@dataclass
class TradingStats:
    """Statistiques de trading."""
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_usdt: float = 0.0
    total_fees_usdt: float = 0.0
    current_capital: float = DEFAULT_CAPITAL
    peak_capital: float = DEFAULT_CAPITAL
    max_drawdown_pct: float = 0.0
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades * 100
    
    @property
    def net_pnl(self) -> float:
        return self.total_pnl_usdt - self.total_fees_usdt


# ============================================
# PAPER TRADER
# ============================================

class PaperTrader:
    """
    Gestionnaire de paper trading.
    
    Simule le trading en temps réel sans argent réel.
    Charge le modèle ML, scanne les paires, et simule les trades.
    """
    
    def __init__(self, config: PaperTradeConfig):
        self._config = config
        self._running = False
        self._trade_counter = 0
        
        # État
        self._open_trades: Dict[str, PaperTrade] = {}  # symbol -> trade
        self._closed_trades: List[PaperTrade] = []
        self._stats = TradingStats(current_capital=config.capital, peak_capital=config.capital)
        
        # Composants (initialisés au démarrage)
        self._model = None
        self._feature_engine = FeatureEngine()
        self._feature_names = get_feature_names()
        self._client: Optional[BinanceClient] = None
        
        # Cache des données
        self._price_cache: Dict[str, float] = {}
        self._klines_cache: Dict[str, pd.DataFrame] = {}
        
        # Timing
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        
        logger.info(f"📄 PaperTrader initialisé | Capital: {config.capital:.2f} USDT")
    
    # =========================================
    # PROPRIÉTÉS
    # =========================================
    
    @property
    def config(self) -> PaperTradeConfig:
        return self._config
    
    @property
    def stats(self) -> TradingStats:
        return self._stats
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def open_positions_count(self) -> int:
        return len(self._open_trades)
    
    # =========================================
    # LIFECYCLE
    # =========================================
    
    async def start(self) -> None:
        """Démarre le paper trading."""
        logger.info("=" * 65)
        logger.info("📄 CryptoScalper AI - Paper Trading")
        logger.info("=" * 65)
        
        # Charger le modèle
        self._load_model()
        
        # Connexion Binance
        await self._connect_binance()
        
        # Charger les données initiales
        await self._load_initial_data()
        
        # Définir le temps de fin si duration spécifiée
        self._start_time = datetime.now()
        if self._config.duration_hours:
            self._end_time = self._start_time + timedelta(hours=self._config.duration_hours)
            logger.info(f"⏱️  Durée: {self._config.duration_hours}h (fin: {self._end_time.strftime('%H:%M')})")
        
        # Afficher la configuration
        self._log_config()
        
        # Démarrer
        self._running = True
        logger.info("✅ Paper Trading démarré!")
        logger.info("-" * 65)
        
        # Boucle principale
        await self._main_loop()
    
    async def stop(self) -> None:
        """Arrête proprement le paper trading."""
        if not self._running:
            return
        
        logger.info("\n🛑 Arrêt du Paper Trading...")
        self._running = False
        
        # Fermer les positions ouvertes
        await self._close_all_positions()
        
        # Déconnecter Binance
        if self._client:
            await self._client.close()
        
        # Afficher le résumé final
        self._print_final_summary()
        
        # Sauvegarder les trades
        self._save_trades_to_csv()
        
        logger.info("✅ Paper Trading arrêté proprement")
    
    # =========================================
    # INITIALISATION
    # =========================================
    
    def _load_model(self) -> None:
        """Charge le modèle ML."""
        import joblib
        
        model_path = Path(self._config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        self._model = joblib.load(model_path)
        logger.info(f"✅ Modèle chargé: {model_path.name}")
    
    async def _connect_binance(self) -> None:
        """Connecte à l'API Binance."""
        self._client = BinanceClient()
        await self._client.connect()
        logger.info("✅ Connexion Binance établie")
    
    async def _load_initial_data(self) -> None:
        """Charge les données initiales (klines)."""
        logger.info("📊 Chargement des données...")
        
        for symbol in self._config.symbols:
            try:
                # Essayer de charger depuis le cache local
                cache_path = Path(self._config.data_dir) / f"{symbol}_1m.parquet"
                if cache_path.exists():
                    df = pd.read_parquet(cache_path).tail(200)
                    self._klines_cache[symbol] = df
                    logger.debug(f"   {symbol}: {len(df)} klines (cache)")
                else:
                    # Sinon récupérer depuis l'API
                    df = await self._fetch_klines_api(symbol, limit=200)
                    self._klines_cache[symbol] = df
                    logger.debug(f"   {symbol}: {len(df)} klines (API)")
                
                # Mettre à jour le prix
                if len(df) > 0:
                    self._price_cache[symbol] = df["close"].iloc[-1]
                    
            except Exception as e:
                logger.warning(f"⚠️  Erreur chargement {symbol}: {e}")
        
        logger.info(f"✅ {len(self._klines_cache)} paires chargées")
    
    async def _fetch_klines_api(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        """Récupère les klines depuis l'API Binance."""
        klines = await self._client._client.get_klines(
            symbol=symbol,
            interval="1m",
            limit=limit
        )
        
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        
        return df
    
    def _log_config(self) -> None:
        """Affiche la configuration."""
        logger.info(f"   Symboles: {', '.join(self._config.symbols)}")
        logger.info(f"   Capital: {self._config.capital:.2f} USDT")
        logger.info(f"   Seuil proba: {self._config.probability_threshold:.0%}")
        logger.info(f"   Position size: {self._config.position_size_pct:.0%}")
        logger.info(f"   TP: {self._config.take_profit_pct:.1%} | SL: {self._config.stop_loss_pct:.1%}")
        logger.info(f"   Timeout: {self._config.timeout_minutes} min")
        logger.info(f"   Scan interval: {self._config.scan_interval}s")
    
    # =========================================
    # BOUCLE PRINCIPALE
    # =========================================
    
    async def _main_loop(self) -> None:
        """Boucle principale de trading."""
        scan_count = 0
        
        while self._running:
            try:
                # Vérifier la durée
                if self._should_stop_by_duration():
                    logger.info("⏱️  Durée atteinte, arrêt...")
                    break
                
                scan_count += 1
                
                # 1. Mettre à jour les prix
                await self._update_prices()
                
                # 2. Vérifier les positions ouvertes
                await self._check_open_positions()
                
                # 3. Scanner les opportunités
                await self._scan_opportunities()
                
                # 4. Afficher le statut périodiquement
                if scan_count % 10 == 0:  # Toutes les 10 scans
                    self._print_status()
                
                # Attendre avant le prochain scan
                await asyncio.sleep(self._config.scan_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Erreur boucle: {e}")
                await asyncio.sleep(5)
    
    def _should_stop_by_duration(self) -> bool:
        """Vérifie si on doit s'arrêter (durée atteinte)."""
        if self._end_time is None:
            return False
        return datetime.now() >= self._end_time
    
    # =========================================
    # MISE À JOUR DES DONNÉES
    # =========================================
    
    async def _update_prices(self) -> None:
        """Met à jour les prix et klines."""
        for symbol in self._config.symbols:
            try:
                # Récupérer le dernier prix
                ticker = await self._client._client.get_symbol_ticker(symbol=symbol)
                self._price_cache[symbol] = float(ticker["price"])
                
                # Mettre à jour les klines (ajouter la dernière)
                if symbol in self._klines_cache:
                    new_kline = await self._fetch_klines_api(symbol, limit=1)
                    self._klines_cache[symbol] = pd.concat([
                        self._klines_cache[symbol].tail(199),
                        new_kline
                    ]).reset_index(drop=True)
                    
            except Exception as e:
                logger.warning(f"⚠️  Erreur update {symbol}: {e}")
    
    # =========================================
    # GESTION DES POSITIONS
    # =========================================
    
    async def _check_open_positions(self) -> None:
        """Vérifie les positions ouvertes (SL/TP/Timeout)."""
        for symbol, trade in list(self._open_trades.items()):
            current_price = self._price_cache.get(symbol, 0)
            
            if current_price <= 0:
                continue
            
            now = datetime.now()
            
            # Check Stop Loss
            if current_price <= trade.stop_loss_price:
                await self._close_trade(trade, current_price, TradeStatus.CLOSED_SL)
                continue
            
            # Check Take Profit
            if current_price >= trade.take_profit_price:
                await self._close_trade(trade, current_price, TradeStatus.CLOSED_TP)
                continue
            
            # Check Timeout
            if now >= trade.timeout_time:
                await self._close_trade(trade, current_price, TradeStatus.CLOSED_TIMEOUT)
                continue
    
    async def _close_trade(
        self,
        trade: PaperTrade,
        exit_price: float,
        status: TradeStatus
    ) -> None:
        """Ferme un trade."""
        trade.status = status
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        
        # Calculer le PnL
        price_change_pct = (exit_price - trade.entry_price) / trade.entry_price
        gross_pnl = trade.size_usdt * price_change_pct
        fees = trade.size_usdt * TRADING_FEES_PCT
        trade.pnl_usdt = gross_pnl - fees
        trade.pnl_pct = price_change_pct * 100 - (TRADING_FEES_PCT * 100)
        
        # Mettre à jour les stats
        self._update_stats_on_close(trade, fees)
        
        # Déplacer vers closed
        del self._open_trades[trade.symbol]
        self._closed_trades.append(trade)
        
        # Logger
        emoji = "🟢" if trade.pnl_usdt > 0 else "🔴"
        status_str = status.value.replace("CLOSED_", "")
        logger.info(
            f"{emoji} CLOSE {trade.symbol} | {status_str} | "
            f"PnL: {trade.pnl_usdt:+.2f} USDT ({trade.pnl_pct:+.2f}%) | "
            f"Durée: {trade.duration_minutes:.0f}min"
        )
    
    def _update_stats_on_close(self, trade: PaperTrade, fees: float) -> None:
        """Met à jour les statistiques après fermeture."""
        self._stats.total_trades += 1
        self._stats.total_pnl_usdt += trade.pnl_usdt
        self._stats.total_fees_usdt += fees
        self._stats.current_capital += trade.pnl_usdt
        
        if trade.pnl_usdt > 0:
            self._stats.winning_trades += 1
        else:
            self._stats.losing_trades += 1
        
        # Mise à jour peak et drawdown
        if self._stats.current_capital > self._stats.peak_capital:
            self._stats.peak_capital = self._stats.current_capital
        
        drawdown = (self._stats.peak_capital - self._stats.current_capital) / self._stats.peak_capital
        if drawdown > self._stats.max_drawdown_pct:
            self._stats.max_drawdown_pct = drawdown
    
    async def _close_all_positions(self) -> None:
        """Ferme toutes les positions ouvertes."""
        for symbol, trade in list(self._open_trades.items()):
            current_price = self._price_cache.get(symbol, trade.entry_price)
            await self._close_trade(trade, current_price, TradeStatus.CLOSED_TIMEOUT)
    
    # =========================================
    # SCANNER & SIGNAUX
    # =========================================
    
    async def _scan_opportunities(self) -> None:
        """Scanne les paires pour trouver des opportunités."""
        # Skip si déjà une position ouverte
        if self.open_positions_count > 0:
            return
        
        best_signal: Optional[tuple] = None  # (symbol, probability)
        
        for symbol in self._config.symbols:
            try:
                probability = self._compute_prediction(symbol)
                
                if probability is None:
                    continue
                
                if self._config.verbose:
                    signal_emoji = "🟢" if probability >= self._config.probability_threshold else "⚪"
                    logger.debug(f"   {signal_emoji} {symbol}: {probability:.2%}")
                
                # Garder le meilleur signal au-dessus du seuil
                if probability >= self._config.probability_threshold:
                    if best_signal is None or probability > best_signal[1]:
                        best_signal = (symbol, probability)
                        
            except Exception as e:
                logger.warning(f"⚠️  Erreur scan {symbol}: {e}")
        
        # Exécuter le meilleur signal
        if best_signal:
            await self._open_trade(best_signal[0], best_signal[1])
    
    def _compute_prediction(self, symbol: str) -> Optional[float]:
        """Calcule la prédiction ML pour un symbole."""
        df = self._klines_cache.get(symbol)
        if df is None or len(df) < 100:
            return None
        
        # Calculer les features
        feature_set = self._feature_engine.compute_features(df.tail(100), symbol=symbol)
        
        # Vérifier les NaN
        features = feature_set.features
        nan_count = sum(1 for v in features.values() if np.isnan(v))
        if nan_count > 5:  # Trop de NaN
            return None
        
        # Préparer le vecteur
        X = np.array([[features.get(name, 0) for name in self._feature_names]])
        
        # Remplacer les NaN par 0
        X = np.nan_to_num(X, nan=0.0)
        
        # Prédiction
        probability = self._model.predict_proba(X)[0, 1]
        
        return probability
    
    # =========================================
    # OUVERTURE DE TRADE
    # =========================================
    
    async def _open_trade(self, symbol: str, probability: float) -> None:
        """Ouvre un nouveau trade."""
        current_price = self._price_cache.get(symbol)
        if current_price is None or current_price <= 0:
            return
        
        # Calculer la taille de position
        size_usdt = self._stats.current_capital * self._config.position_size_pct
        quantity = size_usdt / current_price
        
        # Calculer SL et TP
        take_profit_price = current_price * (1 + self._config.take_profit_pct)
        stop_loss_price = current_price * (1 - self._config.stop_loss_pct)
        timeout_time = datetime.now() + timedelta(minutes=self._config.timeout_minutes)
        
        # Créer le trade
        self._trade_counter += 1
        trade = PaperTrade(
            id=self._trade_counter,
            symbol=symbol,
            entry_time=datetime.now(),
            entry_price=current_price,
            quantity=quantity,
            size_usdt=size_usdt,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            timeout_time=timeout_time,
            probability=probability,
        )
        
        self._open_trades[symbol] = trade
        
        # Logger
        logger.info(
            f"🔵 OPEN {symbol} | "
            f"Prix: {current_price:.4f} | "
            f"Size: {size_usdt:.2f} USDT | "
            f"TP: {take_profit_price:.4f} | "
            f"SL: {stop_loss_price:.4f} | "
            f"Proba: {probability:.2%}"
        )
    
    # =========================================
    # AFFICHAGE
    # =========================================
    
    def _print_status(self) -> None:
        """Affiche le statut actuel."""
        uptime = (datetime.now() - self._start_time).total_seconds() / 60
        
        logger.info("-" * 50)
        logger.info(f"📊 Status | Uptime: {uptime:.0f}min")
        logger.info(
            f"   Capital: {self._stats.current_capital:.2f} USDT | "
            f"PnL: {self._stats.net_pnl:+.2f} USDT"
        )
        logger.info(
            f"   Trades: {self._stats.total_trades} | "
            f"WR: {self._stats.win_rate:.1f}% | "
            f"Open: {self.open_positions_count}"
        )
        
        # Afficher les prix actuels
        prices_str = " | ".join([
            f"{s}: {self._price_cache.get(s, 0):.2f}"
            for s in self._config.symbols[:3]
        ])
        logger.info(f"   {prices_str}")
        logger.info("-" * 50)
    
    def _print_final_summary(self) -> None:
        """Affiche le résumé final."""
        duration = (datetime.now() - self._start_time).total_seconds() / 3600
        
        logger.info("\n" + "=" * 65)
        logger.info("📊 RÉSUMÉ FINAL - Paper Trading")
        logger.info("=" * 65)
        logger.info(f"   Durée: {duration:.1f} heures")
        logger.info(f"   Capital initial: {self._config.capital:.2f} USDT")
        logger.info(f"   Capital final: {self._stats.current_capital:.2f} USDT")
        logger.info(f"   PnL net: {self._stats.net_pnl:+.2f} USDT ({self._stats.net_pnl/self._config.capital*100:+.1f}%)")
        logger.info("-" * 65)
        logger.info(f"   Total trades: {self._stats.total_trades}")
        logger.info(f"   Gagnants: {self._stats.winning_trades}")
        logger.info(f"   Perdants: {self._stats.losing_trades}")
        logger.info(f"   Win Rate: {self._stats.win_rate:.1f}%")
        logger.info(f"   Max Drawdown: {self._stats.max_drawdown_pct*100:.1f}%")
        logger.info(f"   Frais totaux: {self._stats.total_fees_usdt:.2f} USDT")
        logger.info("=" * 65)
        
        # Statut final
        if self._stats.win_rate >= 40 and self._stats.net_pnl > 0:
            logger.info("✅ OBJECTIF ATTEINT: WR >= 40% et PnL positif")
        elif self._stats.win_rate >= 40:
            logger.info("⚠️  Win Rate OK mais PnL négatif")
        else:
            logger.info("❌ Win Rate insuffisant (< 40%)")
    
    # =========================================
    # EXPORT
    # =========================================
    
    def _save_trades_to_csv(self) -> None:
        """Sauvegarde les trades dans un fichier CSV."""
        log_path = Path(self._config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ajouter un timestamp au nom
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = log_path.parent / f"paper_trades_{timestamp}.csv"
        
        with open(final_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "symbol", "entry_time", "exit_time", "entry_price",
                "exit_price", "size_usdt", "pnl_usdt", "pnl_pct", "status",
                "probability", "duration_min"
            ])
            
            for trade in self._closed_trades:
                writer.writerow([
                    trade.id,
                    trade.symbol,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat() if trade.exit_time else "",
                    trade.entry_price,
                    trade.exit_price or "",
                    trade.size_usdt,
                    f"{trade.pnl_usdt:.4f}" if trade.pnl_usdt else "",
                    f"{trade.pnl_pct:.4f}" if trade.pnl_pct else "",
                    trade.status.value,
                    f"{trade.probability:.4f}",
                    f"{trade.duration_minutes:.1f}" if trade.duration_minutes else "",
                ])
        
        logger.info(f"💾 Trades sauvegardés: {final_path}")


# ============================================
# CLI
# ============================================

def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="CryptoScalper AI - Paper Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    python scripts/paper_trading.py
    python scripts/paper_trading.py --capital 30 --threshold 0.20
    python scripts/paper_trading.py --duration 2h --verbose
    python scripts/paper_trading.py --symbols BTCUSDT,ETHUSDT
        """
    )
    
    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_CAPITAL,
        help=f"Capital initial (défaut: {DEFAULT_CAPITAL} USDT)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Chemin du modèle (défaut: {DEFAULT_MODEL_PATH})"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(DEFAULT_SYMBOLS),
        help=f"Symboles à scanner (défaut: {','.join(DEFAULT_SYMBOLS)})"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_PROBABILITY_THRESHOLD,
        help=f"Seuil de probabilité (défaut: {DEFAULT_PROBABILITY_THRESHOLD})"
    )
    
    parser.add_argument(
        "--duration",
        type=str,
        default=None,
        help="Durée (ex: 1h, 2h, 24h, 7d)"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_SCAN_INTERVAL,
        help=f"Intervalle de scan en secondes (défaut: {DEFAULT_SCAN_INTERVAL})"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Dossier des données (défaut: {DEFAULT_DATA_DIR})"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbose (affiche chaque scan)"
    )
    
    return parser.parse_args()


def parse_duration(duration_str: str) -> float:
    """Parse une durée (1h, 2h, 24h, 7d) en heures."""
    if duration_str is None:
        return None
    
    duration_str = duration_str.lower().strip()
    
    if duration_str.endswith("h"):
        return float(duration_str[:-1])
    elif duration_str.endswith("d"):
        return float(duration_str[:-1]) * 24
    else:
        return float(duration_str)


def setup_logging(verbose: bool = False) -> None:
    """Configure le logging."""
    logger.remove()
    
    level = "DEBUG" if verbose else "INFO"
    
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # Log file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "paper_trading.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
        rotation="10 MB"
    )


# ============================================
# MAIN
# ============================================

async def main() -> int:
    """Point d'entrée principal."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Créer la config
    config = PaperTradeConfig(
        capital=args.capital,
        symbols=args.symbols.split(","),
        model_path=args.model,
        data_dir=args.data_dir,
        probability_threshold=args.threshold,
        scan_interval=args.interval,
        verbose=args.verbose,
        duration_hours=parse_duration(args.duration),
    )
    
    # Créer le trader
    trader = PaperTrader(config)
    
    # Gérer les signaux
    def signal_handler(sig, frame):
        logger.info("\n⚠️  Signal reçu, arrêt en cours...")
        asyncio.create_task(trader.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Démarrer
    try:
        await trader.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return 1
    finally:
        await trader.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))