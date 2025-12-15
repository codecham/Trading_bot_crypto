#!/usr/bin/env python3
# cryptoscalper/main.py
"""
CryptoScalper AI - Orchestrateur Principal.

Point d'entrée du bot de trading automatique.
Coordonne tous les modules : collecte, prédiction, signaux, risque, exécution.

Usage:
    # Paper trading (défaut, recommandé pour commencer)
    python main.py --mode paper
    
    # Live trading (ATTENTION: trades réels!)
    python main.py --mode live
    
    # Options
    python main.py --mode paper --capital 30 --log-level DEBUG

Fonctionnalités:
- Scan multi-paires en temps réel
- Prédictions ML avec XGBoost
- Gestion du risque stricte
- Exécution automatique (paper/live)
- Arrêt propre sur SIGINT/SIGTERM
"""

import argparse
import asyncio
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

# ============================================
# IMPORTS INTERNES
# ============================================

from cryptoscalper.config.settings import get_settings, Settings
from cryptoscalper.utils.logger import (
    setup_logger,
    logger,
    log_bot_status,
    log_trade_signal,
    log_trade_executed,
    log_trade_result,
)
from cryptoscalper.utils.exceptions import (
    CryptoScalperError,
    APIConnectionError,
    ConfigurationError,
)

# Import CloseReason pour la gestion des positions
from cryptoscalper.trading.executor import CloseReason


# ============================================
# ENUMS & TYPES
# ============================================

class BotMode(Enum):
    """Mode de fonctionnement du bot."""
    
    PAPER = "paper"   # Simulation sans ordres réels
    LIVE = "live"     # Trading réel


class BotStatus(Enum):
    """Statut du bot."""
    
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


# ============================================
# CONFIGURATION
# ============================================

@dataclass
class BotConfig:
    """
    Configuration du bot de trading.
    
    Attributes:
        mode: Mode paper ou live
        initial_capital: Capital initial en USDT
        model_path: Chemin vers le modèle ML
        scan_interval: Intervalle entre les scans (secondes)
        max_pairs: Nombre max de paires à surveiller
        min_volume_24h: Volume minimum 24h en USDT
        log_level: Niveau de log
    """
    
    mode: BotMode = BotMode.PAPER
    initial_capital: float = 30.0
    model_path: str = "models/saved/xgb_model_latest.joblib"
    scan_interval: float = 2.0
    max_pairs: int = 150
    min_volume_24h: float = 1_000_000
    log_level: str = "INFO"
    
    # Seuils ML (temporairement bas pour tester le pipeline)
    min_probability: float = 0.50  # 10% - TEMPORAIRE
    min_confidence: float = 0.50   # 50%
    
    # Risk
    max_position_pct: float = 0.20
    default_stop_loss_pct: float = 0.004  # 0.4%
    default_take_profit_pct: float = 0.003  # 0.3%
    position_timeout_seconds: int = 300  # 5 minutes
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BotConfig":
        """Crée une config depuis les arguments CLI."""
        return cls(
            mode=BotMode(args.mode),
            initial_capital=args.capital,
            model_path=args.model,
            scan_interval=args.interval,
            log_level=args.log_level.upper(),
        )


# ============================================
# BOT STATE
# ============================================

@dataclass
class BotState:
    """
    État global du bot.
    
    Contient toutes les métriques et statistiques en temps réel.
    """
    
    # Status
    status: BotStatus = BotStatus.INITIALIZING
    start_time: Optional[datetime] = None
    
    # Capital
    initial_capital: float = 30.0
    current_capital: float = 30.0
    peak_capital: float = 30.0
    
    # Positions
    open_positions: int = 0
    open_symbols: Set[str] = field(default_factory=set)
    
    # Trading stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    
    # Scan stats
    total_scans: int = 0
    signals_generated: int = 0
    signals_executed: int = 0
    
    # Errors
    consecutive_errors: int = 0
    last_error: Optional[str] = None
    
    @property
    def win_rate(self) -> float:
        """Taux de réussite."""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.0
        return self.winning_trades / total
    
    @property
    def uptime_seconds(self) -> float:
        """Durée de fonctionnement en secondes."""
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def uptime_str(self) -> str:
        """Durée de fonctionnement formatée."""
        seconds = int(self.uptime_seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    @property
    def drawdown_pct(self) -> float:
        """Drawdown actuel en pourcentage."""
        if self.peak_capital <= 0:
            return 0.0
        return (self.peak_capital - self.current_capital) / self.peak_capital


# ============================================
# TRADING BOT
# ============================================

class TradingBot:
    """
    Orchestrateur principal du bot de trading.
    
    Coordonne tous les modules:
    - DataCollector: collecte temps réel
    - MultiPairScanner: détection opportunités
    - FeatureEngine: calcul features
    - MLPredictor: prédictions
    - SignalGenerator: génération signaux
    - RiskManager: gestion risque
    - TradeExecutor: exécution ordres
    - TradeLogger: logging trades
    
    Usage:
        config = BotConfig(mode=BotMode.PAPER)
        bot = TradingBot(config)
        
        await bot.start()
        # Le bot tourne jusqu'à SIGINT
        await bot.wait_for_shutdown()
    """
    
    def __init__(self, config: BotConfig):
        """
        Initialise le bot.
        
        Args:
            config: Configuration du bot
        """
        self._config = config
        self._state = BotState(
            initial_capital=config.initial_capital,
            current_capital=config.initial_capital,
            peak_capital=config.initial_capital,
        )
        
        # Modules (initialisés dans start())
        self._binance_client = None  # Client Binance principal
        self._collector = None
        self._scanner = None
        self._feature_engine = None
        self._predictor = None
        self._signal_generator = None
        self._risk_manager = None
        self._executor = None
        self._trade_logger = None
        self._symbols_manager = None
        
        # Control
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._main_task: Optional[asyncio.Task] = None
        
        # Daily reset
        self._last_daily_reset = datetime.now().date()
    
    # =========================================
    # PROPERTIES
    # =========================================
    
    @property
    def config(self) -> BotConfig:
        """Configuration du bot."""
        return self._config
    
    @property
    def state(self) -> BotState:
        """État actuel du bot."""
        return self._state
    
    @property
    def is_running(self) -> bool:
        """Le bot est-il en cours d'exécution?"""
        return self._running
    
    @property
    def mode(self) -> BotMode:
        """Mode actuel (paper/live)."""
        return self._config.mode
    
    # =========================================
    # LIFECYCLE
    # =========================================
    
    async def start(self) -> None:
        """
        Démarre le bot.
        
        1. Initialise tous les modules
        2. Lance la boucle principale
        3. Configure les handlers de signaux
        """
        if self._running:
            logger.warning("⚠️ Bot déjà en cours d'exécution")
            return
        
        self._state.status = BotStatus.INITIALIZING
        logger.info("=" * 65)
        logger.info("🤖 CryptoScalper AI - Démarrage")
        logger.info("=" * 65)
        logger.info(f"   Mode: {self._config.mode.value.upper()}")
        logger.info(f"   Capital: {self._config.initial_capital:.2f} USDT")
        logger.info(f"   Modèle: {self._config.model_path}")
        logger.info("=" * 65)
        
        try:
            # 1. Initialiser les modules
            await self._initialize_modules()
            
            # 2. Configurer les handlers de signaux
            self._setup_signal_handlers()
            
            # 3. Démarrer
            self._running = True
            self._state.status = BotStatus.RUNNING
            self._state.start_time = datetime.now()
            
            logger.info("✅ Bot démarré avec succès!")
            logger.info("-" * 65)
            
            # 4. Lancer la boucle principale
            self._main_task = asyncio.create_task(self._main_loop())
            
        except Exception as e:
            self._state.status = BotStatus.ERROR
            self._state.last_error = str(e)
            logger.error(f"❌ Erreur au démarrage: {e}")
            raise
    
    async def stop(self) -> None:
        """
        Arrête proprement le bot.
        
        1. Ferme les positions ouvertes (optionnel)
        2. Arrête les modules
        3. Sauvegarde les logs
        """
        if not self._running:
            return
        
        logger.info("-" * 65)
        logger.info("🛑 Arrêt du bot demandé...")
        
        self._state.status = BotStatus.STOPPING
        self._running = False
        self._shutdown_event.set()
        
        # Attendre la fin de la boucle principale
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        
        # Arrêter les modules
        await self._shutdown_modules()
        
        # Afficher le résumé final
        self._print_final_summary()
        
        self._state.status = BotStatus.STOPPED
        logger.info("✅ Bot arrêté proprement")
    
    async def wait_for_shutdown(self) -> None:
        """Attend le signal d'arrêt."""
        await self._shutdown_event.wait()
    
    # =========================================
    # INITIALIZATION
    # =========================================
    
    async def _initialize_modules(self) -> None:
        """Initialise tous les modules du bot."""
        
        # Import dynamique pour éviter les imports circulaires
        from cryptoscalper.data.binance_client import BinanceClient
        from cryptoscalper.data.websocket_manager import WebSocketManager
        from cryptoscalper.data.collector import DataCollector, CollectorConfig
        from cryptoscalper.data.symbols import SymbolsManager
        from cryptoscalper.data.multi_pair_scanner import (
            MultiPairScanner,
            ScannerConfig,
        )
        from cryptoscalper.data.features import FeatureEngine
        from cryptoscalper.models.predictor import MLPredictor
        from cryptoscalper.trading.signals import SignalGenerator, SignalConfig
        from cryptoscalper.trading.risk_manager import RiskManager, RiskConfig
        from cryptoscalper.trading.executor import TradeExecutor, ExecutorConfig, CloseReason
        from cryptoscalper.utils.trade_logger import TradeLogger
        
        logger.info("📦 Initialisation des modules...")
        
        # 0. Binance Client (nécessaire pour SymbolsManager)
        logger.info("  0/10 BinanceClient...")
        self._binance_client = BinanceClient()
        await self._binance_client.connect()
        logger.info("      ✓ Connecté à Binance")
        
        # 1. Symbols Manager - Sélection dynamique des paires
        logger.info("  1/10 SymbolsManager...")
        self._symbols_manager = SymbolsManager(
            client=self._binance_client._client,
            min_volume_24h=self._config.min_volume_24h,
            max_pairs=self._config.max_pairs,
        )
        await self._symbols_manager.start()
        symbols = self._symbols_manager.symbols
        logger.info(f"      ✓ {len(symbols)} paires sélectionnées")
        
        # 2. Data Collector
        logger.info("  2/10 DataCollector...")
        collector_config = CollectorConfig(
            symbols=symbols,
            enable_websocket=True,
            subscribe_klines=True,
            subscribe_depth=True,
        )
        self._collector = DataCollector(collector_config)
        await self._collector.start()
        logger.info("      ✓ Collecteur démarré")
        
        # 3. Multi-Pair Scanner
        logger.info("  3/10 MultiPairScanner...")
        scanner_config = ScannerConfig(
            min_score_for_alert=0.3,
        )
        self._scanner = MultiPairScanner(
            self._collector._ws_manager,
            scanner_config,
        )
        await self._scanner.start(scan_interval=self._config.scan_interval)
        logger.info("      ✓ Scanner actif")
        
        # 4. Feature Engine
        logger.info("  4/10 FeatureEngine...")
        self._feature_engine = FeatureEngine()
        logger.info(f"      ✓ {self._feature_engine.feature_count} features")
        
        # 5. ML Predictor
        logger.info("  5/10 MLPredictor...")
        model_path = Path(self._config.model_path)
        if model_path.exists():
            self._predictor = MLPredictor.from_file(model_path)
            logger.info("      ✓ Modèle chargé")
        else:
            logger.warning(f"      ⚠️ Modèle non trouvé: {model_path}")
            logger.warning("      → Le bot fonctionnera sans ML (signaux scanner uniquement)")
            self._predictor = None
        
        # 6. Signal Generator
        logger.info("  6/10 SignalGenerator...")
        signal_config = SignalConfig(
            min_probability=self._config.min_probability,
            min_confidence=self._config.min_confidence,
            default_stop_loss_pct=self._config.default_stop_loss_pct,
            default_take_profit_pct=self._config.default_take_profit_pct,
        )
        self._signal_generator = SignalGenerator(signal_config)
        logger.info("      ✓ Générateur de signaux prêt")
        
        # 7. Risk Manager
        logger.info("  7/10 RiskManager...")
        risk_config = RiskConfig(
            initial_capital=self._config.initial_capital,
            max_position_pct=self._config.max_position_pct,
            max_open_positions=1,  # Un seul trade à la fois
        )
        self._risk_manager = RiskManager(risk_config)
        logger.info("      ✓ Risk manager actif")
        
        # 8. Trade Executor
        logger.info("  8/10 TradeExecutor...")
        is_paper = self._config.mode == BotMode.PAPER
        executor_config = ExecutorConfig(
            testnet=True,  # Toujours testnet pour la sécurité
            paper_trading=is_paper,
        )
        self._executor = TradeExecutor(executor_config)
        await self._executor.connect()
        mode_str = "PAPER" if is_paper else "LIVE"
        logger.info(f"      ✓ Executor mode {mode_str}")
        
        # 9. Trade Logger
        logger.info("  9/10 TradeLogger...")
        self._trade_logger = TradeLogger(csv_path="logs/trades.csv")
        logger.info("      ✓ Logger trades actif")
        
        logger.info("✅ Tous les modules initialisés")
    
    async def _shutdown_modules(self) -> None:
        """Arrête tous les modules proprement."""
        logger.info("📦 Arrêt des modules...")
        
        # Ordre inverse d'initialisation
        if self._executor:
            await self._executor.disconnect()
            logger.info("  ✓ Executor arrêté")
        
        if self._scanner:
            await self._scanner.stop()
            logger.info("  ✓ Scanner arrêté")
        
        if self._collector:
            await self._collector.stop()
            logger.info("  ✓ Collector arrêté")
        
        if self._symbols_manager:
            await self._symbols_manager.stop()
            logger.info("  ✓ SymbolsManager arrêté")
        
        if self._binance_client:
            await self._binance_client.disconnect()
            logger.info("  ✓ BinanceClient déconnecté")
    
    # =========================================
    # SIGNAL HANDLERS
    # =========================================
    
    def _setup_signal_handlers(self) -> None:
        """Configure les handlers pour SIGINT et SIGTERM."""
        loop = asyncio.get_event_loop()
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s))
            )
    
    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Gère les signaux d'arrêt."""
        sig_name = sig.name
        logger.info(f"\n📡 Signal reçu: {sig_name}")
        await self.stop()
    
    # =========================================
    # MAIN LOOP
    # =========================================
    
    async def _main_loop(self) -> None:
        """
        Boucle principale du bot.
        
        Cycle:
        1. Vérifier le reset journalier
        2. Scanner les opportunités
        3. Analyser les candidats (ML)
        4. Générer les signaux
        5. Vérifier le risque
        6. Exécuter les trades
        7. Gérer les positions ouvertes
        8. Afficher le statut
        """
        logger.info("🔄 Boucle principale démarrée")
        
        status_interval = 30  # Afficher le statut toutes les 30s
        last_status_time = datetime.now()
        
        while self._running:
            try:
                # 1. Reset journalier
                self._check_daily_reset()
                
                # 2. Vérifier les positions ouvertes (timeouts, etc.)
                await self._manage_open_positions()
                
                # 3. Scanner les opportunités
                candidates = self._scanner.get_top_opportunities(n=10)
                self._state.total_scans += 1
                
                if candidates and self._can_open_new_trade():
                    # 4. Analyser les candidats
                    await self._analyze_candidates(candidates)
                
                # 5. Afficher le statut périodiquement
                if (datetime.now() - last_status_time).total_seconds() >= status_interval:
                    self._log_status()
                    last_status_time = datetime.now()
                
                # Reset erreurs consécutives
                self._state.consecutive_errors = 0
                
                # Attendre avant le prochain cycle
                await asyncio.sleep(self._config.scan_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._state.consecutive_errors += 1
                self._state.last_error = str(e)
                logger.error(f"❌ Erreur dans la boucle principale: {e}")
                
                # Si trop d'erreurs consécutives, arrêter
                if self._state.consecutive_errors >= 5:
                    logger.critical("🚨 Trop d'erreurs consécutives, arrêt d'urgence!")
                    self._state.status = BotStatus.ERROR
                    break
                
                await asyncio.sleep(5)  # Pause avant de réessayer
    
    # =========================================
    # TRADING LOGIC
    # =========================================
    
    def _can_open_new_trade(self) -> bool:
        """Vérifie si on peut ouvrir un nouveau trade."""
        # Pas de trade si kill switch actif
        if self._risk_manager.is_kill_switch_active:
            return False
        
        # Vérifier via risk manager
        can_trade, reason = self._risk_manager.can_open_trade()
        if not can_trade:
            return False
        
        return True
    
    async def _analyze_candidates(self, candidates: list) -> None:
        """
        Analyse les candidats et exécute si opportunité trouvée.
        
        Args:
            candidates: Liste des PairState des meilleures opportunités
        """
        for pair_state in candidates:
            symbol = pair_state.symbol
            
            # Skip si déjà en position
            if symbol in self._state.open_symbols:
                continue
            
            try:
                # Calculer les features
                features = await self._compute_features_for_symbol(symbol)
                
                if features is None:
                    logger.debug(f"Features NULL pour {symbol}")
                    continue
                
                # Prédiction ML (si modèle disponible)
                if self._predictor:
                    prediction = self._predictor.predict(features)
                    
                    # LOG DES PRÉDICTIONS pour debug
                    logger.debug(
                        f"🔮 ML {symbol}: proba={prediction.probability_up:.1%} "
                        f"conf={prediction.confidence:.1%} "
                        f"(seuils: {self._config.min_probability:.0%}/{self._config.min_confidence:.0%})"
                    )
                    
                    # Vérifier les seuils
                    if prediction.probability_up < self._config.min_probability:
                        continue
                    if prediction.confidence < self._config.min_confidence:
                        continue
                    
                    # Signal validé !
                    logger.info(
                        f"✨ Signal ML validé {symbol}: "
                        f"proba={prediction.probability_up:.1%} conf={prediction.confidence:.1%}"
                    )
                    
                    # Générer le signal
                    signal = self._signal_generator.generate_signal(
                        prediction=prediction,
                        current_price=pair_state.current_price,
                    )
                    
                    if signal:
                        self._state.signals_generated += 1
                        await self._execute_signal(signal)
                        break  # Un seul trade à la fois
                
                else:
                    # Sans ML, utiliser le score du scanner
                    if pair_state.score >= 0.7:  # Seuil arbitraire
                        logger.info(
                            f"📊 Opportunité scanner: {symbol} | "
                            f"Score: {pair_state.score:.2f}"
                        )
                
            except Exception as e:
                logger.error(f"Erreur analyse {symbol}: {e}")
    
    async def _compute_features_for_symbol(self, symbol: str):
        """
        Calcule les features pour un symbole.
        
        Returns:
            FeatureSet ou None si erreur
        """
        try:
            # Récupérer les données (retourne List[Kline])
            klines_list = await self._collector.fetch_klines(symbol, limit=100)
            depth_data = self._collector.get_depth(symbol)
            
            # Vérifier qu'on a des données
            if not klines_list or len(klines_list) < 50:
                return None
            
            # Convertir List[Kline] en DataFrame
            import pandas as pd
            klines_data = []
            for k in klines_list:
                klines_data.append({
                    'open_time': k.open_time,
                    'open': k.open,
                    'high': k.high,
                    'low': k.low,
                    'close': k.close,
                    'volume': k.volume,
                    'close_time': k.close_time,
                    'quote_volume': k.quote_volume,
                    'trades_count': k.trades_count,
                })
            df = pd.DataFrame(klines_data)
            
            if df.empty:
                return None
            
            # Préparer l'orderbook (si disponible)
            orderbook = None
            if depth_data:
                from cryptoscalper.data.features import OrderbookData
                # Convertir DepthData en OrderbookData pour FeatureEngine
                orderbook = OrderbookData(
                    bids=depth_data.bids if hasattr(depth_data, 'bids') else [],
                    asks=depth_data.asks if hasattr(depth_data, 'asks') else [],
                )
            
            # Calculer les features
            features = self._feature_engine.compute_features(
                df=df,
                orderbook=orderbook,
                symbol=symbol,
            )
            
            return features
            
        except Exception as e:
            logger.debug(f"Erreur features {symbol}: {e}")
            return None
    
    async def _execute_signal(self, signal) -> None:
        """
        Exécute un signal de trading.
        
        Args:
            signal: TradeSignal à exécuter
        """
        symbol = signal.symbol
        
        # Calculer la taille de position
        size_usdt = self._risk_manager.calculate_position_size(
            signal=signal,
            current_capital=self._state.current_capital,
        )
        
        if size_usdt <= 0:
            logger.debug(f"Position trop petite pour {symbol}")
            return
        
        # Log du signal
        log_trade_signal(
            symbol=symbol,
            action="BUY",
            price=signal.entry_price or 0.0,
            confidence=signal.probability,
        )
        
        # Exécuter
        try:
            position = await self._executor.execute_signal(
                signal=signal,
                size_usdt=size_usdt,
            )
            
            if position:
                # Mise à jour de l'état
                self._state.open_positions += 1
                self._state.open_symbols.add(symbol)
                self._state.signals_executed += 1
                self._state.total_trades += 1
                self._state.daily_trades += 1
                
                log_trade_executed(
                    symbol=symbol,
                    side="BUY",
                    quantity=position.quantity,
                    price=position.entry_price,
                    order_id=position.entry_order_id,
                )
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution {symbol}: {e}")
    
    async def _manage_open_positions(self) -> None:
        """Gère les positions ouvertes (timeouts, etc.)."""
        if not self._executor:
            return
        
        positions = self._executor.open_positions
        
        for position in positions:
            # Vérifier timeout
            duration = (datetime.now() - position.entry_time).total_seconds()
            
            if duration > self._config.position_timeout_seconds:
                logger.info(f"⏰ Timeout position {position.symbol}")
                
                try:
                    completed = await self._executor.close_position(
                        position,
                        reason=CloseReason.TIMEOUT,
                    )
                    
                    if completed:
                        self._record_trade_result(completed)
                        
                except Exception as e:
                    logger.error(f"Erreur fermeture {position.symbol}: {e}")
    
    def _record_trade_result(self, completed_trade) -> None:
        """Enregistre le résultat d'un trade fermé."""
        symbol = completed_trade.symbol
        pnl = completed_trade.pnl_usdt
        
        # Mise à jour état local
        self._state.open_positions = max(0, self._state.open_positions - 1)
        self._state.open_symbols.discard(symbol)
        self._state.total_pnl += pnl
        self._state.daily_pnl += pnl
        self._state.current_capital += pnl
        
        if pnl > 0:
            self._state.winning_trades += 1
        else:
            self._state.losing_trades += 1
        
        # Mise à jour peak
        if self._state.current_capital > self._state.peak_capital:
            self._state.peak_capital = self._state.current_capital
        
        # Enregistrer dans le risk manager
        if self._risk_manager:
            try:
                self._risk_manager.register_trade_result(completed_trade)
            except Exception as e:
                # Kill switch activated?
                logger.warning(f"⚠️ RiskManager exception: {e}")
        
        # Log
        log_trade_result(
            symbol=symbol,
            pnl_usdt=pnl,
            pnl_percent=completed_trade.pnl_percent,
            duration_seconds=int(completed_trade.duration_seconds),
        )
        
        # Sauvegarder dans trade logger
        if self._trade_logger:
            self._trade_logger.log_trade(completed_trade)
    
    # =========================================
    # DAILY RESET
    # =========================================
    
    def _check_daily_reset(self) -> None:
        """Vérifie et effectue le reset journalier si nécessaire."""
        today = datetime.now().date()
        
        if today > self._last_daily_reset:
            logger.info("🌅 Nouveau jour - Reset des statistiques journalières")
            
            self._state.daily_pnl = 0.0
            self._state.daily_trades = 0
            
            if self._risk_manager:
                self._risk_manager.reset_daily_stats()
            
            self._last_daily_reset = today
    
    # =========================================
    # STATUS & LOGGING
    # =========================================
    
    def _log_status(self) -> None:
        """Affiche le statut périodique du bot."""
        log_bot_status(
            status=self._state.status.value,
            capital=self._state.current_capital,
            open_positions=self._state.open_positions,
            daily_pnl=self._state.daily_pnl,
        )
        
        # Stats supplémentaires en debug
        logger.debug(
            f"📊 Stats: Scans={self._state.total_scans} | "
            f"Trades={self._state.total_trades} | "
            f"WR={self._state.win_rate:.1%} | "
            f"Uptime={self._state.uptime_str}"
        )
    
    def _print_final_summary(self) -> None:
        """Affiche le résumé final à l'arrêt."""
        logger.info("=" * 65)
        logger.info("📊 RÉSUMÉ FINAL")
        logger.info("=" * 65)
        logger.info(f"   Durée: {self._state.uptime_str}")
        logger.info(f"   Mode: {self._config.mode.value.upper()}")
        logger.info("-" * 65)
        logger.info(f"   Capital initial: {self._state.initial_capital:.2f} USDT")
        logger.info(f"   Capital final: {self._state.current_capital:.2f} USDT")
        logger.info(f"   PnL total: {self._state.total_pnl:+.2f} USDT")
        logger.info("-" * 65)
        logger.info(f"   Trades: {self._state.total_trades}")
        logger.info(f"   Gagnants: {self._state.winning_trades}")
        logger.info(f"   Perdants: {self._state.losing_trades}")
        logger.info(f"   Win rate: {self._state.win_rate:.1%}")
        logger.info("-" * 65)
        logger.info(f"   Scans effectués: {self._state.total_scans}")
        logger.info(f"   Signaux générés: {self._state.signals_generated}")
        logger.info(f"   Signaux exécutés: {self._state.signals_executed}")
        logger.info("=" * 65)


# ============================================
# CLI
# ============================================

def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="CryptoScalper AI - Bot de Trading Automatique",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py --mode paper              # Paper trading (défaut)
  python main.py --mode live               # Live trading (ATTENTION!)
  python main.py --mode paper --capital 50 # Paper avec 50€
  python main.py --mode paper --log-level DEBUG
        """,
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["paper", "live"],
        default="paper",
        help="Mode de trading: paper (simulation) ou live (réel)",
    )
    
    parser.add_argument(
        "--capital",
        type=float,
        default=30.0,
        help="Capital initial en USDT (défaut: 30)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/saved/xgb_model_latest.joblib",
        help="Chemin vers le modèle ML",
    )
    
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Intervalle de scan en secondes (défaut: 2)",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Niveau de log (défaut: INFO)",
    )
    
    return parser.parse_args()


async def main() -> int:
    """Point d'entrée principal."""
    args = parse_args()
    
    # Setup logging
    setup_logger(level=args.log_level.upper())
    
    # Confirmation pour le mode live
    if args.mode == "live":
        print("\n" + "=" * 65)
        print("⚠️  ATTENTION: MODE LIVE TRADING")
        print("=" * 65)
        print("Vous êtes sur le point de lancer le bot en mode LIVE.")
        print("Cela signifie que de VRAIS ordres seront passés sur Binance.")
        print(f"Capital à risque: {args.capital:.2f} USDT")
        print("=" * 65)
        
        confirm = input("Tapez 'CONFIRM' pour continuer: ")
        if confirm != "CONFIRM":
            print("Annulé.")
            return 1
        print()
    
    # Créer la configuration
    config = BotConfig.from_args(args)
    
    # Créer et lancer le bot
    bot = TradingBot(config)
    
    try:
        await bot.start()
        await bot.wait_for_shutdown()
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⌨️ Interruption clavier")
        await bot.stop()
        return 0
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        await bot.stop()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))