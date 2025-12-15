# cryptoscalper/data/multi_pair_scanner.py
"""
Scanner multi-paires temps réel pour détecter les opportunités de trading.

Surveille 100-200 paires simultanément et détecte :
- Volume spikes (pics de volume anormaux)
- Momentum (accélération des prix)
- Breakouts (cassure de niveaux)

Architecture en 2 étapes :
1. Scan large : filtres rapides sur toutes les paires
2. Analyse profonde : ML complet sur les candidates uniquement
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable

from cryptoscalper.data.websocket_manager import WebSocketManager, PairState, TickerData
from cryptoscalper.data.symbols import SymbolsManager
from cryptoscalper.config.settings import get_settings
from cryptoscalper.utils.logger import logger


class AlertType(Enum):
    """Types d'alertes du scanner."""
    
    VOLUME_SPIKE = "VOLUME_SPIKE"      # Pic de volume anormal
    MOMENTUM = "MOMENTUM"              # Momentum fort et accélérant
    BREAKOUT = "BREAKOUT"              # Cassure proche du high 24h
    REVERSAL = "REVERSAL"              # Retournement potentiel
    COMBINED = "COMBINED"              # Plusieurs signaux combinés


@dataclass
class ScannerAlert:
    """
    Alerte générée par le scanner.
    
    Représente une opportunité potentielle détectée.
    """
    
    symbol: str
    alert_type: AlertType
    score: float  # 0.0 à 1.0
    details: Dict
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return f"[{self.alert_type.value}] {self.symbol} (score: {self.score:.2f})"


@dataclass
class ScannerConfig:
    """Configuration du scanner multi-paires."""
    
    # Filtres de base
    min_volume_24h: float = 1_000_000  # Volume min 24h en USDT
    max_spread_percent: float = 0.15    # Spread max acceptable (0.15%)
    
    # Seuils de détection
    volume_spike_ratio: float = 3.0     # Ratio vs moyenne pour spike
    momentum_threshold_1m: float = 0.15  # Mouvement min 1m (0.15%)
    momentum_threshold_5m: float = 0.30  # Mouvement min 5m (0.30%)
    breakout_distance_percent: float = 0.5  # Distance du high 24h
    
    # Scoring
    min_score_for_alert: float = 0.3    # Score min pour générer une alerte
    
    # Callback
    alert_callback: Optional[Callable[[ScannerAlert], None]] = None


@dataclass
class ScannerStats:
    """Statistiques du scanner."""
    
    start_time: Optional[datetime] = None
    scans_count: int = 0
    alerts_generated: int = 0
    alerts_by_type: Dict[str, int] = field(default_factory=dict)
    last_scan_time: Optional[datetime] = None
    last_scan_duration_ms: float = 0.0
    
    @property
    def uptime_seconds(self) -> float:
        """Durée depuis le démarrage."""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0


class MultiPairScanner:
    """
    Scanner temps réel pour 100-200 paires.
    
    Détecte les opportunités via des filtres rapides et génère
    des alertes pour les paires les plus prometteuses.
    
    Usage:
        scanner = MultiPairScanner(ws_manager, config)
        await scanner.start()
        
        # Récupérer les meilleures opportunités
        opportunities = scanner.get_top_opportunities(n=10)
        
        await scanner.stop()
    """
    
    def __init__(
        self,
        ws_manager: WebSocketManager,
        config: Optional[ScannerConfig] = None
    ):
        """
        Initialise le scanner.
        
        Args:
            ws_manager: WebSocketManager connecté et actif
            config: Configuration du scanner
        """
        self._ws_manager = ws_manager
        self._config = config or ScannerConfig()
        self._settings = get_settings()
        
        # État
        self._running = False
        self._scan_task: Optional[asyncio.Task] = None
        
        # Alertes récentes (garde les 100 dernières)
        self._recent_alerts: List[ScannerAlert] = []
        self._max_alerts = 100
        
        # Cache des moyennes de volume (calculé périodiquement)
        self._volume_averages: Dict[str, float] = {}
        
        # Stats
        self.stats = ScannerStats()
    
    @property
    def is_running(self) -> bool:
        """Vérifie si le scanner est actif."""
        return self._running
    
    @property
    def alerts(self) -> List[ScannerAlert]:
        """Liste des alertes récentes."""
        return list(self._recent_alerts)
    
    async def start(self, scan_interval: float = 2.0) -> None:
        """
        Démarre le scanner.
        
        Args:
            scan_interval: Intervalle entre les scans en secondes
        """
        if self._running:
            logger.warning("Scanner déjà en cours d'exécution")
            return
        
        logger.info("🔍 Démarrage du scanner multi-paires...")
        
        self._running = True
        self.stats = ScannerStats(start_time=datetime.now())
        
        # Initialiser les alertes par type
        for alert_type in AlertType:
            self.stats.alerts_by_type[alert_type.value] = 0
        
        # Lancer la boucle de scan
        self._scan_task = asyncio.create_task(
            self._scan_loop(scan_interval)
        )
        
        logger.info("✅ Scanner multi-paires démarré")
    
    async def stop(self) -> None:
        """Arrête le scanner."""
        if not self._running:
            return
        
        logger.info("🛑 Arrêt du scanner...")
        
        self._running = False
        
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Scanner arrêté")
    
    async def _scan_loop(self, interval: float) -> None:
        """Boucle principale de scan."""
        while self._running:
            try:
                await self._perform_scan()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur dans le scan: {e}")
                await asyncio.sleep(1)
    
    async def _perform_scan(self) -> None:
        """Effectue un scan complet de toutes les paires."""
        start_time = datetime.now()
        
        # Récupérer tous les états des paires
        all_states = self._ws_manager.get_all_states()
        
        if not all_states:
            return
        
        # Scanner chaque paire
        for symbol, state in all_states.items():
            self._scan_pair(symbol, state)
        
        # Mettre à jour les stats
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        self.stats.scans_count += 1
        self.stats.last_scan_time = datetime.now()
        self.stats.last_scan_duration_ms = duration_ms
    
    def _scan_pair(self, symbol: str, state: PairState) -> None:
        """
        Scanne une paire et génère des alertes si opportunité.
        
        DOIT ÊTRE RAPIDE (< 1ms par paire).
        """
        # Vérifications de base
        if not self._passes_basic_filters(state):
            return
        
        # Détecter les patterns
        alerts = []
        
        # 1. Volume Spike
        volume_alert = self._detect_volume_spike(state)
        if volume_alert:
            alerts.append(volume_alert)
        
        # 2. Momentum
        momentum_alert = self._detect_momentum(state)
        if momentum_alert:
            alerts.append(momentum_alert)
        
        # 3. Breakout
        breakout_alert = self._detect_breakout(state)
        if breakout_alert:
            alerts.append(breakout_alert)
        
        # Générer les alertes
        for alert in alerts:
            self._emit_alert(alert)
    
    def _passes_basic_filters(self, state: PairState) -> bool:
        """Vérifie les filtres de base (rapide)."""
        # Prix valide
        if state.current_price <= 0:
            return False
        
        # Volume minimum
        if state.volume_24h < self._config.min_volume_24h:
            return False
        
        # Spread acceptable (si orderbook disponible)
        if state.current_depth:
            spread = state.current_depth.spread_percent
            if spread and spread > self._config.max_spread_percent:
                return False
        
        return True
    
    def _detect_volume_spike(self, state: PairState) -> Optional[ScannerAlert]:
        """
        Détecte un pic de volume anormal.
        
        Compare le volume récent à la moyenne.
        """
        # Pour un vrai spike, il faudrait comparer au volume moyen
        # Ici on utilise une heuristique simple basée sur le volume 24h
        
        # Volume moyen par minute sur 24h
        avg_volume_per_min = state.volume_24h / 1440 if state.volume_24h > 0 else 0
        
        if avg_volume_per_min <= 0:
            return None
        
        # On n'a pas le volume par minute en temps réel dans PairState actuel
        # Cette détection sera améliorée quand on aura les trades en temps réel
        
        return None  # À implémenter avec plus de données
    
    def _detect_momentum(self, state: PairState) -> Optional[ScannerAlert]:
        """
        Détecte un momentum fort et accélérant.
        
        Critères :
        - Mouvement > seuil sur 1 min
        - Accélération (1m > 5m proportionnellement)
        """
        change_1m = state.change_1m_percent
        change_5m = state.change_5m_percent
        
        if change_1m is None:
            return None
        
        # Vérifier le seuil de momentum
        if abs(change_1m) < self._config.momentum_threshold_1m:
            return None
        
        # Calculer le score
        score = self._calculate_momentum_score(change_1m, change_5m)
        
        if score < self._config.min_score_for_alert:
            return None
        
        # Direction du momentum
        direction = "UP" if change_1m > 0 else "DOWN"
        
        return ScannerAlert(
            symbol=state.symbol,
            alert_type=AlertType.MOMENTUM,
            score=score,
            details={
                "change_1m": change_1m,
                "change_5m": change_5m,
                "direction": direction,
                "price": state.current_price
            }
        )
    
    def _calculate_momentum_score(
        self,
        change_1m: float,
        change_5m: Optional[float]
    ) -> float:
        """Calcule le score de momentum (0-1)."""
        score = 0.0
        
        # Score basé sur l'amplitude du mouvement
        amplitude = abs(change_1m)
        if amplitude > 0.5:
            score += 0.4
        elif amplitude > 0.3:
            score += 0.3
        elif amplitude > 0.15:
            score += 0.2
        
        # Bonus si accélération
        if change_5m is not None and change_5m != 0:
            # Ratio d'accélération : change_1m devrait être > change_5m/5
            expected_1m = change_5m / 5
            if abs(change_1m) > abs(expected_1m) * 1.5:
                score += 0.3  # Forte accélération
            elif abs(change_1m) > abs(expected_1m):
                score += 0.15  # Accélération modérée
        
        # Bonus si même direction sur les deux périodes
        if change_5m is not None:
            if (change_1m > 0 and change_5m > 0) or (change_1m < 0 and change_5m < 0):
                score += 0.1
        
        return min(score, 1.0)
    
    def _detect_breakout(self, state: PairState) -> Optional[ScannerAlert]:
        """
        Détecte un breakout potentiel.
        
        Critères :
        - Prix proche du high 24h
        - Momentum positif
        """
        if state.high_24h <= 0 or state.current_price <= 0:
            return None
        
        # Distance du high 24h
        distance_from_high = (state.high_24h - state.current_price) / state.high_24h * 100
        
        # Vérifier si proche du high
        if distance_from_high > self._config.breakout_distance_percent:
            return None
        
        # Vérifier le momentum positif
        change_1m = state.change_1m_percent
        if change_1m is None or change_1m <= 0:
            return None
        
        # Calculer le score
        score = self._calculate_breakout_score(distance_from_high, change_1m)
        
        if score < self._config.min_score_for_alert:
            return None
        
        return ScannerAlert(
            symbol=state.symbol,
            alert_type=AlertType.BREAKOUT,
            score=score,
            details={
                "distance_from_high_percent": distance_from_high,
                "high_24h": state.high_24h,
                "price": state.current_price,
                "change_1m": change_1m
            }
        )
    
    def _calculate_breakout_score(
        self,
        distance_from_high: float,
        change_1m: float
    ) -> float:
        """Calcule le score de breakout (0-1)."""
        score = 0.0
        
        # Plus on est proche du high, meilleur est le score
        if distance_from_high < 0.1:  # Nouveau high
            score += 0.5
        elif distance_from_high < 0.2:
            score += 0.4
        elif distance_from_high < 0.3:
            score += 0.3
        else:
            score += 0.2
        
        # Bonus pour momentum positif fort
        if change_1m > 0.3:
            score += 0.3
        elif change_1m > 0.15:
            score += 0.2
        else:
            score += 0.1
        
        return min(score, 1.0)
    
    def _emit_alert(self, alert: ScannerAlert) -> None:
        """Émet une alerte."""
        # Ajouter à la liste (max 100)
        self._recent_alerts.append(alert)
        if len(self._recent_alerts) > self._max_alerts:
            self._recent_alerts.pop(0)
        
        # Mettre à jour les stats
        self.stats.alerts_generated += 1
        self.stats.alerts_by_type[alert.alert_type.value] += 1
        
        # Log
        logger.debug(f"📢 Alerte: {alert}")
        
        # Callback
        if self._config.alert_callback:
            try:
                self._config.alert_callback(alert)
            except Exception as e:
                logger.error(f"Erreur callback alerte: {e}")
    
    def get_top_opportunities(self, n: int = 10) -> List[PairState]:
        """
        Retourne les N meilleures opportunités actuelles.
        
        Combine les différents signaux pour un score global.
        
        Args:
            n: Nombre d'opportunités à retourner
            
        Returns:
            Liste de PairState triée par score décroissant
        """
        all_states = self._ws_manager.get_all_states()
        
        if not all_states:
            return []
        
        # Calculer un score pour chaque paire
        scored_pairs = []
        
        for symbol, state in all_states.items():
            if not self._passes_basic_filters(state):
                continue
            
            score = self._calculate_opportunity_score(state)
            if score > 0:
                scored_pairs.append((state, score))
        
        # Trier par score décroissant
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner le top N
        return [pair for pair, _ in scored_pairs[:n]]
    
    def _calculate_opportunity_score(self, state: PairState) -> float:
        """
        Calcule un score d'opportunité global (0-1).
        
        Combine momentum, proximité breakout, et autres facteurs.
        """
        score = 0.0
        
        # 1. Momentum récent (40% du score max)
        change_1m = state.change_1m_percent
        if change_1m is not None and change_1m > 0:
            if change_1m > 0.3:
                score += 0.4
            elif change_1m > 0.15:
                score += 0.25
            elif change_1m > 0.05:
                score += 0.1
        
        # 2. Proximité du breakout (30% du score max)
        if state.high_24h > 0 and state.current_price > 0:
            distance_from_high = (state.high_24h - state.current_price) / state.high_24h * 100
            if distance_from_high < 0.5:
                score += 0.3
            elif distance_from_high < 1.0:
                score += 0.2
            elif distance_from_high < 2.0:
                score += 0.1
        
        # 3. Spread faible (15% du score max)
        if state.current_depth:
            spread = state.current_depth.spread_percent
            if spread is not None:
                if spread < 0.03:
                    score += 0.15
                elif spread < 0.05:
                    score += 0.1
                elif spread < 0.1:
                    score += 0.05
        
        # 4. Imbalance positif dans l'orderbook (15% du score max)
        if state.current_depth:
            imbalance = state.current_depth.imbalance
            if imbalance > 0.3:
                score += 0.15
            elif imbalance > 0.1:
                score += 0.1
            elif imbalance > 0:
                score += 0.05
        
        return min(score, 1.0)
    
    def get_recent_alerts(
        self,
        alert_type: Optional[AlertType] = None,
        min_score: float = 0.0,
        limit: int = 20
    ) -> List[ScannerAlert]:
        """
        Récupère les alertes récentes filtrées.
        
        Args:
            alert_type: Filtrer par type d'alerte
            min_score: Score minimum
            limit: Nombre max d'alertes
            
        Returns:
            Liste d'alertes filtrées
        """
        alerts = self._recent_alerts
        
        # Filtrer par type
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        # Filtrer par score
        alerts = [a for a in alerts if a.score >= min_score]
        
        # Trier par timestamp (plus récent en premier)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        
        return alerts[:limit]
    
    def get_stats(self) -> dict:
        """Retourne les statistiques du scanner."""
        return {
            "running": self._running,
            "uptime_seconds": self.stats.uptime_seconds,
            "scans_count": self.stats.scans_count,
            "alerts_generated": self.stats.alerts_generated,
            "alerts_by_type": self.stats.alerts_by_type,
            "last_scan_duration_ms": self.stats.last_scan_duration_ms,
            "recent_alerts_count": len(self._recent_alerts)
        }