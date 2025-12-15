# 🤖 Projet CryptoScalper AI - Spécification Complète

## 📋 Table des Matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Spécifications fonctionnelles](#2-spécifications-fonctionnelles)
3. [Architecture technique](#3-architecture-technique)
4. [Modules détaillés](#4-modules-détaillés)
   - 4.1 Data Collector
   - 4.2 Multi-Pair Scanner ⭐ (nouveau)
   - 4.3 Feature Engine
   - 4.4 ML Predictor
   - 4.5 Signal Generator
   - 4.6 Risk Manager
   - 4.7 Executor
   - 4.8 Logger
   - 4.9 Trade Logger
5. [Modèle IA / Machine Learning](#5-modèle-ia--machine-learning)
6. [Gestion du risque](#6-gestion-du-risque)
7. [Structure du code](#7-structure-du-code)
8. [Plan de développement](#8-plan-de-développement)
9. [Configuration et déploiement](#9-configuration-et-déploiement)
10. [Tests et validation](#10-tests-et-validation)
11. [Contraintes et limitations Binance](#11-contraintes-et-limitations-binance)
12. [Glossaire](#12-glossaire)

---

## 1. Vue d'ensemble du projet

### 1.1 Objectif

Développer un bot de trading automatique utilisant l'intelligence artificielle pour effectuer du scalping sur Binance. Le bot doit être capable de :

- Scanner plusieurs paires de trading simultanément
- Identifier des opportunités de profit à très court terme (2-5 minutes)
- Exécuter automatiquement des ordres d'achat et de revente
- Fonctionner de manière 100% autonome 24h/24, 7j/7
- Protéger le capital avec une gestion du risque stricte

### 1.2 Paramètres du projet

| Paramètre | Valeur |
|-----------|--------|
| Capital initial | 20-30 € (USDT) |
| Drawdown maximum acceptable | 25% |
| Disponibilité | 24/7 |
| Style de trading | Scalping (haute fréquence de trades) |
| Durée moyenne d'une position | 2-5 minutes |
| Type de marché | Spot uniquement (pas de futures/levier) |
| Paires | Multi-paires (à définir dynamiquement) |
| Hébergement | Local (machine personnelle) |
| Monitoring | Logs console + fichiers (dashboard ultérieur) |

### 1.3 Philosophie du bot

> **"Ne jamais rester longtemps dans une position pour ne pas bloquer l'argent inutilement"**

Le bot privilégie :
- Des gains petits mais fréquents
- Une rotation rapide du capital
- La minimisation du temps d'exposition au marché
- La sortie rapide en cas de mouvement défavorable

---

## 2. Spécifications fonctionnelles

### 2.1 Fonctionnalités principales

#### F1 - Collecte de données en temps réel
- Connexion WebSocket à Binance pour les flux live
- Récupération des prix, orderbook, et trades pour N paires
- Calcul des indicateurs techniques en temps réel
- Stockage temporaire en mémoire (pas de base de données lourde)

#### F2 - Analyse et prédiction IA
- Analyse simultanée de plusieurs paires
- Scoring de chaque paire : probabilité de hausse dans les 2-5 min
- Classement des opportunités par score de confiance
- Seuil minimum de confiance pour déclencher un trade

#### F3 - Exécution automatique des trades
- Passage d'ordres market pour entrée rapide
- Placement automatique du stop-loss
- Placement automatique du take-profit
- Gestion des ordres partiellement remplis

#### F4 - Gestion des positions
- Suivi en temps réel des positions ouvertes
- Trailing stop optionnel
- Sortie forcée après X minutes (timeout)
- Maximum une position par paire à la fois

#### F5 - Gestion du risque
- Limite de perte journalière
- Limite de perte par trade
- Taille de position calculée dynamiquement
- Kill switch en cas de drawdown excessif

#### F6 - Logging et monitoring
- Logs détaillés de chaque décision
- Historique des trades (CSV/JSON)
- Statistiques de performance en temps réel
- Alertes en cas d'erreur critique

### 2.2 Flux principal du bot

```
BOUCLE PRINCIPALE (toutes les X secondes):

1. COLLECTER les données à jour de toutes les paires surveillées
   
2. CALCULER les features/indicateurs pour chaque paire

3. PRÉDIRE avec le modèle ML :
   - Score de probabilité de hausse (0-100%)
   - Niveau de confiance de la prédiction
   
4. FILTRER les opportunités :
   - Score > seuil minimum (ex: 70%)
   - Confiance > seuil minimum
   - Pas de position déjà ouverte sur cette paire
   - Risk manager autorise le trade
   
5. Si opportunité valide :
   a. CALCULER la taille de position
   b. EXÉCUTER l'ordre d'achat (market)
   c. PLACER stop-loss et take-profit
   d. LOGGER la décision
   
6. GÉRER les positions ouvertes :
   - Vérifier si SL/TP atteint
   - Vérifier le timeout
   - Mettre à jour le trailing stop si activé
   
7. METTRE À JOUR les statistiques

8. ATTENDRE avant la prochaine itération
```

### 2.3 Sélection des paires à trader

#### Critères de sélection automatique des paires :
- Volume 24h minimum (liquidité suffisante)
- Spread bid/ask raisonnable (< 0.1%)
- Volatilité suffisante (sinon pas de profit possible)
- Paires en USDT uniquement (simplicité)
- Exclusion des stablecoins (USDC, BUSD, etc.)

#### Liste initiale suggérée (à affiner) :
```
BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, XRP/USDT,
ADA/USDT, DOGE/USDT, MATIC/USDT, DOT/USDT, AVAX/USDT,
LINK/USDT, UNI/USDT, ATOM/USDT, LTC/USDT, ETC/USDT
```

Le bot peut aussi sélectionner dynamiquement les paires les plus volatiles du moment.

---

## 3. Architecture technique

### 3.1 Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MAIN ORCHESTRATOR                               │
│                         (main.py - boucle principale)                        │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Config    │  │   Logger    │  │  Scheduler  │  │   State Manager     │ │
│  │   Manager   │  │   System    │  │             │  │   (positions, etc)  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────────┐              ┌─────────────────────┐
│   DATA COLLECTOR    │              │      EXECUTOR       │
│                     │              │                     │
│ • WebSocket Manager │              │ • Order Manager     │
│ • Price Feeds       │              │ • Position Tracker  │
│ • Orderbook Feeds   │              │ • SL/TP Manager     │
│ • Kline/Candle Data │              │ • Binance API calls │
└─────────────────────┘              └─────────────────────┘
         │                                     ▲
         ▼                                     │
┌─────────────────────┐              ┌─────────────────────┐
│  FEATURE ENGINE     │              │   RISK MANAGER      │
│                     │              │                     │
│ • Technical Indic.  │              │ • Position Sizing   │
│ • Orderbook Features│              │ • Daily Loss Limit  │
│ • Momentum Signals  │              │ • Drawdown Monitor  │
│ • Volatility Metrics│              │ • Kill Switch       │
└─────────────────────┘              └─────────────────────┘
         │                                     ▲
         ▼                                     │
┌─────────────────────┐              ┌─────────────────────┐
│    ML PREDICTOR     │─────────────▶│  SIGNAL GENERATOR   │
│                     │              │                     │
│ • Load Model        │              │ • Score Threshold   │
│ • Feature Prep      │              │ • Confidence Filter │
│ • Batch Prediction  │              │ • Opportunity Rank  │
│ • Confidence Score  │              │ • Trade Signals     │
└─────────────────────┘              └─────────────────────┘
```

### 3.2 Stack technique

| Composant | Technologie | Justification |
|-----------|-------------|---------------|
| Langage | Python 3.11+ | Écosystème ML, rapidité de dev |
| API Binance | python-binance | Lib officielle, bien maintenue |
| WebSocket | websockets / aiohttp | Flux temps réel performant |
| ML Framework | scikit-learn + XGBoost | Léger, efficace pour tabular data |
| Indicateurs | pandas-ta ou ta-lib | Indicateurs techniques optimisés |
| Data | pandas + numpy | Standard pour la manipulation |
| Config | pydantic + YAML | Validation + lisibilité |
| Logging | loguru | Simple et puissant |
| Async | asyncio | Gestion des flux concurrent |

### 3.3 Flux de données (avec Scanner Multi-Paires)

```
Binance WebSocket (1 connexion, 150 streams)
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│              MULTI-PAIR SCANNER                         │
│  • Reçoit données de 100-150 paires                     │
│  • Filtrage ultra-rapide (< 1ms par ticker)             │
│  • Détection: volume spike, momentum, breakout          │
│  • Output: Top 5-10 paires candidates                   │
└────────────────────────┬────────────────────────────────┘
                         │
        Seulement les paires prometteuses (5-10)
                         │
                         ▼
┌─────────────────┐
│ Feature Engine  │  (calcul ~42 features, plus lourd)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Model       │  (prédiction + confidence)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Signal (BUY?)   │  (score > threshold?)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Risk Check      │  (autorisé par risk manager?)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Execute Order   │  (market buy + SL + TP)
└─────────────────┘
```

---

## 4. Modules détaillés

### 4.1 Module : Data Collector (`data/collector.py`)

#### Responsabilités :
- Maintenir les connexions WebSocket avec Binance
- Recevoir et parser les données en temps réel
- Gérer les reconnexions en cas de déconnexion
- Fournir les données aux autres modules

#### Classes principales :

```python
class BinanceDataCollector:
    """
    Gère la collecte de données en temps réel depuis Binance.
    """
    
    def __init__(self, symbols: List[str]):
        """
        Args:
            symbols: Liste des paires à surveiller (ex: ['BTCUSDT', 'ETHUSDT'])
        """
        pass
    
    async def start(self):
        """Démarre les connexions WebSocket."""
        pass
    
    async def stop(self):
        """Arrête proprement les connexions."""
        pass
    
    def get_current_price(self, symbol: str) -> float:
        """Retourne le dernier prix connu pour une paire."""
        pass
    
    def get_orderbook(self, symbol: str, depth: int = 10) -> dict:
        """Retourne l'orderbook (bids/asks) pour une paire."""
        pass
    
    def get_recent_trades(self, symbol: str, limit: int = 50) -> List[dict]:
        """Retourne les trades récents."""
        pass
    
    def get_klines(self, symbol: str, interval: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Retourne les chandeliers récents."""
        pass
```

#### Données collectées par paire :

| Donnée | Source | Fréquence |
|--------|--------|-----------|
| Prix bid/ask | WebSocket ticker | Temps réel |
| Orderbook (20 niveaux) | WebSocket depth | ~100ms |
| Trades | WebSocket trades | Temps réel |
| Klines 1m | WebSocket kline | Chaque minute |
| Klines 5m | WebSocket kline | Toutes les 5 min |

### 4.2 Module : Multi-Pair Scanner (`data/scanner.py`)

#### Responsabilités :
- Surveiller 100-200 paires simultanément en temps réel
- Détecter rapidement les anomalies et opportunités
- Pré-filtrer les paires intéressantes pour analyse approfondie
- Minimiser la latence de détection

#### Pourquoi un scanner dédié ?

Le bot doit pouvoir identifier la **meilleure opportunité parmi des centaines de paires** à tout instant. Plutôt que d'analyser en profondeur chaque paire (trop lent), on utilise une approche en 2 étapes :

```
┌─────────────────────────────────────────────────────────────────┐
│                    ÉTAPE 1 : SCAN LARGE                         │
│              (100-200 paires, filtrage ultra-rapide)            │
│                                                                 │
│  WebSocket unique → Mise à jour prix → Filtres simples → 5-10  │
│                                         paires candidates       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ÉTAPE 2 : ANALYSE PROFONDE                     │
│              (5-10 paires, ML complet)                          │
│                                                                 │
│  Features complètes → Modèle ML → Score final → Signal trade   │
└─────────────────────────────────────────────────────────────────┘
```

#### Capacités WebSocket Binance :

| Ressource | Limite | Notre usage |
|-----------|--------|-------------|
| Streams par connexion | 1024 max | ~200 streams |
| Connexions WebSocket | 5 par IP | 1-2 connexions |
| Messages reçus | ~500-1000/sec pour 200 paires | Gérable |

#### Types de streams utilisés :

```python
# Pour chaque paire, on peut souscrire à :

# 1. Mini Ticker (RECOMMANDÉ - léger et suffisant)
# ~24 bytes par message, très fréquent
f"{symbol.lower()}@miniTicker"
# Données: prix, volume 24h, high/low 24h

# 2. Ticker complet (plus de données)
f"{symbol.lower()}@ticker"
# Données: tout le mini ticker + plus de stats

# 3. Kline/Candlestick (pour indicateurs)
f"{symbol.lower()}@kline_1m"
# Données: OHLCV de la bougie en cours

# 4. Depth (orderbook) - LOURD
f"{symbol.lower()}@depth20@100ms"
# Données: 20 niveaux bid/ask, updates 100ms
```

#### Stratégie de souscription recommandée :

```python
# Niveau 1 : Scan large (toutes les paires)
scan_streams = [f"{s.lower()}@miniTicker" for s in all_symbols]

# Niveau 2 : Données enrichies (paires candidates seulement)
# Souscription dynamique quand une paire devient intéressante
detailed_streams = [
    f"{s.lower()}@kline_1m",
    f"{s.lower()}@depth20@100ms"
]
```

#### Interface du Scanner :

```python
from dataclasses import dataclass
from typing import List, Dict, Callable
from datetime import datetime
import asyncio

@dataclass
class PairState:
    """État temps réel d'une paire."""
    symbol: str
    price: float
    price_1m_ago: float
    price_5m_ago: float
    volume_24h: float
    volume_1m: float
    high_24h: float
    low_24h: float
    bid_price: float
    ask_price: float
    last_update: datetime
    
    @property
    def spread_pct(self) -> float:
        return (self.ask_price - self.bid_price) / self.bid_price * 100
    
    @property
    def change_1m_pct(self) -> float:
        if self.price_1m_ago == 0:
            return 0
        return (self.price - self.price_1m_ago) / self.price_1m_ago * 100
    
    @property
    def change_5m_pct(self) -> float:
        if self.price_5m_ago == 0:
            return 0
        return (self.price - self.price_5m_ago) / self.price_5m_ago * 100
    
    @property
    def distance_from_high_pct(self) -> float:
        return (self.high_24h - self.price) / self.high_24h * 100
    
    @property
    def distance_from_low_pct(self) -> float:
        return (self.price - self.low_24h) / self.low_24h * 100


@dataclass 
class ScannerAlert:
    """Alerte générée par le scanner."""
    symbol: str
    alert_type: str  # "VOLUME_SPIKE", "MOMENTUM", "BREAKOUT", "REVERSAL"
    score: float  # 0-1
    details: Dict
    timestamp: datetime


class MultiPairScanner:
    """
    Scanner temps réel pour 100-200 paires.
    Détecte les opportunités via des filtres rapides.
    """
    
    def __init__(self, config: ScannerConfig):
        """
        Args:
            config: Configuration du scanner incluant:
                - symbols: Liste des paires à surveiller
                - alert_callback: Fonction appelée lors d'une alerte
                - min_volume_24h: Volume minimum pour considérer une paire
                - max_spread_pct: Spread maximum acceptable
        """
        self.config = config
        self.pairs: Dict[str, PairState] = {}
        self.price_history: Dict[str, List[tuple]] = {}  # {symbol: [(timestamp, price), ...]}
        self.alerts: List[ScannerAlert] = []
        self._running = False
    
    async def start(self):
        """Démarre le scanner et les connexions WebSocket."""
        self._running = True
        await self._connect_websockets()
    
    async def stop(self):
        """Arrête le scanner proprement."""
        self._running = False
        await self._disconnect_websockets()
    
    async def _connect_websockets(self):
        """Établit les connexions WebSocket."""
        pass
    
    async def _handle_ticker(self, msg: dict):
        """
        Traite un message ticker entrant.
        Appelé des centaines de fois par seconde.
        DOIT ÊTRE ULTRA-RAPIDE.
        """
        symbol = msg['s']
        
        # Mise à jour de l'état
        if symbol not in self.pairs:
            self.pairs[symbol] = PairState(symbol=symbol, ...)
        
        state = self.pairs[symbol]
        state.price = float(msg['c'])
        state.volume_24h = float(msg['v'])
        state.high_24h = float(msg['h'])
        state.low_24h = float(msg['l'])
        state.last_update = datetime.now()
        
        # Historique des prix (garde 5 min)
        self._update_price_history(symbol, state.price)
        
        # Vérification des alertes (filtres rapides)
        self._check_alerts(symbol)
    
    def _update_price_history(self, symbol: str, price: float):
        """Maintient un historique glissant des prix."""
        now = datetime.now()
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        history = self.price_history[symbol]
        history.append((now, price))
        
        # Nettoyer les entrées > 5 minutes
        cutoff = now - timedelta(minutes=5)
        self.price_history[symbol] = [(t, p) for t, p in history if t > cutoff]
        
        # Mettre à jour price_1m_ago et price_5m_ago
        state = self.pairs[symbol]
        state.price_1m_ago = self._get_price_at(symbol, minutes_ago=1)
        state.price_5m_ago = self._get_price_at(symbol, minutes_ago=5)
    
    def _get_price_at(self, symbol: str, minutes_ago: int) -> float:
        """Récupère le prix approximatif il y a N minutes."""
        target_time = datetime.now() - timedelta(minutes=minutes_ago)
        history = self.price_history.get(symbol, [])
        
        for timestamp, price in history:
            if timestamp <= target_time:
                return price
        
        return self.pairs.get(symbol, PairState()).price or 0
    
    def _check_alerts(self, symbol: str):
        """
        Vérifie si une paire mérite une alerte.
        DOIT ÊTRE RAPIDE - pas de calculs lourds ici.
        """
        state = self.pairs[symbol]
        
        # Filtre de base : volume et spread acceptables
        if state.volume_24h < self.config.min_volume_24h:
            return
        if state.spread_pct > self.config.max_spread_pct:
            return
        
        # Détection des patterns
        alerts = []
        
        # 1. VOLUME SPIKE
        if self._detect_volume_spike(symbol):
            alerts.append(ScannerAlert(
                symbol=symbol,
                alert_type="VOLUME_SPIKE",
                score=0.3,
                details={"volume_ratio": ...},
                timestamp=datetime.now()
            ))
        
        # 2. MOMENTUM FORT
        if self._detect_momentum(symbol):
            alerts.append(ScannerAlert(
                symbol=symbol,
                alert_type="MOMENTUM",
                score=0.3,
                details={"change_1m": state.change_1m_pct},
                timestamp=datetime.now()
            ))
        
        # 3. BREAKOUT
        if self._detect_breakout(symbol):
            alerts.append(ScannerAlert(
                symbol=symbol,
                alert_type="BREAKOUT",
                score=0.4,
                details={"distance_from_high": state.distance_from_high_pct},
                timestamp=datetime.now()
            ))
        
        # Notifier si alertes
        for alert in alerts:
            self.alerts.append(alert)
            if self.config.alert_callback:
                self.config.alert_callback(alert)
    
    def _detect_volume_spike(self, symbol: str) -> bool:
        """Détecte un spike de volume anormal."""
        state = self.pairs[symbol]
        avg_volume_1m = state.volume_24h / 1440  # Volume moyen par minute
        
        return state.volume_1m > avg_volume_1m * 3  # 3x le volume normal
    
    def _detect_momentum(self, symbol: str) -> bool:
        """Détecte un momentum fort et accélérant."""
        state = self.pairs[symbol]
        
        # Momentum positif qui accélère
        return (
            state.change_1m_pct > 0.15 and  # > 0.15% en 1 min
            state.change_1m_pct > state.change_5m_pct / 5 * 1.5  # Accélération
        )
    
    def _detect_breakout(self, symbol: str) -> bool:
        """Détecte un breakout du range récent."""
        state = self.pairs[symbol]
        
        # Prix proche du high 24h (dans les 0.5%)
        return state.distance_from_high_pct < 0.5
    
    def get_top_opportunities(self, n: int = 10) -> List[PairState]:
        """
        Retourne les N paires les plus intéressantes.
        Utilisé par le moteur principal pour l'analyse ML.
        """
        # Scoring rapide de toutes les paires
        scored = []
        for symbol, state in self.pairs.items():
            score = self._quick_score(state)
            if score > 0:
                scored.append((score, state))
        
        # Trier et retourner le top N
        scored.sort(key=lambda x: x[0], reverse=True)
        return [state for _, state in scored[:n]]
    
    def _quick_score(self, state: PairState) -> float:
        """
        Score rapide d'une paire (0-1).
        Utilisé pour le classement, pas pour la décision finale.
        """
        score = 0.0
        
        # Momentum récent
        if state.change_1m_pct > 0.1:
            score += min(state.change_1m_pct / 0.5, 0.3)  # Max 0.3
        
        # Proximité du breakout
        if state.distance_from_high_pct < 2:
            score += (2 - state.distance_from_high_pct) / 2 * 0.3  # Max 0.3
        
        # Spread faible (liquidité)
        if state.spread_pct < 0.05:
            score += 0.2
        elif state.spread_pct < 0.1:
            score += 0.1
        
        # Volume
        # (nécessite comparaison avec moyenne, simplifié ici)
        score += 0.2  # Placeholder
        
        return min(score, 1.0)


@dataclass
class ScannerConfig:
    """Configuration du scanner multi-paires."""
    symbols: List[str]
    min_volume_24h: float = 1_000_000  # 1M USDT minimum
    max_spread_pct: float = 0.15  # 0.15% max spread
    alert_callback: Callable[[ScannerAlert], None] = None
    price_history_minutes: int = 5
```

#### Sélection dynamique des paires à surveiller :

```python
async def get_tradeable_symbols(client: AsyncClient, min_volume: float = 1_000_000) -> List[str]:
    """
    Récupère dynamiquement les meilleures paires à surveiller.
    Appelé au démarrage et périodiquement (toutes les heures).
    """
    
    # Récupérer toutes les paires USDT
    exchange_info = await client.get_exchange_info()
    usdt_pairs = [
        s['symbol'] for s in exchange_info['symbols']
        if s['symbol'].endswith('USDT')
        and s['status'] == 'TRADING'
        and s['isSpotTradingAllowed']
    ]
    
    # Récupérer les volumes 24h
    tickers = await client.get_ticker()
    volumes = {t['symbol']: float(t['quoteVolume']) for t in tickers}
    
    # Filtrer par volume minimum
    tradeable = [
        symbol for symbol in usdt_pairs
        if volumes.get(symbol, 0) >= min_volume
    ]
    
    # Exclure les stablecoins
    excluded = {'USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'EURUSDT', 'DAIUSDT'}
    tradeable = [s for s in tradeable if s not in excluded]
    
    # Trier par volume décroissant
    tradeable.sort(key=lambda s: volumes.get(s, 0), reverse=True)
    
    # Retourner le top 150 (ou moins)
    return tradeable[:150]
```

#### Intégration avec la boucle principale :

```python
async def main_loop():
    """Boucle principale intégrant le scanner."""
    
    # Initialisation
    symbols = await get_tradeable_symbols(client)
    scanner = MultiPairScanner(ScannerConfig(
        symbols=symbols,
        min_volume_24h=1_000_000,
        alert_callback=on_scanner_alert
    ))
    
    await scanner.start()
    
    while running:
        # Le scanner tourne en background via WebSocket
        # On récupère les meilleures opportunités périodiquement
        
        await asyncio.sleep(2)  # Check toutes les 2 secondes
        
        # Récupérer le top 10 des paires intéressantes
        candidates = scanner.get_top_opportunities(n=10)
        
        if not candidates:
            continue
        
        # Analyse ML approfondie seulement sur les candidates
        for state in candidates:
            # Calcul features complètes
            features = feature_engine.compute_features(state.symbol, collector)
            
            # Prédiction ML
            prediction = predictor.predict(features)
            
            # Génération signal si assez confiant
            if prediction.probability_up > 0.65 and prediction.confidence > 0.6:
                signal = create_trade_signal(state, prediction)
                await execute_if_allowed(signal)
                break  # Un seul trade à la fois


def on_scanner_alert(alert: ScannerAlert):
    """Callback appelé quand le scanner détecte quelque chose."""
    logger.info(f"Scanner alert: {alert.alert_type} on {alert.symbol} (score: {alert.score:.2f})")
```

#### Métriques de performance du scanner :

| Métrique | Cible | Mesure |
|----------|-------|--------|
| Latence détection | < 100ms | Temps entre event et alerte |
| CPU usage | < 20% | Pendant scan actif |
| Mémoire | < 500 MB | Pour 200 paires |
| Faux positifs | < 50% | Alertes sans suite |
| Couverture | > 90% | Opportunités détectées |

### 4.3 Module : Feature Engine (`data/features.py`)

#### Responsabilités :
- Calculer les indicateurs techniques
- Extraire les features de l'orderbook
- Préparer les données pour le modèle ML
- Normaliser les features si nécessaire

#### Features à calculer :

##### A. Indicateurs de momentum (10 features)
```python
- RSI (14 périodes)
- RSI (7 périodes) - plus réactif
- Stochastic %K
- Stochastic %D
- Williams %R
- ROC (Rate of Change) 5 périodes
- ROC 10 périodes
- Momentum 5 périodes
- CCI (Commodity Channel Index)
- CMO (Chande Momentum Oscillator)
```

##### B. Indicateurs de tendance (8 features)
```python
- EMA 5 / Prix actuel (ratio)
- EMA 10 / Prix actuel
- EMA 20 / Prix actuel
- MACD line
- MACD signal
- MACD histogram
- ADX (force de tendance)
- Aroon Oscillator
```

##### C. Indicateurs de volatilité (6 features)
```python
- Bollinger Band Width
- Position dans les Bollinger Bands (0-1)
- ATR (Average True Range)
- ATR % du prix
- Écart-type des returns (20 périodes)
- High-Low range moyen
```

##### D. Features d'orderbook (8 features)
```python
- Bid/Ask spread (%)
- Imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
- Bid depth (somme des 10 premiers niveaux)
- Ask depth (somme des 10 premiers niveaux)
- Bid/Ask depth ratio
- Pression achat (volume bid proche du prix)
- Pression vente (volume ask proche du prix)
- Midprice vs last trade
```

##### E. Features de volume (5 features)
```python
- Volume relatif (vs moyenne 20 périodes)
- OBV (On Balance Volume) slope
- Volume delta (buy vs sell volume)
- VWAP distance (prix vs VWAP)
- Accumulation/Distribution
```

##### F. Features de price action (5 features)
```python
- Returns 1 min
- Returns 5 min
- Returns 15 min
- Nombre de chandeliers verts consécutifs
- Taille relative de la dernière bougie
```

#### Interface :

```python
class FeatureEngine:
    """
    Calcule toutes les features pour le modèle ML.
    """
    
    def __init__(self, config: FeatureConfig):
        pass
    
    def compute_features(self, symbol: str, collector: BinanceDataCollector) -> pd.Series:
        """
        Calcule toutes les features pour une paire donnée.
        
        Returns:
            pd.Series avec ~42 features nommées
        """
        pass
    
    def compute_features_batch(self, symbols: List[str], collector: BinanceDataCollector) -> pd.DataFrame:
        """
        Calcule les features pour plusieurs paires.
        
        Returns:
            pd.DataFrame avec une ligne par paire
        """
        pass
```

### 4.4 Module : ML Predictor (`models/predictor.py`)

#### Responsabilités :
- Charger le modèle entraîné
- Préparer les features pour l'inférence
- Retourner les prédictions avec score de confiance

#### Interface :

```python
class MLPredictor:
    """
    Gère les prédictions du modèle ML.
    """
    
    def __init__(self, model_path: str):
        """Charge le modèle depuis le disque."""
        pass
    
    def predict(self, features: pd.Series) -> PredictionResult:
        """
        Prédit pour une seule paire.
        
        Returns:
            PredictionResult(
                probability_up: float,  # Probabilité de hausse (0-1)
                confidence: float,      # Confiance de la prédiction (0-1)
                predicted_move: float,  # Move prédit en %
            )
        """
        pass
    
    def predict_batch(self, features_df: pd.DataFrame) -> List[PredictionResult]:
        """Prédit pour plusieurs paires."""
        pass


@dataclass
class PredictionResult:
    symbol: str
    probability_up: float      # 0.0 à 1.0
    confidence: float          # 0.0 à 1.0
    predicted_move_pct: float  # ex: 0.3 pour +0.3%
    timestamp: datetime
```

### 4.5 Module : Signal Generator (`trading/signals.py`)

#### Responsabilités :
- Filtrer les prédictions selon les seuils
- Classer les opportunités
- Générer les signaux de trading

#### Interface :

```python
class SignalGenerator:
    """
    Génère des signaux de trading à partir des prédictions ML.
    """
    
    def __init__(self, config: SignalConfig):
        """
        Config inclut:
        - min_probability: float (ex: 0.65)
        - min_confidence: float (ex: 0.6)
        - min_predicted_move: float (ex: 0.002 pour 0.2%)
        """
        pass
    
    def generate_signals(
        self, 
        predictions: List[PredictionResult],
        current_positions: List[str]  # paires déjà en position
    ) -> List[TradeSignal]:
        """
        Filtre et classe les opportunités.
        
        Returns:
            Liste de TradeSignal triée par score décroissant
        """
        pass


@dataclass
class TradeSignal:
    symbol: str
    action: str  # "BUY" uniquement pour le scalping
    score: float  # Score composite (0-1)
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    position_size_pct: float  # Suggestion de taille (% du capital)
    timestamp: datetime
    reasoning: str  # Explication pour le log
```

### 4.6 Module : Risk Manager (`trading/risk_manager.py`)

#### Responsabilités :
- Contrôler la taille des positions
- Suivre les pertes journalières
- Activer le kill switch si nécessaire
- Approuver ou rejeter les trades

#### Paramètres de risque :

```python
@dataclass
class RiskConfig:
    # Capital
    initial_capital: float = 30.0  # USDT
    
    # Position sizing
    max_position_pct: float = 0.20  # Max 20% du capital par trade
    min_position_usdt: float = 5.0  # Minimum requis par Binance
    
    # Limites de perte
    max_loss_per_trade_pct: float = 0.02  # -2% du capital max par trade
    max_daily_loss_pct: float = 0.10  # -10% du capital = stop trading
    max_drawdown_pct: float = 0.25  # -25% du capital initial = kill switch
    
    # Limites de trades
    max_open_positions: int = 1  # Une seule position à la fois (capital limité)
    max_trades_per_hour: int = 20  # Éviter l'overtrading
    max_trades_per_day: int = 100
    
    # Timeout
    max_position_duration_sec: int = 300  # 5 minutes max
    
    # Stop-loss / Take-profit par défaut
    default_stop_loss_pct: float = 0.005  # -0.5%
    default_take_profit_pct: float = 0.003  # +0.3%
```

#### Interface :

```python
class RiskManager:
    """
    Gère tous les aspects de risque du bot.
    """
    
    def __init__(self, config: RiskConfig):
        pass
    
    def can_open_trade(self, signal: TradeSignal) -> Tuple[bool, str]:
        """
        Vérifie si un trade peut être ouvert.
        
        Returns:
            (True, "") si autorisé
            (False, "raison") si refusé
        """
        pass
    
    def calculate_position_size(self, signal: TradeSignal, current_capital: float) -> float:
        """
        Calcule la taille de position optimale.
        
        Returns:
            Taille en USDT
        """
        pass
    
    def register_trade_result(self, trade: CompletedTrade):
        """Enregistre le résultat d'un trade pour le suivi."""
        pass
    
    def is_kill_switch_active(self) -> bool:
        """Vérifie si le kill switch est activé."""
        pass
    
    def get_daily_pnl(self) -> float:
        """Retourne le PnL du jour."""
        pass
    
    def reset_daily_stats(self):
        """Reset des stats journalières (à appeler à minuit)."""
        pass
```

### 4.7 Module : Executor (`trading/executor.py`)

#### Responsabilités :
- Passer les ordres sur Binance
- Gérer les positions ouvertes
- Suivre les stop-loss et take-profit
- Gérer les erreurs d'exécution

#### Interface :

```python
class TradeExecutor:
    """
    Exécute les trades sur Binance.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        pass
    
    async def execute_signal(self, signal: TradeSignal, size_usdt: float) -> Optional[Position]:
        """
        Exécute un signal de trading.
        
        1. Place un ordre market BUY
        2. Place un ordre OCO (SL + TP)
        
        Returns:
            Position object si succès, None si échec
        """
        pass
    
    async def close_position(self, position: Position, reason: str) -> CompletedTrade:
        """
        Ferme une position (market sell).
        """
        pass
    
    async def update_stop_loss(self, position: Position, new_sl_price: float):
        """Met à jour le stop-loss (trailing stop)."""
        pass
    
    def get_open_positions(self) -> List[Position]:
        """Retourne toutes les positions ouvertes."""
        pass
    
    async def sync_with_exchange(self):
        """Synchronise l'état local avec Binance."""
        pass


@dataclass
class Position:
    symbol: str
    side: str  # "LONG" (pour spot, c'est toujours long)
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss_price: float
    take_profit_price: float
    stop_loss_order_id: str
    take_profit_order_id: str


@dataclass
class CompletedTrade:
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl_usdt: float
    pnl_pct: float
    exit_reason: str  # "TP", "SL", "TIMEOUT", "MANUAL"
    fees_usdt: float
```

### 4.8 Module : Logger (`utils/logger.py`)

#### Niveaux de log :

```python
# DEBUG - Détails techniques (pas affiché par défaut)
logger.debug(f"Features calculées pour {symbol}: {features}")

# INFO - Flux normal
logger.info(f"Signal généré: BUY {symbol} @ {price}")
logger.info(f"Trade exécuté: {symbol} qty={qty}")
logger.info(f"Position fermée: {symbol} PnL={pnl_pct:.2f}%")

# WARNING - Attention requise
logger.warning(f"Rate limit proche: {remaining} requêtes restantes")
logger.warning(f"Position timeout: {symbol} fermée après {duration}s")

# ERROR - Erreur récupérable
logger.error(f"Échec ordre {symbol}: {error}")
logger.error(f"WebSocket déconnecté, reconnexion...")

# CRITICAL - Kill switch ou crash
logger.critical(f"KILL SWITCH ACTIVÉ - Drawdown {drawdown:.2f}%")
logger.critical(f"Erreur fatale: {error}")
```

#### Format des logs :

```
2024-01-15 14:32:15.123 | INFO     | Position ouverte: BTCUSDT BUY 0.00025 @ 42150.00 (SL: 41940, TP: 42280)
2024-01-15 14:34:22.456 | INFO     | Position fermée: BTCUSDT TP atteint @ 42280.00 | PnL: +0.31% (+0.09 USDT)
2024-01-15 14:34:22.458 | INFO     | Stats: Trades=15 | Win=10 (66.7%) | PnL jour=+1.23 USDT (+4.1%)
```

### 4.9 Module : Trade Logger (`utils/trade_logger.py`)

#### Responsabilités :
- Sauvegarder l'historique des trades
- Calculer les statistiques de performance
- Exporter les données pour analyse

#### Fichier CSV des trades :

```csv
timestamp,symbol,side,entry_price,exit_price,quantity,pnl_usdt,pnl_pct,duration_sec,exit_reason,fees
2024-01-15T14:32:15,BTCUSDT,BUY,42150.00,42280.00,0.00025,0.09,0.31,127,TP,0.02
2024-01-15T14:45:33,ETHUSDT,BUY,2250.50,2244.25,0.0045,-0.08,-0.28,45,SL,0.02
```

#### Statistiques calculées :

```python
@dataclass
class TradingStats:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl_usdt: float
    total_pnl_pct: float
    avg_win_usdt: float
    avg_loss_usdt: float
    profit_factor: float  # gross profit / gross loss
    max_drawdown_pct: float
    avg_trade_duration_sec: float
    best_trade_pnl: float
    worst_trade_pnl: float
    sharpe_ratio: float  # si assez de données
```

---

## 5. Modèle IA / Machine Learning

### 5.1 Approche choisie

**Classification binaire** : Prédire si le prix va monter de X% dans les Y prochaines minutes.

| Paramètre | Valeur |
|-----------|--------|
| Type de modèle | XGBoost Classifier (ou LightGBM) |
| Target | Prix monte de ≥0.2% dans les 3 prochaines minutes |
| Features | ~42 features techniques et orderbook |
| Output | Probabilité (0-1) + confiance |

### 5.2 Préparation des données d'entraînement

#### Source des données :
- API Binance historical klines
- Minimum 3-6 mois de données
- Intervalle : 1 minute
- Plusieurs paires pour généralisation

#### Création du label :

```python
def create_label(df: pd.DataFrame, horizon_minutes: int = 3, threshold_pct: float = 0.002):
    """
    Label = 1 si le prix monte de ≥0.2% dans les 3 prochaines minutes
    Label = 0 sinon
    """
    future_max = df['high'].rolling(horizon_minutes).max().shift(-horizon_minutes)
    current_price = df['close']
    
    df['label'] = ((future_max - current_price) / current_price >= threshold_pct).astype(int)
    
    return df
```

#### Split temporel (CRUCIAL) :

```python
# JAMAIS de split random pour les séries temporelles !

# Données : 6 mois
# Training : mois 1-4 (66%)
# Validation : mois 5 (17%)
# Test : mois 6 (17%)

train_end = "2024-04-30"
val_end = "2024-05-31"

train = df[df.index < train_end]
val = df[(df.index >= train_end) & (df.index < val_end)]
test = df[df.index >= val_end]
```

### 5.3 Entraînement du modèle

```python
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

# Features sélectionnées
feature_cols = [...]  # ~42 features

# Entraînement
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.5,  # Ajuster si classes déséquilibrées
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc'
)

model.fit(
    train[feature_cols], 
    train['label'],
    eval_set=[(val[feature_cols], val['label'])],
    early_stopping_rounds=20,
    verbose=True
)

# Évaluation
y_pred_proba = model.predict_proba(test[feature_cols])[:, 1]
print(f"AUC: {roc_auc_score(test['label'], y_pred_proba):.4f}")
```

### 5.4 Calibration et seuils

Le modèle doit être **calibré** pour que les probabilités soient fiables.

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
calibrated_model.fit(train[feature_cols], train['label'])
```

#### Choix du seuil :

- Seuil haut (0.7-0.8) : Moins de trades, meilleure précision
- Seuil bas (0.5-0.6) : Plus de trades, plus de faux positifs

Pour le scalping avec petit capital, privilégier un **seuil élevé** (0.65-0.75).

### 5.5 Mesure de confiance

En plus de la probabilité, calculer un score de confiance :

```python
def calculate_confidence(model, features):
    """
    Confiance basée sur :
    1. Écart à 0.5 (probabilité extrême = plus confiant)
    2. Accord des arbres (variance des prédictions)
    """
    proba = model.predict_proba(features.reshape(1, -1))[0, 1]
    
    # Plus la proba est loin de 0.5, plus on est confiant
    confidence_from_proba = abs(proba - 0.5) * 2
    
    # Variance des arbres (si disponible)
    # ...
    
    return confidence_from_proba
```

### 5.6 Réentraînement

Le modèle doit être **réentraîné régulièrement** car les marchés évoluent.

- Fréquence suggérée : toutes les 2-4 semaines
- Garder les anciens modèles pour comparaison
- Monitorer la performance pour détecter la dégradation

---

## 6. Gestion du risque

### 6.1 Règles de position sizing

Avec un capital de 30€, le position sizing est crucial :

```python
def calculate_position_size(capital: float, signal: TradeSignal, config: RiskConfig) -> float:
    """
    Calcule la taille de position.
    """
    # Méthode 1 : Pourcentage fixe du capital
    size_by_pct = capital * config.max_position_pct
    
    # Méthode 2 : Basé sur le risque (stop-loss)
    risk_per_trade = capital * config.max_loss_per_trade_pct
    stop_distance_pct = abs(signal.entry_price - signal.stop_loss_price) / signal.entry_price
    size_by_risk = risk_per_trade / stop_distance_pct
    
    # Prendre le minimum
    size = min(size_by_pct, size_by_risk)
    
    # Vérifier le minimum Binance
    if size < config.min_position_usdt:
        return 0  # Trade non viable
    
    return size
```

### 6.2 Stop-loss et Take-profit

Pour le scalping ultra-court terme :

| Type | Valeur | Justification |
|------|--------|---------------|
| Stop-loss | -0.3% à -0.5% | Limiter les pertes rapidement |
| Take-profit | +0.2% à +0.4% | Prendre les gains vite |
| Ratio R:R | ~1:1 ou légèrement négatif | Compensé par win rate élevé |

**Note** : Avec des frais de 0.1% par trade (0.2% aller-retour), un TP de 0.2% ne laisse que 0% de profit net. Il faut viser au moins 0.3% de TP.

### 6.3 Kill switch

```python
class KillSwitch:
    def __init__(self, initial_capital: float, max_drawdown_pct: float = 0.25):
        self.initial_capital = initial_capital
        self.max_drawdown_pct = max_drawdown_pct
        self.peak_capital = initial_capital
        self.is_active = False
    
    def update(self, current_capital: float):
        # Mettre à jour le peak
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        
        # Calculer le drawdown
        drawdown = (self.peak_capital - current_capital) / self.peak_capital
        
        if drawdown >= self.max_drawdown_pct:
            self.is_active = True
            logger.critical(f"KILL SWITCH: Drawdown {drawdown:.1%} >= {self.max_drawdown_pct:.1%}")
            # Fermer toutes les positions
            # Arrêter le trading
```

### 6.4 Limites journalières

```python
class DailyLimits:
    def __init__(self, config: RiskConfig):
        self.max_daily_loss = config.initial_capital * config.max_daily_loss_pct
        self.max_trades = config.max_trades_per_day
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now().date()
    
    def check_and_reset(self):
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = today
    
    def can_trade(self) -> Tuple[bool, str]:
        self.check_and_reset()
        
        if self.daily_pnl <= -self.max_daily_loss:
            return False, f"Perte journalière max atteinte ({self.daily_pnl:.2f} USDT)"
        
        if self.daily_trades >= self.max_trades:
            return False, f"Nombre max de trades atteint ({self.daily_trades})"
        
        return True, ""
```

---

## 7. Structure du code

### 7.1 Arborescence complète

```
cryptoscalper/
│
├── config/
│   ├── __init__.py
│   ├── settings.py          # Dataclasses de configuration
│   ├── default_config.yaml  # Config par défaut
│   └── .env.example         # Template pour les secrets
│
├── data/
│   ├── __init__.py
│   ├── collector.py         # WebSocket et collecte temps réel
│   ├── scanner.py           # Scanner multi-paires temps réel ⭐
│   ├── features.py          # Calcul des features
│   ├── historical.py        # Téléchargement données historiques
│   └── symbols.py           # Sélection dynamique des paires
│
├── models/
│   ├── __init__.py
│   ├── predictor.py         # Inférence du modèle
│   ├── trainer.py           # Entraînement du modèle
│   ├── features_config.py   # Liste des features utilisées
│   └── saved/               # Dossier des modèles sauvegardés
│       └── .gitkeep
│
├── trading/
│   ├── __init__.py
│   ├── signals.py           # Génération de signaux
│   ├── risk_manager.py      # Gestion du risque
│   ├── executor.py          # Exécution des ordres
│   └── position.py          # Dataclasses Position, Trade
│
├── utils/
│   ├── __init__.py
│   ├── logger.py            # Configuration logging
│   ├── trade_logger.py      # Historique des trades
│   └── helpers.py           # Fonctions utilitaires
│
├── backtest/
│   ├── __init__.py
│   ├── engine.py            # Moteur de backtest
│   ├── simulator.py         # Simulation des ordres
│   └── reports.py           # Génération de rapports
│
├── scripts/
│   ├── download_data.py     # Script pour télécharger l'historique
│   ├── train_model.py       # Script d'entraînement
│   ├── backtest.py          # Script de backtest
│   └── optimize.py          # Optimisation hyperparamètres
│
├── tests/
│   ├── __init__.py
│   ├── test_scanner.py      # Tests du scanner multi-paires
│   ├── test_features.py
│   ├── test_risk_manager.py
│   ├── test_executor.py
│   └── test_signals.py
│
├── logs/                    # Dossier des logs (gitignore)
│   └── .gitkeep
│
├── data_cache/              # Cache des données (gitignore)
│   └── .gitkeep
│
├── main.py                  # Point d'entrée principal
├── requirements.txt         # Dépendances Python
├── .gitignore
├── .env                     # Secrets (gitignore)
└── README.md
```

### 7.2 Fichier requirements.txt

```
# Core
python-binance>=1.0.19
pandas>=2.0.0
numpy>=1.24.0

# Technical Analysis
pandas-ta>=0.3.14b
# ou ta-lib (nécessite installation système)

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
# lightgbm>=4.0.0  # Alternative à XGBoost

# Async
asyncio
aiohttp>=3.8.0
websockets>=11.0

# Config & Utils
pydantic>=2.0.0
python-dotenv>=1.0.0
pyyaml>=6.0.0

# Logging
loguru>=0.7.0

# Backtest (optionnel)
vectorbt>=0.26.0
# ou backtrader>=1.9.78

# Dev
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

### 7.3 Fichier de configuration (default_config.yaml)

```yaml
# Configuration CryptoScalper AI

# === Trading ===
trading:
  mode: "paper"  # "paper" ou "live"
  symbols:
    - "BTCUSDT"
    - "ETHUSDT"
    - "BNBUSDT"
    - "SOLUSDT"
    - "XRPUSDT"
  max_symbols_active: 5

# === Scanner Multi-Paires ===
scanner:
  enabled: true
  dynamic_symbols: true           # Sélection automatique des paires
  max_pairs: 150                  # Nombre max de paires à surveiller
  min_volume_24h: 1000000         # Volume minimum (1M USDT)
  max_spread_pct: 0.15            # Spread max acceptable (0.15%)
  refresh_symbols_hours: 1        # Rafraîchir la liste des paires toutes les X heures
  
  # Seuils de détection rapide
  alerts:
    volume_spike_ratio: 3.0       # Alerte si volume > 3x la moyenne
    momentum_threshold_pct: 0.15  # Alerte si mouvement > 0.15% en 1 min
    breakout_distance_pct: 0.5    # Alerte si prix < 0.5% du high 24h

# === Timing ===
timing:
  scan_interval_seconds: 2  # Fréquence de vérification des opportunités
  position_timeout_seconds: 300  # 5 minutes max
  
# === Risk ===
risk:
  initial_capital: 30.0
  max_position_pct: 0.20
  min_position_usdt: 5.0
  max_loss_per_trade_pct: 0.02
  max_daily_loss_pct: 0.10
  max_drawdown_pct: 0.25
  max_open_positions: 1
  max_trades_per_hour: 20
  max_trades_per_day: 100
  default_stop_loss_pct: 0.004  # 0.4%
  default_take_profit_pct: 0.003  # 0.3%

# === Signal ===
signal:
  min_probability: 0.65
  min_confidence: 0.55
  min_predicted_move_pct: 0.002

# === Model ===
model:
  path: "models/saved/xgb_model_latest.joblib"
  features_version: "v1"

# === Binance ===
binance:
  testnet: true  # IMPORTANT: true pour paper trading
  recv_window: 5000

# === Logging ===
logging:
  level: "INFO"
  file_enabled: true
  file_path: "logs/cryptoscalper.log"
  trade_log_path: "logs/trades.csv"
```

---

## 8. Plan de développement

### Phase 1 : Fondations (Semaine 1-2)

#### Objectifs :
- [ ] Structure du projet créée
- [ ] Configuration et logging fonctionnels
- [ ] Connexion à Binance testnet
- [ ] Collecte de données basique

#### Livrables :
1. `config/settings.py` - Classes de configuration
2. `utils/logger.py` - Système de logging
3. `data/collector.py` - Version basique (REST API)
4. Tests de connexion à Binance

#### Critère de succès :
```python
# Ce code doit fonctionner :
collector = BinanceDataCollector(["BTCUSDT"])
price = collector.get_current_price("BTCUSDT")
print(f"BTC: {price}")  # Affiche le prix actuel
```

### Phase 2 : Features & Données (Semaine 2-3)

#### Objectifs :
- [ ] Collecte temps réel via WebSocket
- [ ] Calcul des 42 features
- [ ] Téléchargement données historiques
- [ ] Préparation des données d'entraînement

#### Livrables :
1. `data/collector.py` - Version WebSocket complète
2. `data/features.py` - Feature engine
3. `data/historical.py` - Téléchargement historique
4. `scripts/download_data.py`

#### Critère de succès :
```python
# Ce code doit fonctionner :
features = feature_engine.compute_features("BTCUSDT", collector)
print(f"Features: {len(features)} colonnes")  # ~42 features
```

### Phase 2b : Scanner Multi-Paires (Semaine 2-3) ⭐

#### Objectifs :
- [ ] Connexion WebSocket multi-streams
- [ ] Surveillance de 100-150 paires simultanément
- [ ] Filtres de détection rapide (volume, momentum, breakout)
- [ ] Intégration avec la boucle principale

#### Livrables :
1. `data/scanner.py` - Scanner complet
2. `data/symbols.py` - Sélection dynamique des paires
3. Tests de performance (latence, CPU)

#### Critère de succès :
```python
# Ce code doit fonctionner :
scanner = MultiPairScanner(config)
await scanner.start()

# Après quelques secondes de données
candidates = scanner.get_top_opportunities(n=10)
print(f"Top opportunités: {[c.symbol for c in candidates]}")

# Les alertes doivent se déclencher sur les mouvements
# Latence détection < 100ms
```

#### Points clés :
- Un seul WebSocket pour toutes les paires (via multiplex)
- Filtres ultra-rapides (pas de calculs lourds)
- Historique glissant des prix (5 min)
- Callback pour notifier les alertes en temps réel

### Phase 3 : Modèle ML (Semaine 3-4)

#### Objectifs :
- [ ] Préparation dataset avec labels
- [ ] Entraînement modèle XGBoost
- [ ] Évaluation et calibration
- [ ] Sauvegarde du modèle

#### Livrables :
1. `models/trainer.py` - Pipeline d'entraînement
2. `models/predictor.py` - Classe de prédiction
3. `scripts/train_model.py`
4. Modèle entraîné dans `models/saved/`

#### Critère de succès :
```python
# Ce code doit fonctionner :
predictor = MLPredictor("models/saved/model.joblib")
result = predictor.predict(features)
print(f"Proba up: {result.probability_up:.2%}")  # ex: 72%
```

### Phase 4 : Backtest (Semaine 4-5)

#### Objectifs :
- [ ] Moteur de backtest fonctionnel
- [ ] Simulation réaliste (frais, slippage)
- [ ] Rapport de performance
- [ ] Validation de la stratégie

#### Livrables :
1. `backtest/engine.py`
2. `backtest/simulator.py`
3. `backtest/reports.py`
4. `scripts/backtest.py`

#### Critère de succès :
- Backtest sur 1 mois de données
- Rapport avec métriques : win rate, PnL, drawdown, Sharpe
- Profit positif après frais (sinon, itérer)

### Phase 5 : Trading Engine (Semaine 5-6)

#### Objectifs :
- [ ] Signal generator fonctionnel
- [ ] Risk manager complet
- [ ] Executor (paper trading)
- [ ] Boucle principale

#### Livrables :
1. `trading/signals.py`
2. `trading/risk_manager.py`
3. `trading/executor.py`
4. `main.py` - Orchestrateur

#### Critère de succès :
```bash
# Le bot tourne en paper trading :
python main.py --mode paper
# Logs montrent les décisions prises
```

### Phase 6 : Paper Trading (Semaine 6-8)

#### Objectifs :
- [ ] Bot tourne 24/7 en paper
- [ ] Monitoring des performances
- [ ] Correction des bugs
- [ ] Optimisation des paramètres

#### Actions :
1. Laisser tourner 2-4 semaines minimum
2. Analyser les trades perdants
3. Ajuster les seuils si nécessaire
4. Vérifier la stabilité (pas de crash)

#### Critère de succès :
- Minimum 500 trades simulés
- Win rate > 50%
- Profit factor > 1.0
- Aucun crash sur 2 semaines

### Phase 7 : Live Trading (Semaine 8+)

#### Objectifs :
- [ ] Passage en mode live
- [ ] Petit capital (20-30€)
- [ ] Monitoring intensif
- [ ] Itérations continues

#### Checklist avant live :
- [ ] 2+ semaines de paper trading stable
- [ ] Kill switch testé et fonctionnel
- [ ] Clés API avec permissions minimales (pas de withdraw)
- [ ] Backup du code et des modèles
- [ ] Plan en cas de perte totale du capital

---

## 9. Configuration et déploiement

### 9.1 Variables d'environnement

Fichier `.env` (NE JAMAIS COMMITER) :

```bash
# Binance API
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Mode (paper/live)
TRADING_MODE=paper

# Testnet (pour paper trading)
BINANCE_TESTNET=true
```

### 9.2 Démarrage du bot

```bash
# Installation
pip install -r requirements.txt

# Télécharger les données historiques
python scripts/download_data.py --symbols BTCUSDT,ETHUSDT --days 180

# Entraîner le modèle
python scripts/train_model.py --config config/default_config.yaml

# Backtest
python scripts/backtest.py --start 2024-05-01 --end 2024-05-31

# Lancer en paper trading
python main.py --mode paper

# Lancer en live (ATTENTION)
python main.py --mode live
```

### 9.3 Fonctionnement 24/7 (local)

Pour que le bot tourne en continu sur ta machine :

**Option 1 : tmux/screen (Linux/Mac)**
```bash
tmux new -s cryptobot
python main.py --mode paper
# Ctrl+B puis D pour détacher
# tmux attach -t cryptobot pour revenir
```

**Option 2 : Processus en arrière-plan**
```bash
nohup python main.py --mode paper > output.log 2>&1 &
```

**Option 3 : Systemd service (Linux)**
```ini
# /etc/systemd/system/cryptobot.service
[Unit]
Description=CryptoScalper AI Bot
After=network.target

[Service]
User=your_user
WorkingDirectory=/path/to/cryptoscalper
ExecStart=/path/to/venv/bin/python main.py --mode paper
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## 10. Tests et validation

### 10.1 Tests unitaires

```python
# tests/test_features.py
def test_rsi_calculation():
    """Le RSI doit être entre 0 et 100."""
    df = pd.DataFrame({'close': [100, 101, 99, 102, 98, 103, 97]})
    rsi = calculate_rsi(df['close'], period=7)
    assert 0 <= rsi.iloc[-1] <= 100

# tests/test_risk_manager.py
def test_position_size_respects_max():
    """La taille ne doit pas dépasser max_position_pct."""
    rm = RiskManager(config)
    size = rm.calculate_position_size(signal, capital=30)
    assert size <= 30 * config.max_position_pct

def test_kill_switch_activates():
    """Le kill switch doit s'activer au drawdown max."""
    rm = RiskManager(config)
    rm.update_capital(30)  # Initial
    rm.update_capital(22)  # -26% drawdown
    assert rm.is_kill_switch_active() == True
```

### 10.2 Tests d'intégration

```python
# tests/test_integration.py
async def test_full_cycle():
    """Test du cycle complet : data -> features -> prediction -> signal."""
    collector = BinanceDataCollector(["BTCUSDT"], testnet=True)
    await collector.start()
    
    features = feature_engine.compute_features("BTCUSDT", collector)
    prediction = predictor.predict(features)
    signals = signal_generator.generate_signals([prediction], [])
    
    await collector.stop()
    
    assert features is not None
    assert 0 <= prediction.probability_up <= 1
```

### 10.3 Validation du modèle

**Métriques à suivre :**

| Métrique | Minimum acceptable | Bon | Excellent |
|----------|-------------------|-----|-----------|
| AUC-ROC | > 0.55 | > 0.60 | > 0.65 |
| Précision @ seuil | > 0.50 | > 0.55 | > 0.60 |
| Win rate backtest | > 48% | > 52% | > 56% |
| Profit factor | > 1.0 | > 1.2 | > 1.5 |

**Attention aux red flags :**
- AUC > 0.80 en test → Probablement de l'overfitting ou data leakage
- Win rate > 70% en backtest → Trop beau pour être vrai
- Résultats très différents train/test → Overfitting

---

## 11. Contraintes et limitations Binance

### 11.1 Rate limits

| Endpoint | Limite | Reset |
|----------|--------|-------|
| Ordres | 10/seconde | Par seconde |
| Requêtes générales | 1200/minute | Par minute |
| WebSocket connections | 5 par IP | - |
| WebSocket messages | 5/seconde/connection | Par seconde |

### 11.2 Minimum order size

La plupart des paires ont un minimum de **10 USDT** par ordre (vérifier par paire).

```python
# Vérifier les filtres d'une paire
info = client.get_symbol_info('BTCUSDT')
for f in info['filters']:
    if f['filterType'] == 'MIN_NOTIONAL':
        print(f"Min order: {f['minNotional']} USDT")
```

### 11.3 Frais

| Type | Frais standard | Avec BNB |
|------|---------------|----------|
| Maker | 0.10% | 0.075% |
| Taker | 0.10% | 0.075% |

**Impact sur le scalping :**
- Un aller-retour coûte 0.2% (0.15% avec BNB)
- Un TP de 0.3% laisse seulement 0.1% de profit net
- Garder du BNB pour payer les frais est recommandé

### 11.4 Testnet vs Production

| Aspect | Testnet | Production |
|--------|---------|------------|
| URL | testnet.binance.vision | api.binance.com |
| Argent | Fictif | Réel |
| Liquidité | Simulée | Réelle |
| Comportement | Peut différer | Référence |

**Important** : Le testnet peut avoir des comportements différents (fills instantanés, pas de slippage). Les résultats en paper trading sont souvent optimistes.

---

## 12. Glossaire

| Terme | Définition |
|-------|------------|
| **Scalping** | Style de trading avec des positions très courtes (secondes à minutes) |
| **Stop-loss (SL)** | Ordre automatique pour limiter les pertes |
| **Take-profit (TP)** | Ordre automatique pour prendre les gains |
| **Drawdown** | Perte depuis le plus haut point du capital |
| **Win rate** | Pourcentage de trades gagnants |
| **Profit factor** | Gains bruts / Pertes brutes |
| **Slippage** | Différence entre prix demandé et prix exécuté |
| **Orderbook** | Carnet d'ordres (bids et asks) |
| **OHLCV** | Open, High, Low, Close, Volume (chandelier) |
| **Feature** | Variable d'entrée pour le modèle ML |
| **Label** | Variable cible à prédire (0 ou 1) |
| **Paper trading** | Trading simulé sans argent réel |
| **Kill switch** | Mécanisme d'arrêt d'urgence |
| **OCO** | One-Cancels-Other (ordre combiné SL+TP) |

---

## 🚀 Prêt à coder !

Ce document contient toutes les spécifications nécessaires pour développer le bot. Commence par la **Phase 1** et avance étape par étape.

**Rappels importants :**
1. Toujours commencer en **paper trading**
2. Ne jamais investir plus que ce que tu peux perdre
3. Le bot n'est pas une garantie de profit
4. Monitorer régulièrement les performances
5. Être prêt à arrêter si ça ne fonctionne pas

Bonne chance ! 🤖📈
