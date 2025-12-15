# 📊 CryptoScalper AI - Project Tracker

> **Dernière mise à jour:** 2025-12-15
> **Phase actuelle:** Phase 3 - Scanner Multi-Paires
> **Progression globale:** ~30%

---

## 🎯 État Actuel

📍 ON EN EST OÙ ?
├── Phase: 10 - Boucle Principale ✅
├── Tâche en cours: PHASE 10 COMPLÈTE
├── Prochaine action: Phase 11 - Paper Trading Extended
└── Bloqueurs: Aucun

### Notes de la dernière session:
- Session du 15 décembre 2025
- Phase 10 COMPLÉTÉE ✅
  - main.py orchestrateur complet
  - TradingBot avec boucle async
  - Intégration de tous les modules
  - Gestion SIGINT/SIGTERM
  - Mode paper/live via CLI
  - 19/19 tests passés
- **PHASE 10 COMPLÈTE** 🎉
- Prochaine étape: Phase 11 - Paper Trading Extended


## 📋 Phases du Projet

### Phase 0: Setup Environnement ✅
> Préparer l'environnement de développement

- [x] **0.1 Structure projet**
  - [x] Créer l'arborescence des dossiers
  - [x] Initialiser git + .gitignore
  - [x] Créer requirements.txt de base
  - [x] Créer le fichier .env.example

- [x] **0.2 Configuration**
  - [x] Créer `config/settings.py` (dataclasses Pydantic)
  - [x] Créer `config/default_config.yaml`
  - [x] Créer le loader de configuration

- [x] **0.3 Logging**
  - [x] Créer `utils/logger.py` avec loguru
  - [x] Définir les formats de log
  - [x] Test du système de logging

---

### Phase 1: Connexion Binance ✅
> Se connecter à Binance et récupérer des données de base

- [x] **1.1 Client Binance**
  - [x] Créer `data/binance_client.py` (wrapper)
  - [x] Connexion testnet fonctionnelle
  - [x] Gestion des erreurs API
  - [x] Test: récupérer le prix BTC

- [x] **1.2 Données REST (basique)**
  - [x] Récupérer prix spot d'une paire
  - [x] Récupérer orderbook
  - [x] Récupérer klines (chandeliers)
  - [x] Tests unitaires

---

### Phase 2: Collecte Temps Réel ✅
> WebSocket pour données live

- [x] **2.1 WebSocket Manager**
  - [x] Créer `data/websocket_manager.py`
  - [x] Connexion WebSocket unique
  - [x] Gestion reconnexion auto
  - [x] Gestion des erreurs

- [x] **2.2 Streams de données**
  - [x] Stream ticker (prix)
  - [x] Stream klines 1m
  - [x] Stream orderbook
  - [x] Buffer circulaire pour historique court

- [x] **2.3 Data Collector**
  - [x] Créer `data/collector.py`
  - [x] Interface unifiée pour accéder aux données
  - [x] Tests d'intégration

---

### Phase 3: Scanner Multi-Paires ✅ COMPLÉTÉE
> Surveiller 100+ paires simultanément

- [x] **3.1 Sélection des paires**
  - [x] Créer `data/symbols.py`
  - [x] Récupérer toutes les paires USDT
  - [x] Filtrer par volume minimum
  - [x] Exclure stablecoins
  - [x] Rafraîchissement périodique

- [x] **3.2 Scanner Core**
  - [x] Créer `data/scanner.py` (version basique: pair_scanner.py)
  - [x] Dataclass `PairState`
  - [x] Dataclass `ScannerAlert`
  - [x] Historique prix glissant (5 min)

- [x] **3.3 Détection d'opportunités**
  - [x] Détection volume spike
  - [x] Détection momentum
  - [x] Détection breakout
  - [x] Scoring rapide des paires
  - [x] Méthode `get_top_opportunities()`

- [x] **3.4 Tests Scanner**
  - [x] Test latence < 100ms
  - [x] Test CPU < 20%
  - [x] Test 150 paires simultanées


---

### Phase 4: Feature Engine ✅ COMPLÉTÉE

- [x] **4.1 Indicateurs Momentum** (10 features)
  - [x] RSI (14 et 7 périodes)
  - [x] Stochastic %K, %D
  - [x] Williams %R
  - [x] ROC (5 et 10)
  - [x] Momentum, CCI, CMO

- [x] **4.2 Indicateurs Tendance** (8 features)
  - [x] EMA 5/10/20 ratios
  - [x] MACD (line, signal, histogram)
  - [x] ADX
  - [x] Aroon Oscillator

- [x] **4.3 Indicateurs Volatilité** (6 features)
  - [x] Bollinger Bands (width + position)
  - [x] ATR (absolu et %)
  - [x] Écart-type returns
  - [x] Range High-Low

- [x] **4.4 Features Orderbook** (8 features)
  - [x] Spread bid/ask
  - [x] Imbalance
  - [x] Depth bid/ask
  - [x] Pression achat/vente

- [x] **4.5 Features Volume** (5 features)
  - [x] Volume relatif
  - [x] OBV slope
  - [x] Volume delta
  - [x] VWAP distance
  - [x] A/D line

- [x] **4.6 Features Price Action** (5 features)
  - [x] Returns 1m/5m/15m
  - [x] Chandeliers consécutifs
  - [x] Taille relative bougie

- [x] **4.7 Feature Engine**
  - [x] Créer `data/features.py`
  - [x] Classe `FeatureEngine`
  - [x] Méthode `compute_features()`
  - [x] Méthode `compute_features_batch()`
  - [x] Tests unitaires complets

---

### Phase 5: Données Historiques & Training 🔴
> Préparer et entraîner le modèle ML

- [x] **5.1 Téléchargement historique**
  - [x] Créer `data/historical.py`
  - [x] Télécharger 6 mois de données
  - [x] Stocker en CSV/Parquet
  - [x] Script `scripts/download_data.py`

- [x] **5.2 Préparation dataset**
  - [x] Calcul des features sur historique
  - [x] Création des labels (hausse ≥0.2% en 3min)
  - [x] Split temporel train/val/test
  - [x] Vérification équilibre des classes

- [x] **5.3 Entraînement modèle**
  - [x] Créer `models/trainer.py`
  - [x] Pipeline XGBoost
  - [x] Calibration des probabilités
  - [x] Sauvegarde modèle
  - [x] Script `scripts/train_model.py`

- [x] **5.4 Évaluation**
  - [x] Métriques: AUC, précision, recall
  - [x] Courbe ROC
  - [x] Feature importance
  - [x] Validation sur test set

---

### Phase 6: Predictor & Signals ✅ COMPLÉTÉE
> Inférence et génération de signaux

- [x] **6.1 ML Predictor**
  - [x] Créer `models/predictor.py`
  - [x] Chargement modèle
  - [x] Prédiction single + batch
  - [x] Calcul confiance

- [x] **6.2 Signal Generator**
  - [x] Créer `trading/signals.py`
  - [x] Filtrage par seuils
  - [x] Dataclass `TradeSignal`
  - [x] Ranking des opportunités

---

### Phase 7: Risk Management ✅ COMPLÉTÉE
> Gestion du risque stricte

- [x] **7.1 Risk Manager Core**
  - [x] Créer `trading/risk_manager.py`
  - [x] Dataclass `RiskConfig`
  - [x] Position sizing
  - [x] Vérification autorisation trade

- [x] **7.2 Limites**
  - [x] Limite perte par trade
  - [x] Limite perte journalière
  - [x] Limite nombre de trades
  - [x] Reset quotidien

- [x] **7.3 Kill Switch**
  - [x] Tracking du drawdown
  - [x] Activation automatique
  - [x] Fermeture positions d'urgence

---

### Phase 8: Executor 🔴
> Exécution des ordres

- [x] **8.1 Order Manager**
  - [x] Créer `trading/executor.py`
  - [x] Ordre market BUY
  - [x] Ordre OCO (SL+TP)
  - [x] Gestion erreurs ordres

- [x] **8.2 Position Tracker**
  - [x] Dataclass `Position`
  - [x] Dataclass `CompletedTrade`
  - [x] Suivi positions ouvertes
  - [x] Synchronisation avec exchange

- [x] **8.3 Trade Logger**
  - [x] Créer `utils/trade_logger.py`
  - [x] Sauvegarde CSV des trades
  - [x] Calcul statistiques
  - [x] Export pour analyse

---

### Phase 9: Backtest ✅
- [x] **9.1 Backtest Engine**
  - [x] Créer `backtest/engine.py`
  - [x] Simulation des ordres
  - [x] Prise en compte frais
  - [x] Simulation slippage

- [x] **9.2 Rapports**
  - [x] Créer `backtest/reports.py`
  - [x] Métriques: win rate, PnL, Sharpe
  - [x] Graphiques performance
  - [x] Script `scripts/backtest.py`

---

### Phase 10: Boucle Principale 🔴
> Assemblage final

- [x] **10.1 Orchestrateur**
  - [x] Créer `main.py`
  - [x] Boucle principale async
  - [x] Intégration tous les modules
  - [x] Gestion arrêt propre (SIGINT)

- [x] **10.2 Mode Paper Trading**
  - [x] Flag --mode paper/live
  - [x] Simulation sans ordres réels
  - [x] Logging détaillé

---

### Phase 11: Paper Trading Extended 🔴
> Validation sur plusieurs semaines

- [ ] **11.1 Monitoring**
  - [ ] Laisser tourner 2+ semaines
  - [ ] Collecter statistiques
  - [ ] Identifier bugs

- [ ] **11.2 Optimisation**
  - [ ] Ajuster seuils si nécessaire
  - [ ] Analyser trades perdants
  - [ ] Fine-tuning paramètres

---

### Phase 12: Live Trading 🔴
> Passage en réel (avec précaution!)

- [ ] **12.1 Checklist pré-live**
  - [ ] 2+ semaines paper stable
  - [ ] Kill switch testé
  - [ ] Clés API sans withdraw
  - [ ] Capital risque uniquement

- [ ] **12.2 Go Live**
  - [ ] Basculer en mode live
  - [ ] Monitoring intensif
  - [ ] Prêt à couper si problème

---

## 📈 Métriques de Suivi

| Métrique | Objectif | Actuel |
|----------|----------|--------|
| Tests unitaires | > 80% coverage | - |
| Latence scanner | < 100ms | - |
| Win rate (backtest) | > 52% | - |
| Profit factor | > 1.2 | - |
| Uptime paper | > 99% | - |

---

## 🐛 Bugs & Issues Connus

| # | Description | Priorité | Status |
|---|-------------|----------|--------|
| 1 | pandas-ta incompatible → remplacé par ta | - | ✅ Résolu |
| 2 | Testnet peu d'activité → mode hybride ajouté | - | ✅ Résolu |

---

## 📚 Fichiers de Référence

- `PROJET_TRADING_BOT_IA.md` - Spécifications complètes
- `CLEAN_CODE_RULES.md` - Règles de code à respecter
- `CLAUDE_INSTRUCTIONS.md` - Instructions pour Claude

---

## 🔄 Historique des Sessions

| Date | Phase | Accomplissements |
|------|-------|------------------|
| 2025-12-15 | 0→2 | Setup complet (config, logger, exceptions), Client Binance avec mode hybride, WebSocket complet (ticker, klines, orderbook), DataCollector interface unifiée. Tous tests passent. |
| 2025-12-15 | 3 | Créé symbols.py (SymbolsManager avec rafraîchissement auto), multi_pair_scanner.py (détection momentum/breakout, ScannerAlert, scoring), collector.py (interface unifiée). Tests Phase 3 passés. |
| 2025-12-15 | 4 | Feature Engine complet (42 features): Momentum (RSI, Stochastic, Williams, ROC, CCI, CMO), Tendance (EMA, MACD, ADX, Aroon), Volatilité (BB, ATR), Orderbook (spread, imbalance, depth), Volume (OBV, VWAP, A/D), Price Action (returns, chandeliers). Performance ~71ms. |
| 2025-12-15 | 6 | MLPredictor (chargement, prédiction single/batch, confiance), SignalGenerator (filtrage seuils, scoring, lifecycle), TradeSignal (SL/TP auto, statuts). 31 tests passent. |
| 2025-12-15 | 8 | Executor complet: TradeExecutor (ordres market BUY/SELL, OCO SL+TP, paper trading avec simulation slippage/fees), Position (suivi temps réel, calcul PnL), CompletedTrade (conversion depuis Position), SymbolInfo (validation ordres, rounding). TradeLogger: sauvegarde CSV automatique, TradingStatistics (win_rate, profit_factor, best/worst symbol), export JSON et résumé texte. 14/19 tests passent (5 skippés: réseau). |
| 2025-12-15 | 9 | Backtest complet: BacktestEngine (simulation frais 0.1%, slippage 0.05%, SL/TP/timeout), BacktestTrade/Result avec 24+ métriques, Reports (texte, JSON, HTML Chart.js), métriques avancées (Sharpe, Sortino, Calmar), stats mensuelles/horaires. Script CLI backtest.py. 20/20 tests passés. |
| 2025-12-15 | 10 | Orchestrateur complet: main.py avec TradingBot (boucle async, intégration modules), BotConfig (mode, capital, seuils), BotState (stats temps réel), CLI argparse (--mode paper/live), gestion SIGINT/SIGTERM, reset journalier. 19/19 tests passés. |


---

## 📁 Fichiers Créés

### Phase 0
- `cryptoscalper/config/settings.py` - Configuration Pydantic
- `cryptoscalper/config/constants.py` - Constantes du projet
- `cryptoscalper/utils/logger.py` - Logging avec loguru
- `cryptoscalper/utils/exceptions.py` - Exceptions personnalisées
- `scripts/test_setup.py` - Test de configuration

### Phase 1
- `cryptoscalper/data/binance_client.py` - Client Binance async avec mode hybride
- `scripts/test_binance_connection.py` - Test connexion Binance

### Phase 2
- `cryptoscalper/data/websocket_manager.py` - WebSocket avec reconnexion auto
- `cryptoscalper/data/collector.py` - Interface unifiée REST + WebSocket
- `cryptoscalper/data/pair_scanner.py` - Scanner basique (à améliorer)
- `scripts/test_pair_scanner.py` - Test du scanner
- `scripts/test_phase2.py` - Tests d'intégration Phase 2

### Phase 3
- `cryptoscalper/data/symbols.py`
- `cryptoscalper/data/multi_pair_scanner.py`
- `cryptoscalper/data/collector.py`
- `scripts/test_multi_pair_scanner.py`

### Phase 4
- `cryptoscalper/data/features.py` - Feature Engine avec 42 indicateurs
- `scripts/test_features.py` - Tests d'intégration Phase 4

### Phase 5.1
- `cryptoscalper/data/historical.py` - Téléchargement données historiques
- `scripts/download_data.py` - Script CLI téléchargement
- `scripts/test_historical.py` - Tests d'intégration Phase 5.1

### Phase 5.2
- `cryptoscalper/data/dataset.py` - Préparation dataset ML
- `scripts/prepare_dataset.py` - Script CLI préparation
- `scripts/test_dataset.py` - Tests d'intégration Phase 5.2
- `datasets/.gitkeep` - Dossier pour datasets préparés

### Phase 5.3
- `cryptoscalper/models/trainer.py` - Module d'entraînement XGBoost avec calibration
- `cryptoscalper/models/__init__.py` - Init du module models
- `scripts/train_model.py` - Script CLI pour lancer l'entraînement
- `scripts/test_trainer.py` - Tests d'intégration (12 tests ✅)

### Phase 5.4
- `cryptoscalper/models/evaluator.py` - Évaluation et visualisations
- `cryptoscalper/models/__init__.py` - Exports mis à jour
- `scripts/evaluate_model.py` - Script CLI évaluation
- `scripts/test_evaluator.py` - Tests d'intégration (14 tests)
- `reports/.gitkeep` - Dossier rapports

### Phase 6
- `cryptoscalper/models/predictor.py` - MLPredictor et PredictionResult
- `cryptoscalper/trading/signals.py` - SignalGenerator et TradeSignal
- `scripts/test_predictor.py` - Tests Phase 6.1 (14 tests)
- `scripts/test_signals.py` - Tests Phase 6.2 (17 tests)

### Phase 7:
- `cryptoscalper/trading/risk_manager.py` - Module Risk Management complet
- `cryptoscalper/trading/__init__.py` - Module init mis à jour
- `scripts/test_risk_manager.py` - Tests Phase 7 (28 tests)

### Phase 8:
- `cryptoscalper/trading/executor.py` | Order Manager complet (TradeExecutor, Position, CompletedTrade, SymbolInfo, OrderResult) |
- `cryptoscalper/trading/__init__.py` | Exports du module trading (mis à jour) |
- `cryptoscalper/utils/trade_logger.py` | Trade Logger CSV avec statistiques et exports |
- `scripts/test_executor.py` | Tests Phase 8 (19 tests) |

### Phase 9:
- `cryptoscalper/backtest/engine.py` | Moteur de backtest complet (BacktestEngine, BacktestConfig, BacktestTrade, BacktestResult, CloseReason) |
- `cryptoscalper/backtest/reports.py` | Génération rapports (BacktestReport, AdvancedMetrics, ASCII charts, export JSON/HTML) |
- `cryptoscalper/backtest/__init__.py` | Exports du module backtest |
- `scripts/backtest.py` | Script CLI pour lancer des backtests |
- `scripts/test_backtest.py` | Tests Phase 9 (20 tests) |

### Phase 10:
- `cryptoscalper/main.py` | Orchestrateur principal (TradingBot, BotConfig, BotState, BotMode, CLI argparse) |
- `scripts/test_main.py` | Tests Phase 10 (19 tests) |


---

## 🔧 Notes Techniques

- **pandas-ta** remplacé par **ta** (problème de compatibilité Python)
- **Mode hybride** : `BinanceClient(use_production_data=True)` par défaut pour avoir des données live (testnet a peu d'activité)
- **Testnet** : Garder les clés pour les trades, mais données viennent de production