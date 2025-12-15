# 📊 CryptoScalper AI - Project Tracker

> **Dernière mise à jour:** 2025-01-XX
> **Phase actuelle:** Phase 0 - Setup
> **Progression globale:** 0%

---

## 🎯 État Actuel (À LIRE EN PREMIER)

```
📍 ON EN EST OÙ ?
├── Phase: 0 - Setup Environnement
├── Tâche en cours: Configuration initiale du projet
├── Prochaine action: Créer la structure des dossiers
└── Bloqueurs: Aucun
```

### 📝 Notes de la dernière session
- Session initiale - Création du tracker
- Définition des règles clean code
- Setup de l'environnement Claude

---

## 📋 Phases du Projet

### Phase 0: Setup Environnement ⏳
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
  - [x] Créer `utils/exceptions.py`
  - [x] Test du système de logging

---

### Phase 1: Connexion Binance 🔴
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

### Phase 2: Collecte Temps Réel 🔴
> WebSocket pour données live

- [ ] **2.1 WebSocket Manager**
  - [ ] Créer `data/websocket_manager.py`
  - [ ] Connexion WebSocket unique
  - [ ] Gestion reconnexion auto
  - [ ] Gestion des erreurs

- [ ] **2.2 Streams de données**
  - [ ] Stream ticker (prix)
  - [ ] Stream klines 1m
  - [ ] Stream orderbook
  - [ ] Buffer circulaire pour historique court

- [ ] **2.3 Data Collector**
  - [ ] Créer `data/collector.py`
  - [ ] Interface unifiée pour accéder aux données
  - [ ] Tests d'intégration

---

### Phase 3: Scanner Multi-Paires 🔴
> Surveiller 100+ paires simultanément

- [ ] **3.1 Sélection des paires**
  - [ ] Créer `data/symbols.py`
  - [ ] Récupérer toutes les paires USDT
  - [ ] Filtrer par volume minimum
  - [ ] Exclure stablecoins
  - [ ] Rafraîchissement périodique

- [ ] **3.2 Scanner Core**
  - [ ] Créer `data/scanner.py`
  - [ ] Dataclass `PairState`
  - [ ] Dataclass `ScannerAlert`
  - [ ] Historique prix glissant (5 min)

- [ ] **3.3 Détection d'opportunités**
  - [ ] Détection volume spike
  - [ ] Détection momentum
  - [ ] Détection breakout
  - [ ] Scoring rapide des paires
  - [ ] Méthode `get_top_opportunities()`

- [ ] **3.4 Tests Scanner**
  - [ ] Test latence < 100ms
  - [ ] Test CPU < 20%
  - [ ] Test 150 paires simultanées

---

### Phase 4: Feature Engine 🔴
> Calcul des indicateurs techniques

- [ ] **4.1 Indicateurs Momentum** (10 features)
  - [ ] RSI (14 et 7 périodes)
  - [ ] Stochastic %K, %D
  - [ ] Williams %R
  - [ ] ROC (5 et 10)
  - [ ] Momentum, CCI, CMO

- [ ] **4.2 Indicateurs Tendance** (8 features)
  - [ ] EMA 5/10/20 ratios
  - [ ] MACD (line, signal, histogram)
  - [ ] ADX
  - [ ] Aroon Oscillator

- [ ] **4.3 Indicateurs Volatilité** (6 features)
  - [ ] Bollinger Bands (width + position)
  - [ ] ATR (absolu et %)
  - [ ] Écart-type returns
  - [ ] Range High-Low

- [ ] **4.4 Features Orderbook** (8 features)
  - [ ] Spread bid/ask
  - [ ] Imbalance
  - [ ] Depth bid/ask
  - [ ] Pression achat/vente

- [ ] **4.5 Features Volume** (5 features)
  - [ ] Volume relatif
  - [ ] OBV slope
  - [ ] Volume delta
  - [ ] VWAP distance
  - [ ] A/D line

- [ ] **4.6 Features Price Action** (5 features)
  - [ ] Returns 1m/5m/15m
  - [ ] Chandeliers consécutifs
  - [ ] Taille relative bougie

- [ ] **4.7 Feature Engine**
  - [ ] Créer `data/features.py`
  - [ ] Classe `FeatureEngine`
  - [ ] Méthode `compute_features()`
  - [ ] Méthode `compute_features_batch()`
  - [ ] Tests unitaires complets

---

### Phase 5: Données Historiques & Training 🔴
> Préparer et entraîner le modèle ML

- [ ] **5.1 Téléchargement historique**
  - [ ] Créer `data/historical.py`
  - [ ] Télécharger 6 mois de données
  - [ ] Stocker en CSV/Parquet
  - [ ] Script `scripts/download_data.py`

- [ ] **5.2 Préparation dataset**
  - [ ] Calcul des features sur historique
  - [ ] Création des labels (hausse ≥0.2% en 3min)
  - [ ] Split temporel train/val/test
  - [ ] Vérification équilibre des classes

- [ ] **5.3 Entraînement modèle**
  - [ ] Créer `models/trainer.py`
  - [ ] Pipeline XGBoost
  - [ ] Calibration des probabilités
  - [ ] Sauvegarde modèle
  - [ ] Script `scripts/train_model.py`

- [ ] **5.4 Évaluation**
  - [ ] Métriques: AUC, précision, recall
  - [ ] Courbe ROC
  - [ ] Feature importance
  - [ ] Validation sur test set

---

### Phase 6: Predictor & Signals 🔴
> Inférence et génération de signaux

- [ ] **6.1 ML Predictor**
  - [ ] Créer `models/predictor.py`
  - [ ] Chargement modèle
  - [ ] Prédiction single + batch
  - [ ] Calcul confiance

- [ ] **6.2 Signal Generator**
  - [ ] Créer `trading/signals.py`
  - [ ] Filtrage par seuils
  - [ ] Dataclass `TradeSignal`
  - [ ] Ranking des opportunités

---

### Phase 7: Risk Management 🔴
> Gestion du risque stricte

- [ ] **7.1 Risk Manager Core**
  - [ ] Créer `trading/risk_manager.py`
  - [ ] Dataclass `RiskConfig`
  - [ ] Position sizing
  - [ ] Vérification autorisation trade

- [ ] **7.2 Limites**
  - [ ] Limite perte par trade
  - [ ] Limite perte journalière
  - [ ] Limite nombre de trades
  - [ ] Reset quotidien

- [ ] **7.3 Kill Switch**
  - [ ] Tracking du drawdown
  - [ ] Activation automatique
  - [ ] Fermeture positions d'urgence

---

### Phase 8: Executor 🔴
> Exécution des ordres

- [ ] **8.1 Order Manager**
  - [ ] Créer `trading/executor.py`
  - [ ] Ordre market BUY
  - [ ] Ordre OCO (SL+TP)
  - [ ] Gestion erreurs ordres

- [ ] **8.2 Position Tracker**
  - [ ] Dataclass `Position`
  - [ ] Dataclass `CompletedTrade`
  - [ ] Suivi positions ouvertes
  - [ ] Synchronisation avec exchange

- [ ] **8.3 Trade Logger**
  - [ ] Créer `utils/trade_logger.py`
  - [ ] Sauvegarde CSV des trades
  - [ ] Calcul statistiques
  - [ ] Export pour analyse

---

### Phase 9: Backtest 🔴
> Validation de la stratégie

- [ ] **9.1 Backtest Engine**
  - [ ] Créer `backtest/engine.py`
  - [ ] Simulation des ordres
  - [ ] Prise en compte frais
  - [ ] Simulation slippage

- [ ] **9.2 Rapports**
  - [ ] Créer `backtest/reports.py`
  - [ ] Métriques: win rate, PnL, Sharpe
  - [ ] Graphiques performance
  - [ ] Script `scripts/backtest.py`

---

### Phase 10: Boucle Principale 🔴
> Assemblage final

- [ ] **10.1 Orchestrateur**
  - [ ] Créer `main.py`
  - [ ] Boucle principale async
  - [ ] Intégration tous les modules
  - [ ] Gestion arrêt propre (SIGINT)

- [ ] **10.2 Mode Paper Trading**
  - [ ] Flag --mode paper/live
  - [ ] Simulation sans ordres réels
  - [ ] Logging détaillé

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
| - | Aucun pour l'instant | - | - |

---

## 📚 Fichiers de Référence

- `PROJET_TRADING_BOT_IA.md` - Spécifications complètes
- `CLEAN_CODE_RULES.md` - Règles de code à respecter
- `CLAUDE_INSTRUCTIONS.md` - Instructions pour Claude

---

## 🔄 Historique des Sessions

| Date | Phase | Accomplissements |
|------|-------|------------------|
| 2025-12-15 | 0 | Setup initial, création tracker |

---

## 📝 Notes de Session

### Session 1 - 15 décembre 2024
- ✅ Phase 0 complétée
- Remplacement de pandas-ta par ta (problème compatibilité)
- Tous les tests passent
- Prochaine étape : Phase 1 - Connexion Binance

