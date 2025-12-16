# 📊 CryptoScalper AI - Project Tracker

> **Dernière mise à jour:** 2025-12-16
> **Phase actuelle:** Phase 10.5 - Préparation Paper Trading
> **Stratégie:** Swing Trading (TP 2%, SL 1%)
> **Progression globale:** ~85%

---

## 🎯 État Actuel

```
📍 ON EN EST OÙ ?
├── Phase: 10.5 - Préparation Paper Trading
├── Stratégie: SWING TRADING (changé du scalping)
├── Modèle: swing_final_model.joblib ✅
├── Backtest: +31.27% PnL, 43.5% WR, 1001 trades ✅
├── Audit code: COMPLET ✅
├── Prochaine action: Implémenter paper_trading.py
└── Bloqueurs: Aucun
```

### Paramètres de la stratégie actuelle

| Paramètre | Valeur |
|-----------|--------|
| Take Profit | 2% |
| Stop Loss | 1% |
| Timeout | 120 min (2h) |
| Seuil probabilité | 20% |
| Position size | 20% du capital |
| Win Rate requis | 40% |
| Win Rate obtenu | 43.5% ✅ |

### Notes de la dernière session
- Session du 16 décembre 2025
- **CHANGEMENT MAJEUR**: Passage du scalping au swing trading
  - Scalping non rentable (frais 0.2% mangeaient les gains)
  - Swing trading rentable (+31% backtest)
- **Correction critique des features**:
  - Alignement features.py et dataset.py
  - bb_position: ajout -0.5
  - vwap_distance: rolling 20
  - obv_slope, ad_line: normalisation identique
  - momentum_5: normalisé par prix
- **Validation**: Script validate_features.py créé
- **Audit complet**: Tous les composants testés et fonctionnels
- **Nettoyage**: Suppression datasets/modèles obsolètes
- **Documentation**: USER_GUIDE.md et COMPRENDRE_LE_PROJET.md créés

---

## 📋 Phases du Projet

### Phase 0: Setup Environnement ✅
- [x] Structure projet, git, requirements
- [x] Configuration Pydantic
- [x] Logging avec loguru

### Phase 1: Connexion Binance ✅
- [x] Client Binance async
- [x] Mode hybride (testnet + production data)
- [x] Gestion erreurs API

### Phase 2: Collecte Temps Réel ✅
- [x] WebSocket Manager
- [x] Streams ticker, klines, orderbook
- [x] Reconnexion automatique

### Phase 3: Scanner Multi-Paires ✅
- [x] SymbolsManager
- [x] MultiPairScanner
- [x] Détection momentum/breakout

### Phase 4: Feature Engine ✅
- [x] 42 features techniques
- [x] Momentum, Tendance, Volatilité, Volume, Price Action
- [x] **CORRIGÉ**: Alignement avec dataset.py

### Phase 5: Pipeline ML ✅
- [x] **5.1** Téléchargement données historiques
- [x] **5.2** Préparation dataset (labels SL/TP)
- [x] **5.3** Entraînement XGBoost avec calibration
- [x] **5.4** Évaluation et métriques

### Phase 6: Prédicteur ✅
- [x] MLPredictor
- [x] SignalGenerator
- [x] **CORRIGÉ**: Validation min_probability 0.1-1.0 (pour swing)

### Phase 7: Risk Manager ✅
- [x] Position sizing
- [x] Limites journalières
- [x] Kill switch

### Phase 8: Executor ✅
- [x] Order Manager
- [x] Position Tracker
- [x] Trade Logger

### Phase 9: Backtest ✅
- [x] BacktestEngine
- [x] Rapports (texte, JSON, HTML)
- [x] Métriques avancées (Sharpe, Sortino, Calmar)

### Phase 10: Boucle Principale ✅
- [x] Orchestrateur main.py
- [x] TradingBot avec boucle async
- [x] CLI --mode paper/live
- [x] Gestion SIGINT/SIGTERM

### Phase 10.5: Préparation Paper Trading 🔄 (EN COURS)
- [x] **Changement stratégie**: Scalping → Swing Trading
- [x] **Correction features**: Alignement dataset.py / features.py
- [x] **Validation**: Script validate_features.py
- [x] **Backtest rentable**: +31.27%, 1001 trades, 43.5% WR
- [x] **Audit code**: Tous composants fonctionnels
- [x] **Nettoyage**: Suppression fichiers obsolètes
- [x] **Documentation**: USER_GUIDE.md, COMPRENDRE_LE_PROJET.md
- [ ] **Script paper_trading.py**: À implémenter

### Phase 11: Paper Trading Extended 🔴
- [ ] **11.1** Lancer paper trading 1-2 semaines
- [ ] **11.2** Collecter statistiques réelles
- [ ] **11.3** Valider WR > 40% en conditions réelles

### Phase 12: Live Trading 🔴
- [ ] **12.1** Checklist pré-live (2+ semaines paper stable)
- [ ] **12.2** Démarrer avec petit capital (5-10€)
- [ ] **12.3** Monitoring intensif

---

## 📈 Métriques Actuelles

| Métrique | Objectif | Actuel | Status |
|----------|----------|--------|--------|
| Win Rate (backtest) | > 40% | **43.5%** | ✅ |
| PnL (backtest 90j) | > 0% | **+31.27%** | ✅ |
| Trades (backtest) | - | 1,001 | ✅ |
| Features alignées | 100% | **100%** | ✅ |
| Audit code | Complet | **Complet** | ✅ |

---

## 📁 Fichiers Actuels

### Données
```
data_cache/           # 90 jours (5 cryptos)
├── BTCUSDT_1m.parquet
├── ETHUSDT_1m.parquet
├── SOLUSDT_1m.parquet
├── XRPUSDT_1m.parquet
└── DOGEUSDT_1m.parquet

data_cache_6m/        # 180 jours (backup)
└── [mêmes fichiers]
```

### Datasets
```
datasets/
├── swing_final.parquet           # Dataset complet (648k samples)
├── swing_final_train.parquet     # 70% entraînement
├── swing_final_val.parquet       # 15% validation
└── swing_final_test.parquet      # 15% test
```

### Modèles
```
models/saved/
├── swing_final_model.joblib      # ⭐ Modèle actuel
├── feature_importance.csv
└── metrics_by_threshold.csv
```

### Documentation
```
docs/
├── USER_GUIDE.md                 # Guide utilisateur complet
├── COMPRENDRE_LE_PROJET.md       # Vulgarisation du projet
└── SESSION_SUMMARY.md            # Résumé de la session
```

---

## 🔧 Corrections Importantes (Session 16/12/2025)

### 1. Pourquoi le scalping ne fonctionnait pas

```
Frais Binance: 0.2% aller-retour
TP scalping: 0.5%
Gain net: 0.5% - 0.2% = 0.3%

SL scalping: 0.3%
Perte nette: 0.3% + 0.2% = 0.5%

Win Rate requis: 62.5% (impossible avec ML)
```

### 2. Pourquoi le swing trading fonctionne

```
TP swing: 2%
Gain net: 2% - 0.2% = 1.8%

SL swing: 1%
Perte nette: 1% + 0.2% = 1.2%

Win Rate requis: 40% ✅ (atteignable)
Win Rate obtenu: 43.5% ✅
```

### 3. Features corrigées

| Feature | Problème | Solution |
|---------|----------|----------|
| bb_position | Manquait -0.5 | Ajouté `- 0.5` dans features.py |
| vwap_distance | Méthode différente | Rolling 20 périodes |
| obv_slope | Non normalisé | Divisé par obv_mean |
| ad_line | Non normalisé | Divisé par ad_mean |
| momentum_5 | Valeur absolue | Normalisé par prix × 100 |
| Validation | Seuil min 0.5 | Changé à 0.1 dans signals.py |

---

## 🔄 Historique des Sessions

| Date | Phase | Accomplissements |
|------|-------|------------------|
| 2025-12-15 | 0→10 | Setup complet, tous modules créés, tests passés |
| **2025-12-16** | **10.5** | **PIVOT STRATÉGIQUE**: Scalping → Swing. Correction features. Backtest +31%. Audit complet. Documentation. |

---

## 🚀 Prochaines Étapes

1. **Implémenter paper_trading.py**
   - Script de simulation temps réel
   - Logging des trades simulés
   - Dashboard de monitoring

2. **Lancer le paper trading** (1-2 semaines)
   - Valider WR > 40% en conditions réelles
   - Identifier les bugs éventuels

3. **Live trading** (après validation)
   - Commencer avec 5-10€
   - Monitoring intensif

---

## 📞 Commandes Utiles

```bash
# Télécharger données (90 jours)
python scripts/download_data.py --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT --days 90

# Préparer dataset swing
python scripts/prepare_dataset.py --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT --data-dir data_cache/ --output datasets/swing.parquet --horizon 120 --threshold 0.020 --stop-loss 0.010 --split

# Entraîner modèle
python scripts/train_model.py --train datasets/swing_train.parquet --val datasets/swing_val.parquet --test datasets/swing_test.parquet

# Valider features
python scripts/validate_features.py

# Scan rapide
python -c "
import pandas as pd
import numpy as np
from cryptoscalper.data.features import FeatureEngine, get_feature_names
import joblib

model = joblib.load('models/saved/swing_final_model.joblib')
engine = FeatureEngine()
feature_names = get_feature_names()

for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    df = pd.read_parquet(f'data_cache/{symbol}_1m.parquet').tail(100)
    fs = engine.compute_features(df, symbol=symbol)
    X = np.array([[fs.features[name] for name in feature_names]])
    prob = model.predict_proba(X)[0, 1]
    signal = '🟢' if prob >= 0.20 else '⚪'
    print(f'{signal} {symbol}: {prob:.2%}')
"
```

---

*Dernière mise à jour: 16 décembre 2025*