# 📘 Guide Utilisateur - CryptoScalper AI

> **Version** : 2.0 (Swing Trading)  
> **Date** : 16 décembre 2025  
> **Stratégie** : Swing Trading (TP 2%, SL 1%, Horizon 2h)

---

## 📋 Table des matières

1. [Présentation](#1-présentation)
2. [Installation](#2-installation)
3. [Configuration](#3-configuration)
4. [Télécharger les données](#4-télécharger-les-données)
5. [Préparer le dataset](#5-préparer-le-dataset)
6. [Entraîner le modèle](#6-entraîner-le-modèle)
7. [Valider les features](#7-valider-les-features)
8. [Backtest](#8-backtest)
9. [Paper Trading](#9-paper-trading)
10. [Live Trading](#10-live-trading)
11. [Maintenance](#11-maintenance)
12. [Dépannage](#12-dépannage)
13. [Commandes rapides](#13-commandes-rapides)

---

## 1. Présentation

### Qu'est-ce que CryptoScalper AI ?

Un bot de trading automatique qui utilise le Machine Learning pour prédire les hausses de cryptomonnaies.

### Stratégie actuelle : Swing Trading

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| **Take Profit** | +2% | Objectif de gain |
| **Stop Loss** | -1% | Limite de perte |
| **Timeout** | 2 heures | Durée max d'un trade |
| **Seuil proba** | 20% | Minimum pour acheter |
| **Position** | 20% du capital | Taille par trade |

### Résultats attendus

- **Win Rate** : ~43%
- **PnL** : ~+31% sur 90 jours (backtest)
- **Trades** : ~10-15 par semaine

### Cryptos supportées

- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)
- SOLUSDT (Solana)
- XRPUSDT (Ripple)
- DOGEUSDT (Dogecoin)

---

## 2. Installation

### Prérequis

- Python 3.11+
- macOS, Linux ou Windows
- Compte Binance (pour le trading réel)

### Installation

```bash
# 1. Cloner le projet
git clone https://github.com/ton-username/cryptoscalper.git
cd cryptoscalper

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # macOS/Linux
# ou: venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Vérifier l'installation
python -c "import cryptoscalper; print('✅ Installation OK')"
```

### Structure du projet

```
cryptoscalper/
├── config/              # Configuration
├── data/                # Gestion des données
│   ├── features.py      # Calcul des features (temps réel)
│   ├── dataset.py       # Création du dataset (entraînement)
│   └── historical.py    # Téléchargement Binance
├── models/              # Machine Learning
│   ├── predictor.py     # Prédiction
│   └── trainer.py       # Entraînement
├── trading/             # Trading
│   ├── signals.py       # Génération de signaux
│   ├── risk_manager.py  # Gestion du risque
│   └── executor.py      # Exécution des trades
├── scripts/             # Scripts CLI
├── data_cache/          # Données téléchargées
├── datasets/            # Datasets préparés
└── models/saved/        # Modèles entraînés
```

---

## 3. Configuration

### Fichier de configuration principal

Créer/modifier `config/settings.yaml` :

```yaml
# Configuration Swing Trading
trading:
  strategy: "swing"
  take_profit_pct: 0.02      # 2%
  stop_loss_pct: 0.01        # 1%
  timeout_minutes: 120       # 2 heures
  min_probability: 0.20      # 20%
  position_size_pct: 0.20    # 20% du capital

risk:
  initial_capital: 25.0      # Capital en USDT
  max_daily_loss_pct: 0.10   # 10% max perte/jour
  max_drawdown_pct: 0.25     # 25% kill switch
  max_open_positions: 1      # 1 position à la fois

symbols:
  - BTCUSDT
  - ETHUSDT
  - SOLUSDT
  - XRPUSDT
  - DOGEUSDT
```

### Configuration Binance (pour trading réel)

Créer `config/secrets.yaml` (⚠️ NE PAS COMMIT) :

```yaml
binance:
  api_key: "ta_clé_api"
  api_secret: "ton_secret_api"
  testnet: true  # true pour tester, false pour réel
```

---

## 4. Télécharger les données

### Télécharger 90 jours (recommandé)

```bash
python scripts/download_data.py \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT \
    --days 90 \
    --output-dir data_cache \
    --verbose
```

### Télécharger 180 jours (plus de données)

```bash
python scripts/download_data.py \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT \
    --days 180 \
    --output-dir data_cache_6m \
    --verbose
```

### Vérifier les données téléchargées

```bash
python -c "
import pandas as pd
import os

for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']:
    path = f'data_cache/{symbol}_1m.parquet'
    if os.path.exists(path):
        df = pd.read_parquet(path)
        days = (df['open_time'].max() - df['open_time'].min()).days
        print(f'{symbol}: {len(df):,} bougies, {days} jours')
"
```

**Sortie attendue** :
```
BTCUSDT: 129,660 bougies, 90 jours
ETHUSDT: 129,660 bougies, 90 jours
...
```

---

## 5. Préparer le dataset

### Créer le dataset Swing Trading

```bash
python scripts/prepare_dataset.py \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT \
    --data-dir data_cache/ \
    --output datasets/swing_final.parquet \
    --horizon 120 \
    --threshold 0.020 \
    --stop-loss 0.010 \
    --split \
    --verbose
```

### Paramètres expliqués

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `--horizon` | 120 | Timeout en minutes (2h) |
| `--threshold` | 0.020 | Take Profit 2% |
| `--stop-loss` | 0.010 | Stop Loss 1% |
| `--split` | - | Créer train/val/test |

### Vérifier le dataset

```bash
python -c "
from cryptoscalper.data.dataset import PreparedDataset

dataset = PreparedDataset.load('datasets/swing_final.parquet')
print(f'Total: {len(dataset):,} samples')
print(f'Features: {dataset.stats.feature_count}')
print(f'Positifs: {dataset.stats.label_1_count:,} ({dataset.stats.label_ratio:.1%})')
print(f'Négatifs: {dataset.stats.label_0_count:,}')
"
```

**Sortie attendue** :
```
Total: 648,205 samples
Features: 42
Positifs: 34,687 (5.4%)
Négatifs: 613,518
```

---

## 6. Entraîner le modèle

### Lancer l'entraînement

```bash
python scripts/train_model.py \
    --train datasets/swing_final_train.parquet \
    --val datasets/swing_final_val.parquet \
    --test datasets/swing_final_test.parquet \
    --output models/saved/swing_final_model.joblib \
    --verbose
```

### Sauvegarder le modèle avec un nom explicite

```bash
cp models/saved/xgb_model_latest.joblib models/saved/swing_final_model.joblib
```

### Vérifier le modèle

```bash
python -c "
import joblib

model = joblib.load('models/saved/swing_final_model.joblib')
print(f'Type: {type(model).__name__}')
print('✅ Modèle chargé avec succès')
"
```

### Métriques attendues

- **AUC** : ~0.75-0.80 (capacité à distinguer bons/mauvais trades)
- **Feature importance** : ATR et volatilité en tête (~50%)

---

## 7. Valider les features

### Pourquoi c'est important ?

Les features calculées pour l'entraînement (dataset.py) DOIVENT être identiques à celles calculées en temps réel (features.py). Sinon le modèle ne fonctionne pas !

### Lancer la validation

```bash
python scripts/validate_features.py
```

### Sortie attendue

```
✅ Toutes les features sont alignées (<10% de différence)
```

### Si des features sont mal alignées

Les features avec >10% de différence doivent être corrigées dans `cryptoscalper/data/features.py` pour correspondre à `cryptoscalper/data/dataset.py`.

---

## 8. Backtest

### Backtest rapide (dans le terminal)

```bash
python -c "
import pandas as pd
import numpy as np
from cryptoscalper.data.features import FeatureEngine, get_feature_names
import joblib

model = joblib.load('models/saved/swing_final_model.joblib')
engine = FeatureEngine()
feature_names = get_feature_names()

print('📊 BACKTEST')
print('=' * 50)

total_trades, total_wins = 0, 0
total_pnl = 0

for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT']:
    df = pd.read_parquet(f'data_cache/{symbol}_1m.parquet')
    wins, losses = 0, 0
    
    i = 100
    while i < len(df) - 130:
        try:
            fs = engine.compute_features(df.iloc[:i+1].tail(100), symbol=symbol)
            X = np.array([[fs.features[name] for name in feature_names]])
            prob = model.predict_proba(X)[0, 1]
            
            if prob >= 0.20:
                entry = df.iloc[i]['close']
                
                for j in range(1, 121):
                    if df.iloc[i+j]['low'] <= entry * 0.99:
                        losses += 1
                        total_pnl -= 1.2
                        i += j
                        break
                    elif df.iloc[i+j]['high'] >= entry * 1.02:
                        wins += 1
                        total_pnl += 1.8
                        i += j
                        break
                else:
                    exit_p = df.iloc[i+120]['close']
                    pnl = (exit_p/entry - 1) * 100 - 0.2
                    total_pnl += pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    i += 120
            else:
                i += 5
        except:
            i += 5
    
    total_trades += wins + losses
    total_wins += wins
    wr = wins/(wins+losses)*100 if wins+losses > 0 else 0
    print(f'{symbol}: {wins+losses:3d} trades, WR: {wr:5.1f}%')

print('=' * 50)
wr = total_wins/total_trades*100 if total_trades > 0 else 0
print(f'TOTAL: {total_trades} trades, WR: {wr:.1f}%, PnL: {total_pnl:+.2f}%')
"
```

### Résultats attendus

```
📊 BACKTEST
==================================================
BTCUSDT:  15 trades, WR: 46.7%
ETHUSDT: 165 trades, WR: 44.8%
SOLUSDT: 293 trades, WR: 40.6%
XRPUSDT: 223 trades, WR: 45.7%
DOGEUSDT: 305 trades, WR: 43.6%
==================================================
TOTAL: 1001 trades, WR: 43.5%, PnL: +31.27%
```

---

## 9. Paper Trading

### Qu'est-ce que le Paper Trading ?

Simuler le trading en temps réel **sans argent réel** pour valider que tout fonctionne.

### Lancer le Paper Trading

```bash
python scripts/paper_trading.py \
    --model models/saved/swing_final_model.joblib \
    --capital 25 \
    --threshold 0.20 \
    --duration 24h \
    --verbose
```

*(Script à implémenter)*

### Ce qu'il fait

1. Scanne les 5 cryptos toutes les minutes
2. Quand proba ≥ 20% → simule un achat
3. Surveille le trade (SL/TP)
4. Log tous les résultats

### Durée recommandée

- **Minimum** : 1 semaine
- **Recommandé** : 2 semaines
- **Objectif** : Valider WR > 40% en conditions réelles

---

## 10. Live Trading

### ⚠️ Avertissement

Le trading de cryptomonnaies comporte des risques. Ne tradez qu'avec de l'argent que vous pouvez vous permettre de perdre.

### Prérequis

1. ✅ Paper trading validé pendant 1-2 semaines
2. ✅ Win Rate > 40%
3. ✅ Compte Binance configuré
4. ✅ API keys avec permissions de trading

### Configuration API Binance

1. Aller sur [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. Créer une nouvelle API key
3. Activer "Enable Spot Trading"
4. Ajouter l'IP de ton serveur (optionnel mais recommandé)

### Lancer le Live Trading

```bash
python scripts/live_trading.py \
    --model models/saved/swing_final_model.joblib \
    --capital 25 \
    --threshold 0.20 \
    --verbose
```

*(Script à implémenter)*

### Bonnes pratiques

1. **Commencer petit** : 5-10€ les premières semaines
2. **Monitorer** : Vérifier les résultats chaque jour
3. **Arrêter si problème** : Kill switch automatique si -25%

---

## 11. Maintenance

### Quand ré-entraîner le modèle ?

| Situation | Action |
|-----------|--------|
| Win Rate < 40% pendant 1 semaine | Ré-entraîner |
| PnL négatif pendant 2 semaines | Ré-entraîner |
| Changement majeur du marché | Ré-entraîner |
| Tous les 1-3 mois | Ré-entraîner (préventif) |

### Procédure de ré-entraînement

```bash
# 1. Télécharger les données récentes
python scripts/download_data.py \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT \
    --days 90 \
    --output-dir data_cache \
    --verbose

# 2. Recréer le dataset
python scripts/prepare_dataset.py \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT \
    --data-dir data_cache/ \
    --output datasets/swing_$(date +%Y%m%d).parquet \
    --horizon 120 \
    --threshold 0.020 \
    --stop-loss 0.010 \
    --split \
    --verbose

# 3. Ré-entraîner
python scripts/train_model.py \
    --train datasets/swing_$(date +%Y%m%d)_train.parquet \
    --val datasets/swing_$(date +%Y%m%d)_val.parquet \
    --test datasets/swing_$(date +%Y%m%d)_test.parquet \
    --verbose

# 4. Sauvegarder
cp models/saved/xgb_model_latest.joblib models/saved/swing_$(date +%Y%m%d)_model.joblib

# 5. Valider
python scripts/validate_features.py
```

### Sauvegarder les anciens modèles

```bash
# Garder une copie datée
mv models/saved/swing_final_model.joblib models/saved/swing_final_model_backup_$(date +%Y%m%d).joblib
```

---

## 12. Dépannage

### Erreur : "Module not found"

```bash
# Vérifier que l'environnement virtuel est activé
source venv/bin/activate

# Réinstaller les dépendances
pip install -r requirements.txt
```

### Erreur : "Pas assez de données"

```bash
# Vérifier les données
ls -la data_cache/

# Re-télécharger si nécessaire
python scripts/download_data.py --symbols BTCUSDT --days 90
```

### Probabilités toujours à 0%

Cela signifie que les features ne sont pas alignées. Lancer :

```bash
python scripts/validate_features.py
```

### Le bot ne trade pas

Normal si le marché est en surachat ! Vérifier les conditions :

```bash
python -c "
from cryptoscalper.data.features import FeatureEngine
import pandas as pd

engine = FeatureEngine()
df = pd.read_parquet('data_cache/BTCUSDT_1m.parquet').tail(100)
fs = engine.compute_features(df, symbol='BTCUSDT')

print(f'RSI: {fs.features[\"rsi_14\"]:.1f} (cible: <35)')
print(f'Stoch: {fs.features[\"stoch_k\"]:.1f} (cible: <25)')
"
```

Le modèle attend des conditions de **survente** (RSI < 35) pour acheter.

### Performances dégradées

1. Vérifier le Win Rate sur les 50 derniers trades
2. Si WR < 40% → ré-entraîner le modèle (voir section 11)

---

## 13. Commandes rapides

### Cheatsheet

```bash
# === DONNÉES ===
# Télécharger 90 jours
python scripts/download_data.py --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT --days 90 --output-dir data_cache

# === DATASET ===
# Créer dataset swing
python scripts/prepare_dataset.py --symbols BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT --data-dir data_cache/ --output datasets/swing.parquet --horizon 120 --threshold 0.020 --stop-loss 0.010 --split

# === MODÈLE ===
# Entraîner
python scripts/train_model.py --train datasets/swing_train.parquet --val datasets/swing_val.parquet --test datasets/swing_test.parquet

# Sauvegarder
cp models/saved/xgb_model_latest.joblib models/saved/swing_model.joblib

# === VALIDATION ===
# Valider features
python scripts/validate_features.py

# === TRADING ===
# Paper trading (à implémenter)
python scripts/paper_trading.py --model models/saved/swing_model.joblib --capital 25

# Live trading (à implémenter)
python scripts/live_trading.py --model models/saved/swing_model.joblib --capital 25
```

### Vérification rapide du système

```bash
python -c "
import pandas as pd
import numpy as np
from cryptoscalper.data.features import FeatureEngine, get_feature_names
import joblib

model = joblib.load('models/saved/swing_final_model.joblib')
engine = FeatureEngine()
feature_names = get_feature_names()

print('🔍 SCAN RAPIDE')
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

## 📞 Support

### Logs

Les logs sont dans `logs/cryptoscalper.log`

```bash
tail -f logs/cryptoscalper.log
```

### Debug

Ajouter `--verbose` à n'importe quelle commande pour plus de détails.

---

*Guide mis à jour le 16 décembre 2025*