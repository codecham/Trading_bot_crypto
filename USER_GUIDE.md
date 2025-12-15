# 🤖 CryptoScalper AI - Guide d'Utilisation Complet

> **Version:** 1.0  
> **Dernière mise à jour:** 15 décembre 2025  
> **Auteur:** CryptoScalper Team

---

## 📑 Table des Matières

1. [Introduction](#1-introduction)
2. [Prérequis](#2-prérequis)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Préparation des Données](#5-préparation-des-données)
6. [Entraînement du Modèle ML](#6-entraînement-du-modèle-ml)
7. [Lancement du Bot](#7-lancement-du-bot)
8. [Monitoring & Logs](#8-monitoring--logs)
9. [Backtesting](#9-backtesting)
10. [Dépannage](#10-dépannage)
11. [FAQ](#11-faq)
12. [Glossaire](#12-glossaire)

---

## 1. Introduction

### 🎯 Qu'est-ce que CryptoScalper AI ?

CryptoScalper AI est un bot de trading automatique qui :
- **Scanne** 100-150 paires crypto en temps réel
- **Prédit** les hausses à court terme (2-5 minutes) via Machine Learning
- **Exécute** automatiquement des trades avec une gestion du risque stricte
- **Apprend** de ses erreurs grâce à un modèle XGBoost

### 💰 Pour qui ?

- Capital recommandé : **20-50€** (petit capital, rotation rapide)
- Niveau technique : **Intermédiaire** (Python, ligne de commande)
- Temps requis : **Setup initial ~2h**, puis automatique

### ⚠️ Avertissement

> **TRADING = RISQUE**  
> Ce bot peut vous faire perdre de l'argent. N'investissez que ce que vous pouvez vous permettre de perdre. Commencez TOUJOURS en mode paper trading.

---

## 2. Prérequis

### 💻 Système

| Requis | Minimum | Recommandé |
|--------|---------|------------|
| OS | macOS / Linux / Windows | macOS / Linux |
| Python | 3.10+ | 3.11 |
| RAM | 4 GB | 8 GB |
| Stockage | 2 GB | 5 GB |
| Internet | Stable | Fibre |

### 🔑 Comptes nécessaires

1. **Compte Binance** (ou Binance Testnet pour commencer)
   - Créer un compte : https://www.binance.com
   - Testnet (gratuit) : https://testnet.binance.vision

2. **Clés API Binance**
   - Aller dans : Profil → API Management
   - Créer une nouvelle clé API
   - **IMPORTANT** : Ne jamais activer "Enable Withdrawals"

---

## 3. Installation

### Étape 1 : Cloner le projet

```bash
# Cloner le repository
git clone https://github.com/votre-repo/cryptoscalper-ai.git
cd cryptoscalper-ai
```

### Étape 2 : Créer l'environnement virtuel

```bash
# Créer l'environnement
python -m venv venv

# Activer l'environnement
# Sur macOS/Linux :
source venv/bin/activate

# Sur Windows :
venv\Scripts\activate
```

### Étape 3 : Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Étape 4 : Vérifier l'installation

```bash
# Tester que tout fonctionne
python scripts/test_setup.py
```

Vous devriez voir :
```
✅ Python 3.11.x
✅ Toutes les dépendances installées
✅ Configuration OK
```

---

## 4. Configuration

### 4.1 Créer le fichier .env

```bash
# Copier le template
cp .env.example .env

# Éditer avec votre éditeur préféré
nano .env  # ou code .env, vim .env, etc.
```

### 4.2 Contenu du fichier .env

```env
# ============================================
# BINANCE API KEYS
# ============================================

# Pour le TESTNET (recommandé pour commencer)
BINANCE_TESTNET_API_KEY=votre_cle_testnet_ici
BINANCE_TESTNET_API_SECRET=votre_secret_testnet_ici

# Pour le LIVE (trading réel)
BINANCE_API_KEY=votre_cle_live_ici
BINANCE_API_SECRET=votre_secret_live_ici

# ============================================
# MODE
# ============================================

# true = utiliser le testnet, false = production
BINANCE_TESTNET=true

# ============================================
# LOGGING
# ============================================

LOG_LEVEL=INFO
LOG_FILE=logs/cryptoscalper.log
```

### 4.3 Obtenir les clés API Testnet (Gratuit)

1. Aller sur https://testnet.binance.vision
2. Se connecter avec GitHub
3. Cliquer sur "Generate HMAC_SHA256 Key"
4. Copier l'API Key et le Secret dans `.env`

### 4.4 Obtenir les clés API Live (Trading réel)

1. Se connecter à https://www.binance.com
2. Aller dans **Profil** → **API Management**
3. Cliquer sur **Create API**
4. Choisir **System generated**
5. Nommer la clé (ex: "CryptoScalper Bot")
6. **Permissions à activer** :
   - ✅ Enable Reading
   - ✅ Enable Spot & Margin Trading
   - ❌ Enable Withdrawals (JAMAIS !)
7. Optionnel mais recommandé : Restreindre à votre IP
8. Copier l'API Key et le Secret dans `.env`

### 4.5 Configuration avancée (optionnel)

Le fichier `config/default_config.yaml` contient tous les paramètres :

```yaml
# Capital et risque
risk:
  initial_capital: 30.0        # Capital en USDT
  max_position_pct: 0.20       # 20% max par trade
  max_daily_loss_pct: 0.10     # Stop si -10% dans la journée
  max_drawdown_pct: 0.25       # Kill switch si -25% du capital initial
  default_stop_loss_pct: 0.004 # Stop-loss à -0.4%
  default_take_profit_pct: 0.003 # Take-profit à +0.3%

# Modèle ML
signal:
  min_probability: 0.65        # Proba minimum pour trader
  min_confidence: 0.55         # Confiance minimum

# Scanner
scanner:
  max_pairs: 150               # Nombre de paires à surveiller
  min_volume_24h: 1000000      # Volume min 1M USDT
```

---

## 5. Préparation des Données

> ⏱️ **Temps estimé** : 10-30 minutes selon la quantité de données

### 5.1 Télécharger les données historiques

```bash
# Télécharger 60 jours de données pour les top paires
# ⚠️ Les symboles sont séparés par des VIRGULES (pas d'espaces)
python scripts/download_data.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT --days 60

# Pour plus de paires (recommandé)
python scripts/download_data.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,MATICUSDT --days 90

# Ou utiliser une liste prédéfinie (plus simple)
python scripts/download_data.py --preset default --days 60

# Ou télécharger le top 20 des paires par volume
python scripts/download_data.py --top 20 --days 60
```

**Options disponibles :**

| Option | Description | Défaut |
|--------|-------------|--------|
| `--symbols` | Liste séparée par virgules (ex: BTCUSDT,ETHUSDT) | - |
| `--preset` | Liste prédéfinie: `minimal`, `default`, `all` | - |
| `--top` | Télécharger les N paires avec le plus de volume | - |
| `--days` | Nombre de jours d'historique | 180 |
| `--interval` | Intervalle des bougies | 1m |
| `--output-dir` | Dossier de sortie | data_cache |
| `--format` | Format de sauvegarde: `parquet` ou `csv` | parquet |

### 5.2 Vérifier les données téléchargées

```bash
# Lister les fichiers téléchargés
ls -lh data_cache/

# Exemple de sortie :
# BTCUSDT_1m.parquet   (150 MB)
# ETHUSDT_1m.parquet   (145 MB)
# ...
```

### 5.3 Préparer le dataset d'entraînement

```bash
# Calculer les features et créer les labels
# ⚠️ Les symboles sont séparés par des VIRGULES
python scripts/prepare_dataset.py \
    --symbols BTCUSDT,ETHUSDT,BNBUSDT \
    --output datasets/dataset.parquet \
    --horizon 3 \
    --threshold 0.002

# Avec les splits (train/val/test)
python scripts/prepare_dataset.py \
    --symbols BTCUSDT,ETHUSDT \
    --output datasets/dataset.parquet \
    --split
```

**Options importantes :**

| Option | Description | Défaut |
|--------|-------------|--------|
| `--symbols` | Liste séparée par virgules | (requis) |
| `--horizon` | Horizon de prédiction en minutes | 3 |
| `--threshold` | Seuil de hausse pour label=1 (0.002 = 0.2%) | 0.002 |
| `--data-dir` | Dossier des données sources | data_cache |
| `--output` | Fichier de sortie | datasets/prepared_dataset.parquet |
| `--split` | Sauvegarder aussi les splits train/val/test | false |

**Sortie attendue :**
```
📊 Dataset préparé:
   - Échantillons: 1,234,567
   - Features: 42
   - Labels positifs: 45.2%
   - Labels négatifs: 54.8%
   - Split: 70% train / 15% val / 15% test
```

---

## 6. Entraînement du Modèle ML

> ⏱️ **Temps estimé** : 5-30 minutes selon la taille du dataset

### 6.1 Entraîner le modèle

```bash
# Entraînement standard
python scripts/train_model.py \
    --dataset datasets/dataset.parquet \
    --output models/saved/

# Entraînement avec plus d'itérations (meilleur mais plus long)
python scripts/train_model.py \
    --dataset datasets/dataset.parquet \
    --output models/saved/ \
    --n-estimators 500 \
    --early-stopping 50
```

**Options d'entraînement :**

| Option | Description | Défaut |
|--------|-------------|--------|
| `--n-estimators` | Nombre d'arbres | 200 |
| `--max-depth` | Profondeur max des arbres | 6 |
| `--learning-rate` | Taux d'apprentissage | 0.1 |
| `--early-stopping` | Arrêt si pas d'amélioration | 30 |
| `--calibrate` | Calibrer les probabilités | true |

### 6.2 Évaluer le modèle

```bash
python scripts/evaluate_model.py \
    --model models/saved/xgb_model_latest.joblib \
    --output reports/
```

**Métriques à surveiller :**

| Métrique | Objectif | Description |
|----------|----------|-------------|
| AUC-ROC | > 0.55 | Capacité à distinguer hausse/baisse |
| Précision | > 50% | % de prédictions correctes |
| Recall | > 40% | % de hausses détectées |
| Profit Factor | > 1.0 | Gains / Pertes (backtest) |

### 6.3 Interpréter les résultats

```
📊 Évaluation du modèle:
   - AUC-ROC: 0.58 ✅ (> 0.55)
   - Précision: 54.2% ✅
   - Recall: 47.8%
   - F1-Score: 0.51

📈 Top 10 Features:
   1. rsi_14 (12.3%)
   2. macd_histogram (9.8%)
   3. volume_relative (8.5%)
   ...
```

**Si les métriques sont mauvaises :**
- Télécharger plus de données historiques
- Ajuster `--target-gain` (essayer 0.15% ou 0.25%)
- Augmenter `--n-estimators`

---

## 7. Lancement du Bot

### 7.1 Mode Paper Trading (Simulation)

> ⚠️ **Commencez TOUJOURS par le paper trading !**

```bash
# Lancement basique
python -m cryptoscalper.main --mode paper

# Avec logs détaillés
python -m cryptoscalper.main --mode paper --log-level DEBUG

# Avec capital personnalisé
python -m cryptoscalper.main --mode paper --capital 50
```

### 7.2 Toutes les options CLI

```bash
python -m cryptoscalper.main --help
```

| Option | Description | Défaut |
|--------|-------------|--------|
| `--mode` | `paper` ou `live` | paper |
| `--capital` | Capital initial (USDT) | 30.0 |
| `--model` | Chemin du modèle ML | models/saved/xgb_model_latest.joblib |
| `--interval` | Intervalle de scan (sec) | 2.0 |
| `--log-level` | DEBUG, INFO, WARNING, ERROR | INFO |

### 7.3 Mode Live (Trading Réel)

> ⚠️ **ATTENTION : ARGENT RÉEL !**

**Checklist avant de passer en live :**

- [ ] 2+ semaines de paper trading stable
- [ ] Win rate > 50% en paper
- [ ] Kill switch testé et fonctionnel
- [ ] Clés API SANS permission de retrait
- [ ] Capital que vous pouvez perdre

```bash
# Lancement en live (confirmation requise)
python -m cryptoscalper.main --mode live --capital 30
```

Vous devrez taper `CONFIRM` pour confirmer.

### 7.4 Lancer en arrière-plan (Linux/macOS)

```bash
# Avec nohup
nohup python -m cryptoscalper.main --mode paper > bot.log 2>&1 &

# Vérifier que ça tourne
ps aux | grep cryptoscalper

# Voir les logs en temps réel
tail -f bot.log

# Arrêter le bot
pkill -f "cryptoscalper.main"
```

### 7.5 Lancer avec screen (recommandé)

```bash
# Créer une session screen
screen -S cryptobot

# Lancer le bot
python -m cryptoscalper.main --mode paper

# Détacher la session : Ctrl+A puis D

# Rattacher la session plus tard
screen -r cryptobot

# Arrêter proprement : Ctrl+C dans la session
```

---

## 8. Monitoring & Logs

### 8.1 Structure des logs

```
logs/
├── cryptoscalper.log      # Log principal
├── trades.csv             # Historique des trades
└── trades.log             # Détails des trades
```

### 8.2 Lire les logs en temps réel

```bash
# Log principal
tail -f logs/cryptoscalper.log

# Filtrer par type
tail -f logs/cryptoscalper.log | grep "SIGNAL"
tail -f logs/cryptoscalper.log | grep "TRADE"
tail -f logs/cryptoscalper.log | grep "ERROR"
```

### 8.3 Comprendre les logs

```
# Signal détecté
🟢 SIGNAL | BTCUSDT | BUY @ 42150.00 | Confiance: 72.5%

# Trade exécuté
✅ TRADE | BTCUSDT | BUY | Qty: 0.00071 @ 42150.00 | Order: 12345

# Résultat d'un trade
🎉 RESULT | BTCUSDT | PnL: +0.0850 USDT (+0.28%) | Durée: 145s

# Statut périodique
🤖 STATUS | RUNNING | Capital: 30.85 USDT | Positions: 0 | PnL jour: +0.85 USDT
```

### 8.4 Analyser les performances

```bash
# Statistiques des trades
python scripts/analyze_trades.py --input logs/trades.csv

# Exemple de sortie :
📊 Statistiques de trading:
   Période: 2025-12-01 → 2025-12-15
   
   Trades: 156
   ├── Gagnants: 89 (57.1%)
   ├── Perdants: 62 (39.7%)
   └── Breakeven: 5 (3.2%)
   
   PnL Total: +4.25 USDT (+14.2%)
   Profit Factor: 1.42
   Meilleur trade: +0.45 USDT (SOLUSDT)
   Pire trade: -0.32 USDT (DOGEUSDT)
```

---

## 9. Backtesting

### 9.1 Lancer un backtest

```bash
# Backtest sur données historiques
python scripts/backtest.py \
    --data data/historical/BTCUSDT_1m.parquet \
    --model models/saved/xgb_model_latest.joblib \
    --capital 30 \
    --output reports/

# Backtest sur plusieurs paires
python scripts/backtest.py \
    --data data/historical/ \
    --model models/saved/xgb_model_latest.joblib \
    --capital 30
```

### 9.2 Options de backtest

| Option | Description | Défaut |
|--------|-------------|--------|
| `--capital` | Capital initial | 30.0 |
| `--fee` | Frais par trade (%) | 0.1% |
| `--slippage` | Slippage simulé (%) | 0.05% |
| `--stop-loss` | Stop-loss (%) | 0.4% |
| `--take-profit` | Take-profit (%) | 0.3% |

### 9.3 Rapport de backtest

```
═══════════════════════════════════════════════════
📊 RAPPORT DE BACKTEST
═══════════════════════════════════════════════════
Période: 2025-10-01 → 2025-12-01 (61 jours)
Capital initial: 30.00 USDT
Capital final: 38.45 USDT

📈 Performance:
   PnL: +8.45 USDT (+28.2%)
   Trades: 423
   Win Rate: 54.6%
   Profit Factor: 1.38
   
📉 Risque:
   Max Drawdown: -12.3%
   Sharpe Ratio: 1.85
   Sortino Ratio: 2.12
   
⏱️ Timing:
   Durée moyenne trade: 2.8 min
   Meilleure heure: 14:00-15:00 UTC
   Pire heure: 03:00-04:00 UTC
═══════════════════════════════════════════════════
```

---

## 10. Dépannage

### ❌ Erreur : "API Key invalid"

**Cause :** Clés API incorrectes ou expirées.

**Solution :**
```bash
# Vérifier le .env
cat .env | grep BINANCE

# Tester la connexion
python scripts/test_binance_connection.py
```

### ❌ Erreur : "Insufficient balance"

**Cause :** Pas assez de fonds sur le compte.

**Solution :**
- Vérifier le solde sur Binance
- Réduire `--capital` ou `max_position_pct`
- En testnet : demander des fonds de test

### ❌ Erreur : "Model not found"

**Cause :** Le modèle ML n'existe pas.

**Solution :**
```bash
# Vérifier si le modèle existe
ls models/saved/

# Si vide, entraîner un modèle (voir section 6)
# OU lancer sans ML (le bot utilisera le scanner)
```

### ❌ Le bot ne trade pas

**Causes possibles :**
1. Seuils ML trop élevés
2. Marché calme (pas d'opportunités)
3. Kill switch activé

**Diagnostic :**
```bash
# Logs détaillés
python -m cryptoscalper.main --mode paper --log-level DEBUG

# Vérifier les signaux
tail -f logs/cryptoscalper.log | grep -E "SIGNAL|ALERT"
```

### ❌ Erreur : "Connection reset"

**Cause :** Problème de connexion WebSocket.

**Solution :**
- Vérifier votre connexion internet
- Le bot se reconnecte automatiquement (attendre 30s)
- Si persistant, redémarrer le bot

### ❌ Trop de pertes

**Actions :**
1. Arrêter le bot (`Ctrl+C`)
2. Analyser les trades perdants
3. Ajuster les paramètres :
   - Augmenter `min_probability` (0.70+)
   - Réduire `max_position_pct` (0.10)
   - Augmenter `stop_loss_pct`

---

## 11. FAQ

### Q: Combien puis-je gagner ?

**R:** Impossible à prédire. Les performances passées ne garantissent pas les performances futures. En backtest, le bot a montré des gains de 10-30% sur 2 mois, mais les conditions réelles sont différentes.

### Q: Le bot peut-il perdre tout mon argent ?

**R:** Le kill switch arrête le bot si le drawdown atteint 25% du capital initial. Mais oui, vous pouvez perdre une partie significative de votre capital.

### Q: Puis-je laisser le bot tourner 24/7 ?

**R:** Oui, c'est prévu pour. Utilisez `screen` ou `nohup` pour le laisser tourner même si vous fermez le terminal.

### Q: Faut-il un VPS ?

**R:** Non obligatoire pour commencer, mais recommandé pour le live trading. Un VPS proche des serveurs Binance (Singapour, Tokyo) réduit la latence.

### Q: Le bot fonctionne-t-il sur Binance Futures ?

**R:** Non, actuellement uniquement le spot trading. Les futures pourraient être ajoutés dans une version future.

### Q: Puis-je modifier les stratégies ?

**R:** Oui ! Le code est modulaire. Vous pouvez :
- Modifier les features dans `data/features.py`
- Ajuster les seuils dans `config/default_config.yaml`
- Créer de nouveaux détecteurs dans `data/multi_pair_scanner.py`

---

## 12. Glossaire

| Terme | Définition |
|-------|------------|
| **Scalping** | Stratégie de trading à très court terme (secondes à minutes) |
| **Paper Trading** | Simulation de trading sans argent réel |
| **Stop-Loss (SL)** | Ordre automatique pour limiter les pertes |
| **Take-Profit (TP)** | Ordre automatique pour sécuriser les gains |
| **Drawdown** | Perte maximale depuis un pic de capital |
| **Win Rate** | Pourcentage de trades gagnants |
| **Profit Factor** | Ratio gains totaux / pertes totales |
| **Kill Switch** | Arrêt d'urgence si trop de pertes |
| **OCO** | One-Cancels-Other : ordre combiné SL + TP |
| **Features** | Variables utilisées par le modèle ML |
| **XGBoost** | Algorithme de Machine Learning utilisé |
| **WebSocket** | Connexion temps réel pour recevoir les prix |
| **Testnet** | Environnement de test Binance (faux argent) |

---

## 📞 Support

- **Issues GitHub :** [Lien vers les issues]
- **Documentation :** Ce fichier !
- **Logs :** Toujours inclure les logs lors d'un rapport de bug

---

## 📜 Licence

Ce projet est fourni "tel quel", sans garantie. Utilisez-le à vos propres risques.

---

**Bon trading ! 🚀**