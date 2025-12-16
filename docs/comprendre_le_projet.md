# 🎓 Comprendre le Projet CryptoScalper AI

> Ce document explique en détail comment fonctionne le bot de trading, étape par étape, de manière accessible.

---

## 📋 Table des matières

1. [L'objectif du projet](#1-lobjectif-du-projet)
2. [Les données : le carburant du modèle](#2-les-données--le-carburant-du-modèle)
3. [Les features : transformer les données en signaux](#3-les-features--transformer-les-données-en-signaux)
4. [Les labels : apprendre au modèle ce qu'est un "bon" trade](#4-les-labels--apprendre-au-modèle-ce-quest-un-bon-trade)
5. [Le modèle ML : le cerveau du bot](#5-le-modèle-ml--le-cerveau-du-bot)
6. [La stratégie de trading](#6-la-stratégie-de-trading)
7. [Le backtest : tester avant de risquer de l'argent](#7-le-backtest--tester-avant-de-risquer-de-largent)
8. [Les fichiers du projet](#8-les-fichiers-du-projet)
9. [Glossaire](#9-glossaire)

---

## 1. L'objectif du projet

### En une phrase
> **Prédire quand une crypto va monter de 2% avant de descendre de 1%, et acheter automatiquement à ce moment.**

### L'idée générale

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   DONNÉES         →    MODÈLE ML    →    DÉCISION              │
│   (historique)         (cerveau)         (acheter ou non)      │
│                                                                 │
│   Prix, volume,        Analyse les       "Proba de succès:     │
│   indicateurs...       patterns          25% → J'ACHÈTE !"     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pourquoi le Machine Learning ?

Un humain ne peut pas :
- Surveiller 5 cryptos 24h/24
- Analyser 42 indicateurs simultanément
- Prendre des décisions sans émotions
- Réagir en quelques secondes

Le ML peut faire tout ça !

---

## 2. Les données : le carburant du modèle

### Qu'est-ce qu'une "bougie" (candle) ?

Une bougie représente **1 minute** de trading. Elle contient :

```
        │
        │  ← High (prix max atteint)
   ┌────┴────┐
   │         │  ← Corps (ouverture → fermeture)
   │  VERT   │     Vert = le prix a monté
   │         │     Rouge = le prix a baissé
   └────┬────┘
        │
        │  ← Low (prix min atteint)
        
   + Volume (quantité échangée)
```

### Données téléchargées

```
data_cache/BTCUSDT_1m.parquet
├── open_time      → 2025-09-16 13:07:00
├── open           → 104,500.00 (prix d'ouverture)
├── high           → 104,550.00 (plus haut)
├── low            → 104,480.00 (plus bas)
├── close          → 104,520.00 (prix de fermeture)
├── volume         → 125.5 BTC (volume échangé)
└── ... (autres colonnes)
```

### Combien de données ?

| Période | Bougies par crypto | Total (5 cryptos) |
|---------|-------------------|-------------------|
| 1 jour | 1,440 | 7,200 |
| 90 jours | ~130,000 | 650,000 |
| 180 jours | ~260,000 | 1,300,000 |

Plus on a de données, plus le modèle peut apprendre de patterns différents !

---

## 3. Les features : transformer les données en signaux

### C'est quoi une "feature" ?

Une feature est un **indicateur calculé** à partir des données brutes.

**Analogie** : Imagine que tu dois prédire s'il va pleuvoir.
- Donnée brute : température = 15°C
- Features : 
  - Température moyenne sur 7 jours
  - Écart avec la normale saisonnière
  - Tendance (monte ou descend ?)

### Les 42 features du projet

#### 🎯 Momentum (10 features) - "La crypto a-t-elle de l'élan ?"

| Feature | Signification | Valeurs |
|---------|--------------|---------|
| `rsi_14` | Force relative sur 14 périodes | 0-100 (>70 = surachat, <30 = survente) |
| `rsi_7` | Force relative sur 7 périodes (plus réactif) | 0-100 |
| `stoch_k` | Stochastique %K | 0-100 (>80 = surachat, <20 = survente) |
| `stoch_d` | Stochastique %D (moyenne de %K) | 0-100 |
| `williams_r` | Williams %R | -100 à 0 |
| `roc_5` | Rate of Change 5 min | % de variation |
| `roc_10` | Rate of Change 10 min | % de variation |
| `momentum_5` | Variation sur 5 min | % (normalisé) |
| `cci` | Commodity Channel Index | -200 à +200 typiquement |
| `cmo` | Chande Momentum Oscillator | -100 à +100 |

**Exemple concret RSI** :
```
RSI = 25 → "La crypto est survendue, beaucoup ont vendu"
         → Potentiel rebond à la hausse ?
         
RSI = 75 → "La crypto est surachetée, beaucoup ont acheté"
         → Potentiel correction à la baisse ?
```

#### 📈 Tendance (8 features) - "Dans quelle direction va le prix ?"

| Feature | Signification |
|---------|--------------|
| `ema_5_ratio` | Prix actuel / Moyenne mobile 5 min |
| `ema_10_ratio` | Prix actuel / Moyenne mobile 10 min |
| `ema_20_ratio` | Prix actuel / Moyenne mobile 20 min |
| `macd_line` | Différence entre 2 moyennes mobiles |
| `macd_signal` | Moyenne du MACD |
| `macd_histogram` | MACD - Signal (momentum de la tendance) |
| `adx` | Force de la tendance (0-100) |
| `aroon_oscillator` | Direction de la tendance (-100 à +100) |

**Exemple EMA ratio** :
```
ema_5_ratio = 1.002 → Le prix est 0.2% AU-DESSUS de sa moyenne 5 min
                    → Tendance haussière court terme

ema_5_ratio = 0.998 → Le prix est 0.2% EN-DESSOUS de sa moyenne 5 min
                    → Tendance baissière court terme
```

#### 🌊 Volatilité (6 features) - "Le prix bouge-t-il beaucoup ?"

| Feature | Signification |
|---------|--------------|
| `bb_width` | Largeur des bandes de Bollinger (% du prix) |
| `bb_position` | Position dans les bandes (-0.5 à +0.5) |
| `atr` | Average True Range (volatilité moyenne) |
| `atr_percent` | ATR en % du prix |
| `returns_std` | Écart-type des rendements (volatilité) |
| `hl_range_avg` | Range moyen High-Low |

**Exemple ATR** :
```
ATR = 0.1% → Le prix bouge peu, marché calme
ATR = 0.5% → Le prix bouge beaucoup, marché agité

Un ATR élevé = plus de chances d'atteindre le TP de 2%
           mais aussi plus de risque de toucher le SL !
```

#### 📊 Volume (5 features) - "Y a-t-il de l'intérêt pour cette crypto ?"

| Feature | Signification |
|---------|--------------|
| `volume_relative` | Volume actuel / Volume moyen 20 min |
| `obv_slope` | Pente de l'On-Balance Volume |
| `volume_delta` | Volume × direction du prix |
| `vwap_distance` | Distance au prix moyen pondéré par volume |
| `ad_line` | Accumulation/Distribution |

**Exemple volume_relative** :
```
volume_relative = 3.0 → Volume 3× supérieur à la normale !
                      → Quelque chose se passe, fort intérêt

volume_relative = 0.5 → Volume 2× inférieur à la normale
                      → Marché peu actif
```

#### 🕯️ Price Action (5 features) - "Comment se comportent les bougies ?"

| Feature | Signification |
|---------|--------------|
| `returns_1m` | Rendement sur 1 minute (%) |
| `returns_5m` | Rendement sur 5 minutes (%) |
| `returns_15m` | Rendement sur 15 minutes (%) |
| `consecutive_green` | Nombre de bougies vertes consécutives |
| `candle_body_ratio` | Taille du corps / Taille totale |

#### 📖 Orderbook (8 features) - Placeholders pour données temps réel

Ces features sont à 0 dans le dataset historique (pas de données orderbook), mais utilisables en trading réel.

### Pourquoi normaliser les features ?

**Le problème** :
```
BTC price  = 100,000 USD
DOGE price = 0.30 USD

ATR de BTC  = 500 USD (absolu)
ATR de DOGE = 0.015 USD (absolu)
```

Si on utilise les valeurs absolues, le modèle ne peut pas comparer !

**La solution - Normaliser** :
```
ATR de BTC  = 500 / 100,000 × 100 = 0.5%
ATR de DOGE = 0.015 / 0.30 × 100 = 5.0%

→ Maintenant on peut comparer : DOGE est plus volatile que BTC !
```

---

## 4. Les labels : apprendre au modèle ce qu'est un "bon" trade

### Le concept

Pour entraîner un modèle ML, il faut lui montrer des exemples :
- "Voici une situation → c'était un BON moment pour acheter (label = 1)"
- "Voici une situation → c'était un MAUVAIS moment pour acheter (label = 0)"

### Notre stratégie de labeling : SL/TP

```
                     TP (+2%)
                   ┌─────────────────────────
                   │
     Prix ─────────┼─── Entry Point (achat)
       ↑           │
                   │
                   └─────────────────────────
                     SL (-1%)
                     
     ──────────────────────────────────────────→ Temps
                   0        ...        120 min
```

**Règles** :
- **Label = 1** (bon trade) : Le prix touche +2% (TP) AVANT de toucher -1% (SL)
- **Label = 0** (mauvais trade) : Le prix touche -1% (SL) en premier, OU ni l'un ni l'autre en 120 min

### Exemple concret

```python
# À 10:00, BTC = 100,000 USD
Entry = 100,000

TP = 100,000 × 1.02 = 102,000  # +2%
SL = 100,000 × 0.99 = 99,000   # -1%

# Scénario A : Le prix monte à 102,500 à 10:45
→ TP touché en premier → Label = 1 ✅

# Scénario B : Le prix descend à 98,500 à 10:30
→ SL touché en premier → Label = 0 ❌

# Scénario C : Le prix reste entre 99,500 et 101,500 pendant 2h
→ Timeout, ni TP ni SL → Label = 0 ❌
```

### Distribution dans notre dataset

```
Total samples: 648,205

Labels positifs (TP atteint): 34,687 (5.4%)
Labels négatifs (SL/timeout): 613,518 (94.6%)

→ Seulement 5.4% des moments sont de "bons" moments pour acheter !
```

C'est **normal** : on ne veut acheter que dans les meilleures conditions.

---

## 5. Le modèle ML : le cerveau du bot

### XGBoost - C'est quoi ?

XGBoost = "Extreme Gradient Boosting"

**Analogie simplifiée** :

Imagine une équipe de médecins qui doivent diagnostiquer un patient :
1. Le 1er médecin donne son avis (arbre de décision #1)
2. Le 2ème médecin corrige les erreurs du 1er (arbre #2)
3. Le 3ème corrige les erreurs restantes (arbre #3)
4. ... et ainsi de suite

XGBoost combine des centaines de "petits experts" (arbres) pour faire une prédiction finale.

### Comment ça marche concrètement ?

```
ENTRÉE (42 features)          MODÈLE                    SORTIE
─────────────────────         ──────                    ──────
RSI = 32 ───────────┐
Stoch = 25 ─────────┤
ATR = 0.08 ─────────┼────→   XGBoost    ────→   Probabilité = 28%
Volume = 2.5 ───────┤        (500 arbres)
...                 │
MACD = -0.02 ───────┘

                                              "28% de chances que
                                               le prix monte de 2%
                                               avant de baisser de 1%"
```

### L'entraînement

```
┌─────────────────────────────────────────────────────────────┐
│  DATASET D'ENTRAÎNEMENT (453,743 samples)                   │
│                                                             │
│  Features          Label                                    │
│  ─────────         ─────                                    │
│  [RSI=28, ...]  →  1 (bon trade)   ← Le modèle apprend     │
│  [RSI=65, ...]  →  0 (mauvais)        les patterns qui     │
│  [RSI=31, ...]  →  1 (bon trade)      mènent au succès     │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘

                         ↓ Entraînement

┌─────────────────────────────────────────────────────────────┐
│  MODÈLE ENTRAÎNÉ                                            │
│                                                             │
│  "J'ai appris que quand RSI < 35 ET ATR > 0.05             │
│   ET volume_relative > 2... il y a plus de chances         │
│   de succès !"                                              │
└─────────────────────────────────────────────────────────────┘
```

### Métriques d'évaluation

| Métrique | Signification | Notre résultat |
|----------|---------------|----------------|
| **AUC** | Capacité à distinguer bons/mauvais trades (0.5 = hasard, 1.0 = parfait) | 0.77 |
| **Precision** | Parmi les trades prédits positifs, combien le sont vraiment ? | Variable selon seuil |
| **Recall** | Parmi les vrais positifs, combien sont détectés ? | Variable selon seuil |

### L'importance des features

Le modèle nous dit quelles features sont les plus utiles :

```
🏆 TOP 5 FEATURES:
1. atr          29% ████████████████  ← La volatilité est CLEF !
2. atr_percent  22% ████████████
3. hl_range_avg  8% █████
4. momentum_5    3% ██
5. volume_delta  3% ██
```

**Interprétation** : Le modèle se base principalement sur la **volatilité** (ATR) pour prédire. C'est logique : pour atteindre +2%, il faut que le prix bouge suffisamment !

---

## 6. La stratégie de trading

### Les paramètres clés

```
┌─────────────────────────────────────────────────────────────┐
│                 STRATÉGIE SWING TRADING                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 Take Profit (TP)     : +2.0%                           │
│  🛑 Stop Loss (SL)       : -1.0%                           │
│  ⏱️  Timeout             : 120 minutes (2 heures)          │
│                                                             │
│  💰 Position Size        : 20% du capital par trade        │
│  🎯 Seuil de probabilité : 20% minimum pour entrer         │
│                                                             │
│  📈 Ratio Risk/Reward    : 2:1 (je risque 1 pour gagner 2) │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Pourquoi ces valeurs ?

#### Le ratio 2:1 (TP=2%, SL=1%)

```
Si je gagne : +2.0% - 0.2% frais = +1.8% net
Si je perds : -1.0% - 0.2% frais = -1.2% net

Pour être rentable :
   Gains × WinRate = Pertes × LossRate
   1.8% × WR = 1.2% × (1 - WR)
   
   WR = 1.2 / (1.8 + 1.2) = 40%
   
→ Il suffit de gagner 40% des trades pour être rentable !
```

#### Le seuil de 20%

```
Plus le seuil est haut → Moins de trades mais meilleure qualité
Plus le seuil est bas  → Plus de trades mais moins de qualité

Seuil 15% : Beaucoup de trades, WR trop bas
Seuil 20% : Bon compromis (WR ~43%)
Seuil 25% : Peu de trades, WR légèrement meilleur
```

### Les frais Binance

C'est crucial de comprendre l'impact des frais !

```
Frais Binance Spot : 0.1% par transaction

Un trade = 2 transactions (achat + vente)
Frais total = 0.1% + 0.1% = 0.2% par trade

Exemple avec 25€ :
├── Achat : 25€ × 0.1% = 0.025€ de frais
├── Vente : 25€ × 0.1% = 0.025€ de frais
└── Total : 0.05€ de frais par trade
```

**Pourquoi le scalping ne fonctionne pas** :
```
Scalping (TP=0.5%) :
   Gain brut : +0.5%
   Frais     : -0.2%
   Gain net  : +0.3%  ← Les frais mangent 40% du gain !

Swing (TP=2%) :
   Gain brut : +2.0%
   Frais     : -0.2%
   Gain net  : +1.8%  ← Les frais ne prennent que 10%
```

---

## 7. Le backtest : tester avant de risquer de l'argent

### C'est quoi un backtest ?

Simuler la stratégie sur des données historiques pour voir si elle aurait été rentable.

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION (Backtest)                    │
│                                                             │
│  Données historiques (90 jours)                             │
│  ─────────────────────────────────                          │
│                                                             │
│  Pour chaque minute :                                       │
│    1. Calculer les features                                 │
│    2. Demander au modèle : "Probabilité de succès ?"        │
│    3. Si proba ≥ 20% → Simuler le trade                    │
│    4. Enregistrer si WIN ou LOSS                            │
│                                                             │
│  Résultat final :                                           │
│  ├── Nombre de trades                                       │
│  ├── Win Rate                                               │
│  └── PnL total                                              │
└─────────────────────────────────────────────────────────────┘
```

### Notre résultat de backtest

```
📊 BACKTEST SUR 90 JOURS
════════════════════════════════════════════════════

Trades totaux : 1,001
├── Gagnants  : 435 (43.5%)
└── Perdants  : 566 (56.5%)

PnL total : +31.27%

Par crypto :
├── BTCUSDT  :  15 trades, WR 46.7%
├── ETHUSDT  : 165 trades, WR 44.8%
├── SOLUSDT  : 293 trades, WR 40.6%
├── XRPUSDT  : 223 trades, WR 45.7%
└── DOGEUSDT : 305 trades, WR 43.6%

════════════════════════════════════════════════════
```

### Interprétation

```
✅ Win Rate (43.5%) > Win Rate requis (40%)
   → La stratégie est rentable !

✅ PnL positif (+31.27%)
   → Sur 90 jours, on aurait fait +31% de profit

📊 Avec 25€ de capital :
   → 25€ × 1.3127 = 32.82€ (+7.82€ de profit)
```

---

## 8. Les fichiers du projet

### Structure simplifiée

```
cryptoscalper/
│
├── 📊 data/                      # Gestion des données
│   ├── binance_client.py         # Connexion à Binance
│   ├── historical.py             # Téléchargement historique
│   ├── features.py               # ⭐ Calcul des 42 features (TEMPS RÉEL)
│   └── dataset.py                # ⭐ Création du dataset (ENTRAÎNEMENT)
│
├── 🧠 models/                    # Machine Learning
│   ├── predictor.py              # Utiliser le modèle pour prédire
│   └── trainer.py                # Entraîner le modèle
│
├── 💹 trading/                   # Logique de trading
│   ├── signals.py                # Générer les signaux d'achat
│   ├── risk_manager.py           # Gérer le risque
│   └── executor.py               # Exécuter les trades
│
├── 📁 scripts/                   # Scripts à lancer
│   ├── download_data.py          # Télécharger les données
│   ├── prepare_dataset.py        # Préparer le dataset
│   ├── train_model.py            # Entraîner le modèle
│   └── validate_features.py      # Vérifier l'alignement des features
│
├── 📂 data_cache/                # Données téléchargées (90 jours)
│   ├── BTCUSDT_1m.parquet
│   ├── ETHUSDT_1m.parquet
│   └── ...
│
├── 📂 datasets/                  # Datasets pour l'entraînement
│   ├── swing_final_train.parquet # 70% pour entraîner
│   ├── swing_final_val.parquet   # 15% pour valider
│   └── swing_final_test.parquet  # 15% pour tester
│
└── 📂 models/saved/              # Modèles entraînés
    └── swing_final_model.joblib  # Le modèle actuel
```

### Le flux de travail

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE D'ENTRAÎNEMENT                        │
└─────────────────────────────────────────────────────────────────────┘

    ① download_data.py
       ─────────────────
       Binance API → data_cache/*.parquet
       "Télécharge 90/180 jours de bougies"
    
            ↓
    
    ② prepare_dataset.py
       ──────────────────
       data_cache/*.parquet → datasets/*.parquet
       "Calcule les features + labels pour chaque bougie"
    
            ↓
    
    ③ train_model.py
       ─────────────
       datasets/*.parquet → models/saved/*.joblib
       "Entraîne XGBoost sur les données"


┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE DE TRADING                            │
└─────────────────────────────────────────────────────────────────────┘

    Chaque minute :
    
    ① Récupérer les 100 dernières bougies (Binance API)
            ↓
    ② Calculer les 42 features (features.py)
            ↓
    ③ Demander au modèle la probabilité (predictor.py)
            ↓
    ④ Si proba ≥ 20% → Acheter ! (executor.py)
            ↓
    ⑤ Surveiller le trade (SL/TP) et vendre
```

### Fichiers critiques et leur rôle

| Fichier | Rôle | Importance |
|---------|------|------------|
| `features.py` | Calcule les features en TEMPS RÉEL | ⭐⭐⭐ |
| `dataset.py` | Calcule les features pour l'ENTRAÎNEMENT | ⭐⭐⭐ |
| | **Ces deux fichiers DOIVENT calculer les features de la même façon !** | |
| `trainer.py` | Entraîne le modèle XGBoost | ⭐⭐ |
| `predictor.py` | Utilise le modèle pour prédire | ⭐⭐ |

---

## 9. Glossaire

### Termes de trading

| Terme | Définition |
|-------|------------|
| **Long** | Parier sur la hausse (acheter puis vendre plus cher) |
| **Short** | Parier sur la baisse (vendre puis racheter moins cher) |
| **TP (Take Profit)** | Prix auquel on vend pour encaisser le profit |
| **SL (Stop Loss)** | Prix auquel on vend pour limiter la perte |
| **Entry** | Prix d'entrée (achat) |
| **Exit** | Prix de sortie (vente) |
| **PnL** | Profit and Loss (gains et pertes) |
| **Win Rate** | Pourcentage de trades gagnants |
| **Spread** | Différence entre prix d'achat et de vente |
| **Slippage** | Différence entre prix attendu et prix réel |

### Termes techniques

| Terme | Définition |
|-------|------------|
| **Feature** | Variable d'entrée du modèle (indicateur calculé) |
| **Label** | Variable de sortie du modèle (0 ou 1) |
| **Dataset** | Ensemble de données pour entraîner le modèle |
| **Train/Val/Test** | Splits du dataset (entraînement/validation/test) |
| **Overfitting** | Le modèle mémorise au lieu d'apprendre (mauvais) |
| **AUC** | Métrique de qualité du modèle (0.5 à 1.0) |
| **Threshold** | Seuil de probabilité pour déclencher un trade |

### Indicateurs techniques

| Indicateur | Catégorie | Ce qu'il mesure |
|------------|-----------|-----------------|
| **RSI** | Momentum | Force relative (survente/surachat) |
| **MACD** | Tendance | Convergence/divergence des moyennes |
| **Bollinger Bands** | Volatilité | Bandes de prix normales |
| **ATR** | Volatilité | Amplitude moyenne des mouvements |
| **OBV** | Volume | Pression acheteuse/vendeuse |
| **Stochastique** | Momentum | Position du prix dans son range |

---

## 🎯 Résumé en une image

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│     DONNÉES              FEATURES           MODÈLE         ACTION   │
│     ───────              ────────           ──────         ──────   │
│                                                                     │
│   ┌─────────┐          ┌─────────┐       ┌─────────┐    ┌────────┐ │
│   │ Bougies │ ──────→  │ RSI=32  │ ────→ │ XGBoost │ ─→ │ ACHAT  │ │
│   │ OHLCV   │          │ ATR=0.1 │       │         │    │  si    │ │
│   │ Volume  │          │ MACD=.. │       │ Proba = │    │ >20%   │ │
│   └─────────┘          │ ...     │       │  28%    │    └────────┘ │
│                        └─────────┘       └─────────┘               │
│                                                                     │
│   100 dernières        42 features       Probabilité    Trade avec │
│   bougies              calculées         de succès      SL/TP      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

      +31% de profit sur 90 jours avec 1001 trades (43.5% WR)
```

---

*Document créé le 16 décembre 2025*