# 🧹 Clean Code Rules - CryptoScalper AI

> Ces règles sont **obligatoires** pour tout le code du projet.
> Claude doit les respecter à chaque génération de code.

---

## 📏 Règle #1: Fonctions Courtes

```
✅ Maximum 15-20 lignes par fonction
✅ Une fonction = UNE responsabilité
✅ Nom descriptif qui dit ce que ça fait
❌ Pas de fonction "fourre-tout"
❌ Pas de commentaire pour expliquer ce que fait le code (le nom doit suffire)
```

### Exemple ❌ Mauvais
```python
def process_data(data):
    # Nettoyer les données
    cleaned = []
    for item in data:
        if item is not None and item != "":
            cleaned.append(item.strip())
    
    # Calculer la moyenne
    total = 0
    for item in cleaned:
        total += float(item)
    avg = total / len(cleaned) if cleaned else 0
    
    # Formater le résultat
    result = f"Average: {avg:.2f}"
    return result
```

### Exemple ✅ Bon
```python
def clean_data(data: List[str]) -> List[str]:
    """Supprime les valeurs vides et nettoie les espaces."""
    return [item.strip() for item in data if item]


def calculate_average(values: List[str]) -> float:
    """Calcule la moyenne d'une liste de valeurs numériques."""
    if not values:
        return 0.0
    numbers = [float(v) for v in values]
    return sum(numbers) / len(numbers)


def format_average_result(average: float) -> str:
    """Formate le résultat de la moyenne pour affichage."""
    return f"Average: {average:.2f}"


def process_data(data: List[str]) -> str:
    """Pipeline complet: nettoyage → calcul → formatage."""
    cleaned = clean_data(data)
    average = calculate_average(cleaned)
    return format_average_result(average)
```

---

## 📏 Règle #2: Nommage Explicite

```
✅ Variables: nom_descriptif_en_snake_case
✅ Fonctions: verbe_action_snake_case
✅ Classes: NomEnPascalCase
✅ Constantes: MAJUSCULES_AVEC_UNDERSCORE
❌ Pas d'abréviations obscures (sauf standards: df, i, n)
❌ Pas de noms génériques (data, info, result, temp)
```

### Exemples
```python
# ❌ Mauvais
def calc(d):
    r = d['p'] * d['q']
    return r

# ✅ Bon
def calculate_trade_value(trade: dict) -> float:
    return trade['price'] * trade['quantity']


# ❌ Mauvais
x = get_data()
y = process(x)

# ✅ Bon
raw_klines = fetch_klines_from_binance(symbol="BTCUSDT")
processed_candles = normalize_kline_data(raw_klines)
```

---

## 📏 Règle #3: Type Hints Obligatoires

```
✅ Toutes les fonctions ont des type hints
✅ Paramètres ET retour typés
✅ Utiliser Optional, List, Dict, etc. de typing
❌ Pas de fonction sans types
```

### Exemple
```python
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class TradeSignal:
    symbol: str
    price: float
    confidence: float


def generate_signal(
    symbol: str,
    features: Dict[str, float],
    threshold: float = 0.65
) -> Optional[TradeSignal]:
    """Génère un signal si la confiance dépasse le seuil."""
    confidence = calculate_confidence(features)
    
    if confidence < threshold:
        return None
    
    return TradeSignal(
        symbol=symbol,
        price=features['current_price'],
        confidence=confidence
    )
```

---

## 📏 Règle #4: Dataclasses pour les Structures

```
✅ Utiliser @dataclass pour toute structure de données
✅ Utiliser Pydantic BaseModel pour la config (validation)
❌ Pas de dictionnaires "magiques" avec des clés string
❌ Pas de tuples pour des données structurées
```

### Exemple
```python
# ❌ Mauvais
def create_order(symbol, side, price, qty):
    return {
        'symbol': symbol,
        'side': side, 
        'price': price,
        'quantity': qty
    }

order = create_order('BTC', 'BUY', 42000, 0.01)
print(order['pric'])  # KeyError à runtime!


# ✅ Bon
from dataclasses import dataclass

@dataclass
class Order:
    symbol: str
    side: str
    price: float
    quantity: float


def create_order(symbol: str, side: str, price: float, qty: float) -> Order:
    return Order(symbol=symbol, side=side, price=price, quantity=qty)

order = create_order('BTC', 'BUY', 42000, 0.01)
print(order.price)  # Autocomplétion + type checking!
```

---

## 📏 Règle #5: Gestion d'Erreurs Explicite

```
✅ Try/except ciblé (exception spécifique)
✅ Logger les erreurs avec contexte
✅ Fail fast: retourner tôt si problème
❌ Pas de except: générique (bare except)
❌ Pas de pass silencieux dans except
```

### Exemple
```python
# ❌ Mauvais
def get_price(symbol):
    try:
        response = api.get_ticker(symbol)
        return response['price']
    except:
        pass  # Erreur silencieuse!


# ✅ Bon
from loguru import logger

class PriceFetchError(Exception):
    """Erreur lors de la récupération du prix."""
    pass


def get_price(symbol: str) -> float:
    """Récupère le prix actuel d'un symbol."""
    try:
        response = api.get_ticker(symbol)
        return float(response['price'])
    
    except KeyError as e:
        logger.error(f"Clé manquante dans la réponse pour {symbol}: {e}")
        raise PriceFetchError(f"Format de réponse invalide pour {symbol}")
    
    except ConnectionError as e:
        logger.error(f"Erreur connexion API pour {symbol}: {e}")
        raise PriceFetchError(f"Impossible de contacter l'API pour {symbol}")
```

---

## 📏 Règle #6: Pas de Magic Numbers

```
✅ Constantes nommées pour toute valeur fixe
✅ Configuration externalisée quand possible
❌ Pas de nombres "magiques" dans le code
```

### Exemple
```python
# ❌ Mauvais
def is_good_opportunity(score):
    if score > 0.65 and volume > 1000000:
        return True


# ✅ Bon
MIN_CONFIDENCE_THRESHOLD = 0.65
MIN_VOLUME_24H_USDT = 1_000_000


def is_good_opportunity(score: float, volume: float) -> bool:
    """Vérifie si une opportunité répond aux critères minimum."""
    has_enough_confidence = score > MIN_CONFIDENCE_THRESHOLD
    has_enough_volume = volume > MIN_VOLUME_24H_USDT
    return has_enough_confidence and has_enough_volume
```

---

## 📏 Règle #7: Docstrings Utiles

```
✅ Une ligne si la fonction est évidente
✅ Multi-lignes avec Args/Returns si complexe
✅ Expliquer le POURQUOI, pas le COMMENT
❌ Pas de docstring qui répète le nom de la fonction
```

### Exemple
```python
# ❌ Mauvais (répète le nom)
def calculate_rsi(prices):
    """Calcule le RSI."""
    pass


# ✅ Bon (simple)
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index - indicateur de surachat/survente."""
    pass


# ✅ Bon (complexe)
def generate_trade_signal(
    features: Dict[str, float],
    model: MLPredictor,
    risk_manager: RiskManager
) -> Optional[TradeSignal]:
    """
    Génère un signal de trading si les conditions sont réunies.
    
    Le signal n'est généré que si:
    1. Le modèle prédit une hausse avec confiance > seuil
    2. Le risk manager autorise un nouveau trade
    3. Aucune position n'est déjà ouverte sur ce symbol
    
    Args:
        features: Dictionnaire des features techniques calculées
        model: Instance du prédicteur ML
        risk_manager: Instance du gestionnaire de risque
        
    Returns:
        TradeSignal si conditions réunies, None sinon
    """
    pass
```

---

## 📏 Règle #8: Structure des Fichiers

```
✅ Un fichier = un module cohérent
✅ Max ~200-300 lignes par fichier
✅ Imports en haut, groupés (stdlib, external, local)
✅ Constantes après les imports
✅ Classes/fonctions principales ensuite
❌ Pas de fichier "utils" fourre-tout géant
```

### Template de fichier
```python
"""
Module: nom_du_module.py
Description courte de ce que fait ce module.
"""

# === Imports standard ===
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict

# === Imports externes ===
import pandas as pd
from loguru import logger

# === Imports locaux ===
from config.settings import TradingConfig
from utils.helpers import round_price

# === Constantes ===
DEFAULT_TIMEOUT_SECONDS = 30
MAX_RETRY_ATTEMPTS = 3

# === Dataclasses ===
@dataclass
class MyDataClass:
    ...

# === Classes principales ===
class MyMainClass:
    ...

# === Fonctions utilitaires du module ===
def helper_function():
    ...
```

---

## 📏 Règle #9: Tests

```
✅ Chaque module a son fichier test_xxx.py
✅ Tester les cas nominaux ET les erreurs
✅ Tests indépendants (pas d'ordre requis)
✅ Noms de tests descriptifs
❌ Pas de code non testé pour les parties critiques
```

### Exemple
```python
# tests/test_risk_manager.py

def test_position_size_respects_maximum():
    """La taille de position ne doit jamais dépasser le max configuré."""
    config = RiskConfig(max_position_pct=0.20)
    rm = RiskManager(config)
    
    size = rm.calculate_position_size(capital=100, signal=mock_signal)
    
    assert size <= 100 * 0.20


def test_kill_switch_activates_on_max_drawdown():
    """Le kill switch doit s'activer quand le drawdown atteint le seuil."""
    config = RiskConfig(max_drawdown_pct=0.25)
    rm = RiskManager(config)
    
    rm.update_capital(initial=100)
    rm.update_capital(current=74)  # -26% drawdown
    
    assert rm.is_kill_switch_active() is True


def test_trade_rejected_when_daily_loss_exceeded():
    """Un trade doit être refusé si la perte journalière max est atteinte."""
    config = RiskConfig(max_daily_loss_pct=0.10)
    rm = RiskManager(config)
    rm.register_loss(amount=15)  # Sur capital de 100
    
    can_trade, reason = rm.can_open_trade(mock_signal)
    
    assert can_trade is False
    assert "perte journalière" in reason.lower()
```

---

## 📏 Règle #10: Async Propre

```
✅ async/await cohérent (pas de mix sync/async)
✅ Utiliser asyncio.gather pour paralléliser
✅ Timeout sur les opérations réseau
✅ Cleanup propre (finally, context managers)
❌ Pas de time.sleep() dans du code async
```

### Exemple
```python
# ❌ Mauvais
async def fetch_all_prices(symbols):
    prices = []
    for symbol in symbols:
        price = await get_price(symbol)  # Séquentiel!
        prices.append(price)
    return prices


# ✅ Bon
async def fetch_all_prices(symbols: List[str]) -> Dict[str, float]:
    """Récupère les prix en parallèle pour tous les symbols."""
    tasks = [get_price(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    prices = {}
    for symbol, result in zip(symbols, results):
        if isinstance(result, Exception):
            logger.warning(f"Erreur prix {symbol}: {result}")
            continue
        prices[symbol] = result
    
    return prices


# ✅ Bon - avec timeout
async def get_price_with_timeout(symbol: str, timeout: float = 5.0) -> float:
    """Récupère le prix avec un timeout."""
    try:
        return await asyncio.wait_for(
            get_price(symbol),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"Timeout récupération prix {symbol}")
        raise
```

---

## 🎯 Checklist Avant Commit

Avant de valider du code, vérifier:

- [ ] Fonctions < 20 lignes
- [ ] Noms explicites (pas d'abréviations)
- [ ] Type hints sur toutes les fonctions
- [ ] Dataclasses pour les structures
- [ ] Pas de magic numbers
- [ ] Gestion d'erreurs appropriée
- [ ] Docstrings présentes
- [ ] Tests écrits pour le nouveau code

---

## 🔧 Outils Recommandés

```bash
# Formatage automatique
black .

# Tri des imports
isort .

# Linting
ruff check .

# Type checking
mypy .

# Tests
pytest -v
```

Configuration suggérée dans `pyproject.toml`:
```toml
[tool.black]
line-length = 100

[tool.isort]
profile = "black"
line_length = 100

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "N", "UP", "B", "C4"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
```
