#!/usr/bin/env python3
# scripts/test_signals.py
"""
Tests d'intégration pour le module signals (Phase 6.2).

Vérifie:
- TradeSignal dataclass
- SignalGenerator
- Filtrage par seuils
- Ranking des signaux
- Gestion du cycle de vie

Usage:
    python scripts/test_signals.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import setup_logger, logger


# ============================================
# HELPERS
# ============================================

def create_mock_prediction(
    symbol: str,
    probability_up: float,
    confidence: float,
    is_bullish: bool = True
):
    """Crée une PredictionResult mock."""
    from cryptoscalper.models.predictor import PredictionResult
    
    return PredictionResult(
        symbol=symbol,
        probability_up=probability_up,
        probability_down=1 - probability_up,
        predicted_class=1 if is_bullish else 0,
        confidence=confidence,
        features_used=42,
        model_version="test_v1"
    )


def create_varied_predictions(n: int = 10):
    """Crée une liste variée de prédictions."""
    predictions = []
    
    for i in range(n):
        symbol = f"PAIR{i}USDT"
        # Probabilités variées entre 0.4 et 0.9
        prob = 0.4 + (i / n) * 0.5
        # Confiance variée entre 0.3 et 0.8
        conf = 0.3 + (i / n) * 0.5
        
        predictions.append(create_mock_prediction(
            symbol=symbol,
            probability_up=prob,
            confidence=conf,
            is_bullish=prob > 0.5
        ))
    
    return predictions


# ============================================
# TESTS ENUMS
# ============================================

def test_signal_type():
    """Test 6.2.1 - Enum SignalType."""
    print("\n📊 Test 6.2.1 - SignalType enum...")
    
    from cryptoscalper.trading.signals import SignalType
    
    assert SignalType.BUY.value == "BUY"
    assert SignalType.SELL.value == "SELL"
    assert SignalType.HOLD.value == "HOLD"
    
    print("  ✅ SignalType: BUY, SELL, HOLD")
    return True


def test_signal_strength():
    """Test 6.2.2 - Enum SignalStrength."""
    print("\n💪 Test 6.2.2 - SignalStrength enum...")
    
    from cryptoscalper.trading.signals import SignalStrength
    
    # Test from_probability
    assert SignalStrength.from_probability(0.90) == SignalStrength.VERY_STRONG
    assert SignalStrength.from_probability(0.80) == SignalStrength.STRONG
    assert SignalStrength.from_probability(0.70) == SignalStrength.MODERATE
    assert SignalStrength.from_probability(0.55) == SignalStrength.WEAK
    
    print("  ✅ WEAK < MODERATE < STRONG < VERY_STRONG")
    print("  ✅ from_probability() fonctionne")
    
    return True


def test_signal_status():
    """Test 6.2.3 - Enum SignalStatus."""
    print("\n🔄 Test 6.2.3 - SignalStatus enum...")
    
    from cryptoscalper.trading.signals import SignalStatus
    
    assert SignalStatus.PENDING.value == "PENDING"
    assert SignalStatus.EXECUTED.value == "EXECUTED"
    assert SignalStatus.EXPIRED.value == "EXPIRED"
    assert SignalStatus.CANCELLED.value == "CANCELLED"
    
    print("  ✅ SignalStatus: PENDING, EXECUTED, EXPIRED, CANCELLED")
    return True


# ============================================
# TESTS SIGNAL CONFIG
# ============================================

def test_signal_config():
    """Test 6.2.4 - SignalConfig dataclass."""
    print("\n⚙️ Test 6.2.4 - SignalConfig...")
    
    from cryptoscalper.trading.signals import SignalConfig
    
    # Config par défaut
    config = SignalConfig()
    
    assert config.min_probability == 0.65
    assert config.min_confidence == 0.55
    assert config.validity_seconds == 60
    
    print(f"  ✅ Config par défaut:")
    print(f"     min_probability: {config.min_probability}")
    print(f"     min_confidence: {config.min_confidence}")
    print(f"     validity: {config.validity_seconds}s")
    
    # Config personnalisée
    custom = SignalConfig(
        min_probability=0.70,
        min_confidence=0.60,
        validity_seconds=30
    )
    
    custom.validate()  # Ne devrait pas lever d'exception
    
    print("  ✅ Config personnalisée validée")
    
    # Config invalide
    try:
        invalid = SignalConfig(min_probability=1.5)
        invalid.validate()
        print("  ❌ Aurait dû lever une exception")
        return False
    except ValueError:
        print("  ✅ Validation erreur OK")
    
    return True


# ============================================
# TESTS TRADE SIGNAL
# ============================================

def test_trade_signal_creation():
    """Test 6.2.5 - Création TradeSignal."""
    print("\n📡 Test 6.2.5 - TradeSignal création...")
    
    from cryptoscalper.trading.signals import (
        TradeSignal, SignalType, SignalStrength, SignalStatus
    )
    
    signal = TradeSignal(
        symbol="BTCUSDT",
        signal_type=SignalType.BUY,
        probability=0.75,
        confidence=0.50,
        strength=SignalStrength.STRONG,
        score=0.65
    )
    
    assert signal.symbol == "BTCUSDT"
    assert signal.is_buy is True
    assert signal.is_sell is False
    assert signal.status == SignalStatus.PENDING
    assert signal.is_valid is True  # Pas encore expiré
    
    print(f"  ✅ Signal créé: {signal.symbol}")
    print(f"     Type: {signal.signal_type.value}")
    print(f"     Proba: {signal.probability:.0%}")
    print(f"     Force: {signal.strength.value}")
    
    return True


def test_trade_signal_sl_tp():
    """Test 6.2.6 - Calcul SL/TP."""
    print("\n💰 Test 6.2.6 - Calcul SL/TP...")
    
    from cryptoscalper.trading.signals import (
        TradeSignal, SignalType, SignalStrength
    )
    
    signal = TradeSignal(
        symbol="BTCUSDT",
        signal_type=SignalType.BUY,
        probability=0.75,
        confidence=0.50,
        strength=SignalStrength.STRONG,
        stop_loss_pct=0.004,  # 0.4%
        take_profit_pct=0.003  # 0.3%
    )
    
    current_price = 50000.0
    signal.calculate_sl_tp_prices(current_price)
    
    assert signal.entry_price == 50000.0
    assert signal.stop_loss_price == 50000 * (1 - 0.004)  # 49800
    assert signal.take_profit_price == 50000 * (1 + 0.003)  # 50150
    
    print(f"  ✅ Prix calculés:")
    print(f"     Entry: {signal.entry_price}")
    print(f"     SL: {signal.stop_loss_price}")
    print(f"     TP: {signal.take_profit_price}")
    print(f"     R/R: {signal.risk_reward_ratio:.2f}")
    
    return True


def test_trade_signal_lifecycle():
    """Test 6.2.7 - Cycle de vie du signal."""
    print("\n🔄 Test 6.2.7 - Cycle de vie signal...")
    
    from cryptoscalper.trading.signals import (
        TradeSignal, SignalType, SignalStrength, SignalStatus
    )
    
    signal = TradeSignal(
        symbol="BTCUSDT",
        signal_type=SignalType.BUY,
        probability=0.75,
        confidence=0.50,
        strength=SignalStrength.STRONG,
        valid_until=datetime.now() + timedelta(seconds=60)
    )
    
    # Initial: PENDING
    assert signal.status == SignalStatus.PENDING
    assert signal.is_valid is True
    
    # Exécuter
    signal.execute()
    assert signal.status == SignalStatus.EXECUTED
    assert signal.is_valid is False  # Plus valide après exécution
    
    print("  ✅ PENDING → EXECUTED")
    
    # Test expiration
    signal2 = TradeSignal(
        symbol="ETHUSDT",
        signal_type=SignalType.BUY,
        probability=0.70,
        confidence=0.45,
        strength=SignalStrength.MODERATE,
        valid_until=datetime.now() - timedelta(seconds=1)  # Déjà expiré
    )
    
    assert signal2.is_valid is False  # Expiré
    
    print("  ✅ Signal expiré détecté")
    
    # Test annulation
    signal3 = TradeSignal(
        symbol="BNBUSDT",
        signal_type=SignalType.BUY,
        probability=0.65,
        confidence=0.40,
        strength=SignalStrength.MODERATE,
    )
    
    signal3.cancel()
    assert signal3.status == SignalStatus.CANCELLED
    
    print("  ✅ Signal annulé")
    
    return True


def test_trade_signal_to_dict():
    """Test 6.2.8 - Conversion to_dict."""
    print("\n📋 Test 6.2.8 - to_dict()...")
    
    from cryptoscalper.trading.signals import (
        TradeSignal, SignalType, SignalStrength
    )
    
    signal = TradeSignal(
        symbol="BTCUSDT",
        signal_type=SignalType.BUY,
        probability=0.75,
        confidence=0.50,
        strength=SignalStrength.STRONG,
        score=0.65
    )
    
    d = signal.to_dict()
    
    assert "symbol" in d
    assert "signal_type" in d
    assert "probability" in d
    assert "is_valid" in d
    assert d["symbol"] == "BTCUSDT"
    assert d["signal_type"] == "BUY"
    
    print(f"  ✅ to_dict(): {len(d)} clés")
    
    # Test __str__
    s = str(signal)
    assert "BTCUSDT" in s
    assert "BUY" in s
    
    print(f"  ✅ __str__(): {s}")
    
    return True


# ============================================
# TESTS SIGNAL GENERATOR
# ============================================

def test_signal_generator_creation():
    """Test 6.2.9 - Création SignalGenerator."""
    print("\n🏭 Test 6.2.9 - SignalGenerator création...")
    
    from cryptoscalper.trading.signals import SignalGenerator, SignalConfig
    
    # Par défaut
    gen = SignalGenerator()
    
    assert gen.config.min_probability == 0.65
    assert gen.active_signals_count == 0
    
    print(f"  ✅ Générateur créé (défaut)")
    
    # Avec config
    config = SignalConfig(min_probability=0.70, min_confidence=0.60)
    gen2 = SignalGenerator(config)
    
    assert gen2.config.min_probability == 0.70
    
    print(f"  ✅ Générateur avec config personnalisée")
    
    return True


def test_generate_signals_filtering():
    """Test 6.2.10 - Filtrage des signaux."""
    print("\n🔍 Test 6.2.10 - Filtrage signaux...")
    
    from cryptoscalper.trading.signals import SignalGenerator, SignalConfig
    
    config = SignalConfig(
        min_probability=0.65,
        min_confidence=0.50
    )
    gen = SignalGenerator(config)
    
    # Créer des prédictions variées
    predictions = [
        create_mock_prediction("HIGH", 0.80, 0.60),    # ✓ Passe
        create_mock_prediction("MEDIUM", 0.68, 0.55),  # ✓ Passe
        create_mock_prediction("LOW_PROB", 0.55, 0.60),  # ✗ Proba trop basse
        create_mock_prediction("LOW_CONF", 0.70, 0.40),  # ✗ Confiance trop basse
        create_mock_prediction("BEARISH", 0.40, 0.70, is_bullish=False),  # ✗ Bearish
    ]
    
    signals = gen.generate_signals(predictions)
    
    # Seuls HIGH et MEDIUM devraient passer
    assert len(signals) == 2
    
    symbols = [s.symbol for s in signals]
    assert "HIGH" in symbols
    assert "MEDIUM" in symbols
    assert "LOW_PROB" not in symbols
    assert "LOW_CONF" not in symbols
    assert "BEARISH" not in symbols
    
    print(f"  ✅ {len(signals)}/5 signaux générés (filtrage OK)")
    print(f"     Passés: {symbols}")
    
    return True


def test_generate_signals_exclusion():
    """Test 6.2.11 - Exclusion des positions."""
    print("\n🚫 Test 6.2.11 - Exclusion positions...")
    
    from cryptoscalper.trading.signals import SignalGenerator
    
    gen = SignalGenerator()
    
    predictions = [
        create_mock_prediction("BTCUSDT", 0.80, 0.60),
        create_mock_prediction("ETHUSDT", 0.75, 0.55),
        create_mock_prediction("BNBUSDT", 0.70, 0.50),  # En position
    ]
    
    # Exclure BNBUSDT car déjà en position
    signals = gen.generate_signals(
        predictions,
        current_positions=["BNBUSDT"]
    )
    
    assert len(signals) == 2
    symbols = [s.symbol for s in signals]
    assert "BNBUSDT" not in symbols
    
    print(f"  ✅ BNBUSDT exclu (en position)")
    
    return True


def test_signal_scoring_ranking():
    """Test 6.2.12 - Scoring et ranking."""
    print("\n🏆 Test 6.2.12 - Scoring et ranking...")
    
    from cryptoscalper.trading.signals import SignalGenerator
    
    gen = SignalGenerator()
    
    predictions = [
        create_mock_prediction("LOW", 0.65, 0.55),      # Score faible
        create_mock_prediction("HIGH", 0.90, 0.80),    # Score élevé
        create_mock_prediction("MEDIUM", 0.75, 0.60),  # Score moyen
    ]
    
    signals = gen.generate_signals(predictions)
    
    # Doivent être triés par score décroissant
    assert signals[0].symbol == "HIGH"
    assert signals[1].symbol == "MEDIUM"
    assert signals[2].symbol == "LOW"
    
    print(f"  ✅ Ranking correct:")
    for s in signals:
        print(f"     {s.symbol}: score={s.score:.2f}")
    
    return True


def test_get_top_signals():
    """Test 6.2.13 - Top N signaux."""
    print("\n🔝 Test 6.2.13 - Top signaux...")
    
    from cryptoscalper.trading.signals import SignalGenerator
    
    gen = SignalGenerator()
    
    # Générer plusieurs signaux
    predictions = create_varied_predictions(10)
    gen.generate_signals(predictions)
    
    # Top 3
    top = gen.get_top_signals(n=3)
    
    assert len(top) <= 3
    
    # Vérifier l'ordre
    for i in range(len(top) - 1):
        assert top[i].score >= top[i + 1].score
    
    print(f"  ✅ Top {len(top)} signaux récupérés")
    
    return True


def test_signal_lifecycle_management():
    """Test 6.2.14 - Gestion cycle de vie."""
    print("\n🔄 Test 6.2.14 - Gestion cycle de vie...")
    
    from cryptoscalper.trading.signals import SignalGenerator, SignalStatus
    
    gen = SignalGenerator()
    
    predictions = [
        create_mock_prediction("BTCUSDT", 0.80, 0.60),
        create_mock_prediction("ETHUSDT", 0.75, 0.55),
    ]
    
    signals = gen.generate_signals(predictions)
    
    # Marquer un comme exécuté
    executed = gen.mark_signal_executed("BTCUSDT")
    assert executed is not None
    assert executed.status == SignalStatus.EXECUTED
    
    print("  ✅ Signal marqué comme exécuté")
    
    # Annuler l'autre
    cancelled = gen.cancel_signal("ETHUSDT")
    assert cancelled is not None
    assert cancelled.status == SignalStatus.CANCELLED
    
    print("  ✅ Signal annulé")
    
    # Plus de signaux actifs
    active = gen.get_active_signals()
    assert len(active) == 0
    
    print("  ✅ Aucun signal actif restant")
    
    return True


def test_signal_generator_statistics():
    """Test 6.2.15 - Statistiques."""
    print("\n📊 Test 6.2.15 - Statistiques...")
    
    from cryptoscalper.trading.signals import SignalGenerator
    
    gen = SignalGenerator()
    
    # Générer des signaux
    predictions = create_varied_predictions(5)
    gen.generate_signals(predictions)
    
    # Exécuter quelques-uns
    gen.mark_signal_executed("PAIR4USDT")
    gen.cancel_signal("PAIR3USDT")
    
    stats = gen.get_statistics()
    
    assert "total_generated" in stats
    assert "active_count" in stats
    assert "executed_count" in stats
    assert stats["executed_count"] == 1
    
    print(f"  ✅ Statistiques:")
    print(f"     Générés: {stats['total_generated']}")
    print(f"     Actifs: {stats['active_count']}")
    print(f"     Exécutés: {stats['executed_count']}")
    print(f"     Score moyen: {stats['avg_score']:.2f}")
    
    return True


def test_utility_functions():
    """Test 6.2.16 - Fonctions utilitaires."""
    print("\n🔧 Test 6.2.16 - Fonctions utilitaires...")
    
    from cryptoscalper.trading.signals import (
        create_signal_generator,
        filter_signals_by_strength,
        SignalStrength,
        TradeSignal,
        SignalType
    )
    
    # create_signal_generator
    gen = create_signal_generator(
        min_probability=0.70,
        stop_loss_pct=0.005
    )
    
    assert gen.config.min_probability == 0.70
    assert gen.config.default_stop_loss_pct == 0.005
    
    print("  ✅ create_signal_generator()")
    
    # filter_signals_by_strength
    signals = [
        TradeSignal("A", SignalType.BUY, 0.90, 0.80, SignalStrength.VERY_STRONG),
        TradeSignal("B", SignalType.BUY, 0.75, 0.60, SignalStrength.STRONG),
        TradeSignal("C", SignalType.BUY, 0.65, 0.50, SignalStrength.MODERATE),
        TradeSignal("D", SignalType.BUY, 0.55, 0.40, SignalStrength.WEAK),
    ]
    
    # Filtrer par force >= STRONG
    strong = filter_signals_by_strength(signals, SignalStrength.STRONG)
    assert len(strong) == 2
    assert all(s.symbol in ["A", "B"] for s in strong)
    
    print("  ✅ filter_signals_by_strength()")
    
    return True


def test_duplicate_signal_prevention():
    """Test 6.2.17 - Prévention doublons."""
    print("\n🔒 Test 6.2.17 - Prévention doublons...")
    
    from cryptoscalper.trading.signals import SignalGenerator
    
    gen = SignalGenerator()
    
    prediction = create_mock_prediction("BTCUSDT", 0.80, 0.60)
    
    # Premier signal
    signals1 = gen.generate_signals([prediction])
    assert len(signals1) == 1
    
    # Deuxième génération - devrait être ignorée
    signals2 = gen.generate_signals([prediction])
    assert len(signals2) == 0  # Pas de nouveau signal
    
    print("  ✅ Doublon évité pour BTCUSDT")
    
    # Après expiration ou exécution, peut regénérer
    gen.mark_signal_executed("BTCUSDT")
    
    signals3 = gen.generate_signals([prediction])
    assert len(signals3) == 1  # Nouveau signal possible
    
    print("  ✅ Nouveau signal après exécution")
    
    return True


# ============================================
# MAIN
# ============================================

def main():
    """Exécute tous les tests de la Phase 6.2."""
    print("=" * 65)
    print("🧪 CryptoScalper AI - Tests Phase 6.2: Signal Generator")
    print("=" * 65)
    
    setup_logger(level="WARNING")
    
    results = []
    
    try:
        # Tests Enums
        print("\n" + "─" * 50)
        print("📦 6.2.1-6.2.3 Enums")
        print("─" * 50)
        results.append(("6.2.1 SignalType", test_signal_type()))
        results.append(("6.2.2 SignalStrength", test_signal_strength()))
        results.append(("6.2.3 SignalStatus", test_signal_status()))
        
        # Tests Config
        print("\n" + "─" * 50)
        print("⚙️ 6.2.4 SignalConfig")
        print("─" * 50)
        results.append(("6.2.4 SignalConfig", test_signal_config()))
        
        # Tests TradeSignal
        print("\n" + "─" * 50)
        print("📡 6.2.5-6.2.8 TradeSignal")
        print("─" * 50)
        results.append(("6.2.5 Création", test_trade_signal_creation()))
        results.append(("6.2.6 SL/TP", test_trade_signal_sl_tp()))
        results.append(("6.2.7 Cycle de vie", test_trade_signal_lifecycle()))
        results.append(("6.2.8 to_dict", test_trade_signal_to_dict()))
        
        # Tests SignalGenerator
        print("\n" + "─" * 50)
        print("🏭 6.2.9-6.2.17 SignalGenerator")
        print("─" * 50)
        results.append(("6.2.9 Création", test_signal_generator_creation()))
        results.append(("6.2.10 Filtrage", test_generate_signals_filtering()))
        results.append(("6.2.11 Exclusion positions", test_generate_signals_exclusion()))
        results.append(("6.2.12 Scoring/Ranking", test_signal_scoring_ranking()))
        results.append(("6.2.13 Top signaux", test_get_top_signals()))
        results.append(("6.2.14 Cycle de vie", test_signal_lifecycle_management()))
        results.append(("6.2.15 Statistiques", test_signal_generator_statistics()))
        results.append(("6.2.16 Utils", test_utility_functions()))
        results.append(("6.2.17 Anti-doublons", test_duplicate_signal_prevention()))
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Résumé
    print("\n" + "=" * 65)
    print("📊 RÉSULTATS PHASE 6.2")
    print("=" * 65)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("─" * 65)
    print(f"  Total: {passed}/{len(results)} tests passés")
    print("=" * 65)
    
    if failed == 0:
        print("🎉 Phase 6.2 - Signal Generator : VALIDÉE !")
        return 0
    else:
        print(f"⚠️  {failed} test(s) échoué(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())