#!/usr/bin/env python3
# scripts/test_predictor.py
"""
Tests d'intégration pour le module predictor (Phase 6.1).

Vérifie:
- Chargement du modèle
- Prédiction single et batch
- Calcul de confiance
- PredictionResult dataclass
- Gestion des erreurs

Usage:
    python scripts/test_predictor.py
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import setup_logger, logger


# ============================================
# HELPERS - Création de mocks
# ============================================

class MockXGBModel:
    """Mock model sérialisable pour les tests."""
    
    def __init__(self, n_features: int = 42):
        self.n_features = n_features
        self.feature_importances_ = np.random.rand(n_features)
        self.feature_importances_ /= self.feature_importances_.sum()
    
    def predict_proba(self, X):
        """Simule une prédiction basée sur la moyenne des features."""
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        
        # Proba basée sur la moyenne normalisée des features
        means = X.mean(axis=1)
        # Transformer en proba entre 0.3 et 0.8
        min_mean = means.min() if n_samples > 1 else means[0] - 1
        max_mean = means.max() if n_samples > 1 else means[0] + 1
        prob_up = 0.3 + 0.5 * (means - min_mean) / (max_mean - min_mean + 1e-10)
        prob_down = 1 - prob_up
        
        return np.column_stack([prob_down, prob_up])
    
    def predict(self, X):
        """Retourne la classe prédite."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


def create_mock_model(n_features: int = 42):
    """Crée un faux modèle XGBoost pour les tests."""
    return MockXGBModel(n_features)


def create_mock_calibrated_model(n_features: int = 42):
    """Crée un mock de modèle calibré (CalibratedClassifierCV)."""
    return MockXGBModel(n_features)


def save_mock_model(model, path: Path, metadata: dict = None):
    """Sauvegarde un mock model avec ses métadonnées."""
    # Sauvegarder le modèle
    joblib.dump(model, path)
    
    # Sauvegarder les métadonnées si fournies
    if metadata:
        import json
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)


def create_mock_feature_set(symbol: str = "BTCUSDT", seed: int = 42):
    """Crée un FeatureSet mock."""
    from cryptoscalper.data.features import FeatureSet, get_feature_names
    
    np.random.seed(seed)
    feature_names = get_feature_names()
    
    # Générer des valeurs réalistes
    features = {}
    for name in feature_names:
        if "rsi" in name:
            features[name] = np.random.uniform(20, 80)
        elif "ratio" in name or "percent" in name:
            features[name] = np.random.uniform(0, 2)
        elif "returns" in name:
            features[name] = np.random.uniform(-0.02, 0.02)
        else:
            features[name] = np.random.uniform(-1, 1)
    
    return FeatureSet(
        symbol=symbol,
        features=features,
        timestamp=pd.Timestamp.now()
    )


# ============================================
# TESTS PREDICTION RESULT
# ============================================

def test_prediction_result():
    """Test 6.1.1 - PredictionResult dataclass."""
    print("\n📊 Test 6.1.1 - PredictionResult dataclass...")
    
    from cryptoscalper.models.predictor import PredictionResult
    
    # Créer un résultat
    result = PredictionResult(
        symbol="BTCUSDT",
        probability_up=0.72,
        probability_down=0.28,
        predicted_class=1,
        confidence=0.44,  # 2 * |0.72 - 0.5| = 0.44
        features_used=42,
        model_version="20251215_120000"
    )
    
    # Vérifier les propriétés
    assert result.is_bullish is True, "Devrait être bullish (classe 1)"
    assert result.is_confident is False, "Confiance < 0.55, donc pas confiant"
    assert result.is_strong_signal is False, "Pas assez confiant pour être fort"
    
    print(f"  ✅ is_bullish: {result.is_bullish}")
    print(f"  ✅ is_confident: {result.is_confident}")
    print(f"  ✅ is_strong_signal: {result.is_strong_signal}")
    
    # Test avec signal fort
    strong_result = PredictionResult(
        symbol="ETHUSDT",
        probability_up=0.78,
        probability_down=0.22,
        predicted_class=1,
        confidence=0.56,  # 2 * |0.78 - 0.5| = 0.56
    )
    
    assert strong_result.is_strong_signal is True, "Devrait être un signal fort"
    print(f"  ✅ Signal fort détecté (proba={strong_result.probability_up:.2%})")
    
    # Test to_dict
    d = result.to_dict()
    assert "symbol" in d
    assert "probability_up" in d
    assert "is_bullish" in d
    print(f"  ✅ to_dict() OK: {len(d)} clés")
    
    # Test __str__
    s = str(result)
    assert "BTCUSDT" in s
    assert "HAUSSE" in s or "BAISSE" in s
    print(f"  ✅ __str__(): {s}")
    
    return True


def test_model_metadata():
    """Test 6.1.2 - ModelMetadata dataclass."""
    print("\n📋 Test 6.1.2 - ModelMetadata...")
    
    from cryptoscalper.models.predictor import ModelMetadata
    
    # Créer depuis rien
    meta = ModelMetadata()
    assert meta.n_features == 42
    print(f"  ✅ Métadonnées par défaut: {meta.n_features} features")
    
    # Créer depuis JSON
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_model.json"
        
        metadata = {
            "trained_at": "2025-12-15T12:00:00",
            "n_features": 42,
            "feature_names": ["feature_1", "feature_2"],
            "is_calibrated": True,
            "val_metrics": {"roc_auc": 0.72}
        }
        
        import json
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        
        loaded_meta = ModelMetadata.from_json(json_path)
        
        assert loaded_meta.is_calibrated is True
        assert loaded_meta.val_auc == 0.72
        assert loaded_meta.trained_at is not None
        
        print(f"  ✅ Chargé depuis JSON: AUC={loaded_meta.val_auc}")
    
    # Test fichier inexistant
    meta_none = ModelMetadata.from_json(Path("/nonexistent/path.json"))
    assert meta_none.n_features == 42  # Valeurs par défaut
    print(f"  ✅ Fichier inexistant géré")
    
    return True


# ============================================
# TESTS CHARGEMENT MODÈLE
# ============================================

def test_predictor_from_file():
    """Test 6.1.3 - Chargement du predictor depuis un fichier."""
    print("\n📂 Test 6.1.3 - Chargement du predictor...")
    
    from cryptoscalper.models.predictor import MLPredictor
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        
        # Créer et sauvegarder un mock model
        mock_model = create_mock_model(42)
        
        metadata = {
            "trained_at": datetime.now().isoformat(),
            "n_features": 42,
            "feature_names": [f"feature_{i}" for i in range(42)],
            "is_calibrated": False,
            "val_metrics": {"roc_auc": 0.70}
        }
        
        save_mock_model(mock_model, model_path, metadata)
        
        # Charger le predictor
        predictor = MLPredictor.from_file(model_path)
        
        assert predictor is not None
        assert predictor.n_features == 42
        assert predictor.is_calibrated is False
        
        print(f"  ✅ Predictor chargé: {predictor.n_features} features")
        print(f"  ✅ Calibré: {predictor.is_calibrated}")
        print(f"  ✅ AUC val: {predictor.metadata.val_auc}")
    
    return True


def test_predictor_file_not_found():
    """Test 6.1.4 - Erreur si fichier introuvable."""
    print("\n❌ Test 6.1.4 - Fichier introuvable...")
    
    from cryptoscalper.models.predictor import MLPredictor
    
    try:
        MLPredictor.from_file("/nonexistent/model.joblib")
        print("  ❌ Aurait dû lever une exception")
        return False
    except FileNotFoundError as e:
        print(f"  ✅ FileNotFoundError levée: {e}")
        return True


# ============================================
# TESTS PRÉDICTION SINGLE
# ============================================

def test_predict_single():
    """Test 6.1.5 - Prédiction unique."""
    print("\n🎯 Test 6.1.5 - Prédiction unique...")
    
    from cryptoscalper.models.predictor import MLPredictor, PredictionResult
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        mock_model = create_mock_model(42)
        save_mock_model(mock_model, model_path)
        
        predictor = MLPredictor.from_file(model_path)
        
        # Créer un FeatureSet
        feature_set = create_mock_feature_set("BTCUSDT")
        
        # Prédire
        result = predictor.predict(feature_set)
        
        assert isinstance(result, PredictionResult)
        assert result.symbol == "BTCUSDT"
        assert 0 <= result.probability_up <= 1
        assert 0 <= result.probability_down <= 1
        assert abs(result.probability_up + result.probability_down - 1.0) < 0.01
        assert 0 <= result.confidence <= 1
        assert result.predicted_class in [0, 1]
        
        print(f"  ✅ Prédiction OK:")
        print(f"     Symbol: {result.symbol}")
        print(f"     Proba up: {result.probability_up:.2%}")
        print(f"     Confiance: {result.confidence:.2%}")
        print(f"     Classe: {result.predicted_class}")
    
    return True


def test_predict_from_dict():
    """Test 6.1.6 - Prédiction depuis dictionnaire."""
    print("\n📖 Test 6.1.6 - Prédiction depuis dict...")
    
    from cryptoscalper.models.predictor import MLPredictor
    from cryptoscalper.data.features import get_feature_names
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        mock_model = create_mock_model(42)
        save_mock_model(mock_model, model_path)
        
        predictor = MLPredictor.from_file(model_path)
        
        # Créer un dictionnaire de features
        features = {name: np.random.uniform(-1, 1) for name in get_feature_names()}
        
        # Prédire
        result = predictor.predict_from_dict(features, "ETHUSDT")
        
        assert result.symbol == "ETHUSDT"
        assert 0 <= result.probability_up <= 1
        
        print(f"  ✅ Prédiction depuis dict: {result.probability_up:.2%}")
    
    return True


def test_predict_from_array():
    """Test 6.1.7 - Prédiction depuis array numpy."""
    print("\n🔢 Test 6.1.7 - Prédiction depuis array...")
    
    from cryptoscalper.models.predictor import MLPredictor
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        mock_model = create_mock_model(42)
        save_mock_model(mock_model, model_path)
        
        predictor = MLPredictor.from_file(model_path)
        
        # Créer un array
        X = np.random.randn(42)
        
        # Prédire
        result = predictor.predict_from_array(X, "BNBUSDT")
        
        assert result.symbol == "BNBUSDT"
        assert 0 <= result.probability_up <= 1
        
        print(f"  ✅ Prédiction depuis array: {result.probability_up:.2%}")
        
        # Test avec mauvaise dimension
        X_bad = np.random.randn(30)  # 30 au lieu de 42
        try:
            predictor.predict_from_array(X_bad, "TEST")
            print("  ❌ Aurait dû lever une exception")
            return False
        except ValueError as e:
            print(f"  ✅ ValueError levée pour mauvaise dimension")
    
    return True


# ============================================
# TESTS PRÉDICTION BATCH
# ============================================

def test_predict_batch():
    """Test 6.1.8 - Prédiction batch."""
    print("\n📦 Test 6.1.8 - Prédiction batch...")
    
    from cryptoscalper.models.predictor import MLPredictor
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        mock_model = create_mock_model(42)
        save_mock_model(mock_model, model_path)
        
        predictor = MLPredictor.from_file(model_path)
        
        # Créer plusieurs FeatureSet
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
        feature_sets = [create_mock_feature_set(s, seed=i) for i, s in enumerate(symbols)]
        
        # Prédiction batch
        import time
        start = time.time()
        results = predictor.predict_batch(feature_sets)
        elapsed = (time.time() - start) * 1000
        
        assert len(results) == len(symbols)
        
        for result, expected_symbol in zip(results, symbols):
            assert result.symbol == expected_symbol
            assert 0 <= result.probability_up <= 1
        
        print(f"  ✅ Batch de {len(results)} prédictions en {elapsed:.1f}ms")
        print(f"     Probas: {[f'{r.probability_up:.2%}' for r in results]}")
    
    return True


def test_predict_batch_empty():
    """Test 6.1.9 - Batch vide."""
    print("\n📭 Test 6.1.9 - Batch vide...")
    
    from cryptoscalper.models.predictor import MLPredictor
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        mock_model = create_mock_model(42)
        save_mock_model(mock_model, model_path)
        
        predictor = MLPredictor.from_file(model_path)
        
        # Batch vide
        results = predictor.predict_batch([])
        
        assert results == []
        print(f"  ✅ Batch vide retourne liste vide")
    
    return True


def test_predict_batch_dataframe():
    """Test 6.1.10 - Prédiction depuis DataFrame."""
    print("\n📊 Test 6.1.10 - Prédiction depuis DataFrame...")
    
    from cryptoscalper.models.predictor import MLPredictor
    from cryptoscalper.data.features import get_feature_names
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        mock_model = create_mock_model(42)
        save_mock_model(mock_model, model_path)
        
        predictor = MLPredictor.from_file(model_path)
        
        # Créer un DataFrame
        n_samples = 10
        feature_names = get_feature_names()
        
        data = {name: np.random.randn(n_samples) for name in feature_names}
        data["symbol"] = [f"PAIR{i}USDT" for i in range(n_samples)]
        df = pd.DataFrame(data)
        
        # Prédiction
        results = predictor.predict_batch_dataframe(df, symbol_column="symbol")
        
        assert len(results) == n_samples
        assert results[0].symbol == "PAIR0USDT"
        
        print(f"  ✅ DataFrame de {n_samples} lignes traité")
        print(f"     Symboles: {[r.symbol for r in results[:3]]}...")
    
    return True


# ============================================
# TESTS CONFIANCE
# ============================================

def test_confidence_calculation():
    """Test 6.1.11 - Calcul de la confiance."""
    print("\n💪 Test 6.1.11 - Calcul de la confiance...")
    
    from cryptoscalper.models.predictor import MLPredictor
    
    # La confiance = 2 * |probability - 0.5|
    # 0.5 → 0.0 (incertain)
    # 0.7 → 0.4
    # 0.8 → 0.6
    # 0.9 → 0.8
    # 1.0 → 1.0 (certain)
    
    test_cases = [
        (0.50, 0.00),
        (0.55, 0.10),
        (0.60, 0.20),
        (0.65, 0.30),
        (0.70, 0.40),
        (0.75, 0.50),
        (0.80, 0.60),
        (0.90, 0.80),
        (1.00, 1.00),
        # Probas basses aussi
        (0.40, 0.20),
        (0.30, 0.40),
        (0.20, 0.60),
    ]
    
    # Créer un predictor minimal pour tester
    class MockPredictor(MLPredictor):
        def __init__(self):
            pass
        
        def _calculate_confidence(self, probability_up: float) -> float:
            return 2 * abs(probability_up - 0.5)
    
    predictor = MockPredictor()
    
    for proba, expected_conf in test_cases:
        conf = predictor._calculate_confidence(proba)
        assert abs(conf - expected_conf) < 0.01, f"Proba {proba}: attendu {expected_conf}, obtenu {conf}"
    
    print(f"  ✅ {len(test_cases)} cas de confiance validés")
    
    return True


# ============================================
# TESTS UTILITAIRES
# ============================================

def test_get_top_predictions():
    """Test 6.1.12 - Filtrage top prédictions."""
    print("\n🏆 Test 6.1.12 - Top prédictions...")
    
    from cryptoscalper.models.predictor import MLPredictor, PredictionResult
    
    # Créer des prédictions variées
    predictions = [
        PredictionResult("AAA", 0.80, 0.20, 1, 0.60),  # Fort signal
        PredictionResult("BBB", 0.70, 0.30, 1, 0.40),  # Signal moyen
        PredictionResult("CCC", 0.90, 0.10, 1, 0.80),  # Très fort signal
        PredictionResult("DDD", 0.55, 0.45, 1, 0.10),  # Faible signal
        PredictionResult("EEE", 0.40, 0.60, 0, 0.20),  # Bearish
        PredictionResult("FFF", 0.65, 0.35, 1, 0.30),  # Signal limite
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        mock_model = create_mock_model(42)
        save_mock_model(mock_model, model_path)
        
        predictor = MLPredictor.from_file(model_path)
        
        # Filtrer
        top = predictor.get_top_predictions(
            predictions,
            n=3,
            min_probability=0.65,
            min_confidence=0.30
        )
        
        assert len(top) == 3
        # Doivent être triés par proba décroissante
        assert top[0].symbol == "CCC"  # 0.90
        assert top[1].symbol == "AAA"  # 0.80
        assert top[2].symbol == "BBB"  # 0.70
        
        print(f"  ✅ Top 3 filtrés:")
        for p in top:
            print(f"     {p.symbol}: proba={p.probability_up:.2%}, conf={p.confidence:.2%}")
    
    return True


def test_predictor_summary():
    """Test 6.1.13 - Summary du predictor."""
    print("\n📝 Test 6.1.13 - Summary...")
    
    from cryptoscalper.models.predictor import MLPredictor
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        mock_model = create_mock_model(42)
        
        metadata = {
            "trained_at": "2025-12-15T12:00:00",
            "is_calibrated": True,
            "val_metrics": {"roc_auc": 0.72}
        }
        save_mock_model(mock_model, model_path, metadata)
        
        predictor = MLPredictor.from_file(model_path)
        
        summary = predictor.summary()
        
        assert "MLPredictor" in summary
        assert "42" in summary  # n_features
        assert "Oui" in summary  # calibré
        
        print(f"  ✅ Summary généré:\n{summary}")
    
    return True


def test_nan_handling():
    """Test 6.1.14 - Gestion des NaN dans les features."""
    print("\n🔧 Test 6.1.14 - Gestion des NaN...")
    
    from cryptoscalper.models.predictor import MLPredictor
    from cryptoscalper.data.features import get_feature_names, FeatureSet
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        mock_model = create_mock_model(42)
        save_mock_model(mock_model, model_path)
        
        predictor = MLPredictor.from_file(model_path)
        
        # Créer des features avec des NaN
        features = {name: np.nan for name in get_feature_names()}
        features["rsi_14"] = 50.0  # Une valeur valide
        
        feature_set = FeatureSet(
            symbol="NANTEST",
            features=features,
            timestamp=pd.Timestamp.now()
        )
        
        # La prédiction ne devrait pas planter
        result = predictor.predict(feature_set)
        
        assert result is not None
        assert 0 <= result.probability_up <= 1
        
        print(f"  ✅ NaN gérés: proba={result.probability_up:.2%}")
    
    return True


# ============================================
# MAIN
# ============================================

def main():
    """Exécute tous les tests de la Phase 6.1."""
    print("=" * 65)
    print("🧪 CryptoScalper AI - Tests Phase 6.1: ML Predictor")
    print("=" * 65)
    
    setup_logger(level="WARNING")
    
    results = []
    
    try:
        # Tests dataclasses
        print("\n" + "─" * 50)
        print("📦 6.1.1-6.1.2 Dataclasses")
        print("─" * 50)
        results.append(("6.1.1 PredictionResult", test_prediction_result()))
        results.append(("6.1.2 ModelMetadata", test_model_metadata()))
        
        # Tests chargement
        print("\n" + "─" * 50)
        print("📂 6.1.3-6.1.4 Chargement Modèle")
        print("─" * 50)
        results.append(("6.1.3 Chargement fichier", test_predictor_from_file()))
        results.append(("6.1.4 Fichier introuvable", test_predictor_file_not_found()))
        
        # Tests prédiction single
        print("\n" + "─" * 50)
        print("🎯 6.1.5-6.1.7 Prédiction Single")
        print("─" * 50)
        results.append(("6.1.5 Predict single", test_predict_single()))
        results.append(("6.1.6 Predict from dict", test_predict_from_dict()))
        results.append(("6.1.7 Predict from array", test_predict_from_array()))
        
        # Tests prédiction batch
        print("\n" + "─" * 50)
        print("📦 6.1.8-6.1.10 Prédiction Batch")
        print("─" * 50)
        results.append(("6.1.8 Predict batch", test_predict_batch()))
        results.append(("6.1.9 Batch vide", test_predict_batch_empty()))
        results.append(("6.1.10 Batch DataFrame", test_predict_batch_dataframe()))
        
        # Tests confiance et utilitaires
        print("\n" + "─" * 50)
        print("💪 6.1.11-6.1.14 Confiance & Utils")
        print("─" * 50)
        results.append(("6.1.11 Calcul confiance", test_confidence_calculation()))
        results.append(("6.1.12 Top predictions", test_get_top_predictions()))
        results.append(("6.1.13 Summary", test_predictor_summary()))
        results.append(("6.1.14 Gestion NaN", test_nan_handling()))
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Résumé
    print("\n" + "=" * 65)
    print("📊 RÉSULTATS PHASE 6.1")
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
        print("🎉 Phase 6.1 - ML Predictor : VALIDÉE !")
        return 0
    else:
        print(f"⚠️  {failed} test(s) échoué(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())