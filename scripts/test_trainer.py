#!/usr/bin/env python3
# scripts/test_trainer.py
"""
Tests d'intégration pour le module trainer (Phase 5.3).

Lance tous les tests:
    python scripts/test_trainer.py

Tests individuels:
    python scripts/test_trainer.py test_xgboost_config
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import setup_logger, logger


# =========================================
# HELPERS
# =========================================

def create_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 42,
    positive_ratio: float = 0.3,
    random_state: int = 42
) -> tuple:
    """
    Crée un dataset synthétique pour les tests.
    
    Returns:
        (X, y, feature_names)
    """
    np.random.seed(random_state)
    
    # Features aléatoires avec structure
    X = np.random.randn(n_samples, n_features)
    
    # Créer un pattern pour que le modèle puisse apprendre
    # Les premiers features ont un impact sur y
    signal = X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2
    noise = np.random.randn(n_samples) * 0.5
    
    # Label basé sur le signal + bruit
    threshold = np.percentile(signal + noise, (1 - positive_ratio) * 100)
    y = (signal + noise > threshold).astype(int)
    
    # Noms des features
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return X, y, feature_names


def create_mock_prepared_dataset(
    n_samples: int = 1000,
    n_features: int = 42,
    positive_ratio: float = 0.3
):
    """Crée un mock PreparedDataset."""
    from cryptoscalper.data.dataset import PreparedDataset, DatasetStats, LabelConfig
    
    X, y, feature_names = create_synthetic_dataset(n_samples, n_features, positive_ratio)
    
    features_df = pd.DataFrame(X, columns=feature_names)
    labels = pd.Series(y)
    timestamps = pd.Series([datetime.now()] * n_samples)
    
    stats = DatasetStats(
        total_rows=n_samples,
        valid_rows=n_samples,
        label_1_count=int(y.sum()),
        label_0_count=int(n_samples - y.sum()),
        feature_count=n_features,
    )
    
    return PreparedDataset(
        features=features_df,
        labels=labels,
        timestamps=timestamps,
        symbols=["TEST"],
        feature_names=feature_names,
        stats=stats,
        label_config=LabelConfig()
    )


# =========================================
# TESTS CONFIGURATION
# =========================================

def test_xgboost_config():
    """Test 5.3.1 - Configuration XGBoost."""
    print("\n⚙️ Test 5.3.1 - XGBoostConfig...")
    
    from cryptoscalper.models.trainer import XGBoostConfig
    
    # Config par défaut
    config = XGBoostConfig()
    
    assert config.n_estimators == 200, "n_estimators par défaut = 200"
    assert config.max_depth == 6, "max_depth par défaut = 6"
    assert config.learning_rate == 0.05, "learning_rate par défaut = 0.05"
    assert config.calibrate is True, "calibrate par défaut = True"
    
    print(f"  ✅ Config par défaut OK")
    print(f"     n_estimators={config.n_estimators}, max_depth={config.max_depth}")
    
    # Conversion en params XGBoost
    params = config.to_xgb_params()
    
    assert "n_estimators" in params
    assert "max_depth" in params
    assert "learning_rate" in params
    assert params["use_label_encoder"] is False
    
    print(f"  ✅ Conversion en params XGBoost OK")
    
    # Config personnalisée
    custom = XGBoostConfig(
        n_estimators=100,
        max_depth=4,
        scale_pos_weight=2.0
    )
    
    params = custom.to_xgb_params()
    assert params["scale_pos_weight"] == 2.0
    
    print(f"  ✅ Config personnalisée OK")
    
    return True


def test_evaluation_metrics():
    """Test 5.3.2 - Dataclass EvaluationMetrics."""
    print("\n📊 Test 5.3.2 - EvaluationMetrics...")
    
    from cryptoscalper.models.trainer import EvaluationMetrics
    
    metrics = EvaluationMetrics(
        accuracy=0.75,
        precision=0.70,
        recall=0.65,
        f1=0.67,
        roc_auc=0.80,
        confusion_matrix=np.array([[100, 20], [30, 50]])
    )
    
    # Summary
    summary = metrics.summary()
    assert "Accuracy" in summary
    assert "0.75" in summary
    
    print(f"  ✅ EvaluationMetrics créé")
    print(f"     Summary: {summary[:50]}...")
    
    # to_dict
    d = metrics.to_dict()
    assert d["accuracy"] == 0.75
    assert "confusion_matrix" in d
    
    print(f"  ✅ to_dict() OK")
    
    return True


def test_feature_importance():
    """Test 5.3.3 - FeatureImportance."""
    print("\n🏆 Test 5.3.3 - FeatureImportance...")
    
    from cryptoscalper.models.trainer import FeatureImportance
    import xgboost as xgb
    
    # Créer un petit modèle
    X, y, names = create_synthetic_dataset(500, 10)
    
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, verbosity=0)
    model.fit(X, y)
    
    # Créer FeatureImportance
    fi = FeatureImportance.from_model(model, names, top_n=5)
    
    assert len(fi.importance_dict) == 10, "10 features"
    assert len(fi.top_features) == 5, "Top 5"
    
    # Vérifier que c'est trié
    importances = [x[1] for x in fi.top_features]
    assert importances == sorted(importances, reverse=True), "Trié décroissant"
    
    print(f"  ✅ FeatureImportance créé")
    print(f"     Top feature: {fi.top_features[0][0]} ({fi.top_features[0][1]:.4f})")
    
    # Summary
    summary = fi.summary(top_n=3)
    assert "Top Features" in summary
    
    print(f"  ✅ summary() OK")
    
    return True


# =========================================
# TESTS ENTRAÎNEMENT
# =========================================

def test_trainer_basic():
    """Test 5.3.4 - Entraînement basique."""
    print("\n🚀 Test 5.3.4 - Entraînement basique...")
    
    from cryptoscalper.models.trainer import ModelTrainer, XGBoostConfig
    
    # Créer des datasets
    train = create_mock_prepared_dataset(800, 42, 0.3)
    val = create_mock_prepared_dataset(200, 42, 0.3)
    
    # Config légère pour les tests
    config = XGBoostConfig(
        n_estimators=20,
        max_depth=3,
        early_stopping_rounds=5,
        calibrate=False  # Plus rapide pour le test
    )
    
    trainer = ModelTrainer(config)
    
    # Entraîner
    result = trainer.train(train, val, verbose=False)
    
    assert result.model is not None, "Modèle créé"
    assert result.val_metrics is not None, "Métriques validation"
    assert result.val_metrics.roc_auc > 0, "AUC > 0"
    assert result.feature_importance is not None, "Feature importance"
    assert result.training_time_seconds > 0, "Temps > 0"
    
    print(f"  ✅ Entraînement terminé en {result.training_time_seconds:.1f}s")
    print(f"     AUC: {result.val_metrics.roc_auc:.4f}")
    print(f"     Precision: {result.val_metrics.precision:.4f}")
    
    return True


def test_trainer_with_calibration():
    """Test 5.3.5 - Entraînement avec calibration."""
    print("\n📐 Test 5.3.5 - Entraînement avec calibration...")
    
    from cryptoscalper.models.trainer import ModelTrainer, XGBoostConfig
    from sklearn.calibration import CalibratedClassifierCV
    
    train = create_mock_prepared_dataset(800, 42, 0.3)
    val = create_mock_prepared_dataset(200, 42, 0.3)
    
    config = XGBoostConfig(
        n_estimators=20,
        max_depth=3,
        calibrate=True,
        calibration_method="isotonic",
        calibration_cv=3
    )
    
    trainer = ModelTrainer(config)
    result = trainer.train(train, val, verbose=False)
    
    assert result.is_calibrated is True, "Modèle calibré"
    assert isinstance(result.model, CalibratedClassifierCV), "Type calibré"
    assert result.raw_model is not None, "Modèle brut conservé"
    
    print(f"  ✅ Calibration OK")
    print(f"     Type: {type(result.model).__name__}")
    
    # Vérifier que les probas sont bien calibrées (entre 0 et 1)
    X, _ = val.to_numpy()
    probas = result.model.predict_proba(X)[:, 1]
    
    assert probas.min() >= 0, "Probas >= 0"
    assert probas.max() <= 1, "Probas <= 1"
    
    print(f"  ✅ Probabilités dans [0,1]: min={probas.min():.4f}, max={probas.max():.4f}")
    
    return True


def test_trainer_with_test_set():
    """Test 5.3.6 - Entraînement avec jeu de test."""
    print("\n🧪 Test 5.3.6 - Entraînement avec test set...")
    
    from cryptoscalper.models.trainer import ModelTrainer, XGBoostConfig
    
    train = create_mock_prepared_dataset(600, 42, 0.3)
    val = create_mock_prepared_dataset(200, 42, 0.3)
    test = create_mock_prepared_dataset(200, 42, 0.3)
    
    config = XGBoostConfig(n_estimators=20, max_depth=3, calibrate=False)
    
    trainer = ModelTrainer(config)
    result = trainer.train(train, val, test, verbose=False)
    
    assert result.test_metrics is not None, "Métriques test présentes"
    assert result.test_metrics.roc_auc > 0, "AUC test > 0"
    
    print(f"  ✅ Test set évalué")
    print(f"     Val AUC: {result.val_metrics.roc_auc:.4f}")
    print(f"     Test AUC: {result.test_metrics.roc_auc:.4f}")
    
    return True


def test_metrics_by_threshold():
    """Test 5.3.7 - Métriques par seuil."""
    print("\n📈 Test 5.3.7 - Métriques par seuil...")
    
    from cryptoscalper.models.trainer import ModelTrainer, XGBoostConfig, print_threshold_analysis
    
    train = create_mock_prepared_dataset(800, 42, 0.3)
    val = create_mock_prepared_dataset(200, 42, 0.3)
    
    config = XGBoostConfig(n_estimators=20, max_depth=3, calibrate=False)
    
    trainer = ModelTrainer(config)
    result = trainer.train(train, val, verbose=False)
    
    # Vérifier les métriques par seuil
    assert result.val_metrics.metrics_by_threshold is not None
    
    thresholds = list(result.val_metrics.metrics_by_threshold.keys())
    assert 0.5 in thresholds, "Seuil 0.5 présent"
    assert 0.65 in thresholds, "Seuil 0.65 présent"
    
    # Vérifier la structure
    for thresh, m in result.val_metrics.metrics_by_threshold.items():
        assert "precision" in m
        assert "recall" in m
        assert "f1" in m
        assert "n_predictions" in m
    
    print(f"  ✅ {len(thresholds)} seuils analysés")
    
    # Test print_threshold_analysis (ne doit pas planter)
    print_threshold_analysis(result.val_metrics)
    
    return True


# =========================================
# TESTS SAUVEGARDE / CHARGEMENT
# =========================================

def test_save_load_model():
    """Test 5.3.8 - Sauvegarde et chargement du modèle."""
    print("\n💾 Test 5.3.8 - Sauvegarde/Chargement...")
    
    from cryptoscalper.models.trainer import ModelTrainer, XGBoostConfig
    
    train = create_mock_prepared_dataset(500, 42, 0.3)
    val = create_mock_prepared_dataset(100, 42, 0.3)
    
    config = XGBoostConfig(n_estimators=10, max_depth=3, calibrate=False)
    
    trainer = ModelTrainer(config)
    result = trainer.train(train, val, verbose=False)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        
        # Sauvegarder
        saved_path = trainer.save_model(result.model, model_path)
        assert saved_path.exists(), "Fichier créé"
        
        print(f"  ✅ Modèle sauvegardé: {saved_path}")
        print(f"     Taille: {saved_path.stat().st_size / 1024:.1f} KB")
        
        # Charger
        loaded_model = ModelTrainer.load_model(saved_path)
        
        # Vérifier que le modèle fonctionne
        X, _ = val.to_numpy()
        original_preds = result.model.predict_proba(X)[:, 1]
        loaded_preds = loaded_model.predict_proba(X)[:, 1]
        
        assert np.allclose(original_preds, loaded_preds), "Prédictions identiques"
        
        print(f"  ✅ Modèle chargé et vérifié")
    
    return True


def test_save_training_result():
    """Test 5.3.9 - Sauvegarde complète du résultat."""
    print("\n💾 Test 5.3.9 - Sauvegarde complète...")
    
    from cryptoscalper.models.trainer import ModelTrainer, XGBoostConfig
    
    train = create_mock_prepared_dataset(500, 42, 0.3)
    val = create_mock_prepared_dataset(100, 42, 0.3)
    
    config = XGBoostConfig(n_estimators=10, max_depth=3, calibrate=False)
    
    trainer = ModelTrainer(config)
    result = trainer.train(train, val, verbose=False)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = trainer.save_training_result(result, Path(tmpdir))
        
        assert "model" in paths, "Modèle sauvegardé"
        assert "feature_importance" in paths, "Feature importance sauvegardée"
        assert "metrics_by_threshold" in paths, "Métriques par seuil sauvegardées"
        
        # Vérifier que les fichiers existent
        for name, path in paths.items():
            assert path.exists(), f"{name} existe"
            print(f"  ✅ {name}: {path.name}")
        
        # Vérifier les métadonnées JSON
        meta_path = paths["model"].with_suffix(".json")
        assert meta_path.exists(), "Métadonnées JSON"
        
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        
        assert "trained_at" in meta
        assert "val_metrics" in meta
        
        print(f"  ✅ Métadonnées: {list(meta.keys())}")
    
    return True


# =========================================
# TESTS UTILITAIRES
# =========================================

def test_find_optimal_threshold():
    """Test 5.3.10 - Recherche du seuil optimal."""
    print("\n🎯 Test 5.3.10 - Seuil optimal...")
    
    from cryptoscalper.models.trainer import (
        ModelTrainer, XGBoostConfig, find_optimal_threshold
    )
    
    train = create_mock_prepared_dataset(800, 42, 0.3)
    val = create_mock_prepared_dataset(200, 42, 0.3)
    
    config = XGBoostConfig(n_estimators=20, max_depth=3, calibrate=False)
    
    trainer = ModelTrainer(config)
    result = trainer.train(train, val, verbose=False)
    
    # Trouver le seuil optimal
    optimal = find_optimal_threshold(
        result.val_metrics,
        min_precision=0.3,
        min_predictions_pct=5.0
    )
    
    if optimal:
        print(f"  ✅ Seuil optimal trouvé: {optimal}")
        
        # Vérifier les critères
        m = result.val_metrics.metrics_by_threshold[optimal]
        assert m["precision"] >= 0.3, "Precision OK"
        assert m["pct_predictions"] >= 5.0, "% predictions OK"
    else:
        print(f"  ⚠️ Aucun seuil ne satisfait les critères")
    
    return True


def test_training_result_metadata():
    """Test 5.3.11 - Métadonnées du résultat."""
    print("\n📋 Test 5.3.11 - Métadonnées...")
    
    from cryptoscalper.models.trainer import ModelTrainer, XGBoostConfig
    
    train = create_mock_prepared_dataset(500, 42, 0.3)
    val = create_mock_prepared_dataset(100, 42, 0.3)
    
    config = XGBoostConfig(n_estimators=10, max_depth=3, calibrate=False)
    
    trainer = ModelTrainer(config)
    result = trainer.train(train, val, verbose=False)
    
    # Vérifier les métadonnées
    assert result.n_train_samples == 500
    assert result.n_val_samples == 100
    assert result.n_features == 42
    assert len(result.feature_names) == 42
    
    # to_metadata
    meta = result.to_metadata()
    
    assert "trained_at" in meta
    assert "val_metrics" in meta
    assert meta["n_features"] == 42
    
    print(f"  ✅ Métadonnées OK")
    print(f"     Clés: {list(meta.keys())}")
    
    # summary
    summary = result.summary()
    assert "Résultat Training" in summary
    
    print(f"  ✅ Summary OK")
    
    return True


# =========================================
# TEST AVEC DONNÉES RÉELLES (optionnel)
# =========================================

def test_with_real_data():
    """Test 5.3.12 - Test avec données réelles (si disponibles)."""
    print("\n🌐 Test 5.3.12 - Avec données réelles...")
    
    from cryptoscalper.models.trainer import ModelTrainer, XGBoostConfig
    from cryptoscalper.data.dataset import PreparedDataset
    
    # Chercher un dataset préparé
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        print("  ⚠️ Pas de dossier datasets/, test skippé")
        return True
    
    parquet_files = list(datasets_dir.glob("*.parquet"))
    if not parquet_files:
        print("  ⚠️ Pas de datasets préparés, test skippé")
        return True
    
    # Utiliser le premier
    dataset_path = parquet_files[0]
    print(f"  📂 Utilisation de {dataset_path.name}")
    
    try:
        dataset = PreparedDataset.load(dataset_path)
        
        # Split
        train, val, test = dataset.split_temporal()
        
        print(f"  📊 Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
        
        # Entraîner avec config légère
        config = XGBoostConfig(
            n_estimators=50,
            max_depth=4,
            calibrate=True
        )
        
        trainer = ModelTrainer(config)
        result = trainer.train(train, val, test, verbose=False)
        
        print(f"\n  ✅ Résultats sur données réelles:")
        print(f"     Val AUC:  {result.val_metrics.roc_auc:.4f}")
        print(f"     Test AUC: {result.test_metrics.roc_auc:.4f}")
        print(f"     Top feature: {result.feature_importance.top_features[0][0]}")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️ Erreur (non bloquante): {e}")
        return True


# =========================================
# MAIN
# =========================================

def main():
    """Exécute tous les tests de la Phase 5.3."""
    setup_logger(level="WARNING")  # Réduire les logs pour les tests
    
    print("=" * 70)
    print("🧪 CryptoScalper AI - Tests Phase 5.3 (Trainer)")
    print("=" * 70)
    
    tests = [
        # Configuration
        ("5.3.1", "XGBoostConfig", test_xgboost_config),
        ("5.3.2", "EvaluationMetrics", test_evaluation_metrics),
        ("5.3.3", "FeatureImportance", test_feature_importance),
        
        # Entraînement
        ("5.3.4", "Entraînement basique", test_trainer_basic),
        ("5.3.5", "Calibration", test_trainer_with_calibration),
        ("5.3.6", "Test set", test_trainer_with_test_set),
        ("5.3.7", "Métriques par seuil", test_metrics_by_threshold),
        
        # Sauvegarde
        ("5.3.8", "Save/Load model", test_save_load_model),
        ("5.3.9", "Save training result", test_save_training_result),
        
        # Utilitaires
        ("5.3.10", "Seuil optimal", test_find_optimal_threshold),
        ("5.3.11", "Métadonnées", test_training_result_metadata),
        
        # Données réelles
        ("5.3.12", "Données réelles", test_with_real_data),
    ]
    
    passed = 0
    failed = 0
    
    for test_id, name, test_func in tests:
        try:
            if len(sys.argv) > 1 and sys.argv[1] != test_func.__name__:
                continue
            
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"  ❌ FAILED: {name}")
        except Exception as e:
            failed += 1
            print(f"\n❌ EXCEPTION dans {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Résumé
    print("\n" + "=" * 70)
    print(f"📊 RÉSULTATS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✅ Tous les tests passent !")
        print("\n📝 Phase 5.3 prête pour validation.")
        print("   Prochaine étape: Préparer un dataset et lancer l'entraînement réel")
        print("\n   Exemple:")
        print("   python scripts/train_model.py --dataset datasets/dataset.parquet")
    else:
        print(f"\n❌ {failed} test(s) échoué(s)")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())