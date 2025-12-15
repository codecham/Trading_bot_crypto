#!/usr/bin/env python3
# scripts/test_evaluator.py
"""
Tests d'intégration pour le module evaluator (Phase 5.4).

Vérifie:
- ModelEvaluator avec données synthétiques
- Génération de tous les graphiques
- Calcul des métriques (Brier, PR-AUC)
- Recherche du seuil optimal
- Génération du rapport HTML
- EvaluationReport

Usage:
    python scripts/test_evaluator.py
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import setup_logger, logger


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def create_synthetic_predictions(
    n_samples: int = 1000,
    positive_ratio: float = 0.3,
    model_quality: float = 0.7
) -> tuple:
    """
    Crée des prédictions synthétiques pour les tests.
    
    Args:
        n_samples: Nombre d'échantillons
        positive_ratio: Ratio de positifs
        model_quality: Qualité du modèle (0.5 = random, 1.0 = parfait)
        
    Returns:
        (y_true, y_proba)
    """
    np.random.seed(42)
    
    # Labels vrais
    n_positive = int(n_samples * positive_ratio)
    y_true = np.array([1] * n_positive + [0] * (n_samples - n_positive))
    np.random.shuffle(y_true)
    
    # Probabilités - simuler un modèle de qualité variable
    y_proba = np.zeros(n_samples)
    
    for i in range(n_samples):
        if y_true[i] == 1:
            # Positifs: distribution vers les hautes probas
            base = model_quality
            y_proba[i] = np.clip(
                base + np.random.normal(0, 0.2),
                0.01, 0.99
            )
        else:
            # Négatifs: distribution vers les basses probas
            base = 1 - model_quality
            y_proba[i] = np.clip(
                base + np.random.normal(0, 0.2),
                0.01, 0.99
            )
    
    return y_true, y_proba


def create_mock_model(y_proba: np.ndarray, n_features: int = 42):
    """
    Crée un faux modèle qui retourne les probas données.
    
    Args:
        y_proba: Probabilités à retourner
        n_features: Nombre de features pour feature_importances_
    """
    class MockModel:
        def __init__(self, probas, num_features):
            self._probas = probas
            self.feature_importances_ = np.random.rand(num_features)
            self.feature_importances_ /= self.feature_importances_.sum()
        
        def predict_proba(self, X):
            return np.column_stack([1 - self._probas, self._probas])
        
        def predict(self, X):
            return (self._probas >= 0.5).astype(int)
    
    return MockModel(y_proba, n_features)


def create_mock_dataset(n_samples: int, n_features: int = 42, positive_ratio: float = 0.3):
    """Crée un mock PreparedDataset."""
    from cryptoscalper.data.dataset import PreparedDataset, DatasetStats, LabelConfig
    
    np.random.seed(42)
    
    # Features
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    
    # Labels
    n_positive = int(n_samples * positive_ratio)
    y = pd.Series([1] * n_positive + [0] * (n_samples - n_positive))
    y = y.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Timestamps
    timestamps = pd.Series([datetime.now()] * n_samples)
    
    # Stats
    stats = DatasetStats(
        total_rows=n_samples,
        valid_rows=n_samples,
        label_1_count=n_positive,
        label_0_count=n_samples - n_positive,
        feature_count=n_features,
    )
    
    return PreparedDataset(
        features=X,
        labels=y,
        timestamps=timestamps,
        symbols=["TESTUSDT"],  # Liste, pas string
        feature_names=feature_names,
        stats=stats,
        label_config=LabelConfig()
    )


# ============================================
# TESTS
# ============================================

def test_evaluator_initialization():
    """Test 5.4.1 - Initialisation de ModelEvaluator."""
    print("\n🔧 Test 5.4.1 - Initialisation ModelEvaluator...")
    
    from cryptoscalper.models.evaluator import ModelEvaluator
    
    y_true, y_proba = create_synthetic_predictions(500)
    
    # Test avec y_true/y_proba directement
    evaluator = ModelEvaluator(
        model=None,
        y_true=y_true,
        y_proba=y_proba
    )
    
    assert len(evaluator.y_true) == 500, "500 échantillons"
    assert len(evaluator.y_proba) == 500, "500 probabilités"
    
    print(f"  ✅ Initialisation avec y_true/y_proba OK")
    
    # Test avec dataset
    dataset = create_mock_dataset(300)
    
    # Créer un mock model qui retourne des probas
    mock_proba = np.random.rand(300)
    model = create_mock_model(mock_proba, n_features=42)
    
    evaluator2 = ModelEvaluator(model, dataset)
    assert len(evaluator2.y_true) == 300, "300 échantillons"
    
    print(f"  ✅ Initialisation avec dataset OK")
    
    return True


def test_brier_score():
    """Test 5.4.2 - Calcul du Brier Score."""
    print("\n📊 Test 5.4.2 - Brier Score...")
    
    from cryptoscalper.models.evaluator import ModelEvaluator
    
    # Modèle parfait
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    
    evaluator = ModelEvaluator(None, y_true=y_true, y_proba=y_proba)
    brier = evaluator.compute_brier_score()
    
    assert brier == 0.0, f"Brier parfait = 0, got {brier}"
    print(f"  ✅ Brier parfait = {brier}")
    
    # Modèle aléatoire
    y_proba_random = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    evaluator2 = ModelEvaluator(None, y_true=y_true, y_proba=y_proba_random)
    brier2 = evaluator2.compute_brier_score()
    
    assert brier2 == 0.25, f"Brier random = 0.25, got {brier2}"
    print(f"  ✅ Brier aléatoire = {brier2}")
    
    # Modèle réaliste
    y_true, y_proba = create_synthetic_predictions(1000, model_quality=0.7)
    evaluator3 = ModelEvaluator(None, y_true=y_true, y_proba=y_proba)
    brier3 = evaluator3.compute_brier_score()
    
    assert 0 < brier3 < 0.5, f"Brier réaliste entre 0 et 0.5, got {brier3}"
    print(f"  ✅ Brier modèle réaliste = {brier3:.4f}")
    
    return True


def test_pr_auc():
    """Test 5.4.3 - Calcul du PR-AUC."""
    print("\n📈 Test 5.4.3 - PR-AUC...")
    
    from cryptoscalper.models.evaluator import ModelEvaluator
    
    # Modèle de bonne qualité
    y_true, y_proba = create_synthetic_predictions(1000, model_quality=0.8)
    evaluator = ModelEvaluator(None, y_true=y_true, y_proba=y_proba)
    
    pr_auc = evaluator.compute_pr_auc()
    
    assert 0 <= pr_auc <= 1, f"PR-AUC entre 0 et 1, got {pr_auc}"
    assert pr_auc > 0.5, f"Bon modèle devrait avoir PR-AUC > 0.5, got {pr_auc}"
    
    print(f"  ✅ PR-AUC = {pr_auc:.4f}")
    
    return True


def test_optimal_threshold():
    """Test 5.4.4 - Recherche du seuil optimal."""
    print("\n🎯 Test 5.4.4 - Seuil optimal...")
    
    from cryptoscalper.models.evaluator import ModelEvaluator
    
    y_true, y_proba = create_synthetic_predictions(1000, model_quality=0.7)
    evaluator = ModelEvaluator(None, y_true=y_true, y_proba=y_proba)
    
    # Seuil optimal pour F1
    thresh, metrics = evaluator.find_optimal_threshold(metric="f1")
    
    assert 0.3 <= thresh <= 0.85, f"Seuil dans la plage testée, got {thresh}"
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "n_predictions" in metrics
    
    print(f"  ✅ Seuil optimal (F1): {thresh:.2f}")
    print(f"     Precision: {metrics['precision']:.4f}")
    print(f"     Recall: {metrics['recall']:.4f}")
    print(f"     F1: {metrics['f1']:.4f}")
    
    # Seuil optimal pour precision
    thresh_prec, _ = evaluator.find_optimal_threshold(metric="precision")
    print(f"  ✅ Seuil optimal (Precision): {thresh_prec:.2f}")
    
    return True


def test_plot_roc_curve():
    """Test 5.4.5 - Génération courbe ROC."""
    print("\n📊 Test 5.4.5 - Courbe ROC...")
    
    from cryptoscalper.models.evaluator import ModelEvaluator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    y_true, y_proba = create_synthetic_predictions(500)
    evaluator = ModelEvaluator(None, y_true=y_true, y_proba=y_proba)
    
    fig = evaluator.plot_roc_curve()
    
    assert fig is not None, "Figure créée"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "roc.png"
        fig.savefig(path)
        plt.close(fig)
        
        assert path.exists(), "Fichier créé"
        assert path.stat().st_size > 0, "Fichier non vide"
        
        print(f"  ✅ Courbe ROC générée ({path.stat().st_size / 1024:.1f} KB)")
    
    return True


def test_plot_precision_recall():
    """Test 5.4.6 - Génération courbe PR."""
    print("\n📊 Test 5.4.6 - Courbe Precision-Recall...")
    
    from cryptoscalper.models.evaluator import ModelEvaluator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    y_true, y_proba = create_synthetic_predictions(500)
    evaluator = ModelEvaluator(None, y_true=y_true, y_proba=y_proba)
    
    fig = evaluator.plot_precision_recall_curve()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pr.png"
        fig.savefig(path)
        plt.close(fig)
        
        assert path.exists(), "Fichier créé"
        print(f"  ✅ Courbe PR générée ({path.stat().st_size / 1024:.1f} KB)")
    
    return True


def test_plot_feature_importance():
    """Test 5.4.7 - Génération feature importance plot."""
    print("\n🏆 Test 5.4.7 - Feature Importance Plot...")
    
    from cryptoscalper.models.evaluator import ModelEvaluator
    from cryptoscalper.models.trainer import FeatureImportance
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    y_true, y_proba = create_synthetic_predictions(500)
    feature_names = [f"feature_{i}" for i in range(42)]
    
    evaluator = ModelEvaluator(
        model=None,
        y_true=y_true,
        y_proba=y_proba,
        feature_names=feature_names
    )
    
    # Créer un FeatureImportance mock
    importance_dict = {f: np.random.rand() for f in feature_names}
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    fi = FeatureImportance(
        importance_dict=importance_dict,
        top_features=sorted_features[:20]
    )
    
    fig = evaluator.plot_feature_importance(fi, top_n=15)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "fi.png"
        fig.savefig(path)
        plt.close(fig)
        
        assert path.exists(), "Fichier créé"
        print(f"  ✅ Feature importance plot généré ({path.stat().st_size / 1024:.1f} KB)")
    
    return True


def test_plot_calibration():
    """Test 5.4.8 - Génération courbe de calibration."""
    print("\n📐 Test 5.4.8 - Courbe de Calibration...")
    
    from cryptoscalper.models.evaluator import ModelEvaluator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    y_true, y_proba = create_synthetic_predictions(1000)
    evaluator = ModelEvaluator(None, y_true=y_true, y_proba=y_proba)
    
    fig = evaluator.plot_calibration_curve(n_bins=10)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "calib.png"
        fig.savefig(path)
        plt.close(fig)
        
        assert path.exists(), "Fichier créé"
        print(f"  ✅ Courbe de calibration générée ({path.stat().st_size / 1024:.1f} KB)")
    
    return True


def test_plot_probability_distribution():
    """Test 5.4.9 - Génération distribution des probabilités."""
    print("\n📊 Test 5.4.9 - Distribution des Probabilités...")
    
    from cryptoscalper.models.evaluator import ModelEvaluator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    y_true, y_proba = create_synthetic_predictions(1000)
    evaluator = ModelEvaluator(None, y_true=y_true, y_proba=y_proba)
    
    fig = evaluator.plot_probability_distribution()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "proba_dist.png"
        fig.savefig(path)
        plt.close(fig)
        
        assert path.exists(), "Fichier créé"
        print(f"  ✅ Distribution générée ({path.stat().st_size / 1024:.1f} KB)")
    
    return True


def test_plot_threshold_analysis():
    """Test 5.4.10 - Génération analyse par seuil."""
    print("\n📉 Test 5.4.10 - Analyse par Seuil...")
    
    from cryptoscalper.models.evaluator import ModelEvaluator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    y_true, y_proba = create_synthetic_predictions(1000)
    evaluator = ModelEvaluator(None, y_true=y_true, y_proba=y_proba)
    
    fig = evaluator.plot_threshold_analysis()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "thresh.png"
        fig.savefig(path)
        plt.close(fig)
        
        assert path.exists(), "Fichier créé"
        print(f"  ✅ Analyse par seuil générée ({path.stat().st_size / 1024:.1f} KB)")
    
    return True


def test_generate_all_plots():
    """Test 5.4.11 - Génération de tous les graphiques."""
    print("\n📊 Test 5.4.11 - Génération de tous les graphiques...")
    
    from cryptoscalper.models.evaluator import ModelEvaluator
    import matplotlib
    matplotlib.use('Agg')
    
    y_true, y_proba = create_synthetic_predictions(500)
    feature_names = [f"feature_{i}" for i in range(42)]
    
    # Créer un mock model avec feature_importances_
    model = create_mock_model(y_proba, n_features=42)
    
    evaluator = ModelEvaluator(
        model=model,
        y_true=y_true,
        y_proba=y_proba,
        feature_names=feature_names
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        plots = evaluator.generate_all_plots(output_dir)
        
        expected_plots = [
            "roc_curve",
            "pr_curve",
            "feature_importance",
            "calibration_curve",
            "probability_distribution",
            "threshold_analysis"
        ]
        
        for name in expected_plots:
            assert name in plots, f"{name} généré"
            assert plots[name].exists(), f"{name} existe"
        
        print(f"  ✅ {len(plots)} graphiques générés:")
        for name, path in plots.items():
            print(f"     • {name}: {path.stat().st_size / 1024:.1f} KB")
    
    return True


def test_generate_html_report():
    """Test 5.4.12 - Génération rapport HTML."""
    print("\n📄 Test 5.4.12 - Rapport HTML...")
    
    from cryptoscalper.models.evaluator import ModelEvaluator
    import matplotlib
    matplotlib.use('Agg')
    
    y_true, y_proba = create_synthetic_predictions(500)
    evaluator = ModelEvaluator(None, y_true=y_true, y_proba=y_proba)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Générer les plots d'abord
        plots = evaluator.generate_all_plots(output_dir)
        
        # Générer le rapport HTML
        html_path = output_dir / "report.html"
        result_path = evaluator.generate_html_report(
            html_path,
            plots=plots,
            model_name="Test Model"
        )
        
        assert result_path.exists(), "HTML créé"
        
        content = result_path.read_text()
        assert "<html" in content, "Contenu HTML valide"
        assert "ROC-AUC" in content, "Métriques présentes"
        assert "Test Model" in content, "Nom du modèle présent"
        assert "Matrice de Confusion" in content, "Section confusion matrix"
        
        print(f"  ✅ Rapport HTML généré ({result_path.stat().st_size / 1024:.1f} KB)")
    
    return True


def test_evaluation_report_dataclass():
    """Test 5.4.13 - Dataclass EvaluationReport."""
    print("\n📋 Test 5.4.13 - EvaluationReport dataclass...")
    
    from cryptoscalper.models.evaluator import EvaluationReport
    from cryptoscalper.models.trainer import EvaluationMetrics
    
    metrics = EvaluationMetrics(
        accuracy=0.75,
        precision=0.70,
        recall=0.65,
        f1=0.67,
        roc_auc=0.80
    )
    
    report = EvaluationReport(
        metrics=metrics,
        brier_score=0.15,
        pr_auc=0.72,
        recommended_threshold=0.65
    )
    
    assert report.metrics.roc_auc == 0.80
    assert report.brier_score == 0.15
    assert report.recommended_threshold == 0.65
    
    # Test summary
    summary = report.summary()
    assert "ROC-AUC" in summary
    assert "0.8000" in summary
    assert "0.65" in summary
    
    print(f"  ✅ EvaluationReport créé")
    print(f"     Summary preview: {summary[:100]}...")
    
    return True


def test_full_evaluation_workflow():
    """Test 5.4.14 - Workflow complet d'évaluation."""
    print("\n🔄 Test 5.4.14 - Workflow complet...")
    
    from cryptoscalper.models.evaluator import ModelEvaluator, EvaluationReport
    import matplotlib
    matplotlib.use('Agg')
    
    # Simuler un cas réaliste
    y_true, y_proba = create_synthetic_predictions(
        n_samples=2000,
        positive_ratio=0.3,
        model_quality=0.72
    )
    
    feature_names = [f"feature_{i}" for i in range(42)]
    model = create_mock_model(y_proba, n_features=42)
    
    evaluator = ModelEvaluator(
        model=model,
        y_true=y_true,
        y_proba=y_proba,
        feature_names=feature_names
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Générer le rapport complet
        report = evaluator.generate_full_report(output_dir, model_name="Test XGBoost")
        
        # Vérifications
        assert isinstance(report, EvaluationReport)
        assert report.metrics.roc_auc > 0
        assert report.brier_score > 0
        assert report.pr_auc > 0
        assert len(report.plots) >= 6
        
        # Vérifier les fichiers
        assert (output_dir / "evaluation_report.html").exists()
        assert (output_dir / "roc_curve.png").exists()
        
        print(f"  ✅ Workflow complet exécuté")
        print(f"     ROC-AUC: {report.metrics.roc_auc:.4f}")
        print(f"     PR-AUC: {report.pr_auc:.4f}")
        print(f"     Brier: {report.brier_score:.4f}")
        print(f"     Seuil recommandé: {report.recommended_threshold:.2f}")
        print(f"     {len(report.plots)} fichiers générés")
    
    return True


# ============================================
# MAIN
# ============================================

def main() -> int:
    """Point d'entrée principal."""
    setup_logger(level="WARNING")
    
    print("=" * 70)
    print("🧪 CryptoScalper AI - Tests Phase 5.4 (Evaluator)")
    print("=" * 70)
    
    tests = [
        ("5.4.1", "Initialisation", test_evaluator_initialization),
        ("5.4.2", "Brier Score", test_brier_score),
        ("5.4.3", "PR-AUC", test_pr_auc),
        ("5.4.4", "Seuil optimal", test_optimal_threshold),
        ("5.4.5", "Plot ROC", test_plot_roc_curve),
        ("5.4.6", "Plot PR", test_plot_precision_recall),
        ("5.4.7", "Plot Feature Importance", test_plot_feature_importance),
        ("5.4.8", "Plot Calibration", test_plot_calibration),
        ("5.4.9", "Plot Proba Distribution", test_plot_probability_distribution),
        ("5.4.10", "Plot Threshold Analysis", test_plot_threshold_analysis),
        ("5.4.11", "Generate All Plots", test_generate_all_plots),
        ("5.4.12", "Generate HTML Report", test_generate_html_report),
        ("5.4.13", "EvaluationReport", test_evaluation_report_dataclass),
        ("5.4.14", "Workflow Complet", test_full_evaluation_workflow),
    ]
    
    passed = 0
    failed = 0
    
    for test_id, name, test_func in tests:
        try:
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
        print("\n📝 Phase 5.4 prête pour validation.")
        print("\n   Exemple d'utilisation:")
        print("   python scripts/evaluate_model.py \\")
        print("       --model models/saved/xgb_model_latest.joblib \\")
        print("       --dataset datasets/test_dataset.parquet \\")
        print("       --output reports/evaluation/")
    else:
        print(f"\n❌ {failed} test(s) échoué(s)")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())