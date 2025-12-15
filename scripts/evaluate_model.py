#!/usr/bin/env python3
# scripts/evaluate_model.py
"""
Script d'évaluation d'un modèle ML.

Génère:
- Courbe ROC
- Courbe Precision-Recall
- Feature importance plot
- Courbe de calibration
- Distribution des probabilités
- Analyse par seuil
- Rapport HTML complet

Usage:
    # Évaluer sur un dataset de test
    python scripts/evaluate_model.py \\
        --model models/saved/xgb_model_latest.joblib \\
        --dataset datasets/test_dataset.parquet \\
        --output reports/evaluation/
    
    # Avec options
    python scripts/evaluate_model.py \\
        --model models/saved/xgb_model_latest.joblib \\
        --dataset datasets/test_dataset.parquet \\
        --output reports/evaluation/ \\
        --name "XGBoost v1.0" \\
        --no-html
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import setup_logger, logger


def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Évalue un modèle ML et génère des rapports de visualisation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    # Évaluation basique
    python scripts/evaluate_model.py --model model.joblib --dataset test.parquet
    
    # Avec dossier de sortie personnalisé
    python scripts/evaluate_model.py --model model.joblib --dataset test.parquet --output reports/
    
    # Sans génération HTML
    python scripts/evaluate_model.py --model model.joblib --dataset test.parquet --no-html
        """
    )
    
    # Arguments obligatoires
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Chemin du modèle (.joblib)"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Chemin du dataset de test (.parquet)"
    )
    
    # Arguments optionnels
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Dossier de sortie (défaut: reports/evaluation_YYYYMMDD_HHMMSS/)"
    )
    
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="XGBoost Classifier",
        help="Nom du modèle pour le rapport (défaut: 'XGBoost Classifier')"
    )
    
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Ne pas générer le rapport HTML"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Ne pas générer les graphiques"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Seuil pour les métriques principales (défaut: 0.5)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbeux"
    )
    
    return parser.parse_args()


def print_metrics_summary(evaluator, threshold: float = 0.5) -> None:
    """Affiche un résumé des métriques dans la console."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix
    )
    
    y_true = evaluator.y_true
    y_proba = evaluator.y_proba
    y_pred = (y_proba >= threshold).astype(int)
    
    # Métriques
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = evaluator.compute_pr_auc()
    brier = evaluator.compute_brier_score()
    
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DE L'ÉVALUATION")
    print("=" * 70)
    
    print(f"\n📈 MÉTRIQUES PRINCIPALES (seuil = {threshold}):")
    print("-" * 50)
    print(f"   • ROC-AUC:        {roc_auc:.4f}")
    print(f"   • PR-AUC:         {pr_auc:.4f}")
    print(f"   • Brier Score:    {brier:.4f}")
    print(f"   • Accuracy:       {accuracy:.4f}")
    print(f"   • Precision:      {precision:.4f}")
    print(f"   • Recall:         {recall:.4f}")
    print(f"   • F1-Score:       {f1:.4f}")
    
    print(f"\n📋 MATRICE DE CONFUSION:")
    print("-" * 50)
    print(f"                     Prédit 0    Prédit 1")
    print(f"   Vrai 0 (Neg):     {cm[0][0]:8,}    {cm[0][1]:8,}")
    print(f"   Vrai 1 (Pos):     {cm[1][0]:8,}    {cm[1][1]:8,}")
    
    n_samples = len(y_true)
    n_pos = int(np.sum(y_true))
    print(f"\n   Total: {n_samples:,} échantillons")
    print(f"   Positifs: {n_pos:,} ({n_pos/n_samples*100:.1f}%)")
    print(f"   Négatifs: {n_samples - n_pos:,} ({(n_samples-n_pos)/n_samples*100:.1f}%)")
    
    # Seuil optimal
    optimal_thresh, optimal_metrics = evaluator.find_optimal_threshold()
    print(f"\n💡 SEUIL OPTIMAL (basé sur F1):")
    print("-" * 50)
    print(f"   • Seuil:     {optimal_thresh:.2f}")
    print(f"   • Precision: {optimal_metrics['precision']:.4f}")
    print(f"   • Recall:    {optimal_metrics['recall']:.4f}")
    print(f"   • F1:        {optimal_metrics['f1']:.4f}")
    print(f"   • Prédictions: {optimal_metrics['n_predictions']:,}")
    
    # Analyse par seuil
    print(f"\n📉 ANALYSE PAR SEUIL:")
    print("-" * 50)
    print(f"   {'Seuil':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'N Preds':<10}")
    
    for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        y_p = (y_proba >= thresh).astype(int)
        tp = np.sum((y_p == 1) & (y_true == 1))
        fp = np.sum((y_p == 1) & (y_true == 0))
        fn = np.sum((y_p == 0) & (y_true == 1))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_t = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        n_pred = int(np.sum(y_p))
        
        marker = " ← optimal" if abs(thresh - optimal_thresh) < 0.01 else ""
        print(f"   {thresh:<10.2f} {prec:<12.4f} {rec:<12.4f} {f1_t:<12.4f} {n_pred:<10,}{marker}")
    
    print("\n" + "=" * 70)


def main() -> int:
    """Point d'entrée principal."""
    args = parse_args()
    
    # Configuration du logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(level=log_level)
    
    # Vérifier les fichiers
    model_path = Path(args.model)
    dataset_path = Path(args.dataset)
    
    if not model_path.exists():
        logger.error(f"❌ Modèle non trouvé: {model_path}")
        return 1
    
    if not dataset_path.exists():
        logger.error(f"❌ Dataset non trouvé: {dataset_path}")
        return 1
    
    # Dossier de sortie
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"reports/evaluation_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🔬 CryptoScalper AI - Évaluation du Modèle")
    print("=" * 70)
    print(f"\n📂 Modèle:  {model_path}")
    print(f"📂 Dataset: {dataset_path}")
    print(f"📁 Sortie:  {output_dir}")
    print(f"📊 Nom:     {args.name}")
    
    try:
        import joblib
        from cryptoscalper.data.dataset import PreparedDataset
        from cryptoscalper.models.evaluator import ModelEvaluator
        
        # Charger le modèle
        logger.info(f"📂 Chargement du modèle...")
        model = joblib.load(model_path)
        
        # Charger le dataset
        logger.info(f"📂 Chargement du dataset...")
        dataset = PreparedDataset.load(dataset_path)
        logger.info(f"   {len(dataset):,} échantillons, {len(dataset.feature_names)} features")
        
        # Créer l'évaluateur
        evaluator = ModelEvaluator(model, dataset)
        
        # Afficher le résumé console
        print_metrics_summary(evaluator, args.threshold)
        
        # Générer les graphiques si demandé
        if not args.no_plots:
            logger.info(f"\n📊 Génération des graphiques...")
            plots = evaluator.generate_all_plots(output_dir)
            
            print(f"\n📊 GRAPHIQUES GÉNÉRÉS:")
            print("-" * 50)
            for name, path in plots.items():
                print(f"   ✅ {name}: {path}")
        
        # Générer le rapport HTML si demandé
        if not args.no_html:
            logger.info(f"\n📄 Génération du rapport HTML...")
            html_path = output_dir / "evaluation_report.html"
            
            # Charger les métadonnées du modèle si disponibles
            metadata_path = model_path.with_suffix('.json')
            training_result = None
            
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                logger.info(f"   Métadonnées chargées: {metadata_path}")
            
            evaluator.generate_html_report(
                html_path,
                plots=plots if not args.no_plots else None,
                model_name=args.name
            )
            
            print(f"\n📄 RAPPORT HTML:")
            print("-" * 50)
            print(f"   ✅ {html_path}")
            print(f"\n   Ouvrir dans le navigateur:")
            print(f"   open {html_path}")
        
        # Sauvegarder les métriques en JSON
        import json
        import numpy as np
        
        metrics_dict = {
            "model_path": str(model_path),
            "dataset_path": str(dataset_path),
            "evaluated_at": datetime.now().isoformat(),
            "n_samples": len(dataset),
            "n_features": len(dataset.feature_names),
            "metrics": {
                "roc_auc": float(evaluator.compute_pr_auc()),
                "pr_auc": float(evaluator.compute_pr_auc()),
                "brier_score": float(evaluator.compute_brier_score()),
            },
            "optimal_threshold": float(evaluator.find_optimal_threshold()[0]),
        }
        
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"💾 Métriques sauvegardées: {metrics_path}")
        
        print("\n" + "=" * 70)
        print("✅ ÉVALUATION TERMINÉE")
        print("=" * 70)
        print(f"\n📁 Tous les fichiers dans: {output_dir}")
        print(f"\n   Fichiers générés:")
        for f in sorted(output_dir.iterdir()):
            size_kb = f.stat().st_size / 1024
            print(f"   • {f.name} ({size_kb:.1f} KB)")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())