#!/usr/bin/env python3
# scripts/train_model.py
"""
Script d'entraînement du modèle XGBoost.

Usage:
    # Entraîner avec un dataset préparé
    python scripts/train_model.py --dataset datasets/dataset.parquet
    
    # Entraîner avec les splits déjà faits
    python scripts/train_model.py --train datasets/train.parquet \
                                  --val datasets/val.parquet \
                                  --test datasets/test.parquet
    
    # Avec paramètres personnalisés
    python scripts/train_model.py --dataset datasets/dataset.parquet \
                                  --n-estimators 300 \
                                  --max-depth 8 \
                                  --learning-rate 0.03 \
                                  --no-calibrate
    
    # Sauvegarder dans un chemin spécifique
    python scripts/train_model.py --dataset datasets/dataset.parquet \
                                  --output models/saved/my_model.joblib
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.models.trainer import (
    ModelTrainer,
    XGBoostConfig,
    TrainingResult,
    print_threshold_analysis,
    find_optimal_threshold,
)
from cryptoscalper.data.dataset import PreparedDataset, LabelConfig
from cryptoscalper.utils.logger import setup_logger, logger


def parse_args() -> argparse.Namespace:
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Entraîne un modèle XGBoost pour CryptoScalper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Entraîner avec un dataset unique (sera splitté automatiquement)
  python scripts/train_model.py --dataset datasets/dataset.parquet
  
  # Entraîner avec des splits pré-calculés
  python scripts/train_model.py --train datasets/train.parquet \\
                                --val datasets/val.parquet \\
                                --test datasets/test.parquet
  
  # Personnaliser les hyperparamètres
  python scripts/train_model.py --dataset datasets/dataset.parquet \\
                                --n-estimators 300 \\
                                --max-depth 8 \\
                                --learning-rate 0.03
        """
    )
    
    # === Données ===
    data_group = parser.add_argument_group("Données")
    data_group.add_argument(
        "--dataset",
        type=str,
        help="Chemin vers le dataset complet (sera splitté 70/15/15)"
    )
    data_group.add_argument(
        "--train",
        type=str,
        help="Chemin vers le dataset d'entraînement"
    )
    data_group.add_argument(
        "--val",
        type=str,
        help="Chemin vers le dataset de validation"
    )
    data_group.add_argument(
        "--test",
        type=str,
        help="Chemin vers le dataset de test (optionnel)"
    )
    
    # === Hyperparamètres XGBoost ===
    xgb_group = parser.add_argument_group("Hyperparamètres XGBoost")
    xgb_group.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Nombre d'arbres (default: 200)"
    )
    xgb_group.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Profondeur max des arbres (default: 6)"
    )
    xgb_group.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Taux d'apprentissage (default: 0.05)"
    )
    xgb_group.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Ratio de sous-échantillonnage (default: 0.8)"
    )
    xgb_group.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Ratio de colonnes par arbre (default: 0.8)"
    )
    xgb_group.add_argument(
        "--scale-pos-weight",
        type=float,
        default=None,
        help="Poids pour la classe positive (auto si non spécifié)"
    )
    xgb_group.add_argument(
        "--early-stopping",
        type=int,
        default=20,
        help="Early stopping rounds (default: 20)"
    )
    
    # === Calibration ===
    cal_group = parser.add_argument_group("Calibration")
    cal_group.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Désactiver la calibration des probabilités"
    )
    cal_group.add_argument(
        "--calibration-method",
        type=str,
        default="isotonic",
        choices=["isotonic", "sigmoid"],
        help="Méthode de calibration (default: isotonic)"
    )
    cal_group.add_argument(
        "--calibration-cv",
        type=int,
        default=5,
        help="CV folds pour la calibration (default: 5)"
    )
    
    # === Sortie ===
    out_group = parser.add_argument_group("Sortie")
    out_group.add_argument(
        "--output",
        type=str,
        default="models/saved/xgb_model_latest.joblib",
        help="Chemin de sauvegarde du modèle"
    )
    out_group.add_argument(
        "--no-save",
        action="store_true",
        help="Ne pas sauvegarder le modèle"
    )
    
    # === Options ===
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mode verbose"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Mode silencieux (pas de logs XGBoost)"
    )
    
    return parser.parse_args()


def load_datasets(args: argparse.Namespace) -> tuple:
    """
    Charge les datasets selon les arguments.
    
    Returns:
        (train, val, test) - test peut être None
    """
    if args.dataset:
        # Un seul fichier → split automatique
        logger.info(f"📂 Chargement du dataset: {args.dataset}")
        dataset = PreparedDataset.load(Path(args.dataset))
        
        logger.info(f"   {len(dataset):,} lignes, {dataset.stats.feature_count} features")
        logger.info(f"   Labels: {dataset.stats.label_ratio:.1%} positifs")
        
        # Split
        logger.info("📊 Split temporel 70/15/15...")
        train, val, test = dataset.split_temporal()
        
        return train, val, test
    
    elif args.train and args.val:
        # Fichiers séparés
        logger.info(f"📂 Chargement train: {args.train}")
        train = PreparedDataset.load(Path(args.train))
        
        logger.info(f"📂 Chargement val: {args.val}")
        val = PreparedDataset.load(Path(args.val))
        
        test = None
        if args.test:
            logger.info(f"📂 Chargement test: {args.test}")
            test = PreparedDataset.load(Path(args.test))
        
        return train, val, test
    
    else:
        raise ValueError(
            "Spécifiez --dataset OU (--train et --val). "
            "Utilisez --help pour plus d'infos."
        )


def create_config(args: argparse.Namespace) -> XGBoostConfig:
    """Crée la configuration XGBoost depuis les arguments."""
    return XGBoostConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        scale_pos_weight=args.scale_pos_weight,
        early_stopping_rounds=args.early_stopping,
        calibrate=not args.no_calibrate,
        calibration_method=args.calibration_method,
        calibration_cv=args.calibration_cv,
    )


def print_summary(result: TrainingResult) -> None:
    """Affiche un résumé complet de l'entraînement."""
    print("\n" + "=" * 70)
    print("🎯 RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("=" * 70)
    
    print(f"\n⏱️  Temps d'entraînement: {result.training_time_seconds:.1f}s")
    print(f"🌳 Meilleure itération: {result.best_iteration}")
    print(f"📐 Modèle calibré: {'Oui ✅' if result.is_calibrated else 'Non'}")
    
    # Métriques validation
    print("\n📊 MÉTRIQUES VALIDATION:")
    print("-" * 40)
    print(f"   Accuracy:  {result.val_metrics.accuracy:.4f}")
    print(f"   Precision: {result.val_metrics.precision:.4f}")
    print(f"   Recall:    {result.val_metrics.recall:.4f}")
    print(f"   F1-Score:  {result.val_metrics.f1:.4f}")
    print(f"   ROC-AUC:   {result.val_metrics.roc_auc:.4f}")
    
    # Matrice de confusion
    cm = result.val_metrics.confusion_matrix
    print("\n   Matrice de confusion:")
    print(f"                 Prédit 0    Prédit 1")
    print(f"   Vrai 0:       {cm[0][0]:8.0f}    {cm[0][1]:8.0f}")
    print(f"   Vrai 1:       {cm[1][0]:8.0f}    {cm[1][1]:8.0f}")
    
    # Métriques test si disponibles
    if result.test_metrics:
        print("\n📊 MÉTRIQUES TEST:")
        print("-" * 40)
        print(f"   Accuracy:  {result.test_metrics.accuracy:.4f}")
        print(f"   Precision: {result.test_metrics.precision:.4f}")
        print(f"   Recall:    {result.test_metrics.recall:.4f}")
        print(f"   F1-Score:  {result.test_metrics.f1:.4f}")
        print(f"   ROC-AUC:   {result.test_metrics.roc_auc:.4f}")
    
    # Feature importance
    if result.feature_importance:
        print("\n🏆 TOP 15 FEATURES:")
        print("-" * 40)
        for i, (name, score) in enumerate(result.feature_importance.top_features[:15], 1):
            bar = "█" * int(score * 40)
            print(f"   {i:2d}. {name:28s} {score:.4f} {bar}")
    
    # Analyse par seuil
    print_threshold_analysis(result.val_metrics)
    
    # Seuil recommandé
    optimal = find_optimal_threshold(result.val_metrics)
    if optimal:
        print(f"\n💡 Seuil recommandé: {optimal:.2f}")
    
    print("\n" + "=" * 70)


def main() -> int:
    """Point d'entrée principal."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    if args.quiet:
        log_level = "WARNING"
    setup_logger(level=log_level)
    
    print("=" * 70)
    print("🤖 CryptoScalper AI - Entraînement Modèle XGBoost")
    print("=" * 70)
    
    try:
        # Charger les données
        train_data, val_data, test_data = load_datasets(args)
        
        print(f"\n📋 Données chargées:")
        print(f"   Train: {len(train_data):,} samples ({train_data.stats.label_ratio:.1%} positifs)")
        print(f"   Val:   {len(val_data):,} samples ({val_data.stats.label_ratio:.1%} positifs)")
        if test_data:
            print(f"   Test:  {len(test_data):,} samples ({test_data.stats.label_ratio:.1%} positifs)")
        
        # Créer la configuration
        config = create_config(args)
        
        print(f"\n⚙️  Configuration XGBoost:")
        print(f"   n_estimators:    {config.n_estimators}")
        print(f"   max_depth:       {config.max_depth}")
        print(f"   learning_rate:   {config.learning_rate}")
        print(f"   subsample:       {config.subsample}")
        print(f"   colsample:       {config.colsample_bytree}")
        print(f"   early_stopping:  {config.early_stopping_rounds}")
        print(f"   calibration:     {'Oui' if config.calibrate else 'Non'}")
        
        # Entraîner
        print("\n" + "=" * 70)
        print("🚀 ENTRAÎNEMENT EN COURS...")
        print("=" * 70 + "\n")
        
        trainer = ModelTrainer(config)
        result = trainer.train(
            train_data,
            val_data,
            test_data,
            verbose=not args.quiet
        )
        
        # Afficher le résumé
        print_summary(result)
        
        # Sauvegarder
        if not args.no_save:
            output_path = Path(args.output)
            paths = trainer.save_training_result(result, output_path.parent)
            
            print(f"\n💾 FICHIERS SAUVEGARDÉS:")
            for name, path in paths.items():
                print(f"   {name}: {path}")
        
        print("\n✅ Entraînement terminé avec succès!")
        
        # Afficher un rappel si AUC est faible
        if result.val_metrics.roc_auc < 0.55:
            print("\n⚠️  ATTENTION: AUC faible (<0.55)")
            print("   Le modèle n'est pas meilleur qu'un tirage aléatoire.")
            print("   Essayez:")
            print("   - Plus de données d'entraînement")
            print("   - Différents hyperparamètres")
            print("   - Features supplémentaires")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"❌ Fichier non trouvé: {e}")
        return 1
    except ValueError as e:
        logger.error(f"❌ Erreur de configuration: {e}")
        return 1
    except Exception as e:
        logger.exception(f"❌ Erreur inattendue: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())