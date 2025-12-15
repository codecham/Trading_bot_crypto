# cryptoscalper/models/evaluator.py
"""
Module d'évaluation et visualisation des modèles ML.

Responsabilités :
- Génération de courbes ROC et Precision-Recall
- Visualisation de l'importance des features
- Courbe de calibration
- Distribution des probabilités prédites
- Génération de rapports HTML

Usage:
    from cryptoscalper.models.evaluator import ModelEvaluator
    
    evaluator = ModelEvaluator(model, test_data)
    evaluator.generate_all_plots(output_dir="reports/")
    evaluator.generate_html_report("reports/evaluation_report.html")
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Imports matplotlib avec backend non-interactif pour serveur
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from cryptoscalper.data.dataset import PreparedDataset
from cryptoscalper.models.trainer import (
    EvaluationMetrics,
    FeatureImportance,
    TrainingResult,
)
from cryptoscalper.utils.logger import logger


# ============================================
# CONSTANTES
# ============================================

# Style des graphiques
PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
}

# Couleurs
COLOR_PRIMARY = "#2563eb"  # Bleu
COLOR_SECONDARY = "#dc2626"  # Rouge
COLOR_SUCCESS = "#16a34a"  # Vert
COLOR_WARNING = "#f59e0b"  # Orange
COLOR_BASELINE = "#6b7280"  # Gris


# ============================================
# DATACLASS POUR LE RAPPORT
# ============================================

@dataclass
class EvaluationReport:
    """Rapport complet d'évaluation."""
    
    # Métriques principales
    metrics: EvaluationMetrics
    
    # Feature importance
    feature_importance: Optional[FeatureImportance] = None
    
    # Chemins des graphiques générés
    plots: Dict[str, Path] = field(default_factory=dict)
    
    # Score de calibration (Brier score)
    brier_score: float = 0.0
    
    # AUC Precision-Recall
    pr_auc: float = 0.0
    
    # Seuil recommandé
    recommended_threshold: float = 0.65
    
    # Métadonnées
    created_at: datetime = field(default_factory=datetime.now)
    model_path: Optional[str] = None
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Résumé textuel du rapport."""
        lines = [
            "=" * 60,
            "📊 RAPPORT D'ÉVALUATION DU MODÈLE",
            "=" * 60,
            f"Date: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📈 MÉTRIQUES PRINCIPALES:",
            f"   • ROC-AUC:        {self.metrics.roc_auc:.4f}",
            f"   • PR-AUC:         {self.pr_auc:.4f}",
            f"   • Accuracy:       {self.metrics.accuracy:.4f}",
            f"   • Precision:      {self.metrics.precision:.4f}",
            f"   • Recall:         {self.metrics.recall:.4f}",
            f"   • F1-Score:       {self.metrics.f1:.4f}",
            f"   • Brier Score:    {self.brier_score:.4f}",
            "",
            f"💡 Seuil recommandé: {self.recommended_threshold:.2f}",
        ]
        
        if self.plots:
            lines.extend([
                "",
                "📁 GRAPHIQUES GÉNÉRÉS:",
            ])
            for name, path in self.plots.items():
                lines.append(f"   • {name}: {path}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================
# CLASSE PRINCIPALE
# ============================================

class ModelEvaluator:
    """
    Évaluateur de modèle avec génération de visualisations.
    
    Génère:
    - Courbe ROC avec AUC
    - Courbe Precision-Recall
    - Bar chart des features importantes
    - Courbe de calibration
    - Distribution des probabilités
    - Rapport HTML complet
    """
    
    def __init__(
        self,
        model: Any,
        dataset: Optional[PreparedDataset] = None,
        y_true: Optional[np.ndarray] = None,
        y_proba: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialise l'évaluateur.
        
        Args:
            model: Modèle entraîné (XGBoost ou CalibratedClassifierCV)
            dataset: Dataset de test (si y_true/y_proba non fournis)
            y_true: Labels vrais (optionnel si dataset fourni)
            y_proba: Probabilités prédites (optionnel si dataset fourni)
            feature_names: Noms des features (pour importance)
        """
        self._model = model
        self._feature_names = feature_names or []
        
        # Calculer y_true et y_proba si dataset fourni
        if dataset is not None:
            X, y = dataset.to_numpy()
            self._y_true = y
            self._y_proba = model.predict_proba(X)[:, 1]
            self._feature_names = feature_names or dataset.feature_names
        elif y_true is not None and y_proba is not None:
            self._y_true = y_true
            self._y_proba = y_proba
        else:
            raise ValueError("Fournir soit dataset, soit y_true et y_proba")
        
        # Appliquer le style matplotlib
        plt.rcParams.update(PLOT_STYLE)
        
        logger.info(f"📊 ModelEvaluator initialisé avec {len(self._y_true)} échantillons")
    
    # =========================================
    # PROPRIÉTÉS
    # =========================================
    
    @property
    def y_true(self) -> np.ndarray:
        """Labels vrais."""
        return self._y_true
    
    @property
    def y_proba(self) -> np.ndarray:
        """Probabilités prédites."""
        return self._y_proba
    
    # =========================================
    # MÉTRIQUES SUPPLÉMENTAIRES
    # =========================================
    
    def compute_brier_score(self) -> float:
        """
        Calcule le Brier score (mesure de calibration).
        
        Plus le score est bas, meilleure est la calibration.
        0 = parfait, 1 = pire.
        """
        return brier_score_loss(self._y_true, self._y_proba)
    
    def compute_pr_auc(self) -> float:
        """Calcule l'AUC de la courbe Precision-Recall."""
        return average_precision_score(self._y_true, self._y_proba)
    
    def find_optimal_threshold(
        self,
        metric: str = "f1",
        thresholds: Optional[List[float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Trouve le seuil optimal selon une métrique.
        
        Args:
            metric: Métrique à optimiser ('f1', 'precision', 'recall', 'youden')
            thresholds: Liste des seuils à tester
            
        Returns:
            (seuil_optimal, métriques_à_ce_seuil)
        """
        if thresholds is None:
            thresholds = np.arange(0.3, 0.85, 0.05).tolist()
        
        best_threshold = 0.5
        best_score = 0.0
        best_metrics: Dict[str, float] = {}
        
        for thresh in thresholds:
            y_pred = (self._y_proba >= thresh).astype(int)
            
            # Calculer les métriques
            tp = np.sum((y_pred == 1) & (self._y_true == 1))
            fp = np.sum((y_pred == 1) & (self._y_true == 0))
            fn = np.sum((y_pred == 0) & (self._y_true == 1))
            tn = np.sum((y_pred == 0) & (self._y_true == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Youden's J (sensibilité + spécificité - 1)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            youden = recall + specificity - 1
            
            metrics = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "youden": youden,
                "n_predictions": int(np.sum(y_pred)),
            }
            
            # Sélectionner selon la métrique demandée
            score = metrics.get(metric, f1)
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
                best_metrics = metrics
        
        return best_threshold, best_metrics
    
    # =========================================
    # GÉNÉRATION DES GRAPHIQUES
    # =========================================
    
    def plot_roc_curve(
        self,
        ax: Optional[plt.Axes] = None,
        show_threshold_points: bool = True
    ) -> Figure:
        """
        Génère la courbe ROC.
        
        Args:
            ax: Axes matplotlib (crée une figure si None)
            show_threshold_points: Afficher des points pour certains seuils
            
        Returns:
            Figure matplotlib
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure
        
        # Calculer la courbe ROC
        fpr, tpr, thresholds = roc_curve(self._y_true, self._y_proba)
        auc = roc_auc_score(self._y_true, self._y_proba)
        
        # Tracer la courbe
        ax.plot(
            fpr, tpr,
            color=COLOR_PRIMARY,
            linewidth=2,
            label=f"ROC (AUC = {auc:.4f})"
        )
        
        # Ligne de base (random)
        ax.plot(
            [0, 1], [0, 1],
            color=COLOR_BASELINE,
            linestyle="--",
            linewidth=1,
            label="Aléatoire (AUC = 0.5)"
        )
        
        # Points de seuils importants
        if show_threshold_points:
            threshold_points = [0.5, 0.6, 0.7, 0.8]
            for thresh in threshold_points:
                idx = np.argmin(np.abs(thresholds - thresh))
                ax.scatter(
                    fpr[idx], tpr[idx],
                    s=100, zorder=5,
                    label=f"Seuil {thresh}"
                )
                ax.annotate(
                    f"{thresh}",
                    (fpr[idx], tpr[idx]),
                    textcoords="offset points",
                    xytext=(10, -10),
                    fontsize=9
                )
        
        # Mise en forme
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Taux de Faux Positifs (FPR)")
        ax.set_ylabel("Taux de Vrais Positifs (TPR)")
        ax.set_title("Courbe ROC (Receiver Operating Characteristic)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Remplir sous la courbe
        ax.fill_between(fpr, tpr, alpha=0.1, color=COLOR_PRIMARY)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(
        self,
        ax: Optional[plt.Axes] = None
    ) -> Figure:
        """
        Génère la courbe Precision-Recall.
        
        Particulièrement utile pour les classes déséquilibrées.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure
        
        # Calculer la courbe PR
        precision, recall, thresholds = precision_recall_curve(
            self._y_true, self._y_proba
        )
        pr_auc = average_precision_score(self._y_true, self._y_proba)
        
        # Baseline (proportion de positifs)
        baseline = np.mean(self._y_true)
        
        # Tracer la courbe
        ax.plot(
            recall, precision,
            color=COLOR_PRIMARY,
            linewidth=2,
            label=f"PR Curve (AUC = {pr_auc:.4f})"
        )
        
        # Ligne de base
        ax.axhline(
            y=baseline,
            color=COLOR_BASELINE,
            linestyle="--",
            linewidth=1,
            label=f"Baseline ({baseline:.2%} positifs)"
        )
        
        # Mise en forme
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall (Sensibilité)")
        ax.set_ylabel("Precision")
        ax.set_title("Courbe Precision-Recall")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(
        self,
        feature_importance: Optional[FeatureImportance] = None,
        top_n: int = 20,
        ax: Optional[plt.Axes] = None
    ) -> Figure:
        """
        Génère un bar chart de l'importance des features.
        
        Args:
            feature_importance: FeatureImportance (calcule si None)
            top_n: Nombre de features à afficher
            ax: Axes matplotlib
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure
        
        # Calculer l'importance si non fournie
        if feature_importance is None:
            if hasattr(self._model, 'feature_importances_'):
                # XGBoost direct
                importance = self._model.feature_importances_
            elif hasattr(self._model, 'estimator'):
                # CalibratedClassifierCV
                importance = self._model.estimator.feature_importances_
            elif hasattr(self._model, 'calibrated_classifiers_'):
                # Autre forme de CalibratedClassifierCV
                importance = self._model.calibrated_classifiers_[0].estimator.feature_importances_
            else:
                logger.warning("Impossible d'extraire l'importance des features")
                ax.text(0.5, 0.5, "Feature importance non disponible",
                       ha='center', va='center', fontsize=14)
                return fig
            
            feature_importance = FeatureImportance.from_model(
                self._model.estimator if hasattr(self._model, 'estimator') else self._model,
                self._feature_names,
                top_n=top_n
            )
        
        # Préparer les données
        features = [f[0] for f in feature_importance.top_features[:top_n]]
        importances = [f[1] for f in feature_importance.top_features[:top_n]]
        
        # Inverser pour avoir le plus important en haut
        features = features[::-1]
        importances = importances[::-1]
        
        # Couleurs selon l'importance
        colors = [COLOR_PRIMARY if i > np.mean(importances) else COLOR_SECONDARY 
                  for i in importances]
        
        # Barres horizontales
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color=colors, alpha=0.8)
        
        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Features les Plus Importantes")
        ax.grid(True, axis='x', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, (imp, feat) in enumerate(zip(importances, features)):
            ax.text(imp + 0.002, i, f"{imp:.4f}", va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_calibration_curve(
        self,
        n_bins: int = 10,
        ax: Optional[plt.Axes] = None
    ) -> Figure:
        """
        Génère la courbe de calibration.
        
        Compare les probabilités prédites aux fréquences observées.
        Une bonne calibration suit la diagonale.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure
        
        # Calculer la courbe de calibration
        prob_true, prob_pred = calibration_curve(
            self._y_true, self._y_proba, n_bins=n_bins
        )
        
        # Brier score
        brier = brier_score_loss(self._y_true, self._y_proba)
        
        # Tracer la courbe
        ax.plot(
            prob_pred, prob_true,
            color=COLOR_PRIMARY,
            linewidth=2,
            marker='o',
            markersize=8,
            label=f"Modèle (Brier = {brier:.4f})"
        )
        
        # Ligne parfaite
        ax.plot(
            [0, 1], [0, 1],
            color=COLOR_BASELINE,
            linestyle="--",
            linewidth=1,
            label="Calibration Parfaite"
        )
        
        # Mise en forme
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel("Probabilité Prédite Moyenne")
        ax.set_ylabel("Fraction de Positifs")
        ax.set_title("Courbe de Calibration")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_probability_distribution(
        self,
        ax: Optional[plt.Axes] = None,
        bins: int = 50
    ) -> Figure:
        """
        Génère l'histogramme des probabilités prédites.
        
        Montre la distribution séparée pour les vrais positifs et négatifs.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        # Séparer par classe réelle
        proba_positive = self._y_proba[self._y_true == 1]
        proba_negative = self._y_proba[self._y_true == 0]
        
        # Histogrammes
        ax.hist(
            proba_negative, bins=bins, alpha=0.6,
            color=COLOR_SECONDARY, label=f"Négatifs (n={len(proba_negative)})",
            density=True
        )
        ax.hist(
            proba_positive, bins=bins, alpha=0.6,
            color=COLOR_SUCCESS, label=f"Positifs (n={len(proba_positive)})",
            density=True
        )
        
        # Lignes verticales pour les seuils
        for thresh in [0.5, 0.65, 0.75]:
            ax.axvline(x=thresh, color=COLOR_BASELINE, linestyle='--', 
                      alpha=0.7, label=f"Seuil {thresh}")
        
        ax.set_xlabel("Probabilité Prédite")
        ax.set_ylabel("Densité")
        ax.set_title("Distribution des Probabilités Prédites par Classe")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_threshold_analysis(
        self,
        ax: Optional[plt.Axes] = None
    ) -> Figure:
        """
        Génère le graphique d'analyse des métriques par seuil.
        
        Montre precision, recall et F1 en fonction du seuil.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        thresholds = np.arange(0.3, 0.85, 0.02)
        precisions = []
        recalls = []
        f1s = []
        
        for thresh in thresholds:
            y_pred = (self._y_proba >= thresh).astype(int)
            
            tp = np.sum((y_pred == 1) & (self._y_true == 1))
            fp = np.sum((y_pred == 1) & (self._y_true == 0))
            fn = np.sum((y_pred == 0) & (self._y_true == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        # Tracer les courbes
        ax.plot(thresholds, precisions, color=COLOR_PRIMARY, linewidth=2, label="Precision")
        ax.plot(thresholds, recalls, color=COLOR_SUCCESS, linewidth=2, label="Recall")
        ax.plot(thresholds, f1s, color=COLOR_WARNING, linewidth=2, label="F1-Score")
        
        # Marquer le seuil optimal (F1 max)
        best_idx = np.argmax(f1s)
        best_thresh = thresholds[best_idx]
        ax.axvline(x=best_thresh, color=COLOR_SECONDARY, linestyle='--', 
                  label=f"Seuil optimal (F1): {best_thresh:.2f}")
        
        ax.set_xlabel("Seuil de Probabilité")
        ax.set_ylabel("Score")
        ax.set_title("Métriques en Fonction du Seuil de Décision")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.3, 0.85])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        return fig
    
    # =========================================
    # GÉNÉRATION GROUPÉE
    # =========================================
    
    def generate_all_plots(
        self,
        output_dir: Path,
        feature_importance: Optional[FeatureImportance] = None,
        prefix: str = ""
    ) -> Dict[str, Path]:
        """
        Génère tous les graphiques et les sauvegarde.
        
        Args:
            output_dir: Dossier de sortie
            feature_importance: FeatureImportance (optionnel)
            prefix: Préfixe pour les noms de fichiers
            
        Returns:
            Dictionnaire {nom_plot: chemin_fichier}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        # ROC Curve
        fig = self.plot_roc_curve()
        path = output_dir / f"{prefix}roc_curve.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plots["roc_curve"] = path
        logger.info(f"📊 ROC curve: {path}")
        
        # Precision-Recall Curve
        fig = self.plot_precision_recall_curve()
        path = output_dir / f"{prefix}pr_curve.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plots["pr_curve"] = path
        logger.info(f"📊 PR curve: {path}")
        
        # Feature Importance
        fig = self.plot_feature_importance(feature_importance)
        path = output_dir / f"{prefix}feature_importance.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plots["feature_importance"] = path
        logger.info(f"📊 Feature importance: {path}")
        
        # Calibration Curve
        fig = self.plot_calibration_curve()
        path = output_dir / f"{prefix}calibration_curve.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plots["calibration_curve"] = path
        logger.info(f"📊 Calibration curve: {path}")
        
        # Probability Distribution
        fig = self.plot_probability_distribution()
        path = output_dir / f"{prefix}probability_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plots["probability_distribution"] = path
        logger.info(f"📊 Probability distribution: {path}")
        
        # Threshold Analysis
        fig = self.plot_threshold_analysis()
        path = output_dir / f"{prefix}threshold_analysis.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plots["threshold_analysis"] = path
        logger.info(f"📊 Threshold analysis: {path}")
        
        logger.info(f"✅ {len(plots)} graphiques générés dans {output_dir}")
        return plots
    
    # =========================================
    # RAPPORT HTML
    # =========================================
    
    def generate_html_report(
        self,
        output_path: Path,
        training_result: Optional[TrainingResult] = None,
        plots: Optional[Dict[str, Path]] = None,
        model_name: str = "XGBoost Classifier"
    ) -> Path:
        """
        Génère un rapport HTML complet.
        
        Args:
            output_path: Chemin du fichier HTML
            training_result: Résultat d'entraînement (optionnel)
            plots: Dictionnaire des chemins des graphiques
            model_name: Nom du modèle pour le titre
            
        Returns:
            Chemin du fichier HTML généré
        """
        output_path = Path(output_path)
        
        # Calculer les métriques
        roc_auc = roc_auc_score(self._y_true, self._y_proba)
        pr_auc = self.compute_pr_auc()
        brier = self.compute_brier_score()
        
        # Trouver le seuil optimal
        optimal_thresh, optimal_metrics = self.find_optimal_threshold()
        
        # Métriques à seuil 0.5
        y_pred_50 = (self._y_proba >= 0.5).astype(int)
        tp_50 = np.sum((y_pred_50 == 1) & (self._y_true == 1))
        fp_50 = np.sum((y_pred_50 == 1) & (self._y_true == 0))
        fn_50 = np.sum((y_pred_50 == 0) & (self._y_true == 1))
        tn_50 = np.sum((y_pred_50 == 0) & (self._y_true == 0))
        
        # Générer le HTML
        html_content = self._build_html_report(
            model_name=model_name,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            brier=brier,
            optimal_thresh=optimal_thresh,
            optimal_metrics=optimal_metrics,
            confusion_matrix=(tp_50, fp_50, fn_50, tn_50),
            n_samples=len(self._y_true),
            n_positive=int(np.sum(self._y_true)),
            plots=plots,
            training_result=training_result,
        )
        
        # Sauvegarder
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding='utf-8')
        
        logger.info(f"📄 Rapport HTML généré: {output_path}")
        return output_path
    
    def _build_html_report(
        self,
        model_name: str,
        roc_auc: float,
        pr_auc: float,
        brier: float,
        optimal_thresh: float,
        optimal_metrics: Dict[str, float],
        confusion_matrix: Tuple[int, int, int, int],
        n_samples: int,
        n_positive: int,
        plots: Optional[Dict[str, Path]],
        training_result: Optional[TrainingResult],
    ) -> str:
        """Construit le contenu HTML du rapport."""
        
        tp, fp, fn, tn = confusion_matrix
        
        # Préparer les images en base64 ou chemins relatifs
        plot_sections = ""
        if plots:
            for name, path in plots.items():
                title = name.replace("_", " ").title()
                plot_sections += f"""
                <div class="plot-container">
                    <h3>{title}</h3>
                    <img src="{path.name}" alt="{title}" />
                </div>
                """
        
        # Feature importance table si disponible
        feature_table = ""
        if training_result and training_result.feature_importance:
            rows = ""
            for i, (feat, imp) in enumerate(training_result.feature_importance.top_features[:15], 1):
                bar_width = int(imp * 200)
                rows += f"""
                <tr>
                    <td>{i}</td>
                    <td>{feat}</td>
                    <td>{imp:.4f}</td>
                    <td><div class="bar" style="width: {bar_width}px;"></div></td>
                </tr>
                """
            
            feature_table = f"""
            <section class="section">
                <h2>🏆 Top 15 Features</h2>
                <table class="feature-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Feature</th>
                            <th>Importance</th>
                            <th>Visualisation</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </section>
            """
        
        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport d'Évaluation - {model_name}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #16a34a;
            --warning: #f59e0b;
            --danger: #dc2626;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--primary);
        }}
        
        h1 {{
            font-size: 2rem;
            color: var(--primary);
            margin-bottom: 0.5rem;
        }}
        
        .subtitle {{
            color: var(--text-muted);
        }}
        
        .section {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}
        
        .metric-card {{
            background: var(--bg);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary);
        }}
        
        .metric-value.success {{ color: var(--success); }}
        .metric-value.warning {{ color: var(--warning); }}
        .metric-value.danger {{ color: var(--danger); }}
        
        .metric-label {{
            font-size: 0.875rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}
        
        .confusion-matrix {{
            display: grid;
            grid-template-columns: auto auto auto;
            gap: 0.5rem;
            max-width: 400px;
            margin: 1rem auto;
        }}
        
        .cm-cell {{
            padding: 1rem;
            text-align: center;
            border-radius: 4px;
        }}
        
        .cm-tp {{ background: #dcfce7; color: #166534; }}
        .cm-tn {{ background: #dbeafe; color: #1e40af; }}
        .cm-fp {{ background: #fee2e2; color: #991b1b; }}
        .cm-fn {{ background: #fef3c7; color: #92400e; }}
        .cm-header {{ font-weight: bold; color: var(--text-muted); }}
        
        .plot-container {{
            margin: 1.5rem 0;
            text-align: center;
        }}
        
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .plot-container h3 {{
            margin-bottom: 1rem;
            color: var(--text-muted);
        }}
        
        .plots-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 1.5rem;
        }}
        
        .feature-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .feature-table th, .feature-table td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        .feature-table th {{
            background: var(--bg);
            font-weight: 600;
        }}
        
        .bar {{
            height: 20px;
            background: linear-gradient(90deg, var(--primary), #60a5fa);
            border-radius: 4px;
        }}
        
        .recommendation {{
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border-left: 4px solid var(--success);
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin-top: 1rem;
        }}
        
        .recommendation strong {{
            color: var(--success);
        }}
        
        footer {{
            text-align: center;
            color: var(--text-muted);
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Rapport d'Évaluation du Modèle</h1>
            <p class="subtitle">{model_name} • {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </header>
        
        <section class="section">
            <h2>📈 Métriques Principales</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value {'success' if roc_auc > 0.7 else 'warning' if roc_auc > 0.6 else 'danger'}">{roc_auc:.4f}</div>
                    <div class="metric-label">ROC-AUC</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'success' if pr_auc > 0.5 else 'warning' if pr_auc > 0.3 else 'danger'}">{pr_auc:.4f}</div>
                    <div class="metric-label">PR-AUC</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'success' if brier < 0.2 else 'warning' if brier < 0.3 else 'danger'}">{brier:.4f}</div>
                    <div class="metric-label">Brier Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{optimal_thresh:.2f}</div>
                    <div class="metric-label">Seuil Optimal (F1)</div>
                </div>
            </div>
            
            <div class="recommendation">
                <strong>💡 Recommandation:</strong> Utiliser un seuil de <strong>{optimal_thresh:.2f}</strong> 
                pour un F1-Score de {optimal_metrics.get('f1', 0):.4f} 
                (Precision: {optimal_metrics.get('precision', 0):.4f}, Recall: {optimal_metrics.get('recall', 0):.4f})
            </div>
        </section>
        
        <section class="section">
            <h2>🎯 Métriques au Seuil 0.5</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{(tp + tn) / (tp + tn + fp + fn):.4f}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{tp / (tp + fn) if (tp + fn) > 0 else 0:.4f}</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{n_samples:,}</div>
                    <div class="metric-label">Échantillons</div>
                </div>
            </div>
        </section>
        
        <section class="section">
            <h2>📋 Matrice de Confusion (Seuil 0.5)</h2>
            <div class="confusion-matrix">
                <div class="cm-cell cm-header"></div>
                <div class="cm-cell cm-header">Prédit 0</div>
                <div class="cm-cell cm-header">Prédit 1</div>
                
                <div class="cm-cell cm-header">Vrai 0</div>
                <div class="cm-cell cm-tn">TN: {tn:,}</div>
                <div class="cm-cell cm-fp">FP: {fp:,}</div>
                
                <div class="cm-cell cm-header">Vrai 1</div>
                <div class="cm-cell cm-fn">FN: {fn:,}</div>
                <div class="cm-cell cm-tp">TP: {tp:,}</div>
            </div>
            <p style="text-align: center; color: var(--text-muted); margin-top: 0.5rem;">
                Positifs: {n_positive:,} ({n_positive/n_samples*100:.1f}%) • 
                Négatifs: {n_samples - n_positive:,} ({(n_samples-n_positive)/n_samples*100:.1f}%)
            </p>
        </section>
        
        {feature_table}
        
        <section class="section">
            <h2>📊 Visualisations</h2>
            <div class="plots-grid">
                {plot_sections}
            </div>
        </section>
        
        <footer>
            <p>Généré par CryptoScalper AI • ModelEvaluator</p>
        </footer>
    </div>
</body>
</html>"""
        
        return html
    
    # =========================================
    # RAPPORT COMPLET
    # =========================================
    
    def generate_full_report(
        self,
        output_dir: Path,
        training_result: Optional[TrainingResult] = None,
        model_name: str = "XGBoost Classifier"
    ) -> EvaluationReport:
        """
        Génère un rapport complet avec graphiques et HTML.
        
        Args:
            output_dir: Dossier de sortie
            training_result: Résultat d'entraînement (optionnel)
            model_name: Nom du modèle
            
        Returns:
            EvaluationReport avec tous les chemins
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature importance
        feature_importance = None
        if training_result and training_result.feature_importance:
            feature_importance = training_result.feature_importance
        
        # Générer tous les graphiques
        plots = self.generate_all_plots(output_dir, feature_importance)
        
        # Générer le rapport HTML
        html_path = output_dir / "evaluation_report.html"
        self.generate_html_report(html_path, training_result, plots, model_name)
        plots["html_report"] = html_path
        
        # Calculer les métriques
        roc_auc = roc_auc_score(self._y_true, self._y_proba)
        pr_auc = self.compute_pr_auc()
        brier = self.compute_brier_score()
        optimal_thresh, _ = self.find_optimal_threshold()
        
        # Construire le rapport
        y_pred = (self._y_proba >= 0.5).astype(int)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix as cm
        
        metrics = EvaluationMetrics(
            accuracy=accuracy_score(self._y_true, y_pred),
            precision=precision_score(self._y_true, y_pred, zero_division=0),
            recall=recall_score(self._y_true, y_pred, zero_division=0),
            f1=f1_score(self._y_true, y_pred, zero_division=0),
            roc_auc=roc_auc,
            confusion_matrix=cm(self._y_true, y_pred),
        )
        
        report = EvaluationReport(
            metrics=metrics,
            feature_importance=feature_importance,
            plots=plots,
            brier_score=brier,
            pr_auc=pr_auc,
            recommended_threshold=optimal_thresh,
            dataset_info={
                "n_samples": len(self._y_true),
                "n_positive": int(np.sum(self._y_true)),
                "positive_ratio": float(np.mean(self._y_true)),
            }
        )
        
        logger.info(f"✅ Rapport complet généré dans {output_dir}")
        return report


# ============================================
# FONCTION UTILITAIRE
# ============================================

def evaluate_model_from_file(
    model_path: Path,
    dataset_path: Path,
    output_dir: Path,
) -> EvaluationReport:
    """
    Évalue un modèle depuis des fichiers.
    
    Args:
        model_path: Chemin du modèle (.joblib)
        dataset_path: Chemin du dataset (.parquet)
        output_dir: Dossier de sortie
        
    Returns:
        EvaluationReport
    """
    import joblib
    from cryptoscalper.data.dataset import PreparedDataset
    
    # Charger le modèle
    model = joblib.load(model_path)
    logger.info(f"📂 Modèle chargé: {model_path}")
    
    # Charger le dataset
    dataset = PreparedDataset.load(dataset_path)
    logger.info(f"📂 Dataset chargé: {dataset_path} ({len(dataset)} échantillons)")
    
    # Créer l'évaluateur
    evaluator = ModelEvaluator(model, dataset)
    
    # Générer le rapport
    return evaluator.generate_full_report(output_dir)


# Exports
__all__ = [
    "ModelEvaluator",
    "EvaluationReport",
    "evaluate_model_from_file",
]