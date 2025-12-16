#!/usr/bin/env python3
# scripts/prepare_dataset.py
"""
Script pour préparer le dataset d'entraînement avec affichage visuel.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import setup_logger, logger
from cryptoscalper.data.dataset import (
    DatasetBuilder,
    LabelConfig,
    SplitConfig,
    prepare_dataset,
    analyze_class_balance,
)
from cryptoscalper.data.historical import is_data_cached


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Prépare le dataset pour l'entraînement ML"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Liste de symboles séparés par des virgules"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Horizon de prédiction en minutes (défaut: 3)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.002,
        help="Seuil Take Profit (défaut: 0.002 = 0.2%%)"
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.003,
        help="Seuil Stop Loss (défaut: 0.003 = 0.3%%)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/prepared_dataset.parquet",
        help="Fichier de sortie"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data_cache",
        help="Dossier des données sources"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Sauvegarder aussi les splits train/val/test"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbose"
    )
    
    return parser.parse_args()


def main():
    """Point d'entrée principal."""
    args = parse_args()
    
    # Setup logging (silencieux pour le visuel)
    log_level = "DEBUG" if args.verbose else "WARNING"
    setup_logger(level=log_level)
    
    # Header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🔧 CryptoScalper AI - Préparation Dataset[/bold cyan]",
        border_style="cyan"
    ))
    console.print()
    
    # Parser les symboles
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # Vérifier que les données existent
    data_dir = Path(args.data_dir)
    
    with console.status("[bold green]Vérification des données...") as status:
        missing = [s for s in symbols if not is_data_cached(s, data_dir)]
    
    if missing:
        console.print(f"[red]❌ Données manquantes pour: {missing}[/red]")
        console.print("[yellow]Lancez d'abord: python scripts/download_data.py --symbols ...[/yellow]")
        return 1
    
    console.print(f"[green]✅ Données trouvées pour {len(symbols)} symbole(s)[/green]")
    
    # Afficher la configuration
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Param", style="cyan")
    config_table.add_column("Valeur", style="white")
    
    config_table.add_row("📊 Symboles", ", ".join(symbols))
    config_table.add_row("⏱️  Horizon", f"{args.horizon} minutes (timeout)")
    config_table.add_row("🎯 Take Profit", f"+{args.threshold:.2%}")
    config_table.add_row("🛑 Stop Loss", f"-{args.stop_loss:.2%}")
    config_table.add_row("📈 Risk/Reward", f"{args.threshold/args.stop_loss:.2f}x")
    config_table.add_row("📂 Sortie", args.output)
    
    console.print()
    console.print(Panel(config_table, title="[bold]Configuration[/bold]", border_style="blue"))
    console.print()
    
    # Configuration
    label_config = LabelConfig(
        horizon_minutes=args.horizon,
        threshold_percent=args.threshold,
        stop_loss_percent=args.stop_loss
    )
    
    # Construire le dataset
    builder = DatasetBuilder(label_config=label_config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        
        task = progress.add_task("[cyan]Construction du dataset...", total=None)
        
        try:
            dataset = builder.build_from_symbols(symbols, data_dir)
            progress.update(task, completed=True)
        except Exception as e:
            console.print(f"[red]❌ Erreur construction dataset: {e}[/red]")
            return 1
    
    console.print("[green]✅ Dataset construit ![/green]")
    console.print()
    
    # Analyse de l'équilibre
    balance = analyze_class_balance(dataset.labels)
    
    # Tableau des statistiques
    stats_table = Table(title="📊 Statistiques du Dataset", show_header=True)
    stats_table.add_column("Métrique", style="cyan")
    stats_table.add_column("Valeur", style="green", justify="right")
    
    stats_table.add_row("Total samples", f"{len(dataset):,}")
    stats_table.add_row("Features", str(dataset.stats.feature_count))
    stats_table.add_row("", "")
    positive_count = int(balance['positive_ratio'] * len(dataset))
    negative_count = len(dataset) - positive_count
    stats_table.add_row("✅ Labels positifs (TP atteint)", f"{positive_count:,} ({balance['positive_ratio']:.1%})")
    stats_table.add_row("❌ Labels négatifs (SL/timeout)", f"{negative_count:,} ({balance['negative_ratio']:.1%})")
    stats_table.add_row("Déséquilibre", f"{balance['imbalance_ratio']:.2f}x")
    
    console.print(stats_table)
    console.print()
    
    # Sauvegarder
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with console.status("[bold green]Sauvegarde du dataset..."):
        dataset.save(output_path)
    
    console.print(f"[green]✅ Sauvegardé: {output_path}[/green]")
    
    # Split si demandé
    if args.split:
        console.print()
        with console.status("[bold green]Création des splits train/val/test..."):
            train, val, test = dataset.split_temporal()
            
            base_name = output_path.stem
            train.save(output_path.parent / f"{base_name}_train.parquet")
            val.save(output_path.parent / f"{base_name}_val.parquet")
            test.save(output_path.parent / f"{base_name}_test.parquet")
        
        # Tableau des splits
        split_table = Table(title="📂 Splits", show_header=True)
        split_table.add_column("Split", style="cyan")
        split_table.add_column("Samples", justify="right")
        split_table.add_column("Positifs", justify="right")
        split_table.add_column("Fichier", style="dim")
        
        for name, ds in [("Train", train), ("Validation", val), ("Test", test)]:
            bal = analyze_class_balance(ds.labels)
            split_table.add_row(
                name,
                f"{len(ds):,}",
                f"{bal['positive_ratio']:.1%}",
                f"{base_name}_{name.lower()}.parquet"
            )
        
        console.print(split_table)
    
    # Résumé final
    console.print()
    console.print(Panel.fit(
        f"[bold green]✅ DATASET PRÊT ![/bold green]\n\n"
        f"📂 {output_path}\n"
        f"📊 {len(dataset):,} samples\n"
        f"🎯 {balance['positive_ratio']:.1%} positifs",
        border_style="green"
    ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())