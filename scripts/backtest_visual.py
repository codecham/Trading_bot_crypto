#!/usr/bin/env python3
# scripts/backtest_visual.py
"""
Backtest avec affichage visuel en temps réel.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.text import Text

from cryptoscalper.backtest import BacktestEngine, BacktestConfig
from cryptoscalper.models.predictor import MLPredictor
from cryptoscalper.data.features import FeatureEngine
from cryptoscalper.utils.logger import setup_logger

console = Console()


def create_stats_table(stats: dict) -> Table:
    """Crée le tableau des statistiques."""
    table = Table(title="📊 Statistiques Live", show_header=False, box=None)
    table.add_column("Métrique", style="cyan")
    table.add_column("Valeur", style="green")
    
    table.add_row("💰 Capital", f"{stats.get('capital', 0):.2f} USDT")
    table.add_row("📈 Equity", f"{stats.get('equity', 0):.2f} USDT")
    table.add_row("📊 PnL", f"{stats.get('pnl', 0):+.2f} USDT ({stats.get('pnl_pct', 0):+.1f}%)")
    table.add_row("", "")
    table.add_row("🎯 Trades", f"{stats.get('total_trades', 0)}")
    table.add_row("✅ Gagnants", f"{stats.get('wins', 0)}")
    table.add_row("❌ Perdants", f"{stats.get('losses', 0)}")
    table.add_row("📈 Win Rate", f"{stats.get('win_rate', 0):.1f}%")
    table.add_row("", "")
    table.add_row("📉 Max Drawdown", f"{stats.get('max_dd', 0):.2f}%")
    table.add_row("🔥 Meilleur", f"{stats.get('best_trade', 0):+.2f}%")
    table.add_row("💀 Pire", f"{stats.get('worst_trade', 0):+.2f}%")
    
    return table


def create_trades_table(trades: list) -> Table:
    """Crée le tableau des derniers trades."""
    table = Table(title="📋 Derniers Trades", show_lines=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("Heure", style="cyan", width=12)
    table.add_column("PnL", width=10)
    table.add_column("Raison", width=12)
    
    for trade in trades[-8:]:  # 8 derniers
        pnl = trade.get('pnl', 0)
        pnl_style = "green" if pnl >= 0 else "red"
        pnl_str = f"{pnl:+.3f}%"
        
        reason_emoji = {
            'take_profit': '🎯 TP',
            'stop_loss': '🛑 SL',
            'timeout': '⏰ TO',
        }.get(trade.get('reason', ''), '❓')
        
        table.add_row(
            str(trade.get('id', '-')),
            trade.get('time', '-'),
            Text(pnl_str, style=pnl_style),
            reason_emoji
        )
    
    return table


def create_equity_chart(equity_history: list, width: int = 50) -> str:
    """Crée un mini graphique ASCII de l'equity."""
    if len(equity_history) < 2:
        return "En attente de données..."
    
    # Prendre les N derniers points
    data = equity_history[-width:]
    
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val if max_val > min_val else 1
    
    height = 8
    chart_lines = []
    
    for h in range(height, -1, -1):
        line = ""
        threshold = min_val + (h / height) * range_val
        for val in data:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        chart_lines.append(line)
    
    # Ajouter les labels
    chart = f"  {max_val:.2f} ┤" + chart_lines[0] + "\n"
    for line in chart_lines[1:-1]:
        chart += "           │" + line + "\n"
    chart += f"  {min_val:.2f} ┤" + chart_lines[-1]
    
    return chart


def run_visual_backtest(
    data_path: str,
    model_path: str,
    capital: float = 25.0,
    prob_threshold: float = 0.65,
    sl: float = 0.005,
    tp: float = 0.004,
):
    """Lance le backtest avec affichage visuel."""
    
    console.print("\n[bold cyan]🤖 CryptoScalper AI - Backtest Visuel[/bold cyan]\n")
    
    # Charger les données
    with console.status("[bold green]Chargement des données..."):
        df = pd.read_parquet(data_path)
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        console.print(f"  ✅ {len(df):,} bougies chargées")
    
    # Charger le modèle
    with console.status("[bold green]Chargement du modèle ML..."):
        predictor = MLPredictor.from_file(model_path)
        console.print(f"  ✅ Modèle chargé (calibré: {predictor.is_calibrated})")
    
    # Config
    config = BacktestConfig(
        initial_capital=capital,
        position_size_pct=0.2,
        stop_loss_pct=sl,
        take_profit_pct=tp,
        min_probability=prob_threshold,
        fee_percent=0.001,
        slippage_percent=0.0005,
    )
    
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Capital: {capital:.2f} USDT | SL: {sl:.2%} | TP: {tp:.2%} | Seuil: {prob_threshold:.0%}")
    console.print()
    
    # État du backtest
    stats = {
        'capital': capital,
        'equity': capital,
        'pnl': 0,
        'pnl_pct': 0,
        'total_trades': 0,
        'wins': 0,
        'losses': 0,
        'win_rate': 0,
        'max_dd': 0,
        'best_trade': 0,
        'worst_trade': 0,
    }
    trades_history = []
    equity_history = [capital]
    
    feature_engine = FeatureEngine()
    engine = BacktestEngine(config)
    
    warmup = 50
    n_rows = len(df)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        
        task = progress.add_task("[cyan]Backtest en cours...", total=n_rows - warmup)
        
        # Initialiser l'état
        current_capital = capital
        peak_capital = capital
        open_trade = None
        trade_id = 0
        
        for i in range(warmup, n_rows):
            progress.update(task, advance=1)
            
            # Données jusqu'à maintenant
            current_data = df.iloc[:i+1]
            current_row = df.iloc[i]
            timestamp = df.index[i]
            
            current_price = float(current_row['close'])
            high_price = float(current_row['high'])
            low_price = float(current_row['low'])
            
            # Vérifier position ouverte
            if open_trade:
                # Check SL
                sl_price = open_trade['entry'] * (1 - sl)
                if low_price <= sl_price:
                    pnl_pct = -sl * 100
                    current_capital *= (1 - sl - 0.002)  # fees
                    trades_history.append({
                        'id': open_trade['id'],
                        'time': str(timestamp)[-8:-3] if hasattr(timestamp, 'strftime') else str(i),
                        'pnl': pnl_pct,
                        'reason': 'stop_loss'
                    })
                    stats['losses'] += 1
                    stats['worst_trade'] = min(stats['worst_trade'], pnl_pct)
                    open_trade = None
                    
                # Check TP
                elif high_price >= open_trade['entry'] * (1 + tp):
                    pnl_pct = tp * 100
                    current_capital *= (1 + tp - 0.002)  # fees
                    trades_history.append({
                        'id': open_trade['id'],
                        'time': str(timestamp)[-8:-3] if hasattr(timestamp, 'strftime') else str(i),
                        'pnl': pnl_pct,
                        'reason': 'take_profit'
                    })
                    stats['wins'] += 1
                    stats['best_trade'] = max(stats['best_trade'], pnl_pct)
                    open_trade = None
            
            # Nouvelle prédiction si pas de position
            if not open_trade and i % 5 == 0:  # Check toutes les 5 bougies
                try:
                    feature_set = feature_engine.compute_features(
                        current_data.tail(60),
                        symbol="BTCUSDT"
                    )
                    prediction = predictor.predict(feature_set)
                    
                    if prediction.probability_up >= prob_threshold:
                        trade_id += 1
                        open_trade = {
                            'id': trade_id,
                            'entry': current_price,
                            'time': timestamp
                        }
                        stats['total_trades'] += 1
                        
                except Exception:
                    pass
            
            # Update stats
            equity_history.append(current_capital)
            peak_capital = max(peak_capital, current_capital)
            drawdown = (peak_capital - current_capital) / peak_capital * 100
            
            stats['capital'] = current_capital
            stats['equity'] = current_capital
            stats['pnl'] = current_capital - capital
            stats['pnl_pct'] = (current_capital / capital - 1) * 100
            stats['max_dd'] = max(stats['max_dd'], drawdown)
            
            total = stats['wins'] + stats['losses']
            stats['win_rate'] = (stats['wins'] / total * 100) if total > 0 else 0
            
            # Afficher toutes les 1000 bougies
            if i % 1000 == 0:
                progress.console.print(
                    f"  [dim]#{i:,}[/dim] | "
                    f"💰 {current_capital:.2f} | "
                    f"📊 {stats['total_trades']} trades | "
                    f"✅ {stats['win_rate']:.0f}%"
                )
    
    # Résultats finaux
    console.print("\n" + "=" * 60)
    console.print("[bold green]✅ BACKTEST TERMINÉ[/bold green]")
    console.print("=" * 60)
    
    final_table = Table(show_header=False, box=None)
    final_table.add_column("", style="cyan")
    final_table.add_column("", style="bold")
    
    pnl_style = "green" if stats['pnl'] >= 0 else "red"
    
    final_table.add_row("Capital Initial", f"{capital:.2f} USDT")
    final_table.add_row("Capital Final", f"{stats['capital']:.2f} USDT")
    final_table.add_row("PnL", Text(f"{stats['pnl']:+.2f} USDT ({stats['pnl_pct']:+.1f}%)", style=pnl_style))
    final_table.add_row("", "")
    final_table.add_row("Total Trades", str(stats['total_trades']))
    final_table.add_row("Gagnants", f"{stats['wins']} ({stats['win_rate']:.1f}%)")
    final_table.add_row("Perdants", str(stats['losses']))
    final_table.add_row("", "")
    final_table.add_row("Max Drawdown", f"{stats['max_dd']:.2f}%")
    final_table.add_row("Meilleur Trade", f"{stats['best_trade']:+.2f}%")
    final_table.add_row("Pire Trade", f"{stats['worst_trade']:+.2f}%")
    
    console.print(final_table)
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Backtest visuel")
    parser.add_argument("--data", required=True, help="Fichier de données")
    parser.add_argument("--model", required=True, help="Fichier du modèle")
    parser.add_argument("--capital", type=float, default=25.0)
    parser.add_argument("--prob", type=float, default=0.65)
    parser.add_argument("--sl", type=float, default=0.005)
    parser.add_argument("--tp", type=float, default=0.004)
    
    args = parser.parse_args()
    
    setup_logger(level="WARNING")
    
    run_visual_backtest(
        args.data,
        args.model,
        args.capital,
        args.prob,
        args.sl,
        args.tp,
    )


if __name__ == "__main__":
    main()