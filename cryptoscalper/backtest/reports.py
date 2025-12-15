# cryptoscalper/backtest/reports.py
"""
Backtest Reports - Génération de rapports et visualisations.

Ce module génère des rapports détaillés à partir des résultats
du backtest, incluant:
- Métriques de performance (Sharpe, Sortino, Calmar)
- Graphiques (equity curve, distribution PnL, monthly returns)
- Export en différents formats (HTML, JSON, CSV)

Usage:
    from cryptoscalper.backtest.reports import BacktestReport
    
    report = BacktestReport(result)
    report.save_html("reports/backtest_report.html")
    report.save_json("reports/backtest_result.json")
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import pandas as pd
import numpy as np

from cryptoscalper.backtest.engine import BacktestResult, BacktestTrade, TradeCloseReason
from cryptoscalper.utils.logger import logger


# ============================================
# MÉTRIQUES AVANCÉES
# ============================================

@dataclass
class AdvancedMetrics:
    """
    Métriques avancées de performance.
    
    Attributes:
        sharpe_ratio: Ratio de Sharpe (rendement / volatilité)
        sortino_ratio: Ratio de Sortino (rendement / volatilité négative)
        calmar_ratio: Ratio de Calmar (rendement / max drawdown)
        max_drawdown: Drawdown maximum
        max_drawdown_duration_days: Durée du pire drawdown
        recovery_factor: PnL total / Max drawdown
        win_rate: Taux de gain
        profit_factor: Gains bruts / Pertes brutes
        expectancy: Espérance mathématique par trade
        avg_win: Gain moyen
        avg_loss: Perte moyenne
        max_consecutive_wins: Gains consécutifs max
        max_consecutive_losses: Pertes consécutives max
    """
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: float = 0.0
    recovery_factor: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    @classmethod
    def from_result(cls, result: BacktestResult) -> "AdvancedMetrics":
        """
        Calcule les métriques avancées depuis un BacktestResult.
        
        Args:
            result: Résultat du backtest
            
        Returns:
            AdvancedMetrics calculées
        """
        trades = result.trades
        if not trades:
            return cls()
        
        pnls = [t.pnl_usdt for t in trades if t.pnl_usdt is not None]
        if not pnls:
            return cls()
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        # Ratios de base
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        win_rate = len(wins) / len(pnls) if pnls else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Sharpe ratio (annualisé)
        returns = np.array(pnls) / result.initial_capital
        if len(returns) > 1 and np.std(returns) > 0:
            # Assumant des trades sur 5 min, environ 12 trades/heure possible
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 24 * 12)
        else:
            sharpe = 0.0
        
        # Sortino ratio (ne compte que la volatilité négative)
        negative_returns = [r for r in returns if r < 0]
        if negative_returns and len(negative_returns) > 1:
            downside_std = np.std(negative_returns)
            sortino = (np.mean(returns) / downside_std) * np.sqrt(252 * 24 * 12) if downside_std > 0 else 0
        else:
            sortino = 0.0
        
        # Calmar ratio
        calmar = result.total_return_pct / result.max_drawdown if result.max_drawdown > 0 else 0
        
        # Recovery factor
        recovery = result.total_pnl / (result.max_drawdown * result.initial_capital) if result.max_drawdown > 0 else 0
        
        # Séries consécutives
        max_wins, max_losses = _calculate_consecutive_streaks(pnls)
        
        # Durée du drawdown max
        dd_duration = _calculate_max_drawdown_duration(result.equity_curve)
        
        return cls(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=result.max_drawdown,
            max_drawdown_duration_days=dd_duration,
            recovery_factor=recovery,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
        )


def _calculate_consecutive_streaks(pnls: List[float]) -> Tuple[int, int]:
    """
    Calcule les séries de gains/pertes consécutifs.
    
    Args:
        pnls: Liste des PnL
        
    Returns:
        (max_wins, max_losses)
    """
    if not pnls:
        return 0, 0
    
    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0
    
    for pnl in pnls:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
    
    return max_wins, max_losses


def _calculate_max_drawdown_duration(
    equity_curve: List[Tuple[datetime, float]]
) -> float:
    """
    Calcule la durée du drawdown maximum en jours.
    
    Args:
        equity_curve: Liste de (timestamp, equity)
        
    Returns:
        Durée en jours
    """
    if not equity_curve:
        return 0.0
    
    peak = equity_curve[0][1]
    peak_time = equity_curve[0][0]
    max_duration = timedelta(0)
    current_dd_start = None
    
    for timestamp, equity in equity_curve:
        if equity >= peak:
            # Nouveau peak, fin du drawdown
            if current_dd_start is not None:
                duration = timestamp - current_dd_start
                max_duration = max(max_duration, duration)
            peak = equity
            peak_time = timestamp
            current_dd_start = None
        else:
            # En drawdown
            if current_dd_start is None:
                current_dd_start = peak_time
    
    return max_duration.total_seconds() / (24 * 3600)


# ============================================
# ANALYSE PAR PÉRIODE
# ============================================

@dataclass
class PeriodStats:
    """Statistiques pour une période donnée."""
    
    period: str
    trades: int
    pnl: float
    win_rate: float
    best_trade: float
    worst_trade: float


def calculate_monthly_stats(
    result: BacktestResult
) -> List[PeriodStats]:
    """
    Calcule les statistiques mensuelles.
    
    Args:
        result: Résultat du backtest
        
    Returns:
        Liste de PeriodStats par mois
    """
    trades = result.trades
    if not trades:
        return []
    
    # Grouper par mois
    monthly: Dict[str, List[BacktestTrade]] = {}
    for trade in trades:
        if trade.entry_time:
            month_key = trade.entry_time.strftime("%Y-%m")
            if month_key not in monthly:
                monthly[month_key] = []
            monthly[month_key].append(trade)
    
    stats = []
    for month, month_trades in sorted(monthly.items()):
        pnls = [t.pnl_usdt for t in month_trades if t.pnl_usdt is not None]
        wins = [p for p in pnls if p > 0]
        
        stats.append(PeriodStats(
            period=month,
            trades=len(month_trades),
            pnl=sum(pnls),
            win_rate=len(wins) / len(pnls) if pnls else 0,
            best_trade=max(pnls) if pnls else 0,
            worst_trade=min(pnls) if pnls else 0,
        ))
    
    return stats


def calculate_hourly_stats(
    result: BacktestResult
) -> Dict[int, Dict]:
    """
    Calcule les statistiques par heure de la journée.
    
    Utile pour identifier les meilleures heures de trading.
    
    Args:
        result: Résultat du backtest
        
    Returns:
        Dict avec stats par heure (0-23)
    """
    trades = result.trades
    if not trades:
        return {}
    
    hourly: Dict[int, List[float]] = {h: [] for h in range(24)}
    
    for trade in trades:
        if trade.entry_time and trade.pnl_usdt is not None:
            hour = trade.entry_time.hour
            hourly[hour].append(trade.pnl_usdt)
    
    stats = {}
    for hour, pnls in hourly.items():
        if pnls:
            wins = [p for p in pnls if p > 0]
            stats[hour] = {
                "trades": len(pnls),
                "pnl": sum(pnls),
                "win_rate": len(wins) / len(pnls),
                "avg_pnl": np.mean(pnls),
            }
        else:
            stats[hour] = {
                "trades": 0,
                "pnl": 0,
                "win_rate": 0,
                "avg_pnl": 0,
            }
    
    return stats


# ============================================
# GÉNÉRATION DE GRAPHIQUES (ASCII)
# ============================================

def plot_equity_curve_ascii(
    equity_curve: List[Tuple[datetime, float]],
    width: int = 60,
    height: int = 15,
) -> str:
    """
    Génère un graphique ASCII de l'equity curve.
    
    Args:
        equity_curve: Liste de (timestamp, equity)
        width: Largeur du graphique
        height: Hauteur du graphique
        
    Returns:
        Graphique ASCII
    """
    if not equity_curve:
        return "Pas de données"
    
    equities = [e[1] for e in equity_curve]
    min_eq = min(equities)
    max_eq = max(equities)
    
    if max_eq == min_eq:
        max_eq = min_eq + 1
    
    # Réduire les points si trop nombreux
    if len(equities) > width:
        step = len(equities) // width
        equities = equities[::step][:width]
    
    lines = []
    for row in range(height):
        line = ""
        threshold = max_eq - (row / (height - 1)) * (max_eq - min_eq)
        
        for eq in equities:
            if eq >= threshold:
                line += "█"
            else:
                line += " "
        
        # Label sur les bords
        if row == 0:
            line += f" {max_eq:.2f}"
        elif row == height - 1:
            line += f" {min_eq:.2f}"
        
        lines.append(line)
    
    # Axe X
    x_axis = "─" * len(equities)
    lines.append(x_axis)
    
    return "\n".join(lines)


def plot_pnl_distribution_ascii(
    trades: List[BacktestTrade],
    bins: int = 10,
    width: int = 40,
) -> str:
    """
    Génère un histogramme ASCII de la distribution des PnL.
    
    Args:
        trades: Liste des trades
        bins: Nombre de bins
        width: Largeur des barres
        
    Returns:
        Histogramme ASCII
    """
    pnls = [t.pnl_usdt for t in trades if t.pnl_usdt is not None]
    if not pnls:
        return "Pas de données"
    
    # Créer les bins
    min_pnl = min(pnls)
    max_pnl = max(pnls)
    
    if min_pnl == max_pnl:
        return f"Tous les trades: {min_pnl:.4f} USDT"
    
    bin_size = (max_pnl - min_pnl) / bins
    counts = [0] * bins
    
    for pnl in pnls:
        idx = min(int((pnl - min_pnl) / bin_size), bins - 1)
        counts[idx] += 1
    
    max_count = max(counts) if counts else 1
    
    lines = []
    for i, count in enumerate(counts):
        bar_len = int((count / max_count) * width)
        bar = "█" * bar_len
        
        range_start = min_pnl + i * bin_size
        range_end = range_start + bin_size
        
        label = f"{range_start:+.4f} to {range_end:+.4f}"
        lines.append(f"{label:>25} | {bar} ({count})")
    
    return "\n".join(lines)


# ============================================
# RAPPORT COMPLET
# ============================================

class BacktestReport:
    """
    Générateur de rapports de backtest.
    
    Crée des rapports complets avec métriques, graphiques
    et analyse détaillée des résultats.
    
    Usage:
        report = BacktestReport(result)
        print(report.generate_text())
        report.save_html("report.html")
    """
    
    def __init__(self, result: BacktestResult):
        """
        Initialise le générateur de rapport.
        
        Args:
            result: Résultat du backtest
        """
        self._result = result
        self._metrics = AdvancedMetrics.from_result(result)
        self._monthly_stats = calculate_monthly_stats(result)
        self._hourly_stats = calculate_hourly_stats(result)
    
    @property
    def result(self) -> BacktestResult:
        """Résultat du backtest."""
        return self._result
    
    @property
    def metrics(self) -> AdvancedMetrics:
        """Métriques avancées."""
        return self._metrics
    
    def generate_text(self) -> str:
        """
        Génère un rapport textuel complet.
        
        Returns:
            Rapport formaté
        """
        lines = [
            "=" * 70,
            "📊 RAPPORT DE BACKTEST - CryptoScalper AI",
            "=" * 70,
            "",
            self._result.summary(),
            "",
            "=" * 70,
            "📈 MÉTRIQUES AVANCÉES",
            "=" * 70,
            "",
            f"  Sharpe Ratio:           {self._metrics.sharpe_ratio:>10.2f}",
            f"  Sortino Ratio:          {self._metrics.sortino_ratio:>10.2f}",
            f"  Calmar Ratio:           {self._metrics.calmar_ratio:>10.2f}",
            f"  Recovery Factor:        {self._metrics.recovery_factor:>10.2f}",
            "",
            f"  Max Drawdown Duration:  {self._metrics.max_drawdown_duration_days:>10.1f} jours",
            "",
            f"  Gain moyen:             {self._metrics.avg_win:>+10.4f} USDT",
            f"  Perte moyenne:          {self._metrics.avg_loss:>+10.4f} USDT",
            f"  Espérance:              {self._metrics.expectancy:>+10.4f} USDT",
            "",
            f"  Série gains max:        {self._metrics.max_consecutive_wins:>10}",
            f"  Série pertes max:       {self._metrics.max_consecutive_losses:>10}",
            "",
        ]
        
        # Stats mensuelles
        if self._monthly_stats:
            lines.extend([
                "=" * 70,
                "📅 PERFORMANCE MENSUELLE",
                "=" * 70,
                "",
                f"  {'Mois':<10} {'Trades':>8} {'PnL':>12} {'Win Rate':>10}",
                "  " + "-" * 42,
            ])
            
            for stat in self._monthly_stats:
                lines.append(
                    f"  {stat.period:<10} {stat.trades:>8} "
                    f"{stat.pnl:>+12.4f} {stat.win_rate:>10.1%}"
                )
            
            lines.append("")
        
        # Equity curve ASCII
        lines.extend([
            "=" * 70,
            "📈 EQUITY CURVE",
            "=" * 70,
            "",
            plot_equity_curve_ascii(self._result.equity_curve),
            "",
        ])
        
        # Distribution PnL
        lines.extend([
            "=" * 70,
            "📊 DISTRIBUTION DES PnL",
            "=" * 70,
            "",
            plot_pnl_distribution_ascii(self._result.trades),
            "",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def save_text(self, filepath: str) -> None:
        """
        Sauvegarde le rapport en texte.
        
        Args:
            filepath: Chemin du fichier
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.generate_text())
        
        logger.info(f"📄 Rapport texte sauvegardé: {filepath}")
    
    def save_json(self, filepath: str) -> None:
        """
        Sauvegarde les résultats en JSON.
        
        Args:
            filepath: Chemin du fichier
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "result": self._result.to_dict(),
            "metrics": {
                "sharpe_ratio": self._metrics.sharpe_ratio,
                "sortino_ratio": self._metrics.sortino_ratio,
                "calmar_ratio": self._metrics.calmar_ratio,
                "max_drawdown": self._metrics.max_drawdown,
                "max_drawdown_duration_days": self._metrics.max_drawdown_duration_days,
                "recovery_factor": self._metrics.recovery_factor,
                "win_rate": self._metrics.win_rate,
                "profit_factor": self._metrics.profit_factor,
                "expectancy": self._metrics.expectancy,
                "avg_win": self._metrics.avg_win,
                "avg_loss": self._metrics.avg_loss,
                "max_consecutive_wins": self._metrics.max_consecutive_wins,
                "max_consecutive_losses": self._metrics.max_consecutive_losses,
            },
            "monthly_stats": [
                {
                    "period": s.period,
                    "trades": s.trades,
                    "pnl": s.pnl,
                    "win_rate": s.win_rate,
                }
                for s in self._monthly_stats
            ],
            "hourly_stats": self._hourly_stats,
            "generated_at": datetime.now().isoformat(),
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"📄 Résultats JSON sauvegardés: {filepath}")
    
    def save_trades_csv(self, filepath: str) -> None:
        """
        Sauvegarde la liste des trades en CSV.
        
        Args:
            filepath: Chemin du fichier
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        trades_data = []
        for t in self._result.trades:
            trades_data.append({
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "size_usdt": t.size_usdt,
                "stop_loss_price": t.stop_loss_price,
                "take_profit_price": t.take_profit_price,
                "pnl_usdt": t.pnl_usdt,
                "pnl_percent": t.pnl_percent,
                "fees_paid": t.fees_paid,
                "close_reason": t.close_reason.value if t.close_reason else None,
                "probability": t.probability,
                "confidence": t.confidence,
                "duration_minutes": t.duration_minutes,
            })
        
        df = pd.DataFrame(trades_data)
        df.to_csv(path, index=False)
        
        logger.info(f"📄 Trades CSV sauvegardés: {filepath}")
    
    def save_html(self, filepath: str) -> None:
        """
        Sauvegarde le rapport en HTML.
        
        Args:
            filepath: Chemin du fichier
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        html = self._generate_html()
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        
        logger.info(f"📄 Rapport HTML sauvegardé: {filepath}")
    
    def _generate_html(self) -> str:
        """Génère le rapport HTML."""
        r = self._result
        m = self._metrics
        
        # Couleurs selon performance
        pnl_color = "#22c55e" if r.total_return >= 0 else "#ef4444"
        wr_color = "#22c55e" if r.win_rate >= 0.5 else "#ef4444"
        
        # Données pour le graphique equity
        equity_data = [
            {"x": ts.isoformat(), "y": eq}
            for ts, eq in r.equity_curve[::max(1, len(r.equity_curve)//100)]
        ]
        
        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport Backtest - CryptoScalper AI</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                background: #0f172a; color: #e2e8f0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ font-size: 2rem; margin-bottom: 30px; text-align: center; }}
        h2 {{ font-size: 1.5rem; margin: 20px 0 15px; color: #94a3b8; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 20px; }}
        .card-title {{ font-size: 0.875rem; color: #94a3b8; margin-bottom: 8px; }}
        .card-value {{ font-size: 1.5rem; font-weight: bold; }}
        .positive {{ color: #22c55e; }}
        .negative {{ color: #ef4444; }}
        .chart-container {{ background: #1e293b; border-radius: 12px; padding: 20px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ color: #94a3b8; font-weight: 500; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; }}
        .badge-win {{ background: #166534; color: #22c55e; }}
        .badge-loss {{ background: #7f1d1d; color: #ef4444; }}
        footer {{ text-align: center; margin-top: 40px; color: #64748b; font-size: 0.875rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Rapport de Backtest</h1>
        
        <div class="card" style="margin-bottom: 20px;">
            <p><strong>Symbole:</strong> {r.symbol} | 
               <strong>Période:</strong> {r.start_date.strftime('%Y-%m-%d')} → {r.end_date.strftime('%Y-%m-%d')} 
               ({r.duration_days} jours)</p>
        </div>
        
        <h2>💰 Performance</h2>
        <div class="grid">
            <div class="card">
                <div class="card-title">Capital Initial</div>
                <div class="card-value">{r.initial_capital:.2f} USDT</div>
            </div>
            <div class="card">
                <div class="card-title">Capital Final</div>
                <div class="card-value">{r.final_capital:.2f} USDT</div>
            </div>
            <div class="card">
                <div class="card-title">Rendement</div>
                <div class="card-value" style="color: {pnl_color};">{r.total_return:+.2f} USDT ({r.total_return_pct:+.2%})</div>
            </div>
            <div class="card">
                <div class="card-title">Max Drawdown</div>
                <div class="card-value negative">{r.max_drawdown:.2%}</div>
            </div>
        </div>
        
        <h2>📊 Trades</h2>
        <div class="grid">
            <div class="card">
                <div class="card-title">Total Trades</div>
                <div class="card-value">{r.total_trades}</div>
            </div>
            <div class="card">
                <div class="card-title">Win Rate</div>
                <div class="card-value" style="color: {wr_color};">{r.win_rate:.1%}</div>
            </div>
            <div class="card">
                <div class="card-title">Profit Factor</div>
                <div class="card-value">{r.profit_factor:.2f}</div>
            </div>
            <div class="card">
                <div class="card-title">Sharpe Ratio</div>
                <div class="card-value">{m.sharpe_ratio:.2f}</div>
            </div>
        </div>
        
        <div class="grid" style="margin-top: 15px;">
            <div class="card">
                <div class="card-title">Take Profit</div>
                <div class="card-value positive">{r.take_profit_count}</div>
            </div>
            <div class="card">
                <div class="card-title">Stop Loss</div>
                <div class="card-value negative">{r.stop_loss_count}</div>
            </div>
            <div class="card">
                <div class="card-title">Timeout</div>
                <div class="card-value">{r.timeout_count}</div>
            </div>
            <div class="card">
                <div class="card-title">Frais Payés</div>
                <div class="card-value">{r.total_fees:.4f} USDT</div>
            </div>
        </div>
        
        <h2>📈 Equity Curve</h2>
        <div class="chart-container">
            <canvas id="equityChart"></canvas>
        </div>
        
        <h2>📉 Métriques Avancées</h2>
        <div class="grid">
            <div class="card">
                <div class="card-title">Sortino Ratio</div>
                <div class="card-value">{m.sortino_ratio:.2f}</div>
            </div>
            <div class="card">
                <div class="card-title">Calmar Ratio</div>
                <div class="card-value">{m.calmar_ratio:.2f}</div>
            </div>
            <div class="card">
                <div class="card-title">Recovery Factor</div>
                <div class="card-value">{m.recovery_factor:.2f}</div>
            </div>
            <div class="card">
                <div class="card-title">Espérance / Trade</div>
                <div class="card-value">{m.expectancy:+.4f} USDT</div>
            </div>
        </div>
        
        <div class="grid" style="margin-top: 15px;">
            <div class="card">
                <div class="card-title">Gain Moyen</div>
                <div class="card-value positive">{m.avg_win:+.4f} USDT</div>
            </div>
            <div class="card">
                <div class="card-title">Perte Moyenne</div>
                <div class="card-value negative">{m.avg_loss:+.4f} USDT</div>
            </div>
            <div class="card">
                <div class="card-title">Série Gains Max</div>
                <div class="card-value">{m.max_consecutive_wins}</div>
            </div>
            <div class="card">
                <div class="card-title">Série Pertes Max</div>
                <div class="card-value">{m.max_consecutive_losses}</div>
            </div>
        </div>
        
        <footer>
            <p>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | CryptoScalper AI</p>
        </footer>
    </div>
    
    <script>
        const equityData = {json.dumps(equity_data)};
        
        new Chart(document.getElementById('equityChart'), {{
            type: 'line',
            data: {{
                labels: equityData.map(d => d.x.split('T')[0]),
                datasets: [{{
                    label: 'Equity (USDT)',
                    data: equityData.map(d => d.y),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }},
                }},
                scales: {{
                    x: {{
                        display: true,
                        grid: {{ color: '#334155' }},
                        ticks: {{ color: '#94a3b8', maxTicksLimit: 10 }}
                    }},
                    y: {{
                        display: true,
                        grid: {{ color: '#334155' }},
                        ticks: {{ color: '#94a3b8' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        return html


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def generate_report(
    result: BacktestResult,
    output_dir: str = "reports",
    prefix: str = "backtest",
) -> BacktestReport:
    """
    Génère tous les rapports pour un backtest.
    
    Args:
        result: Résultat du backtest
        output_dir: Dossier de sortie
        prefix: Préfixe des fichiers
        
    Returns:
        BacktestReport généré
    """
    report = BacktestReport(result)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(output_dir)
    
    # Générer tous les fichiers
    report.save_text(base_path / f"{prefix}_{timestamp}.txt")
    report.save_json(base_path / f"{prefix}_{timestamp}.json")
    report.save_trades_csv(base_path / f"{prefix}_{timestamp}_trades.csv")
    report.save_html(base_path / f"{prefix}_{timestamp}.html")
    
    logger.info(f"✅ Tous les rapports générés dans {output_dir}")
    
    return report


# Aliases pour compatibilité
def plot_equity_curve(result: BacktestResult) -> str:
    """Génère un graphique ASCII de l'equity curve."""
    return plot_equity_curve_ascii(result.equity_curve)


def plot_pnl_distribution(result: BacktestResult) -> str:
    """Génère un histogramme ASCII de la distribution des PnL."""
    return plot_pnl_distribution_ascii(result.trades)


def plot_monthly_returns(result: BacktestResult) -> str:
    """Génère un tableau des rendements mensuels."""
    stats = calculate_monthly_stats(result)
    if not stats:
        return "Pas de données mensuelles"
    
    lines = [
        f"{'Mois':<10} {'Trades':>8} {'PnL':>12} {'Win Rate':>10}",
        "-" * 42,
    ]
    
    for s in stats:
        lines.append(f"{s.period:<10} {s.trades:>8} {s.pnl:>+12.4f} {s.win_rate:>10.1%}")
    
    return "\n".join(lines)


__all__ = [
    # Dataclasses
    "AdvancedMetrics",
    "PeriodStats",
    # Classes
    "BacktestReport",
    # Functions
    "generate_report",
    "calculate_monthly_stats",
    "calculate_hourly_stats",
    "plot_equity_curve",
    "plot_pnl_distribution",
    "plot_monthly_returns",
    "plot_equity_curve_ascii",
    "plot_pnl_distribution_ascii",
]