# cryptoscalper/utils/trade_logger.py
"""
Module de logging et statistiques des trades.

Responsabilités :
- Sauvegarder les trades dans un fichier CSV
- Calculer les statistiques de trading
- Exporter les données pour analyse
- Générer des résumés de performance

Usage:
    trade_logger = TradeLogger("logs/trades.csv")
    
    # Logger un trade terminé
    trade_logger.log_trade(completed_trade)
    
    # Obtenir les statistiques
    stats = trade_logger.get_statistics()
    print(f"Win rate: {stats.win_rate:.1%}")
"""

import csv
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json

from cryptoscalper.utils.logger import logger


# ============================================
# DATACLASSES
# ============================================

@dataclass
class TradeRecord:
    """
    Enregistrement d'un trade pour le CSV.
    
    Attributes:
        trade_id: ID unique du trade
        symbol: Paire tradée
        side: BUY ou SELL
        entry_time: Timestamp d'entrée (ISO)
        exit_time: Timestamp de sortie (ISO)
        entry_price: Prix d'entrée
        exit_price: Prix de sortie
        quantity: Quantité tradée
        pnl_usdt: Profit/perte en USDT
        pnl_percent: Profit/perte en %
        close_reason: Raison de fermeture
        duration_seconds: Durée du trade
        commission: Commission totale
        signal_score: Score du signal (optionnel)
    """
    
    trade_id: str
    symbol: str
    side: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl_usdt: float
    pnl_percent: float
    close_reason: str
    duration_seconds: int
    commission: float = 0.0
    signal_score: Optional[float] = None
    
    @classmethod
    def from_completed_trade(cls, trade) -> "TradeRecord":
        """Crée un TradeRecord depuis un CompletedTrade."""
        return cls(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            side=trade.side.value if hasattr(trade.side, 'value') else str(trade.side),
            entry_time=trade.entry_time.isoformat(),
            exit_time=trade.exit_time.isoformat(),
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            quantity=trade.quantity,
            pnl_usdt=trade.pnl_usdt,
            pnl_percent=trade.pnl_percent,
            close_reason=trade.close_reason.value if hasattr(trade.close_reason, 'value') else str(trade.close_reason),
            duration_seconds=trade.duration_seconds,
            commission=trade.commission_total,
            signal_score=trade.signal_score,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return asdict(self)


@dataclass
class TradingStatistics:
    """
    Statistiques de trading agrégées.
    
    Attributes:
        total_trades: Nombre total de trades
        winning_trades: Nombre de trades gagnants
        losing_trades: Nombre de trades perdants
        win_rate: Taux de réussite (0-1)
        total_pnl_usdt: PnL total en USDT
        avg_pnl_usdt: PnL moyen par trade
        max_win_usdt: Plus gros gain
        max_loss_usdt: Plus grosse perte
        avg_win_usdt: Gain moyen
        avg_loss_usdt: Perte moyenne
        profit_factor: Ratio gains/pertes
        avg_duration_seconds: Durée moyenne des trades
        total_commission: Commission totale payée
        best_symbol: Meilleur symbole
        worst_symbol: Pire symbole
    """
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl_usdt: float = 0.0
    avg_pnl_usdt: float = 0.0
    max_win_usdt: float = 0.0
    max_loss_usdt: float = 0.0
    avg_win_usdt: float = 0.0
    avg_loss_usdt: float = 0.0
    profit_factor: float = 0.0
    avg_duration_seconds: float = 0.0
    total_commission: float = 0.0
    best_symbol: str = ""
    worst_symbol: str = ""
    
    # Par close_reason
    take_profit_count: int = 0
    stop_loss_count: int = 0
    timeout_count: int = 0
    manual_count: int = 0
    
    # Période
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class DailyStatistics:
    """
    Statistiques journalières.
    """
    
    date: str
    trades_count: int = 0
    winning_trades: int = 0
    pnl_usdt: float = 0.0
    commission: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0


# ============================================
# TRADE LOGGER
# ============================================

class TradeLogger:
    """
    Logger de trades avec sauvegarde CSV et statistiques.
    
    Fonctionnalités:
    - Sauvegarde automatique des trades en CSV
    - Calcul des statistiques en temps réel
    - Export pour analyse externe
    - Résumés journaliers/hebdomadaires/mensuels
    
    Usage:
        trade_logger = TradeLogger("logs/trades.csv")
        trade_logger.log_trade(completed_trade)
        stats = trade_logger.get_statistics()
    """
    
    # Colonnes du fichier CSV
    CSV_COLUMNS = [
        "trade_id", "symbol", "side", "entry_time", "exit_time",
        "entry_price", "exit_price", "quantity", "pnl_usdt", "pnl_percent",
        "close_reason", "duration_seconds", "commission", "signal_score"
    ]
    
    def __init__(
        self,
        csv_path: str = "logs/trades.csv",
        auto_create: bool = True
    ):
        """
        Initialise le trade logger.
        
        Args:
            csv_path: Chemin du fichier CSV
            auto_create: Créer le fichier s'il n'existe pas
        """
        self._csv_path = Path(csv_path)
        self._trades: List[TradeRecord] = []
        
        # Créer le dossier si nécessaire
        if auto_create:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Charger les trades existants
        self._load_existing_trades()
        
        logger.info(
            f"📊 TradeLogger initialisé | "
            f"Fichier: {self._csv_path} | "
            f"Trades chargés: {len(self._trades)}"
        )
    
    # =========================================
    # PROPRIÉTÉS
    # =========================================
    
    @property
    def csv_path(self) -> Path:
        """Chemin du fichier CSV."""
        return self._csv_path
    
    @property
    def trades_count(self) -> int:
        """Nombre total de trades."""
        return len(self._trades)
    
    @property
    def trades(self) -> List[TradeRecord]:
        """Liste des trades."""
        return self._trades.copy()
    
    # =========================================
    # LOGGING DES TRADES
    # =========================================
    
    def log_trade(self, trade) -> None:
        """
        Log un trade terminé.
        
        Args:
            trade: CompletedTrade ou objet similaire
        """
        # Convertir en TradeRecord
        if isinstance(trade, TradeRecord):
            record = trade
        else:
            record = TradeRecord.from_completed_trade(trade)
        
        # Ajouter à la liste
        self._trades.append(record)
        
        # Sauvegarder dans le CSV
        self._append_to_csv(record)
        
        # Log
        emoji = "🎉" if record.pnl_usdt > 0 else "😤"
        logger.info(
            f"{emoji} Trade logged | {record.trade_id} | "
            f"{record.symbol} | PnL: {record.pnl_usdt:+.4f} USDT"
        )
    
    def log_trades_batch(self, trades: List) -> None:
        """
        Log plusieurs trades d'un coup.
        
        Args:
            trades: Liste de CompletedTrade
        """
        for trade in trades:
            self.log_trade(trade)
    
    # =========================================
    # STATISTIQUES
    # =========================================
    
    def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbol: Optional[str] = None
    ) -> TradingStatistics:
        """
        Calcule les statistiques de trading.
        
        Args:
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)
            symbol: Filtrer par symbole (optionnel)
            
        Returns:
            TradingStatistics avec toutes les métriques
        """
        # Filtrer les trades
        filtered_trades = self._filter_trades(start_date, end_date, symbol)
        
        if not filtered_trades:
            return TradingStatistics()
        
        # Calculs de base
        total = len(filtered_trades)
        winners = [t for t in filtered_trades if t.pnl_usdt > 0]
        losers = [t for t in filtered_trades if t.pnl_usdt < 0]
        
        winning_count = len(winners)
        losing_count = len(losers)
        
        # PnL
        total_pnl = sum(t.pnl_usdt for t in filtered_trades)
        total_wins = sum(t.pnl_usdt for t in winners)
        total_losses = abs(sum(t.pnl_usdt for t in losers))
        
        # Max/Min
        max_win = max((t.pnl_usdt for t in winners), default=0.0)
        max_loss = min((t.pnl_usdt for t in losers), default=0.0)
        
        # Moyennes
        avg_pnl = total_pnl / total if total > 0 else 0.0
        avg_win = total_wins / winning_count if winning_count > 0 else 0.0
        avg_loss = total_losses / losing_count if losing_count > 0 else 0.0
        
        # Profit factor
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Duration
        avg_duration = sum(t.duration_seconds for t in filtered_trades) / total if total > 0 else 0.0
        
        # Commission
        total_commission = sum(t.commission for t in filtered_trades)
        
        # Par close_reason
        tp_count = sum(1 for t in filtered_trades if "take_profit" in t.close_reason.lower())
        sl_count = sum(1 for t in filtered_trades if "stop_loss" in t.close_reason.lower())
        timeout_count = sum(1 for t in filtered_trades if "timeout" in t.close_reason.lower())
        manual_count = sum(1 for t in filtered_trades if "manual" in t.close_reason.lower())
        
        # Par symbole
        symbol_pnl = self._calculate_symbol_pnl(filtered_trades)
        best_symbol = max(symbol_pnl.items(), key=lambda x: x[1])[0] if symbol_pnl else ""
        worst_symbol = min(symbol_pnl.items(), key=lambda x: x[1])[0] if symbol_pnl else ""
        
        # Période
        start = min(t.entry_time for t in filtered_trades)
        end = max(t.exit_time for t in filtered_trades)
        
        return TradingStatistics(
            total_trades=total,
            winning_trades=winning_count,
            losing_trades=losing_count,
            win_rate=winning_count / total if total > 0 else 0.0,
            total_pnl_usdt=total_pnl,
            avg_pnl_usdt=avg_pnl,
            max_win_usdt=max_win,
            max_loss_usdt=max_loss,
            avg_win_usdt=avg_win,
            avg_loss_usdt=avg_loss,
            profit_factor=profit_factor,
            avg_duration_seconds=avg_duration,
            total_commission=total_commission,
            best_symbol=best_symbol,
            worst_symbol=worst_symbol,
            take_profit_count=tp_count,
            stop_loss_count=sl_count,
            timeout_count=timeout_count,
            manual_count=manual_count,
            start_date=start,
            end_date=end,
        )
    
    def get_daily_statistics(
        self,
        days: int = 7
    ) -> List[DailyStatistics]:
        """
        Calcule les statistiques par jour.
        
        Args:
            days: Nombre de jours à inclure
            
        Returns:
            Liste de DailyStatistics
        """
        daily_stats = []
        today = date.today()
        
        for i in range(days):
            target_date = today - timedelta(days=i)
            date_str = target_date.isoformat()
            
            # Filtrer les trades du jour
            day_trades = [
                t for t in self._trades
                if t.exit_time.startswith(date_str)
            ]
            
            if not day_trades:
                daily_stats.append(DailyStatistics(date=date_str))
                continue
            
            pnl = sum(t.pnl_usdt for t in day_trades)
            winners = sum(1 for t in day_trades if t.pnl_usdt > 0)
            commission = sum(t.commission for t in day_trades)
            best = max(t.pnl_usdt for t in day_trades)
            worst = min(t.pnl_usdt for t in day_trades)
            
            daily_stats.append(DailyStatistics(
                date=date_str,
                trades_count=len(day_trades),
                winning_trades=winners,
                pnl_usdt=pnl,
                commission=commission,
                best_trade_pnl=best,
                worst_trade_pnl=worst,
            ))
        
        return daily_stats
    
    def get_symbol_statistics(self) -> Dict[str, TradingStatistics]:
        """
        Calcule les statistiques par symbole.
        
        Returns:
            Dict[symbol, TradingStatistics]
        """
        # Grouper par symbole
        symbols = set(t.symbol for t in self._trades)
        
        stats_by_symbol = {}
        for symbol in symbols:
            stats_by_symbol[symbol] = self.get_statistics(symbol=symbol)
        
        return stats_by_symbol
    
    # =========================================
    # EXPORT
    # =========================================
    
    def export_to_json(self, filepath: str) -> None:
        """
        Exporte les trades en JSON.
        
        Args:
            filepath: Chemin du fichier JSON
        """
        data = {
            "trades": [t.to_dict() for t in self._trades],
            "statistics": asdict(self.get_statistics()),
            "exported_at": datetime.utcnow().isoformat(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"📤 Trades exportés vers {filepath}")
    
    def export_summary(self, filepath: str) -> None:
        """
        Exporte un résumé textuel des performances.
        
        Args:
            filepath: Chemin du fichier de résumé
        """
        stats = self.get_statistics()
        daily = self.get_daily_statistics(7)
        
        lines = [
            "=" * 60,
            "📊 RÉSUMÉ DE TRADING",
            "=" * 60,
            "",
            "📈 STATISTIQUES GLOBALES",
            "-" * 40,
            f"  Total trades: {stats.total_trades}",
            f"  Win rate: {stats.win_rate:.1%}",
            f"  PnL total: {stats.total_pnl_usdt:+.4f} USDT",
            f"  PnL moyen: {stats.avg_pnl_usdt:+.4f} USDT",
            f"  Profit factor: {stats.profit_factor:.2f}",
            f"  Durée moyenne: {stats.avg_duration_seconds:.0f}s",
            "",
            f"  🎉 Trades gagnants: {stats.winning_trades}",
            f"     Gain moyen: +{stats.avg_win_usdt:.4f} USDT",
            f"     Max gain: +{stats.max_win_usdt:.4f} USDT",
            "",
            f"  😤 Trades perdants: {stats.losing_trades}",
            f"     Perte moyenne: -{stats.avg_loss_usdt:.4f} USDT",
            f"     Max perte: {stats.max_loss_usdt:.4f} USDT",
            "",
            "📊 PAR TYPE DE SORTIE",
            "-" * 40,
            f"  Take Profit: {stats.take_profit_count}",
            f"  Stop Loss: {stats.stop_loss_count}",
            f"  Timeout: {stats.timeout_count}",
            f"  Manual: {stats.manual_count}",
            "",
            f"  Meilleur symbole: {stats.best_symbol}",
            f"  Pire symbole: {stats.worst_symbol}",
            "",
            "📅 DERNIERS 7 JOURS",
            "-" * 40,
        ]
        
        for day in daily:
            if day.trades_count > 0:
                lines.append(
                    f"  {day.date}: {day.trades_count} trades | "
                    f"PnL: {day.pnl_usdt:+.4f} USDT | "
                    f"Win: {day.winning_trades}/{day.trades_count}"
                )
            else:
                lines.append(f"  {day.date}: Aucun trade")
        
        lines.extend([
            "",
            "=" * 60,
            f"Généré le: {datetime.utcnow().isoformat()}",
        ])
        
        with open(filepath, 'w') as f:
            f.write("\n".join(lines))
        
        logger.info(f"📤 Résumé exporté vers {filepath}")
    
    def print_summary(self) -> None:
        """Affiche un résumé dans la console."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 50)
        print("📊 TRADING SUMMARY")
        print("=" * 50)
        print(f"  Total trades: {stats.total_trades}")
        print(f"  Win rate: {stats.win_rate:.1%}")
        print(f"  PnL total: {stats.total_pnl_usdt:+.4f} USDT")
        print(f"  Profit factor: {stats.profit_factor:.2f}")
        print(f"  Avg duration: {stats.avg_duration_seconds:.0f}s")
        print("=" * 50 + "\n")
    
    # =========================================
    # MÉTHODES PRIVÉES
    # =========================================
    
    def _load_existing_trades(self) -> None:
        """Charge les trades existants depuis le CSV."""
        if not self._csv_path.exists():
            return
        
        try:
            with open(self._csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Convertir les types
                    record = TradeRecord(
                        trade_id=row["trade_id"],
                        symbol=row["symbol"],
                        side=row["side"],
                        entry_time=row["entry_time"],
                        exit_time=row["exit_time"],
                        entry_price=float(row["entry_price"]),
                        exit_price=float(row["exit_price"]),
                        quantity=float(row["quantity"]),
                        pnl_usdt=float(row["pnl_usdt"]),
                        pnl_percent=float(row["pnl_percent"]),
                        close_reason=row["close_reason"],
                        duration_seconds=int(row["duration_seconds"]),
                        commission=float(row.get("commission", 0)),
                        signal_score=float(row["signal_score"]) if row.get("signal_score") else None,
                    )
                    self._trades.append(record)
                    
        except Exception as e:
            logger.warning(f"Erreur chargement trades existants: {e}")
    
    def _append_to_csv(self, record: TradeRecord) -> None:
        """Ajoute un trade au fichier CSV."""
        file_exists = self._csv_path.exists()
        
        try:
            with open(self._csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                
                # Écrire l'en-tête si nouveau fichier
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(record.to_dict())
                
        except Exception as e:
            logger.error(f"Erreur écriture CSV: {e}")
    
    def _filter_trades(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbol: Optional[str] = None
    ) -> List[TradeRecord]:
        """Filtre les trades selon les critères."""
        filtered = self._trades
        
        if symbol:
            filtered = [t for t in filtered if t.symbol == symbol]
        
        if start_date:
            start_str = start_date.isoformat()
            filtered = [t for t in filtered if t.entry_time >= start_str]
        
        if end_date:
            end_str = end_date.isoformat()
            filtered = [t for t in filtered if t.exit_time <= end_str]
        
        return filtered
    
    def _calculate_symbol_pnl(self, trades: List[TradeRecord]) -> Dict[str, float]:
        """Calcule le PnL par symbole."""
        symbol_pnl = defaultdict(float)
        
        for trade in trades:
            symbol_pnl[trade.symbol] += trade.pnl_usdt
        
        return dict(symbol_pnl)


# ============================================
# FONCTIONS UTILITAIRES
# ============================================

def create_trade_logger(
    csv_path: str = "logs/trades.csv"
) -> TradeLogger:
    """
    Crée un TradeLogger avec les paramètres par défaut.
    
    Args:
        csv_path: Chemin du fichier CSV
        
    Returns:
        TradeLogger configuré
    """
    return TradeLogger(csv_path=csv_path)


__all__ = [
    # Dataclasses
    "TradeRecord",
    "TradingStatistics",
    "DailyStatistics",
    # Classes
    "TradeLogger",
    # Functions
    "create_trade_logger",
]