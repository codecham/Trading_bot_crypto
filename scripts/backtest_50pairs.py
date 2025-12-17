#!/usr/bin/env python3
# scripts/backtest_50pairs.py
"""
Backtest du modèle XGBoost sur les données historiques.

Simule le trading avec les mêmes règles que le paper trading:
- TP: 2%
- SL: 1%
- Timeout: 120 min
- Frais: 0.2%

Usage:
    python scripts/backtest_50pairs.py
    python scripts/backtest_50pairs.py --threshold 0.35
    python scripts/backtest_50pairs.py --threshold 0.40 --max-positions 3
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
import joblib
from loguru import logger


# ============================================
# CONFIGURATION
# ============================================

@dataclass
class BacktestConfig:
    """Configuration du backtest."""
    # Fichiers
    model_path: str = "models/saved/swing_50pairs_model.joblib"
    test_data_path: str = "datasets/swing_50pairs_test.parquet"
    
    # Capital
    initial_capital: float = 25.0
    position_size_pct: float = 0.20  # 20% du capital par trade
    
    # Stratégie
    take_profit_pct: float = 0.02   # 2%
    stop_loss_pct: float = 0.01     # 1%
    timeout_minutes: int = 120       # 2h
    trading_fee_pct: float = 0.001   # 0.1% par trade (0.2% aller-retour)
    
    # Filtres
    probability_threshold: float = 0.35
    max_open_positions: int = 1      # Positions simultanées max
    min_minutes_between_trades: int = 5  # Éviter les trades trop rapprochés
    
    # Output
    output_dir: str = "backtest_results"


class TradeStatus(Enum):
    """Statut d'un trade."""
    OPEN = "OPEN"
    WIN_TP = "WIN_TP"
    LOSS_SL = "LOSS_SL"
    LOSS_TIMEOUT = "LOSS_TIMEOUT"
    WIN_TIMEOUT = "WIN_TIMEOUT"


@dataclass
class Trade:
    """Représente un trade."""
    id: int
    symbol: str
    entry_time: datetime
    entry_price: float
    size_usdt: float
    probability: float
    tp_price: float
    sl_price: float
    timeout_time: datetime
    
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    pnl_usdt: float = 0.0
    pnl_pct: float = 0.0
    
    @property
    def duration_minutes(self) -> float:
        if self.exit_time and self.entry_time:
            return (self.exit_time - self.entry_time).total_seconds() / 60
        return 0.0


@dataclass
class BacktestResult:
    """Résultats du backtest."""
    # Config
    config: BacktestConfig
    
    # Trades
    trades: List[Trade] = field(default_factory=list)
    
    # Métriques
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # PnL
    total_pnl_usdt: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # Capital
    final_capital: float = 0.0
    peak_capital: float = 0.0
    
    # Equity curve
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades * 100
    
    @property
    def avg_win_pct(self) -> float:
        wins = [t for t in self.trades if t.pnl_pct > 0]
        if not wins:
            return 0.0
        return sum(t.pnl_pct for t in wins) / len(wins)
    
    @property
    def avg_loss_pct(self) -> float:
        losses = [t for t in self.trades if t.pnl_pct <= 0]
        if not losses:
            return 0.0
        return sum(t.pnl_pct for t in losses) / len(losses)
    
    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_usdt for t in self.trades if t.pnl_usdt > 0)
        gross_loss = abs(sum(t.pnl_usdt for t in self.trades if t.pnl_usdt < 0))
        if gross_loss == 0:
            return float('inf')
        return gross_profit / gross_loss
    
    @property
    def avg_trade_duration(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.duration_minutes for t in self.trades) / len(self.trades)


def get_feature_names() -> List[str]:
    """Retourne la liste des 42 features."""
    return [
        "rsi_14", "rsi_7", "stoch_k", "stoch_d", "williams_r",
        "roc_5", "roc_10", "momentum_5", "cci", "cmo",
        "ema_5_ratio", "ema_10_ratio", "ema_20_ratio",
        "macd_line", "macd_signal", "macd_histogram",
        "adx", "aroon_oscillator",
        "bb_width", "bb_position", "atr", "atr_percent",
        "returns_std", "hl_range_avg",
        "spread_percent", "orderbook_imbalance", "bid_depth", "ask_depth",
        "depth_ratio", "bid_pressure", "ask_pressure", "midprice_distance",
        "volume_relative", "obv_slope", "volume_delta", "vwap_distance", "ad_line",
        "returns_1m", "returns_5m", "returns_15m",
        "consecutive_green", "candle_body_ratio"
    ]


# ============================================
# BACKTEST ENGINE
# ============================================

class BacktestEngine:
    """Moteur de backtest."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.model = None
        self.data = None
        self.feature_names = get_feature_names()
        
        # État
        self.capital = config.initial_capital
        self.peak_capital = config.initial_capital
        self.open_trades: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0
        
        # Tracking
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.max_drawdown = 0.0
        
        # Pour éviter les trades trop rapprochés
        self.last_trade_time: Dict[str, datetime] = {}
    
    def load_model(self) -> bool:
        """Charge le modèle."""
        try:
            self.model = joblib.load(self.config.model_path)
            logger.info(f"✅ Modèle chargé: {self.config.model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            return False
    
    def load_data(self) -> bool:
        """Charge les données de test."""
        try:
            self.data = pd.read_parquet(self.config.test_data_path)
            
            # S'assurer que open_time est datetime
            if not pd.api.types.is_datetime64_any_dtype(self.data['open_time']):
                self.data['open_time'] = pd.to_datetime(self.data['open_time'])
            
            # Trier par temps
            self.data = self.data.sort_values(['open_time', 'symbol']).reset_index(drop=True)
            
            logger.info(f"✅ Données chargées: {len(self.data):,} rows")
            logger.info(f"   Période: {self.data['open_time'].min()} → {self.data['open_time'].max()}")
            logger.info(f"   Symboles: {self.data['symbol'].nunique()}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Erreur chargement données: {e}")
            return False
    
    def get_prediction(self, row: pd.Series) -> float:
        """Obtient la probabilité de hausse pour une ligne."""
        features = row[self.feature_names].values.reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        proba = self.model.predict_proba(features)[0, 1]
        return proba
    
    def can_open_trade(self, symbol: str, current_time: datetime) -> bool:
        """Vérifie si on peut ouvrir un trade."""
        # Vérifier le nombre de positions ouvertes
        if len(self.open_trades) >= self.config.max_open_positions:
            return False
        
        # Vérifier si on a déjà une position sur ce symbole
        if symbol in self.open_trades:
            return False
        
        # Vérifier le délai minimum entre trades
        if symbol in self.last_trade_time:
            time_since_last = (current_time - self.last_trade_time[symbol]).total_seconds() / 60
            if time_since_last < self.config.min_minutes_between_trades:
                return False
        
        return True
    
    def open_trade(self, row: pd.Series, probability: float) -> Trade:
        """Ouvre un nouveau trade."""
        self.trade_counter += 1
        
        entry_price = row['close']
        symbol = row['symbol']
        current_time = row['open_time']
        
        # Calculer la taille
        size_usdt = self.capital * self.config.position_size_pct
        
        # Calculer TP et SL
        tp_price = entry_price * (1 + self.config.take_profit_pct)
        sl_price = entry_price * (1 - self.config.stop_loss_pct)
        
        # Timeout
        timeout_time = current_time + timedelta(minutes=self.config.timeout_minutes)
        
        trade = Trade(
            id=self.trade_counter,
            symbol=symbol,
            entry_time=current_time,
            entry_price=entry_price,
            size_usdt=size_usdt,
            probability=probability,
            tp_price=tp_price,
            sl_price=sl_price,
            timeout_time=timeout_time,
        )
        
        self.open_trades[symbol] = trade
        self.last_trade_time[symbol] = current_time
        
        return trade
    
    def check_trade_exit(self, trade: Trade, row: pd.Series) -> bool:
        """Vérifie si un trade doit être fermé."""
        current_time = row['open_time']
        high = row['high']
        low = row['low']
        close = row['close']
        
        # Check Stop Loss (sur le low)
        if low <= trade.sl_price:
            self._close_trade(trade, trade.sl_price, current_time, TradeStatus.LOSS_SL)
            return True
        
        # Check Take Profit (sur le high)
        if high >= trade.tp_price:
            self._close_trade(trade, trade.tp_price, current_time, TradeStatus.WIN_TP)
            return True
        
        # Check Timeout
        if current_time >= trade.timeout_time:
            status = TradeStatus.WIN_TIMEOUT if close > trade.entry_price else TradeStatus.LOSS_TIMEOUT
            self._close_trade(trade, close, current_time, status)
            return True
        
        return False
    
    def _close_trade(self, trade: Trade, exit_price: float, exit_time: datetime, status: TradeStatus):
        """Ferme un trade."""
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.status = status
        
        # Calculer le PnL
        gross_pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        fee_pct = self.config.trading_fee_pct * 2  # Aller-retour
        trade.pnl_pct = (gross_pnl_pct - fee_pct) * 100
        trade.pnl_usdt = trade.size_usdt * (gross_pnl_pct - fee_pct)
        
        # Mettre à jour le capital
        self.capital += trade.pnl_usdt
        
        # Tracker le peak et drawdown
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        current_dd = (self.peak_capital - self.capital) / self.peak_capital * 100
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
        
        # Déplacer vers closed
        if trade.symbol in self.open_trades:
            del self.open_trades[trade.symbol]
        self.closed_trades.append(trade)
    
    def run(self) -> BacktestResult:
        """Exécute le backtest."""
        if self.model is None or self.data is None:
            raise ValueError("Modèle ou données non chargés")
        
        logger.info(f"\n🚀 Démarrage du backtest...")
        logger.info(f"   Seuil: {self.config.probability_threshold:.0%}")
        logger.info(f"   Capital: {self.config.initial_capital} USDT")
        logger.info(f"   Position size: {self.config.position_size_pct:.0%}")
        logger.info(f"   Max positions: {self.config.max_open_positions}")
        
        start_time = time.time()
        
        # Grouper par timestamp pour simuler le temps réel
        grouped = self.data.groupby('open_time')
        total_timestamps = len(grouped)
        
        processed = 0
        last_log = 0
        
        for timestamp, group in grouped:
            processed += 1
            
            # Log progression tous les 10%
            progress = processed / total_timestamps * 100
            if progress - last_log >= 10:
                logger.info(f"   Progression: {progress:.0f}% ({processed:,}/{total_timestamps:,})")
                last_log = progress
            
            # Enregistrer l'equity curve (toutes les heures environ)
            if processed % 60 == 0:
                self.equity_curve.append((timestamp, self.capital))
            
            # Pour chaque symbole à ce timestamp
            for _, row in group.iterrows():
                symbol = row['symbol']
                
                # Vérifier les trades ouverts
                if symbol in self.open_trades:
                    trade = self.open_trades[symbol]
                    self.check_trade_exit(trade, row)
                
                # Vérifier si on peut ouvrir un nouveau trade
                if self.can_open_trade(symbol, timestamp):
                    proba = self.get_prediction(row)
                    
                    if proba >= self.config.probability_threshold:
                        self.open_trade(row, proba)
        
        # Fermer les trades restants au prix de clôture final
        logger.info(f"   Fermeture des {len(self.open_trades)} positions restantes...")
        last_row = self.data.iloc[-1]
        for symbol, trade in list(self.open_trades.items()):
            # Trouver le dernier prix pour ce symbole
            symbol_data = self.data[self.data['symbol'] == symbol]
            if len(symbol_data) > 0:
                final_row = symbol_data.iloc[-1]
                self._close_trade(trade, final_row['close'], final_row['open_time'], TradeStatus.LOSS_TIMEOUT)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Backtest terminé en {elapsed:.1f}s")
        
        # Compiler les résultats
        result = self._compile_results()
        
        return result
    
    def _compile_results(self) -> BacktestResult:
        """Compile les résultats du backtest."""
        winning = [t for t in self.closed_trades if t.pnl_usdt > 0]
        losing = [t for t in self.closed_trades if t.pnl_usdt <= 0]
        
        total_pnl_usdt = sum(t.pnl_usdt for t in self.closed_trades)
        total_pnl_pct = (self.capital - self.config.initial_capital) / self.config.initial_capital * 100
        
        result = BacktestResult(
            config=self.config,
            trades=self.closed_trades,
            total_trades=len(self.closed_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            total_pnl_usdt=total_pnl_usdt,
            total_pnl_pct=total_pnl_pct,
            max_drawdown_pct=self.max_drawdown,
            final_capital=self.capital,
            peak_capital=self.peak_capital,
            equity_curve=self.equity_curve,
        )
        
        return result


# ============================================
# AFFICHAGE DES RÉSULTATS
# ============================================

def print_results(result: BacktestResult):
    """Affiche les résultats du backtest."""
    
    print("\n" + "=" * 65)
    print("📊 RÉSULTATS DU BACKTEST")
    print("=" * 65)
    
    # Configuration
    print(f"\n⚙️  Configuration:")
    print(f"   Seuil probabilité: {result.config.probability_threshold:.0%}")
    print(f"   Capital initial: {result.config.initial_capital:.2f} USDT")
    print(f"   Position size: {result.config.position_size_pct:.0%}")
    print(f"   TP: {result.config.take_profit_pct:.1%} | SL: {result.config.stop_loss_pct:.1%}")
    print(f"   Timeout: {result.config.timeout_minutes} min")
    
    # Performance globale
    print(f"\n📈 Performance globale:")
    print(f"   Capital final: {result.final_capital:.2f} USDT")
    print(f"   PnL total: {result.total_pnl_usdt:+.2f} USDT ({result.total_pnl_pct:+.1f}%)")
    print(f"   Max Drawdown: {result.max_drawdown_pct:.1f}%")
    
    # Trades
    print(f"\n🎯 Statistiques des trades:")
    print(f"   Total trades: {result.total_trades}")
    print(f"   Gagnants: {result.winning_trades} ({result.win_rate:.1f}%)")
    print(f"   Perdants: {result.losing_trades} ({100 - result.win_rate:.1f}%)")
    print(f"   Profit Factor: {result.profit_factor:.2f}")
    
    # Moyennes
    print(f"\n📊 Moyennes:")
    print(f"   Gain moyen: {result.avg_win_pct:+.2f}%")
    print(f"   Perte moyenne: {result.avg_loss_pct:.2f}%")
    print(f"   Durée moyenne: {result.avg_trade_duration:.0f} min")
    
    # Par type de sortie
    tp_trades = [t for t in result.trades if t.status == TradeStatus.WIN_TP]
    sl_trades = [t for t in result.trades if t.status == TradeStatus.LOSS_SL]
    timeout_trades = [t for t in result.trades if t.status in [TradeStatus.WIN_TIMEOUT, TradeStatus.LOSS_TIMEOUT]]
    
    print(f"\n🏷️  Par type de sortie:")
    print(f"   Take Profit: {len(tp_trades)} ({len(tp_trades)/result.total_trades*100:.1f}%)")
    print(f"   Stop Loss: {len(sl_trades)} ({len(sl_trades)/result.total_trades*100:.1f}%)")
    print(f"   Timeout: {len(timeout_trades)} ({len(timeout_trades)/result.total_trades*100:.1f}%)")
    
    # Par symbole (top 10)
    print(f"\n🏆 Top 10 symboles par PnL:")
    symbol_pnl = {}
    symbol_trades = {}
    for trade in result.trades:
        if trade.symbol not in symbol_pnl:
            symbol_pnl[trade.symbol] = 0
            symbol_trades[trade.symbol] = 0
        symbol_pnl[trade.symbol] += trade.pnl_usdt
        symbol_trades[trade.symbol] += 1
    
    sorted_symbols = sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True)
    for symbol, pnl in sorted_symbols[:10]:
        trades_count = symbol_trades[symbol]
        print(f"   {symbol:<12} {pnl:>+8.2f} USDT ({trades_count} trades)")
    
    # Pires symboles
    print(f"\n📉 Pires 5 symboles:")
    for symbol, pnl in sorted_symbols[-5:]:
        trades_count = symbol_trades[symbol]
        print(f"   {symbol:<12} {pnl:>+8.2f} USDT ({trades_count} trades)")
    
    # Résumé final
    print("\n" + "=" * 65)
    if result.total_pnl_pct > 0:
        print(f"✅ BACKTEST RENTABLE: {result.total_pnl_pct:+.1f}%")
    else:
        print(f"❌ BACKTEST NON RENTABLE: {result.total_pnl_pct:+.1f}%")
    print("=" * 65)


def save_results(result: BacktestResult, output_dir: str):
    """Sauvegarde les résultats."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder les trades en CSV
    trades_data = []
    for t in result.trades:
        trades_data.append({
            'id': t.id,
            'symbol': t.symbol,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'size_usdt': t.size_usdt,
            'probability': t.probability,
            'pnl_usdt': t.pnl_usdt,
            'pnl_pct': t.pnl_pct,
            'status': t.status.value,
            'duration_min': t.duration_minutes,
        })
    
    trades_df = pd.DataFrame(trades_data)
    trades_path = output_path / f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    trades_df.to_csv(trades_path, index=False)
    logger.info(f"💾 Trades sauvegardés: {trades_path}")
    
    # Sauvegarder l'equity curve
    if result.equity_curve:
        equity_df = pd.DataFrame(result.equity_curve, columns=['timestamp', 'capital'])
        equity_path = output_path / f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        equity_df.to_csv(equity_path, index=False)
        logger.info(f"💾 Equity curve sauvegardée: {equity_path}")
    
    # Sauvegarder le résumé
    summary = {
        'threshold': result.config.probability_threshold,
        'initial_capital': result.config.initial_capital,
        'final_capital': result.final_capital,
        'total_pnl_usdt': result.total_pnl_usdt,
        'total_pnl_pct': result.total_pnl_pct,
        'total_trades': result.total_trades,
        'win_rate': result.win_rate,
        'max_drawdown_pct': result.max_drawdown_pct,
        'profit_factor': result.profit_factor,
        'avg_trade_duration': result.avg_trade_duration,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = output_path / f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"💾 Résumé sauvegardé: {summary_path}")


# ============================================
# MAIN
# ============================================

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Backtest du modèle 50 paires")
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.20,
        help="Seuil de probabilité (défaut: 0.20)"
    )
    parser.add_argument(
        "--tp",
        type=float,
        default=2.0,
        help="Take Profit en %% (défaut: 2.0)"
    )
    parser.add_argument(
        "--sl",
        type=float,
        default=1.0,
        help="Stop Loss en %% (défaut: 1.0)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout en minutes (défaut: 120)"
    )
    parser.add_argument(
        "--max-positions", "-m",
        type=int,
        default=1,
        help="Nombre max de positions simultanées (défaut: 1)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=25.0,
        help="Capital initial (défaut: 25 USDT)"
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.20,
        help="Taille de position en %% du capital (défaut: 0.20)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/saved/swing_50pairs_model.joblib",
        help="Chemin du modèle"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/swing_50pairs_test.parquet",
        help="Chemin des données de test"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Sauvegarder les résultats"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    print("=" * 65)
    print("📊 CryptoScalper AI - Backtest 50 Paires")
    print("=" * 65)
    
    # Configuration
    config = BacktestConfig(
        model_path=args.model,
        test_data_path=args.data,
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        take_profit_pct=args.tp / 100,  # Convertir % en décimal
        stop_loss_pct=args.sl / 100,    # Convertir % en décimal
        timeout_minutes=args.timeout,
        probability_threshold=args.threshold,
        max_open_positions=args.max_positions,
    )
    
    # Créer le moteur de backtest
    engine = BacktestEngine(config)
    
    # Charger le modèle et les données
    if not engine.load_model():
        return 1
    
    if not engine.load_data():
        return 1
    
    # Exécuter le backtest
    result = engine.run()
    
    # Afficher les résultats
    print_results(result)
    
    # Sauvegarder si demandé
    if args.save:
        save_results(result, config.output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())