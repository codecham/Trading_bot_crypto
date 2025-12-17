#!/usr/bin/env python3
# scripts/test_multiple_configs.py
"""
Teste plusieurs configurations de backtest et affiche un résumé.
"""

import subprocess
import sys
import re

# Configurations à tester
# (TP%, SL%, Timeout_min, Threshold)
CONFIGS = [
    # TP élevé avec SL serré (ratio R:R élevé)
    (2.0, 0.3, 60, 0.20),
    (2.0, 0.3, 120, 0.20),
    (2.0, 0.5, 60, 0.20),
    (2.0, 0.5, 120, 0.20),
    
    # TP moyen avec SL serré
    (1.5, 0.3, 60, 0.20),
    (1.5, 0.5, 60, 0.20),
    (1.5, 0.5, 120, 0.20),
    
    # TP petit avec SL très serré
    (1.0, 0.3, 60, 0.20),
    (1.0, 0.5, 60, 0.20),
    
    # Config originale pour comparaison
    (2.0, 1.0, 120, 0.35),
    (2.0, 1.0, 120, 0.20),
]

def run_backtest(tp, sl, timeout, threshold):
    """Lance un backtest et récupère les résultats."""
    cmd = [
        "python", "scripts/backtest_50pairs.py",
        "--tp", str(tp),
        "--sl", str(sl),
        "--timeout", str(timeout),
        "--threshold", str(threshold),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max
        )
        output = result.stdout + result.stderr
        return parse_output(output)
    except Exception as e:
        return {"error": str(e)}


def parse_output(output):
    """Parse la sortie du backtest."""
    results = {}
    
    # PnL
    match = re.search(r"PnL total: ([+-]?\d+\.?\d*) USDT \(([+-]?\d+\.?\d*)%\)", output)
    if match:
        results["pnl_usdt"] = float(match.group(1))
        results["pnl_pct"] = float(match.group(2))
    
    # Trades
    match = re.search(r"Total trades: (\d+)", output)
    if match:
        results["trades"] = int(match.group(1))
    
    # Win rate
    match = re.search(r"Gagnants: \d+ \((\d+\.?\d*)%\)", output)
    if match:
        results["win_rate"] = float(match.group(1))
    
    # Max DD
    match = re.search(r"Max Drawdown: (\d+\.?\d*)%", output)
    if match:
        results["max_dd"] = float(match.group(1))
    
    # Profit Factor
    match = re.search(r"Profit Factor: (\d+\.?\d*)", output)
    if match:
        results["profit_factor"] = float(match.group(1))
    
    # Exit types
    match = re.search(r"Take Profit: \d+ \((\d+\.?\d*)%\)", output)
    if match:
        results["tp_pct"] = float(match.group(1))
    
    match = re.search(r"Stop Loss: \d+ \((\d+\.?\d*)%\)", output)
    if match:
        results["sl_pct"] = float(match.group(1))
    
    match = re.search(r"Timeout: \d+ \((\d+\.?\d*)%\)", output)
    if match:
        results["to_pct"] = float(match.group(1))
    
    return results


def main():
    print("=" * 80)
    print("🔬 CryptoScalper AI - Test de Configurations Multiples")
    print("=" * 80)
    print()
    
    all_results = []
    
    for i, (tp, sl, timeout, threshold) in enumerate(CONFIGS, 1):
        print(f"[{i}/{len(CONFIGS)}] Test: TP={tp}% SL={sl}% T/O={timeout}m Seuil={threshold*100:.0f}%...")
        
        results = run_backtest(tp, sl, timeout, threshold)
        results["tp"] = tp
        results["sl"] = sl
        results["timeout"] = timeout
        results["threshold"] = threshold
        
        if "error" in results:
            print(f"   ❌ Erreur: {results['error']}")
        else:
            pnl = results.get("pnl_pct", 0)
            wr = results.get("win_rate", 0)
            trades = results.get("trades", 0)
            emoji = "✅" if pnl > 0 else "❌"
            print(f"   {emoji} PnL: {pnl:+.1f}% | WR: {wr:.1f}% | Trades: {trades}")
        
        all_results.append(results)
    
    # Trier par PnL
    all_results.sort(key=lambda x: x.get("pnl_pct", -999), reverse=True)
    
    # Afficher le tableau récapitulatif
    print()
    print("=" * 80)
    print("📊 RÉCAPITULATIF")
    print("=" * 80)
    print()
    print(f"{'#':<3} {'TP':>5} {'SL':>5} {'T/O':>5} {'Seuil':>6} {'Trades':>7} {'WR':>6} {'PnL':>8} {'MaxDD':>6} {'PF':>5} {'TP%':>5} {'SL%':>5} {'TO%':>5}")
    print("-" * 80)
    
    for i, r in enumerate(all_results, 1):
        if "error" in r:
            print(f"{i:<3} Erreur")
            continue
        
        emoji = "✅" if r.get("pnl_pct", 0) > 0 else "❌"
        
        print(f"{emoji}{i:<2} {r['tp']:>4.1f}% {r['sl']:>4.1f}% {r['timeout']:>4}m {r['threshold']*100:>5.0f}% "
              f"{r.get('trades', 0):>7} {r.get('win_rate', 0):>5.1f}% {r.get('pnl_pct', 0):>+7.1f}% "
              f"{r.get('max_dd', 0):>5.1f}% {r.get('profit_factor', 0):>4.2f} "
              f"{r.get('tp_pct', 0):>4.0f}% {r.get('sl_pct', 0):>4.0f}% {r.get('to_pct', 0):>4.0f}%")
    
    # Meilleure config
    print()
    print("-" * 80)
    
    profitable = [r for r in all_results if r.get("pnl_pct", 0) > 0]
    
    if profitable:
        best = profitable[0]
        print(f"🏆 MEILLEURE CONFIGURATION:")
        print(f"   TP: {best['tp']}%")
        print(f"   SL: {best['sl']}%")
        print(f"   Timeout: {best['timeout']} min")
        print(f"   Seuil: {best['threshold']*100:.0f}%")
        print(f"   → PnL: {best.get('pnl_pct', 0):+.1f}%")
        print(f"   → Win Rate: {best.get('win_rate', 0):.1f}%")
        print(f"   → Trades: {best.get('trades', 0)}")
    else:
        print("❌ AUCUNE CONFIGURATION RENTABLE")
        print()
        print("Recommandations:")
        print("1. Le modèle actuel ne prédit pas efficacement les mouvements de prix")
        print("2. Envisager de changer l'approche de labeling")
        print("3. Tester avec plus de features (order book réel, sentiment)")
        print("4. Essayer un autre type de modèle (LSTM, Transformer)")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()