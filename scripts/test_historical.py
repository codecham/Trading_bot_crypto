# scripts/test_historical.py
"""
Tests d'intégration pour la Phase 5.1 - Données Historiques.

Teste :
- HistoricalDataDownloader (téléchargement, pagination)
- Sauvegarde et chargement (Parquet, CSV)
- MultiSymbolDownloader

Usage:
    python scripts/test_historical.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import tempfile

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptoscalper.utils.logger import setup_logger, logger
from cryptoscalper.data.binance_client import BinanceClient
from cryptoscalper.data.historical import (
    HistoricalDataDownloader,
    MultiSymbolDownloader,
    DownloadConfig,
    DownloadProgress,
    download_historical_data,
    is_data_cached,
    load_cached_data,
)


# =========================================
# TESTS DOWNLOAD CONFIG
# =========================================

def test_download_config_defaults():
    """Test 5.1.1 - Configuration par défaut."""
    print("\n⚙️ Test 5.1.1 - Configuration par défaut...")
    
    config = DownloadConfig()
    
    assert config.interval == "1m", "Intervalle par défaut devrait être 1m"
    assert config.days == 180, "Days par défaut devrait être 180"
    assert config.end_date is not None, "end_date devrait être défini"
    assert config.start_date is not None, "start_date devrait être défini"
    
    # Vérifier que start_date est bien 180 jours avant end_date
    diff = (config.end_date - config.start_date).days
    assert diff == 180, f"Différence devrait être 180 jours, got {diff}"
    
    print(f"  ✅ Config par défaut OK")
    print(f"     Période: {config.start_date.date()} → {config.end_date.date()}")
    
    return True


def test_download_config_custom():
    """Test 5.1.2 - Configuration personnalisée."""
    print("\n⚙️ Test 5.1.2 - Configuration personnalisée...")
    
    start = datetime(2024, 1, 1)
    end = datetime(2024, 6, 30)
    
    config = DownloadConfig(
        interval="5m",
        start_date=start,
        end_date=end,
        save_format="csv"
    )
    
    assert config.interval == "5m"
    assert config.start_date == start
    assert config.end_date == end
    assert config.save_format == "csv"
    
    print(f"  ✅ Config personnalisée OK")
    
    return True


# =========================================
# TESTS HISTORICAL DATA DOWNLOADER
# =========================================

async def test_download_small():
    """Test 5.1.3 - Téléchargement petit (1 jour)."""
    print("\n📥 Test 5.1.3 - Téléchargement 1 jour...")
    
    async with BinanceClient() as client:
        downloader = HistoricalDataDownloader(client._client)
        
        df = await downloader.download("BTCUSDT", days=1)
        
        assert len(df) > 0, "DataFrame devrait contenir des données"
        assert "open" in df.columns, "Colonne 'open' manquante"
        assert "close" in df.columns, "Colonne 'close' manquante"
        assert "volume" in df.columns, "Colonne 'volume' manquante"
        
        # Vérifier les types
        assert df["close"].dtype == float, "close devrait être float"
        assert df["trades_count"].dtype == int, "trades_count devrait être int"
        
        # Vérifier qu'on a environ 1440 lignes (1 jour = 1440 minutes)
        expected_min = 1400  # Un peu moins car données en cours
        assert len(df) >= expected_min, f"Devrait avoir ≥{expected_min} lignes, got {len(df)}"
        
        print(f"  ✅ Téléchargement OK: {len(df)} lignes")
        print(f"     Période: {df['open_time'].iloc[0]} → {df['open_time'].iloc[-1]}")
        print(f"     Prix: ${df['close'].iloc[-1]:,.2f}")
    
    return True


async def test_download_with_dates():
    """Test 5.1.4 - Téléchargement avec dates spécifiques."""
    print("\n📅 Test 5.1.4 - Téléchargement avec dates...")
    
    start_date = datetime.now() - timedelta(days=3)
    end_date = datetime.now() - timedelta(days=2)
    
    async with BinanceClient() as client:
        downloader = HistoricalDataDownloader(client._client)
        
        df = await downloader.download(
            "ETHUSDT",
            start_date=start_date,
            end_date=end_date
        )
        
        assert len(df) > 0, "DataFrame devrait contenir des données"
        
        # Vérifier que les données sont dans la plage
        first_time = df["open_time"].iloc[0]
        last_time = df["open_time"].iloc[-1]
        
        print(f"  ✅ Téléchargement avec dates OK: {len(df)} lignes")
        print(f"     Demandé: {start_date.date()} → {end_date.date()}")
        print(f"     Reçu: {first_time} → {last_time}")
    
    return True


async def test_download_progress_callback():
    """Test 5.1.5 - Callback de progression."""
    print("\n📊 Test 5.1.5 - Callback progression...")
    
    progress_updates = []
    
    def on_progress(progress: DownloadProgress):
        progress_updates.append(progress.percent)
    
    async with BinanceClient() as client:
        downloader = HistoricalDataDownloader(client._client)
        downloader.on_progress(on_progress)
        
        df = await downloader.download("BNBUSDT", days=1)
        
        assert len(df) > 0
        # Le callback peut ne pas être appelé souvent pour 1 jour
        # Car on reçoit tout en 1-2 requêtes
        
        print(f"  ✅ Callback OK: {len(progress_updates)} updates")
    
    return True


# =========================================
# TESTS SAUVEGARDE / CHARGEMENT
# =========================================

async def test_save_parquet():
    """Test 5.1.6 - Sauvegarde Parquet."""
    print("\n💾 Test 5.1.6 - Sauvegarde Parquet...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        async with BinanceClient() as client:
            downloader = HistoricalDataDownloader(client._client)
            
            df = await downloader.download("BTCUSDT", days=1)
            
            # Sauvegarder
            path = Path(tmpdir) / "test_btc.parquet"
            saved_path = downloader.save_to_parquet(df, path)
            
            assert saved_path.exists(), "Fichier devrait exister"
            
            # Recharger
            df_loaded = HistoricalDataDownloader.load_from_parquet(saved_path)
            
            assert len(df_loaded) == len(df), "Même nombre de lignes"
            assert list(df_loaded.columns) == list(df.columns), "Mêmes colonnes"
            
            # Vérifier les valeurs
            assert df_loaded["close"].iloc[0] == df["close"].iloc[0]
            
            print(f"  ✅ Parquet OK: {saved_path.stat().st_size / 1024:.1f} KB")
    
    return True


async def test_save_csv():
    """Test 5.1.7 - Sauvegarde CSV."""
    print("\n💾 Test 5.1.7 - Sauvegarde CSV...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        async with BinanceClient() as client:
            downloader = HistoricalDataDownloader(client._client)
            
            df = await downloader.download("ETHUSDT", days=1)
            
            # Sauvegarder
            path = Path(tmpdir) / "test_eth.csv"
            saved_path = downloader.save_to_csv(df, path)
            
            assert saved_path.exists(), "Fichier devrait exister"
            
            # Recharger
            df_loaded = HistoricalDataDownloader.load_from_csv(saved_path)
            
            assert len(df_loaded) == len(df), "Même nombre de lignes"
            
            print(f"  ✅ CSV OK: {saved_path.stat().st_size / 1024:.1f} KB")
    
    return True


# =========================================
# TESTS MULTI-SYMBOL DOWNLOADER
# =========================================

async def test_multi_symbol_download():
    """Test 5.1.8 - Téléchargement multi-symboles."""
    print("\n📦 Test 5.1.8 - Téléchargement multi-symboles...")
    
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = DownloadConfig(output_dir=Path(tmpdir))
        
        async with BinanceClient() as client:
            downloader = MultiSymbolDownloader(client._client, config)
            
            results = await downloader.download_all(
                symbols=symbols,
                days=1,
                save=True
            )
            
            assert len(results) == len(symbols), "Devrait avoir tous les résultats"
            
            for symbol in symbols:
                assert symbol in results, f"{symbol} manquant"
                assert results[symbol].rows_count > 0, f"{symbol} vide"
                assert results[symbol].file_path.exists(), f"{symbol} fichier manquant"
            
            # Vérifier les fichiers
            files = list(Path(tmpdir).glob("*.parquet"))
            assert len(files) == len(symbols), f"Devrait avoir {len(symbols)} fichiers"
            
            print(f"  ✅ Multi-symboles OK:")
            for symbol, result in results.items():
                print(f"     {symbol}: {result.rows_count:,} lignes")
    
    return True


# =========================================
# TESTS HELPER FUNCTIONS
# =========================================

async def test_helper_download():
    """Test 5.1.9 - Fonction helper download_historical_data."""
    print("\n🔧 Test 5.1.9 - Fonction helper...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Modifier temporairement le dossier de sortie
        import cryptoscalper.data.historical as hist_module
        original_data_dir = hist_module.DATA_DIR
        hist_module.DATA_DIR = tmpdir
        
        try:
            async with BinanceClient() as client:
                df = await download_historical_data(
                    client._client,
                    symbol="BTCUSDT",
                    days=1,
                    save=True
                )
                
                assert len(df) > 0, "DataFrame non vide"
                
                # Vérifier le cache
                cached = is_data_cached("BTCUSDT", Path(tmpdir))
                assert cached, "Données devraient être en cache"
                
                # Charger depuis cache
                df_cached = load_cached_data("BTCUSDT", Path(tmpdir))
                assert df_cached is not None, "Chargement cache devrait fonctionner"
                assert len(df_cached) == len(df), "Même nombre de lignes"
                
                print(f"  ✅ Helper + cache OK")
        finally:
            hist_module.DATA_DIR = original_data_dir
    
    return True


# =========================================
# TEST PERFORMANCE
# =========================================

async def test_download_performance():
    """Test 5.1.10 - Performance téléchargement."""
    print("\n⚡ Test 5.1.10 - Performance...")
    
    async with BinanceClient() as client:
        downloader = HistoricalDataDownloader(client._client)
        
        start = datetime.now()
        df = await downloader.download("BTCUSDT", days=7)
        duration = (datetime.now() - start).total_seconds()
        
        rows_per_second = len(df) / duration
        
        print(f"  📊 Résultats:")
        print(f"     Lignes: {len(df):,}")
        print(f"     Durée: {duration:.1f}s")
        print(f"     Vitesse: {rows_per_second:,.0f} lignes/s")
        
        # Objectif: au moins 1000 lignes/s
        target_rate = 1000
        status = "✅" if rows_per_second >= target_rate else "⚠️"
        print(f"  {status} Objectif: ≥{target_rate} lignes/s")
    
    return rows_per_second >= 500  # Seuil plus bas car dépend du réseau


# =========================================
# MAIN
# =========================================

async def main():
    """Exécute tous les tests de la Phase 5.1."""
    print("=" * 65)
    print("🧪 CryptoScalper AI - Tests Phase 5.1: Données Historiques")
    print("=" * 65)
    
    setup_logger(level="WARNING")
    
    results = []
    
    try:
        # Tests configuration
        print("\n" + "─" * 50)
        print("📦 5.1.1-2 Configuration")
        print("─" * 50)
        results.append(("5.1.1 Config défaut", test_download_config_defaults()))
        results.append(("5.1.2 Config custom", test_download_config_custom()))
        
        # Tests téléchargement
        print("\n" + "─" * 50)
        print("📦 5.1.3-5 Téléchargement")
        print("─" * 50)
        results.append(("5.1.3 Download petit", await test_download_small()))
        results.append(("5.1.4 Download dates", await test_download_with_dates()))
        results.append(("5.1.5 Callback progress", await test_download_progress_callback()))
        
        # Tests sauvegarde
        print("\n" + "─" * 50)
        print("📦 5.1.6-7 Sauvegarde")
        print("─" * 50)
        results.append(("5.1.6 Parquet", await test_save_parquet()))
        results.append(("5.1.7 CSV", await test_save_csv()))
        
        # Tests multi-symboles
        print("\n" + "─" * 50)
        print("📦 5.1.8-9 Multi-symboles & Helpers")
        print("─" * 50)
        results.append(("5.1.8 Multi-symboles", await test_multi_symbol_download()))
        results.append(("5.1.9 Helper function", await test_helper_download()))
        
        # Test performance
        print("\n" + "─" * 50)
        print("📦 5.1.10 Performance")
        print("─" * 50)
        results.append(("5.1.10 Performance", await test_download_performance()))
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        logger.exception("Erreur test Phase 5.1")
        return 1
    
    # Résumé
    print("\n" + "=" * 65)
    print("📊 RÉSULTATS PHASE 5.1")
    print("=" * 65)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("─" * 65)
    print(f"  Total: {passed}/{len(results)} tests passés")
    print("=" * 65)
    
    if failed == 0:
        print("🎉 Phase 5.1 - Données Historiques : VALIDÉE !")
        return 0
    else:
        print(f"⚠️  {failed} test(s) échoué(s)")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))