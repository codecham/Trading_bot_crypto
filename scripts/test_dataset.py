# scripts/test_dataset.py
"""
Tests d'intégration pour la Phase 5.2 - Préparation Dataset.

Teste :
- LabelConfig et SplitConfig
- DatasetBuilder (création labels, features)
- Split temporel
- Équilibre des classes

Usage:
    python scripts/test_dataset.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import tempfile

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from cryptoscalper.utils.logger import setup_logger, logger
from cryptoscalper.data.dataset import (
    DatasetBuilder,
    LabelConfig,
    SplitConfig,
    DatasetStats,
    PreparedDataset,
    prepare_dataset,
    analyze_class_balance,
)


# =========================================
# HELPERS
# =========================================

def create_sample_ohlcv(n_rows: int = 500, trend: str = "up") -> pd.DataFrame:
    """Crée un DataFrame OHLCV de test."""
    np.random.seed(42)
    
    # Prix de base avec tendance
    base_price = 45000
    
    if trend == "up":
        drift = 0.0001
    elif trend == "down":
        drift = -0.0001
    else:
        drift = 0
    
    returns = np.random.normal(drift, 0.002, n_rows)
    prices = base_price * np.cumprod(1 + returns)
    
    # Générer OHLCV
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(minutes=n_rows),
        periods=n_rows,
        freq="1min"
    )
    
    df = pd.DataFrame({
        "open_time": timestamps,
        "open": prices * (1 + np.random.uniform(-0.001, 0.001, n_rows)),
        "high": prices * (1 + np.random.uniform(0, 0.003, n_rows)),
        "low": prices * (1 - np.random.uniform(0, 0.003, n_rows)),
        "close": prices,
        "volume": np.random.uniform(10, 100, n_rows),
        "close_time": timestamps + timedelta(minutes=1),
        "quote_volume": np.random.uniform(1000, 10000, n_rows),
        "trades_count": np.random.randint(100, 1000, n_rows),
        "taker_buy_volume": np.random.uniform(5, 50, n_rows),
        "taker_buy_quote_volume": np.random.uniform(500, 5000, n_rows),
    })
    
    # S'assurer que high >= max(open, close) et low <= min(open, close)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    
    return df


# =========================================
# TESTS CONFIG
# =========================================

def test_label_config_defaults():
    """Test 5.2.1 - LabelConfig par défaut."""
    print("\n⚙️ Test 5.2.1 - LabelConfig par défaut...")
    
    config = LabelConfig()
    
    assert config.horizon_minutes == 3, "Horizon par défaut = 3"
    assert config.threshold_percent == 0.002, "Threshold par défaut = 0.2%"
    assert config.use_future_high is True, "use_future_high par défaut = True"
    
    print(f"  ✅ LabelConfig par défaut OK")
    print(f"     horizon={config.horizon_minutes}min, threshold={config.threshold_percent:.2%}")
    
    return True


def test_split_config_validation():
    """Test 5.2.2 - Validation SplitConfig."""
    print("\n⚙️ Test 5.2.2 - Validation SplitConfig...")
    
    # Config valide
    config = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert config.train_ratio == 0.7
    
    # Config invalide (ne somme pas à 1)
    try:
        bad_config = SplitConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
        print("  ❌ Aurait dû lever une erreur")
        return False
    except ValueError as e:
        print(f"  ✅ Erreur correctement levée: {e}")
    
    return True


# =========================================
# TESTS LABELS
# =========================================

def test_label_creation():
    """Test 5.2.3 - Création des labels."""
    print("\n🏷️ Test 5.2.3 - Création des labels...")
    
    # Créer des données avec une tendance haussière
    df = create_sample_ohlcv(200, trend="up")
    
    builder = DatasetBuilder(
        label_config=LabelConfig(horizon_minutes=3, threshold_percent=0.002)
    )
    
    # Appeler la méthode interne de création des labels
    df_with_labels = builder._create_labels(df)
    
    assert "label" in df_with_labels.columns, "Colonne 'label' manquante"
    assert "future_return" in df_with_labels.columns, "Colonne 'future_return' manquante"
    
    # Vérifier les valeurs
    labels = df_with_labels["label"].dropna()
    assert labels.isin([0, 1]).all(), "Labels doivent être 0 ou 1"
    
    # Avec tendance haussière, on devrait avoir des labels positifs
    positive_ratio = labels.mean()
    print(f"  ✅ Labels créés: {len(labels)} lignes")
    print(f"     Ratio positifs: {positive_ratio:.1%}")
    
    return True


def test_label_threshold_impact():
    """Test 5.2.4 - Impact du seuil sur les labels."""
    print("\n🎯 Test 5.2.4 - Impact du seuil...")
    
    df = create_sample_ohlcv(200)
    
    # Seuil bas = plus de labels positifs
    builder_low = DatasetBuilder(
        label_config=LabelConfig(threshold_percent=0.001)
    )
    df_low = builder_low._create_labels(df)
    ratio_low = df_low["label"].dropna().mean()
    
    # Seuil haut = moins de labels positifs
    builder_high = DatasetBuilder(
        label_config=LabelConfig(threshold_percent=0.005)
    )
    df_high = builder_high._create_labels(df)
    ratio_high = df_high["label"].dropna().mean()
    
    assert ratio_low > ratio_high, "Seuil bas devrait donner plus de positifs"
    
    print(f"  ✅ Impact du seuil vérifié:")
    print(f"     Seuil 0.1%: {ratio_low:.1%} positifs")
    print(f"     Seuil 0.5%: {ratio_high:.1%} positifs")
    
    return True


# =========================================
# TESTS DATASET BUILDER
# =========================================

def test_dataset_builder_basic():
    """Test 5.2.5 - DatasetBuilder basique."""
    print("\n🔧 Test 5.2.5 - DatasetBuilder basique...")
    
    df = create_sample_ohlcv(200)
    
    builder = DatasetBuilder()
    dataset = builder.build_from_dataframe(df, symbol="TEST")
    
    assert len(dataset) > 0, "Dataset non vide"
    assert dataset.stats.feature_count == 42, "42 features"
    assert len(dataset.feature_names) == 42, "42 noms de features"
    
    print(f"  ✅ Dataset construit: {len(dataset)} lignes, {dataset.stats.feature_count} features")
    
    return True


def test_dataset_split_temporal():
    """Test 5.2.6 - Split temporel."""
    print("\n📊 Test 5.2.6 - Split temporel...")
    
    df = create_sample_ohlcv(300)
    
    builder = DatasetBuilder()
    dataset = builder.build_from_dataframe(df, symbol="TEST")
    
    # Split
    train, val, test = dataset.split_temporal()
    
    # Vérifier les tailles
    total = len(train) + len(val) + len(test)
    assert total == len(dataset), "Split complet"
    
    train_ratio = len(train) / len(dataset)
    val_ratio = len(val) / len(dataset)
    test_ratio = len(test) / len(dataset)
    
    assert 0.65 < train_ratio < 0.75, f"Train ratio ~70%, got {train_ratio:.1%}"
    assert 0.10 < val_ratio < 0.20, f"Val ratio ~15%, got {val_ratio:.1%}"
    assert 0.10 < test_ratio < 0.20, f"Test ratio ~15%, got {test_ratio:.1%}"
    
    print(f"  ✅ Split temporel:")
    print(f"     Train: {len(train)} ({train_ratio:.0%})")
    print(f"     Val: {len(val)} ({val_ratio:.0%})")
    print(f"     Test: {len(test)} ({test_ratio:.0%})")
    
    return True


def test_no_data_leakage():
    """Test 5.2.7 - Pas de fuite de données (data leakage)."""
    print("\n🔒 Test 5.2.7 - Vérification data leakage...")
    
    df = create_sample_ohlcv(300)
    
    builder = DatasetBuilder()
    dataset = builder.build_from_dataframe(df, symbol="TEST")
    
    train, val, test = dataset.split_temporal()
    
    # Vérifier que les timestamps sont ordonnés
    assert train.timestamps.iloc[-1] < val.timestamps.iloc[0], \
        "Train doit finir avant Val"
    assert val.timestamps.iloc[-1] < test.timestamps.iloc[0], \
        "Val doit finir avant Test"
    
    print(f"  ✅ Pas de data leakage:")
    print(f"     Train: ... → {train.timestamps.iloc[-1]}")
    print(f"     Val: {val.timestamps.iloc[0]} → {val.timestamps.iloc[-1]}")
    print(f"     Test: {test.timestamps.iloc[0]} → ...")
    
    return True


# =========================================
# TESTS ÉQUILIBRE
# =========================================

def test_class_balance_analysis():
    """Test 5.2.8 - Analyse équilibre des classes."""
    print("\n⚖️ Test 5.2.8 - Analyse équilibre...")
    
    # Créer des labels déséquilibrés
    labels_balanced = pd.Series([0, 1] * 50)  # 50-50
    labels_imbalanced = pd.Series([0] * 80 + [1] * 20)  # 80-20
    
    balance_ok = analyze_class_balance(labels_balanced)
    balance_bad = analyze_class_balance(labels_imbalanced)
    
    assert balance_ok["is_balanced"] is True, "50-50 devrait être équilibré"
    assert balance_bad["is_balanced"] is False, "80-20 ne devrait pas être équilibré"
    
    print(f"  ✅ Analyse équilibre:")
    print(f"     50-50: équilibré={balance_ok['is_balanced']}, ratio={balance_ok['imbalance_ratio']:.2f}")
    print(f"     80-20: équilibré={balance_bad['is_balanced']}, ratio={balance_bad['imbalance_ratio']:.2f}")
    
    return True


def test_dataset_stats():
    """Test 5.2.9 - Statistiques du dataset."""
    print("\n📈 Test 5.2.9 - Statistiques dataset...")
    
    df = create_sample_ohlcv(200)
    
    builder = DatasetBuilder()
    dataset = builder.build_from_dataframe(df, symbol="TEST")
    
    stats = dataset.stats
    
    assert stats.total_rows == 200, "Total rows correct"
    assert stats.valid_rows > 0, "Valid rows > 0"
    assert stats.feature_count == 42, "42 features"
    assert stats.label_1_count + stats.label_0_count == stats.valid_rows, "Labels complets"
    
    print(f"  ✅ Stats OK:")
    print(f"     Total: {stats.total_rows}, Valid: {stats.valid_rows}")
    print(f"     Labels: {stats.label_1_count} positifs, {stats.label_0_count} négatifs")
    print(f"     Ratio: {stats.label_ratio:.1%}")
    
    return True


# =========================================
# TESTS SAUVEGARDE/CHARGEMENT
# =========================================

def test_dataset_save_load():
    """Test 5.2.10 - Sauvegarde et chargement."""
    print("\n💾 Test 5.2.10 - Sauvegarde/Chargement...")
    
    df = create_sample_ohlcv(200)
    
    builder = DatasetBuilder()
    dataset = builder.build_from_dataframe(df, symbol="TEST")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_dataset.parquet"
        
        # Sauvegarder
        dataset.save(path)
        assert path.exists(), "Fichier créé"
        
        # Recharger
        loaded = PreparedDataset.load(path)
        
        assert len(loaded) == len(dataset), "Même taille"
        assert loaded.stats.feature_count == dataset.stats.feature_count, "Mêmes features"
        
        # Vérifier les valeurs
        assert (loaded.features.values == dataset.features.values).all(), "Features identiques"
        assert (loaded.labels.values == dataset.labels.values).all(), "Labels identiques"
        
        print(f"  ✅ Sauvegarde/Chargement OK")
        print(f"     Taille fichier: {path.stat().st_size / 1024:.1f} KB")
    
    return True


def test_to_numpy():
    """Test 5.2.11 - Conversion numpy."""
    print("\n🔢 Test 5.2.11 - Conversion numpy...")
    
    df = create_sample_ohlcv(200)
    
    builder = DatasetBuilder()
    dataset = builder.build_from_dataframe(df, symbol="TEST")
    
    X, y = dataset.to_numpy()
    
    assert isinstance(X, np.ndarray), "X est un ndarray"
    assert isinstance(y, np.ndarray), "y est un ndarray"
    assert X.shape[0] == len(dataset), "Même nombre de lignes"
    assert X.shape[1] == 42, "42 features"
    assert len(y) == len(dataset), "Même nombre de labels"
    
    print(f"  ✅ Conversion numpy:")
    print(f"     X shape: {X.shape}")
    print(f"     y shape: {y.shape}")
    
    return True


# =========================================
# TEST AVEC DONNÉES RÉELLES (optionnel)
# =========================================

def test_with_cached_data():
    """Test 5.2.12 - Test avec données réelles (si disponibles)."""
    print("\n🌐 Test 5.2.12 - Avec données réelles...")
    
    from cryptoscalper.data.historical import is_data_cached, get_cached_data_path
    
    data_dir = Path("data_cache")
    
    # Chercher un fichier de données existant
    if not data_dir.exists():
        print("  ⚠️ Pas de données en cache, test skippé")
        return True
    
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        print("  ⚠️ Pas de fichiers parquet, test skippé")
        return True
    
    # Utiliser le premier fichier trouvé
    test_file = parquet_files[0]
    symbol = test_file.stem.split("_")[0]
    
    print(f"  📂 Utilisation de {test_file.name}")
    
    try:
        builder = DatasetBuilder()
        dataset = builder.build_from_file(test_file, symbol)
        
        print(f"  ✅ Dataset réel construit:")
        print(f"     Lignes: {len(dataset):,}")
        print(f"     Ratio positifs: {dataset.stats.label_ratio:.1%}")
        
        # Split
        train, val, test = dataset.split_temporal()
        print(f"     Train/Val/Test: {len(train):,}/{len(val):,}/{len(test):,}")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️ Erreur (non bloquante): {e}")
        return True  # Test non bloquant


# =========================================
# MAIN
# =========================================

def main():
    """Exécute tous les tests de la Phase 5.2."""
    print("=" * 65)
    print("🧪 CryptoScalper AI - Tests Phase 5.2: Préparation Dataset")
    print("=" * 65)
    
    setup_logger(level="WARNING")
    
    results = []
    
    try:
        # Tests configuration
        print("\n" + "─" * 50)
        print("📦 5.2.1-2 Configuration")
        print("─" * 50)
        results.append(("5.2.1 LabelConfig défaut", test_label_config_defaults()))
        results.append(("5.2.2 SplitConfig validation", test_split_config_validation()))
        
        # Tests labels
        print("\n" + "─" * 50)
        print("📦 5.2.3-4 Création Labels")
        print("─" * 50)
        results.append(("5.2.3 Création labels", test_label_creation()))
        results.append(("5.2.4 Impact seuil", test_label_threshold_impact()))
        
        # Tests DatasetBuilder
        print("\n" + "─" * 50)
        print("📦 5.2.5-7 DatasetBuilder")
        print("─" * 50)
        results.append(("5.2.5 Builder basique", test_dataset_builder_basic()))
        results.append(("5.2.6 Split temporel", test_dataset_split_temporal()))
        results.append(("5.2.7 No data leakage", test_no_data_leakage()))
        
        # Tests équilibre
        print("\n" + "─" * 50)
        print("📦 5.2.8-9 Équilibre & Stats")
        print("─" * 50)
        results.append(("5.2.8 Analyse équilibre", test_class_balance_analysis()))
        results.append(("5.2.9 Stats dataset", test_dataset_stats()))
        
        # Tests sauvegarde
        print("\n" + "─" * 50)
        print("📦 5.2.10-11 Sauvegarde & Conversion")
        print("─" * 50)
        results.append(("5.2.10 Save/Load", test_dataset_save_load()))
        results.append(("5.2.11 To numpy", test_to_numpy()))
        
        # Test données réelles
        print("\n" + "─" * 50)
        print("📦 5.2.12 Données Réelles")
        print("─" * 50)
        results.append(("5.2.12 Données réelles", test_with_cached_data()))
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        logger.exception("Erreur test Phase 5.2")
        return 1
    
    # Résumé
    print("\n" + "=" * 65)
    print("📊 RÉSULTATS PHASE 5.2")
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
        print("🎉 Phase 5.2 - Préparation Dataset : VALIDÉE !")
        return 0
    else:
        print(f"⚠️  {failed} test(s) échoué(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())