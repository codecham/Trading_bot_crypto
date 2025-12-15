# scripts/test_setup.py
"""
Script de test pour vérifier que la configuration est correcte.

Usage:
    python scripts/test_setup.py
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Teste que tous les imports fonctionnent."""
    print("🔍 Test des imports...")
    
    try:
        from cryptoscalper.config.settings import Settings, get_settings
        print("  ✅ settings.py importé")
    except ImportError as e:
        print(f"  ❌ Erreur import settings: {e}")
        return False
    
    try:
        from cryptoscalper.config.constants import (
            BINANCE_API_URL,
            QUOTE_ASSET,
            RSI_PERIOD_DEFAULT
        )
        print("  ✅ constants.py importé")
    except ImportError as e:
        print(f"  ❌ Erreur import constants: {e}")
        return False
    
    try:
        from cryptoscalper.utils.logger import logger, setup_logger
        print("  ✅ logger.py importé")
    except ImportError as e:
        print(f"  ❌ Erreur import logger: {e}")
        return False
    
    try:
        from cryptoscalper.utils.exceptions import (
            CryptoScalperError,
            TradingError,
            InsufficientBalanceError
        )
        print("  ✅ exceptions.py importé")
    except ImportError as e:
        print(f"  ❌ Erreur import exceptions: {e}")
        return False
    
    return True


def test_settings():
    """Teste le chargement des settings."""
    print("\n🔍 Test des settings...")
    
    from cryptoscalper.config.settings import Settings
    
    try:
        settings = Settings()
        print("  ✅ Settings créés avec succès")
        print(f"     - Testnet: {settings.binance.testnet}")
        print(f"     - Capital: {settings.trading.initial_capital} USDT")
        print(f"     - Risque max/trade: {settings.trading.max_risk_per_trade:.1%}")
        print(f"     - Log level: {settings.logging.level}")
        return True
    except Exception as e:
        print(f"  ❌ Erreur création settings: {e}")
        return False


def test_logger():
    """Teste le système de logging."""
    print("\n🔍 Test du logger...")
    
    from cryptoscalper.utils.logger import logger, setup_logger
    
    try:
        setup_logger(level="DEBUG")
        logger.debug("Message DEBUG (devrait apparaître)")
        logger.info("Message INFO")
        logger.warning("Message WARNING")
        print("  ✅ Logger fonctionne correctement")
        return True
    except Exception as e:
        print(f"  ❌ Erreur logger: {e}")
        return False


def test_exceptions():
    """Teste les exceptions personnalisées."""
    print("\n🔍 Test des exceptions...")
    
    from cryptoscalper.utils.exceptions import (
        InsufficientBalanceError,
        InvalidSymbolError,
        KillSwitchActivatedError
    )
    
    try:
        # Test InsufficientBalanceError
        error = InsufficientBalanceError(required=100.0, available=50.0)
        assert "50" in str(error)
        assert "100" in str(error)
        print("  ✅ InsufficientBalanceError OK")
        
        # Test InvalidSymbolError
        error = InvalidSymbolError("INVALIDUSDT")
        assert "INVALIDUSDT" in str(error)
        print("  ✅ InvalidSymbolError OK")
        
        # Test KillSwitchActivatedError
        error = KillSwitchActivatedError(drawdown_percent=0.30, max_drawdown=0.25)
        assert "KILL SWITCH" in str(error)
        print("  ✅ KillSwitchActivatedError OK")
        
        return True
    except AssertionError as e:
        print(f"  ❌ Assertion échouée: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Erreur exceptions: {e}")
        return False


def main():
    """Exécute tous les tests."""
    print("=" * 50)
    print("🧪 CryptoScalper AI - Test de Configuration")
    print("=" * 50)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Settings", test_settings()))
    results.append(("Logger", test_logger()))
    results.append(("Exceptions", test_exceptions()))
    
    print("\n" + "=" * 50)
    print("📊 RÉSULTATS")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 Tous les tests sont passés ! Configuration OK.")
        return 0
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
        return 1


if __name__ == "__main__":
    sys.exit(main())