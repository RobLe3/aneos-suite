"""monitoring.alerts must load even when ml.prediction is absent."""

import importlib
import sys


def test_alerts_imports_without_ml(monkeypatch):
    """monitoring.alerts must load even when ml.prediction is absent."""
    monkeypatch.setitem(sys.modules, "aneos_core.ml.prediction", None)
    if "aneos_core.monitoring.alerts" in sys.modules:
        del sys.modules["aneos_core.monitoring.alerts"]
    mod = importlib.import_module("aneos_core.monitoring.alerts")
    assert mod is not None
