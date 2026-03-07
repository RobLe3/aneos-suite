"""Network integration tests — run with: pytest -m network
Skip in CI with: pytest -m 'not network'
"""
import pytest


@pytest.mark.network
def test_neodys_apophis():
    from aneos_core.config.settings import APIConfig
    from aneos_core.data.sources.neodys import NEODySSource
    r = NEODySSource(APIConfig())._make_request("99942")
    assert r is not None and 0.8 < r["a"] < 1.1


@pytest.mark.network
def test_mpc_apophis():
    from aneos_core.config.settings import APIConfig
    from aneos_core.data.sources.mpc import MPCSource
    r = MPCSource(APIConfig())._make_request("99942")
    assert r is not None and 0.8 < r["a"] < 1.1
