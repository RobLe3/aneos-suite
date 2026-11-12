"""Keep provider scheduling state isolated from users and other test cases."""
import pytest


@pytest.fixture(autouse=True)
def isolated_provider_scheduling(monkeypatch, tmp_path, request):
    if request.node.get_closest_marker("network") is not None:
        return
    monkeypatch.setenv('ANEOS_PROVIDER_STATE_DIR', str(tmp_path / 'provider-coordination'))
    # Explicit pacing tests set their own interval; unrelated parser tests do
    # not need wall-clock sleeps or share a previous test's outage state.
    monkeypatch.setenv('ANEOS_JPL_MIN_INTERVAL', '0')
