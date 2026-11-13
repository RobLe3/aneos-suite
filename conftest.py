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


@pytest.fixture(autouse=True)
def deny_unmarked_network(monkeypatch, request):
    """An offline selection must not silently query a live source on a stale seam."""
    if request.node.get_closest_marker('network') is not None:
        yield
        return
    import socket
    attempts = []
    def denied(*args, **kwargs):
        attempts.append('unmarked network attempt')
        raise AssertionError('Live network requires an explicit pytest network marker')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'getaddrinfo', denied)
    yield
    assert not attempts, 'Offline test attempted live network; inject the actual source seam or mark an authorized network integration test'
