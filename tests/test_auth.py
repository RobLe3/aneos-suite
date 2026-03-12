"""
Phase 21D tests — JWT authentication (G-031 partial).

Tests that:
1. Mock tokens still work in dev mode (ANEOS_ENV=development, no SECRET_KEY).
2. create_access_token raises RuntimeError when ANEOS_SECRET_KEY is unset.
3. Real JWT encode/decode round-trip works when SECRET_KEY is set.
4. Invalid tokens are rejected (return None from _decode_bearer_token).
"""

import os
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auth_manager():
    from aneos_api.auth import AuthManager
    return AuthManager({})


# ---------------------------------------------------------------------------
# Test 1 — mock tokens work in dev without SECRET_KEY
# ---------------------------------------------------------------------------

def test_mock_tokens_still_work_in_dev_without_secret(monkeypatch):
    """In dev mode with no SECRET_KEY, hardcoded mock tokens must be accepted."""
    monkeypatch.setenv('ANEOS_ENV', 'development')
    monkeypatch.delenv('ANEOS_SECRET_KEY', raising=False)
    import aneos_api.auth as auth_mod
    monkeypatch.setattr(auth_mod, '_JWT_SECRET', '')

    mgr = _auth_manager()
    result = mgr.authenticate_bearer_token('mock_admin_token')
    assert result is not None, "mock_admin_token should be accepted in dev mode without SECRET_KEY"
    assert result.get('username') == 'admin'


# ---------------------------------------------------------------------------
# Test 2 — create_access_token raises without SECRET_KEY
# ---------------------------------------------------------------------------

def test_create_access_token_raises_without_secret_key(monkeypatch):
    """create_access_token must raise RuntimeError when ANEOS_SECRET_KEY is not set."""
    import aneos_api.auth as auth_mod
    monkeypatch.setattr(auth_mod, '_JWT_SECRET', '')
    with pytest.raises(RuntimeError, match='ANEOS_SECRET_KEY'):
        auth_mod.create_access_token('user_001', 'viewer')


# ---------------------------------------------------------------------------
# Test 3 — real JWT encode/decode round-trip
# ---------------------------------------------------------------------------

def test_real_jwt_decode_roundtrip(monkeypatch):
    """With a set SECRET_KEY, create_access_token + _decode_bearer_token must round-trip."""
    import aneos_api.auth as auth_mod
    if not auth_mod._HAS_JOSE:
        pytest.skip('python-jose not installed')

    secret = 'test-secret-key-for-unit-tests-only-do-not-use-in-prod'
    monkeypatch.setattr(auth_mod, '_JWT_SECRET', secret)

    token = auth_mod.create_access_token('admin_001', 'admin')
    assert isinstance(token, str) and len(token) > 20

    payload = auth_mod._decode_bearer_token(token)
    assert payload is not None, "Valid token should decode successfully"
    assert payload['sub'] == 'admin_001'
    assert payload['role'] == 'admin'


# ---------------------------------------------------------------------------
# Test 4 — invalid token returns None
# ---------------------------------------------------------------------------

def test_invalid_token_returns_none(monkeypatch):
    """_decode_bearer_token must return None for garbage input."""
    import aneos_api.auth as auth_mod
    if not auth_mod._HAS_JOSE:
        pytest.skip('python-jose not installed')

    secret = 'test-secret-key-for-unit-tests-only-do-not-use-in-prod'
    monkeypatch.setattr(auth_mod, '_JWT_SECRET', secret)

    result = auth_mod._decode_bearer_token('not.a.valid.jwt.token')
    assert result is None, "Invalid token must return None"


# ---------------------------------------------------------------------------
# Test 5 — auth_endpoints module importable and router defined
# ---------------------------------------------------------------------------

def test_auth_endpoints_router_importable():
    """auth_endpoints must be importable and expose a 'router' attribute."""
    from aneos_api.endpoints import auth_endpoints
    assert hasattr(auth_endpoints, 'router'), "auth_endpoints must expose 'router'"
