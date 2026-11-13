"""
Phase 23 tests — SA 2.0, Auth DB wiring, ADR-032 closure, warning cleanup.

Tests cover:
  - 23A: SQLAlchemy 2.0 declarative_base import (no MovedIn20Warning)
  - 23B: _load_users_from_db, authenticate_bearer_token DB fallback, create_user API key
  - 23C: WebhookNotificationChannel aiohttp guard, AlertManager ML debug log
"""
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 23A — SQLAlchemy 2.0 import: no MovedIn20Warning from sqlalchemy.orm path
# ---------------------------------------------------------------------------

def test_declarative_base_uses_orm_import():
    """Importing declarative_base from sqlalchemy.orm must emit no DeprecationWarning."""
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            from sqlalchemy.orm import declarative_base  # noqa: F401
        # Reached here — no warning raised
    except ImportError:
        pytest.skip("SQLAlchemy not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_db_user(user_id="db_user_001", username="dbuser",
                       email="db@aneos.local", role="analyst",
                       api_keys=None, is_active=True):
    u = MagicMock()
    u.user_id = user_id
    u.username = username
    u.email = email
    u.role = role
    u.api_keys = api_keys if api_keys is not None else ["aneos_testkey123"]
    u.is_active = is_active
    u.created_at = None
    u.last_login = None
    return u


# ---------------------------------------------------------------------------
# 23B — _load_users_from_db
# (SessionLocal is imported locally inside _load_users_from_db, so we patch
#  at aneos_api.database.* where the function-level import resolves from.)
# ---------------------------------------------------------------------------

def test_load_users_from_db_populates_api_key_map():
    """_load_users_from_db() inserts DB users' keys into API_KEY_MAP."""
    from aneos_api import auth as auth_mod

    fake_user = _make_fake_db_user(api_keys=["aneos_unique_key_xyz"])
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [fake_user]

    with patch("aneos_api.database.SessionLocal", return_value=mock_db), \
         patch("aneos_api.database.HAS_SQLALCHEMY", True):
        auth_mod.API_KEY_MAP.pop("aneos_unique_key_xyz", None)
        auth_mod._load_users_from_db()

    assert "aneos_unique_key_xyz" in auth_mod.API_KEY_MAP
    assert auth_mod.API_KEY_MAP["aneos_unique_key_xyz"]["username"] == "dbuser"


def test_load_users_from_db_is_idempotent():
    """Calling _load_users_from_db() twice does not duplicate keys."""
    from aneos_api import auth as auth_mod

    key = "aneos_idempotent_key"
    fake_user = _make_fake_db_user(api_keys=[key])
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [fake_user]

    with patch("aneos_api.database.SessionLocal", return_value=mock_db), \
         patch("aneos_api.database.HAS_SQLALCHEMY", True):
        auth_mod.API_KEY_MAP.pop(key, None)
        auth_mod._load_users_from_db()
        auth_mod._load_users_from_db()  # second call — key already present, skipped

    assert auth_mod.API_KEY_MAP[key]["username"] == "dbuser"


def test_load_users_from_db_handles_no_sqlalchemy():
    """_load_users_from_db() is a no-op when HAS_SQLALCHEMY is False."""
    from aneos_api import auth as auth_mod

    before = set(auth_mod.API_KEY_MAP.keys())
    with patch("aneos_api.database.HAS_SQLALCHEMY", False):
        auth_mod._load_users_from_db()  # should not raise, no keys added

    assert set(auth_mod.API_KEY_MAP.keys()) == before


def test_authenticate_bearer_db_fallback():
    """JWT sub not in MOCK_USERS is resolved via DB fallback."""
    from aneos_api.auth import AuthManager
    import aneos_api.auth as auth_mod

    mgr = AuthManager({})
    fake_user = _make_fake_db_user(user_id="db_only_user_23", username="dbonly23")
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = fake_user

    with patch.object(auth_mod, "_decode_bearer_token",
                      return_value={"sub": "db_only_user_23"}), \
         patch("aneos_api.database.SessionLocal", return_value=mock_db), \
         patch("aneos_api.database.HAS_SQLALCHEMY", True):
        result = mgr.authenticate_bearer_token("fake.jwt.token")

    assert result is not None
    assert result["username"] == "dbonly23"


from tests.test_durable_network_api import api


@pytest.mark.asyncio
async def test_create_user_api_endpoint_registers_api_key(api,monkeypatch):
    """Verified HTTP admin persists in the same DB and registers the actual key."""
    import aneos_api.auth as auth_mod
    from aneos_api.database import User
    client,_,sessions,*_=api
    key='synthetic-phase23-admin-'+'p'*40
    monkeypatch.setenv('ANEOS_ADMIN_API_KEY',key)
    before=set(auth_mod.API_KEY_MAP)
    response=client.post('/api/v1/admin/users',headers={'X-API-Key':key},json={
        'username':'newuser23b','email':'new23b@example.invalid',
        'password':'synthetic-test-password-only','role':'viewer'})
    assert response.status_code==200,response.text
    with sessions() as db:
        user=db.query(User).filter_by(username='newuser23b').one()
        generated=user.api_keys[0]
        assert generated.startswith('aneos_') and generated not in before
        assert auth_mod.API_KEY_MAP[generated]['user_id']==user.user_id==response.json()['user_id']
    assert set(auth_mod.API_KEY_MAP)-before=={generated}


# ---------------------------------------------------------------------------
# 23C — WebhookNotificationChannel aiohttp guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_webhook_channel_returns_false_without_aiohttp():
    """WebhookNotificationChannel.send_notification() returns False when aiohttp absent."""
    from aneos_core.monitoring.alerts import (
        WebhookNotificationChannel, Alert, AlertType, AlertLevel
    )

    channel = WebhookNotificationChannel(
        channel_id="test_webhook",
        webhook_url="http://localhost:9999/webhook",
    )
    fake_alert = Alert(
        alert_id="test_001",
        rule_id="r1",
        alert_type=AlertType.SYSTEM_ERROR,
        alert_level=AlertLevel.LOW,
        title="Test",
        message="Test alert",
        timestamp=datetime.now(),
    )

    with patch.dict("sys.modules", {"aiohttp": None}):
        result = await channel.send_notification(fake_alert)

    assert result is False


@pytest.mark.asyncio
async def test_webhook_channel_logs_warning_without_aiohttp(caplog):
    """WebhookNotificationChannel logs a warning when aiohttp is absent."""
    from aneos_core.monitoring.alerts import (
        WebhookNotificationChannel, Alert, AlertType, AlertLevel
    )

    channel = WebhookNotificationChannel(
        channel_id="test_webhook_warn",
        webhook_url="http://localhost:9999/webhook",
    )
    fake_alert = Alert(
        alert_id="test_002",
        rule_id="r1",
        alert_type=AlertType.SYSTEM_ERROR,
        alert_level=AlertLevel.LOW,
        title="Test",
        message="Test alert",
        timestamp=datetime.now(),
    )

    with caplog.at_level(logging.WARNING, logger="aneos_core.monitoring.alerts"):
        with patch.dict("sys.modules", {"aiohttp": None}):
            await channel.send_notification(fake_alert)

    assert any("aiohttp" in r.message for r in caplog.records)


def test_ml_alerts_debug_log_when_unavailable(caplog):
    """check_anomaly_alert emits DEBUG log when _HAS_ML_ALERTS is False."""
    import aneos_core.monitoring.alerts as alerts_mod
    from aneos_core.monitoring.alerts import AlertManager

    mgr = AlertManager()

    with patch.object(alerts_mod, "_HAS_ML_ALERTS", False):
        with caplog.at_level(logging.DEBUG, logger="aneos_core.monitoring.alerts"):
            mgr.check_anomaly_alert(MagicMock(), MagicMock())

    assert any("ML alert subsystem unavailable" in r.message for r in caplog.records)
