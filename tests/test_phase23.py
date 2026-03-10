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


@pytest.mark.asyncio
async def test_create_user_api_endpoint_registers_api_key():
    """Admin create_user generates an API key and registers it in API_KEY_MAP."""
    import aneos_api.auth as auth_mod
    from aneos_api.endpoints.admin import create_user
    from aneos_api.models import CreateUserRequest

    # Build a mock DB that simulates commit/refresh
    class FakeDB:
        def __init__(self):
            self._added = None

        def add(self, obj):
            # Simulate DB assigning user_id and created_at
            if not obj.user_id:
                obj.user_id = "new_usr_23"
            obj.created_at = datetime.now()
            self._added = obj

        def commit(self):
            pass

        def refresh(self, obj):
            pass

        def close(self):
            pass

    request = CreateUserRequest(
        username="newuser23b",
        email="new23b@aneos.local",
        password="securepassword123",
        role="viewer",
    )
    current_user = {"username": "admin", "role": "admin", "user_id": "admin_001"}

    initial_keys = set(auth_mod.API_KEY_MAP.keys())

    with patch("aneos_api.database.SessionLocal", FakeDB), \
         patch("aneos_api.endpoints.admin.HAS_SQLALCHEMY", True), \
         patch("aneos_api.database.HAS_SQLALCHEMY", True):
        await create_user(request, current_user)

    new_keys = set(auth_mod.API_KEY_MAP.keys()) - initial_keys
    assert len(new_keys) == 1, f"Expected 1 new API key, got: {new_keys}"
    assert list(new_keys)[0].startswith("aneos_")


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
