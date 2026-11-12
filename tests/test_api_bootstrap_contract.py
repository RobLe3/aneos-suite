"""Explicit full-API schema startup, not import-time database mutation."""
from unittest.mock import Mock,AsyncMock
import pytest


@pytest.mark.asyncio
async def test_full_api_startup_confirms_database_before_services_and_shutdown(monkeypatch):
    from aneos_api import app,database
    db=Mock(return_value=True)
    monkeypatch.setattr(database,'init_database',db)
    services=Mock(initialize_services=AsyncMock(),shutdown_services=AsyncMock())
    monkeypatch.setattr(app,'get_aneos_app',lambda:services)
    async with app.lifespan(Mock()):
        db.assert_called_once()
        services.initialize_services.assert_awaited_once()
    services.shutdown_services.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('unconfirmed',[False,None,'yes'])
async def test_full_api_schema_failure_cannot_claim_success_or_run_services(monkeypatch,unconfirmed):
    from aneos_api import app,database
    monkeypatch.setattr(database,'init_database',Mock(return_value=unconfirmed))
    services=Mock(initialize_services=AsyncMock(),shutdown_services=AsyncMock())
    monkeypatch.setattr(app,'get_aneos_app',lambda:services)
    with pytest.raises(RuntimeError,match='schema unavailable'):
        async with app.lifespan(Mock()):raise AssertionError('startup must not yield')
    services.initialize_services.assert_not_called()


@pytest.mark.asyncio
async def test_started_api_services_are_shutdown_even_when_lifespan_body_raises(monkeypatch):
    from aneos_api import app,database
    monkeypatch.setattr(database,'init_database',Mock(return_value=True))
    services=Mock(initialize_services=AsyncMock(),shutdown_services=AsyncMock())
    monkeypatch.setattr(app,'get_aneos_app',lambda:services)
    with pytest.raises(RuntimeError,match='synthetic body failure'):
        async with app.lifespan(Mock()):raise RuntimeError('synthetic body failure')
    services.shutdown_services.assert_awaited_once()


@pytest.mark.asyncio
async def test_partial_service_startup_failure_still_runs_cleanup(monkeypatch):
    from aneos_api import app,database
    monkeypatch.setattr(database,'init_database',Mock(return_value=True))
    services=Mock(initialize_services=AsyncMock(side_effect=RuntimeError('partial startup')),
                  shutdown_services=AsyncMock())
    monkeypatch.setattr(app,'get_aneos_app',lambda:services)
    with pytest.raises(RuntimeError,match='partial startup'):
        async with app.lifespan(Mock()):raise AssertionError('must not yield')
    services.shutdown_services.assert_awaited_once()
