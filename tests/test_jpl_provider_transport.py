"""Provider coordination controls: no live requests or scientific calibration."""
import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
import requests


def transport():
    from aneos_core.data import provider_http
    return provider_http


@pytest.fixture(autouse=True)
def isolated(monkeypatch,tmp_path):
    import socket
    monkeypatch.setenv('ANEOS_PROVIDER_STATE_DIR',str(tmp_path/'provider-state'))
    monkeypatch.setenv('ANEOS_JPL_MIN_INTERVAL','0')
    monkeypatch.setattr(socket.socket,'connect',Mock(side_effect=AssertionError('external socket forbidden')))


class Response:
    def __init__(self,status=200,headers=None,body=b'{"signature":{"version":"synthetic"}}'):
        self.status_code=status;self.headers=headers or {};self._content=body
        self.closed=False
    @property
    def content(self): return self._content
    def close(self): self.closed=True
    def json(self): return json.loads(self._content)
    def raise_for_status(self):
        if self.status_code>=400: raise requests.HTTPError(f'HTTP {self.status_code}')


class Clock:
    def __init__(self): self.value=1000.;self.sleeps=[]
    def time(self): return self.value
    def monotonic(self): return self.value
    def sleep(self,seconds): self.sleeps.append(seconds);self.value+=seconds


def test_known_host_matching_cannot_be_spoofed():
    p=transport()
    assert p.is_jpl_api('https://ssd-api.jpl.nasa.gov/sbdb.api')
    assert p.is_jpl_api('https://ssd.jpl.nasa.gov/api/horizons.api')
    for url in ['https://ssd-api.jpl.nasa.gov.evil.invalid/sbdb.api','https://evil.invalid/?ssd-api.jpl.nasa.gov',
                'https://ssd.jpl.nasa.gov/not-an-api']:
        assert not p.is_jpl_api(url)


def test_application_header_and_response_are_preserved(monkeypatch):
    p=transport();reply=Response();send=Mock(return_value=reply)
    result=p.get('https://ssd-api.jpl.nasa.gov/sbdb.api',params={'sstr':'1998 KY26'},_send=send)
    assert result is reply and reply.closed
    assert send.call_args.kwargs['params']=={'sstr':'1998 KY26'}
    assert send.call_args.kwargs['allow_redirects'] is False
    assert send.call_args.kwargs['stream'] is False
    ua=send.call_args.kwargs['headers']['User-Agent']
    from aneos_core import __version__
    assert __version__ in ua and 'https://github.com/RobLe3/aneos-suite/issues' in ua


def test_threads_and_distinct_jpl_endpoints_share_one_wire_lease():
    p=transport();active=0;maximum=0;lock=threading.Lock()
    class Slow(Response):
        @property
        def content(self): time.sleep(.02);return self._content
        def close(self):
            nonlocal active
            if not self.closed:
                with lock:active-=1
            super().close()
    def send(*a,**kw):
        nonlocal active,maximum
        with lock:active+=1;maximum=max(maximum,active)
        return Slow()
    urls=['https://ssd-api.jpl.nasa.gov/sbdb.api','https://ssd.jpl.nasa.gov/api/horizons.api']*3
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda url:p.get(url,_send=send),urls))
    assert maximum==1 and active==0


@pytest.mark.parametrize('retry_header',['8','Thu, 01 Jan 1970 00:16:48 GMT'])
def test_retry_after_is_shared_and_never_retried_early(retry_header):
    p=transport();clock=Clock();send=Mock(side_effect=[Response(429,{'Retry-After':retry_header}),Response()])
    with pytest.raises(p.ProviderUnavailableError):
        p.get('https://ssd-api.jpl.nasa.gov/cad.api',_send=send,_clock=clock)
    assert send.call_count==1
    state=p.provider_status()
    assert state['next_allowed_at']>=1008
    p.get('https://ssd.jpl.nasa.gov/api/horizons.api',_send=send,_clock=clock)
    assert clock.value>=1008 and send.call_count==2


def test_long_retry_after_exceeds_budget_without_another_request():
    p=transport();clock=Clock();send=Mock(return_value=Response(503,{'Retry-After':'3600'}))
    with pytest.raises(p.ProviderUnavailableError):p.get('https://ssd-api.jpl.nasa.gov/cad.api',_send=send,_clock=clock)
    with pytest.raises(p.ProviderUnavailableError,match='budget|cooldown'):
        p.get('https://ssd-api.jpl.nasa.gov/sbdb.api',_send=send,_clock=clock,_max_wait=1)
    assert send.call_count==1
    assert p.provider_status()['next_allowed_at']>=4600


def test_permanent_4xx_does_not_install_long_outage_or_retry():
    p=transport();clock=Clock();send=Mock(return_value=Response(400))
    with pytest.raises(p.ProviderUnavailableError):p.get('https://ssd-api.jpl.nasa.gov/sbdb.api',_send=send,_clock=clock)
    assert send.call_count==1
    assert p.provider_status()['failures']==0


def test_transport_timeout_sets_cooldown_and_no_empty_json_success():
    p=transport();clock=Clock();send=Mock(side_effect=requests.Timeout('synthetic timeout'))
    with pytest.raises(p.ProviderUnavailableError):p.get('https://ssd-api.jpl.nasa.gov/sbdb.api',_send=send,_clock=clock)
    assert p.provider_status()['next_allowed_at']>1000
    assert send.call_count==1


def test_corrupt_coordination_state_fails_closed(monkeypatch,tmp_path):
    p=transport();directory=tmp_path/'corrupt';directory.mkdir();(directory/'jpl-ssd-v1.json').write_text('not json')
    monkeypatch.setenv('ANEOS_PROVIDER_STATE_DIR',str(directory));send=Mock()
    with pytest.raises(p.ProviderUnavailableError,match='state|coordination'):
        p.get('https://ssd-api.jpl.nasa.gov/sbdb.api',_send=send)
    send.assert_not_called()


def test_non_jpl_positive_control_is_not_blocked_by_jpl_cooldown():
    p=transport();clock=Clock()
    with pytest.raises(p.ProviderUnavailableError):
        p.get('https://ssd-api.jpl.nasa.gov/cad.api',_send=Mock(return_value=Response(429,{'Retry-After':'3600'})),_clock=clock)
    reply=Response();send=Mock(return_value=reply)
    assert p.get('https://example.invalid/own-source',_send=send) is reply
    send.assert_called_once()


def test_redirects_are_not_followed_as_implicit_source_changes():
    p=transport();send=Mock(return_value=Response(302,{'Location':'https://example.invalid/other'}))
    with pytest.raises(p.ProviderUnavailableError):p.get('https://ssd-api.jpl.nasa.gov/sbdb.api',_send=send)
    assert send.call_count==1


def test_redundant_health_probe_cannot_veto_an_authorized_real_fetch():
    from aneos_core.data.fetcher import DataFetcher
    from aneos_core.data.sources.base import FetchResult
    source=Mock();source.health_check=Mock(side_effect=AssertionError('redundant probe'))
    source.fetch_orbital_elements=Mock(return_value=FetchResult(True,data={'semi_major_axis':1.,'eccentricity':.2,'inclination':15.},source='SBDB'))
    fetcher=DataFetcher.__new__(DataFetcher);fetcher.logger=Mock()
    fetcher.fetch_stats={'source_successes':{'SBDB':0},'source_failures':{'SBDB':0}}
    result=fetcher._fetch_from_source('SBDB',source,'marker')
    assert result is not None
    source.health_check.assert_not_called()
    source.fetch_orbital_elements.assert_called_once_with('marker')


def test_first_process_blocks_second_until_buffered_body_is_closed(tmp_path):
    """Real child processes and flock, with a synthetic sender, no HTTP server."""
    import os
    import subprocess
    import sys
    script = '''
import socket,sys,time
from pathlib import Path
from aneos_core.data import provider_http as p
socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError('no network'))
events=Path(sys.argv[1])
class Reply:
 status_code=200
 headers={}
 @property
 def content(self):
  time.sleep(.12)
  return b'{}'
 def close(self):
  with events.open('a') as f: f.write('close\\n')
def send(*a,**k):
 with events.open('a') as f: f.write('send\\n')
 return Reply()
p.get(sys.argv[2],_send=send)
'''
    events = tmp_path / 'events'
    urls = ['https://ssd-api.jpl.nasa.gov/cad.api',
            'https://ssd.jpl.nasa.gov/api/horizons.api']
    children = [subprocess.Popen([sys.executable, '-c', script, str(events), url],
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE) for url in urls]
    for child in children:
        out, err = child.communicate(timeout=15)
        assert child.returncode == 0, (out, err)
    assert events.read_text().splitlines() == ['send', 'close', 'send', 'close']


def test_killed_process_keeps_quiet_window_without_next_wire_attempt(tmp_path):
    import subprocess
    import sys
    script = '''
import time
from pathlib import Path
from aneos_core.data import provider_http as p
def send(*a,**k):
 Path(__import__('sys').argv[1]).write_text('entered')
 time.sleep(60)
p.get('https://ssd-api.jpl.nasa.gov/sbdb.api',_send=send,timeout=30)
'''
    entered = tmp_path / 'entered'
    child = subprocess.Popen([sys.executable, '-c', script, str(entered)])
    try:
        end = time.monotonic() + 10
        while not entered.exists() and time.monotonic() < end:
            time.sleep(.02)
        assert entered.exists()
    finally:
        child.kill()
        child.wait(timeout=5)
    p = transport()
    state = p.provider_status()
    assert state['inflight_until'] > time.time()
    send = Mock()
    with pytest.raises(p.ProviderUnavailableError, match='cooldown'):
        p.get('https://ssd-api.jpl.nasa.gov/cad.api', _send=send, _max_wait=0)
    send.assert_not_called()


@pytest.mark.parametrize('bad_state', [
    {}, {'version': 2}, {'version': 1, 'failures': 0, 'next_allowed_at': float('nan'), 'inflight_until': 0},
    {'version': 1, 'failures': True, 'next_allowed_at': 0, 'inflight_until': 0},
])
def test_invalid_state_never_becomes_a_fresh_schedule(bad_state, tmp_path):
    import os
    directory = tmp_path / 'provider-state'
    directory.mkdir()
    path = directory / 'jpl-ssd-v1.json'
    path.write_text(json.dumps(bad_state))
    path.chmod(0o600)
    p = transport()
    send = Mock()
    with pytest.raises(p.ProviderUnavailableError):
        p.get('https://ssd-api.jpl.nasa.gov/cad.api', _send=send)
    send.assert_not_called()


def test_symlink_state_cannot_redirect_scheduler_writes(tmp_path):
    target = tmp_path / 'do-not-change'
    target.write_text('original')
    directory = tmp_path / 'provider-state'
    directory.mkdir()
    (directory / 'jpl-ssd-v1.json').symlink_to(target)
    p = transport()
    send = Mock()
    with pytest.raises(p.ProviderUnavailableError):
        p.get('https://ssd-api.jpl.nasa.gov/cad.api', _send=send)
    assert target.read_text() == 'original'
    send.assert_not_called()


@pytest.mark.parametrize('status', [None, True, Mock(), '200', 999])
def test_absent_or_invalid_status_is_not_assumed_to_be_200(status):
    p = transport()
    reply = Response(status)
    with pytest.raises(p.ProviderUnavailableError, match='invalid'):
        p.get('https://ssd-api.jpl.nasa.gov/cad.api', _send=Mock(return_value=reply))
    assert reply.closed


def test_invalid_json_is_not_an_empty_success():
    p = transport()
    with pytest.raises(p.ProviderUnavailableError, match='JSON'):
        p.get_json('https://ssd-api.jpl.nasa.gov/cad.api',
                   _send=Mock(return_value=Response(body=b'not json')))


@pytest.mark.parametrize('url', ['http://ssd-api.jpl.nasa.gov/cad.api',
    'https://user:secret@ssd-api.jpl.nasa.gov/cad.api', 'https://ssd-api.jpl.nasa.gov:8080/cad.api'])
def test_insecure_or_credentialed_jpl_urls_fail_without_wire_or_secret_message(url):
    p = transport()
    send = Mock()
    with pytest.raises(p.ProviderUnavailableError) as caught:
        p.get(url, _send=send)
    assert 'secret' not in str(caught.value)
    send.assert_not_called()


def test_session_retries_disabled_for_jpl_only():
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    p = transport()
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=5, status_forcelist=[429, 503])))
    original = session.get_adapter('https://other.invalid/')
    session.send = Mock(return_value=Response())
    p.session_get(session, 'https://ssd-api.jpl.nasa.gov:443/sbdb.api')
    assert session.get_adapter('https://ssd-api.jpl.nasa.gov:443/sbdb.api').max_retries.total == 0
    assert session.get_adapter('https://other.invalid/') is original
    assert original.max_retries.total == 5
    assert session.send.call_count == 1


def test_configured_pacing_is_separate_from_a_published_quota(monkeypatch):
    monkeypatch.setenv('ANEOS_JPL_MIN_INTERVAL', '0.25')
    p = transport()
    clock = Clock()
    send = Mock(return_value=Response())
    p.get('https://ssd-api.jpl.nasa.gov/cad.api', _send=send, _clock=clock)
    p.get('https://ssd.jpl.nasa.gov/api/horizons.api', _send=send, _clock=clock)
    assert clock.value == 1000.25 and send.call_count == 2


@pytest.mark.asyncio
async def test_async_cancellation_does_not_release_active_thread_lease():
    p = transport()
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    class Blocking(Response):
        @property
        def content(self):
            entered.set()
            assert release.wait(timeout=5)
            return b'{}'
        def close(self):
            super().close()
            closed.set()
    task = asyncio.create_task(asyncio.to_thread(p.get, 'https://ssd-api.jpl.nasa.gov/sbdb.api',
                                                _send=Mock(return_value=Blocking())))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        send = Mock()
        with pytest.raises(p.ProviderUnavailableError, match='budget'):
            await asyncio.to_thread(p.get, 'https://ssd-api.jpl.nasa.gov/cad.api',
                                    _send=send, _max_wait=.05)
        send.assert_not_called()
    finally:
        release.set()
        assert await asyncio.to_thread(closed.wait, 5)
    p.get('https://ssd-api.jpl.nasa.gov/cad.api', _send=Mock(return_value=Response()))


@pytest.mark.asyncio
async def test_jpl_base_adapter_uses_encoded_query_and_buffered_transport(monkeypatch):
    from aneos_core.data.sources.sbdb import SBDBSource
    from aneos_core.config.settings import APIConfig
    p = transport()
    reply = Response(body=b'{"synthetic":true}')
    request = Mock(return_value=reply)
    monkeypatch.setattr(requests, 'get', request)
    source = SBDBSource(APIConfig())
    result = await source._http_get('', {'sstr': 'synthetic + &= KY26', 'unused': None})
    assert result == {'synthetic': True} and reply.closed
    from urllib.parse import urlsplit, parse_qs
    url = request.call_args.args[0]
    assert urlsplit(url).path == '/sbdb.api'  # No invented trailing slash.
    assert parse_qs(urlsplit(url).query) == {'sstr': ['synthetic + &= KY26']}
    assert p.provider_status()['failures'] == 0


@pytest.mark.asyncio
async def test_explicit_health_get_accepts_expected_400_without_an_extra_probe(monkeypatch):
    from aneos_core.data.sources.sbdb import SBDBSource
    from aneos_core.config.settings import APIConfig
    reply = Response(400)
    request = Mock(return_value=reply)
    monkeypatch.setattr(requests, 'get', request)
    source = SBDBSource(APIConfig())
    from aneos_core.data.sources.base import HTTPDataSource
    assert await HTTPDataSource.health_check(source) is True
    request.assert_called_once()
    assert reply.closed


@pytest.mark.asyncio
async def test_returned_source_failure_cannot_reset_circuit_or_be_cached():
    from unittest.mock import AsyncMock
    from aneos_core.data.sources.sbdb import SBDBSource
    from aneos_core.data.sources.base import FetchResult
    from aneos_core.config.settings import APIConfig
    source = SBDBSource(APIConfig())
    source._circuit_breaker = Mock()
    source._circuit_breaker.can_execute.return_value = True
    source._rate_limiter = Mock(acquire=AsyncMock())
    source._store_in_cache = AsyncMock()
    source.fetch_orbital_elements = AsyncMock(return_value=FetchResult(False, error_message='synthetic outage'))
    result = await source.fetch_with_resilience('synthetic')
    assert result.success is False
    source._circuit_breaker.record_failure.assert_called_once()
    source._circuit_breaker.record_success.assert_not_called()
    source._store_in_cache.assert_not_called()
    assert source.get_status().failed_requests == 1
    # A genuine recovered source remains an authorized positive control.
    source.fetch_orbital_elements.return_value = FetchResult(True, data={'a': 1.2})
    recovered = await source.fetch_with_resilience('synthetic')
    assert recovered.success
    source._circuit_breaker.record_success.assert_called_once()
    source._store_in_cache.assert_awaited_once()


def test_world_writable_coordination_directory_fails_closed(tmp_path):
    directory = tmp_path / 'provider-state'
    directory.mkdir()
    directory.chmod(0o777)
    p = transport()
    send = Mock()
    with pytest.raises(p.ProviderUnavailableError, match='coordination'):
        p.get('https://ssd-api.jpl.nasa.gov/cad.api', _send=send)
    send.assert_not_called()


def test_close_failure_is_an_outage_not_a_success():
    p = transport()
    reply = Response()
    reply.close = Mock(side_effect=OSError('synthetic close error'))
    with pytest.raises(p.ProviderUnavailableError, match='closed'):
        p.get('https://ssd-api.jpl.nasa.gov/cad.api', _send=Mock(return_value=reply))
    assert p.provider_status()['failures'] == 1
