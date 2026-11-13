"""Buffered, serialized JPL SSD GETs; endpoint schemas remain source responsibilities.

All aNEOS processes must share ANEOS_PROVIDER_STATE_DIR on a filesystem with
working POSIX flock. This is not a distributed lock or a scientific-data cache.
"""
from contextlib import contextmanager
from datetime import timezone
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import stat
import time
from urllib.parse import urlsplit

import requests

from aneos_core import __version__

APPLICATION_HEADERS = {
    'User-Agent': f'aNEOS/{__version__} scientific-research (+https://github.com/RobLe3/aneos-suite/issues)'
}
_STATE_NAME = 'jpl-ssd-v1.json'


class ProviderUnavailableError(requests.RequestException):
    """Unavailable acquisition, never an empty successful evidence response."""

    def __init__(self, message, *, retry_at=None, status_code=None):
        self.retry_at = retry_at
        self.status_code = status_code
        super().__init__(message)


def is_jpl_api(url):
    parsed = urlsplit(url)
    return (parsed.hostname == 'ssd-api.jpl.nasa.gov' or
            (parsed.hostname == 'ssd.jpl.nasa.gov' and parsed.path.startswith('/api/')))


def _number(value, label, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(label)
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(label)
    return number


def _initial():
    return dict(version=1, next_allowed_at=0., failures=0, inflight_until=0.)


def _read(handle):
    handle.seek(0)
    state = json.load(handle)
    if not isinstance(state, dict) or type(state.get('version')) is not int or state['version'] != 1:
        raise ValueError('state version')
    if type(state.get('failures')) is not int or not 0 <= state['failures'] <= 1000000:
        raise ValueError('state failures')
    for key in ('next_allowed_at', 'inflight_until'):
        state[key] = _number(state.get(key), key)
    return state


def _write(handle, state):
    handle.seek(0)
    json.dump(state, handle, allow_nan=False, sort_keys=True)
    handle.truncate()
    handle.flush()
    os.fsync(handle.fileno())


@contextmanager
def _locked(clock, deadline):
    handle = None
    lock_handle = None
    try:
        import fcntl
        directory = Path(os.environ.get('ANEOS_PROVIDER_STATE_DIR',
                                       '~/.cache/aneos/providers')).expanduser()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if (directory.is_symlink() or directory.stat().st_uid != os.getuid()
                or directory.stat().st_mode & 0o022):
            raise ValueError('unsafe coordination directory')
        flags = os.O_RDWR | os.O_NOFOLLOW
        lock_fd = os.open(directory / 'jpl-ssd-v1.lock', flags | os.O_CREAT, 0o600)
        lock_handle = os.fdopen(lock_fd, 'r+', encoding='utf-8')
        info = os.fstat(lock_handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('unsafe coordination lock')
        while True:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if clock.monotonic() >= deadline:
                    raise ProviderUnavailableError('JPL coordination wait budget exhausted')
                clock.sleep(min(.02, max(0., deadline - clock.monotonic())))
        path = directory / _STATE_NAME
        try:
            fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
        except FileExistsError:
            fd = os.open(path, flags)
            created = False
        handle = os.fdopen(fd, 'r+', encoding='utf-8')
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('unsafe coordination file')
        # Initialize only a file created by this call. Empty existing files may
        # reflect an interrupted write, and must not erase a previous cooldown.
        if created:
            _write(handle, _initial())
        state = _read(handle)
    except ProviderUnavailableError:
        if handle:
            handle.close()
        if lock_handle:
            lock_handle.close()
        raise
    except Exception as exc:
        if handle:
            handle.close()
        if lock_handle:
            lock_handle.close()
        raise ProviderUnavailableError('JPL coordination state unavailable or invalid') from exc
    try:
        yield handle, state
    finally:
        handle.close()
        lock_handle.close()  # Releases flock even when the request raises.


def provider_status():
    """Return scheduling metadata only; this is not a provider health claim."""
    with _locked(time, time.monotonic() + 5) as (_, state):
        return dict(state)


def _retry_after(headers, now):
    value = headers.get('Retry-After')
    if value is None:
        return now
    try:
        return now + _number(value, 'Retry-After')
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(now, date.timestamp())
        except (TypeError, ValueError, OverflowError):
            return now


def _disable_hidden_retries(send, url):
    """A requests Session must not retry invisibly within one guarded attempt."""
    session = getattr(send, '__self__', None)
    if isinstance(session, requests.Session):
        # Specific prefixes leave the session's non-JPL adapters unchanged.
        prefix = 'https://' + urlsplit(url).netloc + '/'
        adapter = session.get_adapter(prefix)
        if adapter.max_retries.total != 0:
            session.mount(prefix, requests.adapters.HTTPAdapter(max_retries=0))


def get(url, *, _send=None, _clock=None, _max_wait=30., _accepted_statuses=(200,), **kwargs):
    """One buffered attempt, with shared cooldown; no automatic network retry.

    ``_max_wait`` bounds coordination/pacing waits, not wire duration. Requests'
    connect/read timeout is an inactivity timeout, not an end-to-end deadline.
    Async callers must use to_thread; cancellation does not release its lease.
    """
    send = _send or requests.get
    if not is_jpl_api(url):
        return send(url, **kwargs)
    clock = _clock or time
    parsed = urlsplit(url)
    try:
        if parsed.scheme != 'https' or parsed.port not in (None, 443) or parsed.username or parsed.password:
            raise ValueError('JPL requires uncredentialed HTTPS')
        wait = _number(_max_wait, 'wait budget')
        interval = _number(os.environ.get('ANEOS_JPL_MIN_INTERVAL', '.25'), 'pacing interval')
        timeout = kwargs.get('timeout', 30.)
        parts = timeout if isinstance(timeout, tuple) else (timeout,)
        if len(parts) not in (1, 2):
            raise ValueError('timeout dimensions')
        quiet_window = sum(_number(part, 'timeout', positive=True) for part in parts) + 2.
    except (TypeError, ValueError) as exc:
        raise ProviderUnavailableError('Invalid JPL request/coordination configuration') from exc
    kwargs['timeout'] = timeout
    kwargs['headers'] = {**(kwargs.get('headers') or {}), **APPLICATION_HEADERS}
    # Do not let a lower-case caller/session header override identification.
    kwargs['headers'] = {k: v for k, v in kwargs['headers'].items()
                         if k.lower() != 'user-agent' or k == 'User-Agent'}
    kwargs['allow_redirects'] = False
    kwargs['stream'] = False
    _disable_hidden_retries(send, url)
    deadline = clock.monotonic() + wait
    while True:
        with _locked(clock, deadline) as (handle, state):
            now = clock.time()
            allowed = max(state['next_allowed_at'], state['inflight_until'])
            delay = max(0., allowed - now)
            if delay <= 0:
                state['inflight_until'] = now + quiet_window
                _write(handle, state)
                response = None
                failed = False
                try:
                    response = send(url, **kwargs)
                    # Buffer and close while holding the lease, including failures.
                    response.content
                    status = response.status_code
                    if type(status) is not int or not 100 <= status <= 599:
                        raise ValueError('missing/invalid HTTP status')
                    if status not in _accepted_statuses:
                        failed = status == 429 or status >= 500
                        if failed:
                            state['next_allowed_at'] = max(state['next_allowed_at'],
                                                          _retry_after(response.headers, clock.time()))
                        raise ProviderUnavailableError(f'JPL returned HTTP {status}', status_code=status)
                    state['failures'] = 0
                    return response
                except ProviderUnavailableError:
                    raise
                except Exception as exc:
                    failed = True
                    # Never include query strings, credentials or response bodies.
                    raise ProviderUnavailableError('JPL request failed or response invalid') from exc
                finally:
                    try:
                        if response is not None:
                            try:
                                response.close()
                            except Exception as exc:
                                failed = True
                                raise ProviderUnavailableError('JPL response could not be closed') from exc
                    finally:
                        state['inflight_until'] = 0.
                        state['next_allowed_at'] = max(state['next_allowed_at'], clock.time() + interval)
                        if failed:
                            state['failures'] = min(state['failures'] + 1, 1000000)
                            state['next_allowed_at'] = max(state['next_allowed_at'],
                                clock.time() + min(60., 2. ** min(state['failures'], 6)))
                        _write(handle, state)
        # Do not hold the lock while waiting for a persisted pacing/cooldown time.
        if delay > deadline - clock.monotonic():
            raise ProviderUnavailableError('JPL cooldown exceeds wait budget', retry_at=allowed)
        clock.sleep(delay)


def get_json(url, **kwargs):
    response = get(url, **kwargs)
    try:
        return response.json()
    except Exception as exc:
        raise ProviderUnavailableError('JPL response is not valid JSON') from exc


def session_get(session, url, **kwargs):
    """Explicit Session integration, preserving non-JPL session behavior."""
    return get(url, _send=session.get, **kwargs)
