"""aNEOS API exports, loaded only when requested.

Storage and job tooling must not assemble the web app or import optional ML
runtimes. Public names retain their implementations; laziness is not a fallback
or a claim that those optional services are healthy/validated.
"""
from importlib import import_module

from aneos_version import APPLICATION_VERSION as __version__
__author__ = "aNEOS Project"

_EXPORTS = {
    'create_app':('.app','create_app'), 'ANEOSApp':('.app','ANEOSApp'),
    'APIResponse':('.models','APIResponse'), 'ErrorResponse':('.models','ErrorResponse'),
    'PaginatedResponse':('.models','PaginatedResponse'),
    'AuthManager':('.auth','AuthManager'), 'APIKeyAuth':('.auth','APIKeyAuth'),
    'RateLimiter':('.middleware','RateLimiter'), 'RequestLogger':('.middleware','RequestLogger'),
    'ErrorHandler':('.middleware','ErrorHandler'),
    # These module attributes were historically imported, though not in __all__.
    'analysis':('.endpoints.analysis',None), 'prediction':('.endpoints.prediction',None),
    'monitoring':('.endpoints.monitoring',None), 'admin':('.endpoints.admin',None),
}
__all__ = ['create_app','ANEOSApp','APIResponse','ErrorResponse','PaginatedResponse',
           'AuthManager','APIKeyAuth','RateLimiter','RequestLogger','ErrorHandler']


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module_name, attribute = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attribute) if attribute is not None else module
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
