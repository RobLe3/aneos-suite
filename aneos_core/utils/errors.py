"""Custom exception types for aNEOS pipeline and data subsystems."""


class IntegrationError(RuntimeError):
    """Raised when a required pipeline component cannot be imported."""


class DataSourceUnavailableError(RuntimeError):
    """Raised when all external API data sources are exhausted."""


class EvidencePersistenceError(RuntimeError):
    """Acquired source evidence could not be committed to the local ledger."""


class AmbiguousObjectError(ValueError):
    """A catalogue lookup did not identify a single object."""


class SourceIdentityError(ValueError):
    """Returned native identifiers conflict with the requested object."""


class SourcePayloadError(ValueError):
    """Received source content was not admitted; native payload stays available."""
    def __init__(self, message, *, native_response=None, query=None):
        super().__init__(message)
        self.native_response = native_response
        self.query = query
