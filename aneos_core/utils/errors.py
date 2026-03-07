"""Custom exception types for aNEOS pipeline and data subsystems."""


class IntegrationError(RuntimeError):
    """Raised when a required pipeline component cannot be imported."""


class DataSourceUnavailableError(RuntimeError):
    """Raised when all external API data sources are exhausted."""
