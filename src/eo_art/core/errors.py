class EODataLoadError(Exception):
    """Raised when EO data cannot be loaded or is structurally invalid."""


class CRSMissingError(EODataLoadError):
    """Raised when loaded data has no coordinate reference system."""
