class SecurityValidationError(ValueError):
    """Untrusted input failed a structural or trust-boundary check."""


class ResourceLimitError(SecurityValidationError):
    """Untrusted input exceeds a declared resource budget."""


__all__ = ["ResourceLimitError", "SecurityValidationError"]
