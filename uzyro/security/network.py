from __future__ import annotations

from urllib.parse import urlparse

from .errors import SecurityValidationError


CORE_DOWNLOAD_HOSTS = {"api.github.com", "github.com", "objects.githubusercontent.com", "huggingface.co"}


def validate_download_url(url: str, *, allowed_hosts: set[str] | None = None, allow_file: bool = False) -> str:
    parsed = urlparse(str(url))
    if allow_file and parsed.scheme == "file":
        return str(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise SecurityValidationError("Загрузка разрешена только по HTTPS")
    host = parsed.hostname.lower().rstrip(".")
    hosts = CORE_DOWNLOAD_HOSTS if allowed_hosts is None else {item.lower().rstrip(".") for item in allowed_hosts}
    if host not in hosts:
        raise SecurityValidationError("Источник загрузки не входит в список доверенных")
    if parsed.username or parsed.password:
        raise SecurityValidationError("Учётные данные в URL запрещены")
    return str(url)


def validate_loopback_url(url: str) -> str:
    parsed = urlparse(str(url))
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise SecurityValidationError("Локальный AI может обращаться только к loopback")
    return str(url)


__all__ = ["CORE_DOWNLOAD_HOSTS", "validate_download_url", "validate_loopback_url"]
