from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path


class CredentialStoreError(RuntimeError):
    pass


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob(data: bytes) -> tuple[_DataBlob, ctypes.Array]:
    buffer = ctypes.create_string_buffer(data)
    return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))), buffer


def _crypt(data: bytes, protect: bool) -> bytes:
    if os.name != "nt":
        raise CredentialStoreError("Безопасное хранение ключа поддерживается только в Windows")
    source, source_buffer = _blob(data)
    output = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    function = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    arguments = (ctypes.byref(source), None, None, None, None, 0, ctypes.byref(output)) if protect else (
        ctypes.byref(source), None, None, None, None, 0, ctypes.byref(output)
    )
    if not function(*arguments):
        raise CredentialStoreError(f"Windows DPAPI вернул ошибку {kernel32.GetLastError()}")
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        kernel32.LocalFree(output.pbData)
        del source_buffer


class EncryptedCredentialStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, secret: str) -> None:
        value = str(secret).strip()
        if not value:
            self.delete()
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_bytes(_crypt(value.encode("utf-8"), True))
        os.replace(temporary, self.path)

    def load(self) -> str | None:
        if not self.path.exists():
            return None
        try:
            return _crypt(self.path.read_bytes(), False).decode("utf-8")
        except (OSError, UnicodeDecodeError, CredentialStoreError):
            return None

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)


__all__ = ["CredentialStoreError", "EncryptedCredentialStore"]
