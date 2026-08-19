from __future__ import annotations

import io
import os
from pathlib import Path
import stat
import warnings
import zipfile
import zlib

import numpy as np
from PIL import Image

from .errors import ResourceLimitError, SecurityValidationError
from .limits import LIMITS
from .paths import ensure_within, validate_archive_member, validate_local_input


PIL_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "TIFF", "GIF", "ICO", "TGA", "PPM"}


def validate_dimensions(width: int, height: int, *, channels: int = 4, bytes_per_channel: int = 1, copies: int = 8) -> None:
    if width <= 0 or height <= 0 or width > LIMITS.max_dimension or height > LIMITS.max_dimension:
        raise ResourceLimitError("Размеры изображения выходят за безопасный диапазон")
    pixels = width * height
    if pixels > LIMITS.max_image_pixels:
        raise ResourceLimitError("Изображение превышает безопасное количество пикселей")
    estimate = pixels * max(1, channels) * max(1, bytes_per_channel) * max(1, copies)
    if estimate > LIMITS.memory_budget_bytes():
        raise ResourceLimitError("Для изображения недостаточно безопасного запаса памяти; используйте большой документ")


def validate_regular_file(path: str | Path, *, maximum: int) -> Path:
    source = validate_local_input(path)
    size = source.stat().st_size
    if size <= 0:
        raise SecurityValidationError("Файл пуст")
    if size > maximum:
        raise ResourceLimitError("Файл превышает безопасный размер")
    return source


def inspect_pillow_image(path: str | Path) -> tuple[Path, str, tuple[int, int]]:
    source = validate_regular_file(path, maximum=LIMITS.max_image_file_bytes)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as image:
                image_format = str(image.format or "").upper()
                if image_format not in PIL_FORMATS:
                    raise SecurityValidationError("Содержимое файла не является поддерживаемым изображением")
                validate_dimensions(image.width, image.height)
                frames = int(getattr(image, "n_frames", 1))
                if frames > 1_000:
                    raise ResourceLimitError("Изображение содержит слишком много кадров")
                metadata_bytes = sum(
                    len(value) if isinstance(value, bytes) else len(str(value).encode("utf-8", errors="replace"))
                    for value in image.info.values()
                )
                if metadata_bytes > LIMITS.max_metadata_bytes:
                    raise ResourceLimitError("Метаданные изображения превышают безопасный размер")
                return source, image_format, (image.width, image.height)
    except Image.DecompressionBombError as exc:
        raise ResourceLimitError("Изображение распознано как decompression bomb") from exc
    except (OSError, SyntaxError) as exc:
        raise SecurityValidationError("Файл повреждён или не является поддерживаемым изображением") from exc


def load_pillow_image(path: str | Path, mode: str = "RGBA") -> Image.Image:
    source, _format, size = inspect_pillow_image(path)
    try:
        with Image.open(source) as image:
            image.load()
            if image.size != size:
                raise SecurityValidationError("Размер изображения изменился во время декодирования")
            return image.convert(mode).copy()
    except (OSError, SyntaxError) as exc:
        raise SecurityValidationError("Не удалось безопасно декодировать изображение") from exc


def load_pillow_bytes(payload: bytes, mode: str = "RGBA") -> Image.Image:
    if not payload or len(payload) > LIMITS.max_archive_member_bytes:
        raise ResourceLimitError("Встроенное изображение превышает безопасный размер")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if str(image.format or "").upper() not in PIL_FORMATS:
                raise SecurityValidationError("Встроенные данные не являются поддерживаемым изображением")
            validate_dimensions(image.width, image.height)
            image.load()
            return image.convert(mode).copy()
    except Image.DecompressionBombError as exc:
        raise ResourceLimitError("Встроенное изображение распознано как decompression bomb") from exc
    except (OSError, SyntaxError) as exc:
        raise SecurityValidationError("Встроенное изображение повреждено") from exc


def validate_zip_info(info: zipfile.ZipInfo, *, maximum: int = LIMITS.max_archive_member_bytes) -> str:
    path = validate_archive_member(info.filename)
    normalized = path.as_posix()
    if info.flag_bits & 0x1:
        raise SecurityValidationError("Зашифрованные элементы архива не поддерживаются")
    kind = (info.external_attr >> 16) & 0o170000
    if kind == stat.S_IFLNK:
        raise SecurityValidationError("Символические ссылки внутри архива запрещены")
    if info.file_size < 0 or info.file_size > maximum:
        raise ResourceLimitError("Элемент архива превышает безопасный размер")
    ratio = info.file_size / max(1, info.compress_size)
    if info.file_size > 1024**2 and ratio > LIMITS.max_archive_ratio:
        raise ResourceLimitError("Архив содержит подозрительно сжатый элемент")
    return normalized


def validate_zip_archive(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > LIMITS.max_archive_entries:
        raise ResourceLimitError("Архив содержит слишком много файлов")
    result: dict[str, zipfile.ZipInfo] = {}
    total = 0
    for info in infos:
        normalized = validate_zip_info(info)
        if normalized in result:
            raise SecurityValidationError("Архив содержит повторяющиеся пути")
        total += info.file_size
        if total > LIMITS.max_archive_total_bytes:
            raise ResourceLimitError("Распакованный архив превышает безопасный размер")
        result[normalized] = info
    return result


def read_zip_member(archive: zipfile.ZipFile, name: str, *, maximum: int | None = None) -> bytes:
    infos = validate_zip_archive(archive)
    normalized = validate_archive_member(name).as_posix()
    info = infos.get(normalized)
    if info is None:
        raise SecurityValidationError(f"В архиве отсутствует обязательный файл: {normalized}")
    limit = LIMITS.max_archive_member_bytes if maximum is None else maximum
    if info.file_size > limit:
        raise ResourceLimitError(f"Элемент архива слишком велик: {normalized}")
    payload = archive.read(info)
    if len(payload) != info.file_size:
        raise SecurityValidationError(f"Размер элемента архива не совпадает с описанием: {normalized}")
    return payload


def safe_extract_zip(archive: zipfile.ZipFile, destination: str | Path) -> list[Path]:
    infos = validate_zip_archive(archive)
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, info in infos.items():
        target = ensure_within(root.joinpath(*validate_archive_member(name).parts), root)
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        copied = 0
        try:
            with archive.open(info, "r") as source, target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > info.file_size:
                        raise SecurityValidationError("Архив изменился во время распаковки")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if copied != info.file_size:
                raise SecurityValidationError("Архив изменился во время распаковки")
        except Exception:
            target.unlink(missing_ok=True)
            raise
        written.append(target)
    return written


def validate_array(array: np.ndarray, *, expected_channels: set[int] | None = None) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim not in {2, 3}:
        raise SecurityValidationError("Массив изображения должен иметь два или три измерения")
    channels = 1 if value.ndim == 2 else value.shape[2]
    if expected_channels is not None and channels not in expected_channels:
        raise SecurityValidationError("Некорректное количество каналов изображения")
    validate_dimensions(value.shape[1], value.shape[0], channels=channels, bytes_per_channel=value.dtype.itemsize, copies=2)
    return value


def bounded_zlib_decompress(payload: bytes, *, maximum: int) -> bytes:
    decoder = zlib.decompressobj()
    try:
        output = decoder.decompress(payload, maximum + 1)
        if len(output) > maximum or decoder.unconsumed_tail:
            raise ResourceLimitError("Сжатые данные превышают безопасный размер")
        output += decoder.flush(maximum + 1 - len(output))
    except zlib.error as exc:
        raise SecurityValidationError("Сжатые данные повреждены") from exc
    if len(output) > maximum:
        raise ResourceLimitError("Сжатые данные превышают безопасный размер")
    return output


__all__ = [
    "bounded_zlib_decompress", "inspect_pillow_image", "load_pillow_bytes", "load_pillow_image", "read_zip_member", "safe_extract_zip", "validate_array",
    "validate_dimensions", "validate_regular_file", "validate_zip_archive", "validate_zip_info",
]
