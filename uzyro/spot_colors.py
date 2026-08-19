from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import struct
import uuid
from typing import Any, Iterable

import numpy as np

from .security.files import validate_regular_file
from .security.paths import validate_local_input
from .security.validation import bounded_string, load_bounded_json


ASE_HEADER = b"ASEF\x00\x01\x00\x00"
ASE_COLOR_BLOCK = 0x0001
ASE_GROUP_START = 0xC001
ASE_GROUP_END = 0xC002


@dataclass(frozen=True)
class SpotColor:
    name: str
    lab: tuple[float, float, float]
    alternate_rgb: tuple[int, int, int]
    source: str = "Пользовательские"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        name = bounded_string(self.name, "Название цвета", 256).strip()
        if not name:
            raise ValueError("Название плашечной краски не может быть пустым")
        lab = tuple(float(value) for value in self.lab)
        rgb = tuple(int(value) for value in self.alternate_rgb)
        if len(lab) != 3 or not 0.0 <= lab[0] <= 100.0 or any(not -128.0 <= value <= 127.0 for value in lab[1:]):
            raise ValueError("Lab должен быть задан как L 0–100, a/b от -128 до 127")
        if len(rgb) != 3 or any(not 0 <= value <= 255 for value in rgb):
            raise ValueError("Экранный RGB должен находиться в диапазоне 0–255")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "lab", lab)
        object.__setattr__(self, "alternate_rgb", rgb)
        object.__setattr__(self, "source", bounded_string(self.source, "Источник цвета", 256).strip() or "Пользовательские")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["lab"] = list(self.lab)
        value["alternate_rgb"] = list(self.alternate_rgb)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SpotColor":
        return cls(
            name=str(value["name"]),
            lab=tuple(value["lab"]),
            alternate_rgb=tuple(value.get("alternate_rgb") or lab_to_srgb(tuple(value["lab"]))),
            source=str(value.get("source", "Пользовательские")),
            id=str(value.get("id") or uuid.uuid4().hex),
        )


def _srgb_compand(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * np.power(np.maximum(value, 0), 1 / 2.4) - 0.055)


def _srgb_expand(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.04045, value / 12.92, np.power((value + 0.055) / 1.055, 2.4))


def lab_to_srgb(lab: tuple[float, float, float]) -> tuple[int, int, int]:
    l_value, a_value, b_value = (float(value) for value in lab)
    fy = (l_value + 16.0) / 116.0
    fx = fy + a_value / 500.0
    fz = fy - b_value / 200.0
    epsilon, kappa = 216 / 24389, 24389 / 27

    def inverse(value: float) -> float:
        cube = value**3
        return cube if cube > epsilon else (116 * value - 16) / kappa

    xyz_d50 = np.array([0.96422 * inverse(fx), inverse(fy), 0.82521 * inverse(fz)])
    xyz_d65 = np.array(
        [[0.9555766, -0.0230393, 0.0631636], [-0.0282895, 1.0099416, 0.0210077], [0.0122982, -0.0204830, 1.3299098]]
    ) @ xyz_d50
    linear = np.array(
        [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]]
    ) @ xyz_d65
    return tuple(int(round(value)) for value in np.clip(_srgb_compand(linear) * 255.0, 0, 255))


def srgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    linear = _srgb_expand(np.array(rgb, dtype=np.float64) / 255.0)
    xyz_d65 = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
    ) @ linear
    xyz = np.array(
        [[1.0478112, 0.0228866, -0.0501270], [0.0295424, 0.9904844, -0.0170491], [-0.0092345, 0.0150436, 0.7521316]]
    ) @ xyz_d65
    normalized = xyz / np.array([0.96422, 1.0, 0.82521])
    epsilon, kappa = 216 / 24389, 24389 / 27
    values = np.where(normalized > epsilon, np.cbrt(normalized), (kappa * normalized + 16) / 116)
    return float(116 * values[1] - 16), float(500 * (values[0] - values[1])), float(200 * (values[1] - values[2]))


def spot_settings(metadata: dict[str, Any]) -> dict[str, Any]:
    settings = metadata.setdefault("spot_colors", {})
    settings.setdefault("colors", [])
    settings.setdefault("assignments", {})
    return settings


def document_spot_colors(document) -> list[SpotColor]:
    return [SpotColor.from_dict(value) for value in spot_settings(document.metadata).get("colors", [])]


def replace_document_spot_colors(document, colors: Iterable[SpotColor]) -> None:
    values = list(colors)
    settings = spot_settings(document.metadata)
    settings["colors"] = [color.to_dict() for color in values]
    valid_ids = {color.id for color in values}
    settings["assignments"] = {
        layer_id: color_id for layer_id, color_id in settings.get("assignments", {}).items() if color_id in valid_ids
    }
    document.dirty = True


def assign_spot_color(document, layer_id: str, color_id: str | None) -> None:
    settings = spot_settings(document.metadata)
    assignments = settings["assignments"]
    if color_id is None:
        assignments.pop(layer_id, None)
    elif color_id not in {color.id for color in document_spot_colors(document)}:
        raise ValueError("Плашечная краска отсутствует в библиотеке документа")
    else:
        assignments[layer_id] = color_id
    document.dirty = True


def assigned_spot_color(document, layer_id: str) -> SpotColor | None:
    color_id = spot_settings(document.metadata)["assignments"].get(layer_id)
    return next((color for color in document_spot_colors(document) if color.id == color_id), None)


def _ase_string(payload: bytes, offset: int) -> tuple[str, int]:
    if offset + 2 > len(payload):
        raise ValueError("Повреждённая строка ASE")
    units = struct.unpack_from(">H", payload, offset)[0]
    offset += 2
    byte_length = units * 2
    if units < 1 or offset + byte_length > len(payload):
        raise ValueError("Повреждённая длина строки ASE")
    raw = payload[offset : offset + byte_length - 2]
    return raw.decode("utf-16-be"), offset + byte_length


def _ase_rgb(model: bytes, values: tuple[float, ...]) -> tuple[int, int, int]:
    if model == b"RGB ":
        return tuple(int(round(np.clip(value, 0, 1) * 255)) for value in values[:3])
    if model == b"CMYK":
        c, m, y, k = (np.clip(value, 0, 1) for value in values[:4])
        return tuple(int(round(255 * (1 - value) * (1 - k))) for value in (c, m, y))
    if model == b"Gray":
        gray = int(round(np.clip(values[0], 0, 1) * 255))
        return gray, gray, gray
    raise ValueError(f"Неподдерживаемая модель ASE: {model!r}")


def load_ase(path: str | Path, include_process: bool = True) -> list[SpotColor]:
    source = validate_local_input(path, allowed_suffixes={".ase"})
    data = validate_regular_file(source, maximum=32 * 1024 * 1024).read_bytes()
    if len(data) < 12 or data[:8] != ASE_HEADER:
        raise ValueError("Файл не является библиотекой Adobe Swatch Exchange")
    block_count = struct.unpack_from(">I", data, 8)[0]
    if block_count > 100_000:
        raise ValueError("Библиотека ASE содержит слишком много элементов")
    offset, group, colors = 12, Path(path).stem, []
    for _ in range(block_count):
        if offset + 6 > len(data):
            raise ValueError("Библиотека ASE обрезана")
        kind, size = struct.unpack_from(">HI", data, offset)
        offset += 6
        payload = data[offset : offset + size]
        if len(payload) != size:
            raise ValueError("Библиотека ASE обрезана")
        offset += size
        if kind == ASE_GROUP_START:
            group, _ = _ase_string(payload, 0)
            continue
        if kind == ASE_GROUP_END or kind != ASE_COLOR_BLOCK:
            continue
        name, cursor = _ase_string(payload, 0)
        if cursor + 6 > len(payload):
            raise ValueError("Повреждённый цвет ASE")
        model = payload[cursor : cursor + 4]
        cursor += 4
        component_count = {b"RGB ": 3, b"CMYK": 4, b"LAB ": 3, b"Gray": 1}.get(model)
        if component_count is None or cursor + component_count * 4 + 2 > len(payload):
            raise ValueError(f"Неподдерживаемая модель цвета ASE: {model!r}")
        values = struct.unpack_from(">" + "f" * component_count, payload, cursor)
        color_type = struct.unpack_from(">H", payload, cursor + component_count * 4)[0]
        if color_type != 1 and not include_process:
            continue
        if model == b"LAB ":
            lab = (values[0] * 100.0 if values[0] <= 1.0001 else values[0], values[1], values[2])
            rgb = lab_to_srgb(lab)
        else:
            rgb = _ase_rgb(model, values)
            lab = srgb_to_lab(rgb)
        colors.append(SpotColor(name, lab, rgb, group))
    return colors


def _encode_ase_string(value: str) -> bytes:
    encoded = value.encode("utf-16-be") + b"\x00\x00"
    return struct.pack(">H", len(encoded) // 2) + encoded


def save_ase(path: str | Path, colors: Iterable[SpotColor], group_name: str = "UZYRO") -> None:
    blocks: list[bytes] = []
    group = _encode_ase_string(group_name)
    blocks.append(struct.pack(">HI", ASE_GROUP_START, len(group)) + group)
    for color in colors:
        payload = _encode_ase_string(color.name) + b"LAB "
        payload += struct.pack(">fffH", color.lab[0] / 100.0, color.lab[1], color.lab[2], 1)
        blocks.append(struct.pack(">HI", ASE_COLOR_BLOCK, len(payload)) + payload)
    blocks.append(struct.pack(">HI", ASE_GROUP_END, 0))
    Path(path).write_bytes(ASE_HEADER + struct.pack(">I", len(blocks)) + b"".join(blocks))


def load_library(path: str | Path) -> list[SpotColor]:
    source = validate_local_input(path, allowed_suffixes={".ase", ".json", ".prswatches"})
    if source.suffix.lower() == ".ase":
        return load_ase(source)
    data = load_bounded_json(source, maximum=8 * 1024 * 1024)
    values = data.get("colors") if isinstance(data, dict) else data
    if not isinstance(values, list) or len(values) > 100_000:
        raise ValueError("В библиотеке отсутствует список colors")
    return [SpotColor.from_dict(value) for value in values]


def save_library(path: str | Path, colors: Iterable[SpotColor]) -> None:
    target = Path(path)
    values = list(colors)
    if target.suffix.lower() == ".ase":
        save_ase(target, values)
        return
    target.write_text(
        json.dumps({"format": "UZYRO spot colors", "version": 1, "colors": [color.to_dict() for color in values]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = [name for name in globals() if not name.startswith("__")]
