from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import BlendMode, ProtectedFlags, Resource, Tag
from psd_tools.psd.image_resources import ImageResource, ResoulutionInfo
from psd_tools.psd.tagged_blocks import ProtectedSetting

from .color_management import color_settings, display_rgba, profile_name
from .core_shared import pil_to_rgba_array, rgba_array_to_pil
from .layer import Layer
from .render_ops import render_layer_pixels, render_layer_style

if TYPE_CHECKING:
    from .document import Document


PSD_EXTENSIONS = {".psd", ".psb"}
BLEND_TO_PSD = {
    "Normal": BlendMode.NORMAL,
    "Multiply": BlendMode.MULTIPLY,
    "Screen": BlendMode.SCREEN,
    "Overlay": BlendMode.OVERLAY,
    "Soft Light": BlendMode.SOFT_LIGHT,
    "Linear Light": BlendMode.LINEAR_LIGHT,
    "Darken": BlendMode.DARKEN,
    "Lighten": BlendMode.LIGHTEN,
    "Difference": BlendMode.DIFFERENCE,
    "Color": BlendMode.COLOR,
    "Luminosity": BlendMode.LUMINOSITY,
}
PSD_TO_BLEND = {value.value: key for key, value in BLEND_TO_PSD.items()}
PSD_EFFECT_TAGS = {
    Tag.EFFECTS_LAYER.value, Tag.OBJECT_BASED_EFFECTS_LAYER_INFO.value,
    Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V0.value, Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V1.value,
}
PSD_VECTOR_TAGS = {
    Tag.VECTOR_MASK_AS_GLOBAL_MASK.value, Tag.VECTOR_MASK_SETTING1.value, Tag.VECTOR_MASK_SETTING2.value,
    Tag.VECTOR_ORIGINATION_DATA.value, Tag.VECTOR_ORIGINATION_UNKNOWN.value,
    Tag.VECTOR_STROKE_DATA.value, Tag.VECTOR_STROKE_CONTENT_DATA.value,
}


class PSDCompatibilityError(RuntimeError):
    pass


def _resource_data(psd: PSDImage, key: Resource) -> Any:
    try:
        return psd.image_resources.get_data(key)
    except Exception:
        return None


def _dpi(psd: PSDImage) -> int:
    resolution = _resource_data(psd, Resource.RESOLUTION_INFO)
    if resolution is None:
        return 300
    return max(1, min(2400, round(float(resolution.horizontal) / 65536.0)))


def _leaf_layers(group, prefix: tuple[str, ...] = ()):
    for source in group:
        if source.is_group():
            yield from _leaf_layers(source, (*prefix, str(source.name)))
        else:
            yield source, prefix


def _layer_image(source) -> Image.Image:
    image = source.topil(apply_icc=False)
    if image is None:
        image = source.composite(force=True, apply_icc=False)
    if image is None:
        width, height = max(1, source.width), max(1, source.height)
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))
    return image.convert("RGBA")


def _compatibility_risks(source) -> list[str]:
    name = str(getattr(source, "name", "Слой"))
    kind = str(getattr(source, "kind", "pixel")).lower()
    risks: list[str] = []
    if kind == "type":
        risks.append(f"Текстовый слой «{name}» будет сохранён в PSD с приблизительными параметрами")
    if kind == "smartobject":
        risks.append(f"Photoshop Smart Object «{name}» при совместимом экспорте станет пиксельным слоем")
    blocks = getattr(source, "tagged_blocks", {})
    keys = {getattr(key, "value", key) for key in getattr(blocks, "keys", lambda: ())()}
    if keys & PSD_EFFECT_TAGS:
        risks.append(f"Эффекты Photoshop слоя «{name}» не останутся редактируемыми")
    if keys & PSD_VECTOR_TAGS:
        risks.append(f"Векторные данные слоя «{name}» не останутся редактируемыми")
    return risks


def _mask(source, size: tuple[int, int]) -> np.ndarray | None:
    if source.mask is None:
        return None
    image = source.mask.topil(layer_sized=True)
    if image is None:
        return None
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image.convert("L"), dtype=np.uint8)


def _text_data(source, pixels: np.ndarray) -> dict[str, Any]:
    alpha = pixels[:, :, 3]
    opaque = pixels[:, :, :3][alpha > 16]
    color = [255, 255, 255, 255] if not opaque.size else [*np.median(opaque, axis=0).astype(int).tolist(), 255]
    size = max(8, min(500, round(max(1, source.height) * 0.72)))
    return {
        "text": str(getattr(source, "text", "")), "x": int(source.left), "y": int(source.top),
        "color": color, "size": size, "font_family": "arial.ttf", "box_width": max(0, int(source.width)),
        "align": "left", "line_spacing": max(2, size // 5), "tracking": 0, "bold": False, "italic": False,
        "underline": False, "path_mode": "none", "path_amount": 0, "path_points": [], "path_start": 0.0,
        "path_end": 1.0, "path_side": 1, "path_reverse": False, "baseline_shift": 0, "rotation": 0.0,
        "psd_import_approximation": True,
    }


def _embedded_smart_document(source, document_type) -> dict[str, Any] | None:
    try:
        smart = source.smart_object
        if str(smart.kind) != "data":
            return None
        data = smart.data
        detected = str(smart.detected_filetype or "").lower()
        if detected in {"8bps", "8bpb", "psd", "psb"}:
            nested = _load_psd_image(PSDImage.open(BytesIO(data)), document_type, "embedded.psd")
        else:
            image = Image.open(BytesIO(data)).convert("RGBA")
            pixels = pil_to_rgba_array(image)
            nested = document_type(width=image.width, height=image.height, background=(0, 0, 0, 0))
            nested.layers = [Layer("Содержимое Smart Object", pixels)]
        return nested.snapshot()
    except Exception:
        return None


def _preserve_working_pixels(layer: Layer, source, depth: int, color_model: str) -> None:
    if depth <= 8:
        return
    try:
        values = source.numpy()
        if values is not None and values.ndim == 3 and values.shape[2] in {3, 4} and values.shape[:2] == layer.pixels.shape[:2]:
            layer.set_working_rgba(values, depth, color_model)
        else:
            layer.set_working_rgba(layer.pixels, depth, color_model)
    except Exception:
        layer.set_working_rgba(layer.pixels, depth, color_model)


def _import_layer(source, groups: tuple[str, ...], document_type, document_size: tuple[int, int], depth: int, color_model: str) -> tuple[Layer, str | None]:
    image = _layer_image(source)
    pixels = pil_to_rgba_array(image)
    name = " / ".join((*groups, str(source.name))) if groups else str(source.name)
    blend_value = getattr(source.blend_mode, "value", source.blend_mode)
    common = {
        "name": name, "opacity": float(source.opacity) / 255.0, "visible": bool(source.is_visible()),
        "locked": bool(int(source.locks or 0)),
        "mask": _mask(source, image.size), "blend_mode": PSD_TO_BLEND.get(blend_value, "Normal"),
        "clipping": bool(source.clipping),
    }
    if source.kind == "type":
        full = np.zeros((document_size[1], document_size[0], 4), dtype=np.uint8)
        x1, y1 = max(0, source.left), max(0, source.top)
        x2, y2 = min(document_size[0], source.left + pixels.shape[1]), min(document_size[1], source.top + pixels.shape[0])
        if x1 < x2 and y1 < y2:
            full[y1:y2, x1:x2] = pixels[y1 - source.top:y2 - source.top, x1 - source.left:x2 - source.left]
        if common["mask"] is not None:
            full_mask = np.zeros((document_size[1], document_size[0]), dtype=np.uint8)
            if x1 < x2 and y1 < y2:
                full_mask[y1:y2, x1:x2] = common["mask"][y1 - source.top:y2 - source.top, x1 - source.left:x2 - source.left]
            common["mask"] = full_mask
        layer = Layer(pixels=full, x=0, y=0, kind="text", text_data=_text_data(source, pixels), **common)
        _preserve_working_pixels(layer, source, depth, color_model)
        return layer, "Текст импортирован редактируемым с приблизительными параметрами шрифта"
    if source.kind == "smartobject":
        nested = _embedded_smart_document(source, document_type)
        smart_data = {
            "linked": False, "content_type": "document" if nested else "image",
            "original_size": [pixels.shape[1], pixels.shape[0]],
            "transform": {"width": pixels.shape[1], "height": pixels.shape[0], "angle": 0.0, "flip_horizontal": False, "flip_vertical": False},
            "psd_import": True,
        }
        if nested:
            smart_data["nested_document"] = nested
        layer = Layer(pixels=pixels, x=int(source.left), y=int(source.top), kind="embedded", smart_source=pixels.copy(), smart_data=smart_data, **common)
        _preserve_working_pixels(layer, source, depth, color_model)
        return layer, None if nested else "Smart Object импортирован по сохранённому изображению без исходного содержимого"
    layer = Layer(pixels=pixels, x=int(source.left), y=int(source.top), **common)
    _preserve_working_pixels(layer, source, depth, color_model)
    warning = None if source.kind == "pixel" else f"Слой типа {source.kind} импортирован как растровый"
    return layer, warning


def _load_psd_image(psd: PSDImage, document_type, source_name: str) -> "Document":
    metadata: dict[str, Any] = {"source": "PSD/PSB", "source_path": source_name}
    profile = _resource_data(psd, Resource.ICC_PROFILE)
    if isinstance(profile, bytes) and profile:
        settings = color_settings(metadata)
        settings["profile_name"] = profile_name(profile)
        settings["icc_base64"] = base64.b64encode(profile).decode("ascii")
    mode_name = str(getattr(psd.color_mode, "name", psd.color_mode)).upper()
    color_model = "CMYK" if mode_name == "CMYK" else "Lab" if mode_name == "LAB" else "RGBA"
    document = document_type(
        width=int(psd.width), height=int(psd.height), dpi=_dpi(psd), bit_depth=int(psd.depth),
        color_model=color_model, background=(0, 0, 0, 0), metadata=metadata,
    )
    warnings: list[str] = []
    for source, groups in _leaf_layers(psd):
        layer, warning = _import_layer(source, groups, document_type, psd.size, int(psd.depth), color_model)
        document.layers.append(layer)
        for risk in [warning, *_compatibility_risks(source)]:
            if risk and risk not in warnings:
                warnings.append(risk)
    if not document.layers:
        composite = psd.composite(apply_icc=False)
        document.layers.append(Layer("Сведённое изображение", pil_to_rgba_array(composite.convert("RGBA"))))
    document.active_layer = len(document.layers) - 1
    document.path = source_name
    document.dirty = False
    document.metadata["psd_compatibility"] = {
        "version": int(psd.version), "format": "PSB" if int(psd.version) == 2 else "PSD", "warnings": warnings,
    }
    return document


def load_psd(path: str | Path, document_type) -> "Document":
    try:
        return _load_psd_image(PSDImage.open(path), document_type, str(Path(path).resolve()))
    except Exception as exc:
        raise PSDCompatibilityError(f"Не удалось открыть PSD/PSB: {exc}") from exc


def _set_image_resource(psd: PSDImage, key: Resource, data: bytes) -> None:
    psd.image_resources[key] = ImageResource(key=int(key), data=data)


def _legacy_layer_name(name: str) -> str:
    try:
        name.encode("macroman")
        return name
    except UnicodeEncodeError:
        return "Layer"


def _create_pixel_layer(psd: PSDImage, pixels: np.ndarray, name: str, x: int, y: int, opacity: float, blend_mode: str):
    created = psd.create_pixel_layer(
        rgba_array_to_pil(display_rgba(pixels)), name=_legacy_layer_name(name), top=int(y), left=int(x),
        opacity=max(0, min(255, round(opacity * 255))), blend_mode=BLEND_TO_PSD.get(blend_mode, BlendMode.NORMAL),
    )
    created.name = name
    return created


def export_psd(document: "Document", path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.suffix.lower() not in PSD_EXTENSIONS:
        raise PSDCompatibilityError("Расширение файла должно быть .psd или .psb")
    psd = PSDImage.new("RGBA", (document.width, document.height), color=(0, 0, 0, 0), depth=document.bit_depth)
    if target.suffix.lower() == ".psb":
        psd._record.header.version = 2
    warnings: list[str] = []
    has_adjustments = any(layer.kind == "adjustment" for layer in document.layers)
    for source in document.layers:
        if source.kind == "adjustment":
            warnings.append("Корректирующие слои добавлены в сведённую визуальную копию")
            continue
        pixels = render_layer_pixels(source)
        effects, styled_pixels = render_layer_style(source, pixels)
        for index, (effect_pixels, x, y, opacity, blend_mode) in enumerate(effects, 1):
            effect = _create_pixel_layer(psd, effect_pixels, f"{source.name} · эффект {index}", x, y, opacity, blend_mode)
            effect.visible = bool(source.visible) and not has_adjustments
        created = _create_pixel_layer(psd, styled_pixels, source.name, source.x, source.y, source.opacity, source.blend_mode)
        created.visible = bool(source.visible) and not has_adjustments
        created.clipping = bool(source.clipping)
        if source.locked:
            created.tagged_blocks.set_data(Tag.PROTECTED_SETTING, ProtectedSetting(int(ProtectedFlags.COMPLETE)))
        if source.mask is not None:
            created.create_mask(Image.fromarray(source.mask.astype(np.uint8), "L"), top=int(source.y), left=int(source.x))
            if not source.mask_enabled or source.mask_density != 1.0 or source.mask_feather != 0.0:
                warnings.append(f"Дополнительные параметры маски «{source.name}» требуют проверки в Photoshop")
        if source.kind != "raster" or source.effects or source.filters or source.transform_data:
            warnings.append(f"Слой «{source.name}» экспортирован как пиксельный")
    if has_adjustments:
        merged = _create_pixel_layer(psd, document.composite(False), "Сведённая визуальная копия", 0, 0, 1.0, "Normal")
        merged.visible = True
    dpi = max(1, int(document.dpi))
    resolution = ResoulutionInfo(dpi * 65536, 1, 1, dpi * 65536, 1, 1)
    _set_image_resource(psd, Resource.RESOLUTION_INFO, resolution.tobytes())
    encoded = color_settings(document.metadata).get("icc_base64")
    if encoded:
        _set_image_resource(psd, Resource.ICC_PROFILE, base64.b64decode(encoded))
    try:
        psd.save(target)
    except Exception as exc:
        raise PSDCompatibilityError(f"Не удалось сохранить PSD/PSB: {exc}") from exc
    return {"path": str(target), "format": "PSB" if target.suffix.lower() == ".psb" else "PSD", "warnings": warnings}


__all__ = ["PSDCompatibilityError", "PSD_EXTENSIONS", "export_psd", "load_psd"]
