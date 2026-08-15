from __future__ import annotations

import ctypes
import os
import sys
import unicodedata
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path


def _load_bundled_fribidi() -> object | None:
    if os.name != "nt":
        return None
    roots = [Path(__file__).resolve().parent / "assets" / "native"]
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        roots.insert(0, Path(bundled))
    for root in roots:
        library = root / "fribidi-0.dll"
        if library.is_file():
            return ctypes.WinDLL(str(library))
    return None


_FRIBIDI_HANDLE = _load_bundled_fribidi()

from PIL import ImageDraw, ImageFont, features


@dataclass(frozen=True)
class TextStyle:
    family: str = "Arial"
    size: int = 48
    bold: bool = False
    italic: bool = False
    tracking: float = 0.0
    kerning: bool = True
    standard_ligatures: bool = True
    discretionary_ligatures: bool = False
    stylistic_set: int = 0
    direction: str = "auto"
    language: str = ""


@dataclass(frozen=True)
class GlyphRun:
    text: str
    font: ImageFont.ImageFont
    advance: float
    bbox: tuple[int, int, int, int]
    direction: str | None
    language: str | None
    features: tuple[str, ...]
    clusters: tuple[str, ...]


def grapheme_clusters(text: str) -> list[str]:
    """A small Unicode grapheme fallback that keeps marks, selectors and ZWJ sequences together."""
    clusters: list[str] = []
    join_next = False
    for character in text:
        code = ord(character)
        combining = bool(unicodedata.combining(character)) or 0xFE00 <= code <= 0xFE0F or 0x1F3FB <= code <= 0x1F3FF
        if clusters and (combining or join_next or character == "\u200d"):
            clusters[-1] += character
        else:
            clusters.append(character)
        join_next = character == "\u200d"
    return clusters


def text_direction(text: str, requested: str = "auto") -> str | None:
    normalized = str(requested).lower()
    if normalized in {"ltr", "rtl", "ttb"}:
        return normalized
    for character in text:
        bidi = unicodedata.bidirectional(character)
        if bidi in {"R", "AL", "AN"}:
            return "rtl"
        if bidi == "L":
            return "ltr"
    return None


def opentype_features(style: TextStyle) -> tuple[str, ...]:
    result = ["kern" if style.kerning else "-kern", "liga" if style.standard_ligatures else "-liga"]
    result.append("dlig" if style.discretionary_ligatures else "-dlig")
    if 1 <= int(style.stylistic_set) <= 20:
        result.append(f"ss{int(style.stylistic_set):02d}")
    return tuple(result)


class FontResolver:
    def __init__(self, cache_capacity: int = 64) -> None:
        self.cache_capacity = max(8, int(cache_capacity))
        self._cache: OrderedDict[tuple[str, int, bool, bool, bool], ImageFont.ImageFont] = OrderedDict()

    def _remember(self, key: tuple[str, int, bool, bool, bool], font: ImageFont.ImageFont) -> ImageFont.ImageFont:
        self._cache[key] = font
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)
        return font

    @staticmethod
    def candidates(family: str, bold: bool, italic: bool) -> list[str]:
        name = family.strip() or "Arial"
        compact = name.lower().replace(" ", "").removesuffix(".ttf")
        values: list[str] = []
        if compact == "arial":
            values.append("arialbi.ttf" if bold and italic else "arialbd.ttf" if bold else "ariali.ttf" if italic else "arial.ttf")
        elif compact in {"segoeuiemoji", "seguiemj"}:
            values.append("seguiemj.ttf")
        elif compact in {"seguisymbol", "seguisym"}:
            values.append("seguisym.ttf")
        if not Path(name).suffix and (bold or italic):
            suffix = " Bold Italic" if bold and italic else " Bold" if bold else " Italic"
            values.extend([f"{name}{suffix}.ttf", f"{compact}{'bi' if bold and italic else 'bd' if bold else 'i'}.ttf"])
        values.append(name)
        if not Path(name).suffix:
            values.extend([f"{name}.ttf", f"{compact}.ttf"])
        values.extend(["segoeui.ttf", "seguisym.ttf", "seguiemj.ttf", "arial.ttf"])
        return list(dict.fromkeys(values))

    def load(self, style: TextStyle) -> ImageFont.ImageFont:
        use_raqm = bool(features.check_feature("raqm"))
        key = (style.family, int(style.size), bool(style.bold), bool(style.italic), use_raqm)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        engine = ImageFont.Layout.RAQM if use_raqm else ImageFont.Layout.BASIC
        for candidate in self.candidates(style.family, style.bold, style.italic):
            try:
                font = ImageFont.truetype(candidate, max(4, int(style.size)), layout_engine=engine)
                return self._remember(key, font)
            except OSError:
                continue
        font = ImageFont.load_default()
        return self._remember(key, font)


class TextShaper(ABC):
    @abstractmethod
    def shape(self, text: str, style: TextStyle) -> GlyphRun:
        raise NotImplementedError

    @abstractmethod
    def draw(self, draw: ImageDraw.ImageDraw, xy: tuple[float, float], run: GlyphRun, fill, stroke_width: int = 0) -> None:
        raise NotImplementedError


class PillowRaqmTextShaper(TextShaper):
    def __init__(self, resolver: FontResolver | None = None) -> None:
        self.resolver = resolver or FontResolver()
        self.available = bool(features.check_feature("raqm"))

    def shape(self, text: str, style: TextStyle) -> GlyphRun:
        if any(ord(character) >= 0x1F000 for character in text):
            style = replace(style, family="Segoe UI Emoji", bold=False, italic=False)
        font = self.resolver.load(style)
        direction = text_direction(text, style.direction) if self.available else None
        language = style.language.strip() or None
        feature_list = opentype_features(style) if self.available else ()
        kwargs = {"direction": direction, "language": language, "features": list(feature_list)} if self.available else {}
        advance = float(font.getlength(text or " ", **kwargs))
        clusters = tuple(grapheme_clusters(text))
        advance += max(0, len(clusters) - 1) * float(style.tracking)
        bbox = tuple(int(value) for value in font.getbbox(text or " ", **kwargs))
        return GlyphRun(text, font, max(0.0, advance), bbox, direction, language, feature_list, clusters)

    def draw(self, draw: ImageDraw.ImageDraw, xy: tuple[float, float], run: GlyphRun, fill, stroke_width: int = 0) -> None:
        kwargs = {
            "direction": run.direction, "language": run.language, "features": list(run.features),
            "stroke_width": max(0, int(stroke_width)), "stroke_fill": fill, "embedded_color": True,
        } if self.available else {"stroke_width": max(0, int(stroke_width)), "stroke_fill": fill}
        complex_joining = any(unicodedata.bidirectional(character) in {"R", "AL", "AN"} for character in run.text)
        if complex_joining or len(run.clusters) <= 1 or abs(run.advance - float(run.font.getlength(run.text or " ", **({"direction": run.direction, "language": run.language, "features": list(run.features)} if self.available else {})))) < 0.01:
            draw.text(xy, run.text, fill=fill, font=run.font, **kwargs)
            return
        # Tracking is applied to grapheme clusters, never to individual code points.
        metric_kwargs = {"direction": run.direction, "language": run.language, "features": list(run.features)} if self.available else {}
        base_advance = float(run.font.getlength(run.text or " ", **metric_kwargs))
        tracking = (run.advance - base_advance) / max(1, len(run.clusters) - 1)
        prefix = ""
        for index, cluster in enumerate(run.clusters):
            cursor = float(xy[0]) + float(run.font.getlength(prefix, **metric_kwargs)) + tracking * index
            draw.text((cursor, xy[1]), cluster, fill=fill, font=run.font, **kwargs)
            prefix += cluster


DEFAULT_TEXT_SHAPER = PillowRaqmTextShaper()


def style_from_data(data: dict) -> TextStyle:
    return TextStyle(
        family=str(data.get("font_family", "Arial")), size=max(4, int(data.get("size", 48))),
        bold=bool(data.get("bold", False)), italic=bool(data.get("italic", False)),
        tracking=float(data.get("tracking", 0)), kerning=bool(data.get("kerning_enabled", True)),
        standard_ligatures=bool(data.get("standard_ligatures", True)),
        discretionary_ligatures=bool(data.get("discretionary_ligatures", False)),
        stylistic_set=int(data.get("stylistic_set", 0)), direction=str(data.get("direction", "auto")),
        language=str(data.get("language", "")),
    )


__all__ = [
    "DEFAULT_TEXT_SHAPER", "FontResolver", "GlyphRun", "PillowRaqmTextShaper", "TextShaper", "TextStyle",
    "grapheme_clusters", "opentype_features", "style_from_data", "text_direction",
]
