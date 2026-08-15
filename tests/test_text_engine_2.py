from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from uzyro.core import Document
from uzyro.text_layout import TextLayoutEngine
from uzyro.text_shaper import DEFAULT_TEXT_SHAPER, FontResolver, TextStyle, grapheme_clusters, text_direction


def test_bundled_fribidi_runtime_is_present_and_verified() -> None:
    library = Path(__file__).parents[1] / "uzyro" / "assets" / "native" / "fribidi-0.dll"
    assert library.is_file()
    assert hashlib.sha256(library.read_bytes()).hexdigest() == (
        "463c7479d434a8681a9dbde16d0675e28d09d38fa43e3e18d826e345584ba18d"
    )


def alpha_bounds(pixels: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(pixels[:, :, 3] > 0)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def test_raqm_uses_real_font_kerning_independently_from_tracking() -> None:
    shaper = DEFAULT_TEXT_SHAPER
    kerned = shaper.shape("AV To WA", TextStyle("Arial", 60, kerning=True))
    unkerned = shaper.shape("AV To WA", TextStyle("Arial", 60, kerning=False))
    tracked = shaper.shape("AV To WA", TextStyle("Arial", 60, kerning=True, tracking=8))
    assert shaper.available
    assert kerned.advance < unkerned.advance
    assert tracked.advance == kerned.advance + (len(grapheme_clusters("AV To WA")) - 1) * 8


def test_ligatures_combining_marks_and_rtl_are_shaped_as_runs() -> None:
    with_ligature = DEFAULT_TEXT_SHAPER.shape("office", TextStyle("Calibri", 60, standard_ligatures=True))
    without_ligature = DEFAULT_TEXT_SHAPER.shape("office", TextStyle("Calibri", 60, standard_ligatures=False))
    assert with_ligature.advance < without_ligature.advance
    assert grapheme_clusters("A\u0301") == ["A\u0301"]
    arabic = "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"
    run = DEFAULT_TEXT_SHAPER.shape(arabic, TextStyle("Arial", 48))
    assert run.direction == "rtl"
    assert text_direction("English " + arabic) == "ltr"
    assert run.advance > 20 and run.bbox[2] > run.bbox[0]
    emoji = DEFAULT_TEXT_SHAPER.shape("Фото \U0001f3a8", TextStyle("Arial", 48))
    assert "seguiemj" in str(getattr(emoji.font, "path", "")).lower()


def test_paragraph_layout_wraps_long_words_and_reports_vertical_overflow() -> None:
    engine = TextLayoutEngine()
    data = {
        "text": "Оченьдлинноесловобезпробелов и обычный текст для переноса",
        "x": 10, "y": 15, "font_family": "Arial", "size": 28,
        "box_width": 150, "box_height": 78, "text_mode": "paragraph",
        "line_spacing": 4, "tracking": 0,
    }
    layout = engine.layout(data)
    assert len(layout.lines) >= 1
    assert layout.overflow
    assert all(line.width <= 151 for line in layout.lines)
    point = engine.layout({**data, "text_mode": "point", "box_height": 0})
    assert len(point.lines) == 1


def test_paragraph_box_clips_rendering_and_point_text_expands() -> None:
    text = "Первая строка длинного абзаца в блоке. Вторая часть должна уйти ниже."
    document = Document.new(500, 260, (0, 0, 0, 0))
    paragraph = document.add_text_layer(text, 20, 20, (15, 25, 35, 255), 30, "Arial", 190, box_height=62, text_mode="paragraph")
    _x1, _y1, _x2, paragraph_bottom = alpha_bounds(paragraph.pixels)
    assert paragraph_bottom <= 82
    point = document.add_text_layer(text, 20, 120, (15, 25, 35, 255), 30, "Arial", 0, text_mode="point")
    point_bounds = alpha_bounds(point.pixels)
    assert point_bounds[2] - point_bounds[0] > 190


def test_opentype_and_paragraph_properties_roundtrip(tmp_path) -> None:
    document = Document.new(420, 220, (0, 0, 0, 0))
    layer = document.add_text_layer(
        "office AV Русский", 15, 20, (0, 0, 0, 255), 34, "Calibri", 260,
        box_height=120, text_mode="paragraph", kerning_enabled=False,
        standard_ligatures=False, discretionary_ligatures=True, stylistic_set=3,
        direction="ltr", language="ru",
    )
    path = tmp_path / "text-engine-2.prdx"
    document.save_project(path)
    restored = Document.open_project(path)
    for key in (
        "box_height", "text_mode", "kerning_enabled", "standard_ligatures",
        "discretionary_ligatures", "stylistic_set", "direction", "language",
    ):
        assert restored.layer.text_data[key] == layer.text_data[key]


def test_font_resolver_uses_bounded_lru_cache() -> None:
    resolver = FontResolver(cache_capacity=12)
    first = resolver.load(TextStyle("Arial", 8))
    for size in range(9, 50):
        resolver.load(TextStyle("Arial", size))
    assert len(resolver._cache) == 12
    assert first not in resolver._cache.values()
