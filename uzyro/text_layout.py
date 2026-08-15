from __future__ import annotations

from dataclasses import dataclass

from .text_shaper import DEFAULT_TEXT_SHAPER, TextShaper, TextStyle, grapheme_clusters, style_from_data


@dataclass(frozen=True)
class TextObject:
    text: str
    x: int
    y: int
    box_width: int
    box_height: int
    mode: str
    align: str
    leading: int
    indent_left: int
    indent_right: int
    first_line_indent: int
    spacing_before: int
    spacing_after: int
    baseline_shift: int
    style: TextStyle

    @classmethod
    def from_data(cls, data: dict) -> "TextObject":
        width = max(0, int(data.get("box_width", 0) or 0))
        return cls(
            text=str(data.get("text", "")), x=int(data.get("x", 0)), y=int(data.get("y", 0)),
            box_width=width, box_height=max(0, int(data.get("box_height", 0) or 0)),
            mode=str(data.get("text_mode", "paragraph" if width else "point")),
            align=str(data.get("align", "left")).lower(),
            leading=max(0, int(data.get("line_spacing", max(2, int(data.get("size", 48)) // 5)))),
            indent_left=max(0, int(data.get("indent_left", 0))),
            indent_right=max(0, int(data.get("indent_right", 0))),
            first_line_indent=int(data.get("first_line_indent", 0)),
            spacing_before=max(0, int(data.get("spacing_before", 0))),
            spacing_after=max(0, int(data.get("spacing_after", 0))),
            baseline_shift=int(data.get("baseline_shift", 0)), style=style_from_data(data),
        )


@dataclass(frozen=True)
class LayoutLine:
    text: str
    x: float
    y: float
    width: float
    available: float
    justify: bool


@dataclass(frozen=True)
class TextLayout:
    lines: tuple[LayoutLine, ...]
    line_height: int
    overflow: bool


class TextLayoutEngine:
    def __init__(self, shaper: TextShaper | None = None) -> None:
        self.shaper = shaper or DEFAULT_TEXT_SHAPER

    def measure(self, text: str, style: TextStyle) -> float:
        return self.shaper.shape(text or " ", style).advance if text else 0.0

    def _break_word(self, word: str, width: float, style: TextStyle) -> list[str]:
        if width <= 0 or self.measure(word, style) <= width:
            return [word]
        chunks: list[str] = []
        current = ""
        for cluster in grapheme_clusters(word):
            candidate = current + cluster
            if current and self.measure(candidate, style) > width:
                chunks.append(current)
                current = cluster
            else:
                current = candidate
        if current or not chunks:
            chunks.append(current)
        return chunks

    def _wrap(self, paragraph: str, width: float, style: TextStyle) -> list[str]:
        if width <= 0:
            return [paragraph]
        if not paragraph:
            return [""]
        lines: list[str] = []
        current = ""
        for word in paragraph.split(" "):
            pieces = self._break_word(word, width, style)
            for piece_index, piece in enumerate(pieces):
                separator = " " if current and piece_index == 0 else ""
                candidate = current + separator + piece
                if current and self.measure(candidate, style) > width:
                    lines.append(current)
                    current = piece
                else:
                    current = candidate
                if piece_index < len(pieces) - 1:
                    lines.append(current)
                    current = ""
        lines.append(current)
        return lines

    def layout(self, data: dict) -> TextLayout:
        text_object = TextObject.from_data(data)
        font = self.shaper.shape(" ", text_object.style).font
        try:
            ascent, descent = font.getmetrics()
            natural_height = max(1, int(ascent + descent))
        except AttributeError:
            bbox = font.getbbox("Mg")
            natural_height = max(1, int(bbox[3] - bbox[1]))
        line_height = natural_height + text_object.leading
        paragraph_width = max(1, text_object.box_width - text_object.indent_left - text_object.indent_right) if text_object.mode == "paragraph" and text_object.box_width else 0
        lines: list[LayoutLine] = []
        cursor_y = float(text_object.y)
        overflow = False
        paragraphs = text_object.text.split("\n")
        for paragraph_index, paragraph in enumerate(paragraphs):
            cursor_y += text_object.spacing_before
            wrapped = self._wrap(paragraph, paragraph_width, text_object.style)
            for line_index, line in enumerate(wrapped):
                first_offset = text_object.first_line_indent if line_index == 0 else 0
                available = max(1.0, paragraph_width - first_offset) if paragraph_width else 0.0
                width = self.measure(line, text_object.style)
                x = float(text_object.x + text_object.indent_left + first_offset)
                if available and text_object.align == "center":
                    x += max(0.0, (available - width) * 0.5)
                elif available and text_object.align == "right":
                    x += max(0.0, available - width)
                if text_object.mode == "paragraph" and text_object.box_height and cursor_y + natural_height > text_object.y + text_object.box_height:
                    overflow = True
                    continue
                lines.append(LayoutLine(line, x, cursor_y - text_object.baseline_shift, width, available, bool(available and text_object.align == "justify" and line_index < len(wrapped) - 1 and " " in line)))
                cursor_y += line_height
            cursor_y += text_object.spacing_after
            if paragraph_index < len(paragraphs) - 1 and not wrapped:
                cursor_y += line_height
        return TextLayout(tuple(lines), line_height, overflow)


DEFAULT_TEXT_LAYOUT = TextLayoutEngine()


__all__ = ["DEFAULT_TEXT_LAYOUT", "LayoutLine", "TextLayout", "TextLayoutEngine", "TextObject"]
