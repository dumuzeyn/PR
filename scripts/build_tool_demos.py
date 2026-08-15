from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tool_demo_actual import DemoFrame, SIZE, actual_frame, selection_outline
from uzyro.app_shared import TOOL_DEFINITIONS


ASSET_DIR = ROOT / "uzyro" / "assets" / "tool_demos"
SOURCE_DIR = ROOT / "design_assets" / "tool_demo_sources"
FRAME_COUNT = 18
ENGINE_TAG = "UZYRO actual tool engine v1"
TOOLS = tuple(tool_id for _label, tool_id, _description in TOOL_DEFINITIONS)
TOOL_LABELS = {tool_id: label for label, tool_id, _description in TOOL_DEFINITIONS}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    filename = "segoeuib.ttf" if bold else "segoeui.ttf"
    try:
        return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size)
    except OSError:
        return ImageFont.load_default()


def cover(path: Path) -> Image.Image:
    return ImageOps.fit(Image.open(path).convert("RGB"), SIZE, Image.Resampling.LANCZOS)


def progress(index: int) -> float:
    if index < 3:
        return 0.0
    if index >= FRAME_COUNT - 4:
        return 1.0
    return (index - 3) / (FRAME_COUNT - 8)


def dashed_segment(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    phase: int,
    color: str = "#151515",
) -> None:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max(1, round(math.hypot(dx, dy)))
    for position in range(-phase, length, 8):
        low, high = max(0, position), min(length, position + 4)
        if high <= low:
            continue
        first = (start[0] + dx * low / length, start[1] + dy * low / length)
        second = (start[0] + dx * high / length, start[1] + dy * high / length)
        draw.line((first, second), fill="#ffffff", width=3)
        draw.line((first, second), fill=color, width=1)


def dashed_path(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], phase: int, closed: bool = False) -> None:
    path = points + [points[0]] if closed and len(points) > 2 else points
    for first, second in zip(path, path[1:]):
        dashed_segment(draw, first, second, phase)


def draw_cursor(image: Image.Image, point: tuple[int, int], ring: int = 0) -> None:
    x, y = point
    draw = ImageDraw.Draw(image)
    if ring:
        draw.ellipse((x - ring, y - ring, x + ring, y + ring), outline="#ffffff", width=3)
        draw.ellipse((x - ring - 1, y - ring - 1, x + ring + 1, y + ring + 1), outline="#151515")
    points = [(x, y), (x + 3, y + 17), (x + 7, y + 12), (x + 12, y + 19), (x + 16, y + 16), (x + 10, y + 9), (x + 17, y + 7)]
    draw.polygon(points, fill="#ffffff", outline="#111111")


def draw_selection(draw: ImageDraw.ImageDraw, frame: DemoFrame, phase: int) -> None:
    if frame.selection is not None:
        overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        alpha = Image.fromarray((frame.selection.astype("float32") * 0.18).astype("uint8"), mode="L")
        overlay.paste((44, 145, 230, 255), (0, 0, *SIZE), alpha)
        draw._image.alpha_composite(overlay)
        for contour in selection_outline(frame.selection):
            dashed_path(draw, contour, phase, True)
    if frame.guide:
        dashed_path(draw, frame.guide, phase, frame.guide_closed)


def draw_path_nodes(draw: ImageDraw.ImageDraw, frame: DemoFrame) -> None:
    for index, node in enumerate(frame.nodes):
        anchor = tuple(round(float(value)) for value in node["anchor"])
        incoming = tuple(round(float(value)) for value in node.get("in", node["anchor"]))
        outgoing = tuple(round(float(value)) for value in node.get("out", node["anchor"]))
        draw.line((incoming, anchor, outgoing), fill="#f0b84f", width=1)
        for handle in (incoming, outgoing):
            draw.ellipse((handle[0] - 3, handle[1] - 3, handle[0] + 3, handle[1] + 3), fill="#ffffff", outline="#222222")
        color = "#845cf6" if index == frame.active_node else "#ffffff"
        draw.rectangle((anchor[0] - 5, anchor[1] - 5, anchor[0] + 5, anchor[1] + 5), fill=color, outline="#161616", width=2)


def draw_source_marker(draw: ImageDraw.ImageDraw, source: tuple[int, int]) -> None:
    x, y = source
    draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline="#ffffff", width=3)
    draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline="#845cf6", width=2)
    draw.line((x - 14, y, x + 14, y), fill="#ffffff", width=1)
    draw.line((x, y - 14, x, y + 14), fill="#ffffff", width=1)


def draw_swatch(draw: ImageDraw.ImageDraw, color: tuple[int, int, int, int]) -> None:
    draw.rectangle((242, 12, 279, 49), fill="#ffffff", outline="#141414", width=2)
    draw.rectangle((247, 17, 274, 44), fill=color)


def banner(image: Image.Image, tool: str, index: int) -> None:
    if index < 3:
        text, color = "ДО", "#4f5660"
    elif index >= FRAME_COUNT - 4:
        text, color = "РЕЗУЛЬТАТ", "#167447"
    else:
        text, color = TOOL_LABELS[tool].upper(), "#5f3fc4"
    draw = ImageDraw.Draw(image, "RGBA")
    label_font = font(10, True)
    bounds = draw.textbbox((0, 0), text, font=label_font)
    width = min(SIZE[0] - 12, bounds[2] - bounds[0] + 14)
    draw.rounded_rectangle((6, 6, 6 + width, 25), radius=3, fill=color + "EC")
    draw.text((13, 8), text, font=label_font, fill="#ffffff")


def compose(frame: DemoFrame, tool: str, index: int) -> Image.Image:
    image = frame.image.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    draw_selection(draw, frame, index % 8)
    if frame.nodes:
        draw_path_nodes(draw, frame)
    if frame.source is not None:
        draw_source_marker(draw, frame.source)
    if frame.swatch is not None:
        draw_swatch(draw, frame.swatch)
    ring = 9 if tool in {"brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing", "quick_selection"} else 0
    draw_cursor(image, frame.cursor, ring)
    banner(image, tool, index)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, SIZE[1] - 3, SIZE[0], SIZE[1]), fill="#202228")
    elapsed = round(SIZE[0] * (index + 1) / FRAME_COUNT)
    draw.rectangle((0, SIZE[1] - 3, elapsed, SIZE[1]), fill="#845cf6")
    return image.convert("RGB")


def make_frames(tool: str, still: Image.Image, landscape: Image.Image) -> list[Image.Image]:
    return [compose(actual_frame(tool, progress(index), still, landscape), tool, index) for index in range(FRAME_COUNT)]


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_animation(tool: str, frames: list[Image.Image]) -> None:
    output_path = ASSET_DIR / f"{tool}.gif"
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    frames[0].save(
        temporary,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=110,
        loop=0,
        optimize=True,
        disposal=2,
        comment=ENGINE_TAG.encode("ascii"),
    )
    os.replace(temporary, output_path)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    still_path = SOURCE_DIR / "still_life.png"
    landscape_path = SOURCE_DIR / "landscape.png"
    still, landscape = cover(still_path), cover(landscape_path)
    manifest = {
        "format": "UZYRO tool demos v2",
        "generator": ENGINE_TAG,
        "frame_count": FRAME_COUNT,
        "size": list(SIZE),
        "sources": {still_path.name: source_hash(still_path), landscape_path.name: source_hash(landscape_path)},
        "tools": {},
    }
    for tool in TOOLS:
        frames = make_frames(tool, still, landscape)
        save_animation(tool, frames)
        manifest["tools"][tool] = {"file": f"{tool}.gif", "engine": "uzyro"}
        print(f"created {tool}.gif")
    (ASSET_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
