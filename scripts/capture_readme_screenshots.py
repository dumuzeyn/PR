from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "docs" / "screenshots"
SOURCE = ROOT / "design_assets" / "tool_demo_sources"


class BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("size", wintypes.DWORD),
        ("width", wintypes.LONG),
        ("height", wintypes.LONG),
        ("planes", wintypes.WORD),
        ("bit_count", wintypes.WORD),
        ("compression", wintypes.DWORD),
        ("image_size", wintypes.DWORD),
        ("x_pixels_per_meter", wintypes.LONG),
        ("y_pixels_per_meter", wintypes.LONG),
        ("colors_used", wintypes.DWORD),
        ("colors_important", wintypes.DWORD),
    ]


class BitmapInfo(ctypes.Structure):
    _fields_ = [("header", BitmapInfoHeader), ("colors", wintypes.DWORD * 3)]


class Rect(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG), ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


def pump(window, seconds: float = 0.35) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        window.update()
        time.sleep(0.01)


def capture_window(window, target: Path, max_width: int = 1500) -> None:
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    client = int(window.winfo_id())
    handle = int(user32.GetParent(client)) or client
    bounds = Rect()
    if not user32.GetWindowRect(handle, ctypes.byref(bounds)):
        raise OSError("Could not read the UZYRO window bounds")
    width, height = bounds.right - bounds.left, bounds.bottom - bounds.top
    window_dc = user32.GetWindowDC(handle)
    memory_dc = gdi32.CreateCompatibleDC(window_dc)
    bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
    previous = gdi32.SelectObject(memory_dc, bitmap)
    try:
        if not user32.PrintWindow(handle, memory_dc, 2):
            raise OSError("Windows could not render the UZYRO window")
        header = BitmapInfoHeader(
            ctypes.sizeof(BitmapInfoHeader), width, -height, 1, 32, 0,
            width * height * 4, 0, 0, 0, 0,
        )
        info = BitmapInfo(header)
        buffer = ctypes.create_string_buffer(width * height * 4)
        if not gdi32.GetDIBits(memory_dc, bitmap, 0, height, buffer, ctypes.byref(info), 0):
            raise OSError("Could not copy pixels from the UZYRO window")
        image = Image.frombuffer("RGB", (width, height), buffer, "raw", "BGRX", 0, 1).copy()
        if image.width > max_width:
            image.thumbnail((max_width, round(image.height * max_width / image.width)), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, optimize=True)
    finally:
        gdi32.SelectObject(memory_dc, previous)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(handle, window_dc)


def source_image(name: str, size: tuple[int, int] = (1600, 900)) -> Image.Image:
    return ImageOps.fit(Image.open(SOURCE / name).convert("RGBA"), size, Image.Resampling.LANCZOS)


def document_from(image: Image.Image, name: str):
    from uzyro.core import Document

    pixels = np.asarray(image, dtype=np.uint8).copy()
    document = Document.new(image.width, image.height, (0, 0, 0, 0))
    document.layer.name = name
    document.layer.pixels = pixels
    document.layer.touch_pixels()
    document.dirty = False
    return document


def show_document(app, document, tool: str) -> None:
    app.open_document_session(document, replace_startup=True)
    app.tool.set(tool)
    pump(app, 0.65)
    app.state("normal")
    app.geometry("1500x920+120+70")
    app.update_idletasks()
    app.fit_to_screen()
    pump(app, 0.5)


def vector_showcase(app) -> None:
    document = document_from(source_image("landscape.png"), "Пейзаж")
    document.add_shape_layer(
        "rectangle", (310, 72, 1130, 270), (18, 22, 31, 190), (255, 255, 255, 105), 3,
    ).name = "Подложка"
    document.add_text_layer("UZYRO", 380, 98, (255, 255, 255, 255), 112, "segoeuib.ttf").name = "Заголовок"
    document.add_text_layer(
        "Графика без лишних границ", 440, 222, (205, 190, 255, 255), 32, "segoeui.ttf",
    ).name = "Подпись"
    document.active_layer = len(document.layers) - 2
    document.dirty = False
    show_document(app, document, "move")
    app.selected_layer_ids = {document.layer.id}
    app.refresh_layers()
    app.refresh()
    pump(app)


def painting_showcase(app) -> None:
    from uzyro.brush_engine import BrushSettings, PixelBrushStroke

    document = document_from(source_image("still_life.png"), "Натюрморт")
    document.add_layer("Мазок кисти", np.zeros((document.height, document.width, 4), dtype=np.uint8))
    stroke = PixelBrushStroke(
        document.layer,
        BrushSettings(radius=24, hardness=1.0, opacity=0.92, flow=1.0, spacing=0.0),
        (132, 92, 246, 255),
    )
    points = [(210 + index * 34, 220 + round(np.sin(index * 0.5) * 75)) for index in range(31)]
    for first, second in zip(points, points[1:]):
        for amount in np.linspace(0.0, 1.0, 8, endpoint=False):
            stroke.dab(round(first[0] + (second[0] - first[0]) * amount), round(first[1] + (second[1] - first[1]) * amount))
    document.layer.touch_pixels()
    document.dirty = False
    show_document(app, document, "brush")


def main() -> None:
    if os.name != "nt":
        raise SystemExit("README screenshots are captured from the Windows application")
    with tempfile.TemporaryDirectory(prefix="uzyro-readme-") as profile:
        os.environ["LOCALAPPDATA"] = profile
        from uzyro.app_mixins.startup_screen import StartupScreenMixin

        StartupScreenMixin.read_clipboard_image = staticmethod(lambda: None)
        from uzyro.app import UZYROApp

        app = UZYROApp()
        try:
            app.state("normal")
            pump(app)
            capture_window(app, OUTPUT / "01-start-screen.png")

            vector_showcase(app)
            capture_window(app, OUTPUT / "02-workspace.png")

            painting_showcase(app)
            capture_window(app, OUTPUT / "03-painting.png")
        finally:
            app.destroy()


if __name__ == "__main__":
    main()
