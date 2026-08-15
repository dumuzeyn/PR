from __future__ import annotations

from pathlib import Path

from PIL import Image

from uzyro.core import Document, render_shape_layer


ROOT = Path(__file__).resolve().parents[1]
DESIGN_DIR = ROOT / "design_assets" / "branding"
ASSET_DIR = ROOT / "uzyro" / "assets" / "branding"
SIZE = 2048
BACKGROUND = (18, 19, 22, 255)
WHITE = (250, 250, 252, 255)
PURPLE = (155, 111, 255, 255)


def add_rectangle(document: Document, box: tuple[int, int, int, int], color, rotation: float, radius: int, name: str):
    layer = document.add_shape_layer("rectangle", box, color)
    layer.name = name
    layer.shape_data["rotation"] = rotation
    layer.shape_data["corner_radius"] = radius
    render_shape_layer(layer)
    layer.touch_pixels()
    return layer


def build_logo() -> Document:
    document = Document.new(SIZE, SIZE, BACKGROUND)
    document.layers[0].name = "Тёмный фон"

    ring = document.add_shape_layer(
        "ellipse",
        (344, 344, 1704, 1704),
        (0, 0, 0, 0),
        WHITE,
        132,
    )
    ring.name = "Белое кольцо"

    add_rectangle(document, (356, 918, 1692, 1130), BACKGROUND, -40.0, 34, "Разрыв кольца")
    add_rectangle(document, (398, 950, 1650, 1098), PURPLE, -40.0, 26, "Фиолетовая диагональ")
    document.active_layer = len(document.layers) - 1
    document.metadata.update({"title": "UZYRO logo", "generator": "UZYRO vector engine"})
    return document


def main() -> None:
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    document = build_logo()
    project_path = DESIGN_DIR / "UZYRO-logo.prdx"
    source_path = DESIGN_DIR / "UZYRO-logo-2048.png"
    app_png_path = ASSET_DIR / "uzyro-icon.png"
    ico_path = ASSET_DIR / "uzyro.ico"

    document.save_project(project_path)
    source = Image.fromarray(document.composite(False), "RGBA")
    source.save(source_path, optimize=True)
    source.resize((512, 512), Image.Resampling.LANCZOS).save(app_png_path, optimize=True)
    source.save(
        ico_path,
        format="ICO",
        sizes=[(16, 16), (20, 20), (24, 24), (32, 32), (40, 40), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(project_path)
    print(source_path)
    print(app_png_path)
    print(ico_path)


if __name__ == "__main__":
    main()
