from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = (ROOT / "uzyro", ROOT / "tests")
MAX_SOURCE_LINES = 500


def test_python_sources_stay_within_reviewable_size() -> None:
    oversized: list[str] = []
    for directory in SOURCE_DIRECTORIES:
        for path in directory.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            if line_count > MAX_SOURCE_LINES:
                oversized.append(f"{path.relative_to(ROOT)}: {line_count} строк")
    assert not oversized, "Файлы необходимо разделить по предметным обязанностям:\n" + "\n".join(oversized)


def test_public_facades_keep_existing_entry_points() -> None:
    from uzyro.app import UZYROApp
    from uzyro.core import Document, Layer, apply_filter_stack, content_aware_fill
    from uzyro.history import History, LayerPropertyCommand
    from uzyro.ui.tool_options import ToolOptionsPanel
    from uzyro.ui.tool_palette import ToolPalette, ToolPaletteDialog

    assert UZYROApp.__name__ == "UZYROApp"
    assert Document.__name__ == "Document"
    assert Layer.__name__ == "Layer"
    assert callable(apply_filter_stack)
    assert callable(content_aware_fill)
    assert History.__name__ == "History"
    assert LayerPropertyCommand.__name__ == "LayerPropertyCommand"
    assert ToolOptionsPanel.__name__ == "ToolOptionsPanel"
    assert ToolPalette.__name__ == "ToolPalette"
    assert ToolPaletteDialog.__name__ == "ToolPaletteDialog"
