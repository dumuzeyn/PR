from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = ROOT / "uzyro"
TEST_ROOT = ROOT / "tests"
MAX_PRODUCTION_LINES = 500
MAX_TEST_LINES = 550


def oversized_python_files(directory: Path, limit: int) -> list[str]:
    oversized: list[str] = []
    for path in directory.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > limit:
            oversized.append(f"{path.relative_to(ROOT)}: {line_count} строк")
    return oversized


def test_production_modules_stay_within_reviewable_size() -> None:
    oversized = oversized_python_files(PRODUCTION_ROOT, MAX_PRODUCTION_LINES)
    assert not oversized, "Production-модули необходимо разделить по обязанностям:\n" + "\n".join(oversized)


def test_test_modules_do_not_become_monolithic_suites() -> None:
    oversized = oversized_python_files(TEST_ROOT, MAX_TEST_LINES)
    assert not oversized, "Файлы необходимо разделить по предметным обязанностям:\n" + "\n".join(oversized)


def test_public_facades_keep_existing_entry_points() -> None:
    from uzyro.app import UZYROApp
    from uzyro.core import Document, Layer
    from uzyro.history import History, LayerPropertyCommand
    from uzyro.ui.tool_options import ToolOptionsPanel
    from uzyro.ui.tool_palette import ToolPalette, ToolPaletteDialog

    public_types = {
        "UZYROApp": UZYROApp,
        "Document": Document,
        "Layer": Layer,
        "History": History,
        "LayerPropertyCommand": LayerPropertyCommand,
        "ToolOptionsPanel": ToolOptionsPanel,
        "ToolPalette": ToolPalette,
        "ToolPaletteDialog": ToolPaletteDialog,
    }
    assert {name: value.__name__ for name, value in public_types.items()} == {
        name: name for name in public_types
    }
    assert all(value.__module__.startswith("uzyro") for value in public_types.values())
