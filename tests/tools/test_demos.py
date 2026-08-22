import json

from PIL import Image, ImageChops

from uzyro.app import TOOL_DEFINITIONS, tool_demo_path


def test_every_tool_has_an_animated_demo() -> None:
    expected_tools = {tool_id for _label, tool_id, _description in TOOL_DEFINITIONS}
    for _label, tool_id, _description in TOOL_DEFINITIONS:
        path = tool_demo_path(tool_id)
        assert path.is_file(), f"missing demo for {tool_id}"
        with Image.open(path) as animation:
            assert animation.format == "GIF"
            assert animation.size == (288, 162)
            assert animation.n_frames == 18
            assert animation.info.get("comment") == b"UZYRO actual tool engine v1"
            animation.seek(0)
            first = animation.convert("RGB")
            animation.seek(8)
            middle = animation.convert("RGB")
            assert ImageChops.difference(first, middle).getbbox() is not None

    manifest = json.loads((tool_demo_path("brush").parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "UZYRO tool demos v2"
    assert manifest["generator"] == "UZYRO actual tool engine v1"
    assert set(manifest["tools"]) == expected_tools
    assert all(item["engine"] == "uzyro" for item in manifest["tools"].values())
