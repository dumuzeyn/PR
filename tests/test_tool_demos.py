from PIL import Image

from uzyro.app import TOOL_DEFINITIONS, tool_demo_path


def test_every_tool_has_an_animated_demo() -> None:
    for _label, tool_id, _description in TOOL_DEFINITIONS:
        path = tool_demo_path(tool_id)
        assert path.is_file(), f"missing demo for {tool_id}"
        with Image.open(path) as animation:
            assert animation.format == "GIF"
            assert animation.size == (288, 162)
            assert animation.n_frames >= 12
