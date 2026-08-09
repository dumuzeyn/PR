from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable

import cv2

from .core import Document, adjust_brightness_contrast, adjust_saturation, apply_filter_stack, rotate_bound


@dataclass
class ActionStep:
    command: str
    params: dict[str, Any]
    label: str = ""


class ActionRecorder:
    def __init__(self) -> None:
        self.recording = False
        self.steps: list[ActionStep] = []

    def start(self) -> None:
        self.steps.clear()
        self.recording = True

    def stop(self) -> None:
        self.recording = False

    def record(self, command: str, params: dict[str, Any] | None = None, label: str = "") -> None:
        if self.recording:
            self.steps.append(ActionStep(command, dict(params or {}), label))

    def save(self, path: str | Path, name: str | None = None) -> None:
        data = {"format": "PhotoRedactor action v2", "name": name or Path(path).stem, "steps": [asdict(step) for step in self.steps]}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class ActionRunner:
    def __init__(self) -> None:
        self.commands: dict[str, Callable[[Document, dict[str, Any]], None]] = {
            "resize_image": self._resize_image,
            "resize_canvas": self._resize_canvas,
            "flatten": lambda document, params: document.flatten(),
            "rotate": self._rotate,
            "flip": self._flip,
            "filter_stack": self._filter_stack,
            "brightness_contrast": self._brightness_contrast,
            "saturation": self._saturation,
            "set_bit_depth": lambda document, params: document.set_bit_depth(int(params["bit_depth"])),
            "set_color_model": lambda document, params: document.set_color_model(str(params["color_model"])),
        }

    def register(self, name: str, callback: Callable[[Document, dict[str, Any]], None]) -> None:
        if not name or name in self.commands:
            raise ValueError(f"Action command already exists: {name}")
        self.commands[name] = callback

    def run(self, document: Document, action: str | Path | dict[str, Any]) -> int:
        data = json.loads(Path(action).read_text(encoding="utf-8")) if isinstance(action, (str, Path)) else action
        count = 0
        for raw in data.get("steps", []):
            if isinstance(raw, str):
                continue
            command = str(raw.get("command", ""))
            callback = self.commands.get(command)
            if callback is None:
                raise ValueError(f"Unknown action command: {command}")
            callback(document, dict(raw.get("params") or {}))
            count += 1
        return count

    def batch(self, action: str | Path | dict[str, Any], sources: list[str | Path], destination: str | Path, suffix: str = ".png") -> list[Path]:
        output = Path(destination)
        output.mkdir(parents=True, exist_ok=True)
        results: list[Path] = []
        for source in sources:
            document = Document.from_image(source)
            self.run(document, action)
            target = output / f"{Path(source).stem}{suffix}"
            document.export_flat(target)
            results.append(target)
        return results

    @staticmethod
    def _resize_image(document: Document, params: dict[str, Any]) -> None:
        document.resize_image(int(params["width"]), int(params["height"]))

    @staticmethod
    def _resize_canvas(document: Document, params: dict[str, Any]) -> None:
        document.resize_canvas(int(params["width"]), int(params["height"]), str(params.get("anchor", "center")))

    @staticmethod
    def _rotate(document: Document, params: dict[str, Any]) -> None:
        angle = float(params.get("angle", 0.0))
        for layer in document.layers:
            layer.pixels = rotate_bound(layer.pixels, angle, cv2.INTER_CUBIC)
            layer.x = 0
            layer.y = 0
            layer.touch_pixels()
        if angle % 180:
            document.width, document.height = document.height, document.width
        document.dirty = True

    @staticmethod
    def _flip(document: Document, params: dict[str, Any]) -> None:
        axis = 1 if str(params.get("axis", "horizontal")) == "horizontal" else 0
        for layer in document.layers:
            layer.pixels = cv2.flip(layer.pixels, axis)
            layer.touch_pixels()
        document.dirty = True

    @staticmethod
    def _filter_stack(document: Document, params: dict[str, Any]) -> None:
        document.layer.pixels = apply_filter_stack(document.layer.pixels, list(params.get("filters") or []))
        document.layer.touch_pixels()
        document.dirty = True

    @staticmethod
    def _brightness_contrast(document: Document, params: dict[str, Any]) -> None:
        document.layer.pixels = adjust_brightness_contrast(document.layer.pixels, int(params.get("brightness", 0)), float(params.get("contrast", 1.0)))
        document.layer.touch_pixels()
        document.dirty = True

    @staticmethod
    def _saturation(document: Document, params: dict[str, Any]) -> None:
        document.layer.pixels = adjust_saturation(document.layer.pixels, float(params.get("amount", 1.0)))
        document.layer.touch_pixels()
        document.dirty = True


def action_template(name: str = "Новое действие") -> dict[str, Any]:
    return {"format": "PhotoRedactor action v2", "name": name, "steps": []}
