from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Callable

import cv2

from .action_history import apply_history_payload, history_command_to_payload
from .core import Document, adjust_brightness_contrast, adjust_saturation, apply_filter_stack, rotate_bound
from .security.errors import ResourceLimitError
from .security.limits import LIMITS
from .security.validation import bounded_int, bounded_string, finite_float, load_bounded_json, validate_json_tree


ACTION_FORMAT = "UZYRO action v3"
LEGACY_ACTION_FORMATS = {
    f"{'Photo' + 'Redactor'} action v2",
    f"{'Photo' + 'Redactor'} action v3",
}
ERROR_POLICIES = {"stop", "continue", "skip_file"}
CONDITION_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains", "in", "exists"}


@dataclass
class ActionStep:
    command: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str = ""
    condition: dict[str, Any] | list[dict[str, Any]] | None = None
    on_error: str = "stop"
    enabled: bool = True
    stop_after: bool = False


@dataclass
class StepError:
    index: int
    label: str
    message: str


@dataclass
class ActionReport:
    executed: int = 0
    skipped: int = 0
    stopped: bool = False
    stop_message: str = ""
    errors: list[StepError] = field(default_factory=list)


class SkipBatchFile(RuntimeError):
    pass


class ActionRecorder:
    def __init__(self) -> None:
        self.recording = False
        self.steps: list[ActionStep] = []

    def start(self, *, append: bool = False) -> None:
        if not append:
            self.steps.clear()
        self.recording = True

    def stop(self) -> None:
        self.recording = False

    def record(
        self,
        command: str,
        params: dict[str, Any] | None = None,
        label: str = "",
        **options: Any,
    ) -> None:
        if self.recording:
            self.steps.append(ActionStep(command, dict(params or {}), label, **options))

    def record_history_command(self, command: Any, document: Document) -> None:
        if not self.recording:
            return
        payload = history_command_to_payload(command, document)
        self.steps.append(ActionStep("history", payload, str(getattr(command, "label", "Операция"))))

    def add_stop(self, message: str = "Остановка действия", condition: dict[str, Any] | None = None) -> None:
        self.steps.append(ActionStep("stop", {"message": message}, message, condition=condition))

    def save(self, path: str | Path, name: str | None = None, on_error: str = "stop") -> None:
        if on_error not in ERROR_POLICIES:
            raise ValueError(f"Неизвестная политика ошибок: {on_error}")
        data = {
            "format": ACTION_FORMAT,
            "name": name or Path(path).stem,
            "on_error": on_error,
            "steps": [asdict(step) for step in self.steps],
        }
        validate_action(data)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_action(action: str | Path | dict[str, Any]) -> dict[str, Any]:
    data = load_bounded_json(action, maximum=LIMITS.max_action_bytes) if isinstance(action, (str, Path)) else dict(action)
    validate_action(data)
    return data


def validate_action(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("Файл действия должен содержать JSON-объект")
    action_format = str(data.get("format", ""))
    if action_format not in {"UZYRO action v2", ACTION_FORMAT, *LEGACY_ACTION_FORMATS}:
        raise ValueError("Неподдерживаемый формат действия")
    if data.get("on_error", "stop") not in ERROR_POLICIES:
        raise ValueError("Некорректная общая политика ошибок")
    steps = data.get("steps")
    if not isinstance(steps, list):
        raise ValueError("В действии отсутствует список шагов")
    if len(steps) > LIMITS.max_action_steps:
        raise ResourceLimitError("Действие содержит слишком много шагов")
    bounded_string(str(data.get("name", "Действие")), "Название действия", LIMITS.max_name_chars)
    for index, raw in enumerate(steps):
        if isinstance(raw, str):
            bounded_string(raw, f"Шаг {index + 1}", LIMITS.max_name_chars)
            continue
        if not isinstance(raw, dict) or not str(raw.get("command", "")):
            raise ValueError(f"Шаг {index + 1} не содержит команду")
        command = bounded_string(str(raw["command"]), f"Команда шага {index + 1}", 128)
        params = raw.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"Параметры шага {index + 1} должны быть объектом")
        validate_json_tree(params)
        _validate_builtin_params(command, params, index)
        if raw.get("on_error", "stop") not in ERROR_POLICIES:
            raise ValueError(f"Шаг {index + 1} содержит неизвестную политику ошибок")
        _validate_condition(raw.get("condition"), index)


def _validate_builtin_params(command: str, params: dict[str, Any], index: int) -> None:
    label = f"Параметры шага {index + 1}"
    if command in {"resize_image", "resize_canvas"}:
        bounded_int(params.get("width"), f"{label}: ширина", 1, LIMITS.max_dimension)
        bounded_int(params.get("height"), f"{label}: высота", 1, LIMITS.max_dimension)
    elif command == "rotate":
        finite_float(params.get("angle", 0), f"{label}: угол", -360_000.0, 360_000.0)
    elif command == "set_bit_depth" and int(params.get("bit_depth", 0)) not in {8, 16, 32}:
        raise ValueError(f"{label}: глубина цвета должна быть 8, 16 или 32")
    elif command == "set_color_model" and str(params.get("color_model", "")) not in {"RGBA", "CMYK", "Lab", "GRAY"}:
        raise ValueError(f"{label}: неизвестная цветовая модель")
    elif command == "brightness_contrast":
        bounded_int(params.get("brightness", 0), f"{label}: яркость", -255, 255)
        finite_float(params.get("contrast", 1.0), f"{label}: контраст", 0.0, 10.0)
    elif command == "saturation":
        finite_float(params.get("amount", 1.0), f"{label}: насыщенность", 0.0, 10.0)


def _validate_condition(condition: Any, index: int) -> None:
    if condition is None:
        return
    conditions = condition if isinstance(condition, list) else [condition]
    if not conditions or not all(isinstance(item, dict) for item in conditions):
        raise ValueError(f"Условие шага {index + 1} должно быть объектом или списком объектов")
    for item in conditions:
        if not str(item.get("field", "")):
            raise ValueError(f"Условие шага {index + 1} не содержит поле")
        if item.get("operator", "eq") not in CONDITION_OPERATORS:
            raise ValueError(f"Условие шага {index + 1} содержит неизвестный оператор")


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
            "history": self._history,
        }

    def register(self, name: str, callback: Callable[[Document, dict[str, Any]], None]) -> None:
        if not name or name in self.commands:
            raise ValueError(f"Команда действия уже существует: {name}")
        self.commands[name] = callback

    def run(self, document: Document, action: str | Path | dict[str, Any]) -> int:
        return self.run_with_report(document, action).executed

    def batch(
        self,
        action: str | Path | dict[str, Any],
        sources: list[str | Path],
        destination: str | Path,
        suffix: str = ".png",
    ) -> list[Path]:
        from .batch_queue import BatchQueue

        queue = BatchQueue(self)
        job = queue.enqueue(action, sources, destination, suffix=suffix)
        queue.run_all()
        failures = [item for item in job.items if item.state == "failed"]
        if failures:
            raise RuntimeError(failures[0].message)
        return [Path(item.target) for item in job.items if item.state == "completed"]

    def run_with_report(
        self,
        document: Document,
        action: str | Path | dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> ActionReport:
        data = load_action(action)
        report = ActionReport()
        execution_context = dict(context or {})
        execution_context.setdefault("layer_ids", {})
        steps = data.get("steps", [])
        for index, raw in enumerate(steps):
            if cancelled and cancelled():
                report.stopped = True
                report.stop_message = "Выполнение отменено"
                break
            if isinstance(raw, str) or not bool(raw.get("enabled", True)):
                report.skipped += 1
                continue
            label = str(raw.get("label") or raw.get("command") or f"Шаг {index + 1}")
            if not self._matches(document, raw.get("condition"), execution_context):
                report.skipped += 1
                if progress:
                    progress(index + 1, len(steps), f"Пропущено: {label}")
                continue
            command = str(raw.get("command", ""))
            if command == "stop":
                report.stopped = True
                report.stop_message = str((raw.get("params") or {}).get("message", label))
                break
            callback = self.commands.get(command)
            try:
                if callback is None:
                    raise ValueError(f"Неизвестная команда действия: {command}")
                params = dict(raw.get("params") or {})
                if command == "history":
                    apply_history_payload(document, params, execution_context)
                else:
                    callback(document, params)
                report.executed += 1
            except Exception as exc:
                report.errors.append(StepError(index, label, str(exc)))
                policy = str(raw.get("on_error") or data.get("on_error", "stop"))
                if policy == "skip_file":
                    raise SkipBatchFile(str(exc)) from exc
                if policy == "stop":
                    report.stopped = True
                    report.stop_message = f"Ошибка на шаге «{label}»: {exc}"
                    break
            if progress:
                progress(index + 1, len(steps), label)
            if bool(raw.get("stop_after", False)):
                report.stopped = True
                report.stop_message = f"Остановка после шага «{label}»"
                break
        return report

    @staticmethod
    def _matches(document: Document, condition: Any, context: dict[str, Any]) -> bool:
        if condition is None:
            return True
        conditions = condition if isinstance(condition, list) else [condition]
        return all(_evaluate_condition(document, item, context) for item in conditions)

    @staticmethod
    def _history(document: Document, params: dict[str, Any]) -> None:
        apply_history_payload(document, params, {})

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
        document.layer.pixels = adjust_brightness_contrast(
            document.layer.pixels,
            int(params.get("brightness", 0)),
            float(params.get("contrast", 1.0)),
        )
        document.layer.touch_pixels()
        document.dirty = True

    @staticmethod
    def _saturation(document: Document, params: dict[str, Any]) -> None:
        document.layer.pixels = adjust_saturation(document.layer.pixels, float(params.get("amount", 1.0)))
        document.layer.touch_pixels()
        document.dirty = True


def _condition_value(document: Document, field_name: str, context: dict[str, Any]) -> Any:
    fields = {
        "document.width": document.width,
        "document.height": document.height,
        "document.layer_count": len(document.layers),
        "document.has_selection": document.selection_mask is not None,
        "document.bit_depth": document.bit_depth,
        "document.color_model": document.color_model,
        "layer.kind": document.layer.kind,
        "layer.name": document.layer.name,
        "source.extension": str(context.get("source_extension", "")).lower(),
        "source.name": str(context.get("source_name", "")),
    }
    if field_name not in fields:
        raise ValueError(f"Неизвестное поле условия: {field_name}")
    return fields[field_name]


def _evaluate_condition(document: Document, condition: dict[str, Any], context: dict[str, Any]) -> bool:
    actual = _condition_value(document, str(condition.get("field", "")), context)
    expected = condition.get("value")
    operator = str(condition.get("operator", "eq"))
    if operator == "eq":
        return actual == expected
    if operator == "ne":
        return actual != expected
    if operator == "gt":
        return actual > expected
    if operator == "gte":
        return actual >= expected
    if operator == "lt":
        return actual < expected
    if operator == "lte":
        return actual <= expected
    if operator == "contains":
        return expected in actual
    if operator == "in":
        return actual in expected
    if operator == "exists":
        return bool(actual) is bool(expected)
    raise ValueError(f"Неизвестный оператор условия: {operator}")


def action_template(name: str = "Новое действие") -> dict[str, Any]:
    return {"format": ACTION_FORMAT, "name": name, "on_error": "stop", "steps": []}


__all__ = [
    "ACTION_FORMAT",
    "ActionRecorder",
    "ActionReport",
    "ActionRunner",
    "ActionStep",
    "SkipBatchFile",
    "action_template",
    "load_action",
    "validate_action",
]
