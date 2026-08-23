from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from PIL import ImageTk

from .theme import TOKENS


class PropertySection(ttk.Frame):
    def __init__(self, master: tk.Misc, text: str) -> None:
        super().__init__(master, style="Panel.TFrame")
        ttk.Label(self, text=text, style="SectionTitle.TLabel").pack(side=tk.LEFT)
        ttk.Separator(self).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))


class AccentScale(tk.Canvas):
    def __init__(
        self,
        master: tk.Misc,
        *,
        from_: float = 0.0,
        to: float = 1.0,
        variable: tk.Variable | None = None,
        command: Callable[[str], object] | None = None,
        length: int = 100,
        orient: str = tk.HORIZONTAL,
        **kwargs,
    ) -> None:
        self.minimum = float(from_)
        self.maximum = float(to)
        self.variable = variable or tk.DoubleVar(master, value=self.minimum)
        self.command = command
        self.hovered = False
        self.dragging = False
        height = int(kwargs.pop("height", 18))
        background = kwargs.pop("background", TOKENS.PANEL_BG)
        super().__init__(
            master,
            width=length,
            height=height,
            background=background,
            highlightthickness=1,
            highlightbackground=background,
            highlightcolor=TOKENS.FOCUS,
            borderwidth=0,
            takefocus=True,
            cursor="hand2",
            **kwargs,
        )
        self._trace = self.variable.trace_add("write", self._variable_changed)
        self.bind("<Configure>", self._redraw)
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<ButtonPress-1>", self._pointer)
        self.bind("<B1-Motion>", self._pointer)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Left>", lambda _event: self._step(-1))
        self.bind("<Right>", lambda _event: self._step(1))
        self.bind("<FocusIn>", self._redraw)
        self.bind("<FocusOut>", self._redraw)

    def configure(self, cnf=None, **kwargs):
        options = dict(cnf) if isinstance(cnf, dict) else {}
        options.update(kwargs)
        variable = options.pop("variable", None)
        command = options.pop("command", None)
        if variable is not None and variable is not self.variable:
            try:
                self.variable.trace_remove("write", self._trace)
            except (tk.TclError, ValueError):
                pass
            self.variable = variable
            self._trace = self.variable.trace_add("write", self._variable_changed)
            self._redraw()
        if command is not None:
            self.command = command
        if isinstance(cnf, str):
            return super().configure(cnf)
        return super().configure(**options)

    config = configure

    def _fraction(self) -> float:
        span = self.maximum - self.minimum
        value = float(self.variable.get())
        return 0.0 if span == 0 else max(0.0, min(1.0, (value - self.minimum) / span))

    def _redraw(self, _event=None) -> None:
        if not self.winfo_exists():
            return
        self.delete("all")
        width = max(16, self.winfo_width())
        height = max(12, self.winfo_height())
        x1, x2, y = 7, width - 7, height // 2
        position = round(x1 + (x2 - x1) * self._fraction())
        self.create_line(x1, y, x2, y, fill=TOKENS.BORDER_STRONG, width=3, capstyle=tk.ROUND)
        self.create_line(x1, y, position, y, fill=TOKENS.ACCENT, width=3, capstyle=tk.ROUND)
        thumb = TOKENS.ACCENT_HOVER if self.hovered or self.dragging else TOKENS.TEXT_PRIMARY
        self.create_oval(position - 4, y - 4, position + 4, y + 4, fill=thumb, outline=TOKENS.CONTROL_PRESSED, width=1)
        if self.focus_get() is self:
            self.create_rectangle(1, 1, width - 2, height - 2, outline=TOKENS.FOCUS, width=1)

    def _variable_changed(self, *_args) -> None:
        try:
            self.after_idle(self._redraw)
        except tk.TclError:
            pass

    def _enter(self, _event) -> None:
        self.hovered = True
        self._redraw()

    def _leave(self, _event) -> None:
        self.hovered = False
        self._redraw()

    def _pointer(self, event) -> str:
        self.focus_set()
        self.dragging = True
        width = max(16, self.winfo_width())
        fraction = max(0.0, min(1.0, (event.x - 7) / max(1, width - 14)))
        self.set(self.minimum + fraction * (self.maximum - self.minimum), notify=True)
        return "break"

    def _release(self, _event) -> str:
        self.dragging = False
        self._redraw()
        return "break"

    def _step(self, direction: int) -> str:
        step = (self.maximum - self.minimum) / 100.0
        self.set(float(self.variable.get()) + direction * step, notify=True)
        return "break"

    def get(self) -> float:
        return float(self.variable.get())

    def set(self, value: float, *, notify: bool = False) -> None:
        bounded = max(min(float(value), max(self.minimum, self.maximum)), min(self.minimum, self.maximum))
        self.variable.set(bounded)
        if notify and self.command is not None:
            self.command(str(bounded))


class SlimScrollbar(tk.Canvas):
    def __init__(self, master: tk.Misc, *, orient: str = tk.VERTICAL, command=None, **kwargs) -> None:
        self.orient = orient
        self.command = command
        self.first = 0.0
        self.last = 1.0
        self.hovered = False
        self.drag_offset: float | None = None
        dimensions = {"width": 10, "height": kwargs.pop("height", 100)} if orient == tk.VERTICAL else {"height": 10, "width": kwargs.pop("width", 100)}
        background = kwargs.pop("background", TOKENS.PANEL_BG)
        super().__init__(master, background=background, highlightthickness=0, borderwidth=0, cursor="hand2", **dimensions, **kwargs)
        self.bind("<Configure>", self._redraw)
        self.bind("<Enter>", lambda _event: self._set_hover(True))
        self.bind("<Leave>", lambda _event: self._set_hover(False))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", lambda _event: setattr(self, "drag_offset", None))

    def set(self, first, last) -> None:
        self.first, self.last = float(first), float(last)
        self._redraw()

    def _axis_length(self) -> int:
        return self.winfo_height() if self.orient == tk.VERTICAL else self.winfo_width()

    def _coordinate(self, event) -> int:
        return int(event.y if self.orient == tk.VERTICAL else event.x)

    def _thumb(self) -> tuple[float, float]:
        length = max(1, self._axis_length())
        start = self.first * length
        end = max(start + 18, self.last * length)
        if end > length:
            start -= end - length
            end = length
        return start, end

    def _redraw(self, _event=None) -> None:
        if not self.winfo_exists():
            return
        self.delete("all")
        start, end = self._thumb()
        color = TOKENS.TEXT_MUTED if self.hovered else TOKENS.BORDER_STRONG
        if self.orient == tk.VERTICAL:
            self.create_rectangle(3, start, 7, end, fill=color, outline="")
        else:
            self.create_rectangle(start, 3, end, 7, fill=color, outline="")

    def _set_hover(self, value: bool) -> None:
        self.hovered = value
        self._redraw()

    def _press(self, event) -> str:
        coordinate = self._coordinate(event)
        start, end = self._thumb()
        if start <= coordinate <= end:
            self.drag_offset = coordinate - start
        else:
            self.drag_offset = (end - start) / 2
            self._move_thumb(coordinate)
        return "break"

    def _drag(self, event) -> str:
        if self.drag_offset is not None:
            self._move_thumb(self._coordinate(event))
        return "break"

    def _move_thumb(self, coordinate: float) -> None:
        if self.command is None:
            return
        length = max(1, self._axis_length())
        start, end = self._thumb()
        thumb_length = end - start
        fraction = max(0.0, min(1.0, (coordinate - float(self.drag_offset or 0.0)) / max(1.0, length - thumb_length)))
        self.command("moveto", fraction)


class LayerList(tk.Canvas):
    ROW_HEIGHT = 42

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        kwargs.pop("height", None)
        kwargs.pop("exportselection", None)
        kwargs.pop("selectmode", None)
        kwargs.pop("activestyle", None)
        kwargs.pop("font", None)
        kwargs.setdefault("width", 240)
        super().__init__(master, background=TOKENS.PANEL_BG, highlightthickness=0, borderwidth=0, takefocus=True, **kwargs)
        self._items: list[str] = []
        self._layers: list[object] = []
        self._selection: set[int] = set()
        self._active = 0
        self._hover = -1
        self._anchor = 0
        self._thumbnail_factory = None
        self._thumbnail_cache: dict[tuple[object, ...], ImageTk.PhotoImage] = {}
        self._row_images: list[ImageTk.PhotoImage] = []
        self.bind("<Configure>", self._redraw)
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda _event: self._set_hover(-1))
        self.bind("<ButtonPress-1>", self._select_pointer, add="+")
        self.bind("<Up>", lambda _event: self._keyboard(-1))
        self.bind("<Down>", lambda _event: self._keyboard(1))

    def set_layers(self, layers: list[object], thumbnail_factory) -> None:
        self._layers = layers
        self._thumbnail_factory = thumbnail_factory
        live_ids = {getattr(layer, "id", None) for layer in layers}
        self._thumbnail_cache = {key: value for key, value in self._thumbnail_cache.items() if key[0] in live_ids}
        self._redraw()

    def _normalize_index(self, index, *, insertion: bool = False) -> int:
        if index in (tk.END, "end"):
            return len(self._items) if insertion else max(0, len(self._items) - 1)
        if index in (tk.ACTIVE, "active"):
            return self._active
        return max(0, int(index))

    def insert(self, index, *elements) -> None:
        position = self._normalize_index(index, insertion=True)
        for element in elements:
            self._items.insert(position, str(element))
            position += 1
        self._redraw()

    def delete(self, first, last=None) -> None:
        start = self._normalize_index(first)
        end = self._normalize_index(last) if last is not None else start
        if self._items:
            del self._items[start : end + 1]
        self._selection = {row for row in self._selection if row < len(self._items)}
        self._redraw()

    def get(self, first, last=None):
        start = self._normalize_index(first)
        if last is None:
            return self._items[start]
        end = self._normalize_index(last)
        return tuple(self._items[start : end + 1])

    def selection_clear(self, first, last=None) -> None:
        start = self._normalize_index(first)
        end = self._normalize_index(last) if last is not None else start
        self._selection.difference_update(range(start, end + 1))
        self._redraw()

    def selection_set(self, first, last=None) -> None:
        start = self._normalize_index(first)
        end = self._normalize_index(last) if last is not None else start
        self._selection.update(range(start, min(end, len(self._items) - 1) + 1))
        self._redraw()

    def curselection(self) -> tuple[int, ...]:
        return tuple(sorted(self._selection))

    def activate(self, index) -> None:
        self._active = self._normalize_index(index)
        self._redraw()

    def index(self, index) -> int:
        return self._normalize_index(index)

    def nearest(self, y: int) -> int:
        return max(0, min(len(self._items) - 1, int(self.canvasy(y) // self.ROW_HEIGHT))) if self._items else 0

    def bbox(self, index):
        row = self._normalize_index(index)
        if not (0 <= row < len(self._items)):
            return None
        return (0, round(row * self.ROW_HEIGHT - self.canvasy(0)), self.winfo_width(), self.ROW_HEIGHT)

    def see(self, index) -> None:
        row = self._normalize_index(index)
        total = max(1, len(self._items) * self.ROW_HEIGHT)
        self.yview_moveto(max(0.0, min(1.0, row * self.ROW_HEIGHT / total)))

    def _motion(self, event) -> None:
        self._set_hover(self.nearest(event.y))

    def _set_hover(self, row: int) -> None:
        if row != self._hover:
            self._hover = row
            self._redraw()

    def _select_pointer(self, event) -> None:
        if event.x <= 34 or not self._items:
            return
        row = self.nearest(event.y)
        state = int(getattr(event, "state", 0))
        if state & 0x0001:
            start, end = sorted((self._anchor, row))
            self._selection = set(range(start, end + 1))
        elif state & 0x0004:
            self._selection.symmetric_difference_update({row})
            self._anchor = row
        else:
            self._selection = {row}
            self._anchor = row
        self._active = row
        self._redraw()
        self.event_generate("<<ListboxSelect>>", when="tail")

    def _keyboard(self, direction: int) -> str:
        if not self._items:
            return "break"
        self._active = max(0, min(len(self._items) - 1, self._active + direction))
        self._selection = {self._active}
        self.see(self._active)
        self._redraw()
        self.event_generate("<<ListboxSelect>>", when="tail")
        return "break"

    def _thumbnail(self, layer) -> ImageTk.PhotoImage | None:
        if self._thumbnail_factory is None or layer is None:
            return None
        key = (
            getattr(layer, "id", id(layer)),
            getattr(layer, "pixels_revision", 0),
            getattr(layer, "mask_revision", 0),
            repr(getattr(layer, "filters", None)),
            repr(getattr(layer, "effects", None)),
            getattr(layer, "opacity", 1.0),
            getattr(layer, "mask_enabled", True),
            getattr(layer, "mask_density", 1.0),
            28,
        )
        image = self._thumbnail_cache.get(key)
        if image is None:
            image = ImageTk.PhotoImage(self._thumbnail_factory(layer, 28), master=self)
            layer_id = key[0]
            self._thumbnail_cache = {
                cached_key: cached_image
                for cached_key, cached_image in self._thumbnail_cache.items()
                if cached_key[0] != layer_id
            }
            self._thumbnail_cache[key] = image
        return image

    def _draw_eye(self, y: int, visible: bool) -> None:
        color = TOKENS.TEXT_SECONDARY if visible else TOKENS.TEXT_DISABLED
        self.create_oval(10, y - 5, 24, y + 5, outline=color, width=1)
        self.create_oval(15, y - 2, 19, y + 2, fill=color, outline="")
        if not visible:
            self.create_line(9, y + 7, 25, y - 7, fill=TOKENS.DANGER, width=1)

    def _redraw(self, _event=None) -> None:
        if not self.winfo_exists():
            return
        super().delete("all")
        self._row_images = []
        width = max(120, self.winfo_width())
        for row, text in enumerate(self._items):
            y1 = row * self.ROW_HEIGHT
            y2 = y1 + self.ROW_HEIGHT
            selected = row in self._selection
            background = TOKENS.CONTROL_SELECTED if selected else TOKENS.CONTROL_HOVER if row == self._hover else TOKENS.PANEL_BG
            self.create_rectangle(0, y1, width, y2, fill=background, outline="")
            if selected:
                self.create_rectangle(0, y1, 3, y2, fill=TOKENS.ACCENT, outline="")
            layer = self._layers[row] if row < len(self._layers) else None
            self._draw_eye(y1 + self.ROW_HEIGHT // 2, bool(getattr(layer, "visible", True)))
            thumbnail = self._thumbnail(layer)
            if thumbnail is not None:
                self._row_images.append(thumbnail)
                self.create_rectangle(37, y1 + 6, 67, y1 + 36, outline=TOKENS.BORDER_STRONG, width=1)
                self.create_image(38, y1 + 7, image=thumbnail, anchor=tk.NW)
            name = str(getattr(layer, "name", text)).strip()
            self.create_text(74, y1 + self.ROW_HEIGHT // 2, text=name, anchor=tk.W, fill=TOKENS.TEXT_PRIMARY, font=("Segoe UI", 9), width=max(60, width - 112))
            states = ""
            if bool(getattr(layer, "locked", False)):
                states += "L"
            if getattr(layer, "mask", None) is not None:
                states += " M"
            if getattr(layer, "effects", None) or getattr(layer, "filters", None):
                states += " FX"
            if states:
                self.create_text(width - 8, y1 + self.ROW_HEIGHT // 2, text=states.strip(), anchor=tk.E, fill=TOKENS.TEXT_MUTED, font=("Segoe UI Semibold", 7))
            self.create_line(8, y2 - 1, width - 8, y2 - 1, fill=TOKENS.BORDER_SUBTLE)
        self.configure(scrollregion=(0, 0, width, max(1, len(self._items) * self.ROW_HEIGHT)))
