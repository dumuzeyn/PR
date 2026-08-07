from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from .scrollable_frame import ScrollableFrame

ToolDefinition = tuple[str, str, str]


def normalize_tool_order(saved_order: list[str] | None, definitions: list[ToolDefinition]) -> list[str]:
    valid = [value for _label, value, _description in definitions]
    seen: set[str] = set()
    order: list[str] = []
    for value in saved_order or []:
        if value in valid and value not in seen:
            order.append(value)
            seen.add(value)
    for value in valid:
        if value not in seen:
            order.append(value)
    return order


def normalize_visible_tools(saved_visible: list[str] | None, order: list[str]) -> list[str]:
    visible = [value for value in saved_visible or [] if value in order]
    if not visible:
        visible = list(order)
    return visible


class ToolPalette(ttk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        *,
        definitions: list[ToolDefinition],
        tool_var: tk.StringVar,
        order: list[str],
        visible: list[str],
        select_tool,
        configure_tools,
        tooltip_factory=None,
    ) -> None:
        super().__init__(parent)
        self.definitions = definitions
        self.tool_var = tool_var
        self.order = list(order)
        self.visible = list(visible)
        self.select_tool = select_tool
        self.tooltip_factory = tooltip_factory
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(header, text="Инструменты").pack(side=tk.LEFT)
        ttk.Button(header, text="Настроить...", command=configure_tools).pack(side=tk.RIGHT)
        self.scroller = ScrollableFrame(self, height=280)
        self.scroller.pack(fill=tk.BOTH, expand=True, padx=(8, 4), pady=(0, 8))
        self.buttons: dict[str, ttk.Radiobutton] = {}
        self.render()

    def set_configuration(self, order: list[str], visible: list[str]) -> None:
        self.order = list(order)
        self.visible = list(visible)
        self.render()

    def render(self) -> None:
        for child in self.scroller.content.winfo_children():
            child.destroy()
        self.buttons.clear()
        by_id = {value: (label, value, description) for label, value, description in self.definitions}
        for value in self.order:
            if value not in self.visible or value not in by_id:
                continue
            label, _value, description = by_id[value]
            button = ttk.Radiobutton(
                self.scroller.content,
                text=label,
                value=value,
                variable=self.tool_var,
                command=lambda v=value: self.select_tool(v),
            )
            button.pack(fill=tk.X, padx=2, pady=2)
            if self.tooltip_factory is not None:
                self.tooltip_factory(button, description)
            self.buttons[value] = button


class ToolPaletteDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Widget,
        *,
        definitions: list[ToolDefinition],
        order: list[str],
        visible: list[str],
    ) -> None:
        super().__init__(parent)
        self.title("Настроить панель инструментов")
        self.transient(parent)
        self.grab_set()
        self.resizable(True, True)
        self.result: tuple[list[str], list[str]] | None = None
        self.definitions = definitions
        self.order = list(order)
        self.visible = set(visible)
        self.label_by_id = {value: label for label, value, _description in definitions}
        self._drag_tool: str | None = None
        self._drag_start_y = 0
        self._dragged = False
        self.geometry("420x520")

        ttk.Label(self, text="Инструменты панели").pack(anchor=tk.W, padx=12, pady=(12, 6))
        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=12)
        list_area = ttk.Frame(body)
        list_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(list_area, exportselection=False, activestyle="dotbox", cursor="hand2")
        scrollbar = ttk.Scrollbar(list_area, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        side = ttk.Frame(body)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        ttk.Button(side, text="Вверх", command=self.move_up).pack(fill=tk.X, pady=(0, 4))
        ttk.Button(side, text="Вниз", command=self.move_down).pack(fill=tk.X, pady=(0, 12))
        self.visible_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text="Показывать", variable=self.visible_var, command=self.toggle_selected).pack(fill=tk.X)
        ttk.Button(side, text="Показать все", command=self.show_all).pack(fill=tk.X, pady=(12, 4))
        ttk.Button(side, text="Скрыть все", command=self.hide_all).pack(fill=tk.X, pady=(0, 4))
        ttk.Button(side, text="По умолчанию", command=self.reset_default).pack(fill=tk.X)

        buttons = ttk.Frame(self)
        buttons.pack(fill=tk.X, padx=12, pady=12)
        ttk.Button(buttons, text="Применить", command=self.apply).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Отмена", command=self.cancel).pack(side=tk.RIGHT, padx=(0, 8))
        self.listbox.bind("<<ListboxSelect>>", lambda _event: self.update_selected_state())
        self.listbox.bind("<ButtonPress-1>", self.begin_drag)
        self.listbox.bind("<B1-Motion>", self.drag_tool)
        self.listbox.bind("<ButtonRelease-1>", self.end_drag)
        self.listbox.bind("<space>", self.toggle_selected_from_keyboard)
        self.listbox.bind("<Return>", self.toggle_selected_from_keyboard)
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.refresh_list()
        if self.order:
            self.listbox.selection_set(0)
            self.update_selected_state()

    def refresh_list(self, selected_value: str | None = None) -> None:
        selected = selected_value if selected_value is not None else self.current_tool()
        self.listbox.delete(0, tk.END)
        for value in self.order:
            mark = "☑" if value in self.visible else "☐"
            self.listbox.insert(tk.END, f"{mark} {self.label_by_id.get(value, value)}")
        if selected in self.order:
            index = self.order.index(selected)
            self.listbox.selection_set(index)
            self.listbox.see(index)
        self.update_selected_state()

    def current_tool(self) -> str | None:
        selection = self.listbox.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if 0 <= index < len(self.order):
            return self.order[index]
        return None

    def update_selected_state(self) -> None:
        value = self.current_tool()
        self.visible_var.set(value in self.visible if value else False)

    def toggle_selected(self) -> None:
        value = self.current_tool()
        if value is None:
            return
        if self.visible_var.get():
            self.visible.add(value)
        else:
            self.visible.discard(value)
        self.refresh_list(value)

    def toggle_tool(self, value: str) -> None:
        if value not in self.order:
            return
        if value in self.visible:
            self.visible.discard(value)
        else:
            self.visible.add(value)
        self.refresh_list(value)

    def toggle_selected_from_keyboard(self, _event=None) -> str:
        value = self.current_tool()
        if value is not None:
            self.toggle_tool(value)
        return "break"

    def list_index_at(self, y: int) -> int | None:
        if not self.order:
            return None
        index = int(self.listbox.nearest(y))
        bounds = self.listbox.bbox(index)
        if bounds is None:
            return None
        _x, row_y, _width, row_height = bounds
        if y < row_y or y >= row_y + row_height:
            return None
        return index

    def begin_drag(self, event) -> str:
        index = self.list_index_at(event.y)
        if index is None:
            return "break"
        self._drag_tool = self.order[index]
        self._drag_start_y = int(event.y)
        self._dragged = False
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self.listbox.activate(index)
        self.update_selected_state()
        return "break"

    def drag_tool(self, event) -> str:
        if self._drag_tool is None:
            return "break"
        if abs(int(event.y) - self._drag_start_y) >= 4:
            self._dragged = True
        if event.y < 12:
            self.listbox.yview_scroll(-1, "units")
        elif event.y > self.listbox.winfo_height() - 12:
            self.listbox.yview_scroll(1, "units")
        target = int(self.listbox.nearest(event.y))
        target = max(0, min(len(self.order) - 1, target))
        self.move_tool_to(self._drag_tool, target)
        return "break"

    def end_drag(self, event) -> str:
        value = self._drag_tool
        should_toggle = value is not None and not self._dragged and int(event.x) <= 34
        self._drag_tool = None
        self._dragged = False
        if should_toggle and value is not None:
            self.toggle_tool(value)
        return "break"

    def move_tool_to(self, value: str, new_index: int) -> None:
        if value not in self.order:
            return
        old_index = self.order.index(value)
        new_index = max(0, min(len(self.order) - 1, int(new_index)))
        if old_index == new_index:
            return
        self.order.pop(old_index)
        self.order.insert(new_index, value)
        self.refresh_list(value)

    def move_up(self) -> None:
        self.move_selected(-1)

    def move_down(self) -> None:
        self.move_selected(1)

    def move_selected(self, delta: int) -> None:
        value = self.current_tool()
        if value is None:
            return
        index = self.order.index(value)
        new_index = index + delta
        if new_index < 0 or new_index >= len(self.order):
            return
        self.move_tool_to(value, new_index)

    def show_all(self) -> None:
        self.visible = set(self.order)
        self.refresh_list()

    def hide_all(self) -> None:
        value = self.current_tool()
        self.visible = {value} if value is not None else set()
        self.refresh_list()

    def reset_default(self) -> None:
        self.order = [value for _label, value, _description in self.definitions]
        self.visible = set(self.order)
        self.refresh_list()

    def apply(self) -> None:
        visible = [value for value in self.order if value in self.visible]
        if not visible:
            messagebox.showwarning("Инструменты", "Нельзя скрыть все инструменты.")
            return
        self.result = (list(self.order), visible)
        self.destroy()

    def cancel(self) -> None:
        self.result = None
        self.destroy()
