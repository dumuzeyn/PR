from __future__ import annotations

from .tool_palette_shared import *


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
        self._drag_insert_index: int | None = None
        self._drag_ghost: tk.Toplevel | None = None
        self._indicator_after_id: str | None = None
        self._indicator_phase = 0
        self._highlight_after_ids: list[str] = []
        self._drag_from_handle = False
        self.geometry("520x560")
        self.minsize(480, 430)

        ttk.Label(self, text="Инструменты панели").pack(anchor=tk.W, padx=12, pady=(12, 6))
        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=12)
        list_area = ttk.Frame(body)
        list_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(list_area, exportselection=False, activestyle="dotbox", cursor="hand2")
        scrollbar = ttk.Scrollbar(list_area, orient=tk.VERTICAL, command=self.scroll_list)
        self.scrollbar = scrollbar
        self.listbox.configure(yscrollcommand=self.on_listbox_scroll)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        side = ttk.Frame(body)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        ttk.Button(side, text="Вверх", command=self.move_up).pack(fill=tk.X, pady=(0, 4))
        ttk.Button(side, text="Вниз", command=self.move_down).pack(fill=tk.X, pady=(0, 12))
        self.visible_var = tk.BooleanVar(value=True)
        self.visibility_status = ttk.Label(side, text="", anchor=tk.CENTER)
        self.visibility_status.pack(fill=tk.X)
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
        self.listbox.bind("<Configure>", lambda _event: self.position_drag_handle())
        self.listbox.bind("<space>", self.toggle_selected_from_keyboard)
        self.listbox.bind("<Return>", self.toggle_selected_from_keyboard)
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self._drop_indicator = tk.Frame(self.listbox, height=3, background=TOKENS.ACCENT)
        self._drag_handle = tk.Label(
            self.listbox,
            text="≡",
            cursor="fleur",
            relief=tk.RAISED,
            borderwidth=1,
            padx=4,
            background=TOKENS.SURFACE_ELEVATED,
            foreground=TOKENS.ACCENT_HOVER,
            font=("Segoe UI", 11, "bold"),
            takefocus=False,
        )
        self._drag_handle.bind("<ButtonPress-1>", self.begin_handle_drag)
        self._drag_handle.bind("<B1-Motion>", self.drag_from_handle)
        self._drag_handle.bind("<ButtonRelease-1>", self.end_handle_drag)
        self.refresh_list()
        if self.order:
            self.listbox.selection_set(0)
            self.update_selected_state()

    def refresh_list(self, selected_value: str | None = None) -> None:
        selected = selected_value if selected_value is not None else self.current_tool()
        self._drag_handle.place_forget()
        self.listbox.delete(0, tk.END)
        for value in self.order:
            state = "на панели" if value in self.visible else "скрыт"
            self.listbox.insert(tk.END, f"{self.label_by_id.get(value, value)}  ({state})")
            if value not in self.visible:
                self.listbox.itemconfigure(tk.END, foreground=TOKENS.TEXT_DISABLED)
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
        is_visible = value in self.visible if value else False
        self.visible_var.set(is_visible)
        self.visibility_status.configure(text="Добавлен" if is_visible else "Скрыт")
        self.position_drag_handle()

    def scroll_list(self, *args) -> None:
        self.listbox.yview(*args)
        self.after_idle(self.position_drag_handle)

    def on_listbox_scroll(self, first: str, last: str) -> None:
        self.scrollbar.set(first, last)
        self.after_idle(self.position_drag_handle)

    def position_drag_handle(self) -> None:
        if not hasattr(self, "_drag_handle") or self._drag_tool is not None:
            return
        selection = self.listbox.curselection()
        if not selection:
            self._drag_handle.place_forget()
            return
        bounds = self.listbox.bbox(int(selection[0]))
        if bounds is None:
            self._drag_handle.place_forget()
            return
        _row_x, row_y, _row_width, row_height = bounds
        handle_width = 28
        handle_x = max(2, self.listbox.winfo_width() - handle_width - 3)
        self._drag_handle.place(x=handle_x, y=row_y, width=handle_width, height=row_height)
        self._drag_handle.lift()

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
        return self.start_drag(index, int(event.y), from_handle=False)

    def start_drag(self, index: int, y: int, *, from_handle: bool) -> str:
        self.clear_drag_feedback()
        self._drag_tool = self.order[index]
        self._drag_start_y = y
        self._dragged = False
        self._drag_from_handle = from_handle
        self._drag_insert_index = index
        self._drag_handle.place_forget()
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self.listbox.activate(index)
        self.update_selected_state()
        return "break"

    def handle_event_for_listbox(self, event) -> SimpleNamespace:
        return SimpleNamespace(
            x=int(event.x_root) - self.listbox.winfo_rootx(),
            y=int(event.y_root) - self.listbox.winfo_rooty(),
        )

    def begin_handle_drag(self, _event) -> str:
        value = self.current_tool()
        if value is None or value not in self.order:
            return "break"
        index = self.order.index(value)
        bounds = self.listbox.bbox(index)
        if bounds is None:
            return "break"
        return self.start_drag(index, int(bounds[1] + bounds[3] / 2), from_handle=True)

    def drag_from_handle(self, event) -> str:
        return self.drag_tool(self.handle_event_for_listbox(event))

    def end_handle_drag(self, event) -> str:
        return self.end_drag(self.handle_event_for_listbox(event))

    def drag_tool(self, event) -> str:
        if self._drag_tool is None:
            return "break"
        if abs(int(event.y) - self._drag_start_y) >= 4:
            self._dragged = True
        if event.y < 12:
            self.listbox.yview_scroll(-1, "units")
        elif event.y > self.listbox.winfo_height() - 12:
            self.listbox.yview_scroll(1, "units")
        if self._dragged:
            self._drag_insert_index = self.insertion_index_at(event.y)
            self.show_drag_feedback(event)
        return "break"

    def end_drag(self, event) -> str:
        value = self._drag_tool
        was_dragged = self._dragged
        insert_index = self._drag_insert_index
        dragged_from_handle = self._drag_from_handle
        should_toggle = value is not None and not was_dragged and not dragged_from_handle
        self._drag_tool = None
        self._dragged = False
        self._drag_from_handle = False
        self._drag_insert_index = None
        self.clear_drag_feedback()
        if should_toggle and value is not None:
            self.toggle_tool(value)
        elif was_dragged and value is not None and insert_index is not None:
            self.move_tool_to_insertion(value, insert_index)
        else:
            self.position_drag_handle()
        return "break"

    def insertion_index_at(self, y: int) -> int:
        if not self.order:
            return 0
        target = max(0, min(len(self.order) - 1, int(self.listbox.nearest(y))))
        if self._drag_tool in self.order:
            old_index = self.order.index(self._drag_tool)
            if target > old_index:
                return target + 1
            if target < old_index:
                return target
        bounds = self.listbox.bbox(target)
        if bounds is None:
            return target
        _x, row_y, _width, row_height = bounds
        return target + (1 if y >= row_y + row_height / 2 else 0)

    def show_drag_feedback(self, event) -> None:
        if self._drag_tool is None or self._drag_insert_index is None:
            return
        if self._drag_ghost is None:
            ghost = tk.Toplevel(self)
            ghost.overrideredirect(True)
            try:
                ghost.attributes("-alpha", 0.92)
                ghost.attributes("-topmost", True)
            except tk.TclError:
                pass
            ttk.Label(
                ghost,
                text=f"≡  {self.label_by_id.get(self._drag_tool, self._drag_tool)}",
                padding=(12, 6),
                relief=tk.SOLID,
            ).pack()
            self._drag_ghost = ghost
        root_x = self.listbox.winfo_rootx() + max(12, int(event.x) + 14)
        root_y = self.listbox.winfo_rooty() + int(event.y) + 10
        self._drag_ghost.geometry(f"+{root_x}+{root_y}")
        self.place_drop_indicator(self._drag_insert_index)
        if self._indicator_after_id is None:
            self.animate_drop_indicator()

    def place_drop_indicator(self, insert_index: int) -> None:
        if not self.order:
            return
        if insert_index >= len(self.order):
            bounds = self.listbox.bbox(len(self.order) - 1)
            if bounds is None:
                return
            y = bounds[1] + bounds[3] - 2
        else:
            bounds = self.listbox.bbox(max(0, insert_index))
            if bounds is None:
                return
            y = bounds[1] - 1
        self._drop_indicator.place(x=2, y=y, relwidth=1.0, width=-4, height=3)
        self._drop_indicator.lift()

    def animate_drop_indicator(self) -> None:
        if self._drag_tool is None or not self._dragged:
            self._indicator_after_id = None
            return
        colors = (TOKENS.ACCENT, TOKENS.ACCENT_HOVER, TOKENS.FOCUS, TOKENS.ACCENT_HOVER)
        self._drop_indicator.configure(background=colors[self._indicator_phase % len(colors)])
        self._indicator_phase += 1
        self._indicator_after_id = self.after(85, self.animate_drop_indicator)

    def clear_drag_feedback(self) -> None:
        if self._indicator_after_id is not None:
            self.after_cancel(self._indicator_after_id)
            self._indicator_after_id = None
        if hasattr(self, "_drop_indicator"):
            self._drop_indicator.place_forget()
        if self._drag_ghost is not None:
            self._drag_ghost.destroy()
            self._drag_ghost = None

    def move_tool_to_insertion(self, value: str, insert_index: int) -> None:
        if value not in self.order:
            return
        old_index = self.order.index(value)
        self.order.pop(old_index)
        if insert_index > old_index:
            insert_index -= 1
        new_index = max(0, min(len(self.order), int(insert_index)))
        self.order.insert(new_index, value)
        self.refresh_list(value)
        self.animate_moved_row(new_index)

    def animate_moved_row(self, index: int) -> None:
        for after_id in self._highlight_after_ids:
            try:
                self.after_cancel(after_id)
            except tk.TclError:
                pass
        self._highlight_after_ids.clear()
        normal = str(self.listbox.cget("background"))
        selected = str(self.listbox.cget("selectbackground"))
        colors = ((TOKENS.SELECTION, TOKENS.ACCENT), (TOKENS.SURFACE_SELECTED, TOKENS.ACCENT_HOVER), (TOKENS.SURFACE_HOVER, TOKENS.FOCUS), (normal, selected))
        for step, (background, selected_background) in enumerate(colors):
            after_id = self.after(
                step * 70,
                lambda bg=background, sbg=selected_background, i=index: self.listbox.itemconfigure(
                    i,
                    background=bg,
                    selectbackground=sbg,
                ),
            )
            self._highlight_after_ids.append(after_id)

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
        self.clear_drag_feedback()
        self.destroy()

    def cancel(self) -> None:
        self.clear_drag_feedback()
        self.result = None
        self.destroy()

__all__ = [name for name in globals() if not name.startswith("__")]
