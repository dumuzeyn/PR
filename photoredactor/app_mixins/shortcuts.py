from __future__ import annotations

from ..app_shared import *


class ShortcutsMixin:
    def _build_ui(self) -> None:
        self._build_menu()
        self.editor_root = ttk.Frame(self, style="App.TFrame")
        self.editor_root.pack(fill=tk.BOTH, expand=True)
        options_bar = ttk.Frame(self.editor_root, style="Topbar.TFrame", height=46)
        options_bar.pack(fill=tk.X)
        options_bar.pack_propagate(False)
        self._build_tool_options(options_bar)
        ttk.Separator(self.editor_root).pack(fill=tk.X)
        root = ttk.PanedWindow(self.editor_root, orient=tk.HORIZONTAL)
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root, width=188, style="Panel.TFrame")
        center = ttk.Frame(root, style="Workspace.TFrame")
        right = ttk.Frame(root, width=292, style="Panel.TFrame")
        root.add(left, weight=0)
        root.add(center, weight=1)
        root.add(right, weight=0)

        self._build_tools(left)
        self._build_canvas(center)
        self._build_panels(right)

        self.status_frame = ttk.Frame(self, style="Status.TFrame")
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status = ttk.Label(self.status_frame, text="", anchor=tk.W, style="Status.TLabel")
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.status_coords = ttk.Label(self.status_frame, text="", style="Status.TLabel", width=13, anchor=tk.E)
        self.status_coords.pack(side=tk.RIGHT)
        self.status_zoom = ttk.Label(self.status_frame, text="100%", style="Status.TLabel", width=7, anchor=tk.E)
        self.status_zoom.pack(side=tk.RIGHT)
        self.zoom_label = self.status_zoom
        self.status_size = ttk.Label(self.status_frame, text="", style="Status.TLabel", width=14, anchor=tk.E)
        self.status_size.pack(side=tk.RIGHT)
        self._build_shortcuts()
        self.bind_all("<KeyPress-space>", self.space_down)
        self.bind_all("<KeyRelease-space>", self.space_up)
        for sequence in ("<Left>", "<Right>", "<Up>", "<Down>", "<Shift-Left>", "<Shift-Right>", "<Shift-Up>", "<Shift-Down>"):
            self.bind_all(sequence, self.nudge_selected_object)

    def _build_shortcuts(self) -> None:
        bindings = {
            "<Delete>": self.shortcut_delete,
            "<Escape>": self.cancel_incomplete_interaction,
            "<Return>": self.shortcut_enter,
        }
        for sequence, callback in bindings.items():
            self.bind_all(sequence, callback)
        self.bind_all("<Control-KeyPress>", self.shortcut_control_key)
        self.bind_all("<KeyPress>", self.shortcut_plain_key)

    def shortcut_control_key(self, event):
        if event_key(event) == "p" and not (int(getattr(event, "state", 0)) & 0x0001):
            return self.shortcut_print(event)
        callbacks = {
            "undo": self.shortcut_undo, "redo": self.shortcut_redo, "save": self.shortcut_save,
            "save_as": self.shortcut_save_as, "open": self.shortcut_open, "new_document": self.shortcut_new,
            "select_all": self.shortcut_select_all, "deselect": self.shortcut_deselect,
            "invert_selection": self.shortcut_invert_selection, "copy": self.shortcut_copy,
            "cut": self.shortcut_cut, "paste": self.shortcut_paste, "new_layer": self.shortcut_new_layer,
            "duplicate_layer": self.shortcut_duplicate_layer, "merge_down": self.shortcut_merge_down,
            "flatten": self.shortcut_flatten, "free_transform": self.shortcut_free_transform,
            "fit_to_screen": self.shortcut_fit_to_screen, "actual_size": self.shortcut_actual_size,
        }
        callback = callbacks.get(command_for_event(event))
        return callback(event) if callback is not None else None

    def shortcut_plain_key(self, event):
        if int(getattr(event, "state", 0)) & 0x0004:
            return None
        key = event_key(event)
        if key == "x":
            return self.shortcut_swap_colors(event)
        if key == "d":
            return self.shortcut_reset_colors(event)
        if key == "+":
            return self.shortcut_zoom_in(event)
        if key == "-":
            return self.shortcut_zoom_out(event)
        if key in {"[", "]"}:
            return self.shortcut_brush_size(1 if key == "]" else -1)
        tools = TOOL_SHORTCUT_GROUPS.get(key)
        if tools and self.shortcut_context() == "canvas" and self._editor_active:
            current = self.tool.get()
            target = tools[(tools.index(current) + 1) % len(tools)] if current in tools else tools[0]
            return self.shortcut_tool(target)
        return None

    def shortcut_tool(self, tool: str):
        if self.shortcut_context() == "canvas" and self._editor_active:
            self.select_tool(tool)
            return "break"
        return None

    @staticmethod
    def _widget_is_descendant(widget, parent) -> bool:
        current = widget
        while current is not None:
            if current is parent:
                return True
            current = getattr(current, "master", None)
        return False

    def shortcut_context(self) -> str:
        focus = self.focus_get()
        if focus is not None and isinstance(focus, (tk.Text, tk.Entry, ttk.Entry, ttk.Spinbox)):
            return "text"
        if hasattr(self, "layer_list") and focus is not None and self._widget_is_descendant(focus, self.layer_list):
            return "layers"
        grabbed = self.grab_current()
        if grabbed is not None and grabbed is not self:
            return "modal"
        return "canvas"

    def shortcut_undo(self, _event=None):
        if self.shortcut_context() == "text":
            return None
        if self._editor_active:
            self.undo()
        return "break"

    def shortcut_redo(self, _event=None):
        if self.shortcut_context() == "text":
            return None
        if self._editor_active:
            self.redo()
        return "break"

    def shortcut_save(self, _event=None):
        if self._editor_active:
            self.save()
        return "break"

    def shortcut_save_as(self, _event=None):
        if self._editor_active:
            self.save_as_project()
        return "break"

    def shortcut_open(self, _event=None):
        self.open_file()
        return "break"

    def shortcut_new(self, _event=None):
        self.new_document()
        return "break"

    def shortcut_print(self, _event=None):
        if self._editor_active:
            self.system_print_document()
        return "break"

    def shortcut_select_all(self, _event=None):
        context = self.shortcut_context()
        if context == "text":
            focus = self.focus_get()
            if isinstance(focus, tk.Text):
                focus.tag_add(tk.SEL, "1.0", "end-1c")
                focus.mark_set(tk.INSERT, "1.0")
            elif focus is not None:
                focus.selection_range(0, tk.END)
            return "break"
        if context == "layers":
            self.layer_list.selection_set(0, tk.END)
            self.layer_selected(None)
            return "break"
        if context == "canvas" and self._editor_active:
            self.select_all()
            return "break"
        return None

    def shortcut_deselect(self, _event=None):
        if self.shortcut_context() == "canvas" and self._editor_active:
            self.clear_selection()
            return "break"
        return None

    def shortcut_invert_selection(self, _event=None):
        if self.shortcut_context() == "canvas" and self._editor_active:
            self.invert_selection()
            return "break"
        return None

    def shortcut_delete(self, _event=None):
        context = self.shortcut_context()
        if context == "text":
            return None
        if context == "layers":
            self.delete_layer()
        elif context == "canvas":
            if not self.delete_selected_anchors():
                self.delete_selected_pixels()
        return "break"

    def shortcut_copy(self, _event=None):
        if self.shortcut_context() == "text":
            return None
        self.copy_pixels()
        return "break"

    def shortcut_cut(self, _event=None):
        if self.shortcut_context() == "text":
            return None
        self.copy_pixels()
        self.delete_selected_pixels()
        return "break"

    def shortcut_paste(self, _event=None):
        if self.shortcut_context() == "text":
            return None
        self.paste_pixels()
        return "break"

    def shortcut_enter(self, _event=None):
        if self.shortcut_context() == "text":
            return None
        if self.tool.get() == "crop":
            self.apply_crop_overlay()
            return "break"
        if self.tool.get() == "patch" and self.apply_patch_preview():
            return "break"
        return None

    def shortcut_zoom_in(self, _event=None):
        if self._editor_active and self.shortcut_context() != "text":
            self.set_zoom(self.zoom.get() * 1.25)
            return "break"
        return None

    def shortcut_zoom_out(self, _event=None):
        if self._editor_active and self.shortcut_context() != "text":
            self.set_zoom(self.zoom.get() / 1.25)
            return "break"
        return None

    def shortcut_brush_size(self, direction: int):
        sized_tools = {"brush", "eraser", "blur_tool", "sharpen_tool", "dodge", "burn", "clone", "healing", "spot_healing", "quick_selection"}
        if self.shortcut_context() != "canvas" or not self._editor_active or self.tool.get() not in sized_tools:
            return None
        current = int(self.brush_size.get())
        step = max(1, current // 10)
        updated = max(1, min(220, current + step * int(direction)))
        self.brush_size.set(updated)
        self.status_text(f"Размер инструмента: {updated} px")
        return "break"

    def shortcut_new_layer(self, _event=None):
        if self.shortcut_context() != "text" and self._editor_active:
            self.new_layer()
            return "break"
        return None

    def shortcut_duplicate_layer(self, _event=None):
        if self.shortcut_context() != "text" and self._editor_active:
            self.duplicate_layer()
            return "break"
        return None

    def shortcut_merge_down(self, _event=None):
        if self.shortcut_context() != "text" and self._editor_active:
            self.merge_down()
            return "break"
        return None

    def shortcut_flatten(self, _event=None):
        if self.shortcut_context() != "text" and self._editor_active:
            self.flatten()
            return "break"
        return None

    def shortcut_free_transform(self, _event=None):
        if self.shortcut_context() == "canvas" and self._editor_active:
            self.free_transform_layer()
            return "break"
        return None

    def shortcut_fit_to_screen(self, _event=None):
        if self.shortcut_context() == "canvas" and self._editor_active:
            self.fit_to_screen()
            return "break"
        return None

    def shortcut_actual_size(self, _event=None):
        if self.shortcut_context() == "canvas" and self._editor_active:
            self.set_zoom(1.0)
            return "break"
        return None

    def nudge_selected_object(self, event):
        if self.shortcut_context() != "canvas" or self.tool.get() != "move" or self.doc.layer.id not in self.selected_layer_ids:
            return None
        layer = self.doc.layer
        if layer.kind not in {"shape", "text"} or layer.locked:
            return None
        step = 10 if event.state & 0x0001 else 1
        dx = -step if event.keysym == "Left" else step if event.keysym == "Right" else 0
        dy = -step if event.keysym == "Up" else step if event.keysym == "Down" else 0
        before = (layer.x, layer.y)
        layer.x += dx
        layer.y += dy
        self.doc.dirty = True
        self.push_command(LayerMoveCommand("Сдвинуть объект", layer.id, before, (layer.x, layer.y)))
        self.refresh()
        return "break"

    def shortcut_swap_colors(self, _event=None):
        if self.shortcut_context() == "canvas":
            self.swap_colors()
            return "break"
        return None

    def shortcut_reset_colors(self, _event=None):
        if self.shortcut_context() == "canvas":
            self.reset_colors()
            return "break"
        return None

    def cancel_incomplete_interaction(self, _event=None) -> str:
        if self._text_editor is not None:
            self.cancel_text_edit()
            self.status_text("Редактирование текста отменено")
            return "break"
        if hasattr(self, "canvas"):
            self.clear_drag_preview()
            self.clear_lasso_overlay()
            self.clear_quick_selection_preview()
            self.clear_gradient_preview()
            self.clear_patch_preview()
        self.drag_start = None
        self.last_point = None
        self._shape_drag_options = None
        self._crop_box = None
        self._crop_drag_handle = None
        self._crop_drag_origin_box = None
        self._clone_anchor_target = None
        self._clone_anchor_source = None
        if hasattr(self, "canvas"):
            self.update_selection_overlay()
        return "break"
