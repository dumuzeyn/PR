from __future__ import annotations

from ..app_shared import *


class CommandsMixin:
    def push_command(self, command) -> None:
        self.history.push(command)
        self._edit_generation += 1
        self.record_action(command.label)
        self.status_text(command.label)
        self.refresh_history_panel()

    def run_document_command(self, label: str, fn) -> None:
        before = self.doc.raw_state()
        fn()
        after = self.doc.raw_state()
        command = self.compact_document_command(label, before, after)
        if command is not None:
            self.history.push(command)
        self._edit_generation += 1
        self.record_action(label)
        self.status_text(label)
        self.refresh_history_panel()

    @classmethod
    def state_value_equal(cls, left, right) -> bool:
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and left.shape == right.shape and np.array_equal(left, right)
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(cls.state_value_equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(cls.state_value_equal(a, b) for a, b in zip(left, right))
        return left == right

    def compact_document_command(self, label: str, before: dict, after: dict):
        before_layers = before.get("layers", [])
        after_layers = after.get("layers", [])
        same_layer_order = [layer.get("id") for layer in before_layers] == [layer.get("id") for layer in after_layers]
        document_keys = (set(before) | set(after)) - {"layers"}
        changed_document = {
            key: (before.get(key), after.get(key))
            for key in document_keys
            if not self.state_value_equal(before.get(key), after.get(key))
        }
        changed_layers: list[tuple[str, dict, dict]] = []
        if same_layer_order:
            for old_layer, new_layer in zip(before_layers, after_layers):
                changed_fields = {
                    key
                    for key in set(old_layer) | set(new_layer)
                    if not self.state_value_equal(old_layer.get(key), new_layer.get(key))
                }
                if changed_fields:
                    changed_layers.append(
                        (
                            str(old_layer["id"]),
                            {key: old_layer.get(key) for key in changed_fields},
                            {key: new_layer.get(key) for key in changed_fields},
                        )
                    )
        if same_layer_order and len(changed_layers) == 1 and not changed_document:
            layer_id, old_fields, new_fields = changed_layers[0]
            return LayerFieldsCommand(label, layer_id, old_fields, new_fields)
        if same_layer_order and not changed_layers and changed_document:
            return DocumentFieldsCommand(
                label,
                {key: value[0] for key, value in changed_document.items()},
                {key: value[1] for key, value in changed_document.items()},
            )
        if same_layer_order and not changed_layers and not changed_document:
            return None
        return DocumentStateCommand(label, before, after)

    def run_selection_command(self, label: str, fn) -> None:
        before = None if self.doc.selection_mask is None else self.doc.selection_mask.copy()
        fn()
        after = None if self.doc.selection_mask is None else self.doc.selection_mask.copy()
        self.history.push(SelectionMaskCommand(label, before, after))
        self._edit_generation += 1
        self.record_action(label)
        self.selection_box = self.doc.selection_bounds()
        self._selection_contour_signature = None
        self.update_selection_overlay()
        self.status_text(label)

    def record_action(self, label: str) -> None:
        if not self.action_recorder.recording:
            return
        normalized = label.lower()
        command = ""
        params: dict[str, object] = {}
        if "resize image" in normalized or "размер изображения" in normalized:
            command, params = "resize_image", {"width": self.doc.width, "height": self.doc.height}
        elif "resize canvas" in normalized or "размер холста" in normalized:
            command, params = "resize_canvas", {"width": self.doc.width, "height": self.doc.height, "anchor": "center"}
        elif "flatten" in normalized or "свести" in normalized:
            command = "flatten"
        elif "rotate" in normalized or "повернуть" in normalized:
            command, params = "rotate", {"angle": 180 if "180" in normalized else 90}
        elif "flip" in normalized or "отразить" in normalized:
            command, params = "flip", {"axis": "vertical" if "vertical" in normalized or "вертик" in normalized else "horizontal"}
        elif "bit depth" in normalized or "глубина" in normalized:
            command, params = "set_bit_depth", {"bit_depth": self.doc.bit_depth}
        elif "color model" in normalized or "цветовая модель" in normalized:
            command, params = "set_color_model", {"color_model": self.doc.color_model}
        elif self.doc.layer.filters:
            command, params = "filter_stack", {"filters": copy.deepcopy(self.doc.layer.filters)}
        if command:
            self.action_recorder.record(command, params, label)

    def start_action_recording(self) -> None:
        self.action_recorder.start()
        self.status_text("Запись действия начата")

    def stop_action_recording(self) -> None:
        self.action_recorder.stop()
        self.status_text(f"Запись остановлена: {len(self.action_recorder.steps)} шагов")

    def clear_action_recording(self) -> None:
        self.action_recorder.steps.clear()
        self.status_text("Запись действия очищена")

    def save_action_recording(self) -> None:
        if not self.action_recorder.steps:
            messagebox.showinfo("Действия", "Нет записанных исполняемых шагов.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Action JSON", "*.json")])
        if not path:
            return
        self.action_recorder.save(path)
        self.status_text(f"Действие сохранено: {path}")

    def run_action_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Действие PhotoRedactor", "*.json"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            self.run_document_command("Выполнить действие", lambda: self.action_runner.run(self.doc, path))
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Действия", str(exc))

    def batch_action_file(self) -> None:
        action = filedialog.askopenfilename(filetypes=[("Действие PhotoRedactor", "*.json")])
        if not action:
            return
        sources = filedialog.askopenfilenames(filetypes=[("Изображения", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff")])
        if not sources:
            return
        destination = filedialog.askdirectory(title="Папка результата")
        if not destination:
            return
        self.run_background(
            "Пакетное действие",
            lambda: self.action_runner.batch(action, list(sources), destination),
            lambda results: messagebox.showinfo("Действия", f"Обработано файлов: {len(results)}"),
        )

    def schedule_autosave(self) -> None:
        self.after(60000, self.autosave_tick)

    def autosave_tick(self) -> None:
        self.autosave_recovery()
        self.schedule_autosave()

    def autosave_recovery(self) -> None:
        if not getattr(self, "doc", None) or not self.doc.dirty:
            return
        try:
            self.recovery_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot = self.document_copy()
            snapshot.save_project(self.recovery_path)
            self.doc.dirty = True
        except Exception:
            pass

    def check_recovery_file(self) -> None:
        if self.recovery_path.exists() and messagebox.askyesno("Recovery", "A recovery file was found. Open it?"):
            self.open_recovery()

    def open_recovery(self) -> None:
        if not self.recovery_path.exists():
            messagebox.showinfo("Восстановление", "Файл восстановления не найден.")
            return
        self.doc = Document.open_project(self.recovery_path)
        self.history.clear()
        self.selection_box = self.doc.selection_bounds()

    def run_pixel_delta_command(self, label: str, fn) -> tuple[int, int, int, int] | None:
        layer = self.doc.layer
        if layer.locked or layer.kind == "adjustment":
            return None
        layer_id = layer.id
        before = layer.pixels.copy()
        before_working = layer.working_rgba().copy() if layer.working_pixels is not None else None
        fn()
        target = self.doc.get_layer(layer_id)
        if target is None or target.pixels.shape != before.shape:
            return None
        changed = np.any(target.pixels != before, axis=2)
        if not np.any(changed):
            return None
        ys, xs = np.where(changed)
        rect = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        x1, y1, x2, y2 = rect
        target.touch_pixels()
        after_working = target.working_rgba() if before_working is not None else None
        self.push_command(PixelPatchCommand(
            label,
            layer_id,
            rect,
            before[y1:y2, x1:x2].copy(),
            target.pixels[y1:y2, x1:x2].copy(),
            None if before_working is None else before_working[y1:y2, x1:x2].copy(),
            None if after_working is None else after_working[y1:y2, x1:x2].copy(),
        ))
        self.doc.dirty = True
        self.request_canvas_refresh(self.local_to_document_rect(rect, target), target, "pixels")
        self.refresh_layers()
        return rect

    def set_layer_property(self, label: str, attribute: str, value, affects_canvas: bool = True, preserve_render_cache: bool = True) -> bool:
        layer = self.doc.layer
        before = copy.deepcopy(getattr(layer, attribute))
        after = copy.deepcopy(value)
        if before == after:
            return False
        setattr(layer, attribute, after)
        if attribute == "mask":
            layer.touch_mask()
        self.doc.dirty = True
        self.push_command(LayerPropertyCommand(label, layer.id, attribute, before, after))
        if affects_canvas:
            self.refresh(preserve_render_cache=preserve_render_cache)
        else:
            self.refresh_layers()
        return True
        self.show_editor()
        self.status_text(f"Открыто восстановление: {self.recovery_path}")

    def clear_recovery(self) -> None:
        if self.recovery_path.exists():
            self.recovery_path.unlink()
        self.status_text("Recovery cleared")

    def undo(self) -> None:
        label = self.history.undo(self.doc)
        if label:
            self._edit_generation += 1
            self.refresh_history_command(self.history.last_command)
            self.status_text(f"Undo: {label}")

    def redo(self) -> None:
        label = self.history.redo(self.doc)
        if label:
            self._edit_generation += 1
            self.refresh_history_command(self.history.last_command)
            self.status_text(f"Redo: {label}")

    def refresh_history_command(self, command) -> None:
        if isinstance(command, (PixelPatchCommand, PixelTilePatchCommand, MaskPatchCommand, MaskTilePatchCommand)):
            layer = self.doc.get_layer(command.layer_id)
            if layer is not None:
                kind = "mask" if isinstance(command, (MaskPatchCommand, MaskTilePatchCommand)) else "pixels"
                for rect in command.dirty_rects:
                    self.render_engine.invalidate_region(self.doc, self.local_to_document_rect(rect, layer), layer, kind)
                self._composite_dirty = True
                self._view_dirty = True
                self.refresh_canvas()
                self.refresh_layers()
                self.info.configure(text=f"{self.doc.width} x {self.doc.height}px\nРЎР»РѕРµРІ: {len(self.doc.layers)}\nРђРєС‚РёРІРЅС‹Р№: {self.doc.layer.name}")
                return
        if isinstance(command, LayerPropertyCommand):
            if command.attribute in {"name", "locked", "mask_linked"}:
                self.refresh_layers()
                return
            self.refresh(preserve_render_cache=command.attribute not in {"filters", "effects", "mask", "mask_feather"})
            return
        if isinstance(command, (LayerOpacityCommand, LayerBlendModeCommand, LayerVisibilityCommand)):
            self.refresh(preserve_render_cache=True)
            return
        if isinstance(command, LayerFieldsCommand):
            changed = set(command.before) | set(command.after)
            self.refresh(preserve_render_cache=not bool(changed & {"pixels", "filters", "effects", "mask", "mask_feather"}))
            return
        self.refresh()
