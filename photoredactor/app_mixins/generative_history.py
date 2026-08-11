from __future__ import annotations

from ..app_shared import *


class GenerativeHistoryMixin:
    def _generated_layer_pixels(self, result: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pixels = result.copy()
        pixels[:, :, 3] = np.clip(
            pixels[:, :, 3].astype(np.float32) * (mask.astype(np.float32) / 255.0), 0, 255,
        ).astype(np.uint8)
        pixels[mask == 0, :3] = 0
        return pixels

    def _apply_generative_fill(
        self, result: np.ndarray, mask: np.ndarray, settings: dict[str, object], existing: Layer | None,
    ) -> None:
        pixels = self._generated_layer_pixels(result, mask)
        if existing is not None:
            before = {"pixels": existing.pixels.copy(), "generation_data": copy.deepcopy(existing.generation_data), "name": existing.name}
            existing.pixels = pixels
            existing.generation_data = copy.deepcopy(settings)
            existing.name = f"Генеративная заливка · {settings['seed']}"
            existing.touch_pixels()
            after = {"pixels": pixels.copy(), "generation_data": copy.deepcopy(settings), "name": existing.name}
            self.push_command(LayerFieldsCommand("Повторить генеративную заливку", existing.id, before, after))
        else:
            layer = Layer(f"Генеративная заливка · {settings['seed']}", pixels, generation_data=copy.deepcopy(settings))
            index = min(len(self.doc.layers), self.doc.active_layer + 1)
            self.doc.layers.insert(index, layer)
            self.doc.active_layer = index
            self.doc.dirty = True
            self.push_command(LayerInsertCommand("Генеративная заливка", index, copy.deepcopy(layer)))
        self.invalidate_pixels()
        self.refresh()

    def _apply_generative_expand(
        self, result: np.ndarray, margins: tuple[int, int, int, int],
        settings: dict[str, object], existing: Layer | None,
    ) -> None:
        left, top, right, bottom = margins
        pixels = result.copy()
        pixels[top:top + result.shape[0] - top - bottom, left:left + result.shape[1] - left - right] = 0
        if existing is not None:
            before = {"pixels": existing.pixels.copy(), "generation_data": copy.deepcopy(existing.generation_data), "name": existing.name}
            existing.pixels = pixels
            existing.generation_data = copy.deepcopy(settings)
            existing.name = f"Генеративное расширение · {settings['seed']}"
            existing.touch_pixels()
            after = {"pixels": pixels.copy(), "generation_data": copy.deepcopy(settings), "name": existing.name}
            self.push_command(LayerFieldsCommand("Повторить генеративное расширение", existing.id, before, after))
        else:
            layer = Layer(f"Генеративное расширение · {settings['seed']}", pixels, generation_data=copy.deepcopy(settings))
            command = GeneratedExpandCommand(
                "Генеративное расширение", layer, margins, (self.doc.width, self.doc.height), self.doc.active_layer,
                None if self.doc.selection_mask is None else self.doc.selection_mask.copy(),
                {name: value.copy() for name, value in self.doc.saved_selections.items()},
            )
            command.redo(self.doc)
            self.push_command(command)
        self.invalidate_pixels()
        self.refresh()

    def generative_history_dialog(self) -> None:
        layers = [layer for layer in self.doc.layers if layer.generation_data]
        if not layers:
            messagebox.showinfo("История генераций", "В документе ещё нет применённых генераций.")
            return
        dialog = tk.Toplevel(self)
        dialog.title("История генераций")
        dialog.transient(self)
        dialog.grab_set()
        tree = ttk.Treeview(dialog, columns=("type", "provider", "seed", "prompt"), show="headings", selectmode="browse")
        columns = (
            ("type", "Режим", 110), ("provider", "Модель", 170),
            ("seed", "Seed", 100), ("prompt", "Запрос", 360),
        )
        for key, label, width in columns:
            tree.heading(key, text=label)
            tree.column(key, width=width)
        tree.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 6))
        ordered = list(reversed(layers))
        for index, layer in enumerate(ordered):
            data = layer.generation_data or {}
            provider = data.get("local_model_id") if data.get("provider") == "local" else "Stability AI"
            tree.insert(
                "", tk.END, iid=str(index),
                values=("Заливка" if data.get("operation") == "fill" else "Расширение", provider, data.get("seed", ""), data.get("prompt", "")),
            )

        def repeat() -> None:
            if not tree.selection():
                return
            layer = ordered[int(tree.selection()[0])]
            self.doc.active_layer = self.doc.layers.index(layer)
            dialog.destroy()
            self._open_generative_workspace(str(layer.generation_data.get("operation", "fill")), layer)

        footer = ttk.Frame(dialog, padding=(12, 6, 12, 12))
        footer.pack(fill=tk.X)
        ttk.Button(footer, text="Повторить", command=repeat, style="Primary.TButton").pack(side=tk.RIGHT)
        ttk.Button(footer, text="Закрыть", command=dialog.destroy).pack(side=tk.RIGHT, padx=6)
        tree.bind("<Double-Button-1>", lambda _event: repeat())
        self._generative_history_tree = tree
        self._generative_history_repeat = repeat
        self.center_toplevel(dialog, 860, 460)
