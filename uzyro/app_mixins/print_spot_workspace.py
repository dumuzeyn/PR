from __future__ import annotations

from ..app_shared import *


class PrintSpotWorkspaceMixin:
    def system_print_document(self) -> None:
        try:
            owner = int(self.winfo_id())
            if print_document(self.doc, owner=owner, fit_to_page=True):
                self.status_text("Документ передан в очередь печати")
        except Exception as exc:
            messagebox.showerror("Печать", f"Не удалось напечатать документ:\n{exc}", parent=self)

    def export_separations_dialog(self, profile: str, intent: str, black_point: bool, parent=None) -> None:
        if not profile or not Path(profile).exists():
            messagebox.showinfo("Цветоделение", "Сначала выберите CMYK ICC-профиль.", parent=parent or self)
            return
        directory = filedialog.askdirectory(title="Папка для печатных форм", parent=parent or self)
        if not directory:
            return
        try:
            manifest = export_color_separations(self.doc, directory, profile, intent, black_point)
            messagebox.showinfo("Цветоделение", f"Печатные формы сохранены:\n{manifest.parent}", parent=parent or self)
        except Exception as exc:
            messagebox.showerror("Цветоделение", str(exc), parent=parent or self)

    def _edit_spot_color(self, parent, current: SpotColor | None = None) -> SpotColor | None:
        dialog = tk.Toplevel(parent)
        dialog.title("Плашечная краска")
        dialog.transient(parent)
        dialog.grab_set()
        name = tk.StringVar(value=current.name if current else "")
        source = tk.StringVar(value=current.source if current else "Пользовательские")
        initial_lab = current.lab if current else (50.0, 0.0, 0.0)
        lab_values = [tk.DoubleVar(value=value) for value in initial_lab]
        rgb = list(current.alternate_rgb if current else lab_to_srgb(initial_lab))
        result: list[SpotColor] = []

        form = ttk.Frame(dialog, padding=14)
        form.pack(fill=tk.BOTH, expand=True)
        ttk.Label(form, text="Название").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=name, width=34).grid(row=0, column=1, columnspan=3, sticky="ew", padx=(8, 0))
        ttk.Label(form, text="Библиотека / производитель").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(form, textvariable=source).grid(row=1, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(8, 0))
        for index, (label, variable) in enumerate(zip(("L", "a", "b"), lab_values)):
            ttk.Label(form, text=label).grid(row=2, column=index + 1, sticky="w", pady=(12, 0))
            ttk.Spinbox(
                form,
                from_=0 if index == 0 else -128,
                to=100 if index == 0 else 127,
                increment=0.1,
                textvariable=variable,
                width=9,
            ).grid(row=3, column=index + 1, sticky="w", pady=(3, 0))

        swatch = tk.Canvas(form, width=42, height=42, highlightthickness=1, highlightbackground=TOKENS.BORDER)
        swatch.grid(row=2, column=0, rowspan=2, sticky="w", pady=(12, 0))

        def refresh_swatch(*_args) -> None:
            try:
                rgb[:] = lab_to_srgb(tuple(value.get() for value in lab_values))
                swatch.configure(background=f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
            except (tk.TclError, ValueError):
                pass

        for variable in lab_values:
            variable.trace_add("write", refresh_swatch)
        refresh_swatch()

        def save() -> None:
            try:
                result.append(
                    SpotColor(
                        name.get(),
                        tuple(value.get() for value in lab_values),
                        tuple(rgb),
                        source.get(),
                        current.id if current else uuid.uuid4().hex,
                    )
                )
            except (ValueError, tk.TclError) as exc:
                messagebox.showerror("Плашечная краска", str(exc), parent=dialog)
                return
            dialog.destroy()

        footer = ttk.Frame(form)
        footer.grid(row=4, column=0, columnspan=4, sticky="e", pady=(16, 0))
        ttk.Button(footer, text="Отмена", command=dialog.destroy).pack(side=tk.LEFT)
        ttk.Button(footer, text="Сохранить", command=save, style="Primary.TButton").pack(side=tk.LEFT, padx=(6, 0))
        form.columnconfigure(1, weight=1)
        dialog.bind("<Return>", lambda _event: save())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        self.center_toplevel(dialog, 500, 260)
        dialog.wait_window()
        return result[0] if result else None

    @staticmethod
    def _merge_spot_colors(existing: list[SpotColor], imported: list[SpotColor]) -> list[SpotColor]:
        result = list(existing)
        keys = {(color.source.casefold(), color.name.casefold()): index for index, color in enumerate(result)}
        for color in imported:
            key = color.source.casefold(), color.name.casefold()
            if key in keys:
                original_id = result[keys[key]].id
                result[keys[key]] = SpotColor(color.name, color.lab, color.alternate_rgb, color.source, original_id)
            else:
                keys[key] = len(result)
                result.append(color)
        return result

    def spot_colors_workspace(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Плашечные краски")
        dialog.transient(self)
        dialog.grab_set()
        colors = document_spot_colors(self.doc)

        body = ttk.Frame(dialog, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(body, columns=("source", "lab", "rgb"), show="tree headings", selectmode="browse")
        tree.heading("#0", text="Название")
        tree.heading("source", text="Библиотека")
        tree.heading("lab", text="Lab")
        tree.heading("rgb", text="Экранный RGB")
        tree.column("#0", width=210)
        tree.column("source", width=190)
        tree.column("lab", width=150)
        tree.column("rgb", width=110)
        tree.grid(row=0, column=0, sticky="nsew")
        scroll = SlimScrollbar(body, orient=tk.VERTICAL, command=tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=scroll.set)
        status = ttk.Label(body, style="Secondary.TLabel")
        status.grid(row=1, column=0, sticky="w", pady=(8, 0))

        def selected_index() -> int | None:
            selection = tree.selection()
            return int(selection[0]) if selection else None

        def refresh(select: int | None = None) -> None:
            tree.delete(*tree.get_children())
            assigned = assigned_spot_color(self.doc, self.doc.layer.id)
            for index, color in enumerate(colors):
                marker = " ●" if assigned and assigned.id == color.id else ""
                lab = f"{color.lab[0]:.1f}, {color.lab[1]:.1f}, {color.lab[2]:.1f}"
                rgb = "#{:02X}{:02X}{:02X}".format(*color.alternate_rgb)
                tree.insert("", tk.END, iid=str(index), text=color.name + marker, values=(color.source, lab, rgb))
            if colors:
                index = min(select if select is not None else 0, len(colors) - 1)
                tree.selection_set(str(index))
                tree.focus(str(index))
            status.configure(
                text=f"Активный слой: {self.doc.layer.name}  |  Краска: {assigned.name if assigned else 'не назначена'}"
            )

        def commit(label: str, updated: list[SpotColor]) -> None:
            colors[:] = updated
            self.run_document_command(label, lambda: replace_document_spot_colors(self.doc, colors))
            refresh()
            self.refresh_layers()

        def add() -> None:
            color = self._edit_spot_color(dialog)
            if color:
                commit("Добавить плашечную краску", colors + [color])
                refresh(len(colors) - 1)

        def edit() -> None:
            index = selected_index()
            if index is None:
                return
            color = self._edit_spot_color(dialog, colors[index])
            if color:
                updated = list(colors)
                updated[index] = color
                commit("Изменить плашечную краску", updated)
                refresh(index)

        def remove() -> None:
            index = selected_index()
            if index is not None:
                commit("Удалить плашечную краску", colors[:index] + colors[index + 1 :])

        def import_colors() -> None:
            path = filedialog.askopenfilename(
                title="Импорт библиотеки красок",
                filetypes=[("Библиотеки красок", "*.ase *.prswatches *.json"), ("Все файлы", "*.*")],
                parent=dialog,
            )
            if not path:
                return
            try:
                imported = load_library(path)
                commit("Импортировать библиотеку красок", self._merge_spot_colors(colors, imported))
                messagebox.showinfo("Библиотека красок", f"Импортировано цветов: {len(imported)}", parent=dialog)
            except Exception as exc:
                messagebox.showerror("Библиотека красок", str(exc), parent=dialog)

        def export_colors() -> None:
            path = filedialog.asksaveasfilename(
                title="Экспорт библиотеки красок",
                defaultextension=".ase",
                filetypes=[("Adobe Swatch Exchange", "*.ase"), ("UZYRO", "*.prswatches")],
                parent=dialog,
            )
            if path:
                try:
                    save_library(path, colors)
                except Exception as exc:
                    messagebox.showerror("Библиотека красок", str(exc), parent=dialog)

        def apply_to_layer() -> None:
            index = selected_index()
            if index is None:
                return
            color = colors[index]
            self.run_document_command(
                "Назначить плашечную краску слою", lambda: assign_spot_color(self.doc, self.doc.layer.id, color.id)
            )
            self.foreground = (*color.alternate_rgb, 255)
            self.refresh_color_control()
            self.refresh_layers()
            refresh(index)

        def clear_layer() -> None:
            self.run_document_command("Снять плашечную краску со слоя", lambda: assign_spot_color(self.doc, self.doc.layer.id, None))
            self.refresh_layers()
            refresh(selected_index())

        toolbar = ttk.Frame(body)
        toolbar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        for label, command in (("Добавить", add), ("Изменить", edit), ("Удалить", remove), ("Импорт...", import_colors), ("Экспорт...", export_colors)):
            ttk.Button(toolbar, text=label, command=command).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="Снять со слоя", command=clear_layer).pack(side=tk.RIGHT)
        ttk.Button(toolbar, text="Назначить слою", command=apply_to_layer, style="Primary.TButton").pack(side=tk.RIGHT, padx=(0, 6))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        tree.bind("<Double-1>", lambda _event: apply_to_layer())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        self._spot_color_tree = tree
        self._spot_color_refresh = refresh
        self.center_toplevel(dialog, 820, 520)
        refresh()


__all__ = [name for name in globals() if not name.startswith("__")]
