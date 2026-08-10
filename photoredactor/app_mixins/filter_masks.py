from __future__ import annotations

from ..app_shared import *


class FilterMasksMixin:
    def filter_mask_editor(self, initial_mask: np.ndarray, selection_mask: np.ndarray | None = None) -> np.ndarray | None:
        mask = np.asarray(initial_mask, dtype=np.uint8).copy()
        result: np.ndarray | None = None
        dialog = tk.Toplevel(self)
        dialog.title("Редактирование маски фильтра")
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.grab_set()
        canvas_width, canvas_height = 520, 360
        canvas = tk.Canvas(dialog, width=canvas_width, height=canvas_height, background="#202226", highlightthickness=0)
        canvas.grid(row=0, column=0, rowspan=7, padx=12, pady=12)
        mode = tk.StringVar(value="Белая кисть")
        size = tk.DoubleVar(value=40)
        ttk.Label(dialog, text="Кисть").grid(row=0, column=1, sticky="w", padx=(0, 12), pady=(12, 4))
        ttk.Combobox(dialog, textvariable=mode, values=["Белая кисть", "Чёрная кисть"], state="readonly", width=20).grid(row=1, column=1, sticky="ew", padx=(0, 12))
        ttk.Label(dialog, text="Размер").grid(row=2, column=1, sticky="w", padx=(0, 12), pady=(12, 4))
        ttk.Scale(dialog, from_=2, to=max(20, min(mask.shape) // 2), variable=size, orient=tk.HORIZONTAL).grid(row=3, column=1, sticky="ew", padx=(0, 12))
        ttk.Button(dialog, text="Из текущего выделения", command=lambda: set_from_selection()).grid(row=4, column=1, sticky="ew", padx=(0, 12), pady=(14, 4))
        ttk.Button(dialog, text="Инвертировать", command=lambda: invert()).grid(row=5, column=1, sticky="ew", padx=(0, 12), pady=4)
        fill_buttons = ttk.Frame(dialog)
        fill_buttons.grid(row=6, column=1, sticky="ew", padx=(0, 12), pady=4)
        ttk.Button(fill_buttons, text="Белая", command=lambda: fill(255)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(fill_buttons, text="Чёрная", command=lambda: fill(0)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        bottom = ttk.Frame(dialog)
        bottom.grid(row=7, column=0, columnspan=2, sticky="e", padx=12, pady=(0, 12))
        last_point: list[tuple[int, int] | None] = [None]
        scale = min(canvas_width / max(1, mask.shape[1]), canvas_height / max(1, mask.shape[0]))
        shown_width = max(1, round(mask.shape[1] * scale))
        shown_height = max(1, round(mask.shape[0] * scale))
        offset_x = (canvas_width - shown_width) // 2
        offset_y = (canvas_height - shown_height) // 2

        def refresh_mask() -> None:
            shown = cv2.resize(mask, (shown_width, shown_height), interpolation=cv2.INTER_NEAREST)
            rgba = np.dstack([shown, shown, shown, np.full_like(shown, 255)])
            self._filter_mask_editor_image = ImageTk.PhotoImage(rgba_array_to_pil(rgba))
            canvas.delete("all")
            canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self._filter_mask_editor_image)

        def canvas_point(event) -> tuple[int, int] | None:
            x = int((event.x - offset_x) / max(scale, 1e-6))
            y = int((event.y - offset_y) / max(scale, 1e-6))
            return (x, y) if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] else None

        def paint(event) -> None:
            point = canvas_point(event)
            if point is None:
                return
            value = 255 if mode.get() == "Белая кисть" else 0
            radius = max(1, int(size.get()) // 2)
            previous = last_point[0] or point
            cv2.line(mask, previous, point, value, radius * 2, cv2.LINE_AA)
            cv2.circle(mask, point, radius, value, -1, cv2.LINE_AA)
            last_point[0] = point
            refresh_mask()

        def end_stroke(_event=None) -> None:
            last_point[0] = None

        def set_from_selection() -> None:
            if selection_mask is None or not np.any(selection_mask):
                messagebox.showinfo("Маска фильтра", "Текущее выделение отсутствует.", parent=dialog)
                return
            mask[:] = cv2.resize(selection_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            refresh_mask()

        def invert() -> None:
            mask[:] = 255 - mask
            refresh_mask()

        def fill(value: int) -> None:
            mask.fill(value)
            refresh_mask()

        def accept() -> None:
            nonlocal result
            result = mask.copy()
            dialog.destroy()

        canvas.bind("<Button-1>", paint)
        canvas.bind("<B1-Motion>", paint)
        canvas.bind("<ButtonRelease-1>", end_stroke)
        ttk.Button(bottom, text="ОК", command=accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(bottom, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT)
        self._filter_mask_editor_mask = mask
        self._filter_mask_editor_invert = invert
        self._filter_mask_editor_accept = accept
        refresh_mask()
        self.center_toplevel(dialog, 760, 430)
        dialog.wait_window()
        return result

    def normalize_filter_item(self, item: dict) -> dict | None:
        kind = str(item.get("type", "")).lower()
        if kind not in FILTER_TYPES:
            return None
        return self.make_filter_item(kind, self.filter_primary_value(item), item)

    def make_filter_item(self, kind: str, value: float, original: dict | None = None) -> dict:
        original = original or {}
        if kind == "blur":
            item = {"type": "blur", "radius": max(1, int(float(value)))}
        elif kind == "sharpen":
            item = {"type": "sharpen", "amount": max(0.0, float(value))}
        elif kind == "noise":
            item = {"type": "noise", "amount": max(0.0, min(1.0, float(value))), "seed": int(original.get("seed", 12345))}
        elif kind == "median":
            item = {"type": "median", "size": max(3, int(float(value)) | 1)}
        elif kind == "edge":
            item = {"type": "edge", "strength": max(0.0, min(1.0, float(value)))}
        else:
            item = {"type": "emboss", "strength": max(0.0, min(1.0, float(value)))}
        item["enabled"] = bool(original.get("enabled", True))
        item["opacity"] = max(0.0, min(1.0, float(original.get("opacity", 1.0))))
        blend_mode = str(original.get("blend_mode", "Normal"))
        item["blend_mode"] = blend_mode if blend_mode in BLEND_MODES else "Normal"
        channel = str(original.get("channel", "RGB"))
        item["channel"] = channel if channel in CHANNEL_LABELS else "RGB"
        item["mask_inverted"] = bool(original.get("mask_inverted", False))
        item["mask_density"] = max(0.0, min(1.0, float(original.get("mask_density", 1.0))))
        item["mask_feather"] = max(0.0, float(original.get("mask_feather", 0.0)))
        if isinstance(original.get("mask"), str) and original.get("mask"):
            item["mask"] = original["mask"]
        return item

    @staticmethod
    def filter_primary_value(item: dict) -> float:
        kind = str(item.get("type", "")).lower()
        if kind == "blur":
            return float(item.get("radius", 3))
        if kind == "sharpen":
            return float(item.get("amount", 1.0))
        if kind == "noise":
            return float(item.get("amount", 0.03))
        if kind == "median":
            return float(item.get("size", 3))
        if kind in {"edge", "emboss"}:
            return float(item.get("strength", 1.0))
        return 1.0

    def effects_to_text(self, effects: dict) -> str:
        parts = []
        stroke = effects.get("stroke")
        if stroke:
            parts.append(f"stroke,{stroke.get('size', 4)},{stroke.get('opacity', 1.0)}")
        shadow = effects.get("drop_shadow")
        if shadow:
            parts.append(f"shadow,{shadow.get('x', 10)},{shadow.get('y', 10)},{shadow.get('blur', 12)},{shadow.get('opacity', 0.55)}")
        glow = effects.get("outer_glow")
        if glow:
            parts.append(f"glow,{glow.get('blur', 18)},{glow.get('opacity', 0.5)}")
        return ";".join(parts)

    def parse_effects(self, raw: str) -> dict:
        effects = {}
        if not raw.strip():
            return effects
        for chunk in raw.split(";"):
            parts = [part.strip() for part in chunk.split(",") if part.strip()]
            if not parts:
                continue
            kind = parts[0].lower()
            if kind == "stroke" and len(parts) == 3:
                effects["stroke"] = {"enabled": True, "size": int(float(parts[1])), "opacity": float(parts[2]), "color": list(self.background)}
            elif kind == "shadow" and len(parts) == 5:
                effects["drop_shadow"] = {"enabled": True, "x": int(float(parts[1])), "y": int(float(parts[2])), "blur": int(float(parts[3])), "opacity": float(parts[4]), "color": [0, 0, 0, 255]}
            elif kind == "glow" and len(parts) == 3:
                effects["outer_glow"] = {"enabled": True, "blur": int(float(parts[1])), "opacity": float(parts[2]), "color": list(self.foreground)}
            else:
                raise ValueError
        return effects

    def filters_to_text(self, filters: list[dict]) -> str:
        parts = []
        for item in filters:
            kind = str(item.get("type", "")).lower()
            if kind == "blur":
                parts.append(self.filter_text_chunk("blur", item.get("radius", 3), item))
            elif kind == "sharpen":
                parts.append(self.filter_text_chunk("sharpen", item.get("amount", 1.0), item))
            elif kind == "noise":
                parts.append(self.filter_text_chunk("noise", item.get("amount", 0.03), item))
            elif kind == "median":
                parts.append(self.filter_text_chunk("median", item.get("size", 3), item))
            elif kind == "edge":
                parts.append(self.filter_text_chunk("edge", item.get("strength", 1.0), item))
            elif kind == "emboss":
                parts.append(self.filter_text_chunk("emboss", item.get("strength", 1.0), item))
        return ";".join(parts)

    @staticmethod
    def filter_text_chunk(kind: str, value: float, item: dict) -> str:
        opacity = float(item.get("opacity", 1.0))
        blend_mode = str(item.get("blend_mode", "Normal"))
        if abs(opacity - 1.0) <= 0.001 and blend_mode == "Normal":
            return f"{kind},{value}"
        return f"{kind},{value},{opacity},{blend_mode}"

    def parse_filters(self, raw: str) -> list[dict]:
        filters: list[dict] = []
        if not raw.strip():
            return filters
        for chunk in raw.split(";"):
            parts = [part.strip() for part in chunk.split(",") if part.strip()]
            if not parts:
                continue
            kind = parts[0].lower()
            if len(parts) not in {2, 4}:
                raise ValueError
            metadata: dict[str, object] = {}
            if len(parts) == 4:
                metadata["opacity"] = max(0.0, min(1.0, float(parts[2])))
                metadata["blend_mode"] = parts[3] if parts[3] in BLEND_MODES else "Normal"
            if kind == "blur":
                filters.append(self.make_filter_item("blur", max(1, int(float(parts[1]))), metadata))
            elif kind == "sharpen":
                filters.append(self.make_filter_item("sharpen", max(0.0, float(parts[1])), metadata))
            elif kind == "noise":
                filters.append(self.make_filter_item("noise", max(0.0, min(1.0, float(parts[1]))), metadata))
            elif kind == "median":
                filters.append(self.make_filter_item("median", max(3, int(float(parts[1])) | 1), metadata))
            elif kind == "edge":
                filters.append(self.make_filter_item("edge", max(0.0, min(1.0, float(parts[1]))), metadata))
            elif kind == "emboss":
                filters.append(self.make_filter_item("emboss", max(0.0, min(1.0, float(parts[1]))), metadata))
            else:
                raise ValueError
        return filters
