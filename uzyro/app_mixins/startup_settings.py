from __future__ import annotations

from ..app_shared import *
from ..brush_config import default_brush_config, is_test_polluted_brush_settings, normalize_brush_config
from ..brand import user_data_directory
class StartupSettingsMixin:
    def __init__(self) -> None:
        super().__init__()
        self.theme = TOKENS
        self.ui_style = configure_theme(self)
        self.configure(background=TOKENS.BACKGROUND)
        self.withdraw()
        self.title("UZYRO - редактор изображений")
        self.geometry("1440x920")
        self.minsize(1000, 640)

        self.doc = Document.new()
        self.history = History()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.app_data_dir = user_data_directory(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))))
        self.recovery_path = self.app_data_dir / "recovery.prdx"
        self.settings_path = self.app_data_dir / "settings.json"
        self.recent_files: list[str] = []
        self.action_recorder = ActionRecorder()
        self.action_runner = ActionRunner()
        self.batch_queue = BatchQueue(self.action_runner)
        self.plugin_registry = PluginRegistry()
        self.plugin_registry.discover()
        self._plugin_action_names = set(self.plugin_registry.action_commands)
        for name, callback in self.plugin_registry.action_commands.items():
            self.action_runner.register(name, callback)
        self._edit_generation = 0
        self.adjustment_presets = {name: dict(value) for name, value in ADJUSTMENT_PRESETS.items()}
        self.tool = tk.StringVar(value="brush")
        self.auto_select = tk.StringVar(value="Слой")
        self.tool_order = [value for _label, value, _description in TOOL_DEFINITIONS]
        self.visible_tools = list(self.tool_order)
        self.tool_pane_position = 360
        self.custom_canvas_width = 1280
        self.custom_canvas_height = 900
        self.custom_canvas_dpi = 72
        self.custom_canvas_background = "Белый"
        self.generative_settings = {
            "provider": "local",
            "local_model_id": "realistic-vision-51-inpaint", "local_backend": "auto",
            "variants": 3, "style": "photographic", "creativity": 0.5,
            "performance_profile": "balanced", "steps": 6, "cfg_scale": 1.5,
            "strength": 0.88, "sampler": "LCM",
        }
        self.paint_target = tk.StringVar(value="pixels")
        self.selection_mode = tk.StringVar(value="replace")
        self.retouch_preset = tk.StringVar(value="Средняя ретушь")
        self.zoom = tk.DoubleVar(value=1.0)
        self.brush_size = tk.IntVar(value=28)
        self.opacity = tk.DoubleVar(value=1.0)
        self.hardness = tk.DoubleVar(value=1.0)
        self.brush_flow = tk.DoubleVar(value=1.0)
        self.brush_spacing = tk.DoubleVar(value=0.0)
        self.brush_smoothing = tk.DoubleVar(value=0.15)
        self.brush_blend_mode = tk.StringVar(value="Normal")
        self.pressure_size = tk.BooleanVar(value=False)
        self.pressure_opacity = tk.BooleanVar(value=False)
        self.pressure_flow = tk.BooleanVar(value=False)
        self.brush_preset = tk.StringVar(value="Круглая кисть")
        self.brush_preset_name = tk.StringVar(value="Мой пресет")
        self.brush_presets = copy.deepcopy(BRUSH_PRESET_DEFAULTS)
        self.brush_advanced = default_brush_config()
        self.retouch_strength = tk.DoubleVar(value=0.25)
        self.exposure = tk.DoubleVar(value=0.15)
        self.tonal_range = tk.StringVar(value="Средние тона")
        self.tool_settings = copy.deepcopy(TOOL_SETTINGS_DEFAULTS)
        self._active_settings_tool = "brush"
        self.shape_stroke_width = tk.IntVar(value=2)
        self.shape_from_center = tk.BooleanVar(value=False)
        self.polygon_sides = tk.IntVar(value=5)
        self.star_points_count = tk.IntVar(value=5)
        self.star_inner_ratio = tk.DoubleVar(value=0.5)
        self.custom_shape_preset = tk.StringVar(value=next(iter(CUSTOM_SHAPE_PRESETS)))
        self.selection_feather = tk.IntVar(value=0)
        self.selection_antialias = tk.BooleanVar(value=True)
        self.magic_contiguous = tk.BooleanVar(value=True)
        self.magic_sample_all_layers = tk.BooleanVar(value=False)
        self.color_range_sample_all_layers = tk.BooleanVar(value=False)
        self.tolerance = tk.IntVar(value=24)
        self.color_range_sample_hex = tk.StringVar(value="#000000")
        self.clone_aligned = tk.BooleanVar(value=True)
        self.clone_sampling = tk.StringVar(value="Текущий слой")
        self.clone_source_x = tk.IntVar(value=0)
        self.clone_source_y = tk.IntVar(value=0)
        self.clone_offset_x = tk.IntVar(value=0)
        self.clone_offset_y = tk.IntVar(value=0)
        self.clone_scale_x = tk.DoubleVar(value=100.0)
        self.clone_scale_y = tk.DoubleVar(value=100.0)
        self.clone_rotation = tk.DoubleVar(value=0.0)
        self.clone_flip_horizontal = tk.BooleanVar(value=False)
        self.clone_flip_vertical = tk.BooleanVar(value=False)
        self.clone_overlay_visible = tk.BooleanVar(value=True)
        self.clone_overlay_opacity = tk.DoubleVar(value=0.45)
        self.healing_diffusion = tk.IntVar(value=4)
        self.spot_healing_mode = tk.StringVar(value="С учётом содержимого")
        self.patch_structure = tk.IntVar(value=5)
        self.patch_color_adaptation = tk.DoubleVar(value=8.0)
        self.patch_sample_all_layers = tk.BooleanVar(value=False)
        self.gradient_type = tk.StringVar(value="Линейный")
        self.gradient_mode = tk.StringVar(value="Заливка")
        self.gradient_shape = tk.StringVar(value="Прямоугольник")
        self.gradient_object_fill = tk.StringVar(value="Градиент")
        self.gradient_texture = tk.StringVar(value="Шахматная")
        self.gradient_mid_enabled = tk.BooleanVar(value=False)
        self.gradient_mid_position = tk.DoubleVar(value=0.5)
        self.gradient_mid_color = (255, 90, 80, 255)
        self.gradient_definition: dict[str, object] | None = None
        self.crop_aspect = tk.StringVar(value="Свободно")
        self.crop_custom_width = tk.IntVar(value=1920)
        self.crop_custom_height = tk.IntVar(value=1080)
        self.text_font_family = tk.StringVar(value="Arial")
        self.text_size = tk.IntVar(value=48)
        self.text_bold = tk.BooleanVar(value=False)
        self.text_italic = tk.BooleanVar(value=False)
        self.text_underline = tk.BooleanVar(value=False)
        self.text_align = tk.StringVar(value="left")
        self.text_line_spacing = tk.IntVar(value=10)
        self.text_tracking = tk.IntVar(value=0)
        self.text_rotation = tk.DoubleVar(value=0.0)
        self.text_box_width = tk.IntVar(value=0)
        self.text_box_height = tk.IntVar(value=0)
        self.quick_smooth = tk.IntVar(value=1)
        self.quick_edge_radius = tk.IntVar(value=2)
        self.quick_edge_strength = tk.DoubleVar(value=0.5)
        self.grid_visible = tk.BooleanVar(value=False)
        self.grid_spacing = tk.IntVar(value=64)
        self.view_channel = tk.StringVar(value="RGB")
        self.mask_preview = tk.StringVar(value=MASK_PREVIEW_NORMAL)
        self.foreground = (30, 120, 255, 255)
        self.background = (255, 255, 255, 255)
        self.drag_start: tuple[int, int] | None = None
        self.last_point: tuple[int, int] | None = None
        self._move_layer_id: str | None = None
        self._move_start: tuple[int, int] | None = None
        self._move_start_mask: np.ndarray | None = None
        self._move_group_starts: dict[str, tuple[int, int]] = {}
        self._move_group_masks: dict[str, np.ndarray | None] = {}
        self._move_last_bounds: tuple[int, int, int, int] | None = None
        self._move_selection_start: tuple[int, int] | None = None
        self._move_selection_bounds: tuple[int, int, int, int] | None = None
        self._move_selection_delta = (0, 0)
        self._object_resize_handle: str | None = None
        self._object_resize_before: dict | None = None
        self._object_resize_layer_id: str | None = None
        self._object_resize_rendered_bounds: tuple[int, int, int, int] | None = None
        self._last_object_resize_render = 0.0
        self._object_bounds_ids: list[int] = []
        self._stroke_layer_id: str | None = None
        self._stroke_kind = "pixels"
        self._stroke_rect: tuple[int, int, int, int] | None = None
        self._stroke_before: np.ndarray | None = None
        self._stroke_tiles: dict[tuple[int, int], tuple[tuple[int, int, int, int], np.ndarray]] = {}
        self._stroke_selection_mask: np.ndarray | None = None
        self._retouch_stroke: RetouchStroke | None = None
        self._active_brush_stroke: PixelBrushStroke | MaskBrushStroke | None = None
        self._brush_path: BrushPathSampler | None = None
        self._opacity_layer_id: str | None = None
        self._opacity_before: float | None = None
        self._space_down = False
        self._panning = False
        self.selection_id: int | None = None
        self._selection_overlay_ids: list[int] = []
        self._selection_contours: list[np.ndarray] = []
        self._selection_contour_signature: tuple[int, int, int] | None = None
        self._selection_dash_phase = 0
        self._selection_animation_id: str | None = None
        self._drag_preview_ids: list[int] = []
        self._gradient_preview_id: int | None = None
        self._gradient_preview_image: ImageTk.PhotoImage | None = None
        self._last_gradient_preview_at = 0.0
        self._gradient_preview_after_id: str | None = None
        self._gradient_preview_pending: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._crop_box: tuple[int, int, int, int] | None = None
        self._crop_overlay_ids: list[int] = []
        self._crop_drag_handle: str | None = None
        self._crop_drag_origin_box: tuple[int, int, int, int] | None = None
        self._text_editor: tk.Text | None = None
        self._text_editor_window: int | None = None
        self._text_editor_layer_id: str | None = None
        self._text_editor_origin = (0, 0)
        self._text_editor_box_width = 0
        self._text_editor_box_height = 0
        self._text_editor_before: dict[str, object] | None = None
        self._text_property_after_id: str | None = None
        self._text_property_before: dict[str, object] | None = None
        self._loading_text_properties = False
        self._shape_drag_options: dict | None = None
        self.selection_box: tuple[int, int, int, int] | None = None
        self._lasso_points: list[tuple[int, int]] = []
        self._polygon_points: list[tuple[int, int]] = []
        self._polygon_ids: list[int] = []
        self._magnetic_edges: np.ndarray | None = None
        self._quick_points: list[tuple[int, int]] = []
        self._quick_mode = "replace"
        self._quick_preview_mask: np.ndarray | None = None
        self._quick_preview_processed = 0
        self._quick_base_selection: np.ndarray | None = None
        self._quick_preview_id: int | None = None
        self._quick_preview_image: ImageTk.PhotoImage | None = None
        self._last_quick_preview_time = 0.0
        self._brush_preview_ids: list[int] = []
        self._last_pointer_event = None
        self._clone_source: tuple[int, int] | None = None
        self._clone_anchor_target: tuple[int, int] | None = None
        self._clone_anchor_source: tuple[int, int] | None = None
        self._source_anchor = SourceAnchor()
        self._path_overlay_ids: list[int] = []
        self._path_selected_nodes: set[int] = set()
        self._path_drag_target: tuple[int, str] | None = None
        self._path_drag_before: dict[str, object] | None = None
        self._path_drag_origin = None
        self._clone_sample_pixels: np.ndarray | None = None
        self._clone_sample_origin = (0, 0)
        self._clone_source_marker_ids: list[int] = []
        self._source_retouch_stroke = None
        self._spot_last_point: tuple[int, int] | None = None
        self._clone_overlay_id: int | None = None
        self._clone_overlay_image: ImageTk.PhotoImage | None = None
        self._clone_source_dialog: tk.Toplevel | None = None
        self.selected_layer_ids: set[str] = {self.doc.layer.id}
        self._pixel_clipboard: np.ndarray | None = None
        self._pixel_clipboard_origin = (0, 0)
        self._patch_start_bounds: tuple[int, int, int, int] | None = None
        self._patch_preview_id: int | None = None
        self._patch_image_id: int | None = None
        self._patch_preview_image: ImageTk.PhotoImage | None = None
        self._patch_pending_bounds: tuple[int, int, int, int] | None = None
        self._patch_sample_pixels: np.ndarray | None = None
        self._patch_sample_origin = (0, 0)
        self._last_patch_preview_time = 0.0
        self._guide_doc_lines: list[tuple[str, int]] = []
        self._overlay_ids: list[int] = []
        self._preview_image: ImageTk.PhotoImage | None = None
        self._layer_thumb_image: ImageTk.PhotoImage | None = None
        self._mask_thumb_image: ImageTk.PhotoImage | None = None
        self._select_mask_preview_image: ImageTk.PhotoImage | None = None
        self._mask_edge_preview_image: ImageTk.PhotoImage | None = None
        self._filter_stack_preview_image: ImageTk.PhotoImage | None = None
        self._adjustment_preview_image: ImageTk.PhotoImage | None = None
        self._transform_preview_image: ImageTk.PhotoImage | None = None
        self._warp_preview_image: ImageTk.PhotoImage | None = None
        self._bezier_preview_image: ImageTk.PhotoImage | None = None
        self._text_path_preview_image: ImageTk.PhotoImage | None = None
        self._frequency_preview_images: list[ImageTk.PhotoImage] = []
        self._portrait_preview_images: list[ImageTk.PhotoImage] = []
        self._startup_clipboard_preview: ImageTk.PhotoImage | None = None
        self._new_document_preview: ImageTk.PhotoImage | None = None
        self.startup_frame: tk.Frame | None = None
        self._editor_active = False
        self._canvas_image_id: int | None = None
        self._canvas_tile_ids: dict[tuple[int, int], int] = {}
        self._canvas_tile_images: dict[tuple[int, int], ImageTk.PhotoImage] = {}
        self._canvas_view_signature: tuple[object, ...] | None = None
        self._layer_thumbnail_cache: dict[tuple[object, ...], Image.Image] = {}
        self._mask_thumbnail_cache: dict[tuple[object, ...], Image.Image] = {}
        self._canvas_origin = (0, 0)
        self._render_after_id: str | None = None
        self._initial_fit_after_id: str | None = None
        self._performing_initial_fit = False
        self._last_render_time = 0.0
        self._composite_cache = None
        self._composite_dirty = True
        self._view_dirty = True
        self.render_engine = RenderEngine(tile_size=256)

        self._build_ui()
        self.tool.trace_add("write", self.tool_changed)
        self.brush_size.trace_add("write", self.brush_size_changed)
        self.tolerance.trace_add("write", self.quick_preview_settings_changed)
        self.quick_smooth.trace_add("write", self.quick_preview_settings_changed)
        self.quick_edge_radius.trace_add("write", self.quick_preview_settings_changed)
        self.quick_edge_strength.trace_add("write", self.quick_preview_settings_changed)
        self.selection_mode.trace_add("write", self.selection_mode_changed)
        for variable in (
            self.clone_source_x,
            self.clone_source_y,
            self.clone_offset_x,
            self.clone_offset_y,
            self.clone_scale_x,
            self.clone_scale_y,
            self.clone_rotation,
            self.clone_flip_horizontal,
            self.clone_flip_vertical,
            self.clone_overlay_visible,
            self.clone_overlay_opacity,
            self.clone_sampling,
        ):
            variable.trace_add("write", self.clone_source_settings_changed)
        for variable in (self.patch_structure, self.patch_color_adaptation, self.patch_sample_all_layers):
            variable.trace_add("write", self.patch_preview_settings_changed)
        for variable in (
            self.text_font_family, self.text_size, self.text_bold, self.text_italic,
            self.text_underline, self.text_align, self.text_line_spacing,
            self.text_tracking, self.text_rotation, self.text_box_width, self.text_box_height,
        ):
            variable.trace_add("write", self.text_properties_changed)
        self.load_settings()
        self.refresh_recent_menu()
        self.refresh()
        self.show_start_screen()
        self.deiconify()
        self.schedule_autosave()
    def destroy(self) -> None:
        from ..local_generative import shutdown_local_servers
        if self._selection_animation_id is not None:
            try:
                self.after_cancel(self._selection_animation_id)
            except tk.TclError:
                pass
            self._selection_animation_id = None
        if self._initial_fit_after_id is not None:
            try:
                self.after_cancel(self._initial_fit_after_id)
            except tk.TclError:
                pass
            self._initial_fit_after_id = None
        self.autosave_recovery()
        self.save_settings()
        self.executor.shutdown(wait=False, cancel_futures=True)
        shutdown_local_servers()
        self.render_engine.scratch.close()
        super().destroy()

    def load_settings(self) -> None:
        try:
            if self.settings_path.exists():
                data = json.loads(self.settings_path.read_text(encoding="utf-8"))
                gpu_mode = str(data.get("gpu_mode", os.environ.get("UZYRO_GPU", "auto"))).lower()
                os.environ["UZYRO_GPU"] = gpu_mode if gpu_mode in {"auto", "force", "off"} else "auto"
                reset_acceleration_metrics()
                self.render_engine.gpu = gpu_status()
                self.render_engine.mipmaps.gpu = dict(self.render_engine.gpu)
                self.recent_files = [str(path) for path in data.get("recent_files", []) if Path(path).exists()]
                self.tool_order = normalize_tool_order(data.get("tool_order"), TOOL_DEFINITIONS)
                self.visible_tools = normalize_visible_tools(data.get("visible_tools"), self.tool_order)
                if int(data.get("tool_schema_version", 1)) < 2 and "spot_healing" not in self.visible_tools:
                    self.visible_tools.append("spot_healing")
                if int(data.get("tool_schema_version", 1)) < 3:
                    for tool_id in ("path_select", "direct_select", "add_anchor", "delete_anchor", "convert_anchor"):
                        if tool_id not in self.visible_tools:
                            self.visible_tools.append(tool_id)
                self.tool_pane_position = int(data.get("tool_pane_position", self.tool_pane_position))
                saved_tool_settings = data.get("tool_settings", {})
                if isinstance(saved_tool_settings, dict):
                    for tool_id, defaults in TOOL_SETTINGS_DEFAULTS.items():
                        saved = saved_tool_settings.get(tool_id)
                        if isinstance(saved, dict):
                            self.tool_settings[tool_id] = {**defaults, **{key: saved[key] for key in defaults if key in saved}}
                saved_brush_presets = data.get("brush_presets", {})
                if isinstance(saved_brush_presets, dict):
                    for name, values in saved_brush_presets.items():
                        if isinstance(name, str) and name.strip() and isinstance(values, dict):
                            self.brush_presets[name.strip()] = dict(values)
                self.brush_advanced.update(normalize_brush_config(data.get("brush_advanced")))
                if is_test_polluted_brush_settings(self.tool_settings.get("brush"), self.brush_advanced):
                    self.tool_settings["brush"] = copy.deepcopy(TOOL_SETTINGS_DEFAULTS["brush"])
                    self.brush_advanced.clear()
                    self.brush_advanced.update(default_brush_config())
                elif int(data.get("brush_default_version", 1)) < 2:
                    brush = self.tool_settings.get("brush", {})
                    old_defaults = {
                        "hardness": 0.5,
                        "spacing": 0.25,
                        "opacity": 1.0,
                        "flow": 1.0,
                        "blend_mode": "Normal",
                    }
                    if all(brush.get(key) == value for key, value in old_defaults.items()):
                        brush.update(hardness=1.0, spacing=0.0)
                    round_brush = self.brush_presets.get("Круглая кисть", {})
                    if round_brush.get("hardness") == 0.8 and round_brush.get("spacing") == 0.25:
                        round_brush.update(hardness=1.0, spacing=0.0)
                if int(data.get("brush_default_version", 1)) < 3:
                    for tool_id, defaults in TOOL_SETTINGS_DEFAULTS.items():
                        settings = self.tool_settings.get(tool_id, {})
                        if "hardness" in defaults:
                            settings["hardness"] = 1.0
                        if "spacing" in defaults:
                            settings["spacing"] = 0.0
                shape_settings = data.get("shape_settings", {})
                if isinstance(shape_settings, dict):
                    self.shape_stroke_width.set(max(0, min(100, int(shape_settings.get("stroke_width", 2)))))
                    self.shape_from_center.set(bool(shape_settings.get("from_center", False)))
                    self.polygon_sides.set(max(3, min(64, int(shape_settings.get("polygon_sides", 5)))))
                    self.star_points_count.set(max(3, min(64, int(shape_settings.get("star_points", 5)))))
                    self.star_inner_ratio.set(float(np.clip(shape_settings.get("star_inner_ratio", 0.5), 0.05, 0.95)))
                    preset = str(shape_settings.get("custom_shape_preset", self.custom_shape_preset.get()))
                    if preset in CUSTOM_SHAPE_PRESETS:
                        self.custom_shape_preset.set(preset)
                source_settings = data.get("source_retouch", {})
                if isinstance(source_settings, dict):
                    self.load_source_retouch_settings(source_settings)
                custom_canvas = data.get("custom_canvas", {})
                self.custom_canvas_width = max(1, min(50000, int(custom_canvas.get("width", self.custom_canvas_width))))
                self.custom_canvas_height = max(1, min(50000, int(custom_canvas.get("height", self.custom_canvas_height))))
                self.custom_canvas_dpi = max(1, min(2400, int(custom_canvas.get("dpi", self.custom_canvas_dpi))))
                saved_background = str(custom_canvas.get("background", self.custom_canvas_background))
                if saved_background in DOCUMENT_BACKGROUNDS:
                    self.custom_canvas_background = saved_background
                generative = data.get("generative", {})
                if isinstance(generative, dict):
                    self.generative_settings = {
                        "provider": "local",
                        "local_model_id": str(generative.get("local_model_id", "realistic-vision-51-inpaint")),
                        "local_backend": str(generative.get("local_backend", "auto")),
                        "variants": max(1, min(4, int(generative.get("variants", 3)))),
                        "style": str(generative.get("style", "photographic")),
                        "creativity": float(np.clip(generative.get("creativity", 0.5), 0.0, 1.0)),
                        "performance_profile": str(generative.get(
                            "performance_profile", "balanced" if generative.get("sampler") == "LCM" else "quality",
                        )),
                        "steps": max(4, min(80, int(generative.get("steps", 22)))),
                        "cfg_scale": float(np.clip(generative.get("cfg_scale", 7.0), 1.0, 20.0)),
                        "strength": float(np.clip(generative.get("strength", 0.88), 0.05, 1.0)),
                        "sampler": str(generative.get("sampler", "DPM++ 2M")),
                    }
                if self.tool.get() not in self.tool_order:
                    self.tool.set(self.visible_tools[0])
                self.load_active_tool_settings(self.tool.get())
                if hasattr(self, "tool_palette"):
                    self.tool_palette.set_configuration(self.tool_order, self.visible_tools)
                if hasattr(self, "tool_split"):
                    self.after_idle(self.apply_tool_pane_position)
                self.refresh_tool_menu()
        except Exception:
            self.recent_files = []
            self.tool_order = normalize_tool_order(None, TOOL_DEFINITIONS)
            self.visible_tools = normalize_visible_tools(None, self.tool_order)
    def save_settings(self) -> None:
        try:
            self.app_data_dir.mkdir(parents=True, exist_ok=True)
            self.capture_tool_pane_position()
            self.save_active_tool_settings()
            self.settings_path.write_text(
                json.dumps(
                    {
                        "recent_files": self.recent_files[:12],
                        "tool_order": self.tool_order,
                        "visible_tools": self.visible_tools,
                        "tool_schema_version": 3,
                        "gpu_mode": os.environ.get("UZYRO_GPU", "auto"),
                        "tool_pane_position": self.tool_pane_position,
                        "tool_settings": self.tool_settings,
                        "brush_presets": self.brush_presets,
                        "brush_advanced": self.brush_advanced,
                        "brush_default_version": 3,
                        "shape_settings": {
                            "stroke_width": int(self.shape_stroke_width.get()),
                            "from_center": bool(self.shape_from_center.get()),
                            "polygon_sides": int(self.polygon_sides.get()),
                            "star_points": int(self.star_points_count.get()),
                            "star_inner_ratio": float(self.star_inner_ratio.get()),
                            "custom_shape_preset": self.custom_shape_preset.get(),
                        },
                        "source_retouch": self.source_retouch_settings_payload(),
                        "custom_canvas": {
                            "width": self.custom_canvas_width,
                            "height": self.custom_canvas_height,
                            "dpi": self.custom_canvas_dpi,
                            "background": self.custom_canvas_background,
                        },
                        "generative": self.generative_settings,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass
    def add_recent_file(self, path: str) -> None:
        resolved = str(Path(path))
        self.recent_files = [item for item in self.recent_files if item.lower() != resolved.lower()]
        self.recent_files.insert(0, resolved)
        self.recent_files = self.recent_files[:12]
        self.save_settings()
        self.refresh_recent_menu()
    def refresh_recent_menu(self) -> None:
        if not hasattr(self, "recent_menu"):
            return
        self.recent_menu.delete(0, tk.END)
        if not self.recent_files:
            self.recent_menu.add_command(label="Нет недавних файлов", state=tk.DISABLED)
            self.refresh_startup_recent()
            return
        for path in self.recent_files:
            label = Path(path).name
            self.recent_menu.add_command(label=label, command=lambda p=path: self.open_path(p))
        self.recent_menu.add_separator()
        self.recent_menu.add_command(label="Очистить список", command=self.clear_recent_files)
        self.refresh_startup_recent()
    def clear_recent_files(self) -> None:
        self.recent_files.clear()
        self.save_settings()
        self.refresh_recent_menu()
