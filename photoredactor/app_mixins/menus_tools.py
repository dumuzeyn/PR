from __future__ import annotations

from ..app_shared import *


class MenusToolsMixin:
    def _build_menu(self) -> None:
        menu = tk.Menu(self)
        self.editor_menu = menu
        self.config(menu=menu)

        file_menu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Файл", menu=file_menu)
        self.file_menu = file_menu
        file_menu.add_command(label="Новый холст", command=self.new_document, accelerator=accelerator("new_document"))
        file_menu.add_command(label="Открыть изображение/проект", command=self.open_file, accelerator=accelerator("open"))
        self.recent_menu = tk.Menu(file_menu, tearoff=False)
        file_menu.add_cascade(label="Недавние файлы", menu=self.recent_menu)
        file_menu.add_command(label="Поместить встроенное", command=self.place_embedded)
        file_menu.add_command(label="Поместить связанное", command=self.place_linked)
        file_menu.add_command(label="Загрузить файлы как слои", command=self.load_files_as_layers)
        file_menu.add_separator()
        file_menu.add_command(label="Сохранить проект", command=self.save, accelerator=accelerator("save"))
        file_menu.add_command(label="Сохранить проект как", command=self.save_as_project, accelerator=accelerator("save_as"))
        file_menu.add_command(label="Экспорт изображения", command=self.export_image)
        file_menu.add_command(label="Экспорт слоев", command=self.export_layers)
        file_menu.add_separator()
        file_menu.add_command(label="Открыть восстановление", command=self.open_recovery)
        file_menu.add_command(label="Очистить восстановление", command=self.clear_recovery)
        file_menu.add_separator()
        file_menu.add_command(label="Пакетный размер/конвертация", command=self.batch_process)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.destroy)

        edit = tk.Menu(menu, tearoff=False)
        self.edit_menu = edit
        menu.add_cascade(label="Правка", menu=edit)
        edit.add_command(label="Отменить", command=self.undo, accelerator=accelerator("undo"))
        edit.add_command(label="Повторить", command=self.redo, accelerator=accelerator("redo"))
        edit.add_separator()
        edit.add_command(label="Вырезать", command=self.shortcut_cut, accelerator=accelerator("cut"))
        edit.add_command(label="Копировать", command=self.shortcut_copy, accelerator=accelerator("copy"))
        edit.add_command(label="Вставить", command=self.shortcut_paste, accelerator=accelerator("paste"))
        edit.add_command(label="Удалить выбранные пиксели", command=self.shortcut_delete, accelerator="Delete")
        edit.add_separator()
        edit.add_command(label="Снять выделение", command=self.clear_selection, accelerator=accelerator("deselect"))

        self.tools_menu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Инструменты", menu=self.tools_menu)
        self.refresh_tool_menu()

        select = tk.Menu(menu, tearoff=False)
        self.select_menu = select
        menu.add_cascade(label="Выделение", menu=select)
        select.add_command(label="Выделить все", command=self.select_all, accelerator=accelerator("select_all"))
        select.add_command(label="Инвертировать выделение", command=self.invert_selection, accelerator=accelerator("invert_selection"))
        select.add_command(label="Снять выделение", command=self.clear_selection, accelerator=accelerator("deselect"))
        select.add_separator()
        select.add_command(label="Выделить непрозрачные пиксели", command=self.select_opaque_pixels)
        select.add_command(label="Выделить объект", command=self.select_subject)
        select.add_command(label="Выделить фон", command=self.select_background)
        select.add_command(label="Выделить небо", command=self.select_sky)
        select.add_command(label="Автоматическое выделение...", command=self.automatic_selection_workspace)
        select.add_command(label="Одна строка", command=self.single_row_selection)
        select.add_command(label="Один столбец", command=self.single_column_selection)
        select.add_separator()
        select.add_command(label="Растушевка", command=self.feather_selection)
        select.add_command(label="Сгладить", command=self.smooth_selection)
        select.add_command(label="Расширить", command=self.grow_selection)
        select.add_command(label="Сжать", command=self.shrink_selection)
        select.add_command(label="Граница", command=self.border_selection)
        select.add_command(label="Уточнить край", command=self.refine_selection)
        select.add_command(label="Умная очистка края", command=self.cleanup_selection_edges)
        select.add_command(label="Коррекция края по уверенности", command=self.correct_selection_edges)
        select.add_command(label="Выделить и маска", command=self.select_and_mask_workspace)
        select.add_separator()
        select.add_command(label="Сохранить выделение", command=self.save_selection)
        select.add_command(label="Загрузить выделение", command=self.load_selection)
        select.add_command(label="Удалить сохраненное выделение", command=self.delete_saved_selection)

        image = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Изображение", menu=image)
        image.add_command(label="Размер изображения", command=self.resize_image)
        image.add_command(label="Размер холста", command=self.resize_canvas)
        image.add_command(label="Генеративное расширение холста", command=self.generative_expand_dialog)
        image.add_command(label="Обрезать по выделению", command=self.crop_to_selection)
        image.add_command(label="Обрезать прозрачные пиксели", command=self.trim_transparent)
        image.add_command(label="Показать все слои", command=self.reveal_all)
        image.add_separator()
        image.add_command(label="Повернуть на 90 по часовой", command=lambda: self.rotate(90))
        image.add_command(label="Повернуть на 180", command=lambda: self.rotate(180))
        image.add_command(label="Отразить горизонтально", command=lambda: self.flip(horizontal=True))
        image.add_command(label="Отразить вертикально", command=lambda: self.flip(horizontal=False))

        color_menu = tk.Menu(image, tearoff=False)
        image.add_separator()
        image.add_cascade(label="Управление цветом", menu=color_menu)
        color_menu.add_command(label="Назначить ICC-профиль", command=self.assign_icc_profile)
        color_menu.add_command(label="Преобразовать в ICC-профиль", command=self.convert_icc_profile)
        color_menu.add_separator()
        color_menu.add_command(label="Цветопроба и подготовка к печати", command=self.color_proof_workspace)
        color_menu.add_command(label="Включить / выключить цветопробу", command=self.toggle_soft_proof)
        model_menu = tk.Menu(color_menu, tearoff=False)
        color_menu.add_cascade(label="Цветовая модель", menu=model_menu)
        for model in COLOR_MODELS:
            model_menu.add_command(label=model, command=lambda value=model: self.change_color_model(value))
        depth_menu = tk.Menu(color_menu, tearoff=False)
        color_menu.add_cascade(label="Глубина каналов", menu=depth_menu)
        for depth in BIT_DEPTHS:
            depth_menu.add_command(label=f"{depth} бит", command=lambda value=depth: self.change_bit_depth(value))

        layer = tk.Menu(menu, tearoff=False)
        self.layer_menu = layer
        menu.add_cascade(label="Слой", menu=layer)
        layer.add_command(label="Новый слой", command=self.new_layer, accelerator=accelerator("new_layer"))
        layer.add_command(label="Дублировать слой", command=self.duplicate_layer, accelerator=accelerator("duplicate_layer"))
        layer.add_command(label="Удалить слой", command=self.delete_layer)
        layer.add_command(label="Переименовать слой", command=self.rename_layer)
        layer.add_command(label="Заблокировать/разблокировать", command=self.toggle_layer_lock)
        layer.add_command(label="Редактировать текстовый слой", command=self.edit_text_layer)
        layer.add_command(label="Редактировать контур текста", command=self.edit_text_path)
        layer.add_command(label="Трансформировать текстовый блок", command=self.free_transform_layer)
        layer.add_command(label="Редактировать фигуру", command=self.edit_shape_layer)
        layer.add_command(label="Редактировать точки Безье", command=self.edit_bezier_points)
        layer.add_command(label="Булева операция фигур", command=self.boolean_shape_layers)
        layer.add_command(label="\u0420\u0435\u0434\u0430\u043a\u0442\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u043a\u043e\u0440\u0440\u0435\u043a\u0442\u0438\u0440\u0443\u044e\u0449\u0438\u0439 \u0441\u043b\u043e\u0439", command=self.edit_adjustment_layer)
        layer.add_separator()
        layer.add_command(label="Поднять выше", command=lambda: self.move_layer(1))
        layer.add_command(label="Опустить ниже", command=lambda: self.move_layer(-1))
        layer.add_command(label="Свободная трансформация", command=self.free_transform_layer, accelerator=accelerator("free_transform"))
        layer.add_command(label="Трансформировать выделенные пиксели", command=self.transform_selected_pixels)
        layer.add_command(label="Перспективная трансформация", command=self.perspective_transform_layer)
        layer.add_command(label="Деформация слоя", command=self.warp_layer)
        layer.add_command(label="Обновить связанный слой", command=self.update_linked_layer)
        layer.add_command(label="Перелинковать слой", command=self.relink_layer)
        smart_menu = tk.Menu(layer, tearoff=False)
        layer.add_cascade(label="Smart Object", menu=smart_menu)
        smart_menu.add_command(label="Преобразовать выбранные слои", command=self.convert_to_smart_object)
        smart_menu.add_command(label="Редактировать содержимое", command=self.edit_smart_object_contents)
        smart_menu.add_separator()
        smart_menu.add_command(label="Показать статус связи", command=self.show_linked_layer_status)
        smart_menu.add_command(label="Разрешить конфликт связи", command=self.resolve_linked_conflict_dialog)
        smart_menu.add_command(label="Заменить содержимое", command=self.replace_smart_contents)
        smart_menu.add_command(label="Редактировать фильтры", command=self.edit_layer_filters)
        smart_menu.add_command(label="Преобразовать во встроенный", command=self.convert_smart_to_embedded)
        smart_menu.add_command(label="Сбросить трансформацию", command=self.reset_smart_transform)
        layer.add_command(label="Переключить обтравочную маску", command=self.toggle_clipping_mask)
        layer.add_command(label="Стили слоя", command=self.edit_layer_styles)
        layer.add_command(label="Фильтры слоя", command=self.edit_layer_filters)
        layer.add_command(label="Очистить фильтры слоя", command=self.clear_layer_filters)
        layer.add_command(label="Объединить с нижним", command=self.merge_down, accelerator=accelerator("merge_down"))
        layer.add_command(label="Свести изображение", command=self.flatten, accelerator=accelerator("flatten"))
        layer.add_separator()
        layer.add_command(label="Редактировать маску как канал", command=self.edit_active_mask_channel)
        layer.add_command(label="Добавить маску из выделения", command=self.add_mask_from_selection)
        layer.add_command(label="Добавить белую маску", command=self.add_reveal_all_mask)
        layer.add_command(label="Добавить черную маску", command=self.add_hide_all_mask)
        layer.add_command(label="Инвертировать маску", command=self.invert_layer_mask)
        layer.add_command(label="Включить/выключить маску", command=self.toggle_layer_mask)
        layer.add_command(label="Связать/отвязать маску", command=self.toggle_layer_mask_link)
        layer.add_command(label="Плотность маски", command=self.set_mask_density)
        layer.add_command(label="Растушевка маски", command=self.set_mask_feather)
        layer.add_command(label="Уточнить край маски", command=self.refine_layer_mask)
        layer.add_command(label="Применить маску", command=self.apply_layer_mask)
        layer.add_command(label="Удалить маску", command=self.delete_layer_mask)

        adj = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Коррекция", menu=adj)
        adj.add_command(label="Яркость/контраст", command=self.adjust_brightness_contrast)
        adj.add_command(label="Насыщенность", command=self.adjust_saturation)
        adj.add_command(label="Тон/Насыщенность", command=self.adjust_hue_saturation)
        adj.add_command(label="Экспозиция", command=self.adjust_exposure)
        adj.add_command(label="Цветовой баланс", command=self.adjust_color_balance)
        adj.add_command(label="Уровни", command=self.adjust_levels)
        adj.add_command(label="Кривые", command=self.adjust_curves)
        adj.add_command(label="Порог", command=self.adjust_threshold)
        adj.add_command(label="Постеризация", command=self.adjust_posterize)
        adj.add_command(label="Инверсия", command=self.adjust_invert)
        adj.add_command(label="Черно-белое", command=self.adjust_grayscale)
        adj.add_separator()
        adj.add_command(label="Добавить корректирующий слой", command=self.add_adjustment_layer)
        adj.add_command(label="\u0420\u0435\u0434\u0430\u043a\u0442\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u043a\u043e\u0440\u0440\u0435\u043a\u0442\u0438\u0440\u0443\u044e\u0449\u0438\u0439 \u0441\u043b\u043e\u0439", command=self.edit_adjustment_layer)

        filters = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Фильтр", menu=filters)
        filters.add_command(label="Размытие по Гауссу", command=self.filter_blur)
        filters.add_command(label="Резкость", command=self.filter_sharpen)
        filters.add_command(label="Шум", command=self.filter_noise)
        filters.add_separator()
        filters.add_command(label="Заливка с учетом содержимого", command=self.filter_content_aware_fill)
        filters.add_command(label="Очистка краев выделения", command=self.filter_edge_cleanup)
        filters.add_command(label="Удаление красных глаз", command=self.filter_red_eye)
        filters.add_command(label="Заплатка из источника", command=self.filter_patch_selection)
        self.plugin_filters_menu = tk.Menu(filters, tearoff=False, postcommand=self.refresh_plugin_filter_menu)
        filters.add_separator()
        filters.add_cascade(label="Плагины", menu=self.plugin_filters_menu)

        retouch = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Ретушь", menu=retouch)
        retouch.add_command(label="Частотное разложение", command=self.frequency_separation_layers)
        retouch.add_command(label="Портретная обработка", command=self.portrait_cleanup_layer)
        retouch.add_separator()
        retouch.add_command(label="Выбрать точечное восстановление", command=lambda: self.tool.set("spot_healing"))
        retouch.add_command(label="Удаление красных глаз", command=self.filter_red_eye)
        retouch.add_command(label="Заплатка из источника", command=self.filter_patch_selection)

        analysis = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Анализ", menu=analysis)
        analysis.add_command(label="Статистика изображения", command=self.show_image_statistics)
        analysis.add_command(label="Гистограмма", command=self.show_histogram)
        analysis.add_command(label="Метаданные / EXIF", command=self.show_metadata)
        analysis.add_command(label="Редактировать метаданные", command=self.edit_metadata)
        analysis.add_command(label="Состояние кэша и GPU", command=self.show_cache_status)
        analysis.add_command(label="Настройка и тест GPU", command=self.gpu_acceleration_dialog)

        actions = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Действия", menu=actions)
        actions.add_command(label="Начать запись", command=self.start_action_recording)
        actions.add_command(label="Остановить запись", command=self.stop_action_recording)
        actions.add_command(label="Сохранить запись", command=self.save_action_recording)
        actions.add_command(label="Очистить запись", command=self.clear_action_recording)
        actions.add_separator()
        actions.add_command(label="Выполнить действие...", command=self.run_action_file)
        actions.add_command(label="Пакетно выполнить действие...", command=self.batch_action_file)
        actions.add_separator()
        actions.add_command(label="Перезагрузить плагины", command=self.reload_plugins)
        actions.add_command(label="Ошибки плагинов", command=self.show_plugin_errors)

        view = tk.Menu(menu, tearoff=False)
        self.view_menu = view
        menu.add_cascade(label="Вид", menu=view)
        view.add_command(label="Увеличить", command=lambda: self.set_zoom(self.zoom.get() * 1.25), accelerator="+")
        view.add_command(label="Уменьшить", command=lambda: self.set_zoom(self.zoom.get() / 1.25), accelerator="-")
        view.add_command(label="100%", command=lambda: self.set_zoom(1.0), accelerator=accelerator("actual_size"))
        view.add_command(label="По размеру окна", command=self.fit_to_screen, accelerator=accelerator("fit_to_screen"))
        view.add_separator()
        channel = tk.Menu(view, tearoff=False)
        view.add_cascade(label="Канал", menu=channel)
        for label, name in [("RGB", "RGB"), ("Красный", "Red"), ("Зеленый", "Green"), ("Синий", "Blue"), ("Альфа", "Alpha")]:
            channel.add_radiobutton(label=label, value=name, variable=self.view_channel, command=self.set_view_channel)
        mask_view = tk.Menu(view, tearoff=False)
        view.add_cascade(label="Просмотр маски", menu=mask_view)
        for label in MASK_PREVIEW_MODES:
            mask_view.add_radiobutton(label=label, value=label, variable=self.mask_preview, command=self.set_mask_preview)
        view.add_separator()
        view.add_checkbutton(label="Сетка", variable=self.grid_visible, command=self.refresh_canvas)
        view.add_command(label="Шаг сетки", command=self.set_grid_spacing)
        view.add_command(label="Добавить горизонтальную направляющую", command=self.add_horizontal_guide)
        view.add_command(label="Добавить вертикальную направляющую", command=self.add_vertical_guide)
        view.add_command(label="Очистить направляющие", command=self.clear_guides)

        view.add_separator()
        view.add_command(label="Настроить панель инструментов...", command=self.configure_tool_palette)

    def _build_tools(self, parent: ttk.Frame) -> None:
        parent.configure(width=188)
        parent.pack_propagate(False)
        self.tool_palette = ToolPalette(
            parent,
            definitions=TOOL_DEFINITIONS,
            tool_var=self.tool,
            order=self.tool_order,
            visible=self.visible_tools,
            select_tool=self.select_tool,
            configure_tools=self.configure_tool_palette,
            tooltip_factory=ToolTip,
        )
        self.tool_palette.pack(fill=tk.BOTH, expand=True)

    def _build_tool_options(self, parent: ttk.Frame) -> None:
        self.tool_options_panel = ToolOptionsPanel(
            parent,
            tool_var=self.tool,
            definitions=TOOL_DEFINITIONS,
            brush_size=self.brush_size,
            opacity=self.opacity,
            hardness=self.hardness,
            retouch_strength=self.retouch_strength,
            exposure=self.exposure,
            tonal_range=self.tonal_range,
            tolerance=self.tolerance,
            color_range_sample_hex=self.color_range_sample_hex,
            selection_mode=self.selection_mode,
            quick_smooth=self.quick_smooth,
            quick_edge_radius=self.quick_edge_radius,
            quick_edge_strength=self.quick_edge_strength,
            paint_target=self.paint_target,
            retouch_preset=self.retouch_preset,
            retouch_presets=RETOUCH_PRESETS,
            pick_foreground=self.pick_foreground,
            pick_background=self.pick_background,
            set_paint_target=self.set_paint_target,
            apply_retouch_preset=self.apply_retouch_preset,
            shape_stroke_width=self.shape_stroke_width,
            polygon_sides=self.polygon_sides,
            star_points=self.star_points_count,
            star_inner_ratio=self.star_inner_ratio,
            custom_shape_preset=self.custom_shape_preset,
            custom_shape_presets=list(CUSTOM_SHAPE_PRESETS),
            selection_feather=self.selection_feather,
            selection_antialias=self.selection_antialias,
            magic_contiguous=self.magic_contiguous,
            clone_aligned=self.clone_aligned,
            clone_sampling=self.clone_sampling,
            gradient_type=self.gradient_type,
            gradient_mode=self.gradient_mode,
            gradient_shape=self.gradient_shape,
            gradient_object_fill=self.gradient_object_fill,
            gradient_texture=self.gradient_texture,
            gradient_mid_enabled=self.gradient_mid_enabled,
            gradient_mid_position=self.gradient_mid_position,
            pick_gradient_mid=self.pick_gradient_mid,
            crop_aspect=self.crop_aspect,
            crop_custom_width=self.crop_custom_width,
            crop_custom_height=self.crop_custom_height,
            text_font_family=self.text_font_family,
            text_size=self.text_size,
            text_bold=self.text_bold,
            text_italic=self.text_italic,
            text_underline=self.text_underline,
            text_align=self.text_align,
            text_line_spacing=self.text_line_spacing,
            text_tracking=self.text_tracking,
            text_rotation=self.text_rotation,
            text_box_width=self.text_box_width,
            finish_text_edit=self.finish_text_edit,
            edit_active_text=self.edit_active_text_on_canvas,
            edit_text_path=self.edit_text_path,
            tooltip_factory=ToolTip,
            compact=True,
            auto_select=self.auto_select,
            color_provider=lambda: (self.foreground, self.background),
        )
        self.tool_options_panel.pack(fill=tk.BOTH, expand=True)
