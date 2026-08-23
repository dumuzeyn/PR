from __future__ import annotations

from .tool_palette_shared import *
from .menu_bar import DarkMenuButton


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
        self._images: dict[str, tk.PhotoImage] = {}
        self._group_choice = {name: values[0] for name, values in COMBINED_TOOL_GROUPS.items()}
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=8, pady=(8, 5))
        ttk.Label(header, text="Инструменты", style="PanelTitle.TLabel").pack(side=tk.LEFT)
        settings_image = action_icon(self, "settings", 15, TOKENS.TEXT_SECONDARY)
        self._images["settings"] = settings_image
        settings = ttk.Button(
            header,
            image=settings_image,
            text="Настроить",
            compound=tk.LEFT,
            command=configure_tools,
            style="Quiet.TButton",
        )
        settings.pack(side=tk.RIGHT)
        if self.tooltip_factory is not None:
            self.tooltip_factory(settings, "Настроить панель инструментов")
        self.scroller = ScrollableFrame(self, height=280)
        self.scroller.pack(fill=tk.BOTH, expand=True, padx=(4, 2), pady=(0, 5))
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
        settings_image = self._images.get("settings")
        self._images = {"settings": settings_image} if settings_image is not None else {}
        by_id = {value: (label, value, description) for label, value, description in self.definitions}
        group_by_tool = {value: group for group, values in COMBINED_TOOL_GROUPS.items() for value in values}
        consumed_groups: set[str] = set()
        previous_group: str | None = None
        for value in self.order:
            if value not in self.visible or value not in by_id:
                continue
            combined_group = group_by_tool.get(value)
            if combined_group is not None:
                if combined_group in consumed_groups:
                    continue
                consumed_groups.add(combined_group)
                variants = [item for item in COMBINED_TOOL_GROUPS[combined_group] if item in self.visible and item in by_id]
                if len(variants) > 1:
                    self._render_combined_tool(combined_group, variants, by_id, previous_group)
                    previous_group = combined_group
                    continue
            label, _value, description = by_id[value]
            group = TOOL_GROUPS.get(value, value)
            top_pad = 7 if previous_group is not None and group != previous_group else 1
            previous_group = group
            image = tool_icon(self, value, TOKENS.ICON_SIZE)
            self._images[value] = image
            button = ttk.Radiobutton(
                self.scroller.content,
                image=image,
                text=label,
                compound=tk.LEFT,
                value=value,
                variable=self.tool_var,
                command=lambda v=value: self.select_tool(v),
                style="Tool.TRadiobutton",
                takefocus=True,
            )
            button.pack(fill=tk.X, padx=3, pady=(top_pad, 1))
            if self.tooltip_factory is not None:
                shortcut = SHORTCUTS.get(value)
                title = f"{label} ({shortcut})" if shortcut else label
                self.tooltip_factory(button, f"{title}\n{description}", demo=value)
            self.buttons[value] = button

    def _render_combined_tool(self, group: str, variants: list[str], by_id: dict, previous_group: str | None) -> None:
        active = self.tool_var.get()
        if active in variants:
            self._group_choice[group] = active
        selected = self._group_choice.get(group, variants[0])
        if selected not in variants:
            selected = variants[0]
            self._group_choice[group] = selected
        label, _value, description = by_id[selected]
        row = ttk.Frame(self.scroller.content)
        row.pack(fill=tk.X, padx=3, pady=(7 if previous_group is not None else 1, 1))
        image = tool_icon(self, selected, TOKENS.ICON_SIZE)
        self._images[f"group:{group}"] = image
        button = ttk.Radiobutton(
            row, image=image, text=label, compound=tk.LEFT, value=selected,
            variable=self.tool_var, command=lambda value=selected: self.select_tool(value),
            style="Tool.TRadiobutton", takefocus=True,
        )
        picker = DarkMenuButton(row, text="", width=2)
        menu = tk.Menu(picker, tearoff=False)
        for value in variants:
            item_label, _item, item_description = by_id[value]
            menu.add_radiobutton(
                label=item_label, value=value, variable=self.tool_var,
                command=lambda chosen=value, group_name=group: self._choose_group_tool(group_name, chosen),
            )
            self.buttons[value] = button
        picker.configure(menu=menu)
        picker.pack(side=tk.RIGHT, padx=(2, 0))
        button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if self.tooltip_factory is not None:
            shortcut = SHORTCUTS.get(selected)
            title = f"{label} ({shortcut})" if shortcut else label
            self.tooltip_factory(button, f"{title}\n{description}", demo=selected)
            self.tooltip_factory(picker, "Выбрать вариант инструмента")

    def _choose_group_tool(self, group: str, value: str) -> None:
        self._group_choice[group] = value
        self.select_tool(value)
        self.render()

__all__ = [name for name in globals() if not name.startswith("__")]
