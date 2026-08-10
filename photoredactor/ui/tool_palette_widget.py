from __future__ import annotations

from .tool_palette_shared import *


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
        previous_group: str | None = None
        for value in self.order:
            if value not in self.visible or value not in by_id:
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

__all__ = [name for name in globals() if not name.startswith("__")]
