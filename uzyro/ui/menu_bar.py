from __future__ import annotations

import tkinter as tk

from .theme import TOKENS


MENU_ROW_HEIGHT = 28


class DarkMenuPopup(tk.Toplevel):
    def __init__(self, owner: "DesktopMenuBar", menu: tk.Menu, x: int, y: int, parent_popup=None) -> None:
        super().__init__(owner)
        self.owner = owner
        self.menu = menu
        self.parent_popup = parent_popup
        self.child_popup: DarkMenuPopup | None = None
        self.overrideredirect(True)
        self.configure(background=TOKENS.ACTIVE_SHADOW)
        try:
            self.attributes("-topmost", True)
        except tk.TclError:
            pass
        self._run_postcommand()
        self._build_rows()
        self.update_idletasks()
        self.geometry(f"+{x}+{y}")
        self.bind("<Escape>", lambda _event: owner.close_popup())

    def _run_postcommand(self) -> None:
        callback = str(self.menu.cget("postcommand") or "")
        if callback:
            try:
                self.menu.tk.call(callback)
            except tk.TclError:
                pass

    def _build_rows(self) -> None:
        edge = tk.Frame(
            self,
            background=TOKENS.PANEL_RAISED,
            highlightbackground=TOKENS.BORDER_STRONG,
            highlightthickness=1,
        )
        edge.pack(padx=(0, 2), pady=(0, 2))
        end = self.menu.index(tk.END)
        if end is None:
            return
        labels = [str(self.menu.entrycget(index, "label")) for index in range(end + 1) if self.menu.type(index) != "separator"]
        width = max(250, min(420, 118 + max((len(label) for label in labels), default=12) * 7))
        for index in range(end + 1):
            if self.menu.type(index) == "separator":
                separator = tk.Frame(edge, width=width, height=9, background=TOKENS.PANEL_RAISED)
                separator.pack_propagate(False)
                separator.pack()
                tk.Frame(separator, height=1, background=TOKENS.BORDER_SUBTLE).pack(fill=tk.X, padx=10, pady=4)
                continue
            self._build_item(edge, index, width)

    def _build_item(self, parent: tk.Frame, index: int, width: int) -> None:
        kind = self.menu.type(index)
        disabled = str(self.menu.entrycget(index, "state")) == "disabled"
        row = tk.Frame(parent, width=width, height=MENU_ROW_HEIGHT, background=TOKENS.PANEL_RAISED)
        row.pack_propagate(False)
        row.pack(fill=tk.X)
        indicator = ""
        if kind in {"checkbutton", "radiobutton"}:
            variable = str(self.menu.entrycget(index, "variable") or "")
            option = "onvalue" if kind == "checkbutton" else "value"
            selected = str(self.menu.getvar(variable)) == str(self.menu.entrycget(index, option)) if variable else False
            indicator = "✓" if selected else ""
        foreground = TOKENS.TEXT_DISABLED if disabled else TOKENS.TEXT_PRIMARY
        tk.Label(row, text=indicator, width=2, anchor="center", background=TOKENS.PANEL_RAISED, foreground=TOKENS.ACCENT).pack(side=tk.LEFT, padx=(6, 0))
        tk.Label(row, text=self.menu.entrycget(index, "label"), anchor="w", background=TOKENS.PANEL_RAISED, foreground=foreground).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 10))
        accelerator = str(self.menu.entrycget(index, "accelerator") or "")
        tk.Label(row, text=accelerator, anchor="e", background=TOKENS.PANEL_RAISED, foreground=TOKENS.TEXT_MUTED).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(row, text="›" if kind == "cascade" else "", width=2, background=TOKENS.PANEL_RAISED, foreground=TOKENS.TEXT_SECONDARY).pack(side=tk.RIGHT, padx=(0, 4))
        if disabled:
            return
        for widget in (row, *row.winfo_children()):
            widget.bind("<Enter>", lambda _event, item=row, idx=index: self._hover(item, idx), add="+")
            widget.bind("<ButtonRelease-1>", lambda _event, idx=index: self._activate(idx), add="+")

    def _paint_row(self, row: tk.Frame, color: str) -> None:
        row.configure(background=color)
        for child in row.winfo_children():
            child.configure(background=color)

    def _hover(self, row: tk.Frame, index: int) -> None:
        for sibling in row.master.winfo_children():
            if isinstance(sibling, tk.Frame) and int(sibling.winfo_height()) == MENU_ROW_HEIGHT:
                self._paint_row(sibling, TOKENS.PANEL_RAISED)
        self._paint_row(row, TOKENS.CONTROL_HOVER)
        if self.menu.type(index) == "cascade":
            self._open_submenu(index, row)
        elif self.child_popup is not None:
            self.child_popup.destroy_chain()
            self.child_popup = None

    def _submenu(self, index: int) -> tk.Menu | None:
        path = str(self.menu.entrycget(index, "menu") or "")
        try:
            return self.menu.nametowidget(path) if path else None
        except KeyError:
            return None

    def _open_submenu(self, index: int, row: tk.Frame) -> None:
        submenu = self._submenu(index)
        if submenu is None:
            return
        if self.child_popup is not None:
            if self.child_popup.menu is submenu:
                return
            self.child_popup.destroy_chain()
        self.child_popup = DarkMenuPopup(
            self.owner,
            submenu,
            self.winfo_rootx() + self.winfo_width() - 3,
            row.winfo_rooty() - 1,
            parent_popup=self,
        )

    def _activate(self, index: int) -> None:
        if self.menu.type(index) == "cascade":
            row = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
            if row is not None:
                while row.master is not None and row.master is not self:
                    if isinstance(row, tk.Frame) and int(row.winfo_height()) == MENU_ROW_HEIGHT:
                        self._open_submenu(index, row)
                        break
                    row = row.master
            return
        self.owner.close_popup()
        try:
            self.menu.invoke(index)
        except tk.TclError:
            pass

    def destroy_chain(self) -> None:
        if self.child_popup is not None:
            self.child_popup.destroy_chain()
            self.child_popup = None
        if self.winfo_exists():
            self.destroy()


class DesktopMenuBar(tk.Frame):
    def __init__(self, master: tk.Misc, menu: tk.Menu) -> None:
        super().__init__(master, height=30, background=TOKENS.APP_BG)
        self.pack_propagate(False)
        self.menu = menu
        self.popup: DarkMenuPopup | None = None
        self.active_button: tk.Label | None = None
        self._buttons: list[tk.Label] = []
        self._build()
        self.bind_all("<ButtonPress-1>", self._outside_click, add="+")

    def _build(self) -> None:
        end = self.menu.index(tk.END)
        if end is None:
            return
        for index in range(end + 1):
            label = tk.Label(
                self,
                text=self.menu.entrycget(index, "label"),
                background=TOKENS.APP_BG,
                foreground=TOKENS.TEXT_SECONDARY,
                padx=10,
                cursor="hand2",
            )
            label.pack(side=tk.LEFT, fill=tk.Y)
            label.bind("<Enter>", lambda _event, button=label, idx=index: self._enter(button, idx))
            label.bind("<Leave>", lambda _event, button=label: self._leave(button))
            label.bind("<ButtonRelease-1>", lambda _event, button=label, idx=index: self.open_popup(button, idx))
            self._buttons.append(label)

    def _enter(self, button: tk.Label, index: int) -> None:
        button.configure(background=TOKENS.CONTROL_HOVER, foreground=TOKENS.TEXT_PRIMARY)
        if self.popup is not None and button is not self.active_button:
            self.open_popup(button, index)

    def _leave(self, button: tk.Label) -> None:
        if button is not self.active_button:
            button.configure(background=TOKENS.APP_BG, foreground=TOKENS.TEXT_SECONDARY)

    def _submenu(self, index: int) -> tk.Menu | None:
        path = str(self.menu.entrycget(index, "menu") or "")
        try:
            return self.menu.nametowidget(path) if path else None
        except KeyError:
            return None

    def open_popup(self, button: tk.Label, index: int) -> None:
        submenu = self._submenu(index)
        if submenu is None:
            return
        self.close_popup()
        self.active_button = button
        button.configure(background=TOKENS.CONTROL_SELECTED, foreground=TOKENS.TEXT_PRIMARY)
        self.popup = DarkMenuPopup(self, submenu, button.winfo_rootx(), button.winfo_rooty() + button.winfo_height())
        self.popup.focus_force()

    def close_popup(self) -> None:
        if self.popup is not None:
            self.popup.destroy_chain()
            self.popup = None
        if self.active_button is not None:
            self.active_button.configure(background=TOKENS.APP_BG, foreground=TOKENS.TEXT_SECONDARY)
            self.active_button = None

    def _outside_click(self, event) -> None:
        if self.popup is None:
            return
        widget = self.winfo_containing(event.x_root, event.y_root)
        current = widget
        while current is not None:
            if current is self or isinstance(current, DarkMenuPopup):
                return
            current = getattr(current, "master", None)
        self.close_popup()


class DarkMenuButton(tk.Frame):
    def __init__(self, master: tk.Misc, *, text: str, width: int | None = None, **kwargs) -> None:
        super().__init__(master, height=28, background=TOKENS.CONTROL_BG, highlightthickness=1, highlightbackground=TOKENS.BORDER_SUBTLE, **kwargs)
        self.pack_propagate(False)
        self.menu: tk.Menu | None = None
        self.popup: DarkMenuPopup | None = None
        self.active_button = None
        self._label = tk.Label(self, text=text, anchor=tk.W, padx=8, background=TOKENS.CONTROL_BG, foreground=TOKENS.TEXT_PRIMARY, cursor="hand2")
        self._label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._arrow = tk.Label(self, text="▼", width=2, font=("Segoe UI", 7), background=TOKENS.CONTROL_BG, foreground=TOKENS.TEXT_SECONDARY, cursor="hand2")
        self._arrow.pack(side=tk.RIGHT, fill=tk.Y)
        if width is not None:
            self.configure(width=max(28, width * 8 + 22))
        for widget in (self, self._label, self._arrow):
            widget.bind("<Enter>", lambda _event: self._paint(TOKENS.CONTROL_HOVER), add="+")
            widget.bind("<Leave>", self._leave, add="+")
            widget.bind("<ButtonRelease-1>", self._toggle, add="+")
        self.bind_all("<ButtonPress-1>", self._outside_click, add="+")

    def configure(self, cnf=None, **kwargs):
        menu = kwargs.pop("menu", None)
        if menu is not None:
            self.menu = menu
        return super().configure(cnf, **kwargs)

    config = configure

    def _paint(self, color: str) -> None:
        self.configure(background=color)
        self._label.configure(background=color)
        self._arrow.configure(background=color)

    def _leave(self, _event=None) -> None:
        if self.popup is None:
            self._paint(TOKENS.CONTROL_BG)

    def _toggle(self, _event=None) -> str:
        if self.popup is not None:
            self.close_popup()
            return "break"
        if self.menu is None:
            return "break"
        self._paint(TOKENS.CONTROL_SELECTED)
        self.popup = DarkMenuPopup(self, self.menu, self.winfo_rootx(), self.winfo_rooty() + self.winfo_height())
        self.popup.focus_force()
        return "break"

    def close_popup(self) -> None:
        if self.popup is not None:
            self.popup.destroy_chain()
            self.popup = None
        self._paint(TOKENS.CONTROL_BG)

    def _outside_click(self, event) -> None:
        if self.popup is None:
            return
        current = self.winfo_containing(event.x_root, event.y_root)
        while current is not None:
            if current is self or isinstance(current, DarkMenuPopup):
                return
            current = getattr(current, "master", None)
        self.close_popup()
