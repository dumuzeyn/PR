from __future__ import annotations

from ..app_shared import *


class DocumentWorkspaceMixin:
    def _build_document_tabs(self, parent: ttk.Frame) -> None:
        self.document_tabs = ttk.Frame(parent, style="DocumentTabs.TFrame", height=35)
        self.document_tabs.pack(fill=tk.X)
        self.document_tabs.pack_propagate(False)
        self._document_tab_buttons: dict[str, ttk.Button] = {}
        self.refresh_document_tabs()

    def _capture_document_session(self) -> None:
        session = self.document_manager.active
        if session is None or session.document is not self.doc:
            return
        session.history = self.history
        session.selected_layer_ids = set(getattr(self, "selected_layer_ids", set()))
        session.zoom = float(self.zoom.get())
        session.edit_generation = int(self._edit_generation)
        if hasattr(self, "canvas"):
            xview, yview = self.canvas.xview(), self.canvas.yview()
            session.xview = float(xview[0]) if xview else 0.0
            session.yview = float(yview[0]) if yview else 0.0

    def open_document_session(self, document: Document, *, replace_startup: bool = False) -> None:
        self._capture_document_session()
        current = self.document_manager.active
        empty_start = (
            replace_startup
            or not self._editor_active
            and len(self.document_manager.documents) == 1
            and current is not None
            and not current.document.path
            and not current.document.dirty
            and not current.history.undo_stack
        )
        session = self.document_manager.replace_active(document) if empty_start else self.document_manager.add(document)
        self._apply_document_session(session, fit=True)

    def switch_document(self, session_id: str) -> None:
        if self.document_manager.active_document_id == session_id:
            return
        self.cancel_incomplete_interaction()
        self._capture_document_session()
        self._apply_document_session(self.document_manager.activate(session_id), fit=False)

    def cycle_document(self, direction: int = 1) -> str:
        documents = self.document_manager.documents
        if len(documents) < 2:
            return "break"
        current = next((index for index, item in enumerate(documents) if item.id == self.document_manager.active_document_id), 0)
        self.switch_document(documents[(current + direction) % len(documents)].id)
        return "break"

    def close_active_document(self) -> str:
        session = self.document_manager.active
        if session is not None:
            self.close_document_tab(session.id)
        return "break"

    def _apply_document_session(self, session, *, fit: bool) -> None:
        self.doc = session.document
        self.history = session.history
        layer_ids = {layer.id for layer in self.doc.layers}
        self.selected_layer_ids = set(session.selected_layer_ids) & layer_ids
        if not self.selected_layer_ids and self.doc.layers:
            self.selected_layer_ids = {self.doc.layer.id}
        self._edit_generation = session.edit_generation
        self.selection_box = self.doc.selection_bounds()
        self.zoom.set(max(0.01, float(session.zoom)))
        self.render_engine.invalidate_full(self.doc)
        self._composite_cache = None
        self._canvas_view_signature = None
        self._layer_thumbnail_cache.clear()
        self._mask_thumbnail_cache.clear()
        self.refresh_document_tabs()
        self.show_editor()
        if fit:
            return

        def restore_view() -> None:
            if not self.winfo_exists() or self.document_manager.active_document_id != session.id:
                return
            self.canvas.xview_moveto(session.xview)
            self.canvas.yview_moveto(session.yview)
            self.refresh_canvas()

        if self._initial_fit_after_id is not None:
            self.after_cancel(self._initial_fit_after_id)
            self._initial_fit_after_id = None
        self.after_idle(restore_view)

    def refresh_document_tabs(self) -> None:
        if not hasattr(self, "document_tabs"):
            return
        for child in self.document_tabs.winfo_children():
            child.destroy()
        self._document_tab_buttons = {}
        active_id = self.document_manager.active_document_id
        for session in self.document_manager.documents:
            style = "ActiveDocumentTab.TButton" if session.id == active_id else "DocumentTab.TButton"
            tab = ttk.Frame(self.document_tabs, style="DocumentTabs.TFrame")
            tab.pack(side=tk.LEFT, fill=tk.Y)
            button = ttk.Button(tab, text=session.title, style=style, command=lambda sid=session.id: self.switch_document(sid))
            button.pack(side=tk.LEFT, fill=tk.Y)
            tab_surface = TOKENS.CONTROL_SELECTED if session.id == active_id else TOKENS.APP_BG
            dirty = tk.Canvas(tab, width=10, height=35, background=tab_surface, highlightthickness=0, borderwidth=0)
            dirty.pack(side=tk.LEFT, fill=tk.Y)
            if session.document.dirty:
                dirty.create_oval(2, 15, 7, 20, fill=TOKENS.ACCENT_HOVER, outline="")
            close = ttk.Button(tab, text="×", width=3, style="DocumentTabClose.TButton", command=lambda sid=session.id: self.close_document_tab(sid))
            close.pack(side=tk.LEFT, fill=tk.Y)
            indicator = tk.Frame(tab, height=2, background=TOKENS.ACCENT if session.id == active_id else TOKENS.APP_BG)
            indicator.place(x=0, rely=1.0, relwidth=1.0, anchor=tk.SW)
            ToolTip(button, "Переключиться на документ")
            ToolTip(close, "Закрыть документ")
            self._document_tab_buttons[session.id] = button
        add = ttk.Button(self.document_tabs, text="+", width=4, style="DocumentTabClose.TButton", command=self.new_document)
        add.pack(side=tk.LEFT, fill=tk.Y)
        ToolTip(add, "Создать новый документ")
        active = self.document_manager.active
        if active is not None and self._editor_active:
            self.title(f"{active.display_title} - UZYRO")

    def _save_session_before_close(self, session) -> bool:
        if not session.document.dirty:
            return True
        choice = messagebox.askyesnocancel("Несохранённые изменения", f"Сохранить изменения в «{session.title}»?", parent=self)
        if choice is None:
            return False
        if not choice:
            return True
        path = session.document.path
        if not path or not path.lower().endswith(".prdx"):
            path = filedialog.asksaveasfilename(parent=self, defaultextension=".prdx", filetypes=[("Проект UZYRO", "*.prdx")])
        if not path:
            return False
        try:
            session.document.save_project(path)
        except Exception as exc:
            messagebox.showerror("Сохранение", f"Не удалось сохранить документ:\n{exc}", parent=self)
            return False
        session.document.path = path
        session.document.dirty = False
        self.add_recent_file(path)
        return True

    def close_document_tab(self, session_id: str) -> bool:
        session = next((item for item in self.document_manager.documents if item.id == session_id), None)
        if session is None or not self._save_session_before_close(session):
            return False
        was_active = session_id == self.document_manager.active_document_id
        next_session = self.document_manager.close(session_id)
        if next_session is None:
            placeholder = Document.new()
            next_session = self.document_manager.add(placeholder)
            self.doc, self.history = next_session.document, next_session.history
            self.selected_layer_ids = {self.doc.layer.id}
            self.show_start_screen()
        elif was_active:
            self._apply_document_session(next_session, fit=False)
        self.refresh_document_tabs()
        return True

    def close_application(self) -> None:
        self._capture_document_session()
        for session in list(self.document_manager.documents):
            if not self._save_session_before_close(session):
                return
        self.destroy()


__all__ = ["DocumentWorkspaceMixin"]
