from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from .document import Document
from .history import History


@dataclass
class DocumentSession:
    document: Document
    history: History = field(default_factory=History)
    id: str = field(default_factory=lambda: uuid4().hex)
    untitled_number: int | None = None
    selected_layer_ids: set[str] = field(default_factory=set)
    zoom: float = 1.0
    xview: float = 0.0
    yview: float = 0.0
    edit_generation: int = 0

    @property
    def title(self) -> str:
        if self.document.path:
            return Path(self.document.path).name
        number = self.untitled_number or 1
        return "Новый документ" if number == 1 else f"Новый документ {number}"

    @property
    def display_title(self) -> str:
        return f"{self.title}{'*' if self.document.dirty else ''}"


class DocumentManager:
    """Owns open documents while the UI projects one session as active."""

    def __init__(self, document: Document | None = None, history: History | None = None) -> None:
        self.documents: list[DocumentSession] = []
        self.active_document_id: str | None = None
        self._untitled_counter = 0
        if document is not None:
            self.add(document, history=history)

    @property
    def active(self) -> DocumentSession | None:
        return next((item for item in self.documents if item.id == self.active_document_id), None)

    def add(self, document: Document, history: History | None = None) -> DocumentSession:
        self._untitled_counter += 1
        session = DocumentSession(
            document=document,
            history=history or History(),
            untitled_number=None if document.path else self._untitled_counter,
            selected_layer_ids={document.layer.id} if document.layers else set(),
        )
        self.documents.append(session)
        self.active_document_id = session.id
        return session

    def replace_active(self, document: Document, history: History | None = None) -> DocumentSession:
        current = self.active
        if current is None:
            return self.add(document, history)
        replacement = DocumentSession(
            document=document,
            history=history or History(),
            id=current.id,
            untitled_number=current.untitled_number if not document.path else None,
            selected_layer_ids={document.layer.id} if document.layers else set(),
        )
        index = self.documents.index(current)
        self.documents[index] = replacement
        return replacement

    def activate(self, session_id: str) -> DocumentSession:
        session = next((item for item in self.documents if item.id == session_id), None)
        if session is None:
            raise KeyError(session_id)
        self.active_document_id = session.id
        return session

    def close(self, session_id: str) -> DocumentSession | None:
        session = next((item for item in self.documents if item.id == session_id), None)
        if session is None:
            return self.active
        index = self.documents.index(session)
        self.documents.remove(session)
        if self.active_document_id == session_id:
            if self.documents:
                self.active_document_id = self.documents[min(index, len(self.documents) - 1)].id
            else:
                self.active_document_id = None
        return self.active


__all__ = ["DocumentManager", "DocumentSession"]
