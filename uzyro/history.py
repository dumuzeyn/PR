from __future__ import annotations

from .history_base import *
from .history_patches import *


class History:
    def __init__(self, memory_limit_bytes: int = 512 * 1024 * 1024) -> None:
        self.memory_limit_bytes = memory_limit_bytes
        self.undo_stack: list[Command] = []
        self.redo_stack: list[Command] = []
        self.memory_bytes = 0
        self.last_command: Command | None = None

    def clear(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.memory_bytes = 0
        self.last_command = None

    def push(self, command: Command) -> None:
        if command.memory_bytes <= 0:
            return
        self.undo_stack.append(command)
        self.memory_bytes += command.memory_bytes
        self.redo_stack.clear()
        while self.undo_stack and self.memory_bytes > self.memory_limit_bytes:
            dropped = self.undo_stack.pop(0)
            self.memory_bytes -= dropped.memory_bytes

    def undo(self, document: Document) -> str | None:
        if not self.undo_stack:
            return None
        command = self.undo_stack.pop()
        self.memory_bytes -= command.memory_bytes
        with profiler.measure("history.undo"):
            command.undo(document)
        self.redo_stack.append(command)
        self.last_command = command
        return command.label

    def redo(self, document: Document) -> str | None:
        if not self.redo_stack:
            return None
        command = self.redo_stack.pop()
        with profiler.measure("history.redo"):
            command.redo(document)
        self.undo_stack.append(command)
        self.memory_bytes += command.memory_bytes
        self.last_command = command
        return command.label

__all__ = [name for name in globals() if not name.startswith("__")]
