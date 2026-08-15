from .tool_options_shared import *
from .tool_options_base import ToolOptionsBaseMixin
from .tool_options_controls import ToolOptionsControlsMixin


class ToolOptionsPanel(ToolOptionsBaseMixin, ToolOptionsControlsMixin, ttk.Frame):
    pass
