from __future__ import annotations

from pathlib import Path
import tkinter as tk

from PIL import Image, ImageSequence, ImageTk

from .theme import TOKENS


TOOL_DEMO_DIR = Path(__file__).resolve().parents[1] / "assets" / "tool_demos"


def tool_demo_path(tool_id: str) -> Path:
    return TOOL_DEMO_DIR / f"{tool_id}.gif"


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str, delay: int = 450, *, demo: str | None = None) -> None:
        self.widget = widget
        self.text = text
        self.delay = delay
        self.demo = demo
        self._after_id: str | None = None
        self._animation_id: str | None = None
        self._tip: tk.Toplevel | None = None
        self._demo_label: tk.Label | None = None
        self._frames: list[ImageTk.PhotoImage] = []
        self._durations: list[int] = []
        self._frame_index = 0
        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def schedule(self, _event=None) -> None:
        self.cancel()
        self._after_id = self.widget.after(self.delay, self.show)

    def cancel(self) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def show(self) -> None:
        if self._tip is not None:
            return
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        try:
            self._tip.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        shadow = tk.Frame(self._tip, background=TOKENS.ACTIVE_SHADOW, padx=1, pady=1)
        shadow.pack(padx=(0, 2), pady=(0, 2))
        body = tk.Frame(
            shadow,
            background=TOKENS.PANEL_RAISED,
            highlightbackground=TOKENS.BORDER_STRONG,
            highlightthickness=1,
            padx=8,
            pady=7,
        )
        body.pack()
        tk.Label(
            body,
            text=self.text,
            justify=tk.LEFT,
            background=TOKENS.PANEL_RAISED,
            foreground=TOKENS.TEXT_PRIMARY,
            borderwidth=0,
            padx=0,
            pady=0,
            wraplength=288,
        ).pack(anchor=tk.W)
        self._load_demo_frames()
        if self._frames:
            self._demo_label = tk.Label(body, borderwidth=0, background=TOKENS.PANEL_RAISED)
            self._demo_label.pack(pady=(7, 0))
            self._frame_index = 0
            self._animate()

        self._tip.update_idletasks()
        tip_width = self._tip.winfo_reqwidth()
        tip_height = self._tip.winfo_reqheight()
        x = self.widget.winfo_rootx() + self.widget.winfo_width() + 10
        y = self.widget.winfo_rooty()
        screen_width = self.widget.winfo_screenwidth()
        screen_height = self.widget.winfo_screenheight()
        if x + tip_width > screen_width - 8:
            x = self.widget.winfo_rootx() - tip_width - 10
        x = max(8, min(x, screen_width - tip_width - 8))
        y = max(8, min(y, screen_height - tip_height - 8))
        self._tip.wm_geometry(f"+{x}+{y}")

    def _load_demo_frames(self) -> None:
        if self._frames or not self.demo:
            return
        try:
            with Image.open(tool_demo_path(self.demo)) as animation:
                for frame in ImageSequence.Iterator(animation):
                    rendered = frame.convert("RGB")
                    self._frames.append(ImageTk.PhotoImage(rendered, master=self.widget))
                    self._durations.append(max(60, min(250, int(frame.info.get("duration", 110)))))
        except (OSError, tk.TclError):
            self._frames.clear()
            self._durations.clear()

    def _animate(self) -> None:
        if self._tip is None or self._demo_label is None or not self._frames:
            return
        index = self._frame_index % len(self._frames)
        self._demo_label.configure(image=self._frames[index])
        self._frame_index = (index + 1) % len(self._frames)
        self._animation_id = self.widget.after(self._durations[index], self._animate)

    def hide(self, _event=None) -> None:
        self.cancel()
        if self._animation_id is not None:
            try:
                self.widget.after_cancel(self._animation_id)
            except tk.TclError:
                pass
            self._animation_id = None
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None
        self._demo_label = None
