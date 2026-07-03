from __future__ import annotations

from pathlib import Path
import time
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from .core import (
    Document,
    add_noise,
    add_text,
    adjust_brightness_contrast,
    adjust_saturation,
    apply_gradient,
    blur,
    curves,
    draw_brush,
    flood_fill,
    levels,
    rgba_array_to_pil,
    sharpen,
)


class PhotoRedactorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PhotoRedactor")
        self.geometry("1440x920")
        self.minsize(1000, 640)

        self.doc = Document.new()
        self.history: list[dict] = []
        self.redo_stack: list[dict] = []
        self.tool = tk.StringVar(value="brush")
        self.zoom = tk.DoubleVar(value=1.0)
        self.brush_size = tk.IntVar(value=28)
        self.opacity = tk.DoubleVar(value=1.0)
        self.tolerance = tk.IntVar(value=24)
        self.foreground = (30, 120, 255, 255)
        self.background = (255, 255, 255, 255)
        self.drag_start: tuple[int, int] | None = None
        self.last_point: tuple[int, int] | None = None
        self._space_down = False
        self._panning = False
        self.selection_id: int | None = None
        self.selection_box: tuple[int, int, int, int] | None = None
        self._preview_image: ImageTk.PhotoImage | None = None
        self._canvas_image_id: int | None = None
        self._render_after_id: str | None = None
        self._last_render_time = 0.0
        self._composite_cache = None
        self._composite_dirty = True
        self._view_dirty = True

        self._build_ui()
        self.push_history("Initial")
        self.refresh()

    def _build_ui(self) -> None:
        self._build_menu()
        root = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root, width=88)
        center = ttk.Frame(root)
        right = ttk.Frame(root, width=280)
        root.add(left, weight=0)
        root.add(center, weight=1)
        root.add(right, weight=0)

        self._build_tools(left)
        self._build_canvas(center)
        self._build_panels(right)

        self.status = ttk.Label(self, text="", anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self.bind_all("<Control-z>", lambda _e: self.undo())
        self.bind_all("<Control-y>", lambda _e: self.redo())
        self.bind_all("<Control-s>", lambda _e: self.save())
        self.bind_all("<Control-o>", lambda _e: self.open_file())
        self.bind_all("<Control-n>", lambda _e: self.new_document())
        self.bind_all("<plus>", lambda _e: self.set_zoom(self.zoom.get() * 1.25))
        self.bind_all("<minus>", lambda _e: self.set_zoom(self.zoom.get() / 1.25))
        self.bind_all("<KeyPress-space>", self.space_down)
        self.bind_all("<KeyRelease-space>", self.space_up)

    def _build_menu(self) -> None:
        menu = tk.Menu(self)
        self.config(menu=menu)

        file_menu = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_document, accelerator="Ctrl+N")
        file_menu.add_command(label="Open image/project", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Save project", command=self.save, accelerator="Ctrl+S")
        file_menu.add_command(label="Save project as", command=self.save_as_project)
        file_menu.add_command(label="Export image", command=self.export_image)
        file_menu.add_separator()
        file_menu.add_command(label="Batch resize/convert", command=self.batch_process)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)

        edit = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Edit", menu=edit)
        edit.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit.add_separator()
        edit.add_command(label="Clear selection", command=self.clear_selection)

        image = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Image", menu=image)
        image.add_command(label="Resize image", command=self.resize_image)
        image.add_command(label="Resize canvas", command=self.resize_canvas)
        image.add_command(label="Crop to selection", command=self.crop_to_selection)
        image.add_separator()
        image.add_command(label="Rotate 90 CW", command=lambda: self.rotate(90))
        image.add_command(label="Rotate 180", command=lambda: self.rotate(180))
        image.add_command(label="Flip horizontal", command=lambda: self.flip(horizontal=True))
        image.add_command(label="Flip vertical", command=lambda: self.flip(horizontal=False))

        layer = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Layer", menu=layer)
        layer.add_command(label="New layer", command=self.new_layer)
        layer.add_command(label="Duplicate layer", command=self.duplicate_layer)
        layer.add_command(label="Delete layer", command=self.delete_layer)
        layer.add_separator()
        layer.add_command(label="Move up", command=lambda: self.move_layer(1))
        layer.add_command(label="Move down", command=lambda: self.move_layer(-1))
        layer.add_command(label="Merge down", command=self.merge_down)
        layer.add_command(label="Flatten image", command=self.flatten)

        adj = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Adjust", menu=adj)
        adj.add_command(label="Brightness/Contrast", command=self.adjust_brightness_contrast)
        adj.add_command(label="Saturation", command=self.adjust_saturation)
        adj.add_command(label="Levels", command=self.adjust_levels)
        adj.add_command(label="Curves", command=self.adjust_curves)
        adj.add_command(label="Invert", command=self.adjust_invert)
        adj.add_command(label="Grayscale", command=self.adjust_grayscale)

        filters = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="Filter", menu=filters)
        filters.add_command(label="Gaussian blur", command=self.filter_blur)
        filters.add_command(label="Sharpen", command=self.filter_sharpen)
        filters.add_command(label="Noise", command=self.filter_noise)

        view = tk.Menu(menu, tearoff=False)
        menu.add_cascade(label="View", menu=view)
        view.add_command(label="Zoom in", command=lambda: self.set_zoom(self.zoom.get() * 1.25))
        view.add_command(label="Zoom out", command=lambda: self.set_zoom(self.zoom.get() / 1.25))
        view.add_command(label="100%", command=lambda: self.set_zoom(1.0))
        view.add_command(label="Fit", command=self.fit_to_screen)

    def _build_tools(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Tools").pack(pady=(8, 4))
        tools = [
            ("Hand", "hand"),
            ("Move", "move"),
            ("Brush", "brush"),
            ("Eraser", "eraser"),
            ("Fill", "fill"),
            ("Gradient", "gradient"),
            ("Text", "text"),
            ("Select", "select"),
            ("Crop", "crop"),
        ]
        for text, value in tools:
            ttk.Radiobutton(parent, text=text, value=value, variable=self.tool).pack(fill=tk.X, padx=8, pady=2)
        ttk.Separator(parent).pack(fill=tk.X, pady=8)
        ttk.Button(parent, text="FG", command=self.pick_foreground).pack(fill=tk.X, padx=8, pady=2)
        ttk.Button(parent, text="BG", command=self.pick_background).pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(parent, text="Size").pack()
        ttk.Scale(parent, from_=1, to=220, variable=self.brush_size, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8)
        ttk.Label(parent, text="Opacity").pack()
        ttk.Scale(parent, from_=0.01, to=1.0, variable=self.opacity, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8)
        ttk.Label(parent, text="Tolerance").pack()
        ttk.Scale(parent, from_=0, to=128, variable=self.tolerance, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8)

    def _build_canvas(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="-", command=lambda: self.set_zoom(self.zoom.get() / 1.25), width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="+", command=lambda: self.set_zoom(self.zoom.get() * 1.25), width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="100%", command=lambda: self.set_zoom(1.0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Fit", command=self.fit_to_screen).pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(toolbar, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=10)

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(frame, bg="#24262b", highlightthickness=0)
        xbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        ybar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self.pointer_down)
        self.canvas.bind("<B1-Motion>", self.pointer_drag)
        self.canvas.bind("<ButtonRelease-1>", self.pointer_up)
        self.canvas.bind("<ButtonPress-2>", self.pan_down)
        self.canvas.bind("<B2-Motion>", self.pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.pan_up)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)

    def _build_panels(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Layers").pack(anchor=tk.W, padx=8, pady=(8, 4))
        self.layer_list = tk.Listbox(parent, height=16, exportselection=False)
        self.layer_list.pack(fill=tk.BOTH, expand=False, padx=8)
        self.layer_list.bind("<<ListboxSelect>>", self.layer_selected)
        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(buttons, text="+", width=3, command=self.new_layer).pack(side=tk.LEFT)
        ttk.Button(buttons, text="x", width=3, command=self.delete_layer).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Dup", command=self.duplicate_layer).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Up", command=lambda: self.move_layer(1)).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Dn", command=lambda: self.move_layer(-1)).pack(side=tk.LEFT)
        ttk.Label(parent, text="Layer opacity").pack(anchor=tk.W, padx=8)
        self.layer_opacity = tk.DoubleVar(value=1.0)
        ttk.Scale(parent, from_=0.0, to=1.0, variable=self.layer_opacity, command=self.change_layer_opacity).pack(fill=tk.X, padx=8)
        ttk.Button(parent, text="Toggle visible", command=self.toggle_layer_visible).pack(fill=tk.X, padx=8, pady=6)
        ttk.Separator(parent).pack(fill=tk.X, pady=8)
        self.info = ttk.Label(parent, text="", justify=tk.LEFT)
        self.info.pack(anchor=tk.W, padx=8)

    def push_history(self, label: str) -> None:
        self.history.append(self.doc.snapshot())
        if len(self.history) > 40:
            self.history.pop(0)
        self.redo_stack.clear()
        self.status_text(label)

    def restore_snapshot(self, data: dict) -> None:
        path = self.doc.path
        self.doc = Document.restore(data)
        self.doc.path = path
        self.mark_dirty()
        self.refresh()

    def undo(self) -> None:
        if len(self.history) <= 1:
            return
        self.redo_stack.append(self.history.pop())
        self.restore_snapshot(self.history[-1])
        self.status_text("Undo")

    def redo(self) -> None:
        if not self.redo_stack:
            return
        data = self.redo_stack.pop()
        self.history.append(data)
        self.restore_snapshot(data)
        self.status_text("Redo")

    def mark_dirty(self) -> None:
        self._composite_dirty = True
        self._view_dirty = True

    def refresh_canvas(self) -> None:
        if self._composite_dirty or self._composite_cache is None:
            self._composite_cache = self.doc.composite(checker=True)
            self._composite_dirty = False
        image = rgba_array_to_pil(self._composite_cache)
        scale = self.zoom.get()
        if scale != 1.0:
            resample = Image.Resampling.NEAREST if scale >= 4 else Image.Resampling.BILINEAR
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), resample)
        self._preview_image = ImageTk.PhotoImage(image)
        if self._canvas_image_id is None:
            self._canvas_image_id = self.canvas.create_image(0, 0, image=self._preview_image, anchor=tk.NW)
        else:
            self.canvas.itemconfigure(self._canvas_image_id, image=self._preview_image)
        self.canvas.configure(scrollregion=(0, 0, image.width, image.height))
        self.zoom_label.configure(text=f"{round(scale * 100)}%")
        self._last_render_time = time.perf_counter()
        self._view_dirty = False

    def request_canvas_refresh(self) -> None:
        self.mark_dirty()
        if self._render_after_id is not None:
            return
        elapsed_ms = (time.perf_counter() - self._last_render_time) * 1000
        delay = 0 if elapsed_ms >= 33 else int(33 - elapsed_ms)
        self._render_after_id = self.after(delay, self._run_scheduled_canvas_refresh)

    def _run_scheduled_canvas_refresh(self) -> None:
        self._render_after_id = None
        self.refresh_canvas()

    def refresh(self) -> None:
        self.mark_dirty()
        if self._render_after_id is not None:
            self.after_cancel(self._render_after_id)
            self._render_after_id = None
        self.refresh_canvas()
        self.refresh_layers()
        self.info.configure(text=f"{self.doc.width} x {self.doc.height}px\n{len(self.doc.layers)} layers\nActive: {self.doc.layer.name}")

    def refresh_layers(self) -> None:
        self.layer_list.delete(0, tk.END)
        for i, layer in enumerate(reversed(self.doc.layers)):
            marker = "*" if layer.visible else "-"
            self.layer_list.insert(tk.END, f"{marker} {layer.name}  {round(layer.opacity * 100)}%")
        self.layer_list.selection_clear(0, tk.END)
        self.layer_list.selection_set(len(self.doc.layers) - 1 - self.doc.active_layer)
        self.layer_opacity.set(self.doc.layer.opacity)

    def status_text(self, text: str) -> None:
        if hasattr(self, "status"):
            self.status.configure(text=text)

    def canvas_to_doc(self, event) -> tuple[int, int]:
        x = int(self.canvas.canvasx(event.x) / self.zoom.get())
        y = int(self.canvas.canvasy(event.y) / self.zoom.get())
        return x, y

    def space_down(self, _event) -> None:
        self._space_down = True
        self.canvas.configure(cursor="fleur")

    def space_up(self, _event) -> None:
        self._space_down = False
        if not self._panning:
            self.canvas.configure(cursor="")

    def pan_down(self, event) -> None:
        self._panning = True
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def pan_drag(self, event) -> None:
        if self._panning:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def pan_up(self, _event) -> None:
        self._panning = False
        self.canvas.configure(cursor="fleur" if self._space_down else "")

    def pointer_down(self, event) -> None:
        if self.tool.get() == "hand" or self._space_down:
            self.pan_down(event)
            return
        point = self.canvas_to_doc(event)
        self.drag_start = point
        self.last_point = point
        tool = self.tool.get()
        if tool in ["brush", "eraser"]:
            self.paint_at(point)
        elif tool == "fill":
            flood_fill(self.doc.layer, point[0], point[1], self.foreground, int(self.tolerance.get()))
            self.doc.dirty = True
            self.push_history("Fill")
            self.refresh()
        elif tool == "text":
            text = simpledialog.askstring("Text", "Text:")
            if text:
                size = simpledialog.askinteger("Text size", "Size:", initialvalue=48, minvalue=4, maxvalue=500) or 48
                add_text(self.doc.layer, point[0], point[1], text, self.foreground, size)
                self.doc.dirty = True
                self.push_history("Text")
                self.refresh()

    def pointer_drag(self, event) -> None:
        if self._panning:
            self.pan_drag(event)
            return
        point = self.canvas_to_doc(event)
        tool = self.tool.get()
        if tool in ["brush", "eraser"]:
            self.paint_line(self.last_point or point, point)
            self.last_point = point
        elif tool == "move" and self.drag_start:
            dx, dy = point[0] - self.drag_start[0], point[1] - self.drag_start[1]
            self.doc.layer.x += dx
            self.doc.layer.y += dy
            self.drag_start = point
            self.doc.dirty = True
            self.request_canvas_refresh()
        elif tool in ["select", "crop", "gradient"]:
            self.draw_selection(self.drag_start, point)

    def pointer_up(self, event) -> None:
        if self._panning:
            self.pan_up(event)
            return
        point = self.canvas_to_doc(event)
        tool = self.tool.get()
        if tool in ["brush", "eraser"]:
            self.push_history(f"{tool.title()} stroke")
        elif tool == "move":
            self.push_history("Move layer")
        elif tool == "gradient" and self.drag_start:
            apply_gradient(self.doc.layer, (*self.drag_start, *point), self.foreground, self.background)
            self.doc.dirty = True
            self.push_history("Gradient")
            self.refresh()
        elif tool in ["select", "crop"] and self.drag_start:
            self.selection_box = (*self.drag_start, *point)
            self.draw_selection(self.drag_start, point)
        self.drag_start = None
        self.last_point = None

    def paint_at(self, point: tuple[int, int]) -> None:
        draw_brush(self.doc.layer, point[0], point[1], int(self.brush_size.get()), self.foreground, float(self.opacity.get()), self.tool.get() == "eraser")
        self.doc.dirty = True
        self.request_canvas_refresh()

    def paint_line(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        steps = max(1, int(((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5 / max(1, self.brush_size.get() / 4)))
        for i in range(steps + 1):
            t = i / steps
            x = round(start[0] * (1 - t) + end[0] * t)
            y = round(start[1] * (1 - t) + end[1] * t)
            draw_brush(self.doc.layer, x, y, int(self.brush_size.get()), self.foreground, float(self.opacity.get()), self.tool.get() == "eraser")
        self.doc.dirty = True
        self.request_canvas_refresh()

    def draw_selection(self, start: tuple[int, int] | None, end: tuple[int, int]) -> None:
        if not start:
            return
        scale = self.zoom.get()
        coords = [v * scale for v in (*start, *end)]
        if self.selection_id is None:
            self.selection_id = self.canvas.create_rectangle(*coords, outline="#50e3ff", dash=(5, 4), width=2)
        else:
            self.canvas.coords(self.selection_id, *coords)

    def clear_selection(self) -> None:
        self.selection_box = None
        if self.selection_id is not None:
            self.canvas.delete(self.selection_id)
            self.selection_id = None

    def new_document(self) -> None:
        width = simpledialog.askinteger("New document", "Width px:", initialvalue=1280, minvalue=1, maxvalue=50000)
        if not width:
            return
        height = simpledialog.askinteger("New document", "Height px:", initialvalue=900, minvalue=1, maxvalue=50000)
        if not height:
            return
        color = colorchooser.askcolor(title="Background color", initialcolor="#ffffff")[0] or (255, 255, 255)
        self.doc = Document.new(width, height, tuple(map(int, color)) + (255,))
        self.history.clear()
        self.redo_stack.clear()
        self.push_history("New document")
        self.refresh()

    def open_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Supported", "*.prdx *.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All", "*.*")])
        if not path:
            return
        self.doc = Document.open_project(path) if path.lower().endswith(".prdx") else Document.from_image(path)
        self.history.clear()
        self.redo_stack.clear()
        self.push_history("Open")
        self.refresh()

    def save(self) -> None:
        if self.doc.path and self.doc.path.lower().endswith(".prdx"):
            self.doc.save_project(self.doc.path)
            self.status_text(f"Saved {self.doc.path}")
        else:
            self.save_as_project()

    def save_as_project(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".prdx", filetypes=[("PhotoRedactor project", "*.prdx")])
        if path:
            self.doc.save_project(path)
            self.status_text(f"Saved {path}")

    def export_image(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("WebP", "*.webp"), ("TIFF", "*.tiff"), ("BMP", "*.bmp")])
        if path:
            self.doc.export_flat(path)
            self.status_text(f"Exported {path}")

    def pick_foreground(self) -> None:
        color = colorchooser.askcolor(title="Foreground")[0]
        if color:
            self.foreground = tuple(map(int, color)) + (255,)

    def pick_background(self) -> None:
        color = colorchooser.askcolor(title="Background")[0]
        if color:
            self.background = tuple(map(int, color)) + (255,)

    def new_layer(self) -> None:
        self.doc.add_layer(f"Layer {len(self.doc.layers) + 1}")
        self.push_history("New layer")
        self.refresh()

    def duplicate_layer(self) -> None:
        self.doc.duplicate_active_layer()
        self.push_history("Duplicate layer")
        self.refresh()

    def delete_layer(self) -> None:
        self.doc.delete_active_layer()
        self.push_history("Delete layer")
        self.refresh()

    def move_layer(self, delta: int) -> None:
        i = self.doc.active_layer
        j = i + delta
        if 0 <= j < len(self.doc.layers):
            self.doc.layers[i], self.doc.layers[j] = self.doc.layers[j], self.doc.layers[i]
            self.doc.active_layer = j
            self.doc.dirty = True
            self.push_history("Layer reorder")
            self.refresh()

    def merge_down(self) -> None:
        self.doc.merge_down()
        self.push_history("Merge down")
        self.refresh()

    def flatten(self) -> None:
        self.doc.flatten()
        self.push_history("Flatten")
        self.refresh()

    def layer_selected(self, _event) -> None:
        sel = self.layer_list.curselection()
        if sel:
            self.doc.active_layer = len(self.doc.layers) - 1 - sel[0]
            self.refresh_layers()

    def change_layer_opacity(self, _value) -> None:
        self.doc.layer.opacity = float(self.layer_opacity.get())
        self.doc.dirty = True
        self.request_canvas_refresh()

    def toggle_layer_visible(self) -> None:
        self.doc.layer.visible = not self.doc.layer.visible
        self.doc.dirty = True
        self.refresh()

    def resize_image(self) -> None:
        width = simpledialog.askinteger("Resize image", "Width px:", initialvalue=self.doc.width, minvalue=1, maxvalue=100000)
        height = simpledialog.askinteger("Resize image", "Height px:", initialvalue=self.doc.height, minvalue=1, maxvalue=100000)
        if width and height:
            self.doc.resize_image(width, height)
            self.push_history("Resize image")
            self.refresh()

    def resize_canvas(self) -> None:
        width = simpledialog.askinteger("Resize canvas", "Width px:", initialvalue=self.doc.width, minvalue=1, maxvalue=100000)
        height = simpledialog.askinteger("Resize canvas", "Height px:", initialvalue=self.doc.height, minvalue=1, maxvalue=100000)
        if width and height:
            self.doc.resize_canvas(width, height)
            self.push_history("Resize canvas")
            self.refresh()

    def crop_to_selection(self) -> None:
        if not self.selection_box:
            messagebox.showinfo("Crop", "Create a rectangular selection first.")
            return
        self.doc.crop(self.selection_box)
        self.push_history("Crop")
        self.clear_selection()
        self.refresh()

    def rotate(self, angle: int) -> None:
        for layer in self.doc.layers:
            if angle == 90:
                layer.pixels = cv2.rotate(layer.pixels, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                layer.pixels = cv2.rotate(layer.pixels, cv2.ROTATE_180)
        if angle in [90, 270]:
            self.doc.width, self.doc.height = self.doc.height, self.doc.width
        self.doc.dirty = True
        self.push_history("Rotate")
        self.refresh()

    def flip(self, horizontal: bool) -> None:
        code = 1 if horizontal else 0
        for layer in self.doc.layers:
            layer.pixels = cv2.flip(layer.pixels, code)
        self.doc.dirty = True
        self.push_history("Flip")
        self.refresh()

    def apply_to_layer(self, label: str, fn) -> None:
        self.doc.layer.pixels = fn(self.doc.layer.pixels)
        self.doc.dirty = True
        self.push_history(label)
        self.refresh()

    def adjust_brightness_contrast(self) -> None:
        b = simpledialog.askinteger("Brightness", "Brightness -255..255:", initialvalue=0, minvalue=-255, maxvalue=255)
        c = simpledialog.askfloat("Contrast", "Contrast multiplier:", initialvalue=1.1, minvalue=0.0, maxvalue=10.0)
        if b is not None and c is not None:
            self.apply_to_layer("brightness/contrast", lambda arr: adjust_brightness_contrast(arr, b, c))

    def adjust_saturation(self) -> None:
        s = simpledialog.askfloat("Saturation", "Saturation multiplier:", initialvalue=1.2, minvalue=0.0, maxvalue=10.0)
        if s is not None:
            self.apply_to_layer("saturation", lambda arr: adjust_saturation(arr, s))

    def adjust_levels(self) -> None:
        black = simpledialog.askinteger("Levels", "Black point:", initialvalue=0, minvalue=0, maxvalue=254)
        white = simpledialog.askinteger("Levels", "White point:", initialvalue=255, minvalue=1, maxvalue=255)
        gamma = simpledialog.askfloat("Levels", "Gamma:", initialvalue=1.0, minvalue=0.01, maxvalue=10.0)
        if black is not None and white is not None and gamma is not None:
            self.apply_to_layer("levels", lambda arr: levels(arr, black, white, gamma))

    def adjust_curves(self) -> None:
        shadows = simpledialog.askinteger("Curves", "Shadows output:", initialvalue=64, minvalue=0, maxvalue=255)
        midtones = simpledialog.askinteger("Curves", "Midtones output:", initialvalue=128, minvalue=0, maxvalue=255)
        highlights = simpledialog.askinteger("Curves", "Highlights output:", initialvalue=192, minvalue=0, maxvalue=255)
        if shadows is not None and midtones is not None and highlights is not None:
            self.apply_to_layer("curves", lambda arr: curves(arr, shadows, midtones, highlights))

    def adjust_invert(self) -> None:
        self.apply_to_layer("invert", lambda arr: self._invert(arr))

    def adjust_grayscale(self) -> None:
        self.apply_to_layer("grayscale", lambda arr: self._grayscale(arr))

    @staticmethod
    def _invert(arr):
        out = arr.copy()
        out[:, :, :3] = 255 - out[:, :, :3]
        return out

    @staticmethod
    def _grayscale(arr):
        out = arr.copy()
        gray = cv2.cvtColor(out[:, :, :3], cv2.COLOR_RGB2GRAY)
        out[:, :, :3] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return out

    def filter_blur(self) -> None:
        r = simpledialog.askinteger("Gaussian blur", "Radius:", initialvalue=3, minvalue=1, maxvalue=200)
        if r:
            self.apply_to_layer("blur", lambda arr: blur(arr, r))

    def filter_sharpen(self) -> None:
        a = simpledialog.askfloat("Sharpen", "Amount:", initialvalue=1.0, minvalue=0.0, maxvalue=10.0)
        if a is not None:
            self.apply_to_layer("sharpen", lambda arr: sharpen(arr, a))

    def filter_noise(self) -> None:
        a = simpledialog.askfloat("Noise", "Amount 0..1:", initialvalue=0.04, minvalue=0.0, maxvalue=1.0)
        if a is not None:
            self.apply_to_layer("noise", lambda arr: add_noise(arr, a))

    def set_zoom(self, value: float) -> None:
        self.zoom.set(max(0.05, min(16.0, value)))
        self.refresh()

    def fit_to_screen(self) -> None:
        self.update_idletasks()
        w = max(1, self.canvas.winfo_width() - 20)
        h = max(1, self.canvas.winfo_height() - 20)
        self.set_zoom(min(w / self.doc.width, h / self.doc.height))

    def mouse_wheel(self, event) -> None:
        if event.state & 0x0004:
            self.set_zoom(self.zoom.get() * (1.1 if event.delta > 0 else 0.9))
        elif event.state & 0x0001:
            self.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
        else:
            self.canvas.yview_scroll(-3 if event.delta > 0 else 3, "units")

    def batch_process(self) -> None:
        src = filedialog.askdirectory(title="Source folder")
        if not src:
            return
        dst = filedialog.askdirectory(title="Destination folder")
        if not dst:
            return
        width = simpledialog.askinteger("Batch", "Max width px, empty for original:", initialvalue=1920, minvalue=1, maxvalue=50000)
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        count = 0
        for path in Path(src).rglob("*"):
            if path.suffix.lower() not in exts:
                continue
            doc = Document.from_image(path)
            if width and doc.width > width:
                doc.resize_image(width, max(1, round(doc.height * width / doc.width)))
            out = Path(dst) / f"{path.stem}.png"
            doc.export_flat(out)
            count += 1
        messagebox.showinfo("Batch", f"Processed {count} files.")


def main() -> None:
    app = PhotoRedactorApp()
    app.mainloop()
