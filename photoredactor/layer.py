from __future__ import annotations

from .core_shared import *


@dataclass
class Layer:
    name: str
    pixels: np.ndarray
    x: int = 0
    y: int = 0
    opacity: float = 1.0
    visible: bool = True
    locked: bool = False
    mask: np.ndarray | None = None
    mask_enabled: bool = True
    mask_linked: bool = True
    mask_density: float = 1.0
    mask_feather: float = 0.0
    blend_mode: str = "Normal"
    clipping: bool = False
    effects: dict[str, Any] = field(default_factory=dict)
    filters: list[dict[str, Any]] = field(default_factory=list)
    kind: str = "raster"
    text_data: dict[str, Any] | None = None
    shape_data: dict[str, Any] | None = None
    adjustment: dict[str, Any] | None = None
    smart_data: dict[str, Any] | None = None
    generation_data: dict[str, Any] | None = None
    group_id: str | None = None
    smart_source: np.ndarray | None = field(default=None, repr=False, compare=False)
    transform_data: dict[str, Any] | None = None
    transform_source: np.ndarray | None = field(default=None, repr=False, compare=False)
    transform_mask_source: np.ndarray | None = field(default=None, repr=False, compare=False)
    working_pixels: np.ndarray | None = field(default=None, repr=False, compare=False)
    working_model: str = "RGBA"
    working_depth: int = 8
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    pixels_revision: int = field(default=0, repr=False, compare=False)
    mask_revision: int = field(default=0, repr=False, compare=False)

    def touch_pixels(self) -> None:
        self.pixels_revision += 1
        if self.working_pixels is not None:
            precise = working_to_rgba(self.working_pixels, self.working_model)
            previous_display = display_rgba(precise)
            if previous_display.shape != self.pixels.shape:
                self.working_pixels = rgba_to_working(self.pixels, self.working_model, self.working_depth)
                return
            changed = np.any(previous_display != self.pixels, axis=2)
            if np.any(changed):
                edited = normalize_rgba(self.pixels)
                precise[changed] = edited[changed]
                self.working_pixels = rgba_to_working(precise, self.working_model, self.working_depth)

    def working_rgba(self) -> np.ndarray:
        source = self.pixels if self.working_pixels is None else working_to_rgba(self.working_pixels, self.working_model)
        return normalize_rgba(source)

    def set_working_rgba(self, pixels: np.ndarray, bit_depth: int | None = None, color_model: str | None = None) -> None:
        if bit_depth is not None:
            self.working_depth = int(bit_depth)
        if color_model is not None:
            self.working_model = str(color_model)
        normalized = normalize_rgba(pixels)
        self.working_pixels = None if self.working_depth == 8 and self.working_model == "RGBA" else rgba_to_working(
            normalized,
            self.working_model,
            self.working_depth,
        )
        self.pixels = display_rgba(normalized)
        self.pixels_revision += 1

    def replace_pixels(self, pixels: np.ndarray) -> None:
        """Replace a raster result without collapsing an existing high-precision layer."""
        if self.working_pixels is None:
            self.pixels = display_rgba(pixels)
            self.pixels_revision += 1
            return
        self.set_working_rgba(pixels, self.working_depth, self.working_model)

    def sync_display_from_working(self) -> None:
        if self.working_pixels is not None:
            self.pixels = display_rgba(working_to_rgba(self.working_pixels, self.working_model))
        self.pixels_revision += 1

    def touch_mask(self) -> None:
        self.mask_revision += 1

    def clone(self) -> "Layer":
        return Layer(
            name=f"{self.name} copy",
            pixels=self.pixels.copy(),
            x=self.x,
            y=self.y,
            opacity=self.opacity,
            visible=self.visible,
            locked=self.locked,
            mask=None if self.mask is None else self.mask.copy(),
            mask_enabled=self.mask_enabled,
            mask_linked=self.mask_linked,
            mask_density=self.mask_density,
            mask_feather=self.mask_feather,
            blend_mode=self.blend_mode,
            clipping=self.clipping,
            effects=json.loads(json.dumps(self.effects)),
            filters=json.loads(json.dumps(self.filters)),
            kind=self.kind,
            text_data=None if self.text_data is None else json.loads(json.dumps(self.text_data)),
            shape_data=None if self.shape_data is None else json.loads(json.dumps(self.shape_data)),
            adjustment=None if self.adjustment is None else dict(self.adjustment),
            smart_data=None if self.smart_data is None else json.loads(json.dumps(self.smart_data, ensure_ascii=False)),
            generation_data=None if self.generation_data is None else json.loads(json.dumps(self.generation_data, ensure_ascii=False)),
            group_id=self.group_id,
            smart_source=None if self.smart_source is None else self.smart_source.copy(),
            transform_data=None if self.transform_data is None else json.loads(json.dumps(self.transform_data)),
            transform_source=None if self.transform_source is None else self.transform_source.copy(),
            transform_mask_source=None if self.transform_mask_source is None else self.transform_mask_source.copy(),
            working_pixels=None if self.working_pixels is None else self.working_pixels.copy(),
            working_model=self.working_model,
            working_depth=self.working_depth,
        )

__all__ = [name for name in globals() if not name.startswith("__")]
