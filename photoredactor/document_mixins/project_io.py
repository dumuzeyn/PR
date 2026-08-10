from __future__ import annotations

from ..core_shared import *
from ..layer import Layer
from ..geometry_ops import *
from ..selection_ops import *
from ..filter_ops import *
from ..render_ops import *
from ..retouch_ops import *
from ..text_ops import *
from ..shape_ops import *
from ..content_ops import *
from ..adjustment_ops import *


class ProjectIoDocumentMixin:
    def save_project(self, path: str | Path) -> None:
        manifest = {
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "color_model": self.color_model,
            "bit_depth": self.bit_depth,
            "background": list(self.background),
            "active_layer": self.active_layer,
            "selection": "selection.png" if self.selection_mask is not None else None,
            "saved_selections": {},
            "metadata": self.metadata,
            "layers": [],
        }
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if self.selection_mask is not None:
                buf = io.BytesIO()
                rgba_array_to_pil(np.dstack([self.selection_mask] * 4)).save(buf, "PNG")
                zf.writestr("selection.png", buf.getvalue())
            for i, (name, mask) in enumerate(self.saved_selections.items()):
                selection_path = f"selections/{i:04d}.png"
                mask_buf = io.BytesIO()
                rgba_array_to_pil(np.dstack([mask] * 4)).save(mask_buf, "PNG")
                zf.writestr(selection_path, mask_buf.getvalue())
                manifest["saved_selections"][name] = selection_path
            for i, layer in enumerate(self.layers):
                layer_path = f"layers/{i:04d}.png"
                manifest["layers"].append(
                    {
                        "id": layer.id,
                        "name": layer.name,
                        "x": layer.x,
                        "y": layer.y,
                        "opacity": layer.opacity,
                        "visible": layer.visible,
                        "locked": layer.locked,
                        "mask": f"masks/{i:04d}.png" if layer.mask is not None else None,
                        "mask_enabled": layer.mask_enabled,
                        "mask_linked": layer.mask_linked,
                        "mask_density": layer.mask_density,
                        "mask_feather": layer.mask_feather,
                        "blend_mode": layer.blend_mode,
                        "clipping": layer.clipping,
                        "effects": layer.effects,
                        "filters": layer.filters,
                        "kind": layer.kind,
                        "text_data": layer.text_data,
                        "shape_data": layer.shape_data,
                        "adjustment": layer.adjustment,
                        "smart_data": layer.smart_data,
                        "smart_source": f"smart/{i:04d}.png" if layer.smart_source is not None else None,
                        "transform_data": layer.transform_data,
                        "transform_source": f"transforms/{i:04d}.png" if layer.transform_source is not None else None,
                        "transform_mask_source": f"transform_masks/{i:04d}.png" if layer.transform_mask_source is not None else None,
                        "working_pixels": f"working/{i:04d}.npy" if layer.working_pixels is not None else None,
                        "working_model": layer.working_model,
                        "pixels": layer_path,
                    }
                )
                buf = io.BytesIO()
                rgba_array_to_pil(layer.pixels).save(buf, "PNG")
                zf.writestr(layer_path, buf.getvalue())
                if layer.smart_source is not None:
                    smart_buf = io.BytesIO()
                    rgba_array_to_pil(layer.smart_source).save(smart_buf, "PNG")
                    zf.writestr(f"smart/{i:04d}.png", smart_buf.getvalue())
                if layer.transform_source is not None:
                    transform_buf = io.BytesIO()
                    rgba_array_to_pil(layer.transform_source).save(transform_buf, "PNG")
                    zf.writestr(f"transforms/{i:04d}.png", transform_buf.getvalue())
                if layer.transform_mask_source is not None:
                    transform_mask_buf = io.BytesIO()
                    rgba_array_to_pil(np.dstack([layer.transform_mask_source] * 4)).save(transform_mask_buf, "PNG")
                    zf.writestr(f"transform_masks/{i:04d}.png", transform_mask_buf.getvalue())
                if layer.working_pixels is not None:
                    working_buf = io.BytesIO()
                    np.save(working_buf, layer.working_pixels, allow_pickle=False)
                    zf.writestr(f"working/{i:04d}.npy", working_buf.getvalue())
                if layer.mask is not None:
                    mask_buf = io.BytesIO()
                    rgba_array_to_pil(np.dstack([layer.mask] * 4)).save(mask_buf, "PNG")
                    zf.writestr(f"masks/{i:04d}.png", mask_buf.getvalue())
            zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False).encode("utf-8"))
        self.path = str(path)
        self.dirty = False

    @classmethod
    def open_project(cls, path: str | Path) -> "Document":
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            if "manifest.json" not in names and "document.json" in names:
                data = json.loads(zf.read("document.json").decode("utf-8"))
                doc = cls.restore(data)
                doc.path = str(path)
                return doc
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            doc = cls(
                width=int(manifest["width"]),
                height=int(manifest["height"]),
                dpi=int(manifest.get("dpi", 300)),
                color_model=manifest.get("color_model", "RGBA"),
                bit_depth=int(manifest.get("bit_depth", 8)),
                background=tuple(manifest.get("background", [255, 255, 255, 255])),
                metadata=manifest.get("metadata", {}),
            )
            doc.layers = []
            for raw in manifest["layers"]:
                pixels = pil_to_rgba_array(Image.open(io.BytesIO(zf.read(raw["pixels"]))))
                doc.layers.append(
                    Layer(
                        name=raw["name"],
                        pixels=pixels,
                        x=int(raw.get("x", 0)),
                        y=int(raw.get("y", 0)),
                        opacity=float(raw.get("opacity", 1.0)),
                        visible=bool(raw.get("visible", True)),
                        locked=bool(raw.get("locked", False)),
                        mask=None if raw.get("mask") is None else pil_to_rgba_array(Image.open(io.BytesIO(zf.read(raw["mask"]))))[:, :, 0],
                        mask_enabled=bool(raw.get("mask_enabled", True)),
                        mask_linked=bool(raw.get("mask_linked", True)),
                        mask_density=float(raw.get("mask_density", 1.0)),
                        mask_feather=float(raw.get("mask_feather", 0.0)),
                        blend_mode=raw.get("blend_mode", "Normal"),
                        clipping=bool(raw.get("clipping", False)),
                        effects=raw.get("effects", {}),
                        filters=raw.get("filters", []),
                        kind=raw.get("kind", "raster"),
                        text_data=raw.get("text_data"),
                        shape_data=raw.get("shape_data"),
                        adjustment=raw.get("adjustment"),
                        smart_data=raw.get("smart_data"),
                        smart_source=None if not raw.get("smart_source") else pil_to_rgba_array(Image.open(io.BytesIO(zf.read(raw["smart_source"])))),
                        transform_data=raw.get("transform_data"),
                        transform_source=None if not raw.get("transform_source") else pil_to_rgba_array(Image.open(io.BytesIO(zf.read(raw["transform_source"])))),
                        transform_mask_source=None if not raw.get("transform_mask_source") else pil_to_rgba_array(Image.open(io.BytesIO(zf.read(raw["transform_mask_source"]))))[:, :, 0],
                        working_pixels=None if not raw.get("working_pixels") else np.load(io.BytesIO(zf.read(raw["working_pixels"])), allow_pickle=False),
                        working_model=raw.get("working_model", "RGBA"),
                        id=raw.get("id", uuid.uuid4().hex),
                    )
                )
            doc.active_layer = min(int(manifest.get("active_layer", 0)), max(0, len(doc.layers) - 1))
            if manifest.get("selection"):
                doc.selection_mask = pil_to_rgba_array(Image.open(io.BytesIO(zf.read(manifest["selection"]))))[:, :, 0]
            doc.saved_selections = {}
            for name, selection_path in manifest.get("saved_selections", {}).items():
                doc.saved_selections[name] = pil_to_rgba_array(Image.open(io.BytesIO(zf.read(selection_path))))[:, :, 0]
        doc.path = str(path)
        return doc

    def composite(self, checker: bool = False) -> np.ndarray:
        with profiler.measure("render.composite.reference"):
            out = checker_background(self.width, self.height).copy() if checker else blank_rgba(self.width, self.height, (0, 0, 0, 0))
            previous_alpha: np.ndarray | None = None
            for layer in self.layers:
                if layer.visible:
                    if layer.kind == "adjustment" and layer.adjustment is not None:
                        clipping_mask = previous_alpha if layer.clipping and previous_alpha is not None else None
                        apply_adjustment_layer(out, layer, clipping_mask)
                    else:
                        layer_pixels = render_layer_pixels(layer)
                        alpha_mask = effective_layer_mask(layer) if layer.mask_enabled else None
                        if layer.clipping and previous_alpha is not None:
                            clipping_mask = document_alpha_to_layer_mask(previous_alpha, layer)
                            alpha_mask = clipping_mask if alpha_mask is None else np.minimum(alpha_mask, clipping_mask)
                        for pixels, x, y, opacity, blend_mode in render_layer_effects(layer, layer_pixels):
                            alpha_blend_inplace(out, pixels, x, y, opacity, None, 1.0, blend_mode)
                        alpha_blend_inplace(out, layer_pixels, layer.x, layer.y, layer.opacity, alpha_mask, layer.mask_density, layer.blend_mode)
                        previous_alpha = layer_alpha_canvas(self, layer, layer_pixels)
            return out

    def export_flat(self, path: str | Path, quality: int = 95) -> None:
        composite = self.composite(checker=False)
        img = rgba_array_to_pil(composite)
        suffix = Path(path).suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            img.convert("RGB").save(path, quality=max(1, min(100, int(quality))), subsampling=0)
        elif suffix == ".webp":
            img.save(path, quality=max(1, min(100, int(quality))))
        elif suffix in {".tif", ".tiff"} and self.bit_depth == 16:
            bgra = cv2.cvtColor(composite.astype(np.uint16) * 257, cv2.COLOR_RGBA2BGRA)
            if not cv2.imwrite(str(path), bgra):
                raise OSError(f"Could not write TIFF: {path}")
        else:
            img.save(path)
        self.dirty = False

    def set_bit_depth(self, bit_depth: int) -> None:
        bit_depth = int(bit_depth)
        if bit_depth not in BIT_DEPTHS:
            raise ValueError("Bit depth must be 8, 16 or 32")
        for layer in self.layers:
            if layer.working_model == "RGBA":
                source = layer.working_pixels if layer.working_pixels is not None else layer.pixels
                layer.working_pixels = None if bit_depth == 8 else quantize_rgba(source, bit_depth)
                layer.pixels = display_rgba(source)
            layer.pixels_revision += 1
        self.bit_depth = bit_depth
        self.metadata["bit_depth"] = bit_depth
        self.dirty = True

    def set_color_model(self, color_model: str) -> None:
        if color_model not in COLOR_MODELS:
            raise ValueError(f"Unsupported color model: {color_model}")
        for layer in self.layers:
            if layer.kind == "adjustment" or layer.pixels.size == 0:
                continue
            if color_model == "Lab":
                layer.working_pixels = rgb_to_lab(layer.pixels)
            elif color_model == "CMYK":
                alpha = layer.pixels[:, :, 3].astype(np.float32) / 255.0
                layer.working_pixels = np.dstack((rgb_to_cmyk(layer.pixels), alpha))
            else:
                layer.working_pixels = None if self.bit_depth == 8 else quantize_rgba(layer.pixels, self.bit_depth)
            layer.working_model = color_model
            layer.pixels_revision += 1
        self.color_model = color_model
        self.metadata["color_model"] = color_model
        self.dirty = True

    def assign_color_profile(self, profile: str | Path | bytes | None) -> None:
        settings = color_settings(self.metadata)
        settings["profile_name"] = profile_name(profile)
        raw = profile_bytes(profile)
        if raw is None:
            settings.pop("icc_base64", None)
        else:
            settings["icc_base64"] = base64.b64encode(raw).decode("ascii")
        self.dirty = True

    def convert_color_profile(self, destination: str | Path | bytes | None) -> None:
        settings = color_settings(self.metadata)
        encoded = settings.get("icc_base64")
        source: str | bytes | None = base64.b64decode(encoded) if encoded else settings.get("profile_name", "sRGB")
        for layer in self.layers:
            if layer.kind == "adjustment" or layer.pixels.size == 0:
                continue
            layer.pixels = convert_icc(layer.pixels, source, destination)
            layer.touch_pixels()
        self.assign_color_profile(destination)
        self.dirty = True
