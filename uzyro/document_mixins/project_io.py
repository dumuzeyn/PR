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
from ..precision_pipeline import composite_precision
from ..project_tiles import PROJECT_FORMAT_VERSION, PROJECT_TILE_SIZE, is_tiled_array, read_tiled_array, temporary_project_path, tiled_payload_stats, write_tiled_array

import tifffile


class ProjectIoDocumentMixin:
    def save_project(self, path: str | Path) -> None:
        target = Path(path)
        temporary = temporary_project_path(target)
        manifest = {
            "format": "UZYRO project",
            "format_version": PROJECT_FORMAT_VERSION,
            "storage": {"format": "tiles-v1", "tile_size": PROJECT_TILE_SIZE},
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "color_model": self.color_model,
            "bit_depth": self.bit_depth,
            "background": list(self.background),
            "active_layer": self.active_layer,
            "selection": None,
            "saved_selections": {},
            "metadata": self.metadata,
            "layers": [],
        }
        temporary.unlink(missing_ok=True)
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
                if self.selection_mask is not None:
                    manifest["selection"] = write_tiled_array(zf, "document/selection", self.selection_mask)
                for i, (name, mask) in enumerate(self.saved_selections.items()):
                    manifest["saved_selections"][name] = write_tiled_array(zf, f"document/selections/{i:04d}", mask)
                for i, layer in enumerate(self.layers):
                    prefix = f"layers/{i:04d}"
                    manifest["layers"].append(
                        {
                            "id": layer.id,
                            "name": layer.name,
                            "x": layer.x,
                            "y": layer.y,
                            "opacity": layer.opacity,
                            "visible": layer.visible,
                            "locked": layer.locked,
                            "mask": None if layer.mask is None else write_tiled_array(zf, f"{prefix}/mask", layer.mask),
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
                            "generation_data": layer.generation_data,
                            "group_id": layer.group_id,
                            "smart_source": None if layer.smart_source is None else write_tiled_array(zf, f"{prefix}/smart", layer.smart_source),
                            "transform_data": layer.transform_data,
                            "transform_source": None if layer.transform_source is None else write_tiled_array(zf, f"{prefix}/transform", layer.transform_source),
                            "transform_mask_source": None if layer.transform_mask_source is None else write_tiled_array(zf, f"{prefix}/transform_mask", layer.transform_mask_source),
                            "working_pixels": None if layer.working_pixels is None else write_tiled_array(zf, f"{prefix}/working", layer.working_pixels),
                            "working_model": layer.working_model,
                            "working_depth": layer.working_depth,
                            "pixels": write_tiled_array(zf, f"{prefix}/pixels", layer.pixels),
                        }
                    )
                zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False).encode("utf-8"))
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        self.path = str(target)
        self.dirty = False

    @classmethod
    def open_project(cls, path: str | Path, progress=None) -> "Document":
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
            total_layers = len(manifest["layers"])

            def payload(value, mask: bool = False):
                if not value:
                    return None
                if is_tiled_array(value):
                    return read_tiled_array(zf, value)
                decoded = pil_to_rgba_array(Image.open(io.BytesIO(zf.read(value))))
                return decoded[:, :, 0] if mask else decoded

            for layer_index, raw in enumerate(manifest["layers"], 1):
                pixels = payload(raw["pixels"])
                doc.layers.append(
                    Layer(
                        name=raw["name"],
                        pixels=pixels,
                        x=int(raw.get("x", 0)),
                        y=int(raw.get("y", 0)),
                        opacity=float(raw.get("opacity", 1.0)),
                        visible=bool(raw.get("visible", True)),
                        locked=bool(raw.get("locked", False)),
                        mask=payload(raw.get("mask"), True),
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
                        generation_data=raw.get("generation_data"),
                        group_id=raw.get("group_id"),
                        smart_source=payload(raw.get("smart_source")),
                        transform_data=raw.get("transform_data"),
                        transform_source=payload(raw.get("transform_source")),
                        transform_mask_source=payload(raw.get("transform_mask_source"), True),
                        working_pixels=None if not raw.get("working_pixels") else (
                            read_tiled_array(zf, raw["working_pixels"])
                            if is_tiled_array(raw["working_pixels"])
                            else np.load(io.BytesIO(zf.read(raw["working_pixels"])), allow_pickle=False)
                        ),
                        working_model=raw.get("working_model", "RGBA"),
                        working_depth=int(raw.get("working_depth", manifest.get("bit_depth", 8))),
                        id=raw.get("id", uuid.uuid4().hex),
                    )
                )
                if progress is not None:
                    progress(layer_index, total_layers, raw["name"])
            doc.active_layer = min(int(manifest.get("active_layer", 0)), max(0, len(doc.layers) - 1))
            if manifest.get("selection"):
                doc.selection_mask = payload(manifest["selection"], True)
            doc.saved_selections = {}
            for name, selection_path in manifest.get("saved_selections", {}).items():
                doc.saved_selections[name] = payload(selection_path, True)
        doc.path = str(path)
        return doc

    @staticmethod
    def project_storage_info(path: str | Path) -> dict[str, int | str]:
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        totals = {"tiles": 0, "bytes": 0}

        def visit(value) -> None:
            if is_tiled_array(value):
                stats = tiled_payload_stats(value)
                totals["tiles"] += stats["tiles"]
                totals["bytes"] += stats["bytes"]
                return
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(manifest)
        return {
            "format": str(manifest.get("storage", {}).get("format", "legacy")),
            "version": int(manifest.get("format_version", 1)),
            **totals,
        }

    def composite(self, checker: bool = False) -> np.ndarray:
        with profiler.measure("render.composite.reference"):
            if self.bit_depth != 8 or any(layer.working_pixels is not None for layer in self.layers):
                return display_rgba(self.composite_precision(checker))
            out = checker_background(self.width, self.height).copy() if checker else blank_rgba(self.width, self.height, (0, 0, 0, 0))
            previous_alpha: np.ndarray | None = None
            for layer in self.layers:
                if not layer.visible:
                    continue
                if layer.kind == "adjustment" and layer.adjustment is not None:
                    clipping_mask = previous_alpha if layer.clipping and previous_alpha is not None else None
                    apply_adjustment_layer(out, layer, clipping_mask)
                    continue
                layer_pixels = render_layer_pixels(layer)
                alpha_mask = effective_layer_mask(layer) if layer.mask_enabled else None
                if layer.clipping and previous_alpha is not None:
                    clipping_mask = document_alpha_to_layer_mask(previous_alpha, layer)
                    alpha_mask = clipping_mask if alpha_mask is None else np.minimum(alpha_mask, clipping_mask)
                effects, styled_pixels = render_layer_style(layer, layer_pixels)
                for pixels, x, y, opacity, blend_mode in effects:
                    alpha_blend_inplace(out, pixels, x, y, opacity, None, 1.0, blend_mode)
                alpha_blend_inplace(out, styled_pixels, layer.x, layer.y, layer.opacity, alpha_mask, layer.mask_density, layer.blend_mode)
                previous_alpha = layer_alpha_canvas(self, layer, styled_pixels)
            return out

    def composite_precision(self, checker: bool = False) -> np.ndarray:
        return composite_precision(self, checker)

    def export_flat(self, path: str | Path, quality: int = 95) -> None:
        precise = self.composite_precision(checker=False)
        composite = display_rgba(precise)
        img = rgba_array_to_pil(composite)
        suffix = Path(path).suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            img.convert("RGB").save(path, quality=max(1, min(100, int(quality))), subsampling=0)
        elif suffix == ".webp":
            img.save(path, quality=max(1, min(100, int(quality))))
        elif suffix in {".tif", ".tiff"} and self.bit_depth in {16, 32}:
            pixels = quantize_rgba(precise, self.bit_depth)
            settings = color_settings(self.metadata)
            encoded = settings.get("icc_base64")
            icc = base64.b64decode(encoded) if encoded else None
            tifffile.imwrite(
                path,
                pixels,
                photometric="rgb",
                extrasamples=["unassalpha"],
                resolution=(float(self.dpi), float(self.dpi)),
                resolutionunit="INCH",
                iccprofile=icc,
                metadata=None,
            )
        elif suffix == ".png" and self.bit_depth == 16:
            bgra = cv2.cvtColor(quantize_rgba(precise, 16), cv2.COLOR_RGBA2BGRA)
            if not cv2.imwrite(str(path), bgra):
                raise OSError(f"Could not write PNG: {path}")
        else:
            settings = color_settings(self.metadata)
            encoded = settings.get("icc_base64")
            save_options = {"icc_profile": base64.b64decode(encoded)} if encoded else {}
            img.save(path, **save_options)
        self.dirty = False

    def set_bit_depth(self, bit_depth: int) -> None:
        bit_depth = int(bit_depth)
        if bit_depth not in BIT_DEPTHS:
            raise ValueError("Bit depth must be 8, 16 or 32")
        for layer in self.layers:
            if layer.kind != "adjustment" and layer.pixels.size:
                layer.set_working_rgba(layer.working_rgba(), bit_depth, layer.working_model)
        self.bit_depth = bit_depth
        self.metadata["bit_depth"] = bit_depth
        self.dirty = True

    def set_color_model(self, color_model: str) -> None:
        if color_model not in COLOR_MODELS:
            raise ValueError(f"Unsupported color model: {color_model}")
        for layer in self.layers:
            if layer.kind == "adjustment" or layer.pixels.size == 0:
                continue
            layer.set_working_rgba(layer.working_rgba(), self.bit_depth, color_model)
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
            converted = convert_icc(
                layer.working_rgba(),
                source,
                destination,
                settings.get("rendering_intent", "perceptual"),
                bool(settings.get("black_point_compensation", True)),
            )
            layer.set_working_rgba(converted, self.bit_depth, layer.working_model)
        self.assign_color_profile(destination)
        self.dirty = True
