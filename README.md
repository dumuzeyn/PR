# PhotoRedactor

Personal Photoshop-like raster editor for Windows.

This repository starts with a real working editor and an architecture intended to grow toward the full feature list:

- multi-layer raster documents
- custom project format (`.prdx`)
- PNG/JPEG/WebP/BMP/TIFF import and export
- document presets, recent files, embedded and linked image placement, linked layer update/relink, loading several files as layers and layer export
- brush, eraser, fill, gradient, editable paragraph text layers, crop, rectangular/elliptical/freehand/magnetic/polygon selection, quick selection brush with canvas preview, magic wand, color range, move
- layer opacity, visibility, duplication, merge, flatten
- layer locks, raster layer masks, mask thumbnails, thumbnail-click mask channel editing, red-overlay/black-white mask preview, linked/unlinked mask movement, mask painting, mask density/feather controls and blend modes
- clipping masks, adjustment-layer clipping and basic layer styles: stroke, drop shadow and outer glow
- non-destructive per-layer filter stack with a visual editor, built-in/importable/exportable presets, preview, reorder controls, per-filter enable/opacity/blend modes/masks and blur, sharpen, noise, median, edge and emboss filters
- real selection mask with select all, invert, feather, smooth, grow, shrink, border/refine, smart edge cleanup/correction, subject/background/sky selection, richer Select and Mask previews/output including edge confidence, opaque-pixel selection, single row/column selection, magnetic edge snapping and saved alpha selections
- numeric Free Transform, selected-pixel transform, perspective transform and warp presets
- local retouch tools and presets: blur, sharpen, dodge, burn, clone stamp, healing brush and interactive patch tool
- content-aware fill for selected areas, patch selection, edge-aware cleanup and red-eye reduction
- editable shape layers for rectangles, ellipses, lines, Bezier curves, polygons, stars and boolean shape combinations
- text layer font, size, block width, wrapping, alignment, line-spacing and tracking controls
- grid and guide overlays plus RGB/Red/Green/Blue/Alpha channel preview
- Russian main interface labels and hover tooltips for the tool palette
- recovery autosave and manual recovery opening
- simple action recording to JSON logs
- metadata/EXIF view, histogram, image statistics and eyedropper
- non-destructive adjustment layers with presets for brightness/contrast, saturation, hue/saturation, exposure, color balance, levels, curves, threshold, posterize, invert and grayscale
- undo/redo
- command-based undo for strokes, moves and layer edits
- `.prdx` projects with `manifest.json` and separate layer PNG files, without Base64-packed layers
- resize, canvas resize, trim transparent pixels, reveal all layers, rotate, flip
- brightness, contrast, saturation, hue/saturation, exposure, color balance, levels, curves, threshold, posterize, blur, sharpen, noise, median, edge detect, emboss, grayscale, invert
- batch processing for folders
- executable packaging with PyInstaller

## Run from source

```powershell
.\run.ps1
```

## Build exe

```powershell
.\build_exe.ps1
```

The executable will be created in `dist\PhotoRedactor.exe`.

## Tool panel

- The left side is split into a scrollable tool palette and a contextual tool-options panel.
- `Инструменты -> Настроить панель инструментов...` changes visible tools and their order.
- Hidden tools stay available from the main `Инструменты` menu.
- Tool-panel order, visibility and splitter position are saved in `settings.json`.
- The options panel only shows controls that belong to the active tool, such as brush size, opacity, tolerance, paint target, colors or retouch presets.

## Navigation

- Mouse wheel scrolls vertically.
- `Shift` + mouse wheel scrolls horizontally.
- Middle mouse button drags the canvas.
- `Space` + left mouse button drags the canvas.
- The `Hand` tool also drags the canvas.
- Canvas scrollbars move the same viewport state as mouse-wheel and drag panning.
- Zoom keeps the current viewport center, and Fit centers the document in the available work area.
- In Clone/Healing tools, `Alt` + click sets the source point.
- Polygon lasso is finished with a double click.

## Scope

The long requirements file describes a Photoshop-scale application. This codebase treats that as the product target, not a small demo. Features that are not implemented yet are tracked in `ROADMAP.md` instead of being faked in the UI.
