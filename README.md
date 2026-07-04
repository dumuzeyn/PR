# PhotoRedactor

Personal Photoshop-like raster editor for Windows.

This repository starts with a real working editor and an architecture intended to grow toward the full feature list:

- multi-layer raster documents
- custom project format (`.prdx`)
- PNG/JPEG/WebP/BMP/TIFF import and export
- document presets, recent files, embedded and linked image placement, linked layer update/relink, loading several files as layers and layer export
- brush, eraser, fill, gradient, editable paragraph text layers, crop, rectangular/elliptical/freehand/magnetic/polygon selection, quick selection brush, magic wand, color range, move
- layer opacity, visibility, duplication, merge, flatten
- layer locks, raster layer masks, mask painting, mask density/feather controls and blend modes
- clipping masks and basic layer styles: stroke, drop shadow and outer glow
- non-destructive per-layer filter stack with blur, sharpen, noise, median, edge and emboss filters
- real selection mask with select all, invert, feather, smooth, grow, shrink, border/refine, opaque-pixel selection, single row/column selection, magnetic edge snapping and saved alpha selections
- numeric Free Transform and perspective transform for active layers
- local retouch tools: blur, sharpen, dodge, burn, clone stamp and healing brush
- content-aware fill for selected areas, patch selection and red-eye reduction
- editable shape layers for rectangles, ellipses, lines, polygons and stars
- text layer font, size, block width, wrapping, alignment and line-spacing controls
- grid and guide overlays plus RGB/Red/Green/Blue/Alpha channel preview
- Russian main interface labels and hover tooltips for the tool palette
- recovery autosave and manual recovery opening
- simple action recording to JSON logs
- metadata/EXIF view, histogram, image statistics and eyedropper
- non-destructive adjustment layers for brightness/contrast, saturation, levels, curves, invert and grayscale
- undo/redo
- command-based undo for strokes, moves and layer edits
- `.prdx` projects with `manifest.json` and separate layer PNG files, without Base64-packed layers
- resize, canvas resize, trim transparent pixels, reveal all layers, rotate, flip
- brightness, contrast, saturation, levels, curves, blur, sharpen, noise, median, edge detect, emboss, grayscale, invert
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

## Navigation

- Mouse wheel scrolls vertically.
- `Shift` + mouse wheel scrolls horizontally.
- Middle mouse button drags the canvas.
- `Space` + left mouse button drags the canvas.
- The `Hand` tool also drags the canvas.
- In Clone/Healing tools, `Alt` + click sets the source point.
- Polygon lasso is finished with a double click.

## Scope

The long requirements file describes a Photoshop-scale application. This codebase treats that as the product target, not a small demo. Features that are not implemented yet are tracked in `ROADMAP.md` instead of being faked in the UI.
