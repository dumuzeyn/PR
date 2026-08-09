# PhotoRedactor

Personal Photoshop-like raster editor for Windows.

This repository starts with a real working editor and an architecture intended to grow toward the full feature list:

- multi-layer raster documents
- custom project format (`.prdx`)
- PNG/JPEG/WebP/BMP/TIFF import and export
- document presets, recent files, embedded and linked image placement, linked layer update/relink, loading several files as layers and layer export
- dedicated startup workspace for recent projects, file opening, recovery and new-canvas creation before the editor UI appears
- visual new-canvas dialog with screen, social, print and remembered custom presets; clipboard images become the first size preset and can be placed immediately
- centered medium-size startup window, maximized editor, automatic canvas centering and center-preserving zoom
- smooth cached brush strokes, eraser, fill, gradient, editable paragraph/path text layers, crop, rectangular/elliptical/freehand/magnetic/polygon selection, persistent add/subtract/intersect modes, quick selection brush with live result-mask, cursor and image-aware edge refinement, magic wand, color range, move
- layer opacity, visibility, duplication, merge, flatten
- layer locks, raster layer masks, mask thumbnails, thumbnail-click mask channel editing, red-overlay/black-white mask preview, linked/unlinked mask movement, mask painting, density/feather controls, live edge refinement with smart image-aware correction and blend modes
- clipping masks, adjustment-layer clipping and basic layer styles: stroke, drop shadow and outer glow
- non-destructive per-layer filter stack with a visual editor, built-in/importable/exportable presets, preview, reorder controls, per-filter enable/opacity/blend modes/masks and blur, sharpen, noise, median, edge and emboss filters
- real selection mask with select all, invert, feather, smooth, grow, shrink, border/refine, smart edge cleanup/correction, subject/background/sky selection, richer Select and Mask previews/output including edge confidence, opaque-pixel selection, single row/column selection, magnetic edge snapping and saved alpha selections
- visual Free Transform with eight drag handles, selected-pixel transform, perspective transform and live warp preview/presets
- local retouch tools and presets: blur, sharpen, dodge, burn, clone stamp, source-based healing, automatic spot healing and interactive patch tool
- frequency separation into editable tone/texture layers with Linear Light reconstruction and a live three-panel preview
- portrait cleanup with live before/after preview, skin-aware smoothing, texture preservation, tone evening and redness reduction
- content-aware fill for selected areas, patch selection, edge-aware cleanup and red-eye reduction
- editable shape layers for rectangles, ellipses, lines, custom shapes, Bezier curves with draggable control points, polygons, stars and boolean shape combinations
- text layer font, size, block width, wrapping, alignment, line-spacing, tracking, bold/italic/underline, baseline shift, text-on-arc/wave and editable transform handles
- grid and guide overlays plus RGB/Red/Green/Blue/Alpha channel preview
- Russian main interface labels and hover tooltips for the tool palette
- neutral light desktop panels with a dark canvas workspace, a named icon-and-text toolbar, clearly labeled contextual options across the top, tabbed Properties/Layers/History and a compact status bar
- direct Shape/Text selection on the canvas with auto-select, synchronized layers, eight resize handles, modifier-aware resize and arrow-key nudging
- recovery autosave and manual recovery opening
- replayable action recording in JSON v2, action playback and batch execution
- Smart Objects with embedded originals, re-editable transforms and linked-file conflict status
- RAW decoding, ICC profile assignment/conversion, RGB/Lab/CMYK workflows and 8/16/32-bit working channels
- mipmap rendering for distant zoom, scratch-disk cache eviction and optional CUDA detection with CPU fallback
- Python plugin API plus validated external executable filters
- non-destructive generative canvas expansion with a live preview
- metadata/EXIF view, histogram, image statistics and eyedropper
- non-destructive adjustment layers with importable/exportable preset libraries for brightness/contrast, saturation, vibrance, temperature/tint, hue/saturation, exposure, color balance, levels, curves, threshold, posterize, invert and grayscale
- undo/redo
- command-based undo for strokes, moves and layer edits
- tile-based compositor and canvas output: brush, eraser, retouch and healing strokes update only changed 256 px regions
- cached filter stacks, masks and layer effects; opacity and blend changes reuse filtered pixels
- sparse 128 px stroke history, so long diagonal strokes do not store one huge bounding rectangle
- revision-safe background filters that discard stale results after newer document edits
- `.prdx` projects with `manifest.json` and separate layer PNG files, without Base64-packed layers
- resize, canvas resize, trim transparent pixels, reveal all layers, rotate, flip
- brightness, contrast, saturation, hue/saturation, exposure, color balance, levels, curves, threshold, posterize, blur, sharpen, noise, median, edge detect, emboss, grayscale, invert
- batch processing for folders
- executable packaging with PyInstaller

## Run from source

```powershell
.\run.ps1
```

## Dependency policy

PhotoRedactor does not depend on placeholder packages or simulated AI libraries. Runtime dependencies are established projects installed from their real distributions: NumPy, OpenCV, Pillow and rawpy. New dependencies must be checked for a genuine upstream project, an importable implementation and actual use in the packaged application before they are added.

## Build exe

```powershell
.\build_exe.ps1
```

The executable will be created in `dist\PhotoRedactor.exe`.

## Performance diagnostics

Set `PHOTO_REDACTOR_DEBUG_PERF=1` before starting the editor to collect timing counters for composition, dirty tiles, filters, effects and PIL/Tk canvas conversion. Profiling is disabled by default and does not write into the interface.

Run the repeatable rendering benchmark with:

```powershell
python benchmarks\benchmark_rendering.py --include-4k
```

Measured results and the list of correctness fallbacks are recorded in [`PERFORMANCE.md`](PERFORMANCE.md).

## Download the automatic Windows build

Every push to `master` starts the `Build PhotoRedactor for Windows` workflow. Open the repository's `Actions` tab, select the latest successful run and download the `PhotoRedactor-Windows` artifact. It contains the current `PhotoRedactor.exe` built and tested by GitHub.

## Tool panel

- The left side is a scrollable icon-and-name toolbar, so tools remain understandable without memorizing symbols. Shortcuts and explanations also appear in tooltips.
- The contextual options bar is horizontal at the top, starts with `Параметры: <инструмент>` and only shows controls for the active tool. Rare controls stay under `Дополнительно`.
- Gradient controls change with the mode: raster fill, gradient object and texture object do not show each other's irrelevant settings.
- Foreground/background, fill/stroke and gradient colors are shown as clickable color swatches instead of duplicate text buttons.
- `Инструменты -> Настроить панель инструментов...` changes visible tools and their order.
- Click a tool name to show or hide it; drag the row or its visible grip to place it anywhere in the palette order.
- Hidden tools stay available from the main `Инструменты` menu.
- Tool order and visibility are saved in `settings.json`.

## Object interaction

- Shapes are created corner-to-corner. `Shift` keeps proportions, `Alt` creates from the center, and both modifiers can be combined.
- With Move selected, clicking visible Shape/Text content activates the topmost matching layer; hidden layers are ignored and locked layers can be selected but not moved.
- Dragging moves the selected object live. Arrow keys move it by 1 px and `Shift` + arrow by 10 px.
- Shape handles resize on one or two axes. `Shift` keeps proportions and `Alt` resizes around the center.
- One drag creates one compact history command. Move and resize redraw only affected render tiles and do not store full-document snapshots.
- With a selection tool active, right-clicking outside the selected mask clears the selection; right-clicking inside keeps it.

## Plugins and actions

- Put Python plugins in `%APPDATA%\PhotoRedactor\plugins` and reload them from `Действия -> Перезагрузить плагины`.
- The plugin contract and external filter protocol are documented in [`PLUGINS.md`](PLUGINS.md).
- Action v2 files can be replayed on the current document or a group of images. A ready example is in `actions\web_thumbnail.json`.
- `PHOTO_REDACTOR_CACHE_MB` controls the RAM budget for render caches; older arrays automatically move to the scratch folder.
- `PHOTO_REDACTOR_GPU=0` disables optional GPU selection. CPU rendering always remains available.

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

The long requirements file describes a Photoshop-scale application. The roadmap milestones are implemented as local, testable workflows; optional research integrations remain clearly separated from the editor core in `ROADMAP.md`.
