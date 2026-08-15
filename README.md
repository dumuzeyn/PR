# UZYRO

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
- professional gradients with sRGB, Linear RGB and OKLab interpolation, ordered dithering anchored to document coordinates, deterministic noise gradients and editable color/opacity stops
- layer opacity, visibility, duplication, merge, flatten
- layer locks, raster layer masks, mask thumbnails, thumbnail-click mask channel editing, red-overlay/black-white mask preview, linked/unlinked mask movement, mask painting, density/feather controls, live edge refinement with smart image-aware correction and blend modes
- clipping masks, adjustment-layer clipping and basic layer styles: stroke, drop shadow and outer glow
- non-destructive per-layer filter stack with a visual editor, built-in/importable/exportable presets, preview, reorder controls, per-filter enable/opacity/blend modes/masks and blur, sharpen, noise, median, edge and emboss filters
- real selection masks with contextual select all, invert, feather, smooth, grow, shrink, border/refine, perceptual multi-layer Magic Wand, multi-sample Color Range preview, edge-aware Quick Selection, backend-based subject/object ROI/background/sky selection, and a full Select and Mask workspace with standard preview modes, Smart Radius and soft-alpha outputs
- visual Free Transform with eight drag handles, Shift/Alt/Ctrl modifiers, Enter/Escape, selected-pixel and perspective transforms, plus live editable 3x3/4x4/5x5/custom Warp grids with split/remove commands
- local retouch tools and presets: blur, sharpen, dodge, burn, clone stamp, source-based healing, automatic spot healing and interactive patch tool
- frequency separation into editable tone/texture layers with Linear Light reconstruction and a live three-panel preview
- portrait cleanup with live before/after preview, skin-aware smoothing, texture preservation, tone evening and redness reduction
- content-aware fill for selected areas, patch selection, edge-aware cleanup and red-eye reduction
- editable shape layers for rectangles, ellipses, lines, custom shapes, Bezier curves with draggable control points, polygons and stars; boolean compounds keep editable source contours with a live operation preview
- text layer font, size, block width, wrapping, alignment, line-spacing, tracking, bold/italic/underline, baseline shift, text-on-arc/wave and editable transform handles
- grid and guide overlays plus RGB/Red/Green/Blue/Alpha channel preview
- Russian main interface labels and hover tooltips for the tool palette
- neutral light desktop panels with a dark canvas workspace, a named icon-and-text toolbar, clearly labeled contextual options across the top, tabbed Properties/Layers/History and a compact status bar
- direct Shape/Text selection on the canvas with auto-select, synchronized layers, eight resize handles, modifier-aware resize and arrow-key nudging
- recovery autosave and manual recovery opening
- action recording in validated JSON v3 with backward-compatible v2 playback, conditions, stops, per-step error policies and a persistent batch queue
- Smart Objects with embedded originals, re-editable transforms and linked-file conflict status
- native 16-bit RAW decoding; high-precision 8/16/32-bit layers, composition, filters, history and geometry; ICC assignment/conversion, soft proof, gamut warning, print preflight and profile-embedded CMYK TIFF export
- direct tile-by-tile distant-zoom rendering without a full-size composite, scratch-disk cache eviction and measured OpenCL GPU kernels with adaptive CPU fallback
- isolated Plugin SDK v2 with manifests, explicit permissions, filters, document actions, importers, exporters and validated external executables
- downloadable local generative models powered by the real stable-diffusion.cpp engine, with CUDA/Vulkan/CPU selection, verified downloads, LCM speed profiles, masked fill, outpaint, prompts, negative prompts, seeds, variants, history and undo/redo
- tested PSD/PSB compatibility import and export for ordered raster layers, offsets, opacity, visibility, locks, clipping, blend modes, masks, DPI, ICC profiles and 8/16/32-bit document depth; unsupported Photoshop-only structures are reported instead of silently advertised as editable
- metadata/EXIF view, histogram, image statistics and eyedropper
- non-destructive adjustment layers with importable/exportable preset libraries for brightness/contrast, saturation, vibrance, temperature/tint, hue/saturation, exposure, color balance, editable black and white channel mixing, levels, curves, threshold, posterize and invert
- a deterministic non-destructive layer style stack with ten editable effects and live preview
- separate destructive and smart-filter workflows for Gaussian and motion blur, unsharp and smart sharpening, noise reduction/addition and high pass
- undo/redo
- command-based undo for strokes, moves and layer edits
- tile-based compositor and canvas output: brush, eraser, retouch and healing strokes update only changed 256 px regions
- cached filter stacks, masks and layer effects; opacity and blend changes reuse filtered pixels
- sparse 128 px stroke history, so long diagonal strokes do not store one huge bounding rectangle
- revision-safe background filters that discard stale results after newer document edits
- atomic tiled `.prdx` projects with per-tile SHA-256 integrity, high-precision payloads, backward compatibility and background loading of complete documents
- resize, canvas resize, trim transparent pixels, reveal all layers, rotate, flip
- brightness, contrast, saturation, hue/saturation, exposure, color balance, levels, curves, threshold, posterize, blur, sharpen, noise, median, edge detect, emboss, grayscale, invert
- batch processing for folders
- executable packaging with PyInstaller

## Run from source

```powershell
.\run.ps1
```

## Dependency policy

UZYRO does not depend on placeholder packages or simulated AI libraries. Runtime dependencies are established projects installed from their real distributions: NumPy, OpenCV, Pillow, rawpy and psd-tools. New dependencies must be checked for a genuine upstream project, an importable implementation and actual use in the packaged application before they are added.

## Architecture

Public facade modules remain stable while editor, document and rendering behavior is divided by responsibility. Python sources are limited to 500 lines by an automated CI check. The module map and extension rules are documented in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Build exe

```powershell
.\build_exe.ps1
```

The executable will be created in `dist\UZYRO.exe`.

## Run tests

```powershell
.\run_tests.ps1
```

The script discovers every `tests/test_*.py` file automatically. Logic tests run together; every Tk interface test runs in a separate Python process on Windows to prevent Tcl/Tk state from leaking between scenarios. Visual regression covers 35 stored golden scenes, and 18 real-image fixtures validate selection and retouch quality with SSIM, MAE, IoU, precision, recall and boundary F1 metrics.

Intentional golden updates use `python benchmarks\update_goldens.py`; real-image fixture updates use `python benchmarks\update_quality_fixtures.py`. Review changed PNG files before committing them.

## Performance diagnostics

Set `UZYRO_DEBUG_PERF=1` before starting the editor to collect timing counters for composition, dirty tiles, filters, effects and PIL/Tk canvas conversion. Profiling is disabled by default and does not write into the interface.

Run the repeatable rendering benchmark with:

```powershell
python benchmarks\benchmark_rendering.py --include-4k
```

Measured results and the list of correctness fallbacks are recorded in [`PERFORMANCE.md`](PERFORMANCE.md).

## Generative AI

The EXE does not contain model weights. Open the top-level `Модели` menu after `Вид`, choose `Локальные модели...`, and download a model only when it is needed. UZYRO detects NVIDIA CUDA, Vulkan or CPU, downloads the matching official stable-diffusion.cpp Windows engine, verifies the engine and model SHA-256 values, and stores them in the current user's local application data. Realistic Vision 5.1 Inpainting plus its LCM accelerator use about 2.1 GiB; engine size depends on the selected backend.

`Правка -> Генеративная заливка` uses the current selection. `Изображение -> Генеративное расширение холста` adds content around the document. The prompt describes what must appear, while the negative prompt excludes unwanted content. `Быстро` and `Баланс` use the open LCM-LoRA accelerator at 4 or 6 steps; `Качество` uses regular DPM++ 2M sampling, and `Вручную` exposes all values. The local model stays loaded in a private `127.0.0.1` process between variants and is stopped when UZYRO exits.

Both workflows run locally in the background, support cancellation, one to four variants and exact seed repeat, preserve pixels outside the mask, and apply the result as a separate undoable layer. UZYRO does not require an API key and does not send the document to a cloud image provider.

## Download the automatic Windows build

Every push to `master` starts the `Build UZYRO for Windows` workflow. Open the repository's `Actions` tab, select the latest successful run and download the `UZYRO-Windows` artifact. It contains the current `UZYRO.exe` built and tested by GitHub.

## PSD and PSB compatibility

PSD and PSB files can be opened from the regular file dialog. Pixel layers retain their order, names, coordinates, opacity, visibility, locks, clipping, supported blend modes and raster masks. DPI, ICC profile and document depth are imported; 16/32-bit RGB layers retain a high-precision working buffer when the source exposes one.

`Файл -> Экспорт совместимого PSD/PSB` writes a layered file and reports every complex UZYRO layer that had to become a pixel layer. Imported Photoshop text remains editable in UZYRO with approximate font parameters and its original visual cache. The current upstream writer cannot create editable Photoshop type, shape or Smart Object records, so the professional round-trip item remains open in `ROADMAP.md`.

## Printing and spot colors

`Файл -> Печать` opens the native Windows printer dialog and sends the document through the selected printer driver. The image is centered and fitted inside the driver's printable area while preserving its aspect ratio.

`Изображение -> Управление цветом -> Плашечные краски` manages document spot inks and assigns them to layers. Libraries can be exchanged as Adobe Swatch Exchange (`.ase`) or UZYRO (`.prswatches`) files. Licensed Pantone ASE libraries can be imported by their owner; UZYRO does not bundle or fabricate proprietary Pantone color data.

The print preparation window can export four process plates and one grayscale TIFF plate for every assigned spot ink. `separations.json` records the profile, dimensions, Lab alternate values, sources and plate filenames.

## Tool panel

- The left side is a scrollable icon-and-name toolbar, so tools remain understandable without memorizing symbols. Hovering any tool shows its shortcut, explanation and an 18-frame looping demonstration rendered by the same UZYRO engine that performs the edit.
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
- Move Auto-Select offers `Слой`, `Группа` and `Выкл`. Selected layers can be grouped or ungrouped from the Layer menu; group membership is saved in projects and a group drag is one Undo operation.
- Dragging moves the selected object live. Arrow keys move it by 1 px and `Shift` + arrow by 10 px.
- Shape handles resize on one or two axes. `Shift` keeps proportions and `Alt` resizes around the center.
- One drag creates one compact history command. Move and resize redraw only affected render tiles and do not store full-document snapshots.
- Bezier paths are edited directly on the canvas with separate path, node, add, delete and convert tools. Anchors support multi-selection, linked or independent handles and grid snapping.
- Shape and text properties stay in the Properties tab. Shapes expose geometry, appearance and editable gradients; text exposes character, paragraph, vertical type and text-on-path settings.
- With a selection tool active, right-clicking outside the selected mask clears the selection; right-clicking inside keeps it.

## Plugins and actions

- Put Python plugins in `%APPDATA%\UZYRO\plugins` and reload them from `Действия -> Перезагрузить плагины`.
- API v2 plugins use a versioned `plugin.json`, stay disabled until their requested permissions are approved in `Действия -> Управление плагинами`, and execute in a guarded child process. Plugin importers and exporters appear under `Файл`.
- The plugin contract and external filter protocol are documented in [`PLUGINS.md`](PLUGINS.md).
- Action v3 records the editor's actual undoable operations, including paint, masks, selections, vector/text edits, layer properties and layer structure. The action editor supports ordering, conditions, stops and per-step error policies; old v2 files remain readable. A ready example is in `actions\web_thumbnail.json`.
- The batch queue supports multiple jobs, PNG/JPEG/WebP/TIFF/BMP output, rename/overwrite/skip collision policies, continue/stop error policies, progress, cancellation, per-file results and queue save/restore.
- `UZYRO_CACHE_MB` controls the RAM budget for render caches; older arrays automatically move to the scratch folder.
- `UZYRO_GPU=auto` benchmarks OpenCL and uses it only when beneficial; `force` always selects GPU and `off` keeps all work on CPU. The same modes and a live benchmark are available under Analysis > GPU settings.

## Navigation

- Command shortcuts shown in menus come from the same validated registry as their handlers and work by physical key on Russian and English layouts.
- Repeated `M`, `L`, `W`, `U`, `J`, `R`, `O` or `G` presses cycle through tools in that group.
- `[` and `]` decrease and increase the active brush-like tool size.
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

The long requirements file describes a Photoshop-scale application. `ROADMAP.md` separates completed workflows from partial prototypes and unimplemented features; a code path or isolated test is not treated as proof that a milestone is finished.
