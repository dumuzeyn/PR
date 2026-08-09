# PhotoRedactor Roadmap

## Implemented in this foundation

- Local documents: new, open, save, save as, export.
- Workflow: document presets, recent files, layer export and project metadata.
- Startup workflow: centered medium-size editor-free welcome screen with recent files, recovery, clipboard-image detection and a visual preset-based canvas creator; the editor opens maximized, documents open centered and zoom keeps the viewport focus.
- Document geometry: resize image, resize canvas, trim transparent pixels, reveal all layers, rotate, flip, crop.
- Navigation: zoom, fit to screen, pan with scrollbars.
- Raster layers: create, duplicate, delete, reorder, visibility, opacity, merge, flatten.
- Layer compositing: clipping masks including adjustment-layer clipping, basic layer styles, opacity and blend modes.
- Non-destructive filters: per-layer editable filter stack with visual editor, built-in/importable/exportable presets, preview, reordering, per-filter enable/opacity/blend modes/masks, blur, sharpen, noise, median, edge detect and emboss.
- Raster selections: rectangular, elliptical, freehand lasso, magnetic lasso, polygon lasso, persistent replace/add/subtract/intersect modes, quick selection brush with a live result-mask preview and image-aware edge refinement controls, magic wand, color range, single row/column, opaque pixels, subject/background/sky selection, select all, invert, feather, smooth, grow, shrink, border/refine, smart edge cleanup/correction, richer Select and Mask previews/output including edge confidence, saved alpha selections.
- Raster layer masks: reveal all, hide all, from selection, mask thumbnails, thumbnail-click mask channel editing, red-overlay/black-white mask preview, linked/unlinked movement, paint on mask, density, feather, live edge refinement with smooth/contrast/shift and smart image-aware correction, invert, toggle, apply, delete.
- Blend modes: Normal, Multiply, Screen, Overlay, Soft Light, Darken, Lighten, Difference, Color, Luminosity.
- Tools: move, brush, eraser, fill, gradient, editable text, rectangular/elliptical/lasso/magnetic/polygon selection, magic wand, color range, brush/quick-selection canvas previews, interactive patch and crop.
- Text layers: font selection, multiline editing, paragraph width, word wrapping, alignment, line spacing and tracking.
- Local retouching: blur, sharpen, dodge, burn, clone stamp, source-based healing, automatic spot healing, retouch presets, interactive patch selection, edge-aware cleanup, red-eye reduction and content-aware fill.
- Advanced retouching: frequency separation into editable tone/texture layers with Linear Light reconstruction, plus skin-aware portrait cleanup with live preview and controls for smoothing, texture, tone and redness.
- Layer import: place embedded images, place linked images, update/relink linked layers and load several image files as layers.
- Shape layers: editable rectangle, ellipse, line, Bezier curve, polygon, star and boolean-combination layers.
- View helpers: grid overlay, numeric guides and channel preview for RGB/Red/Green/Blue/Alpha.
- UI: centralized neutral DesignTokens theme with light controls and a dark canvas workspace, readable icon-and-name toolbar with Russian tooltips and shortcuts, explicitly labeled horizontal contextual options, tabbed Properties/Layers/History, visible color swatches, canvas-first responsive layout and compact status feedback.
- Object UX: corner-based shape creation with Shift/Alt modifiers, reusable real-shape preview items, direct alpha-aware Shape/Text selection, topmost visible-layer mapping, synchronized canvas/layer selection, screen-stable handles, modifier-aware resize and keyboard nudging.
- Analysis: eyedropper, metadata/EXIF viewer, histogram and image statistics.
- Recovery and automation: autosave recovery file, manual recovery opening and action recording to JSON logs.
- Adjustments: destructive and previewable non-destructive brightness, contrast, saturation, vibrance, temperature/tint, hue/saturation, exposure, color balance, levels, curves, threshold, posterize, grayscale and invert, with importable/exportable preset libraries.
- Filters: Gaussian blur, sharpen, noise, median, edge detect, emboss, content-aware fill and red-eye reduction.
- History: sparse tile commands for strokes, property/list commands for layers and compact changed-field commands, with full snapshots reserved for global document operations.
- Automation: simple batch resize/convert.
- Transform: visual Free Transform with eight handles for scale/move, rotate, position, flips, selected pixels, perspective corner transforms and live warp previews.
- Text: editable paragraph and path text, bold/italic/underline, baseline shift, tracking and transform handles that preserve editability.
- Shapes: custom-shape presets and visual four-point Bezier editing with live handles.
- Tool previews: real ellipse, line, polygon, star, custom shape and Bezier previews before creation.
- Performance: dirty-region 256 px composition, tiled Tk canvas output, cached filters/masks/effects/thumbnails, sparse stroke history, incremental selection previews, revision-safe background filters, regional cached object movement/resize and an opt-in profiler/benchmark.
- Dense shape documents: partial composition rejects non-intersecting Shape layers before filter/cache lookup and blending, keeping direct movement responsive as layer count grows.
- Export: PNG, JPEG, WebP, BMP, TIFF, plus `.prdx` project files.
- Smart Objects: embedded source payloads inside `.prdx`, transforms re-rendered from the original, replace/reset/embed operations and linked-file current/modified/missing status.
- RAW and color management: RAW decoding, assigned/converted ICC profiles, 8/16/32-bit working channels, 16-bit TIFF output, RGB/Lab/CMYK conversions and editable project metadata.
- Large documents: optional CUDA detection with CPU fallback, zoom mipmap pyramids and LRU scratch-disk eviction with a configurable memory budget.
- Extensibility and automation: replayable action v2 commands, batch action execution, Python plugin discovery and validated in-process or external executable filters.
- Generative tools: content-aware fill plus non-destructive local generative canvas expansion with content-aware, mirror and edge modes and live preview.

## Long-term research and optional integrations

The original roadmap milestones are implemented. Future work can add optional cloud model providers, GPU kernels for individual filters and more third-party plugin adapters without changing the local editor's core behavior.
