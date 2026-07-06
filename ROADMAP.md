# PhotoRedactor Roadmap

## Implemented in this foundation

- Local documents: new, open, save, save as, export.
- Workflow: document presets, recent files, layer export and project metadata.
- Document geometry: resize image, resize canvas, trim transparent pixels, reveal all layers, rotate, flip, crop.
- Navigation: zoom, fit to screen, pan with scrollbars.
- Raster layers: create, duplicate, delete, reorder, visibility, opacity, merge, flatten.
- Layer compositing: clipping masks including adjustment-layer clipping, basic layer styles, opacity and blend modes.
- Non-destructive filters: per-layer editable filter stack with visual editor, built-in/importable/exportable presets, preview, reordering, per-filter enable/opacity/blend modes/masks, blur, sharpen, noise, median, edge detect and emboss.
- Raster selections: rectangular, elliptical, freehand lasso, magnetic lasso, polygon lasso, quick selection brush, magic wand, color range, single row/column, opaque pixels, subject selection, select all, invert, feather, smooth, grow, shrink, border/refine, smart edge cleanup, richer Select and Mask previews/output, saved alpha selections.
- Raster layer masks: reveal all, hide all, from selection, mask thumbnails, thumbnail-click mask channel editing, red-overlay/black-white mask preview, linked/unlinked movement, paint on mask, density, feather, invert, toggle, apply, delete.
- Blend modes: Normal, Multiply, Screen, Overlay, Soft Light, Darken, Lighten, Difference, Color, Luminosity.
- Tools: move, brush, eraser, fill, gradient, editable text, rectangular/elliptical/lasso/magnetic/polygon selection, magic wand, color range, interactive patch and crop.
- Text layers: font selection, multiline editing, paragraph width, word wrapping, alignment, line spacing and tracking.
- Local retouching: blur, sharpen, dodge, burn, clone stamp, healing brush, retouch presets, interactive patch selection, edge-aware cleanup, red-eye reduction and content-aware fill.
- Layer import: place embedded images, place linked images, update/relink linked layers and load several image files as layers.
- Shape layers: editable rectangle, ellipse, line, Bezier curve, polygon, star and boolean-combination layers.
- View helpers: grid overlay, numeric guides and channel preview for RGB/Red/Green/Blue/Alpha.
- UI: Russian main menu/tool labels and hover explanations for tools.
- Analysis: eyedropper, metadata/EXIF viewer, histogram and image statistics.
- Recovery and automation: autosave recovery file, manual recovery opening and action recording to JSON logs.
- Adjustments: destructive and previewable non-destructive brightness, contrast, saturation, hue/saturation, exposure, color balance, levels, curves, threshold, posterize, grayscale and invert, with adjustment presets.
- Filters: Gaussian blur, sharpen, noise, median, edge detect, emboss, content-aware fill and red-eye reduction.
- History: undo and redo snapshots.
- Automation: simple batch resize/convert.
- Transform: numeric Free Transform for scale, rotate, position, flips, selected pixels, perspective corner transforms and warp presets.
- Performance baseline: numpy arrays, OpenCV filters, cached composite preview.
- Export: PNG, JPEG, WebP, BMP, TIFF, plus `.prdx` project files.

## Next milestones toward the full list

1. Selection engine: better quick-selection refinement and sky/background selection helpers.
2. Editable masks: stronger mask edge controls.
3. Non-destructive editing: richer editable adjustment controls and preset libraries.
4. Free Transform: transform handles and richer warp preview controls.
5. Text engine: text-on-path, OpenType-style controls and editable transform handles for text boxes.
6. Shapes and paths: custom shapes and richer Bezier point editing.
7. Retouching: frequency separation, spot removal presets and more portrait-oriented cleanup controls.
8. Smart objects: re-editable transforms, embedded payload management and richer linked-file status/conflict handling.
9. RAW/color management: ICC profiles, 16/32-bit channels, Lab/CMYK workflows and richer metadata editing.
10. GPU/tiled rendering: tile cache, dirty-region recomposition, scratch disk, background save.
11. Plugin API and actions: replayable command actions, batch execution, external filters.
12. Advanced AI/generative tools: sky selection, content-aware fill, generative expand.
