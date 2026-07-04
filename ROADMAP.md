# PhotoRedactor Roadmap

## Implemented in this foundation

- Local documents: new, open, save, save as, export.
- Workflow: document presets, recent files, layer export and project metadata.
- Document geometry: resize image, resize canvas, trim transparent pixels, reveal all layers, rotate, flip, crop.
- Navigation: zoom, fit to screen, pan with scrollbars.
- Raster layers: create, duplicate, delete, reorder, visibility, opacity, merge, flatten.
- Layer compositing: clipping masks, basic layer styles, opacity and blend modes.
- Raster selections: rectangular, elliptical, freehand lasso, polygon lasso, quick selection brush, magic wand, color range, single row/column, opaque pixels, select all, invert, feather, grow, shrink, saved alpha selections.
- Raster layer masks: reveal all, hide all, from selection, paint on mask, density, feather, invert, toggle, apply, delete.
- Blend modes: Normal, Multiply, Screen, Overlay, Soft Light, Darken, Lighten, Difference, Color, Luminosity.
- Tools: move, brush, eraser, fill, gradient, editable text, rectangular/elliptical/lasso/polygon selection, magic wand, color range, crop.
- Text layers: font selection, multiline editing, paragraph width, word wrapping, alignment and line spacing.
- Local retouching: blur, sharpen, dodge, burn, clone stamp, healing brush, patch selection, red-eye reduction and content-aware fill.
- Layer import: place embedded image and load several image files as layers.
- Shape layers: editable rectangle, ellipse and line layers.
- View helpers: grid overlay, numeric guides and channel preview for RGB/Red/Green/Blue/Alpha.
- Analysis: eyedropper, metadata/EXIF viewer, histogram and image statistics.
- Recovery and automation: autosave recovery file, manual recovery opening and action recording to JSON logs.
- Adjustments: destructive brightness, contrast, saturation, levels, curves, grayscale, invert plus non-destructive adjustment layers for the same core set.
- Filters: Gaussian blur, sharpen, noise, content-aware fill and red-eye reduction.
- History: undo and redo snapshots.
- Automation: simple batch resize/convert.
- Transform: numeric Free Transform for scale, rotate, position and flips.
- Performance baseline: numpy arrays, OpenCV filters, cached composite preview.
- Export: PNG, JPEG, WebP, BMP, TIFF, plus `.prdx` project files.

## Next milestones toward the full list

1. Selection engine: magnetic lasso, better quick-selection refinement and Refine Edge.
2. Editable masks: mask thumbnails and linked/unlinked mask movement.
3. Non-destructive editing: editable filter stack, adjustment-layer UI previews and clipping mask polish.
4. Free Transform: transform handles, warp, perspective and selected-pixel transforms.
5. Text engine: text-on-path, richer typography controls and editable transform handles for text boxes.
6. Shapes and paths: Bezier editing, polygon/star/custom shapes and shape boolean operations.
7. Retouching: interactive patch tool, retouch presets and stronger edge-aware cleanup.
8. Smart objects: re-editable transforms, embedded payload management and linked file updates.
9. RAW/color management: ICC profiles, 16/32-bit channels, Lab/CMYK workflows and richer metadata editing.
10. GPU/tiled rendering: tile cache, dirty-region recomposition, scratch disk, background save.
11. Plugin API and actions: replayable command actions, batch execution, external filters.
12. Advanced AI/generative tools: subject/sky selection, content-aware fill, generative expand.
