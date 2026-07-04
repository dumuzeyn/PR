# PhotoRedactor Roadmap

## Implemented in this foundation

- Local documents: new, open, save, save as, export.
- Workflow: document presets, recent files, layer export and project metadata.
- Document geometry: resize image, resize canvas, rotate, flip, crop.
- Navigation: zoom, fit to screen, pan with scrollbars.
- Raster layers: create, duplicate, delete, reorder, visibility, opacity, merge, flatten.
- Layer compositing: clipping masks, basic layer styles, opacity and blend modes.
- Raster selections: rectangular, elliptical, freehand lasso, polygon lasso, magic wand, color range, single row/column, opaque pixels, select all, invert, feather, grow, shrink, saved alpha selections.
- Raster layer masks: reveal all, hide all, from selection, paint on mask, invert, toggle, apply, delete.
- Blend modes: Normal, Multiply, Screen, Overlay, Soft Light, Darken, Lighten, Difference, Color, Luminosity.
- Tools: move, brush, eraser, fill, gradient, editable text, rectangular/elliptical/lasso/polygon selection, magic wand, color range, crop.
- Local retouching: blur, sharpen, dodge, burn, clone stamp and healing brush.
- Shape layers: editable rectangle, ellipse and line layers.
- View helpers: grid overlay and numeric guides.
- Analysis: eyedropper, metadata/EXIF viewer, histogram and image statistics.
- Recovery and automation: autosave recovery file, manual recovery opening and action recording to JSON logs.
- Adjustments: destructive brightness, contrast, saturation, levels, curves, grayscale, invert plus non-destructive adjustment layers for the same core set.
- Filters: Gaussian blur, sharpen, noise.
- History: undo and redo snapshots.
- Automation: simple batch resize/convert.
- Transform: numeric Free Transform for scale, rotate, position and flips.
- Performance baseline: numpy arrays, OpenCV filters, cached composite preview.
- Export: PNG, JPEG, WebP, BMP, TIFF, plus `.prdx` project files.

## Next milestones toward the full list

1. Selection engine: quick selection, magnetic lasso and better edge refinement.
2. Editable masks: density/feather controls, mask thumbnails and linked/unlinked mask movement.
3. Non-destructive editing: editable filter stack, adjustment-layer UI previews and clipping mask polish.
4. Free Transform: transform handles, warp, perspective and selected-pixel transforms.
5. Text engine: paragraph boxes, font picker, alignment, wrapping and text-on-path.
6. Shapes and paths: Bezier editing, polygon/star/custom shapes and shape boolean operations.
7. Retouching: patch, red-eye, content-aware fill and source-management presets.
8. Smart objects: embedded image payloads, re-editable transforms, linked file updates.
9. RAW/color management: ICC profiles, 16/32-bit channels, Lab/CMYK workflows and richer metadata editing.
10. GPU/tiled rendering: tile cache, dirty-region recomposition, scratch disk, background save.
11. Plugin API and actions: replayable command actions, batch execution, external filters.
12. Advanced AI/generative tools: subject/sky selection, content-aware fill, generative expand.
